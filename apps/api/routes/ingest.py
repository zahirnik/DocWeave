# apps/api/routes/ingest.py
"""
Ingestion routes — enqueue long-running work (Celery) and report progress.

What this file provides
-----------------------
- POST /ingest/files : accept one or more uploaded files, stage them safely, enqueue a worker job.
- POST /ingest/urls  : accept a list of URLs (e.g., PDFs/CSVs), enqueue a worker job to fetch+ingest.
- GET  /ingest/{id}  : poll job status/progress/result.

Security / RBAC
---------------
- Requires auth (Bearer or X-API-KEY) and scope "rag:ingest".
- Multi-tenant: files are staged under /data/uploads/{tenant_id}/{job_id}/...

Implementation notes
--------------------
- We keep the API thin: quick validation + staging → Celery task does the heavy lifting:
  OCR → parse → chunk → embed → upsert (implemented in apps/worker/tasks.py).
- Files are written via atomic writes; basic size/type checks guard bad inputs.
- We log/audit the action so you have a trail for compliance.

TIP
---
This is tutorial-simple on purpose; swap the local disk staging for S3/GCS in `packages.core.storage`
without touching this route.
"""

from __future__ import annotations

import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

# Auth & scopes
from .auth import get_principal, require_scopes, Principal

# Name resolver to map (tenant, logical_collection) → physical names
from packages.core.naming import resolve_names

# Storage & validation helpers
from packages.core.storage import atomic_write
from packages.ingestion.validators import file_size_ok
from packages.ingestion.metadata import make_metadata  # noqa: F401  # kept for API parity

# Observability & audit
from packages.observability.tracing import trace, set_span_attr, add_event
from packages.core.audit import append_event

# Celery tasks (implemented in apps/worker/tasks.py)
from apps.worker.tasks import ingest_task, ingest_urls_task  # type: ignore[attr-defined]

# NOTE: No router prefix here; main includes this router with prefix="/ingest"
router = APIRouter(tags=["ingest"])

# ---------------------------------
# Request/Response Models
# ---------------------------------
class IngestFilesResponse(BaseModel):
    job_id: str
    collection: str
    accepted_files: List[str]
    skipped_files: List[str] = []
    note: str = Field(
        "Use GET /ingest/{job_id} to poll status. Worker will parse→chunk→embed→upsert.",
        description="Polling info",
    )


class IngestUrlsRequest(BaseModel):
    urls: List[str] = Field(..., min_items=1, description="HTTP(S) URLs to ingest (PDF/CSV/JSON...)")
    collection: str = Field("default", description="Vector collection name (logical)")
    tenant_id: str = Field("t0", description="Tenant id for isolation/tagging")


class IngestUrlsResponse(BaseModel):
    job_id: str
    collection: str
    url_count: int
    note: str = "Use GET /ingest/{job_id} to poll status."


class IngestStatusResponse(BaseModel):
    job_id: str
    state: str = Field(..., example="PENDING")  # PENDING / STARTED / RETRY / FAILURE / SUCCESS
    progress: Optional[float] = Field(None, example=0.42, description="0..1 if reported by task")
    result: Optional[dict] = Field(None, description="Final task result on SUCCESS")
    error: Optional[str] = None


# ---------------------------------
# Constants / helpers (simple guards)
# ---------------------------------
_ALLOWED_EXTS = {".pdf", ".csv", ".xlsx", ".json", ".jsonl", ".txt", ".docx"}
_MAX_MB = 64  # keep tutorial-friendly; adjust as needed


def _ext_ok(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in _ALLOWED_EXTS


def _staging_dir(tenant_id: str, job_id: str) -> str:
    return os.path.join("data", "uploads", tenant_id, job_id)


# ---------------------------------
# Routes
# ---------------------------------

@router.post(
    "/files",
    response_model=IngestFilesResponse,
    summary="Upload files and enqueue ingestion",
    dependencies=[Depends(require_scopes(["rag:ingest"]))],
    include_in_schema=True,  # keep docs enabled; Pydantic v2–safe annotations used
)
@trace("api.ingest.files")
async def ingest_files(
    files: List[UploadFile] = File(..., description="1..N files: PDF/CSV/XLSX/JSON/JSONL/TXT/DOCX"),
    collection: str = Form("default"),
    tenant_id: Optional[str] = Form(None),  # allow None → default to principal.tenant_id / env / 't0'
    principal: Principal = Depends(get_principal),
):
    """
    Stages uploaded files to a per-tenant/job directory and enqueues a Celery job
    that will parse → chunk → embed → upsert into the *tenant-scoped* collection.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Determine effective tenant (prefer explicit form value, else principal, else env, else t0)
    tenant_eff = (tenant_id or getattr(principal, "tenant_id", None) or os.getenv("DEFAULT_TENANT_ID") or "t0").strip()

    # Resolve physical collection names up-front; we pass vector_name to the worker
    names = resolve_names(tenant_eff, collection)
    vector_collection = names.vector_name  # physical name used by vector store

    job_id = uuid.uuid4().hex
    dest_dir = _staging_dir(tenant_eff, job_id)
    os.makedirs(dest_dir, exist_ok=True)

    # Trace
    set_span_attr("tenant_id", tenant_eff)
    set_span_attr("collection.logical", collection)
    set_span_attr("collection.vector_physical", vector_collection)
    set_span_attr("job_id", job_id)

    accepted, skipped = [], []

    for f in files:
        if not f or not getattr(f, "filename", ""):
            skipped.append(getattr(f, "filename", ""))
            continue

        if not _ext_ok(f.filename):
            skipped.append(f.filename)
            continue

        # Simple filename-based check; strict size is enforced post-read below
        if not file_size_ok(f.filename, _MAX_MB):
            # Leave as soft guard; enforce after read
            pass

        dest_path = os.path.join(dest_dir, f.filename)
        try:
            content = await f.read()
        except Exception:
            skipped.append(f.filename)
            continue

        # Final size guard (on disk)
        if len(content) > _MAX_MB * 1024 * 1024:
            skipped.append(f.filename)
            continue

        try:
            atomic_write(dest_path, content)
            accepted.append(dest_path)
        except Exception:
            skipped.append(f.filename)

    if not accepted:
        raise HTTPException(status_code=400, detail=f"No acceptable files. Skipped: {skipped}")

    # Audit trail
    append_event(
        action="ingest.files.enqueue",
        actor=principal.subject,
        details={
            "tenant_id": tenant_eff,
            "collection_logical": collection,
            "collection_vector_physical": vector_collection,
            "files": [os.path.basename(p) for p in accepted],
        },
    )

    # Enqueue Celery job — worker will traverse dest_dir and handle ingestion
    async_result = ingest_task.delay(path=dest_dir, collection=vector_collection, tenant_id=tenant_eff)  # type: ignore[attr-defined]
    add_event(None, "ingest.job_enqueued", {"job_id": async_result.id})

    return IngestFilesResponse(
        job_id=async_result.id,
        collection=vector_collection,
        accepted_files=[os.path.basename(p) for p in accepted],
        skipped_files=skipped,
    )


@router.post(
    "/urls",
    response_model=IngestUrlsResponse,
    summary="Enqueue ingestion for a list of URLs",
    dependencies=[Depends(require_scopes(["rag:ingest"]))],
    include_in_schema=True,
)
@trace("api.ingest.urls")
async def ingest_urls(
    body: IngestUrlsRequest,
    principal: Principal = Depends(get_principal),
):
    """
    Enqueues a worker job that will download the given URLs and ingest their contents.
    We keep downloading in the worker to avoid blocking the API process.
    """
    if not body.urls:
        raise HTTPException(status_code=400, detail="Field 'urls' is required (non-empty).")

    # Determine effective tenant and resolve physical collection
    tenant_eff = (body.tenant_id or getattr(principal, "tenant_id", None) or os.getenv("DEFAULT_TENANT_ID") or "t0").strip()
    names = resolve_names(tenant_eff, body.collection)
    vector_collection = names.vector_name

    # Audit trail
    append_event(
        action="ingest.urls.enqueue",
        actor=principal.subject,
        details={
            "tenant_id": tenant_eff,
            "collection_logical": body.collection,
            "collection_vector_physical": vector_collection,
            "url_count": len(body.urls),
        },
    )

    # Enqueue Celery job
    async_result = ingest_urls_task.delay(urls=body.urls, tenant_id=tenant_eff, collection=vector_collection)  # type: ignore[attr-defined]
    set_span_attr("tenant_id", tenant_eff)
    set_span_attr("collection.logical", body.collection)
    set_span_attr("collection.vector_physical", vector_collection)
    set_span_attr("job_id", async_result.id)
    add_event(None, "ingest.job_enqueued", {"job_id": async_result.id, "kind": "urls"})

    return IngestUrlsResponse(job_id=async_result.id, collection=vector_collection, url_count=len(body.urls))


@router.get(
    "/{job_id}",
    response_model=IngestStatusResponse,
    summary="Poll the ingestion job status/progress",
    dependencies=[Depends(require_scopes(["rag:ingest"]))],
    include_in_schema=True,
)
@trace("api.ingest.status")
async def ingest_status(job_id: str):
    """
    Returns Celery status and (if reported) a progress float 0..1.
    On SUCCESS, includes the worker's result payload (e.g., doc/chunk counts).
    """
    try:
        from celery.result import AsyncResult  # lazy import keeps api fast to import
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Celery not installed/configured: {e}")

    ar = AsyncResult(job_id)
    payload = None
    error = None
    progress = None

    # If your tasks use `self.update_state(meta={'progress': x})`, Celery stores it in ar.info
    if isinstance(ar.info, dict) and "progress" in ar.info:
        try:
            progress = float(ar.info.get("progress"))
        except Exception:
            progress = None

    if ar.state == "FAILURE":
        error = str(ar.info) if ar.info else "unknown error"

    if ar.state == "SUCCESS":
        payload = ar.result if isinstance(ar.result, dict) else {"result": ar.result}

    return IngestStatusResponse(job_id=job_id, state=ar.state, progress=progress, result=payload, error=error)
