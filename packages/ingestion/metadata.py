# packages/ingestion/metadata.py
"""
Metadata helpers — source lineage, tags, tenant/retention labels (tiny & explicit).

What this module provides
-------------------------
- build_document_metadata(... ) -> dict
    Canonical metadata for an ingested **document** (PDF/CSV/DOCX/JSON...).
    Includes lineage (filename/path/url), sizes/checksum, tenant/uploader, tags, and retention hints.

- build_chunk_metadata(doc_meta: dict, *, position: int | None = None, page: int | None = None,
                       section: str | None = None, table_id: str | None = None) -> dict
    Lightweight per-chunk metadata derived from the parent document.

- merge_metadata(base: dict, extra: dict, *, allowlist: set[str] | None = None) -> dict
    Safe merge with optional allowlist of keys to accept from `extra`.

- sanitize_tags(tags: list[str] | set[str] | tuple[str, ...]) -> list[str]
    Normalize, de-duplicate, and bound tag lengths.

Design goals
------------
- Keep the shape **predictable** for downstream indexing/UI.
- Avoid storing secrets; this module is for *descriptive* metadata only.
- Do not import heavy dependencies; pure stdlib and tiny helpers.

Typical usage
-------------
from packages.ingestion.metadata import build_document_metadata, build_chunk_metadata

doc_meta = build_document_metadata(
    filename="Q4_2024_report.pdf",
    storage_path="t0/ingests/abcd1234/Q4_2024_report.pdf",
    mime_type="application/pdf",
    size_bytes=1_234_567,
    checksum="sha256:...",
    tenant_id="t0",
    uploader_id="user_admin",
    source_url="https://example.com/reports/Q4_2024_report.pdf",
    tags=["report","quarterly","finance"]
)

chunk_meta = build_chunk_metadata(doc_meta, position=12, page=7, section="Management Discussion")

Notes on retention/policy
-------------------------
This module does not enforce policy; it only attaches **labels** that other layers
(e.g., policy engine, security) can interpret. See `configs/policies.yaml` for
environment-specific rules. Retention hints here are best-effort:
    {"class": "business", "pii_possible": False, "expires_at": None}
"""

from __future__ import annotations

import datetime as dt
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set

from packages.core.logging import get_logger

log = get_logger(__name__)

# ---------------------------
# Small helpers
# ---------------------------

_TAG_MAX_LEN = 40
_ALLOWED_TAG_RE = re.compile(r"^[a-z0-9_\-\.]+$")  # conservative, UI-friendly


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sanitize_tags(tags: Optional[Iterable[str]]) -> List[str]:
    """
    Normalize user-supplied tags:
      - lower-case
      - strip spaces, replace inner whitespace with underscores
      - keep only [a-z0-9_.-]
      - drop empty
      - de-duplicate while preserving order
      - bound length to _TAG_MAX_LEN (trim end)

    Returns a list suitable for storage/index filters.
    """
    if not tags:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for t in tags:
        s = str(t or "").strip().lower()
        if not s:
            continue
        s = re.sub(r"\s+", "_", s)
        s = "".join(ch for ch in s if _ALLOWED_TAG_RE.match(ch) or ch in "._-" or ch.isalnum())
        s = s[:_TAG_MAX_LEN]
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _basename(path_or_name: Optional[str]) -> Optional[str]:
    if not path_or_name:
        return None
    return os.path.basename(str(path_or_name)).strip() or None


def _ext(path_or_name: Optional[str]) -> Optional[str]:
    b = _basename(path_or_name)
    if not b:
        return None
    _, ext = os.path.splitext(b)
    return ext.lower() or None


def _pii_possible_from_filename(name: Optional[str]) -> bool:
    """
    Very conservative heuristic — filename hints only.
    We do **not** inspect content here (that belongs in security/pii.py).
    """
    if not name:
        return False
    n = name.lower()
    hints = [
        "passport", "idcard", "national_insurance", "nin", "ssn", "social_security",
        "employees", "payroll", "salary", "cv_", "resume", "customer_list",
    ]
    return any(h in n for h in hints)


def _class_from_mime(mime_type: Optional[str]) -> str:
    """
    Coarse document class to help routing & retention UI.
      - "financial" for common finance report types
      - "tabular" for CSV/XLSX
      - "text" for free-form text/word
      - "image" for PNG/JPEG/TIFF (often OCR candidates)
      - "other" default
    """
    mt = (mime_type or "").lower()
    if mt in {
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/html",
    }:
        return "financial"  # most filings/reports will land here
    if mt in {
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/json",
    }:
        return "tabular"
    if mt.startswith("image/"):
        return "image"
    if mt.startswith("text/"):
        return "text"
    return "other"


def merge_metadata(base: Dict[str, Any], extra: Optional[Dict[str, Any]], *, allowlist: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Merge `extra` into `base` with an optional allowlist of keys accepted from `extra`.
    - Nested dicts (1 level) are merged shallowly.
    - Non-dict values in extra overwrite base if key is allowed.

    Returns a new dict.
    """
    if not extra:
        return dict(base)

    allowed = set(allowlist or extra.keys())
    out = dict(base)
    for k, v in extra.items():
        if k not in allowed:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            nv = dict(out[k])
            nv.update(v)
            out[k] = nv
        else:
            out[k] = v
    return out


# ---------------------------
# Public API: document & chunk metadata
# ---------------------------

def build_document_metadata(
    *,
    filename: str,
    storage_path: str,
    mime_type: Optional[str],
    size_bytes: Optional[int],
    checksum: Optional[str],
    tenant_id: str,
    uploader_id: Optional[str],
    source_url: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct a canonical **document** metadata object.

    Args:
      filename     : original filename provided by the client (UI/display)
      storage_path : canonical path/key within storage backend (local/S3)
      mime_type    : best-effort content-type
      size_bytes   : file size, if known
      checksum     : e.g., "sha256:deadbeef..."
      tenant_id    : multi-tenant boundary id (e.g., "t0")
      uploader_id  : user id who initiated ingest (or "system/worker")
      source_url   : original URL (if downloaded)
      tags         : free-form tags (normalized)
      extra        : optional extra metadata; use `allowlist` in merge if needed upstream

    Returns:
      dict with stable fields expected by DB/vectorstore/indexers.
    """
    fn = _basename(filename) or filename
    ext = _ext(fn)
    doc_class = _class_from_mime(mime_type)
    pii_possible = _pii_possible_from_filename(fn)

    base = {
        "created_at": _utc_now_iso(),
        "tenant_id": str(tenant_id),
        "uploader_id": str(uploader_id) if uploader_id else None,
        "filename": fn,
        "ext": ext,
        "storage_path": storage_path,   # where the raw bytes live in Storage
        "source_url": source_url,
        "mime_type": mime_type,
        "size_bytes": int(size_bytes) if (size_bytes is not None) else None,
        "checksum": checksum,
        # High-level classification hints (UI/policy can interpret)
        "retention": {
            "class": doc_class,        # "financial" | "tabular" | "image" | "text" | "other"
            "pii_possible": bool(pii_possible),
            "expires_at": None,        # a policy engine could set this later
        },
        # Free-form labels for filtering and dashboards (normalized)
        "tags": sanitize_tags(tags),
        # Ingestion lineage
        "lineage": {
            "ingest_tool": "api.upload",     # callers can override in `extra`
            "pipeline_version": "v1",
        },
    }

    # Merge extras (unrestricted here; callers can enforce allowlists prior to calling)
    return merge_metadata(base, extra or {})


def build_chunk_metadata(
    doc_meta: Dict[str, Any],
    *,
    position: Optional[int] = None,
    page: Optional[int] = None,
    section: Optional[str] = None,
    table_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct per-chunk metadata derived from the parent document metadata.

    Args:
      doc_meta : parent document metadata (from `build_document_metadata`)
      position : chunk index within document (0-based)
      page     : physical page number for PDFs (1-based recommended)
      section  : logical section title if known (e.g., "MD&A")
      table_id : identifier for table chunks (e.g., "tbl-0003")
      extra    : optional extra key/values

    Returns:
      dict stable for retrieval pipelines.
    """
    base = {
        "tenant_id": doc_meta.get("tenant_id"),
        "document": {
            "filename": doc_meta.get("filename"),
            "storage_path": doc_meta.get("storage_path"),
            "mime_type": doc_meta.get("mime_type"),
            "checksum": doc_meta.get("checksum"),
        },
        "position": int(position) if position is not None else None,
        "page": int(page) if page is not None else None,
        "section": (section or "").strip() or None,
        "table_id": (table_id or "").strip() or None,
        "tags": list(doc_meta.get("tags") or []),
        "retention": dict(doc_meta.get("retention") or {}),
        "created_at": _utc_now_iso(),
    }
    return merge_metadata(base, extra or {})


# --- Compatibility alias for API routes ---  [NEW]
def make_metadata(
    *,
    filename: str,
    storage_path: str,
    mime_type: Optional[str],
    size_bytes: Optional[int],
    checksum: Optional[str],
    tenant_id: str,
    uploader_id: Optional[str],
    source_url: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Backwards-compatible alias used by API routes.
    Delegates to build_document_metadata with the same arguments.
    """
    return build_document_metadata(
        filename=filename,
        storage_path=storage_path,
        mime_type=mime_type,
        size_bytes=size_bytes,
        checksum=checksum,
        tenant_id=tenant_id,
        uploader_id=uploader_id,
        source_url=source_url,
        tags=tags,
        extra=extra,
    )


__all__ = [
    "build_document_metadata",
    "build_chunk_metadata",
    "merge_metadata",
    "sanitize_tags",
    "make_metadata",  # [NEW]
]
