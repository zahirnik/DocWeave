# apps/worker/tasks.py
"""
Celery tasks: ingest → (OCR) → parse → chunk → embed → upsert, with retries & progress.

... (docstring unchanged for brevity)
"""

from __future__ import annotations

import os
import io
import uuid
import time
import shutil
import hashlib
import mimetypes
import tempfile
import json  # [CHANGED]
import re    # [ADDED]
from typing import Dict, Any, List, Tuple, Iterable, Optional

import pandas as pd
from celery import shared_task  # noqa: F401  # (kept; app.task is used)
# Observability & audit
from packages.observability.tracing import start_span, add_event
from packages.core.audit import append_event

# === [ADDED] Tenant/collection resolver ===
from packages.core.naming import resolve_names  # [ADDED]

# Ingestion helpers
from packages.ingestion.loaders_pdf import load_pdf_text
from packages.ingestion.loaders_docx import load_docx_text
from packages.ingestion.loaders_tabular import load_csv, load_xlsx
from packages.ingestion.loaders_json import load_json, load_jsonl
from packages.ingestion.ocr import ocr_image_bytes
from packages.ingestion.normalizers import normalize_text as clean_text  # [CHANGED]

# === Retriever stack ===
from packages.retriever.chunking import chunk_text
from packages.retriever.embeddings import EmbeddingConfig, Embeddings  # [CHANGED]
# [CHANGED] switch to real vector stores (env-selected)
from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # [ADDED]
from packages.retriever.vectorstores.qdrant_store import QdrantStore      # [ADDED]
from packages.retriever.vectorstores.chroma_store import ChromaStore      # [ADDED]
# BM25 backend (Whoosh/Elastic)
from packages.retriever.bm25 import BM25, BM25Config  # [ADDED]

# Celery app config
from .celery_app import app

# HTTP client for KG call
try:
    import requests  # [ADDED]
except Exception:  # pragma: no cover
    requests = None  # [ADDED]


# ---------------------------
# Config / constants
# ---------------------------

_ALLOWED_EXTS = {
    ".pdf", ".docx", ".txt", ".csv", ".xlsx", ".json", ".jsonl",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
}
_MAX_MB = 64
_MAX_CSV_CELLS = 1_000_000

# Embedding config (env-driven)
_EMBED_CFG = EmbeddingConfig(
    provider=os.getenv("EMBEDDINGS_PROVIDER", "local"),
    model=os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5"),
    batch_size=int(os.getenv("EMBEDDINGS_BATCH", "64")),
    normalize=True,
    timeout_s=int(os.getenv("EMBEDDINGS_TIMEOUT_S", "30")),
    max_retries=int(os.getenv("EMBEDDINGS_MAX_RETRIES", "3")),
)

# [ADDED] one Embeddings instance to infer dimension & (optionally) reuse
_EMB = Embeddings(config=_EMBED_CFG)  # lightweight; local sentence-transformers

# [ADDED] KG build integration flags
_KG_ON = (os.getenv("KG_BUILD_ON_INGEST", "false").strip().lower() == "true")
_KG_BASE = os.getenv("KG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_KG_VALIDATE = (os.getenv("KG_VALIDATE", "false").strip().lower() == "true")
_KG_TIMEOUT = int(os.getenv("KG_TIMEOUT_S", "180"))
_KG_ENTITY_KEY_FIXED = os.getenv("KG_ENTITY_KEY")  # if set, use this for all files in the task
_KG_ENTITY_KEY_PREFIX = os.getenv("KG_ENTITY_KEY_PREFIX")  # fallback: prefix + logical collection
_KG_ENTITY_NAME_OVERRIDE = os.getenv("KG_ENTITY_NAME")  # optional override


# ---------------------------
# Utility helpers
# ---------------------------

def _checksum_bytes(b: bytes) -> str:
    m = hashlib.sha256()
    m.update(b)
    return m.hexdigest()


def _slugify(s: str) -> str:  # [ADDED]
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "doc"


def _iter_files(root: str) -> Iterable[str]:
    """Yield absolute paths of files under `root` matching allowed extensions."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = os.path.join(dirpath, name)
            _, ext = os.path.splitext(name.lower())
            if ext in _ALLOWED_EXTS:
                yield os.path.abspath(p)


def _load_any_text(path: str) -> str:
    """Return a best-effort text representation for supported file types."""
    _, ext = os.path.splitext(path.lower())

    if ext == ".pdf":
        return clean_text(load_pdf_text(path))
    if ext == ".docx":
        return clean_text(load_docx_text(path))
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return clean_text(f.read())
    if ext == ".csv":
        df = load_csv(path)
        if df.size > _MAX_CSV_CELLS:
            df = df.head(int(_MAX_CSV_CELLS / max(1, len(df.columns))))
        return df.to_csv(index=False)
    if ext == ".xlsx":
        df = load_xlsx(path)
        if df.size > _MAX_CSV_CELLS:
            df = df.head(int(_MAX_CSV_CELLS / max(1, len(df.columns))))
        return df.to_csv(index=False)
    if ext == ".json":
        obj = load_json(path)
        return clean_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    if ext == ".jsonl":
        rows = load_jsonl(path)
        return "\n".join([str(r) for r in rows])
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        with open(path, "rb") as f:
            data = f.read()
        return clean_text(ocr_image_bytes(data))
    # Fallback: try read as text
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())


def _download_url(url: str, dest_dir: str) -> str:
    """Download a URL to dest_dir and return the saved path."""
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    base = url.split("?")[0].rstrip("/").split("/")[-1] or "download"
    if "." not in base:
        try:
            with urllib.request.urlopen(url) as resp:
                ctype = resp.headers.get("Content-Type", "")
            ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or ".bin"
        except Exception:
            ext = ".bin"
        base = base + ext
    name = f"{uuid.uuid4().hex[:8]}_{base}"
    out = os.path.join(dest_dir, name)
    urllib.request.urlretrieve(url, out)  # nosec (tutorial)
    return out


# ---------------------------
# Vector/BM25 adapters (env-driven)
# ---------------------------

def _guess_dim() -> int:
    v = _EMB.embed_query("ping")
    return len(v)


class _VectorStoreAdapter:
    """Tiny facade so the worker can call `upsert_texts(collection, items)` for any backend."""
    def __init__(self):
        self.kind = (os.getenv("VECTOR_STORE") or "chroma").strip().lower()
        if self.kind == "pgvector":
            dsn = os.getenv("PG_DSN", "")
            if not dsn:
                raise RuntimeError("PG_DSN is required when VECTOR_STORE=pgvector")
            self._store = PgVectorStore(dsn, dimension=_guess_dim())
            self._store.ensure_schema()
        elif self.kind == "qdrant":
            self._store = QdrantStore(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY") or None,
            )
            self._store.ensure_client()
        elif self.kind == "chroma":
            self._store = ChromaStore(persist_dir=os.getenv("CHROMA_DIR", ".chroma"))
            self._store.ensure_client()
        else:
            raise RuntimeError(f"Unknown VECTOR_STORE: {self.kind}")

    def create_collection(self, name: str) -> None:
        dim = _guess_dim()
        if isinstance(self._store, PgVectorStore):
            self._store.create_collection(name, metadata={})
        elif isinstance(self._store, QdrantStore):
            self._store.create_collection(name, dimension=dim, metadata={})
        elif isinstance(self._store, ChromaStore):
            self._store.create_collection(name, dimension=dim, metadata={})

    def upsert_texts(self, collection: str, items: List[dict]) -> int:
        """Each item: {'id': str, 'text': str, 'metadata': dict}."""
        # All adapters implement upsert_texts(collection, items)
        return int(self._store.upsert_texts(collection, items))


def _bm25_client() -> BM25:
    return BM25(config=BM25Config.from_env())


def _make_items(source_path: str, chunks: List[str], tenant_id: str) -> List[dict]:
    """Build stable ids & metadata per chunk."""
    items = []
    base = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:16]
    ext = os.path.splitext(source_path)[1][1:]
    for i, text in enumerate(chunks):
        cid = hashlib.sha1(f"{base}:{i}".encode("utf-8")).hexdigest()
        items.append({
            "id": cid,
            "text": text,
            "metadata": {
                "source": source_path,
                "type": ext,
                "tenant_id": tenant_id,
                "chunk_index": i,
            },
        })
    return items


# ---------------------------
# KG integration helpers [ADDED]
# ---------------------------

def _kg_enabled() -> bool:
    return bool(_KG_ON and requests is not None)


def _derive_entity_key(logical_collection: str, source_path: str) -> str:
    """
    Priority:
      1) KG_ENTITY_KEY (forces a single key for the whole task)
      2) KG_ENTITY_KEY_PREFIX + logical_collection
      3) logical_collection if it looks namespaced (contains ':')
      4) fallback to doc-based key
    """
    if _KG_ENTITY_KEY_FIXED:
        return _KG_ENTITY_KEY_FIXED
    if _KG_ENTITY_KEY_PREFIX:
        return f"{_KG_ENTITY_KEY_PREFIX}{logical_collection}"
    if ":" in (logical_collection or ""):
        return logical_collection
    base = os.path.splitext(os.path.basename(source_path))[0]
    return f"doc:{_slugify(base)}"


def _derive_entity_name(entity_key: str, source_path: str) -> str:
    if _KG_ENTITY_NAME_OVERRIDE:
        return _KG_ENTITY_NAME_OVERRIDE
    if ":" in entity_key:
        return entity_key.split(":", 1)[1] or entity_key
    base = os.path.splitext(os.path.basename(source_path))[0]
    return base


def _derive_doc_id(source_path: str) -> str:
    base = os.path.splitext(os.path.basename(source_path))[0]
    return f"doc:{_slugify(base)}"


def _items_to_kg_chunks(doc_id: str, items: List[dict]) -> List[dict]:
    out = []
    for i, it in enumerate(items):
        out.append({
            "id": f"{doc_id}#chunk{i}",
            "text": it["text"],
            "order": i,
            "meta": {"source": it.get("metadata", {}).get("source", doc_id)},
        })
    return out


def _post_kg_build(
    tenant_id: str,
    logical_collection: str,
    source_path: str,
    items: List[dict],
) -> None:
    if not _kg_enabled():
        return
    try:
        entity_key = _derive_entity_key(logical_collection, source_path)
        entity_name = _derive_entity_name(entity_key, source_path)
        doc_id = _derive_doc_id(source_path)
        chunks = _items_to_kg_chunks(doc_id, items)

        payload = {
            "tenant_id": tenant_id,
            "entity_key": entity_key,
            "entity_name": entity_name,
            "doc_id": doc_id,
            "chunks": chunks,
            "validate": bool(_KG_VALIDATE),
        }

        with start_span("ingest.kg_build", {
            "tenant_id": tenant_id,
            "entity_key": entity_key,
            "doc_id": doc_id,
            "chunks": len(chunks),
        }):
            resp = requests.post(
                f"{_KG_BASE}/kg/build",
                headers={"accept": "application/json", "content-type": "application/json"},
                json=payload,
                timeout=_KG_TIMEOUT,
            )
            ok = 200 <= resp.status_code < 300
            add_event(None, "ingest.kg_build.result", {
                "status": resp.status_code,
                "ok": ok,
                "entity_key": entity_key,
                "doc_id": doc_id,
            })
    except Exception as e:  # never fail the ingestion because of KG
        add_event(None, "ingest.kg_build.error", {
            "error": str(e),
            "source": source_path,
            "kg_base": _KG_BASE,
        })


# ---------------------------
# Core ingest helpers
# ---------------------------

def _ingest_fileset(
    file_paths: List[str],
    tenant_id: str,
    logical_collection: str,
    vector_collection: str,
    bm25_collection: str,
    progress_cb,
) -> Tuple[int, int, List[str]]:
    """
    Core ingestion loop. Returns (files_ingested, chunks_upserted, skipped_files).
    """
    vstore = _VectorStoreAdapter()       # [ADDED] real vector store
    vstore.create_collection(vector_collection)

    bm25 = _bm25_client()                # [ADDED] lexical store (whoosh/elastic)

    files_ingested = 0
    total_chunks = 0
    skipped: List[str] = []

    N = max(1, len(file_paths))
    for i, path in enumerate(file_paths, start=1):
        try:
            if os.path.getsize(path) > _MAX_MB * 1024 * 1024:
                skipped.append(os.path.basename(path))
                progress_cb(i / N)
                continue

            with start_span("ingest.parse", {"path": path}):
                text = _load_any_text(path)

            # Token-aware chunking
            chunks = chunk_text(text, max_tokens=400, overlap=40)

            # Build unified items (id, text, metadata) once
            items = _make_items(path, chunks, tenant_id)

            # Upsert vectors
            with start_span("ingest.upsert.vector", {"chunks": len(items)}):
                total_chunks += vstore.upsert_texts(vector_collection, items)

            # Upsert BM25 (safe to call per file; backends are incremental)
            with start_span("ingest.upsert.bm25", {"chunks": len(items)}):
                bm25.index(bm25_collection, items)

            # [ADDED] Optional: also build KG for this document
            if _kg_enabled():
                _post_kg_build(tenant_id=tenant_id, logical_collection=logical_collection, source_path=path, items=items)

            files_ingested += 1

        except Exception as e:
            skipped.append(os.path.basename(path))
            add_event(None, "ingest.file_error", {"file": path, "error": str(e)})

        finally:
            progress_cb(i / N)

    return files_ingested, total_chunks, skipped


# ---------------------------
# Celery tasks
# ---------------------------

@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 3},
)
def ingest_task(self, path: str, collection: str, tenant_id: str = "t0") -> Dict[str, Any]:
    """
    Ingest a directory of staged files (uploaded earlier through the API).
    The API must pass a *resolved* vector collection; we resolve here again for BM25 symmetry.
    """
    t0 = time.time()
    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")

    # Resolve names (handles both when API passed logical or physical)
    names = resolve_names(tenant_id, collection)  # accepts either; returns both consistently
    logical = names.logical
    vector_collection = names.vector_name
    bm25_collection = names.bm25_name

    file_paths = list(_iter_files(path))
    total = len(file_paths)
    if total == 0:
        return {"files": 0, "chunks": 0, "collection": vector_collection, "skipped": []}

    def _progress(p: float, note: Optional[str] = None):
        self.update_state(state="PROGRESS", meta={"progress": float(p), "note": note or ""})

    append_event(
        "ingest.start",
        actor="worker",
        details={
            "tenant_id": tenant_id,
            "collection_logical": logical,
            "collection_vector_physical": vector_collection,
            "collection_bm25_physical": bm25_collection,
            "files": total,
            "kg_on": _KG_ON,  # [ADDED]
        },
    )

    with start_span("ingest.run", {
        "tenant_id": tenant_id,
        "collection.logical": logical,
        "collection.vector": vector_collection,
        "collection.bm25": bm25_collection,
        "files": total,
        "kg_on": _KG_ON,  # [ADDED]
    }):
        files_ingested, chunks, skipped = _ingest_fileset(
            file_paths=file_paths,
            tenant_id=tenant_id,
            logical_collection=logical,
            vector_collection=vector_collection,
            bm25_collection=bm25_collection,
            progress_cb=lambda p: _progress(p, note="processing"),
        )

    append_event(
        "ingest.finish",
        actor="worker",
        details={
            "tenant_id": tenant_id,
            "collection_vector": vector_collection,
            "collection_bm25": bm25_collection,
            "files": files_ingested,
            "chunks": chunks,
            "skipped": skipped,
        },
    )

    return {
        "files": int(files_ingested),
        "chunks": int(chunks),
        "collection": vector_collection,
        "skipped": skipped,
        "elapsed_s": round(time.time() - t0, 3),
    }


@app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 3},
)
def ingest_urls_task(self, urls: List[str], tenant_id: str = "t0", collection: str = "default") -> Dict[str, Any]:
    """
    Download the given URLs to a temp tenant/job folder, then ingest them.
    """
    names = resolve_names(tenant_id, collection)
    logical = names.logical
    vector_collection = names.vector_name
    bm25_collection = names.bm25_name

    if not urls:
        return {"files": 0, "chunks": 0, "collection": vector_collection, "skipped": []}

    base = os.path.join("data", "uploads", tenant_id, f"urljob_{uuid.uuid4().hex}")
    os.makedirs(base, exist_ok=True)

    # Download phase (20%)
    skipped_dl: List[str] = []
    for i, u in enumerate(urls, start=1):
        try:
            _ = _download_url(u, dest_dir=base)
        except Exception:
            skipped_dl.append(u)
        finally:
            frac = 0.2 * (i / max(1, len(urls)))
            self.update_state(state="PROGRESS", meta={"progress": float(frac), "note": "downloading"})

    # Ingest phase (80%)
    file_paths = list(_iter_files(base))
    if not file_paths:
        return {"files": 0, "chunks": 0, "collection": vector_collection, "skipped": skipped_dl}

    def _progress_ingest(p: float):
        self.update_state(state="PROGRESS", meta={"progress": float(0.2 + 0.8 * p), "note": "processing"})

    append_event(
        "ingest.urls.start",
        actor="worker",
        details={
            "tenant_id": tenant_id,
            "collection_logical": logical,
            "collection_vector_physical": vector_collection,
            "collection_bm25_physical": bm25_collection,
            "url_count": len(urls),
            "kg_on": _KG_ON,  # [ADDED]
        },
    )

    with start_span("ingest.urls", {
        "tenant_id": tenant_id,
        "collection.logical": logical,
        "collection.vector": vector_collection,
        "collection.bm25": bm25_collection,
        "files": len(file_paths),
        "kg_on": _KG_ON,  # [ADDED]
    }):
        files_ingested, chunks, skipped_proc = _ingest_fileset(
            file_paths=file_paths,
            tenant_id=tenant_id,
            logical_collection=logical,
            vector_collection=vector_collection,
            bm25_collection=bm25_collection,
            progress_cb=_progress_ingest,
        )

    try:
        shutil.rmtree(base, ignore_errors=True)
    except Exception:
        pass

    skipped_all = skipped_dl + skipped_proc

    append_event(
        "ingest.urls.finish",
        actor="worker",
        details={
            "tenant_id": tenant_id,
            "collection_vector": vector_collection,
            "collection_bm25": bm25_collection,
            "files": files_ingested,
            "chunks": chunks,
            "skipped": skipped_all,
        },
    )

    return {
        "files": int(files_ingested),
        "chunks": int(chunks),
        "collection": vector_collection,
        "skipped": skipped_all,
    }
