# packages/core/naming.py
"""
Unified, tenant-aware collection/index naming for ALL backends.

Why this exists
---------------
Previously, different parts of the system (ingestion, search, KG) built their
own physical names, which led to "empty results" and subtle mismatches
(e.g., writing vectors to 'ten_t0_demo' but querying BM25 'demo').

This module is the *single source of truth* for:
  1) Computing the effective physical name from (tenant_id, logical_name)
  2) Sanitizing it per backend (OpenSearch/Elasticsearch, pgvector, Chroma, Qdrant, Whoosh)
  3) Exposing convenience helpers so EVERY call site uses the same rules

Environment flags (read here; do not re-read elsewhere)
-------------------------------------------------------
- MULTITENANCY_USE_COLLECTIONS=true|false
- COLLECTION_PREFIX=ten_
- TENANT_DEFAULT=t0                (fallback when a tenant_id isn't provided)

Backends:
- Vector:  VECTOR_STORE=chroma|pgvector|qdrant|simple
- BM25:    BM25_PROVIDER=whoosh|elastic

Usage
-----
from packages.core.naming import (
    resolve_names,            # returns all names (logical, physical, per-backend)
    effective_collection,     # physical name used for both backends
    vector_collection_name,   # backend-safe name for vector store
    bm25_index_name,          # backend-safe name for BM25/lexical store
)

At call sites:
- Ingestion BEFORE writing to ANY backend:
    names = resolve_names(tenant_id, logical_collection)
    vs_name = names.vector_name
    bm_name = names.bm25_name

- Search route AFTER reading tenant + collection:
    names = resolve_names(tenant_id, collection)
    pass names.vector_name and names.bm25_name to the adapters

NEVER construct names manually in other modules.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# ----------------------------- helpers & config ------------------------------------

_ALLOWED_GENERIC = re.compile(r"[^a-z0-9_\-]")  # safe for most local stores

def _get_env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default

def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

@dataclass(frozen=True)
class NamingSettings:
    use_multi: bool
    prefix: str
    tenant_default: str
    vector_store: str
    bm25_provider: str
    pg_max_ident_len: int = 63          # PostgreSQL identifier length
    es_max_index_len: int = 255         # ES/OpenSearch index name bytes (practically safe cap)
    generic_max_len: int = 128          # Chroma/Qdrant/Whoosh safe cap

    @staticmethod
    def load() -> "NamingSettings":
        return NamingSettings(
            use_multi=_to_bool(_get_env("MULTITENANCY_USE_COLLECTIONS", "true")),
            prefix=_get_env("COLLECTION_PREFIX", "ten_"),
            tenant_default=_get_env("TENANT_DEFAULT", _get_env("DEFAULT_TENANT_ID", "t0")),
            vector_store=_get_env("VECTOR_STORE", "chroma").lower(),
            bm25_provider=_get_env("BM25_PROVIDER", "whoosh").lower(),
        )

# ----------------------------- sanitizers per backend ------------------------------

def _sanitize_generic(name: str, max_len: int) -> str:
    """
    Safe for Chroma, Qdrant, Whoosh directories: lowercase, replace disallowed with '_',
    trim separators, limit length.
    """
    s = name.lower().replace(".", "_").replace(" ", "_")
    s = _ALLOWED_GENERIC.sub("_", s)
    s = s.strip("_-")
    if len(s) > max_len:
        s = s[:max_len]
    return s

def _sanitize_pg_identifier(name: str, max_len: int) -> str:
    """
    Postgres/pgvector table/index naming:
      - unquoted identifiers are folded to lowercase
      - must start with a letter or underscore
      - subsequent chars: letters/digits/underscore
      - max length ~63 bytes
    We coerce to snake_case and ensure leading char allowed.
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    if not re.match(r"^[a-z_]", s):
        s = f"t_{s}"
    s = re.sub(r"__+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len]
    return s

def _sanitize_es_index(name: str, max_len: int) -> str:
    """
    Elasticsearch/OpenSearch index name constraints:
      - lowercase only
      - cannot contain \/*?"<>| ,#
      - cannot start with -, _, +
      - cannot be . or ..
      - max length 255
    """
    s = name.lower()
    # replace forbidden chars with '-'
    s = re.sub(r"[\\/*?\"<>|\s,#]+", "-", s)
    # replace dots with '-' to avoid hidden or reserved names, and segment issues
    s = s.replace(".", "-")
    s = s.strip()
    # not exactly '.' or '..'
    if s in {".", ".."}:
        s = s.replace(".", "-")
    # cannot start with '-', '_', '+'
    if s.startswith(("-", "_", "+")):
        s = f"i{s}"
    # collapse multiple '-' and trim
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len]
    # ensure non-empty
    if not s:
        s = "index"
    return s

# ----------------------------- core logic ------------------------------------------

def effective_collection(tenant_id: str | None, logical_name: str, *, settings: NamingSettings | None = None) -> str:
    """
    Compute the physical collection name used uniformly across backends,
    before backend-specific sanitization.

    If multitenancy is ON:
        raw = f"{COLLECTION_PREFIX}{tenant_id}_{logical_name}"
    else:
        raw = logical_name

    Empty or None tenant_id falls back to TENANT_DEFAULT.
    """
    st = settings or NamingSettings.load()
    tid = (tenant_id or "").strip() or st.tenant_default
    raw = f"{st.prefix}{tid}_{logical_name}" if st.use_multi else logical_name
    # basic normalization; backend-specific sanitize happens later
    raw = raw.strip()
    return raw

@dataclass(frozen=True)
class ResolvedNames:
    logical: str          # what the client/API sent (e.g., "filings")
    tenant: str           # effective tenant id used (e.g., "t0")
    effective: str        # unsanitized physical name (e.g., "ten_t0_filings")
    vector_name: str      # sanitized for active vector backend
    bm25_name: str        # sanitized for active BM25 backend

def resolve_names(tenant_id: str | None, logical_name: str, *, settings: NamingSettings | None = None) -> ResolvedNames:
    """
    Return a bundle of names ready for backends. This is the function call sites should use.
    """
    st = settings or NamingSettings.load()
    tid = (tenant_id or "").strip() or st.tenant_default
    eff = effective_collection(tid, logical_name, settings=st)

    # Per-backend sanitization
    # Vector store
    vs = st.vector_store
    if vs == "pgvector":
        vector_physical = _sanitize_pg_identifier(eff, st.pg_max_ident_len)
    elif vs == "qdrant":
        vector_physical = _sanitize_generic(eff, st.generic_max_len)  # Qdrant allows [a-zA-Z0-9_-], we keep lowercase
    elif vs in {"chroma", "simple"}:
        vector_physical = _sanitize_generic(eff, st.generic_max_len)
    else:
        # Unknown vector backend: fall back to conservative generic
        vector_physical = _sanitize_generic(eff, st.generic_max_len)

    # BM25 provider
    bm = st.bm25_provider
    if bm in {"elastic", "elasticsearch", "opensearch"}:
        bm25_physical = _sanitize_es_index(eff, st.es_max_index_len)
    elif bm == "whoosh":
        bm25_physical = _sanitize_generic(eff, st.generic_max_len)
    else:
        # Unknown BM25 backend: conservative
        bm25_physical = _sanitize_generic(eff, st.generic_max_len)

    return ResolvedNames(
        logical=logical_name,
        tenant=tid,
        effective=eff,
        vector_name=vector_physical,
        bm25_name=bm25_physical,
    )

# ----------------------------- convenience wrappers --------------------------------

def vector_collection_name(tenant_id: str | None, logical_name: str) -> str:
    """Backend-appropriate physical name for the active vector store."""
    return resolve_names(tenant_id, logical_name).vector_name

def bm25_index_name(tenant_id: str | None, logical_name: str) -> str:
    """Backend-appropriate physical name for the active BM25/lexical store."""
    return resolve_names(tenant_id, logical_name).bm25_name

# ----------------------------- validation hooks (optional) --------------------------

class NamingError(RuntimeError):
    pass

def assert_backends_known() -> None:
    """
    Fail fast if env points to unsupported backends.
    Call once at startup.
    """
    st = NamingSettings.load()
    supported_vec = {"chroma", "pgvector", "qdrant", "simple"}
    supported_bm = {"whoosh", "elastic", "elasticsearch", "opensearch"}

    if st.vector_store not in supported_vec:
        raise NamingError(f"Unsupported VECTOR_STORE='{st.vector_store}'. Supported: {sorted(supported_vec)}")

    if st.bm25_provider not in supported_bm:
        raise NamingError(f"Unsupported BM25_PROVIDER='{st.bm25_provider}'. Supported: {sorted(supported_bm)}")

def log_name_resolution(logger, tenant_id: str | None, logical_name: str) -> ResolvedNames:
    """
    Convenience: resolve and log a single structured line for diagnostics.
    """
    names = resolve_names(tenant_id, logical_name)
    try:
        logger.info(
            "collection.resolve",
            extra={
                "tenant": names.tenant,
                "logical": names.logical,
                "effective": names.effective,
                "vector_store": NamingSettings.load().vector_store,
                "vector_name": names.vector_name,
                "bm25_provider": NamingSettings.load().bm25_provider,
                "bm25_name": names.bm25_name,
            },
        )
    except Exception:
        # logger may not support 'extra' dict; degrade to plain string
        logger.info(
            f"[collection.resolve] tenant={names.tenant} logical={names.logical} "
            f"effective={names.effective} vector={names.vector_name} bm25={names.bm25_name}"
        )
    return names
