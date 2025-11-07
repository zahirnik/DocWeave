# apps/api/routes/search.py
"""
Search routes — hybrid retrieval (vector + BM25) with simple filters & pagination.

What this file provides
-----------------------
- GET /search :
    - Vector search over the configured collection.
    - Optional BM25 (lexical) search and a simple hybrid combiner.
    - Lightweight metadata filters (type/source_contains) and pagination.

Security / RBAC
---------------
- Requires scope: ["rag:query"].
- Multi-tenant filtering belongs in your vector/BM25 backend; here we only show the pattern.

Notes on BM25 (dev vs prod)
---------------------------
- In this tutorial scaffold, BM25 is a tiny placeholder. For production:
  - Use Elastic/OpenSearch or Whoosh/Lucene.
  - Keep the **same interface** as used below.
  - Return results with `{"score": float, "metadata": {...}}` to match vector hits.

Hybrid scoring (simple & clear)
-------------------------------
- Normalize each result set to [0, 1] using min-max on that set.
- Combine with:  hybrid_score = w_vec * vec_norm + w_bm25 * bm25_norm
- De-duplicate by a stable key (here we use the "source" field from metadata).

This file stays small and explicit on purpose.

[CHG] Fusion is now handled in `HybridSearcher` to avoid drift with backends and naming.
      We keep legacy helpers and the former fusion path commented below for reference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .auth import get_principal, require_scopes, Principal
from packages.observability.tracing import trace, set_span_attr

# Vector search orchestrator (embeds query and searches the active store)
from packages.retriever.search import HybridSearcher

# BM25 (dev placeholder). Swap with a real backend (Elastic/OpenSearch) in production.
# [NOTE] Kept import + factory for compatibility/diagnostics; fusion now happens in HybridSearcher.
from packages.retriever.bm25 import BM25, BM25Config

# [ADD] Unified, tenant-aware collection naming resolver (single source of truth)
from packages.core.naming import resolve_names  # backend-safe physical names

router = APIRouter()

# Lazily-init singletons
_searcher: Optional[HybridSearcher] = None
_bm25: Optional[BM25] = None  # [NOTE] kept for legacy/diagnostics


def _get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher.from_env()
    return _searcher


def _get_bm25() -> BM25:
    """[LEGACY/DIAG] Construct BM25 backend if needed for diagnostics."""
    global _bm25
    if _bm25 is None:
        _bm25 = BM25(config=BM25Config.from_env())
    return _bm25


# ---------------------------
# Schemas
# ---------------------------

class SearchFilters(BaseModel):
    """Simple metadata filter example. Extend as needed."""
    type: Optional[str] = Field(None, description="Filter by metadata.type (e.g., pdf|csv|json)")
    source_contains: Optional[str] = Field(None, description="Substring match against metadata.source")


class SearchHit(BaseModel):
    score: float = Field(..., description="Final score (post-fusion / rerank if enabled)")
    method: str = Field(..., example="vector", description="vector|bm25|hybrid")
    metadata: Dict[str, Any]


class SearchPage(BaseModel):
    page: int
    size: int
    total: int


class SearchResponse(BaseModel):
    items: List[SearchHit]
    page: SearchPage
    notes: Optional[str] = Field(
        None,
        description="Hints about hybrid mode or when BM25 is disabled/unavailable.",
    )


# ---------------------------
# Helpers
# ---------------------------

def _stable_key(hit: Dict[str, Any]) -> str:
    """
    Stable id for de-duplication across result sets.
    We bias toward the document path/URL so chunks from the same file group together.
    [LEGACY] Used by prior route-level fusion; kept for reference/possible future use.
    """
    md = hit.get("metadata") or {}
    src = str(md.get("source") or md.get("doc_id") or "unknown")
    return src


def _apply_filters(hits: List[Dict[str, Any]], flt: SearchFilters) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        md = h.get("metadata") or {}
        if flt.type and str(md.get("type")) != flt.type:
            continue
        if flt.source_contains and flt.source_contains.lower() not in str(md.get("source", "")).lower():
            continue
        out.append(h)
    return out


def _minmax_norm(scores: List[float]) -> List[float]:
    """
    [LEGACY] Route-level normalization helper; left for reference.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _paginate(items: List[SearchHit], page: int, size: int) -> Tuple[List[SearchHit], int]:
    total = len(items)
    if size <= 0:
        return items, total
    start = (page - 1) * size
    end = start + size
    return items[start:end], total


def _infer_method(item: Dict[str, Any], bm25_on: bool) -> str:
    """
    Infer which retrieval contributed to a hit based on raw component scores.
    """
    src = item.get("source") or {}
    v = float(src.get("vector") or 0.0)
    b = float(src.get("bm25") or 0.0)
    if bm25_on and v > 0 and b > 0:
        return "hybrid"
    if b > 0 and v == 0:
        return "bm25"
    return "vector"


# ---------------------------
# Route
# ---------------------------

@router.get(
    "",
    response_model=SearchResponse,
    summary="Hybrid search (vector+optional BM25) with filters & pagination",
    dependencies=[Depends(require_scopes(["rag:query"]))],
)
@trace("api.search.hybrid")
def search(
    q: str = Query(..., description="Query text"),
    collection: str = Query("default", description="Vector collection name"),
    top_k: int = Query(12, ge=1, le=100, description="Candidate pool after fusion/rerank/MMR"),
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    use_bm25: bool = Query(True, description="Enable lexical retrieval (BM25)"),
    # kept for back-compat (fusion weight now comes from FUSION_ALPHA env in the retriever)
    w_vec: float = Query(0.7, ge=0.0, le=1.0, description="(Note) Fusion handled in retriever; see FUSION_ALPHA"),
    w_bm25: float = Query(0.3, ge=0.0, le=1.0, description="(Note) Fusion handled in retriever; see FUSION_ALPHA"),
    f_type: Optional[str] = Query(None, alias="filter.type"),
    f_source_contains: Optional[str] = Query(None, alias="filter.source_contains"),
    principal: Principal = Depends(get_principal),
):
    """
    Returns search results with final fused score and simple pagination.
    [CHG] Vector+BM25 fusion, rerank, and MMR are delegated to `HybridSearcher`.
    """
    # [CHG] Resolve tenant-aware, backend-safe physical names once (single source of truth)
    names = resolve_names(principal.tenant_id, collection)

    # Trace basics (include resolved names for debuggability)
    set_span_attr("tenant_id", names.tenant)
    set_span_attr("collection.logical", names.logical)
    set_span_attr("collection.effective", names.effective)
    set_span_attr("collection.vector_name", names.vector_name)
    set_span_attr("collection.bm25_name", names.bm25_name)
    set_span_attr("top_k", top_k)
    set_span_attr("bm25_enabled", use_bm25)

    filters = SearchFilters(type=f_type, source_contains=f_source_contains)

    # [CHG] Delegate to HybridSearcher (handles vector + optional BM25 + rerank + MMR)
    try:
        hits = _get_searcher().search(
            collection=names.vector_name,            # physical vector name (opaque)
            query=q,
            top_k=top_k,
            filters=None,                            # push auth/tenant filters down at storage in prod
            bm25_collection=names.bm25_name,         # physical BM25 name (opaque)
            bm25_enabled=use_bm25,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")

    # Optional route-level metadata filters
    hits = _apply_filters(hits, filters)

    # Shape to API schema + pagination
    shaped: List[SearchHit] = [
        SearchHit(
            score=float(h.get("score", 0.0)),
            method=_infer_method(h, bm25_on=use_bm25),
            metadata=h.get("metadata") or {},
        )
        for h in hits
    ]

    # The searcher already sorts by relevance; we keep that order
    page_items, total = _paginate(shaped, page=page, size=size)

    note = (
        "Hybrid fusion is handled in the retriever stack (see FUSION_ALPHA env). "
        "Disable BM25 with ?use_bm25=false."
        if use_bm25
        else "Vector-only: set ?use_bm25=true to combine with BM25."
    )

    return SearchResponse(
        items=page_items,
        page=SearchPage(page=page, size=size, total=total),
        notes=note,
    )


# ---------------------------------------------------------------------
# [LEGACY REFERENCE] Prior route-level BM25 retrieval + manual fusion
# ---------------------------------------------------------------------
# The block below shows how we previously performed BM25 retrieval and did
# min-max normalization + weighted combine in the route layer. This is kept
# only for reference. Production code should delegate to HybridSearcher to
# avoid drift and ensure unified naming/backends.
#
# Example (DO NOT ENABLE — left commented intentionally):
#
# def _legacy_route_level_fusion_example(q, names, top_k, filters, w_vec, w_bm25):
#     vec_hits = _get_searcher().search(
#         collection=names.vector_name,
#         query=q,
#         top_k=top_k,
#         filters=None,
#         bm25_collection=None,       # vector only
#         bm25_enabled=False,
#     )
#     vec_hits = _apply_filters(vec_hits, filters)
#
#     bm25_hits_raw = _get_bm25().search(
#         collection=names.bm25_name,
#         query=q,
#         top_k=top_k,
#         filters=None,
#     )
#     bm25_hits = [{"score": float(r.get("score", 0.0)), "metadata": r.get("metadata", {})} for r in bm25_hits_raw]
#     bm25_hits = _apply_filters(bm25_hits, filters)
#
#     vec_scores = [float(h.get("score", 0.0)) for h in vec_hits] if vec_hits else []
#     bm_scores  = [float(h.get("score", 0.0)) for h in bm25_hits] if bm25_hits else []
#     vec_norm   = _minmax_norm(vec_scores)
#     bm_norm    = _minmax_norm(bm_scores)
#
#     combined: Dict[str, Dict[str, Any]] = {}
#     for h, s in zip(vec_hits, vec_norm):
#         k = _stable_key(h)
#         combined.setdefault(k, {"metadata": h.get("metadata") or {}, "vec": 0.0, "bm25": 0.0})
#         combined[k]["vec"] = max(combined[k]["vec"], float(s))
#     for h, s in zip(bm25_hits, bm_norm):
#         k = _stable_key(h)
#         combined.setdefault(k, {"metadata": h.get("metadata") or {}, "vec": 0.0, "bm25": 0.0})
#         combined[k]["bm25"] = max(combined[k]["bm25"], float(s))
#
#     wv = float(w_vec)
#     wb = float(w_bm25)
#     results: List[SearchHit] = []
#     for k, v in combined.items():
#         final = wv * v["vec"] + wb * v["bm25"]
#         method = "hybrid" if (v["vec"] > 0 and v["bm25"] > 0) else ("vector" if v["vec"] > 0 else "bm25")
#         results.append(SearchHit(score=float(final), method=method, metadata=v["metadata"]))
#     results.sort(key=lambda x: x.score, reverse=True)
#     return results
