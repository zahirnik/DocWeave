# apps/api/routes/resolve.py
"""
Lightweight KG resolution endpoint.

GET /kg/resolve
  - kind: "entity" | "metric"
  - tenant_id: str
  - query: str

Uses packages.knowledge_graph.resolution.Neo4jResolver.
No hard check on KG_BACKEND — if resolver can't init, returns 503 with reason.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, HTTPException, Query

# Process-local resolver singleton
_RESOLVER = None
_RESOLVER_ERR: Optional[str] = None

# Lazy import & init so the app can still boot without Neo4j
def _ensure_resolver():
    global _RESOLVER, _RESOLVER_ERR
    if _RESOLVER is not None:
        return
    try:
        from packages.knowledge_graph.resolution import Neo4jResolver  # type: ignore
    except Exception as e:  # module/path issue
        _RESOLVER_ERR = f"Resolver unavailable (module import failed): {e}"
        return
    try:
        _RESOLVER = Neo4jResolver.from_env()
    except Exception as e:  # missing env / bad creds
        _RESOLVER_ERR = f"Resolver unavailable (init failed): {e}"

router = APIRouter(prefix="/kg", tags=["kg"])

@router.get("/resolve")
def resolve(
    kind: Literal["entity", "metric"] = Query(..., description='"entity" or "metric"'),
    tenant_id: str = Query(..., description="Tenant/workspace id (e.g., 'public')"),
    query: str = Query(..., description="Name / alias / metric text to resolve"),
) -> Dict[str, Any]:
    _ensure_resolver()
    if _RESOLVER is None:
        raise HTTPException(status_code=503, detail=_RESOLVER_ERR or "Resolver unavailable")

    q = (query or "").strip()
    if not q:
        return {"hits": []}

    if kind == "entity":
        hits = _RESOLVER.resolve_entity(q, tenant_id=tenant_id, limit=5)
        return {
            "hits": [
                {
                    "id": h.id,
                    "name": getattr(h, "name", h.id),
                    "score": float(getattr(h, "score", 0.0)),
                    "method": getattr(h, "method", "n/a"),
                }
                for h in hits
            ]
        }

    # kind == "metric"
    mhits = _RESOLVER.resolve_metric(q, tenant_id=tenant_id)
    return {
        "hits": [
            {
                "id": h.id,
                "key": getattr(h, "key", h.id),
                "score": float(getattr(h, "score", 0.0)),
                "method": getattr(h, "method", "n/a"),
            }
            for h in mhits
        ]
    }
