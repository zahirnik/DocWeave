# apps/api/routes/kg.py
"""
KG API routes
=============

Endpoints
---------
POST /kg/build
    Build (or rebuild) a small KG slice from parsed document chunks.
    - Idempotent per (tenant_id, node.type, node.key) and (tenant_id, edge tuple).
    - Optionally validate snapshot and return a compact report.

GET /kg/subgraph
    Fetch a small, UI-friendly subgraph around an Entity key.
    - Depth-limited BFS with per-node fan-out cap.

GET /kg/resolve
    Resolve entities or metrics using the Neo4j-backed resolver
    (alias → exact → alt → full-text), multi-tenant aware.

Design notes
------------
- This router keeps zero business logic. It wires request models → builders → store.
- Store selection:
    * If env KG_BACKEND is set to "postgres"/"neo4j" we use that first.
    * Else, if `packages.core.config.settings.KG_BACKEND` is present, we use that.
    * Else fallback to in-memory.
- Responses are deterministic and tutorial-style so they double as examples.

Auth
----
This file assumes your global app has auth middleware (OIDC/JWT, API keys).
If you want per-route RBAC, inject dependency stubs here (e.g., require_role).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# Core KG toolbox
from packages.knowledge_graph import (
    # builders & inputs
    DocumentInput,
    ChunkInput,
    build_graph_for_doc,
    # store
    Store,
    InMemoryStore,
    get_store,  # kept for compatibility (not used directly here)
    # validators (used after persist)
    validate_snapshot,  # imported for API surface parity
    # queries
    subgraph_for_entity,
)

# Optional: read backend choice from settings if available
try:  # keep this optional so the router runs in examples/tests without full core
    from packages.core.config import settings  # type: ignore
except Exception:  # pragma: no cover
    class _FallbackSettings:
        KG_BACKEND: str = "memory"   # "memory" | "postgres" | "neo4j"
        POSTGRES_DSN: Optional[str] = None
        NEO4J_URI: Optional[str] = None
        NEO4J_USER: Optional[str] = None
        NEO4J_PASSWORD: Optional[str] = None
    settings = _FallbackSettings()  # type: ignore

# Optional: Neo4j-backed resolver (alias/exact/alt/FTS)
try:
    from packages.knowledge_graph.resolution import Neo4jResolver  # type: ignore
except Exception:  # pragma: no cover
    Neo4jResolver = None  # type: ignore


router = APIRouter(prefix="/kg", tags=["kg"])

# -----------------------
# Dependency: resolve a Store
# -----------------------

_store_singleton: Optional[Store] = None  # process-local; fine for a single worker

def _read_backend() -> str:
    # Prefer env to allow late-binding in Colab/docker
    return (os.environ.get("KG_BACKEND")
            or getattr(settings, "KG_BACKEND", "memory")
            or "memory")

def get_kg_store() -> Store:
    """
    Return a process-local Store instance.

    Priority:
        1) env KG_BACKEND
        2) settings.KG_BACKEND
        3) default "memory"
    """
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    backend = _read_backend()
    try:
        if backend == "postgres":
            # Expect your core/db.py to expose an Engine or a factory.
            from packages.core.db import engine  # type: ignore
            from packages.knowledge_graph.postgres_store import PostgresStore  # lazy import
            _store_singleton = PostgresStore(engine=engine)

        elif backend == "neo4j":
            from packages.knowledge_graph.neo4j_client import Neo4jStore  # lazy import
            # Read env first, then settings
            uri = os.environ.get("NEO4J_URI") or getattr(settings, "NEO4J_URI", None)
            user = os.environ.get("NEO4J_USER") or getattr(settings, "NEO4J_USER", None)
            pwd  = os.environ.get("NEO4J_PASSWORD") or getattr(settings, "NEO4J_PASSWORD", None)
            if not uri or not user or not pwd:
                raise RuntimeError("Neo4j settings missing (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)")
            _store_singleton = Neo4jStore(uri, auth=(user, pwd))

        else:
            _store_singleton = InMemoryStore()

    except Exception as exc:  # pragma: no cover
        _store_singleton = InMemoryStore()
        # In a real app, use your logger; we avoid importing logging plumbing here.
        print(f"[kg] store init failed for backend={backend!r}: {exc}. Falling back to InMemoryStore.")

    return _store_singleton


# -----------------------
# Optional dependency: Neo4j resolver
# -----------------------

_resolver_singleton: Optional["Neo4jResolver"] = None  # type: ignore[name-defined]

def get_kg_resolver() -> "Neo4jResolver":  # type: ignore[name-defined]
    """
    Return a process-local Neo4jResolver. Requires KG_BACKEND=neo4j and Neo4j env.
    """
    global _resolver_singleton
    if Neo4jResolver is None:  # type: ignore[truthy-bool]
        raise HTTPException(status_code=503, detail="Resolver unavailable (module not importable).")

    if _resolver_singleton is not None:
        return _resolver_singleton

    backend = _read_backend()
    if backend != "neo4j":
        raise HTTPException(status_code=400, detail="Resolver requires KG_BACKEND=neo4j.")

    try:
        # Neo4jResolver.from_env reads NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD
        _resolver_singleton = Neo4jResolver.from_env()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to init resolver: {exc}") from exc
    return _resolver_singleton


# -----------------------
# Request/Response models
# -----------------------

class BuildKGChunk(BaseModel):
    text: str = Field(..., description="Parsed text (a chunk/paragraph).")
    page: Optional[int] = Field(None, description="1-based page index (if known).")
    chunk_id: Optional[str] = Field(None, description="Upstream chunk identifier.")

class BuildKGRequest(BaseModel):
    tenant_id: str
    entity_name: str
    entity_namespace: str = Field("org", description="Use 'org' for companies by default.")
    doc_id: str
    chunks: List[BuildKGChunk]
    metric_aliases: Optional[Dict[str, str]] = Field(
        None, description="Optional regex→canonical metric map to augment defaults."
    )
    # Note: pydantic might warn this shadows BaseModel.validate in some versions; harmless.
    validate: bool = Field(True, description="Run snapshot validators on the built slice.")

class BuildKGResponse(BaseModel):
    nodes_created: int
    nodes_updated: int
    edges_created: int
    edges_ignored: int
    validated: bool
    validation_summary: Optional[str] = None
    errors: int = 0
    warnings: int = 0

class SubgraphResponse(BaseModel):
    meta: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class ResolveResponse(BaseModel):
    kind: str = Field(..., description="entity | metric")
    tenant_id: str
    query: str
    items: List[Dict[str, Any]]


# -----------------------
# Label helpers (route-local to avoid stale exporter imports)
# -----------------------

def _pretty_metric_label_from_key(key: str) -> str:
    # "metric:ghg scope 1" → "GHG Scope 1"
    k = (key or "").strip().lower()
    if ":" in k:
        k = k.split(":", 1)[1].strip()
    return " ".join(w.capitalize() for w in k.split()) or (key or "Metric")

def _node_to_json_with_label(n) -> Dict[str, Any]:
    """
    Convert a Node object to a JSON dict with a guaranteed, non-null 'label'.
    Priority:
        explicit n.label →
        props.text (claim) →
        props.name (metric/entity) →
        props.citation (evidence) →
        pretty(key) (metric) →
        RHS of key (entity) →
        key → type title
    """
    props = getattr(n, "props", {}) or {}
    ntype = getattr(n, "type", None)
    ntype_val = getattr(ntype, "value", None) or str(ntype) or ""
    key = getattr(n, "key", "") or ""

    label = getattr(n, "label", None)
    if not label:
        if ntype_val == "claim":
            label = props.get("text")
        elif ntype_val == "metric":
            label = props.get("name") or _pretty_metric_label_from_key(key)
        elif ntype_val == "entity":
            label = props.get("name") or (key.split(":", 1)[-1].strip().title() if key else None)
        elif ntype_val == "evidence":
            label = props.get("citation")

    if not label:
        # final fallbacks
        if ntype_val == "metric":
            label = _pretty_metric_label_from_key(key)
        elif ntype_val == "entity":
            label = (key.split(":", 1)[-1].strip().title() if key else None)

    if not label:
        label = key or ntype_val.title() or "Node"

    return {
        "id": str(getattr(n, "id")),
        "tenant_id": getattr(n, "tenant_id", ""),
        "type": ntype_val,
        "key": key,
        "label": label,
        "props": props,
    }


# -----------------------
# Routes
# -----------------------

@router.post("/build", response_model=BuildKGResponse)
def build_kg(req: BuildKGRequest, store: Store = Depends(get_kg_store)) -> BuildKGResponse:
    """
    Build a KG slice from parsed chunks. Safe to call multiple times:
    nodes are upserted by (tenant_id, type, key); edges are unique by (tenant_id, type, src, dst).
    """
    # 1) Adapt request → DocumentInput
    doc = DocumentInput(
        tenant_id=req.tenant_id,
        entity_name=req.entity_name,
        entity_namespace=req.entity_namespace,
        doc_id=req.doc_id,
        chunks=[ChunkInput(text=c.text, page=c.page, chunk_id=c.chunk_id) for c in req.chunks],
    )

    # 2) Build (idempotent)
    nodes, edges = build_graph_for_doc(doc, metric_aliases=req.metric_aliases)

    # 3) Persist
    res = store.upsert_nodes_edges(nodes, edges)

    # 4) Optional validation from the store snapshot
    validated = False
    summary = None
    errors = warnings = 0
    if req.validate:
        from packages.knowledge_graph.validators import validate_from_store, Severity
        rpt = validate_from_store(store, req.tenant_id)
        validated = True
        summary = rpt.summary()
        errors = rpt.count(Severity.ERROR)
        warnings = rpt.count(Severity.WARNING)

    return BuildKGResponse(
        nodes_created=res.nodes_created,
        nodes_updated=res.nodes_updated,
        edges_created=res.edges_created,
        edges_ignored=res.edges_ignored,
        validated=validated,
        validation_summary=summary,
        errors=errors,
        warnings=warnings,
    )


@router.get("/subgraph", response_model=SubgraphResponse)
def get_subgraph(
    tenant_id: str = Query(..., description="Tenant/workspace id."),
    entity_key: str = Query(..., description="Canonical entity key, e.g., 'org:acme plc'."),
    depth: int = Query(1, ge=0, le=4, description="BFS depth (small for UIs)."),
    max_neighbours: int = Query(25, ge=1, le=200, description="Fan-out cap per node."),
    store: Store = Depends(get_kg_store),
) -> SubgraphResponse:
    """
    Fetch a small subgraph around an entity (depth-limited).
    Returns a stable JSON shape that front-ends can render easily.
    """
    nodes, edges = subgraph_for_entity(store, tenant_id, entity_key, depth=depth, max_neighbours=max_neighbours)

    # Build JSON here to guarantee non-null labels regardless of exporter version.
    nodes_json = [_node_to_json_with_label(n) for n in nodes]

    # ----------------------------------------------------------------------
    # [FIX] Provide edge endpoints under commonly expected field names
    #       so UIs can resolve node names (avoids "? --rel--> ?").
    #       We keep original src_id/dst_id but ALSO add:
    #         - source / target   (primary)
    #         - src / dst         (aliases)
    #         - source_id / target_id (aliases)
    #         - label             (alias for type.value)
    #         - source_idx / target_idx (optional indices into nodes array)
    # ----------------------------------------------------------------------
    # Build an index map for convenience (id → index)
    id_to_idx: Dict[str, int] = {str(n.get("id")): i for i, n in enumerate(nodes_json)}

    edges_json: List[Dict[str, Any]] = []
    for e in edges:
        src_id = str(e.src_id)
        dst_id = str(e.dst_id)
        etype  = getattr(e.type, "value", str(e.type))

        ej = {
            "id": str(e.id),
            "tenant_id": e.tenant_id,
            "type": etype,
            "label": etype,                 # [FIX] friendly alias for relation label
            "src_id": src_id,               # original fields (kept)
            "dst_id": dst_id,               # original fields (kept)
            "source": src_id,               # [FIX] primary field used by UI
            "target": dst_id,               # [FIX] primary field used by UI
            "src": src_id,                  # [FIX] alias
            "dst": dst_id,                  # [FIX] alias
            "source_id": src_id,            # [FIX] alias
            "target_id": dst_id,            # [FIX] alias
            "source_idx": id_to_idx.get(src_id),  # [FIX] optional index into nodes[]
            "target_idx": id_to_idx.get(dst_id),  # [FIX] optional index into nodes[]
            "props": e.props or {},
        }
        edges_json.append(ej)

    payload = {
        "meta": {
            "tenant_id": tenant_id,
            "entity_key": entity_key,
            "depth": depth,
            "node_count": len(nodes_json),
            "edge_count": len(edges_json),
        },
        "nodes": nodes_json,
        "edges": edges_json,
    }
    return SubgraphResponse(**payload)


@router.get("/resolve", response_model=ResolveResponse)
def resolve(
    kind: str = Query("entity", pattern="^(entity|metric)$", description="Resolve type: entity | metric."),
    tenant_id: str = Query(..., description="Tenant/workspace id."),
    query: str = Query(..., description="Name/key to resolve."),
    limit: int = Query(5, ge=1, le=20, description="Max candidates to return (entity only)."),
    min_score: float = Query(0.55, ge=0.0, le=1.0, description="Minimum score threshold."),
    resolver: "Neo4jResolver" = Depends(get_kg_resolver),  # type: ignore[name-defined]
) -> ResolveResponse:
    """
    Resolve entities or metrics using the Neo4j-backed resolver.
    - Entity: alias → exact → alt → full-text (fts_entity_name)
    - Metric: alias → exact key → exact name → fuzzy over known metrics
    """
    if kind == "entity":
        hits = resolver.resolve_entity(query, tenant_id=tenant_id, limit=limit, min_score=min_score)
        items = [{"id": h.id, "name": h.name, "score": float(h.score), "method": h.method} for h in hits]
        return ResolveResponse(kind="entity", tenant_id=tenant_id, query=query, items=items)

    hits_m = resolver.resolve_metric(query, tenant_id=tenant_id, min_score=min_score)
    items_m = [{"id": h.id, "key": h.key, "score": float(h.score), "method": h.method} for h in hits_m]
    return ResolveResponse(kind="metric", tenant_id=tenant_id, query=query, items=items_m)
