# packages/knowledge_graph/queries.py
"""
Knowledge Graph — query helpers (backend-agnostic)
==================================================

Goal
----
Provide tiny, readable query utilities over the Store interface so API/routes
and agents don’t need to know about backend details.

Design
------
- Works with any implementation of `Store` (in-memory, Postgres, Neo4j).
- Avoids adding new store methods by composing `list_nodes`, `list_edges`,
  and `get_node_by_key`.
- Deterministic ordering in all list returns for test stability.

What you can do
---------------
- claims_about_entity(store, tenant_id, entity_key)
- claims_for_metric(store, tenant_id, metric_key)
- evidence_for_claim(store, tenant_id, claim_id)
- claims_with_evidence_about_entity(store, tenant_id, entity_key)
- entity_metric_pairs(store, tenant_id, entity_key)
- subgraph_for_entity(store, tenant_id, entity_key, depth=1, max_neighbours=25)

Examples
--------
>>> from .store import InMemoryStore
>>> from .builders import build_graph_for_doc, DocumentInput, ChunkInput
>>> st = InMemoryStore()
>>> doc = DocumentInput(
...     tenant_id="t0", entity_name="Acme PLC", doc_id="acme_2024.pdf",
...     chunks=[ChunkInput(text="We will reduce Scope 1 emissions by 30% by 2030.", page=12, chunk_id="c-77")]
... )
>>> nodes, edges = build_graph_for_doc(doc)
>>> _ = st.upsert_nodes_edges(nodes, edges)
>>> claims = claims_about_entity(st, "t0", "org:acme plc")
>>> len(claims) > 0
True
>>> ev = evidence_for_claim(st, "t0", claims[0].id)
>>> len(ev) > 0
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from .schema import Edge, EdgeKind, Node, NodeKind
from .store import Store


# -----------------------
# Small DTOs
# -----------------------

@dataclass
class ClaimWithEvidence:
    claim: Node                 # type=CLAIM
    evidences: List[Node]       # type=EVIDENCE (may be empty)


# -----------------------
# Internal helpers
# -----------------------

def _index_nodes(store: Store, tenant_id: str) -> Dict[UUID, Node]:
    """
    Snapshot and index nodes by id for fast joins.
    """
    nodes = store.list_nodes(tenant_id, kind=None, limit=1_000_000)
    return {n.id: n for n in nodes}


def _sorted_nodes(nodes: List[Node]) -> List[Node]:
    nodes.sort(key=lambda n: (n.type.value, n.key))
    return nodes


def _sorted_edges(edges: List[Edge]) -> List[Edge]:
    edges.sort(key=lambda e: (e.type.value, str(e.src_id), str(e.dst_id)))
    return edges


# -----------------------
# Public query helpers
# -----------------------

def claims_about_entity(store: Store, tenant_id: str, entity_key: str, *, limit: int = 10_000) -> List[Node]:
    """
    Return all Claim nodes that have ABOUT → Entity(entity_key).

    Ordering: by (type, key) which means effectively by claim key.
    """
    idx = _index_nodes(store, tenant_id)

    entity = store.get_node_by_key(tenant_id, NodeKind.ENTITY, entity_key)
    if entity is None:
        return []

    # Find ABOUT edges pointing to the entity (dst_id = entity.id)
    about_edges = store.list_edges(tenant_id, kind=EdgeKind.ABOUT, dst_id=entity.id, limit=limit)
    claims: List[Node] = []
    for e in about_edges:
        src = idx.get(e.src_id)
        if src and src.type == NodeKind.CLAIM:
            claims.append(src)

    return _sorted_nodes(list({c.id: c for c in claims}.values()))  # de-dupe by id


def claims_for_metric(store: Store, tenant_id: str, metric_key: str, *, limit: int = 10_000) -> List[Node]:
    """
    Return all Claim nodes that have QUANTIFIES → Metric(metric_key).
    """
    idx = _index_nodes(store, tenant_id)

    metric = store.get_node_by_key(tenant_id, NodeKind.METRIC, metric_key)
    if metric is None:
        return []

    q_edges = store.list_edges(tenant_id, kind=EdgeKind.QUANTIFIES, dst_id=metric.id, limit=limit)
    claims: List[Node] = []
    for e in q_edges:
        src = idx.get(e.src_id)
        if src and src.type == NodeKind.CLAIM:
            claims.append(src)

    return _sorted_nodes(list({c.id: c for c in claims}.values()))


def evidence_for_claim(store: Store, tenant_id: str, claim_id: UUID, *, limit: int = 1000) -> List[Node]:
    """
    Return Evidence nodes linked via SUPPORTED_BY from a given claim id.
    """
    idx = _index_nodes(store, tenant_id)

    edges = store.list_edges(tenant_id, kind=EdgeKind.SUPPORTED_BY, src_id=claim_id, limit=limit)
    evidences: List[Node] = []
    for e in edges:
        dst = idx.get(e.dst_id)
        if dst and dst.type == NodeKind.EVIDENCE:
            evidences.append(dst)

    return _sorted_nodes(list({n.id: n for n in evidences}.values()))


def claims_with_evidence_about_entity(store: Store, tenant_id: str, entity_key: str, *, limit: int = 10_000) -> List[ClaimWithEvidence]:
    """
    Convenience: bundle claims about the entity with their evidence nodes.
    """
    claims = claims_about_entity(store, tenant_id, entity_key, limit=limit)
    out: List[ClaimWithEvidence] = []
    for cl in claims:
        ev = evidence_for_claim(store, tenant_id, cl.id)
        out.append(ClaimWithEvidence(claim=cl, evidences=ev))
    # Stable order by claim key
    out.sort(key=lambda cwe: cwe.claim.key)
    return out


def entity_metric_pairs(store: Store, tenant_id: str, entity_key: str, *, limit: int = 10_000) -> List[Tuple[Node, Node]]:
    """
    Return (Entity, Metric) pairs linked by MEASURED_BY for a given entity.

    Useful to quickly enumerate the metrics a company is associated with.
    """
    idx = _index_nodes(store, tenant_id)

    entity = store.get_node_by_key(tenant_id, NodeKind.ENTITY, entity_key)
    if entity is None:
        return []

    edges = store.list_edges(tenant_id, kind=EdgeKind.MEASURED_BY, src_id=entity.id, limit=limit)
    pairs: List[Tuple[Node, Node]] = []
    for e in edges:
        metric = idx.get(e.dst_id)
        if metric and metric.type == NodeKind.METRIC:
            pairs.append((entity, metric))

    # Sort by metric key for determinism
    pairs.sort(key=lambda p: p[1].key)
    return pairs


def subgraph_for_entity(
    store: Store,
    tenant_id: str,
    entity_key: str,
    *,
    depth: int = 1,
    max_neighbours: int = 25,
):
    """
    Thin wrapper around store.subgraph_from_entity with stable ordering.
    """
    nodes, edges = store.subgraph_from_entity(
        tenant_id, entity_key, depth=depth, max_neighbours=max_neighbours
    )
    return _sorted_nodes(nodes), _sorted_edges(edges)


__all__ = [
    "ClaimWithEvidence",
    "claims_about_entity",
    "claims_for_metric",
    "evidence_for_claim",
    "claims_with_evidence_about_entity",
    "entity_metric_pairs",
    "subgraph_for_entity",
]
