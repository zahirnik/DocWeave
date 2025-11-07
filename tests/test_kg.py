# tests/test_kg.py
"""
End-to-end smoke tests for the Knowledge Graph package.

Covers:
- builders.build_graph_for_doc  → nodes/edges created with provenance
- store.InMemoryStore           → idempotent upserts & basic queries
- validators.validate_snapshot  → no errors on happy path
- queries.*                     → claims/evidence & entity-metric pairs
- exporters.to_json/graphml     → stable, parseable payloads
"""

from __future__ import annotations

import json
from uuid import UUID

from packages.knowledge_graph import (
    # schema helpers
    NodeKind,
    EdgeKind,
    # builder inputs & build
    ChunkInput,
    DocumentInput,
    build_graph_for_doc,
    # store
    InMemoryStore,
    # validators
    validate_snapshot,
    Severity,
    # queries
    claims_about_entity,
    evidence_for_claim,
    entity_metric_pairs,
    subgraph_for_entity,
    # exporters
    to_json,
    to_graphml,
)


def _sample_doc():
    return DocumentInput(
        tenant_id="t0",
        entity_name="Acme PLC",
        entity_namespace="org",
        doc_id="acme_2024_report.pdf",
        chunks=[
            ChunkInput(
                text="Acme will reduce Scope 1 emissions by 30% by 2030 versus 2019.",
                page=12,
                chunk_id="c-77",
            ),
            ChunkInput(
                text="Revenues grew in 2024. Net zero ambition is reiterated.",
                page=13,
                chunk_id="c-88",
            ),
        ],
    )


def test_build_and_upsert():
    doc = _sample_doc()

    # Build
    nodes, edges = build_graph_for_doc(doc)
    assert len(nodes) > 0 and len(edges) > 0

    # Basic shape
    kinds = sorted({n.type for n in nodes}, key=lambda k: k.value)
    assert set(k.value for k in kinds) >= {"claim", "entity", "evidence", "metric"}

    # Persist (idempotent)
    store = InMemoryStore()
    res1 = store.upsert_nodes_edges(nodes, edges)
    res2 = store.upsert_nodes_edges(nodes, edges)  # repeat exact same batch

    assert res1.nodes_created > 0
    assert res2.nodes_created == 0  # idempotent
    assert res2.edges_created == 0

    # Validate snapshot from store (happy path → no errors)
    rpt = validate_snapshot(
        store.list_nodes("t0", kind=None, limit=1000),
        store.list_edges("t0", kind=None, limit=1000),
        require_provenance=True,
        min_confidence=0.5,
    )
    assert rpt.count(Severity.ERROR) == 0


def test_queries_and_subgraph():
    store = InMemoryStore()
    nodes, edges = build_graph_for_doc(_sample_doc())
    store.upsert_nodes_edges(nodes, edges)

    # Claims about entity
    claims = claims_about_entity(store, "t0", "org:acme plc")
    assert len(claims) >= 1
    assert all(c.type == NodeKind.CLAIM for c in claims)

    # Evidence for first claim
    ev = evidence_for_claim(store, "t0", claims[0].id)
    assert len(ev) >= 1
    assert all(e.type == NodeKind.EVIDENCE for e in ev)

    # Entity ↔ Metric pairs
    pairs = entity_metric_pairs(store, "t0", "org:acme plc")
    assert len(pairs) >= 1
    ent, met = pairs[0]
    assert ent.type == NodeKind.ENTITY
    assert met.type == NodeKind.METRIC

    # Subgraph (depth=1)
    sub_nodes, sub_edges = subgraph_for_entity(store, "t0", "org:acme plc", depth=1)
    assert any(n.type == NodeKind.ENTITY for n in sub_nodes)
    assert any(e.type in {EdgeKind.MEASURED_BY, EdgeKind.ABOUT, EdgeKind.SUPPORTED_BY} for e in sub_edges)


def test_exporters_roundtrip():
    store = InMemoryStore()
    doc = _sample_doc()
    nodes, edges = build_graph_for_doc(doc)
    store.upsert_nodes_edges(nodes, edges)

    ns = store.list_nodes(doc.tenant_id, kind=None, limit=1000)
    es = store.list_edges(doc.tenant_id, kind=None, limit=1000)

    # JSON export
    payload = to_json(ns, es, tenant_id=doc.tenant_id, meta={"entity_key": "org:acme plc"})
    assert payload["meta"]["tenant_id"] == doc.tenant_id
    assert payload["meta"]["node_count"] == len(ns)
    # sanity: IDs are UUIDs as strings
    _ = UUID(payload["nodes"][0]["id"])

    # GraphML export (bytes, XML header present)
    xml_bytes = to_graphml(ns, es, graph_id="kg_t0_acme", directed=True, pretty=False)
    assert xml_bytes.startswith(b'<?xml version="1.0" encoding="utf-8"?>')
