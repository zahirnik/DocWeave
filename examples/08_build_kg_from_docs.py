# examples/08_build_kg_from_docs.py
"""
Example 08 — Build a tiny Knowledge Graph from parsed chunks
============================================================

What this does
--------------
1) Defines a toy document for an entity ("Acme PLC") with two text chunks.
2) Uses the KG builders to extract claims + provenance → Nodes & Edges.
3) Upserts into the in-memory store (idempotent).
4) Validates the snapshot (no errors expected on happy path).
5) Exports a small subgraph as JSON and GraphML for inspection.

How to run
---------
$ python examples/08_build_kg_from_docs.py
# outputs written to: data/outputs/kg_acme.json, data/outputs/kg_acme.graphml

Dependencies
------------
- Only relies on the knowledge_graph package (no DB required).
"""

from __future__ import annotations

import json
from pathlib import Path

from packages.knowledge_graph import (
    # inputs & build
    ChunkInput,
    DocumentInput,
    build_graph_for_doc,
    # store
    InMemoryStore,
    # validation
    validate_snapshot,
    Severity,
    # exporters
    to_json,
    to_graphml,
    # queries
    subgraph_for_entity,
)

OUT_DIR = Path("data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # 1) Define a toy document (what your ingest/parse stage would produce)
    doc = DocumentInput(
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

    # 2) Build nodes/edges (idempotent within this run)
    nodes, edges = build_graph_for_doc(doc)
    print(f"[build] nodes={len(nodes)} edges={len(edges)}")

    # 3) Upsert into a store (in-memory for this example)
    store = InMemoryStore()
    res = store.upsert_nodes_edges(nodes, edges)
    print(
        f"[store] nodes_created={res.nodes_created} nodes_updated={res.nodes_updated} "
        f"edges_created={res.edges_created} edges_ignored={res.edges_ignored}"
    )

    # 4) Validate snapshot
    rpt = validate_snapshot(
        store.list_nodes(doc.tenant_id, kind=None, limit=1000),
        store.list_edges(doc.tenant_id, kind=None, limit=2000),
        require_provenance=True,
        min_confidence=0.50,
    )
    print(f"[validate] {rpt.summary()}")
    if rpt.count(Severity.ERROR) > 0:
        print("[validate] Errors found — inspect and fix before using this graph.")

    # 5) Get a small subgraph around the entity (depth=1 is great for UIs)
    sub_nodes, sub_edges = subgraph_for_entity(store, doc.tenant_id, "org:acme plc", depth=1)
    payload = to_json(sub_nodes, sub_edges, tenant_id=doc.tenant_id, meta={"entity_key": "org:acme plc", "depth": 1})
    xml_bytes = to_graphml(sub_nodes, sub_edges, graph_id="kg_t0_acme", directed=True, pretty=False)

    # 6) Write exports
    json_path = OUT_DIR / "kg_acme.json"
    graphml_path = OUT_DIR / "kg_acme.graphml"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    graphml_path.write_bytes(xml_bytes)
    print(f"[export] wrote {json_path}")
    print(f"[export] wrote {graphml_path}")


if __name__ == "__main__":
    main()
