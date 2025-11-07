# scripts/build_kg.py
"""
CLI — Build a Knowledge Graph slice from parsed chunks
======================================================

What this does
--------------
- Reads a small JSON describing a document's parsed chunks for an entity.
- Uses the KG builders to create nodes/edges.
- Upserts into the selected store (default: InMemory).
- Optionally validates and exports a subgraph (JSON/GraphML).

Why this file exists
--------------------
- Handy for smoke testing the KG end-to-end without spinning up the API.
- Good for CI and demos on Colab (no DB required).

Input JSON shape
----------------
{
  "tenant_id": "t0",
  "entity_name": "Acme PLC",
  "entity_namespace": "org",
  "doc_id": "acme_2024_report.pdf",
  "chunks": [
    {"text": "We will reduce Scope 1 emissions by 30% by 2030.", "page": 12, "chunk_id": "c-77"}
  ],
  "metric_aliases": { "\\bscope\\s*1\\b": "GHG Scope 1" }
}

Usage
-----
$ python scripts/build_kg.py run ./data/samples/acme_chunks.json \
    --depth 1 --out-json ./data/outputs/kg_acme.json --out-graphml ./data/outputs/kg_acme.graphml

Dependencies
------------
- typer (optional; if not installed, `pip install typer[all]`)
- The knowledge_graph package (pure-Python)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from packages.knowledge_graph import (
    # inputs & build
    ChunkInput,
    DocumentInput,
    build_graph_for_doc,
    # store
    get_store,
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

app = typer.Typer(help="Build a small Knowledge Graph slice from parsed chunks.")


def _load_input(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["tenant_id", "entity_name", "doc_id", "chunks"]
    missing = [k for k in required if k not in data]
    if missing:
        raise typer.BadParameter(f"Missing required fields in input JSON: {missing}")
    return data


@app.command()
def run(
    input_json: Path = typer.Argument(..., help="Path to input JSON (see shape in module docstring)."),
    backend: str = typer.Option("memory", "--backend", "-b", help="KG backend: memory | postgres | neo4j"),
    depth: int = typer.Option(1, "--depth", help="Subgraph BFS depth for export (UI-friendly)."),
    out_json: Optional[Path] = typer.Option(None, "--out-json", help="Write subgraph JSON payload here."),
    out_graphml: Optional[Path] = typer.Option(None, "--out-graphml", help="Write subgraph GraphML here."),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Run validators after upsert."),
):
    """
    Build from chunks, upsert to store, optional validate + export.
    """
    data = _load_input(input_json)

    # 1) Document input
    doc = DocumentInput(
        tenant_id=data["tenant_id"],
        entity_name=data["entity_name"],
        entity_namespace=data.get("entity_namespace", "org"),
        doc_id=data["doc_id"],
        chunks=[ChunkInput(**c) for c in data["chunks"]],
    )
    metric_aliases = data.get("metric_aliases")

    # 2) Build nodes/edges
    nodes, edges = build_graph_for_doc(doc, metric_aliases=metric_aliases)
    typer.echo(f"[build] nodes={len(nodes)} edges={len(edges)}")

    # 3) Store (defaults to in-memory; adapters optional)
    try:
        store = get_store(backend)
    except Exception as exc:
        # Safest fallback for Colab/AWS quick tests
        typer.echo(f"[warn] backend={backend!r} not available ({exc}); using InMemoryStore.")
        store = InMemoryStore()

    res = store.upsert_nodes_edges(nodes, edges)
    typer.echo(
        f"[store] nodes_created={res.nodes_created} nodes_updated={res.nodes_updated} "
        f"edges_created={res.edges_created} edges_ignored={res.edges_ignored}"
    )

    # 4) Validate snapshot
    if validate:
        rpt = validate_snapshot(
            store.list_nodes(doc.tenant_id, kind=None, limit=100_000),
            store.list_edges(doc.tenant_id, kind=None, limit=200_000),
            require_provenance=True,
            min_confidence=0.50,
        )
        typer.echo(f"[validate] {rpt.summary()}")
        if rpt.count(Severity.ERROR) > 0:
            typer.echo("[validate] Errors found — inspect before using this graph.")

    # 5) Optional exports (small subgraph around the entity)
    if out_json or out_graphml:
        sub_nodes, sub_edges = subgraph_for_entity(
            store, doc.tenant_id, entity_key=f"{doc.entity_namespace}:{doc.entity_name.lower()}", depth=depth
        )
        if out_json:
            payload = to_json(sub_nodes, sub_edges, tenant_id=doc.tenant_id, meta={"entity_key": f"{doc.entity_namespace}:{doc.entity_name.lower()}", "depth": depth})
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            typer.echo(f"[export] wrote JSON → {out_json}")
        if out_graphml:
            xml_bytes = to_graphml(sub_nodes, sub_edges, graph_id=f"kg_{doc.tenant_id}_{doc.entity_name.lower()}", directed=True, pretty=False)
            out_graphml.parent.mkdir(parents=True, exist_ok=True)
            out_graphml.write_bytes(xml_bytes)
            typer.echo(f"[export] wrote GraphML → {out_graphml}")


if __name__ == "__main__":
    app()
