# examples/09_reporting_demo.py
"""
Reporting demo — deterministic Six-Capitals scorecard + peer benchmark
=====================================================================

What this script does (no LLMs):
1) Builds a tiny Knowledge Graph slice in memory:
   - Entity → Claims → (Metrics) with Evidence provenance.
2) Persists into the default InMemoryStore (Colab/AWS friendly).
3) Runs deterministic scoring using YAML frameworks:
   packages/reporting/frameworks/six_capitals.yaml
   packages/reporting/frameworks/sector_profiles.yaml  (optional)
4) Writes results to ./data/outputs/reporting_demo/ as JSON/CSV.
5) Computes a small peer benchmark across two entities.

Run:
    uv run python examples/09_reporting_demo.py
or:
    python examples/09_reporting_demo.py

Ensure the YAMLs exist at:
    packages/reporting/frameworks/six_capitals.yaml
    packages/reporting/frameworks/sector_profiles.yaml  (optional)
"""

from __future__ import annotations

import os
from pathlib import Path

# --- KG primitives & store ---
from packages.knowledge_graph.builders import DocumentInput, ChunkInput, build_graph_for_doc
from packages.knowledge_graph.store import InMemoryStore
from packages.knowledge_graph.queries import subgraph_for_entity

# --- Reporting: scoring/benchmark + exports ---
from packages.reporting.scoring import score_entity
from packages.reporting.narrative_benchmark import rank_peers
from packages.reporting.exports import write_scorecard_json, write_scorecard_csv, write_benchmark_csv


OUT_DIR = Path("data/outputs/reporting_demo").resolve()
FRAMEWORK = "packages/reporting/frameworks/six_capitals.yaml"
SECTORS   = "packages/reporting/frameworks/sector_profiles.yaml"  # optional; ignored if missing


def _seed_store(store: InMemoryStore):
    """
    Create two tiny docs for two entities, demonstrating:
      - quantified climate claim (target+period+metric)
      - a vague claim (to trigger detectors)
    """
    # --- Entity A: Acme PLC ---
    doc_a = DocumentInput(
        tenant_id="t0",
        entity_name="Acme PLC",
        entity_namespace="org",
        doc_id="acme_2024.pdf",
        chunks=[
            ChunkInput(
                text="Acme will reduce Scope 1 emissions by 30% by 2030 versus a 2019 baseline.",
                page=12, chunk_id="a-c1"
            ),
            ChunkInput(
                text="We aim to improve energy management in coming years.",
                page=13, chunk_id="a-c2"
            ),
        ],
    )
    nodes_a, edges_a = build_graph_for_doc(doc_a)
    store.upsert_nodes_edges(nodes_a, edges_a)

    # --- Entity B: Globex ---
    doc_b = DocumentInput(
        tenant_id="t0",
        entity_name="Globex",
        entity_namespace="org",
        doc_id="globex_2024.pdf",
        chunks=[
            ChunkInput(
                text="Globex targets a 40% reduction in Scope 2 emissions by 2030.",
                page=7, chunk_id="b-c1"
            ),
            ChunkInput(
                text="Net zero ambition is reiterated; renewable electricity share will be increased.",
                page=8, chunk_id="b-c2"
            ),
        ],
    )
    nodes_b, edges_b = build_graph_for_doc(doc_b)
    store.upsert_nodes_edges(nodes_b, edges_b)


def _safe_path(p: str) -> str | None:
    return p if (p and os.path.exists(p)) else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store = InMemoryStore()

    # 1) Seed in-memory KG
    _seed_store(store)

    # 2) Score a single entity (Acme PLC)
    entity_key = "org:acme plc"
    nodes, edges = subgraph_for_entity(store, tenant_id="t0", entity_key=entity_key, depth=2)
    if not nodes or not edges:
        # Fallback: full tenant slice (scorer will filter by entity_key)
        nodes = store.list_nodes("t0", kind=None, limit=100_000)
        edges = store.list_edges("t0", kind=None, limit=200_000)

    sc = score_entity(
        tenant_id="t0",
        entity_key=entity_key,
        nodes=nodes,
        edges=edges,
        framework_yaml_path=FRAMEWORK,
        sector_profiles_yaml_path=_safe_path(SECTORS),
    )

    # 3) Export scorecard
    write_scorecard_json(sc, OUT_DIR / "acme_scorecard.json")
    write_scorecard_csv(sc, OUT_DIR / "acme_scorecard_csv")

    print(f"[OK] Scorecard for {entity_key!r}")
    print(f"     Overall: {sc.overall_score:.3f}")
    for cap in sc.capitals:
        print(f"     {cap.name:<22} score={cap.score:.3f} weight={cap.weight:.2f} metrics={len(cap.metrics)}")

    # 4) Small peer benchmark (Acme vs Globex)
    #    Build per-entity subgraphs; if empty, fall back to full slice.
    peers = ["org:acme plc", "org:globex"]
    corpus = {}
    for ek in peers:
        ns, es = subgraph_for_entity(store, tenant_id="t0", entity_key=ek, depth=2)
        if not ns or not es:
            ns = store.list_nodes("t0", kind=None, limit=100_000)
            es = store.list_edges("t0", kind=None, limit=200_000)
        corpus[ek] = (ns, es)

    bench = rank_peers(
        focus_entity="org:acme plc",
        corpus=corpus,
        framework_yaml_path=FRAMEWORK,
        sector_profiles_yaml_path=_safe_path(SECTORS),
    )

    # 5) Export benchmark CSV
    write_benchmark_csv(bench, OUT_DIR / "benchmark_csv")

    print("[OK] Benchmark peers (overall rank ↓):")
    for p in bench.peers:
        print(f"     {p.entity_key:<20} score={p.scorecard.overall_score:.3f} rank={p.rank_overall} coverage={p.coverage:.2f}")

    print(f"\nArtifacts written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
