# tests/test_reporting.py
"""
Deterministic reporting tests (Six-Capitals)
===========================================

Covers:
- Scoring: score_entity over a tiny in-memory KG slice
- Exports: JSON + CSV writers
- Benchmark: rank_peers across two entities

Notes
-----
- Uses the in-memory KG store (no Postgres/Neo4j).
- Skips if the framework YAML is missing.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# --- KG primitives & store ---
from packages.knowledge_graph.builders import DocumentInput, ChunkInput, build_graph_for_doc
from packages.knowledge_graph.store import InMemoryStore
from packages.knowledge_graph.queries import subgraph_for_entity

# --- Reporting ---
from packages.reporting.scoring import score_entity
from packages.reporting.narrative_benchmark import rank_peers
from packages.reporting.exports import (
    scorecard_to_json_dict,
    write_scorecard_json,
    write_scorecard_csv,
    write_benchmark_csv,
)

FRAMEWORK = "packages/reporting/frameworks/six_capitals.yaml"
SECTORS   = "packages/reporting/frameworks/sector_profiles.yaml"  # optional


def _require_framework():
    if not os.path.exists(FRAMEWORK):
        pytest.skip(f"Framework YAML missing at {FRAMEWORK}")


def _seed_store(store: InMemoryStore) -> None:
    """Create two tiny docs for two entities."""
    # Entity: Acme PLC — quantified Scope 1 claim (target + period)
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

    # Entity: Globex — Scope 2 claim
    doc_b = DocumentInput(
        tenant_id="t0",
        entity_name="Globex",
        entity_namespace="org",
        doc_id="globex_2024.pdf",
        chunks=[
            ChunkInput(
                text="Globex targets a 40% reduction in Scope 2 emissions by 2030.",
                page=7, chunk_id="b-c1"
            )
        ],
    )
    nodes_b, edges_b = build_graph_for_doc(doc_b)
    store.upsert_nodes_edges(nodes_b, edges_b)


def test_scoring_and_exports():
    _require_framework()
    store = InMemoryStore()
    _seed_store(store)

    entity_key = "org:acme plc"
    nodes, edges = subgraph_for_entity(store, tenant_id="t0", entity_key=entity_key, depth=2)
    if not nodes or not edges:
        # Fallback to full tenant slice
        nodes = store.list_nodes("t0", kind=None, limit=100_000)
        edges = store.list_edges("t0", kind=None, limit=200_000)

    sc = score_entity(
        tenant_id="t0",
        entity_key=entity_key,
        nodes=nodes,
        edges=edges,
        framework_yaml_path=FRAMEWORK,
        sector_profiles_yaml_path=SECTORS if os.path.exists(SECTORS) else None,
    )

    # Sanity checks
    assert 0.0 <= sc.overall_score <= 1.0
    assert len(sc.capitals) > 0
    # Ensure at least one metric considered
    assert any(len(m.considered_claim_ids) > 0 for c in sc.capitals for m in c.metrics)

    # Exports
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_scorecard_json(sc, td / "scorecard.json")
        write_scorecard_csv(sc, td / "sc_csv")

        # files exist
        assert (td / "scorecard.json").exists()
        assert (td / "sc_csv" / "capitals.csv").exists()
        assert (td / "sc_csv" / "metrics.csv").exists()

        # JSON dict shape sanity
        payload = scorecard_to_json_dict(sc)
        assert payload["entity_key"] == entity_key
        assert isinstance(payload["capitals"], list)


def test_peer_benchmark():
    _require_framework()
    store = InMemoryStore()
    _seed_store(store)

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
        sector_profiles_yaml_path=SECTORS if os.path.exists(SECTORS) else None,
    )

    assert len(bench.peers) == 2
    # Ranks are 1..N
    assert 1 <= bench.focus.rank_overall <= len(bench.peers)

    # Export CSV
    with tempfile.TemporaryDirectory() as td:
        write_benchmark_csv(bench, td)
        assert Path(td, "peers.csv").exists()
