# packages/reporting/narrative_benchmark.py
"""
Narrative benchmarking over Six-Capitals scorecards
===================================================

What this module does
---------------------
Given a small corpus of entities → (nodes, edges), it:
  • computes deterministic scorecards (via scoring.score_entity),
  • derives simple coverage stats (how many metrics had at least one qualifying claim),
  • ranks peers by overall score and per-capital scores,
  • returns a compact, auditable benchmark summary.

No LLMs. Dependencies: pyyaml (already required by scoring).

Typical use
-----------
>>> from packages.reporting.narrative_benchmark import rank_peers
>>> corpus = {
...   "org:acme plc": (nodes_acme, edges_acme),
...   "org:globex":   (nodes_globex, edges_globex),
... }
>>> bench = rank_peers(
...   focus_entity="org:acme plc",
...   corpus=corpus,
...   framework_yaml_path="packages/reporting/frameworks/six_capitals.yaml",
...   sector_profiles_yaml_path="packages/reporting/frameworks/sector_profiles.yaml",
... )
>>> bench.focus.rank_overall, bench.focus.coverage

Design choices
--------------
- We reuse scoring.score_entity to avoid duplicating flag logic.
- Coverage is defined as: (# metrics with ≥1 considered claim) / (# metrics in framework for the capitals listed).
- Results are plain dataclasses → easy to JSON-serialise if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from packages.knowledge_graph.schema import Node, Edge
from .scoring import Scorecard, score_entity


# -----------------------
# Public DTOs
# -----------------------

@dataclass
class PeerSlice:
    entity_key: str
    scorecard: Scorecard
    coverage: float                      # 0..1
    metric_count_total: int
    metric_count_with_claims: int
    rank_overall: int                    # 1 = best
    ranks_by_capital: Dict[str, int]     # per-capital rank (1 = best)


@dataclass
class BenchmarkResult:
    focus: PeerSlice
    peers: List[PeerSlice]               # includes focus as well; sorted by overall score desc
    notes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "focus": asdict(self.focus),
            "peers": [asdict(p) for p in self.peers],
            "notes": dict(self.notes),
        }


# -----------------------
# Helpers
# -----------------------

def _coverage_from_scorecard(sc: Scorecard) -> Tuple[int, int, float]:
    total_metrics = 0
    with_claims = 0
    for cap in sc.capitals:
        for m in cap.metrics:
            total_metrics += 1
            if len(m.considered_claim_ids) > 0:
                with_claims += 1
    cov = (with_claims / total_metrics) if total_metrics else 0.0
    return total_metrics, with_claims, cov


def _rank(values_desc: List[Tuple[str, float]]) -> Dict[str, int]:
    """
    Assign ranks (1 = best). Stable for ties (same score → same rank, next rank is offset by #items with better score).
    Input: list of (entity_key, score) sorted desc by score.
    """
    ranks: Dict[str, int] = {}
    last_score: Optional[float] = None
    last_rank = 0
    i = 0
    while i < len(values_desc):
        key, val = values_desc[i]
        if last_score is None or val < last_score:
            last_rank = i + 1
            last_score = val
        ranks[key] = last_rank
        i += 1
    return ranks


# -----------------------
# Public API
# -----------------------

def rank_peers(
    *,
    focus_entity: str,
    corpus: Mapping[str, Tuple[Sequence[Node], Sequence[Edge]]],
    framework_yaml_path: str,
    sector_profiles_yaml_path: Optional[str] = None,
) -> BenchmarkResult:
    """
    Build scorecards for all entities in `corpus`, compute simple coverage,
    and produce overall/per-capital rankings.

    Parameters
    ----------
    focus_entity : canonical key, e.g., "org:acme plc"
    corpus       : dict[entity_key] -> (nodes, edges)
    framework_yaml_path : path to six_capitals.yaml
    sector_profiles_yaml_path : optional path to sector profiles yaml

    Returns
    -------
    BenchmarkResult
    """
    # 1) Score everyone deterministically
    scored: Dict[str, Scorecard] = {}
    for ek, (nodes, edges) in corpus.items():
        # tenants: assume already filtered slice per-entity; we don't need tenant id for scoring logic here
        # use the tenant from the first node/edge if available, else "t0"
        tenant_id = (nodes[0].tenant_id if nodes else (edges[0].tenant_id if edges else "t0"))
        scored[ek] = score_entity(
            tenant_id=tenant_id,
            entity_key=ek,
            nodes=nodes,
            edges=edges,
            framework_yaml_path=framework_yaml_path,
            sector_profiles_yaml_path=sector_profiles_yaml_path,
        )

    # 2) Coverage stats
    coverage_meta: Dict[str, Tuple[int, int, float]] = {}
    for ek, sc in scored.items():
        coverage_meta[ek] = _coverage_from_scorecard(sc)

    # 3) Rankings (overall and per-capital)
    overall_sorted = sorted(((ek, sc.overall_score) for ek, sc in scored.items()), key=lambda x: x[1], reverse=True)
    ranks_overall = _rank(overall_sorted)

    # Per-capital ranks: build map capital -> list[(entity, score)]
    cap_names = {cap.name for sc in scored.values() for cap in sc.capitals}
    ranks_by_capital: Dict[str, Dict[str, int]] = {}
    for cap in cap_names:
        arr: List[Tuple[str, float]] = []
        for ek, sc in scored.items():
            # find capital
            val = 0.0
            for c in sc.capitals:
                if c.name == cap:
                    val = c.score
                    break
            arr.append((ek, val))
        arr.sort(key=lambda x: x[1], reverse=True)
        ranks_by_capital[cap] = _rank(arr)

    # 4) Assemble PeerSlice list
    peers: List[PeerSlice] = []
    for ek, sc in scored.items():
        total_m, have_m, cov = coverage_meta[ek]
        peers.append(
            PeerSlice(
                entity_key=ek,
                scorecard=sc,
                coverage=cov,
                metric_count_total=total_m,
                metric_count_with_claims=have_m,
                rank_overall=ranks_overall.get(ek, 0),
                ranks_by_capital={cap: ranks_by_capital.get(cap, {}).get(ek, 0) for cap in cap_names},
            )
        )

    # sort by overall desc
    peers.sort(key=lambda p: p.scorecard.overall_score, reverse=True)

    # 5) Focus entity slice
    focus = next((p for p in peers if p.entity_key.strip().lower() == focus_entity.strip().lower()), None)
    if focus is None:
        # Create an empty placeholder if focus not in corpus
        focus = PeerSlice(
            entity_key=focus_entity,
            scorecard=Scorecard(entity_key=focus_entity, overall_score=0.0, capitals=[], notes={"warning": "focus entity not in corpus"}),
            coverage=0.0,
            metric_count_total=0,
            metric_count_with_claims=0,
            rank_overall=0,
            ranks_by_capital={},
        )

    notes = {
        "entities": str(len(peers)),
        "framework": framework_yaml_path,
        "sector_profiles": sector_profiles_yaml_path or "",
    }

    return BenchmarkResult(focus=focus, peers=peers, notes=notes)


__all__ = ["PeerSlice", "BenchmarkResult", "rank_peers"]
