# apps/api/routes/reports.py
"""
Reporting routes — deterministic scorecards & peer benchmarks (no LLM)
=====================================================================

Endpoints
---------
POST /reports/scorecard
    Body: ScoreRequest
    → Runs rule-based detectors + scoring over a small KG subgraph for the entity.

POST /reports/benchmark
    Body: BenchmarkRequest
    → Builds scorecards for a set of entities and returns overall/per-capital rankings.

Notes
-----
- Works with the default in-memory KG store (no Postgres needed).
- Uses YAML frameworks under packages/reporting/frameworks/.
- If a YAML file is missing or invalid, returns 400 with a clear message.
"""

from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException

from apps.api.schemas.reports import (
    ScoreRequest,
    ScoreResponse,
    CapitalView,
    MetricView,
    MetricFlags,
    BenchmarkRequest,
    BenchmarkResponse,
    PeerView,
)

# KG store & queries
from packages.knowledge_graph.store import InMemoryStore
try:
    # optional convenience factory (if your store module exposes it)
    from packages.knowledge_graph.store import get_store  # type: ignore
except Exception:  # pragma: no cover
    get_store = None  # type: ignore

from packages.knowledge_graph.queries import subgraph_for_entity

# Deterministic scoring/benchmark (no LLMs)
from packages.reporting.scoring import score_entity
from packages.reporting.narrative_benchmark import rank_peers


router = APIRouter(prefix="/reports", tags=["reports"])


# -----------------------
# Helpers
# -----------------------

def _store():
    """
    Resolve a KG store. Defaults to in-memory, which is perfect for Colab/AWS demos/tests.
    To switch, export:  KG_BACKEND=postgres|neo4j|memory
    """
    backend = os.getenv("KG_BACKEND", "memory").strip().lower()
    if get_store is None:
        return InMemoryStore()
    try:
        return get_store(backend)  # type: ignore
    except Exception:
        # Safe fallback
        return InMemoryStore()


def _to_score_response(sd) -> ScoreResponse:
    """
    Convert scoring.Scorecard (dataclass) → ScoreResponse (pydantic).
    Kept explicit so OpenAPI shows clean shapes.
    """
    caps: list[CapitalView] = []
    for cap in sd.capitals:
        mets: list[MetricView] = []
        for m in cap.metrics:
            mets.append(
                MetricView(
                    key=m.key,
                    label=m.label,
                    weight=m.weight,
                    score=m.score,
                    flags=MetricFlags(
                        target_present=m.flags.target_present,
                        period_present=m.flags.period_present,
                        evidence_linked=m.flags.evidence_linked,
                        specificity_good=m.flags.specificity_good,
                        metric_aligned=m.flags.metric_aligned,
                    ),
                    considered_claim_ids=[str(x) for x in (m.considered_claim_ids or [])],
                )
            )
        caps.append(CapitalView(name=cap.name, weight=cap.weight, score=cap.score, metrics=mets))
    return ScoreResponse(entity_key=sd.entity_key, overall_score=sd.overall_score, capitals=caps, notes=dict(sd.notes or {}))


def _yaml_guard(path: str):
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"Framework file not found: {path}")


# -----------------------
# Routes
# -----------------------

@router.post("/scorecard", response_model=ScoreResponse)
def scorecard(req: ScoreRequest) -> ScoreResponse:
    """
    Build a deterministic Six-Capitals scorecard for one entity.
    """
    # Validate YAML paths early for clean 400s
    _yaml_guard(req.framework_yaml_path)
    if req.sector_profiles_yaml_path:
        if not os.path.exists(req.sector_profiles_yaml_path):
            # Non-fatal: allow empty profiles path if client forgot to ship it
            req = req.copy(update={"sector_profiles_yaml_path": None})

    # Pull a compact subgraph around the entity (depth=2 is enough for claims/metrics/evidence)
    store = _store()
    try:
        nodes, edges = subgraph_for_entity(store, req.tenant_id, req.entity_key, depth=2)
        # If subgraph is empty, fallback to full tenant slice as a safety net
        if not nodes or not edges:
            nodes = store.list_nodes(req.tenant_id, kind=None, limit=100_000)
            edges = store.list_edges(req.tenant_id, kind=None, limit=200_000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"KG query failed: {exc}")

    # Score deterministically (no LLM)
    try:
        sd = score_entity(
            tenant_id=req.tenant_id,
            entity_key=req.entity_key,
            nodes=nodes,
            edges=edges,
            framework_yaml_path=req.framework_yaml_path,
            sector_profiles_yaml_path=req.sector_profiles_yaml_path,
        )
    except RuntimeError as exc:
        # Typically YAML parse/missing dependency errors
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

    return _to_score_response(sd)


@router.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    """
    Build peer benchmark: overall rank + per-capital ranks and coverage.
    """
    # Validate framework YAML
    _yaml_guard(req.framework_yaml_path)
    if req.sector_profiles_yaml_path and not os.path.exists(req.sector_profiles_yaml_path):
        req = req.copy(update={"sector_profiles_yaml_path": None})

    # Assemble per-entity subgraphs (depth=2)
    store = _store()
    corpus = {}
    try:
        for ek in req.entity_keys:
            ns, es = subgraph_for_entity(store, req.tenant_id, ek, depth=2)
            if not ns or not es:
                # fallback: full tenant slice (entity filter will be applied in scoring)
                ns = store.list_nodes(req.tenant_id, kind=None, limit=100_000)
                es = store.list_edges(req.tenant_id, kind=None, limit=200_000)
            corpus[ek] = (ns, es)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"KG query failed: {exc}")

    # Rank peers
    try:
        bench = rank_peers(
            focus_entity=req.focus_entity,
            corpus=corpus,
            framework_yaml_path=req.framework_yaml_path,
            sector_profiles_yaml_path=req.sector_profiles_yaml_path,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}")

    # Convert to API shape
    def _peer_view(p) -> PeerView:
        scores_by_cap = {c.name: c.score for c in p.scorecard.capitals}
        return PeerView(
            entity_key=p.entity_key,
            overall_score=p.scorecard.overall_score,
            rank_overall=p.rank_overall,
            coverage=p.coverage,
            metrics_total=p.metric_count_total,
            metrics_with_claims=p.metric_count_with_claims,
            scores_by_capital=scores_by_cap,
            ranks_by_capital=dict(p.ranks_by_capital or {}),
        )

    focus_v = _peer_view(bench.focus)
    peers_v = [_peer_view(pp) for pp in bench.peers]

    return BenchmarkResponse(focus=focus_v, peers=peers_v, notes=dict(bench.notes or {}))
