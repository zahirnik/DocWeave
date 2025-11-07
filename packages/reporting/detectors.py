# packages/reporting/detectors.py
"""
Detectors — rule-based checks over the Knowledge Graph
======================================================

What this module does
---------------------
Given a small KG slice (nodes + edges) for one tenant, run deterministic
checks over CLAIM nodes and return issues (missing target/period/evidence,
low confidence, vague language, and metric alignment problems).

Inputs we rely on (from packages.knowledge_graph.schema):
- Node.props["text"]         : original claim sentence (string)
- Node.props["target"]       : optional target string (e.g., "30%")
- Node.props["period"]       : optional period string (e.g., "by 2030")
- Node.props["confidence"]   : float in [0,1]
- Node.props["metric_key"]   : canonical metric key (e.g., "metric:ghg scope 1")

Edges we rely on:
- EdgeKind.SUPPORTED_BY      : Claim → Evidence
- EdgeKind.QUANTIFIES        : Claim → Metric

Typical use
-----------
>>> from packages.knowledge_graph import to_json
>>> cfg = DetectorsConfig(min_confidence=0.5, min_evidence_per_claim=1,
...                       vague_terms=["aim","intend","explore"])
>>> issues = detect_all(nodes, edges, config=cfg)
>>> # issues is a list of DetectorIssue to feed into scoring or a report

Design notes
------------
- No external dependencies (pure Python).
- Works entirely on the in-memory structures (no Store required at runtime).
- Severity enum aligns with validators.Severity, with a safe fallback if the
  validators module isn't available for some reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID
import re

# --- Knowledge Graph primitives ---
from packages.knowledge_graph.schema import Node, Edge, NodeKind, EdgeKind

# Align with validators.Severity if present; else define a local fallback
try:  # pragma: no cover
    from packages.knowledge_graph.validators import Severity  # type: ignore
except Exception:  # pragma: no cover
    class Severity(str, Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"


# -----------------------
# Public types
# -----------------------

class IssueCode(str, Enum):
    MISSING_TARGET = "missing_target"
    MISSING_PERIOD = "missing_period"
    LOW_CONFIDENCE = "low_confidence"
    NO_EVIDENCE = "no_evidence"
    VAGUE_LANGUAGE = "vague_language"
    METRIC_MISMATCH = "metric_mismatch"  # metric_key present but no QUANTIFIES edge (or vice versa)


@dataclass
class DetectorIssue:
    tenant_id: str
    claim_id: UUID
    code: IssueCode
    severity: Severity
    message: str
    evidence_ids: List[UUID] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class DetectorsConfig:
    min_confidence: float = 0.50
    min_evidence_per_claim: int = 1
    vague_terms: Sequence[str] = field(default_factory=lambda: ["aim", "intend", "explore", "consider", "where possible", "as appropriate"])


# -----------------------
# Helper context (indexes)
# -----------------------

@dataclass
class _KGContext:
    tenant_id: str
    nodes_by_id: Dict[UUID, Node]
    claims: List[Node]
    metrics_by_key: Dict[str, Node]
    evidence_by_id: Dict[UUID, Node]
    # adjacency
    out_by_src: Dict[UUID, List[Edge]]


def _build_context(tenant_id: str, nodes: Sequence[Node], edges: Sequence[Edge]) -> _KGContext:
    nodes_by_id = {n.id: n for n in nodes}
    claims = [n for n in nodes if n.type == NodeKind.CLAIM]
    metrics_by_key = {n.key: n for n in nodes if n.type == NodeKind.METRIC}
    evidence_by_id = {n.id: n for n in nodes if n.type == NodeKind.EVIDENCE}
    out_by_src: Dict[UUID, List[Edge]] = {}
    for e in edges:
        out_by_src.setdefault(e.src_id, []).append(e)
    return _KGContext(
        tenant_id=tenant_id,
        nodes_by_id=nodes_by_id,
        claims=claims,
        metrics_by_key=metrics_by_key,
        evidence_by_id=evidence_by_id,
        out_by_src=out_by_src,
    )


# -----------------------
# Detectors
# -----------------------

def detect_missing_target(ctx: _KGContext) -> List[DetectorIssue]:
    out: List[DetectorIssue] = []
    for c in ctx.claims:
        if not (c.props.get("target") or "").strip():
            out.append(
                DetectorIssue(
                    tenant_id=ctx.tenant_id,
                    claim_id=c.id,
                    code=IssueCode.MISSING_TARGET,
                    severity=Severity.WARNING,
                    message="Claim lacks a target (e.g., '30%' or 'net zero').",
                )
            )
    return out


def detect_missing_period(ctx: _KGContext) -> List[DetectorIssue]:
    out: List[DetectorIssue] = []
    for c in ctx.claims:
        if not (c.props.get("period") or "").strip():
            out.append(
                DetectorIssue(
                    tenant_id=ctx.tenant_id,
                    claim_id=c.id,
                    code=IssueCode.MISSING_PERIOD,
                    severity=Severity.WARNING,
                    message="Claim lacks a time bound (e.g., 'by 2030').",
                )
            )
    return out


def detect_low_confidence(ctx: _KGContext, min_confidence: float) -> List[DetectorIssue]:
    out: List[DetectorIssue] = []
    for c in ctx.claims:
        conf = float(c.props.get("confidence", 0.0))
        if conf < float(min_confidence):
            out.append(
                DetectorIssue(
                    tenant_id=ctx.tenant_id,
                    claim_id=c.id,
                    code=IssueCode.LOW_CONFIDENCE,
                    severity=Severity.INFO,
                    message=f"Claim confidence {conf:.2f} < min {min_confidence:.2f}.",
                    meta={"confidence": f"{conf:.2f}"},
                )
            )
    return out


def detect_no_evidence(ctx: _KGContext, min_evidence: int) -> List[DetectorIssue]:
    out: List[DetectorIssue] = []
    for c in ctx.claims:
        ev_ids = [
            e.dst_id for e in ctx.out_by_src.get(c.id, [])
            if e.type == EdgeKind.SUPPORTED_BY and e.dst_id in ctx.evidence_by_id
        ]
        if len(ev_ids) < int(min_evidence):
            out.append(
                DetectorIssue(
                    tenant_id=ctx.tenant_id,
                    claim_id=c.id,
                    code=IssueCode.NO_EVIDENCE,
                    severity=Severity.ERROR if min_evidence > 0 else Severity.WARNING,
                    message=f"Claim has {len(ev_ids)} evidence link(s); requires ≥ {min_evidence}.",
                    evidence_ids=ev_ids,
                    meta={"have": str(len(ev_ids)), "need": str(min_evidence)},
                )
            )
    return out


def detect_vague_language(ctx: _KGContext, vague_terms: Sequence[str]) -> List[DetectorIssue]:
    out: List[DetectorIssue] = []
    if not vague_terms:
        return out
    # Build a single regex like r"\b(aim|intend|explore)\b"
    escaped = [re.escape(t.strip().lower()) for t in vague_terms if t.strip()]
    if not escaped:
        return out
    pat = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
    for c in ctx.claims:
        text = (c.props.get("text") or "").lower()
        if text and pat.search(text):
            out.append(
                DetectorIssue(
                    tenant_id=ctx.tenant_id,
                    claim_id=c.id,
                    code=IssueCode.VAGUE_LANGUAGE,
                    severity=Severity.WARNING,
                    message="Claim uses vague wording (e.g., 'aim', 'intend', 'explore').",
                )
            )
    return out


def detect_metric_mismatch(ctx: _KGContext) -> List[DetectorIssue]:
    """
    Flag claims where:
      - props.metric_key is set but there is NO QUANTIFIES edge to that metric, or
      - there is a QUANTIFIES edge to a metric but props.metric_key is empty.
    """
    out: List[DetectorIssue] = []
    for c in ctx.claims:
        prop_key = (c.props.get("metric_key") or "").strip().lower()
        edge_keys = []
        for e in ctx.out_by_src.get(c.id, []):
            if e.type == EdgeKind.QUANTIFIES:
                m = ctx.nodes_by_id.get(e.dst_id)
                if m and m.type == NodeKind.METRIC:
                    edge_keys.append(m.key.strip().lower())

        # Cases:
        if prop_key and prop_key not in edge_keys:
            out.append(
                DetectorIssue(
                    tenant_id=ctx.tenant_id,
                    claim_id=c.id,
                    code=IssueCode.METRIC_MISMATCH,
                    severity=Severity.WARNING,
                    message=f"metric_key={prop_key!r} but no QUANTIFIES edge to that metric.",
                    meta={"metric_key": prop_key, "edge_metrics": ",".join(edge_keys)},
                )
            )
        elif not prop_key and edge_keys:
            out.append(
                DetectorIssue(
                    tenant_id=ctx.tenant_id,
                    claim_id=c.id,
                    code=IssueCode.METRIC_MISMATCH,
                    severity=Severity.WARNING,
                    message="QUANTIFIES edge exists but claim.props.metric_key is empty.",
                    meta={"edge_metrics": ",".join(edge_keys)},
                )
            )
    return out


# -----------------------
# Orchestrator
# -----------------------

def detect_all(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    *,
    tenant_id: Optional[str] = None,
    config: Optional[DetectorsConfig] = None,
) -> List[DetectorIssue]:
    """
    Run all detectors and return a flat list of issues.

    Parameters
    ----------
    nodes, edges : KG slice
    tenant_id    : if omitted, we try to infer from the first node/edge
    config       : DetectorsConfig with thresholds and vague terms
    """
    if config is None:
        config = DetectorsConfig()

    # Best-effort tenant inference
    t_id = tenant_id or (nodes[0].tenant_id if nodes else (edges[0].tenant_id if edges else "t0"))

    ctx = _build_context(t_id, nodes, edges)

    issues: List[DetectorIssue] = []
    issues.extend(detect_missing_target(ctx))
    issues.extend(detect_missing_period(ctx))
    issues.extend(detect_low_confidence(ctx, config.min_confidence))
    issues.extend(detect_no_evidence(ctx, config.min_evidence_per_claim))
    issues.extend(detect_vague_language(ctx, config.vague_terms))
    issues.extend(detect_metric_mismatch(ctx))
    return issues


__all__ = [
    "IssueCode",
    "DetectorIssue",
    "DetectorsConfig",
    "detect_all",
    "detect_missing_target",
    "detect_missing_period",
    "detect_low_confidence",
    "detect_no_evidence",
    "detect_vague_language",
    "detect_metric_mismatch",
]
