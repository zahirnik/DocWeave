# packages/reporting/scoring.py
"""
Scoring — deterministic Six-Capitals scorecards from a KG slice
===============================================================

What this module does
---------------------
Given:
  • a small KG slice (nodes + edges) for a single tenant,
  • an entity key (e.g., "org:acme plc"),
  • a YAML framework file (six_capitals.yaml),
  • (optional) a sector profiles YAML,

…it produces a reproducible scorecard (no LLMs):
  - per-metric scores in [0,1] via weighted flags
  - per-capital weighted sums
  - overall score via capital weights
  - rationales with evidence links and detector issues

Dependencies: only `pyyaml`. If missing: `pip install pyyaml`.

Design choices
--------------
- Flags come from *detector issues* (packages.reporting.detectors):
    target_present   := NOT MISSING_TARGET
    period_present   := NOT MISSING_PERIOD
    evidence_linked  := NOT NO_EVIDENCE
    specificity_good := NOT VAGUE_LANGUAGE
    metric_aligned   := NOT METRIC_MISMATCH
- Claim selection per metric:
    QUANTIFIES edges Claim→Metric dominate; we also accept claims whose
    props.metric_key == that metric key if edges are absent.
- Confidence screen:
    Ignore claims with confidence < config.min_confidence (from YAML).
- Aggregation:
    For each flag, we take OR over qualifying claims (i.e., "any claim satisfies").
    Metric score = weighted sum of the 5 flags (weights per-metric override defaults).
    Capital score = Σ metric_weight * metric_score   (weights are intra-capital).
    Overall score = Σ capital_weight * capital_score (must ~sum to 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import UUID

# YAML
try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyyaml not installed. Install with: `pip install pyyaml`") from exc

# KG primitives
from packages.knowledge_graph.schema import Node, Edge, NodeKind, EdgeKind

# Detectors
from .detectors import (
    DetectorsConfig,
    DetectorIssue,
    IssueCode,
    detect_all,
)


# -----------------------
# Public DTOs
# -----------------------

@dataclass
class MetricFlags:
    target_present: bool = False
    period_present: bool = False
    evidence_linked: bool = False
    specificity_good: bool = False
    metric_aligned: bool = False


@dataclass
class MetricResult:
    key: str
    label: str
    weight: float
    flags: MetricFlags
    score: float
    considered_claim_ids: List[UUID] = field(default_factory=list)
    issues: List[DetectorIssue] = field(default_factory=list)  # only those related to considered claims


@dataclass
class CapitalResult:
    name: str                 # capital id (e.g., "natural")
    weight: float             # global capital weight
    score: float
    metrics: List[MetricResult] = field(default_factory=list)


@dataclass
class Scorecard:
    entity_key: str
    overall_score: float
    capitals: List[CapitalResult]
    notes: Dict[str, str] = field(default_factory=dict)  # e.g., weight sums, profile applied


# -----------------------
# Internal helpers
# -----------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_entity(nodes: Sequence[Node], entity_key: str) -> Optional[Node]:
    for n in nodes:
        if n.type == NodeKind.ENTITY and n.key.strip().lower() == entity_key.strip().lower():
            return n
    return None


def _claims_about_entity(nodes: Sequence[Node], edges: Sequence[Edge], entity_id: UUID) -> List[Node]:
    claims = [n for n in nodes if n.type == NodeKind.CLAIM]
    out: List[Node] = []
    # Claim --ABOUT--> Entity
    about_map = {e.src_id: True for e in edges if e.type == EdgeKind.ABOUT and e.dst_id == entity_id}
    for c in claims:
        if about_map.get(c.id, False):
            out.append(c)
    return out


def _metric_node_by_key(nodes: Sequence[Node], key: str) -> Optional[Node]:
    key_low = key.strip().lower()
    for n in nodes:
        if n.type == NodeKind.METRIC and n.key.strip().lower() == key_low:
            return n
    return None


def _claims_linked_to_metric(
    claims: Sequence[Node],
    metric_node: Optional[Node],
    metric_key: str,
    edges: Sequence[Edge],
) -> List[Node]:
    # Prefer explicit QUANTIFIES
    out: List[Node] = []
    if metric_node is not None:
        quant = {e.src_id for e in edges if e.type == EdgeKind.QUANTIFIES and e.dst_id == metric_node.id}
        out = [c for c in claims if c.id in quant]
        if out:
            return out
    # Fallback: props.metric_key equals this metric key
    mk_low = metric_key.strip().lower()
    for c in claims:
        if (c.props.get("metric_key") or "").strip().lower() == mk_low:
            out.append(c)
    return out


def _evidence_map(edges: Sequence[Edge]) -> Dict[UUID, List[UUID]]:
    # Claim id -> list of Evidence ids
    m: Dict[UUID, List[UUID]] = {}
    for e in edges:
        if e.type == EdgeKind.SUPPORTED_BY:
            m.setdefault(e.src_id, []).append(e.dst_id)
    return m


def _apply_sector_overrides(
    cfg: dict,
    profiles: Optional[dict],
    entity: Optional[Node],
) -> Tuple[dict, Optional[str]]:
    """
    Apply sector overrides if enabled in six_capitals.yaml and a matching profile
    exists in sector_profiles.yaml. Returns (possibly modified cfg, profile_name_used).
    """
    if not cfg or not isinstance(cfg, dict):
        return cfg, None

    so = cfg.get("sector_overrides") or {}
    if not so.get("enabled", False):
        return cfg, None

    key_field = (so.get("profile_key_field") or "sector").strip()
    if not entity or key_field not in (entity.props or {}):
        return cfg, None

    profile_name = str(entity.props.get(key_field)).strip().lower()
    if not profiles or "profiles" not in profiles:
        return cfg, None

    profile = (profiles["profiles"] or {}).get(profile_name)
    if not profile:
        return cfg, None

    # Apply metric weight overrides per capital
    apply = profile.get("apply") or {}
    caps = cfg.get("capitals") or {}
    for cap_name, cap_over in apply.items():
        cap = caps.get(cap_name)
        if not cap:
            continue
        metrics_cfg = cap.get("metrics") or []
        # Build quick index by metric key
        idx = {m.get("key"): m for m in metrics_cfg}
        over_m = ((cap_over.get("metrics") or {}) if isinstance(cap_over, dict) else {})
        for m_key, m_patch in over_m.items():
            if m_key in idx and isinstance(m_patch, dict):
                # Only allow weight override here; detector weight tweaks would be in 'detector_overrides'
                if "weight" in m_patch:
                    idx[m_key]["weight"] = float(m_patch["weight"])
        # Reassign to ensure in-place mutate is kept
        caps[cap_name]["metrics"] = list(idx.values())

    # Detector overrides (e.g., vague_terms) — we merge into detector_thresholds
    det_over = profile.get("detector_overrides") or {}
    if det_over:
        dt = cfg.setdefault("detector_thresholds", {})
        # Merge arrays (e.g., vague_terms)
        for k, v in det_over.items():
            if isinstance(v, list):
                base = set((dt.get(k) or []))
                base.update(v)
                dt[k] = sorted(base)
            else:
                dt[k] = v

    return cfg, profile_name


# -----------------------
# Main API
# -----------------------

def score_entity(
    *,
    tenant_id: str,
    entity_key: str,
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    framework_yaml_path: str,
    sector_profiles_yaml_path: Optional[str] = None,
) -> Scorecard:
    """
    Build a deterministic scorecard for `entity_key` using the provided KG slice.

    Returns
    -------
    Scorecard dataclass with per-metric, per-capital and overall scores.
    """
    cfg = _load_yaml(framework_yaml_path)
    profiles = _load_yaml(sector_profiles_yaml_path) if sector_profiles_yaml_path else None

    # Possibly apply sector overrides
    entity = _find_entity(nodes, entity_key)
    cfg, profile_used = _apply_sector_overrides(cfg, profiles, entity)

    min_conf = float(((cfg.get("info") or {}).get("min_confidence")) or 0.0)

    # Run detectors once on the full slice
    det_cfg = DetectorsConfig(
        min_confidence=min_conf,
        min_evidence_per_claim=int(((cfg.get("detector_thresholds") or {}).get("min_evidence_per_claim")) or 1),
        vague_terms=((cfg.get("detector_thresholds") or {}).get("vague_terms")) or DetectorsConfig().vague_terms,
    )
    issues = detect_all(nodes, edges, tenant_id=tenant_id, config=det_cfg)

    # Index issues by claim id
    issues_by_claim: Dict[UUID, List[DetectorIssue]] = {}
    for it in issues:
        issues_by_claim.setdefault(it.claim_id, []).append(it)

    # Identify claims ABOUT the entity (and keep only those meeting confidence)
    if not entity:
        # No such entity; return empty but well-formed scorecard
        return Scorecard(entity_key=entity_key, overall_score=0.0, capitals=[], notes={"warning": "entity not found"})

    claims = [c for c in _claims_about_entity(nodes, edges, entity.id) if float(c.props.get("confidence", 0.0)) >= min_conf]

    # Evidence map for quick checks
    ev_map = _evidence_map(edges)

    # Defaults
    det_w_default = (cfg.get("detector_weights_default") or {})
    caps_cfg: Dict[str, dict] = cfg.get("capitals") or {}
    cap_weights: Dict[str, float] = cfg.get("capital_weights") or {}

    capital_results: List[CapitalResult] = []
    overall_score = 0.0

    for cap_name, cap_cfg in caps_cfg.items():
        metrics_cfg: List[dict] = cap_cfg.get("metrics") or []
        cap_weight = float(cap_weights.get(cap_name, 0.0))

        metric_results: List[MetricResult] = []
        cap_score = 0.0

        for m in metrics_cfg:
            m_key = str(m.get("key"))
            m_label = str(m.get("label", m_key))
            m_weight = float(m.get("weight", 0.0))
            m_det_w = dict(det_w_default)
            m_det_w.update(m.get("detector_weights") or {})

            # Resolve metric node & qualifying claims
            metric_node = _metric_node_by_key(nodes, m_key)
            m_claims = _claims_linked_to_metric(claims, metric_node, m_key, edges)

            # Aggregate flags across qualifying claims
            flags = MetricFlags()
            considered_ids: List[UUID] = []
            related_issues: List[DetectorIssue] = []

            for c in m_claims:
                considered_ids.append(c.id)
                c_issues = issues_by_claim.get(c.id, [])
                related_issues.extend(c_issues)

                # invert detectors → flags
                codes = {iss.code for iss in c_issues}
                # target/period
                flags.target_present   = flags.target_present   or (IssueCode.MISSING_TARGET not in codes)
                flags.period_present   = flags.period_present   or (IssueCode.MISSING_PERIOD not in codes)
                # evidence: also double-check actual links
                has_ev = len(ev_map.get(c.id, [])) >= det_cfg.min_evidence_per_claim
                flags.evidence_linked  = flags.evidence_linked  or (IssueCode.NO_EVIDENCE not in codes and has_ev)
                # specificity
                flags.specificity_good = flags.specificity_good or (IssueCode.VAGUE_LANGUAGE not in codes)
                # alignment (edge or prop)
                flags.metric_aligned   = flags.metric_aligned   or (IssueCode.METRIC_MISMATCH not in codes)

            # Compute metric score (weighted sum of booleans)
            s = (
                float(m_det_w.get("target_present", 0.0))   * (1.0 if flags.target_present   else 0.0) +
                float(m_det_w.get("period_present", 0.0))   * (1.0 if flags.period_present   else 0.0) +
                float(m_det_w.get("evidence_linked", 0.0))  * (1.0 if flags.evidence_linked  else 0.0) +
                float(m_det_w.get("specificity_good", 0.0)) * (1.0 if flags.specificity_good else 0.0) +
                float(m_det_w.get("metric_aligned", 0.0))   * (1.0 if flags.metric_aligned   else 0.0)
            )
            # clamp to [0,1]
            s = max(0.0, min(1.0, s))

            metric_results.append(
                MetricResult(
                    key=m_key,
                    label=m_label,
                    weight=m_weight,
                    flags=flags,
                    score=s,
                    considered_claim_ids=considered_ids,
                    issues=related_issues,
                )
            )

            cap_score += m_weight * s

        capital_results.append(
            CapitalResult(
                name=cap_name,
                weight=cap_weight,
                score=cap_score,
                metrics=metric_results,
            )
        )
        overall_score += cap_weight * cap_score

    # Normalisation & notes
    # (We do not hard-normalise weights; we record the sums for auditors.)
    notes = {
        "capital_weights_sum": f"{sum(cap_weights.values()):.3f}",
        "profile_used": profile_used or "",
        "min_confidence": f"{min_conf:.2f}",
    }

    return Scorecard(
        entity_key=entity_key,
        overall_score=max(0.0, min(1.0, overall_score)),
        capitals=capital_results,
        notes=notes,
    )


__all__ = ["MetricFlags", "MetricResult", "CapitalResult", "Scorecard", "score_entity"]
