# packages/knowledge_graph/validators.py
"""
Knowledge Graph — validators (snapshot consistency checks)
==========================================================

Goal
----
Provide tiny, deterministic validations you can run on a *batch* of nodes/edges
(before upsert) or on a snapshot fetched from a store. Keep the rules simple
and explicit so issues are clear to developers and reviewers.

What we check (snapshot-level)
------------------------------
1) Duplicate node keys per (tenant_id, type, key)              → ERROR
2) Duplicate edges per (tenant_id, type, src_id, dst_id)       → WARNING
3) Orphan edges (src/dst not present in node list)             → ERROR
4) Type sanity for edges:
   - ABOUT:         Claim → Entity                              ERROR
   - SUPPORTED_BY:  Claim → Evidence                            ERROR
   - QUANTIFIES:    Claim → Metric                              ERROR
   - MEASURED_BY:   Entity → Metric                             ERROR
5) Claim payload sanity:
   - missing `entity_key`                                      → ERROR
   - missing provenance (doc_id/page/chunk_id)                 → WARNING (configurable)
   - low confidence (< min_confidence)                         → WARNING
   - ABOUT edge entity.key must match Claim.props.entity_key   → WARNING
6) Evidence payload sanity:
   - missing `doc_id`                                          → ERROR

Return shape
------------
A ValidationReport with:
- severity counts
- issues list (code, severity, message, refs)
- helper `ok` bool and `summary()` string

Usage
-----
>>> report = validate_snapshot(nodes, edges, require_provenance=True, min_confidence=0.5)
>>> if not report.ok:
...     print(report.summary())
...     for issue in report.issues: print(issue.code, issue.message)

You can also build a snapshot from a store and validate:
>>> from .store import Store
>>> def validate_from_store(store: Store, tenant_id: str, limit=10000):
...     nodes = store.list_nodes(tenant_id, kind=None, limit=limit)
...     edges = store.list_edges(tenant_id, kind=None, limit=limit)
...     return validate_snapshot(nodes, edges)

Notes
-----
- Keep this module dependency-light (stdlib + pydantic from schema models).
- All checks are best-effort; we avoid raising exceptions and aggregate issues instead.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from .schema import Edge, EdgeKind, Node, NodeKind


# -----------------------
# Issue types & severities
# -----------------------

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueCode(str, Enum):
    DUPLICATE_NODE_KEY = "duplicate_node_key"
    DUPLICATE_EDGE_TUPLE = "duplicate_edge_tuple"
    ORPHAN_EDGE = "orphan_edge"
    EDGE_TYPE_MISMATCH = "edge_type_mismatch"
    CLAIM_MISSING_ENTITY_KEY = "claim_missing_entity_key"
    CLAIM_MISSING_PROVENANCE = "claim_missing_provenance"
    CLAIM_LOW_CONFIDENCE = "claim_low_confidence"
    CLAIM_ENTITY_KEY_MISMATCH = "claim_entity_key_mismatch"
    EVIDENCE_MISSING_DOC_ID = "evidence_missing_doc_id"


class ValidationIssue(BaseModel):
    code: IssueCode
    severity: Severity
    message: str
    refs: Dict[str, str] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    issues: List[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(i.severity != Severity.ERROR for i in self.issues)

    def count(self, severity: Severity) -> int:
        return sum(1 for i in self.issues if i.severity == severity)

    def summary(self) -> str:
        e = self.count(Severity.ERROR)
        w = self.count(Severity.WARNING)
        i = self.count(Severity.INFO)
        return f"Validation — errors={e}, warnings={w}, info={i}"


# -----------------------
# Public API
# -----------------------

def validate_snapshot(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    *,
    require_provenance: bool = True,
    min_confidence: float = 0.50,
) -> ValidationReport:
    """
    Validate a (nodes, edges) snapshot. Returns a report with aggregated issues.
    """
    rpt = ValidationReport()
    if not nodes:
        return rpt  # nothing to validate

    # Build indices
    by_id: Dict[UUID, Node] = {n.id: n for n in nodes}
    key_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)  # (tenant, kind, key)
    for n in nodes:
        key_counts[(n.tenant_id, n.type.value, n.key)] += 1

    # 1) Duplicate node keys
    for (tenant, kind, key), cnt in key_counts.items():
        if cnt > 1:
            rpt.issues.append(
                ValidationIssue(
                    code=IssueCode.DUPLICATE_NODE_KEY,
                    severity=Severity.ERROR,
                    message=f"Duplicate node key for ({tenant}, {kind}, {key})",
                    refs={"tenant": tenant, "kind": kind, "key": key},
                )
            )

    # 2) Duplicate edges (by (tenant, type, src, dst)) and 3) Orphans
    edge_seen: Dict[Tuple[str, str, UUID, UUID], int] = defaultdict(int)
    for e in edges:
        key = (e.tenant_id, e.type.value, e.src_id, e.dst_id)
        edge_seen[key] += 1

        # Orphan?
        if e.src_id not in by_id or e.dst_id not in by_id:
            rpt.issues.append(
                ValidationIssue(
                    code=IssueCode.ORPHAN_EDGE,
                    severity=Severity.ERROR,
                    message="Edge endpoints not found in snapshot nodes",
                    refs={
                        "tenant": e.tenant_id,
                        "edge_type": e.type.value,
                        "src_id": str(e.src_id),
                        "dst_id": str(e.dst_id),
                    },
                )
            )

    for (tenant, kind, src, dst), cnt in edge_seen.items():
        if cnt > 1:
            rpt.issues.append(
                ValidationIssue(
                    code=IssueCode.DUPLICATE_EDGE_TUPLE,
                    severity=Severity.WARNING,
                    message=f"Duplicate edge tuple for ({tenant}, {kind}, {src} → {dst})",
                    refs={"tenant": tenant, "edge_type": kind, "src_id": str(src), "dst_id": str(dst)},
                )
            )

    # 4) Edge type sanity
    for e in edges:
        src = by_id.get(e.src_id)
        dst = by_id.get(e.dst_id)
        if not src or not dst:
            # already covered as ORPHAN_EDGE
            continue

        ok = True
        if e.type == EdgeKind.ABOUT:
            ok = (src.type == NodeKind.CLAIM and dst.type == NodeKind.ENTITY)
        elif e.type == EdgeKind.SUPPORTED_BY:
            ok = (src.type == NodeKind.CLAIM and dst.type == NodeKind.EVIDENCE)
        elif e.type == EdgeKind.QUANTIFIES:
            ok = (src.type == NodeKind.CLAIM and dst.type == NodeKind.METRIC)
        elif e.type == EdgeKind.MEASURED_BY:
            ok = (src.type == NodeKind.ENTITY and dst.type == NodeKind.METRIC)

        if not ok:
            rpt.issues.append(
                ValidationIssue(
                    code=IssueCode.EDGE_TYPE_MISMATCH,
                    severity=Severity.ERROR,
                    message=f"Edge type {e.type.value} has incompatible endpoint types ({src.type.value} → {dst.type.value})",
                    refs={"edge_type": e.type.value, "src_type": src.type.value, "dst_type": dst.type.value},
                )
            )

    # 5) Claim payload sanity (scan nodes + their edges)
    # Build quick lookups for edges by src to reason about claim links
    edges_by_src: Dict[UUID, List[Edge]] = defaultdict(list)
    for e in edges:
        edges_by_src[e.src_id].append(e)

    for n in nodes:
        if n.type == NodeKind.CLAIM:
            props = n.props or {}
            entity_key = str(props.get("entity_key") or "")
            text = str(props.get("text") or "")
            doc_id = props.get("doc_id")
            page = props.get("page")
            chunk_id = props.get("chunk_id")
            confidence = float(props.get("confidence") or 0.0)

            if not entity_key:
                rpt.issues.append(
                    ValidationIssue(
                        code=IssueCode.CLAIM_MISSING_ENTITY_KEY,
                        severity=Severity.ERROR,
                        message="Claim.props.entity_key is missing",
                        refs={"claim_id": str(n.id), "claim_key": n.key, "text": text[:96]},
                    )
                )

            if require_provenance and (not doc_id or page is None or not chunk_id):
                rpt.issues.append(
                    ValidationIssue(
                        code=IssueCode.CLAIM_MISSING_PROVENANCE,
                        severity=Severity.WARNING,
                        message="Claim is missing full provenance (doc_id/page/chunk_id)",
                        refs={"claim_id": str(n.id), "doc_id": str(doc_id), "page": str(page), "chunk_id": str(chunk_id)},
                    )
                )

            if confidence < min_confidence:
                rpt.issues.append(
                    ValidationIssue(
                        code=IssueCode.CLAIM_LOW_CONFIDENCE,
                        severity=Severity.WARNING,
                        message=f"Claim confidence {confidence:.2f} below threshold {min_confidence:.2f}",
                        refs={"claim_id": str(n.id), "claim_key": n.key},
                    )
                )

            # ABOUT edge must exist and target Entity's key should match entity_key
            about_targets: List[Node] = []
            for e in edges_by_src.get(n.id, []):
                if e.type == EdgeKind.ABOUT:
                    tgt = by_id.get(e.dst_id)
                    if tgt:
                        about_targets.append(tgt)

            if not about_targets:
                rpt.issues.append(
                    ValidationIssue(
                        code=IssueCode.EDGE_TYPE_MISMATCH,
                        severity=Severity.ERROR,
                        message="Claim lacks ABOUT edge to an Entity",
                        refs={"claim_id": str(n.id), "claim_key": n.key},
                    )
                )
            else:
                # If entity_key present, ensure one ABOUT target matches it
                if entity_key and not any(t.key == entity_key for t in about_targets):
                    rpt.issues.append(
                        ValidationIssue(
                            code=IssueCode.CLAIM_ENTITY_KEY_MISMATCH,
                            severity=Severity.WARNING,
                            message="Claim.props.entity_key does not match any ABOUT(Entity) target",
                            refs={"claim_id": str(n.id), "entity_key": entity_key},
                        )
                    )

            # SUPPORTED_BY edge must exist
            has_evidence = any(
                e.type == EdgeKind.SUPPORTED_BY and by_id.get(e.dst_id) and by_id[e.dst_id].type == NodeKind.EVIDENCE
                for e in edges_by_src.get(n.id, [])
            )
            if not has_evidence:
                rpt.issues.append(
                    ValidationIssue(
                        code=IssueCode.EDGE_TYPE_MISMATCH,
                        severity=Severity.ERROR,
                        message="Claim lacks SUPPORTED_BY edge to an Evidence node",
                        refs={"claim_id": str(n.id), "claim_key": n.key},
                    )
                )

        elif n.type == NodeKind.EVIDENCE:
            props = n.props or {}
            if not props.get("doc_id"):
                rpt.issues.append(
                    ValidationIssue(
                        code=IssueCode.EVIDENCE_MISSING_DOC_ID,
                        severity=Severity.ERROR,
                        message="Evidence.props.doc_id is required",
                        refs={"evidence_id": str(n.id), "evidence_key": n.key},
                    )
                )

    return rpt


# -----------------------
# Convenience helper (store snapshot)
# -----------------------

def validate_from_store(store, tenant_id: str, *, limit_nodes: int = 100_000, limit_edges: int = 200_000) -> ValidationReport:
    """
    Pull a snapshot from a Store-like object (requires list_nodes/list_edges API)
    and validate it. Use this in admin scripts or tests.
    """
    try:
        nodes = store.list_nodes(tenant_id, kind=None, limit=limit_nodes)
        edges = store.list_edges(tenant_id, kind=None, limit=limit_edges)
    except Exception as exc:  # pragma: no cover
        rpt = ValidationReport()
        rpt.issues.append(
            ValidationIssue(
                code=IssueCode.DUPLICATE_NODE_KEY,  # generic placeholder
                severity=Severity.ERROR,
                message=f"Failed to fetch snapshot from store: {exc}",
                refs={"tenant": tenant_id},
            )
        )
        return rpt

    return validate_snapshot(nodes, edges)
