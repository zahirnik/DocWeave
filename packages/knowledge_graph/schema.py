# packages/knowledge_graph/schema.py
"""
Knowledge Graph — schema (nodes, edges, and payload contracts)
==============================================================

This module defines the *portable* data contracts for our KG layer.
It is backend-agnostic (works whether you store nodes/edges in Postgres
tables or a native graph DB like Neo4j). Keep it small and boring.

ASCII map
---------
    [Entity] --(measured_by)--> [Metric]
        |                            ^
        | supports                   |
   (supported_by)                    |
        v                            |
     [Claim] ----------------(quantifies)------+
        |                                       \
        +--(evidence)--> [Evidence(doc_id#p:chunk)]

Design choices
--------------
- We use a single "Node" type with an explicit NodeKind (ENTITY, METRIC, CLAIM, EVIDENCE, DOCUMENT)
  for storage simplicity. "Payload" models encode the typed fields we actually care about and are
  placed into `Node.props` as JSON, so we keep strict structure without schema drift.
- Edges are directional and typed via EdgeKind. Keep a small enum; add new kinds sparingly.
- Provenance is first-class: Claim/Evidence payloads carry doc_id, page, chunk_id so every answer
  is traceable.

Dependencies
------------
- Pydantic (v2): for validation and clear error messages.
- Standard library only otherwise.

Examples
--------
>>> # Create an Entity node (company)
>>> ent = EntityPayload(namespace="org", name="Acme PLC", identifiers={"LEI": "5493001KJTIIGC8Y1R12"})
>>> node_entity = ent.to_node(tenant_id="t0")
>>> node_entity.type.name
'ENTITY'

>>> # Create a Metric node (Scope 1 emissions)
>>> met = MetricPayload(name="GHG Scope 1", unit="tCO2e", standard="GHG Protocol", dimension="scope1")
>>> node_metric = met.to_node(tenant_id="t0")

>>> # Create a Claim with evidence linkage (doc/page/chunk)
>>> claim = ClaimPayload(
...     text="Acme targets a 30% reduction in Scope 1 by 2030 vs 2019 baseline.",
...     entity_key=node_entity.key, metric_key=node_metric.key,
...     period="2019–2030", target="30%", confidence=0.86,
...     doc_id="acme_2024_report.pdf", page=12, chunk_id="c-77",
...     normalized_text="reduction value:30% baseline_year:2019 period:2019–2030", unit="%", baseline="2019"
... )
>>> node_claim = claim.to_node(tenant_id="t0")
>>> # node_claim.props['hash'] now contains a stable content hash suitable for indexing
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4
import hashlib
import unicodedata
import re

from pydantic import BaseModel, Field, field_validator


# -----------------------------
# tiny helpers (no dependencies)
# -----------------------------

def now_utc() -> datetime:
    """UTC timestamp helper. Centralised for easy testing/mocking."""
    return datetime.now(timezone.utc)


def canonical_key(namespace: str, name: str) -> str:
    """
    Create a stable "type-scoped" key used for cross-linking nodes
    (e.g., 'org:acme plc', 'metric:ghg scope 1').

    - Lowercases & strips outer whitespace.
    - Internal whitespace is normalised to single spaces (no aggressive slugging).
    """
    ns = (namespace or "").strip().lower()
    nm = " ".join((name or "").strip().lower().split())
    return f"{ns}:{nm}" if ns else nm


def truncate(text: str, max_len: int = 96) -> str:
    """Short label helper for UI chips and logs."""
    if not text:
        return ""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _canon_text(s: Optional[str]) -> str:
    """Light canonicalisation used for hashing."""
    if not s:
        return ""
    t = _strip_accents(str(s).lower().strip())
    t = re.sub(r"[\s\-_\./]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _stable_hash(*parts: str, length: int = 40) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:length]


# ---------------
# KG base enums
# ---------------

class NodeKind(str, Enum):
    """Small, explicit set of node types."""
    ENTITY = "entity"      # company, person, regulator, supplier, etc.
    METRIC = "metric"      # e.g., "GHG Scope 1", "Revenue", "LTIFR"
    CLAIM = "claim"        # natural-language assertion extracted from docs
    EVIDENCE = "evidence"  # specific evidence pointer: doc/page/chunk/span
    DOCUMENT = "document"  # (optional) a source document node


class EdgeKind(str, Enum):
    """Directional edge types. Add sparingly."""
    MEASURED_BY = "measured_by"     # Entity --measured_by--> Metric
    QUANTIFIES = "quantifies"       # Claim --quantifies--> Metric
    ABOUT = "about"                 # Claim --about--> Entity
    SUPPORTED_BY = "supported_by"   # Claim --supported_by--> Evidence
    CITES = "cites"                 # Document/Evidence --cites--> Document/Evidence
    RELATES_TO = "relates_to"       # generic neighbour link (fallback)
    CONTRADICTS = "contradicts"     # Claim --contradicts--> Claim
    REFERS_TO = "refers_to"         # Evidence --refers_to--> Document


# ----------------------
# Storage-level contracts
# ----------------------

class Node(BaseModel):
    """
    Portable storage-level node.

    - `key` is a natural key scoped per (tenant_id, type).
      For entities/metrics we use `canonical_key(namespace, name)`.
      For claims/evidence we derive stable keys from their payload content.

    - `props` contains the typed payload (see payload models below) as JSON.
    """
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str = Field(..., description="Tenant/Workspace identifier")
    type: NodeKind
    key: str = Field(..., description="Stable natural key per tenant+type")
    label: str = Field(..., description="Short human label for chips/tooltips")
    props: Dict[str, Any] = Field(default_factory=dict, description="Typed payload as JSON")
    created_at: datetime = Field(default_factory=now_utc)

    model_config = dict(str_strip_whitespace=True)


class Edge(BaseModel):
    """
    Portable storage-level edge (directional).

    `src_id` and `dst_id` must reference existing Node ids in the same tenant.
    """
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    type: EdgeKind
    src_id: UUID
    dst_id: UUID
    props: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now_utc)

    model_config = dict(str_strip_whitespace=True)


# -----------------------
# Typed payload contracts
# -----------------------

class EntityPayload(BaseModel):
    """
    Payload captured for Entity nodes.

    `namespace` lets you scope types of entities (org, person, regulator).
    Keep it simple: "org" for companies is usually enough.

    `identifiers` can hold LEI, ISIN, Companies House id, etc.
    """
    namespace: str = Field("org", examples=["org", "person", "regulator"])
    name: str
    alt_names: list[str] = Field(default_factory=list)
    identifiers: Dict[str, str] = Field(default_factory=dict)

    def to_node(self, tenant_id: str) -> Node:
        key = canonical_key(self.namespace, self.name)
        return Node(
            tenant_id=tenant_id,
            type=NodeKind.ENTITY,
            key=key,
            label=truncate(self.name, 80),
            props=self.model_dump(),
        )


class MetricPayload(BaseModel):
    """
    Payload captured for Metric nodes.

    `dimension` is a light hint to group variants (e.g., scope1, scope2, intensity).
    `direction_good` indicates whether a higher value is good (True) or bad (False).
    """
    name: str = Field(..., examples=["GHG Scope 1", "Revenue"])
    unit: Optional[str] = Field(None, examples=["tCO2e", "USD", "£", "%"])
    standard: Optional[str] = Field(None, examples=["GHG Protocol", "IFRS"])
    dimension: Optional[str] = Field(None, examples=["scope1", "intensity"])
    direction_good: Optional[bool] = Field(
        None, description="If set, indicates whether increasing is desirable"
    )

    def to_node(self, tenant_id: str) -> Node:
        key = canonical_key("metric", self.name)
        return Node(
            tenant_id=tenant_id,
            type=NodeKind.METRIC,
            key=key,
            label=truncate(self.name, 80),
            props=self.model_dump(),
        )


class EvidencePayload(BaseModel):
    """
    Payload for Evidence nodes: a precise locator inside a document.

    - `doc_id` should correspond to your storage/source id (e.g., filename or UUID).
    - `page` and `chunk_id` help anchor citations.
    - `span_start`/`span_end` can carry character offsets within the chunk (optional).
    """
    doc_id: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    citation: Optional[str] = Field(
        None, description="Optional short citation text for UI tooltips"
    )
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    def to_node(self, tenant_id: str) -> Node:
        # Key is doc-scoped; simple and deterministic
        locator = f"{self.doc_id}#p{self.page if self.page is not None else 'NA'}:{self.chunk_id or 'NA'}"
        key = canonical_key("evidence", locator)
        label = truncate(self.citation or locator, 96)
        return Node(
            tenant_id=tenant_id,
            type=NodeKind.EVIDENCE,
            key=key,
            label=label,
            props=self.model_dump(),
        )


class ClaimPayload(BaseModel):
    """
    Payload for Claim nodes: a normalised assertion we extracted.

    Required: `text` (normalised), `entity_key` (usually from EntityPayload.to_node),
    Optional: `metric_key`, `period`, `target`.

    `confidence`: 0..1 extraction confidence from our NLP/regex pipeline.

    Provenance: `doc_id`, `page`, `chunk_id` should be set to link to Evidence later.

    Extras:
    - normalized_text: compact canonical form (e.g., "reduction value:30% baseline_year:2019 period:by 2030")
    - baseline: baseline year if present (e.g., "2019")
    - unit: detected unit token (e.g., "%", "tCO2e")
    - hash: stable content-hash used by backends like Neo4j for fast de-dup/index
    """
    text: str
    entity_key: str
    metric_key: Optional[str] = None
    period: Optional[str] = None       # e.g., "2019–2030"
    target: Optional[str] = None       # e.g., "30%" or "net zero"
    confidence: float = 0.7
    doc_id: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None

    # NEW structured fields
    normalized_text: Optional[str] = None
    baseline: Optional[str] = None
    unit: Optional[str] = None
    hash: Optional[str] = Field(default=None, description="Stable content hash for the claim")

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("confidence must be within [0, 1]")
        return v

    def _compute_hash(self, tenant_id: str) -> str:
        """
        Compute a stable hash using tenant + entity/metric + normalized text + period/baseline/unit.
        """
        parts = [
            _canon_text(tenant_id),
            _canon_text(self.entity_key),
            _canon_text(self.metric_key or ""),
            _canon_text(self.normalized_text or self.text),
            _canon_text(self.period or ""),
            _canon_text(self.baseline or ""),
            _canon_text(self.unit or ""),
        ]
        return _stable_hash(*parts, length=40)

    def to_node(self, tenant_id: str) -> Node:
        # Derive a stable key from entity + truncated claim text (kept for backward-compat idempotency).
        base = truncate(self.text, 64)
        key = canonical_key("claim", f"{self.entity_key}|{base}")

        # Ensure hash exists in props for stores that surface it to top-level (Neo4j)
        props = self.model_dump()
        if not props.get("hash"):
            props["hash"] = self._compute_hash(tenant_id)

        return Node(
            tenant_id=tenant_id,
            type=NodeKind.CLAIM,
            key=key,
            label=truncate(self.text, 96),
            props=props,
        )


# -----------------------
# Convenience edge makers
# -----------------------

def make_edge(
    tenant_id: str,
    kind: EdgeKind,
    src: Node,
    dst: Node,
    props: Optional[Dict[str, Any]] = None,
) -> Edge:
    """Typed, tiny helper to reduce boilerplate in builders."""
    return Edge(
        tenant_id=tenant_id,
        type=kind,
        src_id=src.id,
        dst_id=dst.id,
        props=props or {},
    )


__all__ = [
    # enums
    "NodeKind", "EdgeKind",
    # storage contracts
    "Node", "Edge",
    # payloads
    "EntityPayload", "MetricPayload", "EvidencePayload", "ClaimPayload",
    # helpers
    "canonical_key", "truncate", "now_utc", "make_edge",
]
