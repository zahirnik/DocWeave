# packages/knowledge_graph/builders.py
"""
Knowledge Graph — builders (documents → nodes & edges)
======================================================

Goal
----
Take parsed documents (already split into *chunks* for RAG) and build a small,
traceable knowledge graph:
    Entity ──measured_by──> Metric
       ↑           ^
       | about     | quantifies
      Claim ───────┘
       |
       └─supported_by→ Evidence(doc_id#page:chunk)

Design choices
--------------
- **Idempotent in-memory build**: we dedupe by *natural keys* so repeated runs
  for the same doc/chunk won't create duplicate nodes/edges *within the run*.
  (Your persistent store should upsert by (tenant_id, type, key) later.)
- **Provenance-first**: every Claim carries (doc_id, page, chunk_id) and we
  create one Evidence node *per locator* (doc_id#page:chunk) and reuse it.
- **Heuristic extractors**: uses `extractors.py` (regex-based) to create
  ClaimPayloads with metric/period/target when detected. You can later augment
  this with LLM extraction, but keep this deterministic path as a fallback.
- **Deterministic Claim.hash**: each Claim node is also tagged with a stable
  content-hash using (tenant_id, entity_key, metric_key?, normalized_text,
  period, baseline, unit) so backends (e.g., Neo4j) can index it efficiently.

Minimal inputs
--------------
We define small Pydantic models for inputs:
- DocumentInput: tenant_id, entity (name, namespace), doc_id, chunks[]
- ChunkInput: text + optional page, chunk_id

Public API
----------
- build_graph_for_doc(doc: DocumentInput, metric_aliases: dict|None)
    → tuple[list[Node], list[Edge]]

Example
-------
>>> doc = DocumentInput(
...   tenant_id="t0",
...   entity_name="Acme PLC",
...   entity_namespace="org",
...   doc_id="acme_2024_report.pdf",
...   chunks=[ChunkInput(text="We will reduce Scope 1 emissions by 30% by 2030.", page=12, chunk_id="c-77")]
... )
>>> nodes, edges = build_graph_for_doc(doc)
>>> [n.type.value for n in nodes]
['entity', 'metric', 'claim', 'evidence']
>>> [(e.type.value) for e in edges]
['measured_by', 'about', 'quantifies', 'supported_by']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from .schema import (
    ClaimPayload,
    Edge,
    EdgeKind,
    EntityPayload,
    EvidencePayload,
    MetricPayload,
    Node,
    NodeKind,
    canonical_key,
    make_edge,
    truncate,
)
from .extractors import extract_claims_from_text


# -----------------------
# Input data contracts
# -----------------------

class ChunkInput(BaseModel):
    text: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None


class DocumentInput(BaseModel):
    tenant_id: str
    entity_name: str
    entity_namespace: str = Field("org", description="Use 'org' for companies by default")
    doc_id: str
    chunks: List[ChunkInput]


# -----------------------
# Internal caches (per run)
# -----------------------

@dataclass
class _Caches:
    # Node caches by natural key
    entities: Dict[str, Node]
    metrics: Dict[str, Node]
    claims: Dict[str, Node]
    evidences: Dict[str, Node]
    # Edge de-duplication: (type, src_key, dst_key)
    edges_seen: Set[Tuple[str, str, str]]

    def __init__(self) -> None:
        self.entities = {}
        self.metrics = {}
        self.claims = {}
        self.evidences = {}
        self.edges_seen = set()


# -----------------------
# Node helpers
# -----------------------

def _get_or_make_entity(tenant_id: str, caches: _Caches, name: str, namespace: str = "org") -> Node:
    key = canonical_key(namespace, name)
    if key in caches.entities:
        return caches.entities[key]
    node = EntityPayload(namespace=namespace, name=name).to_node(tenant_id)
    caches.entities[key] = node
    return node


def _get_or_make_metric_from_key(tenant_id: str, caches: _Caches, metric_key: str) -> Node:
    """
    We receive a canonical metric key like 'metric:ghg scope 1'.
    Reconstruct a readable name for the MetricPayload.
    """
    if metric_key in caches.metrics:
        return caches.metrics[metric_key]

    # Recover canonical name (right side of "metric:...")
    try:
        _, canon_name = metric_key.split(":", 1)
    except ValueError:
        canon_name = metric_key

    node = MetricPayload(name=canon_name).to_node(tenant_id)
    # Ensure the generated key matches the incoming metric_key
    # (in practice it should, given canonical_key rules)
    caches.metrics[node.key] = node
    return node


def _get_or_make_evidence(tenant_id: str, caches: _Caches, *, doc_id: str, page: Optional[int], chunk_id: Optional[str], citation: Optional[str] = None) -> Node:
    payload = EvidencePayload(doc_id=doc_id, page=page, chunk_id=chunk_id, citation=citation)
    node = payload.to_node(tenant_id)
    if node.key in caches.evidences:
        return caches.evidences[node.key]
    caches.evidences[node.key] = node
    return node


def _get_or_make_claim(tenant_id: str, caches: _Caches, payload: ClaimPayload) -> Node:
    node = payload.to_node(tenant_id)
    if node.key in caches.claims:
        return caches.claims[node.key]
    caches.claims[node.key] = node
    return node


def _add_edge_once(caches: _Caches, edge: Edge, *, src_key: str, dst_key: str) -> Optional[Edge]:
    sig = (edge.type.value, src_key, dst_key)
    if sig in caches.edges_seen:
        return None
    caches.edges_seen.add(sig)
    return edge


# -----------------------
# Public builder
# -----------------------

def build_graph_for_doc(
    doc: DocumentInput,
    *,
    metric_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[List[Node], List[Edge]]:
    """
    Build a small KG slice for a single document.

    Steps
    -----
    1) Ensure the Entity node exists.
    2) For each chunk, extract ClaimPayloads (with provenance).
    3) For each claim:
       - create/reuse Claim node (by key)
       - create/reuse Metric node (if metric_key present)
       - create/reuse Evidence node for (doc_id#page:chunk)
       - add edges:
            Entity --MEASURED_BY--> Metric     (once per metric)
            Claim  --ABOUT--> Entity
            Claim  --QUANTIFIES--> Metric     (if metric present)
            Claim  --SUPPORTED_BY--> Evidence

    Returns
    -------
    (nodes, edges)
        Deduplicated lists you can upsert into your persistent store.

    Determinism
    -----------
    - Nodes/edges are returned in a stable order (by type, then key) to make tests easy.
    """
    caches = _Caches()
    tenant_id = doc.tenant_id

    # (1) Entity
    entity_node = _get_or_make_entity(tenant_id, caches, doc.entity_name, doc.entity_namespace)
    entity_key = entity_node.key

    # (2) Iterate chunks → extract claims
    edges: List[Edge] = []
    for ch in doc.chunks:
        claims = extract_claims_from_text(
            entity_key=entity_key,
            text=ch.text,
            doc_id=doc.doc_id,
            page=ch.page,
            chunk_id=ch.chunk_id,
            metric_aliases=metric_aliases,
        )
        if not claims:
            continue

        # Make one Evidence node per locator and reuse for all claims in this chunk
        citation_text = _make_citation_preview(ch.text)
        evidence_node = _get_or_make_evidence(
            tenant_id,
            caches,
            doc_id=doc.doc_id,
            page=ch.page,
            chunk_id=ch.chunk_id,
            citation=citation_text,
        )
        evidence_key = evidence_node.key

        for cl_payload in claims:
            # Claim node
            claim_node = _get_or_make_claim(tenant_id, caches, cl_payload)
            claim_key = claim_node.key

            # NEW: attach deterministic content-hash for Claim (for backend indexing)
            try:
                _maybe_attach_claim_hash(
                    claim_node,
                    tenant_id=tenant_id,
                    entity_key=entity_key,
                    metric_key=getattr(cl_payload, "metric_key", None),
                    normalized_text=_first_present(
                        getattr(cl_payload, "normalized_text", None),
                        getattr(cl_payload, "normalized", None),
                        getattr(cl_payload, "norm_text", None),
                        default=getattr(cl_payload, "text", None),
                    ),
                    period=_first_present(
                        getattr(cl_payload, "period", None),
                        getattr(cl_payload, "target_period", None),
                        getattr(cl_payload, "year", None),
                    ),
                    baseline=_first_present(
                        getattr(cl_payload, "baseline", None),
                        getattr(cl_payload, "baseline_year", None),
                        getattr(cl_payload, "base_year", None),
                    ),
                    unit=getattr(cl_payload, "unit", None),
                )
            except Exception:
                # Non-fatal; hashing is a bonus property for stores that use it.
                pass

            # about(Entity)
            e = make_edge(tenant_id, EdgeKind.ABOUT, claim_node, entity_node)
            maybe = _add_edge_once(caches, e, src_key=claim_key, dst_key=entity_key)
            if maybe:
                edges.append(maybe)

            # supported_by(Evidence)
            e = make_edge(tenant_id, EdgeKind.SUPPORTED_BY, claim_node, evidence_node)
            maybe = _add_edge_once(caches, e, src_key=claim_key, dst_key=evidence_key)
            if maybe:
                edges.append(maybe)

            # metric edges when present
            if getattr(cl_payload, "metric_key", None):
                metric_node = _get_or_make_metric_from_key(tenant_id, caches, cl_payload.metric_key)  # type: ignore[arg-type]
                metric_key = metric_node.key

                # Claim → Metric
                e = make_edge(tenant_id, EdgeKind.QUANTIFIES, claim_node, metric_node)
                maybe = _add_edge_once(caches, e, src_key=claim_key, dst_key=metric_key)
                if maybe:
                    edges.append(maybe)

                # Entity → Metric (measured_by)
                e = make_edge(tenant_id, EdgeKind.MEASURED_BY, entity_node, metric_node)
                maybe = _add_edge_once(caches, e, src_key=entity_key, dst_key=metric_key)
                if maybe:
                    edges.append(maybe)

    # Collect nodes in stable order (by type then key)
    nodes: List[Node] = []
    for bucket in (caches.entities, caches.metrics, caches.claims, caches.evidences):
        nodes.extend(bucket.values())
    nodes.sort(key=lambda n: (n.type.value, n.key))
    edges.sort(key=lambda e: (e.type.value, str(e.src_id), str(e.dst_id)))

    return nodes, edges


# -----------------------
# Tiny helpers
# -----------------------

def _make_citation_preview(text: str, max_len: int = 96) -> str:
    """
    Build a short tooltip-friendly citation snippet from a chunk of text.
    """
    return truncate(text.strip().replace("\n", " "), max_len)


# -------- Claim.hash helpers (local, no extra imports) -------------------------

def _first_present(*vals, default: Optional[str] = None) -> Optional[str]:
    for v in vals:
        if v is not None and str(v).strip() != "":
            return str(v)
    return default

def _strip_accents(s: str) -> str:
    try:
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    except Exception:
        return s

def _canon(s: Optional[str]) -> str:
    if not s:
        return ""
    t = _strip_accents(str(s).lower().strip())
    import re as _re
    t = _re.sub(r"[\\s\\-_\\./]+", " ", t)
    t = _re.sub(r"\\s+", " ", t).strip()
    return t

def _stable_hash(parts: List[str], length: int = 40) -> str:
    import hashlib
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:length]

def _maybe_attach_claim_hash(
    claim_node: Node,
    *,
    tenant_id: str,
    entity_key: str,
    metric_key: Optional[str],
    normalized_text: Optional[str],
    period: Optional[str],
    baseline: Optional[str],
    unit: Optional[str],
) -> None:
    """
    Compute and attach a deterministic 'hash' to the Claim node.
    Tries to store in claim_node.props['hash']; if props not present, sets attribute 'hash'.
    """
    parts = [
        _canon(tenant_id),
        _canon(entity_key),
        _canon(metric_key or ""),
        _canon(normalized_text or ""),
        _canon(period or ""),
        _canon(baseline or ""),
        _canon(unit or ""),
    ]
    chash = _stable_hash(parts, length=40)

    # Prefer props dict if the Node carries one
    props = getattr(claim_node, "props", None)
    if isinstance(props, dict):
        if "hash" not in props:
            props["hash"] = chash
        return

    # Fallback: set as attribute (backend adapter can read it)
    if not hasattr(claim_node, "hash"):
        setattr(claim_node, "hash", chash)
