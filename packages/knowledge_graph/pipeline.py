# packages/knowledge_graph/pipeline.py
"""
Knowledge Graph — pipeline (build → persist → optional LLM fallback → validate)
==============================================================================

Purpose
-------
Provide a *single* call that:
1) Builds a KG slice from parsed chunks using the heuristic extractors.
2) Upserts nodes/edges via a Store (default in-memory; no DB needed).
3) Optionally applies an LLM fallback per chunk if no claims were found.
4) Optionally validates the resulting snapshot.

Works on Colab/AWS with InMemoryStore. No external services required unless
you enable the LLM fallback (needs OPENAI_API_KEY and openai SDK).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Type-only imports to satisfy linters (not needed at runtime)
if TYPE_CHECKING:
    from .builders import DocumentInput
    from .schema import Node, Edge, ClaimPayload

# Runtime imports (actually used at execution)
from .builders import build_graph_for_doc
from .schema import (
    EdgeKind,
    NodeKind,
    EntityPayload,
    EvidencePayload,
    MetricPayload,
    make_edge,
    truncate,
)
from .store import Store
from .validators import Severity, validate_from_store

# Optional (only if llm_fallback=True)
try:
    from .llm_extractors import llm_extract_claims  # type: ignore
except Exception:  # pragma: no cover
    llm_extract_claims = None  # type: ignore


# -----------------------
# Result DTO
# -----------------------

@dataclass
class BuildReport:
    tenant_id: str
    nodes_created: int
    nodes_updated: int
    edges_created: int
    edges_ignored: int
    validated: bool
    errors: int
    warnings: int
    used_llm_fallback: bool

    @property
    def ok(self) -> bool:
        return self.errors == 0


# -----------------------
# Public API
# -----------------------

def build_and_persist_kg(
    doc: "DocumentInput",
    store: Store,
    *,
    metric_aliases: Optional[Dict[str, str]] = None,
    validate: bool = True,
    min_confidence: float = 0.50,
    require_provenance: bool = True,
    llm_fallback: bool = False,
    llm_model: str = "gpt-4o-mini",
    llm_client: Any = None,
) -> BuildReport:
    """
    Build a KG slice, upsert it, optionally apply an LLM fallback, and validate.

    - Heuristic pass: always runs (fast, deterministic).
    - LLM fallback: only runs if enabled *and* the heuristic pass yields zero claims.
    """
    # 1) Heuristic build
    nodes, edges = build_graph_for_doc(doc, metric_aliases=metric_aliases)
    res = store.upsert_nodes_edges(nodes, edges)

    # 2) Did we find any claims?
    found_claims = any(n.type == NodeKind.CLAIM for n in nodes)
    used_llm = False

    # 3) Optional LLM fallback (only if no heuristic claims)
    if llm_fallback and not found_claims:
        if llm_extract_claims is None:
            print("[kg:pipeline] LLM fallback requested but llm_extract_claims unavailable. Install OpenAI SDK or disable fallback.")
        else:
            used_llm = True
            llm_nodes, llm_edges = _build_from_llm(doc, llm_model=llm_model, llm_client=llm_client)
            if llm_nodes or llm_edges:
                r2 = store.upsert_nodes_edges(llm_nodes, llm_edges)
                res.nodes_created += r2.nodes_created
                res.nodes_updated += r2.nodes_updated
                res.edges_created += r2.edges_created
                res.edges_ignored += r2.edges_ignored

    # 4) Optional validation (against persisted snapshot)
    errors = warnings = 0
    validated = False
    if validate:
        rpt = validate_from_store(store, doc.tenant_id)
        validated = True
        errors = rpt.count(Severity.ERROR)
        warnings = rpt.count(Severity.WARNING)

    return BuildReport(
        tenant_id=doc.tenant_id,
        nodes_created=res.nodes_created,
        nodes_updated=res.nodes_updated,
        edges_created=res.edges_created,
        edges_ignored=res.edges_ignored,
        validated=validated,
        errors=errors,
        warnings=warnings,
        used_llm_fallback=used_llm,
    )


# -----------------------
# Internal — LLM-backed builder (small and explicit)
# -----------------------

def _build_from_llm(
    doc: "DocumentInput",
    *,
    llm_model: str,
    llm_client: Any,
) -> Tuple[List["Node"], List["Edge"]]:
    """
    Build nodes/edges by calling the LLM extractor when heuristic pass found none.
    We:
      - create/reuse Entity node
      - per chunk: create Evidence node (doc_id#page:chunk)
      - per claim: Claim node; edges ABOUT(Entity), SUPPORTED_BY(Evidence)
      - if claim.metric_key present: Metric node; edges QUANTIFIES + MEASURED_BY
    """
    assert llm_extract_claims is not None, "llm_extract_claims not available"

    tenant_id = doc.tenant_id
    nodes: List["Node"] = []
    edges: List["Edge"] = []

    # Entity
    ent = EntityPayload(namespace=getattr(doc, "entity_namespace", "org"), name=doc.entity_name).to_node(tenant_id)
    nodes.append(ent)

    # De-dupe maps
    metrics_by_key: Dict[str, "Node"] = {}
    claims_by_key: Dict[str, "Node"] = {}
    evid_by_key: Dict[str, "Node"] = {}

    for ch in doc.chunks:
        # Evidence node
        ev = EvidencePayload(
            doc_id=doc.doc_id,
            page=ch.page,
            chunk_id=ch.chunk_id,
            citation=truncate((ch.text or "").strip().replace("\n", " "), 96),
        ).to_node(tenant_id)
        if ev.key not in evid_by_key:
            evid_by_key[ev.key] = ev
            nodes.append(ev)
        ev_node = evid_by_key[ev.key]

        # LLM claims (structured JSON → ClaimPayloads)
        claims: List["ClaimPayload"] = llm_extract_claims(
            entity_key=ent.key,
            text=ch.text or "",
            doc_id=doc.doc_id,
            page=ch.page,
            chunk_id=ch.chunk_id,
            model=llm_model,
            client=llm_client,
        )

        for cl in claims:
            cl_node = cl.to_node(tenant_id)
            if cl_node.key not in claims_by_key:
                claims_by_key[cl_node.key] = cl_node
                nodes.append(cl_node)
            cl_node = claims_by_key[cl_node.key]

            # ABOUT(Entity)
            edges.append(make_edge(tenant_id, EdgeKind.ABOUT, cl_node, ent))
            # SUPPORTED_BY(Evidence)
            edges.append(make_edge(tenant_id, EdgeKind.SUPPORTED_BY, cl_node, ev_node))

            # Metric edges if present
            if cl.metric_key:
                try:
                    _, canon_name = cl.metric_key.split(":", 1)
                except ValueError:
                    canon_name = cl.metric_key
                met = metrics_by_key.get(cl.metric_key)
                if met is None:
                    met = MetricPayload(name=canon_name).to_node(tenant_id)
                    metrics_by_key[met.key] = met
                    nodes.append(met)
                edges.append(make_edge(tenant_id, EdgeKind.QUANTIFIES, cl_node, met))
                edges.append(make_edge(tenant_id, EdgeKind.MEASURED_BY, ent, met))

    nodes.sort(key=lambda n: (n.type.value, n.key))
    edges.sort(key=lambda e: (e.type.value, str(e.src_id), str(e.dst_id)))
    return nodes, edges


__all__ = ["BuildReport", "build_and_persist_kg"]
