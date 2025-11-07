# packages/knowledge_graph/__init__.py
"""
Knowledge Graph package (portable & testable)
=============================================

This package turns parsed documents into a small, traceable KG and exposes
a backend-agnostic Store interface. It is intentionally *boring* and easy
to read, mirroring tutorial-style clarity.

What you’ll typically import
----------------------------
- Data contracts (nodes/edges/payloads): from .schema
- Heuristic extractors (text/table → claims): from .extractors
- Builders (docs/chunks → nodes & edges): from .builders
- Store API (in-memory reference): from .store
- Validators (snapshot checks): from .validators
- Exporters (JSON/GraphML): from .exporters
- Queries (helper lookups): from .queries
"""

__version__ = "0.1.0"

# Schema (storage contracts & payloads)
from .schema import (
    Node,
    Edge,
    NodeKind,
    EdgeKind,
    EntityPayload,
    MetricPayload,
    EvidencePayload,
    ClaimPayload,
    canonical_key,
    truncate,
    now_utc,
    make_edge,
)

# Extractors (heuristics)
from .extractors import (
    DEFAULT_METRIC_ALIASES,
    extract_claims_from_text,
    extract_claims_from_table,
)

# Builders (documents → nodes & edges)
from .builders import (
    ChunkInput,
    DocumentInput,
    build_graph_for_doc,
)

# Store (backend-agnostic + in-memory reference)
from .store import (
    PersistResult,
    Store,
    InMemoryStore,
    get_store,
)

# Validators (snapshot-level checks)
from .validators import (
    Severity,
    IssueCode,
    ValidationIssue,
    ValidationReport,
    validate_snapshot,
    validate_from_store,
)

# Exporters (JSON / GraphML)
from .exporters import (
    to_json,
    to_graphml,
)

# Queries (helper lookups)
from .queries import (
    ClaimWithEvidence,
    claims_about_entity,
    claims_for_metric,
    evidence_for_claim,
    claims_with_evidence_about_entity,
    entity_metric_pairs,
    subgraph_for_entity,
)

__all__ = [
    # Schema
    "Node",
    "Edge",
    "NodeKind",
    "EdgeKind",
    "EntityPayload",
    "MetricPayload",
    "EvidencePayload",
    "ClaimPayload",
    "canonical_key",
    "truncate",
    "now_utc",
    "make_edge",
    # Extractors
    "DEFAULT_METRIC_ALIASES",
    "extract_claims_from_text",
    "extract_claims_from_table",
    # Builders
    "ChunkInput",
    "DocumentInput",
    "build_graph_for_doc",
    # Store
    "PersistResult",
    "Store",
    "InMemoryStore",
    "get_store",
    # Validators
    "Severity",
    "IssueCode",
    "ValidationIssue",
    "ValidationReport",
    "validate_snapshot",
    "validate_from_store",
    # Exporters
    "to_json",
    "to_graphml",
    # Queries
    "ClaimWithEvidence",
    "claims_about_entity",
    "claims_for_metric",
    "evidence_for_claim",
    "claims_with_evidence_about_entity",
    "entity_metric_pairs",
    "subgraph_for_entity",
]
