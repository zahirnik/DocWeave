# apps/api/schemas/kg.py
"""
KG API Schemas
==============

Purpose
-------
Typed request/response models for Knowledge Graph endpoints. Keeping these
in a separate module makes the route handlers tiny and the models reusable
(e.g., CLI, workers, or tests).

Notes
-----
- Pydantic v2 models.
- Include `json_schema_extra` examples to improve OpenAPI docs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -----------------------
# Build KG
# -----------------------

class BuildKGChunk(BaseModel):
    text: str = Field(..., description="Parsed text (one chunk/paragraph).")
    page: Optional[int] = Field(None, description="1-based page index (if known).")
    chunk_id: Optional[str] = Field(None, description="Upstream chunk identifier.")

    model_config = {
        "json_schema_extra": {
            "example": {"text": "We will reduce Scope 1 emissions by 30% by 2030.", "page": 12, "chunk_id": "c-77"}
        }
    }


class BuildKGRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant/workspace id (e.g., 't0').")
    entity_name: str = Field(..., description="Display name of the entity (e.g., 'Acme PLC').")
    entity_namespace: str = Field("org", description="Namespace for the entity key (default: 'org').")
    doc_id: str = Field(..., description="Source document id or filename.")
    chunks: List[BuildKGChunk]
    metric_aliases: Optional[Dict[str, str]] = Field(
        None, description="Optional regex → canonical metric name (overrides/extends defaults)."
    )
    validate: bool = Field(True, description="Run validators after upsert.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenant_id": "t0",
                "entity_name": "Acme PLC",
                "entity_namespace": "org",
                "doc_id": "acme_2024_report.pdf",
                "chunks": [
                    {"text": "We will reduce Scope 1 emissions by 30% by 2030.", "page": 12, "chunk_id": "c-77"}
                ],
                "validate": True,
            }
        }
    }


class BuildKGResponse(BaseModel):
    nodes_created: int
    nodes_updated: int
    edges_created: int
    edges_ignored: int
    validated: bool
    validation_summary: Optional[str] = None
    errors: int = 0
    warnings: int = 0

    model_config = {
        "json_schema_extra": {
            "example": {
                "nodes_created": 4,
                "nodes_updated": 0,
                "edges_created": 4,
                "edges_ignored": 0,
                "validated": True,
                "validation_summary": "Validation — errors=0, warnings=0, info=0",
                "errors": 0,
                "warnings": 0,
            }
        }
    }


# -----------------------
# Subgraph
# -----------------------

class SubgraphResponse(BaseModel):
    meta: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

    model_config = {
        "json_schema_extra": {
            "example": {
                "meta": {"tenant_id": "t0", "entity_key": "org:acme plc", "depth": 1, "node_count": 4, "edge_count": 4},
                "nodes": [
                    {"id": "4d13...", "type": "entity", "key": "org:acme plc", "label": "Acme PLC", "props": {}},
                    {"id": "b8b0...", "type": "metric", "key": "metric:ghg scope 1", "label": "GHG Scope 1", "props": {}},
                    {"id": "a1f2...", "type": "claim", "key": "claim:org:acme plc|acme will reduce scope 1…", "label": "Acme will reduce Scope 1...", "props": {"confidence": 0.9}},
                    {"id": "c77e...", "type": "evidence", "key": "evidence:acme_2024_report.pdf#p12:c-77", "label": "acme_2024_report.pdf#p12:c-77", "props": {}}
                ],
                "edges": [
                    {"id": "e1", "type": "about", "src": "a1f2...", "dst": "4d13..."},
                    {"id": "e2", "type": "supported_by", "src": "a1f2...", "dst": "c77e..."},
                    {"id": "e3", "type": "quantifies", "src": "a1f2...", "dst": "b8b0..."},
                    {"id": "e4", "type": "measured_by", "src": "4d13...", "dst": "b8b0..."}
                ],
            }
        }
    }


__all__ = [
    "BuildKGChunk",
    "BuildKGRequest",
    "BuildKGResponse",
    "SubgraphResponse",
]
