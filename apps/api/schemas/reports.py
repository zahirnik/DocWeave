# apps/api/schemas/reports.py
"""
Reporting API Schemas
=====================

Purpose
-------
Typed request/response models for deterministic reporting built
on top of the Knowledge Graph (no LLMs). Kept separate so routes
stay ultra-thin and examples show up nicely in /docs.

Endpoints expected to use these:
- POST /reports/scorecard
- POST /reports/benchmark
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -----------------------
# Scorecard
# -----------------------

class ScoreRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant/workspace id (e.g., 't0').")
    entity_key: str = Field(..., description="Canonical entity key (e.g., 'org:acme plc').")
    framework_yaml_path: str = Field(
        "packages/reporting/frameworks/six_capitals.yaml",
        description="Path to the Six-Capitals framework YAML.",
    )
    sector_profiles_yaml_path: Optional[str] = Field(
        "packages/reporting/frameworks/sector_profiles.yaml",
        description="Optional sector profile overrides YAML.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenant_id": "t0",
                "entity_key": "org:acme plc",
                "framework_yaml_path": "packages/reporting/frameworks/six_capitals.yaml",
                "sector_profiles_yaml_path": "packages/reporting/frameworks/sector_profiles.yaml",
            }
        }
    }


class MetricFlags(BaseModel):
    target_present: bool
    period_present: bool
    evidence_linked: bool
    specificity_good: bool
    metric_aligned: bool


class MetricView(BaseModel):
    key: str
    label: str
    weight: float
    score: float
    flags: MetricFlags
    considered_claim_ids: List[str] = Field(default_factory=list)


class CapitalView(BaseModel):
    name: str
    weight: float
    score: float
    metrics: List[MetricView]


class ScoreResponse(BaseModel):
    entity_key: str
    overall_score: float
    capitals: List[CapitalView]
    notes: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "entity_key": "org:acme plc",
                "overall_score": 0.63,
                "capitals": [
                    {
                        "name": "natural",
                        "weight": 0.40,
                        "score": 0.58,
                        "metrics": [
                            {
                                "key": "metric:ghg scope 1",
                                "label": "GHG Scope 1",
                                "weight": 0.30,
                                "score": 0.67,
                                "flags": {
                                    "target_present": True,
                                    "period_present": True,
                                    "evidence_linked": True,
                                    "specificity_good": True,
                                    "metric_aligned": True
                                },
                                "considered_claim_ids": ["7b6d3b11-..."]
                            }
                        ]
                    }
                ],
                "notes": {"capital_weights_sum": "1.000", "profile_used": "energy", "min_confidence": "0.50"}
            }
        }
    }


# -----------------------
# Benchmark
# -----------------------

class BenchmarkRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant/workspace id.")
    focus_entity: str = Field(..., description="Canonical entity key to highlight.")
    entity_keys: List[str] = Field(..., description="Entities to include in the peer set (must include focus).")
    framework_yaml_path: str = Field(
        "packages/reporting/frameworks/six_capitals.yaml",
        description="Path to the Six-Capitals framework YAML.",
    )
    sector_profiles_yaml_path: Optional[str] = Field(
        "packages/reporting/frameworks/sector_profiles.yaml",
        description="Optional sector profile overrides YAML.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenant_id": "t0",
                "focus_entity": "org:acme plc",
                "entity_keys": ["org:acme plc", "org:globex", "org:initech"],
                "framework_yaml_path": "packages/reporting/frameworks/six_capitals.yaml",
                "sector_profiles_yaml_path": "packages/reporting/frameworks/sector_profiles.yaml",
            }
        }
    }


class PeerView(BaseModel):
    entity_key: str
    overall_score: float
    rank_overall: int
    coverage: float
    metrics_total: int
    metrics_with_claims: int
    scores_by_capital: Dict[str, float] = Field(default_factory=dict)
    ranks_by_capital: Dict[str, int] = Field(default_factory=dict)


class BenchmarkResponse(BaseModel):
    focus: PeerView
    peers: List[PeerView]
    notes: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "focus": {
                    "entity_key": "org:acme plc",
                    "overall_score": 0.63,
                    "rank_overall": 2,
                    "coverage": 0.70,
                    "metrics_total": 10,
                    "metrics_with_claims": 7,
                    "scores_by_capital": {"natural": 0.58, "financial": 0.66},
                    "ranks_by_capital": {"natural": 1, "financial": 3}
                },
                "peers": [
                    {
                        "entity_key": "org:globex",
                        "overall_score": 0.69,
                        "rank_overall": 1,
                        "coverage": 0.80,
                        "metrics_total": 10,
                        "metrics_with_claims": 8,
                        "scores_by_capital": {"natural": 0.60, "financial": 0.72},
                        "ranks_by_capital": {"natural": 1, "financial": 1}
                    }
                ],
                "notes": {"entities": "3", "framework": "packages/reporting/frameworks/six_capitals.yaml"}
            }
        }
    }


__all__ = [
    # scorecard
    "ScoreRequest",
    "ScoreResponse",
    "CapitalView",
    "MetricView",
    "MetricFlags",
    # benchmark
    "BenchmarkRequest",
    "BenchmarkResponse",
    "PeerView",
]
