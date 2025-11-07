# apps/api/schemas/models.py
"""
Pydantic request/response models used by the API.

Why keep them here?
- Clear separation of transport shapes from business logic.
- Easy to document, test, and evolve without touching route code.
- All models include examples to make /docs more helpful.

Pydantic v2 style (BaseModel + Field + json_schema_extra).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =========================
# Auth / Principal Models
# =========================

class TokenResponse(BaseModel):
    access_token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    token_type: str = Field("bearer", example="bearer")
    expires_in: int = Field(..., example=3600)


class Principal(BaseModel):
    """Auth context attached to every request after auth."""
    subject: str = Field(..., description="Unique user id or key id", example="user_123")
    tenant_id: str = Field(..., example="t0")
    roles: List[str] = Field(default_factory=list, example=["user"])
    scopes: List[str] = Field(default_factory=list, example=["rag:query"])
    auth_type: str = Field(..., example="bearer")  # "bearer" | "apikey"


class CreateApiKeyRequest(BaseModel):
    label: str = Field(..., example="ingest-bot-eu-west")
    ttl_days: int = Field(90, ge=1, le=365, description="How long the key will be valid.")


class CreateApiKeyResponse(BaseModel):
    key: str = Field(..., description="The plain API key — will be shown ONCE.")
    key_id: str = Field(..., description="Stored identifier for the key.")
    last4: str = Field(..., description="Convenience mask for logs/UI.")
    tenant_id: str
    roles: List[str]
    scopes: List[str]


# =========================
# Chat Models
# =========================

class ChatRequest(BaseModel):
    tenant_id: str = Field("t0", description="Tenant/organization id")
    query: str = Field(..., description="User question or instruction")
    top_k: int = Field(6, ge=1, le=50, description="Number of passages to retrieve")

    model_config = {
        "json_schema_extra": {
            "example": {"tenant_id": "t0", "query": "Summarize Q4 revenues YoY.", "top_k": 6}
        }
    }


class Source(BaseModel):
    score: float = Field(..., example=0.83)
    metadata: Dict[str, Any] = Field(
        ..., example={"source": "docs/10k_2023.pdf", "page": 14, "type": "pdf"}
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Final grounded answer")
    sources: List[Source] = Field(default_factory=list, description="Top passages with metadata")


# =========================
# Ingestion Models
# =========================

class IngestFilesResponse(BaseModel):
    job_id: str
    collection: str
    accepted_files: List[str]
    skipped_files: List[str] = []
    note: str = Field(
        "Use GET /ingest/{job_id} to poll status. Worker will parse→chunk→embed→upsert.",
        description="Polling info",
    )


class IngestUrlsRequest(BaseModel):
    urls: List[str] = Field(..., min_items=1, description="HTTP(S) URLs to ingest (PDF/CSV/JSON...)")
    collection: str = Field("default", description="Vector collection name")
    tenant_id: str = Field("t0", description="Tenant id for isolation/tagging")

    model_config = {
        "json_schema_extra": {
            "example": {
                "urls": ["https://example.com/report.pdf", "https://example.com/table.csv"],
                "collection": "finance_2024",
                "tenant_id": "t0",
            }
        }
    }


class IngestUrlsResponse(BaseModel):
    job_id: str
    collection: str
    url_count: int
    note: str = "Use GET /ingest/{job_id} to poll status."


class IngestStatusResponse(BaseModel):
    job_id: str
    state: str = Field(..., example="PENDING")  # PENDING / STARTED / RETRY / FAILURE / SUCCESS
    progress: Optional[float] = Field(None, example=0.42, description="0..1 if reported by task")
    result: Optional[dict] = Field(None, description="Final task result on SUCCESS")
    error: Optional[str] = None


# =========================
# Search Models
# =========================

class SearchFilters(BaseModel):
    """Simple metadata filter example. Extend as needed."""
    type: Optional[str] = Field(None, description="Filter by metadata.type (e.g., pdf|csv|json)")
    source_contains: Optional[str] = Field(None, description="Substring match against metadata.source")


class SearchHit(BaseModel):
    score: float = Field(..., description="Final score after hybrid combination (if used)")
    method: str = Field(..., example="vector", description="vector|bm25|hybrid")
    metadata: Dict[str, Any]


class SearchPage(BaseModel):
    page: int
    size: int
    total: int


class SearchResponse(BaseModel):
    items: List[SearchHit]
    page: SearchPage
    notes: Optional[str] = Field(
        None,
        description="Hints about hybrid mode or when BM25 is disabled/unavailable.",
    )


# =========================
# Analytics Models
# =========================

class AnalyticsResponse(BaseModel):
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    metrics: Dict[str, Dict[str, float]]
    head: List[Dict[str, Any]]
    chart_path: Optional[str] = Field(
        None, description="Path to saved PNG chart (when chart=true and x/y provided)"
    )
