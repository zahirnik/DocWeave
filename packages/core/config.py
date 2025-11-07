# packages/core/config.py
"""
Centralized application settings (12-factor friendly, tutorial-clear).

Goals
-----
- A single, tiny source of truth for configuration.
- Explicit env var names with safe defaults for local dev.
- Strict validation where it matters (e.g., APP_ENV, JWT secret in prod).
- Clear docstrings and examples so new teammates can self-serve.

How to use
----------
from packages.core.config import get_settings
settings = get_settings()
print(settings.app_env, settings.vectorstore_backend)

Env var reference (common)
--------------------------
APP_ENV=dev|stage|prod
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
DATA_DIR=./data
OUTPUTS_DIR=./data/outputs
ENABLE_TRACING=true|false
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318

# Auth / API
JWT_SECRET=change-me-in-prod
JWT_ISSUER=convai
JWT_AUDIENCE=convai.clients
JWT_TTL_S=3600

# Backends
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/convai
VECTORSTORE_BACKEND=simple|pgvector|qdrant|chroma

# Models / Providers
OPENAI_API_KEY=...
EMBEDDINGS_PROVIDER=openai|sentence
EMBEDDINGS_MODEL=text-embedding-3-small|BAAI/bge-small-en-v1.5
LLM_MODEL=gpt-4o-mini
TAVILY_API_KEY=...

# LangSmith (optional)
LANGSMITH_TRACING=false|true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=convai-finance-agentic-rag
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    """Small wrapper to allow empty string → None conversion uniformly."""
    val = os.getenv(name, default)
    if val is None:
        return None
    val = val.strip()
    return val if val != "" else None


def _as_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return s.lower() in {"1", "true", "yes", "y", "on"}


class Settings(BaseModel):
    """
    Typed, validated settings object.

    NOTE: This uses plain Pydantic BaseModel to stay dependency-light.
    If you prefer pydantic-settings, you can swap it in later with the same fields.
    """

    # --- App / env ---
    app_env: Literal["dev", "stage", "prod"] = Field(
        default_factory=lambda: _getenv("APP_ENV", "dev") or "dev",
        description="Deployment environment flag controlling strictness & defaults.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default_factory=lambda: _getenv("LOG_LEVEL", "INFO") or "INFO",
        description="Logging verbosity.",
    )
    data_dir: str = Field(
        default_factory=lambda: _getenv("DATA_DIR", "./data") or "./data",
        description="Root for data (uploads, samples).",
    )
    outputs_dir: str = Field(
        default_factory=lambda: _getenv("OUTPUTS_DIR", "./data/outputs") or "./data/outputs",
        description="Where generated artifacts (charts/reports) are written.",
    )

    # --- Observability ---
    enable_tracing: bool = Field(
        default_factory=lambda: _as_bool(_getenv("ENABLE_TRACING"), False),
        description="Enable OpenTelemetry tracing if available.",
    )
    otel_endpoint: Optional[str] = Field(
        default_factory=lambda: _getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        description="OTLP HTTP exporter endpoint (e.g., http://otel-collector:4318).",
    )

    # --- CORS ---
    enable_cors: bool = Field(
        default_factory=lambda: _as_bool(_getenv("ENABLE_CORS", "true"), True),
        description="Allow cross-origin requests (handy in local dev).",
    )
    cors_allow_origins: List[str] = Field(
        default_factory=lambda: (_getenv("CORS_ALLOW_ORIGINS", "*") or "*").split(","),
        description="Comma-separated origins for CORS.",
    )

    # --- Caches / DB ---
    redis_url: Optional[str] = Field(
        default_factory=lambda: _getenv("REDIS_URL") or _getenv("CELERY_BROKER_URL"),
        description="Redis connection URL (also used by Celery unless overridden).",
    )
    database_url: Optional[str] = Field(
        default_factory=lambda: _getenv("DATABASE_URL"),
        description="SQLAlchemy DB URL (Postgres recommended).",
    )

    # --- Vector store choice ---
    vectorstore_backend: Literal["simple", "pgvector", "qdrant", "chroma"] = Field(
        default_factory=lambda: _getenv("VECTORSTORE_BACKEND", "simple") or "simple",
        description="Which vector store driver to use by default.",
    )

    # --- Security / JWT ---
    jwt_secret: str = Field(
        default_factory=lambda: _getenv("JWT_SECRET", "dev-insecure-secret") or "dev-insecure-secret",
        description="HMAC secret for JWT signing (MUST set in prod).",
    )
    jwt_issuer: str = Field(
        default_factory=lambda: _getenv("JWT_ISSUER", "convai") or "convai",
        description="JWT issuer claim.",
    )
    jwt_audience: str = Field(
        default_factory=lambda: _getenv("JWT_AUDIENCE", "convai.clients") or "convai.clients",
        description="JWT audience claim to validate.",
    )
    jwt_ttl_s: int = Field(
        default_factory=lambda: int(_getenv("JWT_TTL_S", "3600") or "3600"),
        description="JWT expiration in seconds.",
    )

    # --- Providers / Models ---
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: _getenv("OPENAI_API_KEY"),
        description="OpenAI API key (if using OpenAI models/embeddings).",
    )
    embeddings_provider: Literal["openai", "sentence"] = Field(
        default_factory=lambda: _getenv("EMBEDDINGS_PROVIDER", "sentence") or "sentence",
        description="Which embedding provider to use by default.",
    )
    embeddings_model: str = Field(
        default_factory=lambda: _getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5") or "BAAI/bge-small-en-v1.5",
        description="Embedding model identifier.",
    )
    llm_model: str = Field(
        default_factory=lambda: _getenv("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        description="Chat/completion model identifier.",
    )
    tavily_api_key: Optional[str] = Field(
        default_factory=lambda: _getenv("TAVILY_API_KEY"),
        description="Tavily API key for web search (optional).",
    )

    # --- LangSmith (optional) ---
    langsmith_tracing: bool = Field(
        default_factory=lambda: _as_bool(_getenv("LANGSMITH_TRACING"), False),
        description="Enable LangSmith traces (if key provided).",
    )
    langsmith_api_key: Optional[str] = Field(
        default_factory=lambda: _getenv("LANGSMITH_API_KEY"),
        description="LangSmith API key (optional).",
    )
    langsmith_project: str = Field(
        default_factory=lambda: _getenv("LANGSMITH_PROJECT", "convai-finance-agentic-rag") or "convai-finance-agentic-rag",
        description="Default LangSmith project name.",
    )

    # --- Rate limit defaults (used by middleware) ---
    rate_limit_rate: float = Field(
        default_factory=lambda: float(_getenv("RATE_LIMIT_RATE", "5") or "5"),
        description="Tokens per second for in-app limiter.",
    )
    rate_limit_burst: int = Field(
        default_factory=lambda: int(_getenv("RATE_LIMIT_BURST", "10") or "10"),
        description="Max bucket capacity for in-app limiter.",
    )

    # --- Derived / utility flags ---
    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"

    # --- Validation hooks ---
    @validator("jwt_secret")
    def _prod_requires_secret(cls, v: str) -> str:
        env = _getenv("APP_ENV", "dev") or "dev"
        if env == "prod" and (not v or v == "dev-insecure-secret"):
            raise ValueError("JWT_SECRET must be set to a strong secret in prod.")
        return v

    @validator("vectorstore_backend")
    def _validate_backend_deps(cls, v: str) -> str:
        """
        Gentle guidance: if a non-'simple' backend is selected but DATABASE_URL or
        vendor URL is missing, we don't block import, but leave an actionable hint.
        Downstream drivers should error clearly when initialized without config.
        """
        return v

    def summary(self) -> dict:
        """Redaction-safe summary for logs (/health) (secrets masked)."""
        return {
            "env": self.app_env,
            "log_level": self.log_level,
            "data_dir": self.data_dir,
            "outputs_dir": self.outputs_dir,
            "enable_tracing": self.enable_tracing,
            "otel_endpoint": bool(self.otel_endpoint),
            "redis_url": bool(self.redis_url),
            "database_url": bool(self.database_url),
            "vectorstore_backend": self.vectorstore_backend,
            "embeddings_provider": self.embeddings_provider,
            "embeddings_model": self.embeddings_model,
            "llm_model": self.llm_model,
            "langsmith_tracing": self.langsmith_tracing,
            "langsmith_project": self.langsmith_project,
            "rate_limit_rate": self.rate_limit_rate,
            "rate_limit_burst": self.rate_limit_burst,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a singleton Settings instance. Values are read once from the process
    environment. Restart (or call `reload_settings()`) to pick up changes.

    Example:
        from packages.core.config import get_settings
        settings = get_settings()
        if settings.enable_tracing:
            ...
    """
    s = Settings()  # type: ignore[call-arg]  # pydantic validates
    # Best-effort: create directories eagerly for friendlier first run.
    try:
        os.makedirs(s.data_dir, exist_ok=True)
        os.makedirs(s.outputs_dir, exist_ok=True)
    except Exception:
        # Do not crash on import; filesystem might be read-only in some environments.
        pass
    return s


def reload_settings() -> Settings:
    """
    Drop the cached Settings and rebuild from env.
    Useful in REPL/tests when you tweak os.environ between runs.
    """
    get_settings.cache_clear()  # type: ignore[attr-defined]
    return get_settings()
