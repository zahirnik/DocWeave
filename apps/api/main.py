# apps/api/main.py
"""
Ultra-clear FastAPI app factory.

What this file does (and only this):
- Builds a FastAPI app instance.
- Wires middlewares (request-id, simple rate-limit).
- Registers all API routers (auth, chat, ingest, search, analytics, kg, reports, ui).
- Initializes logging + tracing in a safe, noop-friendly way.
- Exposes tiny / and /health endpoints for probes & sanity checks.
- [ADD] Validates backend env (vector/bm25) early to fail fast on misconfig.

How to run locally:
    uvicorn apps.api.main:app --reload --port 8000

Environment variables that matter here:
    APP_ENV=dev|stage|prod
    ENABLE_TRACING=true|false
    CONVAI_APP_VERSION=0.1.0
    ENABLE_CORS=true|false
    CORS_ALLOW_ORIGINS=*
"""

from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Core settings/logging/telemetry (small, explicit modules in packages/core & packages/observability)
from packages.core.config import get_settings
from packages.core.logging import setup_json_logging
from packages.observability.tracing import init_tracing
# [ADD] Fail-fast validation of backend selections (VECTOR_STORE / BM25_PROVIDER)
from packages.core.naming import assert_backends_known

# Middlewares (kept tiny and well-commented in apps/api/middlewares/)
from .middlewares.request_id import request_id_middleware
from .middlewares.rate_limit import rate_limit_middleware  # token-bucket w/ sane defaults

# Routers (each file is small and self-explanatory in apps/api/routes/)
from .routes import auth, chat, ingest, search, analytics, kg, reports, ui  # ui provides /ui & /ui/ask


def create_app() -> FastAPI:
    """Create and configure the FastAPI app (single place, easy to read)."""
    settings = get_settings()

    # 1) Logging first (JSON to stdout). Safe to call multiple times.
    setup_json_logging()

    # 2) Tracing is best-effort: becomes a no-op if ENABLE_TRACING is false or Otel isn’t installed.
    init_tracing()

    # 3) [ADD] Validate backend env early so misconfigs crash fast & clearly.
    #     - Ensures VECTOR_STORE and BM25_PROVIDER are supported values.
    #     - Keeps naming/resolution consistent across API/worker.
    assert_backends_known()

    app = FastAPI(
        title="Convai Finance Agentic RAG API",
        version=os.getenv("CONVAI_APP_VERSION", "0.1.0"),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 4) CORS (helpful for local web demo; can be disabled in prod via env)
    if os.getenv("ENABLE_CORS", "true").lower() in {"1", "true", "yes"}:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 5) Middlewares (ordered)
    #    - request-id: attaches x-request-id to all responses for tracing/log correlation
    #    - rate-limit: lightweight token-bucket guard (per-IP or header key inside the impl)
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(rate_limit_middleware)

    # 6) Core routers (auth first; then core features)
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
    app.include_router(search.router, prefix="/search", tags=["search"])
    app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
    app.include_router(kg.router)         # router defines prefix="/kg"
    app.include_router(reports.router)    # router defines prefix="/reports"
    app.include_router(ui.router)         # minimal web UI at /ui (+ /ui/ask)

    # 7) Optional UIs (import *after* app is created to avoid NameError on boot)
    #    - RAG+KG Assistant at /ui/assistant
    try:
        from apps.api.routes.assistant import router as assistant_router
        app.include_router(assistant_router)
    except Exception:
        # Keep API booting even if optional UI is missing
        pass

    #    - KG Explorer (visual) at /ui/kg (optional)
    try:
        from apps.api.routes.kg_explorer import router as kg_explorer_router
        app.include_router(kg_explorer_router)
    except Exception:
        pass

    # 8) Minimal meta endpoints
    @app.get("/", tags=["meta"])
    def root():
        return {
            "name": app.title,
            "version": app.version,
            "env": settings.app_env,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    return app


# Uvicorn/Gunicorn entrypoint:
#   uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
app = create_app()
