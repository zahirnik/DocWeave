# packages/core/telemetry.py
"""
Telemetry glue — small helpers to enable LangSmith + tracing in one place.

What this module provides
-------------------------
- configure_observability():
    - Sets up JSON logging.
    - Initializes OpenTelemetry tracing if enabled.
    - Enables LangSmith tracing if configured (by setting env vars that LangChain reads).

- langchain_callbacks():
    - Returns a list of LangChain callbacks (LangSmithTracer when enabled), otherwise [].
      Keep the import lazy so users who don't install langchain/langsmith aren't penalized.

Usage
-----
from packages.core.telemetry import configure_observability, langchain_callbacks
configure_observability()

# pass callbacks into your chains if desired:
cbs = langchain_callbacks()
chain.invoke(inputs, config={"callbacks": cbs})

Environment (see packages.core.config)
--------------------------------------
ENABLE_TRACING=true|false
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318

LANGSMITH_TRACING=true|false
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=convai-finance-agentic-rag

LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
"""

from __future__ import annotations

import os
from typing import List

from packages.core.config import get_settings
from packages.core.logging import setup_json_logging, get_logger
from packages.observability.tracing import init_tracing

log = get_logger(__name__)


def _enable_langsmith_if_configured() -> None:
    """
    Set environment variables that LangChain/LangSmith read at import time.
    We do not import langchain here to avoid heavy deps when unused.
    """
    st = get_settings()
    if st.langsmith_tracing and st.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", st.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", st.langsmith_project)
        log.info("LangSmith tracing enabled for project=%s", st.langsmith_project)
    else:
        # Ensure it's off when not configured (helps tests be deterministic)
        os.environ.setdefault("LANGSMITH_TRACING", "false")


def configure_observability() -> None:
    """
    One-shot initializer to make logs + traces useful in dev and prod.
    - JSON logs to stdout (level from env).
    - OpenTelemetry tracing (best-effort).
    - LangSmith env toggles.
    """
    setup_json_logging(os.getenv("LOG_LEVEL"))
    _enable_langsmith_if_configured()
    init_tracing()


def langchain_callbacks() -> List[object]:
    """
    Return LangChain callbacks if LangSmith tracing is enabled and installed.
    Otherwise return an empty list. Kept optional to avoid forcing heavy deps.
    """
    try:
        if os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes", "on"}:
            from langsmith import Client  # noqa: F401
            from langchain.callbacks.tracers.langchain import LangChainTracer
            return [LangChainTracer()]
    except Exception as e:
        log.info("LangSmith callbacks unavailable: %s", e)
    return []
