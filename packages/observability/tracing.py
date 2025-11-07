# packages/observability/tracing.py
"""
Tiny tracing helpers — OpenTelemetry + (optional) LangSmith, with graceful no-ops.

What this module provides
-------------------------
- init_tracing(service_name="convai-finance-agentic-rag") -> bool
    Initialise OpenTelemetry OTLP exporter if the SDK is installed and an endpoint
    is configured. Returns True if tracing is active, else False.

- get_tracer(name: str)
    Return an OpenTelemetry tracer if available, otherwise a tiny no-op tracer.

- trace(span_name: str)
    Decorator to trace a function. Adds status on exception. **Preserves the original
    function signature** so FastAPI can still infer parameters correctly.

- start_span(span_name: str)
    Context manager to trace arbitrary code blocks.

- add_span_attr(key, value), record_event(name, attrs=None)
    Convenience helpers to annotate the current span (no-op if tracing disabled).

Environment variables (read at init)
------------------------------------
- OTEL_EXPORTER_OTLP_ENDPOINT   (e.g., "http://otel-collector:4318")
- OTEL_EXPORTER_OTLP_HEADERS    (optional, e.g., "Authorization=Bearer <token>")
- OTEL_SERVICE_NAME             (overrides service_name argument)
- LANGSMITH_API_KEY             (enables LangSmith traces when present)
- LANGSMITH_PROJECT             (optional override; default "convai-finance-agentic-rag")
- LANGSMITH_TRACING             (set to "true" automatically when API key present)

Design goals
------------
- Tutorial-clear. If the OpenTelemetry SDK is missing, everything still runs as no-op.
- Single file; easy to audit and test.
"""

from __future__ import annotations

import os
import time
import contextlib
from typing import Any, Callable, Dict, Optional
from functools import wraps
import inspect

# ---------------------------
# Optional OpenTelemetry setup
# ---------------------------

_OTEL_AVAILABLE = False
_otel_trace = None

try:  # soft import (keeps this file dependency-light)
    from opentelemetry import trace as _otel_trace  # type: ignore
    from opentelemetry.trace.status import Status, StatusCode  # type: ignore
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, OTLPSpanExporter  # type: ignore
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    _OTEL_AVAILABLE = False
    Status = object  # type: ignore
    StatusCode = object  # type: ignore


# ---------------------------
# No-op tracer (fallback)
# ---------------------------

class _NoopSpan:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def set_attribute(self, *_: Any, **__: Any): pass
    def add_event(self, *_: Any, **__: Any): pass
    def set_status(self, *_: Any, **__: Any): pass

class _NoopTracer:
    def start_as_current_span(self, *_: Any, **__: Any): return _NoopSpan()
    def start_span(self, *_: Any, **__: Any): return _NoopSpan()


# Global tracer reference (no-op by default)
_TRACER = _NoopTracer()


def _maybe_init_langsmith() -> None:
    """
    If LANGSMITH_API_KEY is set, ensure LangSmith tracing is enabled for LangChain.
    This module doesn't import LangChain; we just set env toggles.
    """
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "convai-finance-agentic-rag"))


def init_tracing(service_name: str = "convai-finance-agentic-rag") -> bool:
    """
    Initialise OpenTelemetry exporter if available and endpoint configured.
    Safe to call multiple times. Returns True if OTEL tracing is active.
    """
    global _TRACER

    _maybe_init_langsmith()

    if not _OTEL_AVAILABLE:
        _TRACER = _NoopTracer()
        return False

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        # No collector configured → keep no-op (but return a tracer object)
        _TRACER = _otel_trace.get_tracer(service_name or "app")  # still returns a tracer, but no provider/exporter
        return False

    # Allow env override
    svc = os.getenv("OTEL_SERVICE_NAME", service_name)

    # Create provider with resource attrs
    resource = Resource.create({"service.name": svc})
    provider = TracerProvider(resource=resource)

    # OTLP/HTTP exporter (works with most collectors)
    headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)

    # Batch processor for efficiency
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Register provider globally
    _otel_trace.set_tracer_provider(provider)
    _TRACER = _otel_trace.get_tracer(svc)
    return True


def get_tracer(name: str = "app"):
    """Return the active tracer (OpenTelemetry or no-op)."""
    return _TRACER


# ---------------------------
# Decorator & context manager
# ---------------------------

def trace(span_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator: wrap a function in a tracing span.

    IMPORTANT: Uses functools.wraps and copies the original signature to ensure frameworks
    like FastAPI see the correct parameters (avoids 422 errors with phantom args/kwargs).
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            tracer = get_tracer(fn.__module__ or "app")
            with tracer.start_as_current_span(span_name) as sp:
                try:
                    sp.set_attribute("code.function", fn.__name__)
                    return fn(*args, **kwargs)
                except Exception as e:  # annotate the span, then re-raise
                    if _OTEL_AVAILABLE:
                        sp.set_status(Status(StatusCode.ERROR, description=str(e)))  # type: ignore
                    raise
        # Ensure FastAPI/inspect can see the original signature
        try:
            wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        except Exception:
            pass
        return wrapper
    return deco


@contextlib.contextmanager
def start_span(span_name: str):
    """
    Context manager to trace an arbitrary block.

    Example
    -------
    with start_span("retriever.search"):
        hits = retriever.search(q)
    """
    tracer = get_tracer("app")
    with tracer.start_as_current_span(span_name) as sp:
        yield sp


# ---------------------------
# Small helpers
# ---------------------------

def add_span_attr(key: str, value: Any) -> None:
    """
    Attach an attribute to the current span (no-op if tracing disabled).
    """
    try:
        if _OTEL_AVAILABLE and _otel_trace is not None:
            sp = _otel_trace.get_current_span()
            if sp is not None:
                sp.set_attribute(key, value)
    except Exception:
        pass  # never fail user code


def record_event(name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
    """
    Add an event to the current span (no-op if tracing disabled).
    """
    try:
        if _OTEL_AVAILABLE and _otel_trace is not None:
            sp = _otel_trace.get_current_span()
            if sp is not None:
                sp.add_event(name, attributes=attrs or {})
    except Exception:
        pass  # never fail user code


# ---------------------------
# Compatibility shims (keep other modules happy)
# ---------------------------

def add_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Alias of record_event(name, attrs=attributes).
    Provided for compatibility with modules that import `add_event`.
    """
    record_event(name, attrs=attributes)


def set_span_attr(key: str, value: Any) -> None:
    """
    Alias of add_span_attr(key, value).
    Provided for compatibility with modules that import `set_span_attr`.
    """
    add_span_attr(key, value)


__all__ = [
    "init_tracing",
    "get_tracer",
    "trace",
    "start_span",
    "add_span_attr",
    "record_event",
    # shims
    "add_event",
    "set_span_attr",
]
