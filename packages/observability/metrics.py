# packages/observability/metrics.py
"""
Lightweight metrics — Prometheus-style counters/histograms with graceful fallback.

What this module provides
-------------------------
- init_metrics(namespace="convai_finance_rag") -> bool
    Try to initialise Prometheus metrics using `prometheus_client`. Returns True if
    the real client is available; else a no-op in-memory fallback is used.

- counter(name, documentation, labelnames=()) -> Counter
- histogram(name, documentation, labelnames=(), buckets=None) -> Histogram
    Return metric objects with `.labels(...).inc(value)` and `.labels(...).observe(value)`.

- export_prometheus_text() -> str
    Export current metrics in Prometheus text format (works with fallback too).

- register_fastapi_endpoint(app, path="/metrics")
    Add a GET endpoint that returns the metrics text (content-type text/plain).

Design goals
------------
- Tutorial-clear; single file; tiny API surface.
- If `prometheus_client` is missing, an in-memory shim mimics the interface sufficiently
  for basic testing and local development.
- Label values are sanitised lightly to avoid exploding the cardinality by whitespace.

Typical usage
-------------
from packages.observability.metrics import init_metrics, counter, histogram, register_fastapi_endpoint

init_metrics()
REQS = counter("api_requests_total", "Total API requests", labelnames=("route","status"))
LAT  = histogram("api_latency_seconds", "Request latency (s)", labelnames=("route",), buckets=[0.05,0.1,0.25,0.5,1,2,5])

# in your middleware/route:
REQS.labels(route="/chat", status="200").inc(1)
LAT.labels(route="/chat").observe(0.123)
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # soft dependency
    from prometheus_client import (
        Counter as _PromCounter,
        Histogram as _PromHistogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY as _DEFAULT_REGISTRY,
    )
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    _PROM_AVAILABLE = False
    _PromCounter = object  # type: ignore
    _PromHistogram = object  # type: ignore
    CollectorRegistry = object  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    def generate_latest(_: Any = None) -> bytes:  # type: ignore
        return b""


# ---------------------------
# Globals
# ---------------------------

_NAMESPACE = "convai_finance_rag"
_REGISTRY = None  # real prometheus registry or None
_LOCK = threading.RLock()

# Caches for created metrics (name -> metric)
_COUNTERS: Dict[str, Any] = {}
_HISTOS: Dict[str, Any] = {}


# ---------------------------
# Helpers
# ---------------------------

def _norm_label_value(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    # clamp length; replace inner whitespace with single space
    s = " ".join(s.split())
    if len(s) > 120:
        s = s[:117] + "..."
    return s or "none"

def _metric_key(name: str, labelnames: Iterable[str]) -> Tuple[str, Tuple[str, ...]]:
    return (name, tuple(labelnames or ()))


# ---------------------------
# Fallback classes (no prometheus_client)
# ---------------------------

class _ShimChild:
    def __init__(self, parent: "._ShimMetric", labels: Tuple[str, ...], values: Tuple[str, ...]):
        self._parent = parent
        self._labels = labels
        self._values = values

    def inc(self, n: float = 1.0) -> None:
        with _LOCK:
            key = (self._labels, self._values)
            self._parent._store[key] = self._parent._store.get(key, 0.0) + float(n)

    def observe(self, x: float) -> None:
        with _LOCK:
            key = (self._labels, self._values)
            bucketed = self._parent._store.setdefault(key, [])
            bucketed.append(float(x))


class _ShimMetric:
    def __init__(self, name: str, doc: str, labelnames: Iterable[str], typ: str):
        self.name = name
        self.doc = doc
        self.typ = typ  # "counter" | "histogram"
        self.labelnames = tuple(labelnames or ())
        self._store: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Any] = {}

    def labels(self, *values: Any, **kw: Any) -> _ShimChild:
        if kw:
            # support labels(a=..., b=...) in any order
            values = tuple(_norm_label_value(kw.get(k)) for k in self.labelnames)
        else:
            values = tuple(_norm_label_value(v) for v in values)
        # sanity: length must match
        if len(values) != len(self.labelnames):
            raise ValueError("label value count does not match label names")
        return _ShimChild(self, self.labelnames, values)


def _shim_export_text(namespace: str) -> str:
    """
    Very small Prometheus text encoder for our shim metrics.
    Not complete, but good enough for local testing.
    """
    lines: List[str] = []
    # counters
    for m in _COUNTERS.values():
        assert isinstance(m, _ShimMetric)
        lines.append(f"# HELP {namespace}_{m.name} {m.doc}")
        lines.append(f"# TYPE {namespace}_{m.name} counter")
        for (labels, values), total in sorted(m._store.items()):
            if not isinstance(total, (int, float)):
                continue
            lbl = ",".join(f'{k}="{v}"' for k, v in zip(labels, values))
            lines.append(f'{namespace}_{m.name}{{{lbl}}} {total}')
    # histograms (we emit simple summary: count and sum; no buckets)
    for m in _HISTOS.values():
        assert isinstance(m, _ShimMetric)
        lines.append(f"# HELP {namespace}_{m.name} {m.doc}")
        lines.append(f"# TYPE {namespace}_{m.name} summary")
        for (labels, values), arr in sorted(m._store.items()):
            if not isinstance(arr, list):
                continue
            count = len(arr)
            total = sum(arr) if arr else 0.0
            lbl = ",".join(f'{k}="{v}"' for k, v in zip(labels, values))
            lines.append(f'{namespace}_{m.name}_count{{{lbl}}} {count}')
            lines.append(f'{namespace}_{m.name}_sum{{{lbl}}} {total}')
    return "\n".join(lines) + "\n"


# ---------------------------
# Public API
# ---------------------------

def init_metrics(namespace: str = "convai_finance_rag") -> bool:
    """
    Initialise metrics system. Call this once early in your app.

    Returns True if `prometheus_client` is available and active.
    """
    global _REGISTRY, _NAMESPACE
    _NAMESPACE = (namespace or "convai_finance_rag").strip().lower().replace("-", "_")
    if _PROM_AVAILABLE:
        # Use default global registry unless tests want their own
        _REGISTRY = _DEFAULT_REGISTRY
        return True
    _REGISTRY = None
    return False


def counter(name: str, documentation: str, *, labelnames: Iterable[str] = ()) -> Any:
    """
    Create or get a Counter metric.
    """
    key = _metric_key(name, labelnames)
    if key in _COUNTERS:
        return _COUNTERS[key]

    if _PROM_AVAILABLE:
        m = _PromCounter(f"{_NAMESPACE}_{name}", documentation, labelnames=tuple(labelnames or ()))
    else:
        m = _ShimMetric(name, documentation, tuple(labelnames or ()), "counter")
    _COUNTERS[key] = m
    return m


def histogram(
    name: str,
    documentation: str,
    *,
    labelnames: Iterable[str] = (),
    buckets: Optional[Iterable[float]] = None,
) -> Any:
    """
    Create or get a Histogram metric.
    """
    key = _metric_key(name, labelnames)
    if key in _HISTOS:
        return _HISTOS[key]

    if _PROM_AVAILABLE:
        if buckets is not None:
            m = _PromHistogram(f"{_NAMESPACE}_{name}", documentation, labelnames=tuple(labelnames or ()), buckets=tuple(buckets))
        else:
            m = _PromHistogram(f"{_NAMESPACE}_{name}", documentation, labelnames=tuple(labelnames or ()))
    else:
        m = _ShimMetric(name, documentation, tuple(labelnames or ()), "histogram")
    _HISTOS[key] = m
    return m


def export_prometheus_text() -> str:
    """
    Export metrics as Prometheus text (works with real client or shim).
    """
    if _PROM_AVAILABLE and _REGISTRY is not None:
        return generate_latest(_REGISTRY).decode("utf-8", errors="replace")
    return _shim_export_text(_NAMESPACE)


def register_fastapi_endpoint(app, path: str = "/metrics") -> None:
    """
    Register a GET /metrics endpoint on a FastAPI app.
    """
    try:
        from fastapi import Response
    except Exception:
        raise RuntimeError("FastAPI not installed; cannot register /metrics endpoint")

    @app.get(path)
    def _metrics_endpoint() -> Any:
        body = export_prometheus_text()
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)


# ---------------------------
# Predefined, common metrics
# ---------------------------

# These are optional convenience metrics you can reuse across the app
_REQ_COUNTER = None
_LAT_HISTO = None

def get_request_counter():
    global _REQ_COUNTER
    if _REQ_COUNTER is None:
        _REQ_COUNTER = counter(
            "requests_total",
            "Total HTTP requests",
            labelnames=("route", "method", "status"),
        )
    return _REQ_COUNTER

def get_latency_histogram():
    global _LAT_HISTO
    if _LAT_HISTO is None:
        _LAT_HISTO = histogram(
            "request_latency_seconds",
            "HTTP request latency in seconds",
            labelnames=("route", "method"),
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
        )
    return _LAT_HISTO


# ---------------------------
# Tiny middleware helper (FastAPI)
# ---------------------------

def fastapi_metrics_middleware():
    """
    Return a FastAPI-compatible middleware factory that measures request count & latency.

    Example
    -------
    app = FastAPI()
    init_metrics()
    app.middleware("http")(fastapi_metrics_middleware())
    register_fastapi_endpoint(app)
    """
    def _mw(request, call_next):
        start = time.perf_counter()
        route = getattr(getattr(request, "scope", {}), "get", lambda *_: None)("path") if hasattr(request, "scope") else None
        route = route or request.url.path  # type: ignore[attr-defined]
        method = getattr(request, "method", "GET")
        try:
            response = None
            response = call_next(request)
            # If call_next is async (FastAPI), wrap accordingly
            if hasattr(response, "__await__"):
                async def _async():
                    resp = await response
                    status = getattr(resp, "status_code", 200)
                    get_request_counter().labels(route=route, method=method, status=str(status)).inc(1)
                    get_latency_histogram().labels(route=route, method=method).observe(time.perf_counter() - start)
                    return resp
                return _async()
            else:
                resp = response
                status = getattr(resp, "status_code", 200)
                get_request_counter().labels(route=route, method=method, status=str(status)).inc(1)
                get_latency_histogram().labels(route=route, method=method).observe(time.perf_counter() - start)
                return resp
        except Exception:
            get_request_counter().labels(route=route, method=method, status="500").inc(1)
            get_latency_histogram().labels(route=route, method=method).observe(time.perf_counter() - start)
            raise
    return _mw
