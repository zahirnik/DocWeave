# apps/api/middlewares/request_id.py
"""
Request-ID middleware (tiny and explicit).

Purpose
-------
- Ensures every request/response carries a stable `x-request-id` header.
- If the client sends `X-Request-ID` or `X-Correlation-ID`, we honor it.
- If none is provided, we generate a UUIDv4 and attach it.
- Stores the id in a context variable for logs/traces to pick up.

Why this matters
----------------
- Makes log lines traceable across services.
- Lets you correlate API, worker, and database events by a single id.
- Plays nicely with OpenTelemetry: we set a span attribute when available.

Usage
-----
In `apps/api/main.py`:
    app.middleware("http")(request_id_middleware)

In your code (optional), to access current id:
    from apps.api.middlewares.request_id import get_request_id
    rid = get_request_id()  # may be None outside of a request context
"""

from __future__ import annotations

import uuid
import contextvars
from typing import Optional

from starlette.requests import Request
from starlette.responses import Response

# Context variable to carry the id across the request lifetime
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> Optional[str]:
    """Return the current request id if set (None outside of a request context)."""
    return _request_id_var.get()


async def request_id_middleware(request: Request, call_next):
    """
    Starlette/FastAPI middleware function.
    - Reads incoming correlation headers (case-insensitive).
    - Sets a request id in context for downstream code.
    - Adds `x-request-id` to the response headers.
    - Attaches `request_id` to the current trace span (best-effort).
    """
    # Prefer client-provided correlation id if present
    rid = (
        request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
        or uuid.uuid4().hex
    )

    # Store in context var for the lifetime of this request
    token = _request_id_var.set(rid)

    # Best-effort: add as a tracing attribute if tracing is enabled
    try:
        from packages.observability.tracing import set_span_attr  # lazy import
        set_span_attr("request_id", rid)
    except Exception:
        # Tracing is optional; never fail the request
        pass

    try:
        # Call downstream handlers
        response: Response = await call_next(request)
    finally:
        # Restore previous context (important when reusing event loops)
        _request_id_var.reset(token)

    # Always include the id on the way out
    response.headers["x-request-id"] = rid
    # Also mirror as x-correlation-id for tooling that expects that name
    response.headers.setdefault("x-correlation-id", rid)

    return response
