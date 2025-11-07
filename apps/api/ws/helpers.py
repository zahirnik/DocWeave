# apps/api/ws/helpers.py
"""
WebSocket helpers — tiny utilities for consistent streaming.

Why this file exists
--------------------
FastAPI lets you handle WebSockets directly in route handlers, but a few patterns
repeat across endpoints. These helpers keep those patterns tiny and consistent:

- principal_from_ws(ws): derive a Principal (Bearer or API key) from WS headers.
- WSJSON: a small wrapper to send framed JSON events: .event(), .error(), .final().
- stream_tokens(ws, tokens): send incremental tokens with a simple "token" event.
- tool_event(ws, name, status, **meta): normalized tool-call event frames.

Event envelope (consistent shape)
---------------------------------
Every frame sent by these helpers follows:

    {
      "event": "<type>",            # e.g. "token" | "node" | "tool" | "final" | "error"
      "name": "<optional name>",    # e.g. node/tool name
      "data": { ... }               # payload specific to the event
    }

You can use these helpers in routes like /chat/stream to keep the code minimal.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterable, Iterable, Optional

from fastapi import HTTPException, status
from starlette.websockets import WebSocket

# Typed response model for the auth context
from apps.api.schemas.models import Principal  # reuse API model for consistency


# ---------------------------
# Auth helper for WS
# ---------------------------

async def principal_from_ws(ws: WebSocket) -> Principal:
    """
    Build a Principal from WebSocket headers (Bearer preferred, else X-API-KEY).
    Mirrors the logic in HTTP routes but adapted to WS handshake.

    Headers considered:
      - Authorization: Bearer <jwt>
      - X-API-KEY: <key>
      - X-Tenant-ID: <tenant override> (must match token/key metadata)

    Raises HTTPException(401/403) on invalid/mismatched credentials.
    """
    auth = ws.headers.get("authorization")
    api_key = ws.headers.get("x-api-key")
    x_tenant_id = ws.headers.get("x-tenant-id")

    # Defer imports so this module stays import-light
    from packages.core.auth import decode_access_token, verify_api_key

    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        try:
            claims = decode_access_token(token)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
        tenant_id = claims.get("tenant_id") or "t0"
        if x_tenant_id and x_tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Tenant mismatch for bearer token.")
        return Principal(
            subject=claims.get("sub", "unknown"),
            tenant_id=tenant_id,
            roles=claims.get("roles", []) or [],
            scopes=claims.get("scopes", []) or [],
            auth_type="bearer",
        )

    if api_key:
        try:
            meta = verify_api_key(api_key)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
        tenant_id = meta.get("tenant_id", "t0")
        if x_tenant_id and x_tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Tenant mismatch for API key.")
        return Principal(
            subject=meta.get("key_id", "key_unknown"),
            tenant_id=tenant_id,
            roles=meta.get("roles", []) or [],
            scopes=meta.get("scopes", []) or [],
            auth_type="apikey",
        )

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials.")


# ---------------------------
# JSON framing convenience
# ---------------------------

class WSJSON:
    """A tiny JSON sender with a consistent event envelope."""

    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def accept(self) -> None:
        await self.ws.accept()

    async def event(self, event: str, name: Optional[str] = None, data: Any = None) -> None:
        payload = {"event": event}
        if name is not None:
            payload["name"] = name
        if data is not None:
            payload["data"] = data
        await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def error(self, message: str, code: str = "runtime_error", data: Any = None) -> None:
        await self.event("error", data={"message": message, "code": code, "meta": data or {}})

    async def final(self, data: Any) -> None:
        await self.event("final", data=data)

    async def close(self, code: int = 1000) -> None:
        await self.ws.close(code=code)


# ---------------------------
# Streaming helpers
# ---------------------------

async def stream_tokens(ws: WSJSON, tokens: Iterable[str] | AsyncIterable[str]) -> None:
    """
    Send tokens as they arrive. The client receives frames like:
        {"event":"token","data":{"text":"..."}}   # repeated
    """
    if hasattr(tokens, "__aiter__"):
        async for t in tokens:  # type: ignore
            await ws.event("token", data={"text": str(t)})
    else:
        for t in tokens:  # type: ignore
            await ws.event("token", data={"text": str(t)})


async def tool_event(ws: WSJSON, name: str, status: str, **meta: Any) -> None:
    """
    Send a normalized tool-call event:
        {"event":"tool","name":"tavily.search","data":{"status":"start", ...}}
    Common statuses: "start" | "ok" | "error".
    """
    await ws.event("tool", name=name, data={"status": status, **(meta or {})})
