# apps/api/routes/chat.py
"""
Chat routes — POST /chat (sync) and WS /stream (streaming).

This file shows exactly how the LangGraph chat pipeline is invoked:
- POST /chat runs the graph once and returns a final answer with sources.
- WS  /chat/stream streams graph node updates (retrieval/analysis/answer).

Auth & RBAC:
- Both endpoints require authentication (Bearer JWT or X-API-KEY).
- Scopes enforced: ["rag:query"]. See apps/api/routes/auth.py for helpers.

LangGraph notes:
- We call a tiny wrapper `get_chat_graph()` from packages.agent_graph.graph.
- The graph should accept a dict-like state with at least:
    { "tenant_id": str, "query": str, "top_k": int }
  and eventually produce:
    { "answer": str, "hits": List[...]}   # "hits" are retrieval results with metadata/scores
- For streaming, we forward each (event, payload) out to the WebSocket.
  Your graph can yield per-node updates (e.g., "retrieve", "analyze", "answer").

This file is intentionally small and explicit (tutorial style).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

# Auth dependencies & scope checks
from .auth import get_principal, require_scopes, Principal

# Observability
from packages.observability.tracing import trace, add_event, set_span_attr

# LangGraph wrapper (kept simple and documented in packages/agent_graph/graph.py)
from packages.agent_graph.graph import get_chat_graph

router = APIRouter()


# ---------------------------
# Schemas (kept local to make the file self-contained)
# If you prefer, move these to apps/api/schemas/chat.py and import here.
# ---------------------------

class ChatRequest(BaseModel):
    tenant_id: str = Field("t0", description="Tenant/organization id")
    query: str = Field(..., description="User question or instruction")
    top_k: int = Field(6, ge=1, le=50, description="Number of passages to retrieve")


class Source(BaseModel):
    score: float = Field(..., example=0.83)
    metadata: Dict[str, Any] = Field(
        ..., example={"source": "docs/10k_2023.pdf", "page": 14, "type": "pdf"}
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Final grounded answer")
    sources: list[Source] = Field(default_factory=list, description="Top passages with metadata")


# ---------------------------
# REST: sync chat
# ---------------------------

@router.post(
    "",
    response_model=ChatResponse,
    summary="Run a single chat turn through the LangGraph pipeline and return final answer",
    dependencies=[Depends(require_scopes(["rag:query"]))],
)
@trace("api.chat.sync")
def chat(
    body: ChatRequest,
    principal: Principal = Depends(get_principal),
):
    """
    Synchronous chat call.
    - Builds/gets the chat graph.
    - Invokes with the provided state.
    - Returns final answer + sources (hits).
    """
    # Attach a few attributes to the current span for better tracing
    set_span_attr("tenant_id", body.tenant_id)
    set_span_attr("top_k", body.top_k)
    set_span_attr("auth_type", principal.auth_type)

    try:
        graph = get_chat_graph()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph init failed: {e}")

    # Minimal, explicit state (the graph is responsible for the rest)
    state = {
        "tenant_id": body.tenant_id,
        "query": body.query,
        "top_k": body.top_k,
    }

    try:
        final_state = graph.invoke(state)  # type: ignore[attr-defined]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph invocation failed: {e}")

    answer = final_state.get("answer") or ""
    hits = final_state.get("hits") or []
    return ChatResponse(answer=answer, sources=hits)


# ---------------------------
# WebSocket: streaming chat
# ---------------------------

@router.websocket(
    "/stream",
    name="Stream graph updates for a chat turn (WebSocket)",
)
@trace("api.chat.stream")
async def stream_chat(
    websocket: WebSocket,
    principal: Principal = Depends(get_principal),  # Auth on WS via headers (Bearer or X-API-KEY)
    _scopes: Principal = Depends(require_scopes(["rag:query"])),  # enforce scopes
):
    """
    WebSocket protocol (very small and friendly):

    Client → Server (JSON):
        {"type": "query", "tenant_id": "t0", "query": "...", "top_k": 6}

    Server → Client (JSON frames):
        {"event": "node", "name": "retrieve", "data": {...}}   # per-node updates
        {"event": "node", "name": "analyze", "data": {...}}
        {"event": "final", "data": {"answer": "...", "sources": [...]}}

    On error:
        {"event": "error", "message": "..."}

    Close:
        normal close code when finished, or client disconnect.
    """
    # Accept the connection *after* dependencies succeeded
    await websocket.accept()

    # Small helper to send JSON consistently
    async def send(event: str, **data):
        payload = {"event": event, **data}
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    try:
        # Receive a single "query" message for this session (simple protocol)
        msg_text = await websocket.receive_text()
        try:
            msg = json.loads(msg_text)
        except Exception:
            await send("error", message="Invalid JSON.")
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        if not isinstance(msg, dict) or msg.get("type") != "query":
            await send("error", message='Expected message with {"type":"query", ...}.')
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        tenant_id = msg.get("tenant_id") or "t0"
        query = msg.get("query") or ""
        top_k = int(msg.get("top_k") or 6)

        if not query:
            await send("error", message="Field 'query' is required.")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Trace context
        set_span_attr("tenant_id", tenant_id)
        set_span_attr("top_k", top_k)
        set_span_attr("auth_type", principal.auth_type)

        try:
            graph = get_chat_graph()
        except Exception as e:
            await send("error", message=f"Graph init failed: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        # Start the graph stream
        state = {"tenant_id": tenant_id, "query": query, "top_k": top_k}

        try:
            # LangGraph .stream yields (event_name, payload) or a richer tuple depending on mode.
            # We default to simple mode="updates" in our graph wrapper for clarity.
            async for event in graph.astream(state):  # type: ignore[attr-defined]
                # Normalize event to a consistent shape
                # Common patterns:
                #   ("node", {"name": "retrieve", "data": {...}})
                #   ("final", {"answer": "...", "hits": [...]})
                # If your graph yields dicts with keys, adapt here minimally.
                if isinstance(event, tuple) and len(event) == 2:
                    evt_name, payload = event
                    if evt_name == "final":
                        data = payload or {}
                        await send("final", data={"answer": data.get("answer", ""), "sources": data.get("hits", [])})
                        break
                    elif evt_name == "node":
                        payload = payload or {}
                        await send("node", name=payload.get("name", "unknown"), data=payload.get("data", {}))
                    else:
                        await send("event", name=str(evt_name), data=payload or {})
                elif isinstance(event, dict) and "final" in event:
                    data = event["final"] or {}
                    await send("final", data={"answer": data.get("answer", ""), "sources": data.get("hits", [])})
                    break
                else:
                    # Fallback passthrough
                    await send("event", name="update", data=event)

            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)

        except WebSocketDisconnect:
            # Client disconnected mid-stream; nothing else to do
            add_event(None, "ws.client_disconnect")  # safe no-op if span missing
        except Exception as e:
            await send("error", message=f"Graph runtime error: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

    except WebSocketDisconnect:
        # Disconnected before sending the first message
        pass
