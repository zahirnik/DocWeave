# packages/agent_graph/state.py
"""
Typed state + simple checkpointing for the Finance Agent graph.

What this module provides
-------------------------
- ChatMessage: tiny dataclass for chat history rows.
- ToolResult : normalized record for tool outputs (tabular ops, charts, etc.).
- AgentState : canonical in-graph state (serializable).
- Budget     : soft token/time counters (tutorial-clear).
- FileCheckpointer: JSON checkpointing (safe, tiny).

Design goals
------------
- Keep types explicit and serializable (pure dict/list/str/float/bool/None).
- No heavy deps. No globals. Easy to unit-test.
- Fail loudly with helpful messages when required fields are missing.

Typical usage
-------------
from packages.agent_graph.state import AgentState, FileCheckpointer

s = AgentState(
    tenant_id="t0",
    collection="acme_finance",
    query="How did gross margin change in Q2 2024?",
)
s.add_message("user", s.query)

ckpt = FileCheckpointer("./data/checkpoints")
ckpt.save(run_id="demo-1", state=s)
loaded = ckpt.load("demo-1")
"""

from __future__ import annotations

import json
import os
import uuid
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------
# Small helpers
# ---------------------------

def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _truncate(s: str, max_len: int = 10_000) -> str:
    """Defensive truncation to keep state tiny on disk."""
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"\n…[truncated {len(s)-max_len} chars]"

def _clean_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure metadata is JSON-serializable & not huge."""
    if not meta:
        return {}
    try:
        # best-effort: stringify non-serializable values
        json.dumps(meta)
        return meta
    except Exception:
        return {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None), dict, list)) else v)
                for k, v in meta.items()}


# ---------------------------
# Data records
# ---------------------------

@dataclass
class ChatMessage:
    role: str            # "user" | "assistant" | "system" (keep minimal set)
    content: str
    created_at: str = field(default_factory=_utc_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": _truncate(self.content, 12_000),
            "created_at": self.created_at,
            "meta": _clean_metadata(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ChatMessage":
        return ChatMessage(
            role=str(d.get("role") or ""),
            content=str(d.get("content") or ""),
            created_at=str(d.get("created_at") or _utc_iso()),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class ToolResult:
    name: str                 # e.g., "tabular_stats", "chart"
    result: Dict[str, Any]    # arbitrary, but must be JSON-serializable
    created_at: str = field(default_factory=_utc_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "result": _clean_metadata(self.result),
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ToolResult":
        return ToolResult(
            name=str(d.get("name") or ""),
            result=dict(d.get("result") or {}),
            created_at=str(d.get("created_at") or _utc_iso()),
        )


@dataclass
class Budget:
    """Soft budget counters. Use these as *signals*; they do not enforce hard limits."""
    max_tokens: int = 2000
    used_tokens: int = 0
    max_ms: int = 30_000
    used_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "Budget":
        if not d:
            return Budget()
        return Budget(
            max_tokens=int(d.get("max_tokens", 2000)),
            used_tokens=int(d.get("used_tokens", 0)),
            max_ms=int(d.get("max_ms", 30_000)),
            used_ms=int(d.get("used_ms", 0)),
        )

    def add_tokens(self, n: int) -> None:
        self.used_tokens = max(0, int(self.used_tokens) + max(0, int(n)))

    def add_time_ms(self, n: int) -> None:
        self.used_ms = max(0, int(self.used_ms) + max(0, int(n)))


# ---------------------------
# Core state
# ---------------------------

@dataclass
class AgentState:
    # Required identifiers
    tenant_id: str
    collection: str
    query: str

    # Optional filters for retrieval (metadata match)
    filters: Optional[Dict[str, Any]] = None

    # Chat history
    messages: List[ChatMessage] = field(default_factory=list)

    # Retrieved contexts
    contexts: List[Dict[str, Any]] = field(default_factory=list)

    # Tool outputs (optional)
    tool_results: List[ToolResult] = field(default_factory=list)

    # Routing / control
    route: Optional[str] = None

    # Budgets
    budget: Budget = field(default_factory=Budget)

    # Final answer
    answer: Optional[str] = None

    # Run metadata
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)

    # ---- convenience mutators ----

    def touch(self) -> None:
        self.updated_at = _utc_iso()

    def add_message(self, role: str, content: str, **meta: Any) -> None:
        self.messages.append(ChatMessage(role=role, content=content, meta=meta))
        self.touch()

    def add_contexts(self, hits: List[Dict[str, Any]]) -> None:
        # keep top-N small to avoid bloat; caller can decide larger N
        for h in hits:
            # defensively trim text
            if "text" in h:
                h = dict(h)
                h["text"] = _truncate(str(h["text"]), 20_000)
            self.contexts.append(h)
        self.touch()

    def add_tool_result(self, name: str, result: Dict[str, Any]) -> None:
        self.tool_results.append(ToolResult(name=name, result=result))
        self.touch()

    # ---- (de)serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "collection": self.collection,
            "query": self.query,
            "filters": _clean_metadata(self.filters or {}),
            "messages": [m.to_dict() for m in self.messages],
            "contexts": self.contexts,  # assumed serializable by design (text+metadata)
            "tool_results": [t.to_dict() for t in self.tool_results],
            "route": self.route,
            "budget": self.budget.to_dict(),
            "answer": _truncate(self.answer or "", 40_000) if self.answer else None,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentState":
        if not d:
            raise ValueError("State dict is empty")
        msgs = [ChatMessage.from_dict(x) for x in (d.get("messages") or [])]
        tools = [ToolResult.from_dict(x) for x in (d.get("tool_results") or [])]
        return AgentState(
            tenant_id=str(d.get("tenant_id") or ""),
            collection=str(d.get("collection") or ""),
            query=str(d.get("query") or ""),
            filters=dict(d.get("filters") or {}) or None,
            messages=msgs,
            contexts=list(d.get("contexts") or []),
            tool_results=tools,
            route=d.get("route") or None,
            budget=Budget.from_dict(d.get("budget")),
            answer=d.get("answer") or None,
            run_id=str(d.get("run_id") or uuid.uuid4().hex),
            created_at=str(d.get("created_at") or _utc_iso()),
            updated_at=str(d.get("updated_at") or _utc_iso()),
        )

    # ---- validations ----

    def validate(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id is required")
        if not self.collection:
            raise ValueError("collection is required")
        if not (self.query or "").strip():
            raise ValueError("query is required")

    # ---- pretty debug ----

    def short(self) -> str:
        return (
            f"AgentState(run_id={self.run_id}, tenant={self.tenant_id}, coll={self.collection}, "
            f"route={self.route}, msgs={len(self.messages)}, ctx={len(self.contexts)}, "
            f"tools={len(self.tool_results)})"
        )


# ---------------------------
# Checkpointing
# ---------------------------

class FileCheckpointer:
    """
    JSON checkpointing on local disk. One file per run_id:
      <root>/<tenant>/<collection>/<run_id>.json

    Notes
    -----
    - Atomic writes via temp file + rename.
    - Safe to call from a single-process app (tutorial scope).
    - For multi-process / production, consider Redis or a DB table.
    """

    def __init__(self, root_dir: str = "./data/checkpoints"):
        self.root_dir = root_dir
        _ensure_dir(self.root_dir)

    def _path(self, tenant_id: str, collection: str, run_id: str) -> str:
        d = os.path.join(self.root_dir, tenant_id, collection)
        _ensure_dir(d)
        return os.path.join(d, f"{run_id}.json")

    # ---- public API ----

    def save(self, run_id: str, state: AgentState) -> str:
        state.validate()
        path = self._path(state.tenant_id, state.collection, run_id)
        tmp = path + ".tmp"
        data = state.to_dict()
        data["saved_at"] = _utc_iso()
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return path

    def load(self, run_id: str, *, tenant_id: str | None = None, collection: str | None = None) -> AgentState:
        """
        If tenant_id/collection not provided, scans directories to find the first matching run_id.
        """
        path = None
        if tenant_id and collection:
            cand = self._path(tenant_id, collection, run_id)
            if os.path.exists(cand):
                path = cand
        else:
            # scan (small tutorial repo; OK to walk a few dirs)
            for t in os.listdir(self.root_dir):
                tdir = os.path.join(self.root_dir, t)
                if not os.path.isdir(tdir):
                    continue
                for c in os.listdir(tdir):
                    cdir = os.path.join(tdir, c)
                    if not os.path.isdir(cdir):
                        continue
                    cand = os.path.join(cdir, f"{run_id}.json")
                    if os.path.exists(cand):
                        path = cand
                        break
                if path:
                    break

        if not path:
            raise FileNotFoundError(f"checkpoint not found for run_id={run_id}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AgentState.from_dict(data)

    def list_runs(self, *, tenant_id: Optional[str] = None, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return a list of {"tenant_id","collection","run_id","updated_at","path"}.
        """
        out: List[Dict[str, Any]] = []
        base = self.root_dir
        tenants = [tenant_id] if tenant_id else [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        for t in tenants:
            tdir = os.path.join(base, t)
            if not os.path.isdir(tdir):
                continue
            colls = [collection] if collection else [d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d))]
            for c in colls:
                cdir = os.path.join(tdir, c)
                if not os.path.isdir(cdir):
                    continue
                for fn in os.listdir(cdir):
                    if not fn.endswith(".json"):
                        continue
                    p = os.path.join(cdir, fn)
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        out.append({
                            "tenant_id": t,
                            "collection": c,
                            "run_id": data.get("run_id") or fn[:-5],
                            "updated_at": data.get("updated_at"),
                            "path": p,
                        })
                    except Exception:
                        out.append({
                            "tenant_id": t,
                            "collection": c,
                            "run_id": fn[:-5],
                            "updated_at": None,
                            "path": p,
                        })
        # Sort newest first
        out.sort(key=lambda x: (x.get("updated_at") or ""), reverse=True)
        return out
