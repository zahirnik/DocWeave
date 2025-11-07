# packages/agent_graph/memory/chat_memory.py
"""
Chat memory — short vs long memory for the Finance Agent.

What this module provides
-------------------------
- ChatTurn: tiny dataclass for a single chat message (role/content/metadata).
- ChatMemory: simple, explicit memory manager with:
    • add(role, content, **meta)            → append a turn
    • recent(n: int | None)                 → last N turns (default = window)
    • window()                              → sliding window (short-term)
    • long_read() / long_write(summary)     → persist/retrieve long-term summary
    • summarize_recent(llm="openai", ...)   → turn the last K turns into a summary (optional)

Design goals
------------
- Keep tutorial-clear: no hidden global state, no DB dependency.
- File-based persistence for long memory (JSON), safe and explicit.
- Work without any LLM packages; summarization becomes optional.

Typical usage
-------------
mem = ChatMemory(tenant_id="t0", session_id="run-123", persist_dir="./data/memory", window_turns=8)
mem.add("user", "Plot 2023–2024 quarterly revenue for ACME.")
mem.add("assistant", "Do you have the CSV file, or should I look it up?")
print(mem.window())     # short-term context
mem.long_write("User often asks for YoY charts for ACME and prefers PNG outputs.")
print(mem.long_read())  # -> that string

Summarization (optional)
------------------------
txt = mem.summarize_recent(max_tokens=300)  # uses OpenAI if OPENAI_API_KEY is set
"""

from __future__ import annotations

import json
import os
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

def _truncate(s: str, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else (s[: n - 1] + "…")


# ---------------------------
# Data record
# ---------------------------

@dataclass
class ChatTurn:
    role: str                 # "user" | "assistant" | "system"
    content: str
    created_at: str = field(default_factory=_utc_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # defensively clamp huge content to keep the memory light
        d["content"] = _truncate(d["content"], 20_000)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ChatTurn":
        return ChatTurn(
            role=str(d.get("role") or ""),
            content=str(d.get("content") or ""),
            created_at=str(d.get("created_at") or _utc_iso()),
            meta=dict(d.get("meta") or {}),
        )


# ---------------------------
# Memory manager
# ---------------------------

class ChatMemory:
    """
    Chat memory with:
      - Short-term window: last `window_turns` messages.
      - Long-term: a single persisted summary per (tenant_id, session_id).

    Persistence layout (file-based):
      <persist_dir>/<tenant_id>/<session_id>/long.json
        {
          "tenant_id": "...",
          "session_id": "...",
          "summary": "free text",
          "updated_at": "iso"
        }
    """

    def __init__(
        self,
        tenant_id: str,
        session_id: str,
        *,
        window_turns: int = 8,
        persist_dir: str = "./data/memory",
    ):
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not session_id:
            raise ValueError("session_id is required")

        self.tenant_id = tenant_id
        self.session_id = session_id
        self.window_turns = max(1, int(window_turns))
        self.persist_dir = persist_dir

        self._turns: List[ChatTurn] = []

    # ---- add / read ----

    def add(self, role: str, content: str, **meta: Any) -> None:
        """
        Append a new turn to memory.
        """
        self._turns.append(ChatTurn(role=role, content=content, meta=meta))

    def recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return the last `n` turns (default = window_turns).
        """
        k = self.window_turns if n is None else max(1, int(n))
        return [t.to_dict() for t in self._turns[-k:]]

    def window(self) -> List[Dict[str, Any]]:
        """
        Return the sliding short-term window (alias to recent()).
        """
        return self.recent(self.window_turns)

    def all(self) -> List[Dict[str, Any]]:
        """
        Return all turns (defensively truncated per-turn).
        """
        return [t.to_dict() for t in self._turns]

    # ---- long memory persistence ----

    def _long_path(self) -> str:
        d = os.path.join(self.persist_dir, self.tenant_id, self.session_id)
        _ensure_dir(d)
        return os.path.join(d, "long.json")

    def long_write(self, summary: str) -> str:
        """
        Persist/replace a long-term summary string for this (tenant, session).
        """
        path = self._long_path()
        tmp = path + ".tmp"
        data = {
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "summary": _truncate(summary or "", 100_000),
            "updated_at": _utc_iso(),
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return path

    def long_read(self) -> Optional[str]:
        """
        Load the long-term summary if present; else None.
        """
        path = self._long_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return str(data.get("summary") or "")
        except Exception:
            return None

    # ---- summarization (optional) ----

    def summarize_recent(
        self,
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        max_tokens: int = 300,
        temperature: float = 0.2,
        include_long: bool = True,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Summarize the last `window_turns` messages into a stable, reusable memory note.

        Returns a string. If the provider is unavailable (e.g., no API key), a
        fallback extractive summary is returned instead.

        provider="openai" requires OPENAI_API_KEY in env.
        """
        # Build a compact conversation transcript
        recent = self.window()
        lines = []
        for turn in recent:
            role = turn["role"]
            content = (turn.get("content") or "").strip()
            lines.append(f"{role.upper()}: {content}")
        transcript = "\n".join(lines)
        prior = (self.long_read() or "") if include_long else ""

        sys = system_prompt or (
            "You are a concise memory writer for a finance-analytics assistant.\n"
            "Extract stable, reusable facts, preferences, or objectives that will help future turns.\n"
            "Avoid transient details (exact numbers unless persistent), avoid personal data.\n"
            "Write 2–6 bullet points. British English."
        )
        user = (
            f"PRIOR MEMORY (optional):\n{prior or '(none)'}\n\n"
            f"RECENT TURNS (most recent last):\n{transcript}\n\n"
            "Write an updated memory. If nothing durable, return 'No stable memory to add.'"
        )

        if provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
                import os as _os
                if not _os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY missing")
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                text = (resp.choices[0].message.content or "").strip()
                return text
            except Exception:
                # graceful fallback to extractive summary
                pass

        # Fallback: extract key sentences heuristically (no network)
        return _fallback_extractive_summary(recent)


# ---------------------------
# Fallback summariser (no LLM)
# ---------------------------

def _fallback_extractive_summary(turns: List[Dict[str, Any]], *, max_points: int = 5) -> str:
    """
    Very small heuristic summary:
      - Prefer the latest user and assistant turns.
      - Extract sentences mentioning preferences, objectives, or files.
    """
    # Very light keyword-based extraction
    KEYWORDS = (
        "prefer", "like", "want", "objective", "goal",
        "file", "csv", "xlsx", "pdf", "chart", "plot",
        "yoy", "qoq", "quarterly", "annual", "ticker",
    )
    sents: List[str] = []
    for t in reversed(turns):  # most recent first
        content = (t.get("content") or "")
        parts = [p.strip() for p in content.replace("\n", " ").split(".") if p.strip()]
        for p in parts:
            low = p.lower()
            if any(k in low for k in KEYWORDS):
                sents.append(p)
            if len(sents) >= max_points:
                break
        if len(sents) >= max_points:
            break

    if not sents:
        return "No stable memory to add."
    bullets = "\n".join(f"• {x}" for x in sents[:max_points])
    return bullets


__all__ = ["ChatTurn", "ChatMemory"]
