# packages/agent_graph/policies.py
"""
Guardrails & retry policies for the Finance Agent graph.

What this module provides
-------------------------
- PolicyConfig: central knobs controlling **tool budgets**, **LLM retries**, and **safety**.
- with_retries(func): tiny decorator to add exponential backoff to any callable.
- sanitize_tool_args(args): bounds-check inputs passed to tools; trims large strings.
- is_domain_allowed(url, cfg): quick allow/deny for outbound web requests (Tavily/Bing/etc.).
- can_call_tool(state, tool_name, cfg): soft quota checks per turn/run; returns (ok, reason).
- redact_for_prompt(text): best-effort PII redaction for prompts/logs (uses security/pii if available).
- enforce_policies_or_raise(state, cfg): lightweight checks before running a graph step.

Design goals
------------
- Keep **tutorial-clear** and dependency-light.
- Fail fast with friendly errors; keep defaults conservative.
- Centralize knobs so the rest of the code stays clean.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from packages.core.logging import get_logger

log = get_logger(__name__)

# ---------------------------
# Configuration
# ---------------------------

@dataclass
class PolicyConfig:
    # --- Budgets / quotas ---
    max_tool_calls_per_turn: int = 4
    max_total_tool_calls: int = 24
    max_llm_tokens_per_turn: int = 4000
    max_retry_tool: int = 2
    max_retry_llm: int = 2

    # --- Web / file permissions (keep conservative) ---
    allow_web: bool = True
    allowed_domains: Iterable[str] = field(default_factory=lambda: (
        # common finance / docs; adjust in configs/policies.yaml if needed
        "sec.gov", "investor.apple.com", "ft.com", "reuters.com", "bloomberg.com",
        "ecb.europa.eu", "bankofengland.co.uk",
    ))
    banned_domains: Iterable[str] = field(default_factory=lambda: (
        "pastebin.com", "anonfiles.com", "mega.nz", "ipfs.io", "dark.*"
    ))
    allow_file_write: bool = False          # charting/tabular tools save into ./data/outputs internally; no arbitrary writes
    allow_code_exec: bool = False           # disallow arbitrary python exec as a "tool"
    allow_shell: bool = False

    # --- Content safety (very small guardrails; keep policy engine for detailed rules) ---
    blocked_patterns: Iterable[str] = field(default_factory=lambda: (
        r"(?i)api[_\- ]?key\s*[:=]\s*[A-Za-z0-9_\-]{16,}",
        r"(?i)password\s*[:=]",
        r"(?i)secret\s*[:=]",
    ))

    # --- Size limits for tool arguments ---
    max_json_arg_bytes: int = 200_000       # ~200 KB per tool call
    max_string_field_len: int = 50_000      # bound long text fields
    trim_suffix: str = "\n…[trimmed by policies]"

    # --- Networking knobs for web search tools ---
    user_agent: str = "convai-finance-agentic-rag/1.0 (+educational)"

    @staticmethod
    def from_env() -> "PolicyConfig":
        # Kept simple; for full env binding, mirror configs/policies.yaml
        return PolicyConfig()


# ---------------------------
# Simple retry decorator
# ---------------------------

def with_retries(max_tries: int = 2, base_sleep_s: float = 0.5, max_sleep_s: float = 4.0):
    """
    Add exponential backoff retries to a callable. Works for sync callables only.

    Example:
        @with_retries(max_tries=3)
        def call_llm(...): ...
    """
    def deco(fn: Callable[..., Any]):
        def _wrapped(*args, **kwargs):
            tries = max(1, int(max_tries))
            last_err = None
            for attempt in range(tries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # pragma: no cover - exercised at runtime
                    last_err = e
                    if attempt >= tries - 1:
                        break
                    sleep = min(max_sleep_s, base_sleep_s * (2 ** attempt))
                    time.sleep(sleep)
            # Re-raise with context
            raise RuntimeError(f"{fn.__name__} failed after {tries} attempts: {last_err}") from last_err
        return _wrapped
    return deco


# ---------------------------
# PII redaction (best-effort)
# ---------------------------

def redact_for_prompt(text: str) -> str:
    """
    Best-effort PII masking for prompts/logs.
    Delegates to packages.security.pii if available, then applies simple regexes.
    """
    if not text:
        return text
    try:
        from packages.security.pii import mask_pii  # tiny helper in security/pii.py
        text = mask_pii(text)
    except Exception:
        pass

    # Very conservative mask of obvious secrets / long tokens
    text = re.sub(r"(?i)(api[_\- ]?key\s*[:=]\s*)([A-Za-z0-9_\-]{12,})", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(password\s*[:=]\s*)(\S+)", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(secret\s*[:=]\s*)(\S+)", r"\1[REDACTED]", text)
    return text


# ---------------------------
# Tool-argument sanitation
# ---------------------------

def _truncate_str(s: str, limit: int, suffix: str) -> str:
    if len(s) <= limit:
        return s
    head = max(0, limit - len(suffix))
    return s[:head] + suffix

def _json_size(b: Any) -> int:
    try:
        return len(json.dumps(b, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return math.inf

def sanitize_tool_args(args: Any, cfg: Optional[PolicyConfig] = None) -> Any:
    """
    Enforce size limits to avoid accidentally shipping huge payloads to tools.
    - Trims long strings in dict/list.
    - If JSON-serialized size exceeds cfg.max_json_arg_bytes, raises.
    """
    cfg = cfg or PolicyConfig.from_env()
    max_str = int(cfg.max_string_field_len)

    def _walk(x: Any) -> Any:
        if isinstance(x, str):
            return _truncate_str(x, max_str, cfg.trim_suffix)
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_walk(v) for v in x)
        return x

    clean = _walk(args)
    size = _json_size(clean)
    if size > cfg.max_json_arg_bytes:
        raise ValueError(f"tool args too large ({size} bytes > {cfg.max_json_arg_bytes})")
    return clean


# ---------------------------
# Domain allow/deny helpers
# ---------------------------

def _normalize_domain(host: str) -> str:
    return host.lower().strip().lstrip("www.")

def is_domain_allowed(url: str, cfg: Optional[PolicyConfig] = None) -> bool:
    """
    Return True if URL's domain is permitted per policy.
    """
    cfg = cfg or PolicyConfig.from_env()
    try:
        p = urllib.parse.urlparse(url)
        host = _normalize_domain(p.hostname or "")
        if not host:
            return False
        # banned takes precedence
        for pat in cfg.banned_domains:
            pat_s = str(pat).lower().strip()
            if pat_s.endswith(".*"):
                if host.startswith(pat_s[:-2]):
                    return False
            if pat_s and pat_s in host:
                return False
        # allow-list (if set) enforces inclusion
        allowed = list(cfg.allowed_domains) or []
        if allowed:
            return any((_normalize_domain(a) in host) for a in allowed)
        return True
    except Exception:
        return False


# ---------------------------
# Quotas / budgets
# ---------------------------

def _count_tool_calls(state: Dict[str, Any]) -> Tuple[int, int]:
    """
    Return (total_so_far, current_turn) based on light-weight counters in state.
    The graph may store counters in `state["meta"]["tool_calls"]`.
    """
    meta = state.get("meta") or {}
    total = int(meta.get("tool_calls_total", 0))
    turn = int(meta.get("tool_calls_turn", 0))
    return total, turn

def _inc_tool_calls(state: Dict[str, Any]) -> None:
    meta = state.setdefault("meta", {})
    meta["tool_calls_total"] = int(meta.get("tool_calls_total", 0)) + 1
    meta["tool_calls_turn"] = int(meta.get("tool_calls_turn", 0)) + 1

def reset_turn_counters(state: Dict[str, Any]) -> None:
    """
    Call this when a new user turn begins (e.g., new /chat request).
    """
    meta = state.setdefault("meta", {})
    meta["tool_calls_turn"] = 0


def can_call_tool(state: Dict[str, Any], tool_name: str, cfg: Optional[PolicyConfig] = None) -> Tuple[bool, str]:
    """
    Soft quota checks before actually calling a tool.
    Returns (ok, reason_if_not_ok).
    """
    cfg = cfg or PolicyConfig.from_env()
    total, turn = _count_tool_calls(state)

    if turn >= cfg.max_tool_calls_per_turn:
        return (False, f"per-turn tool-call limit reached ({turn}/{cfg.max_tool_calls_per_turn})")
    if total >= cfg.max_total_tool_calls:
        return (False, f"run tool-call limit reached ({total}/{cfg.max_total_tool_calls})")

    if tool_name in {"web_search", "browse", "tavily"} and not cfg.allow_web:
        return (False, "web access is disabled by policy")
    if tool_name in {"shell", "bash"} and not cfg.allow_shell:
        return (False, "shell access is disabled by policy")
    if tool_name in {"python_exec"} and not cfg.allow_code_exec:
        return (False, "arbitrary code execution is disabled by policy")

    return (True, "")

def note_tool_call(state: Dict[str, Any]) -> None:
    """
    Increment counters after a successful tool invocation.
    """
    _inc_tool_calls(state)


# ---------------------------
# LLM token budgeting (very light)
# ---------------------------

def can_consume_llm_tokens(state: Dict[str, Any], want_tokens: int, cfg: Optional[PolicyConfig] = None) -> Tuple[bool, str]:
    """
    Soft check against per-turn token limits. The agent should respect this as a *hint*.
    """
    cfg = cfg or PolicyConfig.from_env()
    budget = (state.get("budget") or {})
    used = int(budget.get("used_tokens", 0))
    max_per_turn = int(cfg.max_llm_tokens_per_turn)
    if used + want_tokens > max_per_turn:
        return (False, f"LLM token budget exceeded ({used + want_tokens} > {max_per_turn})")
    return (True, "")

def note_llm_tokens(state: Dict[str, Any], used_tokens: int) -> None:
    budget = state.setdefault("budget", {})
    budget["used_tokens"] = int(budget.get("used_tokens", 0)) + max(0, int(used_tokens))


# ---------------------------
# Pre-step enforcement
# ---------------------------

def enforce_policies_or_raise(state: Dict[str, Any], cfg: Optional[PolicyConfig] = None) -> None:
    """
    Call at the start of each graph step to perform lightweight sanity checks.
    """
    cfg = cfg or PolicyConfig.from_env()

    # Block obviously unsafe content by regex (very small net)
    text = " ".join(
        str(x) for x in [
            state.get("query") or "",
            *(m.get("content") or "" for m in state.get("messages") or [] if isinstance(m, dict)),
        ]
    )
    for pat in cfg.blocked_patterns:
        if re.search(pat, text):
            raise ValueError("blocked content matched policy pattern; please remove secrets and retry")


# ---------------------------
# Tiny helper for outbound URLs in tools
# ---------------------------

def require_allowed_url(url: str, cfg: Optional[PolicyConfig] = None) -> None:
    cfg = cfg or PolicyConfig.from_env()
    if not cfg.allow_web:
        raise PermissionError("web access is disabled by policy")
    if not is_domain_allowed(url, cfg):
        raise PermissionError(f"domain not allowed by policy: {url}")


__all__ = [
    "PolicyConfig",
    "with_retries",
    "sanitize_tool_args",
    "is_domain_allowed",
    "can_call_tool",
    "note_tool_call",
    "reset_turn_counters",
    "can_consume_llm_tokens",
    "note_llm_tokens",
    "redact_for_prompt",
    "enforce_policies_or_raise",
    "require_allowed_url",
]
