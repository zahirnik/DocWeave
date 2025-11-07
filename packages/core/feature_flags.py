# packages/core/feature_flags.py
"""
Feature flags — tiny, explicit, environment-driven.

Purpose
-------
Provide a single, documented way to toggle behavior at runtime without code changes.
This keeps the rest of the codebase clean and tutorial-clear.

Flag sources (highest precedence first)
---------------------------------------
1) In-memory overrides (tests/dev can call `set_override("FLAG_NAME", True)`).
2) Environment variables:
     - FEATURE_<NAME>=true|false|1|0
     - FLAG_<NAME>=true|false|1|0
   Examples:
     FEATURE_BM25=true
     FLAG_RERANK=false
3) Defaults passed by call sites (e.g., is_enabled("BM25", default=False)).

Design
------
- Flags are **strings** of A–Z, 0–9, and underscores. Normalize to UPPER_SNAKE.
- Booleans are parsed leniently: "1, true, yes, on" → True; "0, false, no, off" → False.
- No file/HTTP fetches; keep it tiny and deterministic.
- Thread-safe reads/writes for overrides (simple `threading.Lock`).

Usage
-----
from packages.core.feature_flags import is_enabled, get_flag, set_override, clear_overrides

if is_enabled("BM25", default=True):
    ...  # run BM25 hybrid path

w = float(get_flag("RERANK_WEIGHT", default="0.35"))  # non-boolean example

Testing tips
------------
- Use set_override("BM25", False) in unit tests to force behavior.
- Clear in teardown with clear_overrides().
"""

from __future__ import annotations

import os
import re
import threading
from typing import Dict, Optional

_VALID = re.compile(r"^[A-Z0-9_]+$")

# In-memory overrides (for tests/dev)
_overrides: Dict[str, str] = {}
_lock = threading.Lock()


def _norm(name: str) -> str:
    """Normalize a flag name to UPPER_SNAKE and validate characters."""
    if not name or not isinstance(name, str):
        raise ValueError("flag name must be a non-empty string")
    n = re.sub(r"[^A-Za-z0-9_]", "_", name).upper()
    if not _VALID.match(n):
        raise ValueError(f"invalid flag name: {name!r}")
    return n


def _as_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    # Fallback: non-empty string → True
    return bool(v)


def set_override(name: str, value: bool | str) -> None:
    """
    Set an in-memory override for a flag. Value stored as string.

    Example:
        set_override("BM25", False)
        set_override("RERANK_WEIGHT", "0.5")
    """
    n = _norm(name)
    with _lock:
        _overrides[n] = str(value)


def clear_override(name: str) -> None:
    """Remove a specific override."""
    n = _norm(name)
    with _lock:
        _overrides.pop(n, None)


def clear_overrides() -> None:
    """Remove all overrides (useful in test teardown)."""
    with _lock:
        _overrides.clear()


def get_flag(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Return the flag value as a string, or `default` if not set.

    Lookup order:
      overrides -> env FEATURE_<NAME> -> env FLAG_<NAME> -> default
    """
    n = _norm(name)
    with _lock:
        if n in _overrides:
            return _overrides[n]

    env1 = os.getenv(f"FEATURE_{n}")
    if env1 is not None:
        return env1

    env2 = os.getenv(f"FLAG_{n}")
    if env2 is not None:
        return env2

    return default


def is_enabled(name: str, default: bool = False) -> bool:
    """
    Return the flag as a boolean with lenient parsing.

    Example:
        if is_enabled("HYBRID_SEARCH", default=True):
            ...
    """
    raw = get_flag(name, default=None)
    return _as_bool(raw, default=default)


def all_flags(prefix: str | None = None) -> Dict[str, str]:
    """
    Return a snapshot of all known flags (overrides + env) for debugging.
    `prefix` filters names starting with that string (after normalization).

    WARNING: This enumerates **environment variables** starting with FEATURE_/FLAG_.
    """
    res: Dict[str, str] = {}
    # env
    for k, v in os.environ.items():
        if k.startswith("FEATURE_") or k.startswith("FLAG_"):
            name = k.split("_", 1)[1]
            try:
                n = _norm(name)
            except Exception:
                continue
            if prefix and not n.startswith(_norm(prefix)):
                continue
            res[n] = v
    # overrides (win over env in view)
    with _lock:
        for k, v in _overrides.items():
            if prefix and not k.startswith(_norm(prefix)):
                continue
            res[k] = v
    return dict(sorted(res.items()))
