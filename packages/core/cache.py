# packages/core/cache.py
"""
Redis cache wrappers — tiny, explicit, and safe-by-default.

Why this module exists
----------------------
- Provide a single, well-documented place to cache:
  - raw bytes (e.g., charts/images)
  - UTF-8 strings (e.g., JSON blobs)
  - Python objects (JSON-serializable dicts/lists)
- Fall back to an in-memory dict when Redis is not configured (great for tests).

Design goals
------------
- Minimal surface area (get/set/delete/exists/ttl).
- Key namespacing helper to avoid collisions.
- Clear TTL defaults and per-call overrides.
- Never crash if Redis is down: log a warning and degrade to in-memory cache.

Environment / Settings
----------------------
- Uses `packages.core.config.get_settings().redis_url`.
- If missing, we use a process-local in-memory cache (not shared across workers).

Example
-------
from packages.core.cache import get_cache, cache_key

cache = get_cache()
k = cache_key("embeddings", "openai", model="text-embedding-3-small", v=1)
cache.set_json(k, {"dim": 1536}, ttl_s=3600)
dim = cache.get_json(k).get("dim")

Notes
-----
- For multi-process deployments, prefer a real Redis.
- In-memory cache is cleared on process restart and isn’t shared across workers.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from packages.core.config import get_settings
from packages.core.logging import get_logger

try:
    import redis  # redis-py
except Exception:  # pragma: no cover
    redis = None  # type: ignore

log = get_logger(__name__)


# ---------------------------
# Key helper
# ---------------------------

def cache_key(*parts: str, **labels: Any) -> str:
    """
    Build a stable namespaced key: "convai:<part1>:<part2>:...:k=v,k2=v2"
    - parts are path-like segments
    - labels are sorted k=v pairs (values converted to str)
    """
    head = "convai"
    segs = [head] + [str(p).strip().replace(" ", "_") for p in parts if p]
    if labels:
        kv = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
        segs.append(kv)
    return ":".join(segs)


# ---------------------------
# In-memory fallback
# ---------------------------

@dataclass
class _MemEntry:
    value: bytes
    expires_at: Optional[float]  # epoch seconds or None


class _MemoryCache:
    """Thread-safe in-memory cache with TTL support (best-effort)."""

    def __init__(self):
        self._data: Dict[str, _MemEntry] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.time()

    def _expired(self, ent: _MemEntry) -> bool:
        return ent.expires_at is not None and self._now() >= ent.expires_at

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            ent = self._data.get(key)
            if not ent:
                return None
            if self._expired(ent):
                self._data.pop(key, None)
                return None
            return ent.value

    def set(self, key: str, value: bytes, ttl_s: Optional[int] = None) -> None:
        with self._lock:
            exp = None if ttl_s is None or ttl_s <= 0 else self._now() + int(ttl_s)
            self._data[key] = _MemEntry(value=value, expires_at=exp)

    def delete(self, key: str) -> int:
        with self._lock:
            return 1 if self._data.pop(key, None) else 0

    def ttl(self, key: str) -> Optional[int]:
        with self._lock:
            ent = self._data.get(key)
            if not ent:
                return None
            if ent.expires_at is None:
                return -1  # Redis semantics: -1 means no expire
            rem = int(ent.expires_at - self._now())
            return rem if rem >= 0 else None


# ---------------------------
# Redis-backed cache
# ---------------------------

class _RedisCache:
    """Thin wrapper around redis-py with a stable interface and JSON helpers."""

    def __init__(self, url: str):
        assert redis is not None, "redis package not installed"
        # decode_responses=False because we want raw bytes; we handle encoding ourselves
        self._r = redis.from_url(url, decode_responses=False, health_check_interval=30)
        try:
            # Warm healthcheck (will raise if not reachable)
            self._r.ping()
        except Exception as e:  # pragma: no cover
            log.warning("Redis not reachable at %s (%s); falling back to memory cache.", url, e)
            raise

    # Raw bytes
    def get(self, key: str) -> Optional[bytes]:
        return self._r.get(key)

    def set(self, key: str, value: bytes, ttl_s: Optional[int] = None) -> None:
        if ttl_s and ttl_s > 0:
            self._r.setex(key, int(ttl_s), value)
        else:
            self._r.set(key, value)

    def delete(self, key: str) -> int:
        return int(self._r.delete(key) or 0)

    def ttl(self, key: str) -> Optional[int]:
        t = self._r.ttl(key)
        if t is None:
            return None
        return int(t)


# ---------------------------
# Public cache API
# ---------------------------

class Cache:
    """
    Unified cache facade exposing:
      - get_bytes / set_bytes
      - get_str / set_str
      - get_json / set_json
      - delete / ttl

    TTL defaults
    ------------
    - Default TTL (if not provided) is None (no expiration).
    - For request-scoped items (e.g., rerank results), a TTL of 5–30 minutes is typically enough.
    """

    def __init__(self):
        st = get_settings()
        self._backend: Any
        if st.redis_url and redis is not None:
            try:
                self._backend = _RedisCache(st.redis_url)
                log.info("Cache: using Redis backend")
            except Exception:
                self._backend = _MemoryCache()
                log.info("Cache: using in-memory backend (Redis unavailable)")
        else:
            self._backend = _MemoryCache()
            log.info("Cache: using in-memory backend")

    # ---- Bytes ----
    def get_bytes(self, key: str) -> Optional[bytes]:
        return self._backend.get(key)

    def set_bytes(self, key: str, value: bytes, ttl_s: Optional[int] = None) -> None:
        self._backend.set(key, value, ttl_s)

    # ---- Strings (utf-8) ----
    def get_str(self, key: str) -> Optional[str]:
        b = self.get_bytes(key)
        return None if b is None else b.decode("utf-8", errors="ignore")

    def set_str(self, key: str, value: str, ttl_s: Optional[int] = None) -> None:
        self.set_bytes(key, value.encode("utf-8"), ttl_s)

    # ---- JSON ----
    def get_json(self, key: str) -> Optional[dict]:
        s = self.get_str(key)
        if s is None:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    def set_json(self, key: str, value: dict, ttl_s: Optional[int] = None) -> None:
        self.set_str(key, json.dumps(value, separators=(",", ":"), ensure_ascii=False), ttl_s)

    # ---- Misc ----
    def delete(self, key: str) -> int:
        return self._backend.delete(key)

    def ttl(self, key: str) -> Optional[int]:
        return self._backend.ttl(key)


# Singleton accessor
_cache_singleton: Optional[Cache] = None


def get_cache() -> Cache:
    """Return the process-wide cache instance."""
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = Cache()
    return _cache_singleton
