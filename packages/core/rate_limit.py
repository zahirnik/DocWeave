# packages/core/rate_limit.py
"""
Token-bucket rate limiter (tiny, explicit, production-lean).

Why this module?
----------------
Some endpoints need rate limiting beyond the tutorial middleware. This module
exposes a small, reusable token-bucket that works with either:
  - Redis (preferred for multi-process / multi-host), or
  - In-memory fallback (single-process dev/tests).

Interface (simple & stable)
---------------------------
from packages.core.rate_limit import get_rate_limiter

rl = get_rate_limiter()
ok = rl.allow(key="ip:127.0.0.1", cost=1.0, rate=5.0, burst=10)  # ~5 rps with burst 10

Semantics
---------
- Each key has:
    tokens  : starts at `burst`, refills by `rate` per second, capped at `burst`
    last_ts : last update time (seconds)
- `allow()` consumes `cost` tokens when enough are available and returns True; else False.

Environment defaults
--------------------
RATE_LIMIT_RATE  (float, default 5.0)   # tokens per second
RATE_LIMIT_BURST (int,   default 10)    # bucket capacity

Notes
-----
- The Redis path uses a tiny Lua script for atomic updates (no race conditions).
- The in-memory path is thread-safe but not process-safe (dev only).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from packages.core.config import get_settings
from packages.core.logging import get_logger

log = get_logger(__name__)

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


# ---------------------------
# Redis-backed implementation
# ---------------------------

_REDIS_LUA = """
-- KEYS[1] = key
-- ARGV = now, rate, burst, cost, ttl
local key   = KEYS[1]
local now   = tonumber(ARGV[1])
local rate  = tonumber(ARGV[2])
local burst = tonumber(ARGV[3])
local cost  = tonumber(ARGV[4])
local ttl   = tonumber(ARGV[5])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts     = tonumber(data[2])

if tokens == nil or ts == nil then
  tokens = burst
  ts = now
else
  local elapsed = now - ts
  if elapsed > 0 then
    tokens = math.min(burst, tokens + elapsed * rate)
    ts = now
  end
end

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', ts)
redis.call('EXPIRE', key, ttl)

return allowed
"""


class _RedisRateLimiter:
    def __init__(self, url: str):
        assert redis is not None, "redis package not installed"
        self._r = redis.from_url(url, decode_responses=False, health_check_interval=30)
        try:
            self._r.ping()
        except Exception as e:  # pragma: no cover
            raise RuntimeEr
