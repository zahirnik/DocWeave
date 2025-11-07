# apps/api/middlewares/rate_limit.py
"""
Simple token-bucket rate limiter middleware (tutorial-clear).

What it does
------------
- Applies a per-client token bucket on each request.
- Defaults target: ~5 requests/second with a small burst.
- Identifies a client by `X-API-KEY` when present, otherwise by remote IP.

Why here (and not API Gateway)?
-------------------------------
- Keeping a *tiny* limiter in-app is great for local dev and tests.
- For production, replicate these semantics at the edge (e.g., API Gateway, CDN).
- The same "key function" concept (API key or IP) still applies.

How it works
------------
- Each client key has:
    tokens: starts at `burst`
    refill: rate tokens per second, capped at burst
- For each request cost=1.0 token (changeable if you want).
- If a bucket has insufficient tokens → respond 429.

Tuning via env
--------------
RATE_LIMIT_RATE=5      # tokens per second
RATE_LIMIT_BURST=10    # max tokens

Implementation notes
--------------------
- In-memory dict is fine for one-process dev.
- For distributed limits, swap in a Redis-backed bucket (atomic script).
"""

from __future__ import annotations

import os
import time
import threading
from typing import Dict

from starlette.requests import Request
from starlette.responses import JSONResponse

# Defaults; override via env
_RATE = float(os.getenv("RATE_LIMIT_RATE", "5"))
_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))


class _TokenBucket:
    """A tiny, thread-safe token bucket."""

    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = float(rate_per_sec)
        self.burst = int(burst)
        self.tokens = float(burst)
        self.last = time.time()
        self.lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            # Refill up to burst cap
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False


# Global in-memory buckets (per process)
_BUCKETS: Dict[str, _TokenBucket] = {}


def _key_from_request(request: Request) -> str:
    """
    Select a rate-limit key.
    - Prefer API key for authenticated service clients.
    - Else, use remote IP (beware of proxies; in prod inspect X-Forwarded-For safely).
    """
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"k:{api_key[-8:]}"  # save memory by hashing/shortening (demo)
    # Remote IP (simple; adjust for your infra)
    client = request.client.host if request.client else "unknown"
    return f"ip:{client}"


async def rate_limit_middleware(request: Request, call_next):
    """
    Return 429 if the client's token bucket is empty.
    """
    key = _key_from_request(request)
    bucket = _BUCKETS.get(key)
    if not bucket:
        bucket = _BUCKETS[key] = _TokenBucket(_RATE, _BURST)

    if not bucket.allow(1.0):
        return JSONResponse({"detail": "Too Many Requests"}, status_code=429)

    return await call_next(request)
