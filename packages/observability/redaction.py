# packages/observability/redaction.py
"""
Centralised redaction — one tiny place to scrub PII and secrets for logs/telemetry.

What this module provides
-------------------------
- scrub_text(text: str, extra_patterns: list[tuple[pattern, repl]] | None = None) -> str
    Apply repo-wide PII masking + conservative secret scrubbing.

- scrub_obj(obj, max_depth=5) -> obj
    Recursively scrub strings inside dict/list/tuple structures (safe for logging).

- RedactingFilter(logging.Filter)
    Drop-in logging filter that scrubs LogRecord message/args + known attributes.

- install_logging_redaction(logger=None) -> None
    Attach RedactingFilter to the given logger (or root logger) once.

Design goals
------------
- Tutorial-clear, zero heavy deps.
- Reuse packages.security.pii for robust UK/EU PII masking.
- Keep redaction **lossy** (err on the side of masking) to avoid leaks.

Usage
-----
import logging
from packages.observability.redaction import install_logging_redaction, scrub_text

install_logging_redaction()
log = logging.getLogger("app")
log.info("User %s, email=%s, card=%s", "Alice", "alice@example.co.uk", "4111 1111 1111 1111")
# -> "User Alice, email=[EMAIL], card=[CARD••1111]"
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Prefer our PII utilities; fall back to identity if missing
try:
    from packages.security.pii import mask_pii, redact_obj
except Exception:  # pragma: no cover
    def mask_pii(text: str) -> str:  # type: ignore
        return text
    def redact_obj(obj: Any) -> Any:  # type: ignore
        return obj


# ---------------------------
# Additional conservative secret scrubs (regex → replacement)
# ---------------------------

# Patterns intentionally broad to avoid accidental leakage
_SECRET_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # apiKey: sk-XXXX, x-api-key, Authorization: Bearer XXX, etc.
    (re.compile(r"(?i)\b(api[_\- ]?key|x-api-key|apikey)\b\s*[:=]\s*[A-Za-z0-9_\-]{8,}"), r"\1: [REDACTED]"),
    (re.compile(r"(?i)\b(authorization)\b\s*[:=]\s*(Bearer|Basic)\s+[A-Za-z0-9._\-+=:/]{8,}"), r"\1: [REDACTED]"),
    (re.compile(r"(?i)\b(password|secret|token|sessionid|cookie)\b\s*[:=]\s*\S+"), r"\1: [REDACTED]"),
    # Cloud-ish keys (very conservative)
    (re.compile(r"(?i)\b(AKIA|ASIA)[A-Z0-9]{12,}\b"), "[REDACTED]"),
    (re.compile(r"(?i)\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[JWT-REDACTED]"),
]

# Headers likely to contain sensitive material (scrub values)
_SENSITIVE_HEADER_KEYS = {
    "authorization", "x-api-key", "apikey", "api-key", "cookie", "set-cookie",
    "x-forwarded-for", "proxy-authorization", "x-auth-token", "x-csrf-token",
    "x-azure-sas", "x-aws-signature", "x-gcp-cred", "x-openai-key",
    "password", "secret", "token",
}


# ---------------------------
# Public API
# ---------------------------

def scrub_text(text: str, extra_patterns: Optional[List[Tuple[re.Pattern, str]]] = None) -> str:
    """
    PII → masked, secrets → redacted, with optional extra regex replacements.

    The order is: PII masking first (emails/phones/cards/NINO/IP/IBAN), then secrets.
    """
    s = text or ""
    # 1) PII masking (emails/phones/cards/ibans/nino/ip etc.)
    s = mask_pii(s)
    # 2) Generic secret scrubbing
    for rx, repl in _SECRET_PATTERNS + (extra_patterns or []):
        try:
            s = rx.sub(repl, s)
        except Exception:
            # best effort; never break the caller
            continue
    # Truncate extremely long lines for logs
    if len(s) > 10_000:
        s = s[:9_997] + "..."
    return s


def scrub_headers(headers: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return a copy of HTTP-style headers with sensitive values redacted.
    Keys are compared case-insensitively.
    """
    if not headers:
        return {}
    out: Dict[str, Any] = {}
    for k, v in headers.items():
        if str(k).strip().lower() in _SENSITIVE_HEADER_KEYS:
            out[k] = "[REDACTED]"
        else:
            out[k] = scrub_text(str(v))
    return out


def scrub_obj(obj: Any, *, max_depth: int = 5) -> Any:
    """
    Recursively scrub any strings found in dicts/lists/tuples.
    Non-strings are passed through unchanged. Depth is clamped to avoid cycles.
    """
    if max_depth <= 0:
        # stop recursing; return simple representation
        if isinstance(obj, str):
            return scrub_text(obj)
        try:
            return json.loads(json.dumps(obj, default=str))  # best-effort stable clone
        except Exception:
            return str(obj)

    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, dict):
        return {k: scrub_obj(v, max_depth=max_depth - 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(x, max_depth=max_depth - 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(scrub_obj(x, max_depth=max_depth - 1) for x in obj)
    return obj


# ---------------------------
# Logging filter
# ---------------------------

class RedactingFilter(logging.Filter):
    """
    A logging.Filter that scrubs:
    - record.msg and record.args (format-time redaction)
    - common HTTP attributes (headers, path, query, body)
    - extra fields (dict-like) passed via LoggerAdapter or extra=...

    It modifies the record in-place, returning True to keep the log.
    """

    def __init__(self, name: str = ""):
        super().__init__(name)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        # Message & args
        try:
            if isinstance(record.msg, str):
                record.msg = scrub_text(record.msg)
            # Scrub args while preserving tuple/dict semantics for logging formatters
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: scrub_obj(v) for k, v in record.args.items()}
                elif isinstance(record.args, tuple):
                    record.args = tuple(scrub_obj(a) for a in record.args)
                else:
                    # Rare: other sequence types
                    try:
                        record.args = tuple(scrub_obj(a) for a in record.args)  # type: ignore
                    except Exception:
                        record.args = ()
        except Exception:
            # never block logging
            pass

        # Common web attributes if frameworks attach them
        for attr in ("path", "query", "url", "route", "client"):
            if hasattr(record, attr):
                try:
                    setattr(record, attr, scrub_text(str(getattr(record, attr))))
                except Exception:
                    pass
        # headers/body often present via adapters
        if hasattr(record, "headers"):
            try:
                headers = getattr(record, "headers")
                setattr(record, "headers", scrub_headers(headers if isinstance(headers, dict) else {}))
            except Exception:
                pass
        if hasattr(record, "body"):
            try:
                setattr(record, "body", scrub_text(str(getattr(record, "body"))))
            except Exception:
                pass
        if hasattr(record, "extra"):
            try:
                setattr(record, "extra", scrub_obj(getattr(record, "extra")))
            except Exception:
                pass
        return True


def install_logging_redaction(logger: Optional[logging.Logger] = None) -> None:
    """
    Attach the RedactingFilter to the given logger (or root logger) once.

    Typical usage (in your API startup):
        import logging
        install_logging_redaction(logging.getLogger("uvicorn.access"))
        install_logging_redaction(logging.getLogger("app"))
    """
    log = logger or logging.getLogger()
    # Avoid duplicate installation
    for f in getattr(log, "filters", []):
        if isinstance(f, RedactingFilter):
            return
    log.addFilter(RedactingFilter())


__all__ = ["scrub_text", "scrub_obj", "scrub_headers", "RedactingFilter", "install_logging_redaction"]
