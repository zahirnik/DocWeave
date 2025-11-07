# packages/core/logging.py
"""
JSON logging — tiny, explicit, and stdout-first.

What this module provides
-------------------------
- setup_json_logging(level="INFO"): configure root logger with a JSON formatter.
- get_logger(name): convenience to get a child logger.
- log structure: time, level, name, message, module/file/line, and request_id (if present).

Design choices
--------------
- Logs go to stdout (12-factor-friendly).
- We avoid extra deps; the formatter emits JSON using `json.dumps`.
- If you later adopt `structlog` or `python-json-logger`, you can swap the formatter
  here without touching call-sites.

Correlating requests
--------------------
- If `apps.api.middlewares.request_id` is installed, it sets a context var.
- We read it here and inject `request_id` on every record automatically.

Usage
-----
from packages.core.logging import setup_json_logging, get_logger
setup_json_logging("DEBUG")
log = get_logger(__name__)
log.info("hello", extra={"foo": "bar"})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional


class _JSONFormatter(logging.Formatter):
    """
    Minimal JSON formatter that keeps messages structured and compact.
    """

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"

    def __init__(self, *, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        # Base fields
        ts = time.gmtime(record.created)
        base: Dict[str, Any] = {
            "time": time.strftime(self.default_time_format, ts) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "file": record.pathname,
            "line": record.lineno,
        }

        # Request correlation (best-effort)
        try:
            from apps.api.middlewares.request_id import get_request_id  # lazy import to avoid cycles
            rid = get_request_id()
            if rid:
                base["request_id"] = rid
        except Exception:
            pass

        # Exceptions
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        # Extra fields (only those not in standard LogRecord)
        if self.include_extra:
            for k, v in record.__dict__.items():
                if k in (
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                ):
                    continue
                # Avoid overwriting base keys
                if k in base:
                    continue
                try:
                    json.dumps(v)  # ensure serializable; else cast to str
                    base[k] = v
                except Exception:
                    base[k] = str(v)

        return json.dumps(base, separators=(",", ":"), ensure_ascii=False)


_configured = False


def setup_json_logging(level: Optional[str] = None) -> None:
    """
    Configure root logger to emit JSON to stdout. Safe to call multiple times.
    Respects LOG_LEVEL env var unless `level` is given.

    Example:
        setup_json_logging("DEBUG")
    """
    global _configured
    if _configured:
        return

    # Level precedence: function arg > env > INFO
    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    try:
        numeric_level = getattr(logging, lvl, logging.INFO)
    except Exception:
        numeric_level = logging.INFO

    # Root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate logs in notebooks/uvicorn reloads
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(_JSONFormatter())

    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Convenience to get a named logger.
    Be sure to call setup_json_logging() once during app startup.
    """
    return logging.getLogger(name)
