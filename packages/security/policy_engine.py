# packages/security/policy_engine.py
"""
Policy engine — explicit allow/deny checks for uploads, web, tenants, and logging.

What this module provides
-------------------------
- PolicyEngine: a tiny, dependency-light evaluator that reads rules (optionally from YAML)
  and exposes small, testable checks for common risks in an agentic RAG system.

Key checks
----------
- allow_upload(file_meta) -> (ok: bool, reason: str)
    Enforces MIME/extension allow-list, file-size caps, and basic antivirus flag.

- allow_web(url) -> (ok: bool, reason: str)
    Enforces domain allow/deny and "http vs https" constraints.

- enforce_tenant_access(user_tenant, resource_tenant) -> None | raises PermissionError
    Ensures strict tenant isolation.

- redact_for_logs(text) -> str
    Applies repo-wide PII masking and simple secret scrubbing for log safety.

- retention_days_for(meta) -> int
    Returns how long an artefact should be kept based on its metadata and policy defaults.

Configuration
-------------
You can provide a YAML/JSON policy file at `configs/policies.yaml` like:

policy_version: 1
uploads:
  max_mb: 50
  allowed_mime:
    - application/pdf
    - text/csv
    - application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
  forbidden_ext:
    - .exe
    - .js
    - .bat
  antivirus_required: false

web:
  allow: true
  allowed_domains:
    - sec.gov
    - reuters.com
    - ft.com
  banned_domains:
    - pastebin.com
    - anonfiles.com
  https_only: true

logging:
  redact: true
  keep_request_bodies: false

retention:
  default_days: 90
  by_tag:
    pii: 30
    raw_upload: 60
    derived_chart: 180

The engine works if the file is missing — it uses conservative defaults.

Design goals
------------
- Tutorial-clear, small, and explicit. Zero heavy deps (PyYAML is optional).
- Pure functions that are easy to unit-test.
- Friendly error messages that can flow straight into API responses.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional YAML support (kept soft)
try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# PII utilities from this repo (optional but recommended)
try:  # pragma: no cover
    from packages.security.pii import mask_pii
except Exception:  # pragma: no cover
    def mask_pii(text: str) -> str:  # type: ignore
        return text


# ---------------------------
# Data models (with safe defaults)
# ---------------------------

@dataclass
class UploadPolicy:
    max_mb: int = 25
    allowed_mime: Iterable[str] = field(default_factory=lambda: (
        "application/pdf",
        "text/plain",
        "text/csv",
        "application/json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ))
    forbidden_ext: Iterable[str] = field(default_factory=lambda: (".exe", ".dll", ".js", ".bat", ".sh"))
    antivirus_required: bool = False  # integrate with AV scanner if available


@dataclass
class WebPolicy:
    allow: bool = True
    allowed_domains: Iterable[str] = field(default_factory=lambda: ("sec.gov", "reuters.com", "ft.com", "bloomberg.com"))
    banned_domains: Iterable[str] = field(default_factory=lambda: ("pastebin.com", "anonfiles.com", "mega.nz"))
    https_only: bool = True


@dataclass
class LoggingPolicy:
    redact: bool = True
    keep_request_bodies: bool = False


@dataclass
class RetentionPolicy:
    default_days: int = 90
    by_tag: Dict[str, int] = field(default_factory=lambda: {"pii": 30, "raw_upload": 60, "derived_chart": 180})


@dataclass
class RootPolicy:
    policy_version: int = 1
    uploads: UploadPolicy = field(default_factory=UploadPolicy)
    web: WebPolicy = field(default_factory=WebPolicy)
    logging: LoggingPolicy = field(default_factory=LoggingPolicy)
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)


# ---------------------------
# Helpers
# ---------------------------

def _norm_domain(host: str) -> str:
    return (host or "").strip().lower().lstrip("www.")

def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def _lower_list(xs: Iterable[str]) -> List[str]:
    return [str(x).strip().lower() for x in xs or []]

def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        if yaml is not None and path.lower().endswith((".yml", ".yaml")):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------
# PolicyEngine
# ---------------------------

class PolicyEngine:
    """
    Evaluate allow/deny and redaction decisions based on a small policy file.

    Construction
    -----------
    pe = PolicyEngine.from_path("./configs/policies.yaml")

    If the file is missing or malformed, conservative defaults are used.
    """

    def __init__(self, root: RootPolicy):
        self.root = root

    # ---- loading ----

    @staticmethod
    def from_path(path: str | None) -> "PolicyEngine":
        data: Dict[str, Any] = _load_yaml_or_json(path) if path else {}
        # uploads
        up = data.get("uploads") or {}
        uploads = UploadPolicy(
            max_mb=_as_int(up.get("max_mb", 25), 25),
            allowed_mime=tuple(up.get("allowed_mime", UploadPolicy().allowed_mime)),
            forbidden_ext=tuple(up.get("forbidden_ext", UploadPolicy().forbidden_ext)),
            antivirus_required=bool(up.get("antivirus_required", False)),
        )
        # web
        wb = data.get("web") or {}
        web = WebPolicy(
            allow=bool(wb.get("allow", True)),
            allowed_domains=tuple(wb.get("allowed_domains", WebPolicy().allowed_domains)),
            banned_domains=tuple(wb.get("banned_domains", WebPolicy().banned_domains)),
            https_only=bool(wb.get("https_only", True)),
        )
        # logging
        lg = data.get("logging") or {}
        logging = LoggingPolicy(
            redact=bool(lg.get("redact", True)),
            keep_request_bodies=bool(lg.get("keep_request_bodies", False)),
        )
        # retention
        rt = data.get("retention") or {}
        retention = RetentionPolicy(
            default_days=_as_int(rt.get("default_days", 90), 90),
            by_tag=dict(rt.get("by_tag") or {"pii": 30, "raw_upload": 60, "derived_chart": 180}),
        )
        root = RootPolicy(
            policy_version=_as_int(data.get("policy_version", 1), 1),
            uploads=uploads,
            web=web,
            logging=logging,
            retention=retention,
        )
        return PolicyEngine(root)

    # ---- uploads ----

    def allow_upload(self, file_meta: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check a file before ingestion.

        file_meta may contain:
          - "path": original filename or path (for extension check)
          - "mime": detected MIME type (e.g., "application/pdf")
          - "size_bytes": integer
          - "antivirus_ok": Optional[bool]

        Returns (ok, reason_if_denied).
        """
        p = self.root.uploads

        # Size
        size = _as_int(file_meta.get("size_bytes", 0), 0)
        if size <= 0:
            return (False, "file is empty or size not provided")
        if size > p.max_mb * 1024 * 1024:
            return (False, f"file is too large (>{p.max_mb} MB)")

        # MIME
        mime = (file_meta.get("mime") or "").lower()
        if mime and p.allowed_mime and mime not in _lower_list(p.allowed_mime):
            return (False, f"MIME '{mime}' not allowed")

        # Extension
        path = str(file_meta.get("path") or "")
        ext = _ext(path)
        if ext and any(ext == e.lower() for e in p.forbidden_ext):
            return (False, f"extension '{ext}' is forbidden")

        # Antivirus
        if p.antivirus_required:
            av = file_meta.get("antivirus_ok")
            if av is not True:
                return (False, "antivirus scan required (and must pass)")

        return (True, "")

    # ---- web ----

    def allow_web(self, url: str) -> Tuple[bool, str]:
        """
        Check whether a URL is allowed for outbound web search/fetch.
        """
        w = self.root.web
        if not w.allow:
            return (False, "web access is disabled by policy")
        try:
            u = urllib.parse.urlparse(url)
        except Exception:
            return (False, "malformed URL")
        scheme = (u.scheme or "").lower()
        host = _norm_domain(u.hostname or "")
        if not host:
            return (False, "missing host")
        if w.https_only and scheme != "https":
            return (False, "only https URLs are permitted")

        # Deny-list takes precedence
        for bd in _lower_list(w.banned_domains):
            if not bd:
                continue
            if bd.endswith(".*"):  # pattern like "dark.*"
                if host.startswith(bd[:-2]):
                    return (False, f"domain '{host}' is banned")
            if bd in host:
                return (False, f"domain '{host}' is banned")

        # Allow-list (if present) must match
        allowed = _lower_list(w.allowed_domains)
        if allowed:
            if not any(a in host for a in allowed):
                return (False, f"domain '{host}' not in allow-list")
        return (True, "")

    # ---- tenants ----

    def enforce_tenant_access(self, user_tenant: str, resource_tenant: str) -> None:
        """
        Raise PermissionError if a user attempts to access a resource outside their tenant.
        """
        a = (user_tenant or "").strip()
        b = (resource_tenant or "").strip()
        if not a or not b:
            raise PermissionError("tenant context is required")
        if a != b:
            raise PermissionError("cross-tenant access denied")

    # ---- logging ----

    def redact_for_logs(self, text: str) -> str:
        """
        Apply global log redaction (PII masking + simple secret scrubbing).
        """
        if not self.root.logging.redact or not text:
            return text or ""
        s = mask_pii(text)
        # Additional conservative secret scrubbing
        s = re.sub(r"(?i)(api[_\- ]?key\s*[:=]\s*)([A-Za-z0-9_\-]{12,})", r"\1[REDACTED]", s)
        s = re.sub(r"(?i)(password\s*[:=]\s*)(\S+)", r"\1[REDACTED]", s)
        s = re.sub(r"(?i)(secret\s*[:=]\s*)(\S+)", r"\1[REDACTED]", s)
        return s

    def keep_request_bodies(self) -> bool:
        """
        Whether to persist raw request bodies in logs (off by default).
        """
        return bool(self.root.logging.keep_request_bodies)

    # ---- retention ----

    def retention_days_for(self, meta: Optional[Dict[str, Any]]) -> int:
        """
        Decide how long to keep an artefact given its metadata tags.
        meta may contain: {"tags": ["pii","raw_upload"], ...}
        """
        r = self.root.retention
        if not meta:
            return int(r.default_days)
        tags = [str(t).lower() for t in (meta.get("tags") or []) if t]
        for t in tags:
            if t in r.by_tag:
                return _as_int(r.by_tag[t], r.default_days)
        return int(r.default_days)


# ---------------------------
# Convenience constructor
# ---------------------------

def load_default_engine() -> PolicyEngine:
    """
    Load policy from ./configs/policies.yaml if present; else use defaults.
    """
    default_path = os.getenv("POLICY_FILE", "./configs/policies.yaml")
    return PolicyEngine.from_path(default_path)


__all__ = ["PolicyEngine", "load_default_engine", "RootPolicy", "UploadPolicy", "WebPolicy", "LoggingPolicy", "RetentionPolicy"]
