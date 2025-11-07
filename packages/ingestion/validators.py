# packages/ingestion/validators.py
"""
Upload validators — size/type/AV checks + checksum/dedupe, tutorial-clear.

What this module provides
-------------------------
- validate_filename(name: str) -> None
    Guard against path traversal and weird names (raises ValueError on failure).

- file_size_bytes(path: str) -> int
    Return file size or raise FileNotFoundError.

- compute_sha256(path: str, chunk_mb: int = 4) -> str
    Streaming SHA-256 for dedupe & integrity logs.

- sniff_mime(path: str, filename_hint: str | None = None) -> str
    Best-effort content type guess (fallback to extension).

- run_antivirus(path: str, timeout_s: int = 8) -> dict
    Best-effort ClamAV scan:
      {"ok": True, "engine": "clamav", "infected": False, "signature": None}
      If unavailable: {"ok": True, "engine": "none", "infected": False}

- validate_upload(path: str, *, filename: str | None = None,
                  max_mb: int | None = None,
                  allow_mimes: set[str] | None = None,
                  allow_exts: set[str] | None = None,
                  antivirus: bool = True) -> dict
    One-stop validator. Returns a structured report:
      {
        "ok": bool,
        "errors": [ ... ],
        "warnings": [ ... ],
        "size_bytes": int,
        "checksum": "sha256:...",
        "mime": "application/pdf",
        "ext": ".pdf",
        "av": {...}  # result of run_antivirus()
      }

Design goals
------------
- Keep it **tiny and explicit** with friendly messages.
- Do NOT introduce hard external dependencies; AV is best-effort.
- Separate *policy* (allowed types, max size) from *mechanism* (checksums, MIME).

Typical usage
-------------
from packages.ingestion.validators import validate_upload

report = validate_upload(
    "/tmp/upload_123.pdf",
    filename="Q4_report.pdf",
    max_mb=64,
    allow_mimes=DEFAULT_ALLOWED_MIME,
    allow_exts=DEFAULT_ALLOWED_EXTS,
    antivirus=True,
)
if not report["ok"]:
    raise ValueError("; ".join(report["errors"]))

Environment knobs
-----------------
MAX_FILE_MB (default 64)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional, Set

from packages.core.logging import get_logger
from packages.core.storage import guess_mime

log = get_logger(__name__)

# ---------------------------
# Defaults (policy)
# ---------------------------

DEFAULT_MAX_MB = int(os.getenv("MAX_FILE_MB", "64"))

# MIME prefixes/types we’re comfortable ingesting by default
DEFAULT_ALLOWED_MIME: Set[str] = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/json",
    "application/vnd.ms-excel",  # legacy xls (rare)
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "text/html",
    "image/png",
    "image/jpeg",
    "image/tiff",
}

# File extensions (lowercase, with dot) we accept by default
DEFAULT_ALLOWED_EXTS: Set[str] = {
    ".pdf", ".txt", ".csv", ".json", ".jsonl",
    ".xlsx", ".xls", ".docx", ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
}


# ---------------------------
# Small helpers
# ---------------------------

def validate_filename(name: str) -> None:
    """
    Reject path traversal and obviously bad names.
    - Must not contain slashes or drive letters
    - Only allow a readable subset of characters (letters, numbers, space, dot, dash, underscore)
    """
    if not name or not isinstance(name, str):
        raise ValueError("filename missing or invalid")

    if "/" in name or "\\" in name:
        raise ValueError("filename must not contain path separators")

    # Reject Windows drive-like prefixes (C: etc.)
    if re.match(r"^[A-Za-z]:", name):
        raise ValueError("filename must not contain drive letters")

    # Conservative character whitelist
    if not re.match(r"^[\w .\-\(\)\[\]]{1,200}$", name):
        raise ValueError("filename contains invalid characters")


def file_size_bytes(path: str) -> int:
    """
    Return file size in bytes (raises if not found).
    """
    return os.path.getsize(path)


def _ext_of(filename: Optional[str]) -> str:
    if not filename:
        return ""
    _, ext = os.path.splitext(filename)
    return ext.lower()


def compute_sha256(path: str, chunk_mb: int = 4) -> str:
    """
    Streaming SHA-256 checksum (hex). Prefixed with 'sha256:' for clarity.
    """
    import hashlib

    h = hashlib.sha256()
    chunk = max(1, int(chunk_mb)) * 1024 * 1024
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return "sha256:" + h.hexdigest()


def sniff_mime(path: str, filename_hint: Optional[str] = None) -> str:
    """
    Guess MIME type using filename extension (safe, deterministic).
    (True content-sniffing would need extra libs; we keep it simple here.)
    """
    fn = filename_hint or os.path.basename(path)
    return guess_mime(fn)


def _human_mb(n_bytes: int) -> str:
    return f"{n_bytes/1024/1024:.1f} MB"


# ---------------------------
# Antivirus (best-effort)
# ---------------------------

def _clamav_available() -> bool:
    """
    Return True if 'clamscan' exists on PATH.
    """
    return shutil.which("clamscan") is not None


def run_antivirus(path: str, timeout_s: int = 8) -> dict:
    """
    Best-effort ClamAV scan. Never raises; returns a result dict.

    Returns:
      {
        "ok": bool,           # scanner ran (or no-op) without internal error
        "engine": "clamav"|"none",
        "infected": bool,
        "signature": str|None,
        "detail": str|None,
      }
    """
    if not _clamav_available():
        return {"ok": True, "engine": "none", "infected": False, "signature": None, "detail": "clamscan not available"}

    try:
        # clamscan exit codes: 0=OK, 1=infected, 2=error
        proc = subprocess.run(
            ["clamscan", "--no-summary", path],
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
            check=False,
        )
        out = proc.stdout.strip() + "\n" + proc.stderr.strip()
        if proc.returncode == 0:
            return {"ok": True, "engine": "clamav", "infected": False, "signature": None, "detail": out.strip()}
        if proc.returncode == 1:
            # Try to parse a signature line like: "<path>: <signature> FOUND"
            sig = None
            for line in out.splitlines():
                if line.endswith("FOUND") and ":" in line:
                    sig = line.split(":", 1)[1].replace("FOUND", "").strip()
                    break
            return {"ok": True, "engine": "clamav", "infected": True, "signature": sig, "detail": out.strip()}
        # returncode == 2 → scanner error
        return {"ok": False, "engine": "clamav", "infected": False, "signature": None, "detail": out.strip()}
    except Exception as e:
        return {"ok": False, "engine": "clamav", "infected": False, "signature": None, "detail": str(e)}


# ---------------------------
# Main entry point
# ---------------------------

def validate_upload(
    path: str,
    *,
    filename: Optional[str] = None,
    max_mb: Optional[int] = None,
    allow_mimes: Optional[Set[str]] = None,
    allow_exts: Optional[Set[str]] = None,
    antivirus: bool = True,
) -> dict:
    """
    Validate an uploaded file BEFORE ingestion.

    Policy parameters:
      max_mb     : coarse size cap (default from env)
      allow_mimes: acceptable MIME types (prefix match NOT done here; exact only)
      allow_exts : acceptable file extensions (lowercase, include dot)

    Returns:
      dict with fields:
        ok, errors, warnings, size_bytes, checksum, mime, ext, av
    """
    report = {
        "ok": False,
        "errors": [],
        "warnings": [],
        "size_bytes": 0,
        "checksum": None,
        "mime": None,
        "ext": None,
        "av": None,
    }

    # ---- Existence ----
    if not os.path.exists(path):
        report["errors"].append("file not found")
        return report

    # ---- Filename check (if provided) ----
    ext = _ext_of(filename)
    try:
        if filename:
            validate_filename(filename)
    except ValueError as e:
        report["errors"].append(f"bad filename: {e}")

    # ---- Size ----
    try:
        size = file_size_bytes(path)
        report["size_bytes"] = size
    except Exception as e:
        report["errors"].append(f"size check failed: {e}")
        return report

    cap = int(max_mb or DEFAULT_MAX_MB)
    if size > cap * 1024 * 1024:
        report["errors"].append(f"file too large ({_human_mb(size)} > {cap} MB)")

    # ---- MIME / EXT policy ----
    mime = sniff_mime(path, filename_hint=filename)
    report["mime"] = mime
    report["ext"] = ext or _ext_of(os.path.basename(path))

    mimes = allow_mimes or DEFAULT_ALLOWED_MIME
    exts = allow_exts or DEFAULT_ALLOWED_EXTS

    if mimes and mime not in mimes:
        report["errors"].append(f"unsupported content-type: {mime}")
    if exts and report["ext"] and report["ext"] not in exts:
        report["errors"].append(f"unsupported file extension: {report['ext']}")

    # ---- AV (best-effort) ----
    av_result = {"ok": True, "engine": "none", "infected": False, "signature": None, "detail": None}
    if antivirus:
        av_result = run_antivirus(path)
        if not av_result.get("ok", False):
            report["warnings"].append("antivirus scan failed (proceeding cautiously)")
        if av_result.get("infected"):
            report["errors"].append("malware detected by antivirus")
    report["av"] = av_result

    # ---- Checksum ----
    try:
        report["checksum"] = compute_sha256(path)
    except Exception as e:
        report["warnings"].append(f"checksum failed: {e}")

    # ---- Finalize ----
    report["ok"] = len(report["errors"]) == 0
    return report


# --- Compatibility wrapper for API imports ---  # [NEW]
def file_size_ok(path: str, max_mb: int | None = None) -> bool:
    """
    Compatibility helper used by API routes.

    Returns True if the file at 'path' is <= max_mb MiB (defaults to DEFAULT_MAX_MB).
    """
    try:
        size = file_size_bytes(path)
    except FileNotFoundError:
        return False
    limit = int(max_mb) if max_mb is not None else DEFAULT_MAX_MB
    return size <= (limit * 1024 * 1024)
