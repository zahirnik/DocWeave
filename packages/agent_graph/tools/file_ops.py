# packages/agent_graph/tools/file_ops.py
"""
File ops tool — safe, tiny helpers for temp/output files and "signed" paths.

What this module provides
-------------------------
Safe, explicit functions the agent/tools can use to read/write artefacts
(e.g., CSVs, charts, small reports) **only** inside project-managed directories.

Roots (relative to repo)
------------------------
- TMP_DIR     = "./data/tmp"       (ephemeral working files)
- OUTPUTS_DIR = "./data/outputs"   (artefacts you may want to show/download)

Functions
---------
- ensure_dirs() -> None
- safe_join(base, *parts) -> str                  # normalises & blocks path traversal
- write_bytes_atomic(path, data: bytes) -> str
- write_text_atomic(path, text: str) -> str
- save_temp_bytes(data: bytes, suffix=".bin", subdir="") -> str
- save_temp_text(text: str, suffix=".txt", subdir="") -> str
- list_outputs(pattern="*") -> list[dict]         # [{"path","size","mtime"}]
- read_text_safe(path, max_bytes=2_000_000) -> str
- read_bytes_safe(path, max_bytes=2_000_000) -> bytes
- remove_file_safe(path) -> bool

"Signed" paths for API handoff (no external service)
----------------------------------------------------
- make_signed_path(rel_path: str, expires_s=3600) -> dict
    Returns a small token bundle:
      {"path": "data/outputs/chart_abc123.png", "exp": 1735689600, "sig": "base64(hmac-sha256)"}
    Your API can later verify with `verify_signed_path(token_dict)` before serving the file.

Security notes
--------------
- All file reads/writes are restricted to TMP_DIR or OUTPUTS_DIR (and subfolders).
- Path traversal is prevented by strict `safe_join`.
- JSON/web endpoints should verify signatures **and** normalise paths again server-side.

No heavy dependencies. Tutorial-clear and easy to test.
"""

from __future__ import annotations

import base64
import glob
import hashlib
import hmac
import io
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Project-local roots
TMP_DIR = "./data/tmp"
OUTPUTS_DIR = "./data/outputs"

# Secret for signing paths (read from env or fall back to process-unique)
_FILE_TOKEN_SECRET = (os.getenv("FILE_TOKEN_SECRET") or uuid.uuid4().hex).encode("utf-8")


# ---------------------------
# Directory management
# ---------------------------

def ensure_dirs() -> None:
    """Create TMP_DIR and OUTPUTS_DIR if they don't exist."""
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------
# Safe path handling
# ---------------------------

def _norm(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")

def _is_within(base: str, path: str) -> bool:
    base_abs = os.path.abspath(base)
    path_abs = os.path.abspath(path)
    try:
        # Python 3.10+: use os.path.commonpath to ensure base is a prefix
        return os.path.commonpath([base_abs]) == os.path.commonpath([base_abs, path_abs])
    except Exception:
        # Fallback: conservative check
        return _norm(path_abs).startswith(_norm(base_abs))

def safe_join(base: str, *parts: str) -> str:
    """
    Join path parts under `base`, normalise, and block path traversal.
    Raises ValueError if the result escapes the base.
    """
    if not base:
        raise ValueError("base is required")
    path = os.path.join(base, *parts)
    path = _norm(path)
    if not _is_within(base, path):
        raise ValueError("unsafe path (path traversal blocked)")
    return path


# ---------------------------
# Atomic writes
# ---------------------------

def write_bytes_atomic(path: str, data: bytes) -> str:
    """
    Atomically write bytes to `path` (via temp file + rename).
    Ensures parent directory exists.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path

def write_text_atomic(path: str, text: str, *, encoding: str = "utf-8") -> str:
    """
    Atomically write text to `path` (UTF-8 by default).
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return write_bytes_atomic(path, text.encode(encoding))


# ---------------------------
# Temp helpers
# ---------------------------

def save_temp_bytes(data: bytes, *, suffix: str = ".bin", subdir: str = "") -> str:
    """
    Save bytes to TMP_DIR/<subdir>/tmp_<uuid><suffix> and return the repo-relative path.
    """
    ensure_dirs()
    name = f"tmp_{uuid.uuid4().hex}{'' if suffix.startswith('.') else '.'}{suffix}"
    path = safe_join(TMP_DIR, subdir, name) if subdir else safe_join(TMP_DIR, name)
    write_bytes_atomic(path, data)
    return _norm(path)

def save_temp_text(text: str, *, suffix: str = ".txt", subdir: str = "") -> str:
    """
    Save text to TMP_DIR/<subdir>/tmp_<uuid><suffix> and return the repo-relative path.
    """
    ensure_dirs()
    name = f"tmp_{uuid.uuid4().hex}{'' if suffix.startswith('.') else '.'}{suffix}"
    path = safe_join(TMP_DIR, subdir, name) if subdir else safe_join(TMP_DIR, name)
    write_text_atomic(path, text)
    return _norm(path)


# ---------------------------
# Outputs listing
# ---------------------------

def list_outputs(pattern: str = "*") -> List[Dict[str, object]]:
    """
    List OUTPUTS_DIR files matching a glob pattern (non-recursive by default).
    Returns [{"path","size","mtime"}].
    """
    ensure_dirs()
    glb = safe_join(OUTPUTS_DIR, pattern)
    out: List[Dict[str, object]] = []
    for p in sorted(glob.glob(glb)):
        try:
            st = os.stat(p)
            out.append({"path": _norm(p), "size": int(st.st_size), "mtime": int(st.st_mtime)})
        except Exception:
            continue
    return out


# ---------------------------
# Safe reads
# ---------------------------

def _require_allowed_roots(path: str) -> None:
    ok = _is_within(TMP_DIR, path) or _is_within(OUTPUTS_DIR, path)
    if not ok:
        raise PermissionError("read/write outside allowed roots is forbidden")

def read_text_safe(path: str, *, max_bytes: int = 2_000_000, encoding: str = "utf-8") -> str:
    """
    Read a small text file within allowed roots, with a hard byte cap.
    """
    p = _norm(path)
    _require_allowed_roots(p)
    size = os.path.getsize(p)
    if size > max_bytes:
        raise ValueError(f"file too large ({size} bytes > {max_bytes})")
    with open(p, "rb") as f:
        data = f.read()
    try:
        return data.decode(encoding, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")

def read_bytes_safe(path: str, *, max_bytes: int = 2_000_000) -> bytes:
    """
    Read a small binary file within allowed roots, with a hard byte cap.
    """
    p = _norm(path)
    _require_allowed_roots(p)
    size = os.path.getsize(p)
    if size > max_bytes:
        raise ValueError(f"file too large ({size} bytes > {max_bytes})")
    with open(p, "rb") as f:
        return f.read()


# ---------------------------
# Removal
# ---------------------------

def remove_file_safe(path: str) -> bool:
    """
    Delete a file within allowed roots. Returns True if removed.
    """
    p = _norm(path)
    _require_allowed_roots(p)
    try:
        os.remove(p)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


# ---------------------------
# "Signed" path tokens
# ---------------------------

def _sign_blob(blob: bytes) -> str:
    sig = hmac.new(_FILE_TOKEN_SECRET, blob, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")

def make_signed_path(rel_path: str, *, expires_s: int = 3600) -> Dict[str, object]:
    """
    Create a small, verifiable token for serving a file via your API.

    Returns:
      {"path": "data/outputs/file.png", "exp": <unix_ts>, "sig": "<b64url>"}

    The API should:
      1) Call `verify_signed_path(token)` to validate.
      2) Re-run safe_join + allowed-roots checks.
      3) Stream the file if ok.
    """
    # Normalise and ensure it's under allowed roots (relative OK)
    p = _norm(rel_path)
    abs_p = os.path.abspath(p)
    _require_allowed_roots(abs_p)

    exp = int(time.time()) + max(1, int(expires_s))
    payload = f"{p}|{exp}".encode("utf-8")
    sig = _sign_blob(payload)
    return {"path": p, "exp": exp, "sig": sig}

def verify_signed_path(token: Dict[str, object]) -> Tuple[bool, str]:
    """
    Verify a token produced by `make_signed_path`.

    Returns: (ok, reason_if_not_ok)
    """
    try:
        p = _norm(str(token.get("path")))
        exp = int(token.get("exp"))
        sig = str(token.get("sig"))
    except Exception:
        return (False, "malformed token")

    if time.time() > exp:
        return (False, "expired")

    payload = f"{p}|{exp}".encode("utf-8")
    expect = _sign_blob(payload)
    if not hmac.compare_digest(expect, sig):
        return (False, "bad signature")

    # final safety: ensure path remains within allowed roots
    if not (_is_within(TMP_DIR, p) or _is_within(OUTPUTS_DIR, p)):
        return (False, "path outside allowed roots")

    return (True, "")


__all__ = [
    "TMP_DIR",
    "OUTPUTS_DIR",
    "ensure_dirs",
    "safe_join",
    "write_bytes_atomic",
    "write_text_atomic",
    "save_temp_bytes",
    "save_temp_text",
    "list_outputs",
    "read_text_safe",
    "read_bytes_safe",
    "remove_file_safe",
    "make_signed_path",
    "verify_signed_path",
]
