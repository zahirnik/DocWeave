# packages/core/audit.py
"""
Append-only audit log (DB + hash-chained JSONL file), tutorial-clear.

What this module provides
-------------------------
- append_event(action, actor, tenant_id=None, details=None) -> str
    Records a security-relevant action both in the SQL DB (AuditEvent table)
    and in an append-only JSONL file with a **hash chain** for tamper-evidence.

- verify_chain() -> dict
    Best-effort verification of the JSONL hash chain; returns a small report.

- last_n(n=50) -> list[dict]
    Read the last N audit entries from the JSONL file for quick diagnostics.

Design goals
------------
- Simple, explicit, and safe: DB write is best-effort; file append is best-effort.
- The JSONL file lives under `outputs_dir/audit.log`. Each line has:
    {
      "id": "<uuid>", "ts": "...", "actor": "...", "tenant_id": "...",
      "action": "ingest.start", "details": {...},
      "prev": "<prev_hash_or_zeros>", "hash": "<sha256_hex_of_line_without_hash>"
    }
- The chain hash is computed over the canonical JSON of the entry **without** the
  `"hash"` key (deterministic separators, sorted keys), concatenated with `prev`.

Notes
-----
- This is a compact, tutorial-friendly approach. In production, consider:
    - shipping logs to a WORM/SIEM system
    - signing entries (KMS-backed) rather than only hashing them
    - storing the chain tip out-of-band (e.g., in a registry)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from packages.core.config import get_settings
from packages.core.logging import get_logger

# DB model (optional — if DB not set up, we still keep file-based log)
try:  # pragma: no cover
    from packages.core.db import session_scope
    from packages.core import models as m
    _HAS_DB = True
except Exception:  # pragma: no cover
    _HAS_DB = False

log = get_logger(__name__)

# ------------- File paths & lock --------------

_lock = threading.Lock()


def _log_path() -> str:
    st = get_settings()
    os.makedirs(st.outputs_dir, exist_ok=True)
    return os.path.join(st.outputs_dir, "audit.log")


def _read_last_hash() -> str:
    """
    Return the previous hash (chain tip) by reading the last non-empty line.
    If file doesn't exist or is empty, return a 64-char zero string.
    """
    path = _log_path()
    tip = "0" * 64
    if not os.path.exists(path):
        return tip
    try:
        with open(path, "rb") as f:
            # seek from end in chunks
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return tip
            block = b""
            step = 4096
            pos = size
            while pos > 0:
                pos = max(0, pos - step)
                f.seek(pos)
                block = f.read(min(step, size - pos)) + block
                lines = block.splitlines()
                if len(lines) >= 1:
                    last = lines[-1]
                    if last.strip():
                        try:
                            obj = json.loads(last.decode("utf-8"))
                            h = str(obj.get("hash") or "")
                            return h if len(h) == 64 else tip
                        except Exception:
                            return tip
            return tip
    except Exception:
        return tip


def _sha256_hex(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


# ------------- Public API ---------------------

def append_event(
    action: str,
    actor: str,
    *,
    tenant_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Append an audit event to DB (if available) and to the hash-chained JSONL file.

    Args:
      action   : verb-like string (e.g., "ingest.start", "api.chat", "admin.key.create")
      actor    : who did it (user id, key id, or "system/worker")
      tenant_id: optional tenant/workspace id
      details  : small JSON-serializable dict (avoid secrets!)
    Returns:
      event id (uuid hex)
    """
    ev_id = uuid.uuid4().hex
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    ten = tenant_id or (details or {}).get("tenant_id")

    # 1) DB write (best-effort)
    if _HAS_DB:
        try:  # pragma: no cover
            with session_scope() as s:
                s.add(
                    m.AuditEvent(  # type: ignore[attr-defined]
                        id=ev_id,
                        ts=dt.datetime.fromisoformat(ts),
                        actor=actor,
                        action=action,
                        tenant_id=str(ten) if ten else None,
                        details=details or {},
                    )
                )
        except Exception as e:
            log.warning("Audit DB write failed: %s", e)

    # 2) File append with hash chain (strongly ordered per process via lock)
    try:
        with _lock:
            prev = _read_last_hash()
            entry_core = {
                "id": ev_id,
                "ts": ts,
                "actor": actor,
                "tenant_id": ten,
                "action": action,
                "details": details or {},
                "prev": prev,
                # NOTE: "hash" is computed over the JSON of this object *without* the "hash" field.
            }
            payload = json.dumps(entry_core, sort_keys=True, separators=(",", ":"))
            chain_hash = _sha256_hex(prev + payload)

            entry = dict(entry_core)
            entry["hash"] = chain_hash

            path = _log_path()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
                f.write("\n")
    except Exception as e:
        log.warning("Audit file append failed: %s", e)

    return ev_id


def verify_chain(limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Verify the hash chain end-to-end (or up to `limit` last entries).
    Returns a small diagnostic dict:
        {"ok": bool, "checked": int, "mismatch_at": Optional[int]}
    """
    path = _log_path()
    if not os.path.exists(path):
        return {"ok": True, "checked": 0, "mismatch_at": None}

    ok = True
    mismatch_at = None
    checked = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if limit is not None and limit > 0:
            lines = lines[-int(limit):]

        prev = "0" * 64
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec_hash = obj.get("hash", "")
            # recompute
            core = dict(obj)
            core.pop("hash", None)
            payload = json.dumps(core, sort_keys=True, separators=(",", ":"))
            expect = _sha256_hex(prev + payload)
            if expect != rec_hash:
                ok = False
                mismatch_at = idx
                break
            prev = rec_hash
            checked += 1

    except Exception as e:
        log.warning("Audit verify failed: %s", e)
        ok = False

    return {"ok": ok, "checked": checked, "mismatch_at": mismatch_at}


def last_n(n: int = 50) -> List[Dict[str, Any]]:
    """
    Return the last N entries from the JSONL audit file (most recent last).
    """
    path = _log_path()
    if not os.path.exists(path) or n <= 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-int(n):]
        return [json.loads(l) for l in lines if l.strip()]
    except Exception:
        return []
