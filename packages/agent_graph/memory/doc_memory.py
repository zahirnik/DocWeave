# packages/agent_graph/memory/doc_memory.py
"""
Document memory — cache useful passages across turns and runs.

What this module provides
-------------------------
A tiny, file-backed cache of previously retrieved passages (chunks) so the agent
can quickly reuse strong contexts without re-querying the vector/BM25 stores.

Class
-----
DocMemory(tenant_id: str, collection: str, root_dir="./data/doc_memory", capacity=5000, ttl_days=90)
  - add(hits: list[dict]) -> int
      Accepts hits from retriever.search(...) with fields:
        {"id","text","metadata","score", ...}
      Stores/updates entries (dedup by id) and refreshes last_access.

  - query(filters: dict | None = None, top_k: int = 12) -> list[dict]
      Returns top passages by a composite of recency + cached score.
      Filters is an AND of exact matches over `metadata` keys.

  - get_recent(n: int = 12) -> list[dict]
      Return most recently accessed entries.

  - invalidate_by_source(source: str) -> int
      Remove entries whose metadata["source"] (or ["filename"] / ["url"]) matches.

  - invalidate_older_than(days: int) -> int
      Remove entries not accessed within `days`.

  - clear() -> None
      Remove all entries for (tenant_id, collection).

Design goals
------------
- Tutorial-clear, zero heavy dependencies.
- Robust to crashes (atomic writes).
- Deterministic, small, and easy to unit-test.

Persistence layout
------------------
<root_dir>/<tenant_id>/<collection>/cache.json

Entry schema
------------
{
  "id": "chunk-id",
  "text": "chunk text (trimmed)",
  "metadata": {...},                # JSON-serializable
  "score": 0.87,                    # fused/rerank score from retrieval (0..1 recommended)
  "first_seen": "iso8601",
  "last_access": "iso8601",
  "access_count": 3
}

Scoring
-------
`query()` sorts by:
  S = 0.7 * score + 0.3 * recency_boost(last_access)
where recency_boost = exp(-age_days / 30). Tuned for "recent but still good".
"""

from __future__ import annotations

import json
import math
import os
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------
# Small helpers
# ---------------------------

def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _iso() -> str:
    return _utc_now().isoformat()

def _parse_iso(s: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return _utc_now()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _truncate(s: str, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else (s[: n - 1] + "…")

def _meta_source(meta: Dict[str, Any]) -> str:
    # Normalise likely source fields
    for k in ("source", "filename", "url", "path"):
        v = meta.get(k)
        if v:
            return str(v)
    return ""

def _match_filters(meta: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
    if not flt:
        return True
    for k, v in flt.items():
        if meta.get(k) != v:
            return False
    return True


# ---------------------------
# In-memory entry model
# ---------------------------

@dataclass
class _Entry:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    first_seen: str = field(default_factory=_iso)
    last_access: str = field(default_factory=_iso)
    access_count: int = 0

    def touch(self) -> None:
        self.last_access = _iso()
        self.access_count = int(self.access_count) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": float(self.score),
            "first_seen": self.first_seen,
            "last_access": self.last_access,
            "access_count": int(self.access_count),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "_Entry":
        return _Entry(
            id=str(d.get("id") or ""),
            text=str(d.get("text") or ""),
            metadata=dict(d.get("metadata") or {}),
            score=float(d.get("score") or 0.0),
            first_seen=str(d.get("first_seen") or _iso()),
            last_access=str(d.get("last_access") or _iso()),
            access_count=int(d.get("access_count") or 0),
        )


# ---------------------------
# DocMemory
# ---------------------------

class DocMemory:
    """
    File-backed cache of passages for (tenant_id, collection).
    """

    def __init__(
        self,
        tenant_id: str,
        collection: str,
        *,
        root_dir: str = "./data/doc_memory",
        capacity: int = 5000,
        ttl_days: int = 90,
        max_text_len: int = 20_000,
    ):
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not collection:
            raise ValueError("collection is required")
        self.tenant_id = tenant_id
        self.collection = collection
        self.root_dir = root_dir
        self.capacity = max(1, int(capacity))
        self.ttl_days = max(1, int(ttl_days))
        self.max_text_len = max(200, int(max_text_len))
        self._entries: Dict[str, _Entry] = {}

        self._load()

    # ------------- paths / IO -------------

    def _dir(self) -> str:
        d = os.path.join(self.root_dir, self.tenant_id, self.collection)
        _ensure_dir(d)
        return d

    def _path(self) -> str:
        return os.path.join(self._dir(), "cache.json")

    def _load(self) -> None:
        p = self._path()
        if not os.path.exists(p):
            self._entries = {}
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or []
            self._entries = {e["id"]: _Entry.from_dict(e) for e in data if e.get("id")}
        except Exception:
            self._entries = {}

    def _save(self) -> None:
        path = self._path()
        tmp = path + ".tmp"
        data = [e.to_dict() for e in self._entries.values()]
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ------------- public API -------------

    def add(self, hits: List[Dict[str, Any]]) -> int:
        """
        Add/refresh hits (dedup by id). Trims to capacity using LRU-ish policy.
        Returns number of upserts.
        """
        if not hits:
            return 0
        n = 0
        now_iso = _iso()
        for h in hits:
            hid = str(h.get("id") or "").strip()
            if not hid:
                continue
            txt = _truncate(h.get("text") or "", self.max_text_len)
            meta = dict(h.get("metadata") or {})
            score = float(h.get("score") or h.get("source", {}).get("vector") or 0.0)

            if hid in self._entries:
                # update existing
                e = self._entries[hid]
                # Refresh only if better score or longer text (keep richer record)
                if score > e.score:
                    e.score = score
                if len(txt) > len(e.text):
                    e.text = txt
                # Merge metadata shallowly (new keys override)
                e.metadata.update(meta)
                e.last_access = now_iso
                e.access_count = int(e.access_count) + 1
            else:
                self._entries[hid] = _Entry(
                    id=hid,
                    text=txt,
                    metadata=meta,
                    score=score,
                    first_seen=now_iso,
                    last_access=now_iso,
                    access_count=1,
                )
            n += 1

        # Evict old entries if over capacity
        self._evict_if_needed()
        self._save()
        return n

    def query(self, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 12) -> List[Dict[str, Any]]:
        """
        Return best cached passages by composite score:
          S = 0.7 * score + 0.3 * recency_boost(last_access)
        """
        self._expire_ttl()
        items = []
        now = _utc_now()
        for e in self._entries.values():
            if not _match_filters(e.metadata, filters):
                continue
            age_days = max(0.0, (now - _parse_iso(e.last_access)).total_seconds() / 86400.0)
            rec = math.exp(-age_days / 30.0)  # 30-day half-life-ish
            s = 0.7 * float(e.score) + 0.3 * rec
            items.append((s, e))

        items.sort(key=lambda t: t[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for _, e in items[: max(1, int(top_k))]:
            # touch on read
            e.touch()
            out.append(
                {
                    "id": e.id,
                    "text": e.text,
                    "metadata": e.metadata,
                    "score": float(e.score),
                    "source": {"memo_recency": e.last_access, "memo_accesses": e.access_count},
                }
            )
        if out:
            self._save()
        return out

    def get_recent(self, n: int = 12) -> List[dict]:
        """
        Return most recently accessed entries (no filtering).
        """
        self._expire_ttl()
        items = sorted(self._entries.values(), key=lambda e: e.last_access, reverse=True)
        out: List[Dict[str, Any]] = []
        for e in items[: max(1, int(n))]:
            out.append({"id": e.id, "text": e.text, "metadata": e.metadata, "score": float(e.score)})
        return out

    def invalidate_by_source(self, source: str) -> int:
        """
        Remove entries whose metadata source/filename/url matches (substring match).
        """
        if not source:
            return 0
        needle = str(source).lower().strip()
        rm: List[str] = []
        for e in self._entries.values():
            s = _meta_source(e.metadata).lower()
            if needle and needle in s:
                rm.append(e.id)
        for k in rm:
            self._entries.pop(k, None)
        if rm:
            self._save()
        return len(rm)

    def invalidate_older_than(self, days: int) -> int:
        """
        Remove entries not accessed within `days`.
        """
        days = max(1, int(days))
        cutoff = _utc_now() - dt.timedelta(days=days)
        rm: List[str] = []
        for k, e in list(self._entries.items()):
            if _parse_iso(e.last_access) < cutoff:
                rm.append(k)
        for k in rm:
            self._entries.pop(k, None)
        if rm:
            self._save()
        return len(rm)

    def clear(self) -> None:
        self._entries.clear()
        # remove file on disk
        p = self._path()
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # ------------- internals -------------

    def _expire_ttl(self) -> None:
        """
        Auto-expire items beyond ttl_days since last_access.
        """
        cutoff = _utc_now() - dt.timedelta(days=self.ttl_days)
        changed = False
        for k in list(self._entries.keys()):
            if _parse_iso(self._entries[k].last_access) < cutoff:
                del self._entries[k]
                changed = True
        if changed:
            self._save()

    def _evict_if_needed(self) -> None:
        """
        Trim to capacity by evicting least recently accessed first (LRU-ish),
        with a small bias to keep higher-scoring items.
        """
        n = len(self._entries)
        if n <= self.capacity:
            return
        # Rank by (age DESC, score ASC) for eviction
        items = list(self._entries.values())
        items.sort(
            key=lambda e: (_parse_iso(e.last_access), float(e.score)),
        )
        # Evict oldest/lowest until within capacity
        to_evict = n - self.capacity
        for e in items[:to_evict]:
            self._entries.pop(e.id, None)

    # ------------- debugging -------------

    def stats(self) -> Dict[str, Any]:
        """
        Quick stats for monitoring/debugging.
        """
        self._expire_ttl()
        if not self._entries:
            return {"count": 0, "oldest": None, "newest": None}
        times = sorted(_parse_iso(e.last_access) for e in self._entries.values())
        return {
            "count": len(self._entries),
            "oldest": times[0].isoformat(),
            "newest": times[-1].isoformat(),
        }


__all__ = ["DocMemory"]
