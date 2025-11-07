# packages/retriever/vectorstores/chroma_store.py
"""
Chroma store — dev-friendly, file-based vector DB wrapper (tutorial-clear).

Why Chroma?
-----------
- Great for **local development** and quick demos.
- Single-process, embedded DB that persists to a directory.
- Not meant for heavy, multi-tenant prod use — use pgvector/Qdrant there.

What this module provides
-------------------------
A tiny interface mirroring the other stores:

Class
-----
ChromaStore(persist_dir: str = ".chroma", collection_prefix: str = "rag")
  - ensure_client() -> None
  - create_collection(name: str, *, dimension: int | None = None, metadata: dict | None = None) -> str
  - delete_collection(name: str) -> bool
  - list_collections() -> list[dict]
  - get_collection(name: str) -> dict | None
  - upsert_texts(collection: str, items: list[dict]) -> int
      items = [
        {
          "id": "optional-uuid",
          "text": "chunk text",
          "metadata": {...},              # JSON-serializable dict
          "embedding": [float, ...]       # length == dimension (not enforced by Chroma)
        }, ...
      ]
  - search(collection: str, query: list[float], top_k: int = 5, filters: dict | None = None)
      -> list[{"id","text","metadata","score"}]     # score = distance (lower is better)
  - stats(collection: str) -> dict
  - get_by_id(collection: str, id: str) -> dict | None
  - delete_by_ids(collection: str, ids: list[str]) -> int

Notes
-----
- Chroma does **not** require a fixed dimension per collection (we keep `dimension`
  as a doc-only hint for parity with other stores).
- Filtering is basic; we apply an AND of exact matches against metadata.
- Distances are backend-defined; we convert to a "score" where lower is better.

Install
-------
pip install chromadb
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Any  # [CHANGED] add Any

from packages.core.logging import get_logger

log = get_logger(__name__)

# ----- IMPORT STRATEGY (robust to NumPy 2 / Chroma API drift) -----
# We avoid hard-failing at import time. We:
#  1) Add a tiny NumPy 2.x shim if needed (some Chroma versions reference np.float_).
#  2) Defer importing `chromadb` until ensure_client().
#  3) Build `include` lists compatible with both Enum and string-based APIs.
# [NEW]
try:
    import numpy as _np  # type: ignore
    if not hasattr(_np, "float_"):  # NumPy 2 removed alias; some deps still expect it
        _np.float_ = _np.float64  # type: ignore[attr-defined]
    if not hasattr(_np, "int_"):
        _np.int_ = _np.int64      # type: ignore[attr-defined]
except Exception:
    pass


def _full_name(prefix: str, name: str) -> str:
    return f"{prefix}_{name}".strip("_")


def _distance_to_score(d: float) -> float:
    """
    Chroma returns distances where **smaller is better** for cosine/Euclidean.
    We pass-through as our 'score' for parity with other stores.
    """
    try:
        return float(d)
    except Exception:
        return 0.0


# [NEW] helper to construct version-tolerant "include" argument
def _include_tuple(chromadb_mod: Any):
    """
    Returns a tuple (include_for_query, include_for_get) compatible with the installed Chroma.
    If IncludeEnum exists, we use it; otherwise we return string keys.
    """
    try:
        from chromadb.api.types import IncludeEnum  # type: ignore
        inc_query = [IncludeEnum.distances, IncludeEnum.metadatas, IncludeEnum.documents]
        inc_get = [IncludeEnum.metadatas, IncludeEnum.documents]
        return inc_query, inc_get
    except Exception:
        # String-based includes (supported in multiple versions)
        return (["distances", "metadatas", "documents"], ["metadatas", "documents"])


class ChromaStore:
    """
    Minimal Chroma-backed vector store for **development** use.
    """

    def __init__(self, persist_dir: str = ".chroma", collection_prefix: str = "rag"):
        """
        Args:
          persist_dir       : directory where Chroma persists data
          collection_prefix : prefix for collection names to avoid collisions
        """
        self.persist_dir = persist_dir
        self.prefix = collection_prefix.strip() or "rag"
        self._client: Optional[Any] = None   # [CHANGED] loosen type (ClientAPI import may vary)
        self._include_query = None           # [NEW]
        self._include_get = None             # [NEW]

    # ---------------------------
    # Client/init
    # ---------------------------

    def ensure_client(self) -> None:
        """Create a client (persisting to disk). Safe to call multiple times."""
        if self._client is not None:
            return
        try:
            import chromadb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "chromadb is required for ChromaStore. Install with:\n"
                "  pip install chromadb"
            ) from e

        os.makedirs(self.persist_dir, exist_ok=True)

        # Prefer PersistentClient; fallback to Client() if older API.  [CHANGED]
        client = None
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=self.persist_dir)  # type: ignore[attr-defined]
        elif hasattr(chromadb, "Client"):
            client = chromadb.Client()  # type: ignore[attr-defined]
        else:
            raise RuntimeError(
                "Unsupported chromadb version: no PersistentClient/Client found. "
                "Try: pip install 'chromadb>=0.4.0'"
            )

        # Basic ping — list collections
        _ = client.list_collections()
        self._client = client

        # Build include lists compatible with installed version  [NEW]
        self._include_query, self._include_get = _include_tuple(chromadb)

    def _get_collection(self, name: str):
        self.ensure_client()
        assert self._client is not None
        full = _full_name(self.prefix, name)
        try:
            return self._client.get_collection(full)
        except Exception:
            return None

    def _get_or_create_collection(self, name: str):
        self.ensure_client()
        assert self._client is not None
        full = _full_name(self.prefix, name)
        try:
            return self._client.get_collection(full)
        except Exception:
            return self._client.create_collection(full)

    # ---------------------------
    # Collections
    # ---------------------------

    def create_collection(self, name: str, *, dimension: Optional[int] = None, metadata: Optional[dict] = None) -> str:
        """
        Create (or get) a collection and return its name.
        - `dimension` is not enforced by Chroma; it's a hint for parity.
        """
        if not name or not name.strip():
            raise ValueError("collection name required")
        _ = self._get_or_create_collection(name)
        return name

    def get_collection(self, name: str) -> Optional[dict]:
        col = self._get_collection(name)
        if not col:
            return None
        try:
            count = col.count()
        except Exception:
            count = None
        return {"name": name, "count": count}

    def list_collections(self) -> List[dict]:
        self.ensure_client()
        assert self._client is not None
        out = []
        for c in self._client.list_collections() or []:
            cname = getattr(c, "name", "") or ""
            short = cname
            pfx = f"{self.prefix}_"
            if cname.startswith(pfx):
                short = cname[len(pfx):]
            out.append({"name": short})
        return out

    def delete_collection(self, name: str) -> bool:
        self.ensure_client()
        assert self._client is not None
        full = _full_name(self.prefix, name)
        try:
            self._client.delete_collection(full)
            return True
        except Exception:
            return False

    # ---------------------------
    # Upsert
    # ---------------------------

    def upsert_texts(self, collection: str, items: List[dict]) -> int:
        """
        Upsert text chunks with embeddings.

        Each item: {"id": str|None, "text": str, "metadata": dict, "embedding": list[float]}
        If "id" absent → Chroma requires ids; we generate simple deterministic ids from order.
        """
        if not items:
            return 0
        col = self._get_or_create_collection(collection)

        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []
        vecs: List[List[float]] = []

        for i, it in enumerate(items):
            pid = str(it.get("id") or f"auto-{i}")
            txt = it.get("text") or ""
            meta = it.get("metadata") or {}
            emb = it.get("embedding")
            if not isinstance(emb, (list, tuple)) or not emb:
                raise ValueError("embedding is required (list[float])")
            ids.append(pid)
            docs.append(txt)
            metas.append(meta)
            vecs.append(list(map(float, emb)))

        try:
            # Chroma uses .add for new points; .upsert is available in newer versions.
            if hasattr(col, "upsert"):
                col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)  # type: ignore[attr-defined]
            else:
                try:
                    col.delete(ids=ids)
                except Exception:
                    pass
                col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)
        except Exception as e:
            raise RuntimeError(f"Chroma upsert failed: {e}") from e

        return len(ids)

    # ---------------------------
    # Search
    # ---------------------------

    def _filter_to_where(self, filters: Optional[Dict]) -> Optional[Dict]:
        """
        Convert {"year":2024,"ticker":"AAPL"} → {"$and":[{"year":{"$eq":2024}}, {"ticker":{"$eq":"AAPL"}}]}
        Keep permissive shape to tolerate Chroma version changes.
        """
        if not filters:
            return None
        clauses = []
        for k, v in filters.items():
            clauses.append({f"{k}": {"$eq": v}})  # [CHANGED] drop "metadata." prefix for wider compat
        return {"$and": clauses} if clauses else None

    def search(
        self,
        collection: str,
        query: Sequence[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[dict]:
        """
        Vector search in Chroma. Returns:
          [{"id","text","metadata","score"}], where score is a distance (lower is better).
        """
        col = self._get_collection(collection)
        if not col:
            raise ValueError(f"collection not found: {collection!r}")

        qvec = [float(x) for x in query]
        where = self._filter_to_where(filters)

        try:
            include_arg = self._include_query or ["distances", "metadatas", "documents"]  # [NEW]
            res = col.query(
                query_embeddings=[qvec],
                n_results=max(1, int(top_k)),
                include=include_arg,   # works with IncludeEnum or strings
                where=where,
            )
        except Exception as e:
            raise RuntimeError(f"Chroma query failed: {e}") from e

        # Normalize result shape
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[dict] = []
        for i in range(min(len(ids), len(docs), len(metas), len(dists))):
            out.append(
                {
                    "id": ids[i],
                    "text": docs[i] or "",
                    "metadata": metas[i] or {},
                    "score": _distance_to_score(dists[i]),
                }
            )
        return out

    # ---------------------------
    # Utilities
    # ---------------------------

    def stats(self, collection: str) -> dict:
        col = self._get_collection(collection)
        if not col:
            return {"count": 0}
        try:
            n = col.count()
        except Exception:
            n = 0
        return {"count": int(n or 0)}

    def get_by_id(self, collection: str, id: str) -> Optional[dict]:
        col = self._get_collection(collection)
        if not col:
            return None
        try:
            include_arg = self._include_get or ["metadatas", "documents"]  # [NEW]
            res = col.get(ids=[id], include=include_arg)  # type: ignore[arg-type]
            if not res or not res.get("ids"):
                return None
            return {
                "id": id,
                "text": (res.get("documents") or [""])[0] or "",
                "metadata": (res.get("metadatas") or [{}])[0] or {},
            }
        except Exception:
            return None

    def delete_by_ids(self, collection: str, ids: Iterable[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        col = self._get_collection(collection)
        if not col:
            return 0
        try:
            col.delete(ids=ids)
            return len(ids)
        except Exception as e:
            raise RuntimeError(f"Chroma delete failed: {e}") from e
