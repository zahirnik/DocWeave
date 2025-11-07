# packages/retriever/vectorstores/qdrant_store.py
"""
Qdrant store — small, explicit wrapper around Qdrant vector DB.

What this module provides
-------------------------
A tutorial-clear interface (mirrors PgVectorStore) for storing/searching
embeddings using Qdrant. It keeps one simple vector field named "text".

Class
-----
QdrantStore(url: str = "http://localhost:6333", api_key: str | None = None, prefer_grpc: bool = False)
  - ensure_client() -> None
  - create_collection(name: str, *, dimension: int, metadata: dict | None = None) -> str
  - delete_collection(name: str) -> bool
  - list_collections() -> list[dict]
  - get_collection(name: str) -> dict | None
  - upsert_texts(collection: str, items: list[dict]) -> int
      items = [
        {
          "id": "optional-uuid-or-int",
          "text": "chunk text",
          "metadata": {...},              # JSON-serializable dict
          "embedding": [float, ...]       # length == dimension
        }, ...
      ]
  - search(collection: str, query: list[float], top_k: int = 5, filters: dict | None = None)
      -> list[{"id","text","metadata","score"}]     # score = distance (lower is better with cosine)
  - stats(collection: str) -> dict
  - get_by_id(collection: str, id: str | int) -> dict | None
  - delete_by_ids(collection: str, ids: list[str | int]) -> int

Design notes
------------
- Uses cosine distance. Prefer L2-normalized embeddings upstream.
- Filters: a tiny helper maps a dict {"year": 2024, "ticker": "AAPL"} to
  a Qdrant Filter with "must" MatchValue conditions.
- Keeps payload fields:
    payload = {
      "text": "...",
      "metadata": {...}
    }

Install
-------
pip install qdrant-client

Qdrant quick start (docker)
---------------------------
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence

from packages.core.logging import get_logger

log = get_logger(__name__)

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "qdrant-client is required for QdrantStore. Install with:\n"
        "  pip install qdrant-client"
    ) from e


# ---------------------------
# Small helpers
# ---------------------------

def _vector_name() -> str:
    return "text"  # single-vector collection (keeps code simple)


def _to_filter(filters: Optional[Dict]) -> Optional[qm.Filter]:
    """
    Convert a simple dict of exact matches to a Qdrant Filter.
    Example:
      {"year": 2024, "ticker": "AAPL"}  →  must: [FieldCondition(key="metadata.year", MatchValue(...)), ...]
    """
    if not filters:
        return None
    conds: List[qm.FieldCondition] = []
    for k, v in filters.items():
        # store under payload["metadata"][k], so key is "metadata.<k>"
        key = f"metadata.{k}"
        conds.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=v)))
    return qm.Filter(must=conds)


# ---------------------------
# Store
# ---------------------------

class QdrantStore:
    """
    Minimal Qdrant-backed vector store mirroring the PgVectorStore shape.
    """

    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None, prefer_grpc: bool = False):
        """
        Args:
          url        : Qdrant endpoint, e.g., http://localhost:6333 or https://<cloud-endpoint>
          api_key    : when using a secured/cloud instance
          prefer_grpc: set True to use gRPC (port 6334), else HTTP
        """
        self.url = url
        self.api_key = api_key
        self.prefer_grpc = bool(prefer_grpc)
        self._client: Optional[QdrantClient] = None

    # ---------------------------
    # Client/init
    # ---------------------------

    def ensure_client(self) -> None:
        """Create a client and ping the server (safe to call multiple times)."""
        if self._client is not None:
            return
        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
            timeout=30,
        )
        try:
            _ = self._client.get_collections()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Could not connect to Qdrant at {self.url}: {e}") from e

    # ---------------------------
    # Collections
    # ---------------------------

    def create_collection(self, name: str, *, dimension: int, metadata: Optional[dict] = None) -> str:
        """
        Create a collection if missing. Returns its name (id).

        Note: Qdrant names are strings; we return `name` to preserve symmetry with PgVector.
        """
        if not name or not name.strip():
            raise ValueError("collection name required")

        self.ensure_client()
        assert self._client is not None

        # Already exists?
        try:
            info = self._client.get_collection(name)
            # If exists but dimension mismatches, we raise to avoid silent errors
            vecs = info.vectors_config  # can be Scalar or Map; we handle single-vector config
            try:
                # Newer qdrant-client returns VectorParams or VectorParamsMap
                if hasattr(vecs, "size"):  # VectorParams
                    if int(vecs.size) != int(dimension):
                        raise RuntimeError(f"Collection '{name}' exists with size={vecs.size}, expected {dimension}")
                elif hasattr(vecs, "configs"):  # VectorParamsMap
                    cfg = vecs.configs.get(_vector_name())
                    if not cfg or int(cfg.size) != int(dimension):
                        raise RuntimeError(f"Collection '{name}' exists with a different vector config")
            except Exception:
                pass
            return name
        except Exception:
            # Create with cosine distance
            self._client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=int(dimension), distance=qm.Distance.COSINE),
                optimizers_config=qm.OptimizersConfigDiff(default_segment_number=2),
            )
            # Optional: set payload schema? Keep flexible — finance metadata varies widely.
            log.info("Qdrant: created collection '%s' (dim=%d)", name, dimension)
            return name

    def get_collection(self, name: str) -> Optional[dict]:
        self.ensure_client()
        assert self._client is not None
        try:
            info = self._client.get_collection(name)
            # Normalize to dict view
            return {
                "name": name,
                "vectors": getattr(info, "vectors_count", None),
                "points": getattr(getattr(info, "points_count", None), "count", None) or None,
            }
        except Exception:
            return None

    def list_collections(self) -> List[dict]:
        self.ensure_client()
        assert self._client is not None
        res = self._client.get_collections()
        items = getattr(res, "collections", []) or []
        out = []
        for c in items:
            cname = getattr(c, "name", None)
            if cname:
                out.append({"name": cname})
        return out

    def delete_collection(self, name: str) -> bool:
        self.ensure_client()
        assert self._client is not None
        try:
            self._client.delete_collection(collection_name=name)
            return True
        except Exception:
            return False

    # ---------------------------
    # Upsert
    # ---------------------------

    def upsert_texts(self, collection: str, items: List[dict]) -> int:
        """
        Upsert text chunks with embeddings into Qdrant.

        Each item:
          id         : optional str/int (Qdrant accepts both)
          text       : raw chunk text
          metadata   : arbitrary JSON-serializable dict
          embedding  : list[float] of length == collection dimension
        """
        if not items:
            return 0

        self.ensure_client()
        assert self._client is not None

        # Build points
        points: List[qm.PointStruct] = []
        for it in items:
            vec = it.get("embedding")
            if not isinstance(vec, (list, tuple)) or not vec:
                raise ValueError("embedding is required (list[float])")
            pid = it.get("id")
            payload = {
                "text": it.get("text") or "",
                "metadata": it.get("metadata") or {},
            }
            points.append(
                qm.PointStruct(
                    id=pid,  # None -> server assigns integer ID
                    payload=payload,
                    vector={_vector_name(): list(map(float, vec))} if hasattr(qm, "PointVectors") else list(map(float, vec)),
                )
            )

        # Upsert
        try:
            self._client.upsert(collection_name=collection, points=points, wait=True)
        except Exception as e:
            raise RuntimeError(f"Qdrant upsert failed: {e}") from e

        return len(points)

    # ---------------------------
    # Search
    # ---------------------------

    def search(
        self,
        collection: str,
        query: Sequence[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[dict]:
        """
        Cosine-distance search (lower is better).
        """
        self.ensure_client()
        assert self._client is not None

        qvec = list(map(float, query))
        flt = _to_filter(filters)

        try:
            # Newer API supports named vectors via dict; we use single-vector name "text"
            res = self._client.search(
                collection_name=collection,
                query_vector=({_vector_name(): qvec} if hasattr(qm, "SearchRequest") else qvec),
                limit=max(1, int(top_k)),
                query_filter=flt,
                with_payload=True,
                with_vectors=False,
                score_threshold=None,  # allow full ranking
            )
        except Exception as e:
            raise RuntimeError(f"Qdrant search failed: {e}") from e

        out: List[dict] = []
        for r in res or []:
            payload = r.payload or {}
            out.append(
                {
                    "id": r.id,
                    "text": payload.get("text") or "",
                    "metadata": payload.get("metadata") or {},
                    "score": float(getattr(r, "score", 0.0)),
                }
            )
        return out

    # ---------------------------
    # Utilities
    # ---------------------------

    def stats(self, collection: str) -> dict:
        """
        Return quick stats for a collection.
        """
        self.ensure_client()
        assert self._client is not None
        try:
            cnt = self._client.count(collection_name=collection, exact=True)
            n = int(getattr(cnt, "count", 0) or 0)
        except Exception:
            n = 0
        return {"count": n}

    def get_by_id(self, collection: str, id: str | int) -> Optional[dict]:
        self.ensure_client()
        assert self._client is not None
        try:
            res = self._client.retrieve(collection_name=collection, ids=[id], with_payload=True, with_vectors=False)
            if not res:
                return None
            p = res[0]
            payload = p.payload or {}
            return {"id": p.id, "text": payload.get("text") or "", "metadata": payload.get("metadata") or {}}
        except Exception:
            return None

    def delete_by_ids(self, collection: str, ids: Iterable[str | int]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        self.ensure_client()
        assert self._client is not None
        try:
            self._client.delete(collection_name=collection, points_selector=qm.PointIdsList(points=ids), wait=True)
            return len(ids)
        except Exception as e:
            raise RuntimeError(f"Qdrant delete failed: {e}") from e
