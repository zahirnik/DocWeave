# packages/retriever/bm25.py
"""
BM25 text search — tiny, explicit wrapper with **two** optional backends:

Backends
--------
1) "whoosh"  (default; dev-friendly, file-based, pure-Python)
   - No external service. Great for local testing and small corpora (<~100k docs).
   - Persists an inverted index per *collection* under `index_dir/<collection>`.

2) "elastic" (requires Elasticsearch/OpenSearch & `elasticsearch` Python client)
   - For bigger data and production-grade indexing.
   - Uses a single index per collection with BM25 scoring (default in ES).

Unified interface
-----------------
BM25(provider="whoosh"|"elastic", **kwargs)
  - index_documents(collection: str, items: list[dict]) -> int
      items: [{"id": "...", "text": "...", "metadata": {...}}]

  - search(collection: str, query: str, top_k: int = 10, filters: dict | None = None) -> list[dict]
      returns: [{"id","text","metadata","score"}]  # BM25 score: **higher is better**

Design goals
------------
- Tutorial-clear code and safe defaults.
- Minimal dependencies. Whoosh path works out-of-the-box for demos.
- Filters are **best-effort**:
    • whoosh: applied post-search by checking `metadata` (fast enough for small N).
    • elastic: translated into term filters in a bool query.

Notes
-----
- Hybrid retrieval will typically combine **vector (cosine; lower is better)** with **BM25 (higher is better)**.
  Your `packages/retriever/search.py` will normalize/merge scores.

Install
-------
# whoosh backend
pip install whoosh

# elastic backend
pip install elasticsearch

Examples
--------
from packages.retriever.bm25 import BM25

bm = BM25(provider="whoosh", index_dir=".whoosh")
bm.index_documents("acme", [
    {"id":"d1","text":"Revenue grew 10% year over year.","metadata":{"year":2024}},
    {"id":"d2","text":"Operating margin compressed due to input costs.","metadata":{"year":2024}},
])
hits = bm.search("acme", "operating margin", top_k=3)
for h in hits:
    print(h["score"], h["id"], h["text"][:60])
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from packages.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------

@dataclass
class BM25Config:
    provider: str = "whoosh"              # "whoosh" | "elastic"
    index_dir: str = ".whoosh"            # for whoosh
    elastic_url: str = "http://localhost:9200"
    index_prefix: str = "rag"             # for elastic index names: <prefix>_<collection>
    elastic_api_key: Optional[str] = None # optional: for Elastic Cloud / secured clusters

    @staticmethod
    def from_env() -> "BM25Config":
        return BM25Config(
            provider=(os.getenv("BM25_PROVIDER") or "whoosh").strip().lower(),
            index_dir=os.getenv("WHOOSH_DIR", ".whoosh"),
            elastic_url=os.getenv("ELASTIC_URL", "http://localhost:9200"),
            index_prefix=os.getenv("BM25_INDEX_PREFIX", "rag"),
            elastic_api_key=os.getenv("ELASTIC_API_KEY") or None,
        )


# ---------------------------------------------------------------------
# Main facade
# ---------------------------------------------------------------------

class BM25:
    """
    Small facade over Whoosh (local) or Elasticsearch (service).
    """

    def __init__(self, config: Optional[BM25Config] = None, **kwargs):
        if config is None:
            config = BM25Config.from_env()
        # allow simple overrides via kwargs: BM25(provider="whoosh", index_dir="...")
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        self.cfg = config
        prov = self.cfg.provider
        if prov not in {"whoosh", "elastic"}:
            raise ValueError("BM25 provider must be 'whoosh' or 'elastic'")
        log.info("BM25 provider=%s", prov)

        # Lazy init per backend
        self._whoosh_ok = None
        self._elastic_client = None

    # ---------------------------
    # Public API
    # ---------------------------

    def index_documents(self, collection: str, items: List[dict]) -> int:
        """
        Index (or re-index) textual documents for BM25. Idempotent-ish:
        - whoosh: adds or updates by id.
        - elastic: upserts via _bulk API.
        """
        if self.cfg.provider == "whoosh":
            return self._index_whoosh(collection, items)
        else:
            return self._index_elastic(collection, items)

    def search(self, collection: str, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[dict]:
        """
        Run a BM25 search and return normalized results with **higher-better** scores.
        """
        if not query or not query.strip():
            return []
        if self.cfg.provider == "whoosh":
            return self._search_whoosh(collection, query, top_k=top_k, filters=filters)
        else:
            return self._search_elastic(collection, query, top_k=top_k, filters=filters)

    # -----------------------------------------------------------------
    # Backend: WHOOSH (dev/local)
    # -----------------------------------------------------------------

    def _ensure_whoosh(self):
        if self._whoosh_ok is not None:
            return
        try:
            import whoosh  # noqa: F401
            from whoosh import fields  # noqa: F401
            self._whoosh_ok = True
        except Exception as e:
            self._whoosh_ok = False
            raise RuntimeError("Whoosh backend requires `whoosh`. Install with: pip install whoosh") from e

    def _whoosh_schema(self):
        """
        Define a tiny schema:
          - id: unique ID (stored)
          - text: main content (indexed for BM25, stored for retrieval)
          - metadata: stored as JSON string (not indexed) — we filter post-hoc in Python
        """
        from whoosh import fields
        return fields.Schema(
            id=fields.ID(stored=True, unique=True),
            text=fields.TEXT(stored=True),
            metadata=fields.STORED,
        )

    def _whoosh_path(self, collection: str) -> str:
        return os.path.join(self.cfg.index_dir, collection)

    def _index_whoosh(self, collection: str, items: List[dict]) -> int:
        self._ensure_whoosh()
        from whoosh import index

        path = self._whoosh_path(collection)
        os.makedirs(path, exist_ok=True)

        if not index.exists_in(path):
            ix = index.create_in(path, schema=self._whoosh_schema())
        else:
            ix = index.open_dir(path)

        n = 0
        with ix.writer(limitmb=256, procs=1) as w:
            for it in items or []:
                doc_id = str(it.get("id") or "").strip()
                text = (it.get("text") or "").strip()
                meta = it.get("metadata") or {}
                if not doc_id or not text:
                    continue
                # Update-or-add (Whoosh: update_document by unique field)
                w.update_document(id=doc_id, text=text, metadata=json.dumps(meta))
                n += 1
        return n

    def _search_whoosh(self, collection: str, query: str, *, top_k: int, filters: Optional[Dict]) -> List[dict]:
        self._ensure_whoosh()
        from whoosh import index
        from whoosh.qparser import QueryParser

        path = self._whoosh_path(collection)
        if not os.path.exists(path) or not index.exists_in(path):
            raise ValueError(f"Whoosh index for collection {collection!r} not found. Index documents first.")

        ix = index.open_dir(path)
        qp = QueryParser("text", schema=ix.schema)

        # Whoosh BM25F is default weighting in many builds; we rely on default for simplicity
        q = qp.parse(query)
        out: List[dict] = []

        with ix.searcher() as s:
            results = s.search(q, limit=max(1, int(top_k)))
            # Convert to our normalized shape
            for hit in results:
                meta = {}
                try:
                    raw = hit.get("metadata")
                    if raw:
                        meta = json.loads(raw)
                except Exception:
                    meta = {}

                # Post-filter by metadata for small corpora (AND on exact matches)
                if filters and not _meta_match(meta, filters):
                    continue

                out.append(
                    {
                        "id": hit.get("id"),
                        "text": hit.get("text") or "",
                        "metadata": meta,
                        "score": float(hit.score or 0.0),  # **higher is better**
                    }
                )
        return out

    # -----------------------------------------------------------------
    # Backend: ELASTIC (service)
    # -----------------------------------------------------------------

    def _elastic_index_name(self, collection: str) -> str:
        return f"{self.cfg.index_prefix}_{collection}".lower()

    def _ensure_elastic(self):
        if self._elastic_client is not None:
            return
        try:
            from elasticsearch import Elasticsearch  # type: ignore
        except Exception as e:
            raise RuntimeError("Elasticsearch backend requires `elasticsearch`. Install with: pip install elasticsearch") from e

        headers = None
        if self.cfg.elastic_api_key:
            # API key header: "Authorization: ApiKey base64(key_id:key)"
            headers = {"Authorization": f"ApiKey {self.cfg.elastic_api_key}"}

        self._elastic_client = Elasticsearch(self.cfg.elastic_url, headers=headers, request_timeout=30)

        try:
            self._elastic_client.info()
        except Exception as e:
            raise RuntimeError(f"Could not connect to Elasticsearch at {self.cfg.elastic_url}: {e}") from e

    def _ensure_elastic_index(self, name: str):
        assert self._elastic_client is not None
        es = self._elastic_client
        if es.indices.exists(index=name):
            return
        # Minimal mapping: text analyzed for BM25, metadata kept as nested-ish object
        es.indices.create(
            index=name,
            body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "text": {"type": "text"},  # BM25 default
                        "metadata": {"type": "object", "enabled": True},  # flattened for term filters
                    }
                },
            },
        )

    def _index_elastic(self, collection: str, items: List[dict]) -> int:
        self._ensure_elastic()
        assert self._elastic_client is not None
        from elasticsearch import helpers  # type: ignore

        name = self._elastic_index_name(collection)
        self._ensure_elastic_index(name)

        actions = []
        for it in items or []:
            doc_id = str(it.get("id") or "").strip()
            text = (it.get("text") or "").strip()
            meta = it.get("metadata") or {}
            if not doc_id or not text:
                continue
            actions.append(
                {
                    "_op_type": "index",
                    "_index": name,
                    "_id": doc_id,
                    "_source": {"id": doc_id, "text": text, "metadata": meta},
                }
            )
        if not actions:
            return 0

        helpers.bulk(self._elastic_client, actions, request_timeout=60)
        self._elastic_client.indices.refresh(index=name)
        return len(actions)

    def _search_elastic(self, collection: str, query: str, *, top_k: int, filters: Optional[Dict]) -> List[dict]:
        self._ensure_elastic()
        assert self._elastic_client is not None
        name = self._elastic_index_name(collection)

        must: List[dict] = []
        if query.strip():
            must.append({"match": {"text": {"query": query}}})
        if filters:
            for k, v in filters.items():
                must.append({"term": {f"metadata.{k}": v}})

        body = {"query": {"bool": {"must": must}}} if must else {"query": {"match_all": {}}}

        try:
            res = self._elastic_client.search(index=name, body=body, size=max(1, int(top_k)))
        except Exception as e:
            raise RuntimeError(f"Elasticsearch search failed: {e}") from e

        hits = (((res or {}).get("hits") or {}).get("hits")) or []
        out: List[dict] = []
        for h in hits:
            src = h.get("_source") or {}
            out.append(
                {
                    "id": src.get("id") or h.get("_id"),
                    "text": src.get("text") or "",
                    "metadata": src.get("metadata") or {},
                    "score": float(h.get("_score") or 0.0),  # **higher is better**
                }
            )
        return out


# ---------------------------------------------------------------------
# Small helper(s)
# ---------------------------------------------------------------------

def _meta_match(meta: Dict, filters: Optional[Dict]) -> bool:
    """
    Post-hoc metadata filtering (AND of exact matches). For whoosh small corpora only.
    """
    if not filters:
        return True
    for k, v in filters.items():
        if meta.get(k) != v:
            return False
    return True


__all__ = ["BM25", "BM25Config"]
