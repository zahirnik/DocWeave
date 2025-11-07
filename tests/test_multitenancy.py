# tests/test_multitenancy.py
"""
Multitenancy smoke tests — ensure **tenant isolation** in vector search.

What we verify (dependency-light)
---------------------------------
1) **Collection-level isolation**: Using separate collections (or separate in-memory
   stores) per tenant means a search in tenant `t0` can never return chunks from `t1`.
2) **Metadata-based filtering** (portable demo): Even if chunks share a collection,
   filtering by `tenant_id` on the client side enforces separation.

Design notes
------------
- We avoid DB migrations and keep tests offline. If a real store adapter is missing,
  we fall back to a tiny in-memory store.
- Embeddings use `EmbeddingClient(provider="auto")` with deterministic, local fallback
  when API keys are not set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from packages.retriever.embeddings import EmbeddingClient

# Optional vector store adapters (duck-typed)
try:
    from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # type: ignore
except Exception:
    PgVectorStore = None  # type: ignore

try:
    from packages.retriever.vectorstores.chroma_store import ChromaStore  # type: ignore
except Exception:
    ChromaStore = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Minimal in-memory vector store (fallback)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _MemItem:
    id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any]

class _MemStore:
    """
    Ultra-small cosine vector store used only in tests:
      - upsert(items) with dicts {id, text, vector, metadata}
      - search(qv, top_k) → [{id, text, score, metadata}]
    """
    def __init__(self, name: str = "mem"):
        self.name = name
        self.items: List[_MemItem] = []

    def upsert(self, items: List[Dict[str, Any]]) -> None:
        idx = {it.id: i for i, it in enumerate(self.items)}
        for d in items:
            mid = str(d["id"])
            rec = _MemItem(
                id=mid,
                text=str(d.get("text") or ""),
                vector=list(d.get("vector") or []),
                metadata=dict(d.get("metadata") or {}),
            )
            if mid in idx:
                self.items[idx[mid]] = rec
            else:
                self.items.append(rec)

    def search(self, qv: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        q = np.asarray(qv, dtype="float32")
        q /= (np.linalg.norm(q) + 1e-12)
        scores: List[float] = []
        for it in self.items:
            v = np.asarray(it.vector, dtype="float32")
            v /= (np.linalg.norm(v) + 1e-12)
            scores.append(float(np.dot(q, v)))
        order = np.argsort(np.asarray(scores))[::-1][: int(top_k)]
        out: List[Dict[str, Any]] = []
        for idx in order:
            it = self.items[int(idx)]
            out.append({"id": it.id, "text": it.text, "score": scores[int(idx)], "metadata": it.metadata})
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder() -> EmbeddingClient:
    return EmbeddingClient(provider="auto", model_alias="mini")


def _make_store(tenant_collection: str):
    """
    Prefer a real store if available (pgvector → chroma), else in-memory.
    We keep the API duck-typed: .upsert(items), .search(qv, top_k).
    """
    # pgvector first (requires DSN in adapter; many adapters also allow default local)
    if PgVectorStore is not None:
        try:
            return PgVectorStore(dsn=None, collection=tenant_collection, create_if_missing=True)  # type: ignore
        except Exception:
            pass
    # chroma next
    if ChromaStore is not None:
        try:
            return ChromaStore(path="./.chroma_test", collection=tenant_collection, create_if_missing=True)  # type: ignore
        except Exception:
            pass
    # fallback
    return _MemStore(name=tenant_collection)


@pytest.fixture
def t0_store(embedder: EmbeddingClient):
    """
    Tenant t0: has only ACME content.
    """
    s = _make_store("unit_test_t0")
    texts = [
        ("acme#p0", "ACME PLC Annual Report 2024: gross margin was 38.2% for the fiscal year.", {"tenant_id": "t0", "company": "ACME PLC"}),
        ("acme#p1", "Operating cash flow improved; inventory normalised.", {"tenant_id": "t0", "company": "ACME PLC"}),
    ]
    vecs = embedder.embed([t for _, t, _ in texts], batch_size=64)
    payloads = []
    for (id_, txt, meta), v in zip(texts, vecs):
        payloads.append({"id": id_, "doc_id": id_.split("#")[0], "text": txt, "vector": v, "metadata": meta})
    s.upsert(payloads)
    return s


@pytest.fixture
def t1_store(embedder: EmbeddingClient):
    """
    Tenant t1: has only Beta content.
    """
    s = _make_store("unit_test_t1")
    texts = [
        ("beta#p0", "Beta Corp Q3 2024 revenue reached $1.26bn, up 14.5% year over year.", {"tenant_id": "t1", "company": "Beta Corp"}),
        ("beta#p1", "Net debt to EBITDA stood at 3.0x in FY2023.", {"tenant_id": "t1", "company": "Beta Corp"}),
    ]
    vecs = embedder.embed([t for _, t, _ in texts], batch_size=64)
    payloads = []
    for (id_, txt, meta), v in zip(texts, vecs):
        payloads.append({"id": id_, "doc_id": id_.split("#")[0], "text": txt, "vector": v, "metadata": meta})
    s.upsert(payloads)
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_isolation_by_collection(embedder: EmbeddingClient, t0_store, t1_store):
    """
    Searching within t0's collection cannot surface t1's chunks (and vice versa),
    because each tenant uses an isolated collection (or isolated in-memory store).
    """
    # Query about Beta in t0 store → should return only ACME docs (best-effort vector match), *never* 'beta#...'
    q_beta = "What was Beta Corp revenue in Q3 2024?"
    qv_beta = embedder.embed([q_beta])[0]
    hits_t0 = t0_store.search(qv_beta, top_k=5)
    assert hits_t0, "t0 search returned no results (unexpected for vector store)"
    assert all(not str(h["id"]).startswith("beta#") for h in hits_t0), "t0 store leaked t1 document IDs!"

    # Symmetric: query about ACME in t1 store → must not show 'acme#...'
    q_acme = "ACME's 2024 gross margin"
    qv_acme = embedder.embed([q_acme])[0]
    hits_t1 = t1_store.search(qv_acme, top_k=5)
    assert hits_t1, "t1 search returned no results (unexpected for vector store)"
    assert all(not str(h["id"]).startswith("acme#") for h in hits_t1), "t1 store leaked t0 document IDs!"


def test_metadata_filtering_single_store(embedder: EmbeddingClient):
    """
    Demonstrate client-side metadata filtering when a backend uses a shared collection.
    (We simulate a shared store by putting both tenants' chunks into a single in-memory store.)
    """
    shared = _MemStore(name="shared")
    rows = [
        ("acme#p0", "ACME PLC Annual Report 2024: gross margin 38.2%.", {"tenant_id": "t0", "company": "ACME PLC"}),
        ("beta#p0", "Beta Corp Q3 2024 revenue $1.26bn.", {"tenant_id": "t1", "company": "Beta Corp"}),
    ]
    vecs = embedder.embed([t for _, t, _ in rows], batch_size=64)
    payloads = []
    for (id_, txt, meta), v in zip(rows, vecs):
        payloads.append({"id": id_, "doc_id": id_.split("#")[0], "text": txt, "vector": v, "metadata": meta})
    shared.upsert(payloads)

    # Search about "revenue" then filter to tenant t0 → should exclude Beta
    q = "What revenue figure was reported?"
    qv = embedder.embed([q])[0]
    hits = shared.search(qv, top_k=5)
    assert hits, "shared store returned no hits"

    # Client-side isolation by tenant_id
    t0_hits = [h for h in hits if (h.get("metadata") or {}).get("tenant_id") == "t0"]
    assert t0_hits, "No results after tenant t0 filter (unexpected)"
    assert all(h["id"].startswith("acme#") for h in t0_hits), "Filtered results include foreign-tenant chunks"


def test_cross_tenant_queries_are_separate(embedder: EmbeddingClient, t0_store, t1_store):
    """
    Ensure running the same query in each tenant's store yields tenant-specific results.
    """
    q = "Summarise operating cash flow trend."
    qv = embedder.embed([q])[0]
    t0_hits = t0_store.search(qv, top_k=3)
    t1_hits = t1_store.search(qv, top_k=3)
    assert t0_hits and t1_hits

    # Heuristic check: top IDs differ across tenants (because corpora differ)
    assert t0_hits[0]["id"].split("#")[0] != t1_hits[0]["id"].split("#")[0], "Top results identical across tenants (unexpected)"
