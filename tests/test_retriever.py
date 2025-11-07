# tests/test_retriever.py
"""
Retriever & vector-store smoke tests (dependency-light, offline-friendly).

What we cover
-------------
- Basic vector upsert + cosine search (pgvector/chroma if available; else an in-memory fallback).
- Simple “hybrid” scoring (vector cosine + tiny BM25-ish term score) to illustrate behavior.
- Metadata presence & ad-hoc filtering on search results.

Notes
-----
- These are **smoke tests**: they validate wiring and shapes rather than strict IR quality.
- We avoid network/LLM calls. Embeddings use `EmbeddingClient(provider="auto")` with a deterministic local fallback if no keys are configured.
- If real vector stores are not installed/configured, we fall back to a tiny in-memory store implemented below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from packages.retriever.embeddings import EmbeddingClient

# Optional stores (duck-typed)
try:
    from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # type: ignore
except Exception:
    PgVectorStore = None  # type: ignore

try:
    from packages.retriever.vectorstores.chroma_store import ChromaStore  # type: ignore
except Exception:
    ChromaStore = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Fallback in-memory store (cosine over precomputed vectors)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _MemItem:
    id: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any]

class _MemStore:
    """
    Minimal vector store drop-in for tests:
      - upsert(items): items are dicts with {id, text, vector, metadata}
      - search(qv, top_k): cosine over vectors, returns [{id,text,score,metadata}]
    """
    def __init__(self):
        self.items: List[_MemItem] = []

    def upsert(self, items: List[Dict[str, Any]]):
        have = {it.id for it in self.items}
        for d in items:
            _id = str(d["id"])
            if _id in have:
                # replace existing
                for i, it in enumerate(self.items):
                    if it.id == _id:
                        self.items[i] = _MemItem(
                            id=_id,
                            text=str(d.get("text") or ""),
                            vector=list(d.get("vector") or []),
                            metadata=dict(d.get("metadata") or {}),
                        )
                        break
            else:
                self.items.append(
                    _MemItem(
                        id=_id,
                        text=str(d.get("text") or ""),
                        vector=list(d.get("vector") or []),
                        metadata=dict(d.get("metadata") or {}),
                    )
                )

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
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder() -> EmbeddingClient:
    return EmbeddingClient(provider="auto", model_alias="mini")


@pytest.fixture
def sample_corpus() -> List[Dict[str, Any]]:
    """
    Three short finance snippets with distinct topics + metadata.
    """
    return [
        {
            "id": "acme#p0",
            "text": "ACME PLC Annual Report 2024: gross margin was 38.2% for the fiscal year.",
            "metadata": {"company": "ACME PLC", "topic": "profitability", "year": 2024},
        },
        {
            "id": "beta#p0",
            "text": "Beta Corp Q3 2024 revenue reached $1.26bn, up 14.5% year over year.",
            "metadata": {"company": "Beta Corp", "topic": "topline", "quarter": "2024Q3"},
        },
        {
            "id": "delta#p0",
            "text": "Delta Inc net debt to EBITDA stood at 3.0x in FY2023.",
            "metadata": {"company": "Delta Inc", "topic": "leverage", "year": 2023},
        },
    ]


@pytest.fixture
def store(embedder: EmbeddingClient, sample_corpus: List[Dict[str, Any]]):
    """
    Try a real vector store (pgvector → chroma), else fall back to the in-memory store.
    """
    # Choose backend (pgvector preferred for parity with prod)
    if PgVectorStore is not None:
        try:
            s = PgVectorStore(dsn=None, collection="unit_test_tmp", create_if_missing=True)  # type: ignore
            # Most PgVectorStore implementations require DSN; if None breaks, we fallback below.
        except Exception:
            s = None
    else:
        s = None

    if s is None and ChromaStore is not None:
        try:
            s = ChromaStore(path="./.chroma_test", collection="unit_test_tmp", create_if_missing=True)  # type: ignore
        except Exception:
            s = None

    if s is None:
        s = _MemStore()

    # Upsert embedded items
    vecs = embedder.embed([d["text"] for d in sample_corpus], batch_size=64)
    payloads: List[Dict[str, Any]] = []
    for d, v in zip(sample_corpus, vecs):
        payloads.append(
            {"id": d["id"], "doc_id": d["id"].split("#")[0], "text": d["text"], "vector": v, "metadata": d["metadata"]}
        )
    s.upsert(payloads)
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_vector_search_top_hit(embedder: EmbeddingClient, store):
    """
    Query specific to ACME's gross margin → ACME chunk should score top-1.
    """
    q = "What was ACME's gross margin in 2024?"
    qv = embedder.embed([q])[0]
    hits = store.search(qv, top_k=3)
    assert len(hits) >= 1
    top = hits[0]["id"]
    assert top.startswith("acme"), f"expected ACME on top, got {top!r}"


def _bm25ish(query: str, text: str) -> float:
    """
    Tiny, deterministic term score (BM25-ish) for hybrid demonstration.
    Counts occurrences of lowercased stem-like tokens.
    """
    q_terms = [t for t in query.lower().replace("'", "").split() if len(t) > 2]
    t = text.lower()
    score = 0.0
    for qt in q_terms:
        score += t.count(qt) * 1.0
    return score


def test_hybrid_rerank(embedder: EmbeddingClient, store):
    """
    Hybrid score = 0.6 * cosine + 0.4 * term_score_scaled
    Validate it still keeps the right document on top for a revenue query.
    """
    q = "Q3 2024 revenue of Beta Corp"
    qv = embedder.embed([q])[0]
    hits = store.search(qv, top_k=3)
    assert hits, "empty search result"

    # Compute hybrid score
    # Normalize BM25-ish to [0,1] over current pool to avoid dominating cosine.
    term_scores = [_bm25ish(q, h["text"]) for h in hits]
    denom = max(term_scores) or 1.0
    hybrid = []
    for h, ts in zip(hits, term_scores):
        s_vec = float(h["score"])
        s_term = float(ts) / float(denom)
        s = 0.6 * s_vec + 0.4 * s_term
        hybrid.append((s, h))
    hybrid.sort(key=lambda x: x[0], reverse=True)
    top_id = hybrid[0][1]["id"]
    assert top_id.startswith("beta"), f"expected Beta on top with hybrid, got {top_id!r}"


def test_metadata_filtering(embedder: EmbeddingClient, store):
    """
    Demonstrate metadata-based filtering (performed client-side here for portability).
    Filter for topic='leverage' → expect Delta hit on top.
    """
    q = "What is the net debt to EBITDA ratio?"
    qv = embedder.embed([q])[0]
    hits = store.search(qv, top_k=5)

    # Client-side filter; a real store backend may offer server-side filters.
    filt = [h for h in hits if (h.get("metadata") or {}).get("topic") == "leverage"]
    assert filt, "no results after metadata filter"
    assert filt[0]["id"].startswith("delta"), f"expected Delta in filtered results, got {filt[0]['id']!r}"


def test_embedding_dimension_consistency(embedder: EmbeddingClient):
    """
    Embedding dim is stable across calls and inputs.
    """
    v1 = embedder.embed(["alpha"])[0]
    v2 = embedder.embed(["beta"])[0]
    v3 = embedder.embed(["alpha"])[0]
    d = len(v1)
    assert d > 8
    assert len(v2) == d and len(v3) == d
    # identical input → cosine ≈ 1
    dot = sum(a * b for a, b in zip(v1, v3))
    n1 = math.sqrt(sum(a * a for a in v1)) + 1e-12
    n2 = math.sqrt(sum(b * b for b in v3)) + 1e-12
    assert (dot / (n1 * n2)) > 0.999
