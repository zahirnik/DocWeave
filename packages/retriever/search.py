# packages/retriever/search.py
"""
Hybrid search orchestrator — vector + BM25 (+ optional rerank) with MMR diversification.

What this module provides
-------------------------
- RetrieverConfig: knobs for providers/backends and scoring.
- HybridSearcher: the single, tiny facade you call from your API/agent graph.

    searcher = HybridSearcher.from_env()
    hits = searcher.search(
        collection="acme_finance",
        query="How did gross margin change in Q2 2024?",
        top_k=8,
        filters={"year": 2024}
    )
    # -> [{"id","text","metadata","score","source":{"bm25":...,"vector":...,"rerank":...}}]

Design goals
------------
- Tutorial-clear, explicit orchestration:
    1) Embed query → vector search (cosine; lower distance is better).
    2) BM25 full-text search (higher is better).
    3) Score normalization into [0,1] (larger is better).
    4) Fuse by weighted sum (alpha controls vector vs. BM25).
    5) Optional Cross-Encoder/API reranking on top-M candidates.
    6) Optional MMR diversification to reduce near-duplicates.

- Small and replaceable: use pgvector, Qdrant, or Chroma via a tiny adapter.

Environment variables (defaults chosen for clarity)
---------------------------------------------------
VECTOR_STORE=pgvector|qdrant|chroma            (default: chroma)
PG_DSN=postgresql://user:pass@host:5432/db     (if VECTOR_STORE=pgvector)
QDRANT_URL=http://localhost:6333               (if VECTOR_STORE=qdrant)
QDRANT_API_KEY=                                (optional)
CHROMA_DIR=.chroma                              (if VECTOR_STORE=chroma)

EMBEDDINGS_*  (see packages.retriever.embeddings.EmbeddingConfig.from_env)
BM25_*        (see packages.retriever.bm25.BM25Config.from_env)
RERANK_*      (see packages.retriever.reranker.RerankConfig.from_env)

FUSION_ALPHA=0.6     # weight for vector score in fusion: score = alpha*vector + (1-alpha)*bm25
RERANK_TOP_M=20      # how many fused candidates to send to reranker (if enabled)
USE_MMR=true|false   # whether to apply MMR diversification
MMR_LAMBDA=0.5       # tradeoff relevance/diversity
MMR_K=8              # final diversified top-k (usually match `top_k`)

Notes
-----
- Vector distance (lower is better) is inverted and normalized.
- BM25 score (higher is better) is normalized by max-score in the candidate set.
- Normalization is per-call (no global stats needed).
- MMR needs an embedding for each candidate text; we reuse the same encoder.

Testing
-------
- Works end-to-end with Chroma + local embeddings (“BAAI/bge-small-en-v1.5”) and Whoosh BM25.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from packages.retriever.embeddings import Embeddings, EmbeddingConfig
from packages.retriever.bm25 import BM25, BM25Config
from packages.retriever.reranker import Reranker, RerankConfig

# Vector store adapters
from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # noqa: F401
from packages.retriever.vectorstores.qdrant_store import QdrantStore      # noqa: F401
from packages.retriever.vectorstores.chroma_store import ChromaStore      # noqa: F401

from packages.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------
# Config
# ---------------------------

@dataclass
class RetrieverConfig:
    # Vector store selection
    vector_store: str = "chroma"  # "pgvector" | "qdrant" | "chroma"

    # Backends DSNs/URLs
    pg_dsn: str = os.getenv("PG_DSN", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    chroma_dir: str = os.getenv("CHROMA_DIR", ".chroma")

    # Fusion and rerank
    fusion_alpha: float = float(os.getenv("FUSION_ALPHA", "0.6"))  # weight on vector
    rerank_top_m: int = int(os.getenv("RERANK_TOP_M", "20"))

    # Diversification (MMR)
    use_mmr: bool = (os.getenv("USE_MMR", "true").lower() in {"1", "true", "yes", "on"})
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.5"))
    mmr_k: int = int(os.getenv("MMR_K", "8"))

    # Limits
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "20"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "20"))

    @staticmethod
    def from_env() -> "RetrieverConfig":
        return RetrieverConfig(
            vector_store=(os.getenv("VECTOR_STORE") or "chroma").strip().lower(),
        )


# ---------------------------
# Hybrid searcher
# ---------------------------

class HybridSearcher:
    """
    Tiny facade orchestrating: embeddings → vector store → BM25 → fusion → rerank → (optional) MMR.
    - Treats provided collection names as *opaque physical names* (already resolved/sanitized upstream).
    """

    def __init__(
        self,
        retr_cfg: Optional[RetrieverConfig] = None,
        emb_cfg: Optional[EmbeddingConfig] = None,
        bm25_cfg: Optional[BM25Config] = None,
        rer_cfg: Optional[RerankConfig] = None,
    ):
        self.rcfg = retr_cfg or RetrieverConfig.from_env()
        self.emb = Embeddings(config=(emb_cfg or EmbeddingConfig.from_env()))
        self.bm25 = BM25(config=(bm25_cfg or BM25Config.from_env()))
        self.reranker = Reranker(config=(rer_cfg or RerankConfig.from_env()))

        # Instantiate vector store adapter
        vs = self.rcfg.vector_store
        if vs == "pgvector":
            if not self.rcfg.pg_dsn:
                raise RuntimeError("PG_DSN is required when VECTOR_STORE=pgvector")
            self._store = PgVectorStore(self.rcfg.pg_dsn, dimension=self._dim_or_guess())
            self._store.ensure_schema()
        elif vs == "qdrant":
            self._store = QdrantStore(url=self.rcfg.qdrant_url, api_key=self.rcfg.qdrant_api_key)
            self._store.ensure_client()
        elif vs == "chroma":
            self._store = ChromaStore(persist_dir=self.rcfg.chroma_dir)
            self._store.ensure_client()
        else:
            raise RuntimeError(f"Unknown VECTOR_STORE: {vs}")

        log.info(
            "HybridSearcher ready: store=%s alpha=%.2f rerank=%s mmr=%s",
            vs, self.rcfg.fusion_alpha, self.reranker.cfg.provider, self.rcfg.use_mmr
        )

    @classmethod
    def from_env(cls) -> "HybridSearcher":
        return cls()

    # ---------------------------
    # Public API
    # ---------------------------

    def search(
        self,
        collection: str,
        query: str,
        *,
        top_k: int = 8,
        filters: Optional[Dict] = None,
        # [ADD] allow separate physical name for BM25 and the option to disable it here
        bm25_collection: Optional[str] = None,
        bm25_enabled: Optional[bool] = None,
    ) -> List[dict]:
        """
        Run hybrid retrieval for a single query.

        Args:
          collection        : physical collection name for the vector store (opaque).
          bm25_collection   : physical index/collection name for BM25 (opaque). Defaults to `collection`.
          bm25_enabled      : override to enable/disable internal BM25 stage. If None, defaults to True
                              unless BM25 provider is 'none'.

        Returns:
          list of dicts, each with:
            "id", "text", "metadata",
            "score",                 # final fused (and reranked) score, higher is better
            "source": {"vector":..., "bm25":..., "rerank":...}  # raw component scores for transparency
        """
        if not query or not query.strip():
            return []

        bm25_name = bm25_collection or collection
        bm25_do = (bm25_enabled if bm25_enabled is not None else True) and self.bm25.cfg.provider != "none"

        # 1) Query embedding → vector search
        qvec = self.emb.embed_query(query)
        vec_hits = self._vector_search(collection, qvec, top_k=self.rcfg.vector_top_k, filters=filters)

        # 2) BM25 search (optional, robust to backend/collection gaps)
        if bm25_do:
            try:
                bm25_hits = self.bm25.search(bm25_name, query, top_k=self.rcfg.bm25_top_k, filters=filters)
            except Exception as e:
                log.info("BM25 search failed for '%s' (%s); falling back to vector-only.", bm25_name, e)
                bm25_hits = []
        else:
            bm25_hits = []

        # 3) Normalize & fuse
        fused = self._fuse(vec_hits, bm25_hits, alpha=self.rcfg.fusion_alpha)

        if not fused:
            return []

        # 4) Optional rerank on top M
        M = max(top_k, self.rcfg.rerank_top_m)
        pool = fused[:M]
        pool = self._maybe_rerank(query, pool)

        # 5) Optional MMR diversification (operates on embeddings)
        if self.rcfg.use_mmr:
            cand_embs = self.emb.embed_documents([c["text"] for c in pool])
            idxs = _mmr_indices(qvec, cand_embs, top_k=min(top_k, len(pool)), lambda_div=self.rcfg.mmr_lambda)
            final = [pool[i] for i in idxs]
        else:
            final = pool[:top_k]

        # Reassign final ranks and clamp score to [0,1]
        out = []
        for rank, item in enumerate(final):
            it = dict(item)
            it["rank"] = rank
            it["score"] = float(min(1.0, max(0.0, it.get("score", 0.0))))
            out.append(it)
        return out

    # ---------------------------
    # Internals
    # ---------------------------

    def _dim_or_guess(self) -> int:
        # If configured in EmbeddingConfig, prefer that; else embed a tiny string to infer.
        if self.emb.cfg.dim:
            return int(self.emb.cfg.dim)
        v = self.emb.embed_query("ping")
        return len(v)

    def _ensure_collection(self, collection: str) -> None:
        """
        Ensure collection exists for the concrete store (dimension known from embeddings).
        The `collection` name is treated as an opaque, already-sanitized physical name.
        """
        if isinstance(self._store, PgVectorStore):
            self._store.create_collection(collection, metadata={})
        elif isinstance(self._store, QdrantStore):
            self._store.create_collection(collection, dimension=self._dim_or_guess(), metadata={})
        elif isinstance(self._store, ChromaStore):
            self._store.create_collection(collection, dimension=self._dim_or_guess(), metadata={})

    def _vector_search(self, collection: str, qvec: Sequence[float], *, top_k: int, filters: Optional[Dict]) -> List[dict]:
        # Make sure the collection exists (no-op if already present)
        self._ensure_collection(collection)
        try:
            hits = self._store.search(collection, qvec, top_k=top_k, filters=filters)
        except Exception as e:
            log.info("Vector search failed (%s); returning empty list.", e)
            hits = []
        # Normalize shape and attach raw distance → converted similarity (0..1, higher better)
        normed = []
        distances = [h["score"] for h in hits] or [1.0]
        d_min = min(distances)
        d_max = max(distances)
        for h in hits:
            dist = float(h.get("score", 0.0))
            sim = _invert_and_normalize_distance(dist, d_min, d_max)  # 0..1
            normed.append(
                {
                    "id": h.get("id"),
                    "text": h.get("text") or "",
                    "metadata": h.get("metadata") or {},
                    "score_vector": sim,   # higher better
                    "score_bm25": 0.0,     # fill later if BM25 hit
                    "source": {"vector": sim, "bm25": 0.0, "rerank": None},
                }
            )
        return normed

    def _fuse(self, vec_hits: List[dict], bm25_hits: List[dict], *, alpha: float) -> List[dict]:
        """
        Join by id where possible; otherwise union. Normalize scores locally.
        """
        # Normalize BM25 scores to 0..1 (higher better)
        bm_max = max([float(h.get("score", 0.0)) for h in bm25_hits] or [1.0])
        bm_norm = {}
        for h in bm25_hits:
            s = float(h.get("score", 0.0))
            s01 = s / bm_max if bm_max > 0 else 0.0
            bm_norm[str(h.get("id"))] = (s01, h)

        # Index vector hits by id
        by_id = {str(h.get("id")): h for h in vec_hits if h.get("id") is not None}

        # Merge: start from union of ids
        merged: Dict[str, dict] = {}

        # 1) add vector hits
        for vid, vh in by_id.items():
            bm = bm_norm.get(vid, (0.0, None))[0]
            fused = alpha * float(vh["score_vector"]) + (1.0 - alpha) * bm
            item = {
                "id": vh["id"],
                "text": vh["text"],
                "metadata": vh["metadata"],
                "score": fused,
                "source": {"vector": float(vh["score_vector"]), "bm25": float(bm), "rerank": None},
            }
            merged[vid] = item

        # 2) add BM25-only hits
        for bid, (bm_s, bh) in bm_norm.items():
            if bid in merged:
                continue
            fused = alpha * 0.0 + (1.0 - alpha) * bm_s
            merged[bid] = {
                "id": bh["id"],
                "text": bh["text"],
                "metadata": bh["metadata"],
                "score": fused,
                "source": {"vector": 0.0, "bm25": float(bm_s), "rerank": None},
            }

        # Order by fused score (desc)
        fused_list = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return fused_list

    def _maybe_rerank(self, query: str, pool: List[dict]) -> List[dict]:
        """
        Apply reranker if configured (provider != none). Preserve pool size; rescore.
        """
        prov = self.reranker.cfg.provider
        if not pool or prov == "none":
            return pool

        try:
            reranked = self.reranker.rerank(query, pool, top_k=len(pool))
        except Exception as e:
            log.info("Reranker failed (%s); skipping rerank.", e)
            return pool

        # Combine rerank score with fused for transparency; keep final order = reranked
        max_rr = max([float(h.get("rerank_score", 0.0)) for h in reranked] or [1.0])
        out = []
        for h in reranked:
            rr = float(h.get("rerank_score", 0.0))
            rr01 = rr / max_rr if max_rr > 0 else 0.0
            base = dict(h)
            base["score"] = rr01
            src = dict(base.get("source") or {})
            src["rerank"] = rr
            base["source"] = src
            out.append(base)
        return out


# ---------------------------
# Math helpers
# ---------------------------

def _invert_and_normalize_distance(dist: float, d_min: float, d_max: float) -> float:
    """
    Convert a distance (lower is better) into a similarity in [0,1] (higher is better).
    We map [d_min .. d_max] → [1 .. 0].
    """
    d_min = float(d_min)
    d_max = float(d_max)
    dist = float(dist)
    if d_max <= d_min + 1e-12:
        return 1.0  # all equal distances
    # similarity decreases linearly with distance
    return float(max(0.0, min(1.0, 1.0 - (dist - d_min) / (d_max - d_min))))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity in [0,1] (assumes non-zero vectors; we clamp).
    """
    num = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        num += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    sim = num / (math.sqrt(na) * math.sqrt(nb))
    # map from [-1,1] to [0,1] for convenience
    return float(0.5 * (sim + 1.0))


def _mmr_indices(
    q: Sequence[float],
    candidates: List[Sequence[float]],
    *,
    top_k: int,
    lambda_div: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) selection.

    Args:
      q          : query embedding
      candidates : list of candidate embeddings
      top_k      : number of items to select
      lambda_div : 0..1 (higher -> favor relevance; lower -> more diversity)

    Returns:
      list of selected indices into `candidates`.
    """
    n = len(candidates)
    if n == 0 or top_k <= 0:
        return []
    top_k = min(top_k, n)

    # Precompute relevance sims to query
    rel = [ _cosine(q, c) for c in candidates ]

    selected: List[int] = []
    remaining = set(range(n))

    # Pick the most relevant first
    first = max(range(n), key=lambda i: rel[i])
    selected.append(first)
    remaining.remove(first)

    # Iteratively add items maximizing MMR criterion
    while len(selected) < top_k and remaining:
        best_i = None
        best_val = -1.0
        for i in list(remaining):
            # diversity term = max similarity to any already selected
            div = 0.0
            for j in selected:
                div = max(div, _cosine(candidates[i], candidates[j]))
            val = lambda_div * rel[i] - (1.0 - lambda_div) * div
            if val > best_val:
                best_i = i
                best_val = val
        selected.append(best_i)          # type: ignore[arg-type]
        remaining.remove(best_i)         # type: ignore[arg-type]

    return selected


# ---------------------------
# Back-compat shim expected by apps/api/routes/search.py
# ---------------------------

def search_collection(
    collection: str,
    query: str,
    top_k: int = 5,
    filters: Optional[Dict] = None,
    retr_cfg: Optional[RetrieverConfig] = None,
    emb_cfg: Optional[EmbeddingConfig] = None,
    bm25_cfg: Optional[BM25Config] = None,
    rer_cfg: Optional[RerankConfig] = None,
) -> List[dict]:
    """
    Back-compat helper: embed query + search the active store.
    Mirrors the old API used by apps/api/routes/search.py.
    """
    searcher = HybridSearcher(retr_cfg=retr_cfg, emb_cfg=emb_cfg, bm25_cfg=bm25_cfg, rer_cfg=rer_cfg)
    return searcher.search(collection=collection, query=query, top_k=top_k, filters=filters)


# Public exports
__all__ = ["RetrieverConfig", "HybridSearcher", "search_collection"]
