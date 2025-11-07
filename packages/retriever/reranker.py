# packages/retriever/reranker.py
"""
Cross-encoder / API reranker — optional, tiny, and cost-aware.

What this module provides
-------------------------
- RerankConfig: config knobs (provider, model, top_k, batch_size, etc.).
- Reranker: small facade with a single method:
    rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]

Where to use it
---------------
Call the reranker **after** you have a candidate set from vector/BM25/hybrid
retrieval (e.g., 20–100 items). It will rescore each (query, doc) pair and
return the best top_k with a uniform "rerank_score" (higher = better).

Providers
---------
- "local": sentence-transformers CrossEncoder (no network; good for dev/test)
    • model default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    • requires: `pip install sentence-transformers`

- "cohere": Cohere Rerank API (paid; fast, strong quality)
    • model default: "rerank-english-v3.0"
    • env: COHERE_API_KEY
    • requires: `pip install cohere`

If the provider is "none", rerank is a no-op (identity order).

Input shape (candidates)
------------------------
Each candidate dict must have:
    {"id": "...", "text": "...", "metadata": {...}, "score": <float>}
Score is the *upstream* score (e.g., vector distance or BM25 score). We do not
use it for reranking, but we pass it through in the output.

Output shape
------------
The returned list has the **same dict shape**, plus:
    "rerank_score": float   (higher is better)
    "rank": int             (0-based rank after reranking)

Design goals
------------
- Keep it tutorial-clear with small, isolated code paths.
- Avoid hard dependencies when a provider is unused.
- Make batch size explicit for local models to control memory.

Examples
--------
from packages.retriever.reranker import Reranker, RerankConfig

rr = Reranker()  # reads env, defaults to "none"
hits = rr.rerank("operating margin trend", candidates)  # candidates from hybrid search
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------
# Config
# ---------------------------

@dataclass
class RerankConfig:
    provider: str = "none"  # "none" | "local" | "cohere"
    model: str = ""         # provider-specific; see below
    top_k: int = 10
    batch_size: int = 32    # for local cross-encoder
    timeout_s: int = 30

    @staticmethod
    def from_env() -> "RerankConfig":
        """
        Environment variables:
          RERANK_PROVIDER=none|local|cohere
          RERANK_MODEL=...
          RERANK_TOP_K=10
          RERANK_BATCH=32
          RERANK_TIMEOUT_S=30
        """
        prov = (os.getenv("RERANK_PROVIDER") or "none").strip().lower()
        # sensible defaults
        if prov == "local":
            default_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        elif prov == "cohere":
            default_model = "rerank-english-v3.0"
        else:
            default_model = ""

        return RerankConfig(
            provider=prov,
            model=os.getenv("RERANK_MODEL", default_model),
            top_k=int(os.getenv("RERANK_TOP_K", "10")),
            batch_size=int(os.getenv("RERANK_BATCH", "32")),
            timeout_s=int(os.getenv("RERANK_TIMEOUT_S", "30")),
        )


# ---------------------------
# Main facade
# ---------------------------

class Reranker:
    """
    Small, explicit reranker facade with provider pluggability.
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        self.cfg = config or RerankConfig.from_env()
        self._provider = self.cfg.provider
        self._model = self.cfg.model

        # Lazy handles
        self._local_model = None  # CrossEncoder
        self._cohere = None       # cohere.Client

    # ---------------------------
    # Public API
    # ---------------------------

    def rerank(self, query: str, candidates: List[dict], top_k: Optional[int] = None) -> List[dict]:
        """
        Rerank `candidates` given a `query`. Returns a new list limited to `top_k`.

        Args:
          query      : user query string
          candidates : list of {"id","text","metadata","score"}
          top_k      : override config.top_k; if None, uses config.top_k

        Returns:
          list of dicts, with added fields:
            "rerank_score": float (higher better)
            "rank": int (0-based)
        """
        if not candidates:
            return []

        k = max(1, int(top_k or self.cfg.top_k))

        if self._provider == "none":
            # Identity order; attach neutral scores
            out = []
            for i, c in enumerate(candidates[:k]):
                item = dict(c)
                item["rerank_score"] = 0.0
                item["rank"] = i
                out.append(item)
            return out

        if self._provider == "local":
            return self._rerank_local(query, candidates, k)

        if self._provider == "cohere":
            return self._rerank_cohere(query, candidates, k)

        # Fallback: identity
        out = []
        for i, c in enumerate(candidates[:k]):
            item = dict(c)
            item["rerank_score"] = 0.0
            item["rank"] = i
            out.append(item)
        return out

    # ---------------------------
    # Provider: Local CrossEncoder
    # ---------------------------

    def _ensure_local(self):
        if self._local_model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Local reranker requires `sentence-transformers`. Install with:\n"
                "  pip install sentence-transformers"
            ) from e

        name = self._model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            self._local_model = CrossEncoder(name)
        except Exception as e:
            raise RuntimeError(f"Failed to load CrossEncoder model '{name}': {e}") from e

    def _rerank_local(self, query: str, candidates: List[dict], top_k: int) -> List[dict]:
        self._ensure_local()
        assert self._local_model is not None

        # Build pairs [(query, doc), ...]
        pairs = [(query, str(c.get("text") or "")) for c in candidates]

        # Batched scoring for memory-friendliness
        scores: List[float] = []
        bsz = max(1, int(self.cfg.batch_size))
        for i in range(0, len(pairs), bsz):
            batch = pairs[i : i + bsz]
            try:
                s = self._local_model.predict(batch, convert_to_numpy=True, show_progress_bar=False)
            except Exception as e:
                raise RuntimeError(f"CrossEncoder predict failed: {e}") from e
            # Ensure list of floats
            scores.extend([float(x) for x in (s.tolist() if hasattr(s, "tolist") else list(s))])

        # Attach and sort (higher score is better)
        aug = []
        for c, sc in zip(candidates, scores):
            item = dict(c)
            item["rerank_score"] = float(sc)
            aug.append(item)

        aug.sort(key=lambda x: x["rerank_score"], reverse=True)

        out = []
        for rank, item in enumerate(aug[:top_k]):
            item = dict(item)
            item["rank"] = rank
            out.append(item)
        return out

    # ---------------------------
    # Provider: Cohere Rerank API
    # ---------------------------

    def _ensure_cohere(self):
        if self._cohere is not None:
            return
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY is not set; cannot use provider 'cohere'.")
        try:
            import cohere  # type: ignore
        except Exception as e:
            raise RuntimeError("Cohere reranker requires `cohere`. Install with: pip install cohere") from e
        self._cohere = cohere.Client(api_key)

    def _rerank_cohere(self, query: str, candidates: List[dict], top_k: int) -> List[dict]:
        self._ensure_cohere()
        assert self._cohere is not None

        # Cohere expects a list of documents (strings); we pass the text field.
        docs = [str(c.get("text") or "") for c in candidates]
        model = self._model or "rerank-english-v3.0"
        try:
            resp = self._cohere.rerank(
                model=model,
                query=query,
                documents=docs,
                top_n=top_k,
            )
        except Exception as e:
            raise RuntimeError(f"Cohere rerank failed: {e}") from e

        # Response contains indices into the original docs with relevance scores
        # Normalize into our candidate dicts
        # Note: some SDKs return resp.results with fields: index, relevance_score
        idx_to_score = {}
        ordered_indices = []
        for r in getattr(resp, "results", []) or []:
            i = int(getattr(r, "index", -1))
            sc = float(getattr(r, "relevance_score", 0.0))
            if 0 <= i < len(candidates):
                idx_to_score[i] = sc
                ordered_indices.append(i)

        out = []
        for rank, i in enumerate(ordered_indices[:top_k]):
            base = candidates[i]
            item = dict(base)
            item["rerank_score"] = float(idx_to_score.get(i, 0.0))
            item["rank"] = rank
            out.append(item)

        # If Cohere returned fewer than requested, pad with identity order (rare)
        if len(out) < top_k:
            seen = {o["id"] for o in out}
            for c in candidates:
                if c.get("id") in seen:
                    continue
                item = dict(c)
                item["rerank_score"] = 0.0
                item["rank"] = len(out)
                out.append(item)
                if len(out) >= top_k:
                    break

        return out


__all__ = ["Reranker", "RerankConfig"]
