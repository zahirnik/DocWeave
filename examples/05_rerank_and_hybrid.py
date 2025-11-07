# examples/05_rerank_and_hybrid.py
"""
Compare BM25 vs Vector vs Hybrid retrieval (tutorial style).

What this example shows
-----------------------
- Build a tiny in-memory corpus from ./data/samples (txt/md/pdf/jsonl[text]).
- Create two retrievers:
    • BM25 (rank_bm25 if available; fallback: TF-IDF or token-overlap)
    • Vector (our EmbeddingClient; cosine similarity)
- Hybrid scoring = α * norm(BM25) + (1-α) * norm(cosine)
- Print the top-k for each method side-by-side with scores.

Why this matters
----------------
Hybrid retrieval often outperforms either method alone on finance PDFs:
BM25 excels at exact term matches (tickers, figures, section names);
vectors excel at semantic paraphrases.

Run
---
  python -m examples.05_rerank_and_hybrid \
    --query "What was ACME's gross margin in 2024?" \
    --top-k 5 --alpha 0.5

Notes
-----
- This script is self-contained and will run even if `rank_bm25`/`sklearn` are missing.
- To keep it focused, we index text directly from files (no pgvector here).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Our embedding client (auto-selects OpenAI if key set; else local fallback)
from packages.retriever.embeddings import EmbeddingClient


# ---------------------------
# Sample loading (same style as other examples)
# ---------------------------

def _read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _read_md(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(p: Path) -> str:
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return ""
    out = []
    with open(p, "rb") as f:
        reader = PyPDF2.PdfReader(f)  # type: ignore
        for page in reader.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(out)

def _read_jsonl(p: Path) -> List[str]:
    out: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                t = d.get("text") or d.get("content") or ""
                if isinstance(t, str) and t.strip():
                    out.append(t)
            except Exception:
                continue
    return out

def load_corpus(base: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (doc_id, text).
    """
    items: List[Tuple[str, str]] = []
    if not base.exists():
        return items
    for pat in ("*.txt", "*.md", "*.pdf", "*.jsonl"):
        for p in sorted(base.rglob(pat)):
            if p.suffix.lower() in {".txt", ".md"}:
                txt = _read_txt(p) if p.suffix.lower() == ".txt" else _read_md(p)
                if txt.strip():
                    items.append((str(p), txt))
            elif p.suffix.lower() == ".pdf":
                txt = _read_pdf(p)
                if txt.strip():
                    items.append((str(p), txt))
            elif p.suffix.lower() == ".jsonl":
                rows = _read_jsonl(p)
                for i, r in enumerate(rows):
                    items.append((f"{p}#r{i}", r))
    return items


# ---------------------------
# BM25 retriever (with graceful fallbacks)
# ---------------------------

def _tokenize(s: str) -> List[str]:
    # simple whitespace + punctuation trim
    import re
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9.%\- ]+", " ", s)
    return [t for t in s.split() if t]

class BM25Retriever:
    """
    BM25 via rank_bm25 if available; else TF-IDF (sklearn); else token overlap.
    """

    def __init__(self, docs: List[str]):
        self.docs = docs
        self.mode = "overlap"
        self._setup()

    def _setup(self):
        # Try rank_bm25
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
            self.tokens = [ _tokenize(d) for d in self.docs ]
            self._bm25 = BM25Okapi(self.tokens)
            self.mode = "bm25"
            return
        except Exception:
            self._bm25 = None

        # Try sklearn TF-IDF cosine
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
            self._tfidf = TfidfVectorizer(tokenizer=_tokenize, lowercase=False)
            self._X = self._tfidf.fit_transform(self.docs)
            self._cos = cosine_similarity
            self.mode = "tfidf"
            return
        except Exception:
            self._tfidf = None
            self._X = None
            self._cos = None

        # Fallback: token overlap (Jaccard)
        self.tokens = [ set(_tokenize(d)) for d in self.docs ]
        self.mode = "overlap"

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.mode == "bm25":
            qtok = _tokenize(query)
            scores = self._bm25.get_scores(qtok)  # type: ignore[attr-defined]
            order = np.argsort(scores)[::-1][:top_k]
            return [(int(i), float(scores[int(i)])) for i in order]
        if self.mode == "tfidf":
            qv = self._tfidf.transform([query])  # type: ignore[attr-defined]
            sims = self._cos(self._X, qv).ravel()  # type: ignore[attr-defined]
            order = np.argsort(sims)[::-1][:top_k]
            return [(int(i), float(sims[int(i)])) for i in order]
        # overlap (Jaccard)
        q = set(_tokenize(query))
        scores = []
        for i, toks in enumerate(self.tokens):
            inter = len(q & toks)
            uni = len(q | toks) or 1
            scores.append(inter / uni)
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[int(i)])) for i in order]


# ---------------------------
# Vector retriever
# ---------------------------

class VectorRetriever:
    def __init__(self, docs: List[str], embedder: EmbeddingClient):
        self.embedder = embedder
        E = np.array(self.embedder.embed(docs, batch_size=64), dtype="float32")
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        self.E = E

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q = np.array(self.embedder.embed([query])[0], dtype="float32")
        q /= (np.linalg.norm(q) + 1e-12)
        sims = (self.E @ q)
        order = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[int(i)])) for i in order]


# ---------------------------
# Hybrid combiner
# ---------------------------

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def hybrid_search(
    query: str,
    bm25: BM25Retriever,
    vec: VectorRetriever,
    top_k: int = 5,
    alpha: float = 0.5,
) -> List[Tuple[int, float]]:
    """
    alpha in [0,1]: 1 → BM25 only, 0 → Vector only.
    """
    # Gather candidate set = union of top-(top_k*3) from each
    kx = max(top_k * 3, 10)
    b_hits = bm25.search(query, top_k=kx)
    v_hits = vec.search(query, top_k=kx)

    # Merge indices
    idxs = sorted(set([i for i, _ in b_hits] + [i for i, _ in v_hits]))

    # Build aligned score arrays (missing → 0)
    b_map = {i: s for i, s in b_hits}
    v_map = {i: s for i, s in v_hits}
    b_scores = np.array([b_map.get(i, 0.0) for i in idxs], dtype="float32")
    v_scores = np.array([v_map.get(i, 0.0) for i in idxs], dtype="float32")

    # Normalise to 0..1 independently (BM25 and cosine are not on the same scale)
    b_norm = _minmax_norm(b_scores)
    v_norm = _minmax_norm(v_scores)

    # Weighted sum
    combo = alpha * b_norm + (1.0 - alpha) * v_norm

    order = np.argsort(combo)[::-1][:top_k]
    return [(int(idxs[int(i)]), float(combo[int(i)])) for i in order]


# ---------------------------
# Pretty printer
# ---------------------------

def _print_results(title: str, hits: List[Tuple[int, float]], ids: List[str], texts: List[str], width: int = 140) -> None:
    print(f"\n{title}")
    print("─" * len(title))
    for rank, (i, s) in enumerate(hits, 1):
        src = ids[i]
        txt = " ".join((texts[i] or "").split())
        snippet = txt[: width] + ("…" if len(txt) > width else "")
        print(f"{rank:>2}. score={s:0.4f}  id={src}")
        print(f"    {snippet}")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare BM25 vs Vector vs Hybrid retrieval")
    ap.add_argument("--samples", default="./data/samples", help="Folder to scan for sample files")
    ap.add_argument("--query", default="What was ACME's gross margin in 2024?", help="Query text")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k results to display")
    ap.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight: 1=BM25 only, 0=Vector only")
    args = ap.parse_args()

    base = Path(args.samples).resolve()
    pairs = load_corpus(base)

    if not pairs:
        # Seed tiny synthetic corpus so demo always runs
        pairs = [
            ("samples://acme-2024", "ACME PLC Annual Report 2024: Gross margin was 38.2% for FY2024, reflecting improved procurement and pricing."),
            ("samples://beta-q3", "Beta Corp 10-Q Q3 2024: Revenue reached $1.26bn, up ~14.5% YoY from $1.10bn."),
            ("samples://gamma-expenses", "Gamma Ltd 2024: Major expense lines include cost of sales, SG&A and R&D."),
            ("samples://delta-leverage", "Delta Inc: Net debt to EBITDA stood at 3.0x in FY2023."),
            ("samples://acme-esg", "ACME PLC Sustainability 2024: Scope 1 emissions 120,000 tCO2e under GHG Protocol."),
        ]

    ids = [pid for pid, _ in pairs]
    texts = [t for _, t in pairs]

    # Build retrievers
    embedder = EmbeddingClient(provider="auto", model_alias="mini")
    bm25 = BM25Retriever(texts)
    vec = VectorRetriever(texts, embedder=embedder)

    # Search
    q = args.query
    top_k = int(args.top_k)
    alpha = float(max(0.0, min(1.0, args.alpha)))

    b_hits = bm25.search(q, top_k=top_k)
    v_hits = vec.search(q, top_k=top_k)
    h_hits = hybrid_search(q, bm25, vec, top_k=top_k, alpha=alpha)

    # Report
    print(f"Query: {q}")
    print(f"Docs : {len(texts)} | BM25 mode: {bm25.mode} | Vector dim: {len(embedder.embed(['x'])[0])}")

    _print_results("BM25", b_hits, ids, texts)
    _print_results("Vector (cosine)", v_hits, ids, texts)
    _print_results(f"Hybrid α={alpha:0.2f} (α*BM25 + (1-α)*Vector)", h_hits, ids, texts)

    print("\nTip: Adjust --alpha to see how the ranking changes (e.g., 0.2, 0.8).")

if __name__ == "__main__":
    main()
