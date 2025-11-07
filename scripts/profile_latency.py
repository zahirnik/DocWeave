# scripts/profile_latency.py
"""
Latency profiler for the Agentic-RAG pipeline (tutorial-clear).

What this script measures
-------------------------
- Embedding latency (per batch).
- Retriever latency (pgvector if configured; else in-memory fallback over ./data/samples).
- Optional LLM latency (OpenAI if key present).
- Optional end-to-end (retrieve → answer) latency.

Outputs
-------
- Pretty console table with p50/p90/p95/p99, mean, std for each step.
- Optional CSV with per-run timings.

Examples
--------
# Full profile with defaults (tries pgvector if PGVECTOR_URL set)
python -m scripts.profile_latency

# Skip LLM (no OpenAI key) and write CSV
python -m scripts.profile_latency --no-llm --out-csv ./data/outputs/latency_profile.csv

# Use local fallback corpus under ./data/samples; 20 runs; top-k=5
python -m scripts.profile_latency --runs 20 --top-k 5 --samples ./data/samples
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as stats
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from packages.core.config import get_settings
from packages.retriever.embeddings import EmbeddingClient

# Optional OpenAI
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# Optional pgvector store
try:
    from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # type: ignore
except Exception:
    PgVectorStore = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: corpus + fallback retriever
# ──────────────────────────────────────────────────────────────────────────────

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(p: Path) -> str:
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return ""
    out = []
    with p.open("rb") as f:
        r = PyPDF2.PdfReader(f)  # type: ignore
        for page in r.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(out)

def load_local_corpus(base: Path) -> List[Tuple[str, str]]:
    """
    Return list[(doc_id, text)] from a few lightweight formats.
    """
    pairs: List[Tuple[str, str]] = []
    if not base.exists():
        return pairs
    for pat in ("*.txt", "*.md"):
        for p in base.rglob(pat):
            t = _read_text(p)
            if t.strip():
                pairs.append((str(p), t))
    for p in base.rglob("*.pdf"):
        t = _read_pdf(p)
        if t.strip():
            pairs.append((str(p), t))
    for p in base.rglob("*.jsonl"):
        try:
            import json
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    t = d.get("text") or d.get("content") or ""
                    if isinstance(t, str) and t.strip():
                        pairs.append((f"{p}#r{i}", t))
        except Exception:
            continue
    return pairs


class LocalRetriever:
    """
    Minimal cosine retriever for fallback timing.
    """
    def __init__(self, embedder: EmbeddingClient, pairs: List[Tuple[str, str]]):
        self.ids = [pid for pid, _ in pairs]
        self.texts = [t for _, t in pairs]
        E = np.array(embedder.embed(self.texts, batch_size=64), dtype="float32")
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        self.E = E
        self.embedder = embedder

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q = np.array(self.embedder.embed([query])[0], dtype="float32")
        q /= (np.linalg.norm(q) + 1e-12)
        sims = (self.E @ q)
        order = sims.argsort()[::-1][:top_k]
        out: List[Dict[str, Any]] = []
        for i in order:
            out.append({"id": self.ids[int(i)], "text": self.texts[int(i)], "score": float(sims[int(i)])})
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SeriesStats:
    name: str
    count: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

def _percentile(x: List[float], q: float) -> float:
    if not x:
        return 0.0
    i = max(0, min(len(x) - 1, int(math.ceil(q * len(x)) - 1)))
    return sorted(x)[i]

def summarise(name: str, ms: List[float]) -> SeriesStats:
    return SeriesStats(
        name=name,
        count=len(ms),
        mean_ms=float(stats.mean(ms)) if ms else 0.0,
        std_ms=float(stats.pstdev(ms)) if len(ms) > 1 else 0.0,
        p50_ms=_percentile(ms, 0.50),
        p90_ms=_percentile(ms, 0.90),
        p95_ms=_percentile(ms, 0.95),
        p99_ms=_percentile(ms, 0.99),
    )

def print_table(rows: List[SeriesStats]) -> None:
    if not rows:
        print("(no rows)")
        return
    hdr = f"{'Step':20} {'N':>5} {'mean':>8} {'std':>8} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}  (ms)"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r.name:20} {r.count:5d} {r.mean_ms:8.1f} {r.std_ms:8.1f} {r.p50_ms:8.1f} {r.p90_ms:8.1f} {r.p95_ms:8.1f} {r.p99_ms:8.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Profilers
# ──────────────────────────────────────────────────────────────────────────────

def profile_embeddings(embedder: EmbeddingClient, runs: int, batch: int) -> List[float]:
    """
    Measure time to embed a batch of short finance-y strings.
    """
    # synthetic batch with light variety so tokenisation isn't identical
    texts = [
        "ACME PLC Annual Report 2024: gross margin 38.2%.",
        "Beta Corp Q3 2024 revenue reached $1.26bn, up 14.5% YoY.",
        "Gamma Ltd net debt to EBITDA stands at 3.0x, leverage policy intact.",
        "Delta Inc guidance reiterated; operating cash flow improved.",
        "EPS diluted 2.31; capex normalised; headcount stable.",
        "Scope 1 emissions 120,000 tCO2e; SBTi targets approved.",
        "Inventory days declined; receivables improved QoQ.",
        "Segment margin expansion driven by procurement savings.",
    ]
    # replicate to reach batch size
    while len(texts) < batch:
        texts.extend(texts[: max(1, batch - len(texts))])
    texts = texts[:batch]

    ms: List[float] = []
    # warmups included? We'll keep explicit warmups in main; here we just time runs.
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = embedder.embed(texts, batch_size=batch)
        t1 = time.perf_counter()
        ms.append((t1 - t0) * 1000.0)
    return ms


def profile_retriever(
    embedder: EmbeddingClient,
    top_k: int,
    runs: int,
    use_pgvector: bool,
    cfg,
    samples_dir: Path,
) -> List[float]:
    """
    Measure search latency. If pgvector is configured, we approximate production:
      - pre-embedded query vector → store.search(qv, top_k)
    Else we use LocalRetriever with in-memory matrix multiplication.
    """
    queries = [
        "What was ACME's gross margin in 2024?",
        "How much revenue did Beta Corp report in Q3 2024?",
        "What is the net debt to EBITDA ratio?",
        "Did operating cash flow improve?",
        "What were Scope 1 emissions?",
    ]

    ms: List[float] = []

    if use_pgvector and PgVectorStore and cfg.pgvector_url:
        # Assume the user already indexed documents in a collection.
        collection = os.getenv("PGV_COLLECTION", "finance_demo")
        store = PgVectorStore(dsn=cfg.pgvector_url, collection=collection, create_if_missing=False)
        # Embed queries up-front so we time only store.search()
        qvecs = [embedder.embed([q])[0] for q in queries]
        for i in range(runs):
            qv = qvecs[i % len(qvecs)]
            t0 = time.perf_counter()
            _ = store.search(qv, top_k=top_k)
            t1 = time.perf_counter()
            ms.append((t1 - t0) * 1000.0)
        return ms

    # Fallback: local in-memory
    pairs = load_local_corpus(samples_dir)
    if not pairs:
        pairs = [
            ("samples://acme", "ACME PLC Annual Report 2024: gross margin was 38.2%."),
            ("samples://beta", "Beta Corp Q3 2024 revenue was $1.26bn, up 14.5% year over year."),
            ("samples://delta", "Delta Inc net debt to EBITDA 3.0x in FY2023."),
            ("samples://gamma", "Operating cash flow improved on inventory normalisation."),
            ("samples://acme-s1", "Scope 1 emissions 120,000 tCO2e reported per GHG Protocol."),
        ]
    retr = LocalRetriever(embedder, pairs)
    for i in range(runs):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        _ = retr.search(q, top_k=top_k)
        t1 = time.perf_counter()
        ms.append((t1 - t0) * 1000.0)
    return ms


def profile_llm(cfg, runs: int, max_tokens: int = 128) -> List[float]:
    """
    Measure LLM chat completion latency with a short finance prompt.
    Skips if OpenAI not configured.
    """
    if not cfg.openai_api_key or OpenAI is None:
        return []

    client = OpenAI(api_key=cfg.openai_api_key)
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    prompt = (
        "You are a concise finance assistant. In one sentence, explain what gross margin represents, "
        "and keep the answer under 25 words."
    )

    ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        t1 = time.perf_counter()
        ms.append((t1 - t0) * 1000.0)
    return ms


def profile_end_to_end(
    embedder: EmbeddingClient,
    cfg,
    runs: int,
    top_k: int,
    use_pgvector: bool,
    samples_dir: Path,
) -> List[float]:
    """
    Retrieve → (optional) LLM answer timing. We keep the LLM call optional to make the
    measurement apples-to-apples across environments.
    """
    if cfg.openai_api_key and OpenAI is not None:
        client = OpenAI(api_key=cfg.openai_api_key)
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    else:
        client = None
        model = None

    queries = [
        "What was ACME's gross margin in 2024?",
        "How much revenue did Beta Corp report in Q3 2024?",
        "What is the net debt to EBITDA ratio?",
        "What were Scope 1 emissions?",
        "Summarise operating cash flow trend briefly.",
    ]

    # Prepare retriever
    if use_pgvector and PgVectorStore and cfg.pgvector_url:
        collection = os.getenv("PGV_COLLECTION", "finance_demo")
        store = PgVectorStore(dsn=cfg.pgvector_url, collection=collection, create_if_missing=False)

        def retrieve(q: str):
            qv = embedder.embed([q])[0]
            return store.search(qv, top_k=top_k)

    else:
        pairs = load_local_corpus(samples_dir)
        if not pairs:
            pairs = [
                ("samples://acme", "ACME PLC Annual Report 2024: gross margin was 38.2%."),
                ("samples://beta", "Beta Corp Q3 2024 revenue was $1.26bn, up 14.5% year over year."),
                ("samples://delta", "Delta Inc net debt to EBITDA 3.0x in FY2023."),
                ("samples://gamma", "Operating cash flow improved on inventory normalisation."),
                ("samples://acme-s1", "Scope 1 emissions 120,000 tCO2e reported per GHG Protocol."),
            ]
        retr = LocalRetriever(embedder, pairs)

        def retrieve(q: str):
            return retr.search(q, top_k=top_k)

    def answer(q: str, ctx_texts: List[str]) -> str:
        if client is None:
            # Heuristic fallback; avoid counting heavy work here
            if "gross margin" in q.lower():
                return "Gross margin was 38.2% [1]."
            return (ctx_texts[0][:160] + ("…" if len(ctx_texts[0]) > 160 else "")) if ctx_texts else "I don't know."
        ctx = "\n\n".join(f"[{i+1}] {c[:400]}" for i, c in enumerate(ctx_texts))
        sys = "Use only the provided contexts and cite as [n]. Keep answer short."
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": f"Q: {q}\n\nContexts:\n{ctx}"}],
            temperature=0.2,
            max_tokens=160,
        )
        return (resp.choices[0].message.content or "").strip()

    ms: List[float] = []
    for i in range(runs):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        hits = retrieve(q)
        ctxs = [h.get("text") or "" for h in hits]
        _ = answer(q, ctxs)
        t1 = time.perf_counter()
        ms.append((t1 - t0) * 1000.0)
    return ms


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Latency profiler for Agentic-RAG")
    ap.add_argument("--runs", type=int, default=10, help="Number of timed runs per step (default: 10)")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup runs to discard (default: 2)")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size (default: 64)")
    ap.add_argument("--top-k", type=int, default=5, help="Retriever top-k (default: 5)")
    ap.add_argument("--samples", default="./data/samples", help="Local samples directory for fallback retriever")
    ap.add_argument("--no-embed", action="store_true", help="Skip embedding profiling")
    ap.add_argument("--no-retrieval", action="store_true", help="Skip retriever profiling")
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM profiling")
    ap.add_argument("--no-e2e", action="store_true", help="Skip end-to-end profiling")
    ap.add_argument("--out-csv", help="Write per-run timings to CSV")
    args = ap.parse_args()

    cfg = get_settings()
    use_pgvector = bool(PgVectorStore and cfg.pgvector_url)
    samples_dir = Path(args.samples).resolve()

    # Embedder
    embedder = EmbeddingClient(provider="auto", model_alias=os.getenv("EMBED_MODEL_ALIAS", "mini"))

    # Warmups
    warm = max(0, int(args.warmup))
    runs = max(1, int(args.runs))
    if not args.no_embed:
        _ = profile_embeddings(embedder, runs=warm, batch=max(1, int(args.batch)))
    if not args.no_retrieval:
        _ = profile_retriever(embedder, top_k=int(args.top_k), runs=warm, use_pgvector=use_pgvector, cfg=cfg, samples_dir=samples_dir)
    if not args.no_llm:
        _ = profile_llm(cfg, runs=warm)
    if not args.no_e2e:
        _ = profile_end_to_end(embedder, cfg, runs=warm, top_k=int(args.top_k), use_pgvector=use_pgvector, samples_dir=samples_dir)

    # Actual timing
    rows: List[SeriesStats] = []
    csv_rows: List[Dict[str, Any]] = []

    if not args.no_embed:
        ms = profile_embeddings(embedder, runs=runs, batch=max(1, int(args.batch)))
        rows.append(summarise("embed_batch", ms))
        csv_rows.extend({"step": "embed_batch", "ms": v} for v in ms)

    if not args.no_retrieval:
        ms = profile_retriever(embedder, top_k=int(args.top_k), runs=runs, use_pgvector=use_pgvector, cfg=cfg, samples_dir=samples_dir)
        rows.append(summarise("retrieve", ms))
        csv_rows.extend({"step": "retrieve", "ms": v} for v in ms)

    if not args.no_llm:
        ms = profile_llm(cfg, runs=runs)
        if ms:
            rows.append(summarise("llm_chat", ms))
            csv_rows.extend({"step": "llm_chat", "ms": v} for v in ms)
        else:
            print("(LLM) skipped: OpenAI not configured or package not installed.")

    if not args.no_e2e:
        ms = profile_end_to_end(embedder, cfg, runs=runs, top_k=int(args.top_k), use_pgvector=use_pgvector, samples_dir=samples_dir)
        rows.append(summarise("e2e_retrieve_answer", ms))
        csv_rows.extend({"step": "e2e_retrieve_answer", "ms": v} for v in ms)

    # Console summary
    print("\nLatency summary")
    print("───────────────")
    print_table(rows)

    # Hints
    print("\nEnvironment")
    print("───────────")
    print(json.dumps({
        "pgvector": use_pgvector,
        "openai": bool(cfg.openai_api_key and OpenAI is not None),
        "embed_model_alias": getattr(embedder, "model_alias", "unknown"),
        "runs": runs,
        "warmup": warm,
        "top_k": int(args.top_k),
        "batch": int(args.batch),
    }, indent=2))

    # CSV
    if args.out_csv:
        import csv as _csv
        out = Path(args.out_csv).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["step", "ms"])
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"\nWrote CSV: {out}")

    print("\nDone.")

if __name__ == "__main__":
    main()
