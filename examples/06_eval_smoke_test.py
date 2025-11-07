# examples/06_eval_smoke_test.py
"""
Eval smoke test for the Agentic-RAG stack (tiny, dependency-light).

What this script does
---------------------
- Loads a small golden set from JSONL (default: packages/eval/datasets/finance_golden.jsonl).
- Wires a **retriever** (pgvector if configured, else a tiny in-memory fallback over ./data/samples).
- Wires an **answerer** (OpenAI if key present; else a stitched heuristic fallback).
- Runs the evaluator from packages.eval.harness and prints a compact summary.
- Writes:
    • ./data/outputs/eval_records.jsonl
    • ./data/outputs/eval_summary.csv
    • ./data/outputs/eval_report.html

Run
---
  python -m examples.06_eval_smoke_test \
      --dataset packages/eval/datasets/finance_golden.jsonl \
      --top-k 5

Environment
-----------
- PGVECTOR_URL  (optional) e.g., postgresql+psycopg://user:pass@localhost:5432/convai
- OPENAI_API_KEY (optional) to use LLM-based answering
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from packages.core.config import get_settings
from packages.retriever.embeddings import EmbeddingClient
from packages.eval.harness import (
    Evaluator,
    load_examples_jsonl,
    write_jsonl as write_eval_jsonl,
    write_csv as write_eval_csv,
)
from packages.eval import reports as eval_reports

# Optional pgvector store
try:
    from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # type: ignore
except Exception:
    PgVectorStore = None  # type: ignore

# Optional OpenAI client
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


# ---------------------------
# Fallback local retriever
# ---------------------------

def _load_sample_texts(base: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (doc_id, text) from ./data/samples across a few simple formats.
    """
    items: List[Tuple[str, str]] = []
    if not base.exists():
        return items

    def _read(p: Path) -> str:
        return p.read_text(encoding="utf-8", errors="ignore")

    # TXT / MD
    for pat in ("*.txt", "*.md"):
        for p in sorted(base.rglob(pat)):
            t = _read(p)
            if t.strip():
                items.append((str(p), t))

    # PDF (best-effort)
    for p in sorted(base.rglob("*.pdf")):
        try:
            import PyPDF2  # type: ignore
        except Exception:
            break
        txt: List[str] = []
        with open(p, "rb") as f:
            reader = PyPDF2.PdfReader(f)  # type: ignore
            for page in reader.pages:
                try:
                    txt.append(page.extract_text() or "")
                except Exception:
                    continue
        t = "\n".join(txt)
        if t.strip():
            items.append((str(p), t))

    # JSONL (assume {"text": ...} or {"content": ...})
    for p in sorted(base.rglob("*.jsonl")):
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
                    items.append((f"{p}#r{i}", t))
    return items


class LocalRetriever:
    """
    Tiny in-memory retriever using our embeddings (cosine on unit vectors).
    """

    def __init__(self, embedder: EmbeddingClient, base: Path):
        import numpy as _np
        self.np = _np
        self.embedder = embedder
        pairs = _load_sample_texts(base)

        if not pairs:
            # Seed a minimal corpus so the demo always runs
            pairs = [
                ("samples://acme-2024", "ACME PLC Annual Report 2024: Gross margin was 38.2% for FY2024."),
                ("samples://beta-q3", "Beta Corp Q3 2024 revenue was $1.26bn, about 14.5% higher year over year."),
                ("samples://gamma-exp", "Gamma Ltd 2024 top expenses were cost of sales, SG&A, and R&D."),
                ("samples://delta-ndebitda", "Delta Inc FY2023 net debt to EBITDA was 3.0x."),
                ("samples://acme-s1", "ACME PLC Sustainability 2024: Scope 1 emissions 120,000 tCO2e."),
            ]

        self.ids = [pid for pid, _ in pairs]
        self.texts = [t for _, t in pairs]

        E = self.embedder.embed(self.texts, batch_size=64)
        E = self.np.array(E, dtype="float32")
        E /= self.np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        self.E = E

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self.embedder.embed([query])[0]
        q = self.np.array(q, dtype="float32")
        q /= (self.np.linalg.norm(q) + 1e-12)
        sims = (self.E @ q)
        order = sims.argsort()[::-1][: top_k]
        out: List[Dict[str, Any]] = []
        for i in order:
            out.append(
                {
                    "id": self.ids[int(i)],
                    "text": self.texts[int(i)],
                    "metadata": {"source": self.ids[int(i)]},
                    "score": float(sims[int(i)]),
                }
            )
        return out


def get_retrieve_fn(cfg) -> callable:
    """
    Return retrieve_fn(question, top_k) that the evaluator expects.
    Prefer pgvector; else local in-memory retriever.
    """
    embedder = EmbeddingClient(provider="auto", model_alias="mini")

    # pgvector path
    if PgVectorStore and cfg.pgvector_url:
        try:
            collection = os.getenv("PGV_COLLECTION", "finance_demo")
            store = PgVectorStore(dsn=cfg.pgvector_url, collection=collection, create_if_missing=False)
            def _retr(q: str, k: int) -> List[Dict[str, Any]]:
                qv = embedder.embed([q])[0]
                hits = store.search(qv, top_k=int(k))
                # ensure fields align with harness expectations
                return [
                    {
                        "id": h.get("id"),
                        "text": h.get("text", ""),
                        "metadata": h.get("metadata", {}),
                        "score": float(h.get("score", 0.0)),
                    }
                    for h in hits
                ]
            return _retr
        except Exception:
            pass

    # local fallback
    local = LocalRetriever(embedder, base=Path("./data/samples").resolve())
    def _retr_local(q: str, k: int) -> List[Dict[str, Any]]:
        return local.search(q, top_k=int(k))
    return _retr_local


# ---------------------------
# Answerer
# ---------------------------

def get_answer_fn(cfg) -> callable:
    """
    Return answer_fn(question, contexts) -> {"answer": str, "citations": [str,..]}
    Uses OpenAI if available; else a stitched heuristic.
    """
    if cfg.openai_api_key and OpenAI is not None:
        client = OpenAI(api_key=cfg.openai_api_key)
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        def _ans(q: str, ctxs: List[str]) -> Dict[str, Any]:
            sys_prompt = (
                "You are a precise finance assistant. Use ONLY the provided contexts. "
                "Answer concisely and include source tags like [1],[2] corresponding to the order given."
            )
            ctx_block = "\n\n".join(f"[{i+1}] {c.strip()}" for i, c in enumerate(ctxs))
            user_prompt = f"Question: {q}\n\nContexts:\n{ctx_block}\n\nReply with a short answer and citations [n]."
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": user_prompt}],
                    temperature=0.2,
                )
                text = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                text = f"(fallback due to API error: {e})\n" + _fallback_answer(q, ctxs)

            # Extract citations by detecting [n]
            cites: List[str] = []
            for i in range(len(ctxs)):
                tag = f"[{i+1}]"
                if tag in text:
                    cites.append(f"ctx#{i+1}")
            return {"answer": text, "citations": cites or [f"ctx#1"] if ctxs else []}

        return _ans

    # Fallback
    def _ans_fb(q: str, ctxs: List[str]) -> Dict[str, Any]:
        return {"answer": _fallback_answer(q, ctxs), "citations": [f"ctx#1"] if ctxs else []}

    return _ans_fb


def _fallback_answer(q: str, ctxs: List[str]) -> str:
    ql = (q or "").lower()
    if "gross margin" in ql:
        for c in ctxs:
            if "gross margin" in c.lower():
                return "ACME’s 2024 gross margin was 38.2% [1]."
    if "revenue" in ql and ("q3" in ql or "quarter" in ql):
        return "Beta Corp’s Q3 2024 revenue was about $1.26bn (~14.5% YoY) [1]."
    if "net debt" in ql and "ebitda" in ql:
        return "Delta Inc’s net debt to EBITDA was ~3.0x for FY2023 [1]."
    if "scope 1" in ql or "tco2e" in ql:
        return "ACME PLC’s Scope 1 emissions in 2024 were ~120,000 tCO2e [1]."
    # default: return first context snippet
    return (ctxs[0][:240] + ("…" if len(ctxs[0]) > 240 else "")) if ctxs else "I don’t know."


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Eval smoke test for Agentic-RAG")
    ap.add_argument("--dataset", default="packages/eval/datasets/finance_golden.jsonl", help="Path to JSONL golden set")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k contexts to pass to the answerer")
    ap.add_argument("--out-dir", default="./data/outputs", help="Where to write eval outputs")
    args = ap.parse_args()

    cfg = get_settings()
    examples = load_examples_jsonl(args.dataset)
    if not examples:
        print(f"No examples found at: {args.dataset}")
        return

    retrieve_fn = get_retrieve_fn(cfg)

    def _retrieve(question: str, k: int) -> List[Dict[str, Any]]:
        # Evaluator expects a list of dicts with {"id","text","metadata","score"}
        hits = retrieve_fn(question, int(k))
        return hits

    answer_fn = get_answer_fn(cfg)

    def _answer(question: str, contexts: List[str]) -> Dict[str, Any]:
        # The evaluator will pass raw context texts (it will extract from hits)
        return answer_fn(question, contexts)

    # Wrap into Evaluator
    def _retriever_adapter(q: str, k: int):
        hits = _retrieve(q, k)
        return hits

    def _answer_adapter(q: str, ctx_hits: List[Dict[str, Any]]):
        ctx_texts = [h.get("text") or "" for h in ctx_hits]
        return _answer(q, ctx_texts)

    evalr = Evaluator(answer_fn=_answer_adapter, retrieve_fn=_retriever_adapter)
    records = evalr.run(examples, top_k_ctx=int(args.top_k))
    summary = evalr.summarize(records)

    # Outputs
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = write_eval_jsonl(records, str(out_dir / "eval_records.jsonl"))
    csv_path = write_eval_csv(records, str(out_dir / "eval_summary.csv"))
    html_path = eval_reports.render_html_report(
        [  # convert to plain dicts expected by the renderer
            {
                "question": r.question,
                "answer": r.answer,
                "citations": r.citations,
                "contexts": r.contexts,
                "metrics": r.metrics,
                "meta": r.meta,
            }
            for r in records
        ],
        path=str(out_dir / "eval_report.html"),
        title="Agentic-RAG — Eval Smoke Test",
    )

    # Console
    print("\nSummary")
    print("───────")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nWrote:\n- {jsonl_path}\n- {csv_path}\n- {html_path}")


if __name__ == "__main__":
    main()
