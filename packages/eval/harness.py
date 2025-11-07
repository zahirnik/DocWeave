# packages/eval/harness.py
"""
Minimal eval harness — smoke-test your Agentic RAG with small, transparent metrics.

What this module provides
-------------------------
- Example dataclass: a single Q/A item (question + gold references).
- load_examples_jsonl(path) -> list[Example]
- Evaluator:
    eval = Evaluator(answer_fn, retrieve_fn)
    results = eval.run(examples, top_k_ctx=6)
    summary = eval.summarize(results)

Answer & retrieve callables
---------------------------
- answer_fn(question: str, contexts: list[str]) -> dict
    Returns: {"answer": str, "citations": list[str]|None}
    You wire this to your chat/graph node that produces a final answer.

- retrieve_fn(question: str, top_k: int) -> list[dict]
    Returns a list of {"id": str, "text": str, "metadata": {...}, "score": float}
    You wire this to your retriever/search pipeline (hybrid or pure vector).

Metrics (lexical, dependency-free)
----------------------------------
For each example:
- exact_match           : bool                          # strict equality (lowercased, normalised)
- jaccard_q2ctx         : float 0..1                    # overlap between question tokens and merged contexts
- ref_coverage          : float 0..1                    # % of reference sentences found in contexts
- ans_supported_ratio   : float 0..1                    # % of answer sentences supported by contexts or refs
- ans_ref_jaccard       : float 0..1                    # overlap between answer and merged references
- ctx_precision_like    : float 0..1                    # % of contexts that are actually relevant to question or refs
- answer_len_tokens     : int                           # rough size (after normalisation)
- num_contexts          : int
- elapsed_ms            : int                           # measured end-to-end for this example

Design goals
------------
- Tutorial-clear; **no heavy deps** (just stdlib).
- Deterministic, small, and easy to run in CI.
- Works even when you don’t have embeddings available.
- If you later install RAGAS/TruLens, you can wrap those here similarly.

Usage
-----
from packages.eval.harness import load_examples_jsonl, Evaluator

def my_retrieve(q, k):
    # call your retriever/search
    return [{"id":"doc1#p3","text":"...","metadata":{}, "score":0.9}, ...]

def my_answer(q, ctxs):
    # call your graph's "answer" node
    return {"answer": "ACME's 2024 gross margin was 38%.", "citations": ["doc1#p3"]}

examples = load_examples_jsonl("./packages/eval/datasets/finance_golden.jsonl")
eval = Evaluator(my_answer, my_retrieve)
results = eval.run(examples, top_k_ctx=6)
print(eval.summarize(results))
"""

from __future__ import annotations

import json
import math
import os
import time
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# ---------------------------
# Data model
# ---------------------------

@dataclass
class Example:
    question: str
    references: List[str] = field(default_factory=list)   # gold supporting snippets (optional)
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Example":
        return Example(
            question=str(d.get("question") or d.get("query") or ""),
            references=[str(x) for x in (d.get("references") or d.get("answers") or [])],
            meta=dict(d.get("meta") or {}),
        )

@dataclass
class EvalRecord:
    question: str
    answer: str
    citations: List[str]
    contexts: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    meta: Dict[str, Any]


# ---------------------------
# Tiny normalisation helpers
# ---------------------------

_PUNCT = str.maketrans({c: " " for c in r",.;:!?()[]{}<>“”‘’\"'`/\\|@#$%^&*_+=~"})

def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.translate(_PUNCT)
    s = " ".join(s.split())
    return s

def _tokens(s: str) -> List[str]:
    return [t for t in _norm_text(s).split() if t]

def _sentences(s: str) -> List[str]:
    # naive sentence split is fine for eval smoke tests
    out: List[str] = []
    for chunk in s.replace("?", ".").replace("!", ".").split("."):
        z = chunk.strip()
        if z:
            out.append(z)
    return out

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


# ---------------------------
# Public loader
# ---------------------------

def load_examples_jsonl(path: str) -> List[Example]:
    """
    Read examples from a JSONL file with fields:
      {"question": "...", "references": ["...", "..."], "meta": {...}}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ex: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ex.append(Example.from_dict(d))
            except Exception:
                continue
    return ex


# ---------------------------
# Evaluator
# ---------------------------

AnswerFn = Callable[[str, List[str]], Dict[str, Any]]
RetrieveFn = Callable[[str, int], List[Dict[str, Any]]]

class Evaluator:
    """
    Wire in your retriever + answerer and get small, deterministic metrics.
    """

    def __init__(self, answer_fn: AnswerFn, retrieve_fn: RetrieveFn):
        self.answer_fn = answer_fn
        self.retrieve_fn = retrieve_fn

    # ---- main loop ----

    def run(
        self,
        examples: List[Example],
        *,
        top_k_ctx: int = 6,
        support_threshold: float = 0.5,      # sentence is "supported" if Jaccard >= 0.5 with any ctx/ref sentence
        ctx_relevance_threshold: float = 0.25,  # context is relevant to Q/refs if Jaccard >= 0.25
        time_budget_s: Optional[float] = None,
    ) -> List[EvalRecord]:
        out: List[EvalRecord] = []
        started = time.time()
        for i, ex in enumerate(examples):
            if time_budget_s is not None and (time.time() - started) > time_budget_s:
                break

            t0 = time.time()
            contexts = self.retrieve_fn(ex.question, int(top_k_ctx)) or []
            merged_ctx_text = " ".join([c.get("text") or "" for c in contexts])
            answer_obj = self.answer_fn(ex.question, [c.get("text") or "" for c in contexts])
            answer_text = str(answer_obj.get("answer") or "")
            citations = [str(x) for x in (answer_obj.get("citations") or [])]
            elapsed_ms = int((time.time() - t0) * 1000)

            metrics = self._compute_metrics(
                question=ex.question,
                answer=answer_text,
                refs=ex.references,
                contexts=contexts,
                support_threshold=support_threshold,
                ctx_relevance_threshold=ctx_relevance_threshold,
            )
            metrics["elapsed_ms"] = elapsed_ms

            out.append(
                EvalRecord(
                    question=ex.question,
                    answer=answer_text,
                    citations=citations,
                    contexts=contexts,
                    metrics=metrics,
                    meta=ex.meta,
                )
            )
        return out

    # ---- metrics ----

    def _compute_metrics(
        self,
        *,
        question: str,
        answer: str,
        refs: List[str],
        contexts: List[Dict[str, Any]],
        support_threshold: float,
        ctx_relevance_threshold: float,
    ) -> Dict[str, Any]:
        q_tok = _tokens(question)
        a_tok = _tokens(answer)
        r_tok = _tokens(" ".join(refs or []))
        c_tok = _tokens(" ".join([c.get("text") or "" for c in contexts]))

        # 1) Exact match baseline (strict)
        exact_match = _norm_text(answer) == _norm_text(" ".join(refs or [])) if refs else False

        # 2) How much do contexts relate to the question?
        j_q2ctx = _jaccard(q_tok, c_tok)

        # 3) Coverage of gold references by retrieved contexts (sentence-level)
        ref_sents = [s for r in refs for s in _sentences(r)]
        ctx_sents = [s for c in contexts for s in _sentences(c.get("text") or "")]
        ref_covered = 0
        for rs in ref_sents:
            rs_tok = _tokens(rs)
            if any(_jaccard(rs_tok, _tokens(cs)) >= support_threshold for cs in ctx_sents):
                ref_covered += 1
        ref_coverage = (ref_covered / len(ref_sents)) if ref_sents else (1.0 if contexts else 0.0)

        # 4) Answer sentences supported by either contexts or refs
        ans_sents = _sentences(answer)
        ans_supported = 0
        for as_ in ans_sents:
            as_tok = _tokens(as_)
            supported = any(_jaccard(as_tok, _tokens(cs)) >= support_threshold for cs in ctx_sents)
            if not supported and refs:
                supported = any(_jaccard(as_tok, _tokens(rs)) >= support_threshold for rs in ref_sents)
            if supported:
                ans_supported += 1
        ans_supported_ratio = (ans_supported / len(ans_sents)) if ans_sents else (1.0 if not answer.strip() else 0.0)

        # 5) Answer vs reference lexical similarity
        ans_ref_jaccard = _jaccard(a_tok, r_tok) if refs else 0.0

        # 6) Context “precision-like”: fraction of contexts that actually relate to Q or refs
        ctx_relevant = 0
        for cs in ctx_sents:
            score_q = _jaccard(_tokens(cs), q_tok)
            score_r = _jaccard(_tokens(cs), r_tok) if refs else 0.0
            if max(score_q, score_r) >= ctx_relevance_threshold:
                ctx_relevant += 1
        ctx_precision_like = (ctx_relevant / len(ctx_sents)) if ctx_sents else 0.0

        return {
            "exact_match": bool(exact_match),
            "jaccard_q2ctx": round(j_q2ctx, 4),
            "ref_coverage": round(ref_coverage, 4),
            "ans_supported_ratio": round(ans_supported_ratio, 4),
            "ans_ref_jaccard": round(ans_ref_jaccard, 4),
            "ctx_precision_like": round(ctx_precision_like, 4),
            "answer_len_tokens": int(len(a_tok)),
            "num_contexts": int(len(contexts)),
        }

    # ---- summarise ----

    def summarize(self, records: List[EvalRecord]) -> Dict[str, Any]:
        """
        Aggregate metrics across records. Returns a small dict with averages.
        """
        if not records:
            return {"count": 0}
        sums: Dict[str, float] = {}
        keys = list(records[0].metrics.keys())
        for r in records:
            for k in keys:
                v = r.metrics.get(k)
                if isinstance(v, (int, float)) and k != "answer_len_tokens" and k != "elapsed_ms" and k != "num_contexts":
                    sums[k] = sums.get(k, 0.0) + float(v)
        n = float(len(records))
        avg = {f"avg_{k}": round(sums.get(k, 0.0) / n, 4) for k in sums}
        avg["count"] = int(n)
        # Add basic latency + token stats
        avg["avg_elapsed_ms"] = round(sum(r.metrics.get("elapsed_ms", 0) for r in records) / n, 1)
        avg["avg_answer_len_tokens"] = round(sum(r.metrics.get("answer_len_tokens", 0) for r in records) / n, 1)
        avg["avg_num_contexts"] = round(sum(r.metrics.get("num_contexts", 0) for r in records) / n, 2)
        return avg


# ---------------------------
# File writers (JSON/CSV)
# ---------------------------

def write_jsonl(records: List[EvalRecord], path: str) -> str:
    """
    Write records to JSONL (answer, citations, metrics). Returns the path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path) or "."), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            row = {
                "question": r.question,
                "answer": r.answer,
                "citations": r.citations,
                "contexts": r.contexts,   # beware: may be large (only for small golden sets)
                "metrics": r.metrics,
                "meta": r.meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_csv(records: List[EvalRecord], path: str) -> str:
    """
    Write a compact CSV of metrics only (one row per example).
    """
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path) or "."), exist_ok=True)
    # Collect all metric keys
    metric_keys = set()
    for r in records:
        metric_keys.update(r.metrics.keys())
    metric_keys = sorted(metric_keys)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["question", "answer_len_tokens"] + metric_keys
        w.writerow(header)
        for r in records:
            row = [r.question, r.metrics.get("answer_len_tokens", 0)] + [r.metrics.get(k, "") for k in metric_keys]
            w.writerow(row)
    return path


__all__ = [
    "Example",
    "EvalRecord",
    "Evaluator",
    "load_examples_jsonl",
    "write_jsonl",
    "write_csv",
]
