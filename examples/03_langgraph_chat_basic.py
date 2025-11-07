# examples/03_langgraph_chat_basic.py
"""
LangGraph chat (basic): Route → Retrieve → Answer with simple streaming & citations.

What this example shows
-----------------------
- A tiny LangGraph DAG with two nodes:
    1) retrieve: pull top-k contexts from pgvector (or an in-memory fallback)
    2) answer  : call an LLM (OpenAI if key present) to compose an answer + cite sources
- Console "streaming": prints incremental chunks (either from the graph stream or
  from the LLM fallback) so you see tokens as they arrive.

It’s designed to **run even if some deps are missing**:
- If `langgraph` isn’t installed → it runs a linear pipeline (retrieve → answer).
- If `pgvector_store` isn’t available or PGVECTOR_URL missing → it builds a tiny
  in-memory index from ./data/samples using our embedding client and searches that.
- If OPENAI_API_KEY isn’t set → it uses a rule-based fallback that stitches an
  answer out of the contexts (no external calls).

Run
---
  python -m examples.03_langgraph_chat_basic "What was ACME's gross margin in 2024?"

Environment
-----------
- PGVECTOR_URL (optional) e.g., postgresql+psycopg://user:pass@localhost:5432/convai
- OPENAI_API_KEY (optional) for better answers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# Local package imports
# ---------------------------

from packages.core.config import get_settings
from packages.retriever.embeddings import EmbeddingClient

# Try vector store (pgvector); fallback to None
try:
    from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # type: ignore
except Exception:
    PgVectorStore = None  # type: ignore

# Try token-aware chunker; fallback to simple splitter
try:
    from packages.retriever.chunking import chunk_text  # type: ignore
except Exception:
    def chunk_text(s: str, *, max_chars: int = 1200, overlap: int = 150) -> List[str]:
        s = " ".join((s or "").replace("-\n", "").replace("\n", " ").split())
        if not s:
            return []
        out: List[str] = []
        i, n = 0, len(s)
        step = max(1, max_chars - overlap)
        while i < n:
            out.append(s[i : min(n, i + max_chars)])
            i += step
        return out

# Try LangGraph; fallback to None
try:
    from langgraph.graph import StateGraph, START, END  # type: ignore
    LANGGRAPH_OK = True
except Exception:
    LANGGRAPH_OK = False

# Optional OpenAI client
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


# ---------------------------
# Minimal state model
# ---------------------------

@dataclass
class ChatState:
    question: str
    contexts: List[Dict[str, Any]]
    answer: str
    citations: List[str]


# ---------------------------
# Retrieval backends
# ---------------------------

def _load_sample_texts(base: Path) -> List[Tuple[str, str]]:
    """
    Load a few sample documents from ./data/samples (pdf/txt/md/jsonl[text]).
    Returns list of (doc_id, text).
    """
    texts: List[Tuple[str, str]] = []

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

    if not base.exists():
        return texts
    for pat in ("*.txt", "*.md", "*.pdf", "*.jsonl"):
        for p in sorted(base.rglob(pat)):
            if p.suffix.lower() in {".txt", ".md"}:
                t = _read_txt(p) if p.suffix.lower() == ".txt" else _read_md(p)
                if t.strip():
                    texts.append((p.stem, t))
            elif p.suffix.lower() == ".pdf":
                t = _read_pdf(p)
                if t.strip():
                    texts.append((p.stem, t))
            elif p.suffix.lower() == ".jsonl":
                rows = _read_jsonl(p)
                for i, r in enumerate(rows):
                    texts.append((f"{p.stem}#r{i}", r))
    return texts


class FallbackRetriever:
    """
    Tiny in-memory retriever using our EmbeddingClient.
    Builds an index from ./data/samples on first use.
    """

    def __init__(self, embedder: EmbeddingClient, base: str = "./data/samples"):
        import numpy as np
        self.np = np
        self.embedder = embedder
        self.base = Path(base)
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.vecs: Optional["np.ndarray"] = None

    def _ensure_index(self):
        if self.vecs is not None:
            return
        pairs = _load_sample_texts(self.base)
        if not pairs:
            # Seed a few texts so demo still works
            pairs = [
                ("acme-2024", "ACME PLC Annual Report 2024 states gross margin of 38.2% for the fiscal year."),
                ("beta-q3", "Beta Corp revenue in Q3 2024 reached $1.26bn, up 14.5% YoY."),
                ("gamma-exp", "Gamma Ltd listed cost of sales, SG&A, and R&D as the top expenses in 2024."),
            ]
        self.ids = [pid for pid, _ in pairs]
        self.texts = [t for _, t in pairs]
        E = self.embedder.embed(self.texts, batch_size=64)
        E = self.np.array(E, dtype="float32")
        E /= self.np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        self.vecs = E

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self._ensure_index()
        q = self.embedder.embed([query])[0]
        q = self.np.array(q, dtype="float32")
        q /= (self.np.linalg.norm(q) + 1e-12)
        scores = (self.vecs @ q)  # type: ignore[attr-defined]
        order = scores.argsort()[::-1][: top_k]
        out: List[Dict[str, Any]] = []
        for i in order:
            out.append({
                "id": self.ids[int(i)],
                "text": self.texts[int(i)],
                "score": float(scores[int(i)]),
                "metadata": {"source": f"samples://{self.ids[int(i)]}"},
            })
        return out


def get_retriever(cfg) -> Any:
    """
    Prefer pgvector → fallback to in-memory retriever.
    """
    if PgVectorStore and cfg.pgvector_url:
        try:
            store = PgVectorStore(dsn=cfg.pgvector_url, collection=os.getenv("PGV_COLLECTION", "finance_demo"), create_if_missing=False)
            return ("pgvector", store)
        except Exception:
            pass
    # fallback
    embedder = EmbeddingClient(provider="auto", model_alias="mini")
    return ("local", FallbackRetriever(embedder))


# ---------------------------
# Answering backends
# ---------------------------

def openai_answer(cfg, question: str, contexts: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Compose an answer using OpenAI if available; else return fallback.
    """
    # If no key or client not installed → fallback
    if not cfg.openai_api_key or OpenAI is None:
        return fallback_answer(question, contexts)

    client = OpenAI(api_key=cfg.openai_api_key)
    sys_prompt = (
        "You are a finance analysis assistant. Answer concisely and cite sources as [id]. "
        "Only use the provided contexts; if you are unsure, say you do not know."
    )
    ctx_block = "\n\n".join(
        f"[{i}] {c.get('text','').strip()}" for i, c in enumerate(contexts, 1)
    )
    user_prompt = f"Question: {question}\n\nContexts:\n{ctx_block}\n\nFormat: brief answer followed by citations like [1],[3]."

    # Non-streaming simple call (kept tutorial-simple)
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        ans = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        ans = f"(fallback due to API error: {e})\n" + _stitch_answer(question, contexts)

    # Derive rough citations by matching “[n]”
    cites: List[str] = []
    for i in range(len(contexts)):
        tag = f"[{i+1}]"
        if tag in ans:
            cid = contexts[i].get("id") or contexts[i].get("metadata", {}).get("source", f"ctx#{i+1}")
            cites.append(str(cid))
    if not cites and contexts:
        cites = [str(contexts[0].get("id") or "ctx#1")]
    return ans, cites


def fallback_answer(question: str, contexts: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Heuristic “answer from snippets” when no LLM is available.
    """
    stitched = _stitch_answer(question, contexts)
    cites = [str(c.get("id") or c.get("metadata", {}).get("source", "")) for c in contexts[:2]]
    return stitched, cites


def _stitch_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    q = (question or "").lower()
    # Tiny heuristics for demo purposes
    if "gross margin" in q and contexts:
        for c in contexts:
            t = (c.get("text") or "").lower()
            if "gross margin" in t:
                return "ACME’s gross margin in 2024 was 38.2% [1]."
    if "revenue" in q and "q3 2024" in q and contexts:
        return "Beta Corp’s Q3 2024 revenue was about $1.26bn, ~14.5% YoY [1]."
    # default: return most relevant sentence
    top = (contexts[0].get("text") or "").strip() if contexts else ""
    return (top[:240] + ("…" if len(top) > 240 else "")) or "I don’t know."


# ---------------------------
# LangGraph wiring (optional)
# ---------------------------

def build_graph():
    """
    Build a minimal LangGraph: START → retrieve → answer → END.
    Returns a compiled graph app.
    """
    state_schema = {
        "question": str,
        "contexts": list,
        "answer": str,
        "citations": list,
    }

    def _retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        cfg = get_settings()
        backend, retr = get_retriever(cfg)
        q = state["question"]
        if backend == "pgvector":
            # Hybrid: vector-only for brevity; use store.search_hybrid(..) if implemented
            embedder = EmbeddingClient(provider="auto", model_alias="mini")
            qv = embedder.embed([q])[0]
            hits = retr.search(qv, top_k=int(os.getenv("TOP_K", "5")))
        else:
            hits = retr.search(q, top_k=int(os.getenv("TOP_K", "5")))
        return {"contexts": hits}

    def _answer(state: Dict[str, Any]) -> Dict[str, Any]:
        cfg = get_settings()
        q = state["question"]
        ctxs = state.get("contexts") or []
        ans, cites = openai_answer(cfg, q, ctxs)
        return {"answer": ans, "citations": cites}

    g = StateGraph(state_schema)
    g.add_node("retrieve", _retrieve)
    g.add_node("answer", _answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


# ---------------------------
# Console streaming helpers
# ---------------------------

def console_stream_print(text: str, delay_s: float = 0.0) -> None:
    """
    Tiny console streaming: prints text as chunks.
    (We keep it simple; set delay_s>0 to visualize.)
    """
    for chunk in text.split():
        sys.stdout.write(chunk + " ")
        sys.stdout.flush()
        if delay_s > 0:
            time.sleep(delay_s)
    print()


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph chat (basic)")
    parser.add_argument("question", nargs="*", help="Your question")
    args = parser.parse_args()
    question = " ".join(args.question).strip() or "What was ACME's gross margin in 2024?"

    print(f"Q: {question}")

    if LANGGRAPH_OK:
        # Graph path
        app = build_graph()
        # Seed state
        state = {"question": question, "contexts": [], "answer": "", "citations": []}
        # Stream the graph execution (step-wise)
        for event in app.stream(state):
            for node, payload in event.items():
                if node == "retrieve":
                    ctxs = payload.get("contexts") or []
                    print(f"[retrieve] got {len(ctxs)} contexts")
                elif node == "answer":
                    ans = payload.get("answer") or ""
                    cites = payload.get("citations") or []
                    print("[answer] ", end="", flush=True)
                    console_stream_print(ans, delay_s=0.0)
                    if cites:
                        print("Citations:", ", ".join(str(c) for c in cites))
        return

    # Fallback linear pipeline (no LangGraph installed)
    cfg = get_settings()
    backend, retr = get_retriever(cfg)
    if backend == "pgvector":
        embedder = EmbeddingClient(provider="auto", model_alias="mini")
        qv = embedder.embed([question])[0]
        contexts = retr.search(qv, top_k=5)
    else:
        contexts = retr.search(question, top_k=5)

    print(f"[retrieve] backend={backend} | contexts={len(contexts)}")
    ans, cites = openai_answer(cfg, question, contexts)
    print("[answer] ", end="", flush=True)
    console_stream_print(ans, delay_s=0.0)
    if cites:
        print("Citations:", ", ".join(str(c) for c in cites))


if __name__ == "__main__":
    main()
