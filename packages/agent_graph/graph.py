# packages/agent_graph/graph.py
"""
Agentic RAG graph (LangGraph-style) — route → retrieve → (optional analyze/tools) → answer.

What this module provides
-------------------------
- FinanceAgentGraph: tiny, explicit wrapper that builds and runs a LangGraph DAG for
  finance-data Q&A with hybrid retrieval (vector+BM25), optional analytics tools, and
  grounded answering.

Graph (high level)
------------------
  ┌─────────┐     ┌───────────┐     ┌──────────────┐     ┌─────────┐
  │  route  ├───▶ │ retrieve  ├───▶ │ analyze_opt? ├───▶ │ answer  │
  └─────────┘     └───────────┘     └──────────────┘     └─────────┘
         │                │                  │
         │                │                  └─▶(skipped if not needed)
         └────────────────┴──────────────────────────────────────────▶ (short-circuit if no retrieval)

State (dict) schema (simple & typed-ish)
----------------------------------------
{
  "tenant_id": "t0",
  "collection": "acme_finance",
  "query": "How did gross margin change in Q2 2024?",
  "filters": {"year": 2024},
  "messages": [ {"role":"user","content":"..."} ],
  "contexts": [ {"id":"...","text":"...","metadata":{...},"score":0.87, "source": {...}}, ... ],
  "tool_results": [ {"name":"tabular_stats","result": {...}}, {"name":"chart","path": "..."} ],
  "budget": {"max_tokens": 2000},
  "answer": "final text with citations",
}

Dependencies
------------
- `langgraph`     → pip install langgraph
- `openai`        → pip install openai
- packages.retriever.search.HybridSearcher
- packages.agent_graph.tools.tabular_stats
- packages.agent_graph.tools.charting
- packages.core.config.get_settings
- packages.core.logging.get_logger
"""

from __future__ import annotations

import os  # [PATCH] for env fallback to OPENAI_API_KEY
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from packages.core.logging import get_logger
from packages.core.config import get_settings
from packages.retriever.search import HybridSearcher

log = get_logger(__name__)

# Optional deps guarded at call sites
try:  # pragma: no cover
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as _e:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = None         # type: ignore

# ---------------------------
# Small helpers
# ---------------------------

def _require_langgraph():
    if StateGraph is None or END is None:
        raise RuntimeError(
            "LangGraph is required for FinanceAgentGraph. Install via:\n"
            "  pip install langgraph"
        )

def _require_openai_client():
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK required. Install with: pip install openai") from e


def _ensure_state_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure minimal keys exist in state; avoid KeyErrors in nodes."""
    s = dict(state or {})
    s.setdefault("messages", [])
    s.setdefault("filters", None)
    s.setdefault("contexts", [])
    s.setdefault("tool_results", [])
    s.setdefault("budget", {"max_tokens": 2000})
    return s


# ---------------------------
# Router (simple rules)
# ---------------------------

_ANALYTICS_HINTS = {
    "csv", "xlsx", "table", "tabular", "chart", "plot", "figure", "regression",
    "yoy", "qoq", "trend", "moving average", "anomaly", "aggregate", "sum", "mean", "std",
    "median", "quantile", "histogram", "bar", "line", "box", "area",
}

def _needs_analytics(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in _ANALYTICS_HINTS)


# ---------------------------
# Node implementations (pure)
# ---------------------------

def node_route(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide whether we need analytics tools in addition to retrieval+answer.

    Output:
      {"route": "qa_only" | "qa_plus_analytics"}
    """
    q = state.get("query") or ""
    route = "qa_plus_analytics" if _needs_analytics(q) else "qa_only"
    out = dict(state)
    out["route"] = route
    log.info("route: %s", route)
    return out


def node_retrieve(searcher: HybridSearcher) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Capture a HybridSearcher in a closure to create a pure node function.
    """
    def _inner(state: Dict[str, Any]) -> Dict[str, Any]:
        s = _ensure_state_defaults(state)
        collection = s.get("collection")
        if not collection:
            raise ValueError("state.collection is required")
        query = s.get("query") or ""
        filters = s.get("filters") or None
        hits = searcher.search(collection=collection, query=query, top_k=12, filters=filters)
        out = dict(s)
        out["contexts"] = hits
        log.info("retrieve: %d hits", len(hits))
        return out
    return _inner


def _format_contexts_for_prompt(hits: List[Dict[str, Any]]) -> str:
    """
    Turn retrieved chunks into a compact citations block for the LLM prompt.
    """
    lines = []
    for i, h in enumerate(hits[:12], start=1):
        meta = h.get("metadata") or {}
        src = meta.get("source") or meta.get("filename") or meta.get("url") or ""
        lines.append(f"[{i}] {src} :: {h.get('text','')[:500]}")
    return "\n".join(lines)


def node_analyze_optional(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the route indicates analytics, call small tools (tabular stats / charting).
    Tools are **optional** — failures are logged and ignored so the graph still answers.
    """
    s = _ensure_state_defaults(state)
    if s.get("route") != "qa_plus_analytics":
        return s

    try:
        from packages.agent_graph.tools.tabular_stats import try_run_tabular_ops  # tiny helper
        from packages.agent_graph.tools.charting import try_make_quick_chart
    except Exception as e:  # pragma: no cover
        log.info("analytics tools unavailable (%s); skipping analytics.", e)
        return s

    query = s.get("query") or ""
    tool_results: List[Dict[str, Any]] = []

    # 1) Try tabular ops
    try:
        res = try_run_tabular_ops(query=query)  # returns [] or [{"name":"tabular_stats","result":{...}}]
        if res:
            tool_results.extend(res)
    except Exception as e:
        log.info("tabular ops failed: %s", e)

    # 2) Try a simple chart (PNG saved under ./data/outputs)
    try:
        chart = try_make_quick_chart(query=query, max_points=500)
        if chart:
            tool_results.append({"name": "chart", **chart})
    except Exception as e:
        log.info("charting failed: %s", e)

    out = dict(s)
    out["tool_results"] = tool_results
    log.info("analyze_opt: %d tool results", len(tool_results))
    return out


def node_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build grounded prompt and call the LLM once to produce a final answer text.
    """
    s = _ensure_state_defaults(state)
    OpenAI = _require_openai_client()
    st = get_settings()

    # --- Build prompt (system + user) ---
    sys = (
        "You are a precise finance analysis assistant. "
        "You MUST:\n"
        " - Ground answers strictly in the provided CONTEXT snippets (with indices like [1], [2]).\n"
        " - If you compute metrics from tool results, show a brief reproducible calculation.\n"
        " - Use British English and concise, verifiable language.\n"
        " - If unsure or missing data, say so and suggest the exact file/section needed.\n"
    )
    ctx = _format_contexts_for_prompt(s.get("contexts") or [])
    tools_block = ""
    tr = s.get("tool_results") or []
    if tr:
        tools_block = "TOOL RESULTS:\n" + "\n".join(
            f"- {it.get('name')}: {str({k:v for k,v in it.items() if k!='name'})[:500]}" for it in tr
        )

    user_q = s.get("query") or ""
    user_msg = (
        f"QUESTION:\n{user_q}\n\n"
        f"CONTEXT (cite with [index] where used; quote minimally):\n{ctx or '(none)'}\n\n"
        f"{tools_block}\n"
        "Answer with a short first paragraph, then 2–5 bullet points with specifics, each with citations like [2], [5]."
    )

    # --- Call model ---
    client = OpenAI(api_key=(getattr(st, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")))  # [PATCH]
    model = getattr(st, "chat_model", None) or "gpt-4o-mini"  # [PATCH]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                *[m for m in (s.get("messages") or []) if m.get("role") in ("system","user","assistant")],
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=int((s.get("budget") or {}).get("max_tokens", 2000)),
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

    out = dict(s)
    out["answer"] = text
    return out


# ---------------------------
# Graph wrapper
# ---------------------------

@dataclass
class FinanceAgentGraph:
    """
    Simple façade to compile and run the agent graph.
    """
    searcher: Optional[HybridSearcher] = None

    def __post_init__(self):
        _require_langgraph()
        self.searcher = self.searcher or HybridSearcher.from_env()
        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(dict)  # state is a plain dictionary

        # Register nodes
        g.add_node("route", node_route)
        g.add_node("retrieve", node_retrieve(self.searcher))  # closure captures searcher
        g.add_node("analyze_opt", node_analyze_optional)
        g.add_node("answer", node_answer)

        # Edges
        g.set_entry_point("route")
        g.add_edge("route", "retrieve")
        # Conditional: decide to branch to analyze or straight to answer
        def _branch(state: Dict[str, Any]) -> str:
            return "analyze_opt" if (state.get("route") == "qa_plus_analytics") else "answer"
        g.add_conditional_edges("retrieve", _branch, {"analyze_opt": "analyze_opt", "answer": "answer"})
        g.add_edge("analyze_opt", "answer")
        g.add_edge("answer", END)

        return g.compile()

    # ----- Public run helpers -----

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the whole graph synchronously and return the final state."""
        s = _ensure_state_defaults(state)
        out = self._graph.invoke(input=s)  # type: ignore[attr-defined]
        return out

    def stream(self, state: Dict[str, Any]):
        """
        Stream node-by-node events. Yields (event_type, payload) tuples.
        (Tutorial-clear event stream; not token streaming.)
        """
        s = _ensure_state_defaults(state)
        for ev in self._graph.stream(input=s, stream_mode="values"):  # type: ignore[attr-defined]
            for node, payload in ev.items():
                yield ("node", {"node": node, "state": payload})
        yield ("end", None)


# ---------------------------
# Exported helper for FastAPI route
# ---------------------------

def get_chat_graph(searcher: Optional[HybridSearcher] = None):  # [ADDED]
    """
    Small exported builder used by FastAPI routes.

    Returns:
      FinanceAgentGraph instance ready to .invoke(...) or .stream(...).
    """
    return FinanceAgentGraph(searcher=searcher)  # [ADDED]


__all__ = [
    "FinanceAgentGraph",
    "get_chat_graph",        # [ADDED]
    "node_route",
    "node_retrieve",
    "node_analyze_optional",
    "node_answer",
]
