# examples/07_web_search_opt_in.py
"""
Web search (opt-in) with Tavily — budget shown in output.

What this example shows
-----------------------
- A tiny, **opt-in** web search step you can plug into your Agentic-RAG flow.
- If `--web` (or USE_WEB=1) and TAVILY_API_KEY are present → runs a live search.
- Otherwise, it skips network calls and explains how to enable them.
- Optional LLM summarisation over the found snippets (OpenAI if key present).

Why opt-in?
-----------
For finance workflows you often want reproducible runs that don’t hit the network
unless explicitly requested. This script makes that toggle obvious.

Run
---
  # dry run (no web)
  python -m examples.07_web_search_opt_in --query "ACME PLC 2024 annual report gross margin"

  # live web (requires TAVILY_API_KEY)
  USE_WEB=1 python -m examples.07_web_search_opt_in --query "ACME PLC 2024 annual report gross margin" --top-k 6

  # live web + LLM summary (requires OPENAI_API_KEY)
  USE_WEB=1 python -m examples.07_web_search_opt_in --query "Beta Corp Q3 2024 revenue" --summarise

Environment
-----------
- TAVILY_API_KEY   (required for live web search)
- OPENAI_API_KEY   (optional; for summarising snippets)
- USE_WEB=1        (or pass --web)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from packages.core.config import get_settings

# Optional deps (we keep the script runnable without them)
try:
    from tavily import TavilyClient  # pip install tavily-python
except Exception:
    TavilyClient = None  # type: ignore

try:
    from openai import OpenAI  # pip install openai
except Exception:
    OpenAI = None  # type: ignore


# ---------------------------
# Helpers
# ---------------------------

def _enabled(flag_cli: bool) -> bool:
    return flag_cli or os.getenv("USE_WEB", "").strip() in {"1", "true", "yes", "on"}

def _wrap(s: str, width: int = 92) -> str:
    return "\n".join(textwrap.wrap(s, width=width)) if s else s

def _print_result(idx: int, item: Dict[str, Any]) -> None:
    title = item.get("title") or "(no title)"
    url = item.get("url") or item.get("source") or ""
    snippet = item.get("content") or item.get("snippet") or item.get("text") or ""
    print(f"{idx:>2}. {title}")
    if url:
        print(f"    {url}")
    if snippet:
        print("    " + _wrap(snippet, 96).replace("\n", "\n    "))

def _estimate_tokens(text: str) -> int:
    # Very rough: ~4 chars/token heuristic
    return max(1, int(len(text) / 4))


# ---------------------------
# Web search (Tavily, with graceful fallback)
# ---------------------------

def run_tavily_search(query: str, top_k: int, api_key: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (results, budget). Each result has {title, url, content}.
    """
    if TavilyClient is None:
        raise RuntimeError("tavily-python not installed. `pip install tavily-python`")

    client = TavilyClient(api_key=api_key)
    # Tavily returns a dict; we standardise to a small list of dicts
    try:
        resp = client.search(
            query=query,
            max_results=int(top_k),
            include_images=False,
            include_answer=False,
            include_raw_content=False,
        )
        items = resp.get("results", []) if isinstance(resp, dict) else (resp or [])
        out: List[Dict[str, Any]] = []
        for it in items:
            out.append({
                "title": it.get("title"),
                "url": it.get("url"),
                "content": it.get("content") or it.get("snippet") or "",
            })
        budget = {"queries": 1, "results": len(out)}
        return out, budget
    except Exception as e:
        raise RuntimeError(f"Tavily search failed: {e}") from e


# ---------------------------
# Optional LLM summary over snippets
# ---------------------------

def summarise_with_openai(question: str, results: List[Dict[str, Any]], api_key: str) -> str:
    """
    Compose a short, source-aware summary across snippets.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai`")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # Prepare a compact context block (cap length)
    blocks = []
    for i, r in enumerate(results[:8], 1):
        title = r.get("title") or ""
        url = r.get("url") or ""
        txt = (r.get("content") or "").strip().replace("\n", " ")
        if len(txt) > 600:
            txt = txt[:600] + " …"
        blocks.append(f"[{i}] {title}\nURL: {url}\n{txt}")
    ctx = "\n\n".join(blocks)

    system = (
        "You are a precise research assistant for financial analysis. "
        "Synthesize a brief, source-backed answer using ONLY the provided snippets. "
        "Cite with [n] where n is the snippet index. If uncertain, say so."
    )
    user = f"Question: {question}\n\nSnippets:\n{ctx}\n\nReturn a short answer with citations like [1],[2]."

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Opt-in web search (Tavily) with optional LLM summary")
    ap.add_argument("--query", required=False, default="ACME PLC 2024 annual report gross margin", help="Query string")
    ap.add_argument("--top-k", type=int, default=6, help="Max results to fetch")
    ap.add_argument("--web", action="store_true", help="Enable live web search (or set USE_WEB=1)")
    ap.add_argument("--summarise", action="store_true", help="Summarise results with OpenAI if key present")
    args = ap.parse_args()

    cfg = get_settings()
    do_web = _enabled(args.web)

    print(f"Query     : {args.query}")
    print(f"Opt-in web: {'ON' if do_web else 'OFF'}")

    if not do_web:
        print("\n(Web disabled) To enable live web search, pass --web or set USE_WEB=1, "
              "and export TAVILY_API_KEY.\n")
        print("Example:")
        print("  USE_WEB=1 TAVILY_API_KEY=tvly_... python -m examples.07_web_search_opt_in --query \"...\"")
        return

    if not cfg.tavily_api_key:
        print("ERROR: TAVILY_API_KEY not configured. Export it and try again.")
        return

    # Run the search
    try:
        results, budget = run_tavily_search(args.query, args.top_k, cfg.tavily_api_key)
    except Exception as e:
        print(f"Search failed: {e}")
        return

    print(f"\nResults ({len(results)}):")
    print("───────────────")
    for i, r in enumerate(results, 1):
        _print_result(i, r)

    # Budget / token estimates
    joined_text = "\n".join((r.get("content") or "") for r in results)
    approx_tokens = _estimate_tokens(joined_text)
    print("\nBudget")
    print("──────")
    print(f"Queries: {budget.get('queries', 1)}")
    print(f"Results: {budget.get('results', len(results))}")
    print(f"Approx tokens in snippets (very rough): ~{approx_tokens}")

    # Optional summary with OpenAI
    if args.summarise:
        if not cfg.openai_api_key:
            print("\n(summarise) Skipped: OPENAI_API_KEY not set.")
            return
        try:
            print("\nSummary")
            print("───────")
            summary = summarise_with_openai(args.query, results, cfg.openai_api_key)
            print(_wrap(summary))
        except Exception as e:
            print(f"(summarise) Failed due to API error: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
