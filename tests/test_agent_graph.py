# tests/test_agent_graph.py
"""
Agent-graph smoke tests (offline, dependency-light).

What we check
-------------
- The graph module can be imported.
- A builder function exists (we try common names).
- The compiled graph exposes an `.invoke()`-like API and returns a dict-ish state.
- A single question run completes without raising and yields some final text field.

Notes
-----
- We do NOT require live LLMs or web search. The repo's graph should pick local
  fallbacks or no-op tools when keys are absent.
- If LangGraph or the graph builder is unavailable (e.g., mid-refactor), we skip.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pytest


def _import_graph_module():
    try:
        mod = __import__("packages.agent_graph.graph", fromlist=["*"])
        return mod
    except ModuleNotFoundError:
        pytest.skip("packages.agent_graph.graph not found in this checkout")
    except Exception as e:
        pytest.fail(f"Import failed for packages.agent_graph.graph: {e}")


def _get_builder(mod) -> Optional[Callable[..., Any]]:
    """
    Try a few conventional factory names to obtain a compiled graph / app.
    The returned callable should accept **kwargs and produce an object with
    `.invoke(state)` or similar.
    """
    for name in (
        "build_graph",          # most common in this repo
        "create_graph",
        "make_graph",
        "build_basic_graph",
        "get_graph",
        "graph",                # sometimes a ready-made compiled graph
    ):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
        # allow a prebuilt object named 'graph' that has .invoke
        if fn is not None and hasattr(fn, "invoke"):
            return lambda **_: fn
    return None


def _has_invoke(app: Any) -> bool:
    return hasattr(app, "invoke") and callable(getattr(app, "invoke"))


def _run_once(app: Any, question: str) -> Dict[str, Any]:
    """
    Minimal state contract used by most LangGraph apps in this repo:
      - input: {"question": str, "history": list} (history optional)
      - output: dict-like state containing one of {"answer","output","final","result"}
    """
    state_in = {"question": question, "history": []}
    try:
        out = app.invoke(state_in)  # LangGraph CompiledGraph API
    except AttributeError:
        # Some apps expose `.run` or are directly callable
        run = getattr(app, "run", None)
        if callable(run):
            out = run(state_in)
        elif callable(app):
            out = app(state_in)
        else:
            raise
    if not isinstance(out, dict):
        # Some graphs return (state, events) — keep the first part
        if isinstance(out, (list, tuple)) and out:
            out = out[0]
    assert isinstance(out, dict), "graph did not return a dict-like state"
    return out


@pytest.mark.timeout(30)
def test_graph_build_and_invoke_basic():
    mod = _import_graph_module()
    builder = _get_builder(mod)
    if builder is None:
        pytest.skip("No graph builder found (expected build_graph/create_graph/etc.)")

    # Build with offline-friendly flags (the builder should tolerate unknown kwargs)
    try:
        app = builder(
            use_web=False,
            use_reranker=False,
            use_pgvector=False,
            model_alias="mini",  # encourage local embedding/LLM fallbacks
        )
    except TypeError:
        # Builder might not accept kwargs; try plain call
        app = builder()

    assert _has_invoke(app) or callable(getattr(app, "run", None)) or callable(app), "Graph app has no invoke/run"

    # One short run
    out = _run_once(app, "What was ACME's gross margin in 2024?")
    # Accept any of these keys as the final answer field
    keys = [k for k in ("answer", "output", "final", "result") if k in out and isinstance(out[k], str)]
    assert keys, f"Expected one of answer/output/final/result in state, got keys: {list(out.keys())}"
    assert any(len(out[k].strip()) > 0 for k in keys), "Empty answer text"

    # If citations/contexts are included, check shape lightly
    if "contexts" in out and out["contexts"] is not None:
        assert isinstance(out["contexts"], (list, tuple))
        if out["contexts"]:
            assert isinstance(out["contexts"][0], (str, dict)), "contexts should be strings or dicts with text/metadata"

    # Optional budget/trace presence (non-fatal if absent)
    for maybe in ("trace", "budget", "used_tokens"):
        if maybe in out:
            assert out[maybe] is not None
