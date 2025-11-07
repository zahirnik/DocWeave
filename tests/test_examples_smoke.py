# tests/test_examples_smoke.py
"""
Smoke-test the runnable examples.

Goal
----
Import each example module and (if it exposes a `main()` function) invoke it with
safe, offline-friendly arguments so it doesn't crash. These are *smoke* tests: we
aren’t asserting semantic outputs, only that the scripts wire up correctly and
exit cleanly under minimal conditions.

Notes
-----
- We avoid network/LLM by not setting API keys and by passing flags that keep
  examples offline (e.g., not enabling web search).
- If an example module is missing in a fresh checkout or is optional, we skip it
  rather than failing CI. This keeps the suite resilient while the repo evolves.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import pytest


# Discoverable list of example modules we expect in this repo.
# If a module isn't present, we mark the test as xfail/skip—this is a smoke suite.
EXAMPLE_MODULES = [
    "examples.00_quickstart_minimal",
    "examples.01_ingest_local_files",
    "examples.02_build_index_pgvector",
    "examples.03_langgraph_chat_basic",
    "examples.04_tabular_stats_demo",
    "examples.05_rerank_and_hybrid",
    "examples.06_eval_smoke_test",
    "examples.07_web_search_opt_in",
]


def _import_optional(modname: str):
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError:
        pytest.skip(f"{modname} not found in this checkout")
    except Exception as e:
        pytest.fail(f"Import failed for {modname}: {e}")


def _get_main(mod) -> Optional[Callable]:
    # Prefer a `main()` callable; if not present, we consider import success a pass.
    return getattr(mod, "main", None)


@pytest.fixture(autouse=True)
def offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Keep examples offline and reproducible during smoke tests.

    - Unset API keys so examples choose fallbacks.
    - Ensure data/outputs paths exist to avoid FileNotFoundError on writes.
    """
    # Scrub keys so examples pick local fallbacks
    for key in [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "BING_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    # Do not enable web search by default
    monkeypatch.delenv("USE_WEB", raising=False)

    # Point outputs to a temp dir where relevant
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common example envs
    monkeypatch.setenv("PGV_COLLECTION", "finance_demo_test")  # harmless if pgvector not configured

    # Ensure repo-style relative data paths exist
    for p in ["data/samples", "data/outputs", "packages/eval/datasets"]:
        d = Path(p)
        d.mkdir(parents=True, exist_ok=True)

    # Seed a tiny sample so ingestion examples always find *something*
    (Path("data/samples") / "smoke_note.txt").write_text(
        "ACME PLC Annual Report 2024: gross margin 38.2%.", encoding="utf-8"
    )


@pytest.mark.parametrize("modname", EXAMPLE_MODULES)
def test_example_import_and_main_runs(modname: str, capsys: pytest.CaptureFixture):
    mod = _import_optional(modname)
    main = _get_main(mod)

    # If no main() present, consider import success a pass.
    if main is None:
        return

    # Prepare safe argv overrides per example:
    argv = []  # default: run with module defaults (which are offline-friendly in this repo)

    if modname.endswith("06_eval_smoke_test"):
        # Keep defaults; the script handles missing dataset gracefully.
        argv = ["--top-k", "3", "--out-dir", "data/outputs"]
    elif modname.endswith("07_web_search_opt_in"):
        # Ensure web is not enabled (no network in CI)
        argv = ["--query", "ACME PLC 2024 gross margin", "--top-k", "2"]
        # DO NOT pass --web
    elif modname.endswith("04_tabular_stats_demo"):
        # Keep simple default; script will write a small PNG under data/outputs
        argv = []
    elif modname.endswith("02_build_index_pgvector"):
        # Works even without pgvector if store fallback exists in the script;
        # but to be safe in smoke, we just import (handled above). If main exists,
        # run with a tiny source and default store selection.
        argv = ["--source", "data/samples", "--collection", "finance_demo_test"]

    # Run main; it should not raise.
    try:
        if argv:
            main(argv)
        else:
            main()
    except SystemExit as e:
        # Some scripts may call sys.exit(0) — treat 0 as success.
        if getattr(e, "code", 0) not in (0, None):
            raise
    except Exception as e:
        # Dump captured stdout/stderr to aid debugging on CI
        out, err = capsys.readouterr()
        pytest.fail(f"{modname}.main errored: {e}\n--- stdout ---\n{out}\n--- stderr ---\n{err}")
