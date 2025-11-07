# packages/reporting/__init__.py
"""
Reporting layer (deterministic scorecards & benchmarks)
======================================================

Purpose
-------
This package sits *on top of* the Knowledge Graph and produces
auditable outputs (scores, rationales, evidence links) without LLMs.

Modules (added next, one by one)
--------------------------------
- frameworks/
    • six_capitals.yaml       # weights/rules template
    • sector_profiles.yaml    # optional sector/materiality tweaks
- detectors.py                # static checks over KG (missing target/period/metric, vagueness…)
- scoring.py                  # pure functions: (detectors + YAML) → scores + rationales
- narrative_benchmark.py      # simple peer/criteria coverage & specificity comparisons
- exports.py                  # JSON/CSV (charts optional)

Runtime behavior
----------------
This __init__ avoids importing submodules so the package can be installed
incrementally (no ImportError before files exist). Import submodules directly:

    from packages.reporting import detectors, scoring

Type checking
-------------
Editors/type-checkers can still resolve symbols via TYPE_CHECKING imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

if TYPE_CHECKING:
    # Hints for IDEs without importing at runtime
    from . import detectors as detectors
    from . import scoring as scoring
    from . import narrative_benchmark as narrative_benchmark
    from . import exports as exports
