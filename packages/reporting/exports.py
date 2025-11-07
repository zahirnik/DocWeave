# packages/reporting/exports.py
"""
Exports — JSON/CSV writers for deterministic scorecards & benchmarks
===================================================================

What this module provides
-------------------------
- scorecard_to_json_dict(scorecard)          → plain dict (safe to json.dump)
- write_scorecard_json(scorecard, path)      → write one JSON file
- write_scorecard_csv(scorecard, dir_path)   → write two CSVs (capitals.csv, metrics.csv)
- write_benchmark_csv(benchmark, dir_path)   → write peers.csv (overall + per-capital ranks)

No heavy dependencies (json/csv/os). Charts are intentionally omitted.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .scoring import Scorecard, CapitalResult, MetricResult
from .narrative_benchmark import BenchmarkResult


# -----------------------
# Scorecard → JSON
# -----------------------

def scorecard_to_json_dict(sc: Scorecard) -> Dict:
    """
    Convert a Scorecard dataclass into a JSON-serialisable dict
    with a stable, UI-friendly shape.
    """
    out = {
        "entity_key": sc.entity_key,
        "overall_score": sc.overall_score,
        "notes": dict(sc.notes or {}),
        "capitals": [],
    }
    for cap in sc.capitals:
        cap_entry = {
            "name": cap.name,
            "weight": cap.weight,
            "score": cap.score,
            "metrics": [],
        }
        for m in cap.metrics:
            cap_entry["metrics"].append(
                {
                    "key": m.key,
                    "label": m.label,
                    "weight": m.weight,
                    "score": m.score,
                    "flags": {
                        "target_present": m.flags.target_present,
                        "period_present": m.flags.period_present,
                        "evidence_linked": m.flags.evidence_linked,
                        "specificity_good": m.flags.specificity_good,
                        "metric_aligned": m.flags.metric_aligned,
                    },
                    "considered_claim_ids": [str(x) for x in (m.considered_claim_ids or [])],
                    # Detector issues can be large; include minimal view
                    "issues": [
                        {
                            "code": iss.code.value if hasattr(iss.code, "value") else str(iss.code),
                            "severity": iss.severity.value if hasattr(iss.severity, "value") else str(iss.severity),
                            "message": iss.message,
                        }
                        for iss in (m.issues or [])
                    ],
                }
            )
        out["capitals"].append(cap_entry)
    return out


def write_scorecard_json(sc: Scorecard, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = scorecard_to_json_dict(sc)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------
# Scorecard → CSVs
# -----------------------

def write_scorecard_csv(sc: Scorecard, dir_path: str | os.PathLike) -> None:
    """
    Write two CSV files into dir_path:
      - capitals.csv  : name, weight, score
      - metrics.csv   : capital, key, label, weight, score, flags...
    """
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    # capitals.csv
    with (d / "capitals.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["capital", "weight", "score"])
        for cap in sc.capitals:
            w.writerow([cap.name, f"{cap.weight:.6f}", f"{cap.score:.6f}"])

    # metrics.csv
    with (d / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "capital",
            "metric_key",
            "metric_label",
            "metric_weight",
            "metric_score",
            "target_present",
            "period_present",
            "evidence_linked",
            "specificity_good",
            "metric_aligned",
            "considered_claims_count",
        ])
        for cap in sc.capitals:
            for m in cap.metrics:
                w.writerow([
                    cap.name,
                    m.key,
                    m.label,
                    f"{m.weight:.6f}",
                    f"{m.score:.6f}",
                    int(bool(m.flags.target_present)),
                    int(bool(m.flags.period_present)),
                    int(bool(m.flags.evidence_linked)),
                    int(bool(m.flags.specificity_good)),
                    int(bool(m.flags.metric_aligned)),
                    len(m.considered_claim_ids or []),
                ])


# -----------------------
# Benchmark → CSV
# -----------------------

def write_benchmark_csv(bench: BenchmarkResult, dir_path: str | os.PathLike) -> None:
    """
    Write peers.csv with overall score, rank, coverage and per-capital ranks/scores.
    """
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    # Collect union of capital names across peers for stable columns
    caps = []
    for p in bench.peers:
        for c in p.scorecard.capitals:
            if c.name not in caps:
                caps.append(c.name)

    header = ["entity_key", "overall_score", "rank_overall", "coverage", "metrics_with_claims", "metrics_total"]
    for cap in caps:
        header += [f"{cap}_score", f"{cap}_rank"]

    with (d / "peers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for p in bench.peers:
            row = [
                p.entity_key,
                f"{p.scorecard.overall_score:.6f}",
                p.rank_overall,
                f"{p.coverage:.6f}",
                p.metric_count_with_claims,
                p.metric_count_total,
            ]
            # per-capital values (fill missing with 0)
            by_name: Dict[str, float] = {c.name: c.score for c in p.scorecard.capitals}
            for cap in caps:
                row.append(f"{by_name.get(cap, 0.0):.6f}")
                row.append(p.ranks_by_capital.get(cap, 0))
            w.writerow(row)


__all__ = [
    "scorecard_to_json_dict",
    "write_scorecard_json",
    "write_scorecard_csv",
    "write_benchmark_csv",
]
