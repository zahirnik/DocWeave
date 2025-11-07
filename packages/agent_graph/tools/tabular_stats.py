# packages/agent_graph/tools/tabular_stats.py
"""
Tabular stats tool — tiny, explicit helpers to run common finance analytics on CSV/XLSX.

What this module provides
-------------------------
- load_tabular(path) -> pandas.DataFrame
- infer_schema(df) -> {"date_col": ..., "value_col": ..., "group_cols": [...]}
- compute_timeseries_ops(df, date_col, value_col, group_cols=[]) -> dict
    Computes:
      • basic_describe (count/mean/std/min/max)
      • resample_Q (quarterly sum if high-frequency)
      • QoQ % change
      • YoY % change
      • rolling_mean(4)
- try_run_tabular_ops(query: str) -> list[{"name":"tabular_stats","result": {...}}]
    Heuristic parser that looks for a file path in the user query and, if found,
    loads and analyzes it. Returns an empty list if nothing actionable is found.

- summarize_csv(path: str, *, nrows: int = 200_000) -> dict   # [ADDED]
    Lightweight structural + numeric summary for a CSV/XLS(X), JSON-serialisable.

Design goals
------------
- Tutorial-clear: minimal pandas, zero hidden magic.
- Defensive: trims big frames; handles missing columns gracefully.
- Side-effect free (no files written). Charting is handled in charting.py.

Assumptions
-----------
- CSV/XLSX contains either:
    * columns: ["date", "<metric>"] or ["date", "ticker", "<metric>"]
    * or a single obvious numeric column (we pick the first).
- "date" can be ISO date, 'YYYY-MM', quarter labels, or Excel datetimes.

Usage
-----
from packages.agent_graph.tools.tabular_stats import try_run_tabular_ops, summarize_csv
out = try_run_tabular_ops("Please analyze ./data/samples/acme_quarterly.csv for YoY/QoQ.")
if out:
    print(out[0]["result"]["yoy_tail"])

info = summarize_csv("./data/samples/acme_quarterly.csv")
"""

from __future__ import annotations

import os
import re
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------
# I/O
# ---------------------------

def load_tabular(path: str, *, max_rows: int = 100_000) -> pd.DataFrame:
    """
    Load a CSV/XLSX file with very small, explicit options.
    - Parses dates if a 'date' column exists.
    - Trims to `max_rows` to keep memory/cost bounded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported tabular format: {ext}")

    # Normalize col names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Try to parse a 'date' column if present
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            # leave as-is if parsing fails
            pass

    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    return df


# ---------------------------
# Schema inference
# ---------------------------

def infer_schema(df: pd.DataFrame) -> Dict[str, object]:
    """
    Identify the date column, value column, and optional group columns.

    Heuristics:
      - date_col: prefer 'date', then columns with 'date' substring, else None
      - value_col: first numeric column that is not an obvious id/counter
      - group_cols: ['ticker'] if present, else any categorical with small cardinality (<=50)
    """
    cols = list(df.columns)

    # date
    date_col = None
    if "date" in cols:
        date_col = "date"
    else:
        for c in cols:
            if "date" in c:
                date_col = c
                break

    # value (first numeric that isn't id-like)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    id_like = {"id", "index", "row", "serial", "count"}
    value_col = None
    for c in num_cols:
        if c in id_like:
            continue
        # Skip columns that look like year/quarter index if they are integer in small range
        if c in {"year", "quarter", "month"}:
            continue
        value_col = c
        break

    # groups
    group_cols: List[str] = []
    if "ticker" in cols:
        group_cols = ["ticker"]
    else:
        # heuristically pick a small-cardinality categorical (but avoid 'date')
        for c in cols:
            if c == date_col:
                continue
            if df[c].dtype == "object":
                nunique = df[c].nunique(dropna=True)
                if 1 < nunique <= 50:
                    group_cols = [c]
                    break

    return {"date_col": date_col, "value_col": value_col, "group_cols": group_cols}


# ---------------------------
# Analytics
# ---------------------------

def _ensure_timeseries(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """If possible, sort and set a DatetimeIndex for resampling."""
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        out = df.sort_values(date_col).set_index(date_col)
        return out
    return df.copy()

def _describe_numeric(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    d = df[value_col].describe()
    return {
        "count": float(d.get("count", 0.0)),
        "mean": _nan_to_none(d.get("mean")),
        "std": _nan_to_none(d.get("std")),
        "min": _nan_to_none(d.get("min")),
        "max": _nan_to_none(d.get("max")),
    }

def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    try:
        return series.pct_change(periods=periods) * 100.0
    except Exception:
        return pd.Series([None] * len(series), index=series.index)

def _nan_to_none(x):
    try:
        if x is None:
            return None
        if isinstance(x, (float, int)) and (pd.isna(x) or (isinstance(x, float) and math.isnan(x))):
            return None
        return float(x)
    except Exception:
        return None


def compute_timeseries_ops(
    df: pd.DataFrame,
    *,
    date_col: Optional[str],
    value_col: str,
    group_cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Compute key finance-friendly aggregates on a time series (optionally per group).
    Returns a dict with small JSON-safe summaries (tails as lists of dicts).
    """
    group_cols = group_cols or []
    res: Dict[str, object] = {"value_col": value_col, "date_col": date_col, "group_cols": group_cols}

    if not value_col or value_col not in df.columns:
        raise ValueError("value_col not found in DataFrame")

    if group_cols:
        out_groups: Dict[str, object] = {}
        for gval, gdf in df.groupby(group_cols):
            key = gval if isinstance(gval, str) else "_".join(map(str, (gval if isinstance(gval, tuple) else (gval,))))
            out_groups[key] = compute_timeseries_ops(gdf.copy(), date_col=date_col, value_col=value_col, group_cols=[])
        res["groups"] = out_groups
        return res

    # No grouping: run direct ops
    res["describe"] = _describe_numeric(df, value_col)

    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        ts = _ensure_timeseries(df, date_col)

        # If higher than quarterly, resample to Q (sum) for finance reports
        try:
            q = ts[value_col].resample("Q").sum(min_count=1)
        except Exception:
            q = ts[value_col]

        # QoQ and YoY in %
        qoq = _pct_change(q, periods=1)
        yoy = _pct_change(q, periods=4)

        # Rolling mean (4 quarters)
        roll = q.rolling(window=4, min_periods=1).mean()

        def _tail_to_dict(series: pd.Series, n: int = 8):
            tail = series.dropna().tail(n)
            return [{"period": str(idx.date()), "value": _nan_to_none(val)} for idx, val in tail.items()]

        res["quarterly_tail"] = _tail_to_dict(q)
        res["qoq_tail"] = _tail_to_dict(qoq)
        res["yoy_tail"] = _tail_to_dict(yoy)
        res["rolling_mean_4_tail"] = _tail_to_dict(roll)
    else:
        # Non-dated series: just basic stats and the last few values
        vals = df[value_col].dropna().tail(8)
        res["tail"] = [{"row": int(i), "value": _nan_to_none(v)} for i, v in zip(vals.index, vals.values)]

    return res


# ---------------------------
# Heuristic "tool entry point"
# ---------------------------

_FILE_PAT = re.compile(r"(?P<path>[\w./\\-]+\.(?:csv|xlsx|xls))", re.IGNORECASE)

def _find_first_path_in_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = _FILE_PAT.search(text)
    if not m:
        return None
    return m.group("path")


def try_run_tabular_ops(query: str, *, max_rows: int = 100_000) -> List[Dict[str, object]]:
    """
    Heuristically detect a CSV/XLSX path in `query`, run analytics, and return
    a tool_result list or [] if nothing to do.

    Returned shape:
      [{"name":"tabular_stats","result": {...}}]
    """
    path = _find_first_path_in_text(query or "")
    if not path:
        return []

    try:
        df = load_tabular(path, max_rows=max_rows)
    except Exception as e:
        # Return a structured error for the caller to display
        return [{"name": "tabular_stats", "error": f"Failed to load '{path}': {e}"}]

    schema = infer_schema(df)
    date_col = schema.get("date_col")  # may be None
    value_col = schema.get("value_col")
    group_cols = schema.get("group_cols") or []

    if not value_col:
        return [{"name": "tabular_stats", "error": f"Could not infer a numeric value column in {path}"}]

    try:
        result = compute_timeseries_ops(df, date_col=date_col, value_col=value_col, group_cols=group_cols)  # type: ignore[arg-type]
    except Exception as e:
        return [{"name": "tabular_stats", "error": f"Computation failed: {e}"}]

    # Attach small provenance
    result["source_path"] = os.path.abspath(path)
    result["rows"] = int(len(df))

    return [{"name": "tabular_stats", "result": result}]


# ---------------------------
# CSV/XLSX summary for API use (NEW)
# ---------------------------

# [ADDED] small helpers used by summarize_csv
def _exists_small(path: str, max_mb: int = 256) -> Tuple[bool, Optional[str]]:
    """Check that file exists and is under a soft size cap (for quick summaries)."""
    if not os.path.exists(path):
        return False, "file not found"
    try:
        size = os.path.getsize(path)
        if size > max_mb * 1024 * 1024:
            return False, f"file too large (> {max_mb} MB)"
    except Exception as e:
        return False, f"size check failed: {e}"
    return True, None

def _dtype_name(s: pd.Series) -> str:  # [ADDED]
    try:
        return str(s.dtype)
    except Exception:
        return "unknown"

def _first_non_null_example(s: pd.Series) -> Optional[str]:  # [ADDED]
    try:
        for v in s.dropna().head(1).tolist():
            return str(v)
    except Exception:
        pass
    return None

def summarize_csv(path: str, *, nrows: int = 200_000) -> Dict[str, object]:
    """
    [ADDED] Summarise a CSV or XLS(X) quickly for UI/agent use.

    Returns (JSON-friendly):
      {
        "ok": bool,
        "path": "<abs path>",
        "rows_read": int,
        "columns": [
          {"name": "...", "dtype": "...", "non_null": int, "nulls": int, "example": "..."},
          ...
        ],
        "numeric_summary": { "<col>": {"count":..., "mean":..., "std":..., "min":..., "p25":..., "p50":..., "p75":..., "max":...}, ...},
        "memory_bytes": int,
        "note": str|None
      }
    """
    ok, err = _exists_small(path)
    if not ok:
        return {"ok": False, "path": path, "note": err}

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in {".csv", ".tsv"}:
            sep = "," if ext == ".csv" else "\t"
            df = pd.read_csv(path, sep=sep, nrows=max(1, int(nrows)), low_memory=False)
        elif ext in {".xlsx", ".xls"}:
            # pandas supports nrows for read_excel in modern versions; if not, read then head().
            try:
                df = pd.read_excel(path, nrows=max(1, int(nrows)))
            except TypeError:
                df = pd.read_excel(path).head(max(1, int(nrows)))
        else:
            return {"ok": False, "path": path, "note": f"unsupported format: {ext}"}
    except Exception as e:
        return {"ok": False, "path": path, "note": f"read failed: {e}"}

    # Normalise column names like load_tabular (for consistency).
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rows = int(len(df))
    # Column metadata
    cols_meta: List[Dict[str, object]] = []
    for name in df.columns.tolist():
        col = df[name]
        nn = int(col.notna().sum())
        na = int(col.isna().sum())
        cols_meta.append({
            "name": str(name),
            "dtype": _dtype_name(col),
            "non_null": nn,
            "nulls": na,
            "example": _first_non_null_example(col),
        })

    # Numeric summary (per-column, with robust percentiles)
    numeric_summary: Dict[str, object] = {}
    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        try:
            q = num_df.quantile([0.25, 0.5, 0.75], interpolation="linear")
        except TypeError:
            # Older pandas may not support 'interpolation' kw — fall back.
            q = num_df.quantile([0.25, 0.5, 0.75])
        for name in num_df.columns:
            s = num_df[name]
            numeric_summary[str(name)] = {
                "count": float(s.count()),
                "mean": _nan_to_none(s.mean()) if s.count() else None,
                "std": _nan_to_none(s.std()) if s.count() > 1 else None,
                "min": _nan_to_none(s.min()) if s.count() else None,
                "p25": _nan_to_none(q.loc[0.25, name]) if name in q.columns else None,
                "p50": _nan_to_none(q.loc[0.50, name]) if name in q.columns else None,
                "p75": _nan_to_none(q.loc[0.75, name]) if name in q.columns else None,
                "max": _nan_to_none(s.max()) if s.count() else None,
            }

    mem = int(df.memory_usage(deep=True).sum())

    return {
        "ok": True,
        "path": os.path.abspath(path),
        "rows_read": rows,
        "columns": cols_meta,
        "numeric_summary": numeric_summary,
        "memory_bytes": mem,
        "note": f"read first {nrows} rows" if rows == nrows else None,
    }


__all__ = [
    "load_tabular",
    "infer_schema",
    "compute_timeseries_ops",
    "try_run_tabular_ops",
    "summarize_csv",  # [ADDED]
]
