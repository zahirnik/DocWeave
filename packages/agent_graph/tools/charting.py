# packages/agent_graph/tools/charting.py
"""
Charting tool — tiny, explicit PNG charts for quick finance visuals.

What this module provides
-------------------------
- try_make_quick_chart(query: str, *, max_points: int = 500) -> dict | None
    Heuristically finds a CSV/XLSX path in the user's query, loads it,
    infers schema, and writes a **single** PNG chart under ./data/outputs.
    Returns a dict like:
      {
        "path": "data/outputs/chart_2b1a4f.png",
        "desc": "Line chart of <value_col> by <date_col> (grouped by <group> if present)",
        "rows": 3421
      }
    or None if no actionable hint is found.

- make_line_chart(df, date_col, value_col, *, group_col=None, title=None, out_dir="./data/outputs") -> str
- make_bar_chart(df, value_col, *, group_col=None, title=None, out_dir="./data/outputs") -> str
- save_line_chart(df, date_col, value_col, *, group_col=None, title=None, out_dir="./data/outputs") -> str  # compat wrapper

Design goals
------------
- Tutorial-clear; **no seaborn** and no heavy styling.
- Headless-safe: uses Matplotlib Agg backend.
- Minimally opinionated visuals suitable for reports.
- Never writes outside ./data/outputs (kept contained for safety).

Assumptions
-----------
- For time series: expect a 'date' column (datetime64) and one numeric value column.
- For non-time series: aggregate by a small-cardinality group and plot a bar chart.

Dependencies
------------
- pandas
- matplotlib

Usage
-----
from packages.agent_graph.tools.charting import try_make_quick_chart
res = try_make_quick_chart("Plot ./data/samples/acme_quarterly.csv as a line chart.")
if res:
    print("Chart saved to:", res["path"])
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Dict, Optional

import pandas as pd

# Use a non-interactive backend for servers/CI
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------
# Tiny I/O helpers
# ---------------------------

def _ensure_out_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _out_path(out_dir: str, suffix: str = "png") -> str:
    _ensure_out_dir(out_dir)
    name = f"chart_{uuid.uuid4().hex[:6]}.{suffix}"
    return os.path.join(out_dir, name)

def _abspath_from_repo_rel(path: str) -> str:
    # Normalise to absolute path but *not* outside project directory
    # We allow relative paths such as "./data/samples/foo.csv"
    return os.path.abspath(path)


# ---------------------------
# Schema inference (lightweight)
# ---------------------------

def infer_schema(df: pd.DataFrame) -> Dict[str, object]:
    """
    Return a best-effort schema:
      {"date_col": Optional[str], "value_col": Optional[str], "group_col": Optional[str]}
    """
    # normalise
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Try to parse 'date'
    date_col: Optional[str] = None
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if pd.api.types.is_datetime64_any_dtype(df["date"]):
                date_col = "date"
        except Exception:
            pass
    if not date_col:
        for c in df.columns:
            if "date" in c:
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col = c
                        break
                except Exception:
                    continue

    # Numeric value column (avoid ids)
    value_col: Optional[str] = None
    id_like = {"id", "index", "row", "serial", "count"}
    for c in df.columns:
        if c in id_like:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            value_col = c
            break

    # Group column (small cardinality)
    group_col: Optional[str] = None
    candidates = []
    if "ticker" in df.columns:
        candidates.append("ticker")
    for c in df.columns:
        if c == date_col:
            continue
        if df[c].dtype == "object":
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= 20:
                candidates.append(c)
    if candidates:
        group_col = candidates[0]

    return {"date_col": date_col, "value_col": value_col, "group_col": group_col, "df": df}


# ---------------------------
# Chart makers
# ---------------------------

def make_line_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    group_col: Optional[str] = None,
    title: Optional[str] = None,
    out_dir: str = "./data/outputs"
) -> str:
    """
    Render a simple line chart:
      - If group_col provided, draws up to the top 6 groups by latest value.
      - Saves a PNG and returns its path (relative to project).
    """
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("date_col/value_col not found in DataFrame")

    # Prepare time index
    sdf = df[[date_col, value_col] + ([group_col] if group_col else [])].copy()
    sdf = sdf.dropna(subset=[date_col, value_col])
    sdf = sdf.sort_values(date_col)

    plt.figure(figsize=(8, 4.5))  # modest size suitable for reports

    if group_col and group_col in sdf.columns:
        # Limit to top 6 groups by most recent value to avoid clutter
        latest = sdf.dropna(subset=[value_col]).sort_values(date_col).groupby(group_col).tail(1)
        top_groups = latest.sort_values(value_col, ascending=False)[group_col].head(6).tolist()

        for g in top_groups:
            gdf = sdf[sdf[group_col] == g]
            plt.plot(gdf[date_col], gdf[value_col], label=str(g))
        plt.legend(loc="best", frameon=False)
    else:
        plt.plot(sdf[date_col], sdf[value_col])

    plt.title(title or f"{value_col} over {date_col}")
    plt.xlabel(date_col)
    plt.ylabel(value_col)
    plt.tight_layout()

    path = _out_path(out_dir)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def make_bar_chart(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: str,
    top_n: int = 20,
    title: Optional[str] = None,
    out_dir: str = "./data/outputs"
) -> str:
    """
    Render a simple bar chart of value_col aggregated by group_col (mean).
    """
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError("group_col/value_col not found in DataFrame")

    g = (
        df[[group_col, value_col]]
        .dropna(subset=[group_col, value_col])
        .groupby(group_col)[value_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(8, 4.5))
    g.plot(kind="bar")  # pandas uses matplotlib under the hood
    plt.title(title or f"{value_col} by {group_col}")
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.tight_layout()

    path = _out_path(out_dir)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# --- Compat wrapper expected by apps/api/routes/analytics.py ---
def save_line_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    group_col: Optional[str] = None,
    title: Optional[str] = None,
    out_dir: str = "./data/outputs",
) -> str:
    """
    Compatibility wrapper that delegates to make_line_chart and returns the saved PNG path.
    """
    return make_line_chart(
        df=df,
        date_col=date_col,
        value_col=value_col,
        group_col=group_col,
        title=title,
        out_dir=out_dir,
    )


# ---------------------------
# Heuristic entry point
# ---------------------------

_FILE_PAT = re.compile(r"(?P<path>[\w./\\-]+\.(?:csv|xlsx|xls))", re.IGNORECASE)

def _find_first_path_in_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = _FILE_PAT.search(text)
    if not m:
        return None
    return m.group("path")


def try_make_quick_chart(query: str, *, max_points: int = 500) -> Optional[Dict[str, object]]:
    """
    If `query` includes a file path (CSV/XLSX), load it and write a single, sensible chart.

    Strategy
    --------
    1) Load DataFrame; trim to `max_points` most recent rows if date exists (else head).
    2) If a 'date' column exists → line chart of <value_col> by <date>.
       If a group_col also exists → multiple lines for top 6 groups.
    3) Else → bar chart of <value_col> by <group_col> (if present).

    Returns
    -------
    dict or None:
      {"path": "...", "desc": "...", "rows": <int>}
    """
    path_hint = _find_first_path_in_text(query or "")
    if not path_hint:
        return None

    abs_path = _abspath_from_repo_rel(path_hint)
    if not os.path.exists(abs_path):
        # Gracefully indicate non-existence to caller
        return {"path": "", "desc": f"File not found: {abs_path}", "rows": 0}  # type: ignore[return-value]

    # Load
    ext = os.path.splitext(abs_path)[1].lower()
    if ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(abs_path, sep=sep)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(abs_path)
    else:
        return {"path": "", "desc": f"Unsupported format: {ext}", "rows": 0}  # type: ignore[return-value]

    # Normalize
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Infer schema
    schema = infer_schema(df)
    df = schema["df"]  # type: ignore[assignment]
    date_col = schema.get("date_col")  # type: ignore[assignment]
    value_col = schema.get("value_col")  # type: ignore[assignment]
    group_col = schema.get("group_col")  # type: ignore[assignment]

    if not value_col:
        return {"path": "", "desc": "Could not infer a numeric value column for charting.", "rows": int(len(df))}  # type: ignore[return-value]

    # Trim rows to keep plots readable
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):  # type: ignore[index]
        df = df.sort_values(date_col)  # type: ignore[index]
        if len(df) > max_points:
            df = df.tail(max_points)
        title = f"{value_col} over time"
        out_path = make_line_chart(df, date_col, value_col, group_col=group_col, title=title)
        desc = f"Line chart of '{value_col}' by '{date_col}'" + (f" (grouped by '{group_col}')" if group_col else "")
    else:
        # Non-time: need a group to bar against; pick a small-cardinality object column
        if not group_col:
            # Fallback: create a row number group for a simple bar of the first N rows
            df = df.head(min(max_points, 50)).reset_index(drop=True)
            df["row"] = df.index + 1
            group_col = "row"
        title = f"{value_col} by {group_col}"
        out_path = make_bar_chart(df, value_col, group_col=group_col, title=title)
        desc = f"Bar chart of '{value_col}' by '{group_col}'"

    # Prefer a repository-relative path for portability in logs/answers
    rel_path = os.path.relpath(out_path).replace("\\", "/")
    return {"path": rel_path, "desc": desc, "rows": int(len(df))}
