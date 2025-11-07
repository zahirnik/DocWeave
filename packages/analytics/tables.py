# packages/analytics/tables.py
"""
Tabular helpers — tiny utilities for KPI pivots and safe exports.

What this module provides
-------------------------
- top_n(df, by, n=10, ascending=False, groupby=None) -> DataFrame
    Return the top/bottom N rows overall or within each group.

- pivot_kpi(df, index, columns, values, aggfunc="sum", fill_value=0) -> DataFrame
    Small wrapper around pandas.pivot_table with sane defaults.

- percent_format(df, cols, decimals=1) -> DataFrame
    Return a copy with percentage columns formatted as 0..100 numbers (not strings).

- to_csv_safe(df, path) -> str
- to_xlsx_safe(df, path, sheet_name="Sheet1") -> str
    Atomic writes (temp + rename) with directory creation.

- kpi_table_from_ratios(df, entity_col=None, period_col=None) -> DataFrame
    Pick common ratio columns and order them nicely.

- melt_wide_to_long(df, id_vars, value_vars, var_name="metric", value_name="value") -> DataFrame

Design goals
------------
- Tutorial-clear: explicit parameters, no surprises.
- Safe: atomic file writes; never creates exotic formats silently.
- No plotting here (see agent_graph/tools/charting.py).

Usage
-----
from packages.analytics.tables import *

df_top = top_n(df, by="revenue", n=5)
table = pivot_kpi(df, index="year", columns="ticker", values="revenue", aggfunc="sum")
to_csv_safe(table, "./data/outputs/revenue_pivot.csv")
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, Optional

import pandas as pd


# ---------------------------
# Core table helpers
# ---------------------------

def top_n(
    df: pd.DataFrame,
    *,
    by: str,
    n: int = 10,
    ascending: bool = False,
    groupby: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return top/bottom N rows overall or per group (stable for ties).
    """
    if groupby:
        g = df.groupby(groupby, group_keys=False)
        return g.apply(lambda d: d.sort_values(by=by, ascending=ascending).head(n))
    return df.sort_values(by=by, ascending=ascending).head(n)


def pivot_kpi(
    df: pd.DataFrame,
    *,
    index: str | List[str],
    columns: Optional[str | List[str]],
    values: str,
    aggfunc: str | callable = "sum",
    fill_value: float | int | None = 0,
) -> pd.DataFrame:
    """
    Small wrapper around pandas.pivot_table with explicit arguments.
    """
    table = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=fill_value,
        dropna=False,
    )
    # Make column order deterministic when MultiIndex
    if isinstance(table.columns, pd.MultiIndex):
        table = table.sort_index(axis=1)
    return table


def percent_format(df: pd.DataFrame, cols: Iterable[str], decimals: int = 1) -> pd.DataFrame:
    """
    Return a copy where ratio columns stored as 0..1 are converted to 0..100 scale.
    (Stays numeric; caller can format as strings at presentation time.)
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (out[c].astype(float) * 100.0).round(decimals)
    return out


def melt_wide_to_long(
    df: pd.DataFrame,
    *,
    id_vars: Iterable[str],
    value_vars: Iterable[str],
    var_name: str = "metric",
    value_name: str = "value",
) -> pd.DataFrame:
    """
    Unpivot wide KPI columns into a tall table.
    """
    return pd.melt(df, id_vars=list(id_vars), value_vars=list(value_vars), var_name=var_name, value_name=value_name)


# ---------------------------
# Safe exports (atomic writes)
# ---------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def to_csv_safe(df: pd.DataFrame, path: str, *, index: bool = True) -> str:
    """
    Atomically write CSV to `path` (UTF-8). Returns the final path.
    """
    _ensure_dir(path)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="") as tmp:
        tmp_path = tmp.name
        df.to_csv(tmp, index=index)
    os.replace(tmp_path, path)
    return path

def to_xlsx_safe(df: pd.DataFrame, path: str, *, sheet_name: str = "Sheet1", index: bool = True) -> str:
    """
    Atomically write XLSX to `path` using openpyxl (installed with pandas).
    """
    _ensure_dir(path)
    # Write to a temp file first
    with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
        tmp_path = tmp.name
    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
    os.replace(tmp_path, path)
    return path


# ---------------------------
# Ratio table convenience
# ---------------------------

_RATIO_ORDER = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "roa",
    "roe",
    "interest_coverage",
    "fcf_margin",
    "eps",
    "pe",
]

def kpi_table_from_ratios(
    df: pd.DataFrame,
    *,
    entity_col: Optional[str] = None,
    period_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a neat KPI table from a DataFrame that already contains ratio columns
    (see packages.analytics.ratios). Keeps entity/period identifiers if provided.
    """
    cols: List[str] = []
    if entity_col and entity_col in df.columns:
        cols.append(entity_col)
    if period_col and period_col in df.columns:
        cols.append(period_col)
    cols += [c for c in _RATIO_ORDER if c in df.columns]

    table = df[cols].copy()
    # Percent-scale margins (0..1 → 0..100), keeping coverage/PE/EPS intact
    pct_cols = [c for c in ["gross_margin", "operating_margin", "net_margin", "ebitda_margin", "fcf_margin"] if c in table.columns]
    table = percent_format(table, pct_cols, decimals=1)
    return table
