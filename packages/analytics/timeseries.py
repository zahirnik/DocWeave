# packages/analytics/timeseries.py
"""
Time-series helpers — clear, compact utilities for finance analytics.

What this module provides
-------------------------
- ensure_datetime_index(df, date_col="date") -> DataFrame
    Normalises a table so the index is a proper DatetimeIndex (if possible).

- pct_change(series, periods) -> Series
- yoy(series, annual_periods=4) -> Series
- qoq(series) -> Series
- rolling_mean(series, window) -> Series
- rolling_std(series, window) -> Series
- moving_average(series, window) -> Series      # alias of rolling_mean
- cumulative_return(series) -> Series           # percentage cumulative change
- log_return(series) -> Series                  # ln(P_t / P_{t-1})

- resample_series(series, rule="Q", how="sum") -> Series
    Down/upsample a Series with a DatetimeIndex using sum/mean/last.

- detect_anomalies_zscore(series, window=12, z=3.0) -> DataFrame
    Rolling z-scores with an 'is_anomaly' boolean flag.

- align_index_left_right(left, right, how="inner") -> (Series, Series)
    Align two time series on a common timeline.

- tail_as_records(series, n=8) -> list[dict]
    Small JSON-friendly tail export useful for API responses.

Design goals
------------
- Tutorial-clear and dependency-light (pandas + numpy only).
- Defensive: gracefully handles missing/invalid dates and NaNs.
- Side-effect free: no file I/O here; tables/exports live in tables.py.

Usage
-----
import pandas as pd
from packages.analytics.timeseries import *

df = pd.read_csv("./data/samples/acme_quarterly.csv")
df = ensure_datetime_index(df, "date")
s = df["revenue"]
print(yoy(s).tail())
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Index normalisation
# ---------------------------

def ensure_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure `df` has a DatetimeIndex by parsing `date_col` when possible.

    If parsing fails, returns the original DataFrame unchanged.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if date_col in df.columns:
        try:
            out = df.copy()
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
            if pd.api.types.is_datetime64_any_dtype(out[date_col]):
                out = out.sort_values(date_col).set_index(date_col)
                return out
        except Exception:
            # fall through to return original
            pass
    return df


# ---------------------------
# Core transforms
# ---------------------------

def pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Percentage change ×100 (NaN-safe)."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    return series.astype(float).pct_change(periods=periods) * 100.0


def yoy(series: pd.Series, annual_periods: int = 4) -> pd.Series:
    """Year-over-year % change for quarterly data by default (periods=4)."""
    return pct_change(series, periods=int(annual_periods))


def qoq(series: pd.Series) -> pd.Series:
    """Quarter-over-quarter % change (periods=1)."""
    return pct_change(series, periods=1)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` observations."""
    return series.astype(float).rolling(window=window, min_periods=1).mean()


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Alias for rolling_mean."""
    return rolling_mean(series, window)


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation over `window` observations."""
    return series.astype(float).rolling(window=window, min_periods=2).std()


def cumulative_return(series: pd.Series) -> pd.Series:
    """
    Cumulative return in %, i.e., (P_t / P_0 - 1) × 100.
    For non-price metrics this becomes cumulative % change from the first non-NaN.
    """
    s = series.astype(float).copy()
    if s.dropna().empty:
        return s * np.nan
    first = s.dropna().iloc[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (s / first - 1.0) * 100.0
    return out


def log_return(series: pd.Series) -> pd.Series:
    """Log returns: ln(P_t / P_{t-1}). Values are in natural log units (≈ % for small changes)."""
    s = series.astype(float)
    return np.log(s / s.shift(1))


# ---------------------------
# Resampling
# ---------------------------

def resample_series(series: pd.Series, rule: str = "Q", how: str = "sum") -> pd.Series:
    """
    Resample a time series using `rule` and an aggregation `how` in {"sum","mean","last"}.

    - "Q" : calendar quarter end; use "QE-DEC" etc. for anchored quarters if needed.
    - "M" : month end
    - "A" : year end
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex to resample")
    how = (how or "sum").lower()
    if how not in {"sum", "mean", "last"}:
        raise ValueError('how must be one of {"sum","mean","last"}')

    if how == "sum":
        return series.resample(rule).sum(min_count=1)
    if how == "mean":
        return series.resample(rule).mean()
    return series.resample(rule).last()


# ---------------------------
# Anomaly detection (rolling z-score)
# ---------------------------

def detect_anomalies_zscore(series: pd.Series, window: int = 12, z: float = 3.0) -> pd.DataFrame:
    """
    Compute rolling z-scores and flag anomalies where |z| >= threshold.

    Returns a DataFrame with columns: value, z, is_anomaly.
    """
    x = series.astype(float)
    mu = x.rolling(window=window, min_periods=max(3, window // 2)).mean()
    sd = x.rolling(window=window, min_periods=max(3, window // 2)).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscores = (x - mu) / sd
    out = pd.DataFrame(
        {"value": x, "z": zscores, "is_anomaly": (zscores.abs() >= float(z))}
    )
    return out


# ---------------------------
# Alignment helpers
# ---------------------------

def align_index_left_right(left: pd.Series, right: pd.Series, how: str = "inner") -> Tuple[pd.Series, pd.Series]:
    """
    Align two series on a shared DatetimeIndex (inner/outer/left/right).
    """
    if not isinstance(left.index, pd.DatetimeIndex) or not isinstance(right.index, pd.DatetimeIndex):
        raise ValueError("both series must have DatetimeIndex")
    l, r = left.align(right, join=how)
    return l, r


# ---------------------------
# Small JSON-friendly export
# ---------------------------

def tail_as_records(series: pd.Series, n: int = 8) -> List[dict]:
    """
    Turn the last N non-NaN points into a list of {"period","value"} records.
    """
    tail = series.dropna().tail(max(1, int(n)))
    if isinstance(tail.index, pd.DatetimeIndex):
        return [{"period": str(idx.date()), "value": float(val)} for idx, val in tail.items()]
    return [{"period": int(idx), "value": float(val)} for idx, val in tail.items()]
