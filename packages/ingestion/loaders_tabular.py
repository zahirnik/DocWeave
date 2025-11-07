# packages/ingestion/loaders_tabular.py
"""
Tabular loaders — CSV/XLSX → pandas.DataFrame (tiny, explicit, with friendly guards).

What this module provides
-------------------------
- load_csv(path: str, *, dtype: dict | None = None, parse_dates: list[str] | None = None) -> pd.DataFrame
    * Robust CSV reader with delimiter sniffing, encoding fallback, and dtype/date parsing knobs.
    * Uses pandas with safe defaults; caps rows/cols optionally via env.

- load_xlsx(path: str, *, sheet: str | int | None = 0, dtype: dict | None = None, parse_dates: list[str] | None = None) -> pd.DataFrame
    * Minimal Excel reader (openpyxl engine). Converts to a tidy DataFrame.

Design goals
------------
- Keep behavior **predictable** and tutorial-clear.
- Fail with clear error messages when files are too large or malformed.
- Provide simple dtype/date parsing hooks (common in finance datasets).
- Avoid magical inference: prefer explicit options from callers.

Environment knobs
-----------------
MAX_CSV_MB=64          # coarse size guard for CSV (default 64 MB)
MAX_XLSX_MB=64         # coarse size guard for XLSX (default 64 MB)
CSV_SNIFFER_SAMPLE=65536   # bytes to sniff for delimiter/encoding (default 64 KiB)
CSV_DEFAULT_DELIM=,        # fallback delimiter if sniffer fails
CSV_DEFAULT_ENCODING=utf-8 # try this first, then fall back to latin-1

Examples
--------
from packages.ingestion.loaders_tabular import load_csv, load_xlsx

df = load_csv("data/samples/quarterly.csv", parse_dates=["quarter_end"])
print(df.dtypes)

df2 = load_xlsx("data/samples/metrics.xlsx", sheet="Sheet1")
print(df2.head())
"""

from __future__ import annotations

import csv
import io
import os
from typing import Dict, Iterable, List, Optional

import pandas as pd


# ---------------------------
# Env knobs / guards
# ---------------------------

_MAX_CSV_MB = int(os.getenv("MAX_CSV_MB", "64"))
_MAX_XLSX_MB = int(os.getenv("MAX_XLSX_MB", "64"))
_SNIFFER_SAMPLE = int(os.getenv("CSV_SNIFFER_SAMPLE", str(64 * 1024)))
_DEFAULT_DELIM = os.getenv("CSV_DEFAULT_DELIM", ",")
_DEFAULT_ENCODING = os.getenv("CSV_DEFAULT_ENCODING", "utf-8")


def _size_ok(path: str, limit_mb: int) -> None:
    try:
        n = os.path.getsize(path)
    except FileNotFoundError:
        raise
    except Exception:
        # If unsure, let pandas decide; keep friendly.
        return
    if n > limit_mb * 1024 * 1024:
        raise RuntimeError(f"file too large (> {limit_mb} MB): {path}")


# ---------------------------
# Utilities
# ---------------------------

def _sniff_delimiter_and_encoding(path: str) -> tuple[str, str]:
    """
    Best-effort delimiter + encoding detection for CSV.
    Returns (delimiter, encoding). Falls back to (_DEFAULT_DELIM, _DEFAULT_ENCODING).
    """
    # 1) Try utf-8 first for sniffing; if that fails, fall back to latin-1 for robustness
    encodings = [_DEFAULT_ENCODING, "latin-1"]
    for enc in encodings:
        try:
            with open(path, "rb") as fb:
                raw = fb.read(_SNIFFER_SAMPLE)
            text = raw.decode(enc, errors="strict" if enc == _DEFAULT_ENCODING else "ignore")
            sample = text[:_SNIFFER_SAMPLE]
            try:
                dialect = csv.Sniffer().sniff(sample)  # type: ignore[arg-type]
                delim = dialect.delimiter
            except Exception:
                # Heuristic: choose the most frequent delimiter among common ones
                cand = [",", ";", "\t", "|"]
                delim = max(cand, key=lambda c: sample.count(c)) or _DEFAULT_DELIM
            return delim, enc
        except Exception:
            continue
    return _DEFAULT_DELIM, _DEFAULT_ENCODING


def _maybe_parse_dates(df: pd.DataFrame, cols: Optional[List[str]]) -> pd.DataFrame:
    if not cols:
        return df
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            except Exception:
                # Keep original if parsing fails; caller can handle.
                pass
    return df


# ---------------------------
# Public API
# ---------------------------

def load_csv(
    path: str,
    *,
    dtype: Optional[Dict[str, str]] = None,
    parse_dates: Optional[List[str]] = None,
    limit_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read a CSV into a DataFrame with friendly defaults.

    Args:
      path:         file path to a .csv
      dtype:        optional dtype mapping, e.g., {"revenue":"Int64","ticker":"string"}
      parse_dates:  list of column names to parse as datetimes (UTC)
      limit_rows:   cap rows (helpful in previews/tests)

    Returns:
      pandas.DataFrame

    Raises:
      FileNotFoundError, RuntimeError (size guard), or pandas errors.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _size_ok(path, _MAX_CSV_MB)

    delim, enc = _sniff_delimiter_and_encoding(path)

    read_kwargs = dict(
        sep=delim,
        encoding=enc,
        dtype=dtype,
        # Keep empty strings as NaN only if needed; in finance, empty often means missing.
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "Null", "NULL"],
        # Use engine="python" for funky delimiters; C engine is faster but stricter
        engine="python",
        # Arrow dtypes are nice when available (pandas >= 2.0 with pyarrow)
        dtype_backend="pyarrow" if "pyarrow" in pd.__dict__.get("get_option", lambda *_: "")() else "numpy",  # type: ignore
    )

    if limit_rows is not None and limit_rows > 0:
        read_kwargs["nrows"] = int(limit_rows)

    try:
        df = pd.read_csv(path, **read_kwargs)  # type: ignore[arg-type]
    except Exception as e:
        # Retry with less strict settings if encoding blew up
        if enc != "latin-1":
            try:
                df = pd.read_csv(path, sep=delim, encoding="latin-1", dtype=dtype, engine="python")  # type: ignore[arg-type]
            except Exception:
                raise RuntimeError(f"CSV parsing failed: {e}") from e
        else:
            raise RuntimeError(f"CSV parsing failed: {e}") from e

    # Trim BOM / whitespace from headers
    df.rename(columns=lambda c: str(c).encode("utf-8").decode("utf-8").strip().lstrip("\ufeff"), inplace=True)

    # Optional date parsing
    df = _maybe_parse_dates(df, parse_dates)

    return df


def load_xlsx(
    path: str,
    *,
    sheet: Optional[str | int] = 0,
    dtype: Optional[Dict[str, str]] = None,
    parse_dates: Optional[List[str]] = None,
    limit_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read an Excel sheet into a DataFrame.

    Args:
      path:         file path to a .xlsx
      sheet:        sheet name or index (default 0)
      dtype:        optional dtype mapping
      parse_dates:  list of column names to parse as datetimes (UTC)
      limit_rows:   cap rows (helpful in previews/tests)

    Returns:
      pandas.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _size_ok(path, _MAX_XLSX_MB)

    # Prefer openpyxl engine for .xlsx
    read_kwargs = dict(
        sheet_name=sheet if sheet is not None else 0,
        dtype=dtype,
        engine="openpyxl",
    )

    try:
        df = pd.read_excel(path, **read_kwargs)  # type: ignore[arg-type]
    except Exception as e:
        # Fallback without specifying engine (pandas will choose)
        try:
            df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, dtype=dtype)  # type: ignore[arg-type]
        except Exception:
            raise RuntimeError(f"XLSX parsing failed: {e}") from e

    # Limit rows if requested
    if limit_rows is not None and limit_rows > 0:
        df = df.head(int(limit_rows))

    # Normalize headers
    df.rename(columns=lambda c: str(c).strip().lstrip("\ufeff"), inplace=True)

    # Optional date parsing
    df = _maybe_parse_dates(df, parse_dates)

    return df
