# packages/ingestion/tables.py
"""
Table extraction helpers — PDF/HTML → pandas.DataFrame (tiny, explicit).

What this module provides
-------------------------
- extract_pdf_tables(path: str, *, pages: str = "1-end", flavor: str = "auto", max_tables: int | None = None) -> list[pd.DataFrame]
    Try Camelot (lattice/stream) → Tabula (Java) to extract tables from PDFs.
    Returns a **list of DataFrames** (may be empty). Best-effort; never crashes on missing deps.

- extract_html_tables(html_or_path: str, *, match: str | None = None, displayed_only: bool = True) -> list[pd.DataFrame]
    Use pandas.read_html to extract tables from HTML (string or file path).

- normalize_table(df: pd.DataFrame, *, drop_empty: bool = True, strip_headers: bool = True, coerce_numeric: bool = True) -> pd.DataFrame
    Clean headers/cells, drop empty rows/cols, coerce obvious numeric columns.

- tables_to_records(df: pd.DataFrame) -> list[dict]
    Convert a tidy DataFrame to a list of dicts (JSON-ready).

Design goals
------------
- Keep the surface area small and tutorial-clear.
- Prefer explicit knobs; avoid magical inference.
- Fail **softly** when optional dependencies (camelot/tabula) or Java/Ghostscript are absent.

Dependencies (optional)
-----------------------
- pandas (required)
- camelot-py[cv] (requires Ghostscript) — better for "lattice" PDFs with clear cell borders.
- tabula-py (requires Java)           — often works on "stream" PDFs.

Usage
-----
from packages.ingestion.tables import extract_pdf_tables, normalize_table

tables = extract_pdf_tables("data/samples/report.pdf", pages="1-5", flavor="auto")
tables = [normalize_table(t) for t in tables]
if tables:
    print(tables[0].head())
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd


# ---------------------------
# Utilities
# ---------------------------

def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _clean_header(s: str) -> str:
    s = str(s or "").strip()
    # Replace weird spaces and collapse inner whitespace
    s = s.replace("\u00a0", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    try:
        # common finance formats: "1,234.56", "(123)", "—", "N/A"
        cleaned = (
            s.map(lambda x: str(x).strip())
             .str.replace(r"[,\s]", "", regex=True)
             .str.replace("(", "-").str.replace(")", "")
             .str.replace("—", "", regex=False)
             .str.replace("–", "", regex=False)
             .str.replace("-", "-", regex=False)
        )
        out = pd.to_numeric(cleaned, errors="ignore")
        return out
    except Exception:
        return s


# ---------------------------
# Public API
# ---------------------------

def extract_pdf_tables(
    path: str,
    *,
    pages: str = "1-end",
    flavor: str = "auto",   # "auto" | "lattice" | "stream"
    max_tables: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Extract tables from a PDF using Camelot (preferred) then Tabula as a fallback.

    Args:
      path       : path to the PDF file
      pages      : Camelot/Tabula pages string (e.g., "1,3-5", "all", or "1-end")
      flavor     : "auto" (try lattice then stream), or force "lattice"/"stream"
      max_tables : cap number of tables returned (None = no cap)

    Returns:
      list of pandas.DataFrame (possibly empty)

    Raises:
      FileNotFoundError if `path` does not exist.
      (Never raises on missing optional deps; returns [] instead.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    tables: List[pd.DataFrame] = []

    # --- Try Camelot first (if installed) ---
    if _has_module("camelot"):
        try:
            import camelot  # type: ignore

            def _camelot(fl: str) -> List[pd.DataFrame]:
                if fl not in {"lattice", "stream"}:
                    return []
                res = camelot.read_pdf(path, pages=pages, flavor=fl)  # type: ignore
                return [t.df for t in res] if getattr(res, "n", 0) > 0 else []

            if flavor == "auto":
                for fl in ("lattice", "stream"):
                    try:
                        out = _camelot(fl)
                        if out:
                            tables.extend(out)
                            break  # choose first successful flavor
                    except Exception:
                        continue
            else:
                tables.extend(_camelot(flavor))
        except Exception:
            # swallow and try Tabula
            pass

    # --- Fallback: Tabula (if installed; Java required) ---
    if not tables and _has_module("tabula"):
        try:
            import tabula  # type: ignore
            # Note: pandas option True returns list of DataFrames
            out = tabula.read_pdf(path, pages=pages, multiple_tables=True)  # type: ignore
            if isinstance(out, list):
                tables.extend(out)
        except Exception:
            pass

    # Cap count if requested
    if max_tables is not None and max_tables > 0:
        tables = tables[: int(max_tables)]

    # Normalize obvious header rows that Camelot/Tabula sometimes create (first row as header)
    normed: List[pd.DataFrame] = []
    for df in tables:
        try:
            # If the first row looks like headers (no duplicates, short strings), promote it
            first = df.iloc[0].astype(str).tolist() if len(df) > 0 else []
            uniq = len(set(first)) == len(first)
            looks_headerish = uniq and all(len(_clean_header(x)) <= 64 for x in first)
            if looks_headerish:
                df2 = df.iloc[1:].copy()
                df2.columns = [_clean_header(x) for x in first]
                normed.append(df2.reset_index(drop=True))
            else:
                # Otherwise, just clean column names
                df2 = df.copy()
                df2.columns = [_clean_header(c) for c in df2.columns]
                normed.append(df2.reset_index(drop=True))
        except Exception:
            normed.append(df.reset_index(drop=True))

    return normed


def extract_html_tables(html_or_path: str, *, match: Optional[str] = None, displayed_only: bool = True) -> List[pd.DataFrame]:
    """
    Extract tables from HTML content or file path using pandas.read_html.

    Args:
      html_or_path  : HTML string or a file path to .html
      match         : optional regex to filter tables by header text
      displayed_only: pass-through to pandas.read_html (hide CSS-hidden tables)

    Returns:
      list of DataFrames (possibly empty)
    """
    try:
        if os.path.exists(html_or_path):
            tables = pd.read_html(html_or_path, match=match, displayed_only=displayed_only)  # type: ignore[arg-type]
        else:
            tables = pd.read_html(html_or_path, match=match, displayed_only=displayed_only)  # type: ignore[arg-type]
        return [t for t in tables if isinstance(t, pd.DataFrame)]
    except Exception:
        return []


def normalize_table(
    df: pd.DataFrame,
    *,
    drop_empty: bool = True,
    strip_headers: bool = True,
    coerce_numeric: bool = True,
) -> pd.DataFrame:
    """
    Clean a table DataFrame: trim headers/cells, drop empty rows/cols, coerce numeric columns.

    Heuristics:
    - Remove columns where ALL values are null/empty after stripping.
    - Remove rows that are entirely empty.
    - Try to coerce numeric-looking columns to numbers.

    Returns:
      A new, cleaned DataFrame.
    """
    out = df.copy()

    # Strip headers
    if strip_headers:
        out.columns = [_clean_header(c) for c in out.columns]

    # Strip cell whitespace and normalize NBSP
    def _strip_cell(x):
        try:
            s = str(x)
            s = s.replace("\u00a0", " ")
            return s.strip()
        except Exception:
            return x

    out = out.applymap(_strip_cell)

    if drop_empty:
        # Drop columns entirely empty or with only NaN/""
        keep_cols = []
        for c in out.columns:
            ser = out[c]
            if ser.dropna().map(lambda v: str(v).strip() != "").any():
                keep_cols.append(c)
        out = out[keep_cols]

        # Drop rows with no content
        mask_nonempty = out.apply(lambda r: any(str(v).strip() != "" for v in r), axis=1)
        out = out.loc[mask_nonempty].reset_index(drop=True)

    if coerce_numeric:
        for c in out.columns:
            # If most cells look numeric, try converting
            ser = out[c]
            sample = ser.dropna().astype(str).head(20).tolist()
            digits_like = sum(1 for s in sample if any(ch.isdigit() for ch in s))
            if sample and digits_like / len(sample) >= 0.6:
                out[c] = _coerce_numeric_series(ser)

    return out


def tables_to_records(df: pd.DataFrame) -> List[dict]:
    """
    Convert a tidy DataFrame to list-of-dicts (records) for JSON output.
    """
    try:
        return df.to_dict(orient="records")
    except Exception:
        # Fallback: manual conversion
        cols = list(map(str, df.columns))
        out: List[dict] = []
        for _, row in df.iterrows():
            out.append({cols[i]: row.iloc[i] for i in range(len(cols))})
        return out
