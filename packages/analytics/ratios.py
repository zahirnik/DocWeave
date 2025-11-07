# packages/analytics/ratios.py
"""
Finance ratios — clear, small helpers for classic fundamental metrics.

What this module provides
-------------------------
- compute_ratios_row(row: dict) -> dict
    Computes common margins, liquidity, leverage, and returns from a **single period**.

- compute_ratios_table(df: pandas.DataFrame, *, entity_col=None, period_col=None) -> pandas.DataFrame
    Vectorised version for a whole table (optionally grouped by company/period).

- common_size_income(df) -> pandas.DataFrame
    Express income statement lines as % of revenue.

Design goals
------------
- Tutorial-clear and dependency-light (only pandas is optional for table ops).
- Tolerant to missing fields; returns None when a ratio can't be computed.
- Column names kept simple and conventional (see “Expected columns” below).

Expected columns (use what you have)
------------------------------------
These names are **suggested**; if your data uses different names, rename before calling.

Income statement (single period):
  revenue, cogs, gross_profit, operating_income, ebit, ebitda, net_income

Balance sheet (end of period):
  current_assets, current_liabilities, cash, receivables, inventory,
  total_assets, total_liabilities, total_equity, interest_expense

Cash flow (optional):
  operating_cash_flow, capex

Market (optional):
  shares_outstanding, price

Notes
-----
- We use **EBIT** when available for margins and coverage; fallback to operating_income.
- Interest coverage uses EBIT / interest_expense (or operating_income if EBIT missing).
- Quick ratio = (current_assets - inventory) / current_liabilities (cash & receivables help if provided).
"""

from __future__ import annotations

from typing import Dict, Optional, Any

try:
    import pandas as pd  # optional
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# ---------------------------
# Small numeric helpers
# ---------------------------

def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        # treat NaNs as None
        if v != v:  # NaN check
            return None
        return v
    except Exception:
        return None

def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None:
        return None
    if abs(d) < 1e-12:
        return None
    return n / d

def _pick(*vals: Optional[float]) -> Optional[float]:
    """Return the first non-None value."""
    for v in vals:
        if v is not None:
            return v
    return None


# ---------------------------
# Row-level ratios
# ---------------------------

def compute_ratios_row(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Compute key finance ratios for a **single period** represented as a dict-like row.

    Returns a dict where values are floats or None if not computable.

    Example
    -------
    compute_ratios_row({
        "revenue": 1000, "cogs": 600, "gross_profit": 400,
        "operating_income": 150, "ebit": 160, "net_income": 120,
        "current_assets": 500, "current_liabilities": 300, "inventory": 100,
        "total_assets": 1200, "total_equity": 600, "interest_expense": 20
    })
    """
    # Pull and normalise
    revenue = _as_float(row.get("revenue"))
    cogs = _as_float(row.get("cogs"))
    gross_profit = _as_float(row.get("gross_profit"))
    operating_income = _as_float(row.get("operating_income"))
    ebit = _as_float(row.get("ebit"))
    ebitda = _as_float(row.get("ebitda"))
    net_income = _as_float(row.get("net_income"))

    current_assets = _as_float(row.get("current_assets"))
    current_liabilities = _as_float(row.get("current_liabilities"))
    cash = _as_float(row.get("cash"))
    receivables = _as_float(row.get("receivables"))
    inventory = _as_float(row.get("inventory"))

    total_assets = _as_float(row.get("total_assets"))
    total_liabilities = _as_float(row.get("total_liabilities"))
    total_equity = _as_float(row.get("total_equity"))

    interest_expense = _as_float(row.get("interest_expense"))

    operating_cash_flow = _as_float(row.get("operating_cash_flow"))
    capex = _as_float(row.get("capex"))

    shares_out = _as_float(row.get("shares_outstanding"))
    price = _as_float(row.get("price"))

    # Derived where needed
    if gross_profit is None and revenue is not None and cogs is not None:
        gross_profit = revenue - cogs

    # Margins (% of revenue)
    gross_margin = _safe_div(gross_profit, revenue)
    op_base = _pick(ebit, operating_income)
    operating_margin = _safe_div(op_base, revenue)
    net_margin = _safe_div(net_income, revenue)
    ebitda_margin = _safe_div(ebitda, revenue)

    # Liquidity
    current_ratio = _safe_div(current_assets, current_liabilities)
    quick_assets = _pick(
        None if current_assets is None or inventory is None else current_assets - inventory,
        None if cash is None or receivables is None or current_liabilities is None else cash + receivables,
    )
    quick_ratio = _safe_div(quick_assets, current_liabilities)

    # Leverage
    debt_to_equity = _safe_div(total_liabilities, total_equity)
    debt_to_assets = _safe_div(total_liabilities, total_assets)

    # Returns
    roa = _safe_div(net_income, total_assets)
    roe = _safe_div(net_income, total_equity)

    # Coverage
    interest_cov = _safe_div(op_base, interest_expense)

    # Cash flow based
    fcf = None
    if operating_cash_flow is not None and capex is not None:
        fcf = operating_cash_flow - capex
    fcf_margin = _safe_div(fcf, revenue)

    # Per-share & valuation (if market fields present)
    eps = _safe_div(net_income, shares_out)
    pe = _safe_div(price, eps) if (price is not None and eps not in (None, 0)) else None

    return {
        # Margins (unitless fractions; multiply by 100 to get %)
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "ebitda_margin": ebitda_margin,

        # Liquidity
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,

        # Leverage
        "debt_to_equity": debt_to_equity,
        "debt_to_assets": debt_to_assets,

        # Returns
        "roa": roa,
        "roe": roe,

        # Coverage
        "interest_coverage": interest_cov,

        # Cash flow
        "fcf": fcf,
        "fcf_margin": fcf_margin,

        # Per-share & valuation
        "eps": eps,
        "pe": pe,
    }


# ---------------------------
# Table-level helpers (pandas)
# ---------------------------

def compute_ratios_table(
    df,  # pandas.DataFrame
    *,
    entity_col: Optional[str] = None,
    period_col: Optional[str] = None,
):
    """
    Vectorised ratio computation over a DataFrame.

    - If `entity_col`/`period_col` are provided, they are preserved in the output.
    - Returns a new DataFrame with ratio columns appended.

    Columns used are the same as for `compute_ratios_row`.
    """
    if pd is None:
        raise RuntimeError("pandas is required for compute_ratios_table")

    cols = df.columns

    def _get(col: str):
        return df[col] if col in cols else None

    revenue = _get("revenue")
    cogs = _get("cogs")
    gross_profit = _get("gross_profit")
    operating_income = _get("operating_income")
    ebit = _get("ebit")
    ebitda = _get("ebitda")
    net_income = _get("net_income")

    current_assets = _get("current_assets")
    current_liabilities = _get("current_liabilities")
    inventory = _get("inventory")

    total_assets = _get("total_assets")
    total_equity = _get("total_equity")
    total_liabilities = _get("total_liabilities")
    interest_expense = _get("interest_expense")

    operating_cash_flow = _get("operating_cash_flow")
    capex = _get("capex")

    shares_out = _get("shares_outstanding")
    price = _get("price")

    import numpy as np

    # Derived
    if gross_profit is None and revenue is not None and cogs is not None:
        gross_profit = revenue - cogs

    def div(a, b):
        a = a.astype(float) if a is not None else None
        b = b.astype(float) if b is not None else None
        if a is None or b is None:
            return np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
            out = out.where(np.isfinite(out))
        return out

    op_base = ebit if ebit is not None else operating_income

    # Ratios
    out = df.copy()
    out["gross_margin"] = div(gross_profit, revenue)
    out["operating_margin"] = div(op_base, revenue)
    out["net_margin"] = div(net_income, revenue)
    out["ebitda_margin"] = div(ebitda, revenue)

    out["current_ratio"] = div(current_assets, current_liabilities)
    if current_assets is not None and inventory is not None:
        out["quick_ratio"] = div(current_assets - inventory, current_liabilities)
    else:
        out["quick_ratio"] = np.nan

    out["debt_to_equity"] = div(total_liabilities, total_equity)
    out["debt_to_assets"] = div(total_liabilities, total_assets)

    out["roa"] = div(net_income, total_assets)
    out["roe"] = div(net_income, total_equity)

    out["interest_coverage"] = div(op_base, interest_expense)

    if operating_cash_flow is not None and capex is not None:
        out["fcf"] = operating_cash_flow - capex
        out["fcf_margin"] = div(out["fcf"], revenue)
    else:
        out["fcf"] = np.nan
        out["fcf_margin"] = np.nan

    if net_income is not None and shares_out is not None:
        out["eps"] = div(net_income, shares_out)
    else:
        out["eps"] = np.nan

    if "eps" in out.columns and price is not None:
        out["pe"] = div(price, out["eps"])
    else:
        out["pe"] = np.nan

    # Keep only useful columns if entity/period provided?
    # We keep all original columns + ratios for clarity.
    return out


# ---------------------------
# Common-size income statement
# ---------------------------

def common_size_income(df):
    """
    Express income statement lines as a percentage of revenue (0..1).

    Input DataFrame should have:
      revenue, cogs, gross_profit, operating_income, ebit, ebitda, net_income
    Missing columns are ignored.

    Returns a new DataFrame with *_pct columns.
    """
    if pd is None:
        raise RuntimeError("pandas is required for common_size_income")
    cols = [c for c in ["cogs", "gross_profit", "operating_income", "ebit", "ebitda", "net_income"] if c in df.columns]
    out = df.copy()
    for c in cols:
        out[f"{c}_pct"] = out[c] / out["revenue"]
    return out
