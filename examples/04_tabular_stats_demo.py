# examples/04_tabular_stats_demo.py
"""
Tabular stats demo: CSV/XLSX → pandas → YoY/QoQ → PNG chart + CSV summary.

What this example shows
-----------------------
- A tiny, **practical** pipeline for finance-style time series analysis.
- Auto-detect date & value columns (with CLI overrides).
- Compute YoY and (if quarterly) QoQ, rolling mean/std, simple anomaly flags.
- Save:
    • ./data/outputs/tabular_stats_chart.png
    • ./data/outputs/tabular_stats_summary.csv
- Prints a compact table to the console.

Run
---
  python -m examples.04_tabular_stats_demo \
      --file ./data/samples/beta_revenue_quarterly.csv \
      --date-col period --value-col revenue

If you omit --file, it will scan ./data/samples for the first CSV/XLSX.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Helpers: discovery & parsing
# ---------------------------

def find_sample_file(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    for pat in ("*.csv", "*.xlsx", "*.xls"):
        files = sorted(base.rglob(pat))
        if files:
            return files[0]
    return None


def _guess_date_col(df: pd.DataFrame) -> Optional[str]:
    # Preferred names first
    for name in df.columns:
        n = str(name).lower()
        if n in {"date", "period", "quarter", "month", "year"}:
            try:
                pd.to_datetime(df[name], errors="raise")
                return name
            except Exception:
                pass
    # Try any column that parses cleanly to datetime (heuristic)
    for name in df.columns:
        try:
            _ = pd.to_datetime(df[name], errors="raise")
            return name
        except Exception:
            continue
    return None


def _guess_value_col(df: pd.DataFrame, exclude: List[str]) -> Optional[str]:
    for name, s in df.items():
        if name in exclude:
            continue
        if pd.api.types.is_numeric_dtype(s):
            return name
        # Try coercion if it's object-like
        try:
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().sum() >= max(3, int(0.6 * len(coerced))):
                return name
        except Exception:
            pass
    return None


def load_table(path: Path, date_col: Optional[str], value_col: Optional[str]) -> Tuple[pd.DataFrame, str, str]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        # CSV: try common separators
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=";")

    # Guess columns if not specified
    date_col = date_col or _guess_date_col(df)
    if not date_col:
        raise ValueError("Unable to detect a date/period column; pass --date-col explicitly.")

    value_col = value_col or _guess_value_col(df, exclude=[date_col])
    if not value_col:
        raise ValueError("Unable to detect a numeric value column; pass --value-col explicitly.")

    # Parse
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Clean
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col).reset_index(drop=True)
    return df, date_col, value_col


# ---------------------------
# Frequency inference & metrics
# ---------------------------

def infer_frequency(dts: pd.Series) -> str:
    """
    Infer coarse frequency: 'M' (monthly), 'Q' (quarterly), 'Y' (annual), else 'U' (unknown).
    """
    if len(dts) < 3:
        return "U"
    diffs = dts.sort_values().diff().dropna().dt.days.to_numpy()
    med = float(np.median(diffs))
    if 25 <= med <= 35:
        return "M"
    if 80 <= med <= 100:
        return "Q"
    if 350 <= med <= 380:
        return "Y"
    return "U"


def compute_metrics(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Compute YoY, QoQ (if quarterly), rolling mean/std and z-score anomalies.
    Returns a new DataFrame indexed by date.
    """
    s = df.set_index(date_col)[value_col].asfreq("D")  # daily to ensure continuity for resample
    # Resample to clean monthly/quarterly/annual using last observation in period (finance style)
    freq = infer_frequency(df[date_col])
    if freq == "M":
        y = s.resample("M").last()
        yoy_periods = 12
        mom_periods = 1
        roll_win = 3
    elif freq == "Q":
        y = s.resample("Q").last()
        yoy_periods = 4
        mom_periods = 1  # QoQ is 1 period on quarterly index
        roll_win = 4
    elif freq == "Y":
        y = s.resample("Y").last()
        yoy_periods = 1
        mom_periods = 1  # YoY on annual rows = Δ vs prior year (periods=1)
        roll_win = 2
    else:
        # Unknown → assume monthly
        y = s.resample("M").last()
        yoy_periods = 12
        mom_periods = 1
        roll_win = 3
        freq = "M"

    out = pd.DataFrame(index=y.index)
    out["value"] = y

    # Changes
    out["pct_change_period"] = out["value"].pct_change(mom_periods)
    out["pct_change_yoy"] = out["value"].pct_change(yoy_periods)

    # Rolling stats
    out["roll_mean"] = out["value"].rolling(roll_win, min_periods=max(2, roll_win // 2)).mean()
    out["roll_std"] = out["value"].rolling(roll_win, min_periods=max(2, roll_win // 2)).std(ddof=0)
    out["zscore"] = (out["value"] - out["roll_mean"]) / (out["roll_std"].replace(0, np.nan))

    # Anomaly flag (simple)
    out["anomaly"] = out["zscore"].abs() >= 2.5

    # Human-readable index column
    out = out.reset_index().rename(columns={out.index.name: "idx", "index": "date"})
    out = out.rename(columns={"idx": "date"}) if "date" not in out.columns else out

    # Consistent date dtype
    out["date"] = pd.to_datetime(out["date"])
    out["freq"] = {"M": "monthly", "Q": "quarterly", "Y": "annual"}.get(freq, "unknown")
    return out


# ---------------------------
# Render: table + chart
# ---------------------------

def write_summary_csv(dfm: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "value", "pct_change_period", "pct_change_yoy", "roll_mean", "zscore", "anomaly", "freq"]
    dfm.loc[:, cols].to_csv(out_path, index=False)
    return out_path


def print_console_table(dfm: pd.DataFrame, rows: int = 12) -> None:
    tail = dfm.tail(rows).copy()
    show = tail[["date", "value", "pct_change_period", "pct_change_yoy", "zscore", "anomaly"]]
    # round for display
    for c in ["value", "pct_change_period", "pct_change_yoy", "zscore"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(4)
    print("\nLatest periods:")
    print(show.to_string(index=False))


def save_chart(dfm: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    # Main series
    ax.plot(dfm["date"], dfm["value"], label="Value")
    if dfm["roll_mean"].notna().any():
        ax.plot(dfm["date"], dfm["roll_mean"], linestyle="--", label="Rolling mean")

    # Secondary: YoY % bars (scaled)
    ax2 = ax.twinx()
    yoy = (dfm["pct_change_yoy"] * 100.0).fillna(0.0)
    ax2.bar(dfm["date"], yoy, alpha=0.25, width=15, label="YoY %")

    # Anomaly markers
    if dfm["anomaly"].any():
        x = dfm.loc[dfm["anomaly"], "date"]
        y = dfm.loc[dfm["anomaly"], "value"]
        ax.scatter(x, y, marker="x", s=40, label="Anomaly")

    # Cosmetics
    ax.set_title("Tabular Stats Demo — Value & Rolling Mean (bars: YoY %)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax2.set_ylabel("YoY %")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    # unified legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=144)
    plt.close(fig)
    return out_path


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Tabular stats demo (YoY/QoQ + chart)")
    parser.add_argument("--file", help="CSV/XLSX path (default: first under ./data/samples)")
    parser.add_argument("--date-col", help="Name of date/period column")
    parser.add_argument("--value-col", help="Name of numeric value column")
    parser.add_argument("--rows", type=int, default=12, help="Console rows to display (default: 12)")
    args = parser.parse_args()

    base_samples = Path("./data/samples").resolve()
    fpath = Path(args.file).resolve() if args.file else find_sample_file(base_samples)
    if not fpath or not Path(fpath).exists():
        print("No input file found.")
        print("Tip: provide --file or drop a CSV/XLSX into ./data/samples and re-run.")
        return

    print(f"Input: {fpath}")
    df, dcol, vcol = load_table(Path(fpath), args.date_col, args.value_col)
    print(f"Detected columns → date: '{dcol}', value: '{vcol}'")

    dfm = compute_metrics(df[[dcol, vcol]], dcol, vcol)

    # Outputs
    out_dir = Path("./data/outputs")
    csv_path = write_summary_csv(dfm, out_dir / "tabular_stats_summary.csv")
    png_path = save_chart(dfm, out_dir / "tabular_stats_chart.png")

    # Console table
    print_console_table(dfm, rows=int(args.rows))

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved PNG : {png_path}")
    print("Done.")

if __name__ == "__main__":
    main()
