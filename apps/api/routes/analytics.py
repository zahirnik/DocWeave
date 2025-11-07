# apps/api/routes/analytics.py
"""
Analytics route — CSV/XLSX → pandas → JSON summary (+ optional PNG chart).

What this file provides
-----------------------
- POST /analytics
  Accepts a CSV/XLSX upload and returns:
    - row/column counts
    - dtype map
    - basic numeric summary (min, max, mean)
    - (optional) a PNG chart saved to disk (and its path returned)

Security / RBAC
---------------
- Requires scope: ["rag:query"]  (keep it simple; you can change to ["rag:analyze"] later)
- Multi-tenant: the chart (if generated) is stored under data/outputs/analytics/{tenant_id}/...

Input format (multipart/form-data)
----------------------------------
- file      : UploadFile   [required]  (.csv or .xlsx)
- tenant_id : str          [optional]  (default "t0")
- x_col     : str          [optional]  (x axis for chart)
- y_col     : str          [optional]  (y axis for chart, numeric column)
- chart     : bool         [optional]  (default False — generate PNG if True)
- title     : str          [optional]  (chart title; default derived from columns)

Output shape
------------
{
  "rows": 4,
  "cols": 3,
  "columns": ["quarter","revenue","cogs"],
  "dtypes": {"quarter":"object","revenue":"int64","cogs":"int64"},
  "metrics": {"revenue":{"min":100,"max":160,"mean":130.0}, "cogs": {...}},
  "head": [{"quarter":"Q1","revenue":100,"cogs":40}, ... up to 5 rows],
  "chart_path": "data/outputs/analytics/t0/analytics_7a8c....png"  # only when chart==True
}

Implementation notes
--------------------
- We keep the route intentionally small and explicit (tutorial style).
- Use `packages.agent_graph.tools.tabular_stats.summarize_csv` for metrics (CSV case).
- For XLSX, we read via pandas, compute metrics here (the helper can be extended similarly).
- PNG chart generation is done by `packages.agent_graph.tools.charting.save_line_chart`.
- All file I/O uses atomic writes; size/type checks are friendly but not exhaustive.
"""

from __future__ import annotations

import io
import os
import uuid
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

# Auth & scopes
from .auth import get_principal, require_scopes, Principal

# Observability & audit
from packages.observability.tracing import trace, set_span_attr, add_event
from packages.core.audit import append_event

# Helpers
from packages.agent_graph.tools.tabular_stats import summarize_csv
from packages.agent_graph.tools.charting import save_line_chart

router = APIRouter()


# ---------------------------
# Response model
# ---------------------------

class AnalyticsResponse(BaseModel):
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    metrics: Dict[str, Dict[str, float]]
    head: List[Dict[str, Any]]
    chart_path: Optional[str] = Field(
        None, description="Path to saved PNG chart (when chart=true and x/y provided)"
    )


# ---------------------------
# Route
# ---------------------------

@router.post(
    "",
    response_model=AnalyticsResponse,
    summary="Analyze a CSV/XLSX file and optionally generate a PNG chart",
    status_code=200,
    dependencies=[Depends(require_scopes(["rag:query"]))],
)
@trace("api.analytics.analyze")
async def analyze(
    file: UploadFile = File(..., description="CSV/XLSX file"),
    tenant_id: str = Form("t0"),
    x_col: Optional[str] = Form(None),
    y_col: Optional[str] = Form(None),
    chart: bool = Form(False),
    title: Optional[str] = Form(None),
    principal: Principal = Depends(get_principal),
) -> AnalyticsResponse:
    """
    Minimal yet practical analytics endpoint:
    - Parses a small CSV/XLSX into pandas.
    - Returns schema + numeric metrics.
    - Optionally writes a PNG chart to data/outputs/analytics/{tenant_id}/.
    """
    # Trace attrs for visibility
    set_span_attr("tenant_id", tenant_id)
    set_span_attr("filename", file.filename)
    set_span_attr("chart", chart)

    # Basic type guard
    fname = (file.filename or "").lower()
    if not (fname.endswith(".csv") or fname.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only .csv or .xlsx files are supported.")

    # Read file into pandas
    try:
        raw = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        if fname.endswith(".csv"):
            buf = io.BytesIO(raw)
            df = pd.read_csv(buf)
        else:
            buf = io.BytesIO(raw)
            df = pd.read_excel(buf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse as table: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="No rows after parsing.")

    # Build response pieces
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Compute metrics (tutorial-simple). If CSV, we can also call summarize_csv for parity.
    metrics: Dict[str, Dict[str, float]] = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        metrics[c] = {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean())}

    # Head preview (up to 5 rows)
    head_rows: List[Dict[str, Any]] = df.head(5).to_dict(orient="records")

    chart_path: Optional[str] = None
    if chart:
        # Validate columns
        if not x_col or not y_col:
            raise HTTPException(status_code=400, detail="x_col and y_col are required when chart=true.")
        if x_col not in df.columns or y_col not in df.columns:
            raise HTTPException(status_code=400, detail="x_col or y_col not found in the data.")
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            raise HTTPException(status_code=400, detail="y_col must be numeric for charting.")

        # Prepare output dir and file name
        out_dir = os.path.join("data", "outputs", "analytics", tenant_id)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"analytics_{uuid.uuid4().hex}.png"
        chart_path = os.path.join(out_dir, fname)

        # Save chart (single line chart for clarity)
        try:
            save_line_chart(
                xs=list(map(str, df[x_col].tolist())),
                ys=df[y_col].tolist(),
                path=chart_path,
                title=title or f"{y_col} by {x_col}",
            )
            add_event(None, "analytics.chart_saved", {"path": chart_path})
        except Exception as e:
            # Do not fail the whole request; return summary without chart
            chart_path = None
            add_event(None, "analytics.chart_error", {"error": str(e)})

    # Audit the action (append-only log)
    append_event(
        action="analytics.analyze",
        actor=principal.subject,
        details={"tenant_id": tenant_id, "filename": file.filename, "chart": chart, "x_col": x_col, "y_col": y_col},
    )

    return AnalyticsResponse(
        rows=int(len(df)),
        cols=int(df.shape[1]),
        columns=list(map(str, df.columns)),
        dtypes=dtypes,
        metrics=metrics,
        head=head_rows,
        chart_path=chart_path,
    )
