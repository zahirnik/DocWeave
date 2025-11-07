# packages/eval/reports.py
"""
Eval reports — tiny HTML/CSV exporters for Agentic-RAG evaluations.

What this module provides
-------------------------
- summarize(records) -> dict
    Compute high-level aggregates identical in spirit to Evaluator.summarize.

- write_csv_summary(records, path="./data/outputs/eval_summary.csv") -> str
    Save a compact CSV of example-level metrics plus a final 'AVERAGES' row.

- render_html_report(records, path="./data/outputs/eval_report.html", *, title="RAG Eval Report") -> str
    Produce a single-file HTML report (inline CSS) with:
      • header metrics (averages, counts, latency)
      • example table (question, answer len, key metrics)
      • expandable per-example contexts and citations

- load_records(path) -> list[dict]
    Convenience loader if you previously wrote JSONL via `harness.write_jsonl`.

Design goals
------------
- Tutorial-clear and dependency-light (stdlib only).
- Self-contained HTML (no external CSS/JS); safe to email or archive.
- Deterministic ordering and explicit field selection.

Usage
-----
from packages.eval.reports import render_html_report, write_csv_summary, load_records

recs = load_records("./data/outputs/last_eval.jsonl")
html_path = render_html_report(recs, "./data/outputs/eval_report.html")
csv_path = write_csv_summary(recs, "./data/outputs/eval_summary.csv")
print(html_path, csv_path)
"""

from __future__ import annotations

import csv
import html
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path) or "."), exist_ok=True)

def _fmt_pct(x: Optional[float], nd: int = 1) -> str:
    if x is None:
        return ""
    try:
        return f"{100.0 * float(x):.{nd}f}%"
    except Exception:
        return ""

def _fmt_num(x: Any) -> str:
    try:
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)
    except Exception:
        return str(x)

def _get(d: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


# ---------------------------
# Public helpers
# ---------------------------

def load_records(path: str) -> List[Dict[str, Any]]:
    """
    Load EvalRecord rows back from JSONL produced by `harness.write_jsonl`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate key metrics across records (mirrors Evaluator.summarize() shape).
    """
    if not records:
        return {"count": 0}
    keys = list((_get(records[0], "metrics") or {}).keys())
    sums: Dict[str, float] = {}
    n = float(len(records))
    for r in records:
        m = r.get("metrics") or {}
        for k, v in m.items():
            if isinstance(v, (int, float)) and k not in {"answer_len_tokens", "elapsed_ms", "num_contexts"}:
                sums[k] = sums.get(k, 0.0) + float(v)
    avg = {f"avg_{k}": round(sums.get(k, 0.0) / n, 4) for k in sums}
    avg["count"] = int(n)
    avg["avg_elapsed_ms"] = round(sum(_get(r, "metrics", "elapsed_ms", default=0) for r in records) / n, 1)
    avg["avg_answer_len_tokens"] = round(sum(_get(r, "metrics", "answer_len_tokens", default=0) for r in records) / n, 1)
    avg["avg_num_contexts"] = round(sum(_get(r, "metrics", "num_contexts", default=0) for r in records) / n, 2)
    return avg


def write_csv_summary(records: List[Dict[str, Any]], path: str = "./data/outputs/eval_summary.csv") -> str:
    """
    Write one row per example with key metrics, plus a trailing AVERAGES row.
    """
    _ensure_dir(path)
    # Collect metric keys
    metric_keys = set()
    for r in records:
        metric_keys.update((r.get("metrics") or {}).keys())
    metric_keys = sorted(metric_keys)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["#",
                  "question",
                  "answer_len_tokens"] + metric_keys + ["elapsed_ms", "num_contexts"]
        w.writerow(header)
        for i, r in enumerate(records, 1):
            m = r.get("metrics") or {}
            row = [
                i,
                _trim(_norm_space(r.get("question") or ""), 140),
                m.get("answer_len_tokens", ""),
            ]
            row += [m.get(k, "") for k in metric_keys]
            row += [m.get("elapsed_ms", ""), m.get("num_contexts", "")]
            w.writerow(row)

        # Averages row
        s = summarize(records)
        avg_row = ["", "AVERAGES", s.get("avg_answer_len_tokens", "")]
        avg_row += [s.get(f"avg_{k}", "") for k in metric_keys]
        avg_row += [s.get("avg_elapsed_ms", ""), s.get("avg_num_contexts", "")]
        w.writerow(avg_row)
    return path


# ---------------------------
# HTML report
# ---------------------------

def render_html_report(
    records: List[Dict[str, Any]],
    path: str = "./data/outputs/eval_report.html",
    *,
    title: str = "RAG Eval Report",
) -> str:
    """
    Produce a single-file, minimal HTML report.
    """
    _ensure_dir(path)
    s = summarize(records)

    head = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8" />
<title>{_esc(title)}</title>
<style>
  :root {{
    --bg:#0b1220; --panel:#111827; --text:#e5e7eb; --muted:#9ca3af; --accent:#22d3ee; --ok:#10b981; --warn:#f59e0b; --bad:#ef4444;
    --line:#1f2937;
  }}
  body {{ background:var(--bg); color:var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin:0; padding:24px; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 8px 0; }}
  .sub {{ color: var(--muted); font-size: 12px; margin-bottom: 18px; }}
  .cards {{ display:grid; grid-template-columns: repeat(6,minmax(0,1fr)); gap:12px; margin-bottom: 18px; }}
  .card {{ background: var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }}
  .card h3 {{ margin:0; font-size:12px; color:var(--muted); font-weight:500; }}
  .card .v {{ font-size:18px; margin-top:6px; }}
  table {{ width:100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align: top; }}
  th {{ color: var(--muted); font-weight:600; }}
  tr:hover td {{ background:#0f172a; }}
  details {{ margin:8px 0; }}
  .k {{ color:var(--muted); }}
  .pct-ok {{ color: var(--ok); }}
  .pct-med {{ color: var(--warn); }}
  .pct-bad {{ color: var(--bad); }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
  .small {{ font-size: 11px; color: var(--muted); }}
</style>
<div class="wrap">
  <h1>{_esc(title)}</h1>
  <div class="sub">Generated: {html.escape(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ'))}</div>
  <div class="cards">
    {_metric_card("Examples", str(s.get("count", 0)))}
    {_metric_card("Avg Jaccard(Q↔Ctx)", _fmt_num(s.get("avg_jaccard_q2ctx")))}
    {_metric_card("Avg Ref Coverage", _fmt_num(s.get("avg_ref_coverage")))}
    {_metric_card("Avg Answer Supported", _fmt_num(s.get("avg_ans_supported_ratio")))}
    {_metric_card("Avg Ctx Precision-like", _fmt_num(s.get("avg_ctx_precision_like")))}
    {_metric_card("Avg Latency (ms)", _fmt_num(s.get("avg_elapsed_ms")))}
  </div>
"""

    # Table header
    body = []
    body.append("<table>")
    body.append("<thead><tr>")
    cols = [
        "#", "Question", "AnsLen", "Jaccard(Q↔Ctx)", "RefCov", "AnsSupported", "AnsRefJ", "CtxPrecision", "Latency(ms)", "Ctx"
    ]
    for c in cols:
        body.append(f"<th>{_esc(c)}</th>")
    body.append("</tr></thead><tbody>")

    # Rows
    for i, r in enumerate(records, 1):
        q = _trim(_norm_space(r.get("question") or ""), 180)
        m = r.get("metrics") or {}
        body.append("<tr>")
        body.append(f"<td class='mono'>{i}</td>")
        body.append(f"<td>{_esc(q)}{_expand_details(r)}</td>")
        body.append(f"<td class='mono'>{_esc(m.get('answer_len_tokens',''))}</td>")
        body.append(f"<td class='mono'>{_fmt_metric(m.get('jaccard_q2ctx'))}</td>")
        body.append(f"<td class='mono'>{_fmt_metric(m.get('ref_coverage'))}</td>")
        body.append(f"<td class='mono'>{_fmt_metric(m.get('ans_supported_ratio'))}</td>")
        body.append(f"<td class='mono'>{_fmt_metric(m.get('ans_ref_jaccard'))}</td>")
        body.append(f"<td class='mono'>{_fmt_metric(m.get('ctx_precision_like'))}</td>")
        body.append(f"<td class='mono'>{_esc(m.get('elapsed_ms',''))}</td>")
        body.append(f"<td class='mono'>{_esc(m.get('num_contexts',''))}</td>")
        body.append("</tr>")
    body.append("</tbody></table>")
    tail = "</div></html>"

    html_text = head + "\n".join(body) + tail
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return path


# ---------------------------
# Internal render helpers
# ---------------------------

def _metric_card(label: str, value: str) -> str:
    return f"""<div class="card"><h3>{_esc(label)}</h3><div class="v mono">{_esc(value)}</div></div>"""

def _norm_space(s: str) -> str:
    return " ".join((s or "").split())

def _trim(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"

def _esc(s: Any) -> str:
    return html.escape(str(s))

def _pct_color(v: Optional[float]) -> str:
    if v is None:
        return ""
    try:
        x = float(v)
    except Exception:
        return ""
    # Color heuristic for 0..1 metrics
    if x >= 0.7:
        return "pct-ok"
    if x >= 0.4:
        return "pct-med"
    return "pct-bad"

def _fmt_metric(v: Optional[float]) -> str:
    if v is None:
        return ""
    try:
        x = float(v)
    except Exception:
        return ""
    cls = _pct_color(x)
    return f"<span class='{cls}'>{x:.3f}</span>"

def _expand_details(r: Dict[str, Any]) -> str:
    """
    Details block with citations and contexts (truncated).
    """
    ans = r.get("answer") or ""
    cits = r.get("citations") or []
    ctxs = r.get("contexts") or []
    parts = []
    if ans:
        parts.append(f"<div><span class='k'>Answer:</span> {_esc(_trim(_norm_space(ans), 300))}</div>")
    if cits:
        parts.append("<div><span class='k'>Citations:</span> <span class='mono'>" + _esc(", ".join(cits)) + "</span></div>")
    if ctxs:
        # show up to 2 contexts inline; include a <details> with the rest
        head = []
        for c in ctxs[:2]:
            txt = _trim(_norm_space(c.get("text") or ""), 220)
            src = c.get("metadata", {}).get("source") or c.get("id") or ""
            head.append(f"<div class='small'><span class='k'>Ctx:</span> {_esc(txt)} <span class='mono k'>[{_esc(src)}]</span></div>")
        tail = ""
        if len(ctxs) > 2:
            extra = []
            for c in ctxs[2:6]:  # cap to keep HTML small
                txt = _trim(_norm_space(c.get("text") or ""), 220)
                src = c.get("metadata", {}).get("source") or c.get("id") or ""
                extra.append(f"<div class='small'><span class='k'>Ctx:</span> {_esc(txt)} <span class='mono k'>[{_esc(src)}]</span></div>")
            tail = "<details><summary class='small'>More contexts</summary>" + "".join(extra) + "</details>"
        parts.append("".join(head) + tail)
    if not parts:
        return ""
    return "<div class='small'>" + "".join(parts) + "</div>"


__all__ = ["load_records", "summarize", "write_csv_summary", "render_html_report"]
