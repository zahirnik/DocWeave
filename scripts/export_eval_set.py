# scripts/export_eval_set.py
"""
Export a tiny **finance** evaluation set (JSONL) for RAG testing.

What this script does
---------------------
- Produces a JSONL file with **question / answers / references / meta** records.
- Two ways to build the set:
  1) **Seed file** (preferred): read your curated Q/A seeds from JSONL or CSV.
     - JSONL lines: {"question": "...", "answers": ["..."], "references": ["..."], "meta": {...}}
     - CSV cols  : question, answers, references, meta
       • answers    = JSON array string or single string
       • references = JSON array string or single string
       • meta       = JSON object string (optional)
  2) **Heuristics over ./data/samples**: scan PDFs/TXTs/MD/JSONL and synthesize simple Q/As
     by detecting common finance phrases (gross margin, revenue, EBITDA, Scope 1, etc.).
     This keeps you productive before you have a curated set.

Why JSONL?
----------
Eval harnesses (RAGAS/TruLens/etc.) love line-delimited JSON. Each line is independent.

Examples
--------
# From curated seeds (JSONL)
python -m scripts.export_eval_set --seeds ./data/samples/finance_seeds.jsonl --out ./packages/eval/datasets/finance_golden.jsonl

# From CSV seeds
python -m scripts.export_eval_set --seeds ./data/samples/finance_seeds.csv --out ./packages/eval/datasets/finance_golden.jsonl

# Heuristic synthesis from docs under ./data/samples (up to 30 items)
python -m scripts.export_eval_set --samples ./data/samples --max-total 30 --out ./packages/eval/datasets/finance_golden.jsonl

Output format (one per line)
----------------------------
{
  "question": "What was ACME's gross margin in 2024?",
  "answers": ["38.2%"],
  "references": ["samples://acme-2024.pdf#p3"],
  "meta": {"topic": "profitability", "year": 2024}
}

Notes
-----
- The heuristic mode is intentionally simple and **deterministic** (no LLM calls).
- For serious evals, replace with your curated seeds (audited by humans).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


# ---------------------------
# Data model
# ---------------------------

@dataclass
class EvalItem:
    question: str
    answers: List[str]
    references: List[str]
    meta: Dict[str, object]


# ---------------------------
# IO helpers
# ---------------------------

def write_jsonl(rows: Iterable[Dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _try_json(s: str) -> Optional[object]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _ensure_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    x = str(x).strip()
    if not x:
        return []
    # Try JSON array in string
    parsed = _try_json(x)
    if isinstance(parsed, list):
        return [str(i) for i in parsed]
    # Allow pipe/comma separated fallbacks
    if "|" in x:
        return [t.strip() for t in x.split("|") if t.strip()]
    if "," in x:
        return [t.strip() for t in x.split(",") if t.strip()]
    return [x]


def load_seeds(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                q = str(d.get("question") or "").strip()
                if not q:
                    continue
                ans = _ensure_list(d.get("answers"))
                refs = _ensure_list(d.get("references"))
                meta = d.get("meta") or {}
                if not isinstance(meta, dict):
                    meta = {}
                items.append(EvalItem(question=q, answers=ans, references=refs, meta=meta))
        return items

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                q = str(row.get("question") or "").strip()
                if not q:
                    continue
                ans = _ensure_list(row.get("answers"))
                refs = _ensure_list(row.get("references"))
                meta_raw = row.get("meta")
                meta = _try_json(meta_raw) if meta_raw else {}
                if not isinstance(meta, dict):
                    meta = {}
                items.append(EvalItem(question=q, answers=ans, references=refs, meta=meta))
        return items

    raise ValueError("Unsupported seeds file (use .jsonl or .csv)")


# ---------------------------
# Heuristic synthesis from docs
# ---------------------------

def _read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _read_md(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(p: Path) -> str:
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return ""
    out = []
    with p.open("rb") as f:
        reader = PyPDF2.PdfReader(f)  # type: ignore
        for page in reader.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(out)

def _read_jsonl(p: Path) -> List[str]:
    t: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("text") or d.get("content") or ""
            if isinstance(s, str) and s.strip():
                t.append(s)
    return t


def _scan_docs(base: Path) -> List[Tuple[str, str]]:
    """Return list of (doc_id, text) from a few basic formats."""
    pairs: List[Tuple[str, str]] = []
    if not base.exists():
        return pairs
    for pat in ("*.txt", "*.md", "*.pdf", "*.jsonl"):
        for p in sorted(base.rglob(pat)):
            if p.suffix.lower() in {".txt"}:
                t = _read_txt(p)
                if t.strip():
                    pairs.append((f"{p.name}", t))
            elif p.suffix.lower() in {".md"}:
                t = _read_md(p)
                if t.strip():
                    pairs.append((f"{p.name}", t))
            elif p.suffix.lower() in {".pdf"}:
                t = _read_pdf(p)
                if t.strip():
                    pairs.append((f"{p.name}", t))
            elif p.suffix.lower() in {".jsonl"}:
                for i, s in enumerate(_read_jsonl(p)):
                    pairs.append((f"{p.name}#r{i}", s))
    return pairs


def _norm(s: str) -> str:
    return " ".join((s or "").replace("-\n", "").replace("\n", " ").split())


# Regexes for simple finance fact extraction
RE_PERCENT = re.compile(r"(\d{1,2}(?:\.\d+)?\s?%)")
RE_MONEY   = re.compile(r"(\$?\d+(?:\.\d+)?\s?(?:bn|billion|m|million|k|thousand))", re.I)
RE_RATIO   = re.compile(r"(\d+(?:\.\d+)?\s?x)")
RE_NUMBER  = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)")

def _first_match(rx: re.Pattern, s: str) -> Optional[str]:
    m = rx.search(s)
    return m.group(1).strip() if m else None


def _make_items_from_text(doc_id: str, text: str, max_per_doc: int = 4) -> List[EvalItem]:
    """
    Produce a handful of Q/A pairs per doc, looking for common finance phrases.
    Deterministic; avoids hallucination by only returning values we matched.
    """
    items: List[EvalItem] = []
    t = _norm(text).lower()

    # 1) Gross margin
    if "gross margin" in t:
        # capture `gross margin ... 38.2%` style
        snippet = next((s for s in t.split(". ") if "gross margin" in s), "")
        val = _first_match(RE_PERCENT, snippet) or _first_match(RE_PERCENT, t)
        if val:
            items.append(EvalItem(
                question="What was the gross margin reported?",
                answers=[val],
                references=[f"samples://{doc_id}"],
                meta={"topic": "profitability", "metric": "gross_margin"}
            ))

    # 2) Revenue with period hints
    if "revenue" in t:
        snippet = next((s for s in t.split(". ") if "revenue" in s), "")
        val = _first_match(RE_MONEY, snippet) or _first_match(RE_MONEY, t)
        if val:
            items.append(EvalItem(
                question="What revenue figure was reported?",
                answers=[val],
                references=[f"samples://{doc_id}"],
                meta={"topic": "topline", "metric": "revenue"}
            ))
        # YoY %
        if "yoy" in t or "year over year" in t:
            pct = _first_match(RE_PERCENT, snippet) or _first_match(RE_PERCENT, t)
            if pct:
                items.append(EvalItem(
                    question="What was the year-over-year revenue change?",
                    answers=[pct],
                    references=[f"samples://{doc_id}"],
                    meta={"topic": "growth", "metric": "revenue_yoy"}
                ))

    # 3) Net debt / EBITDA ratio
    if "net debt" in t and "ebitda" in t:
        val = _first_match(RE_RATIO, t)
        if val:
            items.append(EvalItem(
                question="What was the net debt to EBITDA ratio?",
                answers=[val],
                references=[f"samples://{doc_id}"],
                meta={"topic": "leverage", "metric": "net_debt_to_ebitda"}
            ))

    # 4) Scope 1 emissions
    if "scope 1" in t or "scope i" in t:
        snippet = next((s for s in t.split(". ") if "scope 1" in s or "scope i" in s), "")
        # Try to capture a number possibly followed by tCO2e
        m = re.search(r"(\d[\d,\.]*)\s*(?:tco2e|tco₂e|tonnes|tons)", snippet, re.I) or \
            re.search(r"(\d[\d,\.]*)\s*(?:tco2e|tco₂e|tonnes|tons)", t, re.I)
        if m:
            items.append(EvalItem(
                question="What were the Scope 1 emissions reported?",
                answers=[m.group(1).replace(",", "")],
                references=[f"samples://{doc_id}"],
                meta={"topic": "esg", "metric": "scope1_emissions", "unit": "tCO2e"}
            ))

    # 5) EPS
    if "eps" in t:
        snippet = next((s for s in t.split(". ") if "eps" in s), "")
        val = _first_match(RE_NUMBER, snippet) or _first_match(RE_NUMBER, t)
        if val:
            items.append(EvalItem(
                question="What EPS value was reported?",
                answers=[val],
                references=[f"samples://{doc_id}"],
                meta={"topic": "earnings", "metric": "eps"}
            ))

    # Limit per doc
    return items[:max_per_doc]


def synthesize_from_samples(base: Path, max_per_doc: int, max_total: int) -> List[EvalItem]:
    items: List[EvalItem] = []
    for doc_id, text in _scan_docs(base):
        cand = _make_items_from_text(doc_id, text, max_per_doc=max_per_doc)
        for it in cand:
            items.append(it)
            if len(items) >= max_total:
                return items
    return items


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Export a small finance eval set (JSONL).")
    ap.add_argument("--seeds", help="Path to seeds file (.jsonl or .csv). If omitted, use heuristic mode.")
    ap.add_argument("--samples", default="./data/samples", help="Folder to scan for docs in heuristic mode")
    ap.add_argument("--max-per-doc", type=int, default=4, help="Max heuristic Q/As per document")
    ap.add_argument("--max-total", type=int, default=50, help="Max total heuristic items")
    ap.add_argument("--out", default="./packages/eval/datasets/finance_golden.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.seeds:
        items = load_seeds(Path(args.seeds))
        if not items:
            print(f"No valid rows found in seeds: {args.seeds}")
            return
        rows = [asdict(x) for x in items]
        write_jsonl(rows, out_path)
        print(f"Wrote {len(rows)} items → {out_path}")
        return

    # Heuristic mode
    base = Path(args.samples).resolve()
    items = synthesize_from_samples(base, max_per_doc=int(args.max_per_doc), max_total=int(args.max_total))
    if not items:
        print(f"No items synthesized from: {base}")
        print("Tip: add a few TXT/MD/PDF/JSONL files under ./data/samples or provide --seeds.")
        return
    rows = [asdict(x) for x in items]
    write_jsonl(rows, out_path)
    print(f"Wrote {len(rows)} heuristic items → {out_path}")


if __name__ == "__main__":
    main()
