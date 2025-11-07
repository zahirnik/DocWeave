# examples/01_ingest_local_files.py
"""
Ingest local finance files (PDF/CSV/JSON/TXT) from ./data/samples and print a report.

What this does (tutorial-simple):
- Scans ./data/samples for a few common formats.
- Uses our ingestion loaders if present; otherwise falls back to tiny, safe readers.
- Normalises text lightly and returns a per-file summary (chars/rows, errors).
- Prints a compact report and writes a JSONL to ./data/outputs/ingest_report.jsonl.

You can copy this as a starting point for your own ingestion CLI.

Run:
  python -m examples.01_ingest_local_files
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Try to use the package loaders (preferred); if missing, use tiny fallbacks ---

# PDF
try:
    from packages.ingestion.loaders_pdf import pdf_to_text  # type: ignore
except Exception:
    def pdf_to_text(path: str) -> str:
        # Lightweight fallback using PyPDF2 if available
        try:
            import PyPDF2  # type: ignore
        except Exception as e:
            raise RuntimeError("PDF loader not available (install packages or PyPDF2).") from e
        out = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)  # type: ignore
            for page in reader.pages:
                try:
                    out.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(out).strip()

# DOCX (not required for this example, but we provide a fallback)
try:
    from packages.ingestion.loaders_docx import docx_to_text  # type: ignore
except Exception:
    def docx_to_text(path: str) -> str:
        try:
            import docx  # python-docx  # type: ignore
        except Exception as e:
            raise RuntimeError("DOCX loader not available (install python-docx).") from e
        d = docx.Document(path)  # type: ignore
        return "\n".join(p.text for p in d.paragraphs)

# CSV/XLSX/TSV
try:
    from packages.ingestion.loaders_tabular import load_csv_like, load_xlsx  # type: ignore
except Exception:
    import pandas as _pd  # soft-dep for fallback
    def load_csv_like(path: str) -> "Tuple[_pd.DataFrame, Dict]":
        df = _pd.read_csv(path) if path.lower().endswith(".csv") else _pd.read_table(path)
        return df, {"rows": int(df.shape[0]), "cols": int(df.shape[1])}

    def load_xlsx(path: str) -> "Tuple[Dict[str, _pd.DataFrame], Dict]":
        xls = _pd.read_excel(path, sheet_name=None)
        meta = {name: {"rows": int(df.shape[0]), "cols": int(df.shape[1])} for name, df in xls.items()}
        return xls, {"sheets": len(xls), "by_sheet": meta}

# JSON/JSONL
try:
    from packages.ingestion.loaders_json import load_json_like  # type: ignore
except Exception:
    def load_json_like(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith(".jsonl"):
                return {"records": [json.loads(line) for line in f if line.strip()]}
            return {"json": json.load(f)}

# Normalisers (optional)
try:
    from packages.ingestion.normalizers import normalize_text  # type: ignore
except Exception:
    def normalize_text(s: str) -> str:
        # Minimal whitespace + de-hyphenation at line breaks
        return " ".join((s or "").replace("-\n", "").replace("\n", " ").split())


# ---------------------------
# Data model
# ---------------------------

@dataclass
class IngestResult:
    path: str
    kind: str  # pdf|csv|xlsx|json|jsonl|txt|tsv|unknown
    ok: bool
    error: Optional[str] = None
    chars: Optional[int] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    sheets: Optional[int] = None

    @staticmethod
    def from_text(path: str, text: str) -> "IngestResult":
        return IngestResult(
            path=path,
            kind=_kind_from_suffix(path),
            ok=True,
            error=None,
            chars=len(text or ""),
        )

    @staticmethod
    def from_table(path: str, rows: int, cols: int) -> "IngestResult":
        return IngestResult(
            path=path,
            kind=_kind_from_suffix(path),
            ok=True,
            rows=int(rows),
            cols=int(cols),
        )

    @staticmethod
    def from_xlsx(path: str, sheets: int) -> "IngestResult":
        return IngestResult(
            path=path,
            kind="xlsx",
            ok=True,
            sheets=int(sheets),
        )


# ---------------------------
# Helpers
# ---------------------------

_TEXT_SUFFIXES = {".pdf", ".docx", ".txt"}
_TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx"}
_JSON_SUFFIXES = {".json", ".jsonl"}

def _kind_from_suffix(path: str) -> str:
    s = Path(path).suffix.lower()
    if s == ".pdf": return "pdf"
    if s == ".docx": return "docx"
    if s == ".txt": return "txt"
    if s == ".csv": return "csv"
    if s == ".tsv": return "tsv"
    if s == ".xlsx": return "xlsx"
    if s == ".json": return "json"
    if s == ".jsonl": return "jsonl"
    return "unknown"

def _scan_samples(base: Path) -> List[Path]:
    if not base.exists():
        return []
    pats = ["*.pdf", "*.csv", "*.tsv", "*.xlsx", "*.json", "*.jsonl", "*.txt", "*.docx"]
    out: List[Path] = []
    for pat in pats:
        out.extend(base.rglob(pat))
    # de-dup and sort for deterministic order
    return sorted(set(out))

def _short(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)


# ---------------------------
# Ingestion core
# ---------------------------

def ingest_path(path: Path) -> IngestResult:
    sfx = path.suffix.lower()
    try:
        # TEXT-LIKE
        if sfx == ".pdf":
            text = pdf_to_text(str(path))
            text = normalize_text(text)
            return IngestResult.from_text(str(path), text)
        if sfx == ".docx":
            text = docx_to_text(str(path))
            text = normalize_text(text)
            return IngestResult.from_text(str(path), text)
        if sfx == ".txt":
            text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
            return IngestResult.from_text(str(path), text)

        # TABULAR
        if sfx == ".csv" or sfx == ".tsv":
            df, meta = load_csv_like(str(path))
            return IngestResult.from_table(str(path), meta.get("rows", df.shape[0]), meta.get("cols", df.shape[1]))
        if sfx == ".xlsx":
            sheets, meta = load_xlsx(str(path))
            return IngestResult.from_xlsx(str(path), meta.get("sheets", len(sheets)))

        # JSON / JSONL
        if sfx in (".json", ".jsonl"):
            data = load_json_like(str(path))
            # rough size proxy
            rows = len(data.get("records", [])) if isinstance(data, dict) else 0
            return IngestResult(path=str(path), kind=_kind_from_suffix(str(path)), ok=True, rows=rows)

        # Unknown → treat as text
        text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
        return IngestResult.from_text(str(path), text)

    except Exception as e:
        return IngestResult(path=str(path), kind=_kind_from_suffix(str(path)), ok=False, error=str(e)[:400])


def print_report(results: List[IngestResult], base: Path) -> None:
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]

    print("\nIngestion report")
    print("────────────────")
    print(f"Base directory : {base}")
    print(f"Files scanned  : {len(results)}")
    print(f"OK             : {len(ok)}")
    print(f"Failed         : {len(bad)}")

    if ok:
        print("\nSuccessful:")
        for r in ok:
            extra = []
            if r.chars is not None: extra.append(f"chars={r.chars}")
            if r.rows is not None and r.cols is not None: extra.append(f"rows={r.rows}, cols={r.cols}")
            if r.sheets is not None: extra.append(f"sheets={r.sheets}")
            print(f"  - [{r.kind:<5}] {_short(Path(r.path), base)}  ({', '.join(extra)})")

    if bad:
        print("\nFailed:")
        for r in bad:
            print(f"  - [{r.kind:<5}] {_short(Path(r.path), base)}  ERROR: {r.error}")


def write_jsonl(results: List[IngestResult], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return out_path


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    base = Path("./data/samples").resolve()
    files = _scan_samples(base)
    if not files:
        print(f"No sample files found under: {base}")
        print("Tip: drop a few test PDFs/CSVs/TXTs into ./data/samples and re-run.")
        return

    results: List[IngestResult] = []
    for p in files:
        res = ingest_path(p)
        results.append(res)

    print_report(results, base)
    out = write_jsonl(results, Path("./data/outputs/ingest_report.jsonl"))
    print(f"\nJSONL report written to: {out}")

if __name__ == "__main__":
    main()
