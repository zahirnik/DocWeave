# scripts/build_index.py
"""
Build a vector index from local files (tutorial-clear, dependency-light).

What this script does
---------------------
- Scans one or more sources (folders or files) for text-like content:
  PDF, DOCX, TXT/MD, JSONL({"text": ...}), CSV/XLSX (as rows → joined text).
- Chunks text into small passages (uses repo chunker if available, else a fallback).
- Embeds chunks with our EmbeddingClient (OpenAI if key present; else a fallback).
- Upserts the chunks into your chosen vector store (pgvector | qdrant | chroma).

Why this exists
---------------
It’s the end-to-end “one command” you’ll run most often while developing:
  - drop docs into ./data/samples
  - choose a collection name
  - run this script to (re)build your index

Examples
--------
# Minimal (pgvector recommended)
export PGVECTOR_URL="postgresql+psycopg://user:pass@localhost:5432/convai"
python -m scripts.build_index --collection finance_demo --source ./data/samples

# Chroma (local dev)
python -m scripts.build_index --collection finance_demo --store chroma --source ./data/samples

# Qdrant (remote/self-hosted)
export QDRANT_URL="http://localhost:6333"
python -m scripts.build_index --collection finance_demo --store qdrant --source ./data/samples

Flags (friendly defaults)
-------------------------
--source <path>           # repeatable; files or directories (globs allowed)
--collection <name>       # required; logical index/namespace
--store {pgvector|qdrant|chroma}  (default: pgvector if configured; else chroma)
--batch <int>             # embed batch size (default: 64)
--overwrite               # drop & recreate collection if supported (careful!)
--max-filesize-mb <int>   # skip files larger than N MB (default: 50)
--report-dir <path>       # where to write JSON/CSV report (default: ./data/outputs)

Outputs
-------
- A JSONL run report: <report-dir>/index_report_<timestamp>.jsonl
- A CSV summary:       <report-dir>/index_summary_<timestamp>.csv

Notes
-----
- Network URLs are currently ignored (we focus on local ingestion to keep it simple).
- If repo’s specialised loaders aren’t installed, robust fallbacks are used.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import re
import sys
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------
# Repo imports (with graceful fallbacks)
# ---------------------------

from packages.core.config import get_settings
from packages.retriever.embeddings import EmbeddingClient

# loaders (prefer repo modules; else fallbacks below)
try:
    # [NOTE] Use the repo’s stronger loader
    from packages.ingestion.loaders_pdf import load_pdf_text as pdf_to_text  # type: ignore
except Exception:
    def pdf_to_text(path: str) -> str:
        try:
            import PyPDF2  # type: ignore
        except Exception as e:
            raise RuntimeError("PyPDF2 not available; install it or provide packages.ingestion.loaders_pdf") from e
        out = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)  # type: ignore
            for page in reader.pages:
                try:
                    out.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(out)

try:
    from packages.ingestion.loaders_docx import docx_to_text  # type: ignore
except Exception:
    def docx_to_text(path: str) -> str:
        try:
            import docx  # type: ignore
        except Exception as e:
            raise RuntimeError("python-docx not available; install it or provide packages.ingestion.loaders_docx") from e
        d = docx.Document(path)  # type: ignore
        return "\n".join(p.text for p in d.paragraphs)

try:
    from packages.ingestion.loaders_tabular import load_csv_like, load_xlsx  # type: ignore
except Exception:
    # minimal pandas fallback (optional)
    def load_csv_like(path: str):
        import pandas as pd  # type: ignore
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            # .tsv or unknown → let pandas infer
            df = pd.read_table(path)
        meta = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
        return df, meta

    def load_xlsx(path: str):
        import pandas as pd  # type: ignore
        sheets = pd.read_excel(path, sheet_name=None)
        meta = {"sheets": len(sheets), "by_sheet": {k: {"rows": int(v.shape[0]), "cols": int(v.shape[1])} for k, v in sheets.items()}}
        return sheets, meta

try:
    from packages.ingestion.loaders_json import load_json_like  # type: ignore
except Exception:
    def load_json_like(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith(".jsonl"):
                return {"records": [json.loads(line) for line in f if line.strip()]}
            return {"json": json.load(f)}

try:
    from packages.retriever.chunking import chunk_text  # type: ignore
except Exception:
    def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> List[str]:
        """
        Fallback char-based chunker (token-aware version lives in packages.retriever.chunking).
        """
        s = " ".join((text or "").replace("-\n", "").replace("\n", " ").split())
        if not s:
            return []
        out: List[str] = []
        i, n = 0, len(s)
        step = max(1, max_chars - overlap)
        while i < n:
            out.append(s[i : min(n, i + max_chars)])
            i += step
        return out

# Vector stores (optional)
try:
    from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # type: ignore
except Exception:
    PgVectorStore = None  # type: ignore
try:
    from packages.retriever.vectorstores.qdrant_store import QdrantStore  # type: ignore
except Exception:
    QdrantStore = None  # type: ignore
try:
    from packages.retriever.vectorstores.chroma_store import ChromaStore  # type: ignore
except Exception:
    ChromaStore = None  # type: ignore


# ---------------------------
# Data models
# ---------------------------

@dataclass
class ChunkItem:
    id: str
    doc_id: str
    text: str
    vector: Optional[List[float]] = None
    metadata: Dict[str, str] = None  # noqa

@dataclass
class FileReport:
    path: str
    kind: str
    ok: bool
    chunks: int = 0
    chars: int = 0
    error: Optional[str] = None

@dataclass
class RunSummary:
    collection: str
    store: str
    files_scanned: int
    files_ok: int
    files_failed: int
    total_chunks: int
    created_at: str


# ---------------------------
# Helpers
# ---------------------------

TEXT_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx"}
JSON_SUFFIXES = {".json", ".jsonl"}

def _kind_from_suffix(path: str) -> str:
    s = Path(path).suffix.lower()
    if s in TEXT_SUFFIXES: return s.lstrip(".")
    if s in TABULAR_SUFFIXES: return s.lstrip(".")
    if s in JSON_SUFFIXES: return s.lstrip(".")
    return "unknown"

def _normalize_text(s: str) -> str:
    return " ".join((s or "").replace("-\n", "").replace("\n", " ").split())

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def _scan_sources(sources: List[str], max_mb: int) -> List[Path]:
    out: List[Path] = []
    lim = max_mb * 1024 * 1024
    for src in sources:
        p = Path(src)
        if "*" in src or "?" in src or ("[" in src and "]" in src):
            for m in sorted(Path().glob(src)):
                if m.is_file() and m.stat().st_size <= lim:
                    out.append(m.resolve())
            continue
        if p.is_dir():
            for pat in ["*.pdf", "*.docx", "*.txt", "*.md", "*.csv", "*.tsv", "*.xlsx", "*.json", "*.jsonl"]:
                for f in sorted(p.rglob(pat)):
                    try:
                        if f.is_file() and f.stat().st_size <= lim:
                            out.append(f.resolve())
                    except Exception:
                        continue
        elif p.is_file():
            try:
                if p.stat().st_size <= lim:
                    out.append(p.resolve())
            except Exception:
                continue
        else:
            # best-effort: ignore URLs (out of scope for this script)
            pass
    # stable order, unique
    return sorted(set(out))


# ---------------------------
# Load & chunk
# ---------------------------

def load_text_from_file(path: Path) -> str:
    sfx = path.suffix.lower()
    if sfx == ".pdf":
        return _normalize_text(pdf_to_text(str(path)))
    if sfx == ".docx":
        return _normalize_text(docx_to_text(str(path)))
    if sfx in {".txt", ".md"}:
        return _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    if sfx in {".json", ".jsonl"}:
        data = load_json_like(str(path))
        # promote common fields into a big blob
        texts: List[str] = []
        if isinstance(data, dict) and "records" in data:
            for r in data["records"]:
                t = r.get("text") or r.get("content") or r.get("body") or ""
                if isinstance(t, str) and t.strip():
                    texts.append(t)
        else:
            # fallback: stringify JSON
            texts.append(json.dumps(data, ensure_ascii=False))
        return _normalize_text("\n\n".join(texts))
    if sfx in {".csv", ".tsv"}:
        df, _meta = load_csv_like(str(path))
        # textify rows (keep small to avoid explosion)
        parts: List[str] = []
        for i, row in enumerate(df.head(500).itertuples(index=False)):
            parts.append(" | ".join(str(x) for x in row))
        return _normalize_text("\n".join(parts))
    if sfx == ".xlsx":
        sheets, _meta = load_xlsx(str(path))
        parts: List[str] = []
        for name, df in list(sheets.items())[:4]:
            parts.append(f"## Sheet: {name}")
            for i, row in enumerate(df.head(300).itertuples(index=False)):
                parts.append(" | ".join(str(x) for x in row))
        return _normalize_text("\n".join(parts))
    # unknown → try text
    return _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def to_chunks(path: Path, text: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Returns a list of (chunk_text, metadata) for a given file.
    """
    # [FIX] Support repo token-aware chunker (list[dict]) and graceful fallback.
    out: List[Tuple[str, Dict[str, str]]] = []
    try:
        chs = chunk_text(text)  # repo API → list[dict] with "text", "tokens", maybe "title"
        if chs:
            if isinstance(chs[0], dict):
                for i, d in enumerate(chs):
                    t = (d.get("text") or "").strip()
                    if not t:
                        continue
                    meta = {"source": str(path), "chunk": str(i)}
                    if d.get("title"):
                        meta["title"] = d["title"]
                    out.append((t, meta))
                return out
            if isinstance(chs[0], str):
                for i, t in enumerate(chs):
                    t = (t or "").strip()
                    if not t:
                        continue
                    out.append((t, {"source": str(path), "chunk": str(i)}))
                return out
    except TypeError:
        # If an older/local fallback with a different signature was imported, fall through.
        pass

    # Fallback: simple char-based sliding windows
    s = " ".join((text or "").replace("-\n", "").replace("\n", " ").split())
    if not s:
        return []
    max_chars, overlap = 1200, 150
    step = max(1, max_chars - overlap)
    i = idx = 0
    n = len(s)
    while i < n:
        ch = s[i:min(n, i + max_chars)]
        out.append((ch, {"source": str(path), "chunk": str(idx)}))
        idx += 1
        i += step
    return out


# ---------------------------
# Vector store factory
# ---------------------------

def open_store(store: str, collection: str, overwrite: bool, cfg) -> object:
    """
    Open the requested vector store. Returns an object with:
      - upsert(items: List[dict]) -> None
      - (optional) drop_collection() if overwrite=True and supported
    """
    store = (store or "").lower()
    if store not in {"", "pgvector", "qdrant", "chroma"}:
        raise ValueError("--store must be one of {pgvector|qdrant|chroma}")

    # Auto-pick if not specified
    if not store:
        if getattr(cfg, "pgvector_url", None):
            store = "pgvector"
        elif getattr(cfg, "qdrant_url", None):
            store = "qdrant"
        else:
            store = "chroma"

    if store == "pgvector":
        if PgVectorStore is None or not getattr(cfg, "pgvector_url", None):
            raise RuntimeError("PGVECTOR_URL not set or PgVectorStore unavailable.")
        s = PgVectorStore(dsn=cfg.pgvector_url, collection=collection, create_if_missing=True)
        if overwrite and hasattr(s, "drop_collection"):
            s.drop_collection()
            s = PgVectorStore(dsn=cfg.pgvector_url, collection=collection, create_if_missing=True)
        return s

    if store == "qdrant":
        if QdrantStore is None or not getattr(cfg, "qdrant_url", None):
            raise RuntimeError("QDRANT_URL not set or QdrantStore unavailable.")
        s = QdrantStore(url=cfg.qdrant_url, collection=collection, create_if_missing=True)
        if overwrite and hasattr(s, "drop_collection"):
            s.drop_collection()
            s = QdrantStore(url=cfg.qdrant_url, collection=collection, create_if_missing=True)
        return s

    # ---------------------------
    # CHROMA ADAPTER  [CHANGED]
    # ---------------------------
    if ChromaStore is None:
        raise RuntimeError("ChromaStore unavailable. Install or include packages.retriever.vectorstores.chroma_store.")

    # Read path & prefix from env (stable names)
    chroma_path = os.getenv("CHROMA_PATH", "./.chroma") or "./.chroma"
    collection_prefix = os.getenv("COLLECTION_PREFIX", "rag") or "rag"

    # New ChromaStore API: ChromaStore(persist_dir, collection_prefix) + upsert_texts(collection, items)
    base = ChromaStore(persist_dir=chroma_path, collection_prefix=collection_prefix)

    class _ChromaAdapter:
        """Minimal adapter to match this script's store interface."""
        def __init__(self, base, collection):
            self.base = base
            self.collection = collection

        def upsert(self, items: List[Dict]) -> int:
            payload = []
            for it in items:
                payload.append({
                    "id": it.get("id"),
                    "text": it["text"],
                    "metadata": it.get("metadata", {}),
                    "embedding": it["vector"],
                })
            return self.base.upsert_texts(self.collection, payload)

        def drop_collection(self) -> bool:
            try:
                return bool(self.base.delete_collection(self.collection))
            except Exception:
                return False

    s = _ChromaAdapter(base, collection)
    if overwrite and hasattr(s, "drop_collection"):
        s.drop_collection()
        s = _ChromaAdapter(base, collection)
    return s


# ---------------------------
# Report writers
# ---------------------------

def write_jsonl(rows: Iterable[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path

def write_csv(rows: Iterable[dict], path: Path, field_order: Optional[List[str]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fnames = field_order or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fnames})
    return path


# ---------------------------
# Main build pipeline
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a vector index from local files.")
    ap.add_argument("--source", action="append", required=True, help="File/dir/glob. Repeatable.")
    ap.add_argument("--collection", required=True, help="Collection/index name (e.g., finance_demo)")
    ap.add_argument("--store", choices=["pgvector", "qdrant", "chroma"], default="", help="Vector store backend")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size (default: 64)")
    ap.add_argument("--overwrite", action="store_true", help="Drop & recreate collection if supported")
    ap.add_argument("--max-filesize-mb", type=int, default=50, help="Skip files > N MB (default: 50)")
    ap.add_argument("--report-dir", default="./data/outputs", help="Where to write JSON/CSV reports")
    args = ap.parse_args(argv)

    cfg = get_settings()
    sources = args.source
    files = _scan_sources(sources, max_mb=int(args.max_filesize_mb))
    if not files:
        print("No files found to ingest. Check --source paths and --max-filesize-mb.")
        return 1

    # Open store
    try:
        store = open_store(args.store, args.collection, args.overwrite, cfg)
    except Exception as e:
        print(f"Failed to open vector store: {e}")
        return 2

    # Prepare embedder
    embedder = EmbeddingClient(provider="auto", model_alias="mini")

    # Dedup within this run (content hash)
    seen_hashes: set[str] = set()

    # Collect reports and items
    file_reports: List[FileReport] = []
    items_to_upsert: List[Dict] = []

    for path in files:
        kind = _kind_from_suffix(str(path))
        try:
            text = load_text_from_file(path)
            if not text.strip():
                file_reports.append(FileReport(path=str(path), kind=kind, ok=False, error="empty text"))
                continue

            # Chunks
            chunks = to_chunks(path, text)

            # Dedup chunks by content hash (run-scoped)
            pre = len(chunks)
            filtered: List[Tuple[str, Dict[str, str]]] = []
            for ch_text, meta in chunks:
                h = _hash(ch_text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                filtered.append((ch_text, meta))

            chunks = filtered
            doc_id = Path(path).stem

            # Prepare vectorless items; embed later in batches
            for i, (ch_text, meta) in enumerate(chunks):
                cid = f"{doc_id}#p{i}"
                items_to_upsert.append({
                    "id": cid,
                    "doc_id": doc_id,
                    "text": ch_text,
                    "vector": None,   # fill later
                    "metadata": meta,
                })

            file_reports.append(FileReport(
                path=str(path),
                kind=kind,
                ok=True,
                chunks=len(chunks),
                chars=len(text),
            ))
        except Exception as e:
            file_reports.append(FileReport(path=str(path), kind=kind, ok=False, error=str(e)[:400]))

    # Any items?
    if not items_to_upsert:
        print("No chunks produced after filtering. Nothing to upsert.")
        return 0

    # Embed in batches
    B = max(1, int(args.batch))
    texts = [it["text"] for it in items_to_upsert]
    vectors: List[List[float]] = []
    for i in range(0, len(texts), B):
        batch = texts[i : i + B]
        # [FIX-EMBED] EmbeddingClient.embed() takes only `texts`
        vecs = embedder.embed(batch)
        vectors.extend(vecs)
        print(f"Embedded {i + len(batch)} / {len(texts)}")

    # Attach vectors and upsert in store-sized batches
    for it, v in zip(items_to_upsert, vectors):
        it["vector"] = v

    U = 500
    for i in range(0, len(items_to_upsert), U):
        part = items_to_upsert[i : i + U]
        store.upsert(part)  # store implements uniform upsert(items)
        print(f"Upserted {i + len(part)} / {len(items_to_upsert)}")

    # Reports
    created_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    summary = RunSummary(
        collection=args.collection,
        store=(args.store or "auto"),
        files_scanned=len(files),
        files_ok=sum(1 for r in file_reports if r.ok),
        files_failed=sum(1 for r in file_reports if not r.ok),
        total_chunks=sum(r.chunks for r in file_reports),
        created_at=created_at,
    )

    report_dir = Path(args.report_dir).resolve()
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jsonl_path = report_dir / f"index_report_{stamp}.jsonl"
    csv_path = report_dir / f"index_summary_{stamp}.csv"

    # write per-file JSONL
    write_jsonl((asdict(r) for r in file_reports), jsonl_path)
    # write summary CSV
    write_csv([asdict(summary)], csv_path, field_order=list(asdict(summary).keys()))

    print("\nSummary")
    print("───────")
    print(json.dumps(asdict(summary), indent=2))
    print(f"\nWrote:\n- {jsonl_path}\n- {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
