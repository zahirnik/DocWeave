# examples/02_build_index_pgvector.py
"""
Build a pgvector index from local files (tutorial-simple).

What this does
--------------
- Reads text-like files from ./data/samples (PDF/DOCX/TXT/MD/JSONL[.text]).
- Chunks text into small passages (fallback char-based chunker shown here).
- Embeds chunks with our EmbeddingClient (OpenAI if key present; else fallback).
- Upserts into Postgres (pgvector) using the repo's vector store interface.

Requirements
------------
- A Postgres database with the pgvector extension installed.
- `PGVECTOR_URL` env var set (e.g., postgresql+psycopg://user:pass@localhost:5432/convai)

Run
---
  python -m examples.02_build_index_pgvector --collection finance_demo

Tip
---
If you haven’t created the table/collection yet, the vector store helper will do it.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from packages.core.config import get_settings
from packages.retriever.embeddings import EmbeddingClient

# Prefer our loaders/chunker if available; provide tiny fallbacks to keep the script runnable.

# Loaders (PDF/DOCX/PLAIN/JSONL)
try:
    from packages.ingestion.loaders_pdf import pdf_to_text  # type: ignore
except Exception:
    def pdf_to_text(path: str) -> str:
        try:
            import PyPDF2  # type: ignore
        except Exception as e:
            raise RuntimeError("PyPDF2 not installed; install it or add packages.ingestion.loaders_pdf") from e
        out = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)  # type: ignore
            for p in reader.pages:
                try:
                    out.append(p.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(out).strip()

try:
    from packages.ingestion.loaders_docx import docx_to_text  # type: ignore
except Exception:
    def docx_to_text(path: str) -> str:
        try:
            import docx  # type: ignore
        except Exception as e:
            raise RuntimeError("python-docx not installed; install it or add packages.ingestion.loaders_docx") from e
        d = docx.Document(path)  # type: ignore
        return "\n".join(p.text for p in d.paragraphs)

def txt_to_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def md_to_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def jsonl_to_texts(path: str, text_key: str = "text") -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                t = d.get(text_key)
                if isinstance(t, str):
                    out.append(t)
            except Exception:
                continue
    return out

# Chunker
try:
    from packages.retriever.chunking import chunk_text  # type: ignore
except Exception:
    def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 120) -> List[str]:
        """
        Simple char-based chunker (fallback). Token-aware chunker lives in packages.retriever.chunking.
        """
        s = " ".join((text or "").replace("-\n", "").replace("\n", " ").split())
        if not s:
            return []
        chunks: List[str] = []
        i, n = 0, len(s)
        step = max(1, max_chars - overlap)
        while i < n:
            j = min(n, i + max_chars)
            chunks.append(s[i:j])
            i += step
        return chunks

# Vector store (pgvector)
try:
    from packages.retriever.vectorstores.pgvector_store import PgVectorStore  # type: ignore
except Exception as e:
    PgVectorStore = None  # type: ignore


# ---------------------------
# Helpers
# ---------------------------

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".jsonl"}

def scan_files(base: Path) -> List[Path]:
    if not base.exists():
        return []
    out: List[Path] = []
    for pat in ["*.pdf", "*.docx", "*.txt", "*.md", "*.jsonl"]:
        out.extend(base.rglob(pat))
    # stable order
    out = sorted(set(out))
    return out

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, str]

def load_and_chunk(path: Path, *, doc_id_prefix: str = "") -> List[Chunk]:
    sfx = path.suffix.lower()
    base_id = f"{doc_id_prefix}{path.stem}"
    chunks: List[Chunk] = []

    if sfx == ".pdf":
        txt = pdf_to_text(str(path))
        for k, c in enumerate(chunk_text(txt)):
            chunks.append(Chunk(doc_id=base_id, chunk_id=f"{base_id}#p{k}", text=c, metadata={"source": str(path)}))
        return chunks

    if sfx == ".docx":
        txt = docx_to_text(str(path))
        for k, c in enumerate(chunk_text(txt)):
            chunks.append(Chunk(doc_id=base_id, chunk_id=f"{base_id}#p{k}", text=c, metadata={"source": str(path)}))
        return chunks

    if sfx == ".txt":
        txt = txt_to_text(str(path))
        for k, c in enumerate(chunk_text(txt)):
            chunks.append(Chunk(doc_id=base_id, chunk_id=f"{base_id}#p{k}", text=c, metadata={"source": str(path)}))
        return chunks

    if sfx == ".md":
        txt = md_to_text(str(path))
        for k, c in enumerate(chunk_text(txt)):
            chunks.append(Chunk(doc_id=base_id, chunk_id=f"{base_id}#p{k}", text=c, metadata={"source": str(path)}))
        return chunks

    if sfx == ".jsonl":
        texts = jsonl_to_texts(str(path))
        for i, t in enumerate(texts):
            for k, c in enumerate(chunk_text(t)):
                cid = f"{base_id}#r{i}p{k}"
                chunks.append(Chunk(doc_id=base_id, chunk_id=cid, text=c, metadata={"source": f"{path}#row{i}"}))
        return chunks

    # unsupported (shouldn't happen given SUPPORTED)
    return []


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pgvector index from ./data/samples")
    parser.add_argument("--collection", required=True, help="Collection name (e.g., finance_demo)")
    parser.add_argument("--samples", default="./data/samples", help="Directory to scan for files")
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    cfg = get_settings()
    if not cfg.pgvector_url:
        print("ERROR: PGVECTOR_URL is not set. Example:")
        print("  export PGVECTOR_URL=postgresql+psycopg://user:pass@localhost:5432/convai")
        return
    if PgVectorStore is None:
        print("ERROR: Vector store module not available. You need packages.retriever.vectorstores.pgvector_store.")
        return

    base = Path(args.samples).resolve()
    files = [p for p in scan_files(base) if p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No supported files found under: {base}")
        print("Tip: drop a few PDFs/TXTs/MD/JSONL in ./data/samples and re-run.")
        return

    # 1) Load + chunk
    all_chunks: List[Chunk] = []
    for p in files:
        cs = load_and_chunk(p, doc_id_prefix="")
        if cs:
            all_chunks.extend(cs)

    if not all_chunks:
        print("No chunks produced. Nothing to index.")
        return

    print(f"Files: {len(files)} | Chunks: {len(all_chunks)}")

    # 2) Embeddings
    embedder = EmbeddingClient(provider="auto", model_alias="mini")
    texts = [c.text for c in all_chunks]
    vecs: List[List[float]] = embedder.embed(texts, batch_size=int(args.batch))

    # 3) Upsert to pgvector
    store = PgVectorStore(dsn=cfg.pgvector_url, collection=args.collection, create_if_missing=True)
    # Prepare payloads
    items = []
    for c, v in zip(all_chunks, vecs):
        items.append({
            "id": c.chunk_id,
            "doc_id": c.doc_id,
            "vector": v,
            "text": c.text,
            "metadata": c.metadata,
        })
    # Upsert in batches
    B = 500
    for i in range(0, len(items), B):
        batch_items = items[i : i + B]
        store.upsert(batch_items)
        print(f"Upserted {i + len(batch_items)} / {len(items)}")

    # 4) Test a simple search
    q = "What was ACME's gross margin in 2024?"
    qv = embedder.embed([q])[0]
    hits = store.search(qv, top_k=5)
    print("\nSearch test:")
    for h in hits:
        print(f"- ({h.get('score'):.3f}) {h.get('id')}  ← {h.get('metadata',{}).get('source','')}")
        # Optional: print a snippet
        t = (h.get("text") or "").strip().replace("\n", " ")
        print(f"  {t[:160]}{'…' if len(t)>160 else ''}")

    print("\nDone.")

if __name__ == "__main__":
    main()
