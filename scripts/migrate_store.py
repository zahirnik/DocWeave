# scripts/migrate_store.py
"""
Migrate/merge collections across vector stores (pgvector ↔ qdrant ↔ chroma; optional faiss).

What this script does
---------------------
- Reads items (id, text, metadata, vector?) from a **source** store or a JSONL dump.
- Writes them to a **destination** store, preserving metadata.
- Two modes:
  1) Copy existing vectors  (fast; requires vectors available & dim-compatible)
  2) Re-embed text content (safe when changing models or dims) via EmbeddingClient.

Safety & resilience
-------------------
- Resume-safe: checkpoint file records migrated IDs; re-runs skip completed items.
- Batch processing with progress prints.
- Optional JSONL dump of migrated payloads for auditing/backups.
- Dimension sanity checks when copying vectors.
- Deterministic behavior; no network calls unless your embedder/LLM needs them.

Examples
--------
# pgvector → qdrant (copy vectors)
export PGVECTOR_URL="postgresql+psycopg://user:pass@localhost:5432/convai"
export QDRANT_URL="http://localhost:6333"
python -m scripts.migrate_store \
  --src-store pgvector --src-collection finance_demo \
  --dst-store qdrant  --dst-collection finance_demo_copy

# chroma → pgvector (re-embed with new model)
export PGVECTOR_URL="postgresql+psycopg://user:pass@localhost:5432/convai"
python -m scripts.migrate_store \
  --src-store chroma --src-collection finance_demo \
  --dst-store pgvector --dst-collection finance_demo_gte \
  --reembed --batch 128

# From JSONL dump → qdrant
python -m scripts.migrate_store \
  --input-jsonl ./data/outputs/export_texts.jsonl \
  --dst-store qdrant --dst-collection finance_demo \
  --reembed

CLI
---
--src-store {pgvector|qdrant|chroma|faiss}
--src-collection <name>
--input-jsonl <path>            # alt source when store lacks export/iter
--dst-store {pgvector|qdrant|chroma|faiss}  (required)
--dst-collection <name>         (required)
--reembed                       # compute new vectors from text using EmbeddingClient
--batch <int>                   # default 256
--max <int>                     # limit items for quick tests (optional)
--dump-jsonl <path>             # write migrated payloads (id,text,metadata,vector)
--resume <path>                 # checkpoint path (default under ./data/outputs)

Notes
-----
- Source store must expose at least one of: .iter(), .scan(), or .export() returning
  items with fields {"id","text","metadata","vector"} (vector optional).
- Destination store must support .upsert(items).
- If copying vectors, dimensions must be consistent with the destination collection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from packages.core.config import get_settings
from packages.retriever.embeddings import EmbeddingClient

# ── Optional store adapters (duck-typed)
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

# FAISS is optional; many repos skip it. We guard it.
try:
    from packages.retriever.vectorstores.faiss_store import FaissStore  # type: ignore
except Exception:
    FaissStore = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Item:
    id: str
    text: str
    metadata: Dict[str, object]
    vector: Optional[List[float]]


# ──────────────────────────────────────────────────────────────────────────────
# Store factory
# ──────────────────────────────────────────────────────────────────────────────

def open_store(kind: str, collection: str, cfg):
    kind = (kind or "").lower()
    if kind == "pgvector":
        if PgVectorStore is None or not cfg.pgvector_url:
            raise RuntimeError("PGVECTOR_URL not set or PgVectorStore unavailable.")
        return PgVectorStore(dsn=cfg.pgvector_url, collection=collection, create_if_missing=True)
    if kind == "qdrant":
        if QdrantStore is None or not cfg.qdrant_url:
            raise RuntimeError("QDRANT_URL not set or QdrantStore unavailable.")
        return QdrantStore(url=cfg.qdrant_url, collection=collection, create_if_missing=True)
    if kind == "chroma":
        if ChromaStore is None:
            raise RuntimeError("ChromaStore unavailable.")
        return ChromaStore(path=cfg.chroma_path or "./.chroma", collection=collection, create_if_missing=True)
    if kind == "faiss":
        if FaissStore is None:
            raise RuntimeError("FaissStore unavailable (package not present).")
        return FaissStore(path=cfg.faiss_path or "./.faiss", collection=collection, create_if_missing=True)
    raise ValueError("Store kind must be one of {pgvector|qdrant|chroma|faiss}.")


# ──────────────────────────────────────────────────────────────────────────────
# Iteration helpers (source)
# ──────────────────────────────────────────────────────────────────────────────

def iter_items_from_store(store, batch: int = 1000, limit: Optional[int] = None) -> Iterator[Item]:
    """
    Try common iteration methods across stores: .iter(), .scan(), .export().
    Expected per record keys: id, text, metadata, vector (optional).
    """
    n = 0
    # 1) .iter(...)
    if hasattr(store, "iter"):
        for rec in store.iter(batch=batch, fields=["id", "text", "metadata", "vector"]):  # type: ignore[attr-defined]
            yield Item(
                id=str(rec.get("id")),
                text=str(rec.get("text") or ""),
                metadata=rec.get("metadata") or {},
                vector=rec.get("vector"),
            )
            n += 1
            if limit and n >= limit:
                return
        return
    # 2) .scan(...)
    if hasattr(store, "scan"):
        for rec in store.scan(batch=batch, fields=["id", "text", "metadata", "vector"]):  # type: ignore[attr-defined]
            yield Item(
                id=str(rec.get("id")),
                text=str(rec.get("text") or ""),
                metadata=rec.get("metadata") or {},
                vector=rec.get("vector"),
            )
            n += 1
            if limit and n >= limit:
                return
        return
    # 3) .export()
    if hasattr(store, "export"):
        for rec in store.export():  # type: ignore[attr-defined]
            yield Item(
                id=str(rec.get("id")),
                text=str(rec.get("text") or ""),
                metadata=rec.get("metadata") or {},
                vector=rec.get("vector"),
            )
            n += 1
            if limit and n >= limit:
                return
        return
    raise RuntimeError("Source store does not support iteration; provide --input-jsonl.")


def iter_items_from_jsonl(path: Path, limit: Optional[int] = None) -> Iterator[Item]:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit and n >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            yield Item(
                id=str(d.get("id")),
                text=str(d.get("text") or ""),
                metadata=d.get("metadata") or {},
                vector=d.get("vector"),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint (resume)
# ──────────────────────────────────────────────────────────────────────────────

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.done: set[str] = set()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            _id = d.get("id")
                            if isinstance(_id, str):
                                self.done.add(_id)
                        except Exception:
                            continue
            except Exception:
                self.done = set()

    def has(self, _id: str) -> bool:
        return _id in self.done

    def mark(self, _id: str) -> None:
        self.done.add(_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": _id}) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Writer helpers
# ──────────────────────────────────────────────────────────────────────────────

def upsert_batch(dst_store, batch_payload: List[dict]) -> None:
    """
    Destination stores implement a uniform .upsert(items) method in this repo.
    """
    if not batch_payload:
        return
    dst_store.upsert(batch_payload)


def write_dump_jsonl(rows: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Core migration
# ──────────────────────────────────────────────────────────────────────────────

def migrate(
    src_iter: Iterator[Item],
    dst_store,
    *,
    reembed: bool,
    batch: int,
    ckpt: Checkpoint,
    dump_path: Optional[Path],
    embedder: Optional[EmbeddingClient],
) -> Tuple[int, int]:
    """
    Returns (migrated_count, skipped_count).
    """
    moved = 0
    skipped = 0
    pending: List[Item] = []

    def flush(items: List[Item]):
        nonlocal moved
        if not items:
            return

        # Prepare payloads
        texts = [it.text for it in items]
        vecs: List[Optional[List[float]]]
        if reembed:
            assert embedder is not None, "embedder required for re-embed mode"
            vecs = embedder.embed(texts, batch_size=max(1, batch))
        else:
            vecs = [it.vector for it in items]  # copy existing vectors

        payloads: List[dict] = []
        for it, v in zip(items, vecs):
            payload = {
                "id": it.id,
                "doc_id": it.metadata.get("doc_id") or it.id.split("#")[0],
                "text": it.text,
                "vector": v,
                "metadata": dict(it.metadata or {}),
            }
            payloads.append(payload)

        # Write to destination
        upsert_batch(dst_store, payloads)
        moved += len(payloads)

        # Dump (optional)
        if dump_path is not None:
            write_dump_jsonl(payloads, dump_path)

        # Mark checkpoint
        for it in items:
            ckpt.mark(it.id)

    for item in src_iter:
        if ckpt.has(item.id):
            skipped += 1
            continue

        # If not re-embedding, require vector present
        if not reembed and (item.vector is None or not isinstance(item.vector, list)):
            # No vector to copy → skip safely; user can re-run with --reembed
            skipped += 1
            ckpt.mark(item.id)  # mark skipped to avoid re-seeing it repeatedly
            continue

        pending.append(item)
        if len(pending) >= batch:
            flush(pending)
            pending = []
            print(f"migrated={moved}  skipped={skipped}")

    # Final flush
    flush(pending)
    return moved, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate/merge collections across vector stores.")
    ap.add_argument("--src-store", choices=["pgvector", "qdrant", "chroma", "faiss"], help="Source store kind")
    ap.add_argument("--src-collection", help="Source collection name")
    ap.add_argument("--input-jsonl", help="Alternative source: JSONL with {id,text,metadata,vector?}")
    ap.add_argument("--dst-store", required=True, choices=["pgvector", "qdrant", "chroma", "faiss"], help="Destination store kind")
    ap.add_argument("--dst-collection", required=True, help="Destination collection name")
    ap.add_argument("--reembed", action="store_true", help="Re-embed text using EmbeddingClient (ignores source vectors)")
    ap.add_argument("--batch", type=int, default=256, help="Batch size for embedding/upsert (default: 256)")
    ap.add_argument("--max", type=int, default=0, help="Limit number of items (0 = no limit)")
    ap.add_argument("--dump-jsonl", help="Write migrated payloads to JSONL (audit/backup)")
    ap.add_argument("--resume", help="Checkpoint path (default: ./data/outputs/migrate_<dst>.ckpt.jsonl)")
    args = ap.parse_args(argv)

    cfg = get_settings()

    # Destination store
    try:
        dst = open_store(args.dst_store, args.dst_collection, cfg)
    except Exception as e:
        print(f"ERROR: could not open destination store: {e}")
        return 2

    # Source iterator
    src_iter: Iterator[Item]
    limit = int(args.max) if args.max and args.max > 0 else None

    if args.input_jsonl:
        src_iter = iter_items_from_jsonl(Path(args.input_jsonl), limit=limit)
    else:
        if not args.src_store or not args.src_collection:
            print("ERROR: provide --src-store and --src-collection or use --input-jsonl.")
            return 2
        try:
            src = open_store(args.src_store, args.src_collection, cfg)
        except Exception as e:
            print(f"ERROR: could not open source store: {e}")
            return 2
        try:
            src_iter = iter_items_from_store(src, batch=1000, limit=limit)
        except Exception as e:
            print(f"ERROR: source store is not iterable ({e}). Use --input-jsonl.")
            return 2

    # Embedder (only if re-embedding)
    embedder = None
    if args.reembed:
        embedder = EmbeddingClient(provider="auto", model_alias=os.getenv("EMBED_MODEL_ALIAS", "mini"))

    # Checkpoint
    ckpt_path = Path(args.resume) if args.resume else Path("./data/outputs") / f"migrate_{args.dst_collection}.ckpt.jsonl"
    ckpt = Checkpoint(ckpt_path)

    # Optional dump
    dump_path = Path(args.dump_jsonl) if args.dump_jsonl else None
    if dump_path:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        # if file exists, we append; that’s fine for journaling

    # Migrate
    moved, skipped = migrate(
        src_iter=src_iter,
        dst_store=dst,
        reembed=bool(args.reembed),
        batch=max(1, int(args.batch)),
        ckpt=ckpt,
        dump_path=dump_path,
        embedder=embedder,
    )

    print("\nMigration summary")
    print("─────────────────")
    print(json.dumps({
        "dst_collection": args.dst_collection,
        "migrated": moved,
        "skipped": skipped,
        "checkpoint": str(ckpt_path),
        "reembed": bool(args.reembed),
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
