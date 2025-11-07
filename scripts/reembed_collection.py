# scripts/reembed_collection.py
"""
Re-embed a collection with a new embedding model (resume-safe, parity check).

What this script does
---------------------
- Reads existing items (id, text, metadata[, old vector]) from a vector store **or** from a JSONL dump.
- Re-embeds the texts with a (new) embedding model (OpenAI if key present; else local fallback).
- Upserts results into a **target collection** (default: <source>__<model_alias>) so you can compare safely.
- Writes a checkpoint file to allow resuming long runs without rework.
- Optionally runs a tiny parity/overlap probe to compare old vs new collections.

Why this exists
---------------
- To change embedding models without destroying your current index.
- To support safe migration/rollback and quick A/B checks.

Examples
--------
# Re-embed a pgvector collection into a new pgvector collection
export PGVECTOR_URL="postgresql+psycopg://user:pass@localhost:5432/convai"
python -m scripts.reembed_collection \
  --source-collection finance_demo \
  --target-collection finance_demo_gte \
  --store pgvector --batch 128

# Re-embed from a JSONL dump (each line: {"id","text","metadata":{...}})
python -m scripts.reembed_collection \
  --input-jsonl ./data/outputs/export_texts.jsonl \
  --target-collection finance_demo_gte \
  --store chroma

CLI
---
--source-collection <name>   # required unless --input-jsonl is used
--target-collection <name>   # default: <source>__<model_alias>
--store {pgvector|qdrant|chroma} (default: auto pick; see open_store())
--input-jsonl <path>         # alternative to reading from the store
--batch <int>                # embedding batch size (default: 128)
--resume <path>              # checkpoint jsonl to resume; default under ./data/outputs
--probe-queries <int>        # run tiny parity probe with N queries (default: 0 = skip)

Outputs
-------
- Checkpoint JSONL with processed IDs (for resume)
- Summary printed to stdout

Notes
-----
- We assume vector store interfaces implement:
    • upsert(items: List[dict]) -> None
    • search(vector, top_k: int) -> List[dict] (for parity probe; optional)
    • (Optional) iter/export/scan(...) to iterate items; if not present, use --input-jsonl
- If your store lacks iteration, produce a dump externally (e.g., small export utility) and pass --input-jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from packages.core.config import get_settings
from packages.retriever.embeddings import EmbeddingClient

# Vector stores (duck-typed)
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
class Item:
    id: str
    text: str
    metadata: Dict[str, str]


# ---------------------------
# Store factory
# ---------------------------

def open_store(store: str, collection: str, cfg):
    store = (store or "").lower()
    if not store:
        # auto-pick based on env/config
        if cfg.pgvector_url:
            store = "pgvector"
        elif cfg.qdrant_url:
            store = "qdrant"
        else:
            store = "chroma"

    if store == "pgvector":
        if PgVectorStore is None or not cfg.pgvector_url:
            raise RuntimeError("PGVECTOR_URL not set or PgVectorStore unavailable.")
        return PgVectorStore(dsn=cfg.pgvector_url, collection=collection, create_if_missing=True)

    if store == "qdrant":
        if QdrantStore is None or not cfg.qdrant_url:
            raise RuntimeError("QDRANT_URL not set or QdrantStore unavailable.")
        return QdrantStore(url=cfg.qdrant_url, collection=collection, create_if_missing=True)

    if store == "chroma":
        if ChromaStore is None:
            raise RuntimeError("ChromaStore unavailable.")
        return ChromaStore(path=cfg.chroma_path or "./.chroma", collection=collection, create_if_missing=True)

    raise ValueError("--store must be one of {pgvector|qdrant|chroma}")


# ---------------------------
# Iteration helpers (duck-typed)
# ---------------------------

def iter_store_items(store, batch_size: int = 1000) -> Iterable[Item]:
    """
    Try common iteration methods across stores. Expected output: Item(id, text, metadata).
    If the store doesn’t support iteration, raise and ask for --input-jsonl.
    """
    # 1) Generic .iter(...)
    if hasattr(store, "iter"):
        for rec in store.iter(batch=batch_size, fields=["id", "text", "metadata"]):  # type: ignore[attr-defined]
            yield Item(id=str(rec["id"]), text=rec.get("text") or "", metadata=rec.get("metadata") or {})
        return
    # 2) .scan(...)
    if hasattr(store, "scan"):
        for rec in store.scan(batch=batch_size, fields=["id", "text", "metadata"]):  # type: ignore[attr-defined]
            yield Item(id=str(rec["id"]), text=rec.get("text") or "", metadata=rec.get("metadata") or {})
        return
    # 3) .export() -> Iterable[dict]
    if hasattr(store, "export"):
        for rec in store.export():  # type: ignore[attr-defined]
            yield Item(id=str(rec["id"]), text=rec.get("text") or "", metadata=rec.get("metadata") or {})
        return
    raise RuntimeError("This vector store does not expose an iteration API. Use --input-jsonl.")


def iter_jsonl(path: Path) -> Iterable[Item]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                yield Item(id=str(d["id"]), text=str(d.get("text", "")), metadata=d.get("metadata", {}) or {})
            except Exception:
                continue


# ---------------------------
# Checkpoint (resume-safe)
# ---------------------------

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.processed: set[str] = set()
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
                                self.processed.add(_id)
                        except Exception:
                            continue
            except Exception:
                # corrupt? start fresh
                self.processed = set()

    def has(self, _id: str) -> bool:
        return _id in self.processed

    def mark(self, _id: str):
        # append-line journaling
        self.processed.add(_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": _id}) + "\n")


# ---------------------------
# Parity probe
# ---------------------------

def parity_probe(queries: List[str], embedder: EmbeddingClient, src_store, tgt_store, k: int = 5) -> Dict[str, float]:
    """
    Compare overlap of top-k IDs from source vs target for a few queries.
    Returns a dict with simple overlap stats per query and an average.
    """
    scores: List[float] = []

    def _search_ids(store, q_vec):
        try:
            hits = store.search(q_vec, top_k=k)
            return [str(h.get("id")) for h in hits]
        except Exception:
            return []

    for q in queries:
        qv = embedder.embed([q])[0]
        src_ids = _search_ids(src_store, qv)
        tgt_ids = _search_ids(tgt_store, qv)
        if not src_ids or not tgt_ids:
            continue
        overlap = len(set(src_ids) & set(tgt_ids)) / float(max(1, len(set(src_ids) | set(tgt_ids))))
        scores.append(overlap)

    out = {
        "queries": len(queries),
        "evaluated": len(scores),
        "avg_overlap": float(sum(scores) / len(scores)) if scores else 0.0,
    }
    return out


# ---------------------------
# Main
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Re-embed a collection with a new embedding model.")
    ap.add_argument("--source-collection", help="Existing collection to read from (omit if using --input-jsonl)")
    ap.add_argument("--target-collection", help="Target collection to write to (default: <source>__<model_alias>)")
    ap.add_argument("--store", choices=["pgvector", "qdrant", "chroma"], default="", help="Vector store backend")
    ap.add_argument("--input-jsonl", help="JSONL with lines of {id,text,metadata} (alternative to reading from store)")
    ap.add_argument("--batch", type=int, default=128, help="Embedding batch size")
    ap.add_argument("--resume", help="Checkpoint JSONL path (default: auto under ./data/outputs)")
    ap.add_argument("--probe-queries", type=int, default=0, help="Run tiny parity probe with N queries (0=skip)")
    args = ap.parse_args(argv)

    cfg = get_settings()
    embedder = EmbeddingClient(provider="auto", model_alias=os.getenv("EMBED_MODEL_ALIAS", "mini"))
    model_alias = embedder.model_alias

    # Validate inputs
    if not args.source_collection and not args.input_jsonl:
        print("ERROR: provide --source-collection or --input-jsonl")
        return 2

    # Resolve collections/stores
    source_collection = args.source_collection or f"jsonl_import"
    target_collection = args.target_collection or f"{source_collection}__{model_alias}"

    # Open target store
    try:
        tgt_store = open_store(args.store, target_collection, cfg)
    except Exception as e:
        print(f"Failed to open target store: {e}")
        return 3

    # Optionally open source store (for parity probe or when iterating from store)
    src_store = None
    if args.source_collection:
        try:
            src_store = open_store(args.store, source_collection, cfg)
        except Exception as e:
            print(f"WARNING: could not open source store: {e}")

    # Checkpoint path
    ckpt_path = Path(args.resume) if args.resume else Path("./data/outputs") / f"reembed_{source_collection}_{model_alias}.ckpt.jsonl"
    ckpt = Checkpoint(ckpt_path)

    # Iterator over items
    if args.input_jsonl:
        src_iter: Iterable[Item] = iter_jsonl(Path(args.input_jsonl))
    else:
        if src_store is None:
            print("ERROR: cannot read from source store; supply --input-jsonl.")
            return 4
        try:
            src_iter = iter_store_items(src_store, batch_size=1000)
        except Exception as e:
            print(f"ERROR: cannot iterate source store ({e}); supply --input-jsonl.")
            return 4

    # Main loop (embed → upsert)
    B = max(1, int(args.batch))
    pending: List[Item] = []
    total_read = 0
    total_skipped = 0
    total_upserted = 0

    def flush(batch_items: List[Item]):
        nonlocal total_upserted
        if not batch_items:
            return
        texts = [it.text for it in batch_items]
        vecs = embedder.embed(texts, batch_size=B)
        payloads = []
        for it, v in zip(batch_items, vecs):
            payloads.append({
                "id": it.id,
                "doc_id": it.metadata.get("doc_id") or it.id.split("#")[0],
                "text": it.text,
                "vector": v,
                "metadata": {**(it.metadata or {}), "embedding_model": model_alias},
            })
        # Upsert
        tgt_store.upsert(payloads)
        total_upserted += len(payloads)
        for it in batch_items:
            ckpt.mark(it.id)

    for item in src_iter:
        total_read += 1
        if ckpt.has(item.id):
            total_skipped += 1
            continue
        pending.append(item)
        if len(pending) >= B:
            flush(pending)
            pending = []
            print(f"Processed {total_upserted} items …")

    # Final flush
    flush(pending)

    print("\nRe-embed summary")
    print("────────────────")
    print(json.dumps({
        "source_collection": source_collection,
        "target_collection": target_collection,
        "model_alias": model_alias,
        "read": total_read,
        "skipped_via_resume": total_skipped,
        "upserted": total_upserted,
        "checkpoint": str(ckpt_path),
    }, indent=2))

    # Optional tiny parity probe
    if args.probe_queries and src_store is not None:
        N = int(max(1, args.probe_queries))
        sample_qs = [
            "What was ACME's gross margin in 2024?",
            "Q3 revenue performance Beta Corp",
            "Net debt to EBITDA ratio",
            "Operating cash flow trend",
            "EPS guidance and outlook",
        ]
        random.shuffle(sample_qs)
        qs = sample_qs[:N]
        try:
            stats = parity_probe(qs, embedder, src_store, tgt_store, k=5)
            print("\nParity probe")
            print("────────────")
            print(json.dumps({**stats, "queries_used": qs}, indent=2))
        except Exception as e:
            print(f"(probe) skipped due to store error: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
