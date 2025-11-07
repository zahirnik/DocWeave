# packages/retriever/vectorstores/pgvector_store.py
"""
pgvector store — tiny, explicit wrapper around PostgreSQL + pgvector.

What this module provides
-------------------------
A minimal, tutorial-clear interface for storing and searching embeddings in
PostgreSQL using the pgvector extension.

Class
-----
PgVectorStore(dsn: str, dimension: int, table_prefix: str = "rag")
  - ensure_schema() -> None
  - create_collection(name: str, *, metadata: dict | None = None) -> int
  - delete_collection(name: str) -> bool
  - list_collections() -> list[dict]
  - get_collection(name: str) -> dict | None
  - upsert_texts(collection: str, items: list[dict]) -> int
      items = [
        {
          "id": "optional-uuid",            # if absent, server generates uuid
          "text": "chunk text",
          "metadata": {...},                # JSON-serializable dict
          "embedding": [float, ...]         # length == dimension
        },
        ...
      ]
  - search(collection: str, query: list[float], top_k: int = 5, filters: dict | None = None)
      -> list[{"id","text","metadata","score"}]    # score = cosine distance (lower is better)
  - stats(collection: str) -> dict
  - get_by_id(collection: str, id: str) -> dict | None
  - delete_by_ids(collection: str, ids: list[str]) -> int

Design goals
------------
- Keep SQL readable and explicit; no ORM in this tutorial-level wrapper.
- Safe parameterization for every value **except** the vector literal, which is
  constructed from floats into the pgvector `'[v1,v2,...]'` form (kept safe by
  formatting numbers only).
- Cosine distance search (`<=>` operator). Prefer L2-normalized embeddings.

Requirements
------------
- PostgreSQL 14+ (15+ recommended)
- pgvector extension installed in the database:
    CREATE EXTENSION IF NOT EXISTS vector;
- Python package: psycopg2-binary   (`pip install psycopg2-binary`)
  (You may swap in `psycopg` v3 with small changes if you prefer.)

Schema (created by ensure_schema)
--------------------------------
<tp>_collections(id uuid PK, name text UNIQUE, metadata jsonb, created_at timestamptz)
<tp>_chunks(
    id uuid PK DEFAULT gen_random_uuid(),  -- requires pgcrypto; we guard for fallback
    collection_id uuid REFERENCES <tp>_collections(id) ON DELETE CASCADE,
    text text,
    metadata jsonb,
    embedding vector(<dimension>),
    created_at timestamptz DEFAULT now()
)
Indexes:
- GIN on metadata for simple @> filters
- IVFFLAT on embedding with cosine ops (requires ANALYZE after creation)
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

from packages.core.logging import get_logger

log = get_logger(__name__)

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "psycopg2-binary is required for PgVectorStore. Install with:\n"
        "  pip install psycopg2-binary"
    ) from e


def _vector_literal(vec: Sequence[float]) -> str:
    """
    Build a pgvector literal string like: '[0.1,0.2,0.3]'.

    IMPORTANT: We only format floats; no user-controlled text is interpolated.
    """
    # Coerce to finite floats; replace non-finite with 0.0 to keep insert stable
    clean: List[str] = []
    for x in vec:
        try:
            xf = float(x)
            if not math.isfinite(xf):
                xf = 0.0
        except Exception:
            xf = 0.0
        clean.append(repr(xf))
    return "[" + ",".join(clean) + "]"


@contextmanager
def _conn(dsn: str):
    """
    Connection context manager with autocommit for DDL/DML convenience.
    """
    con = psycopg2.connect(dsn)
    try:
        con.autocommit = True
        yield con
    finally:
        try:
            con.close()
        except Exception:
            pass


class PgVectorStore:
    """
    Minimal pgvector-backed vector store.

    Example
    -------
    store = PgVectorStore(dsn=os.environ["DATABASE_URL"], dimension=1536)
    store.ensure_schema()
    store.create_collection("acme", metadata={"industry":"finance"})

    store.upsert_texts("acme", [
        {"text":"Revenue increased 12% YoY.","metadata":{"year":2024},"embedding": e1},
        {"text":"Operating margin compressed.","metadata":{"year":2024},"embedding": e2},
    ])

    results = store.search("acme", qvec, top_k=3, filters={"year": 2024})
    for r in results:
        print(r["score"], r["text"])
    """

    def __init__(self, dsn: str, dimension: int, table_prefix: str = "rag"):
        if not dsn:
            raise ValueError("dsn is required (e.g., postgres://user:pass@host:5432/db)")
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self.dsn = dsn
        self.dim = int(dimension)
        self.tp = str(table_prefix).strip() or "rag"
        self._tbl_colls = f'{self.tp}_collections'
        self._tbl_chunks = f'{self.tp}_chunks'

    # ---------------------------
    # Schema
    # ---------------------------

    def ensure_schema(self) -> None:
        """
        Create extension/tables/indexes if missing. Safe to call multiple times.
        """
        with _conn(self.dsn) as con:
            cur = con.cursor()
            # Enable required extensions (best-effort)
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as e:
                log.info("Note: couldn't create 'vector' extension automatically: %s", e)
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")  # for gen_random_uuid()
            except Exception:
                pass

            # Collections table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._tbl_colls} (
                    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    name       text UNIQUE NOT NULL,
                    metadata   jsonb,
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )

            # Chunks table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._tbl_chunks} (
                    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    collection_id uuid NOT NULL REFERENCES {self._tbl_colls}(id) ON DELETE CASCADE,
                    text          text,
                    metadata      jsonb,
                    embedding     vector({self.dim}) NOT NULL,
                    created_at    timestamptz NOT NULL DEFAULT now()
                )
                """
            )

            # Useful indexes
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.tp}_chunks_collection ON {self._tbl_chunks}(collection_id)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.tp}_chunks_meta ON {self._tbl_chunks} USING GIN (metadata)")
            # IVFFLAT on embedding (cosine ops). Requires ANALYZE table after sizable inserts.
            try:
                cur.execute(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes WHERE schemaname = ANY(current_schemas(false))
                            AND indexname = 'idx_{self.tp}_chunks_embed_cos'
                        ) THEN
                            EXECUTE 'CREATE INDEX idx_{self.tp}_chunks_embed_cos ON {self._tbl_chunks} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)';
                        END IF;
                    END$$;
                    """
                )
            except Exception as e:
                log.info("Could not create IVFFLAT index (pgvector): %s (OK for dev/small data).", e)

    # ---------------------------
    # Collections
    # ---------------------------

    def create_collection(self, name: str, *, metadata: Optional[dict] = None) -> str:
        """
        Create (or get) a collection and return its id (uuid).
        """
        if not name or not name.strip():
            raise ValueError("collection name required")

        with _conn(self.dsn) as con:
            cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(f"SELECT id FROM {self._tbl_colls} WHERE name=%s", (name,))
            row = cur.fetchone()
            if row:
                return row["id"]

            cur.execute(
                f"INSERT INTO {self._tbl_colls}(name, metadata) VALUES (%s, %s) RETURNING id",
                (name, psycopg2.extras.Json(metadata or {})),
            )
            rid = cur.fetchone()["id"]
            return rid

    def get_collection(self, name: str) -> Optional[dict]:
        with _conn(self.dsn) as con:
            cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(f"SELECT id, name, metadata, created_at FROM {self._tbl_colls} WHERE name=%s", (name,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_collections(self) -> List[dict]:
        with _conn(self.dsn) as con:
            cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(f"SELECT id, name, metadata, created_at FROM {self._tbl_colls} ORDER BY created_at DESC")
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]

    def delete_collection(self, name: str) -> bool:
        with _conn(self.dsn) as con:
            cur = con.cursor()
            # Deleting collection row cascades to chunks via FK
            cur.execute(f"DELETE FROM {self._tbl_colls} WHERE name=%s", (name,))
            return cur.rowcount > 0

    # ---------------------------
    # Upsert
    # ---------------------------

    def _collection_id(self, name: str) -> str:
        got = self.get_collection(name)
        if not got:
            raise ValueError(f"collection not found: {name!r}")
        return got["id"]

    def upsert_texts(self, collection: str, items: List[dict]) -> int:
        """
        Insert or update text chunks with embeddings.

        Each item: {"id": str|None, "text": str, "metadata": dict, "embedding": list[float]}
        If "id" provided and exists → UPDATE; else INSERT.

        Returns: number of upserted rows (best effort).
        """
        if not items:
            return 0

        cid = self._collection_id(collection)
        n = 0
        with _conn(self.dsn) as con:
            cur = con.cursor()
            for it in items:
                text = it.get("text", "")
                meta = it.get("metadata") or {}
                emb = it.get("embedding")
                cid_param = cid

                if not isinstance(emb, (list, tuple)) or len(emb) != self.dim:
                    raise ValueError(f"embedding must be a list[float] of length {self.dim}")

                vec = _vector_literal(emb)

                if it.get("id"):
                    # Try update; if not found, insert with the given id
                    the_id = it["id"]
                    cur.execute(
                        f"""
                        UPDATE {self._tbl_chunks}
                        SET text=%s, metadata=%s, embedding=%s::vector
                        WHERE id=%s AND collection_id=%s
                        """,
                        (text, psycopg2.extras.Json(meta), vec, the_id, cid_param),
                    )
                    if cur.rowcount == 0:
                        cur.execute(
                            f"""
                            INSERT INTO {self._tbl_chunks}(id, collection_id, text, metadata, embedding)
                            VALUES (%s, %s, %s, %s, %s::vector)
                            """,
                            (the_id, cid_param, text, psycopg2.extras.Json(meta), vec),
                        )
                    n += 1
                else:
                    cur.execute(
                        f"""
                        INSERT INTO {self._tbl_chunks}(collection_id, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        """,
                        (cid_param, text, psycopg2.extras.Json(meta), vec),
                    )
                    n += 1
        return n

    # ---------------------------
    # Search
    # ---------------------------

    def search(
        self,
        collection: str,
        query: Sequence[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[dict]:
        """
        Cosine-distance search (lower is better).

        Args:
          collection : collection name
          query      : query embedding (list[float] of length `dimension`)
          top_k      : number of results
          filters    : JSON filter applied via metadata @> <filters_json>

        Returns:
          list of dicts: [{"id","text","metadata","score"}]
        """
        if not isinstance(query, (list, tuple)) or len(query) != self.dim:
            raise ValueError(f"query vector must be length {self.dim}")
        vec = _vector_literal(query)
        cid = self._collection_id(collection)

        where = "c.collection_id = %s"
        params: List = [cid]

        if filters:
            where += " AND c.metadata @> %s"
            params.append(json.dumps(filters))

        # Cosine distance operator in pgvector: <=> (smaller is more similar)
        sql = f"""
            SELECT c.id, c.text, c.metadata, (c.embedding <=> %s::vector) AS score
            FROM {self._tbl_chunks} c
            WHERE {where}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        # We need to bind the vector twice (distance in SELECT and ORDER BY)
        params = [*params[:1], vec, *params[1:], vec, max(1, int(top_k))]

        with _conn(self.dsn) as con:
            cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            rows = cur.fetchall() or []
            out = []
            for r in rows:
                out.append(
                    {
                        "id": str(r["id"]),
                        "text": r.get("text") or "",
                        "metadata": r.get("metadata") or {},
                        "score": float(r.get("score") or 0.0),
                    }
                )
            return out

    # ---------------------------
    # Utilities
    # ---------------------------

    def stats(self, collection: str) -> dict:
        """
        Return quick stats for a collection (count, latest insert time).
        """
        cid = self._collection_id(collection)
        with _conn(self.dsn) as con:
            cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                f"SELECT COUNT(*) AS n, MAX(created_at) AS latest FROM {self._tbl_chunks} WHERE collection_id=%s",
                (cid,),
            )
            row = cur.fetchone() or {"n": 0, "latest": None}
            return {"count": int(row["n"] or 0), "latest": row["latest"]}

    def get_by_id(self, collection: str, id: str) -> Optional[dict]:
        cid = self._collection_id(collection)
        with _conn(self.dsn) as con:
            cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                f"SELECT id, text, metadata FROM {self._tbl_chunks} WHERE collection_id=%s AND id=%s",
                (cid, id),
            )
            r = cur.fetchone()
            if not r:
                return None
            return {"id": str(r["id"]), "text": r.get("text") or "", "metadata": r.get("metadata") or {}}

    def delete_by_ids(self, collection: str, ids: Iterable[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        cid = self._collection_id(collection)
        with _conn(self.dsn) as con:
            cur = con.cursor()
            psycopg2.extras.execute_values(
                cur,
                f"DELETE FROM {self._tbl_chunks} WHERE collection_id = %s AND id IN %s",
                # execute_values needs a tuple of sequence; easiest to pass (cid, tuple(ids)) via a wrapper SELECT
                # but psycopg2 doesn't bind IN %s with sequences directly. We'll build a VALUES list instead.
                # Simpler approach: run a single DELETE with = ANY(%s).
                # So we roll our own here:
                [],
            )
        # The above execute_values approach is clunky; use ANY(...)
        with _conn(self.dsn) as con:
            cur = con.cursor()
            cur.execute(
                f"DELETE FROM {self._tbl_chunks} WHERE collection_id = %s AND id = ANY(%s)",
                (cid, ids),
            )
            return cur.rowcount
