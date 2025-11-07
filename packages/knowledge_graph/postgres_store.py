# packages/knowledge_graph/postgres_store.py
"""
Knowledge Graph — Postgres adapter (SQLAlchemy Core)
====================================================

Purpose
-------
Drop-in `Store` implementation backed by Postgres using SQLAlchemy Core.
It mirrors the minimal Store API so you can switch from the in-memory
reference to a real DB with no call-site changes.

Assumptions
-----------
- Two tables exist (create via Alembic migration):
    kg_nodes(
        id UUID PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        type TEXT NOT NULL,          -- NodeKind value
        key TEXT NOT NULL,
        label TEXT NOT NULL,
        props JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL,
        UNIQUE (tenant_id, type, key)
    );

    kg_edges(
        id UUID PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        type TEXT NOT NULL,          -- EdgeKind value
        src_id UUID NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
        dst_id UUID NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
        props JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL,
        UNIQUE (tenant_id, type, src_id, dst_id)
    );

- You pass an SQLAlchemy `Engine` (or `Session`/`Connection`) from packages/core/db.py.

Design choices
--------------
- Idempotent upserts via `INSERT .. ON CONFLICT .. DO UPDATE`.
- Returns deep-copied Pydantic models (avoid leaking DB-row references).
- Small, readable queries; no ORM models to keep footprint tiny.

Usage
-----
>>> from sqlalchemy import create_engine
>>> from .postgres_store import PostgresStore
>>> eng = create_engine("postgresql+psycopg2://user:pass@localhost:5432/app")
>>> store = PostgresStore(engine=eng)
>>> res = store.upsert_nodes_edges(nodes, edges)
>>> sub_nodes, sub_edges = store.subgraph_from_entity("t0", "org:acme plc", depth=1)

"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import (
    Table, Column, MetaData, String, Text, DateTime, JSON, select, and_, insert, literal_column
)
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, insert as pg_insert

from .schema import Edge, EdgeKind, Node, NodeKind


# -----------------------
# Table metadata (mirror Alembic)
# -----------------------

metadata = MetaData()

kg_nodes = Table(
    "kg_nodes",
    metadata,
    Column("id", PG_UUID(as_uuid=True), primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("type", Text, nullable=False),
    Column("key", Text, nullable=False),
    Column("label", Text, nullable=False),
    Column("props", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

kg_edges = Table(
    "kg_edges",
    metadata,
    Column("id", PG_UUID(as_uuid=True), primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("type", Text, nullable=False),
    Column("src_id", PG_UUID(as_uuid=True), nullable=False),
    Column("dst_id", PG_UUID(as_uuid=True), nullable=False),
    Column("props", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


# -----------------------
# Adapter
# -----------------------

class PostgresStore:
    """
    Drop-in Store backed by Postgres (SQLAlchemy Core).

    Note: We deliberately do not manage transactions here; each method opens
    a short connection and commits (autocommit behavior depends on driver).
    For batched operations, consider wrapping calls in a higher-level
    transaction if your app manages unit-of-work semantics.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # ---- upsert ----

    def upsert_nodes_edges(self, nodes: Sequence[Node], edges: Sequence[Edge]):
        from .store import PersistResult  # avoid circular import at module import time
        res = PersistResult()

        if not nodes and not edges:
            return res

        with self._engine.begin() as conn:
            if nodes:
                res.nodes_created, res.nodes_updated = self._upsert_nodes(conn, nodes)
            if edges:
                res.edges_created, ignored = self._upsert_edges(conn, edges)
                res.edges_ignored += ignored

        return res

    def _upsert_nodes(self, conn: Connection, nodes: Sequence[Node]) -> Tuple[int, int]:
        created = 0
        updated = 0
        for n in nodes:
            stmt = pg_insert(kg_nodes).values(
                id=n.id,
                tenant_id=n.tenant_id,
                type=n.type.value,
                key=n.key,
                label=n.label,
                props=n.props or {},
                created_at=n.created_at,
            ).on_conflict_do_update(
                index_elements=["tenant_id", "type", "key"],
                set_={
                    "label": n.label,
                    "props": n.props or {},
                    # keep created_at as original
                },
            ).returning(literal_column("xmax"))  # xmax>0 indicates update in MVCC
            xmax = conn.execute(stmt).scalar_one()
            if xmax == 0:
                created += 1
            else:
                updated += 1
        return created, updated

    def _upsert_edges(self, conn: Connection, edges: Sequence[Edge]) -> Tuple[int, int]:
        created = 0
        ignored = 0
        # Validate endpoints exist (best-effort)
        existing_ids = self._existing_node_ids(conn, {e.src_id for e in edges} | {e.dst_id for e in edges})
        for e in edges:
            if e.src_id not in existing_ids or e.dst_id not in existing_ids:
                ignored += 1
                continue
            stmt = pg_insert(kg_edges).values(
                id=e.id,
                tenant_id=e.tenant_id,
                type=e.type.value,
                src_id=e.src_id,
                dst_id=e.dst_id,
                props=e.props or {},
                created_at=e.created_at,
            ).on_conflict_do_nothing()  # uniqueness on (tenant_id,type,src_id,dst_id)
            result = conn.execute(stmt)
            if result.rowcount:  # inserted
                created += 1
            else:
                ignored += 1
        return created, ignored

    def _existing_node_ids(self, conn: Connection, ids: Iterable[UUID]) -> set[UUID]:
        if not ids:
            return set()
        stmt = select(kg_nodes.c.id).where(kg_nodes.c.id.in_(list(ids)))
        rows = conn.execute(stmt).all()
        return {r[0] for r in rows}

    # ---- queries ----

    def get_node_by_key(self, tenant_id: str, kind: NodeKind, key: str) -> Optional[Node]:
        with self._engine.connect() as conn:
            stmt = (
                select(kg_nodes)
                .where(
                    and_(
                        kg_nodes.c.tenant_id == tenant_id,
                        kg_nodes.c.type == kind.value,
                        kg_nodes.c.key == key,
                    )
                )
                .limit(1)
            )
            row = conn.execute(stmt).mappings().first()
            if not row:
                return None
            return _row_to_node(row)

    def list_nodes(self, tenant_id: str, kind: Optional[NodeKind] = None, limit: int = 100, offset: int = 0) -> List[Node]:
        with self._engine.connect() as conn:
            where = [kg_nodes.c.tenant_id == tenant_id]
            if kind is not None:
                where.append(kg_nodes.c.type == kind.value)
            stmt = (
                select(kg_nodes)
                .where(and_(*where))
                .order_by(kg_nodes.c.type.asc(), kg_nodes.c.key.asc())
                .offset(offset)
                .limit(limit)
            )
            rows = conn.execute(stmt).mappings().all()
            return [_row_to_node(r) for r in rows]

    def list_edges(
        self,
        tenant_id: str,
        kind: Optional[EdgeKind] = None,
        src_id: Optional[UUID] = None,
        dst_id: Optional[UUID] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Edge]:
        with self._engine.connect() as conn:
            where = [kg_edges.c.tenant_id == tenant_id]
            if kind is not None:
                where.append(kg_edges.c.type == kind.value)
            if src_id is not None:
                where.append(kg_edges.c.src_id == src_id)
            if dst_id is not None:
                where.append(kg_edges.c.dst_id == dst_id)

            stmt = (
                select(kg_edges)
                .where(and_(*where))
                .order_by(kg_edges.c.type.asc(), kg_edges.c.src_id.asc(), kg_edges.c.dst_id.asc())
                .offset(offset)
                .limit(limit)
            )
            rows = conn.execute(stmt).mappings().all()
            return [_row_to_edge(r) for r in rows]

    # ---- subgraph ----

    def subgraph_from_entity(
        self,
        tenant_id: str,
        entity_key: str,
        *,
        depth: int = 1,
        max_neighbours: int = 25,
    ):
        """
        Small BFS in Python with SQL lookups per frontier hop.
        Optimised for small UI neighbourhoods; not intended for bulk export.
        """
        if depth < 0:
            depth = 0

        with self._engine.connect() as conn:
            root_row = conn.execute(
                select(kg_nodes).where(
                    and_(
                        kg_nodes.c.tenant_id == tenant_id,
                        kg_nodes.c.type == NodeKind.ENTITY.value,
                        kg_nodes.c.key == entity_key,
                    )
                ).limit(1)
            ).mappings().first()
            if not root_row:
                return [], []

            root = _row_to_node(root_row)

            seen_nodes: Dict[UUID, Node] = {root.id: root}
            seen_edges: Dict[UUID, Edge] = {}

            frontier: List[UUID] = [root.id]
            for _ in range(max(0, depth)):
                next_frontier: List[UUID] = []

                # Outgoing
                if frontier:
                    out_stmt = (
                        select(kg_edges)
                        .where(
                            and_(
                                kg_edges.c.tenant_id == tenant_id,
                                kg_edges.c.src_id.in_(frontier),
                            )
                        )
                        .limit(max_neighbours * len(frontier))
                    )
                    out_rows = conn.execute(out_stmt).mappings().all()
                    for r in out_rows:
                        e = _row_to_edge(r)
                        seen_edges[e.id] = e
                        # fetch nodes quickly
                        for nid in (e.src_id, e.dst_id):
                            if nid not in seen_nodes:
                                nrow = conn.execute(select(kg_nodes).where(kg_nodes.c.id == nid).limit(1)).mappings().first()
                                if nrow:
                                    n = _row_to_node(nrow)
                                    if n.id not in seen_nodes:
                                        seen_nodes[n.id] = n
                                        next_frontier.append(n.id)

                # Incoming
                if frontier:
                    in_stmt = (
                        select(kg_edges)
                        .where(
                            and_(
                                kg_edges.c.tenant_id == tenant_id,
                                kg_edges.c.dst_id.in_(frontier),
                            )
                        )
                        .limit(max_neighbours * len(frontier))
                    )
                    in_rows = conn.execute(in_stmt).mappings().all()
                    for r in in_rows:
                        e = _row_to_edge(r)
                        seen_edges[e.id] = e
                        for nid in (e.src_id, e.dst_id):
                            if nid not in seen_nodes:
                                nrow = conn.execute(select(kg_nodes).where(kg_nodes.c.id == nid).limit(1)).mappings().first()
                                if nrow:
                                    n = _row_to_node(nrow)
                                    if n.id not in seen_nodes:
                                        seen_nodes[n.id] = n
                                        next_frontier.append(n.id)

                frontier = next_frontier

        nodes = sorted(seen_nodes.values(), key=lambda n: (n.type.value, n.key))
        edges = sorted(seen_edges.values(), key=lambda e: (e.type.value, str(e.src_id), str(e.dst_id)))
        # Deep copies via Pydantic model_copy
        return [n.model_copy(deep=True) for n in nodes], [e.model_copy(deep=True) for e in edges]


# -----------------------
# Row mappers
# -----------------------

def _row_to_node(row) -> Node:
    return Node(
        id=row["id"],
        tenant_id=row["tenant_id"],
        type=NodeKind(row["type"]),
        key=row["key"],
        label=row["label"],
        props=row["props"] or {},
        created_at=row["created_at"],
    )

def _row_to_edge(row) -> Edge:
    return Edge(
        id=row["id"],
        tenant_id=row["tenant_id"],
        type=EdgeKind(row["type"]),
        src_id=row["src_id"],
        dst_id=row["dst_id"],
        props=row["props"] or {},
        created_at=row["created_at"],
    )


__all__ = ["PostgresStore", "kg_nodes", "kg_edges", "metadata"]
