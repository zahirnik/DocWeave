# packages/knowledge_graph/store.py
"""
Knowledge Graph — stores (backend-agnostic + in-memory reference)
=================================================================

Goal
----
Provide a tiny persistence interface for KG nodes/edges with a *working*
in-memory implementation so you can plug the KG into the pipeline today,
write tests, and later add a Postgres or Neo4j adapter without changing
call sites.

Design
------
- `Store` protocol defines the minimal API we need (upsert + queries).
- `InMemoryStore` is deterministic and idempotent:
    * Nodes are deduped by (tenant_id, type, key).
    * Edges are deduped by (tenant_id, type, src_id, dst_id).
- Subgraph API (`subgraph_from_entity`) performs a tiny BFS up to `depth`.

Future adapters
---------------
- Postgres adapter: implement the same methods using your `packages/core/db.py`
  engine/session and simple adjacency tables (see docs/KG.md for schema).
- Neo4j adapter: same interface using cypher queries.

Usage
-----
>>> store = InMemoryStore()
>>> nodes, edges = build_graph_for_doc(doc)                 # from builders.py
>>> res = store.upsert_nodes_edges(nodes, edges)
>>> sub = store.subgraph_from_entity(doc.tenant_id, "org:acme plc", depth=1)

Notes
-----
- We return copies (`model_copy(deep=True)`) to avoid accidental mutation
  of internal state by callers.
- This module has *no* external dependencies beyond pydantic & typing.

"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple
from uuid import UUID

from pydantic import BaseModel

from .schema import Edge, EdgeKind, Node, NodeKind


# -----------------------
# Data contracts
# -----------------------

@dataclass
class PersistResult:
    """Counts from an upsert op (for logs/tests/metrics)."""
    nodes_created: int = 0
    nodes_updated: int = 0
    edges_created: int = 0
    edges_ignored: int = 0  # duplicates


class Store(Protocol):
    """
    Minimal interface a KG backend must implement.
    Keep method names stable; we can extend with optional kwargs over time.
    """

    def upsert_nodes_edges(self, nodes: Sequence[Node], edges: Sequence[Edge]) -> PersistResult: ...

    def get_node_by_key(self, tenant_id: str, kind: NodeKind, key: str) -> Optional[Node]: ...
    def list_nodes(
        self,
        tenant_id: str,
        kind: Optional[NodeKind] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Node]: ...

    def list_edges(
        self,
        tenant_id: str,
        kind: Optional[EdgeKind] = None,
        src_id: Optional[UUID] = None,
        dst_id: Optional[UUID] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Edge]: ...

    def subgraph_from_entity(
        self,
        tenant_id: str,
        entity_key: str,
        *,
        depth: int = 1,
        max_neighbours: int = 25,
    ) -> Tuple[List[Node], List[Edge]]: ...


# -----------------------
# In-memory reference impl
# -----------------------

class InMemoryStore(Store):
    """
    Deterministic, idempotent, in-memory store.

    Implementation details
    ----------------------
    - Nodes:
        * Primary storage: id → Node
        * Secondary index: (tenant_id, kind, key) → id
    - Edges:
        * Primary storage: id → Edge
        * Secondary index: (tenant_id, kind, src_id, dst_id) → id  (enforces uniqueness)
        * Adjacency: src_id → list[edge_id] and dst_id → list[edge_id] for fast neighbourhoods
    """

    def __init__(self) -> None:
        # Nodes
        self._nodes_by_id: Dict[UUID, Node] = {}
        self._node_idx_by_key: Dict[Tuple[str, str, str], UUID] = {}  # (tenant, kind, key) -> node_id

        # Edges
        self._edges_by_id: Dict[UUID, Edge] = {}
        self._edge_idx_by_tuple: Dict[Tuple[str, str, UUID, UUID], UUID] = {}  # (tenant, kind, src, dst) -> edge_id

        # Adjacency
        self._out_edges: Dict[UUID, List[UUID]] = defaultdict(list)
        self._in_edges: Dict[UUID, List[UUID]] = defaultdict(list)

    # ---- upsert ----

    def upsert_nodes_edges(self, nodes: Sequence[Node], edges: Sequence[Edge]) -> PersistResult:
        res = PersistResult()

        # Nodes: dedupe by (tenant, kind, key)
        for n in nodes:
            node_key = (n.tenant_id, n.type.value, n.key)
            existing_id = self._node_idx_by_key.get(node_key)

            if existing_id is None:
                # create
                nn = n.model_copy(deep=True)
                self._nodes_by_id[nn.id] = nn
                self._node_idx_by_key[node_key] = nn.id
                res.nodes_created += 1
            else:
                # update (shallow policy: replace label/props; keep id/created_at)
                cur = self._nodes_by_id[existing_id]
                changed = False
                if cur.label != n.label:
                    cur.label = n.label
                    changed = True
                # Replace props if different (simple equality is fine for JSON-like dicts)
                if cur.props != n.props:
                    cur.props = n.model_dump()["props"]
                    changed = True
                if changed:
                    res.nodes_updated += 1

        # Edges: dedupe by (tenant, kind, src, dst)
        for e in edges:
            e_key = (e.tenant_id, e.type.value, e.src_id, e.dst_id)

            # Validate endpoints exist (within this store). If not, we drop the edge.
            if e.src_id not in self._nodes_by_id or e.dst_id not in self._nodes_by_id:
                res.edges_ignored += 1
                continue

            existing_eid = self._edge_idx_by_tuple.get(e_key)
            if existing_eid is None:
                ee = e.model_copy(deep=True)
                self._edges_by_id[ee.id] = ee
                self._edge_idx_by_tuple[e_key] = ee.id
                self._out_edges[ee.src_id].append(ee.id)
                self._in_edges[ee.dst_id].append(ee.id)
                res.edges_created += 1
            else:
                res.edges_ignored += 1  # duplicate (idempotent)

        return res

    # ---- queries ----

    def get_node_by_key(self, tenant_id: str, kind: NodeKind, key: str) -> Optional[Node]:
        node_id = self._node_idx_by_key.get((tenant_id, kind.value, key))
        if node_id is None:
            return None
        return self._nodes_by_id[node_id].model_copy(deep=True)

    def list_nodes(
        self,
        tenant_id: str,
        kind: Optional[NodeKind] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Node]:
        out: List[Node] = []
        # Fast path: iterate secondary index for tenant/kind filters
        if kind is None:
            # just filter all nodes by tenant
            for n in self._nodes_by_id.values():
                if n.tenant_id == tenant_id:
                    out.append(n)
        else:
            prefix = (tenant_id, kind.value)
            for (t, k, _), node_id in self._node_idx_by_key.items():
                if (t, k) == prefix:
                    out.append(self._nodes_by_id[node_id])

        out.sort(key=lambda n: (n.type.value, n.key))
        return [n.model_copy(deep=True) for n in out[offset : offset + limit]]

    def list_edges(
        self,
        tenant_id: str,
        kind: Optional[EdgeKind] = None,
        src_id: Optional[UUID] = None,
        dst_id: Optional[UUID] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Edge]:
        out: List[Edge] = []

        # Filter via main dict; index is for uniqueness, not for listing
        for e in self._edges_by_id.values():
            if e.tenant_id != tenant_id:
                continue
            if kind is not None and e.type != kind:
                continue
            if src_id is not None and e.src_id != src_id:
                continue
            if dst_id is not None and e.dst_id != dst_id:
                continue
            out.append(e)

        out.sort(key=lambda e: (e.type.value, str(e.src_id), str(e.dst_id)))
        return [e.model_copy(deep=True) for e in out[offset : offset + limit]]

    # ---- subgraph ----

    def subgraph_from_entity(
        self,
        tenant_id: str,
        entity_key: str,
        *,
        depth: int = 1,
        max_neighbours: int = 25,
    ) -> Tuple[List[Node], List[Edge]]:
        """
        BFS outwards from an Entity node by `depth` hops and return a small subgraph.

        - We cap fan-out by `max_neighbours` per node to avoid explosion.
        - Returns copies so callers can't mutate internal state.
        """
        if depth < 0:
            depth = 0

        # Find the root entity node id
        root = self.get_node_by_key(tenant_id, NodeKind.ENTITY, entity_key)
        if root is None:
            return [], []

        visited_nodes: Dict[UUID, Node] = {root.id: root}
        collected_edges: Dict[UUID, Edge] = {}

        q: deque[Tuple[UUID, int]] = deque([(root.id, 0)])

        while q:
            node_id, d = q.popleft()
            if d == depth:
                continue

            # Outgoing edges
            out_ids = self._out_edges.get(node_id, [])
            # Simple cap to avoid exploding subgraphs
            for eid in out_ids[:max_neighbours]:
                e = self._edges_by_id[eid]
                if e.tenant_id != tenant_id:
                    continue
                collected_edges[e.id] = e
                target = self._nodes_by_id[e.dst_id]
                if target.id not in visited_nodes:
                    visited_nodes[target.id] = target
                    q.append((target.id, d + 1))

            # Incoming edges (optional but helpful for context)
            in_ids = self._in_edges.get(node_id, [])
            for eid in in_ids[:max_neighbours]:
                e = self._edges_by_id[eid]
                if e.tenant_id != tenant_id:
                    continue
                collected_edges[e.id] = e
                source = self._nodes_by_id[e.src_id]
                if source.id not in visited_nodes:
                    visited_nodes[source.id] = source
                    q.append((source.id, d + 1))

        nodes = list(visited_nodes.values())
        edges = list(collected_edges.values())
        nodes.sort(key=lambda n: (n.type.value, n.key))
        edges.sort(key=lambda e: (e.type.value, str(e.src_id), str(e.dst_id)))

        # Return deep copies
        return [n.model_copy(deep=True) for n in nodes], [e.model_copy(deep=True) for e in edges]


# Optional: lightweight factory (useful in DI)
def get_store(backend: str = "memory", **kwargs) -> Store:
    """
    Simple store factory. For now supports only "memory".
    Later you can add:
        - backend="postgres" → return PostgresStore(**kwargs)
        - backend="neo4j"    → return Neo4jStore(**kwargs)
    """
    if backend == "memory":
        return InMemoryStore()
    raise ValueError(f"Unknown KG backend: {backend!r}")
