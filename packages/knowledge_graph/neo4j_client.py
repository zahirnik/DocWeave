# packages/knowledge_graph/neo4j_client.py
"""
Knowledge Graph — Neo4j adapter (optional)
- Pure Cypher (no APOC).
- Robust timestamp parsing.
- Subgraph/list_edges return explicit src/dst ids (no reliance on driver relationship node handles).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import UUID
from datetime import datetime, timezone

try:
    from neo4j import GraphDatabase, Driver  # type: ignore
except Exception as _exc:  # pragma: no cover
    GraphDatabase = None  # type: ignore
    Driver = None  # type: ignore
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None

from .schema import Edge, EdgeKind, Node, NodeKind
from .store import PersistResult, Store


def _require_driver() -> None:
    if GraphDatabase is None:
        raise RuntimeError(
            "Neo4j driver not installed. Run `pip install neo4j` "
            f"(import error: {_IMPORT_ERROR})"
        )


def _node_map(n: Node) -> Dict:
    return {
        "id": str(n.id),
        "tenant_id": n.tenant_id,
        "type": n.type.value,                      # e.g., "entity", "metric", "claim", ...
        "key": n.key,
        "label": n.label,
        "props": n.props or {},                    # map form (for surfacing top-level fields)
        "props_json": json.dumps(n.props or {}, ensure_ascii=False, separators=(",", ":")),
        "created_at": n.created_at.isoformat(),
    }


def _edge_map(e: Edge) -> Dict:
    return {
        "id": str(e.id),
        "tenant_id": e.tenant_id,
        "type": e.type.value,
        "src_id": str(e.src_id),
        "dst_id": str(e.dst_id),
        "props_json": json.dumps(e.props or {}, ensure_ascii=False, separators=(",", ":")),
        "created_at": e.created_at.isoformat(),
    }


def _parse_created_at(v) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _record_to_node(rec) -> Node:
    n = rec["n"]
    props = dict(n)
    return Node(
        id=UUID(props["id"]),
        tenant_id=props["tenant_id"],
        type=NodeKind(props["type"]),
        key=props["key"],
        label=props.get("label", ""),
        props=json.loads(props.get("props_json") or "{}"),
        created_at=_parse_created_at(props.get("created_at")),
    )


def _record_to_edge_from_row(row) -> Edge:
    """Row contains: r (relationship props), src_id, dst_id."""
    r = row["r"]
    props = dict(r)
    return Edge(
        id=UUID(props["id"]),
        tenant_id=props["tenant_id"],
        type=EdgeKind(props["type"]),
        src_id=UUID(row["src_id"]),
        dst_id=UUID(row["dst_id"]),
        props=json.loads(props.get("props_json") or "{}"),
        created_at=_parse_created_at(props.get("created_at")),
    )


class Neo4jStore(Store):
    """
    Drop-in Store backed by Neo4j.
    """

    def __init__(self, uri: str, auth: Tuple[str, str], *, database: Optional[str] = None) -> None:
        _require_driver()
        self._driver: Driver = GraphDatabase.driver(uri, auth=auth)
        self._db = database

    # ------------- core upsert -------------

    def upsert_nodes_edges(self, nodes: Sequence[Node], edges: Sequence[Edge]) -> PersistResult:
        res = PersistResult()
        if not nodes and not edges:
            return res

        with self._driver.session(database=self._db) as sess:
            # NODES: Pure Cypher MERGE + conditional label setting (no APOC)
            if nodes:
                params = {"batch": [_node_map(n) for n in nodes]}
                cy_nodes = """
                UNWIND $batch AS n
                MERGE (x:Node {tenant_id:n.tenant_id, type:n.type, key:n.key})
                ON CREATE SET
                    x.id = n.id,
                    x.label = n.label,
                    x.props_json = n.props_json,
                    x.created_at = n.created_at
                SET x.id = n.id,
                    x.tenant_id = n.tenant_id,
                    x.type = n.type,
                    x.key = n.key,
                    x.label = n.label,
                    x.props_json = n.props_json,
                    x.created_at = coalesce(x.created_at, n.created_at)

                // Add TitleCase + UPPERCASE labels without APOC (limited to known kinds)
                FOREACH (_ IN CASE WHEN n.type = 'entity'   THEN [1] ELSE [] END | SET x:Entity:ENTITY)
                FOREACH (_ IN CASE WHEN n.type = 'metric'   THEN [1] ELSE [] END | SET x:Metric:METRIC)
                FOREACH (_ IN CASE WHEN n.type = 'claim'    THEN [1] ELSE [] END | SET x:Claim:CLAIM)
                FOREACH (_ IN CASE WHEN n.type = 'document' THEN [1] ELSE [] END | SET x:Document:DOCUMENT)

                // Surface selected top-level properties by type (for indexes/FTS)
                WITH n, x
                FOREACH (_ IN CASE WHEN n.type = 'entity' THEN [1] ELSE [] END |
                    SET x.name = coalesce(n.props.name, x.name),
                        x.alt_names = coalesce(n.props.alt_names, x.alt_names)
                )
                FOREACH (_ IN CASE WHEN n.type = 'metric' THEN [1] ELSE [] END |
                    SET x.name = coalesce(n.props.name, x.name)
                )
                FOREACH (_ IN CASE WHEN n.type = 'claim' THEN [1] ELSE [] END |
                    SET x.hash = coalesce(n.props.hash, x.hash),
                        x.text = coalesce(n.props.text, x.text),
                        x.normalized_text = coalesce(n.props.normalized_text, x.normalized_text)
                )
                FOREACH (_ IN CASE WHEN n.type = 'document' THEN [1] ELSE [] END |
                    SET x.title = coalesce(n.props.title, x.title),
                        x.published_at = coalesce(n.props.published_at, x.published_at)
                )
                RETURN count(x) AS touched
                """
                summary = sess.run(cy_nodes, params).consume()
                res.nodes_created += summary.counters.nodes_created

            # EDGES: Pure Cypher MERGE on generic :REL with type as a property (no APOC)
            if edges:
                params = {"batch": [_edge_map(e) for e in edges]}
                cy_edges = """
                UNWIND $batch AS e
                MATCH (s:Node {id:e.src_id, tenant_id:e.tenant_id})
                MATCH (d:Node {id:e.dst_id, tenant_id:e.tenant_id})
                MERGE (s)-[r:REL {tenant_id:e.tenant_id, type:e.type, src_id:e.src_id, dst_id:e.dst_id}]->(d)
                ON CREATE SET r.id = e.id, r.created_at = e.created_at, r.props_json = e.props_json
                SET r.props_json = e.props_json
                RETURN count(r) AS touched
                """
                summary = sess.run(cy_edges, params).consume()
                res.edges_created += summary.counters.relationships_created

        return res

    # ------------- queries -------------

    def get_node_by_key(self, tenant_id: str, kind: NodeKind, key: str) -> Optional[Node]:
        q = """
        MATCH (n:Node {tenant_id:$tenant_id, type:$type, key:$key})
        RETURN n
        LIMIT 1
        """
        with self._driver.session(database=self._db) as sess:
            rec = sess.run(q, {"tenant_id": tenant_id, "type": kind.value, "key": key}).single()
            return _record_to_node(rec) if rec else None

    def list_nodes(self, tenant_id: str, kind: Optional[NodeKind] = None, limit: int = 100, offset: int = 0) -> List[Node]:
        if kind is None:
            q = """
            MATCH (n:Node {tenant_id:$tenant_id})
            RETURN n
            SKIP $offset LIMIT $limit
            """
            params = {"tenant_id": tenant_id, "offset": offset, "limit": limit}
        else:
            q = """
            MATCH (n:Node {tenant_id:$tenant_id, type:$type})
            RETURN n
            SKIP $offset LIMIT $limit
            """
            params = {"tenant_id": tenant_id, "type": kind.value, "offset": offset, "limit": limit}
        with self._driver.session(database=self._db) as sess:
            return [_record_to_node(r) for r in sess.run(q, params)]

    def list_edges(
        self,
        tenant_id: str,
        kind: Optional[EdgeKind] = None,
        src_id: Optional[UUID] = None,
        dst_id: Optional[UUID] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Edge]:
        where = ["r.tenant_id = $tenant_id"]
        params: Dict[str, object] = {"tenant_id": tenant_id, "offset": offset, "limit": limit}
        if kind is not None:
            where.append("r.type = $etype")
            params["etype"] = kind.value
        if src_id is not None:
            where.append("s.id = $src")
            params["src"] = str(src_id)
        if dst_id is not None:
            where.append("d.id = $dst")
            params["dst"] = str(dst_id)
        q = f"""
        MATCH (s:Node)-[r]->(d:Node)
        WHERE {' AND '.join(where)}
        RETURN r, s.id AS src_id, d.id AS dst_id
        SKIP $offset LIMIT $limit
        """
        with self._driver.session(database=self._db) as sess:
            return [_record_to_edge_from_row(r) for r in list(sess.run(q, params))]

    def subgraph_from_entity(
        self,
        tenant_id: str,
        entity_key: str,
        *,
        depth: int = 1,
        max_neighbours: int = 25,
    ):
        """
        Small BFS via iterative Cypher to honour per-node caps.
        """
        q_root = """
        MATCH (n:Node {tenant_id:$tenant_id, type:'entity', key:$key})
        RETURN n
        LIMIT 1
        """
        with self._driver.session(database=self._db) as sess:
            rec = sess.run(q_root, {"tenant_id": tenant_id, "key": entity_key}).single()
            if not rec:
                return [], []
            root = _record_to_node(rec)

            seen_nodes: Dict[str, Node] = {str(root.id): root}
            seen_edges: Dict[str, Edge] = {}
            frontier = [str(root.id)]

            for _ in range(max(0, depth)):
                next_frontier: List[str] = []
                for nid in frontier:
                    # outgoing
                    q_out = """
                    MATCH (s:Node {id:$nid})-[r]->(d:Node)
                    WHERE r.tenant_id = $tenant_id
                    RETURN r, s.id AS src_id, d.id AS dst_id
                    LIMIT $limit
                    """
                    for row in sess.run(q_out, {"nid": nid, "tenant_id": tenant_id, "limit": max_neighbours}):
                        e = _record_to_edge_from_row(row)
                        seen_edges[str(e.id)] = e
                        for node_id in (str(e.src_id), str(e.dst_id)):
                            recn = sess.run("MATCH (n:Node {id:$id}) RETURN n LIMIT 1", {"id": node_id}).single()
                            if recn:
                                n = _record_to_node(recn)
                                if str(n.id) not in seen_nodes:
                                    seen_nodes[str(n.id)] = n
                                    next_frontier.append(str(n.id))

                    # incoming
                    q_in = """
                    MATCH (s:Node)-[r]->(d:Node {id:$nid})
                    WHERE r.tenant_id = $tenant_id
                    RETURN r, s.id AS src_id, d.id AS dst_id
                    LIMIT $limit
                    """
                    for row in sess.run(q_in, {"nid": nid, "tenant_id": tenant_id, "limit": max_neighbours}):
                        e = _record_to_edge_from_row(row)
                        seen_edges[str(e.id)] = e
                        for node_id in (str(e.src_id), str(e.dst_id)):
                            recn = sess.run("MATCH (n:Node {id:$id}) RETURN n LIMIT 1", {"id": node_id}).single()
                            if recn:
                                n = _record_to_node(recn)
                                if str(n.id) not in seen_nodes:
                                    seen_nodes[str(n.id)] = n
                                    next_frontier.append(str(n.id))

                frontier = next_frontier

        nodes = sorted(seen_nodes.values(), key=lambda n: (n.type.value, n.key))
        edges = sorted(seen_edges.values(), key=lambda e: (e.type.value, str(e.src_id), str(e.dst_id)))
        return nodes, edges

    # ------------- lifecycle -------------

    def close(self) -> None:
        if getattr(self, "_driver", None):
            self._driver.close()
