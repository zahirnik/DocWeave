#!/usr/bin/env python3
"""
Plot the whole knowledge graph (nodes & edges) for a tenant from Neo4j into an interactive HTML (PyVis).

Usage (after saving this file):
  pip install -q neo4j pyvis
  python scripts/plot_full_graph.py \
    --tenant-id default \
    --limit-nodes 20000 \
    --limit-edges 40000 \
    --outfile graph_all.html

Reads Neo4j creds from env by default:
  NEO4J_URI=neo4j+s://...:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=...

You can also pass them as flags:
  --uri ... --user ... --password ...
"""

from __future__ import annotations
import argparse, os, sys
from typing import List, Dict

try:
    from neo4j import GraphDatabase
except Exception as e:
    print("Missing neo4j driver. Install with: pip install neo4j", file=sys.stderr)
    raise

try:
    from pyvis.network import Network
except Exception as e:
    print("Missing pyvis. Install with: pip install pyvis", file=sys.stderr)
    raise


def fetch_nodes_edges(uri: str, user: str, password: str, tenant: str,
                      limit_nodes: int, limit_edges: int):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes: List[Dict] = []
    edges: List[Dict] = []
    with driver.session() as sess:
        q_nodes = """
        MATCH (n:Node {tenant_id:$tenant})
        RETURN n.id AS id,
               n.type AS type,
               coalesce(n.label, n.key) AS label,
               n.key AS key
        LIMIT $limit_nodes
        """
        nodes = [dict(r) for r in sess.run(q_nodes, {"tenant": tenant, "limit_nodes": limit_nodes})]

        q_edges = """
        MATCH (s:Node {tenant_id:$tenant})-[r]->(d:Node {tenant_id:$tenant})
        RETURN coalesce(r.id, s.id + ':' + type(r) + ':' + d.id) AS id,
               type(r) AS type,
               s.id AS src,
               d.id AS dst
        LIMIT $limit_edges
        """
        edges = [dict(r) for r in sess.run(q_edges, {"tenant": tenant, "limit_edges": limit_edges})]
    driver.close()
    return nodes, edges


def build_net(nodes: List[Dict], edges: List[Dict], height="800px", width="100%"):
    net = Network(height=height, width=width, bgcolor="#ffffff", font_color="#222222", directed=True)
    net.show_buttons(filter_=['physics'])

    # Add nodes (group by type for color/legend)
    for n in nodes:
        nid = n["id"]
        label = n.get("label") or n.get("key") or nid
        title = f"type: {n.get('type','?')}\nkey: {n.get('key','')}\nid: {nid}"
        group = n.get("type", "unknown")
        net.add_node(nid, label=label, title=title, group=group)

    # Add edges
    for e in edges:
        src, dst = e["src"], e["dst"]
        et = e.get("type", "")
        net.add_edge(src, dst, label=et, title=et, arrows="to")

    # ✅ Valid JSON options (fixes JSONDecodeError)
    net.set_options("""
    {
      "nodes": { "shape": "dot", "size": 12 },
      "edges": { "arrows": { "to": { "enabled": true } }, "smooth": false },
      "physics": {
        "barnesHut": { "gravitationalConstant": -8000, "springLength": 190, "springConstant": 0.02 },
        "stabilization": { "iterations": 250 }
      },
      "interaction": { "hover": true, "tooltipDelay": 120, "hideEdgesOnDrag": false }
    }
    """)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant-id", default="default")
    ap.add_argument("--limit-nodes", type=int, default=20000, help="Safety cap; set high to fetch 'everything'")
    ap.add_argument("--limit-edges", type=int, default=40000, help="Safety cap; set high to fetch 'everything'")
    ap.add_argument("--outfile", default="graph_all.html")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    args = ap.parse_args()

    print(f"Connecting to {args.uri} as {args.user}; tenant={args.tenant_id}")
    nodes, edges = fetch_nodes_edges(
        uri=args.uri,
        user=args.user,
        password=args.password,
        tenant=args.tenant_id,
        limit_nodes=args.limit_nodes,
        limit_edges=args.limit_edges,
    )
    print(f"Fetched: {len(nodes)} nodes, {len(edges)} edges")

    net = build_net(nodes, edges)
    net.write_html(args.outfile)
    print(f"Wrote {args.outfile} (open in your browser)")

if __name__ == "__main__":
    main()
