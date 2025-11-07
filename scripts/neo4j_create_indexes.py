#!/usr/bin/env python3
"""
Create Neo4j constraints & indexes (Neo4j 5.x syntax).
Reads NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, optional NEO4J_DATABASE from env.
Safe to re-run: uses IF NOT EXISTS.
"""
import os, sys
from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD  = os.getenv("NEO4J_PASSWORD", "neo4j")
DB   = os.getenv("NEO4J_DATABASE")  # None = default

DDL = [
    # Uniqueness for programmatic IDs
    "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE",

    # Helpful lookups
    "CREATE INDEX node_identity IF NOT EXISTS FOR (n:Node) ON (n.tenant_id, n.type, n.key)",

    # Per-type fast lookups
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX entity_alt_names IF NOT EXISTS FOR (n:Entity) ON (n.alt_names)",
    "CREATE INDEX metric_name IF NOT EXISTS FOR (n:Metric) ON (n.name)",
    "CREATE INDEX claim_hash IF NOT EXISTS FOR (n:Claim) ON (n.hash)",
    "CREATE INDEX claim_norm_text IF NOT EXISTS FOR (n:Claim) ON (n.normalized_text)",
    "CREATE INDEX doc_title_pub IF NOT EXISTS FOR (n:Document) ON (n.title, n.published_at)",

    # Relationship property indexes (we store edge type & ids as properties on :REL)
    "CREATE INDEX rel_type IF NOT EXISTS FOR ()-[r:REL]-() ON (r.type)",
    "CREATE INDEX rel_tenant IF NOT EXISTS FOR ()-[r:REL]-() ON (r.tenant_id)",
    "CREATE INDEX rel_src IF NOT EXISTS FOR ()-[r:REL]-() ON (r.src_id)",
    "CREATE INDEX rel_dst IF NOT EXISTS FOR ()-[r:REL]-() ON (r.dst_id)",
]

def main():
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    with drv.session(database=DB) as s:
        for stmt in DDL:
            try:
                s.run(stmt).consume()
                print("OK:", stmt)
            except Exception as e:
                print("ERR:", stmt, "->", e)
                # keep going; some features vary across editions
    drv.close()
    print("Done.")

if __name__ == "__main__":
    if not (URI and USER and PWD):
        print("Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (and optionally NEO4J_DATABASE).", file=sys.stderr)
        sys.exit(2)
    main()
