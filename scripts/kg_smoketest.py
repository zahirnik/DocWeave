
# scripts/kg_smoketest.py
# Simple Neo4j connectivity + write/read smoke test for the KG setup.
# - Uses plain Cypher MERGE (no APOC required).
# - Fixes the earlier syntax error (d.published_at=$published).
# - Uses timezone-aware UTC timestamps.

import os
import json
from datetime import datetime, timezone
from uuid import uuid4

from neo4j import GraphDatabase

def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def main() -> None:
    uri = env("NEO4J_URI")
    user = env("NEO4J_USER")
    pwd  = env("NEO4J_PASSWORD")

    tenant = os.environ.get("KG_TENANT", "public")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    now_iso = datetime.now(timezone.utc).isoformat()

    doc_id = f"smoke-{uuid4()}".replace("-", "")
    key = f"document:{doc_id}"
    label = "Smoke Test Document"
    props = {"title": "Smoke Test Doc", "published_at": "2025-01-01"}

    cy = """
    // 1) Ping
    RETURN 1 AS ok
    """
    cy_upsert = """
    // 2) Upsert a minimal Document node with both :Node and :Document labels
    MERGE (d:Node:Document {tenant_id:$tenant, type:'document', key:$key})
    ON CREATE SET d.id=$id, d.label=$label, d.props_json=$props_json, d.created_at=$now,
                  d.title=$title, d.published_at=$published
    // keep label in sync on repeated runs
    SET d.label=$label
    RETURN d.id AS id, d.title AS title, d.published_at AS published
    """
    cy_count = """
    // 3) Count documents for this tenant
    MATCH (d:Document {tenant_id:$tenant})
    RETURN count(d) AS n_docs
    """

    with driver.session() as session:
        # 1) ping
        r1 = session.run(cy).single()
        assert r1 and r1["ok"] == 1, "Ping failed"

        # 2) upsert doc
        params = {
            "tenant": tenant,
            "id": doc_id,
            "key": key,
            "label": label,
            "props_json": json.dumps(props, ensure_ascii=False, separators=(",", ":")),
            "now": now_iso,
            "title": props["title"],
            "published": props["published_at"],
        }
        r2 = session.run(cy_upsert, **params).single()
        assert r2, "Upsert returned no row"

        # 3) count
        r3 = session.run(cy_count, tenant=tenant).single()
        n_docs = r3["n_docs"] if r3 else 0

    driver.close()
    print("OK: Neo4j smoke test. Document upserted.",
          f"tenant={tenant} id={doc_id} title={props['title']} total_docs={n_docs}", sep="\n")

if __name__ == "__main__":
    main()
