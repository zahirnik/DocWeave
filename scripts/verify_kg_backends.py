#!/usr/bin/env python3
"""
Verify connectivity to Postgres/pgvector and Neo4j before running the KG.
- Loads .env if present
- Tries common env var names used across RAG/KG stacks
- Prints clear PASS/FAIL and actionable hints

Usage:
  pip install -q python-dotenv psycopg2-binary neo4j
  python scripts/verify_kg_backends.py
"""

import os
import sys
from contextlib import contextmanager

# --- Optional .env loader ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass  # Keep going even if python-dotenv isn't installed

def pick_first(*vals):
    for v in vals:
        if v and str(v).strip():
            return v
    return None

def print_header(title):
    print("\n" + "─" * 8 + f" {title} " + "─" * 8)

@contextmanager
def maybe_import(mod_name, pip_hint=None):
    try:
        mod = __import__(mod_name)
        yield mod
    except ModuleNotFoundError as e:
        print(f"FAIL: Python module '{mod_name}' not found.")
        if pip_hint:
            print(f"      Install with: {pip_hint}")
        raise e

def check_postgres():
    print_header("Postgres / pgvector")
    dsn = pick_first(
        os.getenv("DATABASE_URL"),
        os.getenv("POSTGRES_DSN"),
        os.getenv("PGVECTOR_URL"),
        os.getenv("PG_DSN"),
        os.getenv("PG_URL"),
    )
    if not dsn:
        print("FAIL: No Postgres DSN found in env. Tried: DATABASE_URL, POSTGRES_DSN, PGVECTOR_URL, PG_DSN, PG_URL")
        print("      Example: DATABASE_URL=postgresql://user:pass@host:5432/dbname")
        return False

    sslmode = os.getenv("PG_SSLMODE", "prefer")

    try:
        with maybe_import("psycopg2", "pip install psycopg2-binary"):
            import psycopg2  # type: ignore
            conn = psycopg2.connect(dsn, sslmode=sslmode)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT version();")
            ver = cur.fetchone()[0]
            print(f"PASS: Connected to Postgres ({ver})")
            # Optional pgvector check
            try:
                cur.execute("SELECT extname FROM pg_extension WHERE extname='vector';")
                row = cur.fetchone()
                if row and row[0] == "vector":
                    print("PASS: pgvector extension present")
                else:
                    print("WARN: pgvector extension not found (OK if you are not using pgvector)")
            except Exception as e:
                print(f"WARN: Could not verify pgvector extension: {e}")
            cur.close()
            conn.close()
            return True
    except Exception as e:
        print(f"FAIL: Postgres connection error: {e}")
        print("      Check host/port/firewall, creds, and that the DB is reachable from this environment.")
        return False

def check_neo4j():
    print_header("Neo4j (KG backend)")
    uri = pick_first(os.getenv("NEO4J_URI"), os.getenv("NEO4J_URL"))
    user = pick_first(os.getenv("NEO4J_USER"), os.getenv("NEO4J_USERNAME"))
    pwd  = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not pwd:
        print("FAIL: Missing NEO4J env. Need NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (or NEO4J_URL/NEO4J_USERNAME).")
        print("      Example: NEO4J_URI=bolt://localhost:7687  NEO4J_USER=neo4j  NEO4J_PASSWORD=******")
        return False

    try:
        with maybe_import("neo4j", "pip install neo4j"):
            from neo4j import GraphDatabase  # type: ignore
            driver = GraphDatabase.driver(uri, auth=(user, pwd))
            with driver.session() as session:
                result = session.run("RETURN 1 AS ok")
                ok = result.single()
                if ok and ok.get("ok") == 1:
                    print("PASS: Connected to Neo4j and ran a test query")
                else:
                    print("FAIL: Neo4j test query did not return expected result")
                    driver.close()
                    return False
            driver.close()
            return True
    except Exception as e:
        print(f"FAIL: Neo4j connection error: {e}")
        print("      Check bolt URL, creds, and that the DB allows connections from this environment.")
        return False

def main():
    any_fail = False
    pg_ok = check_postgres()
    if not pg_ok:
        any_fail = True
    neo_ok = check_neo4j()
    if not neo_ok:
        any_fail = True

    print_header("Summary")
    if any_fail:
        print("Some checks FAILED. Fix these before running the KG build.")
        sys.exit(2)
    else:
        print("All checks PASSED. You’re ready to launch the API and run the KG build.")
        sys.exit(0)

if __name__ == "__main__":
    main()
