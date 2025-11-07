#!/usr/bin/env python3
"""
Deep Postgres diagnostics:
- Validates DSN format (without printing secrets)
- Resolves DNS (IPv4/IPv6)
- Tries raw TCP connects to host:port (both families)
- Attempts psycopg2 connection (as-is), then with IPv4 hostaddr, and (if Supabase) pooled port 6543
- Prints clear PASS/FAIL reasons

Usage:
  pip install -q python-dotenv psycopg2-binary
  python scripts/diag_postgres.py --dsn "postgresql://postgres:tDgswkAxH7sW2Q3w@db.pqutgotkkevyfzkcpfrp.supabase.co:5432/postgres"
  # or rely on env:
  export DATABASE_URL="postgresql://postgres:tDgswkAxH7sW2Q3w@db.pqutgotkkevyfzkcpfrp.supabase.co:5432/postgres"
  python scripts/diag_postgres.py
"""

import argparse, os, socket, ipaddress, sys
from contextlib import closing

# Optional .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

def mask(s: str) -> str:
    if not s:
        return s
    return "****" if len(s) <= 8 else (s[:2] + "****" + s[-2:])

def parse_dsn(dsn: str):
    # Minimal URI parser (avoid printing secrets)
    # Expected: postgresql://user:pass@host:port/db?params
    from urllib.parse import urlparse, parse_qs
    u = urlparse(dsn)
    if u.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"Unsupported scheme '{u.scheme}', expected postgresql:// or postgres://")
    user = u.username or ""
    pwd  = u.password or ""
    host = u.hostname or ""
    port = u.port or 5432
    db   = (u.path or "/").lstrip("/") or "postgres"
    qs   = parse_qs(u.query or "")
    return {
        "scheme": u.scheme,
        "user": user,
        "password": pwd,
        "host": host,
        "port": int(port),
        "db": db,
        "params": {k: v[0] for k, v in qs.items()},
        "raw": dsn,
    }

def print_header(t): print("\n" + "─"*8 + f" {t} " + "─"*8)

def resolve_all(host: str):
    v4, v6 = [], []
    try:
        for fam in (socket.AF_INET, socket.AF_INET6):
            try:
                infos = socket.getaddrinfo(host, None, family=fam, type=socket.SOCK_STREAM)
                for ai in infos:
                    addr = ai[4][0]
                    if fam == socket.AF_INET:
                        if addr not in v4: v4.append(addr)
                    else:
                        if addr not in v6: v6.append(addr)
            except socket.gaierror:
                pass
    except Exception as e:
        print(f"DNS error for {host}: {e}")
    return v4, v6

def tcp_probe(addr: str, port: int, timeout=3.0):
    family = socket.AF_INET6 if ":" in addr else socket.AF_INET
    with closing(socket.socket(family, socket.SOCK_STREAM)) as s:
        s.settimeout(timeout)
        try:
            s.connect((addr, port))
            return True, None
        except Exception as e:
            return False, str(e)

def try_psycopg2(params: dict, hostaddr: str | None = None, port_override: int | None = None, require_ssl: bool = True):
    try:
        import psycopg2  # type: ignore
    except ModuleNotFoundError:
        print("psycopg2 not installed. Install with: pip install psycopg2-binary")
        return False, "psycopg2 missing"

    kw = {
        "host": params["host"],
        "dbname": params["db"],
        "user": params["user"],
        "password": params["password"],
        "port": params["port"],
    }
    if hostaddr:
        kw["hostaddr"] = hostaddr  # forces IPv4/IPv6 selection via libpq
    if port_override:
        kw["port"] = port_override
    if require_ssl:
        # If user already set sslmode in DSN, libpq will use that; adding here ensures TLS if none provided.
        kw["sslmode"] = "require"

    try:
        conn = psycopg2.connect(**kw)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()[0]
            cur.execute("SHOW server_version;")
            sv = cur.fetchone()[0]
            cur.execute("SELECT inet_server_addr(), inet_client_addr();")
            addrs = cur.fetchone()
        conn.close()
        return True, {"version": ver, "server_version": sv, "socket": addrs}
    except Exception as e:
        return False, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", help="Postgres DSN URI")
    args = ap.parse_args()

    dsn = args.dsn or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN") or ""
    if not dsn:
        print("FAIL: No DSN provided. Use --dsn or set DATABASE_URL.")
        sys.exit(2)

    print_header("DSN parse")
    try:
        params = parse_dsn(dsn)
    except Exception as e:
        print(f"FAIL: DSN parse error: {e}")
        sys.exit(2)

    safe_user = params["user"]
    safe_host = params["host"]
    print(f"Host: {safe_host}  Port: {params['port']}  DB: {params['db']}  User: {safe_user}")
    if "sslmode" not in (params["params"] or {}):
        print("Note: sslmode not in DSN; will enforce sslmode=require for tests.")

    print_header("DNS resolution")
    v4, v6 = resolve_all(params["host"])
    if v4: print("IPv4:", ", ".join(v4))
    else:  print("IPv4: none")
    if v6: print("IPv6:", ", ".join(v6))
    else:  print("IPv6: none")

    print_header("TCP reachability")
    any_ok = False
    for addr in v4 + v6:
        ok, err = tcp_probe(addr, params["port"])
        tag = "OK" if ok else f"FAIL ({err})"
        print(f"{addr}:{params['port']} -> {tag}")
        any_ok = any_ok or ok
    if not any_ok:
        print("No raw TCP path succeeded. Likely egress or firewall issue (or wrong port).")

    print_header("psycopg2 connect (as-is)")
    ok, info = try_psycopg2(params, hostaddr=None, port_override=None, require_ssl=True)
    if ok:
        print("PASS:", info["server_version"])
        print("Socket:", info["socket"])
        sys.exit(0)
    else:
        print("FAIL:", info)

    # If as-is fails, try forcing IPv4 if available
    if v4:
        print_header("psycopg2 connect (force IPv4 via hostaddr)")
        ok, info = try_psycopg2(params, hostaddr=v4[0], port_override=None, require_ssl=True)
        if ok:
            print("PASS (IPv4):", info["server_version"])
            print("Socket:", info["socket"])
            sys.exit(0)
        else:
            print("FAIL (IPv4):", info)

    # Supabase: also try pooled port 6543 (PgBouncer)
    is_supabase = params["host"].endswith(".supabase.co")
    if is_supabase:
        print_header("psycopg2 connect (Supabase pooled port 6543)")
        ok, info = try_psycopg2(params, hostaddr=(v4[0] if v4 else None), port_override=6543, require_ssl=True)
        if ok:
            print("PASS (pgbouncer 6543):", info["server_version"])
            print("Socket:", info["socket"])
            sys.exit(0)
        else:
            print("FAIL (6543):", info)

    print_header("Result")
    print("Could not establish a Postgres connection. Review the failures above:")
    print("- If TCP to all addresses failed: network/egress/firewall or IPv6-only issue.")
    print("- If TCP succeeded but psycopg2 failed: check user/pass/DB name and URL-encoding for special characters.")
    print("- Managed DBs may require sslmode=require (we enforced it).")
    sys.exit(2)

if __name__ == "__main__":
    main()
