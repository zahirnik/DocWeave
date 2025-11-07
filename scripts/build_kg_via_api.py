#!/usr/bin/env python3
"""
Call /kg/build safely.

- Waits for API /health
- Discovers request schema from /openapi.json
- Builds a payload with --entity-key (default provided) and --set extras
- Optional auth header via env: API_KEY or AUTH_BEARER

Usage:
  pip install -q requests
  python scripts/build_kg_via_api.py --base-url http://127.0.0.1:8000 \
    --entity-key "org:sainsburys ar 2023" --validate false \
    --set source="annual report 2023"
"""

import argparse, json, sys, time, os
from typing import Any, Dict
import requests

def die(msg: str, code: int = 1):
    print(msg)
    sys.exit(code)

def get_headers() -> Dict[str, str]:
    headers = {"accept": "application/json"}
    api_key = os.getenv("API_KEY")  # e.g., x-api-key style
    bearer = os.getenv("AUTH_BEARER")  # raw token (no "Bearer " prefix)
    if api_key:
        headers["x-api-key"] = api_key
    if bearer:
        headers["authorization"] = f"Bearer {bearer}"
    return headers

def fetch_json(url: str, headers: Dict[str, str], timeout=20) -> Any:
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code != 200:
        die(f"FAIL: GET {url} -> {r.status_code} {r.text[:400]}")
    try:
        return r.json()
    except Exception:
        die(f"FAIL: Non-JSON from {url}: {r.text[:400]}")

def resolve_ref(components: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    if "$ref" in schema:
        ref = schema["$ref"]
        parts = ref.split("/")
        if len(parts) >= 4 and parts[1] == "components" and parts[2] == "schemas":
            name = parts[-1]
            return components.get("schemas", {}).get(name, {})
    return schema

def parse_keyvals(kvs):
    out: Dict[str, Any] = {}
    for kv in kvs or []:
        if "=" not in kv:
            die(f"--set expects key=value, got: {kv}")
        k, v = kv.split("=", 1)
        v = v.strip()
        # Try to coerce to bool/int/json when obvious
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        elif v.isdigit():
            out[k] = int(v)
        else:
            try:
                out[k] = json.loads(v)
            except Exception:
                out[k] = v
    return out

def build_payload(schema: Dict[str, Any], defaults: Dict[str, Any]):
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    body = dict(defaults)
    missing = []
    for req in required:
        if req not in body:
            typ = (props.get(req) or {}).get("type")
            if req == "validate":
                body[req] = False
            elif req == "entity_key":
                missing.append(req)
            elif typ == "array":
                body[req] = []
            elif typ == "object":
                body[req] = {}
            elif typ == "boolean":
                body[req] = False
            elif typ in ("number", "integer"):
                body[req] = 0
            else:
                missing.append(req)
    return body, missing

def wait_health(base: str, headers: Dict[str, str], tries=30, delay=1.0):
    url = f"{base}/health"
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                print("Health:", r.json())
                return
        except Exception:
            pass
        time.sleep(delay)
    die(f"API not healthy at {url} after {tries} tries.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--entity-key", default="org:sainsburys ar 2023")
    ap.add_argument("--validate", default="false", help="true|false")
    ap.add_argument("--set", action="append", help='extra fields: key=value (repeatable), value can be JSON')
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    headers = get_headers()

    # 1) Wait for health
    wait_health(base, headers)

    # 2) Discover schema
    spec = fetch_json(f"{base}/openapi.json", headers=headers)
    paths = spec.get("paths", {})
    kg_build = paths.get("/kg/build") or paths.get("/kg/build/")
    if not kg_build or "post" not in kg_build:
        candidates = [p for p in paths if p.startswith("/kg")]
        die(f"Could not find POST /kg/build. Available /kg* paths: {candidates}")

    req_schema = kg_build["post"].get("requestBody", {}) \
        .get("content", {}) \
        .get("application/json", {}) \
        .get("schema", {})
    schema = resolve_ref(spec.get("components", {}), req_schema)
    if not schema:
        die("Could not resolve request schema for /kg/build")

    defaults = {
        "entity_key": args.entity_key,
        "validate": (str(args.validate).lower() == "true"),
    }
    defaults.update(parse_keyvals(args.set))

    body, missing = build_payload(schema, defaults)
    if missing:
        print("NOTE: Required fields missing:", missing)
        print("Schema properties:", list((schema.get("properties") or {}).keys()))
        print("TIP: Re-run with --set field=value for each missing field (value can be JSON).")

    # 3) POST /kg/build
    url = f"{base}/kg/build"
    print(f"\nPOST {url}\nPayload:\n{json.dumps(body, indent=2)}\n")
    r = requests.post(url, headers={"content-type": "application/json", **headers}, json=body, timeout=180)

    print(f"Status: {r.status_code}")
    ctype = r.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            print(json.dumps(r.json(), indent=2)[:4000])
        except Exception:
            print(r.text[:4000])
    else:
        print(r.text[:4000])

    if not (200 <= r.status_code < 300):
        sys.exit(1)

if __name__ == "__main__":
    main()
