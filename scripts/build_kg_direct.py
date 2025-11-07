#!/usr/bin/env python3
"""
Directly call /kg/build (no /openapi.json), providing all required fields.

Adds support for:
  --tenant-id        (or TENANT_ID env; default: "public")
  --entity-name      (auto-derived from --entity-key if omitted)
  --doc-id           (auto-derived slug if omitted)
  --chunks-json PATH (JSON file: list of chunk dicts)
  --text-file  PATH  (plain text file to auto-chunk into chunk dicts)

Chunk format we send (robust, minimal):
  { "id": "<doc_id>#chunk<N>", "text": "<chunk text>", "order": N, "meta": {"source":"<doc_id>"} }

Usage examples:
  pip install -q requests
  python scripts/build_kg_direct.py --base-url http://127.0.0.1:8000 \
    --tenant-id default \
    --entity-key "org:sainsburys ar 2023" \
    --entity-name "Sainsbury's Annual Report 2023" \
    --doc-id "doc:sainsburys_ar_2023" \
    --text-file /content/drive/MyDrive/AAA_Rag/data/sainsburys_ar_2023.txt \
    --validate false

  # Or, if you already have chunk JSON:
  python scripts/build_kg_direct.py --base-url http://127.0.0.1:8000 \
    --entity-key "org:sainsburys ar 2023" \
    --chunks-json /path/to/chunks.json \
    --validate false
"""
import argparse, json, os, sys, time, re
from typing import Dict, Any, List
import requests

def get_headers() -> Dict[str, str]:
    headers = {"accept": "application/json", "content-type": "application/json"}
    if os.getenv("API_KEY"):
        headers["x-api-key"] = os.getenv("API_KEY")
    if os.getenv("AUTH_BEARER"):
        headers["authorization"] = f"Bearer {os.getenv('AUTH_BEARER')}"
    return headers

def wait_health(base: str, headers: Dict[str, str], tries=30, delay=1.0):
    url = f"{base}/health"
    for _ in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=3)
            if r.status_code == 200:
                print("Health:", r.json())
                return True
        except Exception:
            pass
        time.sleep(delay)
    print(f"API not healthy at {url} after {tries} tries.")
    return False

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "doc"

def derive_entity_name(entity_key: str) -> str:
    # e.g., "org:sainsburys ar 2023" -> "sainsburys ar 2023"
    return entity_key.split(":", 1)[-1].strip() if ":" in entity_key else entity_key.strip()

def chunk_text(txt: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return []
    chunks = []
    start = 0
    n = len(txt)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(txt[start:end].strip())
        if end >= n:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]

def build_chunks_from_text(doc_id: str, text_file: str) -> List[Dict[str, Any]]:
    with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    texts = chunk_text(raw)
    chunks = []
    for i, t in enumerate(texts):
        chunks.append({
            "id": f"{doc_id}#chunk{i}",
            "text": t,
            "order": i,
            "meta": {"source": doc_id}
        })
    return chunks

def load_chunks_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("chunks JSON must be a list")
    # Ensure minimal keys exist
    out = []
    for i, c in enumerate(data):
        if isinstance(c, str):
            out.append({"id": f"chunk{i}", "text": c, "order": i})
        elif isinstance(c, dict):
            if "text" not in c:
                raise ValueError(f"chunk at index {i} missing 'text'")
            c.setdefault("id", f"chunk{i}")
            c.setdefault("order", i)
            out.append(c)
        else:
            raise ValueError(f"unsupported chunk type at index {i}: {type(c)}")
    return out

def parse_keyvals(kvs):
    out: Dict[str, Any] = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got: {kv}")
        k, v = kv.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        else:
            try:
                out[k] = json.loads(v)  # allow numbers/lists/objects
            except Exception:
                out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tenant-id", default=os.getenv("TENANT_ID", "public"))
    ap.add_argument("--entity-key", required=True)
    ap.add_argument("--entity-name", default=None)
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--chunks-json", default=None, help="Path to JSON file containing chunk list")
    ap.add_argument("--text-file", default=None, help="Path to a .txt file to auto-chunk")
    ap.add_argument("--validate", default="false", help="true|false")
    ap.add_argument("--set", action="append", help='extra fields: key=value (repeatable), value can be JSON')
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    headers = get_headers()

    # Ensure API is up
    if not wait_health(base, headers):
        sys.exit(1)

    # Derive entity_name and doc_id if missing
    entity_name = args.entity_name or derive_entity_name(args.entity_key)
    doc_id = args.doc_id or f"doc:{slugify(entity_name)}"

    # Build chunks
    chunks: List[Dict[str, Any]] = []
    if args.chunks_json:
        chunks = load_chunks_from_json(args.chunks_json)
    elif args.text_file:
        chunks = build_chunks_from_text(doc_id, args.text_file)
    else:
        # Minimal placeholder chunk so schema validation can pass; replace with real data for production
        chunks = [{"id": f"{doc_id}#chunk0", "text": entity_name, "order": 0, "meta": {"source": doc_id}}]

    # Assemble payload
    body: Dict[str, Any] = {
        "tenant_id": args.tenant_id,
        "entity_key": args.entity_key,
        "entity_name": entity_name,
        "doc_id": doc_id,
        "chunks": chunks,
        "validate": (str(args.validate).lower() == "true"),
    }
    # Merge any additional fields
    body.update(parse_keyvals(args.__dict__.get("set")))

    url = f"{base}/kg/build"
    print(f"\nPOST {url}\nPayload keys: {list(body.keys())}\n"
          f"tenant_id={body['tenant_id']}  entity_key={body['entity_key']}  "
          f"entity_name={body['entity_name']}  doc_id={body['doc_id']}  chunks={len(body['chunks'])}\n")
    try:
        r = requests.post(url, headers=headers, json=body, timeout=600)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(1)

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
        print("\nIf this is a 422 with specific fields, re-run by providing the missing ones.")
        print("If this is a 500, please show `tail -n 120 uvicorn.log`.")
        sys.exit(1)

if __name__ == "__main__":
    main()
