#!/usr/bin/env python3
"""
Ingest a PDF or TXT, chunk it, and POST to /kg/build.

Examples:
  # From a PDF
  python scripts/ingest_doc_and_build.py --base-url http://127.0.0.1:8000 \
    --tenant-id default \
    --entity-key "org:sainsburys ar 2023" \
    --entity-name "Sainsbury's Annual Report 2023" \
    --doc-id doc:sainsburys-ar-2023 \
    --pdf /path/to/Sainsburys_AR_2023.pdf

  # From a TXT
  python scripts/ingest_doc_and_build.py --base-url http://127.0.0.1:8000 \
    --tenant-id default \
    --entity-key "org:sainsburys ar 2023" \
    --text /path/to/sainsburys_ar_2023.txt
"""
import argparse, json, os, re, sys, time
from typing import Any, Dict, List
import requests

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "doc"

def wait_health(base: str, tries=30, delay=1.0):
    url = f"{base}/health"
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    print(f"API not healthy at {url} after {tries} tries.", file=sys.stderr)
    return False

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    # Try pymupdf (best), then pypdf as fallback.
    try:
        import fitz  # pymupdf
        doc = fitz.open(path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        doc.close()
        return "\n".join(texts)
    except Exception:
        try:
            from pypdf import PdfReader  # pip install pypdf
            reader = PdfReader(path)
            texts = []
            for pg in reader.pages:
                try:
                    texts.append(pg.extract_text() or "")
                except Exception:
                    texts.append("")
            return "\n".join(texts)
        except Exception as e:
            print("Failed to read PDF. Install dependencies:\n  pip install -q pymupdf pypdf", file=sys.stderr)
            raise

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(txt: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    txt = normalize_ws(txt)
    if not txt:
        return []
    chunks, n = [], len(txt)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(txt[start:end].strip())
        if end >= n: break
        start = max(0, end - overlap)
    return [c for c in chunks if c]

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
                out[k] = json.loads(v)  # numbers/lists/objects allowed
            except Exception:
                out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tenant-id", default="default")
    ap.add_argument("--entity-key", required=True)
    ap.add_argument("--entity-name", default=None)
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--pdf", default=None, help="Path to a PDF to ingest")
    ap.add_argument("--text", default=None, help="Path to a TXT to ingest")
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--validate", default="false", help="true|false")
    ap.add_argument("--set", action="append", help='extra fields: key=value (repeatable), value can be JSON')
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    if not wait_health(base):
        sys.exit(1)

    entity_name = args.entity_name or (args.entity_key.split(":", 1)[-1].strip())
    doc_id = args.doc_id or f"doc:{slugify(entity_name)}"

    # Load raw text
    if args.pdf:
        raw = read_pdf(args.pdf)
    elif args.text:
        raw = read_txt(args.text)
    else:
        print("Provide --pdf PATH or --text PATH", file=sys.stderr)
        sys.exit(2)

    # Chunk
    texts = chunk_text(raw, max_chars=args.max_chars, overlap=args.overlap)
    if not texts:
        print("No text extracted/chunked; aborting.", file=sys.stderr)
        sys.exit(2)

    # Build chunks payload
    chunks = []
    for i, t in enumerate(texts):
        chunks.append({
            "id": f"{doc_id}#chunk{i}",
            "text": t,
            "order": i,
            "meta": {"source": doc_id}
        })

    body: Dict[str, Any] = {
        "tenant_id": args.tenant_id,
        "entity_key": args.entity_key,
        "entity_name": entity_name,
        "doc_id": doc_id,
        "chunks": chunks,
        "validate": (str(args.validate).lower() == "true"),
    }
    body.update(parse_keyvals(args.set))

    url = f"{base}/kg/build"
    print(f"\nPOST {url}")
    print(f"tenant_id={body['tenant_id']} entity_key={body['entity_key']} entity_name={body['entity_name']} doc_id={body['doc_id']}")
    print(f"chunks={len(chunks)} (max_chars={args.max_chars}, overlap={args.overlap})\n")

    headers = {"accept": "application/json", "content-type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=900)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
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
        print("\nIf 422: add any missing fields via --set key=value")
        print("If 500: tail -n 200 uvicorn.log")
        sys.exit(1)

if __name__ == "__main__":
    main()
