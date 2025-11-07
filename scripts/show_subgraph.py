#!/usr/bin/env python3
"""
Fetch and pretty-print a KG subgraph for a given entity_key.

Usage:
  pip install -q requests
  python scripts/show_subgraph.py --base-url http://127.0.0.1:8000 \
    --tenant-id default --entity-key "org:sainsburys ar 2023" --depth 1
"""
import argparse, json, sys
from typing import Any, Dict, List
import requests

def fetch_json(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print(f"FAIL: GET {url} -> {r.status_code} {r.text[:400]}")
        sys.exit(1)
    try:
        return r.json()
    except Exception:
        print(f"FAIL: Non-JSON from {url}: {r.text[:400]}")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tenant-id", default="default")
    ap.add_argument("--entity-key", required=True)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--limit", type=int, default=50, help="Just for display; not sent to API")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    url = f"{base}/kg/subgraph"

    data = fetch_json(url, {"tenant_id": args.tenant_id, "entity_key": args.entity_key, "depth": args.depth})

    nodes: List[Dict[str, Any]] = data.get("nodes") or data.get("data", {}).get("nodes") or data.get("Nodes") or []
    edges: List[Dict[str, Any]] = data.get("edges") or data.get("data", {}).get("edges") or data.get("Edges") or []

    print(f"\nSubgraph for entity_key='{args.entity_key}' (tenant='{args.tenant_id}', depth={args.depth})")
    print(f"Nodes: {len(nodes)}  Edges: {len(edges)}")

    def safe(v, k): 
        if isinstance(v, dict): 
            return v.get(k)
        return None

    # Show a compact preview
    print("\n— Nodes (up to {args.limit}) —")
    for i, n in enumerate(nodes[:args.limit]):
        nid = safe(n, "id") or n.get("id", "")
        ntype = safe(n, "type") or n.get("type", "")
        key = safe(n, "key") or n.get("key", "")
        label = safe(n, "label") or n.get("label", "")
        print(f"  • [{ntype}] id={nid} key={key} label={label}")

    print("\n— Edges (up to {args.limit}) —")
    for i, e in enumerate(edges[:args.limit]):
        etype = safe(e, "type") or e.get("type", "")
        src = safe(e, "src_id") or e.get("src_id", "")
        dst = safe(e, "dst_id") or e.get("dst_id", "")
        print(f"  • ({etype}) {src} -> {dst}")

    # Dump raw JSON to a file for deeper debugging if needed
    out_path = "subgraph_preview.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved raw subgraph to {out_path}")
    print("Done.")
if __name__ == "__main__":
    main()
