#!/usr/bin/env bash
# One-shot launcher: verify backends → start API → ensure indexes → ingest (PDF or sample)
# → subgraph preview → optional HTML graph.

set -euo pipefail

PROJECT="/content/drive/MyDrive/AAA_Rag"
API_URL="http://127.0.0.1:8000"

TENANT="${TENANT:-default}"
ENTITY_KEY="${ENTITY_KEY:-org:sainsburys ar 2023}"
ENTITY_NAME="${ENTITY_NAME:-Sainsbury's Annual Report 2023}"
DOC_ID="${DOC_ID:-doc:sainsburys-ar-2023}"
PDF_PATH="${PDF_PATH:-}"

cd "$PROJECT"

echo "── KG launch — project: $PROJECT"
echo "tenant: $TENANT"
echo "entity_key: $ENTITY_KEY"
echo

echo "▶ Installing minimal Python deps (quiet)…"
pip -q install requests neo4j pyvis pymupdf pypdf >/dev/null

echo "▶ Verifying backends…"
python scripts/verify_kg_backends.py

# Start (or restart) API and wait for /health
if [ -f scripts/restart_api_debug.sh ]; then
  echo "▶ Starting API via restart_api_debug.sh…"
  bash scripts/restart_api_debug.sh
else
  echo "▶ Starting API inline (uvicorn)…"
  if [ -f uvicorn.pid ] && ps -p "$(cat uvicorn.pid)" >/dev/null 2>&1; then
    echo "  killing prior PID $(cat uvicorn.pid)…"
    kill -9 "$(cat uvicorn.pid)" || true
    rm -f uvicorn.pid
  fi
  nohup python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --log-level info > uvicorn.log 2>&1 &
  echo $! > uvicorn.pid
  echo "  PID $(cat uvicorn.pid) written. Waiting for /health…"
  for i in $(seq 1 30); do
    if curl -s "$API_URL/health" | grep -q '"status"'; then
      echo "  Health: $(curl -s "$API_URL/health")"
      break
    fi
    sleep 1
    if [ "$i" -eq 30 ]; then
      echo "ERROR: API did not get healthy"
      exit 1
    fi
  done
fi

# Ensure Neo4j constraints & indexes if helper exists
if [ -f scripts/ensure_neo4j_indexes.py ]; then
  echo "▶ Ensuring Neo4j constraints & indexes…"
  python scripts/ensure_neo4j_indexes.py || true
else
  echo "ℹ Skipping ensure_neo4j_indexes.py (file not found)."
fi

# Decide which PDF to use (explicit path if exists; else try to discover one)
PDF_TO_USE=""
if [ -n "$PDF_PATH" ] && [ -f "$PDF_PATH" ]; then
  PDF_TO_USE="$PDF_PATH"
else
  # try to find a likely Sainsbury PDF under repo
  PDF_TO_USE="$(find . -type f -iname '*sainsbury*' -iname '*.pdf' 2>/dev/null | head -n 1 || true)"
fi

if [ -n "$PDF_TO_USE" ] && [ -f "$PDF_TO_USE" ]; then
  echo "▶ Ingesting from PDF: $PDF_TO_USE"
  python scripts/ingest_doc_and_build.py \
    --base-url "$API_URL" \
    --tenant-id "$TENANT" \
    --entity-key "$ENTITY_KEY" \
    --entity-name "$ENTITY_NAME" \
    --doc-id "$DOC_ID" \
    --pdf "$PDF_TO_USE" \
    --validate false
else
  echo "▶ No PDF found; posting a minimal sample build to /kg/build…"
  python scripts/build_kg_direct.py \
    --base-url "$API_URL" \
    --tenant-id "$TENANT" \
    --entity-key "$ENTITY_KEY" \
    --entity-name "$ENTITY_NAME" \
    --doc-id "$DOC_ID" \
    --validate false
fi

echo "▶ Subgraph preview (depth=2)…"
python scripts/show_subgraph.py \
  --base-url "$API_URL" \
  --tenant-id "$TENANT" \
  --entity-key "$ENTITY_KEY" \
  --depth 2

# Optional: write an interactive HTML of the full tenant graph
if [ -f scripts/plot_full_graph.py ]; then
  echo "▶ Writing interactive HTML graph (graph_all.html)…"
  python scripts/plot_full_graph.py \
    --tenant-id "$TENANT" \
    --limit-nodes 50000 \
    --limit-edges 100000 \
    --outfile graph_all.html || echo "  Skipping HTML graph if pyvis fails."
  echo "✔ Open: $PROJECT/graph_all.html"
fi

echo "✅ KG launch completed."
