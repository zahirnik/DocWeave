#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/bootstrap_env.sh [REPO_DIR]
# Default repo directory is the common Colab path; override by passing a different path.
REPO_DIR="${1:-/content/drive/MyDrive/AAA_Rag}"
ENV_FILE="$REPO_DIR/.env"
CHROMA_DIR="$REPO_DIR/.chroma"
WHOOSH_DIR="$REPO_DIR/.whoosh"

mkdir -p "$REPO_DIR" "$CHROMA_DIR" "$WHOOSH_DIR"

upsert_env () {
  local k="$1"; local v="$2"
  if [[ -f "$ENV_FILE" ]] && grep -qE "^${k}=" "$ENV_FILE"; then
    sed -i.bak "s|^${k}=.*|${k}=${v}|g" "$ENV_FILE"
  else
    echo "${k}=${v}" >> "$ENV_FILE"
  fi
}

# -------- defaults: single source of truth via ENV --------
upsert_env VECTOR_STORE        "chroma"
upsert_env CHROMA_DIR          "$CHROMA_DIR"

upsert_env BM25_PROVIDER       "whoosh"
upsert_env WHOOSH_DIR          "$WHOOSH_DIR"

upsert_env EMBEDDINGS_PROVIDER "local"
upsert_env EMBEDDINGS_MODEL    "BAAI/bge-small-en-v1.5"

# hybrid weighting (0.0 = BM25-only, 1.0 = vector-only)
upsert_env FUSION_ALPHA        "0.6"

# auth / tenancy
upsert_env JWT_SECRET          "dev-secret"   # 🔒 change in production
upsert_env JWT_ALGORITHM       "HS256"
upsert_env TENANT_DEFAULT      "t0"
upsert_env COLLECTION_DEFAULT  "filings"

# -------- minimal Python deps for retrieval/API demos -------
python3 -m pip -q install --upgrade pip
python3 -m pip -q install "uvicorn[standard]" fastapi chromadb whoosh "sentence-transformers>=3.0.0" PyPDF2 pdfminer.six

echo "✅ .env written at: $ENV_FILE"
echo "   VECTOR_STORE=$(grep -E '^VECTOR_STORE=' \"$ENV_FILE\" | cut -d= -f2-)"
echo "   BM25_PROVIDER=$(grep -E '^BM25_PROVIDER=' \"$ENV_FILE\" | cut -d= -f2-)"
echo "   EMBEDDINGS_MODEL=$(grep -E '^EMBEDDINGS_MODEL=' \"$ENV_FILE\" | cut -d= -f2-)"
echo "   CHROMA_DIR=$CHROMA_DIR"
echo "   WHOOSH_DIR=$WHOOSH_DIR"
echo "✅ Core packages installed."
echo
echo "Next step (load env into your current shell session):"
echo "  set -a; source \"$ENV_FILE\"; set +a"
