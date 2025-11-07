#!/usr/bin/env python3
"""
Run a minimal, reliable demo using the project's own packages.

What it does:
  1) Uses packages.retriever.embeddings (local by default) to embed docs from ./data.
  2) Saves a tiny NPZ index (.demo/index_demo.npz) and queries it with cosine similarity.
  3) Boots the FastAPI app, seeds vector + BM25 stores for BOTH the plain and tenant-mapped
     collection names, waits for readiness, mints a JWT, and calls /search with proper headers.

Environment switches (all optional):
  EMBEDDINGS_PROVIDER=local|openai|bedrock     (default: local)
  EMBEDDINGS_MODEL=BAAI/bge-small-en-v1.5     (default: BAAI/bge-small-en-v1.5)
  OPENAI_API_KEY=...                          (if provider=openai)
  VECTOR_STORE=chroma|simple|pgvector|qdrant  (API search uses 'chroma' by default here)
  CHROMA_DIR=.chroma
  WHOOSH_DIR=.whoosh
  SEARCH_ALPHA=0.6                            (0.0=BM25 only, 1.0=vector only)
  API_REQ_TIMEOUT=45

Usage:
  python run_model_example.py
  python run_model_example.py --no-api         # skip uvicorn part
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import threading
import pathlib
import subprocess
import importlib
import json
from typing import List, Tuple
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DEMO_DIR = REPO_ROOT / ".demo"
DEMO_DIR.mkdir(exist_ok=True)
INDEX_PATH = DEMO_DIR / "index_demo.npz"

# Ensure our repo is importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------- small utilities -------------------------------------
def _pip_install(*pkgs: str) -> None:
    """Install packages via pip (quietly), safe for Colab."""
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    cmd.extend(pkgs)
    subprocess.check_call(cmd)

def _ensure_module(mod_name: str, pip_name: str | None = None) -> None:
    """Ensure a module can be imported; if not, pip install."""
    try:
        importlib.import_module(mod_name)
    except Exception:
        _pip_install(pip_name or mod_name)
        importlib.import_module(mod_name)


# ---- Embeddings via project package -------------------------------------------------
def _ensure_local_embeddings_available():
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        try:
            print("Installing sentence-transformers (for local embeddings)...")
            _pip_install("sentence-transformers>=3.0.0")
        except Exception as e:
            raise SystemExit(
                "Failed to import or install sentence-transformers. "
                "Install it manually:\n  pip install 'sentence-transformers>=3.0.0'\n"
                f"Details: {e}"
            )

def _make_embedder():
    from packages.retriever.embeddings import Embeddings, EmbeddingConfig  # from your project
    provider = os.getenv("EMBEDDINGS_PROVIDER", "local").strip().lower()
    model = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5")

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("EMBEDDINGS_PROVIDER=openai but OPENAI_API_KEY is missing. Falling back to local.")
        provider = "local"

    if provider == "local":
        _ensure_local_embeddings_available()

    cfg = EmbeddingConfig(provider=provider, model=model, normalize=True)
    print(f"[Embeddings] provider={cfg.provider} model={cfg.model} normalize={cfg.normalize}")
    return Embeddings(cfg)

# ---- Simple loaders (text + pdf) ---------------------------------------------------
def _read_text(p: pathlib.Path, limit_chars=6000) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:limit_chars]
    except Exception:
        return ""

def _read_pdf(p: pathlib.Path, max_pages=10, limit_chars=6000) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return ""
    try:
        text = []
        r = PdfReader(str(p))
        for pg in r.pages[:max_pages]:
            try:
                text.append(pg.extract_text() or "")
            except Exception:
                continue
        return "\n".join(text)[:limit_chars]
    except Exception:
        return ""

def _gather_docs() -> list[dict]:
    docs = []
    if DATA_DIR.is_dir():
        candidates: List[pathlib.Path] = []
        for ext in ("*.txt", "*.md", "*.pdf"):
            candidates.extend(DATA_DIR.rglob(ext))
        for p in sorted(candidates)[:50]:  # small cap for demo
            if p.suffix.lower() == ".pdf":
                txt = _read_pdf(p)
            else:
                txt = _read_text(p)
            if txt.strip():
                docs.append({"id": str(p.relative_to(REPO_ROOT)), "text": txt})

    if not docs:
        # Fallback docs include ACME 38.2% to guarantee a match in the smoke test
        docs = [
            {"id": "samples/acme.txt",  "text": "ACME PLC reported a 38.2% gross margin in FY2024 with strong cost control."},
            {"id": "samples/beta.txt",  "text": "Beta Corp revenue in Q3 2024 was $1.26 billion, up 14.5% year over year."},
            {"id": "samples/gamma.txt", "text": "Gamma Ltd major expenses were cost of sales, SG&A, and R&D in 2024."},
        ]
        print("ℹ️ No usable files under ./data; using 3 tiny sample docs.")
    return docs

# ---- Build + save tiny index -------------------------------------------------------
def build_index() -> Tuple[np.ndarray, List[str], List[str]]:
    print("\n== Step 1: Build tiny local index ==")
    embedder = _make_embedder()
    docs = _gather_docs()
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    vecs = np.array(embedder.embed_documents(texts), dtype="float32")
    # Unit-norm (safe even if normalize=True already did it)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    np.savez_compressed(INDEX_PATH, embeddings=vecs, ids=np.array(ids, dtype=object), texts=np.array(texts, dtype=object))
    print(f"✅ Index saved -> {INDEX_PATH} | docs={len(docs)} dim={vecs.shape[1]}")
    return vecs, ids, texts

# ---- Query the index ---------------------------------------------------------------
def query_index(question: str):
    print("\n== Step 2: Query index with cosine similarity ==")
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing index at {INDEX_PATH}. Run build_index() first.")

    arr = np.load(INDEX_PATH, allow_pickle=True)
    E = arr["embeddings"].astype("float32")  # [N, D]
    ids = arr["ids"].tolist()
    texts = arr["texts"].tolist()
    _N, D = E.shape

    embedder = _make_embedder()
    q = np.array(embedder.embed_query(question), dtype="float32")
    if q.shape[0] != D:
        raise SystemExit(
            f"Embedding dimension mismatch.\nIndex dim={D}; query dim={q.shape[0]}.\n"
            "Use the SAME provider/model for querying as used to build the index."
        )

    q /= (np.linalg.norm(q) + 1e-12)
    scores = E @ q
    top = int(np.argmax(scores))
    print(f"Q: {question}")
    print(f"Top doc: {ids[top]} | score={float(scores[top]):.3f}")
    preview = texts[top].replace("\n", " ")
    print(f"Text: {preview[:220] + ('…' if len(preview) > 220 else '')}")
    print("✅ Retrieval OK")


# ---------------------- NEW: seed BOTH collection names -----------------------------
def _seed_chroma_and_whoosh(tenant_id: str = "t0", logical_collection: str = "demo") -> None:
    """
    Seed both Chroma and Whoosh for *both* names:
      1) plain: {logical_collection}
      2) tenant-mapped: ten_{tenant_id}_{logical_collection}
    This covers codepaths where vector store uses tenant-mapped names while BM25 may use plain names.
    """
    # Ensure deps are present
    _ensure_module("chromadb", "chromadb")
    _ensure_module("whoosh", "whoosh")
    _ensure_module("sentence_transformers", "sentence-transformers>=3.0.0")

    # Resolve dirs
    chroma_dir = pathlib.Path(os.getenv("CHROMA_DIR", str(REPO_ROOT / ".chroma")))
    whoosh_dir = pathlib.Path(os.getenv("WHOOSH_DIR", str(REPO_ROOT / ".whoosh")))
    chroma_dir.mkdir(parents=True, exist_ok=True)
    whoosh_dir.mkdir(parents=True, exist_ok=True)

    # Names to seed
    names = [
        logical_collection,
        f"ten_{tenant_id}_{logical_collection}",
    ]

    # Prepare docs
    docs = _gather_docs()
    ids = [pathlib.Path(d["id"]).stem for d in docs]
    texts = [d["text"] for d in docs]

    # --- Chroma
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5"))

    for name in names:
        coll = client.get_or_create_collection(name, embedding_function=ef)
        if coll.count() == 0:
            coll.add(ids=ids, documents=texts)
        print(f"📦 Chroma seeded: {name} | count={coll.count()} | dir={chroma_dir}")

    # --- Whoosh (BM25)
    from whoosh import index
    from whoosh.fields import Schema, TEXT, ID
    from whoosh.analysis import StemmingAnalyzer

    schema = Schema(id=ID(stored=True, unique=True), content=TEXT(analyzer=StemmingAnalyzer()))
    for name in names:
        target = whoosh_dir / name
        target.mkdir(parents=True, exist_ok=True)
        ix = index.create_in(str(target), schema) if not index.exists_in(str(target)) else index.open_dir(str(target))
        with ix.writer(limitmb=256, procs=1, multisegment=True) as w:
            for i, t in zip(ids, texts):
                w.update_document(id=i, content=t)
        print(f"📦 Whoosh seeded: {target} | docs={len(ids)}")


# ---- Optional: start API & smoke test ---------------------------------------------
def _free_port(start=8010) -> int:
    import socket
    p = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1

def start_api_and_test():
    print("\n== Step 3 (optional): Boot FastAPI and smoke-test ==")

    # Hard overrides so config can't re-enable unwanted modes
    os.environ["JWT_SECRET"] = os.getenv("JWT_SECRET", "dev-secret")
    os.environ["JWT_ALGORITHM"] = os.getenv("JWT_ALGORITHM", "HS256")

    # Retrieval stack defaults
    os.environ["EMBEDDINGS_PROVIDER"] = os.getenv("EMBEDDINGS_PROVIDER", "local")
    os.environ["EMBEDDINGS_MODEL"] = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5")
    os.environ["VECTOR_STORE"] = os.getenv("VECTOR_STORE", "chroma")
    os.environ["CHROMA_DIR"] = os.getenv("CHROMA_DIR", str(REPO_ROOT / ".chroma"))
    os.environ["BM25_PROVIDER"] = os.getenv("BM25_PROVIDER", "whoosh")
    os.environ["WHOOSH_DIR"] = os.getenv("WHOOSH_DIR", str(REPO_ROOT / ".whoosh"))
    os.environ["SEARCH_ALPHA"] = os.getenv("SEARCH_ALPHA", "0.6")  # hybrid by default

    # Ensure critical deps present before seeding/starting server
    try:
        _ensure_module("uvicorn", "uvicorn[standard]")
        _ensure_module("requests", "requests")
        _ensure_module("chromadb", "chromadb")
        _ensure_module("whoosh", "whoosh")
    except Exception as e:
        print("⚠️ Missing deps:", e)
        return

    # Seed both store names (plain + tenant-mapped)
    tenant_id = "t0"
    logical_collection = "demo"
    _seed_chroma_and_whoosh(tenant_id=tenant_id, logical_collection=logical_collection)

    # Import API after env is set + stores are seeded
    try:
        from apps.api.main import app  # from your project
    except Exception as e:
        print(f"⚠️ Could not import FastAPI app: {e}")
        print("Skipping API smoke test.")
        return

    import uvicorn, requests  # type: ignore

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="info"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait for /health (poll up to 45s)
    base = f"http://127.0.0.1:{port}"
    ok = False
    for _ in range(45):
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.status_code == 200:
                print("Health:", r.status_code, r.text[:200])
                ok = True
                break
        except Exception:
            pass
        time.sleep(1.0)
    if not ok:
        print("⚠️ /health did not become ready in time.")
        return

    # Mint a token for tenant t0
    headers = {"x-tenant-id": tenant_id}
    try:
        from packages.core.auth import issue_access_token  # from your project
        token, _ttl = issue_access_token(sub="dev", tenant_id=tenant_id, roles=["user"], scopes=["rag:query"], ttl_s=3600)
        headers["Authorization"] = f"Bearer {token}"
    except Exception:
        # Fallback to PyJWT if helper unavailable
        try:
            _ensure_module("jwt", "pyjwt")
            import jwt, time as _time
            payload = {
                "sub": "dev",
                "tenant_id": tenant_id,
                "roles": ["user"],
                "scopes": ["rag:query"],
                "iat": int(_time.time()),
                "exp": int(_time.time()) + 3600,
                "iss": "convai",
                "aud": "convai.clients",
                "jti": "demo-runner",
            }
            token = jwt.encode(payload, os.environ["JWT_SECRET"], algorithm=os.environ["JWT_ALGORITHM"])
            headers["Authorization"] = f"Bearer {token}"
        except Exception as e:
            print("⚠️ Could not mint JWT:", e)
            print("Will try /search without auth (may 401).")

    # Warm-up call (BM25 only) to ensure whoosh/index open + fast second call
    try:
        params = {"q": "warmup", "collection": logical_collection, "top_k": 1, "size": 1, "use_bm25": True}
        requests.get(f"{base}/search", params=params, headers=headers, timeout=30)
    except Exception:
        pass

    # Final search that should hit ACME (hybrid)
    try:
        params = {
            "q": "ACME gross margin 38.2%",
            "collection": logical_collection,  # API may map to ten_{tenant}_{collection}
            "top_k": 3,
            "size": 5,
            "use_bm25": True
        }
        TIMEOUT = float(os.getenv("API_REQ_TIMEOUT", "45"))  # longer timeout for cold starts
        r = requests.get(f"{base}/search", params=params, headers=headers, timeout=TIMEOUT)
        print("Search:", r.status_code, r.headers.get("content-type"))
        try:
            js = r.json()
            print(json.dumps(js, indent=2)[:1200])
        except Exception:
            print((r.text or "")[:800])
    except Exception as e:
        print("⚠️ /search request failed:", e)


# ---- CLI --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run the RAG model example reliably.")
    ap.add_argument("--no-api", action="store_true", help="Skip the FastAPI/uvicorn smoke test.")
    ap.add_argument("--question", default="What was ACME's gross margin in 2024?", help="Query to test retrieval.")
    args = ap.parse_args()

    build_index()
    query_index(args.question)
    if not args.no_api:
        start_api_and_test()

if __name__ == "__main__":
    main()
