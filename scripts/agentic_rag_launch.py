#!/usr/bin/env python3
# scripts/agentic_rag_launch.py
from __future__ import annotations
import os, sys, time, json, signal, hashlib, argparse, textwrap
from pathlib import Path
from typing import Optional, List, Tuple

# -------------------------
# Small helpers
# -------------------------
def e(k: str, d: Optional[str] = None) -> str:
    v = os.getenv(k, d if d is not None else "")
    return "" if v is None else v

def tail(file: Path, n: int = 200) -> str:
    if not file.exists():
        return "(no log file)"
    try:
        lines = file.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(tail error: {exc})"

def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

# -------------------------
# UVicorn lifecycle
# -------------------------
def kill_existing_uvicorn(pid_file: Path) -> None:
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        try: pid_file.unlink()
        except Exception: pass

def start_uvicorn_inline(app: str, host: str, port: int, log_file: Path, pid_file: Path) -> None:
    kill_existing_uvicorn(pid_file)
    # launch uvicorn as a child process in the same interpreter to keep things simple
    cmd = [sys.executable, "-m", "uvicorn", app, "--host", host, "--port", str(port)]
    import subprocess
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    pid_file.write_text(str(proc.pid))

def wait_health(base_url: str, tries: int = 120, delay: float = 1.0) -> bool:
    import requests
    ok = False
    for _ in range(tries):
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.ok and r.json().get("status") == "ok":
                ok = True
                break
        except Exception:
            pass
        time.sleep(delay)
    return ok

# -------------------------
# PDF parsing & chunking
# -------------------------
def read_pdf_to_text(path: Path) -> str:
    try:
        import fitz  # PyMuPDF (fast)
        doc = fitz.open(str(path))
        texts = [pg.get_text("text") for pg in doc]
        doc.close()
        return "\n".join(texts)
    except Exception:
        from pypdf import PdfReader  # fallback (slower)
        rd = PdfReader(str(path))
        parts: List[str] = []
        for pg in rd.pages:
            try:
                parts.append(pg.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)

def chunk_text(text: str, size: int = 1200, overlap: int = 100) -> List[str]:
    text = text.replace("\x00", "")
    out: List[str] = []
    i = 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size - overlap)
    return [t for t in out if t.strip()]

# -------------------------
# KG build via API
# -------------------------
def build_kg_from_folder(base_url: str, folder: Path, tenant_id: str, entity_ns: str = "org") -> None:
    import requests
    pdfs = sorted(folder.rglob("*.pdf"))
    print(f"[KG] Found {len(pdfs)} PDFs in {folder}")
    for pdf in pdfs:
        print(f"[KG] → {pdf.name}")
        raw = read_pdf_to_text(pdf)
        parts = chunk_text(raw, 1200, 100)
        payload = {
            "tenant_id": tenant_id,
            "entity_name": pdf.stem,
            "entity_namespace": entity_ns,
            "doc_id": hashlib.md5(str(pdf).encode()).hexdigest(),
            "chunks": [{"text": parts[i], "page": None, "chunk_id": f"{pdf.stem}:{i}"} for i in range(len(parts))],
            "metric_aliases": None,
            "validate": True
        }
        r = requests.post(f"{base_url}/kg/build", json=payload, timeout=180)
        r.raise_for_status()
        print(f"[KG]   OK: {r.json()}")

# -------------------------
# Chroma ingestion
# -------------------------
def ensure_chroma_index(persist_dir: Path, collection_name: str, embed_model: str):
    import chromadb
    model_dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
    dim = model_dims.get(embed_model, 1536)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        col = client.get_collection(collection_name, embedding_function=None)
    except Exception:
        col = client.create_collection(collection_name, metadata={"dim": dim})
    return client, col, dim

def openai_embed_fn(model: str):
    from openai import OpenAI
    client = OpenAI()
    def _emb(batch: List[str]) -> List[List[float]]:
        resp = client.embeddings.create(model=model, input=batch)
        return [d.embedding for d in resp.data]
    return _emb

def ingest_folder_to_chroma(folder: Path, collection, embed_batch, embed_dim: int) -> int:
    total = 0
    files = [p for p in folder.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".docx"}]
    print(f"[INGEST] Scanning {folder} -> {len(files)} file(s)")
    for fp in files:
        try:
            if fp.suffix.lower() == ".pdf":
                text = read_pdf_to_text(fp)
            elif fp.suffix.lower() == ".docx":
                import docx
                doc = docx.Document(str(fp))
                text = "\n".join(p.text for p in doc.paragraphs)
            else:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(text, 1200, 100)
            if not chunks:
                print(f"[INGEST] skip empty: {fp.name}")
                continue
            ids = [hashlib.md5(f"{fp}:{i}".encode()).hexdigest() for i in range(len(chunks))]
            # [CHANGE] include per-chunk metadata so UI can show "chunk N" instead of "?"
            metas = [
                {
                    "source": fp.name,
                    "path": str(fp),
                    "chunk": i,                            # numeric index for UI
                    "chunk_id": f"{fp.name}:{i}",          # stable id if UI prefers this
                }
                for i in range(len(chunks))
            ]
            embeddings: List[List[float]] = []
            B = 64
            for i in range(0, len(chunks), B):
                embeddings.extend(embed_batch(chunks[i:i+B]))
            collection.upsert(documents=chunks, metadatas=metas, ids=ids, embeddings=embeddings)
            print(f"[INGEST] ✔ {fp.name} (chunks={len(chunks)})")
            total += len(chunks)
        except Exception as exc:
            print(f"[INGEST] ! {fp.name}: {exc}")
    return total

# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Launch API + web UI (/ui), optionally build KG and ingest Chroma."
    )
    ap.add_argument("--ui", action="store_true", help="Start FastAPI (serves /ui).")
    ap.add_argument("--build-kg", action="store_true", help="Build KG from all PDFs under data/mydoc/.")
    ap.add_argument("--ingest", action="store_true", help="Ingest all .pdf/.txt/.docx under data/mydoc/ into Chroma.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    args = ap.parse_args()

    # Required env
    if not os.getenv("OPENAI_API_KEY"):
        die("OPENAI_API_KEY is not set.")

    # Defaults (safe for Colab)
    repo_root   = Path(".").resolve()
    data_dir    = repo_root / "data" / "mydoc"
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("KG_BACKEND", "memory")  # avoid DBs in Colab
    os.environ.setdefault("EMBEDDINGS_PROVIDER", "openai")
    os.environ.setdefault("EMBEDDINGS_MODEL", "text-embedding-3-small")
    os.environ.setdefault("CHAT_MODEL", e("LLM_MODEL", "gpt-4o-mini"))

    chroma_dir  = Path(e("CHROMA_PATH", str(repo_root / ".chroma_oai1536")))
    collection  = f"{e('COLLECTION_PREFIX','rag')}_{e('COLLECTION','mydoc_oai1536')}"
    embed_model = e("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
    base_url    = f"http://{args.host}:{args.port}"

    print(f"ENV: KG_BACKEND={os.environ.get('KG_BACKEND')}  EMBEDDINGS_MODEL={os.environ.get('EMBEDDINGS_MODEL')}  CHAT_MODEL={os.environ.get('CHAT_MODEL')}")
    print(f"CHROMA: {chroma_dir}  collection={collection}  ef=openai:{embed_model}")

    # 1) Start API + health
    uv_pid = Path("uvicorn.pid")
    uv_log = Path("uvicorn.log")
    if args.ui or args.build_kg or args.ingest:
        print("== Step 1/3: Start API & health-check ==")
        start_uvicorn_inline("apps.api.main:app", args.host, args.port, uv_log, uv_pid)
        if not wait_health(base_url, tries=120, delay=1.0):
            print("Health check failed after 120 tries")
            print("\n─── Last 200 lines of uvicorn.log ───")
            print(tail(uv_log, 200))
            sys.exit(1)
        print("Health: OK")

    # 2) Optional KG build
    if args.build_kg:
        print("\n== Step 2/3: Build KG from PDFs ==")
        try:
            build_kg_from_folder(base_url, data_dir, tenant_id=e("TENANT_ID", "default"), entity_ns="org")
        except Exception as exc:
            print(f"[KG] Build error: {exc}")
            print(tail(uv_log, 200))
            sys.exit(2)

    # 3) Optional Chroma ingestion
    if args.ingest:
        print("\n== Step 3/3: Ensure Chroma index & ingest ==")
        try:
            client, col, dim = ensure_chroma_index(chroma_dir, collection, embed_model)
            emb = openai_embed_fn(embed_model)
            added = ingest_folder_to_chroma(data_dir, col, emb, dim)
            print(f"[INGEST] Done. Total chunks added: {added}")
        except Exception as exc:
            print(f"[INGEST] Error: {exc}")
            sys.exit(3)

    # 4) If UI requested, print how to open it
    if args.ui:
        print("\n== UI ==")
        print(f"Open http://{args.host}:{args.port}/ui")
        # Colab hint
        if "COLAB_GPU" in os.environ or os.environ.get("GCE_METADATA_HOST"):
            print(textwrap.dedent("""
                In Colab, run in a Python cell:
                    from google.colab import output
                    url = output.eval_js("google.colab.kernel.proxyPort(8000)")
                    print(url + "/ui")
                    output.eval_js(f"window.open('{url}/ui','_blank')")
            """).strip())

    # Keep parent alive if only UI is requested (so server keeps running)
    if args.ui and not (args.build_kg or args.ingest):
        print("\nPress Ctrl+C to stop. Tail logs with:")
        print("  !tail -n 200 /content/drive/MyDrive/AAA_Rag/uvicorn.log  # in Colab")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
