#!/usr/bin/env python3
# scripts/kg_launch.py
from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional

# -------------------------
# Small helpers
# -------------------------
def env_get(k: str, default: Optional[str] = None) -> str:
    v = os.getenv(k, default if default is not None else "")
    return "" if v is None else v

def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def tail(file: Path, n: int = 200) -> str:
    if not file.exists():
        return "(no log file)"
    try:
        lines = file.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(tail error: {exc})"

# -------------------------
# Uvicorn lifecycle
# -------------------------
def kill_existing_uvicorn(pid_file: Path) -> None:
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            pid_file.unlink()
        except Exception:
            pass

def start_uvicorn(app: str, host: str, port: int, log_file: Path, pid_file: Path) -> None:
    kill_existing_uvicorn(pid_file)
    import subprocess
    cmd = [sys.executable, "-m", "uvicorn", app, "--host", host, "--port", str(port)]
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    pid_file.write_text(str(proc.pid))

def wait_health(base_url: str, tries: int = 120, delay: float = 1.0) -> bool:
    import requests
    for _ in range(tries):
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.ok and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

# -------------------------
# Document parsing & chunking
# -------------------------
def read_any(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            txt = "\n".join(pg.get_text("text") for pg in doc)
            doc.close()
            return txt
        except Exception:
            from pypdf import PdfReader
            rd = PdfReader(str(path))
            return "\n".join((pg.extract_text() or "") for pg in rd.pages)
    elif ext == ".docx":
        import docx
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, size: int = 1200, overlap: int = 100) -> List[str]:
    text = text.replace("\x00", "")
    out: List[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        i += max(1, size - overlap)
    return [t for t in out if t.strip()]

# -------------------------
# KG build via API (/kg/build)
# -------------------------
def build_kg_from_folder(
    base_url: str,
    folder: Path,
    tenant_id: str,
    entity_ns: str = "org",
    max_chars: int = 1200,
    overlap: int = 100,
    exts: tuple = (".pdf", ".txt", ".docx"),
) -> None:
    import requests
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)
    print(f"[KG] Scanning {folder} → {len(files)} file(s)")
    for fp in files:
        try:
            print(f"[KG] → {fp.name}")
            raw = read_any(fp)
            parts = chunk_text(raw, max_chars, overlap)
            payload = {
                "tenant_id": tenant_id,
                "entity_name": fp.stem,
                "entity_namespace": entity_ns,
                "doc_id": hashlib.md5(str(fp).encode()).hexdigest(),
                "chunks": [
                    {"text": parts[i], "page": None, "chunk_id": f"{fp.stem}:{i}"}
                    for i in range(len(parts))
                ],
                "metric_aliases": None,
                "validate": True,
            }
            r = requests.post(f"{base_url}/kg/build", json=payload, timeout=180)
            r.raise_for_status()
            print(f"[KG]   OK: {r.json()}")
        except Exception as exc:
            print(f"[KG] ! {fp.name}: {exc}")

# -------------------------
# (Optional) ngrok tunnel
# -------------------------
def start_ngrok(port: int) -> Optional[str]:
    try:
        from pyngrok import ngrok
    except Exception:
        print("[ngrok] pyngrok not installed. Install with: pip install pyngrok")
        return None
    tunnel = ngrok.connect(port, "http")
    return str(tunnel.public_url)

# -------------------------
# Backend validation helpers
# -------------------------
def validate_backend(backend: str) -> None:
    backend = backend.lower().strip()
    if backend == "memory":
        os.environ["KG_BACKEND"] = "memory"
        return

    if backend == "postgres":
        dsn = os.getenv("POSTGRES_DSN")
        if not dsn:
            die("KG_BACKEND=postgres but POSTGRES_DSN is not set.")
        os.environ["KG_BACKEND"] = "postgres"
        return

    if backend == "neo4j":
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO3J_USER") or os.getenv("NEO4J_USER")  # tolerate common typo
        pwd = os.getenv("NEO4J_PASSWORD")
        if not (uri and user and pwd):
            die("KG_BACKEND=neo4j requires NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.")
        os.environ["KG_BACKEND"] = "neo4j"
        return

    die(f"Unknown backend: {backend}. Use memory|postgres|neo4j")

# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Launch API + (optional) UI and build the KG with memory/postgres/neo4j backends. Optional ngrok tunnel."
    )
    ap.add_argument("--backend", default="memory", choices=["memory", "postgres", "neo4j"], help="KG backend")
    ap.add_argument("--build-kg", action="store_true", help="Build the KG from --folder by calling /kg/build")
    ap.add_argument("--folder", default="data/mydoc", help="Folder with PDFs/TXTs/DOCX")
    ap.add_argument("--tenant-id", default="default")
    ap.add_argument("--entity-namespace", default="org")
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=100)

    ap.add_argument("--ui", action="store_true", help="Keep API running and serve /ui")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ngrok", action="store_true", help="Expose via ngrok; prints public URL")

    ap.add_argument("--set", action="append", help='Set extra envs: KEY=VALUE (repeatable)')
    args = ap.parse_args()

    # Apply extra envs
    if args.set:
        for kv in args.set:
            if "=" in kv:
                k, v = kv.split("=", 1)
                os.environ[k.strip()] = v.strip()

    # Validate backend envs
    validate_backend(args.backend)

    # UI expects these when answering questions; safe defaults:
    os.environ.setdefault("EMBEDDINGS_PROVIDER", "openai")
    os.environ.setdefault("EMBEDDINGS_MODEL", "text-embedding-3-small")
    os.environ.setdefault("CHAT_MODEL", env_get("LLM_MODEL", "gpt-4o-mini"))

    repo_root = Path(".").resolve()
    data_dir = (repo_root / args.folder).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Start API
    app_path = "apps.api.main:app"
    base_url = f"http://{args.host}:{args.port}"
    uv_log = Path("uvicorn.log")
    uv_pid = Path("uvicorn.pid")

    print(f"Backend: {args.backend}  |  Folder: {data_dir}")
    print(f"Starting API on {args.host}:{args.port} …")
    start_uvicorn(app_path, args.host, args.port, uv_log, uv_pid)
    if not wait_health(base_url, tries=120, delay=1.0):
        print("Health check failed after 120 tries.")
        print("\n── uvicorn.log (last 200) ──")
        print(tail(uv_log, 200))
        sys.exit(1)
    print("API: OK")

    # Optional KG build
    if args.build_kg:
        print("\n== Build KG ==")
        build_kg_from_folder(
            base_url=base_url,
            folder=data_dir,
            tenant_id=args.tenant_id,
            entity_ns=args.entity_namespace,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )

    # Optional ngrok
    public_url = None
    if args.ngrok:
        print("\n== ngrok ==")
        public_url = start_ngrok(args.port)
        if public_url:
            print("Public URL:", public_url)
        else:
            print("ngrok not started (pyngrok missing).")

    # Optional UI
    if args.ui:
        print("\n== UI ==")
        print(f"Local UI: {base_url}/ui")
        if public_url:
            print(f"Public UI: {public_url}/ui")

        # Colab hint
        if "COLAB_GPU" in os.environ or os.environ.get("GCE_METADATA_HOST"):
            print(textwrap.dedent("""
                In Colab, run in a Python cell:
                    from google.colab import output
                    url = output.eval_js("google.colab.kernel.proxyPort(8000)")
                    print(url + "/ui")
                    output.eval_js(f"window.open('{url}/ui','_blank')")
            """).strip())

        print("\nPress Ctrl+C to stop. Tail logs with:")
        print("  !tail -n 200 /content/drive/MyDrive/AAA_Rag/uvicorn.log")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
