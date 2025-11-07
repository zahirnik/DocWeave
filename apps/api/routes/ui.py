# apps/api/routes/ui.py
"""
Simple UI router for your RAG system.

Exposes:
- GET  /ui           → Minimal HTML chat page
- POST /ui/ask       → RAG answer with citations (Chroma + OpenAI)

Config via environment (set before running uvicorn):
  OPENAI_API_KEY=...
  CHROMADB_TELEMETRY_DISABLED=1
  CHROMA_PATH=/path/to/.chroma_oai1536
  COLLECTION_PREFIX=rag
  COLLECTION=mydoc_oai1536
  EMBEDDINGS_PROVIDER=openai            # or local|bedrock
  EMBEDDINGS_MODEL=text-embedding-3-small
  EMBEDDINGS_BATCH=128
  CHAT_MODEL=gpt-4o-mini
  TOP_K=6
  MAX_CTX_CHARS=12000
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Repo services
from packages.retriever.vectorstores.chroma_store import ChromaStore
from packages.retriever.embeddings import EmbeddingClient
from openai import OpenAI

router = APIRouter(prefix="/ui", tags=["ui"])

# ---------------------------
# Environment & singletons
# ---------------------------

def _env() -> Dict[str, Any]:
    # Silence Chroma telemetry unless explicitly enabled
    os.environ.setdefault("CHROMADB_TELEMETRY_DISABLED", "1")
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "CHROMA_PATH": os.getenv("CHROMA_PATH", "/content/drive/MyDrive/AAA_Rag/.chroma_oai1536"),
        "COLLECTION_PREFIX": os.getenv("COLLECTION_PREFIX", "rag"),
        "COLLECTION": os.getenv("COLLECTION", "mydoc_oai1536"),
        "EMBEDDINGS_PROVIDER": os.getenv("EMBEDDINGS_PROVIDER", "openai"),
        "EMBEDDINGS_MODEL": os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"),
        "EMBEDDINGS_BATCH": int(os.getenv("EMBEDDINGS_BATCH", "128")),
        "CHAT_MODEL": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "TOP_K": int(os.getenv("TOP_K", "6")),
        "MAX_CTX_CHARS": int(os.getenv("MAX_CTX_CHARS", "12000")),
    }

ENV = _env()

_STORE: Optional[ChromaStore] = None
_COLL = None  # underlying chromadb Collection
_EMB: Optional[EmbeddingClient] = None
_OAI: Optional[OpenAI] = None


def _ensure_services() -> None:
    """Lazy-initialize vector store, embeddings, and LLM client."""
    global _STORE, _COLL, _EMB, _OAI
    if _STORE and _COLL and _EMB and _OAI:
        return

    if not ENV["OPENAI_API_KEY"]:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

    # Vector store (Chroma)
    _STORE = ChromaStore(
        persist_dir=ENV["CHROMA_PATH"],
        collection_prefix=ENV["COLLECTION_PREFIX"],
    )
    _STORE.ensure_client()
    _COLL = _STORE._get_or_create_collection(ENV["COLLECTION"])

    # Embeddings (OpenAI/BEDROCK/local determined by env)
    _EMB = EmbeddingClient(
        provider=ENV["EMBEDDINGS_PROVIDER"],
        model_alias=ENV["EMBEDDINGS_MODEL"],
        batch_size=ENV["EMBEDDINGS_BATCH"],
    )

    # Chat LLM
    _OAI = OpenAI(api_key=ENV["OPENAI_API_KEY"])


# ---------------------------
# Schemas
# ---------------------------

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    max_ctx_chars: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


# ---------------------------
# API
# ---------------------------

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    _ensure_services()

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question.")

    k = int(req.top_k or ENV["TOP_K"])
    max_ctx = int(req.max_ctx_chars or ENV["MAX_CTX_CHARS"])

    # 1) Embed query & retrieve
    qvec = _EMB.embed([q])[0]
    res = _COLL.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # 2) Build grounded context + citations
    ctx_parts: List[str] = []
    cites: List[Dict[str, Any]] = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        src = os.path.basename((meta or {}).get("source", "unknown"))
        chunk = (meta or {}).get("chunk", "?")
        tag = f"[{i}] {src} · chunk {chunk}"
        snippet = " ".join((doc or "").split())
        cites.append({
            "tag": tag,
            "source": src,
            "chunk": chunk,
            "distance": float(dist),
        })
        ctx_parts.append(f"{tag}\n{snippet}\n")

    context = ("\n".join(ctx_parts))[:max_ctx]

    # 3) Constrained generation
    system = (
        "You are a precise analyst. Answer ONLY using the provided CONTEXT.\n"
        "If the answer is not in the context, reply exactly: 'Not found in context.'\n"
        "Be concise and cite sources like [1], [2]."
    )
    user = f"QUESTION:\n{q}\n\nCONTEXT:\n{context}\n\nAnswer with citations."

    try:
        resp = _OAI.chat.completions.create(
            model=ENV["CHAT_MODEL"],
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    return AskResponse(answer=answer, sources=cites)


# ---------------------------
# UI (HTML)
# ---------------------------

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    _ensure_services()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>RAG Chat · {ENV['COLLECTION']}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #0b0c10; color: #e6edf3; }}
  .wrap {{ max-width: 900px; margin: 0 auto; padding: 24px; }}
  h1 {{ margin: 0 0 12px; font-size: 22px; }}
  .bar {{ display:flex; gap:8px; margin: 16px 0 8px; }}
  input, button {{
    background:#161b22; color:#e6edf3; border:1px solid #30363d; border-radius:12px; padding:12px 14px; font-size:14px;
  }}
  button {{ cursor:pointer; }}
  button:hover {{ background:#1f2430; }}
  .ans, .src {{ background:#0f141b; border:1px solid #30363d; border-radius:12px; padding:16px; margin-top:12px; white-space:pre-wrap; }}
  .muted {{ color:#9aa6b2; font-size:12px; }}
  .pill {{ display:inline-block; padding:3px 8px; border:1px solid #30363d; border-radius:999px; margin-right:6px; font-size:12px; color:#c9d1d9; }}
  .row {{ margin:6px 0; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>RAG Chat · <span class="pill">{ENV['COLLECTION']}</span><span class="pill">{ENV['CHAT_MODEL']}</span></h1>
  <div class="bar">
    <input id="q" placeholder="Ask a question…" style="flex:1" />
    <button id="askBtn">Ask</button>
  </div>
  <div class="muted">Answers are grounded in your indexed documents and include citations.</div>
  <div id="out"></div>
</div>
<script>
const elQ = document.getElementById('q');
const elBtn = document.getElementById('askBtn');
const elOut = document.getElementById('out');

async function ask() {{
  const question = elQ.value.trim();
  if (!question) return;
  elBtn.disabled = true;
  elOut.innerHTML = '<div class="ans">Thinking…</div>';

  try {{
    const res = await fetch('/ui/ask', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ question }})
    }});
    if (!res.ok) {{
      const t = await res.text();
      throw new Error('HTTP ' + res.status + ' ' + t);
    }}
    const data = await res.json();
    const srcHtml = (data.sources || []).map((s, i) => {{
      return `<div class="row">[${{i+1}}] <b>${{s.source}}</b> · chunk ${{s.chunk}} · dist ${{(s.distance ?? 0).toFixed(3)}}</div>`;
    }}).join('');
    elOut.innerHTML = `
      <div class="ans">${{data.answer}}</div>
      <div class="src"><b>Sources</b><br/>${{srcHtml || '—'}}</div>
    `;
  }} catch (e) {{
    elOut.innerHTML = `<div class="ans">Error: ${{e.message}}</div>`;
  }} finally {{
    elBtn.disabled = false;
  }}
}}

elBtn.onclick = ask;
elQ.addEventListener('keydown', (e) => {{ if (e.key === 'Enter') ask(); }});
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
