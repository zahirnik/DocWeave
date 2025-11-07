# apps/api/routes/assistant.py
from __future__ import annotations
from fastapi import APIRouter, Response

router = APIRouter()

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>RAG + KG Assistant</title>
<style>
  :root {
    --bg:#0f172a; --panel:#111827; --muted:#9ca3af; --text:#e5e7eb; --accent:#22d3ee;
    --ok:#10b981; --warn:#f59e0b; --err:#ef4444; --card:#0b1020; --border:#1f2937;
  }
  html,body{height:100%;margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,Helvetica Neue,Arial}
  .wrap{display:grid;grid-template-rows:auto 1fr; height:100%}
  header{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--panel);border-bottom:1px solid var(--border)}
  h1{font-size:16px;margin:0}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  label{font-size:12px;color:var(--muted)}
  select,input[type=text]{background:#0b1020;color:var(--text);border:1px solid var(--border);border-radius:10px;padding:8px 10px}
  input[type=text]{min-width:220px}
  button{border:0;border-radius:10px;padding:10px 14px;font-weight:600;cursor:pointer}
  .primary{background:var(--accent);color:#05202a}
  .secondary{background:#1f2937;color:var(--text);border:1px solid var(--border)}
  main{display:grid;grid-template-columns:2fr 1fr;gap:12px;padding:12px;height:calc(100% - 0px);min-height:0}
  .panel{background:var(--card);border:1px solid var(--border);border-radius:12px;display:flex;flex-direction:column;min-height:0}
  .messages{flex:1;overflow:auto;padding:12px}
  .msg{padding:10px 12px;border-radius:10px;margin:8px 0;line-height:1.4}
  .user{background:#0b2e36}
  .bot{background:#101b33}
  .sources{font-size:12px;color:var(--muted);margin-top:6px}
  .inputbar{display:flex;gap:8px;padding:12px;border-top:1px solid var(--border)}
  textarea{flex:1;resize:vertical;min-height:48px;max-height:160px;background:#0b1020;color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px}
  .kgpanel{padding:12px;overflow:auto}
  .small{font-size:12px;color:var(--muted);white-space:pre-wrap}
  .badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#0b2e36;color:var(--accent);font-size:12px;margin-left:8px}
  .err{color:var(--err);font-size:12px;margin-top:6px;white-space:pre-wrap}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="row"><h1>Assistant <span class="badge">RAG + KG</span></h1></div>
    <div class="row">
      <label for="mode">mode</label>
      <select id="mode">
        <option value="rag">RAG</option>
        <option value="hybrid" selected>Hybrid (KG + RAG)</option>
      </select>
      <label for="tenant">tenant_id</label>
      <input id="tenant" type="text" value="default" />
      <label for="ek">entity_key</label>
      <input id="ek" type="text" value="org:sainsburys ar 2023" />
      <button id="fit" class="secondary">Reset</button>
    </div>
  </header>

  <main>
    <section class="panel">
      <div id="msgs" class="messages"></div>
      <div class="inputbar">
        <textarea id="q" placeholder="Ask a question…"></textarea>
        <button id="ask" class="primary">Ask</button>
      </div>
    </section>

    <aside class="panel">
      <div class="kgpanel">
        <div><strong>KG Context (Hybrid mode)</strong></div>
        <div class="small">We fetch /kg/subgraph for the tenant_id & entity_key and distill a compact text summary to seed the LLM.</div>
        <pre id="kgctx" class="small"></pre>
        <div id="err" class="err"></div>
      </div>
    </aside>
  </main>
</div>

<script>
  const msgs  = document.getElementById('msgs');
  const qEl   = document.getElementById('q');
  const modeEl= document.getElementById('mode');
  const ekEl  = document.getElementById('ek');
  const tenantEl = document.getElementById('tenant');
  const kgBox = document.getElementById('kgctx');
  const errBox= document.getElementById('err');

  function addMsg(text, who){
    const div = document.createElement('div');
    div.className = 'msg ' + (who==='user'?'user':'bot');
    div.innerText = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }
  function addSources(items){
    if(!items || !items.length) return;
    const div = document.createElement('div');
    div.className = 'sources';
    div.innerText = "Sources: " + items.map((s,i) => `[${i+1}] ${s}`).join("   ");
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  // --- helpers for robust parsing ---
  function nodeDisplay(n){
    const props = n.props || n.properties || {};
    const name  = n.name || props.title || props.name || props.text || n.label || n.type || n.id || n.key || "";
    const s = String(name || "").trim();
    if (!s) return "(node)";
    return s.length > 120 ? s.slice(0,117) + "..." : s;
  }
  function getNodeKey(n){
    return n.key ?? n.id ?? (n.props && (n.props.key || n.props.id)) ?? null;
  }

  // resolve endpoint (edge endpoint can be id/key/index/dict)
  function resolveEndpoint(ep, idmap, idxmap){
    if (ep === null || ep === undefined) return "?";
    // numeric index
    if (typeof ep === "number") return idxmap[ep] ?? `node[${ep}]`;
    // string id/key or numeric-string index
    if (typeof ep === "string"){
      if (idmap[ep]) return idmap[ep];
      const n = Number(ep);
      if (!Number.isNaN(n)) return idxmap[n] ?? `node[${ep}]`;
      return ep;
    }
    // dict-like
    if (typeof ep === "object"){
      const k = ep.key ?? ep.id ?? ep.u ?? ep.v ?? null;
      if (k && idmap[k]) return idmap[k];
      if (typeof k === "number") return idxmap[k] ?? `node[${k}]`;
      // last resort: try to build a display from object
      return nodeDisplay(ep);
    }
    return "?";
  }

  // Tolerant KG fetch: requires tenant_id and either entity_key or key
  async function fetchKGContext(entityKey){
    errBox.textContent = '';
    kgBox.textContent  = '';
    const tenant = (tenantEl?.value || 'default').trim() || 'default';
    try{
      let url = `/kg/subgraph?tenant_id=${encodeURIComponent(tenant)}&entity_key=${encodeURIComponent(entityKey)}`;
      let res = await fetch(url);
      let body = await res.text();

      // fallback to ?key=
      if(!res.ok){
        const url2 = `/kg/subgraph?tenant_id=${encodeURIComponent(tenant)}&key=${encodeURIComponent(entityKey)}`;
        res  = await fetch(url2);
        body = await res.text();
        if(!res.ok) throw new Error(`${res.status} ${res.statusText}: ${body}`);
      }

      const data = JSON.parse(body);

      // normalize
      const nodes = (data.nodes || data.data?.nodes || []);
      const edges = (data.edges || data.data?.edges || []);

      // Build both id->name and index->name maps
      const idmap = {};
      const idxmap = {};
      nodes.forEach((n, i) => {
        const name = nodeDisplay(n);
        const key  = getNodeKey(n);
        if (key) idmap[String(key)] = name;
        const id   = n.id ?? null;
        if (id) idmap[String(id)] = name;
        idxmap[i] = name;
      });

      // Format triples robustly
      const lines = [];
      for (const e of edges){
        const rel = (e.label || e.type || e.relation || e.rel || "related_to");
        const u = e.source ?? e.src ?? e.from ?? e.u ?? e.u_idx ?? e.source_id ?? null;
        const v = e.target ?? e.dst ?? e.to   ?? e.v ?? e.v_idx ?? e.target_id ?? null;
        const s = resolveEndpoint(u, idmap, idxmap);
        const t = resolveEndpoint(v, idmap, idxmap);
        lines.push(`${s} --${rel}--> ${t}`);
      }

      const header = `TENANT: ${tenant}\\nENTITY_KEY: ${entityKey}\\nNODES: ${nodes.length}  EDGES: ${edges.length}`;
      const bodyText = lines.slice(0, 200).join("\\n");
      const ctx = header + "\\nTRIPLES:\\n" + bodyText;
      kgBox.textContent = ctx;
      return ctx;
    }catch(err){
      errBox.textContent = String(err);
      return "";
    }
  }

  // Send multiple accepted keys to /ui/ask (avoids 422 schema mismatches)
  async function ask(){
    const question = qEl.value.trim();
    if(!question) return;
    addMsg(question,'user');
    qEl.value = "";

    let prompt = question;
    if(modeEl.value === 'hybrid'){
      const kg = await fetchKGContext(ekEl.value.trim());
      if(kg){
        prompt = [
          "You are a careful assistant. First, consider the KG triples, then consult documents via retrieval.",
          "If you cite sources, use the document citations that the backend returns.",
          "KG_CONTEXT:\\n" + kg,
          "QUESTION: " + question
        ].join("\\n\\n");
      }
    }

    const payload = { q: prompt, question: prompt, query: prompt };
    const res = await fetch("/ui/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });

    if(!res.ok){
      let errTxt = "";
      try { errTxt = await res.text(); } catch {}
      addMsg(`(error ${res.status}) ${errTxt.slice(0,300)}`, 'bot');
      return;
    }

    const data = await res.json();
    const answer = data.answer || data.text || JSON.stringify(data);
    addMsg(answer, 'bot');

    const sources = [];
    const items = data.sources || data.citations || data.docs || [];
    for(const it of items){
      const name = it.source || it.file || it.title || it.id || "doc";
      const chunk = (it.chunk !== undefined) ? `chunk ${it.chunk}` : (it.chunk_id ? it.chunk_id : "chunk ?");
      const dist  = (it.dist !== undefined) ? `dist ${Number(it.dist).toFixed(3)}` : undefined;
      sources.push([name, chunk, dist].filter(Boolean).join(" · "));
    }
    addSources(sources);
  }

  document.getElementById('ask').addEventListener('click', ask);
  document.getElementById('fit').addEventListener('click', () => {
    msgs.innerHTML = ""; kgBox.textContent = ""; errBox.textContent = "";
  });
  qEl.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); ask(); }});
</script>
</body>
</html>
"""

@router.get("/ui/assistant", summary="Chat UI that supports RAG or Hybrid (KG + RAG)")
async def ui_assistant() -> Response:
    return Response(content=HTML, media_type="text/html")
