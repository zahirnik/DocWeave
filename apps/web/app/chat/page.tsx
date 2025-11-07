
'use client';
/**
 * Barebones chat UI (POST + optional WS stream) with citations panel.
 * See file header in the previous attempt for full comments.
 */
import React from 'react';

type Source = { score: number; metadata: Record<string, any> };
type ChatTurn = { role: 'user' | 'assistant'; text: string; sources?: Source[] };

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';
const WS_BASE  = process.env.NEXT_PUBLIC_WS_BASE  || (typeof window !== 'undefined'
  ? ((location.protocol === 'https:') ? 'wss://' : 'ws://') + location.host
  : '');

export default function ChatPage() {
  const [tenantId, setTenantId] = React.useState('t0');
  const [token, setToken] = React.useState('');
  const [apiKey, setApiKey] = React.useState('');
  const [query, setQuery] = React.useState('Summarize Q4 revenues YoY.');
  const [topK, setTopK] = React.useState(6);
  const [loading, setLoading] = React.useState(false);
  const [streaming, setStreaming] = React.useState(false);
  const [log, setLog] = React.useState<string[]>([]);
  const [turns, setTurns] = React.useState<ChatTurn[]>([]);
  const appendLog = (line: string) => setLog(prev => [...prev.slice(-500), line]);

  const headers: Record<string, string> = { 'Content-Type': 'application/json', 'X-Tenant-ID': tenantId };
  if (token.trim()) headers['Authorization'] = `Bearer ${token.trim()}`;
  if (apiKey.trim()) headers['X-API-KEY'] = apiKey.trim();

  const doPost = async () => {
    setLoading(true);
    setTurns(t => [...t, { role: 'user', text: query }]);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ tenant_id: tenantId, query, top_k: topK }),
      });
      if (!res.ok) {
        const txt = await res.text();
        appendLog(`HTTP ${res.status}: ${txt}`);
        setTurns(t => [...t, { role: 'assistant', text: `Error: ${res.status} ${txt}` }]);
        return;
      }
      const data = await res.json();
      const answer = data.answer ?? '';
      const sources = (data.sources ?? []) as Source[];
      setTurns(t => [...t, { role: 'assistant', text: answer, sources }]);
    } catch (e: any) {
      appendLog(`POST error: ${e?.message || String(e)}`);
      setTurns(t => [...t, { role: 'assistant', text: `Error: ${e?.message || String(e)}` }]);
    } finally {
      setLoading(false);
    }
  };

  const doStream = async () => {
    setStreaming(true);
    setTurns(t => [...t, { role: 'user', text: query }]);
    try {
      const qs = new URLSearchParams();
      if (token.trim()) qs.set('token', token.trim());
      if (apiKey.trim()) qs.set('api_key', apiKey.trim());
      if (tenantId.trim()) qs.set('tenant_id', tenantId.trim());
      const url = `${WS_BASE}/chat/stream?${qs.toString()}`;
      const ws = new WebSocket(url);
      let answerBuf = '';
      let sources: Source[] = [];
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'query', tenant_id: tenantId, query, top_k: topK }));
        appendLog('WS: opened');
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.event === 'node') {
            appendLog(`node:${msg.name}`);
          } else if (msg.event === 'token') {
            answerBuf += msg.data?.text || '';
            setTurns(t => {
              const copy = t.slice();
              const last = copy[copy.length - 1];
              if (!last || last.role !== 'assistant') {
                copy.push({ role: 'assistant', text: answerBuf });
              } else {
                copy[copy.length - 1] = { ...last, text: answerBuf };
              }
              return copy;
            });
          } else if (msg.event === 'final') {
            const data = msg.data || {};
            answerBuf = data.answer || answerBuf;
            sources = (data.sources || []) as Source[];
            setTurns(t => [...t.filter(turn => !(turn.role === 'assistant' && turn.text === '')), { role: 'assistant', text: answerBuf, sources }]);
          } else if (msg.event === 'error') {
            appendLog(`WS error: ${msg.data?.message || 'unknown'}`);
          } else {
            appendLog(`WS event: ${JSON.stringify(msg)}`);
          }
        } catch (e) {
          appendLog(`WS parse error: ${String(e)}`);
        }
      };
      ws.onerror = (e) => appendLog(`WS onerror: ${JSON.stringify(e)}`);
      ws.onclose = () => { appendLog('WS: closed'); setStreaming(false); };
    } catch (e: any) {
      appendLog(`WS setup error: ${e?.message || String(e)}`);
      setStreaming(false);
    }
  };

  return (
    <div style={{ maxWidth: 960, margin: '40px auto', padding: '0 16px', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Finance Agentic RAG — Chat</h1>
      <p style={{ color: '#666' }}>POST and optional WS streaming demo. Provide credentials to respect RBAC.</p>
      <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <label style={{ display: 'grid' }}>
          <span>Tenant ID</span>
          <input value={tenantId} onChange={(e) => setTenantId(e.target.value)} placeholder="t0" />
        </label>
        <label style={{ display: 'grid' }}>
          <span>Top K</span>
          <input type="number" min={1} max={50} value={topK} onChange={(e) => setTopK(parseInt(e.target.value || '6', 10))} />
        </label>
        <label style={{ display: 'grid' }}>
          <span>Bearer Token (Authorization)</span>
          <input value={token} onChange={(e) => setToken(e.target.value)} placeholder="ey..." />
        </label>
        <label style={{ display: 'grid' }}>
          <span>API Key (X-API-KEY)</span>
          <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk_live_..." />
        </label>
      </section>
      <div style={{ marginTop: 16 }}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={4}
          style={{ width: '100%', fontFamily: 'inherit' }}
          placeholder="Ask a question about your financial documents..."
        />
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button disabled={loading || streaming} onClick={doPost}>{loading ? 'Sending…' : 'Send (POST)'}</button>
        <button disabled={loading || streaming} onClick={doStream} title="Requires WS server to accept query param auth">
          {streaming ? 'Streaming…' : 'Stream (WS)'}
        </button>
      </div>
      <hr style={{ margin: '24px 0' }} />
      <section>
        <h3>Conversation</h3>
        <div style={{ display: 'grid', gap: 12 }}>
          {turns.map((t, i) => (
            <div key={i} style={{ background: t.role === 'user' ? '#f6f8fa' : '#f3fdf8', border: '1px solid #e2e2e2', padding: 12, borderRadius: 8 }}>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>{t.role.toUpperCase()}</div>
              <div style={{ whiteSpace: 'pre-wrap' }}>{t.text}</div>
              {t.sources && t.sources.length > 0 && (
                <details style={{ marginTop: 6 }}>
                  <summary>Citations ({t.sources.length})</summary>
                  <ul style={{ marginTop: 6 }}>
                    {t.sources.map((s, j) => (<li key={j}><code>{(s.metadata?.source ?? 'unknown')}</code> — score {s.score.toFixed(3)}</li>))}
                  </ul>
                </details>
              )}
            </div>
          ))}
        </div>
      </section>
      <section style={{ marginTop: 24 }}>
        <h3>Debug Log</h3>
        <pre style={{ background: '#111', color: '#eee', padding: 12, borderRadius: 8, maxHeight: 220, overflow: 'auto' }}>{log.join('\n')}</pre>
      </section>
    </div>
  );
}
