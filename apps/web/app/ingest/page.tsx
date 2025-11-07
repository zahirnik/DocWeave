
'use client';
/**
 * Drag-and-drop uploader + job progress UI for the ingestion pipeline.
 *
 * Works with the FastAPI endpoints:
 *   - POST /ingest/files  (multipart form upload)
 *   - POST /ingest/urls   (JSON list of URLs)
 *   - GET  /ingest/{id}   (poll job status/progress)
 *
 * Environment:
 *   NEXT_PUBLIC_API_BASE   e.g., http://localhost:8000
 *
 * Auth:
 *   Use Bearer token or X-API-KEY. Tenant id is optional (defaults to "t0").
 */

import React from 'react';

type JobStatus = {
  job_id: string;
  state: 'PENDING' | 'STARTED' | 'RETRY' | 'FAILURE' | 'SUCCESS' | 'PROGRESS' | string;
  progress?: number;
  result?: any;
  error?: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

export default function IngestPage() {
  const [tenantId, setTenantId] = React.useState('t0');
  const [collection, setCollection] = React.useState('default');
  const [token, setToken] = React.useState('');   // Bearer
  const [apiKey, setApiKey] = React.useState(''); // X-API-KEY

  const [files, setFiles] = React.useState<File[]>([]);
  const [urls, setUrls] = React.useState<string>(''); // one per line
  const [busy, setBusy] = React.useState(false);
  const [jobId, setJobId] = React.useState<string | null>(null);
  const [status, setStatus] = React.useState<JobStatus | null>(null);
  const [log, setLog] = React.useState<string[]>([]);

  const appendLog = (line: string) => setLog(prev => [...prev.slice(-500), line]);

  const headers: Record<string, string> = { 'X-Tenant-ID': tenantId };
  if (token.trim()) headers['Authorization'] = `Bearer ${token.trim()}`;
  if (apiKey.trim()) headers['X-API-KEY'] = apiKey.trim();

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const fl = Array.from(e.target.files || []);
    setFiles(fl);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const fl = Array.from(e.dataTransfer.files || []);
    setFiles(fl);
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => e.preventDefault();

  const startUpload = async () => {
    if (files.length === 0) { appendLog('No files to upload.'); return; }
    setBusy(true);
    setStatus(null);
    try {
      const form = new FormData();
      for (const f of files) form.append('files', f, f.name);
      form.append('collection', collection);
      form.append('tenant_id', tenantId);

      const res = await fetch(`${API_BASE}/ingest/files`, {
        method: 'POST',
        headers, // NOTE: do NOT set Content-Type; browser sets multipart boundary
        body: form,
      });
      if (!res.ok) {
        appendLog(`Upload failed: HTTP ${res.status} ${await res.text()}`);
        setBusy(false);
        return;
      }
      const data = await res.json();
      setJobId(data.job_id);
      appendLog(`Enqueued job: ${data.job_id}`);
    } catch (e: any) {
      appendLog(`Upload error: ${e?.message || String(e)}`);
      setBusy(false);
    }
  };

  const startUrls = async () => {
    const list = urls.split('\n').map(s => s.trim()).filter(Boolean);
    if (list.length === 0) { appendLog('No URLs provided.'); return; }
    setBusy(true);
    setStatus(null);
    try {
      const res = await fetch(`${API_BASE}/ingest/urls`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls: list, collection, tenant_id: tenantId }),
      });
      if (!res.ok) {
        appendLog(`URLs enqueue failed: HTTP ${res.status} ${await res.text()}`);
        setBusy(false);
        return;
      }
      const data = await res.json();
      setJobId(data.job_id);
      appendLog(`Enqueued URL job: ${data.job_id}`);
    } catch (e: any) {
      appendLog(`URLs error: ${e?.message || String(e)}`);
      setBusy(false);
    }
  };

  // Poll job if we have an id
  React.useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/ingest/${jobId}`, { headers });
        if (!res.ok) {
          appendLog(`Poll error: HTTP ${res.status}`);
          return;
        }
        const st = await res.json() as JobStatus;
        if (!cancelled) setStatus(st);
        if (st.state === 'SUCCESS' || st.state === 'FAILURE') {
          setBusy(false);
        }
      } catch (e: any) {
        appendLog(`Poll exception: ${e?.message || String(e)}`);
      }
    };
    // First poll immediately, then repeat
    tick();
    const h = setInterval(tick, 1500);
    return () => { cancelled = true; clearInterval(h); };
  }, [jobId]);

  const reset = () => {
    setFiles([]);
    setUrls('');
    setJobId(null);
    setStatus(null);
    setBusy(false);
  };

  const barWidth = Math.round(((status?.progress ?? 0) * 100));

  return (
    <div style={{ maxWidth: 960, margin: '40px auto', padding: '0 16px', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Finance Agentic RAG — Ingest</h1>
      <p style={{ color: '#666' }}>Upload files or provide URLs. The worker parses → chunks → embeds → upserts.</p>

      <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <label style={{ display: 'grid' }}>
          <span>Tenant ID</span>
          <input value={tenantId} onChange={(e) => setTenantId(e.target.value)} placeholder="t0" />
        </label>
        <label style={{ display: 'grid' }}>
          <span>Collection</span>
          <input value={collection} onChange={(e) => setCollection(e.target.value)} placeholder="default" />
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

      <section style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div>
          <h3>File Upload</h3>
          <div onDrop={onDrop} onDragOver={onDragOver}
               style={{ border: '2px dashed #aaa', padding: 16, borderRadius: 8, textAlign: 'center' }}>
            Drag & drop files here
          </div>
          <input type="file" multiple onChange={onFileChange} style={{ marginTop: 8 }} />
          <ul style={{ marginTop: 8 }}>
            {files.map((f, i) => <li key={i}><code>{f.name}</code> — {(f.size/1024/1024).toFixed(2)} MB</li>)}
          </ul>
          <button disabled={busy || files.length === 0} onClick={startUpload}>
            {busy ? 'Working…' : 'Ingest Files'}
          </button>
        </div>

        <div>
          <h3>URLs</h3>
          <textarea
            value={urls}
            onChange={(e) => setUrls(e.target.value)}
            placeholder={"https://example.com/report.pdf\nhttps://example.com/table.csv"}
            rows={8}
            style={{ width: '100%', fontFamily: 'inherit' }}
          />
          <button disabled={busy} onClick={startUrls}>
            {busy ? 'Working…' : 'Ingest URLs'}
          </button>
        </div>
      </section>

      <section style={{ marginTop: 24 }}>
        <h3>Job Status</h3>
        {jobId ? <div><b>Job:</b> <code>{jobId}</code></div> : <div>No job yet.</div>}
        {status && (
          <div style={{ marginTop: 8 }}>
            <div><b>State:</b> {status.state}</div>
            {"progress" in status && typeof status.progress === 'number' && (
              <div style={{ background: '#eee', borderRadius: 6, marginTop: 6 }}>
                <div style={{
                  width: `${barWidth}%`,
                  background: '#4caf50',
                  height: 10,
                  borderRadius: 6
                }} />
              </div>
            )}
            {status.error && <div style={{ color: 'crimson', marginTop: 8 }}>Error: {status.error}</div>}
            {status.result && (
              <details style={{ marginTop: 8 }}>
                <summary>Result</summary>
                <pre style={{ background: '#111', color: '#eee', padding: 12, borderRadius: 8, overflow: 'auto' }}>
                  {JSON.stringify(status.result, null, 2)}
                </pre>
              </details>
            )}
          </div>
        )}
      </section>

      <section style={{ marginTop: 24 }}>
        <button onClick={reset}>Reset</button>
      </section>

      <section style={{ marginTop: 24 }}>
        <h3>Debug Log</h3>
        <pre style={{ background: '#111', color: '#eee', padding: 12, borderRadius: 8, maxHeight: 220, overflow: 'auto' }}>
{log.join('\n')}
        </pre>
      </section>
    </div>
  );
}
