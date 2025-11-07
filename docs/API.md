# HTTP & WebSocket API

Base URL: `http://localhost:8000`

## Health
```
GET /healthz  → 200 OK {"status":"ok"}
```

## Auth
- API keys via `Authorization: Bearer <token>`
- Or OIDC/JWT (see `apps/api/routes/auth.py`)

---

## Chat

### POST /chat
Ask a question; returns answer + citations. For streaming, use WebSocket.

**Request (JSON)**
```json
{
  "question": "What was ACME's gross margin in 2024?",
  "top_k": 5,
  "tenant_id": "t0",
  "tools": ["tabular_stats", "charting"],
  "track": true
}
```

**Response (JSON)**
```json
{
  "answer": "ACME reported a 38.2% gross margin in 2024. [1]",
  "contexts": [
    {"doc_id":"acme_annual_report_2024.pdf", "score":0.82, "snippet":"... gross margin improved to 38.2% ..."}
  ],
  "artifacts": [{"kind":"image/png","path":"outputs/chart_2024Q3.png"}],
  "usage": {"tokens_prompt": 512, "tokens_completion": 78}
}
```

**cURL**
```bash
curl -s -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{"question":"ACME gross margin 2024?","top_k":5}'
```

### WS /stream
Server-sent token stream (words/chunks). See the Next.js demo for a live example.

---

## Ingestion

### POST /ingest
Upload files or URLs to index.
- Accepts `multipart/form-data` or JSON with `urls`.

**cURL (file)**
```bash
curl -s -X POST http://localhost:8000/ingest   -F "files=@data/samples/acme_annual_report_2024.txt"   -F "tenant_id=t0"
```

**cURL (URLs)**
```bash
curl -s -X POST http://localhost:8000/ingest   -H "Content-Type: application/json"   -d '{"urls":["https://example.com/acme/q3-2024"],"tenant_id":"t0"}'
```

Response includes a `job_id` you can poll for progress.

---

## Search

### GET /search
Hybrid search with pagination and filters.

**Query params**
- `q` (string, required)
- `top_k` (int, default 5)
- `tenant_id`, `company`, `topic`, `from`, `to` (optional filters)
- `page`, `per_page`

**Example**
```
GET /search?q=gross%20margin&top_k=5&tenant_id=t0
```

---

## Analytics

### POST /analytics
Upload a CSV/XLSX and run tabular calculations; optionally return a chart.

**cURL**
```bash
curl -s -X POST http://localhost:8000/analytics   -F "file=@data/samples/financials_quarterly.csv"   -F "op=yoy"   -F "column=revenue_usd_m"
```

**Response**
```json
{"table":[["quarter","value","yoy"],["2024Q3",1260,0.145]],"chart":"outputs/revenue_yoy.png"}
```

---

## Admin (examples)
- `GET /metrics` — Prometheus endpoint if enabled.
- `GET /version` — Build/version string.
