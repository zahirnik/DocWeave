# Architecture

This project balances **tutorial clarity** with **industry-grade** patterns.

```
           +--------------------+                 +-----------------------+
           |  apps/api (FastAPI)|  Celery tasks   |  apps/worker (Celery) |
  HTTP/WS  +---------+----------+<----------------+-----------+-----------+
    ▶ /chat          |                                  ingest/embed/upsert
    ▶ /ingest        |                                           |
                     v                                           v
          +-------------------+                      +-----------------------+
          |  agent_graph      |--retrieve/ground-->  |  packages/retriever   |
          |  (LangGraph DAG)  |                      |  (vector/BM25/rerank) |
          +---------+---------+                      +-----------+-----------+
                    |                                             |
                    v                                             v
          +-------------------+                      +-----------------------+
          |  analytics tools  |<--DataFrames-------->|  packages/analytics   |
          |  charts/stats     |                      |  (ratios/timeseries)  |
          +-------------------+                      +-----------------------+

          +-------------------+      +-----------------+      +------------------+
          |  Postgres+pgvector|◀-----| vector stores   |◀-----|  ingestion       |
          +-------------------+      | (Qdrant/Chroma) |      |  loaders/ocr     |
                                     +-----------------+      +------------------+

          +-------------------+
          |  observability    |  (OpenTelemetry + LangSmith + Prometheus)
          +-------------------+
```

## Key packages

- **apps/** — FastAPI service (`/chat`, `/ingest`, `/search`, `/analytics`) and Celery worker.
- **packages/agent_graph/** — LangGraph DAG: *route → retrieve → ground → analyze → answer*.
- **packages/ingestion/** — Loaders for PDF/DOCX/CSV/JSON, OCR, normalisers, validators.
- **packages/retriever/** — Chunking, embeddings, vector stores (pgvector, qdrant, chroma), hybrid search.
- **packages/analytics/** — Pandas/statsmodels helpers; chart export.
- **packages/security/** — PII redaction, firewall (prompt/tool guard), policy engine (allow/deny).
- **packages/core/** — Config, SQLAlchemy models, RBAC, cache, storage, logging, telemetry.

## Flow (chat)

1. **Auth** (OIDC/JWT or API key) → **/chat** accepts a query.
2. **Graph**: route intent → retrieve documents (hybrid + rerank) → ground the answer with quotes.
3. **(Optional)** analytics tool runs DataFrame math and emits charts (PNG/SVG).
4. **Answer** returns with citations + tool artifacts (file links).

## Flow (ingestion)

1. **/ingest** uploads files/URLs → **worker** pipeline runs OCR/parse → normalise → chunk → embed.
2. **Vector upsert** with document-level metadata (tenant, source, tags).
3. **Audit** and metrics recorded; failures retried with backoff.

## Multitenancy

- **Hard isolation** by collection or namespace per tenant in vector store + DB schemas.
- **Soft isolation** by `tenant_id` metadata and RBAC checks in every query path.

## Observability

- Traces via **OpenTelemetry** (export to collector).
- Prompt/tool traces via **LangSmith** (feature-flagged).
- Metrics via **Prometheus** counters/histograms (optional).
