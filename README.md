# DocWeave

DocWeave is an enterprise-ready, agentic Retrieval-Augmented Generation (RAG) platform for document-heavy workflows, with a strong fit for regulated domains.

It helps teams ingest heterogeneous documents, retrieve grounded evidence, and deliver auditable answers through APIs and UI surfaces.

## Why DocWeave

- End-to-end RAG workflow from ingestion to answer generation
- Multi-format document support: PDF, CSV/XLSX, JSON/JSONL, TXT/MD
- Hybrid retrieval (vector + BM25) with optional reranking
- Agentic orchestration for retrieval, grounding, and analysis
- Multi-tenant patterns for isolation and scale
- Production-oriented API, worker, observability, and runbook support

## Core Capabilities

- Ingestion pipeline: parse -> normalize -> chunk -> embed -> index
- Vector backends: `pgvector`, `qdrant`, `chroma`
- LangGraph agent flow: route -> retrieve -> ground -> analyze -> answer
- FastAPI application layer for serving and integration
- Celery worker for asynchronous/background processing
- Security and operations guides for deployment hardening

## Architecture and Docs

- Quickstart: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API: [docs/API.md](docs/API.md)
- Security: [docs/SECURITY.md](docs/SECURITY.md)
- Runbooks: [docs/RUNBOOKS.md](docs/RUNBOOKS.md)

## Quick Start

```bash
make dev
make run
```

In a separate terminal:

```bash
make worker
```

Optional sample indexing:

```bash
make index
```

## Local Development

Available Make targets:

- `make dev` - create virtual environment and install editable dependencies
- `make run` / `make api` - start the FastAPI app with reload
- `make worker` - start Celery worker queues
- `make index` - build sample index from `data/samples`
- `make test` - run test suite
- `make lint` - run Ruff lint checks
- `make type` - run MyPy type checks
- `make docker` - run with Docker Compose

## Deployment Notes

DocWeave includes deployment assets under `deploy/` and `docker/` for containerized and Kubernetes/Helm-based environments.

Before production rollout:

- Configure secrets and environment variables securely
- Validate tenant isolation and data boundaries
- Enable tracing/metrics and log redaction policies
- Review [docs/SECURITY.md](docs/SECURITY.md) and [docs/RUNBOOKS.md](docs/RUNBOOKS.md)

## Project Scope

This repository focuses on practical, production-minded RAG patterns with tutorial-level clarity for fast onboarding.

## License

MIT - see [LICENSE](LICENSE).
