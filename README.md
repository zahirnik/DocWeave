# Convai Finance Agentic RAG

**Agentic RAG** for finance documents (PDF, CSV/XLSX, JSON/JSONL, TXT/MD) with
tutorial-level clarity and production-minded patterns.

- Ingestion → parse/normalize → chunk → embed → vector stores (pgvector/qdrant/chroma)
- Hybrid retrieval (vector+BM25) + (optional) reranker
- LangGraph agent: route → retrieve → ground → analyze (tables/charts) → answer
- FastAPI API + Celery worker, multitenant isolation, observability hooks
- Fully offline-friendly examples — add keys later

## Quickstart
See [docs/QUICKSTART.md](docs/QUICKSTART.md).

## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## API
See [docs/API.md](docs/API.md).

## Security & Runbooks
See [docs/SECURITY.md](docs/SECURITY.md) and [docs/RUNBOOKS.md](docs/RUNBOOKS.md).

## License
MIT — see [LICENSE](LICENSE).
