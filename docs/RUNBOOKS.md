# Runbooks

Operational playbooks for common incidents.

## Ingestion queue jammed

**Symptoms**: `/ingest` returns job IDs that never complete; Redis queue depth grows.

**Checks**
1. `kubectl logs deploy/rag-worker -n convai-rag`
2. Redis live keys: `redis-cli -h redis -n 0 keys "celery*" | wc -l`

**Actions**
- Scale workers: `kubectl scale deploy/rag-worker -n convai-rag --replicas=3`
- Retry stuck tasks: restart the worker deployment (Celery auto requeues).
- Validate storage permissions (`packages/core/storage.py`) and antivirus settings.

## Embedding failures

**Symptoms**: logs show provider timeouts or auth errors.

**Checks**
- `OPENAI_API_KEY` present? Rate-limited?
- Fallback in `EmbeddingClient(provider="auto")` is used?

**Actions**
- Switch model alias in `configs/models.yaml` to a local one (e.g., `gte`).
- Re-run: `python -m scripts.reembed_collection --collection finance_demo --alias gte`.

## DB/pgvector issues

**Symptoms**: connection refused, search returns 0 results.

**Checks**
- `DATABASE_URL` / `PGVECTOR_URL` reachable from API/worker Pods.
- Extension present: `CREATE EXTENSION IF NOT EXISTS vector;`

**Actions**
- Run migrations: `alembic upgrade head`
- Recreate collection or vacuum if large deletions occurred.

## High latency

**Checks**
- `python -m scripts.profile_latency --runs 20 --no-llm`
- p95 > target? Check retriever store CPU/memory saturation.

**Actions**
- Reduce `top_k` or enable reranker only for the top-20.
- Batch embeddings to GPU-enabled workers.
- Enable caching (`packages/core/cache.py`).

## Key rotation / secrets

- Keep secrets out of ConfigMaps. Use Kubernetes Secrets or external secret managers.
- Rolling restart deployments after updating secrets.

## Incident templates

- Timeline, impact, root cause, remediation, follow-ups.
