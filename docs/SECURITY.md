# Security Guide

## Data handling & PII

- **PII detection**: `packages/security/pii.py` uses regex + heuristics to mask emails,
  phone numbers, IBAN/CC, NI numbers. Apply redaction before storing long-term logs.
- **Tenant isolation**: collection-per-tenant (vector store) and schema/row-level
  constraints (SQL) where possible.
- **Retention**: configure retention windows in `configs/policies.yaml`.

## Authentication & Authorisation

- API supports API keys and JWT/OIDC (see `apps/api/routes/auth.py`).
- RBAC helpers in `packages/core/auth.py` apply role/tenant checks per route.

## Secrets management

- Local dev: `.env` (never commit actual secrets).
- Docker/K8s: environment variables via **Secrets**; consider External Secrets or sealed secrets.

## Audit & observability

- Append-only audit logs for sensitive ops (`packages/core/logging.py`/`audit` tables).
- OpenTelemetry traces for all requests/tool-calls; LangSmith prompt traces (feature-flagged).
- Prometheus metrics exported at `/metrics` (if enabled).

## Supply chain

- CI scans dependencies (Trivy/Grype) and pins indirect deps where possible.
- Renovate/Dependabot recommended for automated updates.

## Threats considered

- Prompt injection and tool abuse → guarded by `packages/security/firewall.py` and policy engine.
- Data exfiltration (cross-tenant) → strict RBAC + tenant filters + isolated stores.
- SSRF on ingestion → allowlist domains; restrict URL fetcher.
