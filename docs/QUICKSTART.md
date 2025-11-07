# Quickstart (5–10 minutes)

This repo ships with runnable **tutorial-level** examples and production-minded defaults.

## 0) Requirements
- Python 3.11+
- (Optional) Postgres 16 with `pgvector` extension (or use Docker compose)
- (Optional) Redis 7
- macOS/Linux; Windows (WSL2 recommended)

## 1) Create a virtual environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If you prefer `uv` or `poetry`, the repo works with those as well.

## 2) Configure environment
Copy and edit the sample env (add keys if you have them; otherwise the code uses local fallbacks):
```bash
cp configs/.env.sample .env
# edit .env to set DATABASE_URL, OPENAI_API_KEY, etc. (optional)
```

## 3) Seed sample data (already included)
Tiny PDFs/TXT/CSV/JSONL live in `data/samples/`. You can add more documents later.

## 4) Run an end-to-end example
```bash
python examples/00_quickstart_minimal.py
```
This ingests local samples into an in-memory store and answers a simple finance question
with citations (no external keys required).

Other examples:
```bash
python examples/02_build_index_pgvector.py --collection finance_demo
python examples/03_langgraph_chat_basic.py
python examples/04_tabular_stats_demo.py
```

## 5) Start the API (dev)
```bash
uvicorn apps.api.main:app --reload --port 8000
# Open http://localhost:8000/docs for Swagger UI
```

## 6) Docker compose (optional)
```bash
docker compose -f docker/docker-compose.yml up --build
# API at http://localhost:8000
```

## 7) Tests
```bash
pytest -q
```

## 8) Profiling (latency)
```bash
python -m scripts.profile_latency --runs 20 --no-llm
```

## 9) Re-indexing or migration
```bash
python -m scripts.build_index --source data/samples --collection finance_demo
python -m scripts.migrate_store --src-store chroma --src-collection old_name   --dst-store pgvector --dst-collection new_name --reembed
```

**Tip:** Everything defaults to **offline-friendly** fallbacks so you can get
started immediately. Add keys later for OpenAI/Tavily and switch backends via configs.
