# Simple developer Makefile (tutorial-clear).

.PHONY: help dev run api worker index test lint type docker

help:
	@echo "Targets:"
	@echo "  dev      - create venv and install (editable)"
	@echo "  run      - run API locally (uvicorn)"
	@echo "  api      - same as run"
	@echo "  worker   - run Celery worker"
	@echo "  index    - build a small index from data/samples"
	@echo "  test     - run pytest"
	@echo "  lint     - run ruff"
	@echo "  type     - run mypy"
	@echo "  docker   - docker compose up"

dev:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -e .[dev]

run api:
	uvicorn apps.api.main:app --reload --port 8000

worker:
	celery -A apps.worker.celery_app:celery worker -l INFO -Q default,ingest,embed

index:
	python -m scripts.build_index --source data/samples --collection finance_demo

test:
	pytest -q

lint:
	ruff check .

type:
	mypy packages

docker:
	docker compose -f docker/docker-compose.yml up --build
