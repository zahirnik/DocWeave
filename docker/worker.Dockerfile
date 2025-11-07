# docker/worker.Dockerfile
# Celery worker image for ingestion/embedding/background tasks.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ARG APP_USER=appuser
RUN useradd -m -u 1000 ${APP_USER}
WORKDIR /app

COPY pyproject.toml* requirements.txt* ./

RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi && \
    if [ -f pyproject.toml ]; then pip install --no-cache-dir -e .; fi

COPY . .

USER ${APP_USER}

# Default command runs a single Celery worker. Adjust -Q queues to your routing.
# Expect a Celery app at apps.worker.celery_app:celery (as in the repo plan).
CMD ["celery", "-A", "apps.worker.celery_app:celery", "worker", "-l", "INFO", "-Q", "default,ingest,embed"]
