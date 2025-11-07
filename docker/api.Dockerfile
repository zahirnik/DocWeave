# docker/api.Dockerfile
# Multi-stage, non-root, healthcheck-enabled image for the FastAPI service.
# Built for clarity and reproducibility (tutorial style).

FROM python:3.11-slim AS base

# Prevents Python from writing .pyc files and enables unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1

# System deps (psycopg, pillow, etc.), curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG APP_USER=appuser
RUN useradd -m -u 1000 ${APP_USER}
WORKDIR /app

# Copy dependency manifests first for better layer caching
# We support either pyproject.toml (uv/poetry/pip) or requirements.txt
COPY pyproject.toml* requirements.txt* ./

# Install dependencies (prefer pip for simplicity)
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi && \
    if [ -f pyproject.toml ]; then pip install --no-cache-dir -e .; fi

# Copy the rest of the application
COPY . .

# Expose API port
EXPOSE 8000

# Drop privileges
USER ${APP_USER}

# Healthcheck: FastAPI live endpoint (define /healthz in your app)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
    curl -fsS http://localhost:8000/healthz || exit 1

# Uvicorn entrypoint. Adjust workers/threads as needed.
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
