# apps/worker/celery_app.py
"""
Celery configuration (minimal, well-commented).

What this file does
-------------------
- Creates a Celery application object configured from environment variables.
- Sets sane defaults for JSON serialization, timeouts, and result backend.
- Names a default queue ("ingest") for long-running ingestion/indexing jobs.
- Keeps everything tiny and tutorial-clear.

How to run locally
------------------
# 1) Start a Redis broker (or point to yours via env)
docker run --rm -p 6379:6379 redis:7

# 2) Start the Celery worker
CELERY_BROKER_URL=redis://localhost:6379/0 \
CELERY_RESULT_BACKEND=redis://localhost:6379/0 \
celery -A apps.worker.celery_app.app worker --loglevel=INFO -Q ingest

# (Optional) Start beat if you add periodic tasks later:
celery -A apps.worker.celery_app.app beat --loglevel=INFO

Environment variables
---------------------
CELERY_BROKER_URL       default: redis://localhost:6379/0
CELERY_RESULT_BACKEND   default: redis://localhost:6379/0
CELERY_TASK_TIMEOUT_S   hard time limit per task (seconds), default: 3600
CELERY_SOFT_TIMEOUT_S   soft time limit per task (seconds), default: 3300
"""

from __future__ import annotations

import os
from celery import Celery

# ---- Read env with safe defaults (great for local dev) -----------------------

BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
RESULT_BACKEND = os.getenv(
    "CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://localhost:6379/0")
)
SOFT_TIMEOUT_S = int(os.getenv("CELERY_SOFT_TIMEOUT_S", "3300"))  # 55 minutes
HARD_TIMEOUT_S = int(os.getenv("CELERY_TASK_TIMEOUT_S", "3600"))  # 60 minutes


# ---- Create the Celery app ---------------------------------------------------

app = Celery("convai_finance_agentic_rag", broker=BROKER_URL, backend=RESULT_BACKEND)

# Keep config explicit and compact
app.conf.update(
    task_default_queue="ingest",
    task_queues=None,  # single default queue; easy to add more later
    # Serialization (JSON-only keeps it simple & safe)
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    # Track progress + extended results so /ingest/{id} can display meta info
    task_track_started=True,
    result_extended=True,
    # Timeouts
    task_soft_time_limit=SOFT_TIMEOUT_S,
    task_time_limit=HARD_TIMEOUT_S,
    # Worker behavior
    worker_prefetch_multiplier=1,  # fairness for long tasks
    worker_max_tasks_per_child=100,  # mitigate memory leaks in long-running jobs
    # Timezone/UTC
    enable_utc=True,
    timezone="UTC",
)

# Optional: periodic tasks (uncomment when you add them)
# from celery.schedules import crontab
# app.conf.beat_schedule = {
#     "nightly-reembed": {
#         "task": "apps.worker.tasks.nightly_reembed",
#         "schedule": crontab(hour=2, minute=0),
#         "options": {"queue": "ingest"},
#     }
# }

# Tip: If you need to inspect from a REPL:
# >>> from apps.worker.celery_app import app
# >>> app.control.ping()
