# packages/core/db.py
"""
SQLAlchemy engine/session helpers (tiny, explicit, tutorial-clear).

What this module provides
-------------------------
- build_database_url(): decide on DB URL (env or dev-friendly sqlite fallback)
- get_engine(): create a SQLAlchemy 2.x Engine with safe pool defaults
- SessionLocal: session factory (autocommit=False, autoflush=False)
- session_scope(): context manager for commit/rollback ergonomics
- healthcheck(): simple SELECT 1 to verify connectivity
- create_all(Base): DEV-ONLY helper to create tables without Alembic

How to run locally (dev)
------------------------
Option A — Postgres (recommended)
    docker run --rm -e POSTGRES_PASSWORD=pass -p 5432:5432 postgres:15
    export DATABASE_URL="postgresql+psycopg://postgres:pass@localhost:5432/postgres"
    # (optional) install pgvector in your DB and/or run: CREATE EXTENSION IF NOT EXISTS vector;

Option B — SQLite (zero deps; dev only)
    unset DATABASE_URL
    python -c "from packages.core.db import create_all, get_engine; from packages.core import models as m; create_all(m.Base)"

Migrations
----------
Use Alembic for schema changes:
    alembic init migrations
    # configure env.py to import Settings->DATABASE_URL
    alembic revision -m "create tables"
    alembic upgrade head
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from packages.core.config import get_settings


# ---------------------------
# URL selection
# ---------------------------

def build_database_url() -> str:
    """
    Decide which DB URL to use.

    Priority:
      1) $DATABASE_URL (e.g., postgresql+psycopg://user:pass@host:5432/db)
      2) dev fallback: sqlite file under DATA_DIR

    Returns:
        SQLAlchemy connection URL (driver-qualified)
    """
    settings = get_settings()
    if settings.database_url:
        return settings.database_url

    # Dev-friendly fallback: SQLite file in data dir
    sqlite_path = os.path.join(settings.data_dir, "app.db")
    return f"sqlite+pysqlite:///{sqlite_path}"


# ---------------------------
# Engine / Session factory
# ---------------------------

_ENGINE: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine(echo: bool = False) -> Engine:
    """
    Return a process-wide Engine (singleton).
    Uses pre-ping and conservative pool sizing to behave well in dev & prod.

    Pool guidance (tune per deployment):
      - pool_size=5, max_overflow=10 → modest concurrency
      - pool_pre_ping=True → recovers from stale connections cleanly
    """
    global _ENGINE, _SessionLocal
    if _ENGINE is not None:
        return _ENGINE

    url = build_database_url()
    is_sqlite = url.startswith("sqlite")
    _ENGINE = create_engine(
        url,
        echo=echo,
        future=True,
        pool_pre_ping=True,
        # SQLite has no real pooling; kwargs are ignored for it.
        pool_size=None if is_sqlite else 5,
        max_overflow=None if is_sqlite else 10,
    )

    # Build session factory
    _SessionLocal = sessionmaker(
        bind=_ENGINE,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
        class_=Session,
    )

    return _ENGINE


def _require_session_factory() -> sessionmaker:
    if _SessionLocal is None:
        get_engine()  # initializes _SessionLocal
    assert _SessionLocal is not None  # just for type checkers
    return _SessionLocal


def get_session() -> Session:
    """
    Return a new Session from the global factory.
    Prefer `session_scope()` for automatic commit/rollback.
    """
    return _require_session_factory()()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager to handle commit/rollback automatically.

    Example:
        from packages.core.db import session_scope
        with session_scope() as s:
            s.add(obj)
            ...  # automatic commit on success; rollback on exception
    """
    s = get_session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ---------------------------
# Utilities
# ---------------------------

def healthcheck() -> bool:
    """
    Execute a trivial query to verify connectivity.
    Returns True on success, False on failure (never raises).
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def create_all(Base) -> None:
    """
    DEV-ONLY: Create all tables from a Declarative Base.
    Use Alembic for production migrations.

    Example:
        from packages.core import models as m
        from packages.core.db import create_all
        create_all(m.Base)
    """
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def try_enable_pgvector() -> None:
    """
    Best-effort attempt to enable the 'vector' extension in Postgres.
    - Requires appropriate DB privileges (often superuser).
    - Safe to call even if not on Postgres (it will no-op on non-PG URLs).

    Call this during bootstrap if your deployment expects pgvector:
        from packages.core.db import try_enable_pgvector
        try_enable_pgvector()
    """
    url = build_database_url()
    if not url.startswith("postgresql"):
        return
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception:
        # Do not crash if extension creation is not permitted.
        pass
