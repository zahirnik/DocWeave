# packages/core/models.py
"""
ORM models (SQLAlchemy 2.x) — tiny, explicit, and multi-tenant ready.

Tables included
---------------
- Tenant           : org/workspace boundary (simple id+name).
- User             : app user with roles (denormalized list for tutorial clarity).
- Document         : an ingested source (PDF/CSV/JSON...); checksum + lineage.
- Chunk            : a retrievable text chunk linked to Document (optionally stores embedding id/vector).
- ChatSession      : a chat thread (for UX continuity/history).
- AuditEvent       : append-only audit trail (who did what, when, where).

Design goals
------------
- Keep schemas minimal but **realistic**; add columns later with Alembic migrations.
- Work on both Postgres and SQLite for easy local dev.
- Add obvious indexes (tenant_id, document foreign keys, created_at).
- Avoid heavy ORM magic; keep models straightforward and commented.

Usage
-----
from packages.core.db import get_engine, create_all
from packages.core import models as m

engine = get_engine()
create_all(m.Base)

with session_scope() as s:
    t0 = m.Tenant(id="t0", name="Demo")
    s.add(t0)

Notes on embeddings
-------------------
We store vector embeddings in your vector store (pgvector/qdrant/chroma). In the DB,
`Chunk.embedding_id` (str) can reference the vector store's primary key. If you run
pgvector in Postgres and want to store vectors in-DB, add a VECTOR column via a migration.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    Index,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dialect-friendly helpers (work on SQLite & Postgres)
try:
    from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
    _HAS_PG = True
except Exception:  # pragma: no cover
    PGUUID = None
    JSONB = None
    _HAS_PG = False


# ---------------------------
# Base & mixins
# ---------------------------

class Base(DeclarativeBase):
    """Declarative base for all models."""


def _uuid_col(pk: bool = False) -> mapped_column:
    """
    UUID column that works on SQLite & Postgres.
    We store as TEXT on SQLite; PG uses native UUID when available.
    """
    import uuid
    if _HAS_PG:
        return mapped_column(PGUUID(as_uuid=False), primary_key=pk, default=lambda: uuid.uuid4().hex)
    return mapped_column(String(36), primary_key=pk, default=lambda: uuid.uuid4().hex)  # 32/36 ok


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False,
        index=True,
    )


class TenantMixin:
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False, default="t0")


# ---------------------------
# Tables
# ---------------------------

class Tenant(Base, TimestampMixin):
    """
    A workspace/organization boundary.

    id   : short stable string used in headers/claims (e.g., "t0", "acme").
    name : display name.
    """
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)

    users: Mapped[List["User"]] = relationship(back_populates="tenant", cascade="all,delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tenant id={self.id!r} name={self.name!r}>"


class User(Base, TenantMixin, TimestampMixin):
    """
    App user with denormalized roles/scopes for clarity.

    id       : UUID
    email    : unique per tenant
    roles    : CSV-like string or JSON (choose JSONB in PG); e.g., ["admin","user"]
    is_active: soft-disable flag

    In production you might normalize roles and add SSO identities; we keep it simple here.
    """
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_user_email_per_tenant"),
    )

    id: Mapped[str] = _uuid_col(pk=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)  # RFC 5321 upper bound ~254; we allow 320
    roles: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, doc='Comma-separated roles (e.g., "admin,user")')
    scopes_json: Mapped[Optional[dict]] = mapped_column(JSONB if JSONB else JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    tenant: Mapped["Tenant"] = relationship(back_populates="users")

    def roles_list(self) -> List[str]:
        return [r.strip() for r in (self.roles or "").split(",") if r.strip()]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id[:8]} email={self.email!r} tenant={self.tenant_id!r} roles={self.roles!r}>"


class Document(Base, TenantMixin, TimestampMixin):
    """
    An ingested source document.

    path        : local or remote path used during ingestion (for lineage).
    source_url  : original URL, if downloaded.
    mime_type   : best-effort MIME.
    checksum    : sha256 of original bytes (for dedupe).
    size_bytes  : size of original file (if known).
    meta        : arbitrary metadata (e.g., tags, retention labels).
    """
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_tenant_created", "tenant_id", "created_at"),
        Index("ix_documents_checksum", "checksum"),
    )

    id: Mapped[str] = _uuid_col(pk=True)
    path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB if JSONB else JSON, nullable=True)

    chunks: Mapped[List["Chunk"]] = relationship(back_populates="document", cascade="all,delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Document id={self.id[:8]} tenant={self.tenant_id!r} path={self.path!r}>"


class Chunk(Base, TenantMixin, TimestampMixin):
    """
    A retrievable chunk of text linked to a Document.

    text          : the chunk body used for retrieval (kept small).
    position      : order within the document (e.g., page/chunk index).
    document_id   : FK to Document.
    embedding_id  : optional reference into the vector store (chroma id, qdrant point id, etc.)
    meta          : small metadata blob (page, table id, section title).
    """
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_doc", "document_id"),
        Index("ix_chunks_tenant_doc", "tenant_id", "document_id"),
    )

    id: Mapped[str] = _uuid_col(pk=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB if JSONB else JSON, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover
        snippet = (self.text[:30] + "…") if len(self.text) > 30 else self.text
        return f"<Chunk id={self.id[:8]} doc={self.document_id[:8]} pos={self.position} text={snippet!r}>"


class ChatSession(Base, TenantMixin, TimestampMixin):
    """
    A chat thread for continuity/history.

    title    : optional title shown in UI.
    user_id  : the owner/creator (FK users.id).
    """
    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_chat_sessions_user", "tenant_id", "user_id"),)

    id: Mapped[str] = _uuid_col(pk=True)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    user: Mapped[Optional["User"]] = relationship()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ChatSession id={self.id[:8]} tenant={self.tenant_id!r} title={self.title!r}>"


class AuditEvent(Base):
    """
    Append-only audit log.

    id        : UUID primary key.
    ts        : event timestamp (UTC).
    actor     : who performed the action (user id, key id, or 'system/worker').
    action    : verb-like string, e.g., "ingest.start", "api.chat".
    tenant_id : optional tenant association (useful for multi-tenant filtering).
    details   : JSON with structured metadata (never store secrets).

    Indexes:
    - by timestamp (descending queries)
    - by tenant_id for filtering
    """
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_ts", "ts"),
        Index("ix_audit_tenant", "tenant_id"),
    )

    id: Mapped[str] = _uuid_col(pk=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    actor: Mapped[str] = mapped_column(String(200), nullable=False)
    action: Mapped[str] = mapped_column(String(200), nullable=False)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSONB if JSONB else JSON, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AuditEvent id={self.id[:8]} action={self.action!r} actor={self.actor!r}>"
