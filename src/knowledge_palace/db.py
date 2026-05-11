"""Database models and connection management."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Index,
    text,
    func,
    Boolean,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    author: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)  # "calibre", "file", "manual"
    file_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    calibre_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSONB, default=list)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document {self.title!r}>"


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(Vector(768), nullable=True)  # pgvector
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default=dict)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_embedding", "embedding", postgresql_using="hnsw",
              postgresql_with={"m": 16, "ef_construction": 64},
              postgresql_ops={"embedding": "vector_cosine_ops"}),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.id[:8]}... (doc={self.document_id[:8]}... idx={self.chunk_index})>"


class IngestionLog(Base):
    __tablename__ = "ingestion_log"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # "success", "error", "partial"
    items_processed: Mapped[int] = mapped_column(Integer, default=0)
    items_failed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# --- Database lifecycle ---

_engine = None
_session_factory = None


async def init_db(database_url: str) -> None:
    """Initialize async engine and session factory."""
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=False, pool_size=20, max_overflow=10)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Get a database session (async generator for FastAPI-style dependency injection)."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        yield session


class _SessionContext:
    """Synchronous context manager wrapping an async session for use with `async with`."""

    def __init__(self):
        if _session_factory is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        self._session = _session_factory()
        await self._session.__aenter__()
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session is not None:
            return await self._session.__aexit__(exc_type, exc_val, exc_tb)


def session_context():
    """Return an async context manager that yields a single DB session.

    Usage::

        async with session_context() as session:
            session.add(...)
            await session.commit()
    """
    return _SessionContext()


async def create_tables() -> None:
    """Create all tables (use migrations in production).

    Handles the case where the pgvector extension or tables already exist
    gracefully.  If the database user lacks CREATE EXTENSION permissions,
    the extension is assumed to be pre-installed by a superuser.
    """
    if _engine is None:
        raise RuntimeError("Database not initialized.")
    async with _engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            # The user may not have superuser privileges to create extensions.
            # In that case assume the extension is already available.
            pass
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the engine."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
    _engine = None
    _session_factory = None
