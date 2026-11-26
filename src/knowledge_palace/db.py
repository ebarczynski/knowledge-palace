"""Database models and connection management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .migrations import MIGRATION_LOCK_KEY, apply_migrations

# Schema constants shared with the SQL migrations (the vector column's width
# is fixed by them; changing the embedding model's dimension requires both).
EMBEDDING_DIMENSIONS = 768


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    author: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)  # "calibre", "file", "manual"
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    calibre_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list | None] = mapped_column(JSONB, default=list)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)
    # TIMESTAMPTZ, as in migrations/001_initial.sql.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(
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
    embedding = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=True)  # pgvector
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)

    document: Mapped[Document] = relationship(back_populates="chunks")

    # Same names and parameters as migrations/001_initial.sql, so a schema
    # created by create_all() and one created by the SQL files are identical.
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_embedding_hnsw", "embedding", postgresql_using="hnsw",
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
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# --- Database lifecycle ---

_engine = None
_session_factory = None


async def init_db(database_url: str) -> None:
    """Initialize async engine and session factory.

    Re-initializing (e.g. ``kp ingest`` after another command in the same
    process) disposes the previous engine first, so its pooled connections do
    not leak.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = create_async_engine(database_url, echo=False, pool_size=20, max_overflow=10)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
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
        if _session_factory is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        self._session = _session_factory()
        await self._session.__aenter__()
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session is not None:
            return await self._session.__aexit__(exc_type, exc_val, exc_tb)
        return False


def session_context():
    """Return an async context manager that yields a single DB session.

    Usage::

        async with session_context() as session:
            session.add(...)
            await session.commit()
    """
    return _SessionContext()


async def create_tables() -> None:
    """Create the base tables and bring the schema up to date.

    Idempotent and safe to call on every startup and from several processes at
    once (``kp serve`` and an MCP server starting together): the whole sequence
    runs in one transaction under an advisory lock, so late starters find the
    work done. It also converges: a database created by ``create_all`` alone,
    by hand-applied SQL migrations, or by a mix of both ends up with the same
    tables, columns and indexes.
    """
    if _engine is None:
        raise RuntimeError("Database not initialized.")
    await _ensure_pgvector()
    async with _engine.connect() as conn, conn.begin():
        await conn.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": MIGRATION_LOCK_KEY}
        )
        await conn.run_sync(Base.metadata.create_all)
        await apply_migrations(conn)


async def _ensure_pgvector() -> None:
    """Make sure the ``vector`` type is usable, and say why when it is not.

    Creating the extension needs superuser, so that step is best-effort: the
    extension may already be installed by an administrator. What is *not*
    optional is the type itself, so probe it instead of assuming.
    """
    if _engine is None:
        raise RuntimeError("Database not initialized.")
    create_error = ""
    try:
        await _run_own_connection(_engine, "CREATE EXTENSION IF NOT EXISTS vector", transactional=True)
    except DBAPIError as exc:
        create_error = _first_line(getattr(exc, "orig", exc))
    try:
        await _run_own_connection(_engine, "SELECT '[1,2,3]'::vector")
    except DBAPIError as exc:
        raise RuntimeError(
            "pgvector is not usable on this database: "
            f"{_first_line(getattr(exc, 'orig', exc))}"
            + (f" (CREATE EXTENSION vector failed: {create_error})" if create_error else "")
            + ". Install pgvector for this PostgreSQL major version, or have a "
            "superuser run CREATE EXTENSION vector in the database."
        ) from exc


async def _run_own_connection(engine: AsyncEngine, statement: str, *, transactional: bool = False) -> None:
    """Execute one statement on a connection of its own.

    A pooled connection can hold a prepared plan that a schema change has since
    invalidated. asyncpg reports that once as InvalidCachedStatementError (and
    SQLAlchemy then drops its caches), so it is retried rather than passed off
    as a missing extension.
    """
    for attempt in (1, 2):
        try:
            if transactional:
                async with engine.begin() as conn:
                    await conn.execute(text(statement))
            else:
                async with engine.connect() as conn:
                    await conn.execute(text(statement))
            return
        except DBAPIError as exc:
            stale_plan = type(getattr(exc, "orig", None)).__name__ == "InvalidCachedStatementError"
            if attempt == 2 or not stale_plan:
                raise


def _first_line(error: Exception) -> str:
    return str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__


async def close_db() -> None:
    """Dispose of the engine."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
    _engine = None
    _session_factory = None
