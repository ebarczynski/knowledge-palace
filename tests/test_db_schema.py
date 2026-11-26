"""Schema convergence: however a database was created, one startup brings it to
the one schema the queries need. Runs against a real PostgreSQL + pgvector."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from knowledge_palace.db import create_tables, session_context
from knowledge_palace.search.engine import SearchEngine

pytestmark = pytest.mark.db

CANONICAL_INDEXES = [
    "chunks_pkey",
    "idx_chunks_content_tsvector",
    "ix_chunks_document_id",
    "ix_chunks_embedding_hnsw",
]
MIGRATIONS = ["001_initial.sql", "002_tsvector_column.sql", "003_drop_redundant_indexes.sql"]

# What SQLAlchemy's create_all() produced before the fix: no tsvector column, a
# differently named HNSW index, and timestamps without a time zone.
OLD_ORM_SCHEMA = """
CREATE TABLE documents (id CHAR(32) PRIMARY KEY, title VARCHAR(1024) NOT NULL, author VARCHAR(512),
  source VARCHAR(64) NOT NULL, file_path TEXT, calibre_id INTEGER, content_hash VARCHAR(64) NOT NULL UNIQUE,
  description TEXT, tags JSONB, metadata JSONB,
  created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now());
CREATE TABLE chunks (id CHAR(32) PRIMARY KEY,
  document_id CHAR(32) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  content TEXT NOT NULL, chunk_index INTEGER NOT NULL, embedding vector(768), token_count INTEGER, metadata JSONB);
CREATE INDEX ix_chunks_document_id ON chunks(document_id);
CREATE INDEX ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
"""


async def state() -> dict:
    async with session_context() as session:
        indexes = (await session.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename='chunks' ORDER BY 1"))).scalars().all()
        tsvector = (await session.execute(text(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name='chunks' AND column_name='content_tsvector'"))).scalar()
        applied = (await session.execute(text("SELECT name FROM schema_migrations ORDER BY 1"))).scalars().all()
        triggers = (await session.execute(text(
            "SELECT count(*) FROM pg_trigger WHERE tgname='documents_updated_at'"))).scalar()
        stamp_type = (await session.execute(text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='documents' AND column_name='created_at'"))).scalar()
        return {"indexes": list(indexes), "tsvector": tsvector, "migrations": list(applied),
                "trigger": triggers, "created_at": stamp_type}


async def drop_schema() -> None:
    async with session_context() as session:
        await session.execute(text("DROP SCHEMA public CASCADE"))
        await session.execute(text("CREATE SCHEMA public"))
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await session.commit()


async def test_a_fresh_database_gets_the_whole_schema_from_create_tables(db):
    await create_tables()
    got = await state()
    assert got["indexes"] == CANONICAL_INDEXES
    assert got["tsvector"] == 1 and got["trigger"] == 1
    assert got["migrations"] == MIGRATIONS
    assert got["created_at"] == "timestamp with time zone", "as in migrations/001_initial.sql"


async def test_keyword_search_works_without_any_manual_psql_step(db):
    # At the audited revision create_tables() alone left no content_tsvector, and
    # keyword and hybrid search failed with 'column "c.content_tsvector" does not exist'.
    await create_tables()
    engine = SearchEngine(db)
    assert (await engine.search("anything at all", mode="keyword")).total == 0


async def test_a_second_startup_changes_nothing(db):
    await create_tables()
    before = await state()
    await create_tables()
    assert await state() == before


async def test_a_hand_migrated_database_converges(db):
    """The live installation's shape: both SQL files applied by hand, no bookkeeping."""
    # leftovers of that history: 001's never-used expression index, and the
    # second HNSW index a later create_all() added under the ORM's old name
    async with session_context() as session:
        await session.execute(text(
            "CREATE INDEX ix_chunks_content_fts ON chunks USING GIN (to_tsvector('english', content))"))
        await session.execute(text(
            "CREATE INDEX ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"))
        await session.execute(text("DELETE FROM schema_migrations"))
        await session.commit()
    assert len((await state())["indexes"]) == 6

    await create_tables()

    got = await state()
    assert got["indexes"] == CANONICAL_INDEXES, "dead expression index and duplicate HNSW index removed"
    assert got["migrations"] == MIGRATIONS and got["trigger"] == 1


async def test_a_database_created_by_the_old_create_all_converges_and_keeps_its_rows(db):
    await drop_schema()
    async with session_context() as session:
        for statement in OLD_ORM_SCHEMA.strip().split(";\n"):
            if statement.strip():
                await session.execute(text(statement))
        await session.execute(text(
            "INSERT INTO documents (id, title, source, content_hash) "
            "VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'Old note', 'file', 'h1')"))
        await session.execute(text(
            "INSERT INTO chunks (id, document_id, content, chunk_index) VALUES "
            "('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', "
            "'tokio executor work stealing scheduler', 0)"))
        await session.commit()

    await create_tables()

    got = await state()
    assert got["indexes"] == CANONICAL_INDEXES
    assert got["tsvector"] == 1
    async with session_context() as session:
        kept = (await session.execute(text("SELECT count(*) FROM chunks"))).scalar()
        filled = (await session.execute(text(
            "SELECT content_tsvector IS NOT NULL FROM chunks WHERE id = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'"))).scalar()
    assert kept == 1 and filled, "the pre-existing chunk is kept and its tsvector is computed"
    results = await SearchEngine(db).search("tokio work stealing", mode="keyword")
    assert [r.chunk_id for r in results.results] == ["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]


async def test_concurrent_startups_all_succeed(db):
    # `kp serve` and an MCP server process starting together used to race inside
    # create_all(); three of four processes died with "already exists".
    await drop_schema()
    results = await asyncio.gather(*[create_tables() for _ in range(6)], return_exceptions=True)
    assert [r for r in results if isinstance(r, BaseException)] == []
    assert (await state())["indexes"] == CANONICAL_INDEXES
