"""Search engine against a real PostgreSQL + pgvector; the marked tests also use
the real embedding model."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import text

from knowledge_palace.db import Chunk, Document, session_context
from knowledge_palace.search.engine import SearchEngine

pytestmark = pytest.mark.db

RUST = ("Async tasks in Rust are lazy futures that do nothing until polled by an executor. "
        "The Tokio runtime schedules tasks onto worker threads with a work-stealing scheduler. ")
DB = ("B-tree indexes keep their pages at least half full and split a node in two when it "
      "overflows, which keeps lookups logarithmic in the number of rows stored on disk. ")


async def insert(
    title: str, chunks: list[tuple[str, object]], *, author: str | None = None, tags: list | None = None,
) -> tuple[str, list[str]]:
    """Insert a document; return (document_id, chunk_ids)."""
    async with session_context() as session:
        doc = Document(
            title=title, author=author, source="file", tags=tags or [],
            content_hash=hashlib.sha256(title.encode()).hexdigest(),
        )
        session.add(doc)
        await session.flush()
        chunk_ids = []
        for index, (content, embedding) in enumerate(chunks):
            chunk = Chunk(document_id=doc.id, content=content, chunk_index=index, embedding=embedding)
            session.add(chunk)
            chunk_ids.append(chunk)
        await session.flush()
        returned = [str(chunk.id) for chunk in chunk_ids]
        await session.commit()
        return str(doc.id), returned


async def ids_of(results) -> list[str]:
    return [r.chunk_id for r in results.results]


async def test_tags_filter_matches_documents_by_tag(db):
    # Every tag filter used to fail with a SQL syntax error (":param::jsonb"),
    # and a tag holding a quote built invalid JSON.
    await insert("Rust book", [(RUST, None)], author="Author, Ada", tags=["Rust", 'C++ "systems"'])
    await insert("Python book", [("Async tasks in Rust " + RUST, None)], author="Author, Bob", tags=["Python"])
    engine = SearchEngine(db)
    only_rust = await engine.search("tokio work stealing", mode="keyword", tags=["Rust"])
    only_python = await engine.search("tokio work stealing", mode="keyword", tags=["Python"])
    assert only_rust.total == 1 and only_rust.results[0].title == "Rust book"
    assert only_python.total == 1 and only_python.results[0].title == "Python book"
    assert (await engine.search("tokio", mode="keyword", tags=['C++ "systems"'])).results[0].title == "Rust book"
    assert (await engine.search("tokio", mode="keyword", tags=["Rust", "Python"])).total == 2
    assert (await engine.search("tokio", mode="keyword", tags=["Go"])).total == 0
    assert (await engine.search("tokio", mode="keyword", tags=[])).total == 2, "an empty tag list means no filter"


async def test_author_and_source_filters(db):
    await insert("Rust book", [(RUST, None)], author="Author, Ada")
    await insert("Python book", [(RUST + RUST, None)], author="Author, Bob")
    engine = SearchEngine(db)
    assert {r.title for r in (await engine.search("tokio", mode="keyword", author="ada")).results} == {"Rust book"}
    whitespace = await engine.search("tokio", mode="keyword", author="   ")
    assert {r.title for r in whitespace.results} == {"Rust book", "Python book"}
    assert (await engine.search("tokio", mode="keyword", source_filter="calibre")).total == 0
    assert (await engine.search("tokio", mode="keyword", source_filter="file")).total == 2


async def test_deleting_a_document_removes_its_chunks(db):
    # The ingestion pipeline relies on this cascade when it replaces or prunes.
    from sqlalchemy import delete

    document_id, _ = await insert("Doomed", [(RUST, None), (DB, None)])
    async with session_context() as session:
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()
    async with session_context() as session:
        assert (await session.execute(text("SELECT count(*) FROM chunks"))).scalar() == 0


async def test_keyword_only_hits_are_deduplicated_across_the_overlap(db):
    await insert("Long note", [(RUST * 3, None), (RUST * 2, None)])
    results = await SearchEngine(db).search("tokio work stealing", mode="keyword", limit=10)
    assert len(set(await ids_of(results))) == results.total


@pytest.mark.embedding
async def test_hybrid_score_is_weighted_rrf_with_one_based_ranks(db):
    engine = SearchEngine(db)
    embedding = (await engine.embedding_service.embed_batch([RUST * 3]))[0]
    _, (chunk_id,) = await insert("Rust book", [(RUST * 3, embedding)])
    results = await engine.search(RUST.strip(), mode="hybrid", limit=5)
    assert chunk_id in await ids_of(results)
    top = results.results[0]
    assert top.score == pytest.approx(0.7 / 61 + 0.3 / 61), "rank 1 in both legs: w / (k + 1) with k = 60"
    assert (await engine.search(RUST.strip(), mode="semantic")).results[0].score > 0.5


@pytest.mark.embedding
async def test_fused_context_score_agrees_with_the_sql_convention(db):
    engine = SearchEngine(db)
    embedding = (await engine.embedding_service.embed_batch([RUST * 3]))[0]
    await insert("Rust book", [(RUST * 3, embedding)])
    fused = await engine.retrieve_context(RUST.strip(), limit=5, max_variants=1)
    assert fused.total == 1
    # One variant: keyword, semantic and hybrid lists, the original weighted 2x.
    # A hit ranked first in all three: 2.0 / 61 in each, or only in the lists that returned it.
    score = fused.results[0].score
    assert any(score == pytest.approx(value) for value in (2.0 / 61, 4.0 / 61, 6.0 / 61)), (
        f"one variant x three lists, weight 2x, rank 1 each: a multiple of 2/61, got {score}"
    )
    assert score != pytest.approx(0.1), "0-based ranks would give 0.1"


@pytest.mark.embedding
async def test_context_survives_a_chunk_that_has_no_embedding(db):
    # The schema allows NULL embeddings and the pipeline counts them; the
    # post-fusion gate used to raise TypeError on float(None).
    engine = SearchEngine(db)
    embedding = (await engine.embedding_service.embed_batch([RUST * 3]))[0]
    with_embedding, (with_emb_chunk, null_chunk) = await insert(
        "Rust book", [(RUST * 3, embedding), (RUST * 3 + " tokio executor work stealing " * 5, None)])
    fused = await engine.retrieve_context(RUST.strip(), limit=10)
    assert fused.total >= 1, "the real hit is still returned"
    assert all(r.chunk_id != null_chunk for r in fused.results), "the embedding-less chunk is dropped"
    assert with_emb_chunk in await ids_of(fused)
    assert {r.document_id for r in fused.results} == {with_embedding}


@pytest.mark.embedding
async def test_similarity_floor_zero_really_disables_the_gate(db):
    from dataclasses import replace

    engine = SearchEngine(db)
    query = "tokio executor work stealing"
    opposed = [-value for value in await engine.embedding_service.embed_query(query)]  # cosine = -1
    _, (chunk_id,) = await insert("Opposite", [("zzzz zzzz zzzz", opposed)])
    ungated = SearchEngine(replace(db, search=replace(db.search, min_similarity=0.0)))
    ungated.embedding_service = engine.embedding_service
    assert chunk_id in await ids_of(await ungated.search(query, mode="semantic", limit=10)), (
        "'0.0 disables the gate': with >= 0.0 kept in the SQL, a negative-cosine chunk was still hidden"
    )
    assert chunk_id not in await ids_of(await engine.search(query, mode="semantic", limit=10))


@pytest.mark.embedding
async def test_task_prefixes_are_applied(db):
    # Documents are embedded as "search_document: ...", queries as "search_query: ...".
    service = SearchEngine(db).embedding_service
    assert service._use_prefixes is True
    document = (await service.embed_batch(["tokio executor"]))[0]
    query = await service.embed_query("tokio executor")
    assert service.backend in {"onnx", "pytorch", "mlx"}
    assert len(document) == 768 and len(query) == 768
    assert sum(value * value for value in query) == pytest.approx(1.0, abs=1e-3), "vectors are L2-normalised"
