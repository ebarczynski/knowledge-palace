"""Ingestion pipeline against a real PostgreSQL and the real embedding model:
the index is reconciled with its sources instead of only ever growing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from conftest import TEXT, FakeCalibre, config_for_calibre, config_for_notes, write_note
from knowledge_palace.config import Config
from knowledge_palace.db import session_context
from knowledge_palace.ingest.pipeline import IngestionPipeline
from knowledge_palace.search.engine import SearchEngine

pytestmark = [pytest.mark.db, pytest.mark.embedding]


async def rows(sql: str) -> list[tuple]:
    async with session_context() as session:
        return [tuple(row) for row in (await session.execute(text(sql))).all()]


async def documents(source: str = "file") -> list[tuple]:
    """(title, file_path, calibre_id, chunk count) in a stable order."""
    return await rows(
        "SELECT d.title, coalesce(d.file_path, ''), coalesce(d.calibre_id, 0)::text, "
        "(SELECT count(*) FROM chunks c WHERE c.document_id = d.id) "
        f"FROM documents d WHERE d.source = '{source}' ORDER BY d.title"
    )


async def chunk_count() -> int:
    return (await rows("SELECT count(*) FROM chunks"))[0][0]


async def hits_for(db: Config, query: str) -> list[str]:
    return [r.chunk_id for r in (await SearchEngine(db).search(query, mode="keyword", limit=50)).results]


def seed_notes(notes_dir: Path) -> None:
    for name, key in (("a.md", "rust"), ("b.txt", "db"), ("c.org", "ml")):
        write_note(notes_dir, name, key)


# ------------------------------------------------------------------ notes

async def test_fresh_ingest_indexes_every_document_and_names_every_failure(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    (notes_dir / "dangling.md").symlink_to(notes_dir / "missing.md")
    (notes_dir / "dir.md").mkdir()
    pipeline = IngestionPipeline(config_for_notes(db, notes_dir))
    results = await pipeline.run("files")

    assert (results["extracted"], results["errors"], results["skipped"]) == (3, 2, 0)
    assert results["chunked"] == await chunk_count() == results["embedded"]
    failed = " ".join(f"{name}: {reason}" for name, reason in pipeline._failures)
    assert "dangling.md" in failed and "No such file" in failed
    assert "dir.md" in failed and "Is a directory" in failed, (
        "the audited revision reported only 'Errors: 2', with no file names or reasons"
    )
    log = (await rows("SELECT status, items_processed, items_failed FROM ingestion_log"))[-1]
    assert log == ("partial", 3, 2)


async def test_unchanged_sources_are_skipped_before_any_embedding(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    first = IngestionPipeline(config_for_notes(db, notes_dir))
    await first.run("files")
    embed_calls.clear()

    # A fresh pipeline, like a new `kp ingest` process: an unchanged library
    # must not need the model at all.
    second = IngestionPipeline(config_for_notes(db, notes_dir))
    results = await second.run("files")

    assert (results["extracted"], results["skipped"], results["errors"]) == (0, 3, 0)
    assert embed_calls == [], "nothing new, so nothing is embedded"
    assert second.embedding_service.backend is None, "the model is not even loaded"


async def test_reindex_rebuilds_existing_documents_with_the_new_chunking(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    config = config_for_notes(db, notes_dir)
    pipeline = IngestionPipeline(config)
    await pipeline.run("files")
    titles = [d[0] for d in await documents()]
    embed_calls.clear()

    config.chunking = replace(config.chunking, max_tokens=150)
    # `kp ingest --reindex`: at the audited revision this changed nothing.
    rebuilt = await pipeline.run("files", reindex=True)

    assert rebuilt["extracted"] == 3 and rebuilt["errors"] == 0
    assert embed_calls, "the documents were embedded again"
    assert (await rows("SELECT max(token_count) FROM chunks"))[0][0] <= 150
    assert [d[0] for d in await documents()] == titles, "same documents, no duplicates"


async def test_a_changed_file_replaces_its_document(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    pipeline = IngestionPipeline(config_for_notes(db, notes_dir))
    await pipeline.run("files")
    old_id = (await rows("SELECT id FROM documents WHERE file_path LIKE '%a.md'"))[0][0]
    await hits_for(db, "work-stealing")

    (notes_dir / "a.md").write_text("# Kafka notes\n\n" + TEXT["kafka"] * 40)
    results = await pipeline.run("files")

    assert results["extracted"] == 1 and results["skipped"] == 2
    at_path = await rows("SELECT id FROM documents WHERE file_path LIKE '%a.md'")
    assert len(at_path) == 1, "one document per path (the audited revision kept both)"
    assert at_path[0][0] != old_id and (await documents())[0][0] == "Kafka notes"
    assert await hits_for(db, "work-stealing") == [], "the text that is no longer on disk is gone"


async def test_a_deleted_file_leaves_the_index(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    pipeline = IngestionPipeline(config_for_notes(db, notes_dir))
    await pipeline.run("files")
    assert await hits_for(db, "B-tree indexes")

    (notes_dir / "b.txt").unlink()
    results = await pipeline.run("files")

    assert [d[0] for d in await documents()] == ["ml notes", "rust notes"]
    assert await hits_for(db, "B-tree indexes") == [], "a deleted note must not stay searchable"
    assert results["extracted"] == 0 and results["errors"] == 0


async def test_a_renamed_file_is_repointed_not_reembedded(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    pipeline = IngestionPipeline(config_for_notes(db, notes_dir))
    await pipeline.run("files")
    before = (await rows("SELECT id FROM documents WHERE file_path LIKE '%c.org'"))[0][0]
    chunks_before = await chunk_count()
    embed_calls.clear()

    (notes_dir / "c.org").rename(notes_dir / "c_renamed.org")
    results = await pipeline.run("files")

    after = await rows(f"SELECT id, file_path FROM documents WHERE id = '{before}'")
    assert len(after) == 1 and after[0][1].endswith("c_renamed.org"), "same row, new path"
    assert await chunk_count() == chunks_before and embed_calls == []
    assert results["extracted"] == 0 and results["skipped"] == 3


async def test_identical_content_is_indexed_once(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    (notes_dir / "copy_of_b.txt").write_text((notes_dir / "b.txt").read_text())
    pipeline = IngestionPipeline(config_for_notes(db, notes_dir))
    results = await pipeline.run("files")

    assert results["extracted"] == 3 and results["skipped"] == 1
    assert await chunk_count() == results["chunked"]


async def test_an_emptied_file_drops_its_document(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    pipeline = IngestionPipeline(config_for_notes(db, notes_dir))
    await pipeline.run("files")
    (notes_dir / "a.md").write_text("")
    await pipeline.run("files")
    assert [d[0] for d in await documents()] == ["db notes", "ml notes"]
    assert await hits_for(db, "work-stealing") == []


async def test_a_missing_root_prunes_nothing(db, notes_dir: Path, tmp_path: Path, embed_calls):
    # An unmounted drive must not look like "every file was deleted".
    seed_notes(notes_dir)
    config = config_for_notes(db, notes_dir)
    pipeline = IngestionPipeline(config)
    await pipeline.run("files")
    drive = tmp_path / "drive"
    drive.mkdir()
    write_note(drive, "on-drive.md", "graph")
    config.sources.paths.append(str(drive))
    await pipeline.run("files")
    assert len(await documents()) == 4

    import shutil

    shutil.rmtree(drive)  # the drive is unplugged
    results = await pipeline.run("files")

    assert results["errors"] == 0
    assert len(await documents()) == 4, "documents under the missing root are left alone"
    assert "graph notes" in {d[0] for d in await documents()}


async def test_legacy_duplicates_on_one_path_are_healed(db, notes_dir: Path, embed_calls):
    seed_notes(notes_dir)
    pipeline = IngestionPipeline(config_for_notes(db, notes_dir))
    await pipeline.run("files")
    async with session_context() as session:
        await session.execute(text(
            "INSERT INTO documents (id, title, source, file_path, content_hash) VALUES "
            "('00000000-0000-0000-0000-000000000001', 'OLD a.md', 'file', :p, :h)"),
            {"p": str(notes_dir / "a.md"), "h": "0" * 64})
        await session.execute(text(
            "INSERT INTO chunks (id, document_id, content, chunk_index, metadata) VALUES "
            "(gen_random_uuid(), '00000000-0000-0000-0000-000000000001', 'STALE CHUNK zzzz', 0, '{}')"))
        await session.commit()
    assert len(await documents()) == 4

    (notes_dir / "a.md").write_text("# Changed\n\n" + TEXT["kafka"] * 30)
    await pipeline.run("files")

    assert len(await documents()) == 3, "one document per path again"
    assert (await rows("SELECT count(*) FROM chunks WHERE content LIKE 'STALE CHUNK%'"))[0][0] == 0


# ------------------------------------------------------------------ Calibre

def seed_calibre(calibre: FakeCalibre) -> None:
    calibre.add_book(1, "Rust Book", "EPUB", TEXT["rust"])
    calibre.add_book(2, "Database Book", "PDF", TEXT["db"])
    calibre.add_book(3, "ML Book", "EPUB", TEXT["ml"])
    # Same file content as book 1, different catalog title: a book added twice
    # (extraction embeds the file's own title, so the files must match too).
    calibre.add_book(4, "Rust Book Duplicate", "EPUB", TEXT["rust"], file_title="Rust Book")


async def test_calibre_fresh_ingest_deduplicates_and_reports_missing_files(db, calibre: FakeCalibre, embed_calls):
    seed_calibre(calibre)
    calibre.add_book(5, "Ghost", "EPUB", TEXT["graph"])
    next((calibre.root / "Author, Ada").glob("*(5)/*.epub")).unlink()
    pipeline = IngestionPipeline(config_for_calibre(db, calibre))
    results = await pipeline.run("calibre")

    assert (results["extracted"], results["skipped"], results["errors"]) == (3, 1, 1), pipeline._failures
    assert "Ghost" in " ".join(name for name, _ in pipeline._failures)
    assert {d[0] for d in await documents("calibre")} == {"Database Book", "ML Book", "Rust Book"}


async def test_calibre_unchanged_library_costs_no_embedding(db, calibre: FakeCalibre, embed_calls):
    seed_calibre(calibre)
    pipeline = IngestionPipeline(config_for_calibre(db, calibre))
    await pipeline.run("calibre")
    embed_calls.clear()

    results = await pipeline.run("calibre")

    # The audited revision re-extracted and re-embedded every book on every run
    # and only then discovered, at commit time, that it already had them.
    assert (results["extracted"], results["skipped"], results["errors"]) == (0, 4, 0)
    assert embed_calls == []


async def test_a_book_removed_from_calibre_is_removed_from_the_index(db, calibre: FakeCalibre, embed_calls):
    seed_calibre(calibre)
    pipeline = IngestionPipeline(config_for_calibre(db, calibre))
    await pipeline.run("calibre")
    calibre.remove_book(2)
    await pipeline.run("calibre")
    assert {d[0] for d in await documents("calibre")} == {"ML Book", "Rust Book"}
    assert await hits_for(db, "B-tree indexes") == []


async def test_a_book_read_under_a_new_calibre_id_keeps_its_document(db, calibre: FakeCalibre, embed_calls):
    # Seen on a real library: Calibre re-added the book under a new id while the
    # index kept the old id and a path that no longer exists.
    seed_calibre(calibre)
    pipeline = IngestionPipeline(config_for_calibre(db, calibre))
    await pipeline.run("calibre")
    before = (await rows("SELECT id FROM documents WHERE calibre_id = 2"))[0][0]
    chunks_before = await chunk_count()
    embed_calls.clear()

    calibre.readd_book(2, 902, "Database Book")
    results = await pipeline.run("calibre")

    after = await rows("SELECT id, calibre_id, file_path FROM documents WHERE calibre_id IN (2, 902)")
    assert len(after) == 1, "re-pointed, not duplicated and not pruned"
    assert after[0][0] == before and after[0][1] == 902
    assert Path(after[0][2]).exists(), "provenance points at a file that exists"
    assert await chunk_count() == chunks_before and embed_calls == []
    assert results["extracted"] == 0 and results["errors"] == 0


async def test_an_empty_or_unreadable_catalog_never_empties_the_index(
    db, calibre: FakeCalibre, embed_calls, monkeypatch: pytest.MonkeyPatch
):
    import functools
    import sqlite3

    from knowledge_palace.ingest import pipeline as pipeline_module
    from knowledge_palace.ingest.calibre import CalibreBridge

    # The default busy timeout is 30s; the test only needs to see the failure.
    monkeypatch.setattr(pipeline_module, "CalibreBridge", functools.partial(CalibreBridge, timeout=0.2))
    seed_calibre(calibre)
    pipeline = IngestionPipeline(config_for_calibre(db, calibre))
    await pipeline.run("calibre")
    before = len(await documents("calibre"))

    holder = sqlite3.connect(calibre.db_path, timeout=0.2)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        locked = await pipeline.run("calibre")
    finally:
        holder.rollback()
        holder.close()

    assert locked["errors"] == 1 and "locked" in " ".join(r for _, r in pipeline._failures).lower()
    assert len(await documents("calibre")) == before

    calibre.db_path.write_bytes(b"garbage" * 100)
    corrupt = await pipeline.run("calibre")
    assert corrupt["errors"] == 1 and "not a readable SQLite database" in " ".join(r for _, r in pipeline._failures)
    assert len(await documents("calibre")) == before


async def test_kp_ingest_exits_nonzero_and_names_what_failed(db, notes_dir: Path, tmp_path: Path):
    """The real CLI, in its own process, against the test database."""
    import asyncio
    import sys

    seed_notes(notes_dir)
    (notes_dir / "dir.md").mkdir()
    database = db.database
    config = tmp_path / "config.toml"
    config.write_text(
        f"[database]\nhost = \"{database.host}\"\nport = {database.port}\nname = \"{database.name}\"\n"
        f"user = \"{database.user}\"\npassword = \"{database.password}\"\n"
        f"[sources]\npaths = [\"{notes_dir}\"]\n[embedding]\nprovider = \"auto\"\ndevice = \"cpu\"\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "knowledge_palace.cli", "--config", str(config), "ingest", "--source", "files",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
    output = (stdout + stderr).decode()

    assert process.returncode == 1, output[-800:]
    assert "dir.md" in output and "Is a directory" in output, output[-800:]
    assert "Errors: 1" in output
    assert len(await documents()) == 3, "the three readable notes were indexed despite the failure"
