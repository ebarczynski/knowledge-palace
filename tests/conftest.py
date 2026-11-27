"""Shared fixtures.

Nothing in this suite is mocked. Integration tests talk to a real PostgreSQL
with pgvector and, where marked, to the real embedding model:

- ``KP_TEST_DATABASE_URL`` (a SQLAlchemy URL such as
  ``postgresql+asyncpg://kp:kp@127.0.0.1:5432/knowledge_palace_test``) enables
  tests marked ``db``. Point it at a scratch database: they truncate its tables.
- ``KP_TEST_EMBEDDING=1`` enables tests marked ``embedding`` (they load
  nomic-embed-text-v1.5 from the local Hugging Face cache, or download it).

Without those variables the corresponding tests are skipped, with the reason
shown in the report.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url

from knowledge_palace.config import Config, DatabaseConfig
from knowledge_palace.db import close_db, create_tables, init_db, session_context

DB_URL = os.environ.get("KP_TEST_DATABASE_URL")
EMBEDDING = os.environ.get("KP_TEST_EMBEDDING") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, enabled, reason in (
        ("db", DB_URL is not None, "needs PostgreSQL + pgvector; set KP_TEST_DATABASE_URL"),
        ("embedding", EMBEDDING, "needs the real embedding model; set KP_TEST_EMBEDDING=1"),
    ):
        if enabled:
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


# --------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------

def make_config(base: Config, **overrides) -> Config:
    """A copy of a known-good config (the test database!) with sections replaced.

    Every config used against the database must derive from the fixture's
    ``db`` config. Building one from ``Config()`` silently points at the
    default host:port -- whoever runs Postgres there.
    """
    from dataclasses import replace as dataclass_replace

    config = dataclass_replace(base)
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


@pytest.fixture(scope="session")
def db_url() -> str:
    assert DB_URL is not None, "KP_TEST_DATABASE_URL is not set"
    return DB_URL


@pytest.fixture
def db_config(db_url: str) -> Config:
    url = make_url(db_url)
    config = Config()
    config.database = DatabaseConfig(
        host=url.host or "localhost",
        port=url.port or 5432,
        name=url.database or "knowledge_palace",
        user=url.username or "kp",
        password=url.password or "",
    )
    config.embedding.device = "cpu"
    if config.database.name == "knowledge_palace":
        raise RuntimeError(
            "KP_TEST_DATABASE_URL points at a database named 'knowledge_palace', the "
            "production name; the suite truncates tables and must never run against it"
        )
    return config


@pytest_asyncio.fixture
async def db(db_config: Config) -> AsyncIterator[Config]:
    """A converged schema on empty tables, and a clean engine afterwards."""
    await init_db(db_config.database.url)
    await create_tables()
    async with session_context() as session:
        await session.execute(text("TRUNCATE documents, ingestion_log CASCADE"))
        await session.commit()
    yield db_config
    await close_db()


@pytest_asyncio.fixture
async def api_client(db_config: Config) -> AsyncIterator[httpx.AsyncClient]:
    """A client for the real FastAPI app, with its lifespan (model preload) running."""
    from knowledge_palace.api.app import create_app

    await init_db(db_config.database.url)
    await create_tables()
    app = create_app(db_config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client
    await close_db()


# --------------------------------------------------------------------------
# Embedding call counter (the real model still runs)
# --------------------------------------------------------------------------

@pytest.fixture
def embed_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the size of every ``embed_batch`` call made through the pipeline."""
    from knowledge_palace.ingest import pipeline

    calls: list[int] = []
    original = pipeline.EmbeddingService.embed_batch

    async def counting(self, texts):
        calls.append(len(texts))
        return await original(self, texts)

    monkeypatch.setattr(pipeline.EmbeddingService, "embed_batch", counting)
    return calls


# --------------------------------------------------------------------------
# A Calibre library on disk
# --------------------------------------------------------------------------

CALIBRE_SCHEMA = """
CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL DEFAULT 'Unknown', sort TEXT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, pubdate TIMESTAMP, series_index REAL NOT NULL DEFAULT 1.0,
  author_sort TEXT, isbn TEXT DEFAULT '', path TEXT NOT NULL DEFAULT '', flags INTEGER NOT NULL DEFAULT 1,
  uuid TEXT, has_cover BOOL DEFAULT 0, last_modified TIMESTAMP NOT NULL DEFAULT '2000-01-01 00:00:00+00:00');
CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL, sort TEXT, link TEXT NOT NULL DEFAULT '');
CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL, link TEXT NOT NULL DEFAULT '');
CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, author INTEGER NOT NULL);
CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, tag INTEGER NOT NULL);
CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, format TEXT NOT NULL,
  uncompressed_size INTEGER NOT NULL, name TEXT NOT NULL);
CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER NOT NULL, text TEXT NOT NULL DEFAULT '');
"""


def write_epub(path: Path, title: str, body: str) -> None:
    import ebooklib
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(f"urn:uuid:{title}")
    book.set_title(title)
    chapters = [
        epub.EpubHtml(title=f"c{i}", file_name=f"c{i}.xhtml", uid=f"c{i}",
                      content=f"<html><body><h1>Chapter {i}</h1><p>{body * 4}</p></body></html>")
        for i in range(6)
    ]
    for chapter in chapters:
        book.add_item(chapter)
    book.toc = tuple((c, (c.title, c.file_name)) for c in chapters)
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book, {"ignore_ncx": True})
    assert ebooklib.ITEM_DOCUMENT  # sanity: the library the extractor uses is importable


def write_pdf(path: Path, title: str, body: str) -> None:
    import fitz

    doc = fitz.open()
    for _ in range(6):
        page = doc.new_page()
        for line in range(40):
            page.insert_text((72, 72 + 14 * line), body.strip()[:95], fontsize=7)
    doc.set_metadata({"title": title, "author": "PDF Meta Author"})
    doc.save(str(path))
    doc.close()


class FakeCalibre:
    """A minimal Calibre library: the tables ``CalibreBridge`` reads, plus book files."""

    def __init__(self, root: Path):
        root.mkdir(parents=True)
        self.root = root
        self.db_path = root / "metadata.db"
        with self._connect() as con:
            con.executescript(CALIBRE_SCHEMA)
            con.execute("INSERT INTO authors(id, name, sort) VALUES (1, 'Author, Ada', 'Ada')")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success and is always closed."""
        con = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def add_book(self, book_id: int, title: str, fmt: str, body: str, file_title: str | None = None) -> Path:
        """Add a book. ``file_title`` is the title baked into the file itself.

        Different from the catalog ``title`` this models a book re-added under a
        new title with identical files -- two catalog entries, one content.
        """
        folder = self.root / "Author, Ada" / f"{title} ({book_id})"
        folder.mkdir(parents=True)
        path = folder / f"{(file_title or title).replace(' ', '_')}.{fmt.lower()}"
        if fmt == "EPUB":
            write_epub(path, file_title or title, body)
        elif fmt == "PDF":
            write_pdf(path, file_title or title, body)
        else:
            path.write_bytes(b"\x00" + title.encode())  # resolution tests only; not extracted
        with self._connect() as con:
            con.execute("INSERT INTO books(id, title, uuid) VALUES (?, ?, ?)", (book_id, title, f"uuid-{book_id}"))
            con.execute("INSERT INTO books_authors_link(book, author) VALUES (?, 1)", (book_id,))
            con.execute("INSERT INTO comments(book, text) VALUES (?, ?)", (book_id, f"<p>{title}</p>"))
            con.execute("INSERT INTO data(book, format, uncompressed_size, name) VALUES (?, ?, ?, ?)",
                        (book_id, fmt, 1234, title.replace(" ", "_")))
        return path

    def add_format(self, book_id: int, title: str, fmt: str, body: str) -> Path:
        """Give an existing book one more format (Calibre keeps them in the same folder)."""
        folder = next((self.root / "Author, Ada").glob(f"*({book_id})"))
        path = folder / f"{title.replace(' ', '_')}.{fmt.lower()}"
        if fmt == "EPUB":
            write_epub(path, title, body)
        elif fmt == "PDF":
            write_pdf(path, title, body)
        else:
            path.write_bytes(b"\x00" + title.encode())
        with self._connect() as con:
            con.execute("INSERT INTO data(book, format, uncompressed_size, name) VALUES (?, ?, ?, ?)",
                        (book_id, fmt, 1234, title.replace(" ", "_")))
        return path

    def remove_book(self, book_id: int) -> None:
        import shutil

        with self._connect() as con:
            for table in ("data", "books_authors_link", "books_tags_link", "comments"):
                con.execute(f"DELETE FROM {table} WHERE book = ?", (book_id,))
            con.execute("DELETE FROM books WHERE id = ?", (book_id,))
        for folder in (self.root / "Author, Ada").glob(f"*({book_id})"):
            shutil.rmtree(folder)

    def readd_book(self, old_id: int, new_id: int, title: str | None = None) -> None:
        """What Calibre does when a book is removed and added again: a new id and folder."""

        folder = next((self.root / "Author, Ada").glob(f"*({old_id})"))
        new_title = title or folder.name.rsplit(" (", 1)[0]
        folder.rename(folder.with_name(f"{new_title} ({new_id})"))
        with self._connect() as con:
            for table in ("data", "books_authors_link", "books_tags_link", "comments"):
                con.execute(f"UPDATE {table} SET book = ? WHERE book = ?", (new_id, old_id))
            con.execute("UPDATE books SET id = ?, title = ? WHERE id = ?", (new_id, new_title, old_id))

    @property
    def book_ids(self) -> set[int]:
        with self._connect() as con:
            return {row[0] for row in con.execute("SELECT id FROM books")}


@pytest.fixture
def calibre(tmp_path: Path) -> FakeCalibre:
    return FakeCalibre(tmp_path / "calibre")


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------

TEXT = {
    "rust": ("Async tasks in Rust are lazy futures that do nothing until polled by an executor. "
             "The Tokio runtime schedules tasks onto worker threads with a work-stealing scheduler. "),
    "db": ("B-tree indexes keep their pages at least half full and split a node in two when it "
           "overflows, which keeps lookups logarithmic in the number of rows stored on disk. "),
    "ml": ("LoRA finetuning freezes the base model weights and trains two low-rank matrices whose "
           "product approximates the full weight update at a fraction of the memory cost. "),
    "graph": ("Dijkstra's algorithm relaxes edges in order of increasing tentative distance and "
              "therefore requires every edge weight to be non-negative. "),
    "kafka": ("Kafka partitions are append-only logs and consumers track their position with offsets. "),
}


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    path = tmp_path / "notes"
    path.mkdir()
    return path


def write_note(root: Path, name: str, key: str, repetitions: int = 30) -> Path:
    path = root / name
    path.write_text(f"# {key} notes\n\n" + TEXT[key] * repetitions)
    return path


def config_for_notes(base: Config, notes_dir: Path) -> Config:
    from dataclasses import replace

    return make_config(base, sources=replace(base.sources, paths=[str(notes_dir)]))


def config_for_calibre(base: Config, calibre: FakeCalibre) -> Config:
    from dataclasses import replace

    return make_config(base, calibre=replace(base.calibre, library_path=str(calibre.root)))
