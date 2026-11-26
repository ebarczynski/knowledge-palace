"""Calibre bridge: format resolution, read-only access, and readable failures."""

from __future__ import annotations

import hashlib
import sqlite3
import threading

import pytest

from conftest import TEXT, FakeCalibre
from knowledge_palace.ingest.calibre import (
    FORMAT_PRIORITY,
    BookFileNotFoundError,
    CalibreBridge,
    CalibreLibraryError,
)

ADVERTISED = ["EPUB", "PDF", "MOBI", "AZW", "AZW3", "TXT"]  # README feature list


def add_and_locate(calibre: FakeCalibre, book_id: int, fmt: str) -> object:
    calibre.add_book(book_id, f"Book {fmt}", fmt, TEXT["rust"])
    bridge = CalibreBridge(calibre.root)
    book = next(b for b in bridge.get_books() if b["id"] == book_id)
    return bridge.get_book_file(book)


@pytest.mark.parametrize("fmt", ADVERTISED)
def test_every_advertised_format_is_found_on_disk(calibre: FakeCalibre, fmt: str):
    # AZW3 and AZW were missing from the priority list, so books that only had
    # them were skipped with "No file found" while the run reported 0 errors.
    path = add_and_locate(calibre, 100 + ADVERTISED.index(fmt), fmt)
    assert path is not None, f"{fmt} book not resolved"
    assert path.suffix.lower() == f".{fmt.lower()}"


def test_azw3_and_azw_are_extracted_like_mobi(calibre: FakeCalibre):
    """AZW3/AZW files are resolved and handed to the extractor that supports them."""
    from knowledge_palace.ingest.extractors import get_extractors

    path = add_and_locate(calibre, 7, "AZW3")
    assert any(extractor.can_handle(path) for extractor in get_extractors())


def test_format_preference_keeps_the_existing_order(calibre: FakeCalibre):
    # Changing the order would pick a different file for multi-format books,
    # which changes their content hash and forces a re-ingest of a library.
    assert FORMAT_PRIORITY[:4] == ["EPUB", "PDF", "TXT", "MOBI"]
    calibre.add_book(1, "Multi", "MOBI", TEXT["rust"])
    for fmt in ("PDF", "EPUB"):
        calibre.add_format(1, "Multi", fmt, TEXT["rust"])
    book = next(b for b in CalibreBridge(calibre.root).get_books() if b["id"] == 1)
    assert CalibreBridge(calibre.root).get_book_file(book).suffix == ".epub"


def test_books_without_a_file_raise_a_specific_error(calibre: FakeCalibre):
    calibre.add_book(3, "Ghost", "EPUB", TEXT["rust"])
    path = next((calibre.root / "Author, Ada").glob("*(3)/*.epub"))
    path.unlink()
    bridge = CalibreBridge(calibre.root)
    book = bridge.get_books()[0]
    with pytest.raises(BookFileNotFoundError, match="Ghost"):
        bridge.extract_book(book)


def test_extraction_errors_are_not_swallowed(calibre: FakeCalibre):
    calibre.add_book(4, "Broken", "EPUB", TEXT["rust"])
    next((calibre.root / "Author, Ada").glob("*(4)/*.epub")).write_bytes(b"this is not an epub")
    bridge = CalibreBridge(calibre.root)
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any extractor error; it must surface
        bridge.extract_book(bridge.get_books()[0])
    assert not isinstance(excinfo.value, type(None))


def test_reading_never_modifies_the_library_database(calibre: FakeCalibre):
    calibre.add_book(5, "Intact", "EPUB", TEXT["rust"])
    before = hashlib.sha256(calibre.db_path.read_bytes()).hexdigest()
    CalibreBridge(calibre.root).get_books()
    assert hashlib.sha256(calibre.db_path.read_bytes()).hexdigest() == before
    leftovers = [p.name for p in calibre.root.iterdir() if "-journal" in p.name or "-wal" in p.name]
    assert leftovers == [], f"reading left SQLite side files behind: {leftovers}"


def test_library_locked_by_calibre_is_a_clear_error(calibre: FakeCalibre):
    calibre.add_book(6, "Busy", "EPUB", TEXT["rust"])
    holder = sqlite3.connect(calibre.db_path, timeout=1.0)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        # Before the fix this surfaced as a raw sqlite3.OperationalError traceback.
        with pytest.raises(CalibreLibraryError, match="locked"):
            CalibreBridge(calibre.root, timeout=0.2).get_books()
    finally:
        holder.rollback()
        holder.close()
    assert CalibreBridge(calibre.root, timeout=1.0).get_books(), "readable again once Calibre lets go"


def test_corrupt_database_is_a_clear_error(calibre: FakeCalibre):
    calibre.db_path.write_bytes(b"\x00not a database" * 64)
    with pytest.raises(CalibreLibraryError, match="not a readable SQLite database"):
        CalibreBridge(calibre.root).get_books()


def test_missing_database_is_a_clear_error(tmp_path):
    with pytest.raises(CalibreLibraryError, match="metadata.db not found"):
        CalibreBridge(tmp_path / "nowhere")


def test_relative_and_symlinked_library_paths_open_read_only(calibre: FakeCalibre, tmp_path, monkeypatch):
    calibre.add_book(8, "Relative", "EPUB", TEXT["rust"])
    monkeypatch.chdir(tmp_path)
    assert len(CalibreBridge("calibre").get_books()) == 1
    link = tmp_path / "link-to-library"
    link.symlink_to(calibre.root)
    assert len(CalibreBridge(link).get_books()) == 1


def test_get_books_closes_its_connection(calibre: FakeCalibre):
    calibre.add_book(9, "Closed", "EPUB", TEXT["rust"])
    opened = []
    original = sqlite3.connect

    def tracking(*args, **kwargs):
        conn = original(*args, **kwargs)
        opened.append(conn)
        return conn

    sqlite3.connect = tracking
    try:
        CalibreBridge(calibre.root).get_books()
    finally:
        sqlite3.connect = original
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_connect_is_thread_safe_enough_for_the_ingest_executor(calibre: FakeCalibre):
    calibre.add_book(10, "Threaded", "EPUB", TEXT["rust"])
    calibre.add_book(11, "Threaded2", "EPUB", TEXT["db"])
    bridge = CalibreBridge(calibre.root)
    results, errors = [], []

    def read():
        try:
            results.append(len(bridge.get_books()))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and results == [2, 2, 2, 2]
