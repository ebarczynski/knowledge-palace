"""Extractors on real files: text, EPUB and PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import TEXT, write_epub, write_pdf
from knowledge_palace.ingest.extractors import ExtractedDocument, extract_file, get_extractors


def test_text_file_gets_its_title_from_the_first_markdown_heading(tmp_path: Path):
    path = tmp_path / "note.md"
    path.write_text("# Real Title\n\nbody\n", encoding="utf-8")
    doc = extract_file(path, metadata={"source": "file"})
    assert doc.title == "Real Title" and doc.source == "file" and doc.file_path == str(path)
    assert len(doc.content_hash) == 64 and doc.content == "# Real Title\n\nbody\n"


def test_text_file_title_falls_back_to_the_file_name(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("just text\n")
    assert extract_file(path).title == "note"


def test_null_bytes_are_stripped_everywhere(tmp_path: Path):
    # PostgreSQL rejects 0x00 in text; extraction of real books has produced it.
    path = tmp_path / "nul.md"
    path.write_bytes(b"# Ti\x00tle\n\x00body\x00\n")
    doc = extract_file(path, metadata={"source": "file"})
    assert "\x00" not in doc.title + doc.content and doc.title == "Title"


def test_unreadable_bytes_do_not_crash(tmp_path: Path):
    path = tmp_path / "latin.txt"
    path.write_bytes(("caf\xe9 " + TEXT["db"]).encode("latin-1"))
    doc = extract_file(path)
    assert "caf\ufffd" in doc.content, "invalid UTF-8 is replaced, never silently dropped"


def test_empty_file_extracts_to_empty_content(tmp_path: Path):
    path = tmp_path / "empty.md"
    path.write_bytes(b"")
    assert extract_file(path).content == ""


def test_unsupported_extension_is_rejected(tmp_path: Path):
    path = tmp_path / "book.xyz"
    path.write_bytes(b"whatever")
    with pytest.raises(ValueError, match="No extractor found"):
        extract_file(path)


def test_a_directory_named_like_a_note_fails_loudly(tmp_path: Path):
    # The crawler's glob matches directories too; this used to be counted as an
    # anonymous "error" with no file name.
    (tmp_path / "notes.md").mkdir()
    with pytest.raises(IsADirectoryError):
        extract_file(tmp_path / "notes.md")


def test_broken_symlink_fails_loudly(tmp_path: Path):
    (tmp_path / "dangling.md").symlink_to(tmp_path / "missing.md")
    with pytest.raises(FileNotFoundError):
        extract_file(tmp_path / "dangling.md")


def test_epub_is_extracted_with_metadata(tmp_path: Path):
    path = tmp_path / "book.epub"
    write_epub(path, "EPUB Title", TEXT["rust"])
    doc = extract_file(path, metadata={"title": "EPUB Title", "author": "Author, Ada", "source": "calibre"})
    assert TEXT["rust"].strip() in doc.content
    assert doc.title == "EPUB Title" and doc.author == "Author, Ada" and doc.source == "calibre"
    assert doc.metadata["format"] == "epub"
    assert "Chapter 3" in doc.content


def test_epub_title_falls_back_to_the_embedded_metadata(tmp_path: Path):
    path = tmp_path / "book.epub"
    write_epub(path, "Embedded Title", TEXT["rust"])
    doc = extract_file(path)
    assert doc.title == "Embedded Title"


def test_corrupt_epub_raises_instead_of_returning_garbage(tmp_path: Path):
    path = tmp_path / "book.epub"
    path.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(Exception):  # noqa: B017, PT011 - the point is that it raises
        extract_file(path)


def test_pdf_is_extracted_with_page_count(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    write_pdf(path, "PDF Title", TEXT["db"])
    doc = extract_file(path)
    assert doc.title == "PDF Title" and doc.author == "PDF Meta Author"
    assert doc.metadata["page_count"] == 6 and doc.metadata["format"] == "pdf"
    assert "B-tree indexes" in doc.content


def test_corrupt_pdf_raises(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.7 truncated")
    with pytest.raises(Exception):  # noqa: B017, PT011
        extract_file(path)


def test_extractor_order_prefers_structured_formats():
    names = [type(e).__name__ for e in get_extractors()]
    assert names == ["EpubExtractor", "MobiExtractor", "PdfExtractor", "TextExtractor"]


def test_extension_matching_is_case_insensitive(tmp_path: Path):
    path = tmp_path / "NOTE.MD"
    path.write_text("# Upper\nbody")
    assert extract_file(path).title == "Upper"


def test_extracted_document_sanitises_on_construction():
    doc = ExtractedDocument("ti\x00tle", "au\x00thor", "bo\x00dy", "/p", "h" * 64, {}, "file")
    assert (doc.title, doc.author, doc.content) == ("title", "author", "body")
