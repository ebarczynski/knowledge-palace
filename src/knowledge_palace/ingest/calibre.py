"""Calibre library ingestion."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Generator

from rich.console import Console
from rich.table import Table

from .extractors import extract_file, ExtractedDocument

console = Console()


class CalibreBridge:
    """Reads Calibre's metadata.db and extracts book content."""

    def __init__(self, library_path: str | Path):
        self.library_path = Path(library_path)
        self.db_path = self.library_path / "metadata.db"

        if not self.db_path.exists():
            raise FileNotFoundError(f"Calibre metadata.db not found at {self.db_path}")

        # Build a lookup: book_id -> actual filesystem path (discovered from disk)
        self._book_path_cache: dict[int, Path] | None = None

    def _build_path_cache(self) -> dict[int, Path]:
        """Scan the library directory to discover book files by ID.

        Calibre stores files as: Author/Title (ID)/file.format
        We extract the ID from the directory name pattern '(123)'.
        """
        cache: dict[int, Path] = {}
        import re

        id_pattern = re.compile(r"\((\d+)\)\s*$")

        for item in self.library_path.rglob("*"):
            if not item.is_dir():
                continue
            match = id_pattern.search(item.name)
            if match:
                book_id = int(match.group(1))
                cache[book_id] = item

        return cache

    def _get_book_dir(self, book_id: int) -> Path | None:
        """Find the directory for a book by ID, using the filesystem cache."""
        if self._book_path_cache is None:
            self._book_path_cache = self._build_path_cache()
        return self._book_path_cache.get(book_id)

    def get_books(self) -> list[dict]:
        """Read all books from Calibre metadata.db."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("""
            SELECT
                b.id,
                b.title,
                b.sort AS sort_title,
                b.pubdate,
                b.last_modified,
                b.uuid,
                GROUP_CONCAT(DISTINCT a.name) AS authors,
                GROUP_CONCAT(DISTINCT t.name) AS tags,
                GROUP_CONCAT(DISTINCT d.format) AS formats,
                GROUP_CONCAT(DISTINCT d.name) AS data_names,
                c.text AS comment
            FROM books b
            LEFT JOIN books_authors_link bal ON b.id = bal.book
            LEFT JOIN authors a ON bal.author = a.id
            LEFT JOIN books_tags_link btl ON b.id = btl.book
            LEFT JOIN tags t ON btl.tag = t.id
            LEFT JOIN data d ON b.id = d.book
            LEFT JOIN comments c ON b.id = c.book
            GROUP BY b.id
            ORDER BY b.title
        """)

        books = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return books

    def get_book_file(self, book: dict, preferred_format: str = "EPUB") -> Path | None:
        """Find the actual file path for a book.

        Strategy:
        1. Look up the book's directory by ID from the filesystem cache
        2. Search inside that directory for the preferred format
        3. Fallback to other formats
        4. Last resort: search the entire library for loose files matching the title
        """
        book_id = book.get("id", 0)
        formats_str = book.get("formats") or ""
        available_formats = [f.strip().upper() for f in formats_str.split(",") if f.strip()]

        # Format priority: preferred first, then standard order
        format_priority = [preferred_format]
        for fmt in ["EPUB", "PDF", "TXT", "MOBI"]:
            if fmt not in format_priority:
                format_priority.append(fmt)

        # Strategy 1: Use the path cache to find the book directory
        book_dir = self._get_book_dir(book_id)
        if book_dir and book_dir.is_dir():
            for fmt in format_priority:
                if fmt not in available_formats:
                    continue
                # Look for the file in the book directory
                for f in book_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == f".{fmt.lower()}":
                        return f

        # Strategy 2: Search for loose files at the top level
        # (some Calibre exports put files directly in the library root)
        title = book.get("title", "")
        for fmt in format_priority:
            if fmt not in available_formats:
                continue
            # Check top-level files
            ext = f".{fmt.lower()}"
            for f in self.library_path.iterdir():
                if f.is_file() and f.suffix.lower() == ext:
                    # Rough match: check if title appears in filename
                    if title and title.lower().replace(" ", "")[:20] in f.name.lower().replace(" ", ""):
                        return f

        # Strategy 3: Glob search in author directories
        authors = (book.get("authors") or "Unknown").split(",")[0].strip()
        for fmt in format_priority:
            if fmt not in available_formats:
                continue
            ext = f".{fmt.lower()}"
            # Look in author-name directories
            for f in self.library_path.rglob(f"*{ext}"):
                # Check if book_id in the parent dir name
                parent = f.parent.name
                if f"({book_id})" in parent:
                    return f

        return None

    def extract_book(self, book: dict) -> ExtractedDocument | None:
        """Extract text content from a Calibre book."""
        file_path = self.get_book_file(book)
        if file_path is None:
            console.print(f"[yellow]No file found for: {book['title']}[/yellow]")
            return None

        metadata = {
            "title": book.get("title"),
            "author": book.get("authors"),
            "tags": (book.get("tags") or "").split(",") if book.get("tags") else [],
            "calibre_id": book.get("id"),
            "description": book.get("comment"),
            "source": "calibre",
        }

        try:
            return extract_file(file_path, metadata)
        except Exception as e:
            console.print(f"[red]Error extracting {book['title']}: {e}[/red]")
            return None

    def list_books(self) -> None:
        """Pretty-print all books in the library."""
        books = self.get_books()
        table = Table(title=f"Calibre Library ({len(books)} books)")
        table.add_column("ID", style="dim")
        table.add_column("Title", style="cyan")
        table.add_column("Author")
        table.add_column("Formats")

        for book in books:
            table.add_row(
                str(book["id"]),
                book["title"][:60],
                (book.get("authors") or "Unknown")[:40],
                book.get("formats", ""),
            )

        console.print(table)
