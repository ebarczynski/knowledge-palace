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
        """Find the actual file path for a book, preferring the given format."""
        formats = (book.get("formats") or "").split(",")
        authors = (book.get("authors") or "Unknown").split(",")[0].strip()
        title = book.get("title", "Unknown")
        book_id = book.get("id", 0)

        # Try preferred format first, then fallbacks
        format_priority = [preferred_format, "EPUB", "PDF", "TXT", "MOBI"]
        for fmt in format_priority:
            if fmt in formats:
                # Calibre stores files as: Author Name/Title (ID)/Title - Author.format
                author_dir = authors.replace("/", "_")
                title_dir = f"{title} ({book_id})"
                # Sanitize for filesystem
                for ch in ['\\', '<', '>', ':', '"', '|', '?', '*']:
                    author_dir = author_dir.replace(ch, '_')
                    title_dir = title_dir.replace(ch, '_')
                
                file_name = f"{title} - {authors}"
                for ch in ['\\', '<', '>', ':', '"', '|', '?', '*']:
                    file_name = file_name.replace(ch, '_')
                
                file_name += f".{fmt.lower()}"

                candidates = [
                    self.library_path / author_dir / title_dir / file_name,
                ]
                
                # Try to find the file with glob as fallback
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
                
                # Glob fallback
                pattern = f"**/{book_id}/**/*.{fmt.lower()}"
                matches = list(self.library_path.glob(pattern))
                if matches:
                    return matches[0]

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
