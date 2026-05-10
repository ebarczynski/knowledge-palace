"""Document extraction: EPUB, PDF, plain text."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()


@dataclass
class ExtractedDocument:
    """Result of extracting text from a file."""
    title: str
    author: str | None
    content: str
    file_path: str
    content_hash: str
    metadata: dict
    source: str  # "calibre" | "file"


class Extractor(ABC):
    """Base class for document extractors."""

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        ...

    @abstractmethod
    def extract(self, path: Path, metadata: dict | None = None) -> ExtractedDocument:
        ...


class TextExtractor(Extractor):
    """Handles .txt, .md, .org, .rst files."""

    EXTENSIONS = {".txt", ".md", ".org", ".rst", ".adoc"}

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.EXTENSIONS

    def extract(self, path: Path, metadata: dict | None = None) -> ExtractedDocument:
        content = path.read_text(encoding="utf-8", errors="replace")
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Try to extract title from first heading or first line
        title = metadata.get("title") if metadata else None
        if not title:
            title = self._extract_title(content, path)

        return ExtractedDocument(
            title=title or path.stem,
            author=metadata.get("author") if metadata else None,
            content=content,
            file_path=str(path),
            content_hash=content_hash,
            metadata=metadata or {},
            source=metadata.get("source", "file") if metadata else "file",
        )

    def _extract_title(self, content: str, path: Path) -> str | None:
        """Extract title from markdown heading or first line."""
        for line in content.split("\n")[:5]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            if line.startswith("Title:"):
                return line[6:].strip()
        # Fallback: filename
        return path.stem


class EpubExtractor(Extractor):
    """Handles .epub files via ebooklib."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".epub"

    def extract(self, path: Path, metadata: dict | None = None) -> ExtractedDocument:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        raw_metadata = metadata or {}

        # Extract metadata from EPUB
        title = raw_metadata.get("title")
        if not title:
            titles = book.get_metadata("DC", "title")
            if titles:
                title = titles[0][0]

        author = raw_metadata.get("author")
        if not author:
            authors = book.get_metadata("DC", "creator")
            if authors:
                author = authors[0][0]

        # Extract text from all HTML items
        parts: list[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                parts.append(text)

        content = "\n\n".join(parts)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Extract other metadata
        tags = []
        for subject in book.get_metadata("DC", "subject"):
            tags.append(subject[0])

        return ExtractedDocument(
            title=title or path.stem,
            author=author,
            content=content,
            file_path=str(path),
            content_hash=content_hash,
            metadata={
                **raw_metadata,
                "tags": tags,
                "format": "epub",
            },
            source=raw_metadata.get("source", "calibre"),
        )


class PdfExtractor(Extractor):
    """Handles .pdf files via PyMuPDF (fitz)."""

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract(self, path: Path, metadata: dict | None = None) -> ExtractedDocument:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        raw_metadata = metadata or {}

        # Extract PDF metadata
        pdf_meta = doc.metadata
        title = raw_metadata.get("title") or pdf_meta.get("title") or path.stem
        author = raw_metadata.get("author") or pdf_meta.get("author")

        # Extract text from all pages
        pages: list[str] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append(text)

        content = "\n\n".join(pages)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return ExtractedDocument(
            title=title,
            author=author,
            content=content,
            file_path=str(path),
            content_hash=content_hash,
            metadata={
                **raw_metadata,
                "page_count": len(doc),
                "format": "pdf",
            },
            source=raw_metadata.get("source", "calibre"),
        )


def get_extractors() -> list[Extractor]:
    """Return all available extractors in priority order."""
    return [EpubExtractor(), PdfExtractor(), TextExtractor()]


def extract_file(path: Path, metadata: dict | None = None) -> ExtractedDocument:
    """Auto-detect file type and extract text."""
    for extractor in get_extractors():
        if extractor.can_handle(path):
            return extractor.extract(path, metadata)
    raise ValueError(f"No extractor found for {path.suffix}")
