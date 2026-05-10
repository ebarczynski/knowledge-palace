"""File crawler for scattered text/markdown files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rich.console import Console

from .extractors import extract_file, ExtractedDocument

console = Console()


class FileCrawler:
    """Crawls directories and indexes supported text files."""

    def __init__(
        self,
        paths: list[str],
        extensions: list[str] | None = None,
    ):
        self.paths = [Path(p).expanduser() for p in paths]
        self.extensions = set(extensions or [".md", ".txt", ".org", ".rst", ".adoc"])

    def discover(self) -> list[Path]:
        """Find all supported files in configured paths."""
        files: list[Path] = []
        for base in self.paths:
            if not base.exists():
                console.print(f"[yellow]Path not found: {base}[/yellow]")
                continue

            if base.is_file() and base.suffix.lower() in self.extensions:
                files.append(base)
            elif base.is_dir():
                for ext in self.extensions:
                    files.extend(base.rglob(f"*{ext}"))

        # Deduplicate and sort
        files = sorted(set(files))
        return files

    def extract_file(self, path: Path) -> ExtractedDocument:
        """Extract content from a single file."""
        return extract_file(path, metadata={"source": "file"})

    def extract_all(self) -> list[ExtractedDocument]:
        """Discover and extract all files."""
        files = self.discover()
        console.print(f"Found [cyan]{len(files)}[/cyan] files to index")

        results: list[ExtractedDocument] = []
        for i, path in enumerate(files, 1):
            try:
                doc = self.extract_file(path)
                results.append(doc)
                if i % 50 == 0 or i == len(files):
                    console.print(f"  Extracted {i}/{len(files)}: {path.name}")
            except Exception as e:
                console.print(f"[red]Error extracting {path}: {e}[/red]")

        return results
