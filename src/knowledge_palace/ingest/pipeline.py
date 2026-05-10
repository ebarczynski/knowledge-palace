"""Ingestion pipeline: coordinate extraction, chunking, embedding, and storage."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..config import Config
from ..db import Document, Chunk, IngestionLog, init_db, create_tables, get_session
from ..embedding.service import EmbeddingService
from .calibre import CalibreBridge
from .crawler import FileCrawler
from .chunker import chunk_text
from .extractors import ExtractedDocument

console = Console()


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.embedding_service = EmbeddingService(config.embedding)
        self._calibre: CalibreBridge | None = None

    async def run(
        self,
        source: str = "all",
        reindex: bool = False,
    ) -> dict:
        """Run the ingestion pipeline.

        Args:
            source: "calibre", "files", or "all"
            reindex: If True, reprocess even already-indexed documents

        Returns:
            Summary dict with counts
        """
        await init_db(self.config.database.url)
        await create_tables()

        results = {"extracted": 0, "chunked": 0, "embedded": 0, "errors": 0}

        extracted_docs: list[ExtractedDocument] = []

        if source in ("calibre", "all") and self.config.calibre.library_path:
            calibre_docs = self._ingest_calibre(reindex)
            extracted_docs.extend(calibre_docs)

        if source in ("files", "all") and self.config.sources.paths:
            file_docs = self._ingest_files()
            extracted_docs.extend(file_docs)

        if not extracted_docs:
            console.print("[yellow]No documents to process[/yellow]")
            return results

        # Store documents and chunks
        results = await self._store_documents(extracted_docs, reindex)
        return results

    def _ingest_calibre(self, reindex: bool) -> list[ExtractedDocument]:
        """Extract documents from Calibre library."""
        try:
            bridge = CalibreBridge(self.config.calibre.library_path)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return []

        books = bridge.get_books()
        console.print(f"\n[bold]Calibre: {len(books)} books found[/bold]")

        docs: list[ExtractedDocument] = []
        for i, book in enumerate(books, 1):
            doc = bridge.extract_book(book)
            if doc:
                docs.append(doc)
                if i % 20 == 0 or i == len(books):
                    console.print(f"  Extracted {i}/{len(books)}: {doc.title[:50]}")

        console.print(f"[green]Calibre: {len(docs)} documents extracted[/green]")
        return docs

    def _ingest_files(self) -> list[ExtractedDocument]:
        """Extract documents from file directories."""
        crawler = FileCrawler(
            self.config.sources.paths,
            self.config.sources.file_extensions,
        )
        docs = crawler.extract_all()
        console.print(f"[green]Files: {len(docs)} documents extracted[/green]")
        return docs

    async def _store_documents(
        self,
        docs: list[ExtractedDocument],
        reindex: bool,
    ) -> dict:
        """Store extracted documents, chunk them, embed chunks, and save to DB."""
        from sqlalchemy import select

        results = {"extracted": 0, "chunked": 0, "embedded": 0, "errors": 0}

        async for session in get_session():
            for doc in docs:
                try:
                    # Check for duplicates by content hash
                    if not reindex:
                        exists = await session.execute(
                            select(Document).where(Document.content_hash == doc.content_hash)
                        )
                        if exists.scalar_one_or_none():
                            continue

                    # Create document record
                    db_doc = Document(
                        title=doc.title,
                        author=doc.author,
                        source=doc.source,
                        file_path=doc.file_path,
                        calibre_id=doc.metadata.get("calibre_id"),
                        content_hash=doc.content_hash,
                        description=doc.metadata.get("description"),
                        tags=doc.metadata.get("tags", []),
                        metadata_=doc.metadata,
                    )
                    session.add(db_doc)
                    await session.flush()  # get the ID
                    results["extracted"] += 1

                    # Chunk the document
                    chunks = chunk_text(
                        doc.content,
                        strategy=self.config.chunking.strategy,
                        max_tokens=self.config.chunking.max_tokens,
                        overlap_tokens=self.config.chunking.overlap_tokens,
                        respect_headings=self.config.chunking.respect_headings,
                    )

                    # Embed chunks in batches
                    chunk_texts = [c.content for c in chunks]
                    embeddings = await self.embedding_service.embed_batch(chunk_texts)

                    # Store chunks with embeddings
                    for chunk, embedding in zip(chunks, embeddings):
                        db_chunk = Chunk(
                            document_id=db_doc.id,
                            content=chunk.content,
                            chunk_index=chunk.index,
                            embedding=embedding,
                            token_count=chunk.token_count,
                            metadata_=chunk.metadata,
                        )
                        session.add(db_chunk)
                        results["chunked"] += 1
                        if embedding is not None:
                            results["embedded"] += 1

                    await session.commit()
                    console.print(f"  [green]✓[/green] {doc.title[:50]} ({len(chunks)} chunks)")

                except Exception as e:
                    await session.rollback()
                    console.print(f"  [red]✗[/red] {doc.title[:50]}: {e}")
                    results["errors"] += 1

            # Log ingestion
            log = IngestionLog(
                source="pipeline",
                status="success" if results["errors"] == 0 else "partial",
                items_processed=results["extracted"],
                items_failed=results["errors"],
            )
            session.add(log)
            await session.commit()

        return results
