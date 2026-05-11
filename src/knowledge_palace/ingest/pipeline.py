"""Ingestion pipeline: coordinate extraction, chunking, embedding, and storage.

Processes books in small batches (default 10) to balance:
- Memory: only hold N books' worth of text + embeddings at a time
- Throughput: batch-embed all chunks from N books together (faster than 1-by-1)

Optimizations:
- Pipelined: extraction of batch N+1 overlaps with embedding of batch N
- ONNX backend: 2-3x faster embedding when onnxruntime is installed
- Float16 embeddings: halve memory and DB storage with negligible search quality loss
"""

from __future__ import annotations

import asyncio
import gc
import logging

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from ..config import Config
from ..db import Document, Chunk, IngestionLog, init_db, create_tables, session_context
from ..embedding.service import EmbeddingService
from .calibre import CalibreBridge
from .crawler import FileCrawler
from .chunker import chunk_text
from .extractors import ExtractedDocument

console = Console()
logger = logging.getLogger(__name__)

# Process this many books per batch. Each batch is:
# extracted → chunked → embedded together → committed → freed.
# 10 books × ~350 avg chunks × 768 floats × 2 bytes (fp16) ≈ 5MB embeddings.
# Plus model (~1.2GB) + book text (~5-20MB) + overhead = ~2-3GB peak.
BOOKS_PER_BATCH = 10


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline in small batches."""

    def __init__(self, config: Config):
        self.config = config
        self.embedding_service = EmbeddingService(config.embedding)

    async def run(
        self,
        source: str = "all",
        reindex: bool = False,
    ) -> dict:
        await init_db(self.config.database.url)
        await create_tables()

        results = {"extracted": 0, "chunked": 0, "embedded": 0, "errors": 0}

        if source in ("calibre", "all") and self.config.calibre.library_path:
            r = await self._process_calibre(reindex)
            for k in results:
                results[k] += r[k]

        if source in ("files", "all") and self.config.sources.paths:
            r = await self._process_files(reindex)
            for k in results:
                results[k] += r[k]

        async with session_context() as log_session:
            log = IngestionLog(
                source="pipeline",
                status="success" if results["errors"] == 0 else "partial",
                items_processed=results["extracted"],
                items_failed=results["errors"],
            )
            log_session.add(log)
            await log_session.commit()

        return results

    # ------------------------------------------------------------------
    # Helper: extract and chunk a batch of books (CPU/IO-bound)
    # ------------------------------------------------------------------

    def _extract_batch(self, books, bridge, reindex: bool) -> list:
        """Synchronous: extract and chunk a batch of books. Returns (doc, chunks) list.
        
        Note: dedup check is NOT done here (this runs in a thread, and asyncpg
        connections are bound to the main event loop). Dedup happens in
        _embed_and_commit_batch inside the same session as the insert.
        """
        batch_data = []
        for book in books:
            try:
                doc = bridge.extract_book(book)
                if doc is None:
                    continue

                chunks = chunk_text(
                    doc.content,
                    strategy=self.config.chunking.strategy,
                    max_tokens=self.config.chunking.max_tokens,
                    overlap_tokens=self.config.chunking.overlap_tokens,
                    respect_headings=self.config.chunking.respect_headings,
                )

                if chunks:
                    doc.content = ""  # free large text
                    batch_data.append((doc, chunks))

            except Exception as e:
                logger.exception("Failed to extract %s", book.get("title", "?"))
        return batch_data

    async def _embed_and_commit_batch(
        self,
        batch_data: list,
        results: dict,
        total_processed: list,
    ) -> int:
        """Embed chunks and commit to DB. Returns count of books committed."""
        if not batch_data:
            return 0

        from sqlalchemy import select

        # Collect all chunk texts
        all_texts = []
        for _, chunks in batch_data:
            all_texts.extend(c.content for c in chunks)

        # Embed in sub-batches
        try:
            all_embeddings = []
            embed_batch_size = self.config.embedding.batch_size
            for start in range(0, len(all_texts), embed_batch_size):
                end = min(start + embed_batch_size, len(all_texts))
                batch_embs = await self.embedding_service.embed_batch(
                    all_texts[start:end]
                )
                all_embeddings.extend(batch_embs)
        except Exception as e:
            results["errors"] += len(batch_data)
            logger.exception("Embedding failed for batch")
            return 0

        # Commit each document
        committed = 0
        emb_offset = 0
        for doc, chunks in batch_data:
            try:
                doc_embeddings = all_embeddings[emb_offset:emb_offset + len(chunks)]

                async with session_context() as session:
                    # Dedup check inside session (handles multi-format books)
                    existing = await session.execute(
                        select(Document).where(
                            Document.content_hash == doc.content_hash
                        )
                    )
                    if existing.scalar_one_or_none():
                        emb_offset += len(chunks)
                        continue

                    desc = doc.metadata.get("description")
                    if desc:
                        desc = desc.replace("\x00", "")

                    db_doc = Document(
                        title=doc.title,
                        author=doc.author,
                        source=doc.source,
                        file_path=doc.file_path,
                        calibre_id=doc.metadata.get("calibre_id"),
                        content_hash=doc.content_hash,
                        description=desc,
                        tags=doc.metadata.get("tags", []),
                        metadata_=doc.metadata,
                    )
                    session.add(db_doc)
                    await session.flush()

                    for j, chunk in enumerate(chunks):
                        db_chunk = Chunk(
                            document_id=db_doc.id,
                            content=chunk.content.replace("\x00", ""),
                            chunk_index=chunk.index,
                            embedding=doc_embeddings[j],
                            token_count=chunk.token_count,
                            metadata_=chunk.metadata,
                        )
                        session.add(db_chunk)

                    await session.commit()

                emb_offset += len(chunks)
                results["extracted"] += 1
                results["chunked"] += len(chunks)
                results["embedded"] += sum(
                    1 for e in doc_embeddings if e is not None
                )
                total_processed[0] += 1
                committed += 1

            except Exception as e:
                emb_offset += len(chunks)
                results["errors"] += 1
                logger.exception("Failed to commit %s", doc.title)

        del all_texts, all_embeddings
        return committed

    # ------------------------------------------------------------------
    # Calibre: pipelined extraction + embedding
    # ------------------------------------------------------------------

    async def _process_calibre(self, reindex: bool) -> dict:
        try:
            bridge = CalibreBridge(self.config.calibre.library_path)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return {"extracted": 0, "chunked": 0, "embedded": 0, "errors": 0}

        books = bridge.get_books()
        console.print(f"\n[bold]Calibre: {len(books)} books found[/bold]")

        results = {"extracted": 0, "chunked": 0, "embedded": 0, "errors": 0}
        total_processed = [0]  # mutable counter for closure

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total} books)"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing", total=len(books))

            # Split books into batches
            batches = []
            for i in range(0, len(books), BOOKS_PER_BATCH):
                batches.append(books[i:i + BOOKS_PER_BATCH])

            # Pipeline: extract batch[N+1] in executor while embedding batch[N]
            loop = asyncio.get_event_loop()
            pending_extraction = None

            for batch_idx, batch_books in enumerate(batches):
                batch_end = min((batch_idx + 1) * BOOKS_PER_BATCH, len(books))

                # If we have a pending extraction from previous iteration, await it
                if pending_extraction is not None:
                    batch_data = await pending_extraction
                else:
                    # First batch: extract synchronously
                    batch_data = await loop.run_in_executor(
                        None, self._extract_batch, batch_books, bridge, reindex
                    )

                # Kick off extraction of NEXT batch in background (if exists)
                if batch_idx + 1 < len(batches):
                    next_books = batches[batch_idx + 1]
                    pending_extraction = loop.run_in_executor(
                        None, self._extract_batch, next_books, bridge, reindex
                    )
                else:
                    pending_extraction = None

                # Embed and commit current batch
                if batch_data:
                    await self._embed_and_commit_batch(
                        batch_data, results, total_processed
                    )
                    del batch_data
                    gc.collect()

                progress.update(task, advance=len(batch_books))

                if total_processed[0] > 0 and (total_processed[0] % 20 == 0 or batch_end >= len(books)):
                    console.print(
                        f"  [green]✓[/green] {batch_end}/{len(books)} books processed "
                        f"({results['chunked']} chunks total)"
                    )

        console.print(
            f"\n[bold green]Done:[/bold green] {results['extracted']} docs, "
            f"{results['chunked']} chunks"
            + (f" ({results['errors']} errors)" if results["errors"] else "")
        )
        return results

    # ------------------------------------------------------------------
    # Files: batch of N files at a time
    # ------------------------------------------------------------------

    async def _process_files(self, reindex: bool) -> dict:
        from sqlalchemy import select

        crawler = FileCrawler(
            self.config.sources.paths,
            self.config.sources.file_extensions,
        )
        file_paths = crawler.discover()
        console.print(f"\n[bold]Files: {len(file_paths)} files found[/bold]")

        results = {"extracted": 0, "chunked": 0, "embedded": 0, "errors": 0}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing", total=len(file_paths))

            for batch_start in range(0, len(file_paths), BOOKS_PER_BATCH):
                batch_end = min(batch_start + BOOKS_PER_BATCH, len(file_paths))
                batch_paths = file_paths[batch_start:batch_end]

                batch_data = []
                for file_path in batch_paths:
                    try:
                        doc = crawler.extract_file(file_path)
                        if doc is None:
                            continue

                        if not reindex:
                            async with session_context() as check_session:
                                exists = await check_session.execute(
                                    select(Document).where(
                                        Document.content_hash == doc.content_hash
                                    )
                                )
                                if exists.scalar_one_or_none():
                                    continue

                        chunks = chunk_text(
                            doc.content,
                            strategy=self.config.chunking.strategy,
                            max_tokens=self.config.chunking.max_tokens,
                            overlap_tokens=self.config.chunking.overlap_tokens,
                            respect_headings=self.config.chunking.respect_headings,
                        )

                        if chunks:
                            doc.content = ""
                            batch_data.append((doc, chunks))

                    except Exception as e:
                        results["errors"] += 1

                if batch_data:
                    await self._embed_and_commit_batch(
                        batch_data, results, [0]
                    )
                    del batch_data
                    gc.collect()

                progress.update(task, advance=len(batch_paths))

        console.print(
            f"\n[bold green]Files done:[/bold green] {results['extracted']} docs, "
            f"{results['chunked']} chunks"
        )
        return results
