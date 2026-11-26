"""Ingestion pipeline: coordinate extraction, chunking, embedding, and storage.

Each pass reconciles the index with its source instead of only ever adding to
it. Documents are identified by where they come from (a file path, or a Calibre
book id), and content hashes decide whether anything changed:

- unchanged        skipped before any embedding work
- changed          re-chunked, re-embedded, and the old document replaced
- duplicate        identical text is already indexed elsewhere
- moved / re-added the existing document is re-pointed, its chunks are kept
- gone from source the document is deleted

With ``reindex=True`` (``kp ingest --reindex``) unchanged documents are
rebuilt as well, which is what re-chunking or re-embedding a library requires.

Documents are processed in small batches to bound memory: a batch is extracted,
chunked, embedded and committed, then freed.
"""

from __future__ import annotations

import asyncio
import enum
import gc
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from ..db import Chunk, Document, IngestionLog, create_tables, init_db, session_context
from ..embedding.service import EmbeddingService
from .calibre import CalibreBridge, CalibreError, CalibreLibraryError
from .chunker import Chunk as TextChunk
from .chunker import chunk_text
from .crawler import FileCrawler
from .extractors import ExtractedDocument

if TYPE_CHECKING:
    from ..config import Config

console = Console()
logger = logging.getLogger(__name__)

# Process this many items per batch. Each batch is:
# extracted → chunked → embedded together → committed → freed.
# Embeddings are held as Python lists of floats until commit: about 24 KB per
# 768-dim vector, so 10 books × ~350 chunks ≈ 85 MB (measured), plus the model
# and the batch's text.
BOOKS_PER_BATCH = 10


@dataclass(frozen=True)
class _Indexed:
    """A document row, as the pipeline sees it when deciding what to do."""

    id: str
    title: str
    content_hash: str
    file_path: str | None
    calibre_id: int | None
    source: str


@dataclass
class _Prepared:
    doc: ExtractedDocument
    chunks: list[TextChunk]


class _Action(enum.Enum):
    UNCHANGED = "unchanged"    # same identity, same content: nothing to do
    DUPLICATE = "duplicate"    # same content already indexed elsewhere
    ADOPT = "adopt"            # same content indexed under a vanished identity: re-point it
    INDEX = "index"            # (re)build the document


@dataclass(frozen=True)
class _Decision:
    action: _Action
    # Existing document that the rebuilt one supersedes (same identity, stale content).
    target: str | None = None
    # Documents left over at this identity that no longer correspond to it.
    stale: tuple[str, ...] = ()


# (doc, documents by content hash, documents by file path, documents by Calibre id) -> decision
_Classifier = Callable[
    [ExtractedDocument, dict[str, _Indexed], dict[str, list[_Indexed]], dict[int, list[_Indexed]]],
    _Decision,
]


def _under(path: str, roots: list[Path]) -> bool:
    """True when ``path`` is a file inside (or equal to) one of ``roots``."""
    for root in roots:
        prefix = str(root).rstrip(os.sep) + os.sep
        if path == str(root) or path.startswith(prefix):
            return True
    return False


def classify_file(
    doc: ExtractedDocument,
    by_hash: dict[str, _Indexed],
    by_path: dict[str, list[_Indexed]],
    reindex: bool,
) -> _Decision:
    """Decide what to do with a note file, given what the index holds.

    ``by_path`` may hold several documents for one path on databases written by
    older versions, which never removed superseded documents.
    """
    at_path = by_path.get(doc.file_path, [])
    same_hash = next((d for d in at_path if d.content_hash == doc.content_hash), None)
    others = tuple(d.id for d in at_path if d is not same_hash)

    if same_hash is not None:
        if not reindex:
            return _Decision(_Action.UNCHANGED)
        return _Decision(_Action.INDEX, target=same_hash.id, stale=others)

    indexed = by_hash.get(doc.content_hash)
    if indexed is not None and indexed.source == "file":
        if indexed.file_path and Path(indexed.file_path).exists():
            # Identical text at another path that is still there.
            return _Decision(_Action.DUPLICATE, stale=others)
        # The file moved or was renamed: keep the indexed chunks, re-point the document.
        return _Decision(_Action.ADOPT, target=indexed.id, stale=others)
    if indexed is not None:
        # Same text as an indexed book.
        return _Decision(_Action.DUPLICATE, stale=others)
    return _Decision(_Action.INDEX, target=others[0] if others else None, stale=others[1:])


def classify_calibre(
    doc: ExtractedDocument,
    by_hash: dict[str, _Indexed],
    by_calibre_id: dict[int, list[_Indexed]],
    library_ids: set[int],
    reindex: bool,
) -> _Decision:
    """Decide what to do with a Calibre book, given what the index holds."""
    book_id = int(doc.metadata["calibre_id"])
    at_id = by_calibre_id.get(book_id, [])
    same_hash = next((d for d in at_id if d.content_hash == doc.content_hash), None)
    others = tuple(d.id for d in at_id if d is not same_hash)

    if same_hash is not None:
        if not reindex:
            return _Decision(_Action.UNCHANGED)
        return _Decision(_Action.INDEX, target=same_hash.id, stale=others)

    indexed = by_hash.get(doc.content_hash)
    if indexed is not None:
        if indexed.source == "calibre" and indexed.calibre_id not in library_ids:
            # The book was removed from Calibre and added back under a new id.
            return _Decision(_Action.ADOPT, target=indexed.id, stale=others)
        # Another catalog entry (or a note) holds the same text.
        return _Decision(_Action.DUPLICATE, stale=others)
    return _Decision(_Action.INDEX, target=others[0] if others else None, stale=others[1:])


def _pretend_adopted(
    by_hash: dict[str, _Indexed],
    by_path: dict[str, list[_Indexed]],
    by_calibre_id: dict[int, list[_Indexed]],
    doc: ExtractedDocument,
) -> None:
    """Record a pending adoption in the lookup tables for the rest of the batch.

    Without this, a second catalog entry holding the same text would also try
    to claim the re-pointed document.
    """
    before = by_hash[doc.content_hash]
    calibre_id = doc.metadata.get("calibre_id")
    after = _Indexed(before.id, doc.title, before.content_hash, doc.file_path, calibre_id, before.source)

    old_groups = [by_path.get(before.file_path or "", [])]
    if before.calibre_id is not None:
        old_groups.append(by_calibre_id.get(before.calibre_id, []))
    for group in old_groups:
        if before in group:
            group.remove(before)

    by_hash[after.content_hash] = after
    if after.file_path is not None:
        by_path.setdefault(after.file_path, []).append(after)
    if after.calibre_id is not None:
        by_calibre_id.setdefault(after.calibre_id, []).append(after)


def _reason(error: BaseException) -> str:
    """A one-line, user-facing description of a failure."""
    if isinstance(error, CalibreError):
        return str(error)
    first = str(error).strip().splitlines()[0] if str(error).strip() else ""
    return f"{type(error).__name__}: {first}" if first else type(error).__name__


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

        results = {"extracted": 0, "chunked": 0, "embedded": 0, "errors": 0, "skipped": 0}
        self._failures: list[tuple[str, str]] = []

        if source in ("calibre", "all") and self.config.calibre.library_path:
            await self._process_calibre(reindex, results)
        if source in ("files", "all") and self.config.sources.paths:
            await self._process_files(reindex, results)

        for name, reason in self._failures:
            console.print(f"  [red]✗ {name}: {reason}[/red]")
        if self._failures:
            console.print(f"  [red]{len(self._failures)} item(s) not indexed — see above[/red]")

        async with session_context() as log_session:
            log_session.add(IngestionLog(
                source="pipeline",
                status="success" if results["errors"] == 0 else ("error" if results["extracted"] == 0 else "partial"),
                items_processed=results["extracted"],
                items_failed=results["errors"],
                error_message="; ".join(f"{n}: {r}" for n, r in self._failures[:20]) or None,
            ))
            await log_session.commit()

        return results

    # ------------------------------------------------------------------
    # Shared batch processing
    # ------------------------------------------------------------------

    def _chunk(self, content: str) -> list[TextChunk]:
        c = self.config.chunking
        return chunk_text(
            content,
            strategy=c.strategy,
            max_tokens=c.max_tokens,
            overlap_tokens=c.overlap_tokens,
            respect_headings=c.respect_headings,
        )

    async def _load_indexed(self) -> list[_Indexed]:
        async with session_context() as session:
            rows = (await session.execute(
                select(Document.id, Document.title, Document.content_hash,
                       Document.file_path, Document.calibre_id, Document.source)
            )).all()
        return [_Indexed(*row) for row in rows]

    async def _process_batch(
        self,
        prepared: list[_Prepared],
        classify: _Classifier,
        results: dict,
    ) -> None:
        """Reconcile one batch of extracted documents with the index."""
        if not prepared:
            return
        existing = await self._load_indexed()
        by_hash = {d.content_hash: d for d in existing}
        by_path: dict[str, list[_Indexed]] = {}
        by_calibre_id: dict[int, list[_Indexed]] = {}
        for d in existing:
            if d.file_path:
                by_path.setdefault(d.file_path, []).append(d)
            if d.calibre_id is not None:
                by_calibre_id.setdefault(d.calibre_id, []).append(d)

        to_embed: list[tuple[_Prepared, _Decision]] = []
        to_adopt: list[tuple[_Prepared, _Decision]] = []
        stale: set[str] = set()
        seen_in_batch: set[str] = set()

        for item in prepared:
            decision = classify(item.doc, by_hash, by_path, by_calibre_id)
            if decision.action is _Action.INDEX and item.doc.content_hash in seen_in_batch:
                # An earlier item in this batch carries the same text.
                decision = _Decision(_Action.DUPLICATE, stale=decision.stale)
            seen_in_batch.add(item.doc.content_hash)
            stale.update(decision.stale)
            if decision.action is _Action.ADOPT:
                to_adopt.append((item, decision))
                _pretend_adopted(by_hash, by_path, by_calibre_id, item.doc)
            elif decision.action is _Action.INDEX:
                to_embed.append((item, decision))
            else:
                results["skipped"] += 1

        # Bookkeeping that needs no embedding happens first, in one transaction.
        adopt_targets = {decision.target for _, decision in to_adopt}
        stale -= adopt_targets
        if to_adopt or stale:
            async with session_context() as session:
                for item, decision in to_adopt:
                    if decision.target is not None:
                        await self._repoint(session, decision.target, item.doc)
                if stale:
                    await session.execute(delete(Document).where(Document.id.in_(stale)))
                await session.commit()
            results["skipped"] += len(to_adopt)

        if not to_embed:
            return

        texts = [chunk.content for item, _ in to_embed for chunk in item.chunks]
        try:
            embeddings = await self._embed(texts)
        except Exception:  # noqa: BLE001 - reported per item, the run continues
            logger.exception("Embedding failed for a batch")
            results["errors"] += len(to_embed)
            for item, _ in to_embed:
                self._fail(item.doc.title, "embedding failed (see the log for details)")
            return

        offset = 0
        for item, decision in to_embed:
            count = len(item.chunks)
            try:
                async with session_context() as session:
                    if decision.target:
                        await session.execute(delete(Document).where(Document.id == decision.target))
                    await self._insert(session, item, embeddings[offset:offset + count])
                    await session.commit()
            except IntegrityError:
                # Indexed by a concurrent writer between the check and the commit.
                results["skipped"] += 1
            except Exception as e:  # noqa: BLE001 - one bad document must not stop the run
                logger.exception("Failed to commit %s", item.doc.title)
                results["errors"] += 1
                self._fail(item.doc.title, e)
            else:
                results["extracted"] += 1
                results["chunked"] += count
                results["embedded"] += count
            offset += count

        del embeddings, texts
        gc.collect()

    async def _embed(self, texts: list[str]) -> list[list[float] | None]:
        embeddings: list[list[float] | None] = []
        size = self.config.embedding.batch_size
        for start in range(0, len(texts), size):
            embeddings.extend(await self.embedding_service.embed_batch(texts[start:start + size]))
        return embeddings

    async def _insert(self, session, item: _Prepared, embeddings: list[list[float] | None]) -> None:
        doc = item.doc
        description = doc.metadata.get("description")
        db_doc = Document(
            title=doc.title,
            author=doc.author,
            source=doc.source,
            file_path=doc.file_path,
            calibre_id=doc.metadata.get("calibre_id"),
            content_hash=doc.content_hash,
            description=description.replace("\x00", "") if description else None,
            tags=doc.metadata.get("tags", []),
            metadata_=doc.metadata,
        )
        session.add(db_doc)
        await session.flush()
        session.add_all(
            Chunk(
                document_id=db_doc.id,
                content=chunk.content,
                chunk_index=chunk.index,
                embedding=embedding,
                token_count=chunk.token_count,
                metadata_=chunk.metadata,
            )
            for chunk, embedding in zip(item.chunks, embeddings, strict=True)
        )

    async def _repoint(self, session, existing_id: str, doc: ExtractedDocument) -> None:
        """Attach an indexed document to the identity it now lives under.

        The text is unchanged, so the chunks and their embeddings stay.
        """
        existing = await session.get(Document, existing_id)
        if existing is None:
            return
        existing.title = doc.title
        existing.author = doc.author
        existing.file_path = doc.file_path
        existing.calibre_id = doc.metadata.get("calibre_id")
        existing.tags = doc.metadata.get("tags", [])
        existing.metadata_ = doc.metadata
        description = doc.metadata.get("description")
        existing.description = description.replace("\x00", "") if description else None

    def _fail(self, name: str, error: BaseException | str) -> None:
        reason = _reason(error) if isinstance(error, BaseException) else error
        self._failures.append((name, reason))

    # ------------------------------------------------------------------
    # Calibre: pipelined extraction + embedding
    # ------------------------------------------------------------------

    def _extract_batch(self, bridge: CalibreBridge, books: list[dict]) -> tuple[list[_Prepared], list[tuple[str, str]]]:
        """Synchronous: extract and chunk a batch of books. Runs in a thread."""
        prepared: list[_Prepared] = []
        failures: list[tuple[str, str]] = []
        for book in books:
            try:
                doc = bridge.extract_book(book)
                chunks = self._chunk(doc.content)
                if not chunks:
                    failures.append((book["title"], "no text could be extracted"))
                    continue
                doc.content = ""  # free large text
                prepared.append(_Prepared(doc, chunks))
            except Exception as e:  # noqa: BLE001 - reported per book, the run continues
                failures.append((book["title"], _reason(e)))
        return prepared, failures

    async def _process_calibre(self, reindex: bool, results: dict) -> None:
        try:
            bridge = CalibreBridge(self.config.calibre.library_path)
            books = bridge.get_books()
        except CalibreLibraryError as e:
            self._fail("Calibre library", e)
            results["errors"] += 1
            return
        console.print(f"\n[bold]Calibre: {len(books)} books found[/bold]")
        library_ids = {int(book["id"]) for book in books}

        def classify(doc, by_hash, _by_path, by_calibre_id):
            return classify_calibre(doc, by_hash, by_calibre_id, library_ids, reindex)

        batches = [books[i:i + BOOKS_PER_BATCH] for i in range(0, len(books), BOOKS_PER_BATCH)]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing", total=len(books))
            loop = asyncio.get_running_loop()
            pending: Awaitable | None = None
            for batch_idx, batch_books in enumerate(batches):
                # Extraction of batch N+1 overlaps with embedding of batch N.
                if pending is None:
                    outcome = await loop.run_in_executor(None, self._extract_batch, bridge, batch_books)
                else:
                    outcome = await pending
                if batch_idx + 1 < len(batches):
                    pending = loop.run_in_executor(None, self._extract_batch, bridge, batches[batch_idx + 1])

                prepared, failures = outcome
                results["errors"] += len(failures)
                for title, reason in failures:
                    self._fail(title, reason)
                await self._process_batch(prepared, classify, results)

                progress.update(task, advance=len(batch_books))
                del outcome, prepared
                gc.collect()

        removed = await self._prune_calibre(library_ids)
        console.print(
            f"[bold green]Calibre done:[/bold green] {results['extracted']} indexed, "
            f"{results['skipped']} unchanged or duplicate, {removed} removed, "
            f"{results['errors']} failed"
        )

    async def _prune_calibre(self, library_ids: set[int]) -> int:
        """Delete documents for books that are no longer in the Calibre library."""
        if not library_ids:
            # An empty or unreadable catalog must never empty the index.
            return 0
        existing = await self._load_indexed()
        gone = [
            d.id for d in existing
            if d.source == "calibre" and d.calibre_id is not None and d.calibre_id not in library_ids
        ]
        if gone:
            async with session_context() as session:
                await session.execute(delete(Document).where(Document.id.in_(gone)))
                await session.commit()
        return len(gone)

    # ------------------------------------------------------------------
    # Files: batch of N files at a time
    # ------------------------------------------------------------------

    async def _process_files(self, reindex: bool, results: dict) -> None:
        crawler = FileCrawler(
            self.config.sources.paths,
            self.config.sources.file_extensions,
        )
        file_paths = crawler.discover()
        console.print(f"\n[bold]Files: {len(file_paths)} files found[/bold]")

        def classify(doc, by_hash, by_path, _by_calibre_id):
            return classify_file(doc, by_hash, by_path, reindex)

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
                batch_paths = file_paths[batch_start:batch_start + BOOKS_PER_BATCH]
                prepared: list[_Prepared] = []
                for file_path in batch_paths:
                    try:
                        doc = crawler.extract_file(file_path)
                        chunks = self._chunk(doc.content)
                    except Exception as e:  # noqa: BLE001 - reported per file, the run continues
                        results["errors"] += 1
                        self._fail(str(file_path), e)
                        continue
                    if chunks:
                        doc.content = ""
                        prepared.append(_Prepared(doc, chunks))
                    else:
                        # The file still exists but no longer has text to index.
                        await self._drop_documents_at(str(file_path))

                await self._process_batch(prepared, classify, results)
                progress.update(task, advance=len(batch_paths))
                del prepared
                gc.collect()

        removed = await self._prune_files()
        console.print(
            f"[bold green]Files done:[/bold green] {results['extracted']} indexed, "
            f"{results['skipped']} unchanged or duplicate, {removed} removed, "
            f"{results['errors']} failed"
        )

    async def _drop_documents_at(self, file_path: str) -> int:
        """Delete every document indexed for ``file_path``."""
        async with session_context() as session:
            ids = (await session.execute(select(Document.id).where(Document.file_path == file_path))).scalars().all()
            if ids:
                await session.execute(delete(Document).where(Document.id.in_(ids)))
                await session.commit()
        return len(ids)

    async def _prune_files(self) -> int:
        """Delete documents whose file no longer exists under the configured roots.

        A root that is itself missing (an unmounted drive, a typo) is skipped:
        its files are not gone, only unreachable.
        """
        roots = [Path(p).expanduser() for p in self.config.sources.paths]
        live_roots = [root for root in roots if root.exists()]
        existing = await self._load_indexed()
        gone = [
            d.id for d in existing
            if d.source == "file" and d.file_path
            and _under(d.file_path, live_roots) and not Path(d.file_path).exists()
        ]
        if gone:
            async with session_context() as session:
                await session.execute(delete(Document).where(Document.id.in_(gone)))
                await session.commit()
        return len(gone)


__all__ = ["IngestionPipeline", "BOOKS_PER_BATCH"]
