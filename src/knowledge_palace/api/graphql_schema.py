"""Strawberry GraphQL schema.

Mirrors the REST API: the same ``SearchEngine`` validates every argument, so
an invalid one surfaces as a GraphQL error with a plain message.
"""

from __future__ import annotations

import enum
from typing import cast

import strawberry
from strawberry.scalars import JSON
from strawberry.types import Info

from ..db import Document, session_context
from ..search.engine import SearchEngine, SearchResult, check_limit, check_optional_text, check_source

#: Largest page the ``documents`` field returns.
MAX_DOCUMENTS = 200


def _clean_tags(tags) -> list[str]:
    """Drop non-string entries (Calibre exports can contain JSON nulls in tags)."""
    return [t for t in (tags or []) if isinstance(t, str)]


@strawberry.type
class ChunkType:
    chunk_id: str
    document_id: str
    content: str
    score: float
    title: str
    author: str | None
    source: str
    tags: list[str]
    highlights: list[str]
    provenance: JSON | None = None

    @classmethod
    def from_search_result(cls, r: SearchResult) -> ChunkType:
        return cls(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            content=r.content,
            score=r.score,
            title=r.title,
            author=r.author,
            source=r.source,
            tags=r.tags,
            highlights=r.highlights,
            provenance=cast("JSON | None", r.provenance),
        )


@strawberry.type
class SearchResultType:
    total: int
    mode: str
    query: str
    chunks: list[ChunkType]


@strawberry.type
class DocumentType:
    id: strawberry.ID
    title: str
    author: str | None
    source: str
    tags: list[str]


@strawberry.enum
class SearchMode(enum.Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


def _get_engine() -> SearchEngine:
    """The SearchEngine shared with the REST API (stored on the app module)."""
    from .app import get_search_engine
    return get_search_engine()


@strawberry.type
class Query:
    @strawberry.field
    async def search(
        self,
        _info: Info,
        query: str,
        mode: SearchMode | None = None,
        limit: int | None = None,
        source: str | None = None,
        author: str | None = None,
    ) -> SearchResultType:
        results = await _get_engine().search(
            query,
            mode=mode.value if mode else None,
            limit=limit,
            source_filter=source,
            author=author,
        )
        return SearchResultType(
            total=results.total,
            mode=results.mode,
            query=results.query,
            chunks=[ChunkType.from_search_result(r) for r in results.results],
        )

    @strawberry.field
    async def similar(
        self,
        _info: Info,
        text: str,
        threshold: float | None = 0.7,
        limit: int | None = 5,
    ) -> SearchResultType:
        results = await _get_engine().find_similar(text, threshold=threshold, limit=limit)
        return SearchResultType(
            total=results.total,
            mode=results.mode,
            query=results.query,
            chunks=[ChunkType.from_search_result(r) for r in results.results],
        )

    @strawberry.field
    async def context(
        self,
        _info: Info,
        query: str,
        limit: int | None = 10,
        max_variants: int | None = 6,
        source: str | None = None,
        author: str | None = None,
    ) -> SearchResultType:
        """Multi-query retrieval: expand variants, fuse across modes via RRF."""
        results = await _get_engine().retrieve_context(
            query,
            limit=limit,
            max_variants=max_variants,
            source_filter=source,
            author=author,
        )
        return SearchResultType(
            total=results.total,
            mode=results.mode,
            query=results.query,
            chunks=[ChunkType.from_search_result(r) for r in results.results],
        )

    @strawberry.field
    async def documents(
        self,
        _info: Info,
        source: str | None = None,
        author: str | None = None,
        limit: int | None = None,
    ) -> list[DocumentType]:
        """Documents, newest first; ``author`` is a case-insensitive substring, as elsewhere."""
        from sqlalchemy import select

        source = check_source(source)
        author = check_optional_text(author, "author")
        page = check_limit(limit, maximum=MAX_DOCUMENTS) if limit is not None else 50

        async with session_context() as session:
            stmt = select(Document).order_by(Document.created_at.desc(), Document.id)
            if source:
                stmt = stmt.where(Document.source == source)
            if author:
                stmt = stmt.where(Document.author.ilike(f"%{author}%"))
            docs = (await session.execute(stmt.limit(page))).scalars().all()
            return [
                DocumentType(
                    id=doc.id,
                    title=doc.title,
                    author=doc.author,
                    source=doc.source,
                    tags=_clean_tags(doc.tags),
                )
                for doc in docs
            ]


schema = strawberry.Schema(query=Query)
