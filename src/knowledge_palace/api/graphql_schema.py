"""Strawberry GraphQL schema."""

from __future__ import annotations

from typing import Optional

import enum

import strawberry
from strawberry.types import Info

from ..search.engine import SearchEngine, SearchResult


@strawberry.type
class ChunkType:
    chunk_id: str
    document_id: str
    content: str
    score: float
    title: str
    author: Optional[str]
    source: str
    tags: list[str]
    highlights: list[str]

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
    author: Optional[str]
    source: str
    tags: list[str]


class SearchMode(str, enum.Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


def _get_engine(info: Info) -> SearchEngine:
    """Extract SearchEngine from FastAPI app state."""
    from .app import get_search_engine
    return get_search_engine()


@strawberry.type
class Query:
    @strawberry.field
    async def search(
        self,
        info: Info,
        query: str,
        mode: Optional[SearchMode] = None,
        limit: Optional[int] = None,
        source: Optional[str] = None,
        author: Optional[str] = None,
    ) -> SearchResultType:
        engine = _get_engine(info)
        results = await engine.search(
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
        info: Info,
        text: str,
        threshold: Optional[float] = 0.7,
        limit: Optional[int] = 5,
    ) -> SearchResultType:
        engine = _get_engine(info)
        results = await engine.find_similar(text, threshold=threshold, limit=limit)
        return SearchResultType(
            total=results.total,
            mode="similar",
            query=text,
            chunks=[ChunkType.from_search_result(r) for r in results.results],
        )

    @strawberry.field
    async def documents(
        self,
        info: Info,
        source: Optional[str] = None,
        author: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[DocumentType]:
        from sqlalchemy import select
        from ..db import Document, get_session

        async for session in get_session():
            stmt = select(Document)
            if source:
                stmt = stmt.where(Document.source == source)
            if author:
                stmt = stmt.where(Document.author.ilike(f"%{author}%"))
            stmt = stmt.order_by(Document.title).limit(limit or 50)
            result = await session.execute(stmt)
            docs = result.scalars().all()
            return [
                DocumentType(
                    id=doc.id,
                    title=doc.title,
                    author=doc.author,
                    source=doc.source,
                    tags=doc.tags or [],
                )
                for doc in docs
            ]
        return []


schema = strawberry.Schema(query=Query)
