"""Hybrid search engine: vector similarity + full-text search with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import Document, Chunk, get_session
from ..embedding.service import EmbeddingService
from ..config import Config, SearchConfig


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    title: str
    author: Optional[str]
    source: str
    tags: list[str]
    highlights: list[str]
    metadata: dict


@dataclass
class SearchResults:
    """Collection of search results."""
    results: list[SearchResult]
    total: int
    mode: str
    query: str


class SearchEngine:
    """Hybrid search combining vector similarity and PostgreSQL full-text search."""

    def __init__(self, config: Config):
        self.config = config
        self.search_config = config.search
        self.embedding_service = EmbeddingService(config.embedding)

    async def search(
        self,
        query: str,
        mode: str | None = None,
        limit: int | None = None,
        source_filter: str | None = None,
        tags: list[str] | None = None,
        author: str | None = None,
    ) -> SearchResults:
        """Search the knowledge base.

        Args:
            query: Search query text
            mode: "semantic", "keyword", or "hybrid" (default from config)
            limit: Max results (default from config)
            source_filter: Filter by source ("calibre", "file")
            tags: Filter by tags
            author: Filter by author
        """
        mode = mode or self.search_config.default_mode
        limit = limit or self.search_config.default_limit

        if mode == "semantic":
            return await self._semantic_search(query, limit, source_filter, tags, author)
        elif mode == "keyword":
            return await self._keyword_search(query, limit, source_filter, tags, author)
        elif mode == "hybrid":
            return await self._hybrid_search(query, limit, source_filter, tags, author)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

    async def _semantic_search(
        self,
        query: str,
        limit: int,
        source_filter: str | None,
        tags: list[str] | None,
        author: str | None,
    ) -> SearchResults:
        """Vector similarity search using pgvector."""
        query_embedding = await self.embedding_service.embed_query(query)

        async for session in get_session():
            # Build the query with filters
            filter_clauses = self._build_filters(source_filter, tags, author)

            sql = text(f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    d.title,
                    d.author,
                    d.source,
                    d.tags,
                    d.metadata,
                    1 - (c.embedding <=> :embedding) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                {filter_clauses}
                ORDER BY c.embedding <=> :embedding
                LIMIT :limit
            """)

            result = await session.execute(
                sql,
                {"embedding": str(query_embedding), "limit": limit},
            )
            rows = result.fetchall()

            results = [
                SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    score=float(row.score),
                    title=row.title,
                    author=row.author,
                    source=row.source,
                    tags=row.tags or [],
                    highlights=[],
                    metadata=row.metadata or {},
                )
                for row in rows
            ]

            return SearchResults(
                results=results,
                total=len(results),
                mode="semantic",
                query=query,
            )

        return SearchResults(results=[], total=0, mode="semantic", query=query)

    async def _keyword_search(
        self,
        query: str,
        limit: int,
        source_filter: str | None,
        tags: list[str] | None,
        author: str | None,
    ) -> SearchResults:
        """PostgreSQL full-text search."""
        filter_clauses = self._build_filters(source_filter, tags, author)

        async for session in get_session():
            sql = text(f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    d.title,
                    d.author,
                    d.source,
                    d.tags,
                    d.metadata,
                    ts_rank_cd(
                        to_tsvector('english', c.content),
                        plainto_tsquery('english', :query)
                    ) AS score,
                    ts_headline(
                        'english',
                        c.content,
                        plainto_tsquery('english', :query),
                        'MaxFragments=3, MinWords=5, MaxWords=20'
                    ) AS highlights
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
                {filter_clauses}
                ORDER BY score DESC
                LIMIT :limit
            """)

            result = await session.execute(sql, {"query": query, "limit": limit})
            rows = result.fetchall()

            results = [
                SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    score=float(row.score),
                    title=row.title,
                    author=row.author,
                    source=row.source,
                    tags=row.tags or [],
                    highlights=[row.highlights] if row.highlights else [],
                    metadata=row.metadata or {},
                )
                for row in rows
            ]

            return SearchResults(
                results=results,
                total=len(results),
                mode="keyword",
                query=query,
            )

        return SearchResults(results=[], total=0, mode="keyword", query=query)

    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        source_filter: str | None,
        tags: list[str] | None,
        author: str | None,
    ) -> SearchResults:
        """Hybrid search using Reciprocal Rank Fusion (RRF)."""
        query_embedding = await self.embedding_service.embed_query(query)
        filter_clauses = self._build_filters(source_filter, tags, author)

        # RRF combines vector and FTS results
        # k=60 is the standard RRF constant
        async for session in get_session():
            sql = text(f"""
                WITH vector_results AS (
                    SELECT
                        c.id AS chunk_id,
                        c.embedding <=> :embedding AS distance,
                        ROW_NUMBER() OVER (ORDER BY c.embedding <=> :embedding) AS rank_v
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding IS NOT NULL
                    {filter_clauses}
                    ORDER BY distance
                    LIMIT 50
                ),
                fts_results AS (
                    SELECT
                        c.id AS chunk_id,
                        ts_rank_cd(
                            to_tsvector('english', c.content),
                            plainto_tsquery('english', :query)
                        ) AS rank_score,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank_cd(
                                to_tsvector('english', c.content),
                                plainto_tsquery('english', :query)
                            ) DESC
                        ) AS rank_f,
                        ts_headline(
                            'english',
                            c.content,
                            plainto_tsquery('english', :query),
                            'MaxFragments=3, MinWords=5, MaxWords=20'
                        ) AS highlights
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
                    {filter_clauses}
                    ORDER BY rank_score DESC
                    LIMIT 50
                ),
                rrf AS (
                    SELECT
                        COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
                        (
                            :weight_vector * COALESCE(1.0 / (60 + v.rank_v), 0) +
                            :weight_fts * COALESCE(1.0 / (60 + f.rank_f), 0)
                        ) AS rrf_score,
                        f.highlights
                    FROM vector_results v
                    FULL OUTER JOIN fts_results f ON v.chunk_id = f.chunk_id
                    ORDER BY rrf_score DESC
                    LIMIT :limit
                )
                SELECT
                    rrf.chunk_id,
                    rrf.rrf_score,
                    rrf.highlights,
                    c.content,
                    c.document_id,
                    d.title,
                    d.author,
                    d.source,
                    d.tags,
                    d.metadata
                FROM rrf
                JOIN chunks c ON rrf.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                ORDER BY rrf.rrf_score DESC
            """)

            result = await session.execute(
                sql,
                {
                    "embedding": str(query_embedding),
                    "query": query,
                    "limit": limit,
                    "weight_vector": self.search_config.hybrid_weight_vector,
                    "weight_fts": self.search_config.hybrid_weight_fts,
                },
            )
            rows = result.fetchall()

            results = [
                SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    score=float(row.rrf_score),
                    title=row.title,
                    author=row.author,
                    source=row.source,
                    tags=row.tags or [],
                    highlights=[row.highlights] if row.highlights else [],
                    metadata=row.metadata or {},
                )
                for row in rows
            ]

            return SearchResults(
                results=results,
                total=len(results),
                mode="hybrid",
                query=query,
            )

        return SearchResults(results=[], total=0, mode="hybrid", query=query)

    async def find_similar(
        self,
        text: str,
        threshold: float = 0.7,
        limit: int = 5,
    ) -> SearchResults:
        """Find chunks similar to the given text (vector-only search)."""
        query_embedding = await self.embedding_service.embed_query(text)

        async for session in get_session():
            sql = text("""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    d.title,
                    d.author,
                    d.source,
                    d.tags,
                    d.metadata,
                    1 - (c.embedding <=> :embedding) AS similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                  AND 1 - (c.embedding <=> :embedding) > :threshold
                ORDER BY c.embedding <=> :embedding
                LIMIT :limit
            """)

            result = await session.execute(
                sql,
                {"embedding": str(query_embedding), "threshold": threshold, "limit": limit},
            )
            rows = result.fetchall()

            results = [
                SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    score=float(row.similarity),
                    title=row.title,
                    author=row.author,
                    source=row.source,
                    tags=row.tags or [],
                    highlights=[],
                    metadata=row.metadata or {},
                )
                for row in rows
            ]

            return SearchResults(
                results=results,
                total=len(results),
                mode="similar",
                query=text,
            )

        return SearchResults(results=[], total=0, mode="similar", query=text)

    def _build_filters(
        self,
        source: str | None,
        tags: list[str] | None,
        author: str | None,
    ) -> str:
        """Build SQL WHERE clauses for filtering."""
        clauses = []
        if source:
            clauses.append(f"AND d.source = '{source}'")
        if author:
            clauses.append(f"AND d.author ILIKE '%{author}%'")
        if tags:
            tag_conditions = " OR ".join(
                f"d.tags @> '\"{tag}\"'::jsonb" for tag in tags
            )
            clauses.append(f"AND ({tag_conditions})")
        return " ".join(clauses)
