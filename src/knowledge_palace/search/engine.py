"""Hybrid search engine: vector similarity + full-text search with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text as sql_text, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import Document, Chunk, session_context
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
    provenance: dict = None  # populated from DB columns (chunk_index, file_path, ...)


def _build_provenance(row, doc_metadata: dict) -> dict:
    """Assemble a provenance dict from a result row + document metadata.

    ``row`` exposes ``chunk_index``/``file_path``/``calibre_id`` (added to each
    search SQL query); ``format``/``page_count`` live in the document metadata
    JSONB that every query already fetches. None values are omitted so callers
    get a compact, honest dict. Section headings and exact page numbers are not
    available (extraction discards them) — see README limitations.
    """
    prov = {}
    # chunk-level
    chunk_index = getattr(row, "chunk_index", None)
    if chunk_index is not None:
        prov["chunk_index"] = chunk_index
    # document-level columns
    file_path = getattr(row, "file_path", None)
    if file_path:
        prov["file_path"] = file_path
    calibre_id = getattr(row, "calibre_id", None)
    if calibre_id is not None:
        prov["calibre_id"] = calibre_id
    # document-level fields nested in metadata JSONB
    if doc_metadata:
        for key in ("format", "page_count"):
            val = doc_metadata.get(key)
            if val is not None:
                prov[key] = val
    return prov


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

    async def retrieve_context(
        self,
        query: str,
        limit: int = 10,
        max_variants: int = 6,
        source_filter: str | None = None,
        author: str | None = None,
    ) -> SearchResults:
        """Multi-query retrieval for agent context gathering.

        Generates textual variants of ``query`` (rule-based synonym expansion,
        see :mod:`knowledge_palace.retrieval.expand`), runs each variant in all
        three search modes (keyword / semantic / hybrid), and fuses the ranked
        lists with Reciprocal Rank Fusion, de-duplicating by chunk id.

        This lets an agent harness gather broad, de-duplicated, ranked context
        in a single call instead of orchestrating many searches. The original
        (user) query is weighted higher than auto-expanded variants.

        Args:
            query: the task or question to gather context for.
            limit: number of fused results to return.
            max_variants: cap on textual variants (bounds retrieval cost).
            source_filter / author: passed through to each search.
        """
        from ..retrieval import expand_queries, fuse

        variants = expand_queries(query, max_variants=max_variants)

        # fetch more per sub-search than `limit` so fusion has depth to work with
        per_search = max(limit * 3, 20)
        ranked_lists: list[list[SearchResult]] = []
        weights: list[float] = []

        for i, variant in enumerate(variants):
            # original query weighted 2x; expansions weighted 1x
            weight = 2.0 if i == 0 else 1.0
            for mode in ("keyword", "semantic", "hybrid"):
                res = await self.search(
                    variant,
                    mode=mode,
                    limit=per_search,
                    source_filter=source_filter,
                    author=author,
                )
                if res.results:
                    ranked_lists.append(res.results)
                    weights.append(weight)

        if not ranked_lists:
            return SearchResults(results=[], total=0, mode="context", query=query)

        fused = fuse(ranked_lists, weights=weights)[:limit]
        return SearchResults(results=fused, total=len(fused), mode="context", query=query)

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

        async with session_context() as session:
            # Build the query with filters
            filter_clauses, filter_params = self._build_filters(source_filter, tags, author)

            sql = sql_text(f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    d.title,
                    d.author,
                    d.source,
                    d.tags,
                    d.metadata,
                    d.file_path,
                    d.calibre_id,
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
                {"embedding": str(query_embedding), "limit": limit, **filter_params},
            )
            rows = result.fetchall()

            results = [
                SearchResult(
                    chunk_id=str(row.chunk_id),
                    document_id=str(row.document_id),
                    content=row.content,
                    score=float(row.score),
                    title=row.title,
                    author=row.author,
                    source=row.source,
                    tags=row.tags or [],
                    highlights=[],
                    metadata=row.metadata or {},
                    provenance=_build_provenance(row, row.metadata or {}),
                )
                for row in rows
            ]

            return SearchResults(
                results=results,
                total=len(results),
                mode="semantic",
                query=query,
            )

    async def _keyword_search(
        self,
        query: str,
        limit: int,
        source_filter: str | None,
        tags: list[str] | None,
        author: str | None,
    ) -> SearchResults:
        """PostgreSQL full-text search."""
        filter_clauses, filter_params = self._build_filters(source_filter, tags, author)

        async with session_context() as session:
            sql = sql_text(f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    d.title,
                    d.author,
                    d.source,
                    d.tags,
                    d.metadata,
                    d.file_path,
                    d.calibre_id,
                    ts_rank_cd(
                        c.content_tsvector,
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
                WHERE c.content_tsvector @@ plainto_tsquery('english', :query)
                {filter_clauses}
                ORDER BY score DESC
                LIMIT :limit
            """)

            result = await session.execute(sql, {"query": query, "limit": limit, **filter_params})
            rows = result.fetchall()

            results = [
                SearchResult(
                    chunk_id=str(row.chunk_id),
                    document_id=str(row.document_id),
                    content=row.content,
                    score=float(row.score),
                    title=row.title,
                    author=row.author,
                    source=row.source,
                    tags=row.tags or [],
                    highlights=[row.highlights] if row.highlights else [],
                    metadata=row.metadata or {},
                    provenance=_build_provenance(row, row.metadata or {}),
                )
                for row in rows
            ]

            return SearchResults(
                results=results,
                total=len(results),
                mode="keyword",
                query=query,
            )

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
        filter_clauses, filter_params = self._build_filters(source_filter, tags, author)

        # RRF combines vector and FTS results
        # k=60 is the standard RRF constant
        async with session_context() as session:
            sql = sql_text(f"""
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
                            c.content_tsvector,
                            plainto_tsquery('english', :query)
                        ) AS rank_score,
                        ROW_NUMBER() OVER (
                            ORDER BY ts_rank_cd(
                                c.content_tsvector,
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
                    WHERE c.content_tsvector @@ plainto_tsquery('english', :query)
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
                    c.chunk_index,
                    d.title,
                    d.author,
                    d.source,
                    d.tags,
                    d.metadata,
                    d.file_path,
                    d.calibre_id
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
                    **filter_params,
                },
            )
            rows = result.fetchall()

            results = [
                SearchResult(
                    chunk_id=str(row.chunk_id),
                    document_id=str(row.document_id),
                    content=row.content,
                    score=float(row.rrf_score),
                    title=row.title,
                    author=row.author,
                    source=row.source,
                    tags=row.tags or [],
                    highlights=[row.highlights] if row.highlights else [],
                    metadata=row.metadata or {},
                    provenance=_build_provenance(row, row.metadata or {}),
                )
                for row in rows
            ]

            return SearchResults(
                results=results,
                total=len(results),
                mode="hybrid",
                query=query,
            )

    async def find_similar(
        self,
        text: str,
        threshold: float = 0.7,
        limit: int = 5,
    ) -> SearchResults:
        """Find chunks similar to the given text (vector-only search)."""
        query_embedding = await self.embedding_service.embed_query(text)

        async with session_context() as session:
            sql = sql_text("""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    d.title,
                    d.author,
                    d.source,
                    d.tags,
                    d.metadata,
                    d.file_path,
                    d.calibre_id,
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
                    chunk_id=str(row.chunk_id),
                    document_id=str(row.document_id),
                    content=row.content,
                    score=float(row.similarity),
                    title=row.title,
                    author=row.author,
                    source=row.source,
                    tags=row.tags or [],
                    highlights=[],
                    metadata=row.metadata or {},
                    provenance=_build_provenance(row, row.metadata or {}),
                )
                for row in rows
            ]

            return SearchResults(
                results=results,
                total=len(results),
                mode="similar",
                query=text,
            )

    def _build_filters(
        self,
        source: str | None,
        tags: list[str] | None,
        author: str | None,
    ) -> tuple[str, dict]:
        """Build SQL WHERE clauses for filtering using parameterized queries.

        Returns:
            Tuple of (filter_sql_fragment, params_dict) where params_dict
            contains named parameters to be merged into session.execute() calls.
            Parameter names are prefixed with 'flt_' to avoid collisions with
            existing query parameters (embedding, query, limit, etc.).
        """
        clauses = []
        params: dict = {}

        if source:
            clauses.append("AND d.source = :flt_source")
            params["flt_source"] = source

        if author:
            clauses.append("AND d.author ILIKE :flt_author")
            params["flt_author"] = f"%{author}%"

        if tags:
            tag_conditions = []
            for i, tag in enumerate(tags):
                pname = f"flt_tag_{i}"
                tag_conditions.append(f"d.tags @> :{pname}::jsonb")
                params[pname] = f'"{tag}"'
            clauses.append(f"AND ({' OR '.join(tag_conditions)})")

        return " ".join(clauses), params
