"""Hybrid search engine: vector similarity + full-text search with Reciprocal Rank Fusion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sql_text

from ..config import SEARCH_MODES, Config
from ..db import session_context
from ..embedding.service import EmbeddingService

#: Largest ``limit`` any caller may ask for; keeps one request from pulling the whole index.
MAX_LIMIT = 100
#: Longest accepted search query / question.
MAX_QUERY_CHARS = 2_000
#: Longest accepted ``similar`` text. Well under the embedding model's 8192-token
#: window, so nothing is ever silently truncated by the tokenizer.
MAX_SIMILAR_CHARS = 20_000
#: Document sources a caller may filter on.
SOURCES = ("calibre", "file", "manual")
# RRF smoothing constant, the standard value from Cormack et al. (2009). Shared
# with the Python-side fusion in retrieval/fuse.py.
RRF_K = 60


class InvalidQueryError(ValueError):
    """A caller supplied something the search cannot or should not run with.

    Surfaces as HTTP 422 on REST, as a GraphQL error, and as an MCP tool error.
    """


@dataclass
class SearchResult:
    """A single search result.

    ``score`` is a ranking score whose meaning depends on the mode and is not
    comparable across modes: cosine similarity for ``semantic`` and ``similar``,
    ``ts_rank_cd`` for ``keyword``, and an RRF score for ``hybrid`` and ``context``.
    """

    chunk_id: str
    document_id: str
    content: str
    score: float
    title: str
    author: str | None
    source: str
    tags: list[str]
    highlights: list[str]
    metadata: dict
    provenance: dict | None = None  # populated from DB columns (chunk_index, file_path, ...)


def _clean_tags(tags: Any) -> list[str]:
    """Drop non-string entries (Calibre exports can contain JSON nulls in tags)."""
    return [t for t in (tags or []) if isinstance(t, str)]


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


def _check_text(value: object, name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise InvalidQueryError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise InvalidQueryError(f"{name} must not be empty or whitespace")
    if "\x00" in value:
        raise InvalidQueryError(f"{name} must not contain NUL bytes (PostgreSQL text cannot store them)")
    if len(value) > max_chars:
        raise InvalidQueryError(f"{name} is {len(value)} characters long; the limit is {max_chars}")
    return value


def check_limit(value: object, name: str = "limit", maximum: int = MAX_LIMIT) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidQueryError(f"{name} must be an integer, got {value!r}")
    if not 1 <= value <= maximum:
        raise InvalidQueryError(f"{name} must be between 1 and {maximum}, got {value}")
    return value


def _check_tags(tags: object) -> list[str] | None:
    if tags is None:
        return None
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise InvalidQueryError("tags must be a list of strings")
    return _check_text_list(tags, "tags") or None


def _check_text_list(values: list[str], name: str) -> list[str]:
    checked = []
    for value in values:
        if "\x00" in value:
            raise InvalidQueryError(f"{name} must not contain NUL bytes")
        if value.strip():
            checked.append(value)
    return checked


class SearchEngine:
    """Hybrid search combining vector similarity and PostgreSQL full-text search."""

    def __init__(self, config: Config):
        self.config = config
        self.search_config = config.search
        self.embedding_service = EmbeddingService(config.embedding)

    @property
    def backend(self) -> str | None:
        """Embedding backend in use (``onnx``/``pytorch``/``mlx``); None until the model loads."""
        return self.embedding_service.backend

    async def warm_up(self) -> None:
        """Load the embedding model now so the first request does not pay for it."""
        await self.embedding_service.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            limit: Max results, 1..100 (default from config)
            source_filter: Filter by source ("calibre", "file", "manual")
            tags: Filter by tags (any of them)
            author: Filter by author (case-insensitive substring)

        Raises:
            InvalidQueryError: on any argument the search cannot run with.
        """
        query = _check_text(query, "query", MAX_QUERY_CHARS)
        mode = self.search_config.default_mode if mode is None else mode
        if mode not in SEARCH_MODES:
            raise InvalidQueryError(f"unknown search mode {mode!r}; choose from {', '.join(SEARCH_MODES)}")
        limit = check_limit(self.search_config.default_limit if limit is None else limit)
        source_filter = check_source(source_filter)
        tags = _check_tags(tags)
        author = check_optional_text(author, "author")

        if mode == "semantic":
            return await self._semantic_search(query, limit, source_filter, tags, author)
        if mode == "keyword":
            return await self._keyword_search(query, limit, source_filter, tags, author)
        return await self._hybrid_search(query, limit, source_filter, tags, author)

    async def retrieve_context(
        self,
        query: str,
        limit: int | None = None,
        max_variants: int | None = None,
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
            limit: number of fused results to return, 1..100.
            max_variants: cap on textual variants including the original, 1..12
                (bounds retrieval cost). Default 6.
            source_filter / author: passed through to each search.

        Raises:
            InvalidQueryError: on any argument the retrieval cannot run with.
        """
        from ..retrieval import expand_queries, fuse

        query = _check_text(query, "query", MAX_QUERY_CHARS)
        limit = check_limit(self.search_config.default_limit if limit is None else limit)
        max_variants = check_limit(6 if max_variants is None else max_variants, "max_variants", maximum=12)
        source_filter = check_source(source_filter)
        author = check_optional_text(author, "author")

        variants = expand_queries(query, max_variants=max_variants)

        # fetch more per sub-search than `limit` so fusion has depth to work with
        per_search = min(max(limit * 3, 20), MAX_LIMIT)
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

        # Post-fusion quality gate: semantic/hybrid legs are already gated by
        # min_similarity, but keyword-leg hits enter without vector evidence.
        # Since context() is the grounding tool for agents, require every hit
        # to clear the same similarity bar — keyword-only matches for topics
        # absent from the KB ("welding") must not sneak through.
        if self.search_config.min_similarity > 0:
            fused = await self._gate_by_similarity(query, fused)

        return SearchResults(results=fused, total=len(fused), mode="context", query=query)

    async def find_similar(
        self,
        text: str,
        threshold: float | None = None,
        limit: int | None = None,
    ) -> SearchResults:
        """Find chunks similar to the given text (vector-only search).

        Raises:
            InvalidQueryError: on any argument the search cannot run with.
        """
        text = _check_text(text, "text", MAX_SIMILAR_CHARS)
        threshold = 0.7 if threshold is None else threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise InvalidQueryError(f"threshold must be a number, got {threshold!r}")
        if not 0.0 <= threshold <= 1.0:
            raise InvalidQueryError(f"threshold must be between 0 and 1, got {threshold}")
        limit = check_limit(5 if limit is None else limit, maximum=50)
        query_embedding = await self.embedding_service.embed_query(text)

        async with session_context() as session:
            rows = (await session.execute(
                sql_text("""
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
                """),
                {"embedding": str(query_embedding), "threshold": float(threshold), "limit": limit},
            )).fetchall()
            return self._results(rows, "similar", text, score_attr="similarity")

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    async def _semantic_search(self, query, limit, source_filter, tags, author) -> SearchResults:
        """Vector similarity search using pgvector."""
        query_embedding = await self.embedding_service.embed_query(query)
        filters, filter_params = self._build_filters(source_filter, tags, author)
        gate_sql, gate_params = self._similarity_gate()

        # Every interpolated fragment is a constant chosen in this module; all
        # caller-supplied values travel as bound parameters.
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
              {gate_sql}
              {filters}
            ORDER BY c.embedding <=> :embedding
            LIMIT :limit
        """)  # noqa: S608

        async with session_context() as session:
            rows = (await session.execute(
                sql,
                {"embedding": str(query_embedding), "limit": limit, **gate_params, **filter_params},
            )).fetchall()
            return self._results(rows, "semantic", query)

    async def _keyword_search(self, query, limit, source_filter, tags, author) -> SearchResults:
        """PostgreSQL full-text search."""
        filters, filter_params = self._build_filters(source_filter, tags, author)

        async with session_context() as session:
            rows = (await session.execute(
                sql_text(f"""
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
                    {filters}
                    ORDER BY score DESC
                    LIMIT :limit
                """),  # noqa: S608
                {"query": query, "limit": limit, **filter_params},
            )).fetchall()
            return self._results(rows, "keyword", query, highlight_attr="highlights")

    async def _hybrid_search(self, query, limit, source_filter, tags, author) -> SearchResults:
        """Hybrid search using Reciprocal Rank Fusion (RRF)."""
        query_embedding = await self.embedding_service.embed_query(query)
        filters, filter_params = self._build_filters(source_filter, tags, author)
        gate_sql, gate_params = self._similarity_gate()

        # Each leg is ranked 1..N; the fused score is
        #   w_v / (k + rank_v) + w_f / (k + rank_f)
        # with a missing leg contributing 0, so keyword-only hits are allowed
        # through as literal term matches.
        sql = sql_text(f"""
            WITH vector_results AS (
                SELECT
                    c.id AS chunk_id,
                    c.embedding <=> :embedding AS distance,
                    ROW_NUMBER() OVER (ORDER BY c.embedding <=> :embedding) AS rank_v
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                  {gate_sql}
                  {filters}
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
                {filters}
                ORDER BY rank_score DESC
                LIMIT 50
            ),
            rrf AS (
                SELECT
                    COALESCE(v.chunk_id, f.chunk_id) AS chunk_id,
                    (
                        :weight_vector * COALESCE(1.0 / ({RRF_K} + v.rank_v), 0) +
                        :weight_fts * COALESCE(1.0 / ({RRF_K} + f.rank_f), 0)
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
        """)  # noqa: S608

        async with session_context() as session:
            rows = (await session.execute(
                sql,
                {
                    "embedding": str(query_embedding),
                    "query": query,
                    "limit": limit,
                    "weight_vector": self.search_config.hybrid_weight_vector,
                    "weight_fts": self.search_config.hybrid_weight_fts,
                    **gate_params,
                    **filter_params,
                },
            )).fetchall()
            return self._results(rows, "hybrid", query, score_attr="rrf_score", highlight_attr="highlights")

    async def _gate_by_similarity(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Drop results whose chunk embedding is below ``min_similarity``.

        Keyword-only results carry no similarity score; this fetches their
        embeddings in one query and computes cosine against the query. A chunk
        without an embedding has no evidence either way and is dropped.
        """
        if not results:
            return results
        q_emb = await self.embedding_service.embed_query(query)
        async with session_context() as session:
            rows = (await session.execute(
                sql_text(
                    "SELECT id::text, 1 - (embedding <=> CAST(:e AS vector)) AS sim "
                    "FROM chunks WHERE id = ANY(CAST(:ids AS uuid[]))"
                ),
                {"e": str(q_emb), "ids": [r.chunk_id for r in results]},
            )).fetchall()
        sims = {r[0]: r[1] for r in rows}
        floor = self.search_config.min_similarity
        return [
            r for r in results
            if sims.get(r.chunk_id) is not None and float(sims[r.chunk_id]) >= floor
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _similarity_gate(self) -> tuple[str, dict]:
        """The SQL enforcing ``min_similarity``; empty when the floor is disabled.

        ``min_similarity <= 0`` means "no floor": adding ``>= 0`` instead would
        still hide chunks whose cosine is negative.
        """
        floor = self.search_config.min_similarity
        if floor <= 0:
            return "", {}
        return "AND 1 - (c.embedding <=> :embedding) >= :min_sim", {"min_sim": floor}

    def _results(
        self,
        rows,
        mode: str,
        query: str,
        score_attr: str = "score",
        highlight_attr: str | None = None,
    ) -> SearchResults:
        results = [
            SearchResult(
                chunk_id=str(row.chunk_id),
                document_id=str(row.document_id),
                content=row.content,
                score=float(getattr(row, score_attr)),
                title=row.title,
                author=row.author,
                source=row.source,
                tags=_clean_tags(row.tags),
                highlights=[getattr(row, highlight_attr)] if highlight_attr and getattr(row, highlight_attr) else [],
                metadata=row.metadata or {},
                provenance=_build_provenance(row, row.metadata or {}),
            )
            for row in rows
        ]
        return SearchResults(results=results, total=len(results), mode=mode, query=query)

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
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if source:
            clauses.append("AND d.source = :flt_source")
            params["flt_source"] = source

        if author:
            clauses.append("AND d.author ILIKE :flt_author")
            params["flt_author"] = f"%{author}%"

        if tags:
            # json.dumps() is what keeps a quote or backslash in a tag from
            # turning the literal into invalid JSON.
            tag_conditions = []
            for i, tag in enumerate(tags):
                pname = f"flt_tag_{i}"
                tag_conditions.append(f"d.tags @> CAST(:{pname} AS jsonb)")
                params[pname] = json.dumps(tag)
            clauses.append(f"AND ({' OR '.join(tag_conditions)})")

        return " ".join(clauses), params


def check_source(source: object) -> str | None:
    """An optional source filter: None or blank means "any"; unknown values are rejected."""
    if source is None or (isinstance(source, str) and not source.strip()):
        return None
    if source not in SOURCES:
        raise InvalidQueryError(f"unknown source {source!r}; choose from {', '.join(SOURCES)}")
    return source


def check_optional_text(value: object, name: str) -> str | None:
    """An optional free-text filter: None/blank means "no filter"."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _check_text(value, name, MAX_QUERY_CHARS).strip()


__all__ = [
    "InvalidQueryError", "MAX_LIMIT", "SearchEngine", "SearchResult", "SearchResults",
    "check_limit", "check_optional_text", "check_source",
]
