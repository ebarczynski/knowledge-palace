"""FastAPI application with REST + GraphQL endpoints."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from strawberry.fastapi import GraphQLRouter

from .. import __version__
from ..db import Document, close_db, create_tables, init_db, session_context
from ..search.engine import MAX_QUERY_CHARS, InvalidQueryError, SearchEngine, check_optional_text
from .graphql_schema import schema
from .models import (
    DocumentListItem,
    DocumentResponse,
    DocumentsResponse,
    HealthResponse,
    SearchResponse,
    SearchResultItem,
    SimilarResponse,
)

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

SearchMode = Literal["semantic", "keyword", "hybrid"]
SourceFilter = Literal["calibre", "file", "manual"]

# Global state
_config: Config | None = None
_search_engine: SearchEngine | None = None


def _clean_tags(tags) -> list[str]:
    """Drop non-string entries (Calibre exports can contain JSON nulls in tags)."""
    return [t for t in (tags or []) if isinstance(t, str)]


def get_search_engine() -> SearchEngine:
    if _search_engine is None:
        raise RuntimeError("Search engine not initialized")
    return _search_engine


def get_config() -> Config:
    if _config is None:
        raise RuntimeError("Config not initialized")
    return _config


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and tear down database connections; load the model up front."""
    config = get_config()
    await init_db(config.database.url)
    await create_tables()
    try:
        # Without this the first search pays for loading the model (several
        # seconds) inside the request. A failed warm-up is not fatal: the
        # model is then loaded lazily on first use.
        await get_search_engine().warm_up()
    except Exception:  # noqa: BLE001 - startup must not hinge on the model being loadable right now
        logger.exception("Embedding model could not be preloaded; it will load on first use")
    yield
    await close_db()


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _config, _search_engine

    _config = config
    _search_engine = SearchEngine(config)

    app = FastAPI(
        title="Knowledge Palace",
        description="Personal Knowledge Base with Semantic Search",
        version=__version__,
        lifespan=lifespan,
    )

    # Cross-origin access is opt-in. The API has no authentication and holds a
    # private library: with no allowed origins configured, no CORS headers are
    # sent, so a web page open in the same browser cannot read it.
    if config.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.exception_handler(InvalidQueryError)
    async def _reject_invalid_query(_request: Request, exc: InvalidQueryError) -> JSONResponse:
        """Bad search arguments are the client's mistake: 422, not 500."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # GraphQL
    graphql_app = GraphQLRouter(schema)
    app.include_router(graphql_app, prefix="/graphql")

    # REST API v1
    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok")

    @app.get("/api/v1/search", response_model=SearchResponse)
    async def search(
        q: str = Query(..., min_length=1, max_length=MAX_QUERY_CHARS, description="Search query"),
        mode: SearchMode = Query("hybrid", description="Search mode"),
        limit: int = Query(10, ge=1, le=100),
        source: SourceFilter | None = Query(None, description="Filter by source"),
        author: str | None = Query(
            None, max_length=MAX_QUERY_CHARS, description="Filter by author (case-insensitive substring)"
        ),
    ):
        engine = get_search_engine()
        results = await engine.search(q, mode=mode, limit=limit, source_filter=source, author=author)
        return SearchResponse(
            total=results.total,
            mode=results.mode,
            query=results.query,
            results=[_item(r) for r in results.results],
        )

    @app.get("/api/v1/similar", response_model=SimilarResponse)
    async def find_similar(
        text: str = Query(..., min_length=1, max_length=20_000, description="Text to find similar content for"),
        threshold: float = Query(0.7, ge=0.0, le=1.0),
        limit: int = Query(5, ge=1, le=50),
    ):
        engine = get_search_engine()
        results = await engine.find_similar(text, threshold=threshold, limit=limit)
        return SimilarResponse(total=results.total, results=[_item(r) for r in results.results])

    @app.get("/api/v1/context", response_model=SearchResponse)
    async def gather_context(
        q: str = Query(
            ..., min_length=1, max_length=MAX_QUERY_CHARS, description="Task or topic to gather context for"
        ),
        limit: int = Query(10, ge=1, le=50, description="Max fused results"),
        max_variants: int = Query(6, ge=1, le=12, description="Cap on query variants, the original included"),
        source: SourceFilter | None = Query(None, description="Filter by source"),
        author: str | None = Query(
            None, max_length=MAX_QUERY_CHARS, description="Filter by author (case-insensitive substring)"
        ),
    ):
        """Multi-query retrieval: expand query variants, fuse across modes via RRF.

        Returns the same shape as /search (``SearchResponse``) so callers can
        drop it in, but ``mode`` is ``"context"`` and results are de-duplicated
        fused passages with provenance.
        """
        engine = get_search_engine()
        results = await engine.retrieve_context(
            q, limit=limit, max_variants=max_variants, source_filter=source, author=author
        )
        return SearchResponse(
            total=results.total,
            mode=results.mode,
            query=results.query,
            results=[_item(r) for r in results.results],
        )

    @app.get("/api/v1/documents", response_model=DocumentsResponse)
    async def list_documents(
        source: SourceFilter | None = Query(None, description="Filter by source"),
        author: str | None = Query(
            None, max_length=MAX_QUERY_CHARS, description="Filter by author (case-insensitive substring)"
        ),
        limit: int = Query(20, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        from sqlalchemy import func as sa_func
        from sqlalchemy import select

        author = check_optional_text(author, "author")
        async with session_context() as session:
            query = select(Document).order_by(Document.created_at.desc(), Document.id)
            count_query = select(sa_func.count()).select_from(Document)

            if source is not None:
                query = query.where(Document.source == source)
                count_query = count_query.where(Document.source == source)
            if author is not None:
                # Substring match, as in /search and the GraphQL `documents` field.
                query = query.where(Document.author.ilike(f"%{author}%"))
                count_query = count_query.where(Document.author.ilike(f"%{author}%"))

            total = (await session.execute(count_query)).scalar_one()
            docs = (await session.execute(query.offset(offset).limit(limit))).scalars().all()

            return DocumentsResponse(
                total=total,
                offset=offset,
                limit=limit,
                documents=[
                    DocumentListItem(
                        id=str(doc.id),
                        title=doc.title,
                        author=doc.author,
                        source=doc.source,
                        tags=_clean_tags(doc.tags),
                        created_at=doc.created_at.isoformat(),
                    )
                    for doc in docs
                ],
            )

    @app.get("/api/v1/documents/{document_id}", response_model=DocumentResponse)
    async def get_document(document_id: uuid.UUID):
        """A document by id. A malformed id is a 422; an unknown one a 404."""
        from sqlalchemy import select

        async with session_context() as session:
            doc = (await session.execute(select(Document).where(Document.id == str(document_id)))).scalar_one_or_none()
            if doc is None:
                raise HTTPException(status_code=404, detail="Document not found")
            return DocumentResponse(
                id=str(doc.id),
                title=doc.title,
                author=doc.author,
                source=doc.source,
                file_path=doc.file_path,
                description=doc.description,
                tags=_clean_tags(doc.tags),
                metadata=doc.metadata_ or {},
                created_at=doc.created_at.isoformat(),
            )

    return app


def _item(r) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=str(r.chunk_id),
        document_id=str(r.document_id),
        content=r.content,
        score=r.score,
        title=r.title,
        author=r.author,
        source=r.source,
        tags=r.tags,
        highlights=r.highlights,
        provenance=r.provenance or {},
    )
