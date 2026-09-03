"""FastAPI application with REST + GraphQL endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from ..config import Config
from ..db import init_db, create_tables, close_db, Document
from ..search.engine import SearchEngine
from .graphql_schema import schema
from .models import (
    SearchResponse,
    SearchResultItem,
    DocumentResponse,
    DocumentsResponse,
    DocumentListItem,
    SimilarResponse,
    HealthResponse,
)


# Global state
_config: Config | None = None
_search_engine: SearchEngine | None = None


def get_search_engine() -> SearchEngine:
    if _search_engine is None:
        raise RuntimeError("Search engine not initialized")
    return _search_engine


def get_config() -> Config:
    if _config is None:
        raise RuntimeError("Config not initialized")
    return _config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown database connections."""
    config = get_config()
    await init_db(config.database.url)
    await create_tables()
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
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # GraphQL
    graphql_app = GraphQLRouter(schema)
    app.include_router(graphql_app, prefix="/graphql")

    # REST API v1
    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok")

    @app.get("/api/v1/search", response_model=SearchResponse)
    async def search(
        q: str = Query(..., description="Search query"),
        mode: str = Query("hybrid", description="Search mode: semantic, keyword, hybrid"),
        limit: int = Query(10, ge=1, le=100),
        source: str | None = Query(None, description="Filter by source"),
        author: str | None = Query(None, description="Filter by author"),
    ):
        engine = get_search_engine()
        results = await engine.search(q, mode=mode, limit=limit, source_filter=source, author=author)
        return SearchResponse(
            total=results.total,
            mode=results.mode,
            query=results.query,
            results=[
                SearchResultItem(
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
                for r in results.results
            ],
        )

    @app.get("/api/v1/similar", response_model=SimilarResponse)
    async def find_similar(
        text: str = Query(..., description="Text to find similar content for"),
        threshold: float = Query(0.7, ge=0.0, le=1.0),
        limit: int = Query(5, ge=1, le=50),
    ):
        engine = get_search_engine()
        results = await engine.find_similar(text, threshold=threshold, limit=limit)
        return SimilarResponse(
            total=results.total,
            results=[
                SearchResultItem(
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
                for r in results.results
            ],
        )

    @app.get("/api/v1/context", response_model=SearchResponse)
    async def gather_context(
        q: str = Query(..., description="Task or topic to gather context for"),
        limit: int = Query(10, ge=1, le=50, description="Max fused results"),
        max_variants: int = Query(6, ge=1, le=12, description="Cap on query variants"),
        source: str | None = Query(None, description="Filter by source"),
        author: str | None = Query(None, description="Filter by author"),
    ):
        """Multi-query retrieval: expand query variants, fuse across modes via RRF.

        Returns the same shape as /search (``SearchResponse``) so callers can
        drop it in, but ``mode`` is ``"context"`` and results are de-duplicated
        fused passages with provenance.
        """
        engine = get_search_engine()
        results = await engine.retrieve_context(
            q,
            limit=limit,
            max_variants=max_variants,
            source_filter=source,
            author=author,
        )
        return SearchResponse(
            total=results.total,
            mode=results.mode,
            query=results.query,
            results=[
                SearchResultItem(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=r.score,
                    title=r.title,
                    author=r.author,
                    source=r.source,
                    tags=r.tags,
                    highlights=r.highlights,
                    provenance=r.provenance or {},
                )
                for r in results.results
            ],
        )

    @app.get("/api/v1/documents", response_model=DocumentsResponse)
    async def list_documents(
        source: str | None = Query(None, description="Filter by source"),
        author: str | None = Query(None, description="Filter by author"),
        limit: int = Query(20, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        from sqlalchemy import select, func as sa_func
        from ..db import session_context

        async with session_context() as session:
            query = select(Document).order_by(Document.created_at.desc())
            count_query = select(sa_func.count()).select_from(Document)

            if source is not None:
                query = query.where(Document.source == source)
                count_query = count_query.where(Document.source == source)
            if author is not None:
                query = query.where(Document.author == author)
                count_query = count_query.where(Document.author == author)

            total_result = await session.execute(count_query)
            total = total_result.scalar_one()

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            docs = result.scalars().all()

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
                        tags=doc.tags or [],
                        created_at=doc.created_at.isoformat(),
                    )
                    for doc in docs
                ],
            )

    @app.get("/api/v1/documents/{document_id}", response_model=DocumentResponse)
    async def get_document(document_id: str):
        from sqlalchemy import select
        from ..db import session_context

        async with session_context() as session:
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = result.scalar_one_or_none()
            if doc is None:
                raise HTTPException(404, "Document not found")
            return DocumentResponse(
                id=str(doc.id),
                title=doc.title,
                author=doc.author,
                source=doc.source,
                file_path=doc.file_path,
                description=doc.description,
                tags=doc.tags or [],
                metadata=doc.metadata_ or {},
                created_at=doc.created_at.isoformat(),
            )

    return app
