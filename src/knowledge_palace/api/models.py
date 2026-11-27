"""Pydantic models for REST API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float = Field(
        description=(
            "Ranking score; its meaning depends on `mode` and is not comparable across "
            "modes: cosine similarity (semantic, similar), ts_rank_cd (keyword), "
            "RRF score (hybrid, context)."
        )
    )
    title: str
    author: str | None
    source: str
    tags: list[str]
    highlights: list[str]
    provenance: dict = {}


class SearchResponse(BaseModel):
    total: int
    mode: str
    query: str
    results: list[SearchResultItem]


class SimilarResponse(BaseModel):
    total: int
    results: list[SearchResultItem]


class DocumentResponse(BaseModel):
    id: str
    title: str
    author: str | None
    source: str
    file_path: str | None
    description: str | None
    tags: list[str]
    metadata: dict
    created_at: str


class DocumentListItem(BaseModel):
    id: str
    title: str
    author: str | None
    source: str
    tags: list[str]
    created_at: str


class DocumentsResponse(BaseModel):
    total: int
    offset: int
    limit: int
    documents: list[DocumentListItem]
