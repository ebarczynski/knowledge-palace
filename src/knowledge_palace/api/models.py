"""Pydantic models for REST API."""

from uuid import UUID

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class SearchResultItem(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    title: str
    author: str | None
    source: str
    tags: list[str]
    highlights: list[str]


class SearchResponse(BaseModel):
    total: int
    mode: str
    query: str
    results: list[SearchResultItem]


class SimilarResponse(BaseModel):
    total: int
    results: list[SearchResultItem]


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    author: str | None
    source: str
    file_path: str | None
    description: str | None
    tags: list[str]
    metadata: dict
    created_at: str


class DocumentListItem(BaseModel):
    id: UUID
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
