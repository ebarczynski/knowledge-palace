-- Knowledge Palace: Initial Schema
-- Requires: PostgreSQL 16+ with pgvector extension

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: one row per book/article/note
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(1024) NOT NULL,
    author VARCHAR(512),
    source VARCHAR(64) NOT NULL,  -- 'calibre', 'file', 'manual'
    file_path TEXT,
    calibre_id INTEGER,
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    description TEXT,
    tags JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table: one row per text chunk (for embedding)
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(768),  -- pgvector: 768 dims for nomic-embed-text
    token_count INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Ingestion log: track ingestion runs
CREATE TABLE IF NOT EXISTS ingestion_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL,  -- 'success', 'error', 'partial'
    items_processed INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    error_message TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS ix_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS ix_documents_author ON documents(author);
CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks(document_id);

-- GIN index for full-text search on chunks
CREATE INDEX IF NOT EXISTS ix_chunks_content_fts ON chunks
    USING GIN (to_tsvector('english', content));

-- JSONB index for tags
CREATE INDEX IF NOT EXISTS ix_documents_tags ON documents USING GIN (tags);

-- HNSW index for vector similarity search (fast ANN)
-- Uses cosine distance (<=>) which works well with normalized embeddings
CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
