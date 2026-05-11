-- Migration 002: Add pre-computed tsvector generated column and GIN index
--
-- This adds a STORED generated column that maintains a pre-computed
-- tsvector for the chunk content, eliminating the need to call
-- to_tsvector('english', content) inline on every query.
-- The GIN index enables fast full-text lookups.

-- Add the generated column (safe on existing data; PostgreSQL computes it for all rows)
ALTER TABLE chunks
    ADD COLUMN content_tsvector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- GIN index for fast tsvector lookups
CREATE INDEX idx_chunks_content_tsvector ON chunks USING GIN (content_tsvector);
