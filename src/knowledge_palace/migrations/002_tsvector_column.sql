-- Migration 002: Add pre-computed tsvector generated column and GIN index
--
-- This adds a STORED generated column that maintains a pre-computed
-- tsvector for the chunk content, eliminating the need to call
-- to_tsvector('english', content) inline on every query.
-- The GIN index enables fast full-text lookups.
--
-- Idempotent: the migration runner re-applies this file on databases that
-- were set up by hand before schema_migrations existed.

-- Add the generated column (PostgreSQL computes it for all existing rows)
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsvector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- GIN index for fast tsvector lookups
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsvector
    ON chunks USING GIN (content_tsvector);
