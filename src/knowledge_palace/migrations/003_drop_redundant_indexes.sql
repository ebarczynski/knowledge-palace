-- Migration 003: Drop redundant indexes left behind by the two schema paths
--
-- Two sources of schema used to disagree before migrations were applied
-- automatically:
--
-- 1. 001 built a GIN index on to_tsvector('english', content). No query uses
--    that expression: every search filters on the stored generated column
--    content_tsvector (002), which has its own GIN index. The expression index
--    only cost write time and disk space on every ingest.
--
-- 2. Databases whose tables were first created by SQLAlchemy's create_all()
--    (e.g. by `kp serve` or the MCP server before any migration ran) carry a
--    second HNSW index under the ORM's old name. 001 guarantees that the
--    canonical ix_chunks_embedding_hnsw exists before this file runs.

DROP INDEX IF EXISTS ix_chunks_content_fts;
DROP INDEX IF EXISTS ix_chunks_embedding;
