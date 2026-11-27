-- Migration 004: normalize timestamp columns to timestamptz
--
-- Databases whose tables were first created by SQLAlchemy's create_all()
-- before migrations ran automatically carry `timestamp without time zone`
-- columns; the SQL migrations and the ORM model (db.py) use TIMESTAMPTZ.
-- Converge the old shape. Re-running against timestamptz columns is a no-op.

ALTER TABLE documents
    ALTER COLUMN created_at TYPE timestamptz,
    ALTER COLUMN updated_at TYPE timestamptz;

ALTER TABLE ingestion_log
    ALTER COLUMN timestamp TYPE timestamptz;
