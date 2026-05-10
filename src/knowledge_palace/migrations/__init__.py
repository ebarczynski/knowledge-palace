"""Database migrations runner."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def run_migrations(session: AsyncSession) -> None:
    """Run all SQL migrations in order."""
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    for mf in migration_files:
        print(f"Running migration: {mf.name}")
        sql = mf.read_text()
        # Split on semicolons and execute each statement
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                await session.execute(text(statement))

    await session.commit()
    print(f"Applied {len(migration_files)} migrations")
