"""Database schema migrations.

Migrations are plain ``NNN_name.sql`` files that live next to this module.
:func:`apply_migrations` applies the ones not yet recorded in the
``schema_migrations`` table. ``db.create_tables()`` calls it, so every process
that opens the database (``kp ingest``, ``kp serve``, ``kp mcp``) brings the
schema up to date instead of depending on a manual ``psql -f`` step.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent
MIGRATIONS_TABLE = "schema_migrations"
# Arbitrary constant identifying the advisory lock that serialises concurrent
# startups (e.g. `kp serve` and an MCP server process starting together).
MIGRATION_LOCK_KEY = 0x4B50_4C41

_CREATE_BOOKKEEPING = (  # noqa: S608 - only the constant table name is interpolated
    f"CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} ("
    "name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
)
_SELECT_APPLIED = f"SELECT name FROM {MIGRATIONS_TABLE}"  # noqa: S608 - constant table name
_RECORD_APPLIED = f"INSERT INTO {MIGRATIONS_TABLE} (name) VALUES (:name)"  # noqa: S608 - constant table name

# ``$$`` or a tagged ``$fn$`` opener.
_DOLLAR_TAG = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$")
# Enough to tell a comments-only chunk from a real statement (a stray cut inside
# a string literal can only leave text behind, never erase it all).
_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def migration_files() -> list[Path]:
    """SQL migration files in application order."""
    return sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.name[:3].isdigit())


def split_sql_statements(sql: str) -> list[str]:
    """Split a migration file into statements, preserving each one verbatim.

    Splits on ``;`` only outside single-quoted strings, quoted identifiers,
    ``$tag$`` dollar-quoted bodies (PL/pgSQL functions contain ``;``) and
    comments. Unterminated constructs run to the end of the file rather than
    raising, so a malformed file fails inside PostgreSQL with a real error.
    """
    statements: list[str] = []
    start = i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1 : i + 2]

        if ch == "-" and nxt == "-":  # line comment
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
        elif ch == "/" and nxt == "*":  # block comment
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
        elif ch == "'":
            i = _skip_string(sql, i)
        elif ch == '"':  # quoted identifier: "" escapes a quote
            i += 1
            while i < n:
                if sql[i] == '"':
                    if sql[i + 1 : i + 2] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
        elif ch == "$" and (tag := _DOLLAR_TAG.match(sql, i)) is not None:
            # Dollar-quoted body: ends at the same tag, however many ";" it holds.
            end = sql.find(tag.group(0), tag.end())
            i = n if end == -1 else end + len(tag.group(0))
        elif ch == ";":
            _append_unless_blank(statements, sql[start:i])
            i += 1
            start = i
        else:
            i += 1

    _append_unless_blank(statements, sql[start:])
    return statements


def _append_unless_blank(statements: list[str], chunk: str) -> None:
    """Append ``chunk`` as a statement unless it holds only whitespace or comments."""
    if _COMMENT.sub("", chunk).strip():
        statements.append(chunk.strip())


def _skip_string(sql: str, i: int) -> int:
    """Return the index just past the single-quoted string that starts at ``i``.

    Handles ``''`` escaping and backslash escapes in ``E'...'`` literals.
    """
    n = len(sql)
    backslash_escapes = _has_escape_string_prefix(sql, i)
    i += 1
    while i < n:
        if backslash_escapes and sql[i] == "\\":
            i += 2
        elif sql[i] == "'":
            if sql[i + 1 : i + 2] == "'":
                i += 2
            else:
                return i + 1
        else:
            i += 1
    return n


def _has_escape_string_prefix(sql: str, quote_index: int) -> bool:
    """True when the string at ``quote_index`` is an ``E'...'`` literal.

    The ``E`` must be a standalone token: ``NAME'`` or ``"nameE'`` is not one.
    """
    if quote_index == 0 or sql[quote_index - 1] not in "eE":
        return False
    before = sql[quote_index - 2] if quote_index >= 2 else ""
    return not (before.isalnum() or before in '_$"')


async def apply_migrations(conn: AsyncConnection) -> list[str]:
    """Apply every migration that is not recorded yet; return their names.

    The caller owns the transaction and must already hold
    ``pg_advisory_xact_lock(MIGRATION_LOCK_KEY)`` (see ``db.create_tables``).
    Each file is written to be a no-op on a database that already has its
    objects, which is what lets a database set up by hand before this runner
    existed converge without any bookkeeping row.

    Statements run through the driver directly rather than ``text()`` so a
    ``:word`` inside a migration's string literal or comment is never taken for
    a bind parameter.
    """
    applied: list[str] = []
    await conn.execute(text(_CREATE_BOOKKEEPING))
    done = {row[0] for row in (await conn.execute(text(_SELECT_APPLIED))).fetchall()}
    for path in migration_files():
        if path.name in done:
            continue
        for statement in split_sql_statements(path.read_text(encoding="utf-8")):
            await conn.exec_driver_sql(statement)
        await conn.execute(text(_RECORD_APPLIED), {"name": path.name})
        applied.append(path.name)
        logger.info("applied migration %s", path.name)
    return applied


__all__ = ["apply_migrations", "migration_files", "split_sql_statements", "MIGRATIONS_DIR"]
