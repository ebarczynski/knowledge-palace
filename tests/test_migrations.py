"""Migration runner: statement splitting (pure) and file ordering."""

from __future__ import annotations

import pytest

from knowledge_palace.migrations import MIGRATIONS_DIR, migration_files, split_sql_statements


def test_migrations_dir_points_at_the_sql_files():
    # The audited revision resolved to .../migrations/migrations, which does
    # not exist, so run_migrations() silently "applied" zero files.
    assert MIGRATIONS_DIR.is_dir(), MIGRATIONS_DIR
    assert (MIGRATIONS_DIR / "001_initial.sql").is_file()
    assert [p.name for p in migration_files()] == [
        "001_initial.sql",
        "002_tsvector_column.sql",
        "003_drop_redundant_indexes.sql",
    ]


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        # a PL/pgSQL body holds semicolons that must not split the CREATE FUNCTION
        (
            # a PL/pgSQL body holds semicolons that must not split the statement
            "CREATE FUNCTION f() RETURNS trigger AS $$ BEGIN NEW.a := 1; RETURN NEW; END; $$ LANGUAGE plpgsql;"
            "\nSELECT 1;",
            2,
        ),
        (
            "CREATE FUNCTION g() RETURNS int AS $body$ BEGIN RETURN 1; END; $body$ LANGUAGE sql;",
            1,
        ),
        ("INSERT INTO t VALUES ('a;b', 'it''s; fine', 'note:colon');", 1),
        ('SELECT "weird;name" FROM x; /* block ; comment */ SELECT 2', 2),
        ("SELECT E'it\\'s; ok'; -- line ; comment\nSELECT 3", 2),
        ("SELECT $$a ; b$$; SELECT 2", 2),
        ("-- only a comment ; here\n/* and a block ; */", 0),
        ("SELECT 1;;  ;SELECT 2", 2),
    ],
)
def test_split_respects_strings_identifiers_dollar_quotes_and_comments(sql: str, expected: int):
    statements = split_sql_statements(sql)
    assert len(statements) == expected, statements


def test_split_preserves_statement_text_verbatim():
    sql = "INSERT INTO t VALUES ('keep :this and ; this');\n"
    assert split_sql_statements(sql) == ["INSERT INTO t VALUES ('keep :this and ; this')"]


def test_each_shipped_migration_splits_into_the_expected_statements():
    counts = {path.name: len(split_sql_statements(path.read_text())) for path in migration_files()}
    assert counts == {"001_initial.sql": 13, "002_tsvector_column.sql": 2, "003_drop_redundant_indexes.sql": 2}


def test_shipped_migrations_are_idempotent():
    """No bare CREATE TRIGGER / ADD COLUMN / CREATE INDEX without their IF NOT EXISTS forms."""
    text = "\n".join(path.read_text() for path in migration_files())
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("CREATE INDEX ", "ADD COLUMN ")):
            assert "IF NOT EXISTS" in stripped, stripped
        if stripped.startswith("CREATE TRIGGER"):
            raise AssertionError(f"re-applying fails on an existing trigger: {stripped}")
    assert "CREATE OR REPLACE TRIGGER" in text
