"""CLI behaviour that the README documents."""

from __future__ import annotations

from pathlib import Path

import click.testing
import pytest

from knowledge_palace.cli import cli, pg_dump_argv
from knowledge_palace.config import Config, DatabaseConfig

RUNNER = click.testing.CliRunner()


def invoke(*args: str) -> click.testing.Result:
    return RUNNER.invoke(cli, list(args))


def sample_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text('[api]\nhost = "127.0.0.1"\nport = 8099\n')
    return path


def test_init_config_refuses_to_overwrite_an_existing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A fresh clone already contains config.toml; `kp init-config` used to
    # replace it without a word.
    monkeypatch.chdir(tmp_path)
    original = '[calibre]\nlibrary_path = "/my/real/library"\n'
    (tmp_path / "config.toml").write_text(original)
    result = invoke("init-config")
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert (tmp_path / "config.toml").read_text() == original


def test_init_config_force_writes_a_sample_that_is_safe_and_loadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    result = invoke("init-config", "--force")
    assert result.exit_code == 0, result.output
    text = (tmp_path / "config.toml").read_text()
    assert 'host = "127.0.0.1"' in text and 'host = "0.0.0.0"' not in text
    assert "cors_origins = []" in text
    config = Config.from_file(tmp_path / "config.toml")
    assert config.api.host == "127.0.0.1" and config.api.cors_origins == []


def test_eval_rejects_unknown_options():
    # With ignore_unknown_options a typo such as `--gat` was dropped silently:
    # the eval then ran without gating and exited 0.
    result = invoke("eval", "--gat")
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_a_config_that_would_hang_ingestion_is_reported_cleanly(tmp_path: Path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[chunking]\nmax_tokens = 512\noverlap_tokens = 512\n")
    result = invoke("--config", str(bad), "ingest", "--source", "files")
    assert result.exit_code == 1
    assert "invalid config" in result.output and "overlap_tokens" in result.output
    assert "Traceback" not in result.output


def test_a_config_with_a_syntax_error_is_reported_cleanly(tmp_path: Path):
    bad = tmp_path / "broken.toml"
    bad.write_text("[search\nmin_similarity = 0.7\n")
    result = invoke("--config", str(bad), "search", "query")
    assert result.exit_code == 1
    assert "invalid TOML" in result.output and "Traceback" not in result.output


def test_backup_command_targets_the_configured_port():
    # Without -p, pg_dump talked to whatever listens on 5432, which need not be
    # the database about to be re-embedded.
    argv = pg_dump_argv(DatabaseConfig(host="db.local", port=5543), "/x/dump")
    assert argv[0] == "pg_dump"
    assert argv[argv.index("-p") + 1] == "5543"
    assert argv[argv.index("-h") + 1] == "db.local"


def test_search_limit_is_validated_before_running(tmp_path: Path):
    config = sample_config(tmp_path)
    for value in ("0", "-1", "101"):
        result = invoke("--config", str(config), "search", "query", "--limit", value)
        assert result.exit_code == 2, value


def test_search_rejects_a_blank_query_cleanly(tmp_path: Path):
    # Rejected before any database is needed.
    config = sample_config(tmp_path)
    result = invoke("--config", str(config), "search", "   ")
    assert result.exit_code == 1
    assert "must not be empty" in result.output and "Traceback" not in result.output
