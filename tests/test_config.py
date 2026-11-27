"""Config loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from knowledge_palace.config import Config, ConfigError

OVERLAP_STALLS = "overlap_tokens must satisfy"


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


def test_defaults_bind_to_loopback_and_send_no_cors_headers():
    # The API has no authentication: 0.0.0.0 exposed it to the whole network,
    # and allow_origins=["*"] let any web page read the library.
    assert Config().api.host == "127.0.0.1"
    assert Config().api.cors_origins == []


def test_sections_are_applied(tmp_path: Path):
    config = Config.from_file(write(tmp_path, """
[database]
host = "db.example"
port = 6543

[calibre]
library_path = "/books"

[embedding]
model = "other/model"
batch_size = 8

[search]
default_mode = "keyword"
min_similarity = 0.0
"""))
    assert config.database.host == "db.example" and config.database.port == 6543
    assert config.database.url == "postgresql+asyncpg://kp:kp@db.example:6543/knowledge_palace"
    assert config.calibre.library_path == "/books"
    assert config.embedding.model == "other/model" and config.embedding.batch_size == 8
    assert config.search.default_mode == "keyword" and config.search.min_similarity == 0.0


def test_legacy_sources_directories_table_is_still_accepted(tmp_path: Path):
    config = Config.from_file(write(tmp_path, '[sources.directories]\npaths = ["/a"]\n'))
    assert config.sources.paths == ["/a"]


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("[chunking]\nmax_tokens = 512\noverlap_tokens = 512\n", OVERLAP_STALLS),
        ("[chunking]\nmax_tokens = 512\noverlap_tokens = 600\n", OVERLAP_STALLS),
        ("[chunking]\nmax_tokens = 0\n", "max_tokens must be a positive integer"),
        ("[chunking]\nmax_tokens = -5\n", "max_tokens must be a positive integer"),
        ("[chunking]\noverlap_tokens = -1\n", OVERLAP_STALLS),
        ("[chunking]\nstrategy = \"sliding\"\n", "strategy must be one of"),
        ('[search]\ndefault_mode = "fuzzy"\n', "default_mode must be one of"),
        ("[search]\nmin_similarity = 1.5\n", "min_similarity must be a cosine value"),
        ('[search]\nmin_similarity = "high"\n', "min_similarity must be a cosine value"),
        ("[search]\nhybrid_weight_vector = -0.1\n", "non-negative number"),
        ("[api]\nport = 70000\n", "port must be a valid port"),
    ],
)
def test_unusable_values_are_rejected_when_the_config_is_loaded(tmp_path: Path, body: str, fragment: str):
    # Before the fix every value was accepted blindly; an overlap >= max_tokens
    # then hung `kp ingest` forever.
    with pytest.raises(ConfigError, match=fragment):
        Config.from_file(write(tmp_path, body))


def test_unknown_key_is_reported_not_silently_dropped(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    # A typo such as `min_similarty` used to keep the default without a word.
    with caplog.at_level(logging.WARNING, logger="knowledge_palace.config"):
        config = Config.from_file(write(tmp_path, "[search]\nmin_similarty = 0.5\nmin_similarity = 0.6\n"))
    assert config.search.min_similarity == 0.6
    assert any("min_similarty" in record.getMessage() for record in caplog.records)


def test_valid_edge_values_pass(tmp_path: Path):
    config = Config.from_file(write(tmp_path, "[chunking]\nmax_tokens = 1\noverlap_tokens = 0\n"))
    assert config.chunking.max_tokens == 1 and config.chunking.overlap_tokens == 0


def test_discovery_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    here = tmp_path / "cwd"
    here.mkdir()
    home = tmp_path / "home"
    (home / ".config" / "knowledge-palace").mkdir(parents=True)
    monkeypatch.chdir(here)
    monkeypatch.setattr(Path, "home", lambda: home)
    (home / ".config" / "knowledge-palace" / "config.toml").write_text("[api]\nport = 9001\n")
    assert Config.load().api.port == 9001
    (here / "config.toml").write_text("[api]\nport = 9002\n")
    assert Config.load().api.port == 9002, "./config.toml wins over the home config"
