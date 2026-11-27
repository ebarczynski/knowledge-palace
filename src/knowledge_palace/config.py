"""Configuration management."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import toml

logger = logging.getLogger(__name__)

SEARCH_MODES = ("semantic", "keyword", "hybrid")
CHUNK_STRATEGIES = ("semantic", "fixed")


class ConfigError(ValueError):
    """Raised when a config file contains an invalid or unusable value."""


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "knowledge_palace"
    user: str = "kp"
    password: str = "kp"  # noqa: S105 - local-development default, documented in README

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class CalibreConfig:
    library_path: str = ""
    sync_interval_hours: int = 1


@dataclass
class SourceConfig:
    paths: list[str] = field(default_factory=list)
    file_extensions: list[str] = field(
        default_factory=lambda: [".md", ".txt", ".org", ".rst", ".adoc"]
    )


@dataclass
class EmbeddingConfig:
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    provider: str = "auto"
    batch_size: int = 32
    dimensions: int = 768
    device: str = "cpu"  # "cpu" | "cuda" | "mps"


@dataclass
class ChunkingConfig:
    strategy: str = "semantic"  # "semantic" | "fixed"
    max_tokens: int = 512
    overlap_tokens: int = 50
    respect_headings: bool = True

    def validate(self) -> None:
        """Reject values that would make the chunker loop forever or drop text.

        ``chunk_fixed`` advances by ``max_tokens - overlap_tokens`` per chunk, so
        the step must be at least one token; a non-positive ``max_tokens`` makes
        every slice empty.
        """
        if self.strategy not in CHUNK_STRATEGIES:
            raise ConfigError(
                f"[chunking].strategy must be one of {CHUNK_STRATEGIES}, got {self.strategy!r}"
            )
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ConfigError(f"[chunking].max_tokens must be a positive integer, got {self.max_tokens!r}")
        if not isinstance(self.overlap_tokens, int) or not 0 <= self.overlap_tokens < self.max_tokens:
            raise ConfigError(
                "[chunking].overlap_tokens must satisfy 0 <= overlap_tokens < max_tokens "
                f"(a larger overlap stalls ingestion, a negative one skips text); got "
                f"overlap_tokens={self.overlap_tokens!r}, max_tokens={self.max_tokens!r}"
            )


@dataclass
class SearchConfig:
    default_mode: str = "hybrid"  # "semantic" | "keyword" | "hybrid"
    hybrid_weight_vector: float = 0.7
    hybrid_weight_fts: float = 0.3
    default_limit: int = 10
    # Quality floor for vector matches (cosine similarity). Results below this
    # are dropped from semantic search and from the vector leg of hybrid search,
    # so queries about topics absent from the KB return nothing instead of
    # weakly-related junk. Values <= 0 disable the floor entirely.
    # Calibrated for nomic-embed-text-v1.5 WITH task prefixes: a small sample of
    # in-domain queries bottomed out at 0.720 and out-of-domain probes peaked at
    # 0.718. The repo's own eval set contains in-domain cases at 0.70-0.71 that
    # the floor suppresses -- see evals/README.md before changing this.
    min_similarity: float = 0.72

    def validate(self) -> None:
        if self.default_mode not in SEARCH_MODES:
            raise ConfigError(f"[search].default_mode must be one of {SEARCH_MODES}, got {self.default_mode!r}")
        if not isinstance(self.default_limit, int) or self.default_limit < 1:
            raise ConfigError(f"[search].default_limit must be a positive integer, got {self.default_limit!r}")
        if not _is_finite_number(self.min_similarity) or not -1.0 <= self.min_similarity <= 1.0:
            raise ConfigError(
                f"[search].min_similarity must be a cosine value in [-1, 1] (<= 0 disables it), "
                f"got {self.min_similarity!r}"
            )
        for name in ("hybrid_weight_vector", "hybrid_weight_fts"):
            value = getattr(self, name)
            if not _is_finite_number(value) or value < 0:
                raise ConfigError(f"[search].{name} must be a non-negative number, got {value!r}")


@dataclass
class APIConfig:
    # Loopback by default: the API has no authentication. Set "0.0.0.0" only
    # behind a firewall or reverse proxy.
    host: str = "127.0.0.1"
    port: int = 8080
    # Origins allowed to call the API from a browser. Empty (the default) sends
    # no CORS headers, so web pages cannot read the library through the browser
    # of whoever is running `kp serve`.
    cors_origins: list[str] = field(default_factory=list)


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    calibre: CalibreConfig = field(default_factory=CalibreConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def validate(self) -> None:
        self.chunking.validate()
        self.search.validate()
        if not 1 <= self.api.port <= 65535:
            raise ConfigError(f"[api].port must be a valid port, got {self.api.port!r}")

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load config from a TOML file."""
        raw = toml.load(path)
        config = cls()
        for section_name in ("database", "calibre", "embedding", "chunking", "search", "api"):
            config._apply(section_name, raw.get(section_name, {}))
        # ``[sources]`` is the documented shape; a legacy ``[sources.directories]``
        # table is still accepted.
        config._apply("sources", raw.get("sources", {}).get("directories", raw.get("sources", {})))
        config.validate()
        return config

    def _apply(self, section_name: str, values: dict) -> None:
        section = getattr(self, section_name)
        known = {f.name for f in fields(section)}
        for key, value in values.items():
            if key not in known:
                # Never fail a working install over an unknown key, but do not
                # ignore it silently either: a typo here quietly keeps a default.
                logger.warning("config: unknown key %r in [%s] is ignored", key, section_name)
                continue
            setattr(section, key, value)

    @classmethod
    def load(cls) -> Config:
        """Load config from default locations."""
        candidates = [
            Path("config.toml"),
            Path.home() / ".config" / "knowledge-palace" / "config.toml",
            Path.home() / ".knowledge-palace" / "config.toml",
        ]
        for p in candidates:
            if p.exists():
                return cls.from_file(p)
        return cls()


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
