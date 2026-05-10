"""Configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import toml


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "knowledge_palace"
    user: str = "kp"
    password: str = "kp"

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
    provider: str = "sentence-transformers"
    batch_size: int = 32
    dimensions: int = 768
    device: str = "cpu"  # "cpu" | "cuda" | "mps"


@dataclass
class ChunkingConfig:
    strategy: str = "semantic"  # "semantic" | "fixed"
    max_tokens: int = 512
    overlap_tokens: int = 50
    respect_headings: bool = True


@dataclass
class SearchConfig:
    default_mode: str = "hybrid"  # "semantic" | "keyword" | "hybrid"
    hybrid_weight_vector: float = 0.7
    hybrid_weight_fts: float = 0.3
    default_limit: int = 10


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    calibre: CalibreConfig = field(default_factory=CalibreConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load config from TOML file."""
        raw = toml.load(path)
        config = cls()
        if "database" in raw:
            for k, v in raw["database"].items():
                if hasattr(config.database, k):
                    setattr(config.database, k, v)
        if "calibre" in raw:
            for k, v in raw["calibre"].items():
                if hasattr(config.calibre, k):
                    setattr(config.calibre, k, v)
        if "sources" in raw:
            if "directories" in raw["sources"]:
                config.sources.paths = raw["sources"]["directories"].get("paths", [])
                config.sources.file_extensions = raw["sources"]["directories"].get(
                    "file_extensions", config.sources.file_extensions
                )
            else:
                for k, v in raw["sources"].items():
                    if hasattr(config.sources, k):
                        setattr(config.sources, k, v)
        if "embedding" in raw:
            for k, v in raw["embedding"].items():
                if hasattr(config.embedding, k):
                    setattr(config.embedding, k, v)
        if "chunking" in raw:
            for k, v in raw["chunking"].items():
                if hasattr(config.chunking, k):
                    setattr(config.chunking, k, v)
        if "search" in raw:
            for k, v in raw["search"].items():
                if hasattr(config.search, k):
                    setattr(config.search, k, v)
        if "api" in raw:
            for k, v in raw["api"].items():
                if hasattr(config.api, k):
                    setattr(config.api, k, v)
        return config

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
