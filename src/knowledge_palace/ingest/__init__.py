"""Ingestion subpackage."""

from .pipeline import IngestionPipeline
from .calibre import CalibreBridge
from .crawler import FileCrawler
from .chunker import chunk_text
from .extractors import extract_file, ExtractedDocument

__all__ = [
    "IngestionPipeline",
    "CalibreBridge",
    "FileCrawler",
    "chunk_text",
    "extract_file",
    "ExtractedDocument",
]
