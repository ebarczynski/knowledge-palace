"""Ingestion subpackage."""

from .calibre import CalibreBridge
from .chunker import chunk_text
from .crawler import FileCrawler
from .extractors import ExtractedDocument, extract_file
from .pipeline import IngestionPipeline

__all__ = [
    "IngestionPipeline",
    "CalibreBridge",
    "FileCrawler",
    "chunk_text",
    "extract_file",
    "ExtractedDocument",
]
