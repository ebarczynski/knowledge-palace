"""Multi-query retrieval helpers: query expansion + result fusion.

Used by ``SearchEngine.retrieve_context`` to gather broad, de-duplicated
context for an agent harness in a single call.
"""

from .expand import expand_queries
from .fuse import fuse, RRF_K

__all__ = ["expand_queries", "fuse", "RRF_K"]
