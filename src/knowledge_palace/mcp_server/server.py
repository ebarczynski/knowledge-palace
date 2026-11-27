"""MCP (Model Context Protocol) server exposing the knowledge base to agents.

Transport is stdio: JSON-RPC goes over stdout, so everything meant for human
eyes (status lines, tracebacks) must go to stderr. The tools share one
``SearchEngine`` with the REST and GraphQL API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .. import __version__
from ..config import Config
from ..db import create_tables, init_db
from ..search.engine import MAX_QUERY_CHARS, MAX_SIMILAR_CHARS, InvalidQueryError, SearchEngine, SearchResults

logger = logging.getLogger(__name__)

#: Characters of each chunk that ``search``/``similar`` return unless told otherwise.
DEFAULT_MAX_CHARS = 500
#: Upper bound for the ``max_chars`` argument.
MAX_OUTPUT_CHARS = 20_000

# What ``score`` means per mode; it is not comparable across modes.
_SCORE_LABEL = {"semantic": "cosine", "similar": "cosine", "keyword": "text rank", "hybrid": "rrf"}

_OBJECT = "object"


def _int(description: str, minimum: int, maximum: int) -> dict[str, Any]:
    return {"type": "integer", "description": description, "minimum": minimum, "maximum": maximum}


def _text(description: str, max_length: int, minimum: int = 1) -> dict[str, Any]:
    return {"type": "string", "description": description, "minLength": minimum, "maxLength": max_length}


def _enum(description: str, values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": values, "description": description}


_AUTHOR = _text("Filter by author name (case-insensitive substring)", MAX_QUERY_CHARS)
_SOURCE = _enum("Filter by source type", ["calibre", "file"])
_TRUNCATION = ("Max characters of each hit's text to return (default 500). "
               "0 returns the whole chunk.")
_IDS_NOTE = ("Each hit comes with its chunk id and document id and a provenance line "
             "(format, chunk index, file name).")

CONTEXT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": _OBJECT,
    "properties": {
        "query": {"type": "string"},
        "total": {"type": "integer"},
        "mode": {"type": "string", "enum": ["context"]},
        "results": {
            "type": "array",
            "items": {
                "type": _OBJECT,
                "properties": {
                    "text": {"type": "string"},
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                    "source": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "score": {"type": "number"},
                    "provenance": {"type": _OBJECT},
                },
                "required": ["text", "title", "source", "score"],
            },
        },
    },
    "required": ["query", "total", "mode", "results"],
}

# Tool definitions. Every schema is closed (additionalProperties: false) and
# bounded, so out-of-range limits and misspelled arguments are rejected before
# any code runs.
TOOLS = [
    Tool(
        name="search",
        description=(
            "Search the Knowledge Palace - your personal knowledge base of books, articles, "
            "and notes. Supports semantic, keyword, and hybrid search. Returns readable text, "
            "one block per hit with its score, ids, provenance and the start of its text. "
            + _IDS_NOTE + " Pass max_chars=0 for full chunks."
        ),
        inputSchema={
            "type": _OBJECT,
            "properties": {
                "query": _text("Search query text", MAX_QUERY_CHARS),
                "mode": _enum("Search mode (default: hybrid)", ["semantic", "keyword", "hybrid"]),
                "limit": _int("Max results (default: 10)", 1, 100),
                "source": _SOURCE,
                "author": _AUTHOR,
                "max_chars": _int(_TRUNCATION, 0, MAX_OUTPUT_CHARS),
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="similar",
        description=(
            "Find content in Knowledge Palace that is semantically similar to the given "
            "text. Useful for finding related passages or checking if a problem was already "
            "solved. " + _IDS_NOTE
        ),
        inputSchema={
            "type": _OBJECT,
            "properties": {
                "text": _text("Text to find similar content for", MAX_SIMILAR_CHARS),
                "threshold": {"type": "number", "description": "Minimum cosine similarity, 0-1 (default: 0.7)",
                              "minimum": 0, "maximum": 1},
                "limit": _int("Max results (default: 5)", 1, 50),
                "max_chars": _int(_TRUNCATION, 0, MAX_OUTPUT_CHARS),
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="ask",
        description=(
            "Retrieve passages from your personal knowledge base that help answer a "
            "question (hybrid search). No answer is generated: you read the passages."
        ),
        inputSchema={
            "type": _OBJECT,
            "properties": {
                "question": _text("The question to find passages for", MAX_QUERY_CHARS),
                "limit": _int("Max passages (default: 5)", 1, 50),
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="context",
        description=(
            "Gather broad, de-duplicated context for a task in one call. Expands the "
            "query into synonym/paraphrase variants, runs each across keyword + "
            "semantic + hybrid search, and fuses results via Reciprocal Rank Fusion. "
            "Returns structured JSON: {query, total, mode, results: [{text, title, source, "
            "score, author?, tags?, provenance?}...]} — built for programmatic grounding "
            "(decision engines, RAG pipelines), not for prose answers."
        ),
        inputSchema={
            "type": _OBJECT,
            "properties": {
                "query": _text("The task or topic to gather context for", MAX_QUERY_CHARS),
                "limit": _int("Max fused results (default: 10)", 1, 50),
                "max_variants": _int("Cap on query variants including the original, to bound "
                                     "retrieval cost (default: 6)", 1, 12),
                "source": _SOURCE,
                "author": _AUTHOR,
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        outputSchema=CONTEXT_OUTPUT_SCHEMA,
    ),
]


def format_results(results: SearchResults, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Format search results as readable text, with ids and provenance per hit."""
    if not results.results:
        return "No results found."

    label = _SCORE_LABEL.get(results.mode, "score")
    lines = [f"Found {results.total} results ({results.mode} search, {label} score):\n"]
    for i, r in enumerate(results.results, 1):
        lines.append(f"--- Result {i} ({label}: {r.score:.3f}) ---")
        lines.append(f"Chunk: {r.chunk_id}")
        lines.append(f"Document: {r.document_id}")
        lines.append(f"Title: {r.title}")
        if r.author:
            lines.append(f"Author: {r.author}")
        lines.append(f"Source: {r.source}")
        if r.tags:
            lines.append(f"Tags: {', '.join(r.tags)}")
        if r.provenance:
            lines.append(f"Provenance: {_provenance(r.provenance)}")
        lines.append(f"\n{_clip(r.content, max_chars)}\n")
    return "\n".join(lines)


def _provenance(prov: dict) -> str:
    parts = []
    if prov.get("format"):
        parts.append(str(prov["format"]))
    if prov.get("chunk_index") is not None:
        parts.append(f"chunk {prov['chunk_index']}")
    if prov.get("page_count") is not None:
        parts.append(f"{prov['page_count']} pages")
    if prov.get("file_path"):
        # the file name, not the full path
        parts.append(str(prov["file_path"]).rsplit("/", 1)[-1])
    if prov.get("calibre_id") is not None:
        parts.append(f"calibre {prov['calibre_id']}")
    return " · ".join(parts)


def _clip(content: str, max_chars: int) -> str:
    """Cut ``content`` to ``max_chars`` characters, saying so when it does."""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return (
        f"{content[:max_chars]}\n[... truncated: showing {max_chars} of {len(content)} characters. "
        "Call again with max_chars=0 for the whole chunk.]"
    )


def context_payload(results: SearchResults) -> dict[str, Any]:
    """The ``context`` tool's structured result."""
    hits = []
    for r in results.results:
        hit: dict[str, Any] = {
            "text": r.content,
            "title": r.title,
            "source": r.source,
            "score": round(r.score, 4),
        }
        if r.author:
            hit["author"] = r.author
        if r.tags:
            hit["tags"] = r.tags
        if r.provenance:
            hit["provenance"] = r.provenance
        hits.append(hit)
    return {"query": results.query, "total": results.total, "mode": results.mode, "results": hits}


def ask_text(results: SearchResults) -> str:
    """The ``ask`` tool's passages. The score is an RRF score, not a relevance."""
    if not results.results:
        return "No relevant information found in your knowledge base."
    lines = ["Based on your knowledge base, here are the most relevant passages:\n"]
    for i, r in enumerate(results.results, 1):
        lines.append(f"[{i}] From '{r.title}' by {r.author or 'Unknown'} (rrf: {r.score:.3f}):")
        lines.append(r.content)
        lines.append("")
    return "\n".join(lines)


def _first_line(error: BaseException) -> str:
    return str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__


async def call_tool(engine: SearchEngine, name: str, arguments: dict[str, Any]):
    """Run one tool. Raises ``InvalidQueryError`` for anything the tools cannot run with."""
    if name == "search":
        results = await engine.search(
            query=arguments["query"],
            mode=arguments.get("mode"),
            limit=arguments.get("limit"),
            source_filter=arguments.get("source"),
            author=arguments.get("author"),
        )
        return [TextContent(type="text", text=format_results(results, arguments.get("max_chars", DEFAULT_MAX_CHARS)))]

    if name == "similar":
        results = await engine.find_similar(
            text=arguments["text"],
            threshold=arguments.get("threshold", 0.7),
            limit=arguments.get("limit", 5),
        )
        return [TextContent(type="text", text=format_results(results, arguments.get("max_chars", DEFAULT_MAX_CHARS)))]

    if name == "ask":
        results = await engine.search(query=arguments["question"], mode="hybrid", limit=arguments.get("limit", 5))
        return [TextContent(type="text", text=ask_text(results))]

    if name == "context":
        results = await engine.retrieve_context(
            query=arguments["query"],
            limit=arguments.get("limit", 10),
            max_variants=arguments.get("max_variants", 6),
            source_filter=arguments.get("source"),
            author=arguments.get("author"),
        )
        payload = context_payload(results)
        # Both forms of the same JSON: structured for programmatic clients, text for others.
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))], payload

    raise InvalidQueryError(f"Unknown tool: {name!r}. Available: {', '.join(t.name for t in TOOLS)}")


async def run_mcp_server(config: Config) -> None:
    """Run the MCP server."""
    await init_db(config.database.url)
    await create_tables()
    engine = SearchEngine(config)

    server = Server("knowledge-palace")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def handle_call(name: str, arguments: dict[str, Any]):
        try:
            return await call_tool(engine, name, arguments)
        except InvalidQueryError as e:
            # The SDK turns a raised exception into an error result carrying str(e).
            raise ValueError(str(e)) from None
        except Exception as e:
            # Keep the traceback on stderr; give the agent a one-line reason.
            logger.exception("Tool %s failed", name)
            raise RuntimeError(f"{name} failed: {type(e).__name__}: {_first_line(e)}") from e

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="knowledge-palace",
                server_version=__version__,
                capabilities=server.get_capabilities(NotificationOptions(), {}),
            ),
        )


def main(config_path: str | None = None) -> None:
    """Entry point for MCP server."""
    config = Config.from_file(config_path) if config_path else Config.load()
    asyncio.run(run_mcp_server(config))


__all__ = ["TOOLS", "call_tool", "context_payload", "format_results", "main", "run_mcp_server"]
