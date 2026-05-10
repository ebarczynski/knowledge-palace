"""MCP server for Hermes agent integration."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ..config import Config
from ..db import init_db, create_tables, close_db
from ..search.engine import SearchEngine


# Tool definitions
TOOLS = [
    Tool(
        name="search",
        description="Search the Knowledge Palace - your personal knowledge base of books, articles, and notes. Supports semantic, keyword, and hybrid search.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "keyword", "hybrid"],
                    "description": "Search mode (default: hybrid)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 10)",
                },
                "source": {
                    "type": "string",
                    "enum": ["calibre", "file"],
                    "description": "Filter by source type",
                },
                "author": {
                    "type": "string",
                    "description": "Filter by author name",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="similar",
        description="Find content in Knowledge Palace that is semantically similar to the given text. Useful for finding related passages or checking if a problem was already solved.",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to find similar content for",
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum similarity threshold 0-1 (default: 0.7)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 5)",
                },
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="ask",
        description="Ask a question and get an answer grounded in your personal knowledge base. Returns relevant passages from your books and notes.",
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to answer using your knowledge base",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max context passages (default: 5)",
                },
            },
            "required": ["question"],
        },
    ),
]


def format_results(results) -> str:
    """Format search results as readable text."""
    if not results.results:
        return "No results found."

    lines = [f"Found {results.total} results ({results.mode} search):\n"]
    for i, r in enumerate(results.results, 1):
        lines.append(f"--- Result {i} (score: {r.score:.3f}) ---")
        lines.append(f"Title: {r.title}")
        if r.author:
            lines.append(f"Author: {r.author}")
        lines.append(f"Source: {r.source}")
        if r.tags:
            lines.append(f"Tags: {', '.join(r.tags)}")
        lines.append(f"\n{r.content[:500]}")
        if len(r.content) > 500:
            lines.append("...")
        lines.append("")
    return "\n".join(lines)


async def run_mcp_server(config: Config) -> None:
    """Run the MCP server."""
    await init_db(config.database.url)
    await create_tables()
    search_engine = SearchEngine(config)

    server = Server("knowledge-palace")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "search":
            results = await search_engine.search(
                query=arguments["query"],
                mode=arguments.get("mode"),
                limit=arguments.get("limit"),
                source_filter=arguments.get("source"),
                author=arguments.get("author"),
            )
            return [TextContent(type="text", text=format_results(results))]

        elif name == "similar":
            results = await search_engine.find_similar(
                text=arguments["text"],
                threshold=arguments.get("threshold", 0.7),
                limit=arguments.get("limit", 5),
            )
            return [TextContent(type="text", text=format_results(results))]

        elif name == "ask":
            # RAG-style: search + format as context
            results = await search_engine.search(
                query=arguments["question"],
                mode="hybrid",
                limit=arguments.get("limit", 5),
            )
            if not results.results:
                return [TextContent(type="text", text="No relevant information found in your knowledge base.")]

            lines = ["Based on your knowledge base, here are the most relevant passages:\n"]
            for i, r in enumerate(results.results, 1):
                lines.append(f"[{i}] From '{r.title}' by {r.author or 'Unknown'} (relevance: {r.score:.3f}):")
                lines.append(r.content)
                lines.append("")
            return [TextContent(type="text", text="\n".join(lines))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(config_path: str | None = None):
    """Entry point for MCP server."""
    config = Config.load()
    asyncio.run(run_mcp_server(config))
