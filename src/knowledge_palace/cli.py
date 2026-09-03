"""CLI interface for Knowledge Palace."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import Config

console = Console()
# MCP uses stdio for JSON-RPC, so anything meant for humans on that transport
# must go to stderr to avoid corrupting the protocol stream on stdout.
err_console = Console(stderr=True)


def load_config(config_path: str | None) -> Config:
    """Load config from file or defaults."""
    if config_path:
        return Config.from_file(config_path)
    return Config.load()


@click.group()
@click.option("--config", "-c", default=None, help="Path to config.toml")
@click.pass_context
def cli(ctx, config):
    """Knowledge Palace - Personal Knowledge Base with Semantic Search."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.option("--source", "-s", type=click.Choice(["calibre", "files", "all"]), default="all")
@click.option("--reindex", is_flag=True, help="Reprocess already-indexed documents")
@click.pass_context
def ingest(ctx, source: str, reindex: bool):
    """Ingest documents from Calibre and/or file directories."""
    from .ingest.pipeline import IngestionPipeline

    config = ctx.obj["config"]
    pipeline = IngestionPipeline(config)

    console.print("\n[bold cyan]🏛️  Knowledge Palace — Ingestion[/bold cyan]\n")

    results = asyncio.run(pipeline.run(source=source, reindex=reindex))

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Documents processed: [green]{results['extracted']}[/green]")
    console.print(f"  Chunks created: [green]{results['chunked']}[/green]")
    console.print(f"  Chunks embedded: [green]{results['embedded']}[/green]")
    console.print(f"  Errors: [red]{results['errors']}[/red]")


@cli.command()
@click.argument("query")
@click.option("--mode", "-m", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid")
@click.option("--limit", "-n", default=10)
@click.option("--source", "-s", type=click.Choice(["calibre", "file"]), default=None)
@click.pass_context
def search(ctx, query: str, mode: str, limit: int, source: str | None):
    """Search the knowledge base."""
    from .db import init_db, close_db
    from .search.engine import SearchEngine

    config = ctx.obj["config"]

    async def _search():
        await init_db(config.database.url)
        engine = SearchEngine(config)
        results = await engine.search(query, mode=mode, limit=limit, source_filter=source)
        await close_db()
        return results

    results = asyncio.run(_search())

    console.print(f"\n[bold cyan]Search: '{query}' ({mode}) — {results.total} results[/bold cyan]\n")

    for i, r in enumerate(results.results, 1):
        console.print(f"[bold]{i}.[/bold] [cyan]{r.title}[/cyan] by {r.author or 'Unknown'}")
        console.print(f"   Score: {r.score:.3f} | Source: {r.source} | Tags: {', '.join(r.tags)}")
        console.print(f"   {r.content[:200]}...")
        if r.highlights:
            for h in r.highlights[:2]:
                console.print(f"   [yellow]→ {h}[/yellow]")
        console.print()


@cli.command()
@click.option("--host", default=None, help="Override host")
@click.option("--port", default=None, type=int, help="Override port")
@click.pass_context
def serve(ctx, host: str | None, port: int | None):
    """Start the API server (REST + GraphQL)."""
    import uvicorn
    from .api.app import create_app

    config = ctx.obj["config"]
    host = host or config.api.host
    port = port or config.api.port

    app = create_app(config)

    console.print(f"\n[bold cyan]🏛️  Knowledge Palace API[/bold cyan]")
    console.print(f"  REST:    http://{host}:{port}/api/v1/")
    console.print(f"  GraphQL: http://{host}:{port}/graphql")
    console.print(f"  Docs:    http://{host}:{port}/docs\n")

    # ws="none": the API is HTTP-only (GraphQL has no Subscriptions), so
    # disable WebSocket support. This also prevents uvicorn from importing
    # the deprecated websockets.legacy module (DeprecationWarning on startup).
    uvicorn.run(app, host=host, port=port, ws="none")


@cli.command()
@click.pass_context
def mcp(ctx):
    """Start the MCP server for Hermes agent integration."""
    from .mcp_server.server import run_mcp_server

    config = ctx.obj["config"]
    # MCP uses stdio for JSON-RPC: keep stdout clean and log to stderr instead.
    err_console.print("[dim]Starting Knowledge Palace MCP server...[/dim]")
    asyncio.run(run_mcp_server(config))


@cli.command("calibre-list")
@click.pass_context
def calibre_list(ctx):
    """List books in the Calibre library."""
    from .ingest.calibre import CalibreBridge

    config = ctx.obj["config"]
    if not config.calibre.library_path:
        console.print("[red]No Calibre library path configured[/red]")
        return

    bridge = CalibreBridge(config.calibre.library_path)
    bridge.list_books()


@cli.command("init-config")
@click.argument("path", default="config.toml")
def init_config(path: str):
    """Create a sample configuration file."""
    sample = """# Knowledge Palace Configuration

[database]
host = "localhost"
port = 5432
name = "knowledge_palace"
user = "kp"
password = "kp"

[calibre]
library_path = "/path/to/Calibre Library"
sync_interval_hours = 1

[sources]
paths = [
    "~/notes",
    "~/docs",
]
file_extensions = [".md", ".txt", ".org", ".rst"]

[embedding]
model = "nomic-ai/nomic-embed-text-v1.5"
# Backend is auto-selected: ONNX Runtime (fastest for nomic on Apple Silicon),
# then PyTorch. MLX is used automatically if the model architecture is
# supported by mlx-embeddings (nomic_bert is not).
provider = "auto"
batch_size = 32
dimensions = 768
device = "cpu"

[chunking]
strategy = "semantic"
max_tokens = 512
overlap_tokens = 50
respect_headings = true

[search]
default_mode = "hybrid"
hybrid_weight_vector = 0.7
hybrid_weight_fts = 0.3
default_limit = 10

[api]
host = "0.0.0.0"
port = 8080
"""
    Path(path).write_text(sample)
    console.print(f"[green]Config written to {path}[/green]")


if __name__ == "__main__":
    cli()
