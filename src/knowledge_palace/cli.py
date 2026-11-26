"""CLI interface for Knowledge Palace."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import click
import toml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import Config, ConfigError, DatabaseConfig

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
    try:
        ctx.obj["config"] = load_config(config)
    except ConfigError as e:
        # A bad value in config.toml is a user problem: show it, not a traceback.
        raise click.ClickException(f"invalid config: {e}") from e
    except OSError as e:
        raise click.ClickException(f"cannot read config: {e}") from e
    except toml.TomlDecodeError as e:
        raise click.ClickException(f"invalid TOML in config: {e}") from e


@cli.command()
@click.option("--source", "-s", type=click.Choice(["calibre", "files", "all"]), default="all")
@click.option("--reindex", is_flag=True,
              help="Rebuild documents that are already indexed (after changing chunking "
                   "or embedding settings), not just new or changed ones")
@click.pass_context
def ingest(ctx, source: str, reindex: bool):
    """Ingest documents from Calibre and/or file directories.

    Reconciles the index with the sources: new and changed documents are
    indexed, documents whose file or Calibre book is gone are removed.
    """
    from .ingest.pipeline import IngestionPipeline

    config = ctx.obj["config"]
    pipeline = IngestionPipeline(config)

    console.print("\n[bold cyan]🏛️  Knowledge Palace — Ingestion[/bold cyan]\n")

    results = asyncio.run(pipeline.run(source=source, reindex=reindex))

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Documents processed: [green]{results['extracted']}[/green]")
    console.print(f"  Chunks created: [green]{results['chunked']}[/green]")
    console.print(f"  Chunks embedded: [green]{results['embedded']}[/green]")
    console.print(f"  Unchanged or duplicate: [green]{results['skipped']}[/green]")
    tone = "red" if results["errors"] else "green"
    console.print(f"  Errors: [{tone}]{results['errors']}[/{tone}]")
    if results["errors"]:
        ctx.exit(1)


def pg_dump_argv(database: DatabaseConfig, backup_path: str) -> list[str]:
    """``pg_dump`` command for the pre-reindex backup.

    The port is part of it: without it the backup would silently target
    whatever listens on 5432, which may not be the database being re-embedded.
    """
    return [
        "pg_dump", "-h", database.host, "-p", str(database.port), "-U", database.user,
        "-Fc", "-d", database.name, "-f", backup_path,
    ]


@cli.command("reindex-embeddings")
@click.option("--skip-backup", is_flag=True, help="Skip the automatic pg_dump backup")
@click.option("--start-after", default=0, type=int, metavar="ROWS",
              help="Resume: skip the first ROWS chunks (id-ordered). Use the "
                   "'rows re-embedded' count from an interrupted run.")
@click.option("--provider", type=click.Choice(["auto", "onnx", "pytorch"]), default="auto",
              help="Force an embedding backend (pytorch = CUDA/MPS GPU path)")
@click.option("--batch", default=32, type=int, metavar="N", help="Embedding batch size")
@click.pass_context
def reindex_embeddings(ctx, skip_backup: bool, start_after: int, provider: str, batch: int):
    """Re-embed all chunks with current embedding settings.

    Required after changing [embedding] settings that affect vectors (e.g.
    nomic task prefixes or a different model of the same dimension). Creates
    a pg_dump backup of the database first unless --skip-backup is given.
    """
    import subprocess
    import tempfile
    import time as _time

    from sqlalchemy import text as sql_text

    from .db import close_db, init_db, session_context
    from .embedding.service import EmbeddingService

    config = ctx.obj["config"]
    console.print("\n[bold cyan]🏛️  Knowledge Palace — Embedding Re-index[/bold cyan]\n")

    async def run():
        await init_db(config.database.url)

        # 1. Backup (safety: old embeddings are overwritten in place). A private
        #    directory, not a predictable name in a shared /tmp.
        if not skip_backup:
            backup_path = str(Path(tempfile.mkdtemp(prefix="knowledge_palace_pre_reindex_")) / "dump")
            console.print(f"[dim]Backing up database to {backup_path}...[/dim]")
            # argv only, never a shell; the process runs before any async work starts.
            r = subprocess.run(  # noqa: ASYNC221, S603
                pg_dump_argv(config.database, backup_path),
                env={"PGPASSWORD": config.database.password, "PATH": os.environ["PATH"]},
                capture_output=True, text=True, check=False,
            )
            if r.returncode != 0:
                console.print(f"[red]Backup failed: {r.stderr.strip()}[/red]")
                await close_db()
                return
            console.print("[green]✓ backup written[/green]")

        # 2. Stream (id, content) in batches and re-embed.
        #    Writes are batched: one UPDATE ... FROM (VALUES ...) statement +
        #    one commit per batch. Per-row UPDATEs against the HNSW index are
        #    ~25x slower (134k fsyncs).
        if provider != "auto":
            config.embedding.provider = provider
        svc = EmbeddingService(config.embedding)
        batch_size = batch
        offset = start_after
        done = start_after
        errors = 0
        started = _time.perf_counter()

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(),
            TimeRemainingColumn(), console=console,
        ) as progress:
            total_task = progress.add_task("Re-embedding", total=None)
            while True:
                async with session_context() as session:
                    rows = (await session.execute(
                        sql_text("SELECT id, content FROM chunks ORDER BY id "
                                 "LIMIT :n OFFSET :o"),
                        {"n": batch_size, "o": offset},
                    )).fetchall()
                if not rows:
                    break
                ids = [r[0] for r in rows]
                texts = [r[1] for r in rows]
                embs = await svc.embed_batch(texts)

                updates = [
                    {"id": str(cid), "e": str(emb)}
                    for cid, emb in zip(ids, embs, strict=True)
                    if emb is not None
                ]
                errors += len(rows) - len(updates)
                if updates:
                    async with session_context() as session:
                        await session.execute(
                            sql_text(
                                "UPDATE chunks SET embedding = CAST(v.embedding AS vector) "
                                "FROM (VALUES (:id, :e)) AS v(id, embedding) "
                                "WHERE chunks.id = CAST(v.id AS uuid)"
                            ),
                            updates,
                        )
                        await session.commit()
                done += len(rows)
                offset += batch_size
                progress.update(total_task, completed=done,
                                total=max(done, 1))

        elapsed = _time.perf_counter() - started
        console.print(
            f"\n[bold green]Done:[/bold green] {done} chunks re-embedded "
            f"in {elapsed:.0f}s ({done / max(elapsed, 0.001):.0f}/s), {errors} errors"
        )
        await close_db()

    asyncio.run(run())


@cli.command()
@click.argument("query")
@click.option("--mode", "-m", type=click.Choice(["semantic", "keyword", "hybrid"]), default="hybrid")
@click.option("--limit", "-n", default=10, type=click.IntRange(1, 100), show_default=True)
@click.option("--source", "-s", type=click.Choice(["calibre", "file"]), default=None)
@click.pass_context
def search(ctx, query: str, mode: str, limit: int, source: str | None):
    """Search the knowledge base."""
    from .db import close_db, init_db
    from .search.engine import InvalidQueryError, SearchEngine

    config = ctx.obj["config"]

    async def _search():
        await init_db(config.database.url)
        try:
            engine = SearchEngine(config)
            return await engine.search(query, mode=mode, limit=limit, source_filter=source)
        finally:
            await close_db()

    try:
        results = asyncio.run(_search())
    except InvalidQueryError as e:
        raise click.ClickException(str(e)) from e

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

    console.print("\n[bold cyan]🏛️  Knowledge Palace API[/bold cyan]")
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
    from .ingest.calibre import CalibreBridge, CalibreLibraryError

    config = ctx.obj["config"]
    if not config.calibre.library_path:
        console.print("[red]No Calibre library path configured[/red]")
        return

    bridge = CalibreBridge(config.calibre.library_path)
    try:
        bridge.list_books()
    except CalibreLibraryError as e:
        raise click.ClickException(str(e)) from e


@cli.command("eval")
@click.option("--api", default=None, help="API base URL (default: from config.toml)")
@click.option("--cases-dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="Directory with *.jsonl eval cases (default: evals/cases)")
@click.option("--domains", default=None, help="Comma-separated domain filter (e.g. cpp,rust)")
@click.option("--modes", default="semantic,hybrid,context", show_default=True)
@click.option("-k", type=int, default=5, show_default=True, help="Rank cutoff for hit/mrr/recall")
@click.option("--gate", is_flag=True, help="Exit 1 on quality thresholds or baseline regressions")
@click.option("--baseline", type=click.Path(exists=True), default=None, help="Baseline JSON to compare against")
@click.option("--save-baseline", type=click.Path(), default=None, help="Save current metrics as baseline")
@click.option("--json", "json_out", type=click.Path(), default=None, help="Write metrics JSON here")
@click.option("--show-misses", is_flag=True, help="List cases with no acceptable source in top-k")
@click.pass_context
def eval_cmd(ctx, api, cases_dir, domains, modes, k, gate, baseline, save_baseline, json_out, show_misses):
    """Score retrieval quality against the eval set (evals/cases).

    Run against a live server (kp serve). Use --gate in CI or after changing
    embeddings/chunking/search settings to catch quality regressions.
    """
    from .evalharness import main as harness_main

    argv = []
    if api:
        argv += ["--api", api]
    if cases_dir:
        argv += ["--cases-dir", cases_dir]
    if domains:
        argv += ["--domains", domains]
    argv += ["--modes", modes, "--k", str(k)]
    if gate:
        argv.append("--gate")
    if baseline:
        argv += ["--baseline", baseline]
    if save_baseline:
        argv += ["--save-baseline", save_baseline]
    if json_out:
        argv += ["--json", json_out]
    if show_misses:
        argv.append("--show-misses")
    ctx.exit(harness_main(argv))


@cli.command("init-config")
@click.argument("path", default="config.toml")
@click.option("--force", is_flag=True, help="Overwrite the file if it already exists")
def init_config(path: str, force: bool):
    """Create a sample configuration file.

    Refuses to overwrite an existing file unless --force is given: a config
    holds library paths and settings that are not recoverable from the sample.
    """
    target = Path(path)
    if target.exists() and not force:
        raise click.ClickException(
            f"{path} already exists; pass --force to overwrite it (its current "
            "library paths and settings would be lost)"
        )
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
# Backend is auto-selected: ONNX Runtime (on the CPU), then PyTorch. MLX is
# used automatically only if mlx-embeddings is installed and supports the
# model's architecture (nomic_bert is not supported).
provider = "auto"
batch_size = 32
dimensions = 768
device = "cpu"

[chunking]
strategy = "semantic"
max_tokens = 512
# Must be smaller than max_tokens: 0 <= overlap_tokens < max_tokens
overlap_tokens = 50
respect_headings = true

[search]
default_mode = "hybrid"
hybrid_weight_vector = 0.7
hybrid_weight_fts = 0.3
default_limit = 10
# Vector-match quality floor (cosine). Calibrated for nomic-embed-text-v1.5
# with task prefixes; re-calibrate (kp eval) if you change the embedding model.
# 0 disables the floor.
min_similarity = 0.72

[api]
# 127.0.0.1 = local-only. The API has no authentication: use "0.0.0.0" only
# behind a firewall or reverse proxy.
host = "127.0.0.1"
port = 8080
# Web origins allowed to call the API from a browser. Empty = none.
cors_origins = []
"""
    target.write_text(sample)
    console.print(f"[green]Config written to {path}[/green]")


if __name__ == "__main__":
    cli()
