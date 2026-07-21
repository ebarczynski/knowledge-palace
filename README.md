# Knowledge Palace 🏛️

Self-hosted personal knowledge base with semantic search, Calibre integration, and
agent-accessible API. Ingest books and notes, embed them, and search them with
hybrid (vector + full-text) retrieval over a local PostgreSQL + pgvector database
that only you can access.

## Features

- **Calibre Integration**: Ingest books from your Calibre library (EPUB, PDF, MOBI, AZW, AZW3, TXT)
- **File Crawler**: Index scattered markdown, text, org-mode, and reStructuredText files
- **Semantic Search**: Vector similarity search powered by pgvector (HNSW index)
- **Hybrid Search**: Vector + full-text search combined with Reciprocal Rank Fusion (RRF)
- **Context Gathering**: One-call multi-query retrieval (`context` tool/endpoint) that expands a query into variants, runs each across all search modes, and fuses + de-duplicates results — for comprehensive agent context without orchestrating many searches
- **Provenance**: Every result carries its source (file path, chunk index, format, Calibre id) so agents can cite/verify
- **Agent API**: REST + GraphQL endpoints
- **MCP Server**: Native agent integration via Model Context Protocol
- **Apple Silicon accelerated**: Embeddings run via ONNX Runtime (Metal-accelerated where available), with PyTorch fallback

## Quick Start

### 1. Install PostgreSQL with pgvector

```bash
# macOS (Homebrew)
brew install postgresql@16 pgvector

# Ubuntu/Debian
sudo apt install postgresql-16 postgresql-16-pgvector

# Or via Docker
docker run -d --name kp-postgres \
  -e POSTGRES_DB=knowledge_palace \
  -e POSTGRES_USER=kp \
  -e POSTGRES_PASSWORD=kp \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Helper scripts are provided:

```bash
scripts/setup-postgres.sh         # apt-based install (run with sudo)
scripts/setup-postgres-docker.sh  # Docker-based install (no root needed)
```

### 2. Install Knowledge Palace

Requires **Python 3.14**. This project uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                  # install dependencies into .venv (includes ONNX Runtime)

# Verify the CLI is available
.venv/bin/kp --help
```

### 3. Configure

```bash
kp init-config            # writes a sample config.toml
# Edit config.toml: set [calibre].library_path and/or [sources].paths
```

Key settings:

| Section | Key | Notes |
|---------|-----|-------|
| `[calibre].library_path` | Path to your Calibre folder (contains `metadata.db`) | |
| `[sources].paths` | Directories to crawl for notes | |
| `[embedding].device` | `mps` (Apple Silicon), `cuda` (NVIDIA), or `cpu` | |
| `[embedding].provider` | Embedding backend; ONNX/PyTorch auto-selected (MLX when model is compatible) | |
| `[search].default_mode` | `hybrid` (recommended), `semantic`, or `keyword` | |

### 4. Apply database migrations

```bash
psql -h localhost -U kp -d knowledge_palace \
  -f src/knowledge_palace/migrations/001_initial.sql
psql -h localhost -U kp -d knowledge_palace \
  -f src/knowledge_palace/migrations/002_tsvector_column.sql
```

> **Note:** `kp ingest` and `kp serve` also call `create_tables()` on startup to
> create the base tables, but the `content_tsvector` column required for keyword
> and hybrid search is **only** created by migration `002`. If you skip the
> migrations, keyword/hybrid search will fail with `column "content_tsvector"
> does not exist`. See *Known issues* below.

### 5. Ingest

```bash
kp ingest --source all       # Calibre + files
kp ingest --source calibre   # just Calibre
kp ingest --source files     # just files
kp ingest --source all --reindex   # reprocess already-indexed documents
```

Already-indexed documents are skipped automatically (deduplicated by content hash).

### 6. Search

```bash
# Hybrid search (default)
kp search "how to handle async errors in Rust"

# Semantic only
kp search "async error handling" --mode semantic

# Keyword only
kp search "tokio::spawn" --mode keyword
```

### 7. Serve

```bash
kp serve
# REST API:    http://localhost:8080/api/v1/search?q=your+query
# GraphQL:     http://localhost:8080/graphql
# OpenAPI:     http://localhost:8080/docs
```

### 8. Agent Integration (MCP)

Add to your agent's MCP config (e.g. Hermes `config.yaml`, or your editor's MCP
settings):

```yaml
mcp_servers:
  knowledge-palace:
    transport: stdio
    command: kp
    args: ["mcp"]
```

## API Endpoints

### REST

- `GET /api/v1/health` — Health check
- `GET /api/v1/search?q=...&mode=hybrid&limit=10` — Search (modes: `semantic`, `keyword`, `hybrid`)
- `GET /api/v1/context?q=...&limit=10` — **Multi-query retrieval**: expands query variants, fuses across all modes via RRF, returns de-duplicated ranked context with provenance. Use this for comprehensive context-gathering (e.g. implementing a whole pipeline) instead of orchestrating many `search` calls.
- `GET /api/v1/similar?text=...&threshold=0.7` — Find similar content
- `GET /api/v1/documents` — List documents (supports `source`, `author`, `limit`, `offset`)
- `GET /api/v1/documents/{id}` — Get document details

All search/context/similar responses include a `provenance` object per result
(`chunk_index`, `file_path`, `format`, `calibre_id`, `source`, `page_count` where
available) so callers can cite or verify sources.

### GraphQL

```graphql
query {
  context(query: "implement LoRA finetuning", limit: 10) {
    total
    mode        # "context"
    chunks {
      content
      score
      title
      author
      provenance  # { chunk_index, file_path, format, ... }
    }
  }
}
```

The `search`, `similar`, and `context` fields are all available; `context` does
multi-query fusion, the others take a `mode` (HYBRID/KEYWORD/SEMANTIC).

### MCP Tools (for agents)

- `search(query, mode?, limit?, source?, author?)` — Search knowledge base
- `context(query, limit?, max_variants?, source?, author?)` — **One-call context gathering**: multi-query retrieval + RRF fusion + provenance. Preferred over `search` when an agent needs broad grounded context for a task.
- `similar(text, threshold?, limit?)` — Find similar content
- `ask(question, limit?)` — Ask a question, returns relevant passages (RAG-style)

## Architecture

```
Calibre Library ─┐                     ┌──────────────────────┐
Markdown/Text ───┼──▶ Ingest ──▶ Chunk ──▶ Embed ──▶ PostgreSQL + pgvector
                 │      (pipeline, batched & pipelined)        │
                 └────────────────────────────────────────────┘
                                                          ┌─────┴─────┐
                                                          │  API Layer │
                                                          │ REST + GQL │
                                                          └─────┬─────┘
                                                          ┌─────┴─────┐
                                                          │  MCP Server│
                                                          └───────────┘
```

### Embedding backends

The embedding service tries backends in order of preference and uses the first
that works on your hardware. Backend selection is **model-aware**: the service
reads the model's architecture and only attempts a backend that can actually
load it.

The default model is **nomic-embed-text-v1.5** (`nomic_bert` architecture). For
this model:

1. **MLX** — *skipped*. `mlx-embeddings` does not implement the `nomic_bert`
   architecture (it supports `bert`, `modernbert`, `siglip`, `qwen3`, …). The
   service detects this and skips MLX rather than failing. If you switch to an
   MLX-compatible model and have `mlx-embeddings` installed, MLX will be used.
2. **ONNX Runtime** — the default, and the fastest backend for nomic on Apple
   Silicon (~750+ texts/sec on an M-series Mac). Enabled by `optimum` +
   `onnxruntime`, both core dependencies.
3. **PyTorch** (`sentence-transformers`) — universal fallback if ONNX is
   unavailable.

> Note: `mlx-embeddings` requires `transformers>=5.0` while `optimum` (ONNX)
> requires `transformers<5.0`, so the two cannot coexist. MLX is therefore not
> offered as an installable extra; install it manually only if you use an
> MLX-compatible model and don't need ONNX.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| Web Framework | FastAPI |
| GraphQL | Strawberry |
| Database | PostgreSQL 16 + pgvector (HNSW index) |
| Embeddings | nomic-embed-text-v1.5 (768-dim) via ONNX Runtime / PyTorch |
| Vector Index | pgvector HNSW (`vector_cosine_ops`) |
| EPUB | ebooklib |
| PDF | PyMuPDF (fitz) |
| MOBI/AZW/AZW3 | mobi |
| Chunking | tiktoken + semantic/heading-aware splitting |
| Agent Integration | MCP (Model Context Protocol) |
| CLI | Click + Rich |

## Configuration reference

See `config.toml` and the output of `kp init-config`. Config is discovered from
the first existing path: `./config.toml`,
`~/.config/knowledge-palace/config.toml`, or `~/.knowledge-palace/config.toml`.

## Known issues & limitations

- **Migrations are not run automatically.** `kp ingest` / `kp serve` create the
  base tables via SQLAlchemy, but the `content_tsvector` generated column and its
  GIN index come from migration `002`. Apply the SQL migrations (see step 4) on a
  fresh database, or keyword/hybrid search will fail. (The `run_migrations()`
  helper in `src/knowledge_palace/migrations/__init__.py` exists but is unused
  and currently points at the wrong path.)
- **Embedding dimension is hardcoded to 768.** Changing `[embedding].model` to a
  different-dimension model will break inserts unless the `chunks.embedding`
  column and ORM model are also updated.
- **Chunking token counts use tiktoken `cl100k_base`**, an approximation rather
  than the nomic model's own tokenizer, so `max_tokens` is approximate.
- **No authentication on the API**, and `serve` binds to `0.0.0.0` by default.
  Run behind a firewall/reverse proxy or set `host = "127.0.0.1"` for local-only
  access.
- **Provenance does not include section headings or exact page numbers.** The
  extractors join all PDF pages (discarding page boundaries) and the chunker
  records only a `section_start` boolean, not heading text. Provenance therefore
  carries `chunk_index` + document-level fields (`file_path`, `format`,
  `calibre_id`, `page_count`). Capturing finer-grained locations requires
  extractor changes + a full re-ingest.
- **No dedicated code-snippet index.** EPUB/PDF extraction mangles code (e.g.
  one token per line), so a code index over existing chunks would surface
  broken, unrunnable code. Surfacing code reliably needs code-aware extraction
  first. Until then, code blocks occasionally appear inside `context`/`search`
  prose chunks but are not guaranteed runnable.
- **`context` query expansion is rule-based.** Synonyms cover common SE/ML terms
  (LoRA, finetuning, microservices, async, …); queries with no mapped terms fall
  back to mode-only fusion (still broadens recall via keyword+semantic+hybrid).
- **No tests** are currently included.

## Project layout

```
src/knowledge_palace/
├── cli.py                # Click CLI (kp ingest / search / serve / mcp / ...)
├── config.py             # TOML config → dataclasses
├── db.py                 # SQLAlchemy models + async session management
├── api/                  # FastAPI REST + Strawberry GraphQL
├── mcp_server/           # MCP server for agent integration
├── embedding/            # ONNX → PyTorch embedding service (MLX auto-detected)
├── ingest/               # Calibre bridge, file crawler, extractors, chunker, pipeline
├── retrieval/            # query expansion + RRF fusion (powers `context`)
├── search/               # Hybrid search engine (vector + FTS + RRF) + retrieve_context
└── migrations/           # SQL migration files
```
```
