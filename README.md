# Knowledge Palace 🏛️

Self-hosted personal knowledge base with semantic search, Calibre integration, and
agent-accessible API. Ingest books and notes, embed them, and search them with
hybrid (vector + full-text) retrieval over a local PostgreSQL + pgvector database
that only you can access.

## Features

- **Calibre Integration**: Ingest books from your Calibre library (EPUB, PDF, MOBI, AZW, AZW3, TXT)
- **File Crawler**: Index scattered markdown, text, org-mode, and reStructuredText files
- **Reconciling ingestion**: The index follows your sources — new and changed documents are
  (re-)indexed, files or Calibre books that disappear are removed, unchanged documents are
  skipped *before* any embedding work, and books re-added by Calibre under a new id keep their
  indexed chunks
- **Semantic Search**: Vector similarity search powered by pgvector (HNSW index)
- **Hybrid Search**: Vector + full-text search combined with Reciprocal Rank Fusion (RRF)
- **Context Gathering**: One-call multi-query retrieval (`context` tool/endpoint) that expands a
  query into variants, runs each across all search modes, and fuses + de-duplicates results — for
  comprehensive agent context without orchestrating many searches
- **Provenance**: Every result carries its source (file path, chunk index, format, Calibre id) so
  agents can cite/verify
- **Agent API**: REST + GraphQL endpoints with validated, bounded inputs
- **MCP Server**: Native agent integration via Model Context Protocol, with ids and provenance on
  every hit and structured JSON from `context`
- **Apple Silicon**: Embeddings run via ONNX Runtime (CPU) with PyTorch (CPU/MPS) fallback
- **Tested**: 219 regression tests (`tests/`) covering chunking, fusion, schema convergence,
  ingestion reconciliation, the API surface and the MCP stdio protocol

## Quick Start

### 1. Install PostgreSQL with pgvector

On macOS, the bundled script does everything (installs Homebrew packages, starts the service,
creates the `kp` role + `knowledge_palace` database + pgvector extension — all re-runnable), and
it verifies that pgvector can actually be *loaded*, not merely that it is registered:

```bash
bash scripts/setup-postgres-macos.sh
```

Manual alternatives:

```bash
# macOS (Homebrew)
brew install postgresql@17 pgvector
brew services start postgresql@17

# Ubuntu/Debian
sudo apt install postgresql-16 postgresql-16-pgvector

# Or via Docker (the script scripts/setup-postgres-docker.sh does the same, re-runnably)
docker run -d --name kp-postgres \
  -e POSTGRES_DB=knowledge_palace \
  -e POSTGRES_USER=kp \
  -e POSTGRES_PASSWORD=kp \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

> The pgvector build must match your PostgreSQL major version. Homebrew occasionally ships a
> pgvector that no longer provides a library for the version you run; queries then fail with
> `could not access file "$libdir/vector"` even though `CREATE EXTENSION` once succeeded. The
> setup script's functional probe catches this and prints the fix.

### 2. Install Knowledge Palace

Requires **Python 3.14**. This project uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                  # dependencies + dev tools into .venv

# Verify the CLI is available
.venv/bin/kp --help
```

### 3. Configure

```bash
kp init-config            # writes a sample config.toml (refuses to overwrite an existing one)
# Edit config.toml: set [calibre].library_path and/or [sources].paths
```

Key settings:

| Section | Key | Notes |
|---------|-----|-------|
| `[calibre].library_path` | Path to your Calibre folder (contains `metadata.db`) | Read read-only; a 30 s busy timeout |
| `[sources].paths` | Directories to crawl for notes | A root that is missing is skipped, not treated as "everything deleted" |
| `[embedding].device` | `cpu`, `cuda`, or `mps` | |
| `[embedding].provider` | `auto` (default), `onnx`, or `pytorch` | ONNX is tried first for nomic models |
| `[search].default_mode` | `hybrid` (recommended), `semantic`, or `keyword` | |
| `[api].host` | `127.0.0.1` by default | The API has no authentication |
| `[api].cors_origins` | Browser origins allowed to call the API | Empty = none; a web page cannot read your library |

Invalid values (an overlap that would stall chunking, a non-cosine threshold, an unknown mode)
are rejected when the config is *loaded*, with a message naming the key.

### 4. Ingest

```bash
kp ingest --source all       # Calibre + files
kp ingest --source calibre   # just Calibre
kp ingest --source files     # just files
kp ingest --source all --reindex   # rebuild already-indexed documents too
```

Ingestion reconciles the index with the sources:

- unchanged documents are skipped before any embedding work (a no-op run never loads the model)
- changed documents are rebuilt and replace their old chunks
- deleted files / removed Calibre books are removed from the index (a missing configured root
  prunes nothing — an unmounted drive is not "all files deleted")
- identical content is indexed once; a book re-added by Calibre under a new id is re-pointed,
  keeping its chunks
- every failure is named with its reason, and `kp ingest` exits non-zero if anything failed

The schema (tables, `content_tsvector` column, HNSW/GIN indexes) is created and brought up to
date automatically on `kp ingest` / `kp serve` / `kp mcp` startup — no manual `psql -f` step.
Databases created by older versions (either schema path) converge to the same shape.

### 5. Search

```bash
# Hybrid search (default)
kp search "how to handle async errors in Rust"

# Semantic only
kp search "async error handling" --mode semantic

# Keyword only
kp search "tokio::spawn" --mode keyword
```

**Retrieval quality.** Embeddings use nomic task prefixes (`search_query:` /
`search_document:` — required for nomic-embed models), and vector results are gated by
`[search].min_similarity` (default `0.72`). Queries about topics absent from the library
("cooking", "welding") return **zero** results in semantic mode instead of weakly-related junk;
hybrid mode additionally allows literal keyword matches (real term occurrences). The gate is a
calibration with a thin margin — see `evals/README.md` for what it costs and how to re-measure
it. `0` (or any value ≤ 0) disables it.

If you change embedding-affecting settings (model, prefixes), re-embed:

```bash
kp reindex-embeddings                 # backs up DB first (pg_dump)
kp reindex-embeddings --skip-backup --start-after 67840   # resume an interrupted run
kp reindex-embeddings --provider pytorch --batch 32       # GPU path (MPS/CUDA), low CPU
```

### 6. Serve

```bash
kp serve
# REST API:    http://127.0.0.1:8080/api/v1/search?q=your+query
# GraphQL:     http://127.0.0.1:8080/graphql
# OpenAPI:     http://127.0.0.1:8080/docs
```

The embedding model is preloaded at startup, so the first search is fast. Invalid arguments
(unknown mode, empty query, out-of-range limits, malformed ids) are rejected with HTTP 422 and a
plain message.

### 7. Agent Integration (MCP)

Add to your agent's MCP config:

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
- `GET /api/v1/context?q=...&limit=10` — **Multi-query retrieval**: expands query variants, fuses
  across all modes via RRF, returns de-duplicated ranked context with provenance
- `GET /api/v1/similar?text=...&threshold=0.7` — Find similar content
- `GET /api/v1/documents` — List documents (supports `source`, `author` (case-insensitive
  substring, as everywhere), `limit`, `offset`)
- `GET /api/v1/documents/{id}` — Get document details

`score` means different things per mode and is not comparable across modes: cosine similarity
(`semantic`, `similar`), `ts_rank_cd` (`keyword`), RRF score (`hybrid`, `context`). All
search/context/similar responses include a `provenance` object per result (`chunk_index`,
`file_path`, `format`, `calibre_id`, `page_count` where available).

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

The `search`, `similar`, and `context` fields are all available; `context` does multi-query
fusion, the others take a `mode` (HYBRID/KEYWORD/SEMANTIC). Limits and arguments are validated
by the same engine the REST API uses; violations come back as GraphQL errors with plain messages.

### MCP Tools (for agents)

- `search(query, mode?, limit?, source?, author?, max_chars?)` — Search; each hit carries its
  chunk id, document id, provenance and the start of its text (`max_chars=0` for everything;
  truncation is announced, never silent)
- `context(query, limit?, max_variants?, source?, author?)` — **One-call context gathering**:
  multi-query retrieval + RRF fusion + provenance. Returns structured JSON, validated against an
  output schema. Preferred over `search` when an agent needs broad grounded context
- `similar(text, threshold?, limit?, max_chars?)` — Find similar content
- `ask(question, limit?)` — Retrieve relevant passages (RAG-style); returns passages, not
  generated answers, and labels its score as the RRF score it is

Tool input schemas are closed (`additionalProperties: false`) and bounded; bad arguments are
rejected as tool errors before any code runs.

## Retrieval Quality Eval

`evals/` contains a domain-spanning eval set (general software, C++, Rust, mobile, ML
engineering, project management, architecture & design, plus negative probes) and a scoring
harness that turns it into a regression gate:

```bash
kp serve &                        # eval runs against the live API
kp eval --gate --show-misses      # hit@5 / MRR per domain × mode, negative FP, latency
kp eval --save-baseline evals/baselines/baseline-<date>.json
kp eval --gate --baseline evals/baselines/baseline-2026-10-26.json   # CI-style gate
```

Run it after changing chunking, embedding models, task prefixes, `min_similarity`, hybrid
weights, or fusion — it fails loudly when hit rates drop or negative probes start leaking. See
`evals/README.md` for the metrics, current baseline, and how to add cases.

## Tests

`tests/` holds the regression suite: 219 tests, none of them mocked — unit tests (chunker,
config validation, fusion arithmetic, Calibre bridge on a real SQLite library, migration
splitter, CLI, MCP schemas, extractors on real EPUB/PDF files) plus integration tests against a
real PostgreSQL + pgvector and the real embedding model (schema convergence from every legacy
shape, ingestion reconciliation, the REST/GraphQL surface, and the MCP server over real stdio):

```bash
# unit tests only (no services needed)
.venv/bin/pytest tests -m "not db and not embedding"

# everything: point at a scratch database (tables are truncated!) and allow the model
KP_TEST_DATABASE_URL=postgresql+asyncpg://kp:kp@127.0.0.1:5432/knowledge_palace_test \
KP_TEST_EMBEDDING=1 .venv/bin/pytest tests
```

The suite refuses a database named `knowledge_palace` — use a `*_test` database.

## Using Knowledge Palace from ZCode / coding agents

The MCP server is the intended integration surface. Register it once (user scope, so every
workspace gets it):

```json
{
  "mcpServers": {
    "knowledge-palace": {
      "command": "/path/to/knowledge-palace/.venv/bin/kp",
      "args": ["mcp"]
    }
  }
}
```

Then, when starting a project or task in ZCode:

1. **Kickoff / architecture**: before writing a design, ask
   `context("how to structure <domain> system with <constraints>")` — the multi-query fusion
   endpoint returns broad, de-duplicated, ranked passages with provenance in one call.
2. **Implementation**: use `search` (hybrid) for targeted lookups
   (`"C++ coroutine awaitable pattern"`) and `ask` for RAG-style passages.
3. **Cite and verify**: every result carries provenance (`title`, `author`, `chunk_id`,
   `chunk_index`, `file_path`) — cite it and read the source chunk when the stakes are high.
4. **Trust the empty result**: when `search`/`context` return nothing for a topic, the library
   does not cover it (the similarity gate suppresses weak matches). Fall back to your own
   knowledge or the web — do not invent citations.
5. **Maintain quality**: after ingesting new books or touching retrieval settings, run
   `kp eval --gate`. If hit rates regress, `--show-misses` says exactly which queries broke.
   The tests catch the mechanical regressions: `.venv/bin/pytest tests -m "not db"`.

A ready-made agent skill encapsulating this workflow is versioned at
`.zcode/skills/knowledge-palace/SKILL.md` (source of truth) — copy it to
`~/.agents/skills/knowledge-palace/SKILL.md` to activate it in every workspace.

## Architecture

```
Calibre Library ─┐                     ┌──────────────────────┐
Markdown/Text ───┼──▶ Ingest ──▶ Chunk ──▶ Embed ──▶ PostgreSQL + pgvector
                 │   (reconciling, batched & pipelined)         │
                 └─────────────────────────────────────────────┘
                                                          ┌─────┴─────┐
                                                          │  API Layer │
                                                          │ REST + GQL │
                                                          └─────┬─────┘
                                                          ┌─────┴─────┐
                                                          │  MCP Server│
                                                          └───────────┘
```

### Embedding backends

The embedding service tries backends in order of preference and uses the first that works on
your hardware. Backend selection is **model-aware**: the service reads the model's architecture
and only attempts a backend that can actually load it.

The default model is **nomic-embed-text-v1.5** (`nomic_bert` architecture). For this model:

1. **MLX** — *skipped*. `mlx-embeddings` does not implement the `nomic_bert` architecture. The
   service detects this without any network access and skips MLX rather than failing. If you
   switch to an MLX-compatible model and have `mlx-embeddings` installed, MLX will be used.
2. **ONNX Runtime** — the usual choice: the model's exported `onnx/model.onnx` graph on the CPU
   execution provider. The ONNX and PyTorch backends produce numerically identical vectors
   (differences ~1e-7), so which one ran never changes what is stored.
3. **PyTorch** (`sentence-transformers`) — universal fallback; also the MPS path on Apple
   Silicon when `device = "mps"`.

Speed is a property of your machine, and the model loads from the local Hugging Face cache
without network access once it is cached. Measure before believing any throughput number,
including any you read here.

> Note: `mlx-embeddings` requires `transformers>=5.0` while `optimum` (ONNX) requires
> `transformers<5.0`, so the two cannot coexist. MLX is therefore not offered as an installable
> extra; install it manually only if you use an MLX-compatible model and don't need ONNX.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| Web Framework | FastAPI |
| GraphQL | Strawberry |
| Database | PostgreSQL 16+ + pgvector (HNSW index) |
| Embeddings | nomic-embed-text-v1.5 (768-dim) via ONNX Runtime / PyTorch |
| Vector Index | pgvector HNSW (`vector_cosine_ops`) |
| EPUB | ebooklib |
| PDF | PyMuPDF (fitz) |
| MOBI/AZW/AZW3 | mobi |
| Chunking | tiktoken + semantic/heading-aware splitting, UTF-8-safe at token boundaries |
| Agent Integration | MCP (Model Context Protocol) |
| CLI | Click + Rich |
| Tests | pytest (+ pytest-asyncio), 219 tests |
| Lint / Types | ruff (clean), mypy (clean) |

## Configuration reference

See `config.toml` and the output of `kp init-config`. Config is discovered from the first
existing path: `./config.toml`,
`~/.config/knowledge-palace/config.toml`, or `~/.knowledge-palace/config.toml`. Unknown keys are
warned about (a typo such as `min_similarty` never silently keeps the default).

## Known issues & limitations

- **Embedding dimension is hardcoded to 768.** Changing `[embedding].model` to a
  different-dimension model will break inserts unless the `chunks.embedding` column, the ORM
  model and the migrations are also updated.
- **Chunking token counts use tiktoken `cl100k_base`**, an approximation rather than the nomic
  model's own tokenizer, so `max_tokens` is approximate. Chunks are never cut mid-character:
  token windows are widened to UTF-8 boundaries (verified by tests on CJK and emoji text).
- **No authentication on the API.** `serve` binds to `127.0.0.1` by default and sends no CORS
  headers unless you configure `[api].cors_origins`. Do not expose it without a reverse proxy
  that authenticates.
- **Provenance does not include section headings or exact page numbers.** The extractors join
  all PDF pages (discarding page boundaries) and the chunker records only a `section_start`
  boolean, not heading text. Provenance carries `chunk_index` + document-level fields
  (`file_path`, `format`, `calibre_id`, `page_count`). Capturing finer-grained locations
  requires extractor changes + a full re-ingest.
- **No dedicated code-snippet index.** EPUB/PDF extraction mangles code (e.g. one token per
  line), so a code index over existing chunks would surface broken, unrunnable code. Code blocks
  occasionally appear inside `context`/`search` prose chunks but are not guaranteed runnable.
- **`context` query expansion is rule-based.** Synonyms cover common SE/ML terms (25 groups;
  LoRA, finetuning, microservices, async, …); queries with no mapped terms fall back to
  mode-only fusion.
- **Formats without an extractor are skipped and named.** Calibre libraries may contain entries
  whose only format has no extractor (e.g. `ZIP` containers such as .txtz); such books are
  reported as failures with their titles, not silently dropped. AZW3/AZW are extracted via the
  `mobi` package.
- **Embedding throughput is modest on CPU.** On the machine this repo is developed on (Apple
  M5 Pro), ONNX Runtime embeds ~9 five-hundred-token texts per second single-threaded; the
  PyTorch/MPS path is ~2x faster but reserves several GB of unified memory. A full-library
  re-embed is an overnight job, not a coffee break. Nothing here is Metal-accelerated by
  default; measure on your hardware.
- **The similarity gate's margin is thin** (in-domain floor ≈ out-of-domain peak ≈ 0.72, per the
  calibration sample). The eval set contains legitimate in-domain cases the floor suppresses;
  treat the number as a dial, not ground truth, and re-measure with `kp eval` after any
  embedding change.

## Project layout

```
src/knowledge_palace/
├── cli.py                # Click CLI (kp ingest / search / serve / mcp / eval / ...)
├── config.py             # TOML config → validated dataclasses
├── db.py                 # SQLAlchemy models + async session management + schema convergence
├── api/                  # FastAPI REST + Strawberry GraphQL
├── mcp_server/           # MCP server for agent integration
├── embedding/            # ONNX → PyTorch embedding service (MLX auto-detected)
├── ingest/               # Calibre bridge (read-only), file crawler, extractors, chunker, pipeline
├── retrieval/            # query expansion + RRF fusion (powers `context`)
├── search/               # Hybrid search engine (vector + FTS + RRF) + retrieve_context
├── evalharness.py        # Retrieval-quality eval runner (powers `kp eval`)
└── migrations/           # SQL migrations + the runner applied at startup

evals/
├── cases/                # Eval cases per domain (*.jsonl) + negative probes
├── baselines/            # Saved metric baselines for regression gating
├── run_eval.py           # Standalone entry point (kp eval preferred)
└── README.md             # Metrics, gate thresholds, current baseline

tests/                    # 219 regression tests (unit + real-DB/model integration)
```
