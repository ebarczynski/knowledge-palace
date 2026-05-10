# Knowledge Palace 🏛️

Self-hosted personal knowledge base with semantic search, Calibre integration, and agent-accessible API.

## Features

- **Calibre Integration**: Ingest books from your Calibre library (EPUB, PDF, TXT)
- **File Crawler**: Index scattered markdown, text, and org-mode files
- **Semantic Search**: Vector similarity search powered by pgvector
- **Hybrid Search**: Combines vector + full-text search with Reciprocal Rank Fusion
- **Agent API**: REST + GraphQL endpoints for Hermes and Copilot agents
- **MCP Server**: Native Hermes agent integration via Model Context Protocol
- **Problem-Solution Matching**: Find past solutions to similar problems

## Quick Start

### 1. Install PostgreSQL with pgvector

```bash
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

### 2. Configure

```bash
# Create config
kp init-config
# Edit config.toml with your Calibre path and file directories
```

### 3. Ingest

```bash
# Ingest from all sources
kp ingest --source all

# Or just Calibre
kp ingest --source calibre

# Or just files
kp ingest --source files
```

### 4. Search

```bash
# Hybrid search (default)
kp search "how to handle async errors in Rust"

# Semantic only
kp search "async error handling" --mode semantic

# Keyword only
kp search "tokio::spawn" --mode keyword
```

### 5. Serve

```bash
# Start API server
kp serve

# Then access:
# REST API:    http://localhost:8080/api/v1/search?q=your+query
# GraphQL:     http://localhost:8080/graphql
# OpenAPI:     http://localhost:8080/docs
```

### 6. Hermes Integration (MCP)

Add to your Hermes `config.yaml`:

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
- `GET /api/v1/search?q=...&mode=hybrid&limit=10` — Search
- `GET /api/v1/similar?text=...&threshold=0.7` — Find similar content
- `GET /api/v1/documents/{id}` — Get document details

### GraphQL

```graphql
query {
  search(query: "async error handling", mode: HYBRID, limit: 5) {
    total
    mode
    chunks {
      content
      score
      title
      author
      source
      tags
    }
  }
}
```

### MCP Tools (for Hermes)

- `search(query, mode?, limit?, source?, author?)` — Search knowledge base
- `similar(text, threshold?, limit?)` — Find similar content
- `ask(question, limit?)` — Ask a question with RAG

## Architecture

```
Calibre Library ─┐
Markdown/Text ───┼──▶ Ingest ──▶ Chunk ──▶ Embed ──▶ PostgreSQL + pgvector
File Watcher ─────┘                                              │
                                                          ┌─────┴─────┐
                                                          │  API Layer │
                                                          │ REST + GQL │
                                                          └─────┬─────┘
                                                          ┌─────┴─────┐
                                                          │  MCP Server│
                                                          │ (Hermes)   │
                                                          └───────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| Web Framework | FastAPI |
| GraphQL | Strawberry |
| Database | PostgreSQL 16 + pgvector |
| Embeddings | sentence-transformers (nomic-embed-text) |
| EPUB | ebooklib |
| PDF | PyMuPDF (fitz) |
| Agent Integration | MCP (Model Context Protocol) |
| CLI | Click + Rich |
