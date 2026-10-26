---
name: knowledge-palace
description: Ground engineering work in the user's personal book library (Knowledge Palace) before designing or implementing. Use when starting or realizing a project, choosing an architecture or design approach, or answering domain "how do I" questions in software, C++, Rust, mobile, ML engineering, project management, or system architecture — even when the user doesn't mention the knowledge base. Also use when changing Knowledge Palace itself (retrieval, chunking, embeddings) to run its quality gate.
---

# Knowledge Palace grounding

The user runs a self-hosted knowledge base of ~350 engineering books with
hybrid retrieval, exposed as the `knowledge-palace` MCP tools (`context`,
`search`, `ask`, `similar`). Answers grounded in it cite real books from the
user's own library instead of generic model memory.

This file is the source of truth, versioned in the knowledge-palace repo at
`.zcode/skills/knowledge-palace/SKILL.md`. The active user-level copy lives at
`~/.agents/skills/knowledge-palace/SKILL.md` — update both when changing this
skill.

## When to consult it

- **Project kickoff / architecture / design**: before proposing a design,
  gather context from the library. One `context` call beats many `search` calls.
- **Domain questions during implementation**: concurrency, performance,
  testing, ML pipelines, estimation, API design, etc.
- **Always** when the user asks to base decisions on their books or mentions
  Knowledge Palace.

## How

1. **Broad grounding**: `context` with a task-style query. Example:
   `context(query: "how to structure event-driven microservices with sagas for order processing", limit: 10)`.
   It expands query variants, fuses across modes, and returns ranked,
   de-duplicated passages with provenance.
2. **Targeted lookup**: `search` (hybrid) for specific patterns, e.g.
   `search(query: "C++ coroutine awaitable", mode: "hybrid", limit: 5)`.
3. **Direct question**: `ask(question: "...")` returns relevant passages
   RAG-style.
4. **Cite and verify**: results carry `title`, `author`, `chunk_index`,
   `file_path` provenance. Name the source book when you use a passage; read
   the surrounding content before relying on it for a consequential decision.
5. **Respect the empty result**: zero hits means the library doesn't cover the
   topic (a similarity gate suppresses weak matches). Say so and fall back to
   your own knowledge or the web — never invent a citation.

## Maintaining Knowledge Palace itself

When working in this repo (retrieval, chunking, embeddings, similarity
thresholds, fusion): after changing anything retrieval-affecting, run the
quality gate from the repo root and report the numbers:

```bash
.venv/bin/kp serve &   # if not already running
.venv/bin/kp eval --gate --show-misses --baseline evals/baselines/baseline-2026-10-26.json
```

The gate fails on hit-rate regressions or negative probes leaking. If the
change intentionally improves quality, save a dated baseline
(`--save-baseline evals/baselines/baseline-<date>.json`) and update
`evals/README.md`. New books that change coverage deserve new eval cases —
see `evals/README.md` ("Adding cases").

If the MCP server isn't connected in the current session, fall back to the CLI
from the repo: `.venv/bin/kp search "query" --mode hybrid`.
