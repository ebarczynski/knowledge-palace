# Retrieval Quality Eval

Scores the replies Knowledge Palace returns for realistic engineering questions
and gates regressions. Use it whenever you change anything that affects
retrieval — chunking, the embedding model, task prefixes, `min_similarity`,
hybrid weights, fusion, or the extraction pipeline — and re-baseline when the
change is intentional.

## Domains covered

| File | Domain | Cases |
|------|--------|-------|
| `cases/software.jsonl` | General software engineering (git, testing, CI, algorithms, readability) | 8 |
| `cases/cpp.jsonl` | C++ (concurrency, memory, CMake, patterns, performance, templates, coroutines) | 8 |
| `cases/rust.jsonl` | Rust (async, web, lifetimes, CLI, polyglot) | 6 |
| `cases/mobile.jsonl` | Mobile (SwiftUI, Swift 6, Kotlin/Android, on-device ML) | 6 |
| `cases/ml-engineering.jsonl` | ML engineering (LLM fine-tuning, RAG, MLOps, serving, data hygiene, A/B) | 8 |
| `cases/project-management.jsonl` | Project/program management (agile, estimation, strategy, OSS governance) | 7 |
| `cases/architecture.jsonl` | Architecture & design (microservices, DDD, hexagonal, APIs, ADRs, teams) | 8 |
| `cases/negatives.jsonl` | Negative probes (topics absent from the library — must return nothing) | 8 |

Cases are grounded in books actually present in the library; each `accept`
fragment is a case-insensitive substring of an expected source title.

## Running

The server must be live (`kp serve`):

```bash
kp eval                                    # full report: hit@5, MRR, latency per domain × mode
kp eval --gate                             # exit 1 on quality thresholds or baseline regressions
kp eval --gate --baseline evals/baselines/baseline-2026-10-26.json
kp eval --domains cpp,rust                 # subset
kp eval --show-misses                      # list cases with no acceptable source in top-k
kp eval --save-baseline evals/baselines/baseline-<date>.json
```

The standalone wrapper works without the CLI:

```bash
.venv/bin/python evals/run_eval.py --gate --show-misses
```

## What is measured

- **Hit@k** — fraction of cases where at least one acceptable source appears in
  the top-k results (per domain, per mode: `semantic`, `hybrid`, `context`).
- **MRR@k** — mean reciprocal rank of the first acceptable source.
- **Recall@k** — of the acceptable sources that were returned at all, the
  fraction surfaced within top-k.
- **Negative false positives** — negative probes must return zero results in
  semantic and context mode. This guards the `min_similarity` gate: when the
  gate is lowered or disabled, out-of-domain queries start leaking junk and the
  eval fails.
- **Latency p50** per mode.

## Gate defaults

| Check | Threshold |
|-------|-----------|
| Overall hit@5 (context) | ≥ 0.75 |
| Every domain hit@5 (context) | ≥ 0.50 |
| Negative FP (semantic, context) | 0 |
| Drop vs `--baseline` (any hit@k) | ≤ 0.05 absolute |

Tighten as retrieval improves; the point is to make quality regressions loud,
not to certify perfection.

## Baseline (2026-10-26, nomic-embed-text-v1.5, min_similarity 0.72)

Overall hit@5: 0.86 semantic / 0.86 hybrid / 0.84 context; negatives clean;
p50 ≈ 25 ms (search) / 70 ms (context). Known gaps (tracked, intentional
pressure cases — do not "fix" by loosening accept lists):

- `cpp-cache-misses`, `cpp-pimpl-idiom` — the canonical C++ performance /
  pattern books rank below the similarity gate (top-1 ungated ≈ 0.70).
- `ml-ab-ranking`, `ml-embedding-serving` — *Experimentation for Engineers* /
  *AI-Powered Search* cover these but rank below gate or below noise.
- `rust-async-shared-state` — *Asynchronous Programming in Rust* has the
  content (14 Arc/Mutex chunks) but scores ≈ 0.70 vs gate 0.72.
- `sw-ci-pipeline`, `sw-git-revert` — correct books surface at rank 6+, not top-5.

### The precision/recall dial, quantified

The calibration note in the main README says in-domain similarity bottoms
around 0.72 while out-of-domain probes peak around 0.718 — razor thin. The eval
measures what that costs: 3 of 51 positive queries (0.70–0.71 top-1) are
suppressed entirely by the gate. Lowering `min_similarity` to ~0.70 would
rescue them; the negative probes in this eval tell you immediately whether the
lowered gate still keeps "wine tasting" and "Formula 1" queries clean. Run
`kp eval --gate` after any such change and compare before/after.

## Adding cases

1. Pick a real, recurring question from your ZCode sessions (the eval should
   mirror what agents actually ask).
2. Confirm which library books genuinely answer it (`psql ... WHERE content
   ILIKE '%term%'`, or `kp search`).
3. Add one JSONL line: `id`, `domain`, `query`, `accept` (title fragments),
   optional `note`.
4. Run `kp eval --show-misses --domains <domain>`; if the case misses, decide
   honestly whether it's a bad expectation (fix `accept`) or a retrieval gap
   (keep it — that's the point).
5. If overall metrics legitimately improved, `--save-baseline` with a new date.
