"""Rule-based query expansion for multi-query retrieval.

No LLM, no network, no dependencies. Given a single query, produces a small
set of variations so that an agent harness can retrieve broader, more robust
context in one call (see ``SearchEngine.retrieve_context`` + ``fuse``).

Two kinds of variation are produced:

1. **Term expansion** — key SE/ML terms are detected and rewritten using a
   curated synonym map (e.g. "LoRA" -> "low-rank adaptation"). When no terms
   match, only the original query is returned (which is then run across all
   search modes by the caller).
2. The caller is responsible for running each returned query in multiple
   *modes* (keyword/semantic/hybrid); expansion here is purely textual.

The map is deliberately small and hand-curated for the domains covered by the
knowledge base. It is not a general-purpose NLP system.
"""

from __future__ import annotations

import re

# Curated synonym/paraphrase groups for SE + ML terms found in the corpus.
# Each group lists surface forms; any detected form expands to the others.
# Order within a group doesn't matter; lowercase matching is used.
_SYNONYM_GROUPS: list[list[str]] = [
    # --- ML / LLM fine-tuning ---
    ["lora", "low-rank adaptation", "low rank adaptation"],
    ["qlora", "quantized lora", "4-bit finetuning", "4-bit fine-tuning"],
    ["finetuning", "fine-tuning", "fine tuning", "sft", "supervised finetuning", "instruction tuning"],
    ["peft", "parameter-efficient finetuning", "parameter efficient fine-tuning"],
    ["llm", "large language model", "large language models"],
    ["embedding", "embeddings", "vector embedding", "text embedding"],
    ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    ["transformer", "transformers", "attention model"],
    ["quantization", "quantization", "int8", "int4", "4-bit", "8-bit"],
    ["distillation", "knowledge distillation", "model distillation"],
    ["perplexity", "language model evaluation"],
    # --- Concurrency / async ---
    ["concurrency", "concurrent", "threading", "multithreading", "parallelism", "parallel"],
    ["async", "asynchronous", "asyncio", "futures", "tokio"],
    ["race condition", "data race", "thread safety", "thread-safe"],
    # --- Architecture ---
    ["microservices", "microservice", "service-oriented architecture", "soa"],
    ["event-driven", "event driven", "event sourcing", "pub sub", "pub/sub"],
    ["domain-driven design", "ddd", "domain driven design"],
    ["fitness function", "architectural fitness function"],
    # --- Process ---
    ["technical debt", "tech debt", "code debt"],
    ["tdd", "test-driven development", "test driven development"],
    ["cicd", "ci/cd", "continuous integration", "continuous deployment", "ci cd"],
    ["software estimation", "project estimation", "effort estimation"],
    ["refactoring", "refactor", "code refactoring", "restructure"],
    # --- Data ---
    ["data structure", "data structures", "algorithm", "algorithms"],
    ["graph algorithm", "graph algorithms", "graph traversal"],
]

# Build a lookup: lowercase surface form -> the full group (minus that form)
_EXPANSIONS: dict[str, list[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _term in _group:
        # don't expand a term to itself
        _EXPANSIONS[_term.lower()] = [t for t in _group if t.lower() != _term.lower()]


def _detect_terms(query: str) -> list[tuple[str, list[str]]]:
    """Return (matched_term, expansions) for each synonym group hit in the query.

    Multiple surface forms of the same group may match; we collapse to one
    expansion set per group to avoid producing redundant variants.
    """
    query_lower = query.lower()
    seen_groups: set[frozenset[str]] = set()
    hits: list[tuple[str, list[str]]] = []

    for term, expansions in _EXPANSIONS.items():
        # whole-word / phrase match (boundaries), case-insensitive
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, query_lower):
            group_key = frozenset([term, *expansions])
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            hits.append((term, expansions))
    return hits


def expand_queries(query: str, max_variants: int = 6) -> list[str]:
    """Expand a query into textual variants for multi-query retrieval.

    Always returns at least the original query. When key terms are detected,
    returns the original plus one rewritten variant per detected synonym group
    (the matched term replaced by its first synonym). ``max_variants`` caps the
    total to keep retrieval cost bounded.

    Example::

        >>> expand_queries("LoRA finetuning for LLMs")
        ['LoRA finetuning for LLMs',
         'low-rank adaptation finetuning for LLMs',
         'LoRA fine-tuning for LLMs',
         'LoRA finetuning for large language models']
    """
    query = query.strip()
    if not query:
        return []

    variants: list[str] = [query]
    seen: set[str] = {query.lower()}

    for term, expansions in _detect_terms(query):
        if len(variants) >= max_variants:
            break
        # produce one variant per group: swap the matched term for its first
        # alternative surface form that isn't already in the query.
        for alt in expansions:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            new_variant = pattern.sub(alt, query, count=1)
            if new_variant.lower() not in seen:
                seen.add(new_variant.lower())
                variants.append(new_variant)
                break  # one variant per synonym group

    return variants[:max_variants]


__all__ = ["expand_queries"]
