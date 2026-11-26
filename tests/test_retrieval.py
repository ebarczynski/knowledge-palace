"""Query expansion and Reciprocal Rank Fusion (pure logic, no I/O)."""

from __future__ import annotations

import doctest

import pytest

from knowledge_palace.retrieval import expand
from knowledge_palace.retrieval.expand import _SYNONYM_GROUPS, expand_queries
from knowledge_palace.retrieval.fuse import RRF_K, fuse
from knowledge_palace.search.engine import SearchResult

K = 60


def hit(chunk_id: str, content: str = "x", score: float = 0.5) -> SearchResult:
    return SearchResult(chunk_id, "doc", content, score, "t", None, "file", [], [], {}, {})


# ---------------------------------------------------------------- fusion

def test_top_rank_contributes_weight_over_k_plus_one():
    # RRF counts ranks from 1. The audited revision used enumerate()'s 0, so the
    # same chunk scored 2/60 here against 2/61 in the hybrid SQL.
    fused = fuse([[hit("a")], [hit("a")]], weights=[1.0, 1.0])
    assert fused[0].score == pytest.approx(2.0 / (K + 1))
    assert fused[0].score != pytest.approx(2.0 / K)


def test_fusion_matches_the_sql_convention_for_a_chunk_in_both_lists():
    fused = fuse([[hit("a")], [hit("b"), hit("a")]], weights=[0.7, 0.3])
    assert fused[0].chunk_id == "a"
    assert fused[0].score == pytest.approx(0.7 / (K + 1) + 0.3 / (K + 2))
    assert RRF_K == 60


def test_results_are_deduplicated_and_sorted_with_deterministic_ties():
    lists = [[hit("a"), hit("b"), hit("c")], [hit("c"), hit("b")], [hit("d")]]
    fused = fuse(lists)
    assert [r.chunk_id for r in fused] == ["c", "b", "a", "d"]
    assert len({r.chunk_id for r in fused}) == len(fused)
    tie = fuse([[hit("z")], [hit("y")]])
    assert [r.chunk_id for r in tie] == ["y", "z"], "equal scores tie-break on chunk id"


def test_weight_zero_and_empty_lists_are_ignored():
    fused = fuse([[hit("a")], []], weights=[1.0, 5.0])
    assert [r.chunk_id for r in fused] == ["a"]
    assert fused[0].score == pytest.approx(1.0 / (K + 1))


def test_fuse_keeps_the_richest_copy_and_stamps_the_fused_score():
    short, long_ = hit("a", "short"), hit("a", "a much longer copy", 0.9)
    fused = fuse([[short], [long_]], weights=[2.0, 1.0])
    assert fused[0].content == "a much longer copy"
    assert fused[0].score == pytest.approx(3.0 / (K + 1))


def test_mismatched_weights_are_an_error():
    with pytest.raises(ValueError, match="weights length"):
        fuse([[hit("a")], [hit("b")]], weights=[1.0])
    assert fuse([]) == []


# ---------------------------------------------------------------- expansion

def test_documented_example():
    # The docstring example used to promise a fourth variant ("... large
    # language models") that the code never produced: "LLMs" is not the term "LLM".
    assert doctest.testmod(expand, verbose=False).failed == 0


def test_singular_term_expands_plural_does_not():
    assert expand_queries("LoRA finetuning for LLM") == [
        "LoRA finetuning for LLM",
        "low-rank adaptation finetuning for LLM",
        "LoRA fine-tuning for LLM",
        "LoRA finetuning for large language model",
    ]


def test_variants_are_capped_with_the_original_counted():
    many = "lora qlora finetuning peft llm embedding rag transformer quantization distillation perplexity"
    variants = expand_queries(many, max_variants=6)
    assert len(variants) == 6 and variants[0] == many and len(set(variants)) == 6
    assert expand_queries(many, max_variants=2)[:2] == variants[:2]
    assert expand_queries("nothing maps here") == ["nothing maps here"]


def test_blank_query_expands_to_nothing():
    assert expand_queries("") == [] and expand_queries("   ") == []
    assert expand_queries(" x ") == ["x"]


def test_synonym_map_size():
    assert len(_SYNONYM_GROUPS) == 25
    assert len({term for group in _SYNONYM_GROUPS for term in group}) == 93
