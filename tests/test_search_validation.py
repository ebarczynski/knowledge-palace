"""Search argument validation. No database or model: every case here must be
rejected before the engine touches either, and valid calls must get through."""

from __future__ import annotations

import asyncio
import math

import pytest

from knowledge_palace.config import Config, SearchConfig
from knowledge_palace.search.engine import InvalidQueryError, SearchEngine

#: Raised by ``session_context()``: proof the call passed validation and reached the database.
REACHED_DATABASE = pytest.raises(RuntimeError, match="not initialized")


def go(coro):
    return asyncio.run(coro)


@pytest.fixture
def engine() -> SearchEngine:
    return SearchEngine(Config())


def call(engine: SearchEngine, argument: str, value) -> None:
    go(engine.search(**{"query": "LoRA finetuning", argument: value}))


def test_a_valid_keyword_search_gets_past_validation(engine: SearchEngine):
    # Proof that the rejections below are not "everything is refused".
    with REACHED_DATABASE:
        go(engine.search("tokio executor", mode="keyword", limit=7, source_filter="calibre", author="Ada"))


BAD = [
    ("query", ""),
    ("query", "   "),
    ("query", "\t\n"),
    ("query", "LoRA\x00"),
    ("query", "x" * 2001),
    ("query", 123),
    ("query", None),
    ("mode", "bogus"),
    ("mode", "SEMANTIC"),
    ("limit", 0),
    ("limit", -1),
    ("limit", 101),
    ("limit", 10**9),
    ("limit", 1.5),
    ("limit", "10"),
    ("limit", True),
    ("source_filter", "bogus"),
    ("author", "a\x00b"),
]


@pytest.mark.parametrize(("argument", "value"), BAD, ids=[f"{a}={v!r}" for a, v in BAD])
def test_search_rejects_bad_arguments_with_a_plain_message(engine: SearchEngine, argument: str, value):
    # At the audited revision these reached the database or the model and came
    # back as raw ValueError / asyncpg errors (HTTP 500).
    with pytest.raises(InvalidQueryError) as excinfo:
        call(engine, argument, value)
    message = str(excinfo.value)
    assert message and "sqlalchemy" not in message and "asyncpg" not in message


def test_whitespace_only_author_means_no_filter(engine: SearchEngine):
    with REACHED_DATABASE:
        go(engine.search("x", mode="keyword", author="   "))


def test_tags_must_be_a_list_of_clean_strings(engine: SearchEngine):
    for tags in ("Rust", ["ok", 3], ["a\x00b"]):
        with pytest.raises(InvalidQueryError, match="tags"):
            go(engine.search("x", mode="keyword", tags=tags))
    # A quote inside a tag is legal and must reach the database as JSON.
    with REACHED_DATABASE:
        go(engine.search("x", mode="keyword", tags=['C++ "systems"']))


@pytest.mark.parametrize(
    ("argument", "value"),
    [("limit", 0), ("limit", -3), ("limit", 101), ("max_variants", 0), ("max_variants", 13), ("max_variants", "six")],
)
def test_context_rejects_bad_arguments(engine: SearchEngine, argument: str, value):
    with pytest.raises(InvalidQueryError):
        go(engine.retrieve_context("LoRA finetuning", **{argument: value}))


def test_context_rejects_an_empty_query_instead_of_returning_empty_results(engine: SearchEngine):
    for query in ("", "   "):
        with pytest.raises(InvalidQueryError, match="query must not be empty"):
            go(engine.retrieve_context(query))


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("text", ""),
        ("text", "  "),
        ("text", "x" * 20_001),
        ("threshold", 7),
        ("threshold", -0.1),
        ("threshold", math.nan),
        ("threshold", "0.7"),
        ("limit", 0),
        ("limit", 51),
    ],
)
def test_similar_rejects_bad_arguments(engine: SearchEngine, argument: str, value):
    with pytest.raises(InvalidQueryError):
        go(engine.find_similar(**{"text": "a paragraph of text", argument: value}))


def test_a_disabled_similarity_floor_is_valid_config():
    SearchConfig(min_similarity=0.0).validate()
    SearchConfig(min_similarity=-1.0).validate()
    for bad in (float("nan"), float("inf"), 1.01, "0.7"):
        with pytest.raises(ValueError, match="min_similarity"):
            SearchConfig(min_similarity=bad).validate()


def test_the_error_is_a_value_error_so_graphql_surfaces_its_message():
    assert issubclass(InvalidQueryError, ValueError)
