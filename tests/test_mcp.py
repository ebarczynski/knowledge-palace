"""MCP tool definitions and text formatting (no server process needed)."""

from __future__ import annotations

import json

import jsonschema
import pytest

from knowledge_palace.mcp_server.server import (
    TOOLS,
    ask_text,
    call_tool,
    context_payload,
    format_results,
)
from knowledge_palace.search.engine import InvalidQueryError, SearchResult, SearchResults

PROVENANCE = {"format": "epub", "chunk_index": 2, "file_path": "/lib/Author/Book (7)/book.epub", "calibre_id": 7}
LONG = ("The Tokio runtime schedules tasks onto worker threads with a work-stealing scheduler. " * 12)[:1200]


def results(mode: str = "hybrid", *, content: str = LONG, with_provenance: bool = True) -> SearchResults:
    hit = SearchResult(
        chunk_id="11111111-1111-1111-1111-111111111111",
        document_id="22222222-2222-2222-2222-222222222222",
        content=content, score=0.016393, title="Async notes", author=None, source="file",
        tags=[], highlights=[], metadata={}, provenance=PROVENANCE if with_provenance else None,
    )
    return SearchResults(results=[hit], total=1, mode=mode, query="q")


BY_NAME = {tool.name: tool for tool in TOOLS}


def test_four_tools_are_exposed():
    assert [tool.name for tool in TOOLS] == ["search", "similar", "ask", "context"]


def test_search_output_carries_ids_and_provenance():
    # The audited revision returned neither ids nor provenance, so an agent
    # could neither cite a hit nor ask for the rest of it.
    text = format_results(results())
    assert "Chunk: 11111111-1111-1111-1111-111111111111" in text
    assert "Document: 22222222-2222-2222-2222-222222222222" in text
    assert "Provenance: epub · chunk 2 · book.epub · calibre 7" in text
    assert "/lib/Author" not in text, "provenance shows the file name, not the whole path"


def test_truncation_is_explicit_and_can_be_turned_off():
    text = format_results(results(), max_chars=500)
    assert len(LONG) > 500
    assert f"truncated: showing 500 of {len(LONG)} characters" in text and "max_chars=0" in text
    assert LONG not in text
    assert LONG in format_results(results(), max_chars=0)
    assert LONG in format_results(results(), max_chars=5000)


@pytest.mark.parametrize(
    ("mode", "label"),
    [("semantic", "cosine"), ("similar", "cosine"), ("keyword", "text rank"), ("hybrid", "rrf")],
)
def test_scores_are_labelled_by_what_they_mean(mode: str, label: str):
    assert f"{mode} search, {label} score" in format_results(results(mode))


def test_ask_does_not_present_an_rrf_score_as_relevance():
    # "(relevance: 0.005)" made a top hit look worthless: hybrid scores are RRF.
    text = ask_text(results())
    assert "(rrf: 0.016)" in text and "relevance" not in text
    assert LONG in text, "ask returns whole passages"


def test_empty_results_are_stated():
    empty = SearchResults(results=[], total=0, mode="keyword", query="q")
    assert format_results(empty) == "No results found."
    assert "No relevant information" in ask_text(empty)


def test_context_payload_shape():
    payload = context_payload(results())
    assert set(payload) == {"query", "total", "mode", "results"}
    hit = payload["results"][0]
    assert set(hit) == {"text", "title", "source", "score", "provenance"}, "absent fields are omitted"
    assert hit["score"] == 0.0164 and payload["mode"] == "hybrid", "the mode is passed through"
    bare = context_payload(results(with_provenance=False))["results"][0]
    assert set(bare) == {"text", "title", "source", "score"}


def test_context_payload_is_valid_against_the_declared_output_schema():
    jsonschema.validate(context_payload(results("context")), BY_NAME["context"].outputSchema)
    assert BY_NAME["context"].outputSchema["required"] == ["query", "total", "mode", "results"]


def test_tool_schemas_are_closed_and_bounded():
    for tool in TOOLS:
        schema = tool.inputSchema
        assert schema["additionalProperties"] is False, tool.name
        assert schema["required"], tool.name
    props = BY_NAME["search"].inputSchema["properties"]
    assert (props["limit"]["minimum"], props["limit"]["maximum"]) == (1, 100)
    assert props["query"]["minLength"] == 1 and props["mode"]["enum"] == ["semantic", "keyword", "hybrid"]
    assert props["source"]["enum"] == ["calibre", "file"]
    context = BY_NAME["context"].inputSchema["properties"]
    assert (context["max_variants"]["minimum"], context["max_variants"]["maximum"]) == (1, 12)
    assert (context["limit"]["minimum"], context["limit"]["maximum"]) == (1, 50)
    similar = BY_NAME["similar"].inputSchema["properties"]
    assert (similar["threshold"]["minimum"], similar["threshold"]["maximum"]) == (0, 1)
    assert (similar["limit"]["minimum"], similar["limit"]["maximum"]) == (1, 50)
    assert (BY_NAME["ask"].inputSchema["properties"]["limit"]["maximum"]) == 50


def test_the_schemas_are_valid_json_schema_and_reject_bad_arguments():
    bad = [
        ("search", {"query": "x", "limit": -1}),
        ("search", {"query": "x", "limit": 10**9}),
        ("search", {"query": "x", "source": "bogus"}),
        ("search", {"query": "x", "extra": 1}),
        ("search", {}),
        ("context", {"query": "x", "max_variants": 0}),
        ("context", {"query": "x", "limit": -3}),
        ("similar", {"text": "x", "threshold": 7}),
        ("ask", {"question": ""}),
    ]
    for name, arguments in bad:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(arguments, BY_NAME[name].inputSchema)
    accepted = [("search", {"query": "x", "max_chars": 0}), ("context", {"query": "x", "max_variants": 12})]
    for name, arguments in accepted:
        jsonschema.validate(arguments, BY_NAME[name].inputSchema)


def test_tool_descriptions_do_not_promise_an_answer_or_metal():
    assert "No answer is generated" in BY_NAME["ask"].description
    for tool in TOOLS:
        assert "relevance" not in tool.description
        assert "Hermes" not in tool.description


async def test_unknown_tool_is_reported_as_an_error():
    with pytest.raises(InvalidQueryError, match="Unknown tool: 'nope'.*search, similar, ask, context"):
        await call_tool(None, "nope", {})  # type: ignore[arg-type]


def test_tool_definitions_serialise_to_json():
    assert json.loads(TOOLS[3].model_dump_json(by_alias=True))["outputSchema"]["type"] == "object"
