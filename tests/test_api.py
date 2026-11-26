"""REST + GraphQL API against the real app, database and model."""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest
from sqlalchemy import text

from conftest import TEXT, make_config, write_note
from knowledge_palace.config import Config
from knowledge_palace.db import session_context
from knowledge_palace.ingest.pipeline import IngestionPipeline

pytestmark = [pytest.mark.db, pytest.mark.embedding]


async def seed_one(db: Config, notes_dir) -> None:
    write_note(notes_dir, "a.md", "rust")
    await IngestionPipeline(make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))).run("files")


async def test_health(api_client: httpx.AsyncClient):
    assert (await api_client.get("/api/v1/health")).json() == {"status": "ok"}


async def test_search_returns_provenance_and_ids(api_client: httpx.AsyncClient, db: Config, tmp_path):
    notes = tmp_path / "api-notes"
    notes.mkdir()
    await seed_one(db, notes)
    body = (await api_client.get("/api/v1/search", params={"q": "tokio work stealing", "mode": "keyword"})).json()

    assert body["total"] >= 1 and body["mode"] == "keyword"
    hit = body["results"][0]
    assert set(hit) == {
        "chunk_id", "document_id", "content", "score", "title", "author", "source",
        "tags", "highlights", "provenance",
    }
    assert isinstance(hit["provenance"]["chunk_index"], int), "which chunk wins a tie is arbitrary, the index is not"
    assert hit["provenance"]["file_path"].endswith("a.md")


async def test_validation_errors_are_422_with_plain_messages(api_client: httpx.AsyncClient):
    bad = [
        ({"q": "x", "mode": "bogus"}, "Input should be"),
        ({"q": ""}, "at least 1 character"),
        ({"q": "   "}, "must not be empty or whitespace"),
        ({"q": "x\x00", "mode": "keyword"}, "NUL"),
        ({"q": "x", "limit": 0}, "greater than or equal"),
        ({"q": "x", "limit": 101}, "less than or equal"),
        ({"q": "x", "source": "bogus"}, "Input should be"),
    ]
    for params, fragment in bad:
        response = await api_client.get("/api/v1/search", params=params)
        assert response.status_code == 422, (params, response.status_code, response.text[:120])
        assert fragment in response.text
        assert "sqlalchemy" not in response.text and "asyncpg" not in response.text
    # /context validates its extra argument the same way
    response = await api_client.get("/api/v1/context", params={"q": "x", "max_variants": 0})
    assert response.status_code == 422 and "greater than or equal" in response.text


async def test_whitespace_query_on_keyword_mode_is_rejected_too(api_client: httpx.AsyncClient):
    # The audited revision returned 200 with results for keyword mode.
    response = await api_client.get("/api/v1/search", params={"q": "   ", "mode": "keyword"})
    assert response.status_code == 422


async def test_document_endpoints_validate_the_id(api_client: httpx.AsyncClient):
    assert (await api_client.get("/api/v1/documents/not-a-uuid")).status_code == 422
    unknown = "00000000-0000-0000-0000-000000000000"
    assert (await api_client.get(f"/api/v1/documents/{unknown}")).status_code == 404


async def test_no_cors_headers_without_configuration(api_client: httpx.AsyncClient):
    # With allow_origins=["*"] any web page the user has open could read the
    # library through their own browser.
    response = await api_client.get(
        "/api/v1/search", params={"q": "tokio", "mode": "keyword"}, headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    preflight = await api_client.options(
        "/api/v1/search",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 405, "the page cannot even preflight"


async def test_cors_allows_only_configured_origins(db: Config, notes_dir, tmp_path):
    from knowledge_palace.api.app import create_app

    write_note(notes_dir, "a.md", "rust")
    await IngestionPipeline(make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))).run("files")
    config = make_config(db, api=replace(db.api, cors_origins=["https://good.example"]))
    app = create_app(config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            good = await client.get("/api/v1/search", params={"q": "tokio", "mode": "keyword"},
                                    headers={"Origin": "https://good.example"})
            assert good.headers.get("access-control-allow-origin") == "https://good.example"
            evil = await client.get("/api/v1/search", params={"q": "tokio", "mode": "keyword"},
                                    headers={"Origin": "https://evil.example"})
            assert "access-control-allow-origin" not in evil.headers


async def test_author_filter_agrees_between_rest_search_documents_and_graphql(
    api_client: httpx.AsyncClient, db: Config, notes_dir
):
    write_note(notes_dir, "a.md", "rust")
    config = make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))
    await IngestionPipeline(config).run("files")
    from sqlalchemy import text as _text

    from knowledge_palace.db import session_context as _sc
    async with _sc() as session:
        await session.execute(_text("UPDATE documents SET author = 'Author, Ada'"))
        await session.commit()

    rest_search = await api_client.get("/api/v1/search", params={"q": "tokio", "mode": "keyword", "author": "ada"})
    assert rest_search.status_code == 200 and rest_search.json()["total"] >= 1
    assert (await api_client.get("/api/v1/documents", params={"author": "ada"})).json()["total"] >= 1
    answer = await api_client.post("/graphql", json={"query": '{ documents(author: "ada") { title } }'})
    assert answer.json()["data"]["documents"], "GraphQL author substring matches too"
    nobody = await api_client.get("/api/v1/search", params={"q": "tokio", "mode": "keyword", "author": "nobody"})
    assert nobody.json()["total"] == 0


async def test_graphql_context_field_end_to_end(api_client: httpx.AsyncClient, db: Config, notes_dir):
    write_note(notes_dir, "a.md", "rust")
    config = make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))
    await IngestionPipeline(config).run("files")
    import json

    literal = json.dumps(TEXT["rust"].strip())  # GraphQL string literals use double quotes
    query = (
        f"query {{ context(query: {literal}, limit: 5) "
        "{ total mode chunks { content score title provenance } } }"
    )
    answer = (await api_client.post("/graphql", json={"query": query})).json()
    assert "errors" not in answer, answer.get("errors")
    context = answer["data"]["context"]
    assert context["mode"] == "context" and context["total"] >= 1
    assert context["chunks"][0]["provenance"] is not None


async def test_graphql_reports_bad_limits_as_errors_not_crashes(api_client: httpx.AsyncClient):
    for literal in ("-1", "0", "100000"):
        answer = (
            await api_client.post(
                "/graphql", json={"query": f'{{ search(query: "tokio", mode: KEYWORD, limit: {literal}) {{ total }} }}'}
            )
        ).json()
        assert answer["data"] is None
        assert "limit must be between 1 and 100" in answer["errors"][0]["message"], answer["errors"]


async def test_graphql_rejects_an_empty_query_with_its_message(api_client: httpx.AsyncClient):
    answer = (
        await api_client.post("/graphql", json={"query": '{ search(query: "", mode: KEYWORD) { total } }'})
    ).json()
    assert answer["data"] is None
    assert "must not be empty" in answer["errors"][0]["message"], answer["errors"]


async def test_unknown_graphql_field_is_a_query_error(api_client: httpx.AsyncClient):
    answer = (await api_client.post("/graphql", json={"query": "{ noSuchField }"})).json()
    assert "Cannot query field" in answer["errors"][0]["message"]


async def test_similar_endpoint_respects_the_threshold(api_client: httpx.AsyncClient, db: Config, notes_dir):
    write_note(notes_dir, "a.md", "rust")
    config = make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))
    await IngestionPipeline(config).run("files")
    body = (
        await api_client.get("/api/v1/similar", params={"text": TEXT["rust"] * 2, "threshold": 0.5, "limit": 5})
    ).json()
    assert body["total"] >= 1
    assert all(hit["score"] > 0.5 for hit in body["results"])
    body = (await api_client.get("/api/v1/similar", params={"text": TEXT["rust"] * 2, "threshold": 0.9999})).json()
    assert body["total"] == 0


async def test_context_endpoint_is_the_same_shape_as_search(api_client: httpx.AsyncClient, db: Config, notes_dir):
    write_note(notes_dir, "a.md", "rust")
    config = make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))
    await IngestionPipeline(config).run("files")
    search = (
        await api_client.get("/api/v1/search", params={"q": "tokio work stealing", "mode": "keyword"})
    ).json()
    context = (
        await api_client.get("/api/v1/context", params={"q": TEXT["rust"].strip(), "limit": 5})
    ).json()
    assert set(context) == set(search) == {"total", "mode", "query", "results"}
    assert context["mode"] == "context"
    assert context["results"] and set(context["results"][0]) == set(search["results"][0])


async def test_documents_listing_paginates_and_reports_totals(api_client: httpx.AsyncClient, db: Config, notes_dir):
    write_note(notes_dir, "a.md", "rust")
    write_note(notes_dir, "b.txt", "db")
    config = make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))
    await IngestionPipeline(config).run("files")
    page = (await api_client.get("/api/v1/documents", params={"limit": 1})).json()
    assert page["total"] == 2 and len(page["documents"]) == 1
    other = (await api_client.get("/api/v1/documents", params={"limit": 1, "offset": 1})).json()
    assert other["documents"][0]["id"] != page["documents"][0]["id"]


async def test_tags_with_quotes_and_nulls_survive_the_api(
    api_client: httpx.AsyncClient, db: Config, notes_dir
):
    write_note(notes_dir, "a.md", "rust")
    config = make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))
    await IngestionPipeline(config).run("files")
    async with session_context() as session:
        await session.execute(text("UPDATE documents SET tags = '[\"C++ \\\"systems\\\"\", null]'::jsonb"))
        await session.commit()
    body = (await api_client.get("/api/v1/search", params={"q": "tokio", "mode": "keyword"})).json()
    assert body["results"][0]["tags"] == ['C++ "systems"'], "JSON nulls are dropped, quotes kept"
