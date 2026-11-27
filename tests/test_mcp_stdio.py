"""The MCP server as an agent sees it: a real process speaking JSON-RPC over stdio."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from conftest import write_note
from knowledge_palace.config import Config
from knowledge_palace.ingest.pipeline import IngestionPipeline

pytestmark = pytest.mark.db


class McpSession:
    """A minimal MCP client: newline-delimited JSON-RPC over the server's stdio."""

    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self._next_id = 0
        self.stdout_lines: list[str] = []

    async def call(
        self, method: str, params: dict | None = None, *, notify: bool = False, timeout: float = 60.0
    ) -> dict:
        self._next_id += 1
        message: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notify:
            message["id"] = self._next_id
        assert self.process.stdin
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()
        if notify:
            return {}
        assert self.process.stdout
        raw = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        line = raw.decode()
        self.stdout_lines.append(line.rstrip("\n"))
        return json.loads(line)

    async def drain_until_exit(self, timeout: float = 20.0) -> None:
        assert self.process.stdin
        self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()
        assert self.process.stdout
        rest = await self.process.stdout.read()
        self.stdout_lines.extend(line for line in rest.decode().splitlines() if line)


async def start_server(db: Config, tmp_path: Path) -> McpSession:
    config = tmp_path / "mcp-config.toml"
    database = db.database
    config.write_text(
        f"[database]\nhost = \"{database.host}\"\nport = {database.port}\nname = \"{database.name}\"\n"
        f"user = \"{database.user}\"\npassword = \"{database.password}\"\n"
        "[embedding]\nprovider = \"auto\"\ndevice = \"cpu\"\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "knowledge_palace.cli", "--config", str(config), "mcp",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    return McpSession(process)


@pytest.fixture
async def mcp(db: Config, tmp_path: Path):
    session = await start_server(db, tmp_path)
    await session.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "tests", "version": "0"}})
    await session.call("notifications/initialized", {}, notify=True)
    yield session
    await session.drain_until_exit()
    stderr = await session.process.stderr.read() if session.process.stderr else b""
    assert b"Traceback" not in stderr, stderr.decode()[-800:]


async def seed(db: Config, notes_dir: Path) -> None:
    notes_dir.mkdir(parents=True)
    write_note(notes_dir, "a.md", "rust")
    from dataclasses import replace

    from conftest import make_config
    await IngestionPipeline(make_config(db, sources=replace(db.sources, paths=[str(notes_dir)]))).run("files")


async def test_list_tools_announces_four_documented_tools(mcp: McpSession):
    tools = (await mcp.call("tools/list", {}))["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["search", "similar", "ask", "context"]
    for tool in tools:
        assert tool["inputSchema"]["additionalProperties"] is False


async def test_search_returns_ids_and_provenance(mcp: McpSession, db: Config, tmp_path: Path):
    await seed(db, tmp_path / "mcp-notes")
    arguments = {"query": "tokio work stealing", "mode": "keyword", "limit": 3}
    answer = await mcp.call("tools/call", {"name": "search", "arguments": arguments})
    result = answer["result"]
    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "Chunk:" in text and "Provenance:" in text and "Title:" in text
    assert "tokio" in text.lower()


async def test_every_argument_error_is_a_clean_tool_error(mcp: McpSession):
    bad = [
        ({"query": 123}, "not of type 'string'"),
        ({}, "required property"),
        ({"query": "   "}, "must not be empty"),
        ({"query": "x", "limit": -1}, "minimum of 1"),
        ({"query": "x", "limit": 10**9}, "maximum of 100"),
        ({"query": "x", "mode": "bogus"}, "not one of"),
        ({"query": "x", "source": "bogus"}, "not one of"),
        ({"query": "x", "unexpected": True}, "Additional properties"),
    ]
    for arguments, fragment in bad:
        answer = await mcp.call("tools/call", {"name": "search", "arguments": arguments})
        assert answer["result"]["isError"] is True, arguments
        assert fragment in answer["result"]["content"][0]["text"], (arguments, answer["result"]["content"][0]["text"])


async def test_context_argument_bounds_are_enforced(mcp: McpSession):
    for name, arguments, fragment in (
        ("context", {"query": "x", "max_variants": 0}, "minimum of 1"),
        ("context", {"query": "x", "limit": -3}, "minimum of 1"),
        ("similar", {"text": "x", "threshold": 7}, "maximum of 1"),
        ("ask", {"question": ""}, "should be non-empty"),
    ):
        answer = await mcp.call("tools/call", {"name": name, "arguments": arguments})
        assert answer["result"]["isError"] is True
        assert fragment in answer["result"]["content"][0]["text"]


async def test_unknown_tool_names_the_available_ones(mcp: McpSession):
    answer = await mcp.call("tools/call", {"name": "nope", "arguments": {}})
    assert answer["result"]["isError"] is True
    assert "Unknown tool: 'nope'" in answer["result"]["content"][0]["text"]
    assert "search, similar, ask, context" in answer["result"]["content"][0]["text"]


async def test_context_returns_structured_json(mcp: McpSession, db: Config, tmp_path: Path):
    await seed(db, tmp_path / "mcp-notes-2")
    answer = await mcp.call("tools/call", {"name": "search",
                                           "arguments": {"query": "tokio executor", "mode": "keyword", "max_chars": 0}})
    assert answer["result"]["isError"] is False
    whole = answer["result"]["content"][0]["text"]
    assert "[... truncated" not in whole, "max_chars=0 means whole chunks"


async def test_stdout_carries_nothing_but_jsonrpc(mcp: McpSession, db: Config, tmp_path: Path):
    # Everything the server writes on stdout while we talk to it.
    await seed(db, tmp_path / "mcp-notes-3")
    await mcp.call("tools/call", {"name": "search", "arguments": {"query": "tokio", "mode": "keyword", "limit": 2}})
    await mcp.call("tools/call", {"name": "ask", "arguments": {"question": "tokio work stealing", "limit": 2}})
    await mcp.call("tools/call", {"name": "nope", "arguments": {}})

    await mcp.drain_until_exit()
    assert mcp.stdout_lines, "the server said nothing at all"
    for i, line in enumerate(mcp.stdout_lines):
        message = json.loads(line)  # raises (and fails the test) on any stray print
        assert message.get("jsonrpc") == "2.0" and ("id" in message or "method" in message), (i, line[:120])
