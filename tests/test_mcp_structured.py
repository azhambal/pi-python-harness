"""Pi v1.4 result contracts over SDK 2.x and both owned transports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from pi_agent import create_harness
from pi_agent.events import ToolExecutionEnd
from pi_agent.mcp import McpConnection
from pi_agent.providers import FauxProvider
from pi_agent.types import ToolResultMessage
from test_mcp_http import Peer, configuration, connection, reply
from test_mcp_runtime import settings
from test_mcp_sessions import done, response

VALUES = [None, {}, [], [1, None, {"x": False}], "", "text π", False, True, 0, 1.25]


def setup_server(directory, monkeypatch, transport, wire, schema=None):
    descriptor = {"name": "echo", "inputSchema": {"properties": {}}}
    if schema is not None:
        descriptor["outputSchema"] = schema
    if transport == "stdio":
        (directory / "result.json").write_text(json.dumps(wire))
        (directory / "descriptor.json").write_text(json.dumps(descriptor))
        config = settings(directory, "custom")
        return McpConnection(config.servers[0]), config, {}
    peer = Peer(monkeypatch)
    handle = peer.handle

    async def catalog(request):
        if request.method == "POST":
            message = json.loads(await request.aread())
            if message.get("method") == "tools/list":
                return reply(
                    json.dumps(
                        {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [descriptor]}}
                    ).encode()
                )
        return await handle(request)

    peer.handle = catalog
    peer.override = lambda message: reply(
        json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": wire}).encode()
    )
    return connection(directory), configuration(directory), {"MCP_TOKEN": "synthetic-auth"}


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("is_error", [False, True])
@pytest.mark.parametrize("with_text", [False, True])
async def test_arbitrary_json_preserves_envelope_and_content_precedence(
    tmp_path, monkeypatch, transport, value, is_error, with_text
):
    wire = {
        "structuredContent": value,
        "isError": is_error,
        "_meta": {"private": "top-level-private"},
        "extension": {"revision": 7},
    }
    if with_text:
        wire["content"] = [{"type": "text", "text": "server text", "extension": 8}]
    client, _, _ = setup_server(tmp_path, monkeypatch, transport, wire)
    try:
        tool = next(tool for tool in await client.discover() if tool.provenance.tool == "echo")
        assert tool.schema() == {"type": "object", "properties": {}}
        result = await tool.execute("call", {})
        assert result.is_error is is_error
        expected = {key: value for key, value in wire.items() if key != "_meta"}
        expected.setdefault("content", [])
        assert result.structured_content == expected
        text = "server text" if with_text else json.dumps(value, ensure_ascii=False, indent=2)
        assert result.content[0].text == text
        assert "top-level-private" not in json.dumps(result.to_json())
        assert client.state == "ready"
    finally:
        await client.aclose()


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("is_error", [False, True])
async def test_absent_structured_content_is_distinct_from_null(tmp_path, monkeypatch, transport, is_error):
    client, _, _ = setup_server(tmp_path, monkeypatch, transport, {"isError": is_error})
    try:
        tool = next(tool for tool in await client.discover() if tool.provenance.tool == "echo")
        result = await tool.execute("call", {})
        assert result.structured_content == {"content": [], "isError": is_error}
        if is_error:
            assert result.content[0].text == "MCP tool fixture/echo returned an error"
        else:
            assert result.content == []
    finally:
        await client.aclose()


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize(
    "schema,wire,valid",
    [
        ({"type": "null"}, {"structuredContent": None}, True),
        ({"type": "null"}, {}, False),
        ({"type": "array", "items": {"type": "integer"}}, {"structuredContent": [1, 2]}, True),
        ({"type": "array", "items": {"type": "integer"}}, {"structuredContent": [True]}, False),
        ({"type": "string"}, {"structuredContent": ""}, True),
        ({"type": "boolean"}, {"structuredContent": False}, True),
        ({"type": "number"}, {"structuredContent": 0}, True),
        ({"anyOf": [{"type": "null"}, {"type": "integer"}]}, {"structuredContent": None}, True),
        ({}, {"structuredContent": [False, None]}, True),
        ({"type": "integer"}, {"structuredContent": "error payload", "isError": True}, True),
    ],
)
async def test_output_schema_validates_arbitrary_roots_and_presence(
    tmp_path, monkeypatch, transport, schema, wire, valid
):
    client, _, _ = setup_server(tmp_path, monkeypatch, transport, wire, schema)
    try:
        tool = next(tool for tool in await client.discover() if tool.provenance.tool == "echo")
        result = await tool.execute("call", {})
        if valid:
            assert result.is_error == wire.get("isError", False)
            assert result.structured_content["structuredContent"] == wire["structuredContent"]
        else:
            assert result.is_error and result.details == {"invalid_output_schema": True}
    finally:
        await client.aclose()


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("value", [None, [], False, 0, ""])
async def test_structured_error_survives_events_journal_and_strict_resume(
    config, monkeypatch, transport, value
):
    wire = {"content": [], "structuredContent": value, "isError": True}
    unused, mcp_config, environment = setup_server(config.cwd, monkeypatch, transport, wire)
    await unused.aclose()
    provider = FauxProvider([response(), done()])
    session = await create_harness(
        provider,
        config=config,
        session="new",
        mcp_config=mcp_config,
        mcp_tools=("fixture/echo",),
        mcp_environment=environment,
    )
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("read structured error")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and result.structured_content == wire
        assert result.content[0].text == json.dumps(value, indent=2)
        assert provider.requests[-1].messages[-1] == result
        end = next(event for event in events if isinstance(event, ToolExecutionEnd))
        assert end.is_error and end.result["structured_content"] == wire
        path = session.session_file
    finally:
        await session.aclose()
    before = path.read_bytes()
    restored = await create_harness(
        FauxProvider([]), session=path, mcp_config=mcp_config, mcp_environment=environment
    )
    try:
        assert result in restored.messages
        assert path.read_bytes() == before
        assert len(restored.mcp_bindings) == 1
    finally:
        await restored.aclose()


async def test_full_output_is_owned_and_removed_on_close(tmp_path, monkeypatch):
    wire = {
        "content": [{"type": "text", "text": "head " + "π" * 30_000 + " tail"}],
        "structuredContent": [None] * 100,
    }
    client, _, _ = setup_server(tmp_path, monkeypatch, "stdio", wire)
    tool = (await client.discover())[0]
    result = await tool.execute("call", {})
    path = Path(result.details["full_output_path"])
    assert path.read_text() == wire["content"][0]["text"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert result.structured_content == wire
    assert result.content[0].text.startswith("head ") and " tail" in result.content[0].text
    await client.aclose()
    assert not path.exists()
