from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from pi_agent import PermissionPolicy, ToolHooks, create_harness
from pi_agent.events import event_to_json
from pi_agent.mcp import McpConnection, McpError, McpLimits, McpManifest, binding_name, parse_mcp_config
from pi_agent.mcp.resources import RESOURCE_PREFIX
from pi_agent.providers import FauxProvider
from pi_agent.tools.base import ToolValidationError
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage
from test_mcp_http import Peer, configuration, connection
from test_mcp_runtime import assert_reaped

FIXTURE = Path(__file__).parent / "fixtures/mcp_resource_server.py"


def stdio_config(directory, mode="normal"):
    return parse_mcp_config(
        {
            "mcpServers": {
                "fixture": {
                    "command": sys.executable,
                    "args": [str(FIXTURE), str(directory), mode],
                    "exposure": "direct",
                    "timeout": 2,
                }
            }
        },
        cwd=directory,
    )


def choose(tools, suffix):
    return next(tool for tool in tools if tool.provenance.tool == RESOURCE_PREFIX + suffix)


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_resource_lists_templates_reads_and_binary_notices(tmp_path, monkeypatch, transport):
    peer = Peer(monkeypatch) if transport == "http" else None
    client = (
        connection(tmp_path) if peer else McpConnection(stdio_config(tmp_path).servers[0], environment={})
    )
    tools = await client.discover()
    try:
        listing = choose(tools, "list")
        page = await listing.execute("id", {})
        assert page.structured_content["structuredContent"]["resources"][0]["uri"] == "file:///sample"
        cursor = page.structured_content["structuredContent"]["nextCursor"]
        second = await listing.execute("id", {"cursor": cursor})
        assert second.structured_content["structuredContent"]["resources"][0]["uri"] == "file:///second"
        assert second.structured_content["structuredContent"]["nextCursor"] is None
        templates = await choose(tools, "templates/list").execute("id", {})
        assert (
            templates.structured_content["structuredContent"]["resourceTemplates"][0]["uriTemplate"]
            == "file:///{name}"
        )
        text = await choose(tools, "read").execute("id", {"uri": "file:///sample"})
        assert "resource text" in text.content[0].text
        json_resource = await choose(tools, "read").execute("id", {"uri": "file:///json"})
        assert '{"answer": 42}' in json_resource.content[0].text
        binary = await choose(tools, "read").execute("id", {"uri": "file:///binary"})
        assert "Unsupported MCP resource: binary" in binary.content[0].text
        serialized = json.dumps(
            [result.to_json() for result in (page, second, templates, text, json_resource, binary)]
        )
        assert "synthetic-private" not in serialized and "c3ludGhldGlj" not in serialized
        assert not listing.parallel_safe and listing.provenance.kind == "mcp"
        assert len(McpManifest.from_tools(tools).bindings) == 4
    finally:
        await client.aclose()
    if transport == "stdio":
        assert_reaped(tmp_path)
    else:
        assert all(client.is_closed for client in peer.clients)


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_resources_without_tool_capability_do_not_list_tools(tmp_path, monkeypatch, transport):
    peer = Peer(monkeypatch, mode="resources-only") if transport == "http" else None
    client = (
        connection(tmp_path)
        if peer
        else McpConnection(stdio_config(tmp_path, "resources-only").servers[0], environment={})
    )
    try:
        tools = await client.discover()
        assert len(tools) == 3
        if peer:
            assert peer.count("tools/list") == 0 and peer.count("resources/list") == 0
        else:
            events = (tmp_path / "events").read_text()
            assert "tools/list" not in events and "resources/list" not in events
    finally:
        await client.aclose()


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_resource_text_budget_keeps_full_envelope(tmp_path, monkeypatch, transport):
    peer = Peer(monkeypatch) if transport == "http" else None
    limits = McpLimits(result_bytes=1024)
    client = (
        connection(tmp_path, limits=limits)
        if peer
        else McpConnection(stdio_config(tmp_path).servers[0], limits=limits, environment={})
    )
    try:
        result = await choose(await client.discover(), "read").execute("id", {"uri": "file:///large"})
        assert result.details["truncated"]
        assert len(result.content[0].text.encode()) < 1500
        path = Path(result.details["full_output_path"])
        assert path.read_text() == result.structured_content["content"][0]["text"]
        path.unlink()
    finally:
        await client.aclose()


def call_resource(suffix="read", args=None):
    return AssistantMessage(
        content=[
            ToolCallBlock(
                id="call",
                name=binding_name("fixture", RESOURCE_PREFIX + suffix),
                arguments=args or {"uri": "file:///sample"},
            )
        ],
        stop_reason="toolUse",
    )


def done():
    return AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop")


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("policy", ["allow", "read-only", "ask-denied"])
async def test_resource_permissions_hooks_events_journal_and_resume(config, monkeypatch, transport, policy):
    peer = Peer(monkeypatch) if transport == "http" else None
    settings = configuration(config.cwd) if peer else stdio_config(config.cwd)
    env = {"MCP_TOKEN": "synthetic-auth-secret"} if peer else {}
    confirmations = []
    hooks = []

    def before(context, token):
        hooks.append(context.tool_provenance.tool)

    def confirm(request, token):
        confirmations.append(request.context.tool_provenance.tool)
        return False

    session = await create_harness(
        FauxProvider([call_resource(), done()]),
        config=config,
        session="new",
        mcp_config=settings,
        mcp_tools=["fixture/" + RESOURCE_PREFIX + "read"],
        mcp_environment=env,
        permission_policy=PermissionPolicy(
            read_only=policy == "read-only", default="ask" if policy == "ask-denied" else "allow"
        ),
        hooks=ToolHooks(before_tool_call=before),
        confirm_permission=confirm,
    )
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("read resource")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error == (policy != "allow")
        assert hooks == [RESOURCE_PREFIX + "read"]
        assert confirmations == ([RESOURCE_PREFIX + "read"] if policy == "ask-denied" else [])
        if peer:
            assert peer.count("resources/read") == (1 if policy == "allow" else 0)
        else:
            assert (config.cwd / "events").read_text().count("resources/read") == (
                1 if policy == "allow" else 0
            )
        path = session.session_file
        journal = path.read_text()
        serialized = json.dumps([event_to_json(event) for event in events])
        for secret in (
            "synthetic-auth-secret",
            "synthetic-session-secret",
            "synthetic-private",
            "synthetic.invalid",
        ):
            assert secret not in journal and secret not in serialized
        assert "synthetic-instructions-not-auto-adopted" not in journal
    finally:
        await session.aclose()
    before_bytes = path.read_bytes()
    resumed = await create_harness(FauxProvider([]), session=path, mcp_config=settings, mcp_environment=env)
    try:
        assert [binding.tool for binding in resumed.mcp_bindings] == [RESOURCE_PREFIX + "read"]
        assert path.read_bytes() == before_bytes
    finally:
        await resumed.aclose()


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_missing_resource_error_is_constant_and_terminal(tmp_path, monkeypatch, transport):
    peer = Peer(monkeypatch) if transport == "http" else None
    client = (
        connection(tmp_path) if peer else McpConnection(stdio_config(tmp_path).servers[0], environment={})
    )
    tool = choose(await client.discover(), "read")
    try:
        with pytest.raises(McpError) as error:
            await tool.execute("id", {"uri": "file:///missing"})
        assert "synthetic-private-error" not in str(error.value)
        with pytest.raises(McpError, match="unavailable"):
            await tool.execute("id", {"uri": "file:///sample"})
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "suffix,args",
    [
        ("read", {}),
        ("read", {"uri": "not a URI"}),
        ("read", {"uri": "file:///" + "\u044f" * 5000}),
        ("list", {"cursor": "\u044f" * 600}),
        ("list", {"cursor": ""}),
        ("read", {"uri": "file:///sample", "unexpected": True}),
    ],
)
async def test_resource_argument_checks_precede_dispatch(tmp_path, monkeypatch, suffix, args):
    peer = Peer(monkeypatch)
    client = connection(tmp_path)
    try:
        tools = await client.discover()
        with pytest.raises(ToolValidationError):
            choose(tools, suffix).validate_arguments(args)
        assert peer.count("resources/read") == 0 and peer.count("resources/list") == 0
    finally:
        await client.aclose()


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_reserved_tool_cannot_impersonate_resource_binding(tmp_path, monkeypatch, transport):
    peer = Peer(monkeypatch, mode="reserved-tool") if transport == "http" else None
    client = (
        connection(tmp_path)
        if peer
        else McpConnection(stdio_config(tmp_path, "reserved-tool").servers[0], environment={})
    )
    with pytest.raises(McpError, match="discovery failed"):
        await client.discover()
    await client.aclose()


async def test_resources_only_obey_tool_count_budget(tmp_path, monkeypatch):
    Peer(monkeypatch, mode="resources-only")
    client = connection(tmp_path, limits=McpLimits(max_tools=2))
    with pytest.raises(McpError, match="resource tool count"):
        await client.discover()
    await client.aclose()


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_compaction_preserves_resource_manifest(config, monkeypatch, transport):
    peer = Peer(monkeypatch) if transport == "http" else None
    settings = configuration(config.cwd) if peer else stdio_config(config.cwd)
    env = {"MCP_TOKEN": "synthetic-auth"} if peer else {}
    session = await create_harness(
        FauxProvider([call_resource(), done(), done(), done()]),
        config=replace(config, compaction=replace(config.compaction, keep_recent_tokens=0)),
        session="new",
        mcp_config=settings,
        mcp_tools=["fixture/" + RESOURCE_PREFIX + "read"],
        mcp_environment=env,
    )
    try:
        await session.prompt("read")
        await session.prompt("second turn")
        bindings = session.mcp_bindings
        await session.compact()
        assert session.mcp_bindings == bindings
        path = session.session_file
    finally:
        await session.aclose()
    restored = await create_harness(FauxProvider([]), session=path, mcp_config=settings, mcp_environment=env)
    try:
        assert restored.mcp_bindings == bindings
    finally:
        await restored.aclose()


async def test_resource_capability_removal_rejects_resume_without_mutation(config, monkeypatch):
    peer = Peer(monkeypatch)
    settings = configuration(config.cwd)
    env = {"MCP_TOKEN": "synthetic-auth"}
    session = await create_harness(
        FauxProvider([done()]),
        config=config,
        session="new",
        mcp_config=settings,
        mcp_tools=["fixture/" + RESOURCE_PREFIX + "read"],
        mcp_environment=env,
    )
    await session.prompt("seed")
    path = session.session_file
    await session.aclose()
    original = path.read_bytes()
    peer.mode = "no-resources"
    with pytest.raises(McpError, match="unavailable"):
        await create_harness(FauxProvider([]), session=path, mcp_config=settings, mcp_environment=env)
    assert path.read_bytes() == original
    assert all(client.is_closed for client in peer.clients)


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_large_resource_page_never_returns_partial_pagination(tmp_path, monkeypatch, transport):
    peer = Peer(monkeypatch, mode="large-page") if transport == "http" else None
    client = (
        connection(tmp_path, limits=McpLimits(result_bytes=1024))
        if peer
        else McpConnection(
            stdio_config(tmp_path, "large-page").servers[0],
            limits=McpLimits(result_bytes=1024),
            environment={},
        )
    )
    try:
        result = await choose(await client.discover(), "list").execute("id", {})
        assert not result.is_error and result.details["truncated"]
        payload = result.structured_content["structuredContent"]
        assert payload["nextCursor"] == "page-2"
        assert payload["resources"][0]["description"] == "\x01" * 10000
        path = Path(result.details["full_output_path"])
        assert json.loads(path.read_text()) == payload
        path.unlink()
    finally:
        await client.aclose()


async def test_invalid_resource_cursor_has_constant_diagnostic(tmp_path, monkeypatch):
    Peer(monkeypatch, mode="bad-cursor")
    client = connection(tmp_path)
    try:
        with pytest.raises(McpError, match="Invalid MCP resource cursor"):
            await choose(await client.discover(), "list").execute("id", {})
    finally:
        await client.aclose()
