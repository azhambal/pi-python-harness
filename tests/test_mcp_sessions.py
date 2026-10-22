from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from pi_agent import (
    AfterToolCallResult,
    BeforeToolCallResult,
    PermissionPolicy,
    PermissionRule,
    ToolHooks,
    create_harness,
)
from pi_agent.events import ToolExecutionUpdate, event_to_json
from pi_agent.mcp import McpError, McpRuntime, binding_name
from pi_agent.providers import FauxProvider
from pi_agent.session_store import SessionStore
from pi_agent.tools import create_coding_tools
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage
from test_mcp_runtime import assert_reaped, settings, wait_event


def response(name="echo", args=None):
    return AssistantMessage(
        content=[
            ToolCallBlock(
                id="call",
                name=binding_name("fixture", name),
                arguments={} if args is None else args,
            )
        ],
        stop_reason="toolUse",
    )


def done():
    return AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop")


async def test_mcp_runs_through_hooks_events_and_journal(config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-secret")
    stages = []

    def before(context, token):
        stages.append("before")
        assert context.tool_provenance.kind == "mcp"
        assert context.tool_provenance.server == "fixture"
        assert context.tool_provenance.tool == "echo"
        assert json.loads(context.tool_provenance.annotations_json)["readOnlyHint"] is True
        context.args["message"] = "discarded"

    def after(context, token):
        stages.append("after")
        assert context.tool_provenance.tool == "echo"
        assert context.result.structured_content["structuredContent"]["echo"]["message"] == "hello"
        return AfterToolCallResult(content=[TextBlock(text="host final")])

    runtime = McpRuntime(settings(config.cwd), builtins=tuple(create_coding_tools(config.cwd)))
    provider = FauxProvider([response(args={"message": "hello"}), done()])
    session = await create_harness(
        provider,
        config=config,
        session="new",
        prepare_tools=runtime.prepare_tools,
        hooks=ToolHooks(before, after),
    )
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("echo")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert stages == ["before", "after"]
        assert result.content == [TextBlock(text="host final")]
        assert result.structured_content is None
        assert any(isinstance(event, ToolExecutionUpdate) for event in events)
        journal = session.session_file.read_text()
        serialized = json.dumps([event_to_json(event) for event in events])
        for secret in ("synthetic-provider-secret", "synthetic-meta-secret"):
            assert secret not in journal and secret not in serialized
        assert "mcpServers" not in journal and "mcp_harness_server.py" not in journal
        assert set(tool.name for tool in session.agent.tools) >= {"read", "bash", "edit", "write"}
        with pytest.raises(RuntimeError, match="aclose"):
            session.dispose()
    finally:
        await session.aclose()
    assert_reaped(config.cwd)
    reopened = SessionStore.open(session.session_file)
    reopened.close()


@pytest.mark.parametrize("kind", ["read-only", "deny", "ask-no-callback", "ask-denied", "hook-denied"])
async def test_admission_never_calls_server_when_blocked(config, kind):
    confirms = []

    def confirm(request, token):
        confirms.append(request.context.tool_provenance.tool)
        return False

    policy = PermissionPolicy(
        read_only=kind == "read-only",
        default="ask" if kind.startswith("ask") else "allow",
        rules=(PermissionRule("mcp__*", "deny"),) if kind == "deny" else (),
    )
    hooks = (
        ToolHooks(before_tool_call=lambda ctx, token: BeforeToolCallResult(block=True, reason="host denied"))
        if kind == "hook-denied"
        else None
    )
    runtime = McpRuntime(settings(config.cwd))
    session = await create_harness(
        FauxProvider([response(), done()]),
        config=config,
        permission_policy=policy,
        prepare_tools=runtime.prepare_tools,
        hooks=hooks,
        confirm_permission=confirm if kind == "ask-denied" else None,
    )
    try:
        await session.prompt("blocked")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error
        assert "call:" not in (config.cwd / "events").read_text()
        assert confirms == (["echo"] if kind == "ask-denied" else [])
    finally:
        await session.aclose()


async def test_ask_approved_has_provenance_and_executes(config):
    runtime = McpRuntime(settings(config.cwd))

    def confirm(request, token):
        assert request.context.tool_provenance.tool == "echo"
        assert request.context.tool_provenance.schema_digest
        return True

    session = await create_harness(
        FauxProvider([response(), done()]),
        config=config,
        permission_policy=PermissionPolicy(default="ask"),
        confirm_permission=confirm,
        prepare_tools=runtime.prepare_tools,
    )
    try:
        await session.prompt("approved")
        assert "call:echo" in (config.cwd / "events").read_text()
    finally:
        await session.aclose()


async def test_invalid_arguments_fail_before_hook_permission_or_server(config):
    stages = []
    runtime = McpRuntime(settings(config.cwd))
    session = await create_harness(
        FauxProvider([response(args={"message": 42}), done()]),
        config=config,
        hooks=ToolHooks(before_tool_call=lambda ctx, token: stages.append("before")),
        prepare_tools=runtime.prepare_tools,
    )
    try:
        await session.prompt("invalid")
        assert stages == []
        assert "call:" not in (config.cwd / "events").read_text()
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and "MCP arguments failed validation" in result.content[0].text
    finally:
        await session.aclose()


async def test_abort_closes_server_then_session_can_close(config):
    runtime = McpRuntime(settings(config.cwd))
    session = await create_harness(
        FauxProvider([response("slow"), done()]), config=config, prepare_tools=runtime.prepare_tools
    )
    pending = asyncio.create_task(session.prompt("slow"))
    await wait_event(config.cwd, "call:slow")
    await session.abort()
    await pending
    assert_reaped(config.cwd)
    await session.aclose()


async def test_failed_core_construction_reaps_connected_server(config, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic constructor failure")

    monkeypatch.setattr("pi_agent.harness.create_agent_session", fail)
    runtime = McpRuntime(settings(config.cwd))
    with pytest.raises(ValueError, match="constructor failure"):
        await create_harness(FauxProvider([]), config=config, prepare_tools=runtime.prepare_tools)
    assert_reaped(config.cwd)


async def test_bad_schema_harness_construction_reaps_and_never_opens_journal(config):
    runtime = McpRuntime(settings(config.cwd, "bad-schema"))
    with pytest.raises(McpError, match="discovery failed"):
        await create_harness(
            FauxProvider([]), config=config, session="new", prepare_tools=runtime.prepare_tools
        )
    assert_reaped(config.cwd)
    assert not config.session_dir.exists()


async def test_structured_only_result_visible_to_provider_and_replayed(config):
    from pi_agent.types import message_from_json

    runtime = McpRuntime(settings(config.cwd, "structured-only"))
    provider = FauxProvider([response(), done()])
    session = await create_harness(
        provider, config=config, session="new", prepare_tools=runtime.prepare_tools
    )
    try:
        await session.prompt("structured")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.structured_content["structuredContent"]["provider_key_present"] is False
        assert "provider_key_present" in result.content[0].text
        assert provider.requests[-1].messages[-1] == result
        assert message_from_json(result.to_json()) == result
    finally:
        await session.aclose()
    store = SessionStore.open(session.session_file)
    try:
        restored = next(
            entry.message
            for entry in store.entries
            if hasattr(entry, "message") and isinstance(entry.message, ToolResultMessage)
        )
        assert restored == result
    finally:
        store.close()
