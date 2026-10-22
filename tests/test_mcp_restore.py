from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from pi_agent import ResourceOwner, create_agent_session, create_harness
from pi_agent.mcp import McpError, McpManifest, McpRuntime, binding_name, parse_mcp_config
from pi_agent.providers import FauxProvider
from pi_agent.session_store import SessionLockError, SessionStore
from pi_agent.tools import create_coding_tools
from pi_agent.types import (
    AssistantMessage,
    SystemMessage,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    get_current_system_message,
)

FIXTURE = Path(__file__).parent / "fixtures/mcp_harness_server.py"


def settings(directory, *, unused=False, timeout=2):
    servers = {
        "fixture": {
            "command": sys.executable,
            "args": [str(FIXTURE), str(directory), "normal"],
            "exposure": "direct",
            "timeout": timeout,
        }
    }
    if unused:
        servers["unused"] = {"command": "/must/not/start", "exposure": "direct"}
    return parse_mcp_config({"mcpServers": servers}, cwd=directory)


def done(text="done"):
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop")


def call(name="echo"):
    return AssistantMessage(
        content=[ToolCallBlock(id="call", name=binding_name("fixture", name), arguments={})],
        stop_reason="toolUse",
    )


def assert_dead(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def pid(directory):
    return int((directory / "pid").read_text())


async def seed(config, *, names=("fixture/echo",), provider=None):
    session = await create_harness(
        provider or FauxProvider([done()]),
        config=config,
        session="new",
        mcp_config=settings(config.cwd),
        mcp_tools=names,
    )
    await session.prompt("seed")
    path = session.session_file
    await session.aclose()
    return path


async def test_default_resume_exact_names_ignores_new_tools_and_unused_servers(config):
    path = await seed(config)
    before = path.read_bytes()
    (config.cwd / "variant").write_text("extra")
    session = await create_harness(
        FauxProvider([]), session=path, mcp_config=settings(config.cwd, unused=True)
    )
    try:
        assert {tool.name for tool in session.agent.tools} == {
            "read",
            "bash",
            "edit",
            "write",
            binding_name("fixture", "echo"),
        }
        assert [binding.tool for binding in session.mcp_bindings] == ["echo"]
        assert session.mcp_server_states["unused"] == "inactive"
        assert path.read_bytes() == before
    finally:
        await session.aclose()


@pytest.mark.parametrize("variant", ["input", "output", "description", "removed"])
async def test_changed_or_missing_binding_rejects_resume_without_journal_mutation(config, variant):
    path = await seed(config)
    before = path.read_bytes()
    (config.cwd / "variant").write_text(variant)
    with pytest.raises((ValueError, McpError)):
        await create_harness(FauxProvider([]), session=path, mcp_config=settings(config.cwd))
    assert path.read_bytes() == before
    assert_dead(pid(config.cwd))
    store = SessionStore.open(path)
    store.close()


async def test_selection_does_not_silently_accept_retained_schema_changes(config):
    path = await seed(config)
    (config.cwd / "variant").write_text("output")
    with pytest.raises(ValueError, match="changed"):
        await create_harness(
            FauxProvider([]), session=path, mcp_config=settings(config.cwd), mcp_tools=("fixture/echo",)
        )


async def test_explicit_refresh_accepts_new_digest_in_one_record(config):
    path = await seed(config)
    before = SessionStore.open(path)
    count = before.entry_count
    before.close()
    (config.cwd / "variant").write_text("input")
    session = await create_harness(
        FauxProvider([]), session=path, mcp_config=settings(config.cwd), refresh_mcp=True
    )
    try:
        assert session.store.entry_count == count + 1
        record = session.store.last_entry.message
        assert isinstance(record, SystemMessage) and record.mcp_manifest and record.tools_added
        assert record.tools_added[0].parameters["properties"]["count"] == {"type": "integer"}
        assert McpManifest.from_manifest(record.mcp_manifest) == McpManifest.from_tools(session.agent.tools)
        assert "mcp_harness_server.py" not in path.read_text()
    finally:
        await session.aclose()
    restored = await create_harness(FauxProvider([]), session=path, mcp_config=settings(config.cwd))
    await restored.aclose()


async def test_offline_removal_requires_no_config_or_process_and_survives_resume(config):
    path = await seed(config)
    old_pid = pid(config.cwd)
    with pytest.raises(ValueError):
        create_agent_session(FauxProvider([]), session=path)
    session = await create_harness(FauxProvider([]), session=path, mcp_tools=())
    try:
        assert session.mcp_bindings == () and pid(config.cwd) == old_pid
        record = session.store.last_entry.message
        assert record.mcp_manifest and record.tools_removed
        assert all(not tool.name.startswith("mcp__") for tool in session.agent.tools)
        assert "mcp__" not in session.system_prompt
    finally:
        await session.aclose()
    restored = create_agent_session(FauxProvider([]), session=path)
    restored.dispose()


async def test_resume_without_trusted_config_never_launches_from_manifest(config):
    path = await seed(config)
    old_pid = pid(config.cwd)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        await create_harness(FauxProvider([]), session=path)
    with pytest.raises(McpError, match="trusted configuration"):
        await create_harness(FauxProvider([]), session=path, refresh_mcp=True)
    assert pid(config.cwd) == old_pid and path.read_bytes() == before


async def test_legacy_builtin_journal_does_not_activate_configured_extras(config):
    old = create_agent_session(FauxProvider([done()]), config=config, session="new")
    await old.prompt("legacy")
    path = old.session_file
    old.dispose()
    session = await create_harness(
        FauxProvider([]), session=path, mcp_config=settings(config.cwd, unused=True)
    )
    try:
        assert session.mcp_bindings == ()
        assert not (config.cwd / "pid").exists()
    finally:
        await session.aclose()


async def test_lock_is_held_during_resume_connection_prepare(config):
    path = await seed(config)
    (config.cwd / "variant").write_text("hang")
    pending = asyncio.create_task(
        create_harness(FauxProvider([]), session=path, mcp_config=settings(config.cwd, timeout=10))
    )
    old_pid = pid(config.cwd)
    async with asyncio.timeout(3):
        while pid(config.cwd) == old_pid:
            await asyncio.sleep(0.01)
    with pytest.raises(SessionLockError):
        SessionStore.open(path)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert_dead(pid(config.cwd))
    reopened = SessionStore.open(path)
    reopened.close()


async def test_reconnect_only_future_call_and_strict_digest_check(config):
    provider = FauxProvider([call("crash"), done(), call(), done()])
    session = await create_harness(
        provider,
        config=config,
        session="new",
        mcp_config=settings(config.cwd),
        mcp_tools=("fixture/echo", "fixture/crash"),
    )
    try:
        await session.prompt("crash")
        assert session.mcp_server_states["fixture"] in ("failed", "closed")
        assert (config.cwd / "events").read_text().splitlines().count("call:crash") == 1
        before = session.store.entry_count
        await session.reconnect_mcp("fixture")
        assert session.store.entry_count == before
        await session.prompt("new request")
        assert (config.cwd / "events").read_text().splitlines().count("call:crash") == 1
        assert "call:echo" in (config.cwd / "events").read_text()
        results = [message for message in session.messages if isinstance(message, ToolResultMessage)]
        assert results[0].is_error and not results[-1].is_error
        old_tools = tuple(session.agent.tools)
        old_pid = pid(config.cwd)
        (config.cwd / "variant").write_text("output")
        with pytest.raises(McpError, match="changed"):
            await session.reconnect_mcp("fixture")
        assert tuple(session.agent.tools) == old_tools
        os.kill(old_pid, 0)
        assert_dead(pid(config.cwd))
    finally:
        await session.aclose()


async def test_reload_updates_tools_manifest_prompt_before_publish(config):
    session = await create_harness(
        FauxProvider([]),
        config=config,
        session="new",
        mcp_config=settings(config.cwd),
        mcp_tools=("fixture/echo",),
    )
    try:
        before = session.store.entry_count
        old_pid = pid(config.cwd)
        await session.reload_mcp(("fixture/echo", "fixture/error"))
        assert session.store.entry_count == before + 1
        record = session.store.last_entry.message
        assert record.tools_added and record.mcp_manifest and record.sections
        assert {binding.tool for binding in session.mcp_bindings} == {"echo", "error"}
        assert binding_name("fixture", "error") in session.system_prompt
        assert_dead(old_pid)
        await session.reload_mcp(())
        assert session.mcp_bindings == () and session.mcp_server_states["fixture"] == "inactive"
    finally:
        await session.aclose()


async def test_fsync_failure_does_not_publish_candidates(config, monkeypatch):
    session = await create_harness(
        FauxProvider([done()]),
        config=config,
        session="new",
        mcp_config=settings(config.cwd),
        mcp_tools=("fixture/echo",),
    )
    old_tools = tuple(session.agent.tools)
    old_pid = pid(config.cwd)
    try:
        await session.prompt("persist before testing a failed durable update")

        def fail(_fd):
            raise OSError("synthetic fsync failure")

        with monkeypatch.context() as patch:
            patch.setattr("pi_agent.session_store.os.fsync", fail)
            with pytest.raises(OSError):
                await session.reload_mcp(("fixture/echo", "fixture/error"))
        assert tuple(session.agent.tools) == old_tools
        os.kill(old_pid, 0)
        assert_dead(pid(config.cwd))
        assert not session.store.usable
        path = session.session_file
    finally:
        await session.aclose()
    # The failed write may have reached disk; opening adopts the coherent record
    # actually recovered. No split manifest/declaration patch can be observed.
    store = SessionStore.open(path)
    try:
        messages = [entry.message for entry in store.entries if hasattr(entry, "message")]
        current = get_current_system_message(messages)
        manifest = McpManifest.from_manifest(current.mcp_manifest)
        manifest.validate_declarations(current.tools_added)
    finally:
        store.close()


async def test_cancelled_reload_closes_candidate_preserves_old_registry(config):
    session = await create_harness(
        FauxProvider([]),
        config=config,
        mcp_config=settings(config.cwd, timeout=10),
        mcp_tools=("fixture/echo",),
    )
    old_pid = pid(config.cwd)
    old_tools = tuple(session.agent.tools)
    (config.cwd / "variant").write_text("hang")
    pending = asyncio.create_task(session.reload_mcp())
    try:
        async with asyncio.timeout(3):
            while pid(config.cwd) == old_pid:
                await asyncio.sleep(0.01)
        assert not session.is_idle
        with pytest.raises(RuntimeError, match="busy"):
            await session.prompt("must wait")
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert tuple(session.agent.tools) == old_tools and session.is_idle
        os.kill(old_pid, 0)
        assert_dead(pid(config.cwd))
    finally:
        await session.aclose()


async def test_compaction_preserves_manifest_and_restore(config):
    provider = FauxProvider([done("seed"), done("second"), done("summary")])
    session = await create_harness(
        provider,
        config=replace(config, compaction=replace(config.compaction, keep_recent_tokens=0)),
        session="new",
        mcp_config=settings(config.cwd),
        mcp_tools=("fixture/echo",),
    )
    try:
        await session.prompt("seed")
        await session.prompt("second")
        recorded = session.mcp_bindings
        await session.compact()
        assert session.mcp_bindings == recorded
        path = session.session_file
    finally:
        await session.aclose()
    restored = await create_harness(FauxProvider([]), session=path, mcp_config=settings(config.cwd))
    try:
        assert restored.mcp_bindings == recorded
    finally:
        await restored.aclose()


async def test_unfinished_recorded_call_recovers_unknown_outcome_without_replay(config):
    path = await seed(config)
    store = SessionStore.open(path)
    try:
        store.append_message(
            AssistantMessage(
                content=[
                    ToolCallBlock(
                        id="unfinished",
                        name=binding_name("fixture", "echo"),
                        arguments={},
                    )
                ],
                stop_reason="toolUse",
            )
        )
    finally:
        store.close()
    session = await create_harness(FauxProvider([]), session=path, mcp_config=settings(config.cwd))
    try:
        result = next(
            message
            for message in session.messages
            if isinstance(message, ToolResultMessage) and message.tool_call_id == "unfinished"
        )
        assert result.interrupted and result.is_error and "unknown" in result.content[0].text
        assert "call:echo" not in (config.cwd / "events").read_text()
    finally:
        await session.aclose()


async def test_manifest_declaration_mismatch_fails_before_server_launch(config):
    path = await seed(config)
    old_pid = pid(config.cwd)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        message = row.get("message", {})
        if message.get("mcp_manifest"):
            for tool in message.get("tools_added", []):
                if tool["name"].startswith("mcp__"):
                    tool["description"] = "tampered declaration"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="manifest does not match"):
        await create_harness(FauxProvider([]), session=path, mcp_config=settings(config.cwd))
    assert pid(config.cwd) == old_pid and path.read_bytes() == before


async def test_abort_pending_reload_joins_and_preserves_current_tools(config):
    session = await create_harness(
        FauxProvider([]),
        config=config,
        mcp_config=settings(config.cwd, timeout=20),
        mcp_tools=("fixture/echo",),
    )
    old_pid = pid(config.cwd)
    old_tools = tuple(session.agent.tools)
    (config.cwd / "variant").write_text("hang")
    pending = asyncio.create_task(session.reload_mcp())
    try:
        async with asyncio.timeout(3):
            while pid(config.cwd) == old_pid:
                await asyncio.sleep(0.01)
        await session.abort()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert session.is_idle and tuple(session.agent.tools) == old_tools
        assert_dead(pid(config.cwd))
        os.kill(old_pid, 0)
    finally:
        await session.aclose()


async def test_explicit_saved_canonical_selector_scopes_server_on_resume(config):
    path = await seed(config)
    session = await create_harness(
        FauxProvider([]),
        session=path,
        mcp_config=settings(config.cwd, unused=True),
        mcp_tools=(binding_name("fixture", "echo"),),
    )
    try:
        assert session.mcp_server_states["unused"] == "inactive"
        assert [binding.tool for binding in session.mcp_bindings] == ["echo"]
    finally:
        await session.aclose()


async def test_legacy_matching_host_injection_adopts_manifest(config):
    path = await seed(config)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.get("message", {}).pop("mcp_manifest", None)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    runtime = McpRuntime(
        settings(config.cwd), builtins=tuple(create_coding_tools(config.cwd)), selectors=("fixture/echo",)
    )
    owner = ResourceOwner()
    prepared = await runtime.prepare_tools(owner)
    session = None
    try:
        session = create_agent_session(FauxProvider([]), session=path, tools=prepared)
        assert [binding.tool for binding in session.mcp_bindings] == ["echo"]
        assert session.store.last_entry.message.mcp_manifest
        assert session.store.last_entry.message.tools_added is None
    finally:
        if session is not None:
            session.dispose()
        await owner.aclose()


async def test_legacy_injection_cannot_activate_new_discovery_extra(config):
    path = await seed(config)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.get("message", {}).pop("mcp_manifest", None)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = path.read_bytes()
    owner = ResourceOwner()
    runtime = McpRuntime(settings(config.cwd), builtins=tuple(create_coding_tools(config.cwd)))
    prepared = await runtime.prepare_tools(owner)
    try:
        with pytest.raises(ValueError, match="manifest does not match"):
            create_agent_session(FauxProvider([]), session=path, tools=prepared)
        assert path.read_bytes() == before
    finally:
        await owner.aclose()


async def test_retirement_failure_reports_committed_loadout_and_retains_owner(config, monkeypatch):
    session = await create_harness(
        FauxProvider([call("error"), done()]),
        config=config,
        session="new",
        mcp_config=settings(config.cwd),
        mcp_tools=("fixture/echo",),
    )
    old = next(tool for tool in session.agent.tools if tool.provenance is not None)._connection
    close = old.aclose
    attempts = 0

    async def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic retirement failure")
        await close()

    monkeypatch.setattr(old, "aclose", fail_once)
    try:
        with pytest.raises(McpError, match="loadout committed; cleanup pending"):
            await session.reload_mcp(("fixture/echo", "fixture/error"))
        assert session.is_idle and session.store.usable
        assert {binding.tool for binding in session.mcp_bindings} == {"echo", "error"}
        assert old in session._mcp_runtime._owned_connections
        await session.prompt("new tools remain usable")
        assert "call:error" in (config.cwd / "events").read_text()
    finally:
        await session.aclose()
    assert attempts >= 2


@pytest.mark.parametrize("selector", ["*", "fixture/echo", True, [True]])
async def test_invalid_selection_never_starts_a_process(config, selector):
    with pytest.raises(ValueError, match="sequence"):
        await create_harness(
            FauxProvider([]), config=config, mcp_config=settings(config.cwd), mcp_tools=selector
        )
    assert not (config.cwd / "pid").exists()
