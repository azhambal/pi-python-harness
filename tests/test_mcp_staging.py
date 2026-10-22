from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from pi_agent import ResourceOwner
from pi_agent.mcp import McpConfig, McpConnection, McpError, McpRuntime, binding_name, parse_mcp_config
from pi_agent.mcp.runtime import McpStage
from pi_agent.tools.base import Tool, ToolResult

FIXTURE = Path(__file__).parent / "fixtures/mcp_harness_server.py"


class Builtin(Tool):
    name = "builtin"

    async def execute(self, tool_call_id, args, cancellation=None, on_update=None):
        return ToolResult()


def settings(directory, modes=None, disabled=()):
    modes = {"one": "normal", "two": "normal"} if modes is None else modes
    entries = {}
    for name, mode in modes.items():
        child = directory / name
        child.mkdir(exist_ok=True)
        entries[name] = {
            "command": sys.executable,
            "args": [str(FIXTURE), str(child), mode],
            "cwd": str(child),
            "timeout": 5,
            "exposure": "direct",
            "enabled": name not in disabled,
        }
    return parse_mcp_config({"mcpServers": entries}, cwd=directory)


async def wait_event(directory, event):
    async with asyncio.timeout(5):
        while True:
            path = directory / "events"
            if path.exists() and event in path.read_text().splitlines():
                return
            await asyncio.sleep(0.01)


def assert_reaped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def pid(directory):
    return int((directory / "pid").read_text())


def by_binding(tools):
    return {tool.name: tool for tool in tools}


@pytest.mark.parametrize("selector", ["one/echo", binding_name("one", "echo")])
async def test_initial_filters_exact_servers_and_preserves_all_builtins(tmp_path, selector):
    builtin = Builtin()
    runtime = McpRuntime(
        settings(
            tmp_path,
            {"one": "normal", "two": "initialize-hang", "disabled": "normal"},
            disabled=("disabled",),
        ),
        builtins=(builtin,),
        selectors=[selector],
        servers=["one"],
        environment={},
    )
    owner = ResourceOwner()
    try:
        assert runtime.server_states == {"one": "inactive", "two": "inactive", "disabled": "disabled"}
        tools = await runtime.prepare_tools(owner)
        assert tools[0] is builtin
        assert [tool.name for tool in tools] == ["builtin", binding_name("one", "echo")]
        assert runtime.tools == tools
        assert runtime.server_states == {"one": "ready", "two": "inactive", "disabled": "disabled"}
        assert not (tmp_path / "two/pid").exists()
        assert not (tmp_path / "disabled/pid").exists()
        assert owner.pending
    finally:
        await owner.aclose()
    assert_reaped(pid(tmp_path / "one"))


@pytest.mark.parametrize("servers", [None, ["one"], []])
async def test_empty_selectors_never_start_processes(tmp_path, servers):
    builtin = Builtin()
    runtime = McpRuntime(settings(tmp_path), builtins=(builtin,), selectors=[], servers=servers)
    owner = ResourceOwner()
    try:
        assert await runtime.prepare_tools(owner) == (builtin,)
        assert runtime.server_states == {"one": "inactive", "two": "inactive"}
        stage = await runtime.stage([], servers=servers)
        assert stage.tools == () and stage.connections == ()
        await stage.aclose()
        assert not list(tmp_path.glob("*/pid"))
    finally:
        await owner.aclose()


@pytest.mark.parametrize("servers", [["absent"], ["disabled"], ["one", "one"], "one"])
def test_invalid_initial_server_selection_before_launch(tmp_path, servers):
    config = settings(tmp_path, {"one": "normal", "disabled": "normal"}, disabled=("disabled",))
    with pytest.raises(McpError):
        McpRuntime(config, servers=servers)
    assert not list(tmp_path.glob("*/pid"))


@pytest.mark.parametrize("selectors", ["one/echo", ["one/echo", "one/echo"], [""], [1]])
def test_invalid_initial_selectors_before_launch(tmp_path, selectors):
    with pytest.raises(McpError):
        McpRuntime(settings(tmp_path), selectors=selectors)
    assert not list(tmp_path.glob("*/pid"))


async def test_alias_selector_avoids_unrelated_server_launch(tmp_path):
    runtime = McpRuntime(settings(tmp_path), selectors=["one/echo"], environment={})
    owner = ResourceOwner()
    try:
        tools = await runtime.prepare_tools(owner)
        assert len(tools) == 1
        assert not (tmp_path / "two/pid").exists()
        assert runtime.server_states == {"one": "ready", "two": "inactive"}
    finally:
        await owner.aclose()
    assert_reaped(pid(tmp_path / "one"))


async def test_reload_uses_full_config_and_none_means_fresh_all(tmp_path):
    runtime = McpRuntime(settings(tmp_path), selectors=["one/echo"], servers=["one"], environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    old_pid = pid(tmp_path / "one")
    try:
        stage = await runtime.stage(None)
        assert isinstance(stage, McpStage)
        assert len(stage.tools) == 20
        assert all(tool.provenance.kind == "mcp" for tool in stage.tools)
        assert len(stage.connections) == 2
        assert runtime.tools is initial
        assert runtime.server_states["two"] == "inactive"
        assert not stage.committed
        runtime.validate_stage(stage, stage.tools)
        runtime.publish(stage, stage.tools)
        assert stage.committed and runtime.tools == stage.tools
        assert runtime.server_states == {"one": "ready", "two": "ready"}
        assert (await initial[0].execute("old", {})).is_error is False
        await runtime.retire_previous(stage)
        assert_reaped(old_pid)
        assert not (await runtime.tools[0].execute("new", {})).is_error
        with pytest.raises(McpError, match="committed"):
            await stage.aclose()
        await runtime.retire_previous(stage)
    finally:
        await owner.aclose()


async def test_partial_reconnect_retains_other_active_connections_and_bindings(tmp_path):
    builtin = Builtin()
    runtime = McpRuntime(settings(tmp_path), builtins=(builtin,), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    old_one = pid(tmp_path / "one")
    old_two = pid(tmp_path / "two")
    retained = tuple(tool for tool in initial if tool.provenance and tool.provenance.server == "two")
    try:
        names = [tool.name for tool in initial if tool.provenance and tool.provenance.server == "one"]
        stage = await runtime.stage(names, servers=["one"])
        assert len(stage.tools) == 10 and len(stage.connections) == 1
        assert all(tool.provenance.server == "one" for tool in stage.tools)
        complete = (builtin, *stage.tools, *retained)
        runtime.validate_stage(stage, complete)
        runtime.publish(stage, complete)
        assert runtime.tools == complete
        assert pid(tmp_path / "two") == old_two
        await runtime.retire_previous(stage)
        assert_reaped(old_one)
        assert not (await retained[0].execute("retained", {})).is_error
        assert not (await stage.tools[0].execute("reconnected", {})).is_error
    finally:
        await owner.aclose()
    assert_reaped(old_two)


async def test_discard_preserves_active_registry_and_is_idempotent(tmp_path):
    runtime = McpRuntime(settings(tmp_path), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    active = tuple(runtime._connections)
    old_pids = [pid(tmp_path / name) for name in ("one", "two")]
    try:
        stage = await runtime.stage(["one/echo"], servers=["one"])
        candidate = pid(tmp_path / "one")
        await asyncio.gather(stage.aclose(), stage.aclose())
        assert_reaped(candidate)
        assert runtime._connections == active and runtime.tools is initial
        for old_pid in old_pids:
            os.kill(old_pid, 0)
        assert not (await initial[0].execute("still-active", {})).is_error
        with pytest.raises(McpError, match="cannot be published"):
            runtime.validate_stage(stage)
    finally:
        await owner.aclose()
    for old_pid in old_pids:
        assert_reaped(old_pid)


@pytest.mark.parametrize("selectors", [["one/missing"], ["one/echo", binding_name("one", "echo")]])
async def test_selection_failure_reaps_all_candidates_and_preserves_active(tmp_path, selectors):
    runtime = McpRuntime(settings(tmp_path), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    active = runtime._connections
    try:
        with pytest.raises(McpError):
            await runtime.stage(selectors, servers=["one", "two"])
        assert_reaped(pid(tmp_path / "one"))
        assert_reaped(pid(tmp_path / "two"))
        assert runtime._connections is active and runtime.tools is initial
        assert not (await initial[0].execute("still-active", {})).is_error
        assert runtime._owned_connections == list(active)
    finally:
        await owner.aclose()


async def test_discovery_failure_reaps_prior_candidates_and_keeps_old_bindings(tmp_path):
    runtime = McpRuntime(settings(tmp_path), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    active = runtime._connections
    runtime._config = McpConfig(
        tuple(
            replace(server, args=(*server.args[:-1], "bad-schema")) if server.name == "two" else server
            for server in runtime._config.servers
        )
    )
    try:
        with pytest.raises(McpError, match="discovery failed"):
            await runtime.stage()
        assert_reaped(pid(tmp_path / "one"))
        assert_reaped(pid(tmp_path / "two"))
        assert runtime._connections is active and runtime.tools is initial
        assert not (await initial[0].execute("still-active", {})).is_error
    finally:
        await owner.aclose()


@pytest.mark.parametrize("method", ["stage", "discard", "publish", "validate", "retire"])
async def test_callback_reentry_rejected_without_deadlock(tmp_path, method):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    stage = await runtime.stage(["one/echo"])
    errors = []

    async def update(_value):
        try:
            if method == "stage":
                await runtime.stage()
            elif method == "discard":
                await stage.aclose()
            elif method == "publish":
                runtime.publish(stage, stage.tools)
            elif method == "validate":
                runtime.validate_stage(stage)
            else:
                await runtime.retire_previous(stage)
        except RuntimeError as error:
            errors.append(str(error))

    try:
        async with asyncio.timeout(3):
            assert not (await initial[0].execute("callback", {}, on_update=update)).is_error
        assert errors and all("resource callback" in error for error in errors)
        assert runtime.tools is initial and not stage.committed
    finally:
        await owner.aclose()


async def test_ownership_stale_and_closed_stage_validations(tmp_path):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    other = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    first = await runtime.stage(["one/echo"])
    second = await runtime.stage(["one/echo"])
    try:
        with pytest.raises(McpError, match="another runtime"):
            other.validate_stage(first)
        with pytest.raises(McpError):
            await other.retire_previous(first)
        runtime.publish(first, first.tools)
        with pytest.raises(McpError, match="stale"):
            runtime.validate_stage(second)
        with pytest.raises(McpError):
            runtime.publish(first, first.tools)
        await second.aclose()
        await runtime.retire_previous(first)
        await runtime.aclose()
        with pytest.raises(McpError, match="closed"):
            await runtime.stage()
    finally:
        await runtime.aclose()
        await other.aclose()


async def test_server_states_include_active_failure_without_secrets(tmp_path):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    owner = ResourceOwner()
    tools = by_binding(await runtime.prepare_tools(owner))
    try:
        with pytest.raises(McpError):
            await tools[binding_name("one", "crash")].execute("crash", {})
        assert runtime.server_states["one"] in ("failed", "closed")
        assert "command" not in repr(runtime._config)
        assert "args" not in repr(runtime._config)
        assert str(tmp_path) not in repr(runtime._config)
        snapshot = runtime.server_states
        snapshot["one"] = "changed"
        assert runtime.server_states["one"] != "changed"
    finally:
        await owner.aclose()


async def test_staging_singleflight_and_close_joins_queued_setup(tmp_path, monkeypatch):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    entered = asyncio.Event()
    gate = asyncio.Event()
    original = McpConnection.discover
    in_flight = maximum = 0

    async def blocked(connection):
        nonlocal in_flight, maximum
        in_flight += 1
        maximum = max(maximum, in_flight)
        entered.set()
        try:
            await gate.wait()
            return await original(connection)
        finally:
            in_flight -= 1

    monkeypatch.setattr(McpConnection, "discover", blocked)
    first = asyncio.create_task(runtime.stage())
    await entered.wait()
    second = asyncio.create_task(runtime.stage())
    await asyncio.sleep(0)
    try:
        async with asyncio.timeout(3):
            await runtime.aclose()
            await asyncio.gather(first, second, return_exceptions=True)
        assert maximum == 1
        assert not runtime._owned_connections
        assert not list(tmp_path.glob("*/pid"))
    finally:
        gate.set()
        await runtime.aclose()


async def test_concurrent_successful_stages_discover_serially(tmp_path, monkeypatch):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    original = McpConnection.discover
    in_flight = maximum = 0

    async def measured(connection):
        nonlocal in_flight, maximum
        in_flight += 1
        maximum = max(maximum, in_flight)
        try:
            return await original(connection)
        finally:
            in_flight -= 1

    monkeypatch.setattr(McpConnection, "discover", measured)
    try:
        stages = await asyncio.gather(runtime.stage(), runtime.stage())
        assert maximum == 1
        assert stages[0].connections[0] is not stages[1].connections[0]
        await asyncio.gather(*(stage.aclose() for stage in stages))
        assert not runtime._owned_connections
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("action", ["cancel", "runtime-close"])
async def test_initializing_stage_settlement_and_old_binding_ownership(tmp_path, action):
    runtime = McpRuntime(
        settings(tmp_path, {"one": "normal", "two": "initialize-hang"}),
        selectors=["one/echo"],
        servers=["one"],
        environment={},
    )
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    old_pid = pid(tmp_path / "one")
    pending = asyncio.create_task(runtime.stage())
    await wait_event(tmp_path / "two", "initialize")
    try:
        async with asyncio.timeout(3):
            if action == "cancel":
                pending.cancel()
            else:
                await runtime.aclose()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert_reaped(pid(tmp_path / "one"))
        assert_reaped(pid(tmp_path / "two"))
        assert runtime.tools is initial
        if action == "cancel":
            os.kill(old_pid, 0)
            assert not (await initial[0].execute("old", {})).is_error
        else:
            assert_reaped(old_pid)
    finally:
        await owner.aclose()


async def test_runtime_close_joins_pending_spawn(tmp_path, monkeypatch):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    original = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    release = asyncio.Event()
    processes = []

    async def delayed(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)
    pending = asyncio.create_task(runtime.stage())
    await spawned.wait()
    closing = asyncio.create_task(runtime.aclose())
    try:
        await asyncio.sleep(0.05)
        assert not closing.done()
        closing.cancel()
        release.set()
        async with asyncio.timeout(3):
            with pytest.raises(asyncio.CancelledError):
                await closing
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert processes[0].returncode is not None
        assert_reaped(processes[0].pid)
        assert not runtime._owned_connections
    finally:
        release.set()
        await runtime.aclose()


@pytest.mark.parametrize("action", ["discard", "retire"])
async def test_cancelled_stage_cleanup_shield_joins_process_settlement(tmp_path, monkeypatch, action):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    owner = ResourceOwner()
    await runtime.prepare_tools(owner)
    old = runtime._connections[0]
    stage = await runtime.stage(["one/echo"])
    target = stage.connections[0] if action == "discard" else old
    original = target.aclose
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed():
        entered.set()
        await release.wait()
        await original()

    monkeypatch.setattr(target, "aclose", delayed)
    if action == "retire":
        runtime.publish(stage, stage.tools)
    closing = asyncio.create_task(stage.aclose() if action == "discard" else runtime.retire_previous(stage))
    await entered.wait()
    try:
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        release.set()
        async with asyncio.timeout(3):
            with pytest.raises(asyncio.CancelledError):
                await closing
        assert target._owner.done() and target._transport._connection.process.returncode is not None
        assert target not in runtime._owned_connections
        if action == "retire":
            assert stage.committed and runtime.tools == stage.tools
            assert not (await stage.tools[0].execute("new", {})).is_error
    finally:
        release.set()
        await owner.aclose()


@pytest.mark.parametrize("action", ["discard", "retire", "failed-setup"])
async def test_cleanup_failure_retained_for_retry_and_sanitized(tmp_path, monkeypatch, action):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    old = runtime._connections[0]
    original = McpConnection.aclose
    failing = []

    async def fail(connection):
        await original(connection)
        if connection is not old or action == "retire":
            failing.append(connection)
            raise RuntimeError("synthetic-cleanup-secret")

    try:
        if action == "failed-setup":
            monkeypatch.setattr(McpConnection, "aclose", fail)
            with pytest.raises(ExceptionGroup) as error:
                await runtime.stage(["one/absent"])
            assert runtime.tools is initial
        else:
            stage = await runtime.stage(["one/echo"])
            monkeypatch.setattr(McpConnection, "aclose", fail)
            if action == "retire":
                runtime.publish(stage, stage.tools)
            with pytest.raises(ExceptionGroup) as error:
                if action == "discard":
                    await stage.aclose()
                else:
                    await runtime.retire_previous(stage)
            assert runtime.tools == (stage.tools if action == "retire" else initial)
        assert failing and all(connection in runtime._owned_connections for connection in failing)
        assert "synthetic-cleanup-secret" not in str(error.value)
        assert all("synthetic-cleanup-secret" not in str(child) for child in error.value.exceptions)
        monkeypatch.setattr(McpConnection, "aclose", original)
        if action == "discard":
            await stage.aclose()
            assert not stage._pending
        elif action == "retire":
            await runtime.retire_previous(stage)
            assert not stage._previous
            assert not (await stage.tools[0].execute("committed", {})).is_error
        else:
            await owner.aclose()
        assert all(connection not in runtime._owned_connections for connection in failing)
    finally:
        monkeypatch.setattr(McpConnection, "aclose", original)
        await owner.aclose()


async def test_runtime_close_reaps_active_retired_and_unpublished_stages(tmp_path):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    owner = ResourceOwner()
    await runtime.prepare_tools(owner)
    old_pid = pid(tmp_path / "one")
    replacement = await runtime.stage(["one/echo"])
    new_pid = pid(tmp_path / "one")
    runtime.publish(replacement, replacement.tools)
    unpublished = await runtime.stage(["one/echo"])
    pending_pid = pid(tmp_path / "one")
    await owner.aclose()
    for child in (old_pid, new_pid, pending_pid):
        assert_reaped(child)
    assert not runtime._owned_connections
    await unpublished.aclose()
    await runtime.retire_previous(replacement)


async def test_partial_publish_retires_omitted_old_server_from_final_loadout(tmp_path):
    runtime = McpRuntime(settings(tmp_path), environment={})
    owner = ResourceOwner()
    await runtime.prepare_tools(owner)
    old_one, old_two = [pid(tmp_path / name) for name in ("one", "two")]
    try:
        stage = await runtime.stage(["one/echo"], servers=["one"])
        # Partial targets do not force retention of tools omitted by the parent.
        runtime.validate_stage(stage, stage.tools)
        runtime.publish(stage, stage.tools)
        assert runtime._connections == stage.connections
        assert runtime.server_states == {"one": "ready", "two": "inactive"}
        await runtime.retire_previous(stage)
        assert_reaped(old_one)
        assert_reaped(old_two)
        assert not (await stage.tools[0].execute("new", {})).is_error
    finally:
        await owner.aclose()


async def test_publish_uses_tool_pointers_and_retires_unused_candidates(tmp_path):
    builtin = Builtin()
    runtime = McpRuntime(settings(tmp_path), builtins=(builtin,), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    old_connections = runtime._connections
    old_one, old_two = [pid(tmp_path / name) for name in ("one", "two")]
    retained = tuple(tool for tool in initial if tool.provenance and tool.provenance.server == "two")
    try:
        stage = await runtime.stage()
        candidate_one, candidate_two = [pid(tmp_path / name) for name in ("one", "two")]
        new_one = next(tool for tool in stage.tools if tool.provenance.server == "one")
        complete = (builtin, new_one, *retained)
        runtime.validate_stage(stage, complete)
        runtime.publish(stage, complete)
        assert runtime._connections == (new_one._connection, old_connections[1])
        await runtime.retire_previous(stage)
        assert_reaped(old_one)
        assert_reaped(candidate_two)
        os.kill(old_two, 0)
        os.kill(candidate_one, 0)
        assert not (await retained[0].execute("retained", {})).is_error
        assert not (await new_one.execute("new", {})).is_error
    finally:
        await owner.aclose()


@pytest.mark.parametrize("loadout", ["builtin-only", "empty"])
async def test_final_loadout_can_remove_all_mcp_bindings(tmp_path, loadout):
    builtin = Builtin()
    runtime = McpRuntime(settings(tmp_path), builtins=(builtin,), environment={})
    owner = ResourceOwner()
    await runtime.prepare_tools(owner)
    old_pids = [pid(tmp_path / name) for name in ("one", "two")]
    try:
        stage = await runtime.stage(["one/echo"], servers=["one"])
        candidate = pid(tmp_path / "one")
        complete = (builtin,) if loadout == "builtin-only" else ()
        runtime.validate_stage(stage, complete)
        runtime.publish(stage, complete)
        assert runtime._connections == () and runtime.tools == complete
        assert runtime.server_states == {"one": "inactive", "two": "inactive"}
        await runtime.retire_previous(stage)
        for child in (*old_pids, candidate):
            assert_reaped(child)
    finally:
        await owner.aclose()


async def test_validate_stage_rejects_foreign_and_mixed_connection_pointers(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    runtime = McpRuntime(settings(first_dir, {"one": "normal"}), environment={})
    other = McpRuntime(settings(second_dir, {"one": "normal"}), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    stage = await runtime.stage(["one/echo"])
    foreign = await other.stage(["one/echo"])
    try:
        with pytest.raises(McpError, match="unavailable connection"):
            runtime.validate_stage(stage, foreign.tools)
        with pytest.raises(McpError, match="mixes connections"):
            runtime.validate_stage(stage, (*initial, *stage.tools))
        assert not stage.committed and runtime.tools is initial
        # Keeping an old connection instead of its fresh candidate is permitted.
        runtime.validate_stage(stage, initial)
        runtime.publish(stage, initial)
        await runtime.retire_previous(stage)
        assert not (await initial[0].execute("kept", {})).is_error
    finally:
        await owner.aclose()
        await other.aclose()


async def test_no_tools_server_is_settled_during_initial_preparation(tmp_path):
    runtime = McpRuntime(settings(tmp_path, {"one": "no-tools"}), environment={})
    owner = ResourceOwner()
    try:
        assert await runtime.prepare_tools(owner) == ()
        assert runtime.server_states == {"one": "inactive"}
        assert_reaped(pid(tmp_path / "one"))
        assert runtime._owned_connections == []
    finally:
        await owner.aclose()


async def test_failed_candidate_cleanup_racing_public_cancellation_keeps_retry_owner(tmp_path, monkeypatch):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    active = runtime._connections[0]
    active_pid = pid(tmp_path / "one")
    original = McpConnection.aclose
    entered = asyncio.Event()
    release = asyncio.Event()
    failed = []

    async def delayed_failure(connection):
        if connection is not active:
            entered.set()
            await release.wait()
            await original(connection)
            failed.append(connection)
            raise RuntimeError("synthetic-cleanup-secret")
        await original(connection)

    monkeypatch.setattr(McpConnection, "aclose", delayed_failure)
    pending = asyncio.create_task(runtime.stage(["one/unavailable"]))
    await entered.wait()
    candidate_pid = pid(tmp_path / "one")
    try:
        pending.cancel()
        await asyncio.sleep(0)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
        release.set()
        async with asyncio.timeout(3):
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert_reaped(candidate_pid)
        assert failed and all(connection in runtime._owned_connections for connection in failed)
        assert runtime.tools is initial and runtime._connections == (active,)
        os.kill(active_pid, 0)
        assert not (await initial[0].execute("old", {})).is_error
        monkeypatch.setattr(McpConnection, "aclose", original)
        await owner.aclose()
        assert not runtime._owned_connections and not owner.pending
    finally:
        release.set()
        monkeypatch.setattr(McpConnection, "aclose", original)
        await owner.aclose()


async def test_public_cancellation_after_candidate_setup_completed_discards_result(tmp_path, monkeypatch):
    runtime = McpRuntime(settings(tmp_path, {"one": "normal"}), environment={})
    original = runtime._setup
    caller = None

    async def cancel_at_delivery(selectors, servers):
        stage = await original(selectors, servers)
        assert caller is not None
        asyncio.get_running_loop().call_soon(caller.cancel)
        return stage

    monkeypatch.setattr(runtime, "_setup", cancel_at_delivery)
    caller = asyncio.create_task(runtime.stage())
    try:
        async with asyncio.timeout(3):
            with pytest.raises(asyncio.CancelledError):
                await caller
        assert_reaped(pid(tmp_path / "one"))
        assert runtime.tools == () and runtime._connections == ()
        assert not runtime._owned_connections
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("operation", ["initial", "stage"])
@pytest.mark.parametrize("unused", ["unavailable", "missing-env", "initialize-hang"])
async def test_alias_scope_ignores_unused_broken_configured_servers(tmp_path, operation, unused):
    config = settings(tmp_path, {"one": "normal", "two": "initialize-hang"})
    one, two = config.servers
    if unused == "unavailable":
        two = replace(two, command="/must/not/start-unused-server")
    elif unused == "missing-env":
        two = replace(two, env={"DECLARED": "${MISSING_FIXTURE_VALUE}"})
    config = McpConfig((one, two))
    runtime = McpRuntime(config, selectors=["one/echo"] if operation == "initial" else [], environment={})
    owner = ResourceOwner()
    try:
        async with asyncio.timeout(3):
            tools = await runtime.prepare_tools(owner)
            if operation == "stage":
                stage = await runtime.stage(["one/echo"])
                runtime.validate_stage(stage, stage.tools)
                runtime.publish(stage, stage.tools)
                await runtime.retire_previous(stage)
                tools = runtime.tools
        assert len(tools) == 1 and tools[0].name == binding_name("one", "echo")
        assert runtime.server_states == {"one": "ready", "two": "inactive"}
        assert not (tmp_path / "two/pid").exists()
        assert not (await tools[0].execute("selected", {})).is_error
    finally:
        await owner.aclose()


async def test_known_hash_selector_scopes_server_without_override(tmp_path):
    config = settings(tmp_path, {"one": "normal", "two": "initialize-hang"})
    runtime = McpRuntime(config, selectors=["one/echo"], environment={})
    owner = ResourceOwner()
    initial = await runtime.prepare_tools(owner)
    try:
        # All discovered names are cached, including initially unselected tools.
        stage = await runtime.stage([binding_name("one", "a-b"), binding_name("one", "a_b")])
        assert [connection.config.name for connection in stage.connections] == ["one"]
        assert {tool.name for tool in stage.tools} == {binding_name("one", "a-b"), binding_name("one", "a_b")}
        assert len(stage.tools) == 2
        assert not (tmp_path / "two/pid").exists()
        assert runtime.tools is initial
        await stage.aclose()
    finally:
        await owner.aclose()


async def test_unknown_fresh_hash_falls_back_to_full_enabled_catalog(tmp_path):
    runtime = McpRuntime(settings(tmp_path), selectors=[binding_name("one", "echo")], environment={})
    owner = ResourceOwner()
    try:
        tools = await runtime.prepare_tools(owner)
        assert len(tools) == 1 and tools[0].name == binding_name("one", "echo")
        assert (tmp_path / "two/events").read_text().splitlines() == ["initialize", "list", "list"]
        assert_reaped(pid(tmp_path / "two"))
        assert runtime.server_states == {"one": "ready", "two": "inactive"}
    finally:
        await owner.aclose()


@pytest.mark.parametrize("selector", ["absent/echo", "disabled/echo", "one/"])
async def test_invalid_alias_scopes_fail_before_any_launch(tmp_path, selector):
    runtime = McpRuntime(
        settings(tmp_path, {"one": "normal", "disabled": "normal"}, disabled=("disabled",)), environment={}
    )
    try:
        with pytest.raises(McpError):
            await runtime.stage([selector])
        assert not list(tmp_path.glob("*/pid"))
    finally:
        await runtime.aclose()


async def test_explicit_server_scope_remains_strict_for_known_or_unknown_hash(tmp_path):
    runtime = McpRuntime(settings(tmp_path), environment={})
    try:
        with pytest.raises(McpError, match="unavailable"):
            await runtime.stage([binding_name("two", "echo")], servers=["one"])
        assert_reaped(pid(tmp_path / "one"))
        assert not (tmp_path / "two/pid").exists()
        assert runtime.tools == () and not runtime._owned_connections
    finally:
        await runtime.aclose()
