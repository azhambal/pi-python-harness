from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from pi_agent import Cancellation, ResourceOwner, create_harness
from pi_agent.mcp import (
    McpConnection,
    McpError,
    McpLimits,
    McpRuntime,
    StdioLimits,
    binding_name,
    parse_mcp_config,
)
from pi_agent.providers import FauxProvider

FIXTURE = Path(__file__).parent / "fixtures/mcp_harness_server.py"


def settings(directory, mode="normal", timeout=2):
    return parse_mcp_config(
        {
            "mcpServers": {
                "fixture": {
                    "command": sys.executable,
                    "args": [str(FIXTURE), str(directory), mode],
                    "timeout": timeout,
                    "exposure": "direct",
                }
            }
        },
        cwd=directory,
    )


def connection(directory, mode="normal", timeout=2, **kwargs):
    return McpConnection(settings(directory, mode, timeout).servers[0], **kwargs)


async def wait_event(directory, event):
    async with asyncio.timeout(3):
        while True:
            path = directory / "events"
            if path.exists() and event in path.read_text().splitlines():
                return
            await asyncio.sleep(0.01)


def assert_reaped(directory):
    with pytest.raises(ProcessLookupError):
        os.kill(int((directory / "pid").read_text()), 0)


async def test_single_flight_paginated_discovery_and_result_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-secret")
    client = connection(tmp_path, limits=McpLimits(progress_updates=2))
    try:
        initialized = await asyncio.gather(client.connect(), client.connect())
        assert initialized[0] is initialized[1]
        tools = await client.discover()
        assert len(tools) == 10
        echo = tools[0]
        updates = []

        async def update(value):
            updates.append(value)

        result = await echo.execute("call", {"message": "hello"}, on_update=update)
        assert not result.is_error
        assert result.structured_content["structuredContent"] == {
            "echo": {"message": "hello"},
            "provider_key_present": False,
        }
        assert len(updates) == 2
        assert "synthetic-meta-secret" not in json.dumps(result.to_json())
        assert echo.provenance.server == "fixture" and echo.provenance.tool == "echo"
        assert not echo.parallel_safe
        names = {tool.name for tool in tools}
        assert binding_name("fixture", "a-b") in names and binding_name("fixture", "a_b") in names
        assert binding_name("fixture", "a-b") != binding_name("fixture", "a_b")
        assert all(len(name) <= 64 for name in names)
    finally:
        await client.aclose()
    assert_reaped(tmp_path)
    assert (tmp_path / "events").read_text().count("initialize") == 1


@pytest.mark.parametrize("mode", ["bad-schema", "schema-ref", "duplicate", "cursor-loop"])
async def test_invalid_catalog_is_terminal_and_reaped(tmp_path, mode):
    client = connection(tmp_path, mode)
    with pytest.raises(McpError, match="discovery failed"):
        await client.discover()
    assert client.state in ("closed", "failed")
    assert_reaped(tmp_path)


@pytest.mark.parametrize(
    "limits", [McpLimits(max_pages=1), McpLimits(max_tools=2), McpLimits(catalog_bytes=1024)]
)
async def test_catalog_budgets(tmp_path, limits):
    client = connection(tmp_path, limits=limits)
    with pytest.raises(McpError, match="discovery failed"):
        await client.discover()
    assert_reaped(tmp_path)


async def test_server_without_tools_returns_empty_catalog(tmp_path):
    client = connection(tmp_path, "no-tools")
    try:
        assert await client.discover() == []
        assert (tmp_path / "events").read_text().splitlines() == ["initialize"]
    finally:
        await client.aclose()


async def test_schema_error_binary_notice_and_bounded_result(tmp_path):
    client = connection(tmp_path, limits=McpLimits(result_bytes=2048))
    try:
        tools = {tool.provenance.tool: tool for tool in await client.discover()}
        output = await tools["output"].execute("1", {})
        assert output.is_error and output.details == {"invalid_output_schema": True}
        binary = await tools["binary"].execute("2", {})
        assert "Unsupported MCP content: image" in binary.content[0].text
        assert binary.structured_content["content"][0]["data"] == "c2VudGluZWw="
        large = await tools["large"].execute("3", {})
        assert large.details["truncated"]
        assert large.structured_content["structuredContent"] == {"large": "x" * 20_000}
        path = Path(large.details["full_output_path"])
        assert path.read_text() == large.structured_content["content"][0]["text"]
        assert "output truncated" in large.content[0].text
        assert len(large.content[0].text.encode()) < 2500
        path.unlink()
        error = await tools["error"].execute("4", {})
        assert error.is_error and error.content[0].text == "tool reported error"
    finally:
        await client.aclose()


@pytest.mark.parametrize("kind", ["token", "task", "timeout", "crash", "rpc-error"])
async def test_interruption_disconnects_and_never_replays(tmp_path, kind):
    client = connection(tmp_path, timeout=0.3 if kind == "timeout" else 2)
    tools = {tool.provenance.tool: tool for tool in await client.discover()}
    cancellation = Cancellation()
    name = "crash" if kind == "crash" else "rpc-error" if kind == "rpc-error" else "slow"
    call = asyncio.create_task(tools[name].execute("call", {}, cancellation))
    await wait_event(tmp_path, "call:" + name)
    if kind == "token":
        cancellation.abort()
        result = await call
        assert result.is_error and result.details == {"outcome": "unknown"}
    elif kind == "task":
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
    else:
        with pytest.raises(McpError, match="outcome unknown") as error:
            await call
        assert "synthetic-error-secret" not in str(error.value)
    assert_reaped(tmp_path)
    assert (tmp_path / "events").read_text().splitlines().count("call:" + name) == 1
    with pytest.raises(McpError, match="unavailable"):
        await tools["echo"].execute("new", {})
    await client.aclose()


async def test_close_during_initialize_joins_waiters(tmp_path):
    client = connection(tmp_path, "initialize-hang", timeout=10)
    pending = asyncio.create_task(client.connect())
    await wait_event(tmp_path, "initialize")
    await asyncio.gather(client.aclose(), client.aclose())
    with pytest.raises(McpError):
        await pending
    assert_reaped(tmp_path)


async def test_cancelled_harness_construction_settles_runtime(tmp_path, config):
    runtime = McpRuntime(settings(tmp_path, "initialize-hang", timeout=10))
    task = asyncio.create_task(
        create_harness(FauxProvider([]), config=config, prepare_tools=runtime.prepare_tools)
    )
    await wait_event(tmp_path, "initialize")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_reaped(tmp_path)


async def test_runtime_transfer_and_disabled_config(tmp_path):
    runtime = McpRuntime(
        parse_mcp_config(
            {
                "mcpServers": {
                    "disabled": {
                        "command": "/does/not/exist",
                        "enabled": False,
                    }
                }
            },
            cwd=tmp_path,
        )
    )
    owner = ResourceOwner()
    assert await runtime.prepare_tools(owner) == ()
    with pytest.raises(McpError, match="another harness"):
        await runtime.prepare_tools(ResourceOwner())
    await owner.aclose()
    assert not (tmp_path / "pid").exists()


async def test_progress_callback_failure_is_sanitized(tmp_path, caplog):
    client = connection(tmp_path)
    try:
        echo = (await client.discover())[0]

        async def failed_update(_value):
            raise RuntimeError("synthetic-callback-secret")

        result = await echo.execute("call", {}, on_update=failed_update)
        assert result.is_error and result.details["progress_delivery_failed"]
        assert result.structured_content["structuredContent"]["echo"] == {}
        assert "synthetic-callback-secret" not in caplog.text
    finally:
        await client.aclose()


@pytest.mark.parametrize("target", ["connection", "runtime", "owner", "request", "prepare"])
async def test_progress_callback_reentry_fails_without_deadlock(tmp_path, target):
    owner = ResourceOwner()
    runtime = McpRuntime(settings(tmp_path))
    echo = (await runtime.prepare_tools(owner))[0]
    client = echo._connection

    async def update(_value):
        if target == "connection":
            await client.aclose()
        elif target == "runtime":
            await runtime.aclose()
        elif target == "owner":
            await owner.aclose()
        elif target == "prepare":
            await runtime.prepare_tools(owner)
        else:
            await client.request(lambda sdk: sdk.send_ping())

    try:
        async with asyncio.timeout(3):
            result = await echo.execute("call", {}, on_update=update)
        assert result.is_error and result.details["progress_delivery_failed"]
        assert client.state == "ready"
        assert not (await echo.execute("next", {})).is_error
    finally:
        await owner.aclose()
    assert_reaped(tmp_path)


async def test_invalid_large_argument_diagnostic_is_bounded_before_admission(config):
    from pi_agent import ToolHooks
    from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage

    runtime = McpRuntime(settings(config.cwd))
    calls = []
    provider = FauxProvider(
        [
            AssistantMessage(
                content=[
                    ToolCallBlock(
                        id="call",
                        name=binding_name("fixture", "echo"),
                        arguments={"message": ["synthetic-argument-secret" * 5000]},
                    )
                ],
                stop_reason="toolUse",
            ),
            AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop"),
        ]
    )
    session = await create_harness(
        provider,
        config=config,
        prepare_tools=runtime.prepare_tools,
        hooks=ToolHooks(before_tool_call=lambda ctx, token: calls.append(ctx)),
    )
    try:
        await session.prompt("invalid")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and len(json.dumps(result.to_json()).encode()) < 1024
        assert "synthetic-argument-secret" not in json.dumps(result.to_json())
        assert calls == [] and "call:" not in (config.cwd / "events").read_text()
    finally:
        await session.aclose()


async def test_runtime_cleanup_failure_retained_and_retry_releases_owner(tmp_path, monkeypatch):
    original_spawn = asyncio.create_subprocess_exec
    processes = []
    original_waits = []

    async def hang():
        await asyncio.Event().wait()

    async def delayed_wait(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        processes.append(process)
        original_waits.append(process.wait)
        process.wait = hang
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_wait)
    owner = ResourceOwner()
    runtime = McpRuntime(settings(tmp_path), stdio_limits=StdioLimits(shutdown_timeout=0.05))
    await runtime.prepare_tools(owner)
    try:
        with pytest.raises(ExceptionGroup, match="Resource cleanup failed"):
            async with asyncio.timeout(3):
                await owner.aclose()
        assert owner.pending
        assert processes[0].returncode is not None
        assert_reaped(tmp_path)
    finally:
        processes[0].wait = original_waits[0]
        await owner.aclose()
    assert not owner.pending
    await owner.aclose()


async def test_cancelled_connection_close_joins_all_owned_work(tmp_path):
    client = connection(tmp_path)
    await client.discover()
    closing = asyncio.create_task(client.aclose())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert_reaped(tmp_path)
    assert client._owner.done() and client._close_task.done()
    await client.aclose()


async def test_direct_connection_progress_cannot_close_or_log_raw_exception(tmp_path, caplog):
    client = connection(tmp_path)
    await client.discover()

    async def update(progress, total, message):
        try:
            await client.aclose()
        except RuntimeError:
            raise RuntimeError("synthetic-callback-secret") from None

    try:
        async with asyncio.timeout(3):
            result = await client.call("echo", {}, None, update)
        assert not result.is_error and client.state == "ready"
        assert "synthetic-callback-secret" not in caplog.text
        assert "MCP progress callback failed" in caplog.text
    finally:
        await client.aclose()
    assert_reaped(tmp_path)


async def test_progress_result_honors_small_json_budget(tmp_path):
    client = connection(
        tmp_path, mode="progress-large", limits=McpLimits(result_bytes=1024, progress_updates=1)
    )
    updates = []

    async def update(result):
        updates.append(result)

    try:
        echo = (await client.discover())[0]
        await echo.execute("call", {}, on_update=update)
        assert len(updates) == 1
        encoded = json.dumps(updates[0].to_json(), ensure_ascii=False, separators=(",", ":"))
        assert len(encoded.encode()) <= 1024
        assert "message exceeds output budget" in updates[0].content[0].text
    finally:
        await client.aclose()


@pytest.mark.parametrize("kwargs", [{"max_pages": True}, {"result_bytes": 4}, {"progress_updates": 0}])
def test_invalid_host_limits(kwargs):
    with pytest.raises(ValueError):
        McpLimits(**kwargs)
