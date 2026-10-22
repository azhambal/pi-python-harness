"""Bounded stdio budgets, protocol sanitization, and owned-process lifetime."""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import json
import logging
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from pi_agent.mcp._stdio import DEFAULT_STDIO_LIMITS, StdioLimits

FIXTURE = Path(__file__).parent / "fixtures/mcp_bounded_stdio_server.py"
SECRET = "SECRET_STDOUT_SENTINEL"
RESPONSE = b'{"jsonrpc":"2.0","id":1,"result":{}}'
PING = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
NOTIFICATION = b'{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}'
MALFORMED_NOTIFICATION = (
    '{"jsonrpc":"2.0","method":"notifications/progress",'
    '"params":{"progressToken":null,"progress":"' + SECRET + '"}}\n'
).encode()


@pytest.fixture
def sdk():
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession, types
    from mcp.shared.message import SessionMessage

    from pi_agent.mcp._stdio import bounded_stdio

    return anyio, ClientSession, types, SessionMessage, bounded_stdio


def launch(sdk, directory, mode="raw", payload=b"", limits=DEFAULT_STDIO_LIMITS):
    return sdk[-1](
        sys.executable,
        (str(FIXTURE), str(directory), mode, payload.hex() if isinstance(payload, bytes) else payload),
        directory,
        {},
        limits,
    )


async def ready(directory, filename="ready"):
    async with asyncio.timeout(3):
        while not (directory / filename).exists():
            await asyncio.sleep(0.005)


async def failed(transport, code):
    await asyncio.wait_for(transport.failed.wait(), 3)
    assert transport.failure == code


def assert_reaped(directory):
    pid = int((directory / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def assert_no_workers():
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("mcp-stdio-")]


def test_base_import_does_not_import_optional_dependencies(monkeypatch):
    path = Path(__file__).parents[1] / "src/pi_agent/mcp/_stdio.py"
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("mcp", "anyio"):
            raise AssertionError("Optional dependency imported eagerly")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("bounded_stdio_base_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.StdioLimits().frame_bytes == 1024 * 1024


@pytest.mark.parametrize(
    "name,cap",
    [
        ("frame_bytes", 1024 * 1024),
        ("outgoing_bytes", 1024 * 1024),
        ("stderr_bytes", 64 * 1024),
        ("total_stdout_bytes", 16 * 1024 * 1024),
        ("queued_frames", 16),
        ("shutdown_timeout", 0.5),
    ],
)
def test_limits_are_strict_hard_caps(name, cap):
    for value in (0, -1, True, "1", None, float("inf"), float("nan"), cap + 1, 10**1000):
        with pytest.raises(ValueError, match="Invalid MCP stdio limit"):
            StdioLimits(**{name: value})
    if name != "shutdown_timeout":
        with pytest.raises(ValueError):
            StdioLimits(**{name: 1.0})
    assert getattr(StdioLimits(**{name: cap}), name) == cap
    assert getattr(StdioLimits(**{name: cap / 2 if name == "shutdown_timeout" else 1}), name) > 0


def test_limits_are_immutable():
    with pytest.raises(FrozenInstanceError):
        DEFAULT_STDIO_LIMITS.frame_bytes = 2


async def test_sdk_session_roundtrip_and_explicit_environment(sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_SYNTHETIC_PARENT", "synthetic-only")
    async with launch(sdk, tmp_path, "env") as transport:
        await transport.write_stream.send(sdk[3](TypeAdapter(sdk[2].JSONRPCMessage).validate_json(PING)))
        response = await asyncio.wait_for(transport.read_stream.receive(), 3)
        assert isinstance(response, sdk[3])
        assert response.message.result == {"inherited": False}
        assert not transport.failed.is_set()
    assert_reaped(tmp_path)
    assert_no_workers()


async def test_outgoing_typed_message_and_wire_bytes(sdk, tmp_path):
    _, _, types, SessionMessage, _ = sdk
    request = SessionMessage(TypeAdapter(types.JSONRPCMessage).validate_json(PING))
    async with launch(sdk, tmp_path, "echo") as transport:
        await transport.write_stream.send(request)
        response = await asyncio.wait_for(transport.read_stream.receive(), 3)
        assert isinstance(response, SessionMessage)
        assert response.message.id == 1
        wire = (tmp_path / "input").read_bytes()
        assert json.loads(wire) == json.loads(PING)
        assert wire.endswith(b"\n") and len(wire) == len(PING) + 1
        assert not transport.failed.is_set()
    assert_reaped(tmp_path)


@pytest.mark.parametrize(
    "result",
    [
        {"content": [], "isError": "false"},
        {"content": [{"type": "text", "text": 42}]},
        {"content": "SECRET_STDOUT_SENTINEL"},
    ],
)
async def test_response_type_coercions_fail_before_sdk(sdk, tmp_path, result, caplog):
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode() + b"\n"
    request = sdk[3](
        TypeAdapter(sdk[2].JSONRPCMessage).validate_python(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {}}}
        )
    )
    async with launch(sdk, tmp_path, "response", payload) as transport:
        await transport.write_stream.send(request)
        await failed(transport, "invalid_stdout")
    assert SECRET not in caplog.text
    assert_reaped(tmp_path)


async def test_existing_fixture_sdk_initialize_tools_and_progress(sdk, tmp_path):
    fixture = Path(__file__).parent / "fixtures/mcp_stdio_server.py"
    async with sdk[-1](sys.executable, (str(fixture), str(tmp_path), "normal"), tmp_path, {}) as transport:
        async with sdk[1](transport.read_stream, transport.write_stream) as client:
            initialized = await asyncio.wait_for(client.initialize(), 3)
            assert initialized.protocol_version == "2025-11-25"
            tools = await asyncio.wait_for(client.list_tools(), 3)
            assert [tool.name for tool in tools.tools] == ["echo", "error"]
            updates = []

            async def progress(value, total, message):
                updates.append((value, total, message))

            result = await asyncio.wait_for(
                client.call_tool("echo", {"message": "bounded"}, progress_callback=progress), 3
            )
            assert result.structured_content["echo"] == {"message": "bounded"}
            assert updates == [(1, 2, "fixture progress")]
            assert not transport.failed.is_set()
    assert_reaped(tmp_path)
    assert_no_workers()


@pytest.mark.parametrize("payload", [RESPONSE, PING, NOTIFICATION])
async def test_fragmented_frames_are_typed(sdk, tmp_path, payload):
    split = payload[:10] + b"|" + payload[10:24] + b"|" + payload[24:] + b"\n"
    mode = "split-response" if payload == RESPONSE else "split"
    async with launch(sdk, tmp_path, mode, split) as transport:
        if payload == RESPONSE:
            await transport.write_stream.send(sdk[3](TypeAdapter(sdk[2].JSONRPCMessage).validate_json(PING)))
        result = await asyncio.wait_for(transport.read_stream.receive(), 3)
        assert isinstance(result, sdk[3])
        assert not transport.failed.is_set()
    assert_reaped(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\n",
        b"\n",
        b"[]\n",
        b"not-json-" + SECRET.encode() + b"\n",
        b'{"jsonrpc":"2.0","id":1,"result":{},"id":2}\n',
        b'{"jsonrpc":"2.0","id":true,"result":{}}\n',
        b'{"jsonrpc":"2.0","id":-1,"result":{}}\n',
        b'{"jsonrpc":"2.0","id":"-1","result":{}}\n',
        b'{"jsonrpc":"2.0","id":"' + SECRET.encode() + b'","result":{}}\n',
        '{"jsonrpc":"2.0","id":"²","result":{}}\n'.encode(),
        b'{"jsonrpc":"2.0","id":1.0,"result":{}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"x":NaN}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"x":Infinity}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"x":1e9999}}\n',
        b'{"jsonrpc":"2.0","id":1,"error":{"code":true,"message":"secret"}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"a":1,"a":2}}\n',
        b"[" * 65 + b"0" + b"]" * 65 + b"\n",
        b'{"jsonrpc":"2.0","id":1,"method":"sampling/createMessage","params":{"messages":"'
        + SECRET.encode()
        + b'"}}\n',
        MALFORMED_NOTIFICATION,
        b'{"jsonrpc":"2.0","method":"notifications/unknown"}\n',
    ],
)
async def test_invalid_stdout_is_generic_and_closes_receive(sdk, tmp_path, payload, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    async with launch(sdk, tmp_path, "raw", payload) as transport:
        await failed(transport, "invalid_stdout")
        with pytest.raises(sdk[0].EndOfStream):
            await asyncio.wait_for(transport.read_stream.receive(), 0.2)
        assert SECRET not in repr(transport.failure)
    assert_reaped(tmp_path)
    assert SECRET not in caplog.text + str(capsys.readouterr())
    assert_no_workers()


async def test_malformed_notification_cannot_reach_sdk_logging_or_stall_pending_request(
    sdk, tmp_path, caplog, capsys
):
    caplog.set_level(logging.DEBUG)
    async with launch(sdk, tmp_path, "pending-invalid", MALFORMED_NOTIFICATION) as transport:
        async with sdk[1](transport.read_stream, transport.write_stream) as client:
            await asyncio.wait_for(client.initialize(), 3)
            from mcp.shared.exceptions import MCPError

            with pytest.raises(MCPError, match="Connection closed"):
                await asyncio.wait_for(client.send_ping(), 3)
            await failed(transport, "invalid_stdout")
    assert_reaped(tmp_path)
    assert SECRET not in caplog.text + str(capsys.readouterr())
    assert_no_workers()


async def test_exact_frame_limit_and_unterminated_limit(sdk, tmp_path):
    limits = StdioLimits(frame_bytes=len(RESPONSE))
    async with launch(sdk, tmp_path, "response", RESPONSE + b"\n", limits) as transport:
        await transport.write_stream.send(sdk[3](TypeAdapter(sdk[2].JSONRPCMessage).validate_json(PING)))
        assert isinstance(await asyncio.wait_for(transport.read_stream.receive(), 3), sdk[3])
        assert not transport.failed.is_set()
    assert_reaped(tmp_path)
    async with launch(sdk, tmp_path, "raw", b"x" * (len(RESPONSE) + 1), limits) as transport:
        await failed(transport, "frame_limit")
    assert_reaped(tmp_path)


async def test_total_stdout_budget_includes_delimiters(sdk, tmp_path):
    limits = StdioLimits(total_stdout_bytes=len(NOTIFICATION) + 1)
    async with launch(sdk, tmp_path, "split", NOTIFICATION + b"\n|x", limits) as transport:
        assert isinstance(await asyncio.wait_for(transport.read_stream.receive(), 3), sdk[3])
        await failed(transport, "stdout_limit")
    assert_reaped(tmp_path)


async def test_queue_overflow_disconnects_instead_of_stalling_pipe_reader(sdk, tmp_path):
    async with launch(
        sdk, tmp_path, "raw", (NOTIFICATION + b"\n") * 3, StdioLimits(queued_frames=2)
    ) as transport:
        await failed(transport, "queue_limit")
        for _ in range(2):
            assert isinstance(await transport.read_stream.receive(), sdk[3])
        with pytest.raises(sdk[0].EndOfStream):
            await transport.read_stream.receive()
    assert_reaped(tmp_path)


@pytest.mark.parametrize("mode,payload", [("partial-eof", RESPONSE), ("partial-eof", b"")])
async def test_eof_is_failure_and_partial_frames_are_not_accepted(sdk, tmp_path, mode, payload):
    async with launch(sdk, tmp_path, mode, payload) as transport:
        await asyncio.wait_for(transport.failed.wait(), 3)
        assert transport.failure in ("invalid_stdout", "connection_closed")
        with pytest.raises(sdk[0].EndOfStream):
            await asyncio.wait_for(transport.read_stream.receive(), 0.2)
    assert_reaped(tmp_path)


async def test_stderr_budget_discards_and_never_leaks(sdk, tmp_path, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    async with launch(sdk, tmp_path, "stderr", "10000", StdioLimits(stderr_bytes=32)) as transport:
        await failed(transport, "stderr_limit")
        with pytest.raises(sdk[0].EndOfStream):
            await transport.read_stream.receive()
    assert_reaped(tmp_path)
    assert "SECRET_STDERR_SENTINEL" not in caplog.text + str(capsys.readouterr()) + repr(transport.failure)


@pytest.mark.parametrize(
    "value", ["x" * 1025, "\x01" * 200, "😀" * 300, float("inf"), [[[]]], [0] * 1025, "\ud800"]
)
async def test_invalid_outgoing_is_bounded_and_generic(sdk, tmp_path, value, caplog, capsys):
    _, _, types, SessionMessage, _ = sdk
    if value == [[[]]]:
        value = {}
        for _ in range(65):
            value = {"nested": value}
    request = SessionMessage(
        types.JSONRPCRequest(jsonrpc="2.0", id=1, method="ping", params={"value": value})
    )
    async with launch(sdk, tmp_path, "no-read", limits=StdioLimits(outgoing_bytes=1024)) as transport:
        await ready(tmp_path)
        await transport.write_stream.send(request)
        await failed(transport, "invalid_outgoing")
        with pytest.raises(sdk[0].EndOfStream):
            await transport.read_stream.receive()
    assert_reaped(tmp_path)
    assert SECRET not in caplog.text + str(capsys.readouterr())


async def test_exact_outgoing_budget_is_accepted(sdk, tmp_path):
    request = sdk[3](TypeAdapter(sdk[2].JSONRPCMessage).validate_json(PING))
    async with launch(sdk, tmp_path, "echo", limits=StdioLimits(outgoing_bytes=len(PING))) as transport:
        await transport.write_stream.send(request)
        assert isinstance(await asyncio.wait_for(transport.read_stream.receive(), 3), sdk[3])
        assert not transport.failed.is_set()
    assert_reaped(tmp_path)


@pytest.mark.parametrize("response_id", [1, "1", "0001"])
async def test_numeric_response_ids_are_correlated(sdk, tmp_path, response_id):
    payload = json.dumps({"jsonrpc": "2.0", "id": response_id, "result": {}}).encode() + b"\n"
    async with launch(sdk, tmp_path, "response", payload) as transport:
        await transport.write_stream.send(sdk[3](TypeAdapter(sdk[2].JSONRPCMessage).validate_json(PING)))
        response = await asyncio.wait_for(transport.read_stream.receive(), 3)
        assert response.message.id == response_id
        assert not transport.failed.is_set()
    assert_reaped(tmp_path)


@pytest.mark.parametrize(
    "payload", [RESPONSE + b"\n" + RESPONSE + b"\n", RESPONSE.replace(b'"id":1', b'"id":99') + b"\n"]
)
async def test_duplicate_and_unknown_responses_are_terminal(sdk, tmp_path, payload, caplog, capsys):
    async with launch(sdk, tmp_path, "response", payload) as transport:
        await transport.write_stream.send(sdk[3](TypeAdapter(sdk[2].JSONRPCMessage).validate_json(PING)))
        await failed(transport, "invalid_stdout")
    assert_reaped(tmp_path)
    assert SECRET not in caplog.text + str(capsys.readouterr())


@pytest.mark.parametrize("response_id", [SECRET, -1, "²", 99])
async def test_invalid_correlation_ids_never_reach_sdk_logger(sdk, tmp_path, response_id, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    payload = json.dumps({"jsonrpc": "2.0", "id": response_id, "result": {"secret": SECRET}}).encode() + b"\n"
    async with launch(sdk, tmp_path, "pending-invalid", payload) as transport:
        async with sdk[1](transport.read_stream, transport.write_stream) as client:
            await asyncio.wait_for(client.initialize(), 3)
            from mcp.shared.exceptions import MCPError

            with pytest.raises(MCPError, match="Connection closed"):
                await asyncio.wait_for(client.send_ping(), 3)
            await failed(transport, "invalid_stdout")
    assert_reaped(tmp_path)
    assert SECRET not in caplog.text + str(capsys.readouterr())
    assert_no_workers()


async def test_outstanding_request_ids_are_bounded(sdk, tmp_path):
    async with launch(
        sdk, tmp_path, "no-read", limits=StdioLimits(queued_frames=2, shutdown_timeout=0.05)
    ) as transport:
        await ready(tmp_path)
        for request_id in range(3):
            request = sdk[3](sdk[2].JSONRPCRequest(jsonrpc="2.0", id=request_id, method="ping"))
            await transport.write_stream.send(request)
        await failed(transport, "invalid_outgoing")
    assert_reaped(tmp_path)


async def test_explicit_aclose_is_joined_and_idempotent_after_context(sdk, tmp_path):
    async with launch(sdk, tmp_path, "stubborn", limits=StdioLimits(shutdown_timeout=0.05)) as transport:
        await ready(tmp_path)
        await asyncio.wait_for(asyncio.gather(transport.aclose(), transport.aclose()), 2)
        assert_reaped(tmp_path)
        assert not transport.failed.is_set()
        with pytest.raises(sdk[0].EndOfStream):
            await transport.read_stream.receive()
    await transport.aclose()
    assert_no_workers()


@pytest.mark.parametrize("resource", ["process", "stdin-pipe"])
async def test_cleanup_deadline_is_generic_and_retry_retains_ownership(
    sdk, tmp_path, monkeypatch, resource, caplog, capsys
):
    original_spawn = asyncio.create_subprocess_exec
    processes = []
    restore = []

    async def never_closes():
        await asyncio.Event().wait()

    async def delayed_close(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        processes.append(process)
        if resource == "process":
            restore.append(process.wait)
            process.wait = never_closes
        else:
            restore.append(process.stdin.wait_closed)
            process.stdin.wait_closed = never_closes
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_close)
    async with asyncio.timeout(2):
        with pytest.raises(RuntimeError, match="^MCP stdio cleanup failed$") as caught:
            async with launch(
                sdk, tmp_path, "stubborn", limits=StdioLimits(shutdown_timeout=0.05)
            ) as transport:
                await ready(tmp_path)
    assert caught.value.__suppress_context__
    assert transport.failed.is_set() and transport.failure == "shutdown_failed"
    assert processes[0].returncode is not None
    assert_reaped(tmp_path)
    if resource == "process":
        processes[0].wait = restore[0]
    else:
        processes[0].stdin.wait_closed = restore[0]
    await asyncio.wait_for(transport.aclose(), 2)
    await transport.aclose()
    assert_reaped(tmp_path)
    assert_no_workers()
    assert SECRET not in caplog.text + str(capsys.readouterr()) + str(caught.value)


async def test_failed_termination_keeps_live_process_owned_for_retry(sdk, tmp_path, monkeypatch):
    original_spawn = asyncio.create_subprocess_exec
    original_killpg = os.killpg
    processes = []
    popen_kills = []
    transport = None

    async def temporarily_unkillable(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        processes.append(process)
        popen = process._transport._proc
        popen_kills.append((popen, popen.kill))
        monkeypatch.setattr(popen, "kill", lambda: None)
        return process

    def temporarily_ignore_signals(pid, sig):
        if processes and pid == processes[0].pid and sig != 0:
            return
        original_killpg(pid, sig)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", temporarily_unkillable)
    monkeypatch.setattr(os, "killpg", temporarily_ignore_signals)
    try:
        async with asyncio.timeout(2):
            with pytest.raises(RuntimeError, match="^MCP stdio cleanup failed$"):
                async with launch(
                    sdk, tmp_path, "stubborn", limits=StdioLimits(shutdown_timeout=0.05)
                ) as transport:
                    await ready(tmp_path)
        assert transport.failed.is_set() and transport.failure == "shutdown_failed"
        assert processes[0].returncode is None
        os.kill(processes[0].pid, 0)  # The failed attempt demonstrably did not reap it.
    finally:
        monkeypatch.setattr(os, "killpg", original_killpg)
        for popen, kill in popen_kills:
            monkeypatch.setattr(popen, "kill", kill)
        if transport is not None:
            await asyncio.wait_for(transport.aclose(), 2)
    assert_reaped(tmp_path)
    assert_no_workers()


async def test_cancelled_context_kills_term_ignoring_process_and_joins(sdk, tmp_path):
    async def owner():
        async with launch(sdk, tmp_path, "stubborn", limits=StdioLimits(shutdown_timeout=0.05)):
            await ready(tmp_path)
            await asyncio.Event().wait()

    task = asyncio.create_task(owner())
    await ready(tmp_path)
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert_reaped(tmp_path)
    assert_no_workers()


async def test_anyio_cancel_scope_still_joins_cleanup_in_owner(sdk, tmp_path):
    with sdk[0].CancelScope() as scope:
        async with launch(sdk, tmp_path, "stubborn", limits=StdioLimits(shutdown_timeout=0.05)):
            await ready(tmp_path)
            scope.cancel()
            await asyncio.sleep(0)
    assert_reaped(tmp_path)
    assert_no_workers()


async def test_cancel_during_spawn_joins_creation_and_reaps(sdk, tmp_path, monkeypatch):
    original = asyncio.create_subprocess_exec
    created = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*args, **kwargs):
        process = await original(*args, **kwargs)
        created.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed)

    async def owner():
        async with launch(sdk, tmp_path, "stubborn", limits=StdioLimits(shutdown_timeout=0.05)):
            pytest.fail("Cancelled spawn must never yield")

    task = asyncio.create_task(owner())
    await created.wait()
    await ready(tmp_path)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert_reaped(tmp_path)
    assert_no_workers()


@pytest.mark.parametrize("mode", ["pipe-holder", "pipe-holder-exit"])
async def test_process_group_kills_descendants_holding_pipes(sdk, tmp_path, mode):
    async with launch(sdk, tmp_path, mode, limits=StdioLimits(shutdown_timeout=0.05)) as transport:
        await ready(tmp_path, "child-pid")
        if mode == "pipe-holder-exit":
            await asyncio.wait_for(transport.failed.wait(), 3)
    assert_reaped(tmp_path)
    child_pid = int((tmp_path / "child-pid").read_text())
    async with asyncio.timeout(2):
        while True:
            status = Path(f"/proc/{child_pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            await asyncio.sleep(0.01)
    assert_no_workers()


async def test_close_when_stdin_drain_is_blocked(sdk, tmp_path):
    request = sdk[3](
        sdk[2].JSONRPCRequest(jsonrpc="2.0", id=1, method="ping", params={"value": "x" * 500_000})
    )

    async def owner():
        async with launch(sdk, tmp_path, "no-read", limits=StdioLimits(shutdown_timeout=0.05)) as transport:
            await ready(tmp_path)
            await transport.write_stream.send(request)
            await asyncio.sleep(0.05)

    await asyncio.wait_for(owner(), 2)
    assert_reaped(tmp_path)
    assert_no_workers()


async def test_spawn_error_is_sanitized(sdk, tmp_path, caplog, capsys):
    with pytest.raises(RuntimeError, match="^MCP stdio spawn failed$") as caught:
        async with sdk[-1](str(tmp_path / SECRET), (), tmp_path, {}):
            pytest.fail("Missing executable")
    assert SECRET not in str(caught.value) + caplog.text + str(capsys.readouterr())
    assert caught.value.__suppress_context__
    assert_no_workers()


@pytest.mark.parametrize(
    "overrides",
    [
        {"command": ""},
        {"command": "a\0b"},
        {"args": ["arg"]},
        {"args": (1,)},
        {"env": {"A=B": "x"}},
        {"env": {"A": 1}},
        {"env": {"A": "x\0y"}},
        {"cwd": "."},
        {"limits": None},
    ],
)
async def test_bad_parameters_are_rejected_before_spawn(sdk, tmp_path, overrides, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid parameters must not spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    kwargs = {"command": sys.executable, "args": (), "cwd": tmp_path, "env": {}}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match="Invalid MCP stdio parameters"):
        async with sdk[-1](**kwargs):
            pytest.fail("Invalid parameters must not yield")
