"""CLI MCP boundaries, exercised offline with explicit temporary configurations."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pi_agent import cli, create_agent_session
from pi_agent.chat import run_chat
from pi_agent.mcp import McpConfigError, binding_name
from pi_agent.mcp.manifest import McpBinding
from pi_agent.providers import FauxProvider
from pi_agent.session_store import SessionStore
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage

FIXTURE = Path(__file__).parent / "fixtures/mcp_harness_server.py"


def done():
    return AssistantMessage(content=[TextBlock(text="offline done")], stop_reason="stop")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Never read the launch directory's dotenv, host credentials, or user settings.
    monkeypatch.setattr(
        cli,
        "read_environment",
        lambda path: {"OPENAI_API_KEY": "synthetic-cli-secret", "OPENAI_MODEL": "offline"},
    )
    monkeypatch.setattr(cli, "load_settings", lambda *args, **kwargs: {})


@pytest.fixture
def providers(monkeypatch):
    created = []

    def factory(settings):
        provider = FauxProvider([done(), done()])
        created.append(provider)
        return provider

    monkeypatch.setattr(cli, "_runtime_provider", factory)
    return created


def arguments(tmp_path, *flags, command="run"):
    return cli._build_parser().parse_args(
        [command, "--cwd", str(tmp_path), "--session-dir", str(tmp_path / "sessions"), "--no-skills", *flags]
    )


def config_file(tmp_path, *, mode="normal", unused=False, script=FIXTURE):
    directory = tmp_path / "server"
    directory.mkdir(exist_ok=True)
    servers = {
        "fixture": {
            "command": sys.executable,
            "args": [str(script), str(directory), mode],
            "exposure": "direct",
            "timeout": 1,
        }
    }
    if unused:
        other = tmp_path / "unused"
        other.mkdir(exist_ok=True)
        servers["unused"] = {
            "command": sys.executable,
            "args": [str(FIXTURE), str(other), "normal"],
            "exposure": "direct",
            "timeout": 1,
        }
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


def assert_reaped(directory):
    with pytest.raises(ProcessLookupError):
        os.kill(int((directory / "pid").read_text()), 0)


@pytest.mark.parametrize("command", ["run", "chat"])
def test_mcp_flags_and_distinct_project_trust(tmp_path, command):
    args = arguments(tmp_path, command=command)
    assert args.mcp_config is None and args.mcp_project_config is None
    assert args.mcp_tools is None and not args.mcp_reload and not args.trust_project_mcp
    args = arguments(tmp_path, "--mcp-tools", "", command=command)
    assert args.mcp_tools == ()
    name = binding_name("fixture", "echo")
    args = arguments(tmp_path, "--mcp-tools", f" fixture/echo, {name} ", command=command)
    assert args.mcp_tools == ("fixture/echo", name)
    args = arguments(tmp_path, "--trust-project-resources", command=command)
    assert args.trust_project_resources and not args.trust_project_mcp
    args = arguments(tmp_path, "--trust-project-mcp", "--no-trust-project-mcp", command=command)
    assert not args.trust_project_mcp
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--mcp-config", "one", "--mcp-project-config", "two", command=command)
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--mcp-tools", "fixture/echo,", command=command)


async def test_no_mcp_async_wrapper_keeps_sync_factory_compatibility(tmp_path, monkeypatch):
    expected = object()
    calls = []
    monkeypatch.setattr(cli, "_open_session", lambda args: calls.append(args) or expected)
    args = arguments(tmp_path)
    assert await cli._open_session_async(args) is expected
    assert calls == [args]


@pytest.mark.parametrize("project", [False, True])
async def test_async_factory_passes_shared_options_once(tmp_path, providers, monkeypatch, project):
    path = config_file(tmp_path)
    calls = []
    sources = []
    load_config = cli.load_mcp_config

    def record_source(path, **kwargs):
        sources.append(kwargs)
        return load_config(path, **kwargs)

    monkeypatch.setattr(cli, "load_mcp_config", record_source)

    async def harness(provider, **kwargs):
        calls.append((provider, kwargs.copy()))
        for key in ("mcp_config", "mcp_tools", "refresh_mcp"):
            kwargs.pop(key)
        return create_agent_session(provider, **kwargs)

    monkeypatch.setattr(cli, "create_harness", harness)
    flags = (
        ["--mcp-project-config", str(path), "--trust-project-mcp"] if project else ["--mcp-config", str(path)]
    )
    session, settings, switched, resumed = await cli._open_session_async(
        arguments(
            tmp_path,
            *flags,
            "--mcp-tools",
            "fixture/echo",
            "--tools",
            "read",
            "--read-only",
            "--thinking",
            "low",
            "--tool-execution",
            "parallel",
            "--max-parallel-tools",
            "2",
            "--no-session",
        )
    )
    try:
        assert len(providers) == len(calls) == 1 and calls[0][0] is providers[0]
        assert sources == [
            {"cwd": tmp_path, "source": "project" if project else "explicit", "project_trusted": project}
        ]
        options = calls[0][1]
        assert options["mcp_tools"] == ("fixture/echo",) and not options["refresh_mcp"]
        assert options["mcp_config"].servers[0].cwd == tmp_path
        assert options["tool_names"] == ("read",) and options["permission_policy"].read_only
        assert options["config"].thinking_level == "low" and options["config"].max_parallel_tools == 2
        assert settings.model.id == "offline" and not switched and not resumed
    finally:
        await session.aclose()


@pytest.mark.parametrize("trust", [[], ["--trust-project-resources"], ["--no-trust-project-mcp"]])
async def test_untrusted_project_is_rejected_before_any_config_or_environment_read(
    tmp_path, monkeypatch, trust
):
    def forbidden(*args, **kwargs):
        pytest.fail("untrusted project startup reached a read or factory")

    monkeypatch.setattr(cli, "load_mcp_config", forbidden)
    monkeypatch.setattr(cli, "read_environment", forbidden)
    monkeypatch.setattr(cli, "_runtime_provider", forbidden)
    with pytest.raises(McpConfigError, match="--trust-project-mcp"):
        await cli._open_session_async(arguments(tmp_path, "--mcp-project-config", "missing", *trust))


async def test_invalid_config_does_not_recover_journal_or_create_provider(tmp_path, providers, monkeypatch):
    session, *_ = cli._open_session(arguments(tmp_path))
    await session.prompt("persist")
    path = session.session_file
    await session.aclose()
    with path.open("ab") as stream:
        stream.write(b'{"torn": "\xe2\x82')
    before = path.read_bytes()
    invalid = tmp_path / "sensitive-config-name.json"
    invalid.write_text('{"mcpServers":{"fixture":{"command":"synthetic-command-secret"}}}')
    monkeypatch.setattr(cli, "read_environment", lambda path: pytest.fail("read environment before parse"))
    with pytest.raises(McpConfigError) as error:
        await cli._open_session_async(
            arguments(tmp_path, "--resume", str(path), "--mcp-config", str(invalid))
        )
    assert "synthetic-command-secret" not in str(error.value)
    assert "sensitive-config-name" not in str(error.value)
    assert path.read_bytes() == before and len(providers) == 1
    store = SessionStore.open(path)
    store.close()


async def test_resume_options_use_stored_cwd_and_refresh_flag(tmp_path, providers, monkeypatch):
    session, *_ = cli._open_session(arguments(tmp_path))
    await session.prompt("persist")
    path = session.session_file
    await session.aclose()
    mcp = config_file(tmp_path)
    calls = []

    async def harness(provider, **kwargs):
        calls.append(kwargs.copy())
        for key in ("mcp_config", "mcp_tools", "refresh_mcp"):
            kwargs.pop(key)
        return create_agent_session(provider, **kwargs)

    monkeypatch.setattr(cli, "create_harness", harness)
    args = cli._build_parser().parse_args(
        ["run", "--resume", str(path), "--mcp-config", str(mcp), "--mcp-reload"]
    )
    resumed, _, switch, was_resumed = await cli._open_session_async(args)
    try:
        assert was_resumed and not switch
        assert calls[0]["mcp_config"].servers[0].cwd == tmp_path
        assert calls[0]["cwd"] is None and calls[0]["tool_names"] is None
        assert calls[0]["mcp_tools"] is None and calls[0]["refresh_mcp"]
        assert len(providers) == 2
    finally:
        await resumed.aclose()


@pytest.mark.parametrize("json_mode", [False, True])
async def test_chat_mcp_status_and_async_commands(tmp_path, config, monkeypatch, capsys, json_mode):
    session = create_agent_session(FauxProvider([]), config=config)
    name = binding_name("fixture", "echo")
    binding = McpBinding(name, "fixture", "echo", "0" * 64, "0" * 64)
    monkeypatch.setattr(type(session), "mcp_bindings", property(lambda self: (binding,)))
    monkeypatch.setattr(session.agent.tools[0], "description", "synthetic-description-secret")
    reconnect = AsyncMock()
    reload_mcp = AsyncMock()
    monkeypatch.setattr(
        type(session), "mcp_server_states", property(lambda self: {"fixture": "ready"}), raising=False
    )
    monkeypatch.setattr(session, "reconnect_mcp", reconnect, raising=False)
    monkeypatch.setattr(session, "reload_mcp", reload_mcp, raising=False)
    monkeypatch.setattr(
        cli, "_open_session", lambda args: (session, SimpleNamespace(model=session.model), False, False)
    )
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "/mcp\n/mcp reconnect fixture\n/wait\n/mcp reload\n/wait\n"
            "/mcp reload fixture/echo,fixture/error\n/wait\n/mcp reload all\n/wait\n/mcp reconnect\n"
            "/mcp reload fixture/echo,\n/mcp invalid\n/quit\n"
        ),
    )
    assert await run_chat(arguments(tmp_path, *(["--json"] if json_mode else []), command="chat")) == 0
    reconnect.assert_awaited_once_with("fixture")
    assert [call.args for call in reload_mcp.await_args_list] == [
        (None,),
        (("fixture/echo", "fixture/error"),),
        (("*",),),
    ]
    captured = capsys.readouterr()
    ui = captured.err if json_mode else captured.out
    assert '"fixture": "ready"' in ui and name in ui and '"fixture/echo"' in ui
    assert "synthetic-description-secret" not in captured.out + captured.err
    assert "requires one server name" in captured.err and "empty items" in captured.err
    if json_mode:
        assert not captured.out
    assert session._disposed


async def test_chat_mcp_operation_keeps_input_live_and_rejects_overlapping_changes(
    tmp_path, config, monkeypatch, capsys
):
    session = create_agent_session(FauxProvider([]), config=config)
    entered = asyncio.Event()
    released = asyncio.Event()

    async def reconnect(name):
        entered.set()
        await released.wait()

    reload_mcp = AsyncMock()
    monkeypatch.setattr(session, "reconnect_mcp", reconnect, raising=False)
    monkeypatch.setattr(session, "reload_mcp", reload_mcp, raising=False)
    monkeypatch.setattr(cli, "_open_session", lambda args: (session, SimpleNamespace(), False, False))
    reader, writer = os.pipe()
    with os.fdopen(reader, "r", encoding="utf-8") as source:
        monkeypatch.setattr("sys.stdin", source)
        task = asyncio.create_task(run_chat(arguments(tmp_path, command="chat")))
        try:
            os.write(writer, b"/mcp reconnect fixture\n")
            await asyncio.wait_for(entered.wait(), 1)
            os.write(writer, b"/mcp reload\n/status\n")
            for _ in range(10):
                await asyncio.sleep(0)
            released.set()
            os.write(writer, b"/wait\n/quit\n")
            assert await asyncio.wait_for(task, 1) == 0
        finally:
            released.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            os.close(writer)
    reload_mcp.assert_not_awaited()
    assert "Wait for the active operation" in capsys.readouterr().err


@pytest.mark.parametrize("selectors", [None, "fixture/echo", binding_name("fixture", "echo")])
async def test_fresh_cli_mcp_combines_builtin_selection_and_cleans_up(tmp_path, providers, selectors):
    pytest.importorskip("mcp")
    path = config_file(tmp_path)
    flags = [] if selectors is None else ["--mcp-tools", selectors]
    session, *_ = await cli._open_session_async(
        arguments(tmp_path, "--mcp-config", str(path), "--tools", "read", *flags)
    )
    try:
        names = {tool.name for tool in session.agent.tools}
        expected = 11 if selectors is None else 2
        assert len(names) == expected and "read" in names and binding_name("fixture", "echo") in names
        assert "fixture" in session.mcp_server_states
        assert all(isinstance(binding, McpBinding) for binding in session.mcp_bindings)
        assert {binding.name for binding in session.mcp_bindings} == names - {"read"}
    finally:
        await session.aclose()
    assert_reaped(tmp_path / "server")


async def test_empty_mcp_selection_is_offline_fresh_and_resume(tmp_path, providers):
    args = arguments(tmp_path, "--mcp-tools", "", "--tools", "read")
    session, *_ = await cli._open_session_async(args)
    await session.prompt("persist")
    path = session.session_file
    assert [tool.name for tool in session.agent.tools] == ["read"]
    await session.aclose()
    resumed, *_ = await cli._open_session_async(arguments(tmp_path, "--resume", str(path), "--mcp-tools", ""))
    try:
        assert [tool.name for tool in resumed.agent.tools] == ["read"]
        assert not resumed.mcp_server_states
    finally:
        await resumed.aclose()
    assert not (tmp_path / "server").exists()


async def test_resume_only_launches_recorded_servers_and_explicit_empty_removes_mcp(tmp_path, providers):
    pytest.importorskip("mcp")
    path = config_file(tmp_path)
    first, *_ = await cli._open_session_async(
        arguments(tmp_path, "--mcp-config", str(path), "--mcp-tools", "fixture/echo", "--tools", "read")
    )
    await first.prompt("persist")
    journal = first.session_file
    await first.aclose()
    config_file(tmp_path, unused=True)
    resumed, *_ = await cli._open_session_async(
        arguments(tmp_path, "--resume", str(journal), "--mcp-config", str(path))
    )
    try:
        assert {tool.name for tool in resumed.agent.tools} == {"read", binding_name("fixture", "echo")}
        assert not (tmp_path / "unused/pid").exists()
    finally:
        await resumed.aclose()
    assert_reaped(tmp_path / "server")
    disabled, *_ = await cli._open_session_async(
        arguments(tmp_path, "--resume", str(journal), "--mcp-tools", "")
    )
    try:
        assert [tool.name for tool in disabled.agent.tools] == ["read"]
        assert not disabled.mcp_server_states
    finally:
        await disabled.aclose()


@pytest.mark.parametrize("json_mode", [False, True])
async def test_run_mcp_tool_via_faux_provider_without_secret_diagnostics(
    tmp_path, monkeypatch, capsys, json_mode
):
    pytest.importorskip("mcp")
    path = config_file(tmp_path)
    provider = FauxProvider(
        [
            AssistantMessage(
                content=[
                    ToolCallBlock(
                        id="echo", name=binding_name("fixture", "echo"), arguments={"message": "hello"}
                    )
                ],
                stop_reason="toolUse",
            ),
            done(),
        ]
    )
    monkeypatch.setattr(cli, "_runtime_provider", lambda settings: provider)
    args = arguments(
        tmp_path,
        "hello",
        "--mcp-config",
        str(path),
        "--mcp-tools",
        "fixture/echo",
        *(["--json"] if json_mode else []),
    )
    assert await cli._cmd_run(args) == 0
    assert len(provider.requests) == 2
    result = next(
        message for message in provider.requests[-1].messages if isinstance(message, ToolResultMessage)
    )
    assert not result.is_error and result.structured_content["structuredContent"]["echo"] == {
        "message": "hello"
    }
    captured = capsys.readouterr()
    journal = SessionStore.list_sessions(tmp_path / "sessions")[0].path.read_text()
    for secret in ("synthetic-cli-secret", "synthetic-meta-secret", "mcp_harness_server.py"):
        assert secret not in captured.out + captured.err + journal
    if json_mode:
        assert all(json.loads(line)["type"] for line in captured.out.splitlines())
    else:
        assert captured.out == "offline done\n"
    assert_reaped(tmp_path / "server")


@pytest.mark.parametrize("command", ["run", "chat"])
async def test_startup_cli_change_failure_closes_prepared_mcp_session(
    tmp_path, providers, monkeypatch, command
):
    pytest.importorskip("mcp")
    path = config_file(tmp_path)
    opened = []
    original = cli.create_harness

    async def record_session(*args, **kwargs):
        session = await original(*args, **kwargs)
        opened.append(session)
        return session

    monkeypatch.setattr(cli, "create_harness", record_session)

    def invalid_change(*args):
        raise ValueError("synthetic invalid startup model")

    monkeypatch.setattr(cli, "_apply_cli_changes", invalid_change)
    args = arguments(tmp_path, "hello", "--mcp-config", str(path), command=command)
    with pytest.raises(ValueError, match="invalid startup model"):
        await (cli._cmd_run(args) if command == "run" else run_chat(args))
    assert_reaped(tmp_path / "server")
    assert opened[0]._disposed
    store = SessionStore.open(opened[0].session_file)
    store.close()


async def test_failed_preparation_cleans_server_and_creates_no_journal(tmp_path, providers):
    pytest.importorskip("mcp")
    path = config_file(tmp_path, mode="bad-schema")
    with pytest.raises(RuntimeError):
        await cli._open_session_async(arguments(tmp_path, "--mcp-config", str(path)))
    assert len(providers) == 1 and not SessionStore.list_sessions(tmp_path / "sessions")
    assert_reaped(tmp_path / "server")


async def test_cancelled_preparation_closes_server_and_leaves_no_tasks(tmp_path, providers):
    pytest.importorskip("mcp")
    path = config_file(tmp_path, mode="initialize-hang")
    before = asyncio.all_tasks()
    task = asyncio.create_task(cli._open_session_async(arguments(tmp_path, "--mcp-config", str(path))))
    try:
        async with asyncio.timeout(3):
            events = tmp_path / "server/events"
            while not events.exists() or "initialize" not in events.read_text():
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not SessionStore.list_sessions(tmp_path / "sessions")
        assert_reaped(tmp_path / "server")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before)


@pytest.mark.parametrize("selectors", [None, "fixture/echo", binding_name("fixture", "echo")])
@pytest.mark.parametrize("drift", ["schema", "declaration"])
async def test_resume_schema_changes_require_explicit_cli_reload(tmp_path, providers, selectors, drift):
    pytest.importorskip("mcp")
    path = config_file(tmp_path)
    session, *_ = await cli._open_session_async(
        arguments(tmp_path, "--mcp-config", str(path), "--mcp-tools", "fixture/echo")
    )
    await session.prompt("persist")
    journal = session.session_file
    await session.aclose()
    # Derive a changed declaration from the existing independent server fixture.
    changed = tmp_path / "changed_server.py"
    source = FIXTURE.read_text()
    changed.write_text(
        source.replace('{"type": "string"}', '{"type": "integer"}')
        if drift == "schema"
        else source.replace('"fixture " + name', '"changed " + name')
    )
    config_file(tmp_path, script=changed)
    before = journal.read_bytes()
    flags = [] if selectors is None else ["--mcp-tools", selectors]
    args = arguments(tmp_path, "--resume", str(journal), "--mcp-config", str(path), *flags)
    with pytest.raises(ValueError, match="MCP binding schema or declaration changed; explicitly reload MCP"):
        await cli._open_session_async(args)
    assert journal.read_bytes() == before
    assert_reaped(tmp_path / "server")
    args.mcp_reload = True
    refreshed, *_ = await cli._open_session_async(args)
    try:
        echo = next(tool for tool in refreshed.agent.tools if tool.name == binding_name("fixture", "echo"))
        assert echo.description == ("fixture echo" if drift == "schema" else "changed echo")
        assert echo.schema()["properties"]["message"]["type"] == (
            "integer" if drift == "schema" else "string"
        )
        assert len(refreshed.mcp_bindings) == 1
        assert journal.read_bytes() != before
    finally:
        await refreshed.aclose()
    assert_reaped(tmp_path / "server")


async def test_chat_reconnect_refresh_current_and_reload_all(tmp_path, providers, monkeypatch, capsys):
    pytest.importorskip("mcp")
    path = config_file(tmp_path, unused=True)
    snapshots = []
    opened = []
    original = cli.create_harness

    async def record_session(*args, **kwargs):
        session = await original(*args, **kwargs)
        opened.append(session)
        reconnect = session.reconnect_mcp
        reload_mcp = session.reload_mcp

        async def reconnect_and_record(server):
            await reconnect(server)
            snapshots.append({tool.name for tool in session.agent.tools})

        async def reload_and_record(selectors=None):
            await reload_mcp(selectors)
            snapshots.append({tool.name for tool in session.agent.tools})

        monkeypatch.setattr(session, "reconnect_mcp", reconnect_and_record)
        monkeypatch.setattr(session, "reload_mcp", reload_and_record)
        return session

    monkeypatch.setattr(cli, "create_harness", record_session)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "/mcp reconnect fixture\n/wait\n/mcp reload\n/wait\n/mcp reload all\n/wait\n/mcp\n/quit\n"
        ),
    )
    assert (
        await run_chat(
            arguments(
                tmp_path,
                "--mcp-config",
                str(path),
                "--mcp-tools",
                "fixture/echo",
                "--tools",
                "read",
                command="chat",
            )
        )
        == 0
    )
    selected = {"read", binding_name("fixture", "echo")}
    assert snapshots[:2] == [selected, selected]
    assert len(snapshots[2]) == 21
    assert binding_name("unused", "echo") in snapshots[2]
    assert opened[0]._disposed
    captured = capsys.readouterr()
    assert not captured.err and "MCP servers:" in captured.out and "MCP tools:" in captured.out
    assert "synthetic-cli-secret" not in captured.out
    assert_reaped(tmp_path / "server")
    assert_reaped(tmp_path / "unused")


async def test_chat_keeps_running_after_committed_cleanup_failure(tmp_path, providers, monkeypatch, capsys):
    pytest.importorskip("mcp")
    path = config_file(tmp_path)
    original = cli.create_harness
    opened = []
    attempts = []

    async def setup(*args, **kwargs):
        session = await original(*args, **kwargs)
        opened.append(session)
        await session.prompt("persist before reload")
        old = next(tool for tool in session.agent.tools if tool.provenance is not None)._connection
        close = old.aclose

        async def fail_once():
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError("synthetic-retirement-secret")
            await close()

        monkeypatch.setattr(old, "aclose", fail_once)
        return session

    monkeypatch.setattr(cli, "create_harness", setup)
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("/mcp reload fixture/echo,fixture/error\n/wait\n/mcp\n/status\n/quit\n")
    )
    assert (
        await run_chat(
            arguments(tmp_path, "--mcp-config", str(path), "--mcp-tools", "fixture/echo", command="chat")
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "loadout committed; cleanup pending" in captured.err
    assert "synthetic-retirement-secret" not in captured.err
    assert "MCP servers:" in captured.out and binding_name("fixture", "error") in captured.out
    assert opened[0]._disposed and len(attempts) >= 2
    assert_reaped(tmp_path / "server")


async def test_builtin_selection_preserves_mcp_until_explicit_offline_clear(tmp_path, providers):
    pytest.importorskip("mcp")
    path = config_file(tmp_path)
    first, *_ = await cli._open_session_async(
        arguments(tmp_path, "--mcp-config", str(path), "--mcp-tools", "fixture/echo", "--tools", "read")
    )
    try:
        await first.prompt("persist")
        journal = first.session_file
        recorded = first.mcp_bindings
        assert len(recorded) == 1
    finally:
        await first.aclose()
    before_journal = journal.read_bytes()
    before_events = (tmp_path / "server/events").read_bytes()
    with pytest.raises(McpConfigError, match="trusted configuration"):
        await cli._open_session_async(arguments(tmp_path, "--resume", str(journal), "--tools", "grep"))
    assert journal.read_bytes() == before_journal
    assert (tmp_path / "server/events").read_bytes() == before_events
    retained, *_ = await cli._open_session_async(
        arguments(tmp_path, "--resume", str(journal), "--mcp-config", str(path), "--tools", "grep")
    )
    try:
        assert {tool.name for tool in retained.agent.tools} == {"grep", binding_name("fixture", "echo")}
        assert retained.mcp_bindings == recorded
    finally:
        await retained.aclose()
    assert_reaped(tmp_path / "server")
    before = (tmp_path / "server/events").read_bytes()
    cleared, *_ = await cli._open_session_async(
        arguments(tmp_path, "--resume", str(journal), "--tools", "find", "--mcp-tools", "")
    )
    try:
        assert [tool.name for tool in cleared.agent.tools] == ["find"]
        assert cleared.mcp_bindings == () and not cleared.mcp_server_states
        assert (tmp_path / "server/events").read_bytes() == before
    finally:
        await cleared.aclose()
