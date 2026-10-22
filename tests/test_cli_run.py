import asyncio
import json
import os
import signal
from pathlib import Path

import httpx
import pytest

from pi_agent import cli
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.permissions import PermissionRequest
from pi_agent.providers import OpenAICompatProvider
from pi_agent.runtime_config import _ENV_NAMES, read_environment, resolve_runtime
from pi_agent.session_store import SessionStore
from pi_agent.tool_hooks import BeforeToolCallContext, invoke_hook
from pi_agent.types import AssistantMessage, ToolCallBlock


def tool_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    if payload["messages"][-1]["role"] == "user":
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "write",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": "result.txt", "content": "written"}),
                    },
                }
            ]
        }
        finish = "tool_calls"
    else:
        delta, finish = {"content": "done"}, "stop"
    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
    )


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        'OPENAI_API_KEY="test-only"\nOPENAI_BASE_URL=https://unit.test\n'
        "OPENAI_MODEL=deepseek-flash\nPI_PROVIDER=deepseek\n",
        encoding="utf-8",
    )
    return path


def test_environment_precedence_and_secret_is_not_exported(
    environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "shell-model")
    settings = resolve_runtime(read_environment(environment))
    assert settings.model.id == "shell-model"
    assert "test-only" not in repr(settings)
    assert "OPENAI_API_KEY" not in os.environ
    assert resolve_runtime(read_environment(environment), model_id="cli-model").model.id == "cli-model"


@pytest.mark.parametrize("json_mode", [False, True])
async def test_run_print_json_and_resume(
    environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, json_mode: bool
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.headers["authorization"] == "Bearer test-only"
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "\u041f\u0440\u0438\u0432\u0435\u0442"},
                    "finish_reason": "stop",
                }
            ]
        }
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        parser = cli._build_parser()
        argv = [
            "run",
            "hello",
            "--cwd",
            str(tmp_path),
            "--env-file",
            str(environment),
            "--session-dir",
            str(tmp_path / "sessions"),
        ]
        if json_mode:
            argv.append("--json")
        assert await cli._cmd_run(parser.parse_args(argv)) == 0
        captured = capsys.readouterr()
        assert not captured.err
        if json_mode:
            events = [json.loads(line) for line in captured.out.splitlines()]
            assert events[0]["type"] == "agent_start"
            assert events[-1]["type"] == "agent_settled"
        else:
            assert captured.out == "\u041f\u0440\u0438\u0432\u0435\u0442\n"
        candidates = SessionStore.list_sessions(tmp_path / "sessions", match_cwd=tmp_path)
        assert len(candidates) == 1
        assert "test-only" not in candidates[0].path.read_text()
        assert "https://unit.test" not in candidates[0].path.read_text()
        # The stored identity wins over changed environment defaults on resume.
        monkeypatch.setenv("OPENAI_MODEL", "different-model")
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue"])) == 0
        assert requests[1]["model"] == "deepseek-flash"
        assert [item["role"] for item in requests[1]["messages"]] == ["system", "user", "assistant", "user"]
        with candidates[0].path.open("ab") as target:
            target.write(b'{"torn": "\xe2\x82')
        assert (
            await cli._cmd_run(
                parser.parse_args(
                    [
                        "run",
                        "third",
                        "--resume",
                        str(candidates[0].path),
                        "--env-file",
                        str(environment),
                    ]
                )
            )
            == 0
        )
        assert b'"torn"' not in candidates[0].path.read_bytes()
        with candidates[0].path.open("ab") as target:
            target.write(b'{"torn": "\xe2\x82')
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue"])) == 0
        assert b'"torn"' not in candidates[0].path.read_bytes()


def test_missing_key_is_configuration_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert cli.main(["run", "hi", "--env-file", str(tmp_path / "missing"), "--no-session"]) == 2
    assert capsys.readouterr().err.strip() == "OPENAI_API_KEY is required"


@pytest.mark.parametrize("finish,expected", [("length", 1), ("content_filter", 1)])
async def test_failed_run_exit_status(
    environment: Path, monkeypatch: pytest.MonkeyPatch, capsys, finish: str, expected: int
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        chunk = {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": finish}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        args = cli._build_parser().parse_args(
            ["run", "hello", "--env-file", str(environment), "--no-session"]
        )
        assert await cli._cmd_run(args) == expected
        assert capsys.readouterr().err


def test_ignored_env_patterns(tmp_path: Path) -> None:
    # Validate the actual ignore rules using a temporary Git repository.
    import subprocess

    root = Path(__file__).resolve().parents[1]
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "check-ignore", ".env", ".env.local"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [".env", ".env.local"]
    assert (
        subprocess.run(
            ["git", "-C", str(tmp_path), "check-ignore", ".env.example"], capture_output=True
        ).returncode
        == 1
    )


async def test_cli_signal_aborts_http_and_returns_130(
    environment: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    reached = asyncio.Event()
    closed = asyncio.Event()
    callbacks = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: callbacks.update({sig: callback}))
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: callbacks.pop(sig, None))

    class HangingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
            reached.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=HangingBody())
        )
    ) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        args = cli._build_parser().parse_args(
            ["run", "hello", "--env-file", str(environment), "--no-session"]
        )
        task = asyncio.create_task(cli._cmd_run(args))
        await asyncio.wait_for(reached.wait(), 1)
        callbacks[signal.SIGTERM]()
        assert await asyncio.wait_for(task, 1) == 130
        assert closed.is_set()
        assert not callbacks
        assert "Operation aborted" in capsys.readouterr().err


async def test_cli_read_only_survives_continue_and_preserves_policy_rules(
    environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    policy = tmp_path / "permissions.json"
    policy.write_text(json.dumps({"default": "allow", "rules": [{"tool": "read", "action": "deny"}]}))
    sessions = tmp_path / "sessions"
    argv = [
        "run",
        "write",
        "--cwd",
        str(tmp_path),
        "--env-file",
        str(environment),
        "--session-dir",
        str(sessions),
        "--json",
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(tool_response)) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        parser = cli._build_parser()
        for flags in (["--permissions-file", str(policy), "--read-only"], ["--continue"]):
            assert await cli._cmd_run(parser.parse_args(argv + flags)) == 0
            captured = capsys.readouterr()
            assert not captured.err
            events = [json.loads(line) for line in captured.out.splitlines()]
            end = next(event for event in events if event["type"] == "tool_execution_end")
            assert end["is_error"] and end["result"]["is_error"]
            assert "Read-only" in end["result"]["content"][0]["text"]
            assert not (tmp_path / "result.txt").exists()
        candidate = SessionStore.list_sessions(sessions)[0]
        store = SessionStore.open(candidate.path)
        try:
            stored = store.header.config["permissions"]
            assert stored["read_only"] and stored["rules"][0]["action"] == "deny"
        finally:
            store.close()


@pytest.mark.parametrize(
    "source",
    [
        '{"default":"allow","default":"deny"}',
        '{"rules":[{"tool":"write","action":"allow","action":"deny"}]}',
        '{"read_only":"false"}',
        '{"unknown":true}',
    ],
)
def test_cli_rejects_ambiguous_or_invalid_policy(tmp_path: Path, source: str) -> None:
    policy = tmp_path / "permissions.json"
    policy.write_text(source)
    with pytest.raises(ValueError):
        cli._read_permission_policy(policy)


@pytest.mark.parametrize(
    "answer,approved", [(b"y\n", True), (b"YES\n", True), (b"\n", False), (b"n\n", False), (b"n\ny\n", False)]
)
async def test_cli_confirmation_uses_tty_keeps_json_clean_and_closes_reader(
    environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, answer: bytes, approved: bool
) -> None:
    master, slave = os.openpty()
    opened = asyncio.Event()
    descriptors = []
    actual_open = os.open

    def tty_open(path, flags, *args, **kwargs):
        if path != "/dev/tty":
            return actual_open(path, flags, *args, **kwargs)
        fd = os.dup(slave)
        os.set_blocking(fd, False)
        descriptors.append(fd)
        opened.set()
        return fd

    monkeypatch.setattr(cli.os, "open", tty_open)
    policy = tmp_path / "permissions.json"
    policy.write_text('{"default":"ask"}')
    args = cli._build_parser().parse_args(
        [
            "run",
            "write",
            "--cwd",
            str(tmp_path),
            "--env-file",
            str(environment),
            "--no-session",
            "--json",
            "--permissions-file",
            str(policy),
            "--confirm-permissions",
        ]
    )
    task = None
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(tool_response)) as client:
            monkeypatch.setattr(
                cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
            )
            task = asyncio.create_task(cli._cmd_run(args))
            await asyncio.wait_for(opened.wait(), 1)
            os.write(master, answer)
            assert await asyncio.wait_for(task, 1) == 0
        captured = capsys.readouterr()
        assert "Allow this call once? [y/N]" in captured.err
        events = [json.loads(line) for line in captured.out.splitlines()]
        end = next(event for event in events if event["type"] == "tool_execution_end")
        assert end["is_error"] is not approved
        assert (tmp_path / "result.txt").exists() is approved
        assert len(descriptors) == 1
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize("abort", [False, True])
async def test_cli_confirmation_no_tty_denies_and_signal_during_ask_returns_130(
    environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, abort: bool
) -> None:
    opened = asyncio.Event()
    closed = asyncio.Event()
    callbacks = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: callbacks.update({sig: callback}))
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: callbacks.pop(sig, None))

    if abort:

        async def hanging(request, token):
            opened.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        monkeypatch.setattr(cli, "_confirm_on_tty", hanging)
    else:
        actual_open = os.open

        def no_tty(path, flags, *args, **kwargs):
            if path == "/dev/tty":
                raise OSError("No controlling terminal")
            return actual_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(cli.os, "open", no_tty)
    policy = tmp_path / "permissions.json"
    policy.write_text('{"default":"ask"}')
    args = cli._build_parser().parse_args(
        [
            "run",
            "write",
            "--cwd",
            str(tmp_path),
            "--env-file",
            str(environment),
            "--no-session",
            "--json",
            "--permissions-file",
            str(policy),
            "--confirm-permissions",
        ]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(tool_response)) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        task = asyncio.create_task(cli._cmd_run(args))
        try:
            if abort:
                await asyncio.wait_for(opened.wait(), 1)
                callbacks[signal.SIGTERM]()
            assert await asyncio.wait_for(task, 1) == (130 if abort else 0)
            assert not (tmp_path / "result.txt").exists()
            assert not callbacks
            captured = capsys.readouterr()
            events = [json.loads(line) for line in captured.out.splitlines()]
            assert events[-1]["type"] == "agent_settled"
            assert events[-1]["aborted"] is abort
            assert "Operation aborted" in captured.err if abort else "no terminal" in captured.err
            if abort:
                assert closed.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("external", [False, True])
async def test_real_tty_reader_is_removed_and_closed_on_confirmation_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, external: bool
) -> None:
    master, slave = os.openpty()
    opened = asyncio.Event()
    descriptors = []
    actual_open = os.open
    actual_remove = asyncio.get_running_loop().remove_reader
    removed = []

    def tty_open(path, flags, *args, **kwargs):
        if path != "/dev/tty":
            return actual_open(path, flags, *args, **kwargs)
        fd = os.dup(slave)
        os.set_blocking(fd, False)
        descriptors.append(fd)
        opened.set()
        return fd

    def remove_reader(fd):
        removed.append(fd)
        return actual_remove(fd)

    monkeypatch.setattr(cli.os, "open", tty_open)
    monkeypatch.setattr(asyncio.get_running_loop(), "remove_reader", remove_reader)
    block = ToolCallBlock(id="c", name="write", arguments={"path": "result.txt", "content": "new"})
    context = BeforeToolCallContext(
        AssistantMessage(content=[block]), block, dict(block.arguments), cwd=tmp_path
    )
    request = PermissionRequest(context, "ask", str(tmp_path / "result.txt"))
    token = Cancellation()
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(invoke_hook(cli._confirm_on_tty, request, token))
    try:
        await asyncio.wait_for(opened.wait(), 1)
        if external:
            task.cancel()
        else:
            token.abort()
        with pytest.raises(asyncio.CancelledError if external else OperationAbortedError):
            await asyncio.wait_for(task, 1)
        assert removed == descriptors and len(descriptors) == 1
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
        assert not capsys.readouterr().out
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        os.close(master)
        os.close(slave)
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before_tasks)


async def test_cli_resume_model_thinking_and_limit_changes_are_durable(
    environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        chunk = {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    argv = [
        "run",
        "hello",
        "--cwd",
        str(tmp_path),
        "--env-file",
        str(environment),
        "--session-dir",
        str(tmp_path / "sessions"),
        "--json",
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        parser = cli._build_parser()
        assert await cli._cmd_run(parser.parse_args(argv + ["--read-only"])) == 0
        capsys.readouterr()
        assert (
            await cli._cmd_run(
                parser.parse_args(
                    argv
                    + [
                        "--continue",
                        "--model",
                        "second",
                        "--thinking",
                        "high",
                        "--context-window",
                        "50000",
                        "--max-tokens",
                        "1024",
                    ]
                )
            )
            == 0
        )
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert [e["type"] for e in events[:2]] == ["model_changed", "thinking_level_changed"]
        assert payloads[-1]["model"] == "second" and payloads[-1]["max_tokens"] == 1024
        assert payloads[-1]["thinking"] == {"type": "enabled"}
        monkeypatch.setenv("OPENAI_MODEL", "different-environment")
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue"])) == 0
        capsys.readouterr()
        assert payloads[-1]["model"] == "second" and payloads[-1]["thinking"]["type"] == "enabled"
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue", "--max-tokens", "4096"])) == 0
        capsys.readouterr()
        assert payloads[-1]["model"] == "second" and payloads[-1]["max_tokens"] == 4096
        assert payloads[-1]["thinking"]["type"] == "enabled"
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue", "--thinking", "off"])) == 0
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert events[0]["type"] == "thinking_level_changed"
        assert payloads[-1]["thinking"] == {"type": "disabled"}
    path = SessionStore.list_sessions(tmp_path / "sessions")[0].path
    store = SessionStore.open(path)
    try:
        latest = next(e.config for e in reversed(store.entries) if e.type == "config_change")
        assert latest["model"]["id"] == "second" and latest["model"]["context_window"] == 50000
        assert latest["model"]["max_tokens"] == 4096 and latest["thinking_level"] == "off"
        assert latest["permissions"]["read_only"]
    finally:
        store.close()


def anthropic_response() -> httpx.Response:
    events = [
        {
            "type": "message_start",
            "message": {"id": "response", "model": "claude-test", "usage": {"input_tokens": 3}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "done"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
    )


async def test_cli_switch_provider_and_reject_invalid_thinking_without_config_commit(
    environment: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from pi_agent.providers import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-only")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unit.test")
    payloads = []

    def handler(request):
        payloads.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/v1/messages":
            return anthropic_response()
        chunk = {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    argv = [
        "run",
        "hello",
        "--cwd",
        str(tmp_path),
        "--env-file",
        str(environment),
        "--session-dir",
        str(tmp_path / "sessions"),
        "--json",
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        monkeypatch.setattr(
            cli, "AnthropicProvider", lambda **kwargs: AnthropicProvider(client=client, **kwargs)
        )
        parser = cli._build_parser()
        assert await cli._cmd_run(parser.parse_args(argv)) == 0
        capsys.readouterr()
        assert (
            await cli._cmd_run(
                parser.parse_args(
                    argv
                    + [
                        "--continue",
                        "--provider",
                        "anthropic",
                        "--model",
                        "claude-test",
                        "--thinking",
                        "high",
                        "--max-tokens",
                        "1024",
                    ]
                )
            )
            == 0
        )
        capsys.readouterr()
        assert payloads[-1][0] == "/v1/messages" and payloads[-1][1]["thinking"] == {"type": "adaptive"}
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue", "--thinking", "off"])) == 0
        capsys.readouterr()
        path = SessionStore.list_sessions(tmp_path / "sessions")[0].path
        before = path.read_bytes()
        requests_before = len(payloads)
        monkeypatch.setenv("ANTHROPIC_THINKING_MODE", "budget")
        with pytest.raises(ValueError, match="max_tokens"):
            await cli._cmd_run(parser.parse_args(argv + ["--continue", "--thinking", "high"]))
        assert len(payloads) == requests_before and path.read_bytes() == before
        store = SessionStore.open(path)
        store.close()  # Failed configuration must dispose and release the lock.
