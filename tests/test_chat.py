import asyncio
import io
import json
import os
import signal
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pi_agent import cli, create_agent_session
from pi_agent.chat import LineInput, run_chat
from pi_agent.events import QueueUpdate
from pi_agent.permissions import PermissionPolicy
from pi_agent.providers import FauxProvider
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage


def answer(text="done"):
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop")


def inject(monkeypatch, config, provider, *, persisted=False):
    session = create_agent_session(
        provider,
        config=replace(config, model=replace(config.model, reasoning=True)),
        session="new" if persisted else None,
    )
    monkeypatch.setattr(
        cli, "_open_session", lambda args: (session, SimpleNamespace(model=session.model), False, False)
    )
    return session


@pytest.mark.parametrize("json_mode", [False, True])
async def test_chat_commands_prompts_switch_and_clean_json(config, monkeypatch, capsys, json_mode):
    provider = FauxProvider([answer("first answer"), answer("second answer")])
    session = inject(monkeypatch, config, provider, persisted=True)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "/thinking high\n/parallel parallel 2\nfirst\n/wait\n/model second\nsecond\n/wait\n/status\n/help\n/quit\n"
        ),
    )
    args = cli._build_parser().parse_args(["chat"] + (["--json"] if json_mode else []))
    assert await run_chat(args) == 0
    assert session._disposed and len(provider.requests) == 2
    assert provider.requests[-1].model.id == "second"
    assert session.config.tool_execution == "parallel" and session.thinking_level == "high"
    captured = capsys.readouterr()
    if json_mode:
        events = [json.loads(line) for line in captured.out.splitlines()]
        assert any(event["type"] == "model_changed" for event in events)
        assert "/steer" in captured.err
    else:
        assert "first answer" in captured.out and "second answer" in captured.out
    resumed = create_agent_session(FauxProvider([]), session=session.session_file)
    try:
        assert resumed.model.id == "second" and resumed.config.max_parallel_tools == 2
    finally:
        resumed.dispose()


async def test_pipe_input_remains_live_for_steer_and_follow_up(config, monkeypatch, capsys):
    entered = asyncio.Event()
    released = asyncio.Event()
    queued = asyncio.Event()

    class Gate(FauxProvider):
        async def stream(self, request, options):
            if not self.requests:
                entered.set()
                await released.wait()
            async for event in super().stream(request, options):
                yield event

    provider = Gate([answer(), answer(), answer()])
    session = inject(monkeypatch, config, provider)
    session.subscribe(
        lambda event: queued.set() if isinstance(event, QueueUpdate) and event.follow_up else None
    )
    read_fd, write_fd = os.pipe()
    before_tasks = asyncio.all_tasks()
    with os.fdopen(read_fd, "r", encoding="utf-8") as source:
        monkeypatch.setattr("sys.stdin", source)
        task = asyncio.create_task(run_chat(cli._build_parser().parse_args(["chat"])))
        try:
            os.write(write_fd, b"first\n")
            await asyncio.wait_for(entered.wait(), 1)
            os.write(write_fd, b"/steer steer text\n/follow follow text\n")
            await asyncio.wait_for(queued.wait(), 1)
            released.set()
            os.write(write_fd, b"/wait\n/quit\n")
            assert await asyncio.wait_for(task, 1) == 0
            assert len(provider.requests) == 3
            assert "steer text" in str(provider.requests[1].messages)
            assert "follow text" in str(provider.requests[2].messages)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            os.close(write_fd)
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before_tasks)
    assert not capsys.readouterr().err


@pytest.mark.parametrize("signal_stop", [False, True])
async def test_abort_or_signal_cleans_chat_input_and_model_waiters(config, monkeypatch, capsys, signal_stop):
    from pi_agent.events import StreamStart

    entered = asyncio.Event()
    closed = asyncio.Event()

    class Hanging(FauxProvider):
        async def stream(self, request, options):
            try:
                yield StreamStart()
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

    session = inject(monkeypatch, config, Hanging([]))
    callbacks = {}
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: callbacks.update({sig: callback}))
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: callbacks.pop(sig, None))
    read_fd, write_fd = os.pipe()
    before_tasks = asyncio.all_tasks()
    with os.fdopen(read_fd, "r", encoding="utf-8") as source:
        monkeypatch.setattr("sys.stdin", source)
        task = asyncio.create_task(run_chat(cli._build_parser().parse_args(["chat"])))
        try:
            os.write(write_fd, b"first\n")
            await asyncio.wait_for(entered.wait(), 1)
            if signal_stop:
                callbacks[signal.SIGTERM]()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
            else:
                os.write(write_fd, b"/abort\n/quit\n")
                assert await asyncio.wait_for(task, 1) == 0
            assert closed.is_set() and session._disposed and not callbacks
            os.fstat(read_fd)  # Input is caller-owned and stays open.
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            os.close(write_fd)
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before_tasks)
    assert "Operation aborted" in capsys.readouterr().err


async def test_line_input_split_utf8_and_cancelled_reader_ownership(monkeypatch):
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "r", encoding="utf-8") as source:
        lines = LineInput(source)
        first = asyncio.create_task(lines.readline())
        try:
            encoded = "\u041f\u0440\u0438\u0432\u0435\u0442\nnext".encode()
            os.write(write_fd, encoded[:1])
            await asyncio.sleep(0)
            os.write(write_fd, encoded[1:])
            assert await asyncio.wait_for(first, 1) == "\u041f\u0440\u0438\u0432\u0435\u0442\n"
            os.close(write_fd)
            assert await lines.readline() == "next"
            assert await lines.readline() == ""
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)


@pytest.mark.parametrize("response,approved", [("yes", True), ("", False), ("no", False)])
async def test_chat_permission_answers_use_single_repl_reader_and_wait_stays_live(
    config, monkeypatch, capsys, response, approved
):
    provider = FauxProvider(
        [
            AssistantMessage(
                content=[ToolCallBlock(id="w", name="write", arguments={"path": "result", "content": "new"})],
                stop_reason="toolUse",
            ),
            answer(),
        ]
    )
    config.permissions = PermissionPolicy(default="ask")
    session = inject(monkeypatch, config, provider)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"first\n/wait\n{response}\n/wait\n/quit\n"))
    assert (
        await asyncio.wait_for(run_chat(cli._build_parser().parse_args(["chat", "--confirm-permissions"])), 1)
        == 0
    )
    assert (config.cwd / "result").exists() is approved
    result = next(m for m in session.messages if isinstance(m, ToolResultMessage))
    assert result.is_error is not approved
    assert "Allow this call once?" in capsys.readouterr().err


async def test_chat_eof_denies_ask_even_when_it_arrives_before_confirmation(config, monkeypatch, capsys):
    config.permissions = PermissionPolicy(default="ask")
    provider = FauxProvider(
        [
            AssistantMessage(
                content=[ToolCallBlock(id="w", name="write", arguments={"path": "never", "content": "new"})],
                stop_reason="toolUse",
            ),
            answer(),
        ]
    )
    session = inject(monkeypatch, config, provider)
    monkeypatch.setattr("sys.stdin", io.StringIO("first\n"))
    assert (
        await asyncio.wait_for(run_chat(cli._build_parser().parse_args(["chat", "--confirm-permissions"])), 1)
        == 0
    )
    assert session._disposed and not (config.cwd / "never").exists()
    assert next(m for m in session.messages if isinstance(m, ToolResultMessage)).is_error
