"""Linux REPL: cancellable terminal input stays live during an agent run."""

import argparse
import asyncio
import codecs
import json
import os
import signal
import stat
import sys
from collections.abc import Awaitable, Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Literal, cast

from . import cli
from .cancellation import Cancellation
from .events import AgentSettled, MessageEnd, MessageUpdate, SessionEvent, TextDelta, event_to_json
from .permissions import PermissionRequest
from .runtime_config import read_environment, resolve_runtime
from .session import AgentSession
from .types import AssistantMessage, TextBlock, ThinkingLevel


class LineInput:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.eof = False
        self.decoder = codecs.getincrementaldecoder(stream.encoding or "utf-8")("strict")
        try:
            self.fd = stream.fileno()
            self.direct = stat.S_ISREG(os.fstat(self.fd).st_mode)
        except (AttributeError, OSError, ValueError):
            self.fd = -1
            self.direct = True

    async def readline(self) -> str:
        if self.direct:
            return self.stream.readline()
        loop = asyncio.get_running_loop()
        while "\n" not in self.buffer and not self.eof:
            ready = loop.create_future()

            def readable(ready=ready):
                if ready.done():
                    return
                try:
                    chunk = os.read(self.fd, 4096)
                    self.buffer += self.decoder.decode(chunk, final=not chunk)
                    self.eof = not chunk
                    ready.set_result(None)
                except Exception as error:
                    ready.set_exception(error)

            try:
                loop.add_reader(self.fd, readable)
                await ready
            finally:
                loop.remove_reader(self.fd)
        if "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            return line.rstrip("\r") + "\n"
        line, self.buffer = self.buffer, ""
        return line


class _Chat:
    """Own terminal interaction state while the session owns agent execution."""

    def __init__(self, args: argparse.Namespace, session: AgentSession) -> None:
        self.args = args
        self.session = session
        self.printed = False
        self.fatal = False
        self.job: asyncio.Task[None] | None = None
        self.reading: asyncio.Task[str] | None = None
        self.aborting: asyncio.Task[None] | None = None
        self.pending_approval: asyncio.Future[bool] | None = None
        self.approval_requested = asyncio.Event()
        self.input_closed = False
        self.loop = asyncio.get_running_loop()
        self.caller = asyncio.current_task()
        self.ui = sys.stderr if args.json else sys.stdout

    async def confirm(self, request: PermissionRequest, cancellation: Cancellation) -> bool:
        cancellation.raise_if_aborted()
        if self.input_closed:
            print("Permission denied: confirmation input is closed", file=cli._diagnostics())
            return False
        approval = self.loop.create_future()
        self.pending_approval = approval
        self.approval_requested.set()
        print(
            "Permission required: "
            + json.dumps(
                {
                    "tool": request.context.tool_call.name,
                    "args": request.context.args,
                    "reason": request.reason,
                    "resolved_path": request.resolved_path,
                },
                ensure_ascii=False,
            ),
            file=cli._diagnostics(),
        )
        print("Allow this call once? [y/N]: ", end="", flush=True, file=cli._diagnostics())
        try:
            return await approval
        finally:
            if self.pending_approval is approval:
                self.pending_approval = None
                self.approval_requested.clear()
            print(file=cli._diagnostics())

    def render_event(self, event: SessionEvent) -> None:
        if self.args.json:
            print(json.dumps(event_to_json(event), ensure_ascii=False, allow_nan=False), flush=True)
        elif isinstance(event, MessageUpdate) and isinstance(event.stream_event, TextDelta):
            print(event.stream_event.delta, end="", flush=True)
            self.printed = True
        elif isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            if not self.printed:
                text = "\n".join(
                    block.text for block in event.message.content if isinstance(block, TextBlock)
                )
                if text:
                    print(text, end="", flush=True)
                    self.printed = True
            if self.printed:
                print(flush=True)
            self.printed = False
            if event.message.stop_reason in ("error", "length"):
                print(
                    event.message.error_message or f"Run stopped: {event.message.stop_reason}",
                    file=cli._diagnostics(),
                )
        if isinstance(event, AgentSettled) and event.aborted:
            print("Operation aborted", file=cli._diagnostics())

    async def operation(self, awaitable: Awaitable[object]) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            pass
        except (OSError, ValueError, RuntimeError) as error:
            print(str(error), file=cli._diagnostics())
            self.fatal = not self.session.store.usable

    async def launch(self, awaitable: Awaitable[object]) -> None:
        self.job = asyncio.create_task(self.operation(awaitable))
        await asyncio.sleep(0)  # Claim lifecycle before accepting the next input.

    def interrupt(self, sig: int) -> None:
        if (
            sig == signal.SIGINT
            and not self.session.is_idle
            and (self.aborting is None or self.aborting.done())
        ):
            self.aborting = asyncio.create_task(self.session.abort())
        elif self.caller is not None:
            self.caller.cancel()

    def change_model(self, value: str) -> None:
        if not value or len(value.split()) != 1:
            raise ValueError("/model requires [provider/]ID")
        provider_name, separator, identifier = value.partition("/")
        if (
            separator
            and provider_name in ("openai", "deepseek", "anthropic")
            and provider_name != self.session.model.provider
        ):
            target = resolve_runtime(
                read_environment(Path(self.args.env_file).resolve()),
                provider=provider_name,
                model_id=identifier,
            )
            self.session.set_model(target.model, provider=cli._runtime_provider(target))
        else:
            identifier = identifier if separator and provider_name == self.session.model.provider else value
            self.session.set_model(
                replace(self.session.model, id=identifier, name=identifier, context_window=0)
            )

    async def handle_mcp(self, value: str) -> None:
        if not value:
            print(
                "MCP servers: " + json.dumps(dict(self.session.mcp_server_states), ensure_ascii=True),
                file=self.ui,
                flush=True,
            )
            identities = {
                binding.name: f"{binding.server}/{binding.tool}" for binding in self.session.mcp_bindings
            }
            print("MCP tools: " + json.dumps(identities, ensure_ascii=True), file=self.ui, flush=True)
        else:
            action, _, selectors = value.partition(" ")
            selectors = selectors.strip()
            if action not in ("reconnect", "reload"):
                raise ValueError("Use /mcp, /mcp reconnect NAME, or /mcp reload [SELECTORS|all]")
            if action == "reconnect" and len(selectors.split()) != 1:
                raise ValueError("/mcp reconnect requires one server name")
            if not self.session.is_idle or (self.job is not None and not self.job.done()):
                raise RuntimeError("Wait for the active operation before changing MCP tools")
            if action == "reconnect":
                await self.launch(self.session.reconnect_mcp(selectors))
            else:
                selected: tuple[str, ...] | None
                if selectors == "all":
                    selected = ("*",)
                else:
                    selected = cli._mcp_tool_names(selectors) if selectors else None
                await self.launch(self.session.reload_mcp(selected))

    async def wait(self) -> None:
        if self.job is not None:
            if self.args.confirm_permissions:
                notification = asyncio.create_task(self.approval_requested.wait())
                try:
                    ready, _ = await asyncio.wait(
                        [self.job, notification], return_when=asyncio.FIRST_COMPLETED
                    )
                    if self.job in ready:
                        await self.job
                        self.job = None
                finally:
                    notification.cancel()
                    await asyncio.gather(notification, return_exceptions=True)
            else:
                await self.job
                self.job = None
        if self.job is None:
            await self.session.wait_for_idle()

    async def handle_command(self, line: str) -> int | None:
        command, _, value = line.partition(" ")
        value = value.strip()
        if command in ("/quit", "/exit"):
            await self.session.abort()
            return 0
        if command == "/help":
            print(
                "/steer TEXT  /follow TEXT  /abort  /wait  /compact\n"
                "/model [provider/]ID  /thinking LEVEL  /parallel MODE [LIMIT]\n"
                "/mcp  /mcp reconnect NAME  /mcp reload [SELECTORS|all]\n"
                "/skills  /skill:NAME [ARGS]  /reload  /status  /quit\nText entered during a run queues a follow-up.",
                file=self.ui,
                flush=True,
            )
        elif command in ("/steer", "/follow"):
            if not value:
                raise ValueError(f"{command} requires text")
            if self.job is None or self.job.done():
                await self.launch(self.session.prompt(value))
            elif command == "/steer":
                self.session.steer(value)
            else:
                self.session.follow_up(value)
        elif command == "/abort":
            await self.session.abort()
            if self.job is not None:
                await self.job
                self.job = None
        elif command == "/wait":
            await self.wait()
        elif command == "/compact":
            if not self.session.is_idle:
                raise RuntimeError("Wait for the active run before compacting")
            await self.launch(self.session.compact(value or None))
        elif command == "/thinking":
            self.session.set_thinking_level(cast(ThinkingLevel, value))
        elif command == "/model":
            self.change_model(value)
        elif command == "/parallel":
            tokens = value.split()
            if not 1 <= len(tokens) <= 2:
                raise ValueError("/parallel requires sequential|parallel [positive limit]")
            self.session.set_tool_execution(
                cast(Literal["sequential", "parallel"], tokens[0]),
                max_parallel_tools=int(tokens[1]) if len(tokens) == 2 else None,
            )
        elif command == "/mcp":
            await self.handle_mcp(value)
        elif command == "/skills":
            for skill in self.session.skills:
                print(f"{skill.name}: {skill.description}\n  {skill.file_path}", file=self.ui, flush=True)
        elif command == "/reload":
            diagnostics = self.session.reload_resources()
            for diagnostic in diagnostics:
                print(f"skills: {diagnostic.path}: {diagnostic.message}", file=self.ui, flush=True)
            print(f"Resources reloaded: {len(self.session.skills)} skills", file=self.ui, flush=True)
        elif command == "/status":
            print(
                f"{self.session.model.provider}/{self.session.model.id} thinking={self.session.thinking_level} "
                f"tools={self.session.config.tool_execution} cwd={self.session.cwd}",
                file=self.ui,
                flush=True,
            )
        else:
            raise ValueError("Unknown command; use /help")
        return None

    async def read_commands(self, source: LineInput) -> int:
        while True:
            if self.reading is None:
                if sys.stdin.isatty() and not self.args.json:
                    print("pi> ", end="", flush=True)
                self.reading = asyncio.create_task(source.readline())
            watched = [self.reading] + ([self.job] if self.job is not None else [])
            completed, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            if self.job is not None and self.job in completed:
                await self.job
                self.job = None
                if self.fatal:
                    return 2
            if self.reading not in completed:
                continue
            line = await self.reading
            self.reading = None
            if not line:
                self.input_closed = True
                if self.pending_approval is not None and not self.pending_approval.done():
                    self.pending_approval.set_result(False)
                if self.job is not None:
                    await self.job
                await self.session.wait_for_idle()
                return 2 if self.fatal else 0
            line = line.strip()
            if self.pending_approval is not None and not line.startswith("/"):
                if not self.pending_approval.done():
                    self.pending_approval.set_result(line.lower() in ("y", "yes"))
                    self.approval_requested.clear()
                await asyncio.sleep(0)
                continue
            if not line:
                continue
            try:
                if not line.startswith("/") or line.startswith("/skill:"):
                    if self.job is not None and not self.job.done():
                        self.session.follow_up(line)
                    else:
                        await self.launch(self.session.prompt(line))
                    continue
                result = await self.handle_command(line)
                if result is not None:
                    return result
            except (OSError, ValueError, RuntimeError, argparse.ArgumentTypeError) as error:
                print(str(error), file=cli._diagnostics())
                if not self.session.store.usable:
                    return 2

    async def close(self, unsubscribe: Callable[[], None]) -> None:
        if self.reading is not None:
            self.reading.cancel()
            await asyncio.gather(self.reading, return_exceptions=True)
        if not self.session.is_idle:
            await self.session.abort()
        if self.job is not None:
            await asyncio.gather(self.job, return_exceptions=True)
        if self.aborting is not None:
            await asyncio.gather(self.aborting, return_exceptions=True)
        await self.session.wait_for_idle()
        unsubscribe()
        await self.session.aclose()


async def run_chat(args: argparse.Namespace) -> int:
    session, settings, switch, resumed = await cli._open_session_async(args)
    chat = _Chat(args, session)
    installed = []
    unsubscribe = session.subscribe(chat.render_event)
    try:
        cli._apply_cli_changes(session, args, settings, switch, resumed)
        if args.confirm_permissions:
            # One reader owns both commands and permission answers.
            session.agent._confirm_permission = chat.confirm
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                chat.loop.add_signal_handler(sig, partial(chat.interrupt, sig))
                installed.append(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        source = LineInput(sys.stdin)
        if sys.stdin.isatty():
            print("pi-agent chat — /help lists commands, /quit exits", file=chat.ui, flush=True)
        if args.prompt:
            await chat.launch(session.prompt(" ".join(args.prompt)))
        return await chat.read_commands(source)
    finally:
        for sig in installed:
            chat.loop.remove_signal_handler(sig)
        await chat.close(unsubscribe)
