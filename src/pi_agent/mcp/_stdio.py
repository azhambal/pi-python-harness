"""Bounded MCP stdio transport with same-task SDK ownership and process cleanup."""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from mcp.shared.message import SessionMessage
    from pydantic import BaseModel

_CHUNK = 8192
_JSON_DEPTH = 64


@dataclass(frozen=True, slots=True)
class StdioLimits:
    """Positive limits that may be reduced, but never raised above hard caps."""

    frame_bytes: int = 1024 * 1024
    outgoing_bytes: int = 1024 * 1024
    stderr_bytes: int = 64 * 1024
    total_stdout_bytes: int = 16 * 1024 * 1024
    queued_frames: int = 16
    shutdown_timeout: float = 0.5

    def __post_init__(self) -> None:
        for name, cap in (
            ("frame_bytes", 1024 * 1024),
            ("outgoing_bytes", 1024 * 1024),
            ("stderr_bytes", 64 * 1024),
            ("total_stdout_bytes", 16 * 1024 * 1024),
            ("queued_frames", 16),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= cap:
                raise ValueError(f"Invalid MCP stdio limit: {name}")
        timeout = self.shutdown_timeout
        if type(timeout) not in (int, float) or not 0 < timeout <= 0.5:
            raise ValueError("Invalid MCP stdio limit: shutdown_timeout")


@dataclass(slots=True)
class Transport:
    """SDK-compatible channels and a first-failure notification.

    ``failure`` is None while healthy, otherwise a fixed generic code. ``failed``
    is set before read-stream EOF and process cleanup; callers may disconnect
    immediately without waiting for termination. Context exit joins cleanup.
    """

    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    failed: asyncio.Event = field(default_factory=asyncio.Event)
    failure: str | None = None
    _connection: _Connection | None = field(default=None, init=False, repr=False)

    async def aclose(self) -> None:
        """Join owned cleanup; retry failed cleanup, including after context exit.

        This method does not exit the caller's AnyIO contexts and can be called
        from a separate resource-owner task. Cleanup errors are generic; a failed
        attempt retains ownership so the same Transport can be closed again.
        """
        if self._connection is not None:
            try:
                _, cancelled = await _joined(self._connection.close())
            except Exception:
                if self.failure is None:
                    self.failure = "shutdown_failed"
                    self.failed.set()
                raise RuntimeError("MCP stdio cleanup failed") from None
            if cancelled:
                raise asyncio.CancelledError


DEFAULT_STDIO_LIMITS = StdioLimits()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Invalid MCP JSON")
        result[key] = value
    return result


def _reject_constant(_: str) -> Any:
    raise ValueError("Invalid MCP JSON")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Invalid MCP JSON")
    return result


def _local_id(value: Any) -> int:
    """Normalize only the SDK's nonnegative local request-ID domain."""
    if type(value) is int and value >= 0:
        return value
    if type(value) is str and value and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError("Invalid MCP response ID")


def _check_depth(text: str) -> None:
    # Check before json.loads, so adversarial nesting never reaches its recursive
    # parser. Braces inside strings and escaped quotes do not affect depth.
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > _JSON_DEPTH:
                raise ValueError("Invalid MCP JSON")
        elif char in "]}":
            depth -= 1


def _decode_frame(frame: bytearray) -> SessionMessage:
    from mcp import types
    from mcp.shared.message import SessionMessage
    from pydantic import TypeAdapter

    text = frame.decode("utf-8", errors="strict")
    _check_depth(text)
    data = json.loads(
        text, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=_finite_float
    )
    # Strict validation also rejects boolean IDs/error codes and coercions.
    if not isinstance(data, dict):
        raise ValueError("Invalid MCP message")
    model: (
        type[types.JSONRPCRequest]
        | type[types.JSONRPCNotification]
        | type[types.JSONRPCResponse]
        | type[types.JSONRPCError]
    )
    if "method" in data:
        if "result" in data or "error" in data:
            raise ValueError("Invalid MCP message")
        model = types.JSONRPCRequest if "id" in data else types.JSONRPCNotification
    else:
        if ("result" in data) == ("error" in data):
            raise ValueError("Invalid MCP message")
        model = types.JSONRPCResponse if "result" in data else types.JSONRPCError
    root = model.model_validate(data, strict=True)
    if isinstance(root, types.JSONRPCRequest):
        TypeAdapter(types.ServerRequest).validate_python(
            root.model_dump(by_alias=True, exclude_none=True), strict=True
        )
    elif isinstance(root, types.JSONRPCNotification):
        TypeAdapter(types.ServerNotification).validate_python(
            root.model_dump(by_alias=True, exclude_none=True), strict=True
        )
    else:
        # SDK local IDs are integers. Its normalization logs raw peer messages
        # for arbitrary string IDs, including Unicode isdigit/int mismatches.
        _local_id(root.id)
    return SessionMessage(root)


def _encode_frame(message: SessionMessage, limit: int) -> bytes:
    from mcp import types
    from mcp.shared.message import SessionMessage
    from pydantic import TypeAdapter

    if not isinstance(message, SessionMessage) or not isinstance(
        message.message,
        (types.JSONRPCRequest, types.JSONRPCNotification, types.JSONRPCResponse, types.JSONRPCError),
    ):
        raise ValueError("Invalid MCP message")
    remaining = limit
    wire_models = (
        types.JSONRPCRequest,
        types.JSONRPCNotification,
        types.JSONRPCResponse,
        types.JSONRPCError,
        types.ErrorData,
    )

    def charge(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("MCP outgoing limit exceeded")

    def measure(value: Any, depth: int = 0) -> None:
        # Preflight exact JSON size before dumps: a large single string token or
        # control-character escaping must not allocate an unbounded encoder chunk.
        if type(value) is str:
            charge(2)
            for char in value:
                code = ord(char)
                if 0xD800 <= code <= 0xDFFF:
                    raise ValueError("Invalid MCP JSON")
                if char in '"\\\b\f\n\r\t':
                    charge(2)
                elif code < 32:
                    charge(6)
                else:
                    charge(1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4)
        elif type(value) in wire_models:
            if depth >= _JSON_DEPTH:
                raise ValueError("Invalid MCP JSON")
            charge(2)
            # Walk the small, known wire models without model_dump's recursive
            # copies of arbitrary-size params/results. None model fields are
            # omitted exactly as with exclude_none=True (dict values are kept).
            fields = (
                (info.serialization_alias or info.alias or name, getattr(value, name))
                for name, info in type(value).model_fields.items()
                if getattr(value, name) is not None
            )
            for index, (key, item) in enumerate(chain(fields, (value.model_extra or {}).items())):
                if index:
                    charge(1)
                if type(key) is not str:
                    raise ValueError("Invalid MCP JSON")
                measure(key, depth + 1)
                charge(1)
                measure(item, depth + 1)
        elif type(value) in (dict, list):
            if depth >= _JSON_DEPTH:
                raise ValueError("Invalid MCP JSON")
            charge(2)
            for index, item in enumerate(value):
                if index:
                    charge(1)
                if type(value) is dict:
                    if type(item) is not str:
                        raise ValueError("Invalid MCP JSON")
                    measure(item, depth + 1)
                    charge(1)
                    measure(value[item], depth + 1)
                else:
                    measure(item, depth + 1)
        elif value is None:
            charge(4)
        elif type(value) is bool:
            charge(4 if value else 5)
        elif type(value) is int:
            charge(len(str(value)))
        elif type(value) is float and math.isfinite(value):
            charge(len(json.dumps(value, allow_nan=False)))
        else:
            raise ValueError("Invalid MCP JSON")

    measure(message.message)
    data = message.message.model_dump(by_alias=True, exclude_none=True, warnings=False)
    remaining = limit
    measure(data)
    TypeAdapter(types.JSONRPCMessage).validate_python(data, strict=True)
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


async def _joined(task: asyncio.Task[Any]) -> tuple[Any, bool]:
    """Join shielded work even after repeated asyncio or AnyIO cancellation."""
    import anyio

    cancelled = False
    with anyio.CancelScope(shield=True):
        while True:
            try:
                return await asyncio.shield(task), cancelled
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
    raise RuntimeError("MCP stdio cleanup failed")  # CancelScope does not suppress this work.


class _Connection:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        transport: Transport,
        read_sender: MemoryObjectSendStream[SessionMessage | Exception],
        write_receiver: MemoryObjectReceiveStream[SessionMessage],
        limits: StdioLimits,
    ) -> None:
        self.process = process
        self.transport = transport
        self.read_sender = read_sender
        self.write_receiver = write_receiver
        self.limits = limits
        self.stdout_bytes = self.stderr_bytes = 0
        # Bound correlation state as well as frames. Runtime serializes calls,
        # while this permits up to queued_frames concurrent SDK requests.
        self.pending_ids: dict[int, str] = {}
        self.workers: list[asyncio.Task[None]] = []
        self.cleanup: asyncio.Task[None] | None = None

    def fail(self, code: str) -> None:
        if self.cleanup is not None:
            return  # EOF caused by our own channel closure is not a new failure.
        if self.transport.failure is None:
            self.transport.failure = code
            self.transport.failed.set()
        self.close()

    def close(self) -> asyncio.Task[None]:
        if self.cleanup is None or (
            self.cleanup.done() and (self.cleanup.cancelled() or self.cleanup.exception() is not None)
        ):
            # Synchronous channel closure wakes both SDK receive waiters and
            # blocked senders, independently of process shutdown deadlines.
            self.read_sender.close()
            self.write_receiver.close()
            self.transport.write_stream.close()
            self.cleanup = asyncio.create_task(self._shutdown(), name="mcp-stdio-cleanup")
        return self.cleanup

    def start(self) -> None:
        self.workers = [
            asyncio.create_task(worker(), name=f"mcp-stdio-{name}")
            for name, worker in (
                ("stdout", self._stdout),
                ("stderr", self._stderr),
                ("stdin", self._stdin),
                ("exit", self._exit),
            )
        ]

    async def _stdout(self) -> None:
        import anyio
        from mcp import types

        assert self.process.stdout is not None
        frame = bytearray()
        try:
            while True:
                chunk = await self.process.stdout.read(
                    min(_CHUNK, self.limits.total_stdout_bytes - self.stdout_bytes + 1)
                )
                if not chunk:
                    self.fail("invalid_stdout" if frame else "connection_closed")
                    return
                self.stdout_bytes += len(chunk)
                if self.stdout_bytes > self.limits.total_stdout_bytes:
                    self.fail("stdout_limit")
                    return
                offset = 0
                while offset < len(chunk):
                    end = chunk.find(b"\n", offset)
                    stop = end if end >= 0 else len(chunk)
                    if len(frame) + stop - offset > self.limits.frame_bytes:
                        self.fail("frame_limit")
                        return
                    frame.extend(chunk[offset:stop])
                    if end < 0:
                        break
                    try:
                        message = _decode_frame(frame)
                        root = message.message
                        if isinstance(root, types.JSONRPCResponse | types.JSONRPCError):
                            response_id = _local_id(root.id)
                            if response_id not in self.pending_ids:
                                raise ValueError("Invalid MCP response ID")
                            method = self.pending_ids.pop(response_id)
                            if isinstance(root, types.JSONRPCResponse):
                                result_types: dict[str, type[BaseModel]] = {
                                    "initialize": types.InitializeResult,
                                    "tools/list": types.ListToolsResult,
                                    "tools/call": types.CallToolResult,
                                    "resources/list": types.ListResourcesResult,
                                    "resources/templates/list": types.ListResourceTemplatesResult,
                                    "resources/read": types.ReadResourceResult,
                                    "ping": types.EmptyResult,
                                }
                                if method == "tools/call":
                                    root.result.setdefault("content", [])
                                result_type = result_types.get(method)
                                if result_type is not None:
                                    result_type.model_validate(root.result, strict=True, by_name=False)
                    except Exception:
                        self.fail("invalid_stdout")
                        return
                    frame.clear()
                    try:
                        self.read_sender.send_nowait(message)
                    except anyio.WouldBlock:
                        self.fail("queue_limit")
                        return
                    offset = end + 1
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail("connection_closed")

    async def _stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            while chunk := await self.process.stderr.read(
                min(_CHUNK, self.limits.stderr_bytes - self.stderr_bytes + 1)
            ):
                self.stderr_bytes += len(chunk)
                # Never store, decode, log, or expose stderr.
                if self.stderr_bytes > self.limits.stderr_bytes:
                    self.fail("stderr_limit")
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail("connection_closed")

    async def _stdin(self) -> None:
        from mcp import types

        assert self.process.stdin is not None
        try:
            async for message in self.write_receiver:
                try:
                    payload = _encode_frame(message, self.limits.outgoing_bytes)
                    root = message.message
                    if isinstance(root, types.JSONRPCRequest):
                        request_id = _local_id(root.id)
                        if (
                            request_id in self.pending_ids
                            or len(self.pending_ids) >= self.limits.queued_frames
                        ):
                            raise ValueError("Invalid MCP outgoing requests")
                        self.pending_ids[request_id] = root.method
                except Exception:
                    self.fail("invalid_outgoing")
                    return
                self.process.stdin.write(payload + b"\n")
                await self.process.stdin.drain()
            self.fail("connection_closed")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail("connection_closed")

    async def _exit(self) -> None:
        # Process.wait() can wait for descendant-held pipes after the leader has
        # exited. Watch the reaped returncode independently to trigger cleanup.
        while self.process.returncode is None:
            await asyncio.sleep(0.01)
        self.fail("connection_closed")

    def _signal_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass

    def _group_alive(self) -> bool:
        try:
            os.killpg(self.process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    @staticmethod
    async def _drain(reader: asyncio.StreamReader, budget: int) -> None:
        while budget > 0:
            chunk = await reader.read(min(_CHUNK, budget))
            if not chunk:
                break
            budget -= len(chunk)

    async def _shutdown(self) -> None:
        for task in self.workers:
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        if self.process.stdin is not None:
            self.process.stdin.close()
        drains = [
            asyncio.create_task(self._drain(reader, max(0, budget)))
            for reader, budget in (
                (self.process.stdout, self.limits.total_stdout_bytes - self.stdout_bytes),
                (self.process.stderr, self.limits.stderr_bytes - self.stderr_bytes),
            )
            if reader is not None
        ]
        shutdown_failed = False
        try:
            self._signal_group(signal.SIGTERM)
            deadline = asyncio.get_running_loop().time() + self.limits.shutdown_timeout
            while self._group_alive() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(min(0.01, max(0, deadline - asyncio.get_running_loop().time())))
        except OSError:
            shutdown_failed = True
        finally:
            try:
                self._signal_group(signal.SIGKILL)
            except OSError:
                shutdown_failed = True
            # asyncio exposes no public read-pipe close API. Closing its owned
            # subprocess transport also releases paused/descendant-held readers.
            subprocess_transport = getattr(self.process, "_transport", None)
            if subprocess_transport is not None:
                try:
                    subprocess_transport.close()
                except Exception:
                    shutdown_failed = True
            for task in drains:
                task.cancel()
            await asyncio.gather(*drains, return_exceptions=True)
            try:
                async with asyncio.timeout(self.limits.shutdown_timeout):
                    await self.process.wait()
                    if self.process.returncode is None:
                        raise RuntimeError("MCP stdio cleanup failed")
                    if self.process.stdin is not None:
                        try:
                            await self.process.stdin.wait_closed()
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                    # wait() can return an already-set returncode before pipe
                    # connection_lost callbacks run. Join those closures too.
                    while subprocess_transport is not None and any(
                        pipe is None or not pipe.disconnected for pipe in subprocess_transport._pipes.values()
                    ):
                        await asyncio.sleep(0.001)
            except Exception:
                shutdown_failed = True
            if shutdown_failed:
                if self.transport.failure is None:
                    self.transport.failure = "shutdown_failed"
                    self.transport.failed.set()
                raise RuntimeError("MCP stdio cleanup failed") from None


@asynccontextmanager
async def bounded_stdio(
    command: str,
    args: tuple[str, ...],
    cwd: Path,
    env: dict[str, str],
    limits: StdioLimits = DEFAULT_STDIO_LIMITS,
) -> AsyncIterator[Transport]:
    """Spawn exactly the supplied argv/environment and own it until context exit.

    Spawn failures raise a generic RuntimeError; established connections report
    generic codes on Transport. Cancellation joins spawn and TERM/KILL cleanup
    before propagating. Requires the optional ``mcp`` extra and POSIX asyncio.
    """
    import anyio
    from mcp.shared.message import SessionMessage

    if (
        type(command) is not str
        or not command
        or "\0" in command
        or type(args) is not tuple
        or any(type(arg) is not str or "\0" in arg for arg in args)
        or not isinstance(cwd, Path)
        or type(env) is not dict
        or any(
            type(key) is not str
            or not key
            or "=" in key
            or "\0" in key
            or type(value) is not str
            or "\0" in value
            for key, value in env.items()
        )
        or type(limits) is not StdioLimits
    ):
        raise ValueError("Invalid MCP stdio parameters")
    read_sender, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](
        limits.queued_frames
    )
    write_stream, write_receiver = anyio.create_memory_object_stream[SessionMessage](limits.queued_frames)
    transport = Transport(read_stream, write_stream)
    process: asyncio.subprocess.Process | None = None
    connection: _Connection | None = None
    # All AnyIO contexts are entered/exited in this caller task, including when
    # cleanup itself needs a separate shielded asyncio task.
    async with read_sender, read_stream, write_stream, write_receiver:
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                command,
                *args,
                cwd=cwd,
                env=dict(env),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=min(_CHUNK, limits.frame_bytes),
            ),
            name="mcp-stdio-spawn",
        )
        try:
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                try:
                    process, _ = await _joined(spawn)
                except Exception:
                    pass
                raise
            except Exception:
                raise RuntimeError("MCP stdio spawn failed") from None
            connection = _Connection(process, transport, read_sender, write_receiver, limits)
            transport._connection = connection
            connection.start()
            yield transport
        finally:
            if process is not None:
                if connection is None:
                    connection = _Connection(process, transport, read_sender, write_receiver, limits)
                    transport._connection = connection
                await transport.aclose()
