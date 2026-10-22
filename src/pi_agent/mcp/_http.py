"""Bounded Streamable HTTP channels for the pinned SDK, without request replay.

POST responses support JSON and SSE. Optional GET receives notifications; a 405
disables it. Disconnect/expiry is terminal, and SSE id/retry never trigger replay.
Only explicitly configured headers leave the host; proxy/netrc/cookie inheritance,
redirects, compression, OAuth and legacy SSE are deliberately unsupported.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from http.cookiejar import Cookie, DefaultCookiePolicy
from typing import TYPE_CHECKING, Any

import httpx

from ._stdio import _decode_frame, _encode_frame, _joined, _local_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from mcp.shared.message import SessionMessage
    from pydantic import BaseModel

_OWN_HTTP: ContextVar[bool] = ContextVar("pi_mcp_http_logging", default=False)
_LOGGERS = (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)


class _SafeLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _OWN_HTTP.get():
            record.msg = "MCP HTTP transport event (details omitted)"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


class _NoCookies(DefaultCookiePolicy):
    def set_ok(self, cookie: Cookie, request: Any) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class HttpLimits:
    frame_bytes: int = 1024 * 1024
    outgoing_bytes: int = 1024 * 1024
    total_received_bytes: int = 16 * 1024 * 1024
    queued_frames: int = 16
    shutdown_timeout: float = 0.5

    def __post_init__(self) -> None:
        for name, cap in (
            ("frame_bytes", 1024 * 1024),
            ("outgoing_bytes", 1024 * 1024),
            ("total_received_bytes", 16 * 1024 * 1024),
            ("queued_frames", 16),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= cap:
                raise ValueError("Invalid MCP HTTP limit")
        if type(self.shutdown_timeout) not in (int, float) or not 0 < self.shutdown_timeout <= 0.5:
            raise ValueError("Invalid MCP HTTP shutdown limit")


DEFAULT_HTTP_LIMITS = HttpLimits()


@dataclass(slots=True)
class HttpTransport:
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    failed: asyncio.Event = field(default_factory=asyncio.Event)
    failure: str | None = None
    _connection: _HttpConnection | None = field(default=None, init=False, repr=False)

    async def aclose(self) -> None:
        if self._connection is not None:
            try:
                _, cancelled = await _joined(self._connection.close())
            except Exception:
                raise RuntimeError("MCP HTTP cleanup failed; close may be retried") from None
            if cancelled:
                raise asyncio.CancelledError


def _new_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        trust_env=False,
        follow_redirects=False,
        timeout=timeout,
        limits=httpx.Limits(max_connections=17, max_keepalive_connections=17),
    )


class _HttpConnection:
    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        timeout: float,
        limits: HttpLimits,
        transport: HttpTransport,
        sender: MemoryObjectSendStream[SessionMessage | Exception],
        receiver: MemoryObjectReceiveStream[SessionMessage],
    ) -> None:
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self.limits = limits
        self.transport = transport
        self.sender = sender
        self.receiver = receiver
        self.client = _new_client(timeout)
        # Reject ingestion too: repeatedly returning distinct Set-Cookie headers
        # must not accumulate an unbounded jar even though outgoing Cookie is stripped.
        self.client.cookies.jar.set_policy(_NoCookies())
        self.session_id: str | None = None
        self.protocol: str | None = None
        self.total_bytes = 0
        self.pending: dict[int, str] = {}
        self.workers: set[asyncio.Task[None]] = set()
        self.cleanup: asyncio.Task[None] | None = None
        self.get_started = False
        self.log_filter = _SafeLogs()
        for name in _LOGGERS:
            logging.getLogger(name).addFilter(self.log_filter)

    def start(self) -> None:
        self._start(self._writer())

    def _start(self, operation: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(operation, name="mcp-http-worker")
        self.workers.add(task)
        task.add_done_callback(self.workers.discard)

    def fail(self) -> None:
        if self.cleanup is None:
            self.transport.failure = "http_connection_failed"
            self.transport.failed.set()
            self.close()

    def close(self) -> asyncio.Task[None]:
        if self.cleanup is None or (
            self.cleanup.done() and (self.cleanup.cancelled() or self.cleanup.exception() is not None)
        ):
            self.sender.close()
            self.receiver.close()
            self.transport.write_stream.close()
            self.cleanup = asyncio.create_task(self._shutdown(), name="mcp-http-cleanup")
        return self.cleanup

    def _headers(self) -> dict[str, str]:
        headers = {
            **self.headers,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        if self.session_id is not None:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol is not None:
            headers["Mcp-Protocol-Version"] = self.protocol
        return headers

    async def _writer(self) -> None:
        from mcp import types

        marker = _OWN_HTTP.set(True)
        try:
            async for message in self.receiver:
                payload = _encode_frame(message, self.limits.outgoing_bytes)
                root = message.message
                request_id = _local_id(root.id) if isinstance(root, types.JSONRPCRequest) else None
                if len(self.workers) >= self.limits.queued_frames + 2:
                    raise ValueError
                if request_id is not None:
                    assert isinstance(root, types.JSONRPCRequest)
                    if request_id in self.pending or len(self.pending) >= self.limits.queued_frames:
                        raise ValueError
                    self.pending[request_id] = root.method
                if request_id is None:
                    await self._post(payload, None, getattr(root, "method", None))
                else:
                    assert isinstance(root, types.JSONRPCRequest)
                    self._start(self._post(payload, request_id, root.method))
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail()
        finally:
            _OWN_HTTP.reset(marker)

    async def _chunks(self, response: httpx.Response) -> AsyncIterator[bytes]:
        if response.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
            raise ValueError
        # A chunk_size would make httpx aggregate small SSE events until 8192
        # bytes or EOF, delaying progress and completed responses on open streams.
        # Pinned httpcore HTTP/1.1 reads at most 64 KiB per network read.
        async for chunk in response.aiter_raw():
            self.total_bytes += len(chunk)
            if self.total_bytes > self.limits.total_received_bytes:
                raise ValueError
            yield chunk

    def _deliver(self, payload: bytes | bytearray, expected: int | None) -> bool:
        from mcp import types

        if len(payload) > self.limits.frame_bytes:
            raise ValueError
        message = _decode_frame(bytearray(payload))
        root = message.message
        complete = isinstance(root, (types.JSONRPCResponse, types.JSONRPCError))
        if isinstance(root, (types.JSONRPCResponse, types.JSONRPCError)):
            response_id = _local_id(root.id)
            if response_id != expected or response_id not in self.pending:
                raise ValueError
            method = self.pending.pop(response_id)
            if isinstance(root, types.JSONRPCResponse):
                result_types: dict[str, type[BaseModel]] = {
                    "initialize": types.InitializeResult,
                    "tools/list": types.ListToolsResult,
                    "tools/call": types.CallToolResult,
                    "ping": types.EmptyResult,
                    "resources/list": types.ListResourcesResult,
                    "resources/templates/list": types.ListResourceTemplatesResult,
                    "resources/read": types.ReadResourceResult,
                }
                if method == "tools/call":
                    root.result.setdefault("content", [])
                result_type = result_types.get(method)
                if result_type is not None:
                    result_type.model_validate(root.result, strict=True, by_name=False)
                if method == "initialize":
                    protocol = root.result.get("protocolVersion")
                    if (
                        not isinstance(protocol, str)
                        or not protocol.isascii()
                        or len(protocol) > 128
                        or any(ord(c) < 33 or ord(c) > 126 for c in protocol)
                    ):
                        raise ValueError
                    self.protocol = protocol
        self.sender.send_nowait(message)
        return complete

    async def _body(self, response: httpx.Response, expected: int | None) -> None:
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type == "application/json" and expected is not None:
            payload = bytearray()
            async for chunk in self._chunks(response):
                if len(payload) + len(chunk) > self.limits.frame_bytes:
                    raise ValueError
                payload.extend(chunk)
            if not self._deliver(payload, expected):
                raise ValueError
            return
        if content_type != "text/event-stream":
            raise ValueError
        # Incremental bytes, including non-data SSE fields/comments. Neither an
        # endless line nor endless id/retry/comments may evade byte/deadline caps.
        line = bytearray()
        data: list[bytes] = []
        event = b"message"
        event_bytes = 0
        after_cr = False
        first_line = True
        async for chunk in self._chunks(response):
            for byte in chunk:
                if byte == 10 and after_cr:
                    after_cr = False
                    continue
                after_cr = byte == 13
                if byte not in (10, 13):
                    line.append(byte)
                    event_bytes += 1
                    if event_bytes > self.limits.frame_bytes:
                        raise ValueError
                    continue
                value = bytes(line)
                line.clear()
                if first_line:
                    value = value.removeprefix(b"\xef\xbb\xbf")
                    first_line = False
                value.decode("utf-8", errors="strict")
                if not value:
                    if data and event in (b"", b"message") and self._deliver(b"\n".join(data), expected):
                        return
                    data.clear()
                    event = b"message"
                    event_bytes = 0
                elif value.startswith(b"data:"):
                    data.append(value[5:].removeprefix(b" "))
                elif value.startswith(b"event:"):
                    event = value[6:].removeprefix(b" ")
        # GET ending and POST ending before its final response are terminal.
        raise ValueError

    async def _post(self, payload: bytes, request_id: int | None, method: str | None) -> None:
        try:
            async with asyncio.timeout(self.timeout):
                # httpx's cookie jar is deliberately never used for auth or routing.
                request = self.client.build_request(
                    "POST", self.url, content=payload, headers=self._headers()
                )
                request.headers.pop("cookie", None)
                response = await self.client.send(request, stream=True)
                try:
                    if request_id is None:
                        if response.status_code not in (200, 202, 204):
                            raise ValueError
                        if method == "notifications/initialized" and not self.get_started:
                            self.get_started = True
                            self._start(self._get())
                        return
                    if response.status_code != 200:
                        raise ValueError
                    if method == "initialize":
                        session_id = response.headers.get("mcp-session-id")
                        if session_id is not None and (
                            not session_id
                            or len(session_id) > 256
                            or not session_id.isascii()
                            or any(ord(c) < 33 or ord(c) > 126 for c in session_id)
                        ):
                            raise ValueError
                        self.session_id = session_id
                    await self._body(response, request_id)
                finally:
                    await response.aclose()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail()

    async def _get(self) -> None:
        try:
            request = self.client.build_request("GET", self.url, headers=self._headers())
            request.headers.pop("cookie", None)
            response = await self.client.send(request, stream=True)
            try:
                if response.status_code == 405:
                    return
                if response.status_code != 200:
                    raise ValueError
                await self._body(response, None)
            finally:
                await response.aclose()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail()

    async def _shutdown(self) -> None:
        marker = _OWN_HTTP.set(True)
        try:
            tasks = tuple(self.workers)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.session_id is not None and not self.client.is_closed:
                # Session deletion is best effort; it never retries tool calls.
                try:
                    async with asyncio.timeout(self.limits.shutdown_timeout):
                        request = self.client.build_request("DELETE", self.url, headers=self._headers())
                        request.headers.pop("cookie", None)
                        response = await self.client.send(request, stream=True)
                        await response.aclose()
                except Exception:
                    pass
            async with asyncio.timeout(self.limits.shutdown_timeout):
                await self.client.aclose()
            self.pending.clear()
            self.headers.clear()
            self.session_id = None
            for name in _LOGGERS:
                logging.getLogger(name).removeFilter(self.log_filter)
        finally:
            _OWN_HTTP.reset(marker)


@asynccontextmanager
async def bounded_http(
    url: str, headers: dict[str, str], timeout: float, limits: HttpLimits = DEFAULT_HTTP_LIMITS
) -> AsyncIterator[HttpTransport]:
    import anyio
    from mcp.shared.message import SessionMessage

    sender, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](limits.queued_frames)
    write_stream, receiver = anyio.create_memory_object_stream[SessionMessage](limits.queued_frames)
    transport = HttpTransport(read_stream, write_stream)
    async with sender, read_stream, write_stream, receiver:
        connection = _HttpConnection(url, dict(headers), timeout, limits, transport, sender, receiver)
        transport._connection = connection
        connection.start()
        try:
            yield transport
        finally:
            await transport.aclose()
