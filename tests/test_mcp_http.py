from __future__ import annotations

import asyncio
import json
import logging
import runpy
from pathlib import Path

import httpx
import pytest

pytest.importorskip("mcp")

from pi_agent import Cancellation, ResourceOwner
from pi_agent.mcp import HttpLimits, McpConnection, McpError, McpRuntime, _http, parse_mcp_config

HANDLE = runpy.run_path(str(Path(__file__).parent / "fixtures/mcp_resource_server.py"))["handle"]


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks=(), *, hang=False):
        self.chunks = chunks
        self.hang = hang
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def reply(payload=b"", *, status=200, content_type="application/json", headers=None, stream=None):
    return httpx.Response(
        status, headers={"Content-Type": content_type, **(headers or {})}, stream=stream or Stream([payload])
    )


class Peer:
    def __init__(self, monkeypatch, *, sse=False, mode="normal"):
        self.requests = []
        self.messages = []
        self.clients = []
        self.streams = []
        self.sse = sse
        self.mode = mode
        self.override = None
        self.started = asyncio.Event()
        self.get_stream = None
        self.hang = False

        def factory(timeout):
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(self.handle), trust_env=False, timeout=timeout
            )
            self.clients.append(client)
            return client

        monkeypatch.setattr(_http, "_new_client", factory)

    async def handle(self, request):
        self.requests.append(request)
        if request.method == "DELETE":
            return reply(status=204)
        if request.method == "GET":
            if self.get_stream is not None:
                self.streams.append(self.get_stream)
                return reply(content_type="text/event-stream", stream=self.get_stream)
            return reply(status=405)
        message = json.loads(await request.aread())
        self.messages.append(message)
        method = message.get("method")
        if method == "tools/call":
            self.started.set()
            if self.hang:
                await asyncio.Event().wait()
            if self.override is not None:
                return self.override(message)
        result = HANDLE(message, self.mode)
        if result is None:
            return reply(status=202)
        headers = (
            {
                "mcp-session-id": "synthetic-session-secret",
                "Set-Cookie": "secret-cookie=synthetic-cookie-secret",
            }
            if method == "initialize"
            else {}
        )
        payload = json.dumps(result).encode()
        if self.sse and method == "tools/call":
            token = message.get("params", {}).get("_meta", {}).get("progressToken")
            progress = {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": 1, "total": 2, "message": "progress"},
            }
            body = b"id: ignored\r\nretry: 0\r\n\r\n"
            if token is not None:
                body += b"data: " + json.dumps(progress).encode() + b"\r\n\r\n"
            body += b"data: " + payload + b"\r\n\r\n"
            stream = Stream([body[:5], body[5:]])
            self.streams.append(stream)
            return reply(content_type="text/event-stream", stream=stream)
        stream = Stream([payload])
        self.streams.append(stream)
        return reply(headers=headers, stream=stream)

    def count(self, method):
        return sum(message.get("method") == method for message in self.messages)


def configuration(tmp_path, timeout=2):
    return parse_mcp_config(
        {
            "mcpServers": {
                "fixture": {
                    "type": "http",
                    "url": "https://synthetic.invalid/mcp",
                    "headers": {"Authorization": "${MCP_TOKEN}"},
                    "timeout": timeout,
                    "exposure": "direct",
                }
            }
        },
        cwd=tmp_path,
    )


def connection(tmp_path, **kwargs):
    return McpConnection(
        configuration(tmp_path, kwargs.pop("timeout", 2)).servers[0],
        environment={
            "MCP_TOKEN": "Bearer synthetic-auth-secret",
            "OPENAI_API_KEY": "synthetic-provider-secret",
        },
        **kwargs,
    )


@pytest.mark.parametrize("sse", [False, True])
async def test_http_handshake_progress_headers_and_joined_close(tmp_path, monkeypatch, sse):
    peer = Peer(monkeypatch, sse=sse)
    client = connection(tmp_path)
    updates = []
    try:
        tools = await client.discover()
        assert len(tools) == 4
        result = await tools[0].execute("id", {}, on_update=lambda value: collect(updates, value))
        assert "embedded text" in result.content[0].text
        assert len(updates) == (1 if sse else 0)
        assert peer.count("initialize") == 1
        assert peer.count("tools/call") == 1
        call = next(
            request
            for request in peer.requests
            if request.method == "POST" and b"tools/call" in request.content
        )
        assert call.headers["Mcp-Session-Id"] == "synthetic-session-secret"
        assert call.headers["Mcp-Protocol-Version"] == "2025-11-25"
        assert call.headers["Authorization"] == "Bearer synthetic-auth-secret"
        assert "cookie" not in call.headers
        assert "synthetic-provider-secret" not in str(call.headers)
    finally:
        await client.aclose()
    assert all(item.is_closed for item in peer.clients)
    assert all(not item.cookies for item in peer.clients)
    assert all(item.closed for item in peer.streams)
    assert any(request.method == "DELETE" for request in peer.requests)


async def collect(items, value):
    items.append(value)


@pytest.mark.parametrize(
    "kind",
    [
        "expired",
        "redirect",
        "auth",
        "bad-type",
        "compressed",
        "wrong-id",
        "boolean-id",
        "duplicate-key",
        "nonfinite",
        "bad-result",
        "deep",
        "frame",
        "sse-line",
        "sse-fields",
        "sse-disconnect",
    ],
)
async def test_bad_http_response_is_terminal_without_replay(tmp_path, monkeypatch, kind):
    peer = Peer(monkeypatch)
    client = connection(tmp_path, http_limits=HttpLimits(frame_bytes=1024))
    tools = await client.discover()

    def invalid(message):
        packet = {"jsonrpc": "2.0", "id": message["id"], "result": {"content": []}}
        if kind == "expired":
            return reply(status=404)
        if kind == "redirect":
            return reply(status=307, headers={"Location": "https://must-not-contact.invalid"})
        if kind == "auth":
            return reply(status=401)
        if kind == "bad-type":
            return reply(b"synthetic-private-error", content_type="text/plain")
        if kind == "compressed":
            return reply(b"synthetic-private-error", headers={"Content-Encoding": "gzip"})
        if kind == "wrong-id":
            packet["id"] += 1
        if kind == "boolean-id":
            packet["id"] = True
        if kind == "bad-result":
            packet["result"] = {"content": [{"type": "text", "text": 123}]}
        if kind == "duplicate-key":
            return reply(b'{"jsonrpc":"2.0","id":0,"id":0,"result":{}}')
        if kind == "nonfinite":
            return reply(b'{"jsonrpc":"2.0","id":0,"result":{"bad":NaN}}')
        if kind == "deep":
            return reply(b"[" * 65 + b"0" + b"]" * 65)
        if kind == "frame":
            return reply(b" " * 1025)
        if kind == "sse-line":
            return reply(b":" + b"x" * 1025, content_type="text/event-stream")
        if kind == "sse-fields":
            return reply(b"id: x\n" * 300, content_type="text/event-stream")
        if kind == "sse-disconnect":
            return reply(b"id: secret-event\nretry: 0\n\n", content_type="text/event-stream")
        return reply(json.dumps(packet).encode())

    peer.override = invalid
    try:
        with pytest.raises(McpError, match="outcome unknown"):
            await tools[0].execute("id", {})
        assert peer.count("tools/call") == 1
        with pytest.raises(McpError, match="unavailable"):
            await tools[0].execute("id", {})
        assert peer.count("initialize") == 1
    finally:
        await client.aclose()
    assert all(item.is_closed for item in peer.clients)


@pytest.mark.parametrize("kind", ["cancel-task", "cancel-token", "timeout", "close"])
async def test_interruption_joins_request_and_client(tmp_path, monkeypatch, kind):
    peer = Peer(monkeypatch)
    client = connection(tmp_path, timeout=0.1 if kind == "timeout" else 2)
    tools = await client.discover()
    peer.hang = True
    cancellation = Cancellation()
    task = asyncio.create_task(tools[0].execute("id", {}, cancellation=cancellation))
    await peer.started.wait()
    if kind == "cancel-task":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif kind == "cancel-token":
        cancellation.abort()
        assert (await task).is_error
    else:
        if kind == "close":
            await client.aclose()
        with pytest.raises(McpError):
            await task
    await client.aclose()
    assert peer.count("tools/call") == 1
    assert all(item.is_closed for item in peer.clients)
    assert not any(
        task.get_name().startswith("mcp-http-")
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
    )


async def test_get_stream_closed_with_connection(tmp_path, monkeypatch):
    peer = Peer(monkeypatch)
    peer.get_stream = Stream([b": keepalive\n\n"], hang=True)
    client = connection(tmp_path)
    await client.discover()
    await asyncio.sleep(0)
    await client.aclose()
    assert peer.get_stream.closed


async def test_explicit_stage_reconnect_publishes_fresh_http_binding(tmp_path, monkeypatch):
    peer = Peer(monkeypatch)
    runtime = McpRuntime(configuration(tmp_path), environment={"MCP_TOKEN": "synthetic-auth"})
    owner = ResourceOwner()
    tools = await runtime.prepare_tools(owner)
    old = tools[0]
    stage = await runtime.stage(["fixture/echo"], servers=["fixture"])
    runtime.validate_stage(stage, stage.tools)
    runtime.publish(stage, stage.tools)
    await runtime.retire_previous(stage)
    assert runtime.tools[0] is not old
    with pytest.raises(McpError, match="unavailable"):
        await old.execute("id", {})
    assert not (await runtime.tools[0].execute("id", {})).is_error
    await owner.aclose()
    assert peer.count("initialize") == 2
    assert all(item.is_closed for item in peer.clients)


async def test_http_debug_logs_do_not_expose_transport_secrets(tmp_path, monkeypatch, caplog):
    peer = Peer(monkeypatch)
    client = connection(tmp_path)
    caplog.set_level(logging.DEBUG)
    tools = await client.discover()
    await tools[0].execute("id", {})
    await client.aclose()
    assert peer.requests
    assert "synthetic.invalid" not in caplog.text
    assert "synthetic-auth-secret" not in caplog.text
    assert "synthetic-session-secret" not in caplog.text
    assert "synthetic-cookie-secret" not in caplog.text


@pytest.mark.parametrize(
    "field", ["frame_bytes", "outgoing_bytes", "total_received_bytes", "queued_frames", "shutdown_timeout"]
)
@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan"), 32 * 1024 * 1024])
def test_http_limits_cannot_disable_or_raise_caps(field, value):
    with pytest.raises(ValueError):
        HttpLimits(**{field: value})


async def test_http_lifetime_byte_budget_is_shared_between_requests(tmp_path, monkeypatch):
    peer = Peer(monkeypatch)
    client = connection(tmp_path, http_limits=HttpLimits(total_received_bytes=1024))
    tools = await client.discover()
    try:
        succeeded = 0
        for _ in range(10):
            try:
                await tools[0].execute("id", {})
            except McpError:
                break
            succeeded += 1
        else:
            pytest.fail("Lifetime byte budget did not disconnect")
        assert 0 < succeeded < 10
        assert peer.count("tools/call") == succeeded + 1
    finally:
        await client.aclose()


async def test_http_queue_overflow_disconnects_instead_of_growing(tmp_path, monkeypatch):
    peer = Peer(monkeypatch)
    client = connection(tmp_path, http_limits=HttpLimits(queued_frames=1))
    tools = await client.discover()

    def flooded(message):
        body = b""
        token = message["params"]["_meta"]["progressToken"]
        for index in range(10):
            progress = {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": index},
            }
            body += b"data: " + json.dumps(progress).encode() + b"\n\n"
        body += b"data: " + json.dumps(HANDLE(message)).encode() + b"\n\n"
        return reply(body, content_type="text/event-stream")

    peer.override = flooded
    try:
        with pytest.raises(McpError):
            await tools[0].execute("id", {}, on_update=lambda value: collect([], value))
        assert peer.count("tools/call") == 1
    finally:
        await client.aclose()


async def test_cancel_initialize_joins_all_http_work(tmp_path, monkeypatch):
    peer = Peer(monkeypatch)
    original = peer.handle
    started = asyncio.Event()

    async def hang_initialize(request):
        if request.method == "POST" and b'"initialize"' in request.content:
            started.set()
            await asyncio.Event().wait()
        return await original(request)

    peer.handle = hang_initialize
    client = connection(tmp_path)
    task = asyncio.create_task(client.connect())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(item.is_closed for item in peer.clients)
    assert not any(
        task.get_name().startswith("mcp-http-")
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
    )


async def test_http_cleanup_failure_remains_retryable(tmp_path, monkeypatch):
    peer = Peer(monkeypatch)
    client = connection(tmp_path)
    await client.discover()
    instance = peer.clients[0]
    original = instance.aclose
    calls = 0

    async def fail_twice():
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("synthetic-private-close-error")
        await original()

    monkeypatch.setattr(instance, "aclose", fail_twice)
    with pytest.raises(McpError, match="cleanup failed") as error:
        await client.aclose()
    assert "synthetic-private" not in str(error.value)
    assert not instance.is_closed
    await client.aclose()
    assert instance.is_closed


async def test_sse_final_response_is_delivered_before_stream_eof(tmp_path, monkeypatch):
    peer = Peer(monkeypatch)
    client = connection(tmp_path, timeout=0.5)
    tools = await client.discover()
    stream = None

    def open_stream(message):
        nonlocal stream
        stream = Stream([b"data: " + json.dumps(HANDLE(message)).encode() + b"\n\n"], hang=True)
        return reply(content_type="text/event-stream", stream=stream)

    peer.override = open_stream
    try:
        result = await tools[0].execute("id", {})
        assert "embedded text" in result.content[0].text
        assert stream.closed
    finally:
        await client.aclose()


async def test_get_server_ping_delivered_without_buffering_or_eof(tmp_path, monkeypatch):
    peer = Peer(monkeypatch)
    peer.get_stream = Stream([b'data: {"jsonrpc":"2.0","id":"peer-ping","method":"ping"}\n\n'], hang=True)
    client = connection(tmp_path)
    try:
        await client.discover()
        async with asyncio.timeout(0.5):
            while not any(message.get("id") == "peer-ping" for message in peer.messages):
                await asyncio.sleep(0)
        assert next(message for message in peer.messages if message.get("id") == "peer-ping")["result"] == {}
    finally:
        await client.aclose()
    assert peer.get_stream.closed
