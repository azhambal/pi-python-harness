from __future__ import annotations

import asyncio
import json
import logging

import pytest

pytest.importorskip("mcp")

from pi_agent.mcp import McpConnection, McpError, parse_mcp_config
from test_mcp_http import HANDLE
from test_mcp_resources import choose


class SocketPeer:
    def __init__(self, sse):
        self.sse = sse
        self.messages = []
        self.methods = []
        self.tasks = set()
        self.writers = set()
        self.errors = []
        self.expired = False
        self.server = None

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/mcp"
        return self

    async def __aexit__(self, *args):
        self.server.close()
        await self.server.wait_closed()
        for writer in tuple(self.writers):
            writer.close()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        assert not self.errors

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        self.writers.add(writer)
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            lines = raw.decode("ascii").split("\r\n")
            method = lines[0].split(" ")[0]
            headers = {
                name.lower(): value.strip()
                for name, value in (line.split(":", 1) for line in lines[1:] if line)
            }
            self.methods.append(method)
            assert headers["authorization"] == "Bearer synthetic-auth-secret"
            assert "cookie" not in headers
            payload = await reader.readexactly(int(headers.get("content-length", "0")))
            response_headers = ""
            content_type = "application/json"
            body = b""
            if method == "GET":
                status = 405
            elif method == "DELETE":
                status = 204
            else:
                message = json.loads(payload)
                self.messages.append(message)
                result = HANDLE(message)
                status = 202 if result is None else 200
                if self.expired and message.get("method") != "initialize":
                    status = 404
                    result = None
                if result is not None:
                    body = json.dumps(result).encode()
                    if message["method"] == "initialize":
                        response_headers = "Mcp-Session-Id: synthetic-session-secret\r\nSet-Cookie: secret=synthetic-cookie-secret\r\n"
                    else:
                        assert headers["mcp-session-id"] == "synthetic-session-secret"
                        assert headers["mcp-protocol-version"] == "2025-11-25"
                    if self.sse and message["method"] != "initialize":
                        content_type = "text/event-stream"
                        body = b"\xef\xbb\xbfevent:\rdata: " + body + b"\r\r"
            writer.write(
                f"HTTP/1.1 {status} Fixture\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n{response_headers}\r\n".encode()
            )
            await writer.drain()
            # Force chunked arrival across UTF-8/SSE delimiters on actual TCP.
            writer.write(body[:5])
            await writer.drain()
            await asyncio.sleep(0)
            writer.write(body[5:])
            await writer.drain()
        except (asyncio.CancelledError, ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as error:
            self.errors.append(type(error).__name__)
        finally:
            writer.close()
            await writer.wait_closed()
            self.tasks.discard(task)
            self.writers.discard(writer)


@pytest.mark.parametrize("sse", [False, True])
async def test_real_http_socket_resources_expiry_cleanup_and_debug_redaction(tmp_path, caplog, sse):
    caplog.set_level(logging.DEBUG)
    async with SocketPeer(sse) as peer:
        settings = parse_mcp_config(
            {
                "mcpServers": {
                    "fixture": {
                        "type": "streamable-http",
                        "url": peer.url,
                        "headers": {"Authorization": "${MCP_AUTH}"},
                        "exposure": "direct",
                        "timeout": 2,
                    }
                }
            },
            cwd=tmp_path,
        )
        client = McpConnection(settings.servers[0], environment={"MCP_AUTH": "Bearer synthetic-auth-secret"})
        try:
            tools = await client.discover()
            result = await choose(tools, "read").execute("id", {"uri": "file:///sample"})
            assert "resource text" in result.content[0].text
            peer.expired = True
            with pytest.raises(McpError):
                await tools[0].execute("id", {})
        finally:
            await client.aclose()
        assert peer.methods.count("DELETE") == 1
        assert sum(message.get("method") == "tools/call" for message in peer.messages) == 1
        assert not any(
            task.get_name().startswith("mcp-http-")
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        )
    for secret in ("synthetic-auth-secret", "synthetic-session-secret", "synthetic-cookie-secret", peer.url):
        assert secret not in caplog.text
