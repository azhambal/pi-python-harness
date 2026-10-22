"""Independent offline MCP server for host/tool pipeline integration checks."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


async def serve(directory: Path, mode: str) -> None:
    (directory / "pid").write_text(str(os.getpid()))
    variant_path = directory / "variant"
    variant = variant_path.read_text().strip() if variant_path.exists() else ""
    if variant == "hang":
        mode = "initialize-hang"
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    calls = []

    def log(event):
        with (directory / "events").open("a") as stream:
            stream.write(event + "\n")

    def send(value):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", **value}) + "\n")
        sys.stdout.flush()

    async def call(request):
        params = request["params"]
        name = params["name"]
        log("call:" + name)
        token = params.get("_meta", {}).get("progressToken")
        if token is not None:
            for number in range(1, 6):
                send(
                    {
                        "method": "notifications/progress",
                        "params": {
                            "progressToken": token,
                            "progress": number,
                            "total": 5,
                            "message": "\x00" * 1200 if mode == "progress-large" else "working",
                        },
                    }
                )
                await asyncio.sleep(0.01)
        if mode == "custom":
            send({"id": request["id"], "result": json.loads((directory / "result.json").read_text())})
            return
        if name == "slow":
            try:
                await asyncio.Event().wait()
            finally:
                log("handler-stopped")
        if name == "crash":
            os._exit(7)
        if name == "error":
            result = {"content": [{"type": "text", "text": "tool reported error"}], "isError": True}
        elif name == "binary":
            result = {"content": [{"type": "image", "mimeType": "image/png", "data": "c2VudGluZWw="}]}
        elif name == "output":
            result = {"content": [{"type": "text", "text": "output"}], "structuredContent": {"count": "bad"}}
        elif name == "large":
            result = {
                "content": [{"type": "text", "text": '猫\\"' * 10_000}],
                "structuredContent": {"large": "x" * 20_000},
            }
        elif name == "rpc-error":
            send({"id": request["id"], "error": {"code": -32603, "message": "synthetic-error-secret"}})
            return
        else:
            result = {
                "content": [{"type": "text", "text": "echo result"}],
                "structuredContent": {
                    "echo": params.get("arguments", {}),
                    "provider_key_present": "OPENAI_API_KEY" in os.environ,
                },
                "_meta": {"private": "synthetic-meta-secret"},
            }
            if mode == "structured-only":
                result["content"] = []
        send({"id": request["id"], "result": result})

    try:
        while line := await reader.readline():
            request = json.loads(line)
            method = request.get("method")
            if method == "initialize":
                log("initialize")
                if mode == "initialize-hang":
                    continue
                capabilities = {} if mode == "no-tools" else {"tools": {}}
                send(
                    {
                        "id": request["id"],
                        "result": {
                            "protocolVersion": "2025-11-25",
                            "capabilities": capabilities,
                            "serverInfo": {"name": "fixture", "version": "1"},
                        },
                    }
                )
            elif method == "tools/list":
                log("list")
                cursor = request.get("params", {}).get("cursor")
                names = (
                    ["echo", "error", "output", "binary", "large", "rpc-error"]
                    if cursor is None
                    else ["slow", "crash", "a-b", "a_b"]
                )
                if mode == "duplicate" and cursor is not None:
                    names = ["echo"]
                tools = [
                    {
                        "name": name,
                        "description": "fixture " + name,
                        "inputSchema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "additionalProperties": False,
                        },
                        "annotations": {"readOnlyHint": True, "idempotentHint": True},
                    }
                    for name in names
                ]
                if mode == "bad-schema":
                    tools[0]["inputSchema"] = {"type": "object", "properties": 7}
                if mode == "schema-ref":
                    tools[0]["inputSchema"]["properties"]["message"] = {"$ref": "https://invalid.test/schema"}
                if mode == "missing-type":
                    tools[0]["inputSchema"].pop("type")
                for tool in tools:
                    if tool["name"] == "output":
                        tool["outputSchema"] = {
                            "type": "object",
                            "properties": {"count": {"type": "integer"}},
                            "required": ["count"],
                        }
                    if tool["name"] == "echo":
                        if variant == "input":
                            tool["inputSchema"]["properties"]["count"] = {"type": "integer"}
                        elif variant == "output":
                            tool["outputSchema"] = {"type": "object"}
                        elif variant == "description":
                            tool["description"] = "changed echo description"
                if variant == "removed":
                    tools = [tool for tool in tools if tool["name"] != "echo"]
                if variant == "extra" and cursor is not None:
                    tools.append(
                        {"name": "new", "description": "new fixture tool", "inputSchema": {"type": "object"}}
                    )
                if variant == "reversed":
                    tools.reverse()
                if mode == "custom":
                    tools = [json.loads((directory / "descriptor.json").read_text())]
                result = {"tools": tools}
                if mode != "custom" and (cursor is None or mode == "cursor-loop"):
                    result["nextCursor"] = "second"
                send({"id": request["id"], "result": result})
            elif method == "tools/call":
                calls.append(asyncio.create_task(call(request)))
            elif method == "ping":
                send({"id": request["id"], "result": {}})
    finally:
        transport.close()
        for task in calls:
            task.cancel()
        await asyncio.gather(*calls, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(serve(Path(sys.argv[1]), sys.argv[2]))
