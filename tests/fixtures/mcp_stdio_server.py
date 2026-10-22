"""Local JSON-RPC fixture; deliberately independent of the MCP SDK."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path


async def serve(directory: Path, mode: str) -> None:
    (directory / "pid").write_text(str(os.getpid()))
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    calls: dict[int | str, asyncio.Task] = {}

    def send(data):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", **data}) + "\n")
        sys.stdout.flush()

    def log(event):
        with (directory / "events").open("a") as stream:
            stream.write(event + "\n")

    async def call(request):
        params, request_id = request["params"], request["id"]
        name = params["name"]
        log("call:" + name)
        token = params.get("_meta", {}).get("progressToken")
        if token is not None:
            send(
                {
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": token,
                        "progress": 1,
                        "total": 2,
                        "message": "fixture progress",
                    },
                }
            )
        if name == "slow":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                log("handler-cancelled")
                raise
        if name == "crash":
            os._exit(7)
        if name == "error":
            send(
                {
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "fixture error"}], "isError": True},
                }
            )
            return
        value = {
            "echo": params.get("arguments", {}),
            "provider_key_present": "OPENAI_API_KEY" in os.environ,
        }
        send(
            {
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": "fixture result"}],
                    "structuredContent": value,
                    "isError": False,
                },
            }
        )

    try:
        while line := await reader.readline():
            request = json.loads(line)
            method = request.get("method")
            if method == "initialize":
                log("initialize")
                if mode == "initialize-hang":
                    continue
                if mode == "exit":
                    return
                send(
                    {
                        "id": request["id"],
                        "result": {
                            "protocolVersion": "2025-11-25",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fixture", "version": "1"},
                        },
                    }
                )
            elif method == "tools/list":
                cursor = request.get("params", {}).get("cursor")
                names = ["echo", "error"] if not cursor else ["slow", "crash"]
                result = {
                    "tools": [
                        {"name": name, "description": "fixture " + name, "inputSchema": {"type": "object"}}
                        for name in names
                    ]
                }
                if not cursor:
                    result["nextCursor"] = "second"
                send({"id": request["id"], "result": result})
            elif method == "tools/call":
                task = asyncio.create_task(call(request))
                calls[request["id"]] = task
            elif method == "notifications/cancelled":
                log("peer-cancelled")
                task = calls.get(request["params"]["requestId"])
                if task is not None:
                    task.cancel()
            elif method == "ping":
                send({"id": request["id"], "result": {}})
        if mode == "stubborn":
            await asyncio.Event().wait()
    finally:
        transport.close()
        for task in calls.values():
            task.cancel()
        await asyncio.gather(*calls.values(), return_exceptions=True)
        log("exit")


if __name__ == "__main__":
    asyncio.run(serve(Path(sys.argv[1]), sys.argv[2]))
