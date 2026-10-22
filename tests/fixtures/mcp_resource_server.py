"""Independent stdlib-only protocol fixture, shared with HTTP mock tests."""

import json
import os
import sys
from pathlib import Path


def handle(message, mode="normal"):
    method = message.get("method")
    params = message.get("params", {})
    if "id" not in message:
        return None
    if method == "initialize":
        capabilities = {"resources": {}}
        if mode == "no-resources":
            capabilities = {}
        if mode != "resources-only":
            capabilities["tools"] = {}
        result = {
            "protocolVersion": "2025-11-25",
            "capabilities": capabilities,
            "serverInfo": {"name": "resources-fixture", "version": "1"},
            "instructions": "synthetic-instructions-not-auto-adopted",
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "echo",
                    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                }
            ]
        }
        if mode == "reserved-tool":
            result["tools"][0]["name"] = "@pi:resources/read"
    elif method == "tools/call":
        result = {
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///sample",
                        "text": "embedded text",
                        "_meta": {"private": "synthetic-private-meta"},
                    },
                },
                {"type": "resource_link", "uri": "file:///sample", "name": "sample"},
            ]
        }
    elif method == "resources/list":
        uri = "file:///second" if params.get("cursor") else "file:///sample"
        result = {
            "resources": [
                {
                    "name": "sample",
                    "uri": uri,
                    "mimeType": "text/plain",
                    "_meta": {"private": "synthetic-private-meta"},
                    "privateExtension": "synthetic-private-extension",
                }
            ]
        }
        if not params.get("cursor"):
            result["nextCursor"] = "page-2"
        if mode == "large-page":
            result["resources"][0]["description"] = "\x01" * 10000
        if mode == "bad-cursor":
            result["nextCursor"] = "\ud800"
    elif method == "resources/templates/list":
        result = {
            "resourceTemplates": [
                {"name": "by-name", "uriTemplate": "file:///{name}", "mimeType": "text/plain"}
            ]
        }
    elif method == "resources/read":
        uri = params["uri"]
        if uri.endswith("missing"):
            return {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32002, "message": "synthetic-private-error"},
            }
        if uri.endswith("binary"):
            block = {
                "uri": uri,
                "blob": "c3ludGhldGljLWJpbmFyeS1zZWNyZXQ=",
                "mimeType": "application/octet-stream",
            }
        else:
            block = {
                "uri": uri,
                "text": '{"answer": 42}' if uri.endswith("json") else "resource text",
                "mimeType": "application/json" if uri.endswith("json") else "text/plain",
            }
        if uri.endswith("large"):
            block["text"] = "\x01" * 30000
        block["_meta"] = {"private": "synthetic-private-meta"}
        result = {"contents": [block], "_meta": {"private": "synthetic-private-meta"}}
    else:
        result = {}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
    (directory / "pid").write_text(str(os.getpid()))
    for line in sys.stdin:
        message = json.loads(line)
        with (directory / "events").open("a") as log:
            log.write(message.get("method", "response") + "\n")
        response = handle(message, mode)
        if response is not None:
            print(json.dumps(response), flush=True)
