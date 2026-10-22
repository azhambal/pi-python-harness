"""Stdlib-only paginated catalog/load fixture; no provider or filesystem access."""

import json
import sys


def main():
    count = int(sys.argv[1])
    if not 1 <= count <= 1024:
        raise ValueError("Catalog fixture count must be 1..1024")
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "load-fixture", "version": "1"},
            }
        elif method == "tools/list":
            start = int(request.get("params", {}).get("cursor") or 0)
            end = min(count, start + 64)
            result = {
                "tools": [
                    {
                        "name": f"task_{number}",
                        "description": "Load fixture operation",
                        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                    }
                    for number in range(start, end)
                ]
            }
            if end < count:
                result["nextCursor"] = str(end)
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": "\x01" * 60000}]}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)


if __name__ == "__main__":
    main()
