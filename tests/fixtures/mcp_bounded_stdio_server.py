"""Hostile stdlib-only transport fixture; never reads host credential values."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path


def main() -> None:
    directory, mode, payload = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    (directory / "pid").write_text(str(os.getpid()))
    if mode in ("stubborn", "no-read"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode in ("pipe-holder", "pipe-holder-exit"):
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            (directory / "child-pid").write_text(str(os.getpid()))
            while True:
                time.sleep(1)
        while not (directory / "child-pid").exists():
            time.sleep(0.001)
        if mode == "pipe-holder-exit":
            return
    (directory / "ready").touch()
    if mode == "raw":
        os.write(1, bytes.fromhex(payload))
    elif mode == "response":
        sys.stdin.buffer.readline()
        os.write(1, bytes.fromhex(payload))
    elif mode in ("split", "split-response"):
        if mode == "split-response":
            sys.stdin.buffer.readline()
        for part in bytes.fromhex(payload).split(b"|"):
            os.write(1, part)
            time.sleep(0.02)
    elif mode == "stderr":
        for _ in range(int(payload)):
            os.write(2, b"SECRET_STDERR_SENTINEL\n")
    elif mode == "partial-eof":
        os.write(1, bytes.fromhex(payload))
        return
    elif mode == "env":
        request = json.loads(sys.stdin.buffer.readline())
        os.write(
            1,
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"inherited": "MCP_SYNTHETIC_PARENT" in os.environ},
                }
            ).encode()
            + b"\n",
        )
    elif mode in ("echo", "stubborn"):
        for line in sys.stdin.buffer:
            (directory / "input").write_bytes(line)
            request = json.loads(line)
            os.write(1, json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {}}).encode() + b"\n")
        if mode != "stubborn":
            return
    elif mode == "pending-invalid":
        for line in sys.stdin.buffer:
            request = json.loads(line)
            if request["method"] == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "serverInfo": {"name": "bounded-fixture", "version": "1"},
                    },
                }
                os.write(1, json.dumps(response).encode() + b"\n")
            elif request["method"] == "ping":
                os.write(1, bytes.fromhex(payload))
                break
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
