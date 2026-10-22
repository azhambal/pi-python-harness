"""Bounded search output and ripgrep process ownership."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import shutil
import signal
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path

from pi_agent.cancellation import Cancellation
from pi_agent.types import JsonValue, TextBlock

from .base import ToolResult, resolve_path
from .output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES

MAX_RESULTS = 10_000
MAX_CONTEXT = 100
MAX_RECORD_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 4096


def integer(args: Mapping[str, JsonValue], key: str, default: int, maximum: int, minimum: int = 1) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be an integer between {minimum} and {maximum}")
    return value


def search_path(args: Mapping[str, JsonValue], cwd: Path) -> Path:
    value = args.get("path", ".")
    if not isinstance(value, str) or not value:
        raise ValueError("path must be nonempty text")
    target = resolve_path(value, cwd)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    return target


def ignore_args(path: Path) -> list[str]:
    """Git-aware traversal stops inherited ignore rules at nested repository roots."""
    directory = path if path.is_dir() else path.parent
    inside_repo = any((parent / ".git").exists() for parent in (directory, *directory.parents))
    return [] if inside_repo else ["--no-require-git"]


def display(value: str) -> str:
    """Keep one output row per item, including unusual filesystem names."""
    return (
        json.dumps(value, ensure_ascii=True)[1:-1]
        if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
        else value
    )


def json_path(value: str) -> str:
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


def path_item(value: str, key: str = "path") -> dict[str, JsonValue]:
    item: dict[str, JsonValue] = {key: json_path(value)}
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        item[f"{key}_bytes_base64"] = base64.b64encode(os.fsencode(value)).decode("ascii")
    return item


def matches_glob(path: str, pattern: str) -> bool:
    """Basename globs or root-relative segments; ** spans zero or more directories."""
    negative = pattern.startswith("!")
    pattern = pattern[1:] if negative else pattern
    parts = path.replace(os.sep, "/").split("/")
    if "/" not in pattern:
        matched = fnmatchcase(parts[-1], pattern)
    else:
        positions = {0}
        for component in pattern.removeprefix("./").split("/"):
            if component == "**":
                positions = set(range(min(positions), len(parts) + 1)) if positions else set()
            else:
                positions = {
                    index + 1
                    for index in positions
                    if index < len(parts) and fnmatchcase(parts[index], component)
                }
        matched = len(parts) in positions
    return not matched if negative else matched


class SearchOutput:
    def __init__(self, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1:
            raise ValueError("max_lines must be a positive integer")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 512 <= max_bytes <= MAX_RECORD_BYTES
        ):
            raise ValueError("max_bytes must be between 512 and 1048576")
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.rows: list[str] = []
        self.items: list[JsonValue] = []
        self.reason: str | None = None
        self._bytes = 384  # Envelope and actionable truncation notice reserve.

    def add(self, row: str, item: dict[str, JsonValue]) -> bool:
        cost = (
            len(json.dumps(row, ensure_ascii=True).encode())
            + len(json.dumps(item, ensure_ascii=True).encode())
            + 4
        )
        if len(self.rows) >= self.max_lines or self._bytes + cost > self.max_bytes:
            self.reason = "output budget"
            return False
        self.rows.append(row)
        self.items.append(item)
        self._bytes += cost
        return True

    def result(self, empty: str) -> ToolResult:
        notice = "[Search limit reached; narrow the path/pattern or increase limit within host caps.]"
        while True:
            text = "\n".join(self.rows) if self.rows else ("" if self.reason else empty)
            if self.reason:
                text += ("\n" if text else "") + notice
            result = ToolResult(
                content=[TextBlock(text=text)],
                details={"truncated": self.reason is not None, "reason": self.reason},
                structured_content={"items": list(self.items)},
            )
            if (
                len(text.splitlines()) <= self.max_lines
                and len(json.dumps(result.to_json(), ensure_ascii=True).encode()) <= self.max_bytes
            ):
                return result
            self.reason = "output budget"
            if not self.rows:
                raise ValueError("Output budget cannot hold the result envelope")
            self.rows.pop()
            self.items.pop()


def rg_bytes(value: object) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("Malformed ripgrep text field")
    if isinstance(value.get("text"), str):
        return value["text"].encode("utf-8")
    if isinstance(value.get("bytes"), str):
        return base64.b64decode(value["bytes"], validate=True)
    raise ValueError("Malformed ripgrep text field")


async def _stop(process: asyncio.subprocess.Process, *, drain_stderr: bool = True) -> None:
    async def discard(stream: asyncio.StreamReader) -> None:
        while await stream.read(65536):
            pass

    streams = [process.stdout, *([process.stderr] if drain_stderr else [])]
    drains = [asyncio.create_task(discard(stream)) for stream in streams if stream is not None]
    joined = asyncio.gather(process.wait(), *drains)
    try:
        # Draining after consumers stop avoids wait() deadlock on a full pipe.
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(asyncio.shield(joined), 1)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await joined
    finally:
        for task in drains:
            task.cancel()
        await asyncio.gather(*drains, return_exceptions=True)
        joined.cancel()
        await asyncio.gather(joined, return_exceptions=True)


async def run_rg(
    args: list[str],
    accept: Callable[[bytes], bool],
    cancellation: Cancellation | None,
    *,
    separator: bytes = b"\n",
    timeout_s: float = 30,
    cwd: Path | None = None,
) -> None:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    executable = shutil.which("rg")
    if executable is None:
        raise FileNotFoundError("ripgrep (rg) is required for grep/find; install it on PATH")
    if cancellation is not None:
        cancellation.raise_if_aborted()
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            executable,
            "--no-config",
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            cwd=cwd,
            env={"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C.UTF-8"},
        )
    )
    try:
        process = await asyncio.shield(spawning)
    except BaseException:
        process = await spawning
        await _stop(process)
        raise
    assert process.stdout is not None and process.stderr is not None
    stderr = bytearray()
    limited = False

    async def drain_errors() -> None:
        assert process.stderr is not None
        while chunk := await process.stderr.read(8192):
            stderr.extend(chunk[: max(0, MAX_STDERR_BYTES - len(stderr))])

    async def consume() -> None:
        nonlocal limited
        assert process.stdout is not None
        buffer = bytearray()
        while chunk := await process.stdout.read(65536):
            buffer.extend(chunk)
            while (end := buffer.find(separator)) >= 0:
                if end > MAX_RECORD_BYTES:
                    raise ValueError("ripgrep output record exceeds host limit; narrow the search")
                record = bytes(buffer[:end])
                del buffer[: end + len(separator)]
                if cancellation is not None:
                    cancellation.raise_if_aborted()
                if not accept(record):
                    limited = True
                    return
            if len(buffer) > MAX_RECORD_BYTES:
                raise ValueError("ripgrep output record exceeds host limit; narrow the search")
            await asyncio.sleep(0)
        if buffer:
            raise ValueError("Incomplete ripgrep output record")

    errors = asyncio.create_task(drain_errors())
    reader = asyncio.create_task(consume())

    async def finish() -> None:
        await reader
        if limited:
            await _stop(process, drain_stderr=False)
        code = await process.wait()
        await errors
        if not limited and code not in (0, 1):
            message = stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(message or f"ripgrep exited with status {code}")

    completion = asyncio.create_task(finish())
    aborted = asyncio.create_task(cancellation.wait()) if cancellation is not None else None
    try:
        async with asyncio.timeout(timeout_s):
            if aborted is not None:
                await asyncio.wait((completion, aborted), return_when=asyncio.FIRST_COMPLETED)
                assert cancellation is not None
                cancellation.raise_if_aborted()
            await completion
    except TimeoutError as error:
        raise TimeoutError("Search timed out; narrow the path/pattern") from error
    finally:
        tasks = [completion, reader, errors, *([aborted] if aborted is not None else [])]
        for task in tasks:
            task.cancel()
        # Drain tasks before stopping: no pending reads race subprocess cleanup.
        await asyncio.gather(*tasks, return_exceptions=True)
        await _stop(process)
