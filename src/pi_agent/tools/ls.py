"""One-directory listing; contents are never read."""

from __future__ import annotations

import asyncio
import heapq
import os
from collections.abc import Mapping
from pathlib import Path

from pi_agent.cancellation import Cancellation
from pi_agent.types import JsonValue

from ._search import MAX_RESULTS, SearchOutput, display, integer, path_item, search_path
from .base import Tool, ToolResult, ToolUpdateCallback
from .output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES


class LsTool(Tool):
    name = "ls"
    parallel_safe = True
    description = (
        "List a directory, including dotfiles, sorted by name; directories end in '/'. Results are bounded."
    )
    prompt_snippet = "List directory contents"
    guidelines = ("Use ls to list directories.",)
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
        },
        "additionalProperties": False,
    }

    def __init__(
        self, cwd: Path | str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> None:
        self.cwd = Path(cwd)
        self.max_lines, self.max_bytes = max_lines, max_bytes

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        del tool_call_id, on_update
        if cancellation is not None:
            cancellation.raise_if_aborted()
        path = search_path(args, self.cwd)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        limit = integer(args, "limit", 500, MAX_RESULTS)
        output = SearchOutput(self.max_lines, self.max_bytes)
        # Keep only the smallest limit+1 entries, not the entire directory.
        selected: list[_Entry] = []
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if cancellation is not None:
                    cancellation.raise_if_aborted()
                candidate = _Entry(entry.name, entry.is_dir(), entry.is_symlink())
                if len(selected) < limit + 1:
                    heapq.heappush(selected, candidate)
                elif candidate.key < selected[0].key:
                    heapq.heapreplace(selected, candidate)
                if index % 64 == 0:
                    await asyncio.sleep(0)
        ordered = sorted(selected, key=lambda item: item.key)
        if cancellation is not None:
            cancellation.raise_if_aborted()
        for item in ordered[:limit]:
            suffix = "/" if item.directory else ""
            if not output.add(
                display(item.name) + suffix,
                {**path_item(item.name, "name"), "directory": item.directory, "symlink": item.symlink},
            ):
                break
        if len(ordered) > limit and output.reason is None:
            output.reason = "entry limit"
        return output.result("(empty directory)")


class _Entry:
    def __init__(self, name: str, directory: bool, symlink: bool) -> None:
        self.name, self.directory, self.symlink = name, directory, symlink
        self.key = (name.casefold(), name)

    def __lt__(self, other: _Entry) -> bool:
        return self.key > other.key  # Reverse heap keeps the largest retained name at the root.
