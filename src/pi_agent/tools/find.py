"""Find files using ripgrep ignore rules and a root-relative glob filter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pi_agent.cancellation import Cancellation
from pi_agent.types import JsonValue

from ._search import (
    MAX_RESULTS,
    SearchOutput,
    display,
    ignore_args,
    integer,
    matches_glob,
    path_item,
    run_rg,
    search_path,
)
from .base import Tool, ToolResult, ToolUpdateCallback
from .output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES


class FindTool(Tool):
    name = "find"
    parallel_safe = True
    description = "Find files by glob relative to a directory; respects .gitignore, includes unignored dotfiles. Results are bounded."
    prompt_snippet = "Find files by glob pattern (respects .gitignore)"
    guidelines = ("Use find to locate files by glob pattern.",)
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        cwd: Path | str,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        *,
        timeout_s: float = 30,
    ) -> None:
        self.cwd = Path(cwd)
        self.max_lines, self.max_bytes, self.timeout_s = max_lines, max_bytes, timeout_s

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        del tool_call_id, on_update
        path = search_path(args, self.cwd)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern or pattern.startswith("!"):
            raise ValueError("pattern must be a nonempty positive glob")
        limit = integer(args, "limit", 1000, MAX_RESULTS)
        output = SearchOutput(self.max_lines, self.max_bytes)
        count = 0

        def accept(record: bytes) -> bool:
            nonlocal count
            relative = os.path.relpath(os.fsdecode(record), path).replace(os.sep, "/")
            if not matches_glob(relative, pattern):
                return True
            if not output.add(display(relative), path_item(relative)):
                return False
            count += 1
            if count >= limit:
                output.reason = "result limit"
                return False
            return True

        # Run from the search root so path globs are rooted there, including outside Git.
        await run_rg(
            [
                "--files",
                "--hidden",
                "--null",
                "--sort",
                "path",
                *ignore_args(path),
                "--glob",
                "!.git/**",
                "--",
                str(path),
            ],
            accept,
            cancellation,
            separator=b"\0",
            timeout_s=self.timeout_s,
            cwd=path,
        )
        return output.result("No files found matching pattern")
