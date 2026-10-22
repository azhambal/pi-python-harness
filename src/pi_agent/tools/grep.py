"""Stream match and context rows from ripgrep without rereading files."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from pi_agent.cancellation import Cancellation
from pi_agent.types import JsonValue

from ._search import (
    MAX_CONTEXT,
    MAX_RESULTS,
    SearchOutput,
    display,
    ignore_args,
    integer,
    matches_glob,
    path_item,
    rg_bytes,
    run_rg,
    search_path,
)
from .base import Tool, ToolResult, ToolUpdateCallback
from .output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES


class GrepTool(Tool):
    name = "grep"
    parallel_safe = True
    description = "Search file contents by regex or literal text; respects .gitignore. Returns bounded paths, line numbers, and optional context."
    prompt_snippet = "Search file contents for patterns (respects .gitignore)"
    guidelines = ("Use grep to search file contents; use read for complete matched files.",)
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "glob": {"type": "string", "minLength": 1},
            "ignoreCase": {"type": "boolean"},
            "literal": {"type": "boolean"},
            "context": {"type": "integer", "minimum": 0, "maximum": MAX_CONTEXT},
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
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be nonempty text")
        limit = integer(args, "limit", 100, MAX_RESULTS)
        context = integer(args, "context", 0, MAX_CONTEXT, 0)
        command = [
            "--json",
            "--line-number",
            "--hidden",
            "--sort",
            "path",
            *ignore_args(path),
            "--glob",
            "!.git/**",
            "--context",
            str(context),
        ]
        for key, flag in (("ignoreCase", "--ignore-case"), ("literal", "--fixed-strings")):
            value = args.get(key, False)
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean")
            if value:
                command.append(flag)
        glob = args.get("glob")
        if glob is not None:
            if not isinstance(glob, str) or not glob:
                raise ValueError("glob must be nonempty text")
        command.extend(("--", pattern, str(path)))
        output = SearchOutput(self.max_lines, self.max_bytes)
        matches = 0

        def accept(record: bytes) -> bool:
            nonlocal matches
            event = json.loads(record)
            kind = event.get("type")
            if kind not in ("match", "context"):
                return True
            data = event["data"]
            name = os.fsdecode(rg_bytes(data["path"]))
            relative = os.path.relpath(name, path) if path.is_dir() else path.name
            if isinstance(glob, str) and not matches_glob(relative, glob):
                return True
            text = rg_bytes(data["lines"]).decode("utf-8", errors="replace").rstrip("\r\n")
            long_line = len(text) > 500
            text = text[:500] + ("… [use read for full line]" if long_line else "")
            line = data["line_number"]
            if isinstance(line, bool) or not isinstance(line, int) or line < 1:
                raise ValueError("Malformed ripgrep line number")
            delimiter = ":" if kind == "match" else "-"
            row = f"{display(relative)}{delimiter}{line}{delimiter} {display(text)}"
            if not output.add(
                row,
                {
                    **path_item(relative),
                    "line": line,
                    "text": text,
                    "match": kind == "match",
                    "line_truncated": long_line,
                },
            ):
                return False
            if kind == "match":
                matches += 1
                if matches >= limit:
                    output.reason = "match limit"
                    return False
            return True

        await run_rg(
            command,
            accept,
            cancellation,
            timeout_s=self.timeout_s,
            cwd=path if path.is_dir() else path.parent,
        )
        return output.result("No matches found")
