"""Read UTF-8 text with 1-based offsets, bounded output, and continuation notices."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pi_agent.cancellation import Cancellation
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback, resolve_path
from pi_agent.tools.output import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)
from pi_agent.types import JsonValue, TextBlock


class ReadTool(Tool):
    name = "read"
    parallel_safe = True
    description = (
        "Read the contents of a file. Supports text files; output is truncated to "
        f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        "Use offset/limit for large files. When you need the full file, continue with offset "
        "until complete."
    )
    prompt_snippet = "Read file contents"
    guidelines = ("Use read to examine files instead of cat or sed.",)
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"},
        },
        "required": ["path"],
    }

    def __init__(
        self, cwd: Path | str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> None:
        self.cwd = Path(cwd)
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        del tool_call_id, on_update
        path = str(args["path"])
        if not path:
            raise ValueError("read requires a non-empty 'path' argument")
        offset = _optional_int(args.get("offset"), "offset")
        limit = _optional_int(args.get("limit"), "limit")
        if offset is not None and offset < 1:
            raise ValueError("offset must be a positive 1-indexed line number")
        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive number of lines")

        absolute_path = resolve_path(path, self.cwd)
        if cancellation is not None:
            cancellation.raise_if_aborted()
        if not os.access(absolute_path, os.R_OK):
            raise FileNotFoundError(f"Cannot read file: {path} (resolved: {absolute_path})")
        with open(absolute_path, "rb") as file:
            raw = file.read()
        if cancellation is not None:
            cancellation.raise_if_aborted()

        # Node's toString("utf-8") replaces invalid sequences; do the same so a
        # stray binary byte degrades instead of failing the whole read.
        text_content = raw.decode("utf-8", errors="replace")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)

        # Convert from 1-indexed input to 0-indexed access; offset 0/absent = start.
        start_line = max(0, (offset or 1) - 1)
        start_line_display = start_line + 1
        if start_line >= total_file_lines:
            raise ValueError(f"Offset {offset} is beyond end of file ({total_file_lines} lines total)")

        user_limited_lines: int | None = None
        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])

        truncation: TruncationResult = truncate_head(selected, self.max_lines, self.max_bytes)
        details: dict[str, JsonValue] | None = None

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(self.max_bytes)} limit. "
                f"Use bash: sed -n '{start_line_display}p' {path} | head -c {self.max_bytes}]"
            )
            details = {"truncation": truncation.to_json()}
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                    f"({format_size(self.max_bytes)} limit). Use offset={next_offset} to continue.]"
                )
            details = {"truncation": truncation.to_json()}
        elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
            remaining = total_file_lines - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
        else:
            output_text = truncation.content

        return ToolResult(
            content=[TextBlock(text=output_text)],
            details=details,
            structured_content={"text": output_text},
        )


def _optional_int(value: JsonValue, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be a whole number of lines")
        value = int(value)
    return int(value)
