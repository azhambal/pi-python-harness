"""Search file contents with ripgrep, translated from Pi's ``core/tools/grep.ts``."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path

from pi_agent._js import JS_WHITESPACE, number_to_string
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools._paths import node_relative, resolve_to_cwd
from pi_agent.tools._search import SpawnError, get_tool_path, run_lines
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback
from pi_agent.tools.output import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    format_size,
    truncate_head,
    truncate_line,
)
from pi_agent.types import JsonValue, TextBlock

DEFAULT_LIMIT = 100
#: ``Number.MAX_SAFE_INTEGER``: byte truncation only, the match limit caps rows.
_NO_LINE_LIMIT = 2**53 - 1
_NEWLINES = re.compile(r"\r\n|\r")


class GrepTool(Tool):
    name = "grep"
    argument_conversion = "typebox"
    description = (
        "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
        f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} matches or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Long lines are truncated to "
        f"{GREP_MAX_LINE_LENGTH} chars."
    )
    prompt_snippet = "Search file contents for patterns (respects .gitignore)"
    guidelines = ()
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "required": ["pattern"],
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: current directory)",
            },
            "glob": {
                "type": "string",
                "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
            },
            "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
            "literal": {
                "type": "boolean",
                "description": "Treat pattern as literal string instead of regex (default: false)",
            },
            "context": {
                "type": "number",
                "description": "Number of lines to show before and after each match (default: 0)",
            },
            "limit": {"type": "number", "description": "Maximum number of matches to return (default: 100)"},
        },
    }

    def __init__(
        self, cwd: Path | str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> None:
        del max_lines  # The match limit caps rows, as in Pi.
        self.cwd = Path(cwd)
        self.max_bytes = max_bytes

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        del tool_call_id, on_update
        if cancellation is not None and cancellation.aborted:
            raise OperationAbortedError("Operation aborted")
        pattern = str(args["pattern"])
        search_dir = args.get("path")
        glob = args.get("glob")
        context = args.get("context")
        limit = args.get("limit")

        rg_path = get_tool_path("rg")
        if rg_path is None:
            raise FileNotFoundError("ripgrep (rg) is not available and could not be downloaded")

        search_path = resolve_to_cwd(
            search_dir if isinstance(search_dir, str) and search_dir else ".", self.cwd
        )
        try:
            is_directory = stat.S_ISDIR(os.stat(search_path).st_mode)
        except OSError:
            raise FileNotFoundError(f"Path not found: {search_path}") from None

        context_value: int | float = context if isinstance(context, (int, float)) and context > 0 else 0
        effective_limit: int | float = max(1, limit if isinstance(limit, (int, float)) else DEFAULT_LIMIT)

        def format_path(file_path: str) -> str:
            if is_directory:
                relative = node_relative(search_path, file_path)
                if relative and not relative.startswith(".."):
                    return relative.replace("\\", "/")
            return os.path.basename(file_path)

        file_cache: dict[str, list[str]] = {}

        def get_file_lines(file_path: str) -> list[str]:
            lines = file_cache.get(file_path)
            if lines is None:
                try:
                    with open(file_path, "rb") as file:
                        content = file.read().decode("utf-8", errors="replace")
                    lines = _NEWLINES.sub("\n", content).split("\n")
                except OSError:
                    lines = []
                file_cache[file_path] = lines
            return lines

        lines_truncated = False

        def format_block(file_path: str, line_number: int | float) -> list[str]:
            nonlocal lines_truncated
            relative_path = format_path(file_path)
            lines = get_file_lines(file_path)
            if not lines:
                return [f"{relative_path}:{line_number}: (unable to read file)"]
            block: list[str] = []
            start = max(1, line_number - context_value) if context_value > 0 else line_number
            end = min(len(lines), line_number + context_value) if context_value > 0 else line_number
            current = start
            while current <= end:
                line_text = _line_at(lines, current - 1).replace("\r", "")
                truncated_text, was_truncated = truncate_line(line_text)
                if was_truncated:
                    lines_truncated = True
                if current == line_number:
                    block.append(f"{relative_path}:{number_to_string(current)}: {truncated_text}")
                else:
                    block.append(f"{relative_path}-{number_to_string(current)}- {truncated_text}")
                current += 1
            return block

        # --no-config: the tool contract (literal, glob, limit, ...) must not depend on
        # ambient user config like RIPGREP_CONFIG_PATH.
        command = ["--no-config", "--json", "--line-number", "--color=never", "--hidden"]
        if args.get("ignoreCase"):
            command.append("--ignore-case")
        if args.get("literal"):
            command.append("--fixed-strings")
        if isinstance(glob, str) and glob:
            command.extend(("--glob", glob))
        command.extend(("--", pattern, search_path))

        match_count = 0
        match_limit_reached = False
        matches: list[tuple[str, int | float, str | None]] = []

        def on_line(line: str) -> bool:
            nonlocal match_count, match_limit_reached
            if not line.strip(JS_WHITESPACE) or match_count >= effective_limit:
                return False
            try:
                event = json.loads(line)
            except ValueError:
                return False
            if not isinstance(event, dict) or event.get("type") != "match":
                return False
            match_count += 1
            data = event.get("data")
            if not isinstance(data, dict):
                data = {}
            file_path = _text(data.get("path"))
            line_number = data.get("line_number")
            if file_path and isinstance(line_number, (int, float)) and not isinstance(line_number, bool):
                matches.append((file_path, line_number, _text(data.get("lines"))))
            if match_count >= effective_limit:
                match_limit_reached = True
                return True
            return False

        try:
            outcome = await run_lines(rg_path, command, on_line, cancellation)
        except SpawnError as error:
            raise RuntimeError(f"Failed to run ripgrep: {error}") from None
        if not match_limit_reached and outcome.code not in (0, 1):
            code = "null" if outcome.code is None else str(outcome.code)
            raise RuntimeError(outcome.stderr.strip(JS_WHITESPACE) or f"ripgrep exited with code {code}")
        if match_count == 0:
            return ToolResult(content=[TextBlock(text="No matches found")], details=None)

        output_lines: list[str] = []
        for file_path, line_number, line_text in matches:
            if context_value == 0 and line_text is not None:
                relative_path = format_path(file_path)
                sanitized = line_text.replace("\r\n", "\n").replace("\r", "").removesuffix("\n")
                truncated_text, was_truncated = truncate_line(sanitized)
                if was_truncated:
                    lines_truncated = True
                output_lines.append(f"{relative_path}:{number_to_string(line_number)}: {truncated_text}")
            else:
                output_lines.extend(format_block(file_path, line_number))

        truncation = truncate_head(
            "\n".join(output_lines), max_lines=_NO_LINE_LIMIT, max_bytes=self.max_bytes
        )
        output = truncation.content
        details: dict[str, JsonValue] = {}
        notices: list[str] = []
        if match_limit_reached:
            notices.append(
                f"{number_to_string(effective_limit)} matches limit reached. "
                f"Use limit={number_to_string(effective_limit * 2)} for more, or refine pattern"
            )
            details["matchLimitReached"] = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(self.max_bytes)} limit reached")
            details["truncation"] = truncation.to_json()
        if lines_truncated:
            notices.append(
                f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
            )
            details["linesTruncated"] = True
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return ToolResult(content=[TextBlock(text=output)], details=details or None)


def _text(field: object) -> str | None:
    """``field?.text``: ripgrep sends non-UTF-8 data as ``bytes`` instead."""
    if isinstance(field, dict) and isinstance(field.get("text"), str):
        return field["text"]
    return None


def _line_at(lines: list[str], index: int | float) -> str:
    """``lines[index] ?? ""`` for a possibly fractional JavaScript index."""
    if index == int(index) and 0 <= index < len(lines):
        return lines[int(index)]
    return ""
