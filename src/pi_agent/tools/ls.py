"""List one directory, translated from Pi's ``core/tools/ls.ts``."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

from pi_agent._collation import collation_key
from pi_agent._js import number_to_string
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools._paths import node_fs_error, path_exists, resolve_to_cwd
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback
from pi_agent.tools.output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head
from pi_agent.types import JsonValue, TextBlock

DEFAULT_LIMIT = 500
#: ``Number.MAX_SAFE_INTEGER``: byte truncation only, the entry limit caps rows.
_NO_LINE_LIMIT = 2**53 - 1


class LsTool(Tool):
    name = "ls"
    argument_conversion = "typebox"
    description = (
        "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. "
        f"Includes dotfiles. Output is truncated to {DEFAULT_LIMIT} entries or {DEFAULT_MAX_BYTES // 1024}KB "
        "(whichever is hit first)."
    )
    prompt_snippet = "List directory contents"
    guidelines = ()
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: current directory)"},
            "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
        },
    }

    def __init__(
        self, cwd: Path | str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> None:
        del max_lines  # The entry limit caps rows, as in Pi.
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
        path = args.get("path")
        limit = args.get("limit")
        dir_path = resolve_to_cwd(path if isinstance(path, str) and path else ".", self.cwd)
        effective_limit: int | float = limit if isinstance(limit, (int, float)) else DEFAULT_LIMIT

        if not path_exists(dir_path):
            raise FileNotFoundError(f"Path not found: {dir_path}")
        try:
            dir_stat = os.stat(dir_path)
        except OSError as error:
            raise node_fs_error(error, "stat", dir_path) from None
        if not stat.S_ISDIR(dir_stat.st_mode):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        try:
            names = os.listdir(dir_path)
        except OSError as error:
            raise OSError(f"Cannot read directory: {node_fs_error(error, 'scandir', dir_path)}") from None
        # Node decodes names as UTF-8 with replacement characters.
        entries = [(os.fsencode(name).decode("utf-8", errors="replace"), name) for name in names]
        entries.sort(key=lambda entry: collation_key(entry[0].lower()))

        results: list[str] = []
        entry_limit_reached = False
        for display_name, name in entries:
            if cancellation is not None:
                cancellation.raise_if_aborted()
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break
            try:
                entry_stat = os.stat(os.path.join(dir_path, name))
            except OSError:
                continue  # Skip entries we cannot stat, such as broken symlinks.
            results.append(display_name + ("/" if stat.S_ISDIR(entry_stat.st_mode) else ""))

        if not results:
            return ToolResult(content=[TextBlock(text="(empty directory)")], details=None)

        truncation = truncate_head("\n".join(results), max_lines=_NO_LINE_LIMIT, max_bytes=self.max_bytes)
        output = truncation.content
        details: dict[str, JsonValue] = {}
        notices: list[str] = []
        if entry_limit_reached:
            notices.append(
                f"{number_to_string(effective_limit)} entries limit reached. "
                f"Use limit={number_to_string(effective_limit * 2)} for more"
            )
            details["entryLimitReached"] = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(self.max_bytes)} limit reached")
            details["truncation"] = truncation.to_json()
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return ToolResult(content=[TextBlock(text=output)], details=details or None)
