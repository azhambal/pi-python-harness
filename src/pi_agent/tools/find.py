"""Find files with fd, translated from Pi's ``core/tools/find.ts``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pi_agent._js import JS_WHITESPACE, number_to_string
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools._paths import node_relative, path_exists, resolve_to_cwd
from pi_agent.tools._search import SpawnError, get_tool_path, run_lines
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback
from pi_agent.tools.output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head
from pi_agent.types import JsonValue, TextBlock

DEFAULT_LIMIT = 1000
#: ``Number.MAX_SAFE_INTEGER``: byte truncation only, the result limit caps rows.
_NO_LINE_LIMIT = 2**53 - 1


def relativize_find_result_path(result_path: str, search_path: str) -> str:
    """Pi's ``relativizeFindResultPath``: relative to the search root, posix separators."""
    had_trailing_separator = result_path.endswith(os.sep)
    relative_path = node_relative(search_path, result_path) if os.path.isabs(result_path) else result_path
    posix_path = "/".join(relative_path.split(os.sep))
    return f"{posix_path}/" if had_trailing_separator and not posix_path.endswith("/") else posix_path


class FindTool(Tool):
    name = "find"
    argument_conversion = "typebox"
    description = (
        "Search for files by glob pattern. Returns matching file paths relative to the search directory. "
        f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} results or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
    )
    prompt_snippet = "Find files by glob pattern (respects .gitignore)"
    guidelines = ()
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "required": ["pattern"],
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'",
            },
            "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
            "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
        },
    }

    def __init__(
        self, cwd: Path | str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> None:
        del max_lines  # The result limit caps rows, as in Pi.
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
        limit = args.get("limit")
        search_path = resolve_to_cwd(
            search_dir if isinstance(search_dir, str) and search_dir else ".", self.cwd
        )
        effective_limit: int | float = limit if isinstance(limit, (int, float)) else DEFAULT_LIMIT

        fd_path = get_tool_path("fd")
        if fd_path is None:
            raise FileNotFoundError("fd is not available and could not be downloaded")

        command = ["--glob", "--color=never", "--hidden"]
        # fd normally ignores .gitignore outside git repos, so keep --no-require-git
        # there. Inside repos, use fd's default git-aware behavior so parent
        # .gitignore rules stop at nested repo boundaries.
        inside_git_repo = False
        current = search_path
        while True:
            if path_exists(os.path.join(current, ".git")):
                inside_git_repo = True
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        if not inside_git_repo:
            command.append("--no-require-git")
        command.extend(("--max-results", number_to_string(effective_limit)))

        # fd --glob matches against the basename unless --full-path is set; in --full-path
        # mode it matches against the absolute candidate path, so a path-containing
        # pattern like 'src/**/*.spec.ts' needs a leading '**/' to match anything.
        effective_pattern = pattern
        if "/" in pattern:
            command.append("--full-path")
            if not pattern.startswith("/") and not pattern.startswith("**/") and pattern != "**":
                effective_pattern = f"**/{pattern}"
        command.extend(("--", effective_pattern, search_path))

        lines: list[str] = []
        try:
            outcome = await run_lines(fd_path, command, lambda line: lines.append(line), cancellation)
        except SpawnError as error:
            raise RuntimeError(f"Failed to run fd: {error}") from None
        output = "\n".join(lines)
        if outcome.code != 0:
            error_message = (
                outcome.stderr.strip(JS_WHITESPACE) or f"fd exited with code {_code(outcome.code)}"
            )
            if not output:
                raise RuntimeError(error_message)
        if not output:
            return ToolResult(content=[TextBlock(text="No files found matching pattern")], details=None)

        relativized: list[str] = []
        for raw_line in lines:
            line = raw_line.removesuffix("\r").strip(JS_WHITESPACE)
            if not line:
                continue
            relativized.append(relativize_find_result_path(line, search_path))

        result_limit_reached = len(relativized) >= effective_limit
        truncation = truncate_head("\n".join(relativized), max_lines=_NO_LINE_LIMIT, max_bytes=self.max_bytes)
        result_output = truncation.content
        details: dict[str, JsonValue] = {}
        notices: list[str] = []
        if result_limit_reached:
            notices.append(
                f"{number_to_string(effective_limit)} results limit reached. "
                f"Use limit={number_to_string(effective_limit * 2)} for more, or refine pattern"
            )
            details["resultLimitReached"] = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(self.max_bytes)} limit reached")
            details["truncation"] = truncation.to_json()
        if notices:
            result_output += f"\n\n[{'. '.join(notices)}]"
        return ToolResult(content=[TextBlock(text=result_output)], details=details or None)


def _code(code: int | None) -> str:
    return "null" if code is None else str(code)
