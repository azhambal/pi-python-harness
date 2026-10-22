"""Available builtins; the original coding factory retains its four-tool default."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .base import (
    Tool,
    ToolResult,
    ToolUpdateCallback,
    ToolValidationError,
    resolve_path,
    validate_tool_arguments,
)
from .bash import BashTool
from .edit import EditTool
from .find import FindTool
from .grep import GrepTool
from .ls import LsTool
from .output import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    OutputAccumulator,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_tail,
)
from .read import ReadTool
from .write import WriteTool

CODING_TOOL_NAMES = ("read", "bash", "edit", "write")
TOOL_NAMES = (*CODING_TOOL_NAMES, "grep", "find", "ls")


def validate_builtin_names(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or any(not isinstance(name, str) for name in names):
        raise ValueError("tool_names must be a sequence of builtin names")
    selected = tuple(names)
    if len(set(selected)) != len(selected) or set(selected) - set(TOOL_NAMES):
        raise ValueError("tool_names must contain unique supported builtin names")
    return selected


def resolve_builtin_selection(names: Sequence[str], base: Sequence[str]) -> tuple[str, ...]:
    """Resolve an absolute list or uniform +name/-name edits against a base."""
    if isinstance(names, (str, bytes)) or any(not isinstance(name, str) for name in names):
        raise ValueError("Tool selection must be a sequence of builtin names")
    if not any(name.startswith(("+", "-")) for name in names):
        return validate_builtin_names(names)
    if not all(name.startswith(("+", "-")) for name in names):
        raise ValueError("Use either builtin names or +name/-name entries, not both")
    validate_builtin_names([name[1:] for name in names])
    selected = list(validate_builtin_names(base))
    for entry in names:
        name = entry[1:]
        if entry.startswith("-"):
            if name in selected:
                selected.remove(name)
        elif name not in selected:
            selected.append(name)
    return tuple(selected)


def is_builtin_search(tool: Tool) -> bool:
    """An injected tool cannot acquire read-only admission just by using a builtin name."""
    return type(tool) in (GrepTool, FindTool, LsTool)


def create_builtin_tools(
    cwd: Path | str,
    names: Sequence[str] = TOOL_NAMES,
    *,
    bash_default_timeout_s: float | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Tool]:
    selected = validate_builtin_names(names)
    result: list[Tool] = []
    factories = {"read": ReadTool, "grep": GrepTool, "find": FindTool, "ls": LsTool}
    for name in selected:
        if name in factories:
            result.append(factories[name](cwd, max_lines=max_lines, max_bytes=max_bytes))
        elif name == "bash":
            result.append(
                BashTool(
                    cwd, default_timeout_s=bash_default_timeout_s, max_lines=max_lines, max_bytes=max_bytes
                )
            )
        elif name == "edit":
            result.append(EditTool(cwd))
        elif name == "write":
            result.append(WriteTool(cwd))
    return result


def create_coding_tools(
    cwd: Path | str,
    bash_default_timeout_s: float | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Tool]:
    """The four core tools bound to a cwd, in declaration order."""
    return [
        ReadTool(cwd, max_lines=max_lines, max_bytes=max_bytes),
        BashTool(cwd, default_timeout_s=bash_default_timeout_s, max_lines=max_lines, max_bytes=max_bytes),
        EditTool(cwd),
        WriteTool(cwd),
    ]


__all__ = [
    "CODING_TOOL_NAMES",
    "TOOL_NAMES",
    "BashTool",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "EditTool",
    "FindTool",
    "GrepTool",
    "LsTool",
    "OutputAccumulator",
    "ReadTool",
    "Tool",
    "ToolResult",
    "ToolUpdateCallback",
    "ToolValidationError",
    "TruncationResult",
    "WriteTool",
    "create_coding_tools",
    "create_builtin_tools",
    "format_size",
    "resolve_path",
    "truncate_head",
    "truncate_tail",
    "validate_tool_arguments",
    "validate_builtin_names",
]
