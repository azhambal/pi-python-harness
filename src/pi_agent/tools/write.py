"""Atomic UTF-8 file writes with parent directory creation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pi_agent.cancellation import Cancellation
from pi_agent.tools._atomic import atomic_write
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback, resolve_path
from pi_agent.types import JsonValue, TextBlock


class WriteTool(Tool):
    name = "write"
    parallel_safe = True
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )
    prompt_snippet = "Create or overwrite files"
    guidelines = ("Use write only for new files or complete rewrites.",)
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, cwd: Path | str) -> None:
        self.cwd = Path(cwd)

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        del tool_call_id, on_update
        path = args["path"]
        content = args["content"]
        if not isinstance(path, str) or not isinstance(content, str):
            raise ValueError("Write requires string path and content fields.")
        absolute_path = resolve_path(path, self.cwd)

        def throw_if_aborted() -> None:
            if cancellation is not None:
                cancellation.raise_if_aborted()

        throw_if_aborted()
        parent = absolute_path.parent
        if not parent.exists():
            os.makedirs(parent, exist_ok=True)
        throw_if_aborted()
        atomic_write(absolute_path, content.encode("utf-8"), cancellation=cancellation)
        return ToolResult(
            content=[TextBlock(text=f"Successfully wrote to {path}")],
            details={"path": str(absolute_path)},
        )
