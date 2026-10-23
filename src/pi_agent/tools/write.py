"""Write files, translated from Pi's ``core/tools/write.ts``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools._mutation import file_mutation_queue
from pi_agent.tools._paths import node_fs_error, resolve_to_cwd
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback
from pi_agent.types import JsonValue, TextBlock


def make_directories(directory: str) -> None:
    """Node ``mkdir(dir, { recursive: true })`` with Node's error text."""
    try:
        os.makedirs(directory, exist_ok=True)
    except FileExistsError as error:
        raise node_fs_error(error, "mkdir", directory) from None
    except OSError as error:
        raise node_fs_error(error, "mkdir", getattr(error, "filename", None) or directory) from None


def write_text_file(path: str, content: str) -> None:
    """Node ``writeFile(path, content, "utf-8")``: truncate and write in place."""
    try:
        with open(path, "wb") as file:
            file.write(content.encode("utf-8", errors="surrogatepass"))
    except IsADirectoryError as error:
        raise node_fs_error(error, "open", path) from None
    except OSError as error:
        raise node_fs_error(error, "open", path) from None


class WriteTool(Tool):
    name = "write"
    argument_conversion = "typebox"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )
    prompt_snippet = "Create or overwrite files"
    guidelines = ("Use write only for new files or complete rewrites.",)
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "required": ["path", "content"],
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
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
        path = str(args["path"])
        content = str(args["content"])
        absolute_path = resolve_to_cwd(path, self.cwd)

        def throw_if_aborted() -> None:
            if cancellation is not None and cancellation.aborted:
                raise OperationAbortedError("Operation aborted")

        async with file_mutation_queue(absolute_path):
            throw_if_aborted()
            make_directories(os.path.dirname(absolute_path))
            throw_if_aborted()
            write_text_file(absolute_path, content)
            throw_if_aborted()
            return ToolResult(content=[TextBlock(text=f"Successfully wrote to {path}")], details=None)
