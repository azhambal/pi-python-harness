"""Read text files and images, translated from Pi's ``core/tools/read.ts``."""

from __future__ import annotations

import errno
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from pi_agent._js import number_to_string
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools._images import detect_supported_image_mime_type_from_file, process_image
from pi_agent.tools._paths import node_fs_error, resolve_read_path
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback
from pi_agent.tools.output import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
)
from pi_agent.types import ImageBlock, JsonValue, TextBlock

_NON_VISION_IMAGE_NOTE = (
    "[Current model does not support images. The image will be omitted from this request.]"
)


def js_slice_index(value: float, length: int) -> int:
    """JavaScript ``Array.prototype.slice`` index normalization."""
    if math.isnan(value):
        return 0
    if value < 0:
        return 0 if value == -math.inf else max(0, length + math.ceil(value))
    return length if value == math.inf else min(length, math.floor(value))


class ReadTool(Tool):
    name = "read"
    argument_conversion = "typebox"
    description = (
        "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
        "Images are sent as attachments. For text files, output is truncated to "
        f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        "Use offset/limit for large files. When you need the full file, continue with offset "
        "until complete."
    )
    prompt_snippet = "Read file contents"
    guidelines = ("Use read to examine files instead of cat or sed.",)
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"},
        },
    }

    def __init__(
        self,
        cwd: Path | str,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        *,
        auto_resize_images: bool = True,
        model_supports_images: Callable[[], bool | None] | None = None,
    ) -> None:
        self.cwd = Path(cwd)
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.auto_resize_images = auto_resize_images
        #: Reports whether the current model accepts images (Pi's ``ctx.model``);
        #: None or unknown adds no note.
        self.model_supports_images = model_supports_images

    def _non_vision_note(self) -> str | None:
        if self.model_supports_images is None:
            return None
        return _NON_VISION_IMAGE_NOTE if self.model_supports_images() is False else None

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
        path = str(args["path"])
        offset = args.get("offset")
        limit = args.get("limit")

        absolute_path = resolve_read_path(path, self.cwd)
        try:
            os.stat(absolute_path)
            if not os.access(absolute_path, os.R_OK):
                raise PermissionError(errno.EACCES, "Permission denied")
        except OSError as error:
            raise node_fs_error(error, "access", absolute_path) from None
        try:
            mime_type = detect_supported_image_mime_type_from_file(absolute_path)
        except IsADirectoryError as error:
            raise node_fs_error(error, "read") from None
        except OSError as error:
            raise node_fs_error(error, "open", absolute_path) from None
        try:
            with open(absolute_path, "rb") as file:
                buffer = file.read()
        except IsADirectoryError as error:
            raise node_fs_error(error, "read") from None
        except OSError as error:
            raise node_fs_error(error, "open", absolute_path) from None
        if cancellation is not None:
            cancellation.raise_if_aborted()

        note = self._non_vision_note()
        details: dict[str, JsonValue] | None = None
        if mime_type:
            processed = process_image(buffer, mime_type, auto_resize=self.auto_resize_images)
            if not processed.ok:
                text = f"Read image file [{mime_type}]\n{processed.message}"
                if note:
                    text += f"\n{note}"
                content: list[TextBlock | ImageBlock] = [TextBlock(text=text)]
                structured: JsonValue = text
            else:
                text = f"Read image file [{processed.mime_type}]"
                if processed.hints:
                    text += "\n" + "\n".join(processed.hints)
                if note:
                    text += f"\n{note}"
                content = [
                    TextBlock(text=text),
                    ImageBlock(data=processed.data, mime_type=processed.mime_type),
                ]
                structured = {
                    "type": "image",
                    "data": processed.data,
                    "mimeType": processed.mime_type,
                    "note": text,
                }
            return ToolResult(content=content, details=None, structured_content=structured)

        # Node's toString("utf-8") replaces invalid sequences.
        text_content = buffer.decode("utf-8", errors="replace")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, offset - 1) if offset else 0  # type: ignore[operator]
        start_index = js_slice_index(start_line, total_file_lines)
        start_line_display = start_line + 1
        if start_line >= total_file_lines:
            raise ValueError(
                f"Offset {number_to_string(offset)} is beyond end of file ({total_file_lines} lines total)"  # type: ignore[arg-type]
            )

        user_limited_lines: float | None = None
        if limit is not None:
            end_line = min(start_line + limit, total_file_lines)  # type: ignore[operator]
            selected = "\n".join(all_lines[start_index : js_slice_index(end_line, total_file_lines)])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_index:])

        truncation = truncate_head(selected, self.max_lines, self.max_bytes)
        display_start = number_to_string(start_line_display)
        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_index].encode("utf-8")))
            output_text = (
                f"[Line {display_start} is {first_line_size}, exceeds {format_size(self.max_bytes)} limit. "
                f"Use bash: sed -n '{display_start}p' {path} | head -c {self.max_bytes}]"
            )
            details = {"truncation": truncation.to_json()}
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = number_to_string(end_line_display + 1)
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {display_start}-{number_to_string(end_line_display)} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {display_start}-{number_to_string(end_line_display)} of {total_file_lines} "
                    f"({format_size(self.max_bytes)} limit). Use offset={next_offset} to continue.]"
                )
            details = {"truncation": truncation.to_json()}
        elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
            remaining = total_file_lines - (start_line + user_limited_lines)
            following_offset = start_line + user_limited_lines + 1
            output_text = (
                f"{truncation.content}\n\n[{number_to_string(remaining)} more lines in file. "
                f"Use offset={number_to_string(following_offset)} to continue.]"
            )
        else:
            output_text = truncation.content

        return ToolResult(
            content=[TextBlock(text=output_text)], details=details, structured_content=output_text
        )
