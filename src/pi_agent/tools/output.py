"""UTF-8 truncation and bounded streaming output with temporary spills."""

from __future__ import annotations

import codecs
import os
import secrets
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, Literal

from pi_agent._js import to_fixed, utf16_length, utf16_slice
from pi_agent.types import JsonValue

from ._spill import SpillRegistry

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
GREP_MAX_LINE_LENGTH = 500

TruncatedBy = Literal["lines", "bytes"]


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{to_fixed(num_bytes / 1024, 1)}KB"
    return f"{to_fixed(num_bytes / (1024 * 1024), 1)}MB"


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Pi's ``truncateLine``; lengths count UTF-16 code units as JavaScript does."""
    if utf16_length(line) <= max_chars:
        return line, False
    return f"{utf16_slice(line, max_chars)}... [truncated]", True


def _split_lines_for_counting(content: str) -> list[str]:
    """Split into lines the way Pi does: on '\\n', keeping a trailing empty."""
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    start = len(raw) - max_bytes
    # Advance to the start of a UTF-8 character (skip continuation bytes).
    while start < len(raw) and (raw[start] & 0xC0) == 0x80:
        start += 1
    return raw[start:].decode("utf-8", errors="replace")


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: TruncatedBy | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "truncated": self.truncated,
            "truncated_by": self.truncated_by,
            "total_lines": self.total_lines,
            "total_bytes": self.total_bytes,
            "output_lines": self.output_lines,
            "output_bytes": self.output_bytes,
            "last_line_partial": self.last_line_partial,
            "first_line_exceeds_limit": self.first_line_exceeds_limit,
        }


def _no_truncation(content: str, max_lines: int, max_bytes: int) -> TruncationResult:
    total_bytes = len(content.encode("utf-8"))
    total_lines = len(_split_lines_for_counting(content))
    return TruncationResult(
        content=content,
        truncated=False,
        truncated_by=None,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=total_lines,
        output_bytes=total_bytes,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_head(
    content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> TruncationResult:
    """Keep the first N lines/bytes. Never returns partial lines.

    If the first line alone exceeds the byte limit, returns empty content with
    ``first_line_exceeds_limit`` set so callers can point at an alternative.
    """
    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _no_truncation(content, max_lines, max_bytes)

    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by: TruncatedBy = "lines"
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes += line_bytes
    if len(output_lines) >= max_lines and output_bytes <= max_bytes:
        truncated_by = "lines"
    output_content = "\n".join(output_lines)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=len(output_content.encode("utf-8")),
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(
    content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> TruncationResult:
    """Keep the last N lines/bytes.

    If the final line alone exceeds the byte budget, keeps the final
    ``max_bytes`` of it (``last_line_partial``), mirroring Pi.
    """
    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _no_truncation(content, max_lines, max_bytes)

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by: TruncatedBy = "lines"
    last_line_partial = False
    for line in reversed(lines):
        if len(output_lines) >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if output_lines else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                partial = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, partial)
                output_bytes = len(partial.encode("utf-8"))
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes
    if len(output_lines) >= max_lines and output_bytes <= max_bytes:
        truncated_by = "lines"
    output_content = "\n".join(output_lines)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=len(output_content.encode("utf-8")),
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


@dataclass
class OutputSnapshot:
    content: str
    truncation: TruncationResult
    full_output_path: str | None = None


@dataclass
class FullOutput:
    content: str
    truncated: bool


class OutputAccumulator:
    """Incrementally tracks streaming output with bounded memory.

    Chunks are decoded with a streaming UTF-8 decoder (split multibyte
    characters are held back until complete), only a decoded tail is kept for
    display snapshots, and a 0600 temp file captures the full raw bytes once
    output passes the limits.
    """

    def __init__(
        self,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        temp_file_prefix: str = "pi-output",
        spill_registry: SpillRegistry | None = None,
    ) -> None:
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.max_rolling_bytes = max(max_bytes * 2, 1)
        self.temp_file_prefix = temp_file_prefix
        self._spill_registry = spill_registry
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._raw_chunks: list[bytes] = []
        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self._total_raw_bytes = 0
        self._total_decoded_bytes = 0
        self._completed_lines = 0
        self._total_lines = 0
        self._current_line_bytes = 0
        self._has_open_line = False
        self._finished = False
        self._temp_file_path: str | None = None
        self._temp_file: BinaryIO | None = None

    def append(self, data: bytes) -> None:
        if self._finished:
            raise RuntimeError("Cannot append to a finished output accumulator")
        self._total_raw_bytes += len(data)
        self._append_decoded_text(self._decoder.decode(data))
        if self._temp_file is not None or self._should_use_temp_file():
            self._ensure_temp_file()
            assert self._temp_file is not None
            self._temp_file.write(data)
        elif data:
            self._raw_chunks.append(data)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._append_decoded_text(self._decoder.decode(b"", final=True))
        if self._should_use_temp_file():
            self._ensure_temp_file()

    def snapshot(self, persist_if_truncated: bool = False) -> OutputSnapshot:
        tail_truncation = truncate_tail(self._snapshot_text(), self.max_lines, self.max_bytes)
        truncated = self._total_lines > self.max_lines or self._total_decoded_bytes > self.max_bytes
        truncated_by: TruncatedBy | None = None
        if truncated:
            truncated_by = tail_truncation.truncated_by or (
                "bytes" if self._total_decoded_bytes > self.max_bytes else "lines"
            )
        truncation = TruncationResult(
            content=tail_truncation.content,
            truncated=truncated,
            truncated_by=truncated_by,
            total_lines=self._total_lines,
            total_bytes=self._total_decoded_bytes,
            output_lines=tail_truncation.output_lines,
            output_bytes=tail_truncation.output_bytes,
            last_line_partial=tail_truncation.last_line_partial,
            first_line_exceeds_limit=False,
            max_lines=self.max_lines,
            max_bytes=self.max_bytes,
        )
        if persist_if_truncated and truncation.truncated:
            self._ensure_temp_file()
        return OutputSnapshot(
            content=truncation.content,
            truncation=truncation,
            full_output_path=self._temp_file_path,
        )

    def close_temp_file(self) -> None:
        if self._temp_file is None:
            return
        file = self._temp_file
        self._temp_file = None
        try:
            file.flush()
            os.fsync(file.fileno())
        finally:
            file.close()

    def read_full_output(self, max_bytes: int) -> FullOutput:
        """The complete output for callers allowed more than the display snapshot.

        Call after ``finish()`` and ``close_temp_file()``. Output larger than
        ``max_bytes`` keeps its first and last halves around an omission marker.
        """
        if self._temp_file_path is None:
            return FullOutput(
                content=b"".join(self._raw_chunks).decode("utf-8", errors="replace"),
                truncated=False,
            )
        with open(self._temp_file_path, "rb") as file:
            size = os.fstat(file.fileno()).st_size
            if size <= max_bytes:
                return FullOutput(content=file.read().decode("utf-8", errors="replace"), truncated=False)
            head_bytes = max_bytes // 2
            tail_bytes = max_bytes - head_bytes
            head = file.read(head_bytes)
            file.seek(size - tail_bytes)
            tail = file.read(tail_bytes)
        # Streaming decode holds back an incomplete trailing sequence from the
        # head cut; the tail skips leading continuation bytes, like Pi does.
        head_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        head_text = head_decoder.decode(head)
        tail_start = 0
        while tail_start < len(tail) and (tail[tail_start] & 0xC0) == 0x80:
            tail_start += 1
        tail_text = tail[tail_start:].decode("utf-8", errors="replace")
        omitted = size - head_bytes - tail_bytes
        return FullOutput(
            content=f"{head_text}\n\n[... {omitted} bytes omitted ...]\n\n{tail_text}",
            truncated=True,
        )

    def get_last_line_bytes(self) -> int:
        return self._current_line_bytes

    def _append_decoded_text(self, text: str) -> None:
        if not text:
            return
        byte_count = len(text.encode("utf-8"))
        self._total_decoded_bytes += byte_count
        self._tail_text += text
        self._tail_bytes += byte_count
        if self._tail_bytes > self.max_rolling_bytes * 2:
            self._trim_tail()
        newline_count = text.count("\n")
        if newline_count == 0:
            self._current_line_bytes += byte_count
            self._has_open_line = True
        else:
            self._completed_lines += newline_count
            trailing = text.rsplit("\n", 1)[1]
            self._current_line_bytes = len(trailing.encode("utf-8"))
            self._has_open_line = bool(trailing)
        self._total_lines = self._completed_lines + (1 if self._has_open_line else 0)

    def _trim_tail(self) -> None:
        raw = self._tail_text.encode("utf-8")
        if len(raw) <= self.max_rolling_bytes:
            self._tail_bytes = len(raw)
            return
        start = len(raw) - self.max_rolling_bytes
        while start < len(raw) and (raw[start] & 0xC0) == 0x80:
            start += 1
        self._tail_starts_at_line_boundary = (
            self._tail_starts_at_line_boundary if start == 0 else raw[start - 1] == 0x0A
        )
        self._tail_text = raw[start:].decode("utf-8", errors="replace")
        self._tail_bytes = len(self._tail_text.encode("utf-8"))

    def _snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text
        first_newline = self._tail_text.find("\n")
        return self._tail_text if first_newline == -1 else self._tail_text[first_newline + 1 :]

    def _should_use_temp_file(self) -> bool:
        return (
            self._total_raw_bytes > self.max_bytes
            or self._total_decoded_bytes > self.max_bytes
            or self._total_lines > self.max_lines
        )

    def _ensure_temp_file(self) -> None:
        if self._temp_file_path is not None:
            return
        # Pi's createOutputFilePath: `<tmpdir>/<prefix>-<16 hex>.log`, created exclusively.
        path = os.path.join(tempfile.gettempdir(), f"{self.temp_file_prefix}-{secrets.token_hex(8)}.log")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            if self._spill_registry is not None:
                self._spill_registry.register(path, descriptor)
            self._temp_file = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            os.unlink(path)
            raise
        self._temp_file_path = path
        for chunk in self._raw_chunks:
            self._temp_file.write(chunk)
        self._raw_chunks = []
