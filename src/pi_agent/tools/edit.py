"""Exact and fuzzy multi-edits, translated from Pi's ``edit.ts`` and ``edit-diff.ts``."""

from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools._mutation import file_mutation_queue
from pi_agent.tools._paths import error_code, node_fs_error, resolve_to_cwd
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback
from pi_agent.tools.write import write_text_file
from pi_agent.types import JsonValue, TextBlock

_UTF8_BOM = "﻿"
#: Characters JavaScript's ``String.prototype.trimEnd`` removes.
_JS_TRAILING_SPACE = " \t\n\v\f\r                 　﻿"
_SINGLE_QUOTES = re.compile("[‘’‚‛]")
_DOUBLE_QUOTES = re.compile("[“”„‟]")
_DASHES = re.compile("[‐‑‒–—―−]")
_SPECIAL_SPACES = re.compile("[  -   　]")
_LINES_WITH_ENDINGS = re.compile(r"[^\n]*\n|[^\n]+")


def split_bom(text: str) -> tuple[str, str]:
    """Split a leading UTF-8 BOM off; returns (bom, text)."""
    if text.startswith(_UTF8_BOM):
        return _UTF8_BOM, text[len(_UTF8_BOM) :]
    return "", text


def detect_line_ending(content: str) -> str:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Pi's ``normalizeForFuzzyMatch``."""
    text = unicodedata.normalize("NFKC", text)
    text = "\n".join(line.rstrip(_JS_TRAILING_SPACE) for line in text.split("\n"))
    text = _SINGLE_QUOTES.sub("'", text)
    text = _DOUBLE_QUOTES.sub('"', text)
    text = _DASHES.sub("-", text)
    return _SPECIAL_SPACES.sub(" ", text)


@dataclass
class Edit:
    old_text: str
    new_text: str


@dataclass
class AppliedEdits:
    base_content: str
    new_content: str


@dataclass
class _Match:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


@dataclass
class _FuzzyMatch:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool


def fuzzy_find_text(content: str, old_text: str) -> _FuzzyMatch:
    exact = content.find(old_text)
    if exact != -1:
        return _FuzzyMatch(True, exact, len(old_text), False)
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    index = fuzzy_content.find(fuzzy_old)
    if index == -1:
        return _FuzzyMatch(False, -1, 0, False)
    return _FuzzyMatch(True, index, len(fuzzy_old), True)


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    fuzzy_content = normalize_for_fuzzy_match(content)
    if fuzzy_old == "":
        # JavaScript "abc".split("") yields one piece per UTF-16 code unit.
        return len(fuzzy_content.encode("utf-16-le")) // 2 - 1 if fuzzy_content else 0
    return len(fuzzy_content.split(fuzzy_old)) - 1


def _not_found(path: str, index: int, total: int) -> ValueError:
    if total == 1:
        return ValueError(
            f"Could not find the exact text in {path}. The old text must match exactly including all "
            "whitespace and newlines."
        )
    return ValueError(
        f"Could not find edits[{index}] in {path}. The oldText must match exactly including all whitespace "
        "and newlines."
    )


def _duplicate(path: str, index: int, total: int, occurrences: int) -> ValueError:
    if total == 1:
        return ValueError(
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide "
            "more context to make it unique."
        )
    return ValueError(
        f"Found {occurrences} occurrences of edits[{index}] in {path}. Each oldText must be unique. Please "
        "provide more context to make it unique."
    )


def _apply_replacements(content: str, replacements: list[_Match], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        index = replacement.match_index - offset
        result = result[:index] + replacement.new_text + result[index + replacement.match_length :]
    return result


def _line_spans(content: str) -> list[tuple[int, int]]:
    spans = []
    offset = 0
    for line in _LINES_WITH_ENDINGS.findall(content):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _replacement_line_range(lines: list[tuple[int, int]], replacement: _Match) -> tuple[int, int]:
    start = replacement.match_index
    end = replacement.match_index + replacement.match_length
    start_line = next((i for i, (a, b) in enumerate(lines) if a <= start < b), -1)
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")
    end_line = start_line
    while end_line < len(lines) and lines[end_line][1] < end:
        end_line += 1
    if end_line >= len(lines):
        raise ValueError("Replacement range is outside the base content.")
    return start_line, end_line + 1


def apply_replacements_preserving_unchanged_lines(
    original_content: str, base_content: str, replacements: list[_Match]
) -> str:
    original_lines = _LINES_WITH_ENDINGS.findall(original_content)
    base_lines = _line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise ValueError(
            "Cannot preserve unchanged lines because the base content has a different line count."
        )
    groups: list[list] = []  # [start_line, end_line, replacements]
    for replacement in sorted(replacements, key=lambda item: item.match_index):
        start_line, end_line = _replacement_line_range(base_lines, replacement)
        if groups and start_line < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end_line)
            groups[-1][2].append(replacement)
            continue
        groups.append([start_line, end_line, [replacement]])
    original_index = 0
    result = ""
    for start_line, end_line, members in groups:
        result += "".join(original_lines[original_index:start_line])
        group_start = base_lines[start_line][0]
        group_end = base_lines[end_line - 1][1]
        result += _apply_replacements(base_content[group_start:group_end], members, group_start)
        original_index = end_line
    result += "".join(original_lines[original_index:])
    return result


def apply_edits(normalized_content: str, edits: list[Edit], path: str) -> AppliedEdits:
    """Pi's ``applyEditsToNormalizedContent``."""
    normalized = [Edit(normalize_to_lf(edit.old_text), normalize_to_lf(edit.new_text)) for edit in edits]
    total = len(normalized)
    for index, edit in enumerate(normalized):
        if not edit.old_text:
            raise ValueError(
                f"oldText must not be empty in {path}."
                if total == 1
                else f"edits[{index}].oldText must not be empty in {path}."
            )

    initial = [fuzzy_find_text(normalized_content, edit.old_text) for edit in normalized]
    used_fuzzy = any(match.used_fuzzy_match for match in initial)
    base = normalize_for_fuzzy_match(normalized_content) if used_fuzzy else normalized_content

    matched: list[_Match] = []
    for index, edit in enumerate(normalized):
        match = fuzzy_find_text(base, edit.old_text)
        if not match.found:
            raise _not_found(path, index, total)
        occurrences = _count_occurrences(base, edit.old_text)
        if occurrences > 1:
            raise _duplicate(path, index, total, occurrences)
        matched.append(_Match(index, match.index, match.match_length, edit.new_text))

    matched.sort(key=lambda item: item.match_index)
    for previous, current in zip(matched, matched[1:], strict=False):
        if previous.match_index + previous.match_length > current.match_index:
            raise ValueError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    new_content = (
        apply_replacements_preserving_unchanged_lines(normalized_content, base, matched)
        if used_fuzzy
        else _apply_replacements(base, matched)
    )
    if new_content == normalized_content:
        raise ValueError(
            f"No changes made to {path}. The replacement produced identical content. This might indicate an "
            "issue with special characters or the text not existing as expected."
            if total == 1
            else f"No changes made to {path}. The replacements produced identical content."
        )
    return AppliedEdits(base_content=normalized_content, new_content=new_content)


def _diff_parts(old_content: str, new_content: str) -> list[tuple[str, list[str]]]:
    """Line-level diff parts (``"equal"``/``"removed"``/``"added"``), like jsdiff ``diffLines``."""
    old_lines = _LINES_WITH_ENDINGS.findall(old_content)
    new_lines = _LINES_WITH_ENDINGS.findall(new_content)
    parts: list[tuple[str, list[str]]] = []
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(("equal", old_lines[a0:a1]))
            continue
        if a1 > a0:
            parts.append(("removed", old_lines[a0:a1]))
        if b1 > b0:
            parts.append(("added", new_lines[b0:b1]))
    return parts


def generate_diff_string(
    old_content: str, new_content: str, context_lines: int = 4
) -> tuple[str, int | None]:
    """Pi's display-oriented ``generateDiffString``: (diff, first changed line)."""
    parts = _diff_parts(old_content, new_content)
    output: list[str] = []
    width = len(str(max(len(old_content.split("\n")), len(new_content.split("\n")))))
    old_line = new_line = 1
    last_was_change = False
    first_changed: int | None = None

    def number(value: int) -> str:
        return str(value).rjust(width)

    for index, (kind, lines) in enumerate(parts):
        raw = [line[:-1] if line.endswith("\n") else line for line in lines]
        if kind != "equal":
            if first_changed is None:
                first_changed = new_line
            for line in raw:
                if kind == "added":
                    output.append(f"+{number(new_line)} {line}")
                    new_line += 1
                else:
                    output.append(f"-{number(old_line)} {line}")
                    old_line += 1
            last_was_change = True
            continue
        next_is_change = index < len(parts) - 1 and parts[index + 1][0] != "equal"
        if last_was_change and next_is_change:
            if len(raw) <= context_lines * 2:
                shown, skipped, tail = raw, 0, []
            else:
                shown, skipped, tail = raw[:context_lines], len(raw) - 2 * context_lines, raw[-context_lines:]
            for line in shown:
                output.append(f" {number(old_line)} {line}")
                old_line += 1
                new_line += 1
            if skipped:
                output.append(f" {''.rjust(width)} ...")
                old_line += skipped
                new_line += skipped
                for line in tail:
                    output.append(f" {number(old_line)} {line}")
                    old_line += 1
                    new_line += 1
        elif last_was_change:
            shown = raw[:context_lines]
            for line in shown:
                output.append(f" {number(old_line)} {line}")
                old_line += 1
                new_line += 1
            skipped = len(raw) - len(shown)
            if skipped > 0:
                output.append(f" {''.rjust(width)} ...")
                old_line += skipped
                new_line += skipped
        elif next_is_change:
            skipped = max(0, len(raw) - context_lines)
            if skipped > 0:
                output.append(f" {''.rjust(width)} ...")
                old_line += skipped
                new_line += skipped
            for line in raw[skipped:]:
                output.append(f" {number(old_line)} {line}")
                old_line += 1
                new_line += 1
        else:
            old_line += len(raw)
            new_line += len(raw)
        last_was_change = False
    return "\n".join(output), first_changed


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    """A standard unified patch with file headers only."""
    diff = difflib.unified_diff(
        _LINES_WITH_ENDINGS.findall(old_content),
        _LINES_WITH_ENDINGS.findall(new_content),
        fromfile=path,
        tofile=path,
        n=context_lines,
    )
    return "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in diff
    )


def _is_single_edit(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("oldText"), str)
        and isinstance(value.get("newText"), str)
    )


def prepare_edit_arguments(args: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Pi's ``prepareEditArguments``: accept stringified, single-object and legacy edits."""
    edits = args.get("edits")
    if isinstance(edits, str):
        try:
            parsed = json.loads(edits)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            args["edits"] = parsed
        elif _is_single_edit(parsed):
            args["edits"] = [parsed]
    elif _is_single_edit(edits):
        args["edits"] = [edits]

    old_text, new_text = args.get("oldText"), args.get("newText")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return args
    merged = list(args["edits"]) if isinstance(args.get("edits"), list) else []  # type: ignore[arg-type]
    merged.append({"oldText": old_text, "newText": new_text})
    rest = {key: value for key, value in args.items() if key not in ("oldText", "newText")}
    return {**rest, "edits": merged}


class EditTool(Tool):
    name = "edit"
    argument_conversion = "typebox"
    description = (
        "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
        "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
        "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged "
        "regions just to connect distant changes."
    )
    prompt_snippet = (
        "Make precise file edits with exact text replacement, including multiple disjoint edits in one call"
    )
    guidelines = (
        "Use edit for precise changes (edits[].oldText must match exactly)",
        "When changing multiple separate locations in one file, use one edit call with multiple entries in "
        "edits[] instead of multiple edit calls",
        "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do "
        "not emit overlapping or nested edits. Merge nearby changes into one edit.",
        "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with "
        "large unchanged regions.",
    )
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "required": ["path", "edits"],
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["oldText", "newText"],
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": (
                                "Exact text for one targeted replacement. It must be unique in the original "
                                "file and must not overlap with any other edits[].oldText in the same call."
                            ),
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text for this targeted edit.",
                        },
                    },
                },
                "description": (
                    "One or more targeted replacements. Each edit is matched against the original file, not "
                    "incrementally. Do not include overlapping or nested edits. If two changes touch the same "
                    "block or nearby lines, merge them into one edit instead."
                ),
            },
        },
    }

    def __init__(self, cwd: Path | str) -> None:
        self.cwd = Path(cwd)

    def prepare_arguments(self, args: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return prepare_edit_arguments(args)

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        del tool_call_id, on_update
        raw_edits = args.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
        path = str(args["path"])
        edits = [
            Edit(str(item["oldText"]), str(item["newText"])) for item in raw_edits if isinstance(item, dict)
        ]
        absolute_path = resolve_to_cwd(path, self.cwd)

        def throw_if_aborted() -> None:
            if cancellation is not None and cancellation.aborted:
                raise OperationAbortedError("Operation aborted")

        async with file_mutation_queue(absolute_path):
            throw_if_aborted()
            try:
                os.stat(absolute_path)
                if not os.access(absolute_path, os.R_OK | os.W_OK):
                    raise PermissionError(13, "Permission denied")
            except OSError as error:
                throw_if_aborted()
                raise ValueError(f"Could not edit file: {path}. Error code: {error_code(error)}.") from None
            throw_if_aborted()

            try:
                with open(absolute_path, "rb") as file:
                    raw_content = file.read().decode("utf-8", errors="replace")
            except IsADirectoryError as error:
                raise node_fs_error(error, "read") from None
            except OSError as error:
                raise node_fs_error(error, "open", absolute_path) from None
            throw_if_aborted()

            # Strip the BOM before matching: the model will not include an invisible BOM in oldText.
            bom, content = split_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits(normalized_content, edits, path)
            throw_if_aborted()

            write_text_file(absolute_path, bom + restore_line_endings(applied.new_content, original_ending))
            throw_if_aborted()

            diff, first_changed = generate_diff_string(applied.base_content, applied.new_content)
            patch = generate_unified_patch(path, applied.base_content, applied.new_content)
            details: dict[str, JsonValue] = {"diff": diff, "patch": patch}
            if first_changed is not None:
                details["firstChangedLine"] = first_changed
            return ToolResult(
                content=[TextBlock(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details=details,
            )
