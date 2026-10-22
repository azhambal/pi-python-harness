"""Validated exact and fuzzy multi-edits with atomic file replacement."""

from __future__ import annotations

import difflib
import os
import unicodedata
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pi_agent.cancellation import Cancellation
from pi_agent.tools._atomic import atomic_write, read_file_snapshot
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback, resolve_path
from pi_agent.types import JsonValue, TextBlock

_UTF8_BOM = "\ufeff"
_JS_TRAILING_SPACE = " \t\v\f\r\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_FUZZY_TRANSLATION = str.maketrans(
    {
        **dict.fromkeys("\u2018\u2019\u201a\u201b", "'"),
        **dict.fromkeys("\u201c\u201d\u201e\u201f", '"'),
        **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"),
        **dict.fromkeys(
            "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000", " "
        ),
    }
)


def normalize_for_fuzzy_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "\n".join(line.rstrip(_JS_TRAILING_SPACE) for line in text.split("\n")).translate(
        _FUZZY_TRANSLATION
    )


def split_bom(raw: str) -> tuple[str, str]:
    """Split a leading UTF-8 BOM off; returns (bom, text)."""
    if raw.startswith(_UTF8_BOM):
        return _UTF8_BOM, raw[len(_UTF8_BOM) :]
    return "", raw


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


@dataclass
class Edit:
    old_text: str
    new_text: str


@dataclass
class AppliedEdits:
    base_content: str
    new_content: str


def _find_occurrences(content: str, needle: str) -> list[int]:
    """All start offsets of ``needle``, including overlapping ones.

    ``str.count`` counts only non-overlapping matches, so ``"aaa"`` would
    look like a single match for ``"aa"``. Scanning with a step of one keeps
    genuinely ambiguous regions ambiguous, which the validation below
    rejects.
    """
    positions: list[int] = []
    start = 0
    while True:
        found = content.find(needle, start)
        if found == -1:
            return positions
        positions.append(found)
        start = found + 1


def apply_edits(normalized_content: str, edits: list[Edit], path: str) -> AppliedEdits:
    """Match every edit against one base; overlay fuzzy changes on touched lines.

    Raises ValueError with a Pi-shaped message for empty old_text, missing
    text, ambiguous matches (including overlapping ones), overlaps between
    edits, or a no-op result.
    """
    normalized = [
        Edit(old_text=normalize_to_lf(e.old_text), new_text=normalize_to_lf(e.new_text)) for e in edits
    ]
    total = len(normalized)
    for i, edit in enumerate(normalized):
        if not edit.old_text:
            raise ValueError(
                f"oldText must not be empty in {path}."
                if total == 1
                else f"edits[{i}].oldText must not be empty in {path}."
            )

    fuzzy_content = normalize_for_fuzzy_match(normalized_content)
    used_fuzzy = any(edit.old_text not in normalized_content for edit in normalized)
    base = fuzzy_content if used_fuzzy else normalized_content
    matched: list[tuple[int, int, int, str]] = []  # (edit_index, start, length, new_text)
    for i, edit in enumerate(normalized):
        fuzzy_old = normalize_for_fuzzy_match(edit.old_text)
        needle = fuzzy_old if used_fuzzy else edit.old_text
        if not needle:
            raise ValueError(
                f"edits[{i}].oldText is empty after fuzzy normalization in {path}; include context."
            )
        occurrences = _find_occurrences(base, needle)
        if not occurrences:
            raise ValueError(
                f"Could not find the exact text in {path}. The old text must match exactly "
                "including all whitespace and newlines."
                if total == 1
                else f"Could not find edits[{i}] in {path}. The oldText must match exactly "
                "including all whitespace and newlines."
            )
        # Even an exact occurrence can have an ambiguous normalized twin.
        ambiguity = len(_find_occurrences(fuzzy_content, fuzzy_old)) if fuzzy_old else len(occurrences)
        count = max(len(occurrences), ambiguity)
        if count > 1:
            raise ValueError(
                f"Found {count} occurrences of the text in {path}. The text must be unique. "
                "Please provide more context to make it unique."
                if total == 1
                else f"Found {count} occurrences of edits[{i}] in {path}. Each oldText must be "
                "unique. Please provide more context to make it unique."
            )
        matched.append((i, occurrences[0], len(needle), edit.new_text))

    matched.sort(key=lambda item: item[1])
    # The zip pairs neighbours on purpose; the two lists differ in length by one.
    for previous, current in zip(matched, matched[1:], strict=False):
        if previous[1] + previous[2] > current[1]:
            raise ValueError(
                f"edits[{previous[0]}] and edits[{current[0]}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    if used_fuzzy:
        new_content = _preserve_unchanged_lines(normalized_content, base, matched)
    else:
        new_content = _replace_ranges(base, matched)

    if new_content == normalized_content:
        raise ValueError(
            f"No changes made to {path}. The replacement produced identical content. This might "
            "indicate an issue with special characters or the text not existing as expected."
            if total == 1
            else f"No changes made to {path}. The replacements produced identical content."
        )
    return AppliedEdits(base_content=normalized_content, new_content=new_content)


def _replace_ranges(content: str, matched: list[tuple[int, int, int, str]], offset: int = 0) -> str:
    for _index, start, length, replacement in reversed(matched):
        start -= offset
        content = content[:start] + replacement + content[start + length :]
    return content


def _preserve_unchanged_lines(
    original: str,
    base: str,
    matched: list[tuple[int, int, int, str]],
) -> str:
    # Include the logical last line even when folding removes all its whitespace.
    original_parts = original.split("\n")
    base_parts = base.split("\n")
    original_lines = [line + "\n" for line in original_parts[:-1]] + original_parts[-1:]
    base_lines = [line + "\n" for line in base_parts[:-1]] + base_parts[-1:]
    if len(original_lines) != len(base_lines):
        raise ValueError(
            "Fuzzy normalization changed the line count; cannot safely preserve unchanged lines."
        )
    starts: list[int] = []
    offset = 0
    for line in base_lines:
        starts.append(offset)
        offset += len(line)
    groups: list[tuple[int, int, list[tuple[int, int, int, str]]]] = []
    for replacement in matched:
        _, start, length, _ = replacement
        first = bisect_right(starts, start) - 1
        end = bisect_right(starts, start + length - 1)
        if groups and first < groups[-1][1]:
            group_first, group_end, members = groups[-1]
            members.append(replacement)
            groups[-1] = (group_first, max(group_end, end), members)
        else:
            groups.append((first, end, [replacement]))
    parts: list[str] = []
    original_index = 0
    for first, end, members in groups:
        parts.extend(original_lines[original_index:first])
        parts.append(_replace_ranges("".join(base_lines[first:end]), members, starts[first]))
        original_index = end
    parts.extend(original_lines[original_index:])
    return "".join(parts)


def _physical_lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def first_changed_line(old_content: str, new_content: str) -> int | None:
    matcher = difflib.SequenceMatcher(
        None, _physical_lines(old_content), _physical_lines(new_content), autojunk=False
    )
    for tag, _old_start, _old_end, new_start, _new_end in matcher.get_opcodes():
        if tag != "equal":
            return new_start + 1
    return None


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    diff = difflib.unified_diff(
        _physical_lines(old_content),
        _physical_lines(new_content),
        fromfile=path,
        tofile=path,
        n=context_lines,
    )
    return "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in diff
    )


class EditTool(Tool):
    name = "edit"
    parallel_safe = True
    description = (
        "Edit a single file using text replacement, preferring exact matches. Fuzzy fallback handles "
        "trailing whitespace and Unicode-equivalent quotes, dashes and spacing. Every edits[].oldText must match a unique, "
        "non-overlapping region of the original file. If two changes affect the same block or nearby "
        "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
        "unchanged regions just to connect distant changes."
    )
    prompt_snippet = (
        "Make precise file edits with exact text replacement, including multiple disjoint edits in one call"
    )
    guidelines = (
        "Use edit for precise changes; prefer exact oldText, with whitespace/Unicode fuzzy fallback only when needed",
        "When changing multiple separate locations in one file, use one edit call with multiple entries in "
        "edits[] instead of multiple edit calls",
        "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do "
        "not emit overlapping or nested edits. Merge nearby changes into one edit.",
        "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with "
        "large unchanged regions.",
    )
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
            "edits": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "One or more targeted replacements. Each edit is matched against the original file, "
                    "not incrementally. Do not include overlapping or nested edits. If two changes touch "
                    "the same block or nearby lines, merge them into one edit instead."
                ),
                "items": {
                    "type": "object",
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
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
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
        raw_edits = args.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
        edits = []
        for item in raw_edits:
            if not isinstance(item, dict):
                raise ValueError("Edit replacements must contain string oldText and newText fields.")
            old_text, new_text = item.get("oldText"), item.get("newText")
            if not isinstance(old_text, str) or not isinstance(new_text, str):
                raise ValueError("Edit replacements must contain string oldText and newText fields.")
            edits.append(Edit(old_text=old_text, new_text=new_text))
        absolute_path = resolve_path(path, self.cwd)

        def throw_if_aborted() -> None:
            if cancellation is not None:
                cancellation.raise_if_aborted()

        throw_if_aborted()
        if not os.access(absolute_path, os.R_OK | os.W_OK):
            throw_if_aborted()
            raise PermissionError(f"Could not edit file: {path}. File is missing or not readable/writable.")
        throw_if_aborted()

        try:
            raw_bytes, version = read_file_snapshot(absolute_path)
            raw_content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Could not edit {path}: file is not valid UTF-8; no changes were written."
            ) from error
        throw_if_aborted()

        # Strip the BOM before matching: the model will not include an invisible
        # BOM in oldText, and it is re-attached before writing.
        bom, content = split_bom(raw_content)
        original_ending = detect_line_ending(content)
        normalized_content = normalize_to_lf(content)
        applied = apply_edits(normalized_content, edits, path)
        throw_if_aborted()

        final_content = bom + restore_line_endings(applied.new_content, original_ending)
        atomic_write(
            absolute_path, final_content.encode("utf-8"), cancellation=cancellation, expected=version
        )

        patch = generate_unified_patch(path, applied.base_content, applied.new_content)
        changed_line = first_changed_line(applied.base_content, applied.new_content)
        return ToolResult(
            content=[TextBlock(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
            details={"diff": patch, "patch": patch, "first_changed_line": changed_line},
            structured_content={"patch": patch, "first_changed_line": changed_line},
        )
