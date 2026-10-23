"""Tool-call argument parsing, translated from Pi's ``utils/json-parse.ts``."""

from __future__ import annotations

import json
import re

from pi_agent._js_json import JSONParseError, js_json_parse
from pi_agent.types import JsonValue

_VALID_JSON_ESCAPES = frozenset('"\\/bfnrt')
_HEX4 = re.compile(r"[0-9a-fA-F]{4}")
_CONTROL_ESCAPES = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def repair_json(text: str) -> str:
    """Pi's ``repairJson``: escape raw control characters and invalid escapes inside strings."""
    repaired: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue
        if char == '"':
            repaired.append(char)
            in_string = False
            index += 1
            continue
        if char == "\\":
            next_char = text[index + 1] if index + 1 < len(text) else None
            if next_char is None:
                repaired.append("\\\\")
                index += 1
                continue
            if next_char == "u":
                digits = text[index + 2 : index + 6]
                if _HEX4.fullmatch(digits):
                    repaired.append(f"\\u{digits}")
                    index += 6
                    continue
            if next_char in _VALID_JSON_ESCAPES:
                repaired.append(f"\\{next_char}")
                index += 2
                continue
            repaired.append("\\\\")
            index += 1
            continue
        if ord(char) <= 0x1F:
            repaired.append(_CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}"))
        else:
            repaired.append(char)
        index += 1
    return "".join(repaired)


def parse_json_with_repair(text: str) -> JsonValue:
    """Pi's ``parseJsonWithRepair``; errors carry JavaScript's ``JSON.parse`` message."""
    try:
        return js_json_parse(text)
    except JSONParseError:
        repaired = repair_json(text)
        if repaired != text:
            return js_json_parse(repaired)
        raise


def parse_tool_call_arguments(text: str | None) -> tuple[dict[str, JsonValue], str | None]:
    """Pi's ``parseToolCallArguments``: (arguments, argumentsError) of a finished call."""
    if not text or not text.strip():
        return {}, None
    try:
        parsed = parse_json_with_repair(text)
        if not isinstance(parsed, dict):
            raise JSONParseError("Tool arguments must be a JSON object")
        return parsed, None
    except JSONParseError as error:
        return {}, f"{error}\n\nReceived arguments:\n{text}"


def parse_streaming_json(text: str | None) -> dict[str, JsonValue]:
    """Best-effort arguments of an unfinished call, for progress events only."""
    if not text or not text.strip():
        return {}
    try:
        parsed = parse_json_with_repair(text)
    except JSONParseError:
        parsed = _parse_partial(repair_json(text))
    return parsed if isinstance(parsed, dict) else {}


def _parse_partial(text: str) -> JsonValue:
    """Close open strings, arrays and objects of a JSON prefix, then parse it."""
    closers: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            closers.append("}" if char == "{" else "]")
        elif char in "}]" and closers:
            closers.pop()
    candidate = text + ('"' if in_string else "")
    for _ in range(len(candidate) + 1):
        attempt = candidate.rstrip().rstrip(",:").rstrip() + "".join(reversed(closers))
        try:
            return json.loads(attempt)
        except ValueError:
            # Drop the incomplete trailing token and retry.
            trimmed = re.sub(r'(,\s*"[^"]*"\s*:?\s*[^,{}\[\]]*|,?\s*[^,{}\[\]]+)$', "", candidate)
            if trimmed == candidate:
                return {}
            candidate = trimmed
    return {}
