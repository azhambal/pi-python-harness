"""Tool-call argument parsing, translated from Pi's ``utils/json-parse.ts``."""

from __future__ import annotations

import math
import re
from typing import cast

from pi_agent._js import JS_WHITESPACE
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
    if not text or not text.strip(JS_WHITESPACE):
        return {}, None
    try:
        parsed = parse_json_with_repair(text)
        if not isinstance(parsed, dict):
            raise JSONParseError("Tool arguments must be a JSON object")
        return parsed, None
    except JSONParseError as error:
        return {}, f"{error}\n\nReceived arguments:\n{text}"


def parse_streaming_json(text: str | None) -> dict[str, JsonValue]:
    """Pi's ``parseStreamingJson``, including partial-json 0.1.7's permissive fallback.

    Like Pi's generic TypeScript return type, the annotation describes normal tool
    objects; at runtime unfinished arrays/scalars also pass through until the final
    ``parseToolCallArguments`` validates them. Progress observers see the same value.
    """
    if not text or not text.strip(JS_WHITESPACE):
        return {}
    try:
        return cast(dict[str, JsonValue], parse_json_with_repair(text))
    except JSONParseError:
        try:
            parsed = _PartialParser(text).parse_any()
        except (JSONParseError, RecursionError):
            try:
                parsed = _PartialParser(repair_json(text)).parse_any()
            except (JSONParseError, RecursionError):
                return {}
    return cast(dict[str, JsonValue], {} if parsed is None else parsed)


class _PartialParser:
    """Translation of partial-json 0.1.7 with Allow.ALL, as imported by Pi.

    Copyright (c) 2023 Promplate Dev Team. MIT license: data/licenses/partial-json.txt.
    This intentionally preserves permissive cases (partial literals, missing colons,
    ignored trailing text) rather than imposing the completed JSON grammar.
    """

    def __init__(self, text: str) -> None:
        encoded = text.strip(JS_WHITESPACE).encode("utf-16-le", errors="surrogatepass")
        self.text = "".join(
            chr(int.from_bytes(encoded[i : i + 2], "little")) for i in range(0, len(encoded), 2)
        )
        self.index = 0

    def char(self) -> str:
        return self.text[self.index : self.index + 1]

    def blank(self) -> None:
        while self.char() and self.char() in " \n\r\t":
            self.index += 1

    def substring(self, start: int, end: int) -> str:
        # JavaScript String.substring clamps and swaps its arguments.
        start, end = sorted((max(0, start), max(0, end)))
        return self.text[start:end]

    def parse_any(self) -> JsonValue:
        self.blank()
        if self.index >= len(self.text):
            raise JSONParseError("Unexpected end of partial input")
        char = self.char()
        if char == '"':
            return self.string()
        if char == "{":
            return self.object()
        if char == "[":
            return self.array()
        tail = self.text[self.index :]
        for literal, value in (
            ("null", None),
            ("true", True),
            ("false", False),
            ("Infinity", math.inf),
            ("-Infinity", -math.inf),
            ("NaN", math.nan),
        ):
            if tail.startswith(literal) or (
                literal.startswith(tail) and (literal != "-Infinity" or len(tail) > 1)
            ):
                self.index += len(literal)
                return value
        return self.number()

    def string(self) -> JsonValue:
        start = self.index
        escape = False
        self.index += 1
        while self.index < len(self.text) and (
            self.char() != '"' or (escape and self.text[self.index - 1] == "\\")
        ):
            escape = not escape if self.char() == "\\" else False
            self.index += 1
        if self.char() == '"':
            self.index += 1
            return self.decode_string(self.text[start : self.index - int(escape)])
        try:
            return self.decode_string(self.text[start : self.index - int(escape)] + '"')
        except JSONParseError:
            return self.decode_string(self.substring(start, self.text.rfind("\\")) + '"')

    @staticmethod
    def decode_string(text: str) -> JsonValue:
        value = js_json_parse(text)
        if isinstance(value, str):
            return value.encode("utf-16-le", errors="surrogatepass").decode(
                "utf-16-le", errors="surrogatepass"
            )
        return value

    def object(self) -> dict[str, JsonValue]:
        self.index += 1
        self.blank()
        obj: dict[str, JsonValue] = {}
        try:
            while self.char() != "}":
                self.blank()
                if self.index >= len(self.text):
                    return obj
                key = self.string()
                self.blank()
                self.index += 1  # partial-json skips the colon without validating it.
                try:
                    value = self.parse_any()
                except JSONParseError:
                    return obj
                if isinstance(key, str) and key != "__proto__":
                    obj[key] = value
                self.blank()
                if self.char() == ",":
                    self.index += 1
        except JSONParseError:
            return obj
        self.index += 1
        return obj

    def array(self) -> list[JsonValue]:
        self.index += 1
        arr: list[JsonValue] = []
        try:
            while self.char() != "]":
                arr.append(self.parse_any())
                self.blank()
                if self.char() == ",":
                    self.index += 1
        except JSONParseError:
            return arr
        self.index += 1
        return arr

    def number(self) -> JsonValue:
        start = self.index
        if start == 0:
            if self.text == "-":
                raise JSONParseError("Incomplete minus")
            try:
                return js_json_parse(self.text)
            except JSONParseError:
                return js_json_parse(self.substring(0, self.text.rfind("e")))
        if self.char() == "-":
            self.index += 1
        while self.char() and self.char() not in ",]}":
            self.index += 1
        token = self.text[start : self.index]
        try:
            return js_json_parse(token)
        except JSONParseError:
            if token == "-":
                raise
            return js_json_parse(self.substring(start, self.text.rfind("e")))
