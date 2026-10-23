"""JavaScript ``JSON.parse`` with V8's error messages.

Pi reports malformed tool-call arguments back to the model with the message of
``JSON.parse``; this module reproduces V8's ``JsonParser`` error reporting
(message template, UTF-16 position, line and column, source context) so the
model reads the same text. Valid input is decoded with :mod:`json`.
"""

from __future__ import annotations

import json
import math

from pi_agent.types import JsonValue

_WHITESPACE = " \t\n\r"
_SPECIAL_STRINGS = ("[object Object]", "undefined", "Infinity", "NaN")
_MAX_CONTEXT = 10
_MIN_LENGTH_FOR_CONTEXT = _MAX_CONTEXT * 2 + 1
_VALID_ESCAPES = '"\\/bfnrt'
_HEX = "0123456789abcdefABCDEF"

_EXPECTED_PROPERTY_NAME_OR_RBRACE = "Expected property name or '}' in JSON"
_EXPECTED_COMMA_OR_RBRACE = "Expected ',' or '}' after property value in JSON"
_EXPECTED_COMMA_OR_RBRACK = "Expected ',' or ']' after array element in JSON"
_EXPECTED_COLON = "Expected ':' after property name in JSON"
_EXPECTED_DOUBLE_QUOTED_PROPERTY_NAME = "Expected double-quoted property name in JSON"
_UNEXPECTED_NON_WHITESPACE = "Unexpected non-whitespace character after JSON"
_UNTERMINATED_STRING = "Unterminated string in JSON"
_BAD_CONTROL_CHARACTER = "Bad control character in string literal in JSON"
_BAD_ESCAPED_CHARACTER = "Bad escaped character in JSON"
_BAD_UNICODE_ESCAPE = "Bad Unicode escape in JSON"
_NO_NUMBER_AFTER_MINUS = "No number after minus sign in JSON"
_UNTERMINATED_FRACTION = "Unterminated fractional number in JSON"
_EXPONENT_MISSING_NUMBER = "Exponent part is missing a number in JSON"


class JSONParseError(ValueError):
    """A ``SyntaxError`` from ``JSON.parse``; ``str()`` is V8's message."""


class _Parser:
    def __init__(self, text: str) -> None:
        # V8 works on UTF-16 code units: positions and context count them.
        self.source = text
        self.units = text.encode("utf-16-le", errors="surrogatepass")
        self.length = len(self.units) // 2
        self.pos = 0

    def unit(self, index: int) -> str | None:
        if index >= self.length:
            return None
        return chr(self.units[2 * index] | self.units[2 * index + 1] << 8)

    def peek(self) -> str | None:
        return self.unit(self.pos)

    def skip_whitespace(self) -> str | None:
        while (char := self.peek()) is not None and char in _WHITESPACE:
            self.pos += 1
        return char

    def _location(self) -> str:
        line = 1
        last_break = 0
        index = 0
        while index < self.pos:
            char = self.unit(index)
            if char == "\r" and index < self.pos - 1 and self.unit(index + 1) == "\n":
                index += 1
                char = "\n"
            if char in ("\r", "\n"):
                line += 1
                last_break = index + 1
            index += 1
        column = 1 + index - last_break
        return f"at position {self.pos} (line {line} column {column})"

    def _substring(self, start: int, end: int) -> str:
        return self.units[2 * start : 2 * end].decode("utf-16-le", errors="surrogatepass")

    def fail(self, message: str | None = None) -> JSONParseError:
        """V8's ``ReportUnexpectedToken`` for the token at the cursor."""
        if message is not None:
            return JSONParseError(f"{message} {self._location()}")
        char = self.peek()
        if char is None:
            return JSONParseError("Unexpected end of JSON input")
        if char.isdigit() and char.isascii() or char == "-":
            return JSONParseError(f"Unexpected number in JSON {self._location()}")
        if char == '"':
            return JSONParseError(f"Unexpected string in JSON {self._location()}")
        if self.source in _SPECIAL_STRINGS:
            return JSONParseError(f'"{self.source}" is not valid JSON')
        if self.length < _MIN_LENGTH_FOR_CONTEXT:
            return JSONParseError(f"Unexpected token '{char}', \"{self.source}\" is not valid JSON")
        if self.pos < _MAX_CONTEXT:
            context = self._substring(0, self.pos + _MAX_CONTEXT)
            return JSONParseError(f"Unexpected token '{char}', \"{context}\"... is not valid JSON")
        if self.pos < self.length - _MAX_CONTEXT:
            context = self._substring(self.pos - _MAX_CONTEXT, self.pos + _MAX_CONTEXT)
            return JSONParseError(f"Unexpected token '{char}', ...\"{context}\"... is not valid JSON")
        context = self._substring(self.pos - _MAX_CONTEXT, self.length)
        return JSONParseError(f"Unexpected token '{char}', ...\"{context}\" is not valid JSON")

    def parse(self) -> None:
        self.value()
        if self.skip_whitespace() is not None:
            raise self.fail(_UNEXPECTED_NON_WHITESPACE)

    def value(self) -> None:
        char = self.skip_whitespace()
        if char == "{":
            self.object()
        elif char == "[":
            self.array()
        elif char == '"':
            self.string()
        elif char == "-" or (char is not None and "0" <= char <= "9"):
            self.number()
        elif char == "t":
            self.literal("true")
        elif char == "f":
            self.literal("false")
        elif char == "n":
            self.literal("null")
        else:
            raise self.fail()

    def object(self) -> None:
        self.pos += 1
        char = self.skip_whitespace()
        if char == "}":
            self.pos += 1
            return
        if char != '"':
            raise self.fail(_EXPECTED_PROPERTY_NAME_OR_RBRACE)
        while True:
            self.string()
            if self.skip_whitespace() != ":":
                raise self.fail(_EXPECTED_COLON)
            self.pos += 1
            self.value()
            char = self.skip_whitespace()
            if char == ",":
                self.pos += 1
                if self.skip_whitespace() != '"':
                    raise self.fail(_EXPECTED_DOUBLE_QUOTED_PROPERTY_NAME)
                continue
            if char == "}":
                self.pos += 1
                return
            raise self.fail(_EXPECTED_COMMA_OR_RBRACE)

    def array(self) -> None:
        self.pos += 1
        if self.skip_whitespace() == "]":
            self.pos += 1
            return
        while True:
            self.value()
            char = self.skip_whitespace()
            if char == ",":
                self.pos += 1
                continue
            if char == "]":
                self.pos += 1
                return
            raise self.fail(_EXPECTED_COMMA_OR_RBRACK)

    def string(self) -> None:
        self.pos += 1
        while True:
            char = self.peek()
            if char is None:
                raise self.fail(_UNTERMINATED_STRING)
            if char == '"':
                self.pos += 1
                return
            if char == "\\":
                self.pos += 1
                escape = self.peek()
                if escape is None:
                    raise self.fail()
                if escape == "u":
                    for _ in range(4):
                        self.pos += 1
                        digit = self.peek()
                        if digit is None or digit not in _HEX:
                            raise self.fail(_BAD_UNICODE_ESCAPE)
                elif escape not in _VALID_ESCAPES:
                    raise self.fail(_BAD_ESCAPED_CHARACTER)
                self.pos += 1
                continue
            if ord(char) < 0x20:
                raise self.fail(_BAD_CONTROL_CHARACTER)
            self.pos += 1

    def _digits(self) -> None:
        while (char := self.peek()) is not None and "0" <= char <= "9":
            self.pos += 1

    def number(self) -> None:
        if self.peek() == "-":
            self.pos += 1
            char = self.peek()
            if char is None or not "0" <= char <= "9":
                raise self.fail(_NO_NUMBER_AFTER_MINUS)
        if self.peek() == "0":
            self.pos += 1
            char = self.peek()
            if char is not None and "0" <= char <= "9":
                raise self.fail()
        else:
            self._digits()
        if self.peek() == ".":
            self.pos += 1
            char = self.peek()
            if char is None or not "0" <= char <= "9":
                raise self.fail(_UNTERMINATED_FRACTION)
            self._digits()
        if self.peek() in ("e", "E"):
            self.pos += 1
            if self.peek() in ("+", "-"):
                self.pos += 1
            char = self.peek()
            if char is None or not "0" <= char <= "9":
                raise self.fail(_EXPONENT_MISSING_NUMBER)
            self._digits()

    def literal(self, word: str) -> None:
        # The first character matched already, as in V8's ScanLiteral.
        self.pos += 1
        for expected in word[1:]:
            char = self.peek()
            if char is None:
                raise self.fail()
            if char != expected:
                raise self.fail()
            self.pos += 1


def _finite_or_null(text: str) -> float | None:
    # JSON.parse yields Infinity for 1e999; Pi's journal stores it as null (JSON.stringify).
    value = float(text)
    return value if math.isfinite(value) else None


def js_json_parse(text: str) -> JsonValue:
    """``JSON.parse(text)``: the decoded value, or JSONParseError with V8's message."""
    _Parser(text).parse()
    return json.loads(text, parse_float=_finite_or_null)
