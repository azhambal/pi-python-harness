"""JavaScript value semantics needed to reproduce Pi's model-facing text."""

from __future__ import annotations

import math
import os
import re
import time
from decimal import ROUND_HALF_UP, Decimal

JS_WHITESPACE = "\t\n\v\f\r                  　﻿"
_DECIMAL_LITERAL = re.compile(r"^[+-]?(?:Infinity|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$")
_PREFIXED_LITERAL = re.compile(r"^0(?:[xX][0-9a-fA-F]+|[oO][0-7]+|[bB][01]+)$")


def is_js_number(value: object) -> bool:
    """``typeof value === "number"`` for JSON-decoded Python values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def to_number(text: str) -> int | float | None:
    """``+text``; None stands for NaN."""
    stripped = text.strip(JS_WHITESPACE)
    if stripped == "":
        return 0
    if _PREFIXED_LITERAL.match(stripped):
        return int(stripped, 0)
    if not _DECIMAL_LITERAL.match(stripped):
        return None
    if stripped.lstrip("+-") == "Infinity":
        return -math.inf if stripped.startswith("-") else math.inf
    value = float(stripped)
    return int(value) if value.is_integer() and abs(value) < 2**53 else value


def number_to_string(value: int | float) -> str:
    """``Number.prototype.toString()``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value) if abs(value) < 10**21 else number_to_string(float(value))
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "0"
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    text = repr(value)
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    exponent_value = int(exponent)
    if -7 < exponent_value < 21:
        return format(value, "f").rstrip("0").rstrip(".")
    sign = "+" if exponent_value > 0 else "-"
    return f"{mantissa}e{sign}{abs(exponent_value)}"


_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _quote(text: str) -> str:
    out = ['"']
    for char in text:
        escaped = _ESCAPES.get(char)
        if escaped is not None:
            out.append(escaped)
        elif ord(char) < 0x20 or 0xD800 <= ord(char) <= 0xDFFF:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def json_stringify(value: object, indent: int | None = None) -> str:
    """``JSON.stringify(value, null, indent)`` for JSON-compatible values."""
    gap = " " * indent if indent else ""

    def encode(item: object, current: str) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if is_js_number(item):
            return number_to_string(item) if math.isfinite(item) else "null"  # type: ignore[arg-type]
        if isinstance(item, str):
            return _quote(item)
        inner = current + gap
        if isinstance(item, (list, tuple)):
            if not item:
                return "[]"
            parts = [encode(element, inner) for element in item]
            if not gap:
                return "[" + ",".join(parts) + "]"
            return "[\n" + inner + (",\n" + inner).join(parts) + "\n" + current + "]"
        if isinstance(item, dict):
            if not item:
                return "{}"
            separator = ": " if gap else ":"
            parts = [_quote(str(key)) + separator + encode(element, inner) for key, element in item.items()]
            if not gap:
                return "{" + ",".join(parts) + "}"
            return "{\n" + inner + (",\n" + inner).join(parts) + "\n" + current + "}"
        return _quote(str(item))

    return encode(value, "")


def utf16_length(text: str) -> int:
    """``text.length``: UTF-16 code units, so astral characters count twice."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def utf16_slice(text: str, end: int) -> str:
    """``text.slice(0, end)``; a surrogate pair cut in half becomes U+FFFD."""
    units = text.encode("utf-16-le", errors="surrogatepass")[: max(0, end) * 2]
    return units.decode("utf-16-le", errors="replace")


def uuidv7() -> str:
    """Pi's ``uuidv7``: 48-bit millisecond timestamp, version 7, RFC 4122 variant, random rest."""
    value = bytearray((time.time_ns() // 1_000_000).to_bytes(6, "big") + os.urandom(10))
    value[6] = 0x70 | (value[6] & 0x0F)
    value[8] = 0x80 | (value[8] & 0x3F)
    text = value.hex()
    return f"{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"


def to_fixed(value: float, digits: int) -> str:
    """``Number.prototype.toFixed``: exact binary value, ties round up."""
    return str(Decimal(value).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def _imul(a: int, b: int) -> int:
    """``Math.imul``: 32-bit signed multiplication."""
    product = ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF
    return product - (1 << 32) if product & 0x80000000 else product


def _base36(value: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    text = ""
    while True:
        value, remainder = divmod(value, 36)
        text = digits[remainder] + text
        if value == 0:
            return text


def short_hash(text: str) -> str:
    """Pi's ``shortHash`` (``packages/ai/src/utils/hash.ts``) over UTF-16 code units."""
    h1, h2 = 0xDEADBEEF, 0x41C6CE57
    units = text.encode("utf-16-le", errors="surrogatepass")
    for index in range(0, len(units), 2):
        unit = units[index] | units[index + 1] << 8
        h1 = _imul(h1 ^ unit, 2654435761)
        h2 = _imul(h2 ^ unit, 1597334677)
    h1 = _imul(h1 ^ ((h1 & 0xFFFFFFFF) >> 16), 2246822507) ^ _imul(h2 ^ ((h2 & 0xFFFFFFFF) >> 13), 3266489909)
    h2 = _imul(h2 ^ ((h2 & 0xFFFFFFFF) >> 16), 2246822507) ^ _imul(h1 ^ ((h1 & 0xFFFFFFFF) >> 13), 3266489909)
    return _base36(h2 & 0xFFFFFFFF) + _base36(h1 & 0xFFFFFFFF)


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def sanitize_surrogates(text: str) -> str:
    """Pi's ``sanitizeSurrogates``: drop unpaired surrogates (Python never pairs them)."""
    return _LONE_SURROGATE.sub("", text)
