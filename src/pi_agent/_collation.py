"""Node's ``String.prototype.localeCompare`` as a Python sort key.

Pi sorts ``ls`` entries with ICU's root collation (``en-US``): punctuation
before digits before letters, digits compared one by one (``10`` < ``9``),
accents and case only breaking ties. ``data/collation.json`` holds per-character
ICU ranks dumped from Node by ``scripts/pi_reference/dump_collation.ts``; the
key compares them level by level as the Unicode Collation Algorithm does.
Characters outside the table sort after it by code point, like ICU's implicit
weights for unlisted scripts.
"""

from __future__ import annotations

import json
import unicodedata
from functools import cache
from importlib.resources import files

_IMPLICIT_BASE = 1 << 20


#: Offsets that keep an expansion after the letters it spells (``ae`` < ``æ``).
_EXPANSION_TERTIARY = 1000
_EXPANSION_SECONDARY = 1000

Weight = tuple[int, int, int]


@cache
def _raw() -> dict:
    return json.loads(files("pi_agent").joinpath("data/collation.json").read_text("utf-8"))


@cache
def _atomic() -> frozenset[str]:
    return frozenset(chr(int(code, 16)) for code in _raw()["atomic"])


@cache
def _table() -> dict[int, tuple[Weight, ...]]:
    raw = _raw()
    weights: dict[int, Weight] = {int(code, 16): (p, s, t) for code, (p, s, t) in raw["weights"].items()}
    table: dict[int, tuple[Weight, ...]] = {code: (weight,) for code, weight in weights.items()}
    for code, (spelled, accented) in raw["expansions"].items():
        own_tertiary = weights[int(code, 16)][2]
        expanded = [
            (p, s, t + _EXPANSION_TERTIARY + own_tertiary)
            for p, s, t in (weights.get(ord(char), _implicit(char)) for char in spelled)
        ]
        if accented:
            p, s, t = expanded[0]
            expanded[0] = (p, s + _EXPANSION_SECONDARY, t)
        table[int(code, 16)] = tuple(expanded)
    return table


def _implicit(char: str) -> Weight:
    return (_IMPLICIT_BASE + ord(char), 1, 1)


def collation_key(text: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Strings compare as ``localeCompare`` orders them; ties stay in input order."""
    table = _table()
    primary: list[int] = []
    secondary: list[int] = []
    tertiary: list[int] = []
    atomic = _atomic()
    # ICU compares canonically equivalent strings equally (é == e + U+0301), and
    # sorts some precomposed letters (й) as letters of their own.
    for composed in unicodedata.normalize("NFC", text):
        for char in composed if composed in atomic else unicodedata.normalize("NFD", composed):
            for p, s, t in table.get(ord(char)) or (_implicit(char),):
                if p == 0 and s == 1 and t == 1:
                    continue  # Completely ignorable, such as control characters.
                if p:
                    primary.append(p)
                secondary.append(s)
                tertiary.append(t)
    return tuple(primary), tuple(secondary), tuple(tertiary)
