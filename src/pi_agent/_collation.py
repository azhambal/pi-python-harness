"""Node's localeCompare sort keys using the full ICU collation implementation.

A per-character table cannot reproduce contractions, kana or compatibility
characters. Use the ICU C API (the same collation engine as Node's Intl), with
its default variant sensitivity and the process locale (en-US for C/POSIX). Containers need libicui18n;
PI_ICU_LIBRARY can select the exact ICU library version used by the reference
Node build. No Node executable is used at runtime.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import os
import re
import sys
from functools import cache


class _Collator:
    def __init__(self) -> None:
        path = os.environ.get("PI_ICU_LIBRARY") or ctypes.util.find_library("icui18n")
        if path is None:
            raise RuntimeError("ICU collation requires libicui18n; install the ICU runtime in the container")
        self.library = ctypes.CDLL(path)
        match = re.search(r"(?:\.so\.|\.)(\d+)", path)
        suffixes = [f"_{match[1]}", ""] if match else ["", *(f"_{version}" for version in range(100, 49, -1))]
        for suffix in suffixes:
            try:
                open_collator = getattr(self.library, f"ucol_open{suffix}")
                self.sort_key = getattr(self.library, f"ucol_getSortKey{suffix}")
                break
            except AttributeError:
                continue
        else:
            raise RuntimeError(f"ICU collation API was not found in {path}")
        open_collator.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
        open_collator.restype = ctypes.c_void_p
        self.sort_key.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        self.sort_key.restype = ctypes.c_int32
        self.lower = getattr(self.library, f"u_strToLower{suffix}")
        self.lower.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lower.restype = ctypes.c_int32
        status = ctypes.c_int32(0)
        locale = (
            os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES") or os.environ.get("LANG") or "en_US"
        )
        locale = locale.split(".")[0]
        if locale in ("C", "POSIX"):
            locale = "en_US"
        self.handle = open_collator(locale.encode("ascii"), ctypes.byref(status))
        # Negative UErrorCode values are warnings (for example using the root locale).
        if status.value > 0 or not self.handle:
            raise RuntimeError(f"ICU could not open {locale} collation (error {status.value})")
        close = getattr(self.library, f"ucol_close{suffix}")
        close.argtypes = [ctypes.c_void_p]
        close.restype = None
        atexit.register(close, self.handle)

    def key(self, text: str, *, lowercase: bool = False) -> bytes:
        units = text.encode("utf-16-le" if sys.byteorder == "little" else "utf-16-be", errors="surrogatepass")
        source = ctypes.create_string_buffer(units)
        if lowercase:
            # JS toLowerCase uses root Unicode mappings, not locale-sensitive case folding.
            # Use ICU's Unicode version too (Python's may predate new scripts in Node).
            status = ctypes.c_int32(0)
            count = self.lower(None, 0, source, len(units) // 2, b"", ctypes.byref(status))
            if status.value not in (0, 15) and status.value > 0:  # U_BUFFER_OVERFLOW_ERROR = 15
                raise RuntimeError(f"ICU lowercase failed (error {status.value})")
            output = ctypes.create_string_buffer((count + 1) * 2)
            status = ctypes.c_int32(0)
            self.lower(output, count + 1, source, len(units) // 2, b"", ctypes.byref(status))
            if status.value > 0:
                raise RuntimeError(f"ICU lowercase failed (error {status.value})")
            units = output.raw[: count * 2]
            source = ctypes.create_string_buffer(units)
        size = self.sort_key(self.handle, source, len(units) // 2, None, 0)
        output = ctypes.create_string_buffer(size)
        self.sort_key(self.handle, source, len(units) // 2, output, size)
        return output.raw


@cache
def _collator() -> _Collator:
    return _Collator()


def collation_key(text: str) -> bytes:
    """ICU variant sort key; Python's stable sort preserves equal entries' input order."""
    return _collator().key(text)


def case_insensitive_collation_key(text: str) -> bytes:
    """Pi's ``name.toLowerCase().localeCompare(...)``, using ICU's Unicode data throughout."""
    return _collator().key(text, lowercase=True)
