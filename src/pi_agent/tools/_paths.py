"""Tool path resolution and Node-style filesystem errors, translated from Pi.

Mirrors ``core/tools/path-utils.ts`` and ``utils/paths.ts``: paths are
resolved lexically (Node ``path.resolve``), never through symlinks, after
normalizing Unicode spaces, stripping a leading ``@``, expanding ``~`` and
``file://`` URLs.
"""

from __future__ import annotations

import errno
import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

_UNICODE_SPACES = re.compile("[  -   　]")
_NARROW_NO_BREAK_SPACE = " "

#: libuv ``uv_strerror`` texts that Node puts into fs error messages.
_UV_MESSAGES = {
    errno.E2BIG: "argument list too long",
    errno.EACCES: "permission denied",
    errno.EAGAIN: "resource temporarily unavailable",
    errno.EBADF: "bad file descriptor",
    errno.EBUSY: "resource busy or locked",
    errno.EEXIST: "file already exists",
    errno.EFBIG: "file too large",
    errno.EINTR: "interrupted system call",
    errno.EINVAL: "invalid argument",
    errno.EIO: "i/o error",
    errno.EISDIR: "illegal operation on a directory",
    errno.ELOOP: "too many symbolic links encountered",
    errno.EMFILE: "too many open files",
    errno.ENAMETOOLONG: "name too long",
    errno.ENFILE: "file table overflow",
    errno.ENODEV: "no such device",
    errno.ENOENT: "no such file or directory",
    errno.ENOMEM: "not enough memory",
    errno.ENOSPC: "no space left on device",
    errno.ENOTDIR: "not a directory",
    errno.ENOTEMPTY: "directory not empty",
    errno.ENXIO: "no such device or address",
    errno.EPERM: "operation not permitted",
    errno.EROFS: "read-only file system",
    errno.ESPIPE: "invalid seek",
    errno.ETXTBSY: "text file is busy",
    errno.EXDEV: "cross-device link not permitted",
}


def error_code(error: OSError) -> str:
    """The ``error.code`` Node reports, such as ``ENOENT``."""
    return errno.errorcode.get(error.errno or 0, "UNKNOWN")


class NodeFsError(OSError):
    """An OSError that prints as Node's message, without Python's ``[Errno N]``."""

    def __str__(self) -> str:
        return self.strerror or ""


def node_fs_error(error: OSError, syscall: str, path: str | os.PathLike[str] | None = None) -> OSError:
    """An OSError whose message is Node's: ``CODE: description, syscall 'path'``."""
    code = error_code(error)
    description = _UV_MESSAGES.get(error.errno or 0, (error.strerror or "unknown error").lower())
    message = f"{code}: {description}, {syscall}"
    if path is not None:
        message += f" '{os.fspath(path)}'"
    return NodeFsError(error.errno, message)


def normalize_path(
    text: str, *, normalize_unicode_spaces: bool = False, strip_at_prefix: bool = False
) -> str:
    normalized = text
    if normalize_unicode_spaces:
        normalized = _UNICODE_SPACES.sub(" ", normalized)
    if strip_at_prefix and normalized.startswith("@"):
        normalized = normalized[1:]
    home = os.path.expanduser("~")
    if normalized == "~":
        return home
    if normalized.startswith("~/"):
        return os.path.join(home, normalized[2:])
    if normalized.startswith("file://"):
        return unquote(urlparse(normalized).path)
    return normalized


def _node_resolve(*parts: str) -> str:
    """Node ``path.resolve``: lexical, absolute, no trailing slash."""
    resolved = os.path.normpath(os.path.join(os.getcwd(), *parts))
    if resolved.startswith("//"):
        resolved = "/" + resolved.lstrip("/")
    return resolved


def resolve_to_cwd(file_path: str, cwd: Path | str) -> str:
    """Pi's ``resolveToCwd``."""
    normalized = normalize_path(file_path, normalize_unicode_spaces=True, strip_at_prefix=True)
    base = normalize_path(os.fspath(cwd))
    if os.path.isabs(normalized):
        return _node_resolve(normalized)
    return _node_resolve(base, normalized)


def path_exists(path: str) -> bool:
    """Node ``access(path, F_OK)``: follows symlinks."""
    return os.access(path, os.F_OK)


_exists = path_exists


def resolve_read_path(file_path: str, cwd: Path | str) -> str:
    """Pi's ``resolveReadPath``: also try macOS screenshot name variants."""
    resolved = resolve_to_cwd(file_path, cwd)
    if _exists(resolved):
        return resolved
    am_pm = re.sub(
        r" (AM|PM)\.", lambda match: f"{_NARROW_NO_BREAK_SPACE}{match.group(1)}.", resolved, flags=re.I
    )
    if am_pm != resolved and _exists(am_pm):
        return am_pm
    nfd = unicodedata.normalize("NFD", resolved)
    if nfd != resolved and _exists(nfd):
        return nfd
    curly = resolved.replace("'", "’")
    if curly != resolved and _exists(curly):
        return curly
    nfd_curly = nfd.replace("'", "’")
    if nfd_curly != resolved and _exists(nfd_curly):
        return nfd_curly
    return resolved


def node_relative(from_path: str, to_path: str) -> str:
    """Node ``path.relative``: ``""`` for the same path, ``..`` segments otherwise."""
    relative = os.path.relpath(_node_resolve(to_path), _node_resolve(from_path))
    return "" if relative == "." else relative
