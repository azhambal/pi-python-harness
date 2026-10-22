"""Synchronous same-directory file replacement with an explicit commit boundary.

The sequential harness serializes its own writes. Version checks detect outside
changes before commit; they are not compare-and-swap against arbitrary writers.
Symlinks are resolved by the calling tool. Hard links and non-regular targets
are refused because replacing their inode would change their semantics.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pi_agent.cancellation import Cancellation


class ConcurrentFileChangeError(RuntimeError):
    """The target changed while the replacement was being prepared."""


class AtomicWriteCommittedError(RuntimeError):
    """Content was installed, but directory durability could not be confirmed."""

    committed = True


@dataclass(frozen=True)
class FileVersion:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    uid: int
    gid: int
    links: int

    @classmethod
    def from_stat(cls, info: os.stat_result) -> FileVersion:
        return cls(
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_nlink,
        )


def _regular_version(info: os.stat_result, path: Path) -> FileVersion:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Cannot atomically write {path}: target is not a regular file.")
    if info.st_nlink != 1:
        raise ValueError(
            f"Cannot atomically write {path}: hard-linked files require an explicit in-place write."
        )
    return FileVersion.from_stat(info)


def read_file_snapshot(path: Path) -> tuple[bytes, FileVersion]:
    """Read one regular inode without blocking on a FIFO or following a late symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        version = _regular_version(os.fstat(source.fileno()), path)
        content = source.read()
        if FileVersion.from_stat(os.fstat(source.fileno())) != version:
            raise ConcurrentFileChangeError(
                f"File changed while reading {path}; read it again before editing."
            )
    return content, version


def _target_version(directory_fd: int, name: str, path: Path) -> FileVersion | None:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return _regular_version(info, path)


def atomic_write(
    path: Path,
    content: bytes,
    *,
    cancellation: Cancellation | None = None,
    expected: FileVersion | None = None,
) -> None:
    """Write/fsync a private temporary file, verify the target, replace, fsync directory.

    Preserve owner, group, ordinary permission bits and xattrs. Set-ID bits and
    file capabilities are cleared, as with an ordinary unprivileged content write.
    A new file uses 0666 filtered by umask/default ACL. No cancellation is reported
    after commit as if nothing happened; directory-fsync failure explicitly says
    that the content is installed. Parent-directory creation is owned by WriteTool.
    """
    if cancellation is not None:
        cancellation.raise_if_aborted()
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    temp_name: str | None = None
    try:
        original = _target_version(directory_fd, path.name, path)
        if expected is not None and expected != original:
            raise ConcurrentFileChangeError(
                f"File changed before writing {path}; read it again before editing."
            )
        attributes: dict[str, bytes] = {}
        if original is not None:
            if not os.access(path, os.W_OK):
                raise PermissionError(f"Cannot atomically write {path}: file is not writable.")
            original_fd = os.open(path.name, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            try:
                if FileVersion.from_stat(os.fstat(original_fd)) != original:
                    raise ConcurrentFileChangeError(f"File changed before writing {path}; read it again.")
                attributes = {
                    name: os.getxattr(original_fd, name)
                    for name in os.listxattr(original_fd)
                    if name != "security.capability"
                }
            finally:
                os.close(original_fd)
        candidate = f".pi-write-{uuid4().hex}.tmp"
        temp_fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=directory_fd)
        temp_name = candidate
        with os.fdopen(temp_fd, "wb") as target:
            created = os.fstat(target.fileno())
            mode = original.mode & 0o777 if original is not None else stat.S_IMODE(created.st_mode)
            os.fchmod(target.fileno(), 0o600)
            if original is not None and (created.st_uid, created.st_gid) != (original.uid, original.gid):
                os.fchown(target.fileno(), original.uid, original.gid)
            target.write(content)
            target.flush()
            for name, value in attributes.items():
                # Some security labels cannot be set by an unprivileged user;
                # an identical inherited value needs no change.
                try:
                    inherited = os.getxattr(target.fileno(), name)
                except OSError:
                    inherited = None
                if inherited != value:
                    os.setxattr(target.fileno(), name, value)
            os.fchmod(target.fileno(), mode)
            os.fsync(target.fileno())
            prepared = os.fstat(target.fileno())
        if cancellation is not None:
            cancellation.raise_if_aborted()
        if _target_version(directory_fd, path.name, path) != original:
            raise ConcurrentFileChangeError(
                f"File changed while writing {path}; no new content was installed."
            )
        parent = path.parent.stat()
        opened_parent = os.fstat(directory_fd)
        if (parent.st_dev, parent.st_ino) != (opened_parent.st_dev, opened_parent.st_ino):
            raise ConcurrentFileChangeError(
                f"Directory changed while writing {path}; no content was installed."
            )
        temporary = os.stat(temp_name, dir_fd=directory_fd, follow_symlinks=False)
        if FileVersion.from_stat(temporary) != FileVersion.from_stat(prepared):
            raise ConcurrentFileChangeError(
                f"Temporary file changed while writing {path}; no content was installed."
            )
        os.replace(temp_name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        temp_name = None
        try:
            os.fsync(directory_fd)
        except OSError as error:
            raise AtomicWriteCommittedError(
                f"New content was installed in {path}, but directory fsync failed; durability is unconfirmed. "
                "Inspect the file before retrying."
            ) from error
    finally:
        try:
            if temp_name is not None:
                try:
                    os.unlink(temp_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        finally:
            os.close(directory_fd)
