"""Per-file mutation queues, translated from Pi's ``file-mutation-queue.ts``."""

import asyncio
import contextlib
import errno
import os
from collections.abc import AsyncIterator
from weakref import WeakKeyDictionary, WeakValueDictionary

_LOOPS: WeakKeyDictionary[asyncio.AbstractEventLoop, WeakValueDictionary[str, asyncio.Lock]] = (
    WeakKeyDictionary()
)


def mutation_queue_key(path: str) -> str:
    """The realpath of ``path``, or the path itself while it does not exist."""
    resolved = os.path.abspath(path)
    try:
        os.stat(resolved)
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ENOTDIR):
            return resolved
        raise
    return os.path.realpath(resolved)


def file_lock(key: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _LOOPS.setdefault(loop, WeakValueDictionary())
    lock = locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        locks[key] = lock
    return lock


@contextlib.asynccontextmanager
async def file_mutation_queue(path: str) -> AsyncIterator[None]:
    """Serialize mutations of one file; different files proceed in parallel."""
    async with file_lock(mutation_queue_key(path)):
        yield
