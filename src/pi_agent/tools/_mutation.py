"""Resolved-path FIFO locks shared by harness batches on one event loop."""

import asyncio
from pathlib import Path
from weakref import WeakKeyDictionary, WeakValueDictionary

_LOOPS: WeakKeyDictionary[asyncio.AbstractEventLoop, WeakValueDictionary[Path, asyncio.Lock]] = (
    WeakKeyDictionary()
)


def file_lock(path: Path) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _LOOPS.setdefault(loop, WeakValueDictionary())
    lock = locks.get(path)
    if lock is None:
        lock = asyncio.Lock()
        locks[path] = lock
    return lock
