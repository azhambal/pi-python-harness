"""One-shot, awaitable cancellation shared by providers and tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator


class OperationAbortedError(Exception):
    """Raised by raise_if_aborted() and by tools that observe cancellation."""


class Cancellation:
    """Cancellation within one event loop; abort() wakes all current and future waiters."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def abort(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_aborted(self) -> None:
        if self.aborted:
            raise OperationAbortedError("Operation aborted")


async def iterate_with_cancellation[T](
    stream: AsyncIterator[T], cancellation: Cancellation
) -> AsyncGenerator[T, None]:
    """Interrupt a blocked anext(), even when a provider ignores the token.

    The caller owns stream closure. Pending reads are cancelled and joined before
    returning, so aclose() never races an in-flight async-generator read.
    """
    aborted = asyncio.create_task(cancellation.wait())
    read: asyncio.Future[T] | None = None
    try:
        while True:
            cancellation.raise_if_aborted()
            read = asyncio.ensure_future(anext(stream))
            await asyncio.wait((read, aborted), return_when=asyncio.FIRST_COMPLETED)
            cancellation.raise_if_aborted()
            try:
                yield read.result()
            except StopAsyncIteration:
                return
    finally:
        aborted.cancel()
        if read is not None:
            read.cancel()
            await asyncio.gather(read, aborted, return_exceptions=True)
        else:
            await asyncio.gather(aborted, return_exceptions=True)
