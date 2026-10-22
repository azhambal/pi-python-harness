"""Explicit ownership and cancellation-safe settlement of asynchronous resources."""

from __future__ import annotations

import asyncio
import contextvars
from typing import Protocol

_CLEANING: contextvars.ContextVar[tuple[object, ...]] = contextvars.ContextVar(
    "pi_agent_resource_cleanup", default=()
)
_IN_RESOURCE_CALLBACK: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pi_agent_in_resource_callback", default=False
)


def reject_cleanup_reentry(owner: object, method: str) -> None:
    if _IN_RESOURCE_CALLBACK.get():
        raise RuntimeError(f"Cannot call {method} from a resource callback")
    if any(item is owner for item in _CLEANING.get()):
        raise RuntimeError(f"Cannot call {method} from owned resource cleanup")


class AsyncResource(Protocol):
    async def aclose(self) -> None: ...


async def settle_cleanup(task: asyncio.Task[None]) -> None:
    """Finish owned cleanup even when a waiter is cancelled; then propagate it.

    Shield alone would leave cleanup running after the caller returns. Repeated
    cancellation is remembered while the task is joined, including its errors.
    """
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                break
            cancelled = True
        except Exception:
            break
    if cancelled:
        # Retrieve a failed cleanup too; unfinished resources remain retryable.
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    task.result()


class ResourceOwner:
    """Own explicitly registered resources, close in reverse order, retry failures.

    Registration transfers ownership; injected providers/tools are otherwise
    borrowed. One owner may be transferred to exactly one harness construction.
    Resources must provide idempotent, bounded aclose implementations.
    """

    def __init__(self) -> None:
        self._resources: list[AsyncResource] = []
        self._claimed = False
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def pending(self) -> bool:
        return bool(self._resources)

    def claim(self) -> None:
        if self._claimed or self._closing or self._closed:
            raise RuntimeError("Resource owner was already transferred or closed")
        self._claimed = True

    def add(self, resource: AsyncResource) -> None:
        if self._closing or self._closed:
            raise RuntimeError("Cannot register resources after cleanup began")
        if not callable(getattr(resource, "aclose", None)):
            raise TypeError("Owned resources must implement async aclose()")
        if not any(item is resource for item in self._resources):
            self._resources.append(resource)

    async def aclose(self) -> None:
        reject_cleanup_reentry(self, "aclose")
        task = self._close_task
        if task is asyncio.current_task():
            raise RuntimeError("Resource cleanup cannot await itself")
        if task is None or task.done():
            if self._closed:
                return
            self._closing = True
            task = asyncio.create_task(self._close())
            self._close_task = task
        await settle_cleanup(task)

    async def _close(self) -> None:
        token = _CLEANING.set((*_CLEANING.get(), self))
        try:
            await self._close_resources()
        finally:
            _CLEANING.reset(token)

    async def _close_resources(self) -> None:
        errors: list[Exception] = []
        for resource in reversed(tuple(self._resources)):
            try:
                await resource.aclose()
            except asyncio.CancelledError:
                errors.append(RuntimeError("Owned resource cleanup was independently cancelled"))
            except Exception as error:
                errors.append(error)
            else:
                self._resources = [item for item in self._resources if item is not resource]
        self._closed = not self._resources
        if errors:
            raise ExceptionGroup("Resource cleanup failed; aclose() can retry", errors)
