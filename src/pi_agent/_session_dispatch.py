"""Lifecycle guard and ordered, isolated session observers."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import Callable
from copy import deepcopy
from typing import Literal

from .agent import _IN_AGENT_LISTENER, _caller_cancelled
from .events import (
    SessionEvent,
)

SessionListener = Callable[[SessionEvent], object]
_IN_LISTENER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pi_agent_in_session_listener", default=False
)


def _reap_task(task: asyncio.Task) -> None:
    """Retrieve a done task's outcome so cancellation never warns."""
    if task.cancelled():
        return
    task.exception()


class _BusyState:
    """The single lifecycle guard: one activity at a time."""

    def __init__(self) -> None:
        self.activity: Literal["prompt", "compact", "configure", "close"] | None = None

    def claim(self, activity: Literal["prompt", "compact", "configure", "close"]) -> None:
        if self.activity is not None:
            raise RuntimeError(
                f"Session is busy with an ongoing {self.activity}. Wait for it to finish "
                "(wait_for_idle) or abort() first."
            )
        self.activity = activity

    def release(self) -> None:
        self.activity = None


class ObserverDispatcher:
    def __init__(self) -> None:
        self.listeners: list[SessionListener] = []
        self._queue: asyncio.Queue[tuple[SessionEvent, asyncio.Future[None]]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._stopping_workers: set[asyncio.Task[None]] = set()
        self._futures: list[asyncio.Future[None]] = []
        self._closed = False

    def subscribe(self, listener: SessionListener) -> Callable[[], None]:
        """Subscribe to agent and session events as an observer.

        Contracts and failure semantics: docs/invariants.md.
        """
        if self._closed:
            raise RuntimeError("Session observers were disposed.")
        self.listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self.listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    def _enqueue(self, event: SessionEvent) -> asyncio.Future[None] | None:
        """Queue one event for the dispatch worker; returns its completion future.

        Without a running loop, sync listeners are invoked inline and
        awaitable results are closed (nothing can be scheduled).
        """
        if self._closed:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._deliver_inline(event)
            return None
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._dispatch_loop())
        future: asyncio.Future[None] = loop.create_future()
        self._queue.put_nowait((deepcopy(event), future))
        return future

    def _deliver_inline(self, event: SessionEvent) -> None:
        """Deliver to synchronous listeners only (no event loop available)."""
        session_token = _IN_LISTENER.set(True)
        agent_token = _IN_AGENT_LISTENER.set(True)
        try:
            for listener in list(self.listeners):
                try:
                    result = listener(deepcopy(event))
                except Exception:
                    continue
                if inspect.isawaitable(result) and hasattr(result, "close"):
                    result.close()
        finally:
            _IN_AGENT_LISTENER.reset(agent_token)
            _IN_LISTENER.reset(session_token)

    async def _dispatch_loop(self) -> None:
        """Serial, ordered delivery of queued events to all observers."""
        while True:
            event, future = await self._queue.get()
            session_token = _IN_LISTENER.set(True)
            agent_token = _IN_AGENT_LISTENER.set(True)
            try:
                for listener in list(self.listeners):
                    try:
                        result = listener(deepcopy(event))
                        if inspect.isawaitable(result):
                            await result
                    except asyncio.CancelledError:
                        if _caller_cancelled():
                            raise
                        # Independently cancelled observer: isolate it.
                        continue
                    except Exception:
                        continue
                if not future.cancelled():
                    future.set_result(None)
            except asyncio.CancelledError:
                if not future.cancelled():
                    future.set_result(None)
                raise
            finally:
                _IN_AGENT_LISTENER.reset(agent_token)
                _IN_LISTENER.reset(session_token)

    async def emit(self, event: SessionEvent) -> None:
        """Deliver one event through the ordered dispatcher and wait for it."""
        future = self._enqueue(event)
        if future is None:
            return
        try:
            await future
        except asyncio.CancelledError:
            if _caller_cancelled():
                raise

    def emit_nowait(self, event: SessionEvent) -> None:
        """Queue one event; its delivery is settled later, in order."""
        future = self._enqueue(event)
        if future is not None:
            self._futures.append(future)

    async def settle(self) -> None:
        """Await queued deliveries in order; never leak them.

        A cancelled delivery future (disposal) is skipped unless the caller
        itself is cancelled, in which case the remainder stays tracked and
        settlement returns so lifecycle cleanup can finish.
        """
        futures, self._futures = self._futures, []
        for index, future in enumerate(futures):
            try:
                await future
            except asyncio.CancelledError:
                if _caller_cancelled():
                    self._futures = [*futures[index:], *self._futures]
                    return
            except Exception:
                continue

    def close(self) -> None:
        """Stop the dispatcher and retrieve results; used on disposal."""
        self._closed = True
        worker = self._worker
        self._worker = None
        if worker is not None:
            self._stopping_workers.add(worker)
            worker.cancel()
            worker.add_done_callback(_reap_task)
            worker.add_done_callback(self._stopping_workers.discard)
        futures, self._futures = self._futures, []
        for future in futures:
            if not future.done():
                future.cancel()
        while not self._queue.empty():
            _, future = self._queue.get_nowait()
            if not future.done():
                future.cancel()

    async def aclose(self) -> None:
        """Stop and join the dispatch worker before asynchronous disposal returns."""
        self.close()
        await asyncio.gather(*self._stopping_workers, return_exceptions=True)
