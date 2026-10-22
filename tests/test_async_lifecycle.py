from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_agent import ResourceOwner, UserMessage, create_agent_session, create_harness
from pi_agent.providers.faux import FauxProvider
from pi_agent.session_store import SessionStore
from pi_agent.tools.base import Tool, ToolResult


class TrackedResource:
    def __init__(self, name="resource", calls=None, *, fail=False, gate=None):
        self.name = name
        self.calls = calls if calls is not None else []
        self.fail = fail
        self.gate = gate
        self.entered = asyncio.Event()

    async def aclose(self):
        self.calls.append(self.name)
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise ValueError("fixture cleanup failed")


async def test_owner_reverse_order_dedup_and_idempotence():
    owner = ResourceOwner()
    calls = []
    first, second = TrackedResource("first", calls), TrackedResource("second", calls)
    owner.add(first)
    owner.add(second)
    owner.add(first)
    await owner.aclose()
    await owner.aclose()
    assert calls == ["second", "first"] and not owner.pending
    with pytest.raises(RuntimeError, match="after cleanup"):
        owner.add(first)
    with pytest.raises(RuntimeError, match="closed"):
        owner.claim()


async def test_owner_failure_closes_others_and_retries_only_failed():
    owner = ResourceOwner()
    calls = []
    first, second = TrackedResource("first", calls), TrackedResource("second", calls, fail=True)
    owner.add(first)
    owner.add(second)
    with pytest.raises(ExceptionGroup, match="Resource cleanup failed"):
        await owner.aclose()
    assert calls == ["second", "first"] and owner.pending
    second.fail = False
    await owner.aclose()
    assert calls == ["second", "first", "second"] and not owner.pending


async def test_owner_concurrent_cancelled_waiters_join_cleanup():
    owner = ResourceOwner()
    gate = asyncio.Event()
    resource = TrackedResource(gate=gate)
    owner.add(resource)
    first = asyncio.create_task(owner.aclose())
    await resource.entered.wait()
    second = asyncio.create_task(owner.aclose())
    first.cancel()
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert resource.calls == ["resource"] and not owner.pending


async def test_owner_independent_resource_cancellation_is_retryable():
    class CancelOnce:
        calls = 0

        async def aclose(self):
            self.calls += 1
            if self.calls == 1:
                raise asyncio.CancelledError

    owner = ResourceOwner()
    resource = CancelOnce()
    owner.add(resource)
    with pytest.raises(ExceptionGroup, match="cleanup failed"):
        await owner.aclose()
    await owner.aclose()
    assert resource.calls == 2


async def test_owner_reentry_in_child_task_rejects_without_deadlock():
    owner = ResourceOwner()

    class Reentrant:
        async def aclose(self):
            with pytest.raises(RuntimeError, match="owned resource cleanup"):
                await asyncio.create_task(owner.aclose())

    owner.add(Reentrant())
    await asyncio.wait_for(owner.aclose(), 1)


def test_owner_requires_cleanup_contract():
    with pytest.raises(TypeError, match="aclose"):
        ResourceOwner().add(object())


async def test_sync_dispose_rejects_async_owner_before_teardown(config):
    owner = ResourceOwner()
    resource = TrackedResource()
    owner.add(resource)
    session = await create_harness(FauxProvider([]), config=config, session="new", resource_owner=owner)
    session.store.append_message(UserMessage())
    with pytest.raises(RuntimeError, match="await session.aclose"):
        session.dispose()
    assert not session._disposed and session.store.usable and not resource.calls
    await session.aclose()
    assert session._disposed and resource.calls == ["resource"]
    reopened = SessionStore.open(session.session_file)
    reopened.close()
    session.dispose()


async def test_harness_preparation_success_and_borrowed_provider(config):
    class Provider(FauxProvider):
        closed = False

        async def aclose(self):
            self.closed = True

    provider = Provider([])
    resource = TrackedResource()

    async def prepare(owner):
        owner.add(resource)
        return []

    async with await create_harness(provider, config=config, prepare_tools=prepare) as session:
        assert not session._disposed and not session.agent.tools
    assert resource.calls == ["resource"] and not provider.closed


@pytest.mark.parametrize("failure", ["prepare", "construct", "cancel"])
async def test_harness_preparation_failure_closes_resources(config, failure):
    resource = TrackedResource()

    async def prepare(owner):
        owner.add(resource)
        if failure == "prepare":
            raise ValueError("prepare fixture")
        if failure == "cancel":
            raise asyncio.CancelledError
        return []

    kwargs = {"config": config, "prepare_tools": prepare}
    if failure == "construct":
        kwargs["model"] = object()
    exception = asyncio.CancelledError if failure == "cancel" else (ValueError, AttributeError)
    with pytest.raises(exception):
        await create_harness(FauxProvider([]), **kwargs)
    assert resource.calls == ["resource"]


async def test_harness_cancelled_connect_joins_cancelled_cleanup(config):
    preparing = asyncio.Event()
    gate = asyncio.Event()
    resource = TrackedResource(gate=gate)

    async def prepare(owner):
        owner.add(resource)
        preparing.set()
        await asyncio.Event().wait()
        return []

    task = asyncio.create_task(create_harness(FauxProvider([]), config=config, prepare_tools=prepare))
    await preparing.wait()
    task.cancel()
    await resource.entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert resource.calls == ["resource"]


async def test_harness_constructor_and_cleanup_errors_retained(config):
    resource = TrackedResource(fail=True)

    async def prepare(owner):
        owner.add(resource)
        raise ValueError("prepare fixture")

    with pytest.raises(ExceptionGroup, match="construction and cleanup failed") as caught:
        await create_harness(FauxProvider([]), config=config, prepare_tools=prepare)
    assert str(caught.value.exceptions[0]) == "prepare fixture"
    assert resource.calls == ["resource"]


async def test_owner_transfer_cannot_close_resources_of_other_session(config):
    owner = ResourceOwner()
    resource = TrackedResource()
    owner.add(resource)
    session = await create_harness(FauxProvider([]), config=config, resource_owner=owner)
    try:
        with pytest.raises(RuntimeError, match="already transferred"):
            await create_harness(FauxProvider([]), config=config, resource_owner=owner)
        assert not resource.calls
    finally:
        await session.aclose()


async def test_preparation_and_tool_selection_conflict_does_not_transfer(config):
    owner = ResourceOwner()

    async def prepare(owner):
        pytest.fail("conflicting preparation must not execute")

    with pytest.raises(ValueError, match="cannot be combined"):
        await create_harness(
            FauxProvider([]), config=config, resource_owner=owner, prepare_tools=prepare, tool_names=[]
        )
    session = await create_harness(FauxProvider([]), config=config, resource_owner=owner)
    await session.aclose()


async def test_async_cleanup_failure_releases_lock_and_retry(config):
    owner = ResourceOwner()
    resource = TrackedResource(fail=True)
    owner.add(resource)
    session = await create_harness(FauxProvider([]), config=config, session="new", resource_owner=owner)
    session.store.append_message(UserMessage())
    with pytest.raises(ExceptionGroup, match="Session cleanup failed"):
        await session.aclose()
    assert session.is_idle and session._disposed
    reopened = SessionStore.open(session.session_file)
    reopened.close()
    with pytest.raises(RuntimeError, match="await session.aclose"):
        session.dispose()
    resource.fail = False
    await session.aclose()
    assert resource.calls == ["resource", "resource"]


async def test_close_waiters_and_mutations_during_cleanup(config):
    owner = ResourceOwner()
    gate = asyncio.Event()
    resource = TrackedResource(gate=gate)
    owner.add(resource)
    session = await create_harness(FauxProvider([]), config=config, resource_owner=owner)
    first = asyncio.create_task(session.aclose())
    await resource.entered.wait()
    second = asyncio.create_task(session.aclose())
    with pytest.raises(RuntimeError, match="busy"):
        await session.prompt("late")
    for mutation in (session.steer, session.follow_up):
        with pytest.raises(RuntimeError, match="disposed"):
            mutation("late")
    with pytest.raises(RuntimeError, match="disposed"):
        session.clear_queue()
    with pytest.raises(RuntimeError, match="disposed"):
        session.set_thinking_level("off")
    with pytest.raises(RuntimeError, match="Cannot dispose"):
        session.dispose()
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done() and not session.is_idle
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert resource.calls == ["resource"] and session.is_idle


async def test_session_cleanup_reentry_in_child_task_rejects(config):
    owner = ResourceOwner()
    session = await create_harness(FauxProvider([]), config=config, resource_owner=owner)

    class Reentrant:
        async def aclose(self):
            for method in (session.aclose, session.abort, session.wait_for_idle):
                with pytest.raises(RuntimeError, match="owned resource cleanup"):
                    await asyncio.create_task(method())

    owner.add(Reentrant())
    await asyncio.wait_for(session.aclose(), 1)


async def test_core_async_close_joins_dispatcher_and_is_idempotent(config):
    session = create_agent_session(FauxProvider([]), config=config)
    session.set_thinking_level("off")
    # Schedule a configuration notification without a prompt run.
    session._emit_queue_update()
    worker = session._observers._worker
    await session.aclose()
    assert worker.done() and session._disposed
    await session.aclose()
    session.dispose()
    with pytest.raises(RuntimeError, match="disposed"):
        await session.__aenter__()


async def test_async_close_joins_observers_stopped_by_prior_sync_dispose(config):
    session = create_agent_session(FauxProvider([]), config=config)
    session._emit_queue_update()
    worker = session._observers._worker
    session.dispose()
    await session.aclose()
    assert worker.done() and not session._observers._stopping_workers


@pytest.mark.parametrize("origin", ["observer", "tool_hook"])
async def test_async_close_rejected_inside_observer_and_tool_hook(config, origin):
    from pi_agent import AssistantMessage, TextBlock, ToolCallBlock, ToolHooks

    attempts = []

    async def attempt_close():
        with pytest.raises(RuntimeError, match="re-entrantly"):
            await asyncio.create_task(session.aclose())
        attempts.append(origin)

    async def observer(event):
        if event.type == "message_start":
            await attempt_close()

    async def before(context, cancellation):
        await attempt_close()

    provider = FauxProvider(
        [
            AssistantMessage(
                content=[
                    ToolCallBlock(
                        id="call", name="write", arguments={"path": "result.txt", "content": "fixture"}
                    )
                ],
                stop_reason="toolUse",
            ),
            AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop"),
        ]
    )
    session = create_agent_session(
        provider, config=config, hooks=ToolHooks(before_tool_call=before) if origin == "tool_hook" else None
    )
    if origin == "observer":
        session.subscribe(observer)
    try:
        await session.prompt("write fixture")
        assert attempts and not session._disposed
        assert (config.cwd / "result.txt").read_text() == "fixture"
    finally:
        await session.aclose()


async def test_active_run_requires_abort_before_close(config):
    from pi_agent.events import StreamStart
    from pi_agent.types import AssistantMessage

    entered = asyncio.Event()
    released = asyncio.Event()

    class WaitingProvider:
        async def stream(self, request, options):
            try:
                entered.set()
                yield StreamStart(partial=AssistantMessage(stop_reason="pending"))
                await asyncio.Event().wait()
            finally:
                released.set()

    session = create_agent_session(WaitingProvider(), config=config)
    run = asyncio.create_task(session.prompt("wait"))
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="abort.*first"):
            await session.aclose()
        assert not session._disposed
        await session.abort()
        await run
        assert released.is_set()
    finally:
        await session.abort()
        await asyncio.gather(run, return_exceptions=True)
        await session.aclose()


async def test_custom_async_tool_remains_borrowed(config):
    class Borrowed(Tool):
        name = "borrowed"
        closed = False

        async def execute(self, *args, **kwargs):
            return ToolResult()

        async def aclose(self):
            self.closed = True

    tool = Borrowed()
    session = await create_harness(FauxProvider([]), config=config, tools=[tool])
    await session.aclose()
    assert not tool.closed


async def test_async_close_cleans_bash_spill_even_when_owned_resource_fails(config):
    from pi_agent.tools.bash import BashTool

    tool = BashTool(config.cwd, max_bytes=8)
    owner = ResourceOwner()
    resource = TrackedResource(fail=True)
    owner.add(resource)
    session = await create_harness(
        FauxProvider([]), config=config, session="new", tools=[tool], resource_owner=owner
    )
    session.store.append_message(UserMessage())
    result = await tool.execute("spill", {"command": "printf 'abcdefghijklmnop'"})
    spill = Path(result.details["full_output_path"])
    try:
        with pytest.raises(ExceptionGroup, match="Session cleanup failed"):
            await session.aclose()
        assert not spill.exists()
        reopened = SessionStore.open(session.session_file)
        reopened.close()
    finally:
        resource.fail = False
        await session.aclose()
