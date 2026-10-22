import asyncio

import pytest

from pi_agent import AgentConfig, create_agent_session
from pi_agent._session_dispatch import ObserverDispatcher
from pi_agent.events import MessageEnd, QueueUpdate
from pi_agent.providers.faux import FauxProvider
from pi_agent.tools import Tool, ToolResult
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock


async def test_close_settles_untracked_emit_waiters_and_clears_queued_snapshots() -> None:
    dispatcher = ObserverDispatcher()
    entered = asyncio.Event()

    async def hang(event) -> None:
        entered.set()
        await asyncio.Event().wait()

    dispatcher.subscribe(hang)
    first = asyncio.create_task(dispatcher.emit(QueueUpdate(steering=["one"])))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(dispatcher.emit(QueueUpdate(steering=["two"])))
        await asyncio.sleep(0)
        dispatcher.close()
        await asyncio.wait_for(asyncio.gather(first, second), 1)
        assert dispatcher._queue.empty() and dispatcher._futures == []
    finally:
        dispatcher.close()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


async def test_each_listener_receives_an_independent_enqueued_snapshot() -> None:
    dispatcher = ObserverDispatcher()
    seen = []
    dispatcher.subscribe(lambda event: event.steering.append("listener mutation"))
    dispatcher.subscribe(lambda event: seen.append(event.steering))
    event = QueueUpdate(steering=["original"])
    dispatcher.emit_nowait(event)
    event.steering.append("caller mutation")
    try:
        await dispatcher.settle()
        assert seen == [["original"]]
    finally:
        dispatcher.close()
        await asyncio.sleep(0)


async def test_disposed_dispatcher_cannot_restart_from_late_notifications() -> None:
    dispatcher = ObserverDispatcher()
    seen = []
    dispatcher.subscribe(seen.append)
    dispatcher.close()
    dispatcher.emit_nowait(QueueUpdate())
    await dispatcher.emit(QueueUpdate())
    await dispatcher.settle()
    assert seen == [] and dispatcher._worker is None and dispatcher._queue.empty()
    with pytest.raises(RuntimeError, match="disposed"):
        dispatcher.subscribe(seen.append)


async def test_observer_cannot_rewrite_a_committed_tool_call_before_execution(config: AgentConfig) -> None:
    received = []

    class CaptureTool(Tool):
        name = "capture"

        async def execute(self, tool_call_id, args, cancellation=None, on_update=None):
            received.append(dict(args))
            return ToolResult(content=[TextBlock(text="done")])

    provider = FauxProvider(
        [
            AssistantMessage(
                content=[ToolCallBlock(id="c", name="capture", arguments={"value": "original"})],
                stop_reason="toolUse",
            ),
            AssistantMessage(),
        ]
    )
    session = create_agent_session(provider, config=config, tools=[CaptureTool()])

    def mutate(event) -> None:
        if isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            for block in event.message.content:
                if isinstance(block, ToolCallBlock):
                    block.arguments["value"] = "mutation"

    session.subscribe(mutate)
    try:
        await session.prompt("capture")
        assert received == [{"value": "original"}]
        assert session.store.call_ids == {"c"}
    finally:
        session.dispose()


async def test_disposal_rejects_an_active_direct_agent_run(config: AgentConfig) -> None:
    entered = asyncio.Event()

    class HangingTool(Tool):
        name = "hang"

        async def execute(self, tool_call_id, args, cancellation=None, on_update=None):
            entered.set()
            await asyncio.Event().wait()

    session = create_agent_session(
        FauxProvider([AssistantMessage(content=[ToolCallBlock(id="c", name="hang")], stop_reason="toolUse")]),
        config=config,
        tools=[HangingTool()],
    )
    task = asyncio.create_task(session.agent.prompt("direct"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(RuntimeError, match="Cannot dispose"):
            session.dispose()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()
