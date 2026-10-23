import asyncio
from collections.abc import AsyncIterator, Mapping

import pytest

from pi_agent import Agent, AgentConfig, create_agent_session
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.events import (
    AssistantStreamEvent,
    MessageUpdate,
    StreamDone,
    StreamStart,
    TextDelta,
    ToolExecutionEnd,
)
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.providers import FauxProvider
from pi_agent.providers._normalize import transform_messages
from pi_agent.tools import Tool, ToolResult, ToolUpdateCallback
from pi_agent.types import AssistantMessage, JsonValue, TextBlock, ToolCallBlock, ToolResultMessage


async def test_abort_wakes_every_waiter_and_future_waiters() -> None:
    cancellation = Cancellation()
    waiters = [asyncio.create_task(cancellation.wait()) for _ in range(3)]
    cancellation.abort()
    cancellation.abort()
    await asyncio.wait_for(asyncio.gather(*waiters), 1)
    await asyncio.wait_for(cancellation.wait(), 1)
    with pytest.raises(OperationAbortedError):
        cancellation.raise_if_aborted()


class HangingTool(Tool):
    name = "hang"

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.calls: list[str] = []

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        self.calls.append(tool_call_id)
        self.entered.set()
        await asyncio.Event().wait()  # ignores cooperative cancellation
        return ToolResult()


async def test_abort_escalates_uncooperative_tool(config: AgentConfig, monkeypatch) -> None:
    monkeypatch.setattr("pi_agent.session._ABORT_GRACE_SECONDS", 0.05)
    provider = FauxProvider(
        [AssistantMessage(content=[ToolCallBlock(id="c", name="hang")], stop_reason="toolUse")]
    )
    tool = HangingTool()
    session = create_agent_session(provider, config=config, tools=[tool])
    task = asyncio.create_task(session.prompt("work"))
    try:
        # tool_execution_start precedes admission hooks; wait for actual execute.
        await asyncio.wait_for(tool.entered.wait(), 1)
        await asyncio.wait_for(session.abort(), 1)
        with pytest.raises(asyncio.CancelledError):
            await task
        results = [message for message in session.messages if isinstance(message, ToolResultMessage)]
        assert len(results) == 1
        assert results[0].interrupted
        assert "unknown" in results[0].content[0].text
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
@pytest.mark.parametrize("call_count", [1, 2])
async def test_cancelled_bare_agent_pairs_calls_before_next_request(model, mode, call_count):
    calls = [ToolCallBlock(id=f"call_{index}", name="hang") for index in range(call_count)]
    provider = FauxProvider(
        [
            AssistantMessage(content=calls, stop_reason="toolUse"),
            AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop"),
        ]
    )
    tool = HangingTool()
    agent = Agent(provider, model=model, tools=[tool], tool_execution=mode)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    task = asyncio.create_task(agent.prompt("work"))
    try:
        await asyncio.wait_for(tool.entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.is_idle and not agent.pending_tool_calls
        # Sequential batches stop after the running call, as in pi; parallel calls all ran.
        ran = [call.id for call in (calls if mode == "parallel" else calls[:1])]
        results = [message for message in agent.messages if isinstance(message, ToolResultMessage)]
        assert [result.tool_call_id for result in results] == ran
        assert all(result.interrupted and result.is_error for result in results)
        assert all("outcome unknown" in result.content[0].text for result in results)
        assert [event.tool_call_id for event in events if isinstance(event, ToolExecutionEnd)] == ran
        assert len(provider.requests) == 1

        await agent.prompt("inspect the outcome")
        transcript = transform_messages(provider.requests[-1].messages, model)
        requested = [
            block.id
            for message in transcript
            if isinstance(message, AssistantMessage)
            for block in message.content
            if isinstance(block, ToolCallBlock)
        ]
        completed = [message.tool_call_id for message in transcript if isinstance(message, ToolResultMessage)]
        # The projection answers calls without a result ("No result provided").
        assert requested == completed == [call.id for call in calls]
        assert tool.calls == ran
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class MutatingProvider:
    async def stream(
        self, request: ProviderRequest, options: ProviderOptions
    ) -> AsyncIterator[AssistantStreamEvent]:
        message = AssistantMessage(content=[TextBlock(text="a")])
        yield StreamStart(partial=message)
        yield TextDelta(delta="a", partial=message)
        message.content[0].text = "ab"
        yield TextDelta(delta="b", partial=message)
        message.stop_reason = "stop"
        yield StreamDone(message=message)


async def test_stream_snapshots_are_stable_and_watchers_do_not_leak(config: AgentConfig) -> None:
    # CODE_REVIEW 5.7: providers may mutate a single partial object between yields.
    before = asyncio.all_tasks()
    session = create_agent_session(MutatingProvider(), config=config)
    updates = []
    session.subscribe(lambda event: updates.append(event) if isinstance(event, MessageUpdate) else None)
    try:
        await session.prompt("work")
        assert [event.message.content[0].text for event in updates] == ["a", "a", "ab"]
    finally:
        session.dispose()
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before)
