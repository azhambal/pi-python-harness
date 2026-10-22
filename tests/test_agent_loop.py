import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from pi_agent import AgentConfig, create_agent_session
from pi_agent.events import AssistantStreamEvent, StreamStart, ToolExecutionEnd
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.providers import FauxProvider
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage


def answer(text: str = "done") -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop")


def write(path: str, call_id: str = "call_0") -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCallBlock(id=call_id, name="write", arguments={"path": path, "content": path})],
        stop_reason="toolUse",
    )


async def test_tool_loop_persists_and_resumes(config: AgentConfig, tmp_path: Path) -> None:
    provider = FauxProvider([write("a.txt"), answer()])
    session = create_agent_session(provider, config=config, session="new")
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("write a file")
        assert (tmp_path / "a.txt").read_text() == "a.txt"
        assert [message.role for message in session.messages] == [
            "system",
            "user",
            "assistant",
            "toolResult",
            "assistant",
        ]
        assert [event.type for event in events].count("turn_start") == 2
        assert events[-1].type == "agent_settled"
        path = session.session_file
        assert path is not None
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        assert len(resumed.messages) == 5
        assert resumed.cwd == tmp_path
    finally:
        resumed.dispose()


@pytest.mark.parametrize("same_batch", [False, True])
async def test_reused_tool_ids_execute_and_pair(
    config: AgentConfig, tmp_path: Path, same_batch: bool
) -> None:
    # CODE_REVIEW 4.2: call_0 may be local to each provider response.
    first, second = write("a.txt"), write("b.txt")
    if same_batch:
        first.content.extend(second.content)
        script = [first, answer()]
    else:
        script = [first, second, answer()]
    provider = FauxProvider(script)
    session = create_agent_session(provider, config=config, session="new")
    try:
        await session.prompt("write both files")
        assert (tmp_path / "a.txt").exists()
        assert (tmp_path / "b.txt").exists()
        calls = [
            block.id
            for msg in session.messages
            if isinstance(msg, AssistantMessage)
            for block in msg.content
            if isinstance(block, ToolCallBlock)
        ]
        results = [msg.tool_call_id for msg in session.messages if isinstance(msg, ToolResultMessage)]
        assert len(calls) == len(set(calls)) == 2
        assert calls == results
        assert all(msg.content for msg in session.messages if isinstance(msg, AssistantMessage))
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    resumed.dispose()


async def test_empty_success_is_explicit_failure(config: AgentConfig) -> None:
    session = create_agent_session(FauxProvider([AssistantMessage(stop_reason="toolUse")]), config=config)
    try:
        await session.prompt("work")
        final = session.messages[-1]
        assert isinstance(final, AssistantMessage)
        assert final.stop_reason == "error"
        assert final.error_message
        assert final.content
    finally:
        session.dispose()


class HangingProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def stream(
        self, request: ProviderRequest, options: ProviderOptions
    ) -> AsyncIterator[AssistantStreamEvent]:
        del request, options
        try:
            yield StreamStart(partial=AssistantMessage(content=[TextBlock(text="partial")]))
            self.started.set()
            await asyncio.Event().wait()
        finally:
            self.closed.set()


async def test_abort_interrupts_hung_provider(config: AgentConfig) -> None:
    # CODE_REVIEW 4.3: abort must not depend on the next network chunk.
    provider = HangingProvider()
    session = create_agent_session(provider, config=config)
    task = asyncio.create_task(session.prompt("work"))
    try:
        await asyncio.wait_for(provider.started.wait(), 1)
        await asyncio.wait_for(session.abort(), 1)
        await task
        assert provider.closed.is_set()
        final = session.messages[-1]
        assert isinstance(final, AssistantMessage)
        assert final.stop_reason == "aborted"
        assert final.content[0].text == "partial"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()


async def test_length_truncated_call_never_executes(config: AgentConfig, tmp_path: Path) -> None:
    truncated = write("never.txt")
    truncated.stop_reason = "length"
    session = create_agent_session(FauxProvider([truncated, answer()]), config=config)
    try:
        await session.prompt("work")
        assert not (tmp_path / "never.txt").exists()
        result = next(msg for msg in session.messages if isinstance(msg, ToolResultMessage))
        assert result.is_error
        assert "output token limit" in result.content[0].text
    finally:
        session.dispose()


async def test_steering_waits_for_whole_tool_batch(config: AgentConfig) -> None:
    batch = write("a")
    batch.content.extend(write("b", "call_1").content)
    provider = FauxProvider([batch, answer()])
    session = create_agent_session(provider, config=config)

    def steer(event):
        if isinstance(event, ToolExecutionEnd) and event.tool_call_id == "call_0":
            session.steer("also check this")

    session.subscribe(steer)
    try:
        await session.prompt("work")
        assert [msg.role for msg in provider.requests[-1].messages][-3:] == [
            "toolResult",
            "toolResult",
            "user",
        ]
    finally:
        session.dispose()


async def test_tool_error_contains_message_without_exception_class(config: AgentConfig) -> None:
    # CODE_REVIEW 5.5: model-facing errors should not depend on Python classes.
    bad = AssistantMessage(
        content=[
            ToolCallBlock(
                id="c",
                name="edit",
                arguments={"path": "missing", "edits": [{"oldText": "a", "newText": "b"}]},
            )
        ],
        stop_reason="toolUse",
    )
    session = create_agent_session(FauxProvider([bad, answer()]), config=config)
    try:
        await session.prompt("work")
        result = next(msg for msg in session.messages if isinstance(msg, ToolResultMessage))
        assert result.is_error
        assert not result.content[0].text.startswith("PermissionError:")
    finally:
        session.dispose()


async def test_retry_normalizes_id_reserved_by_omitted_attempt(config: AgentConfig, tmp_path: Path) -> None:
    config.retry.enabled = True
    config.retry.base_delay_ms = 1
    failed = write("a.txt")
    failed.stop_reason = "error"
    failed.error_message = "HTTP 503"
    session = create_agent_session(
        FauxProvider([failed, write("a.txt"), answer()]), config=config, session="new"
    )
    try:
        await session.prompt("work")
        assert (tmp_path / "a.txt").read_text() == "a.txt"
        calls = [
            block.id
            for entry in session.store.entries
            if hasattr(entry, "message") and isinstance(entry.message, AssistantMessage)
            for block in entry.message.content
            if isinstance(block, ToolCallBlock)
        ]
        assert len(calls) == len(set(calls)) == 2
        assert calls[0] == "call_0"
        assert calls[1] != "call_0"
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    resumed.dispose()
