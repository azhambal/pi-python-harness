import asyncio
from collections.abc import AsyncIterator

import pytest

from pi_agent import AgentConfig, create_agent_session
from pi_agent.compaction import CompactionAbortedError
from pi_agent.events import AssistantStreamEvent, StreamStart
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.providers import FauxProvider
from pi_agent.types import AssistantMessage, TextBlock


def response(text: str = "answer") -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop")


async def populate(session) -> None:
    for i in range(3):
        await session.prompt(f"question {i}" + "x" * 400)


async def test_compaction_checkpoint_survives_resume(config: AgentConfig) -> None:
    config.compaction.keep_recent_tokens = 10
    provider = FauxProvider([response() for _ in range(3)] + [response("## Goal\nContinue the task")])
    session = create_agent_session(provider, config=config, session="new")
    try:
        await populate(session)
        result = await session.compact()
        assert result.summary.startswith("## Goal")
        assert provider.requests[-1].tools == []
        projected = [message.to_json() for message in session.messages]
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        assert [message.to_json() for message in resumed.messages] == projected
        assert len(resumed.store.entries) > len(resumed.messages)
    finally:
        resumed.dispose()


@pytest.mark.parametrize(
    ("stop_reason", "error"),
    [("error", RuntimeError), ("length", RuntimeError), ("aborted", CompactionAbortedError)],
)
async def test_failed_summary_does_not_commit_checkpoint(config: AgentConfig, stop_reason, error) -> None:
    config.compaction.keep_recent_tokens = 10
    failed = AssistantMessage(
        content=[TextBlock(text="partial summary")], stop_reason=stop_reason, error_message="failure"
    )
    provider = FauxProvider([response() for _ in range(3)] + [failed])
    session = create_agent_session(provider, config=config)
    try:
        await populate(session)
        before = session.store.entries
        with pytest.raises(error):
            await session.compact()
        assert session.store.entries == before
    finally:
        session.dispose()


class HangingSummaryProvider(FauxProvider):
    def __init__(self) -> None:
        super().__init__([response() for _ in range(3)])
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def stream(
        self, request: ProviderRequest, options: ProviderOptions
    ) -> AsyncIterator[AssistantStreamEvent]:
        if request.tools:
            async for event in super().stream(request, options):
                yield event
        else:
            try:
                yield StreamStart()
                self.started.set()
                await asyncio.Event().wait()
            finally:
                self.closed.set()


async def test_abort_interrupts_hanging_compaction(config: AgentConfig) -> None:
    config.compaction.keep_recent_tokens = 10
    provider = HangingSummaryProvider()
    session = create_agent_session(provider, config=config)
    await populate(session)
    before = session.store.entries
    task = asyncio.create_task(session.compact())
    try:
        await asyncio.wait_for(provider.started.wait(), 1)
        await asyncio.wait_for(session.abort(), 1)
        with pytest.raises(CompactionAbortedError):
            await task
        assert provider.closed.is_set()
        assert session.store.entries == before
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()
