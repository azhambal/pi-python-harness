"""Automatic compaction triggers, ported from Pi's agent-session-compaction.test.ts."""

from __future__ import annotations

from dataclasses import replace

from pi_agent import create_agent_session
from pi_agent.config import CompactionSettings
from pi_agent.events import CompactionEnd, CompactionStart
from pi_agent.providers import FauxProvider
from pi_agent.tools.base import Tool, ToolResult
from pi_agent.types import (
    AssistantMessage,
    TextBlock,
    ToolCallBlock,
    Usage,
    UserMessage,
    now_ms,
)

OVERFLOW_FAILED = (
    "Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or "
    "switching to a larger-context model."
)


def answer(text: str = "", *, stop: str = "stop", usage: Usage | None = None, **fields) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)], stop_reason=stop, usage=usage or Usage(), **fields
    )


def assistant(model, *, stop="stop", total=0, timestamp=None, error=None) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        stop_reason=stop,
        error_message=error,
        provider=model.provider,
        api=model.api,
        model=model.id,
        usage=Usage(input=total, total_tokens=total),
        timestamp=timestamp if timestamp is not None else now_ms(),
    )


def session_for(config, provider, *, window=None, max_tokens=None, **compaction):
    config.compaction = CompactionSettings(**{"enabled": True, **compaction})
    if window is not None:
        config.model = replace(config.model, context_window=window)
    if max_tokens is not None:
        config.model = replace(config.model, max_tokens=max_tokens)
    session = create_agent_session(provider, config=config, session="new")
    events: list = []
    session.subscribe(events.append)
    return session, events


def spy_auto_compaction(session, result=False):
    calls: list[tuple[str, bool]] = []

    async def fake(reason, will_retry):
        calls.append((reason, will_retry))
        return result

    session._run_auto_compaction = fake
    return calls


async def test_length_stop_below_output_limit_compacts_and_resumes(config):
    provider = FauxProvider(
        [answer("partial response", stop="length"), answer("prefix summary"), answer("completed response")]
    )
    session, events = session_for(
        config, provider, window=1000, max_tokens=100, keep_recent_tokens=1, reserve_tokens=0
    )
    try:
        await session.prompt("x" * 5000)
        assert len(provider.requests) == 3
        end = [event for event in events if isinstance(event, CompactionEnd)][-1]
        assert (end.reason, end.aborted, end.will_retry) == ("overflow", False, True)
        assert session.messages[-1].content[0].text == "completed response"
    finally:
        session.dispose()


async def test_length_stop_at_output_limit_does_not_compact(config):
    provider = FauxProvider([answer("x" * 400, stop="length", usage=Usage(output=100))])
    session, events = session_for(config, provider, window=1_000_000, max_tokens=100)
    try:
        await session.prompt("hello")
        assert len(provider.requests) == 1
        assert not any(isinstance(event, CompactionStart) for event in events)
    finally:
        session.dispose()


async def test_second_truncated_response_stops_after_one_recovery(config):
    later = now_ms() + 10_000
    provider = FauxProvider(
        [
            answer("x" * 64, stop="length", timestamp=later),
            answer("prefix summary", timestamp=later),
            answer("y" * 64, stop="length", timestamp=later + 1),
        ]
    )
    session, events = session_for(
        config, provider, window=1_000_000, max_tokens=100, keep_recent_tokens=1, reserve_tokens=0
    )
    try:
        await session.prompt("x" * 5000)
        assert len([request for request in provider.requests if request.tools]) == 2
        assert len([e for e in events if isinstance(e, CompactionStart) and e.reason == "overflow"]) == 1
        assert [e for e in events if isinstance(e, CompactionEnd)][-1].error_message == (
            "Truncated response recovery failed after one compact-and-retry attempt."
        )
    finally:
        session.dispose()


async def test_repeated_length_overflow_keeps_overflow_wording(config):
    session, events = session_for(config, FauxProvider([]), window=100, max_tokens=100)
    try:
        calls = spy_auto_compaction(session)
        message = assistant(config.model, stop="length", total=100)
        await session._check_compaction(message)
        await session._check_compaction(replace(message, timestamp=message.timestamp + 1))
        assert len(calls) == 1
        assert OVERFLOW_FAILED in [e.error_message for e in events if isinstance(e, CompactionEnd)]
    finally:
        session.dispose()


async def test_overflow_recovery_is_attempted_once(config):
    session, events = session_for(config, FauxProvider([]))
    try:
        calls = spy_auto_compaction(session)
        message = assistant(config.model, stop="error", error="prompt is too long")
        await session._check_compaction(message)
        await session._check_compaction(replace(message, timestamp=message.timestamp + 1))
        assert len(calls) == 1
        assert OVERFLOW_FAILED in [e.error_message for e in events if isinstance(e, CompactionEnd)]
    finally:
        session.dispose()


async def test_successful_overflow_response_compacts_without_retry(config):
    provider = FauxProvider([answer("completed answer"), answer("prefix summary")])
    session, events = session_for(
        config, provider, window=1, max_tokens=100, keep_recent_tokens=1, reserve_tokens=0
    )
    try:
        # The completed answer reports more input than the window (silent overflow).
        provider._responses = iter(
            [
                answer("completed answer", usage=Usage(input=50, total_tokens=50)),
                answer("summary"),
                answer("p"),
            ]
        )
        await session.prompt("hello")
        end = [event for event in events if isinstance(event, CompactionEnd)][-1]
        assert (end.reason, end.aborted, end.will_retry) == ("overflow", False, False)
        assert len([request for request in provider.requests if request.tools]) == 1
    finally:
        session.dispose()


async def test_overflow_from_another_model_is_ignored(config):
    session, events = session_for(config, FauxProvider([]))
    try:
        calls = spy_auto_compaction(session)
        message = assistant(replace(config.model, id="other"), stop="error", error="prompt is too long")
        await session._check_compaction(message)
        assert calls == []
    finally:
        session.dispose()


async def test_stale_pre_compaction_usage_is_ignored_on_pre_prompt_check(config):
    session, _ = session_for(config, FauxProvider([]))
    try:
        stale = assistant(config.model, total=610_000, timestamp=now_ms() - 10_000)
        first = session.store.append_message(
            UserMessage(content="before compaction", timestamp=stale.timestamp - 1000)
        )
        session.store.append_message(stale)
        session.store.append_compaction("summary", first, 610_000)
        session.store.append_message(UserMessage(content="after compaction"))
        calls = spy_auto_compaction(session)
        await session._check_compaction(stale, skip_aborted_check=False)
        assert calls == []
    finally:
        session.dispose()


async def test_error_threshold_uses_last_successful_usage(config):
    session, _ = session_for(config, FauxProvider([]), window=200_000)
    try:
        now = now_ms()
        error = assistant(config.model, stop="error", error="529 overloaded", timestamp=now + 1000)
        session.agent.messages = [
            UserMessage(content="hello", timestamp=now - 1000),
            assistant(config.model, total=190_000, timestamp=now),
            UserMessage(content="retry", timestamp=now + 500),
            error,
        ]
        calls = spy_auto_compaction(session)
        await session._check_compaction(error)
        assert calls == [("threshold", False)]
    finally:
        session.dispose()


async def test_error_without_prior_usage_does_not_compact(config):
    session, _ = session_for(config, FauxProvider([]), window=200_000)
    try:
        error = assistant(config.model, stop="error", error="529 overloaded")
        session.agent.messages = [UserMessage(content="hello", timestamp=error.timestamp - 1000), error]
        calls = spy_auto_compaction(session)
        await session._check_compaction(error)
        assert calls == []
    finally:
        session.dispose()


async def test_only_kept_pre_compaction_usage_does_not_compact(config):
    session, _ = session_for(config, FauxProvider([]), window=200_000)
    try:
        before = now_ms() - 10_000
        kept = assistant(config.model, total=190_000, timestamp=before)
        first = session.store.append_message(
            UserMessage(content="before compaction", timestamp=before - 1000)
        )
        session.store.append_message(kept)
        session.store.append_compaction("summary", first, 190_000)
        error = assistant(config.model, stop="error", error="529 overloaded")
        session.agent.messages = [
            UserMessage(content="kept user", timestamp=before - 1000),
            kept,
            UserMessage(content="new prompt", timestamp=error.timestamp - 500),
            error,
        ]
        calls = spy_auto_compaction(session)
        await session._check_compaction(error)
        assert calls == []
    finally:
        session.dispose()


async def test_below_threshold_or_disabled_does_not_compact(config):
    below, _ = session_for(config, FauxProvider([]), window=200_000, reserve_tokens=1000)
    try:
        calls = spy_auto_compaction(below)
        await below._check_compaction(assistant(below.config.model, total=1_000))
        assert calls == []
    finally:
        below.dispose()
    disabled, _ = session_for(config, FauxProvider([]))
    try:
        disabled.config.compaction = CompactionSettings(enabled=False)
        calls = spy_auto_compaction(disabled)
        await disabled._check_compaction(assistant(disabled.config.model, total=1_000_000))
        assert calls == []
    finally:
        disabled.dispose()


async def test_threshold_resumes_when_only_queued_messages_exist(config):
    provider = FauxProvider([answer("one"), answer("two"), answer("summary"), answer("prefix")])
    session, _ = session_for(config, provider, keep_recent_tokens=1)
    try:
        await session.prompt("first")
        await session.prompt("second")
        session.agent.follow_up(UserMessage(content="queued"))
        assert await session._run_auto_compaction("threshold", False) is True
    finally:
        session.dispose()


def tool_call() -> AssistantMessage:
    # Pi's faux provider reports usage for the context it received; this port's does not.
    return AssistantMessage(
        content=[ToolCallBlock(id="big", name="large_result", arguments={})],
        stop_reason="toolUse",
        usage=Usage(input=500, total_tokens=500),
    )


class LargeResultTool(Tool):
    name = "large_result"
    description = "Returns enough content to cross the compaction threshold"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, size: int, terminate: bool = False) -> None:
        self.size = size
        self.terminate = terminate

    async def execute(self, tool_call_id, args, cancellation=None, on_update=None):
        return ToolResult(
            content=[TextBlock(text="large-tool-result:" + "x" * self.size)], terminate=self.terminate
        )


async def test_threshold_compacts_between_turns_of_one_run(config):
    responses = [
        answer("old-history:" + "a" * 800),
        answer("recent-history:" + "b" * 800),
        tool_call(),
        answer("compacted history"),
        answer("turn prefix"),  # the cut splits the running turn
        answer("finished after compaction"),
    ]
    provider = FauxProvider(responses)
    config.compaction = CompactionSettings(enabled=True, reserve_tokens=400, keep_recent_tokens=1750)
    config.model = replace(config.model, context_window=2600, max_tokens=100)
    session = create_agent_session(provider, config=config, tools=[LargeResultTool(8000)], session="new")
    events: list = []
    session.subscribe(events.append)
    try:
        await session.prompt("seed old history")
        await session.prompt("seed recent history")
        await session.prompt("run the large tool")
        assert [e.reason for e in events if isinstance(e, CompactionStart)] == ["threshold"]
        resumed = provider.requests[-1]
        texts = " ".join(
            block.text
            for message in resumed.messages
            for block in getattr(message, "content", []) or []
            if isinstance(block, TextBlock)
        )
        assert "compacted history" in texts and "large-tool-result" in texts
        assert session.messages[-1].content[0].text == "finished after compaction"
    finally:
        session.dispose()


async def test_terminating_tool_result_does_not_compact(config):
    provider = FauxProvider(
        [
            answer("old-history:" + "a" * 800),
            answer("recent-history:" + "b" * 800),
            tool_call(),
        ]
    )
    config.compaction = CompactionSettings(enabled=True, reserve_tokens=400, keep_recent_tokens=1750)
    config.model = replace(config.model, context_window=2600, max_tokens=100)
    session = create_agent_session(
        provider, config=config, tools=[LargeResultTool(6800, terminate=True)], session="new"
    )
    events: list = []
    session.subscribe(events.append)
    try:
        await session.prompt("seed old history")
        await session.prompt("seed recent history")
        await session.prompt("run the terminating tool")
        assert not any(isinstance(event, CompactionStart) for event in events)
        assert len(provider.requests) == 3
    finally:
        session.dispose()


async def test_steering_queued_during_compaction_reaches_resumed_request(config):
    import asyncio

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingSummaries(FauxProvider):
        async def stream(self, request, options):
            if not request.tools and not started.is_set():
                started.set()
                await release.wait()
            async for event in super().stream(request, options):
                yield event

    provider = BlockingSummaries(
        [
            answer("old-history:" + "a" * 800),
            answer("recent-history:" + "b" * 800),
            tool_call(),
            answer("compacted history"),
            answer("turn prefix"),
            answer("finished after compaction"),
            answer("finished after delayed steering"),
        ]
    )
    config.compaction = CompactionSettings(enabled=True, reserve_tokens=400, keep_recent_tokens=1750)
    config.model = replace(config.model, context_window=2600, max_tokens=100)
    session = create_agent_session(provider, config=config, tools=[LargeResultTool(6800)], session="new")
    try:
        await session.prompt("seed old history")
        await session.prompt("seed recent history")
        task = asyncio.create_task(session.prompt("run the large tool"))
        await asyncio.wait_for(started.wait(), 1)
        session.steer("change direction")
        release.set()
        await asyncio.wait_for(task, 5)
        resumed = [request for request in provider.requests if request.tools][3]
        assert any(
            isinstance(message, UserMessage) and "change direction" in str(message.content)
            for message in resumed.messages
        )
        assert len([request for request in provider.requests if request.tools]) == 4
    finally:
        session.dispose()
