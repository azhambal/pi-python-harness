import asyncio
from dataclasses import replace

import pytest

from pi_agent import (
    AfterToolCallResult,
    Agent,
    AgentConfig,
    BeforeToolCallResult,
    ToolHooks,
    create_agent_session,
)
from pi_agent.events import ToolExecutionEnd
from pi_agent.providers import FauxProvider
from pi_agent.tool_hooks import merge_after_result
from pi_agent.tools.base import Tool, ToolResult
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage, Usage


def response(*calls: ToolCallBlock) -> AssistantMessage:
    return AssistantMessage(content=list(calls), stop_reason="toolUse")


def write_call(call_id: str = "write", path: str = "result.txt") -> ToolCallBlock:
    return ToolCallBlock(id=call_id, name="write", arguments={"path": path, "content": "written"})


def done() -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop")


async def test_before_deny_pairs_and_persists_without_execution(config: AgentConfig) -> None:
    stages = []

    def before(context, cancellation):
        stages.append("before")
        assert context.args == {"path": "result.txt", "content": "written"}
        return BeforeToolCallResult(block=True, reason="blocked by host")

    def after(context, cancellation):
        stages.append("after")

    provider = FauxProvider([response(write_call()), done()])
    session = create_agent_session(provider, config=config, session="new", hooks=ToolHooks(before, after))
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("write")
        assert stages == ["before"]
        assert not (config.cwd / "result.txt").exists()
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.tool_call_id == "write" and result.is_error
        assert result.content[0].text == "blocked by host"
        end = next(event for event in events if isinstance(event, ToolExecutionEnd))
        assert end.is_error == end.result["is_error"] == result.is_error
        assert end.duration_ms is None
        assert provider.requests[-1].messages[-1] == result
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        results = [message for message in resumed.messages if isinstance(message, ToolResultMessage)]
        assert len(results) == 1 and results[0] == result
    finally:
        resumed.dispose()


@pytest.mark.parametrize("kind", ["unknown", "invalid", "length"])
async def test_hooks_only_receive_valid_executable_calls(config: AgentConfig, kind: str) -> None:
    calls = []
    block = write_call()
    if kind == "unknown":
        block.name = "missing"
    elif kind == "invalid":
        block.arguments = {"path": "result.txt"}
    message = response(block)
    if kind == "length":
        message.stop_reason = "length"
    hooks = ToolHooks(
        lambda context, token: calls.append("before"), lambda context, token: calls.append("after")
    )
    session = create_agent_session(FauxProvider([message, done()]), config=config, hooks=hooks)
    try:
        await session.prompt("write")
        assert calls == []
        assert not (config.cwd / "result.txt").exists()
    finally:
        session.dispose()


async def test_hook_order_and_snapshots_protect_arguments_and_history(config: AgentConfig) -> None:
    stages = []

    def before(context, token):
        stages.append("before")
        context.args["path"] = "wrong.txt"
        context.tool_call.id = "wrong-id"
        context.tool_call.arguments["path"] = "also-wrong.txt"
        context.assistant_message.content.clear()
        context.messages.clear()
        return None

    def after(context, token):
        stages.append("after")
        assert context.args["path"] == "result.txt"
        assert context.tool_call.id == "write"
        assert context.assistant_message.content
        context.result.content[0].text = "discarded mutation"
        return AfterToolCallResult(content=[TextBlock(text="visible result")], details={"host": "final"})

    session = create_agent_session(
        FauxProvider([response(write_call()), done()]), config=config, hooks=ToolHooks(before, after)
    )
    events = []
    session.subscribe(lambda event: (stages.append(event.type), events.append(event)))
    try:
        await session.prompt("write")
        assert (config.cwd / "result.txt").read_text() == "written"
        assert not (config.cwd / "wrong.txt").exists()
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.content == [TextBlock(text="visible result")]
        assert result.details == {"host": "final"}
        assert result.tool_call_id == "write"
        assert (
            stages.index("tool_execution_start")
            < stages.index("before")
            < stages.index("after")
            < stages.index("tool_execution_end")
        )
        assert next(event.result for event in events if isinstance(event, ToolExecutionEnd))["content"] == [
            {"type": "text", "text": "visible result"}
        ]
    finally:
        session.dispose()


def test_after_override_nullable_fields_and_structured_content_rules() -> None:
    original = ToolResult(
        content=[TextBlock(text="original")],
        details={"keep": 1},
        structured_content={"value": 1},
        usage=Usage(output=2),
        is_error=True,
        terminate=True,
    )
    merged = merge_after_result(original, AfterToolCallResult(is_error=False))
    assert merged.details == {"keep": 1} and merged.structured_content == {"value": 1}
    assert not merged.is_error and merged.terminate
    assert merged.usage.output == 2
    replaced = merge_after_result(original, AfterToolCallResult(content=[TextBlock(text="new")]))
    assert replaced.structured_content is None
    kept = merge_after_result(
        original, AfterToolCallResult(content=[TextBlock(text="new")], structured_content={"value": 2})
    )
    assert kept.structured_content == {"value": 2}
    cleared = merge_after_result(
        original, AfterToolCallResult(details=None, structured_content=None, usage=None, terminate=False)
    )
    assert cleared.details is None and cleared.usage is None and not cleared.terminate
    assert original.content == [TextBlock(text="original")] and original.is_error


async def test_after_receives_execution_exception_and_can_replace_error(config: AgentConfig) -> None:
    observed = []

    class Fails(Tool):
        name = "fails"

        async def execute(self, tool_call_id, args, cancellation=None, on_update=None):
            raise ValueError("actual failure")

    def after(context, token):
        observed.append(context.is_error)
        assert context.result.content[0].text == "actual failure"
        return AfterToolCallResult(content=[TextBlock(text="handled")], is_error=False)

    session = create_agent_session(
        FauxProvider([response(ToolCallBlock(id="c", name="fails")), done()]),
        config=config,
        tools=[Fails()],
        hooks=ToolHooks(after_tool_call=after),
    )
    try:
        await session.prompt("run")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert observed == [True] and not result.is_error
        assert result.content == [TextBlock(text="handled")]
    finally:
        session.dispose()


@pytest.mark.parametrize("stage", ["before", "after"])
@pytest.mark.parametrize("invalid", [False, True])
async def test_hook_failures_are_explicit_and_preserve_known_execution(
    config: AgentConfig, stage: str, invalid: bool
) -> None:
    def hook(context, token):
        if invalid:
            return {"unexpected": "return type"}
        raise RuntimeError("host hook failed")

    hooks = ToolHooks(before_tool_call=hook) if stage == "before" else ToolHooks(after_tool_call=hook)
    session = create_agent_session(
        FauxProvider([response(write_call()), done()]), config=config, session="new", hooks=hooks
    )
    try:
        await session.prompt("write")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        # As in pi, the hook's error message becomes the whole result.
        assert result.is_error and result.details == {}
        expected = (
            f"{'Before' if stage == 'before' else 'After'}-tool hook must return "
            f"{'BeforeToolCallResult' if stage == 'before' else 'AfterToolCallResult'} or None"
            if invalid
            else "host hook failed"
        )
        assert [block.text for block in result.content] == [expected]
        assert (config.cwd / "result.txt").exists() is (stage == "after")
    finally:
        session.dispose()


@pytest.mark.parametrize("stage", ["before", "after"])
@pytest.mark.parametrize("external", [False, True])
async def test_cancelling_hanging_hook_keeps_known_results(
    config: AgentConfig, stage: str, external: bool
) -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()
    before_tasks = asyncio.all_tasks()

    async def hanging(context, token):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    hooks = ToolHooks(before_tool_call=hanging) if stage == "before" else ToolHooks(after_tool_call=hanging)
    session = create_agent_session(
        FauxProvider([response(write_call(), write_call("second", "never.txt"))]),
        config=replace(config, tool_execution="parallel"),
        session="new",
        hooks=hooks,
    )
    task = asyncio.create_task(session.prompt("write"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if external:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            await asyncio.wait_for(session.abort(), 1)
            await task
        assert closed.is_set()
        results = [message for message in session.messages if isinstance(message, ToolResultMessage)]
        assert not any(message.interrupted for message in results)
        if stage == "before":
            # Preparation stops at the aborted call, as in pi: the second call gets no
            # result here (the projection answers it later) and nothing runs.
            assert [message.tool_call_id for message in results] == ["write"]
            assert results[0].is_error
            assert not (config.cwd / "result.txt").exists() and not (config.cwd / "never.txt").exists()
        else:
            # Parallel calls all execute before their after-hooks run, as in pi; the
            # cancelled hooks keep the known results.
            assert [message.tool_call_id for message in results] == ["write", "second"]
            assert (config.cwd / "result.txt").read_text() == "written"
            assert (config.cwd / "never.txt").exists()
            assert not results[0].is_error and results[0].details["executed"]
        assert session.is_idle
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before_tasks)


@pytest.mark.parametrize("method", ["wait_for_idle", "abort", "compact"])
async def test_hook_lifecycle_reentry_fails_without_deadlock(config: AgentConfig, method: str) -> None:
    async def before(context, token):
        await getattr(session, method)()

    session = create_agent_session(
        FauxProvider([response(write_call()), done()]),
        config=config,
        hooks=ToolHooks(before_tool_call=before),
    )
    try:
        await asyncio.wait_for(session.prompt("write"), 1)
        assert not (config.cwd / "result.txt").exists()
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and "re-entrantly" in result.content[0].text
    finally:
        session.dispose()


async def test_bare_agent_hook_guard_applies_to_child_tasks(config: AgentConfig) -> None:
    async def before(context, token):
        await asyncio.create_task(agent.wait_for_idle())

    agent = Agent(
        FauxProvider([response(write_call()), done()]),
        model=config.model,
        tools=[],
        hooks=ToolHooks(before_tool_call=before),
    )
    # Provide a built-in to reach the before hook through the low-level public API.
    from pi_agent.tools import WriteTool

    agent.tools = [WriteTool(config.cwd)]
    await asyncio.wait_for(agent.prompt("write"), 1)
    result = next(message for message in agent.messages if isinstance(message, ToolResultMessage))
    assert result.is_error and not (config.cwd / "result.txt").exists()


async def test_hook_context_excludes_results_of_the_current_batch(config: AgentConfig) -> None:
    # pi adds a batch's results to the context only after the whole batch.
    observed = []

    def before(context, token):
        observed.append(
            [message.tool_call_id for message in context.messages if isinstance(message, ToolResultMessage)]
        )

    session = create_agent_session(
        FauxProvider([response(write_call("a", "a"), write_call("b", "b")), done()]),
        config=config,
        hooks=ToolHooks(before_tool_call=before),
    )
    try:
        await session.prompt("write")
        assert observed == [[], []]
    finally:
        session.dispose()


@pytest.mark.parametrize("all_terminate", [False, True])
async def test_block_termination_requires_every_batch_result(
    config: AgentConfig, all_terminate: bool
) -> None:
    def before(context, token):
        return BeforeToolCallResult(block=True, terminate=all_terminate or context.tool_call.id == "a")

    provider = FauxProvider([response(write_call("a", "a"), write_call("b", "b")), done()])
    session = create_agent_session(provider, config=config, hooks=ToolHooks(before_tool_call=before))
    try:
        await session.prompt("write")
        assert len(provider.requests) == (1 if all_terminate else 2)
        assert len([message for message in session.messages if isinstance(message, ToolResultMessage)]) == 2
    finally:
        session.dispose()


@pytest.mark.parametrize(
    "override",
    [
        AfterToolCallResult(details={"invalid": float("nan")}),
        AfterToolCallResult(content=["invalid block"]),
        AfterToolCallResult(is_error="false"),
    ],
)
async def test_invalid_after_patch_becomes_an_error_result_and_a_valid_journal(
    config: AgentConfig, override
) -> None:
    session = create_agent_session(
        FauxProvider([response(write_call()), done()]),
        config=config,
        session="new",
        hooks=ToolHooks(after_tool_call=lambda ctx, token: override),
    )
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("write")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and result.details == {}  # The hook error replaces the result, as in pi.
        assert (config.cwd / "result.txt").read_text() == "written"
        end = next(event for event in events if isinstance(event, ToolExecutionEnd))
        assert end.is_error == end.result["is_error"] == result.is_error
        path = session.session_file
    finally:
        session.dispose()
    reopened = create_agent_session(FauxProvider([]), session=path)
    try:
        assert (
            next(message for message in reopened.messages if isinstance(message, ToolResultMessage)) == result
        )
    finally:
        reopened.dispose()


@pytest.mark.parametrize("stage", ["before", "after"])
async def test_independently_cancelled_callback_does_not_cancel_run(config: AgentConfig, stage: str) -> None:
    async def cancelled(context, token):
        raise asyncio.CancelledError

    hooks = (
        ToolHooks(before_tool_call=cancelled) if stage == "before" else ToolHooks(after_tool_call=cancelled)
    )
    provider = FauxProvider([response(write_call()), done()])
    session = create_agent_session(provider, config=config, hooks=hooks)
    try:
        await session.prompt("write")
        assert len(provider.requests) == 2 and session.messages[-1].stop_reason == "stop"
        assert (config.cwd / "result.txt").exists() is (stage == "after")
    finally:
        session.dispose()


@pytest.mark.parametrize("all_terminate", [False, True])
async def test_after_termination_requires_every_executed_result(
    config: AgentConfig, all_terminate: bool
) -> None:
    provider = FauxProvider([response(write_call("a", "a"), write_call("b", "b")), done()])
    session = create_agent_session(
        provider,
        config=config,
        hooks=ToolHooks(
            after_tool_call=lambda ctx, token: AfterToolCallResult(
                terminate=all_terminate or ctx.tool_call.id == "a"
            )
        ),
    )
    try:
        await session.prompt("write")
        assert (config.cwd / "a").exists() and (config.cwd / "b").exists()
        assert len(provider.requests) == (1 if all_terminate else 2)
    finally:
        session.dispose()
