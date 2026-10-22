import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest

from pi_agent import PermissionPolicy, ToolHooks, create_agent_session
from pi_agent.events import ToolExecutionEnd
from pi_agent.providers import FauxProvider
from pi_agent.tools import EditTool, ReadTool, WriteTool
from pi_agent.tools.base import Tool, ToolResult
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage


def calls(*blocks):
    return AssistantMessage(content=list(blocks), stop_reason="toolUse")


def done():
    return AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop")


async def test_concurrency_bound_out_of_order_completion_and_ordered_persistence(config):
    started = {key: asyncio.Event() for key in "abc"}
    release = {key: asyncio.Event() for key in "abc"}
    finished = []
    active = 0
    peak = 0

    class Gate(Tool):
        name = "gate"
        parallel_safe = True

        async def execute(self, call_id, args, cancellation=None, on_update=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            started[call_id].set()
            try:
                await release[call_id].wait()
                finished.append(call_id)
                return ToolResult(content=[TextBlock(text=call_id)])
            finally:
                active -= 1

    tool = Gate()
    provider = FauxProvider([calls(*(ToolCallBlock(id=key, name="gate") for key in "abc")), done()])
    session = create_agent_session(
        provider,
        config=replace(config, tool_execution="parallel", max_parallel_tools=2),
        tools=[tool],
        session="new",
    )
    events = []
    session.subscribe(events.append)
    task = asyncio.create_task(session.prompt("run"))
    try:
        await asyncio.wait_for(asyncio.gather(started["a"].wait(), started["b"].wait()), 1)
        assert not started["c"].is_set()
        release["b"].set()
        await asyncio.wait_for(started["c"].wait(), 1)
        release["c"].set()
        await asyncio.sleep(0)
        release["a"].set()
        await asyncio.wait_for(task, 1)
        assert peak == 2 and finished[0] == "b"
        results = [m for m in session.messages if isinstance(m, ToolResultMessage)]
        assert [m.tool_call_id for m in results] == list("abc")
        assert [e.tool_call_id for e in events if isinstance(e, ToolExecutionEnd)] == list("abc")
        assert [
            m.tool_call_id for m in provider.requests[-1].messages if isinstance(m, ToolResultMessage)
        ] == list("abc")
        path = session.session_file
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path, tools=[tool])
    try:
        assert resumed.config.tool_execution == resumed.agent.tool_execution == "parallel"
        assert resumed.config.max_parallel_tools == 2
    finally:
        resumed.dispose()


async def test_resolved_path_fifo_does_not_consume_independent_capacity(config):
    path = config.cwd / "source"
    path.write_text("old")
    (config.cwd / "alias").symlink_to("source")
    (config.cwd / "other").write_text("other")
    entered = asyncio.Event()
    released = asyncio.Event()
    independent = asyncio.Event()
    edit_entered = asyncio.Event()

    class Writer(WriteTool):
        async def execute(self, *args, **kwargs):
            entered.set()
            await released.wait()
            return await super().execute(*args, **kwargs)

    class Editor(EditTool):
        async def execute(self, *args, **kwargs):
            edit_entered.set()
            return await super().execute(*args, **kwargs)

    class Reader(ReadTool):
        async def execute(self, *args, **kwargs):
            independent.set()
            return await super().execute(*args, **kwargs)

    provider = FauxProvider(
        [
            calls(
                ToolCallBlock(id="w", name="write", arguments={"path": "source", "content": "first"}),
                ToolCallBlock(
                    id="e",
                    name="edit",
                    arguments={"path": "alias", "edits": [{"oldText": "first", "newText": "second"}]},
                ),
                ToolCallBlock(id="r", name="read", arguments={"path": "other"}),
            ),
            done(),
        ]
    )
    session = create_agent_session(
        provider,
        config=replace(config, tool_execution="parallel", max_parallel_tools=2),
        tools=[Writer(config.cwd), Editor(config.cwd), Reader(config.cwd)],
    )
    task = asyncio.create_task(session.prompt("write/edit/read"))
    try:
        await asyncio.wait_for(asyncio.gather(entered.wait(), independent.wait()), 1)
        assert not edit_entered.is_set() and path.read_text() == "old"
        released.set()
        await asyncio.wait_for(task, 1)
        assert path.read_text() == "second"
        assert all(not m.is_error for m in session.messages if isinstance(m, ToolResultMessage))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()


async def test_unsafe_custom_tool_is_a_barrier(config):
    timeline = []

    class Safe(Tool):
        name = "safe"
        parallel_safe = True

        async def execute(self, call_id, args, cancellation=None, on_update=None):
            timeline.append(call_id + " start")
            await asyncio.sleep(0)
            timeline.append(call_id + " end")
            return ToolResult()

    class Barrier(Safe):
        name = "barrier"
        parallel_safe = False

    session = create_agent_session(
        FauxProvider(
            [
                calls(
                    ToolCallBlock(id="a", name="safe"),
                    ToolCallBlock(id="b", name="barrier"),
                    ToolCallBlock(id="c", name="safe"),
                ),
                done(),
            ]
        ),
        config=replace(config, tool_execution="parallel"),
        tools=[Safe(), Barrier()],
    )
    try:
        await session.prompt("run")
        assert timeline == ["a start", "a end", "b start", "b end", "c start", "c end"]
    finally:
        session.dispose()


async def test_parallel_confirmation_serialized_and_cannot_mutate_context(config):
    in_confirmation = 0
    before = []

    async def confirm(request, token):
        nonlocal in_confirmation
        assert in_confirmation == 0
        in_confirmation += 1
        try:
            await asyncio.sleep(0)
            request.context.args["path"] = "wrong"
            return True
        finally:
            in_confirmation -= 1

    session = create_agent_session(
        FauxProvider(
            [
                calls(
                    *[
                        ToolCallBlock(id=str(i), name="write", arguments={"path": str(i), "content": "new"})
                        for i in range(3)
                    ]
                ),
                done(),
            ]
        ),
        config=replace(config, tool_execution="parallel"),
        permission_policy=PermissionPolicy(default="ask"),
        confirm_permission=confirm,
        hooks=ToolHooks(before_tool_call=lambda ctx, token: before.append(deepcopy(ctx))),
    )
    try:
        await session.prompt("write")
        assert len(before) == 3 and not (config.cwd / "wrong").exists()
        assert all((config.cwd / str(i)).read_text() == "new" for i in range(3))
    finally:
        session.dispose()


async def test_forced_cancel_keeps_completed_results_and_marks_only_running_outcome_unknown(
    config, monkeypatch
):
    monkeypatch.setattr("pi_agent.session._ABORT_GRACE_SECONDS", 0.05)
    entered = asyncio.Event()
    finished = asyncio.Event()

    class Gate(Tool):
        name = "gate"
        parallel_safe = True

        async def execute(self, call_id, args, cancellation=None, on_update=None):
            if call_id == "a":
                entered.set()
                await asyncio.Event().wait()
            finished.set()
            return ToolResult(content=[TextBlock(text="completed")])

    session = create_agent_session(
        FauxProvider([calls(ToolCallBlock(id="a", name="gate"), ToolCallBlock(id="b", name="gate"))]),
        config=replace(config, tool_execution="parallel"),
        tools=[Gate()],
    )
    before_tasks = asyncio.all_tasks()
    task = asyncio.create_task(session.prompt("run"))
    try:
        await asyncio.wait_for(asyncio.gather(entered.wait(), finished.wait()), 1)
        await session.abort()
        with pytest.raises(asyncio.CancelledError):
            await task
        results = [m for m in session.messages if isinstance(m, ToolResultMessage)]
        assert [m.tool_call_id for m in results] == ["a", "b"]
        assert results[0].interrupted and "unknown" in results[0].content[0].text
        assert not results[1].interrupted and not results[1].is_error
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before_tasks)


async def test_file_queue_is_shared_between_sessions_on_one_loop(config):
    source = config.cwd / "source"
    source.write_text("old")
    (config.cwd / "alias").symlink_to("source")
    entered = asyncio.Event()
    released = asyncio.Event()
    second_entered = asyncio.Event()

    class Writer(WriteTool):
        async def execute(self, call_id, *args, **kwargs):
            if call_id == "first":
                entered.set()
                await released.wait()
            else:
                second_entered.set()
            return await super().execute(call_id, *args, **kwargs)

    sessions = [
        create_agent_session(
            FauxProvider(
                [
                    calls(
                        ToolCallBlock(
                            id=identifier, name="write", arguments={"path": path, "content": identifier}
                        )
                    ),
                    done(),
                ]
            ),
            config=replace(config, tool_execution="parallel"),
            tools=[Writer(config.cwd)],
        )
        for identifier, path in [("first", "source"), ("second", "alias")]
    ]
    first = asyncio.create_task(sessions[0].prompt("write"))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(sessions[1].prompt("write"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not second_entered.is_set()
        released.set()
        await asyncio.wait_for(asyncio.gather(first, second), 1)
        assert source.read_text() == "second"
    finally:
        for task in (first, second):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        for session in sessions:
            session.dispose()


async def test_cancelled_file_queue_waiter_is_known_unexecuted(config):
    (config.cwd / "source").write_text("old")
    entered = asyncio.Event()

    class Writer(WriteTool):
        async def execute(self, call_id, *args, **kwargs):
            assert call_id == "first"
            entered.set()
            await asyncio.Event().wait()

    session = create_agent_session(
        FauxProvider(
            [
                calls(
                    *[
                        ToolCallBlock(
                            id=identifier, name="write", arguments={"path": "source", "content": identifier}
                        )
                        for identifier in ("first", "queued")
                    ]
                )
            ]
        ),
        config=replace(config, tool_execution="parallel"),
        tools=[Writer(config.cwd)],
    )
    task = asyncio.create_task(session.prompt("write"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        results = [m for m in session.messages if isinstance(m, ToolResultMessage)]
        assert results[0].interrupted
        assert not results[1].interrupted and "not executed" in results[1].content[0].text
        assert (config.cwd / "source").read_text() == "old"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()


async def test_actual_bash_processes_overlap_without_reordering_results(config):
    from pi_agent.events import ToolExecutionUpdate

    ready = set()
    both = asyncio.Event()
    command = "printf 'ready\\n'; while [ ! -e release ]; do sleep 0.01; done"
    session = create_agent_session(
        FauxProvider(
            [
                calls(
                    *[
                        ToolCallBlock(id=identifier, name="bash", arguments={"command": command})
                        for identifier in ("a", "b")
                    ]
                ),
                done(),
            ]
        ),
        config=replace(config, tool_execution="parallel", max_parallel_tools=2),
    )

    def watch(event):
        if isinstance(event, ToolExecutionUpdate) and "ready" in str(event.partial_result):
            ready.add(event.tool_call_id)
            if len(ready) == 2:
                both.set()

    session.subscribe(watch)
    task = asyncio.create_task(session.prompt("run both"))
    try:
        await asyncio.wait_for(both.wait(), 2)
        (config.cwd / "release").touch()
        await asyncio.wait_for(task, 2)
        assert [m.tool_call_id for m in session.messages if isinstance(m, ToolResultMessage)] == ["a", "b"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()
