import asyncio
import os
from pathlib import Path

import pytest

from pi_agent import AgentConfig, create_agent_session
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.providers.faux import FauxProvider
from pi_agent.session_store import SessionStore
from pi_agent.tools import BashTool
from pi_agent.tools._spill import SpillRegistry
from pi_agent.tools.output import OutputAccumulator
from pi_agent.types import AssistantMessage, ToolCallBlock, UserMessage


async def test_session_disposal_removes_spills_and_keeps_unowned_paths(config: AgentConfig) -> None:
    tool = BashTool(config.cwd, max_bytes=8)
    sentinel = config.cwd / "unowned.log"
    sentinel.write_text("caller-owned")
    session = create_agent_session(FauxProvider([]), config=config, session="new", tools=[tool])
    try:
        result = await tool.execute("spill", {"command": "printf 'abcdefghijklmnop'"})
        spill = Path(result.details["full_output_path"])
        assert spill.exists() and spill.stat().st_mode & 0o777 == 0o600
        # An arbitrary journal-supplied path is never an ownership record.
        from pi_agent.types import ToolResultMessage

        session.store.append_message(
            ToolResultMessage(
                tool_call_id="foreign",
                tool_name="bash",
                content=[],
                details={"full_output_path": str(sentinel)},
            )
        )
        session.dispose()
        assert not spill.exists() and sentinel.read_text() == "caller-owned"
        session.dispose()
    finally:
        session.dispose()


async def test_cancelled_execution_spill_is_registered_before_a_result_exists(tmp_path: Path) -> None:
    tool = BashTool(tmp_path, max_bytes=8)
    update = asyncio.Event()
    paths: list[Path] = []

    async def on_update(result) -> None:
        path = result.details.get("full_output_path")
        if path:
            paths.append(Path(path))
            update.set()

    task = asyncio.create_task(
        tool.execute(
            "cancel",
            {"command": "printf 'abcdefghijklmnop'; sleep 30"},
            on_update=on_update,
        )
    )
    try:
        await asyncio.wait_for(update.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 4)
        assert all(path.exists() for path in paths)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        tool.close()
    assert paths and all(not path.exists() for path in paths)


async def test_parallel_spills_are_cleaned_after_real_tool_dispatch(config: AgentConfig) -> None:
    config.tool_execution = "parallel"
    tool = BashTool(config.cwd, max_bytes=8)
    provider = FauxProvider(
        [
            AssistantMessage(
                content=[
                    ToolCallBlock(id=f"b{n}", name="bash", arguments={"command": "printf 'abcdefghijklmnop'"})
                    for n in range(2)
                ],
                stop_reason="toolUse",
            ),
            AssistantMessage(),
        ]
    )
    session = create_agent_session(provider, config=config, tools=[tool])
    try:
        await session.prompt("spill twice")
        paths = [Path(m.details["full_output_path"]) for m in session.messages if m.role == "toolResult"]
        assert len(set(paths)) == 2 and all(p.exists() for p in paths)
    finally:
        session.dispose()
    assert all(not p.exists() for p in paths)


@pytest.mark.parametrize("outcome", ["timeout", "abort", "reader_failure"])
async def test_failed_execution_spills_live_until_close(tmp_path: Path, outcome: str) -> None:
    tool = BashTool(tmp_path, max_bytes=8)
    cancellation = Cancellation()
    paths: list[Path] = []

    async def update(result) -> None:
        path = result.details.get("full_output_path")
        if path:
            paths.append(Path(path))
            if outcome == "abort":
                cancellation.abort()
            elif outcome == "reader_failure":
                raise ValueError("reader fixture failure")

    try:
        error_type, error_text = {
            "timeout": (TimeoutError, "Command timed out"),
            "abort": (OperationAbortedError, "Command aborted"),
            "reader_failure": (RuntimeError, "Output reader failed"),
        }[outcome]
        with pytest.raises(error_type, match=error_text):
            await asyncio.wait_for(
                tool.execute(
                    "failed",
                    {"command": "printf 'abcdefghijklmnop'; sleep 30", "timeout": 0.1},
                    cancellation=cancellation,
                    on_update=update,
                ),
                3,
            )
        assert paths and all(p.exists() for p in paths)
    finally:
        tool.close()
    assert all(not p.exists() for p in paths)


def test_replaced_owned_path_is_not_deleted(tmp_path: Path) -> None:
    registry = SpillRegistry()
    original = tmp_path / "spill.log"
    descriptor = os.open(original, os.O_CREAT | os.O_RDWR, 0o600)
    registry.register(str(original), descriptor)
    original.unlink()
    replacement = tmp_path / "replacement"
    replacement.write_text("must survive")
    original.symlink_to(replacement)
    os.close(descriptor)
    registry.close()
    assert original.is_symlink() and replacement.read_text() == "must survive"


def test_standalone_accumulator_without_registry_remains_caller_owned() -> None:
    accumulator = OutputAccumulator(max_bytes=2)
    try:
        accumulator.append(b"complete")
        accumulator.finish()
        spill = Path(accumulator.snapshot().full_output_path)
        accumulator.close_temp_file()
        assert spill.read_bytes() == b"complete"
    finally:
        accumulator.close_temp_file()
        spill.unlink()


async def test_cleanup_failure_releases_journal_lock_and_dispose_can_retry(
    config: AgentConfig, monkeypatch
) -> None:
    tool = BashTool(config.cwd, max_bytes=8)
    session = create_agent_session(FauxProvider([]), config=config, session="new", tools=[tool])
    session.store.append_message(UserMessage())
    path = session.session_file
    result = await tool.execute(
        "spill", {"command": "printf 'abcdefghijklmnop'"}, cancellation=Cancellation()
    )
    spill = Path(result.details["full_output_path"])
    unlink = Path.unlink

    def fail(owned: Path, *args, **kwargs):
        if owned == spill:
            raise PermissionError("cleanup denied")
        return unlink(owned, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail)
            with pytest.raises(ExceptionGroup, match="Session cleanup failed"):
                session.dispose()
            assert session.is_idle and spill.exists()
            reopened = SessionStore.open(path)
            reopened.close()
        session.dispose()
        assert not spill.exists()
    finally:
        session.dispose()
