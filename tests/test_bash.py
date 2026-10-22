import asyncio
import os
import signal
from pathlib import Path

import pytest

from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools import BashTool


async def wait_for_file(path: Path, timeout: float = 2) -> None:
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.01)


async def test_background_job_survives_successful_tool_call(tmp_path: Path) -> None:
    # CODE_REVIEW 4.4: a normal shell exit must not SIGKILL its descendants.
    tool = BashTool(tmp_path)
    result = await tool.execute("c", {"command": "(sleep 0.2; touch alive) >/dev/null 2>&1 & echo started"})
    assert not result.is_error
    await wait_for_file(tmp_path / "alive")


async def test_continuous_post_exit_output_is_drained(tmp_path: Path) -> None:
    # CODE_REVIEW 4.5 / Pi #5303: data extends the idle deadline beyond a fixed grace window.
    tool = BashTool(tmp_path)
    result = await tool.execute(
        "c", {"command": "(for i in $(seq 1 30); do sleep 0.1; echo tick$i; done) & echo leader"}
    )
    assert "tick30\n" in result.content[0].text


async def test_abort_still_works_after_shell_leader_exits(tmp_path: Path) -> None:
    tool = BashTool(tmp_path)
    cancellation = Cancellation()
    seen = asyncio.Event()

    async def update(result):
        if "tick" in result.content[0].text:
            seen.set()

    task = asyncio.create_task(
        tool.execute("c", {"command": "(while true; do echo tick; sleep 0.1; done) &"}, cancellation, update)
    )
    try:
        await asyncio.wait_for(seen.wait(), 1)
        cancellation.abort()
        with pytest.raises(OperationAbortedError):
            await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_timeout_kills_group_and_keeps_output(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="started"):
        await BashTool(tmp_path).execute("c", {"command": "echo started; sleep 30", "timeout": 0.1})


async def test_signal_exit_code(tmp_path: Path) -> None:
    result = await BashTool(tmp_path).execute("c", {"command": "kill -TERM $$"})
    assert result.is_error
    assert result.structured_content["exit_code"] == 128 + signal.SIGTERM


async def test_external_task_cancellation_cleans_up_descendants(tmp_path: Path) -> None:
    tool = BashTool(tmp_path)
    task = asyncio.create_task(tool.execute("c", {"command": "sleep 30 & echo $! > child.pid; wait"}))
    pid = None
    try:
        await wait_for_file(tmp_path / "child.pid")
        pid = int((tmp_path / "child.pid").read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        async with asyncio.timeout(1):
            while not _exited(pid):
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        tool.close()
        if pid is not None and not _exited(pid):
            os.kill(pid, signal.SIGKILL)


async def test_session_dispose_terminates_background_process(config, tmp_path: Path) -> None:
    from pi_agent import create_agent_session
    from pi_agent.providers import FauxProvider
    from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock

    session = create_agent_session(
        FauxProvider(
            [
                AssistantMessage(
                    content=[
                        ToolCallBlock(
                            id="c",
                            name="bash",
                            arguments={"command": "sleep 30 >/dev/null 2>&1 & echo $! > child.pid"},
                        )
                    ],
                    stop_reason="toolUse",
                ),
                AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop"),
            ]
        ),
        config=config,
    )
    pid = None
    try:
        await session.prompt("start service")
        pid = int((tmp_path / "child.pid").read_text())
        os.kill(pid, 0)
        assert not _exited(pid)
        session.dispose()
        async with asyncio.timeout(2):
            while not _exited(pid):
                await asyncio.sleep(0.01)
    finally:
        session.dispose()
        if pid is not None and not _exited(pid):
            os.kill(pid, signal.SIGKILL)


def _exited(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        return True
