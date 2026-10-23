"""ripgrep/fd lookup and line-streamed child processes for grep and find.

Mirrors Pi's ``utils/tools-manager.ts`` lookup (the agent ``bin`` directory
first, then ``PATH``) without its download fallback: a container image is
expected to ship ``rg`` and ``fd``. Child processes follow Node's
``spawn`` + ``readline`` behaviour: inherited environment, stdout split on
``\\n``, ``\\r\\n`` or ``\\r``, and a ``None`` exit code when killed by a signal.
"""

from __future__ import annotations

import asyncio
import codecs
import errno
import os
import re
import shutil
import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pi_agent.cancellation import Cancellation, OperationAbortedError

_SYSTEM_NAMES = {"rg": ("rg",), "fd": ("fd", "fdfind")}
_LINE_END = re.compile(r"\r\n|\n|\r")
_PARTIAL_LINE_END = re.compile(r"\r\n|\n|\r(?!$)")
_KILL_GRACE_S = 1.0


def get_tool_path(tool: Literal["rg", "fd"]) -> str | None:
    """Pi's ``getToolPath``: ``<agent dir>/bin/<tool>``, then the system ``PATH``."""
    from pi_agent.config import DEFAULT_AGENT_DIR  # config imports tools indirectly.

    local = DEFAULT_AGENT_DIR / "bin" / tool
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    for name in _SYSTEM_NAMES[tool]:
        if shutil.which(name) is not None:
            return name
    return None


@dataclass
class ProcessOutcome:
    #: Exit status, or None when a signal ended the process (Node's ``code``).
    code: int | None
    stderr: str


class SpawnError(Exception):
    """The executable could not be started; the message is Node's ``spawn`` error text."""


async def run_lines(
    executable: str,
    args: list[str],
    on_line: Callable[[str], bool | None],
    cancellation: Cancellation | None,
) -> ProcessOutcome:
    """Run a child, feeding each stdout line to ``on_line``; a true return kills the child."""
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            executable,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    )
    try:
        process = await asyncio.shield(spawning)
    except OSError as error:
        code = errno.errorcode.get(error.errno or 0, "UNKNOWN")
        raise SpawnError(f"spawn {executable} {code}") from None
    except BaseException:
        # Cancelled while spawning: the child may still start, and it is ours to stop.
        try:
            orphan = await spawning
        except OSError:
            pass
        else:
            await _terminate(orphan)
        raise
    assert process.stdout is not None and process.stderr is not None
    stderr = bytearray()
    loop = asyncio.get_running_loop()

    def kill() -> None:
        """Node's ``child.kill()``, escalated if the group ignores SIGTERM."""
        if process.returncode is None:
            _signal_group(process, signal.SIGTERM)
            loop.call_later(_KILL_GRACE_S, _signal_group, process, signal.SIGKILL)

    async def read_errors() -> None:
        assert process.stderr is not None
        while chunk := await process.stderr.read(65536):
            stderr.extend(chunk)

    async def read_lines() -> None:
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        stopped = False

        def emit(lines: list[str]) -> None:
            nonlocal stopped
            for line in lines:
                if not stopped and on_line(line):
                    stopped = True
                    kill()

        while chunk := await process.stdout.read(65536):
            pending += decoder.decode(chunk)
            # A trailing \r waits for the next chunk: it may start with \n.
            lines = _PARTIAL_LINE_END.split(pending)
            pending = lines.pop()
            emit(lines)
        lines = _LINE_END.split(pending + decoder.decode(b"", final=True))
        last = lines.pop()
        emit([*lines, last] if last else lines)

    readers = asyncio.gather(read_lines(), read_errors())
    aborted = asyncio.create_task(cancellation.wait()) if cancellation is not None else None
    try:
        if aborted is not None:
            await asyncio.wait((readers, aborted), return_when=asyncio.FIRST_COMPLETED)
            if cancellation is not None and cancellation.aborted:
                raise OperationAbortedError("Operation aborted")
        await readers
        returncode = await process.wait()
    finally:
        pending: list[asyncio.Future[Any]] = [readers]
        if aborted is not None:
            pending.append(aborted)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await _terminate(process)
    return ProcessOutcome(
        code=None if returncode < 0 else returncode,
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def _signal_group(process: asyncio.subprocess.Process, signum: int) -> None:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


async def _terminate(process: asyncio.subprocess.Process) -> None:
    """Stop and reap a child that is still running."""
    if process.returncode is not None:
        return
    _signal_group(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), _KILL_GRACE_S)
    except TimeoutError:
        _signal_group(process, signal.SIGKILL)
        await process.wait()
