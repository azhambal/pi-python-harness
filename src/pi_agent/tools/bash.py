"""POSIX shell execution with bounded output and owned process-group cleanup."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import shutil
import signal
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from pi_agent._js import number_to_string
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools.base import Tool, ToolResult, ToolUpdateCallback
from pi_agent.tools.output import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    OutputAccumulator,
    OutputSnapshot,
    format_size,
)
from pi_agent.types import JsonValue, TextBlock

#: Structured output cap for programmatic callers (mirrors Pi's 1 MiB).
STRUCTURED_OUTPUT_MAX_BYTES = 1024 * 1024

#: SIGTERM-to-SIGKILL grace when killing a group, the drain window given to
#: readers whose pipe is held open by descendants, and the bound for
#: shielded cleanup waits.
KILL_GRACE_SECONDS = 2.0
DRAIN_GRACE_SECONDS = 1.0

_MAX_TIMEOUT_S = 2_147_483_647

#: Process.wait() can wait on inherited pipes; only leader reaping is polled.
_EXIT_POLL_SECONDS = 0.05


def resolve_timeout(timeout: object) -> float | None:
    """Validate a timeout in seconds: finite, positive, not a bool; None passes."""
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if not math.isfinite(timeout):
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > _MAX_TIMEOUT_S:
        raise ValueError(f"Invalid timeout: maximum is {_MAX_TIMEOUT_S} seconds")
    return float(timeout)


#: Session metadata Pi exposes to commands (``resolveSpawnContext``).
SESSION_ENV_KEYS = ("PI_SESSION_ID", "PI_SESSION_FILE", "PI_PROVIDER", "PI_MODEL", "PI_REASONING_LEVEL")


def find_shell(shell_path: str | None = None) -> tuple[str, list[str]]:
    """Pi's ``getShellConfig`` on Unix: shellPath, /bin/bash, bash on PATH, then sh."""
    if shell_path:
        if os.access(shell_path, os.F_OK):
            return shell_path, ["-c"]
        raise FileNotFoundError(f"Custom shell path not found: {shell_path}")
    if os.access("/bin/bash", os.F_OK):
        return "/bin/bash", ["-c"]
    on_path = shutil.which("bash")
    if on_path:
        return on_path, ["-c"]
    return "sh", ["-c"]


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    """Signal a whole process group; an empty or gone group is not an error."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, sig)


async def _wait_bounded(awaitable: Awaitable[object], timeout: float) -> bool:
    """Await with a bound; the underlying task is always cancelled and joined."""
    task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout)
        return True
    except TimeoutError:
        return False
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))


def _task_exception(task: asyncio.Future) -> BaseException | None:
    """A done, uncancelled task's exception, or None."""
    if not task.done() or task.cancelled():
        return None
    with contextlib.suppress(BaseException):
        return task.exception()
    return None


class ProcessRegistry:
    """Own shell process groups until the tool/session is closed.

    Standalone BashTool users call close(); AgentSession does that in dispose().
    Descendants creating a new session (setsid) are outside this ownership.
    """

    def __init__(self) -> None:
        self._groups: set[int] = set()

    def add(self, pgid: int) -> None:
        self._groups.add(pgid)

    def discard(self, pgid: int) -> None:
        self._groups.discard(pgid)

    def close(self) -> None:
        for pgid in self._groups:
            _signal_group(pgid, signal.SIGKILL)
        self._groups.clear()


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever "
        "is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout "
        "in seconds."
    )
    argument_conversion = "typebox"
    prompt_snippet = "Execute bash commands (ls, grep, find, etc.)"
    parameters: dict[str, JsonValue] = {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
        },
    }

    def __init__(
        self,
        cwd: Path | str,
        default_timeout_s: float | None = None,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        *,
        command_prefix: str | None = None,
        shell_path: str | None = None,
        expose_session_environment: bool = True,
        session_environment: Callable[[], Mapping[str, str]] | None = None,
    ) -> None:
        self.cwd = Path(cwd)
        self.default_timeout_s = resolve_timeout(default_timeout_s)
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        #: Prepended to every command on its own line (Pi's ``shellCommandPrefix``).
        self.command_prefix = command_prefix
        self.shell_path = shell_path
        #: Pi's ``exposeSessionEnvironment``: export PI_* session metadata and
        #: tell the model about it.
        self.expose_session_environment = expose_session_environment
        self.session_environment = session_environment
        self.guidelines = (
            ("You can inspect PI_* environment variables for current model and session details.",)
            if expose_session_environment
            else ()
        )
        self.process_registry = ProcessRegistry()

    def close(self) -> None:
        """Terminate owned background groups. Full-output files stay, as in Pi."""
        self.process_registry.close()

    def _spawn_env(self) -> dict[str, str]:
        env = dict(os.environ)
        for key in SESSION_ENV_KEYS:
            env.pop(key, None)
        if self.expose_session_environment and self.session_environment is not None:
            env.update({key: value for key, value in self.session_environment().items() if value})
        return env

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        del tool_call_id
        command = str(args["command"])
        if self.command_prefix:
            command = f"{self.command_prefix}\n{command}"
        raw_timeout = args.get("timeout") if args.get("timeout") is not None else self.default_timeout_s
        timeout = resolve_timeout(raw_timeout)
        cwd = self.cwd
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {cwd}\nCannot execute bash commands.")
        if cancellation is not None and cancellation.aborted:
            raise OperationAbortedError("Operation aborted")

        accumulator = OutputAccumulator(
            max_lines=self.max_lines,
            max_bytes=self.max_bytes,
            temp_file_prefix="pi-bash",
        )
        shell, shell_args = find_shell(self.shell_path)
        started_at = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *([shell, *shell_args, command]),
                cwd=str(cwd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,  # own process group; pgid == process.pid
                env=self._spawn_env(),
            )
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Failed to start shell: {error}") from error

        self.process_registry.add(process.pid)
        output_activity = asyncio.Event()
        pump_task = asyncio.create_task(self._pump(process.stdout, accumulator, on_update, output_activity))
        try:
            outcome, reader_error = await self._race_exit(
                process, pump_task, cancellation, timeout, output_activity
            )
            if outcome in ("timeout", "aborted", "reader-failed"):
                await self._terminate_group(process)
            late_error = await self._settle_pumps(process, pump_task)
            if late_error is not None and reader_error is None:
                reader_error = late_error
            if reader_error is not None:
                await self._terminate_group(process)
        except asyncio.CancelledError:
            # External task cancellation: kill and reap the group, settle the
            # reader, close the spill file (even when finishing throws), then
            # let the cancellation fly.
            await self._hard_cleanup(process, pump_task)
            try:
                accumulator.finish()
            finally:
                accumulator.close_temp_file()
            raise

        except BaseException:
            await self._hard_cleanup(process, pump_task)
            try:
                accumulator.finish()
            finally:
                accumulator.close_temp_file()
            raise
        finally:
            # Keep only groups with live descendants. A normal exit leaves those
            # descendants available to later calls; failures have killed the group.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                self.process_registry.discard(process.pid)

        # All outcome paths converge here: the leader is reaped, readers are
        # settled, and the accumulator holds everything the group wrote.
        # finish()/snapshot()/close_temp_file() are synchronous by design;
        # the spill file closes even when finishing or snapshotting throws.
        try:
            accumulator.finish()
            snapshot = accumulator.snapshot(persist_if_truncated=True)
            empty_text = "" if outcome in ("aborted", "timeout") else "(no output)"
            output_text, details = self._format_output(snapshot, accumulator, empty_text)
        finally:
            accumulator.close_temp_file()
        if reader_error is not None:
            raise RuntimeError(
                f"Output reader failed after {time.monotonic() - started_at:.1f}s; collected output:\n"
                f"{output_text}"
            ) from reader_error
        if outcome == "aborted":
            raise OperationAbortedError(_append_status(output_text, "Command aborted"))
        if outcome == "timeout":
            assert isinstance(raw_timeout, (int, float))  # resolve_timeout accepted it.
            raise TimeoutError(
                _append_status(
                    output_text, f"Command timed out after {number_to_string(raw_timeout)} seconds"
                )
            )

        exit_code = process.returncode if process.returncode is not None else 1
        if process.returncode is not None and process.returncode < 0:
            exit_code = 128 + (-process.returncode)
        wall_time_s = round((time.monotonic() - started_at) * 10) / 10
        full_output = accumulator.read_full_output(STRUCTURED_OUTPUT_MAX_BYTES)
        structured: dict[str, JsonValue] = {
            "output": full_output.content,
            "truncated": full_output.truncated,
            "exit_code": exit_code,
            "wall_time_seconds": wall_time_s,
        }
        if full_output.truncated and snapshot.full_output_path:
            structured["full_output_path"] = snapshot.full_output_path
        if exit_code != 0:
            return ToolResult(
                content=[
                    TextBlock(text=_append_status(output_text, f"Command exited with code {exit_code}"))
                ],
                details=details,
                structured_content=structured,
                is_error=True,
            )
        return ToolResult(
            content=[TextBlock(text=output_text)], details=details, structured_content=structured
        )

    # ------------------------------------------------------------------
    # Process plumbing
    # ------------------------------------------------------------------

    @staticmethod
    async def _pump(
        stream: asyncio.StreamReader | None,
        accumulator: OutputAccumulator,
        on_update: ToolUpdateCallback | None,
        output_activity: asyncio.Event,
    ) -> None:
        """Read the merged pipe into the accumulator, emitting live updates.

        Failures (accumulator or observer callback) propagate: the caller
        races this task and kills the group so a child writing into a full
        pipe cannot hang the call, then reports the failure with the output
        collected so far.
        """
        if stream is None:
            return
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            accumulator.append(chunk)
            output_activity.set()
            if on_update is not None:
                snapshot = accumulator.snapshot(persist_if_truncated=True)
                await on_update(
                    ToolResult(
                        content=[TextBlock(text=snapshot.content)] if snapshot.content else [],
                        details={
                            "truncation": (
                                snapshot.truncation.to_json() if snapshot.truncation.truncated else None
                            ),
                            "full_output_path": snapshot.full_output_path,
                        },
                    )
                )

    @staticmethod
    async def _watch_returncode(process: asyncio.subprocess.Process) -> None:
        """Poll the leader's returncode at a bounded interval.

        This watcher is independent of pipe EOF: ``Process.wait()`` can be
        held up by transport bookkeeping while a background descendant still
        holds the pipe, whereas SIGCHLD-based reaping sets ``returncode`` as
        soon as the leader exits. Observing it here lets the race conclude
        and the pipe be drained instead of hanging.
        """
        while process.returncode is None:
            await asyncio.sleep(_EXIT_POLL_SECONDS)

    @staticmethod
    async def _race_exit(
        process: asyncio.subprocess.Process,
        pump_task: asyncio.Task[None],
        cancellation: Cancellation | None,
        timeout_s: float | None,
        output_activity: asyncio.Event,
    ) -> tuple[str, BaseException | None]:
        """Race leader exit + idle drain against reader errors, abort, and timeout.

        EOF alone never ends a running command. Cancellation/timeout stay active
        throughout post-exit draining. All auxiliary tasks are joined on return.
        """
        exit_task = asyncio.create_task(
            BashTool._wait_for_exit_and_output(process, pump_task, output_activity)
        )
        aux_tasks: set[asyncio.Task[None]] = {exit_task}
        tasks: set[asyncio.Task[None]] = {exit_task, pump_task}
        timeout_wait: asyncio.Task[None] | None = None
        if timeout_s is not None:
            timeout_wait = asyncio.create_task(asyncio.sleep(timeout_s))
            tasks.add(timeout_wait)
            aux_tasks.add(timeout_wait)
        if cancellation is not None:
            abort_watch = asyncio.create_task(cancellation.wait())
            tasks.add(abort_watch)
            aux_tasks.add(abort_watch)

        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                # Reader failure outranks a simultaneous clean exit: the
                # output stream is broken either way.
                if pump_task.done() and not pump_task.cancelled():
                    reader_error: BaseException | None = None
                    if pump_task.exception() is not None:
                        reader_error = pump_task.exception()
                    if reader_error is not None:
                        return "reader-failed", reader_error
                if cancellation is not None and cancellation.aborted:
                    return "aborted", None
                if timeout_wait is not None and timeout_wait.done():
                    return "timeout", None
                if exit_task.done() and not exit_task.cancelled():
                    exit_task.result()
                    return "exit", None
            return "exit", None
        finally:
            for task in aux_tasks:
                task.cancel()
            # Join the auxiliary tasks, including ones that already
            # completed; an external cancellation of this coroutine cannot
            # skip this (the pump is left running on purpose).
            with contextlib.suppress(BaseException):
                await asyncio.shield(asyncio.gather(*aux_tasks, return_exceptions=True))

    @staticmethod
    async def _wait_for_exit_and_output(
        process: asyncio.subprocess.Process,
        pump_task: asyncio.Task[None],
        output_activity: asyncio.Event,
    ) -> None:
        """After leader exit, every output chunk re-arms the idle deadline (Pi #5303)."""
        await BashTool._watch_returncode(process)
        while not pump_task.done():
            output_activity.clear()
            activity = asyncio.create_task(output_activity.wait())
            try:
                done, _ = await asyncio.wait(
                    (pump_task, activity), timeout=DRAIN_GRACE_SECONDS, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    return
            finally:
                activity.cancel()
                await asyncio.gather(activity, return_exceptions=True)

    @staticmethod
    async def _terminate_group(process: asyncio.subprocess.Process) -> None:
        """SIGTERM the group, escalate to SIGKILL, and reap the leader.

        Bounded and tracked: the shielded waits cannot be abandoned by an
        outer cancellation.
        """
        pgid = process.pid  # start_new_session makes the pgid equal to the pid
        if process.returncode is None:
            _signal_group(pgid, signal.SIGTERM)
            if not await _wait_bounded(process.wait(), KILL_GRACE_SECONDS):
                _signal_group(pgid, signal.SIGKILL)
                await _wait_bounded(process.wait(), KILL_GRACE_SECONDS)
        # Even with the leader reaped, the group may still hold descendants.
        _signal_group(pgid, signal.SIGKILL)

    async def _settle_pumps(
        self, process: asyncio.subprocess.Process, pump_task: asyncio.Task[None]
    ) -> BaseException | None:
        """Release inherited pipe handles after EOF or idle, without killing descendants."""
        if not pump_task.done():
            pump_task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.shield(asyncio.gather(pump_task, return_exceptions=True))
        # asyncio.Process exposes no public pipe-close API. Closing its transport
        # after leader exit releases inherited handles without signalling the group.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()
        await _wait_bounded(process.wait(), KILL_GRACE_SECONDS)
        return _task_exception(pump_task)

    @staticmethod
    async def _hard_cleanup(process: asyncio.subprocess.Process, pump_task: asyncio.Task[None]) -> None:
        """Failure-path cleanup: SIGKILL the group, reap, settle the reader."""
        with contextlib.suppress(BaseException):
            await BashTool._terminate_group(process)
        pump_task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.shield(asyncio.gather(pump_task, return_exceptions=True))
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _format_output(
        self, snapshot: OutputSnapshot, accumulator: OutputAccumulator, empty_text: str = "(no output)"
    ) -> tuple[str, dict[str, JsonValue] | None]:
        truncation = snapshot.truncation
        text = snapshot.content or empty_text
        details: dict[str, JsonValue] | None = None
        if truncation.truncated:
            details = {
                "truncation": truncation.to_json(),
                "full_output_path": snapshot.full_output_path,
            }
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                last_line_size = format_size(accumulator.get_last_line_bytes())
                text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line} "
                    f"(line is {last_line_size}). Full output: {snapshot.full_output_path}]"
                )
            elif truncation.truncated_by == "lines":
                text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. Full output: {snapshot.full_output_path}]"
            else:
                text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(self.max_bytes)} limit). Full output: {snapshot.full_output_path}]"
                )
        return text, details


def _append_status(text: str, status: str) -> str:
    return f"{text}\n\n{status}" if text else status
