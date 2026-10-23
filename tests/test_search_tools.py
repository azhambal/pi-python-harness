"""grep/find/ls beyond the pi parity fixtures: limits, truncation, odd names, processes."""

import asyncio
import json
import os
import sys

import pytest

from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools import FindTool, GrepTool, LsTool, _search
from pi_agent.tools._search import run_lines

needs_rg = pytest.mark.skipif(_search.get_tool_path("rg") is None, reason="rg is not installed")
needs_fd = pytest.mark.skipif(_search.get_tool_path("fd") is None, reason="fd is not installed")


def text(result) -> str:
    return result.content[0].text


async def test_ls_fractional_and_string_limits_format_like_javascript(tmp_path):
    for name in "abc":
        (tmp_path / name).write_text("")
    result = await LsTool(tmp_path).execute("c", {"limit": 1.5})
    assert text(result) == "a\nb\n\n[1.5 entries limit reached. Use limit=3 for more]"
    assert result.details == {"entryLimitReached": 1.5}


async def test_ls_byte_truncation_notice_and_details(tmp_path):
    for number in range(40):
        (tmp_path / f"{number:02d}-{'x' * 40}").write_text("")
    result = await LsTool(tmp_path, max_bytes=1024).execute("c", {})
    assert text(result).endswith("\n\n[1.0KB limit reached]")
    assert result.details["truncation"]["truncated_by"] == "bytes"


async def test_ls_names_with_invalid_utf8_use_replacement_characters(tmp_path):
    with open(os.fsencode(tmp_path) + b"/name\xff", "wb"):
        pass
    result = await LsTool(tmp_path).execute("c", {})
    assert text(result) == "name�"
    json.dumps(result.to_json(), ensure_ascii=False).encode("utf-8")


async def test_ls_symlinked_directory_gets_slash(tmp_path):
    (tmp_path / "folder").mkdir()
    (tmp_path / "link").symlink_to("folder")
    assert text(await LsTool(tmp_path).execute("c", {})) == "folder/\nlink/"


@needs_rg
async def test_grep_long_line_counts_utf16_units(tmp_path):
    (tmp_path / "f.txt").write_text("needle " + "\U0001f600" * 300 + "\n")
    result = await GrepTool(tmp_path).execute("c", {"pattern": "needle"})
    row, notice = text(result).split("\n\n")
    # 7 ASCII units + 246 emoji (2 units each) + 1 unit: the cut pair becomes U+FFFD.
    assert row == "f.txt:1: needle " + "\U0001f600" * 246 + "�... [truncated]"
    assert notice == "[Some lines truncated to 500 chars. Use read tool to see full lines]"
    assert result.details == {"linesTruncated": True}


@needs_rg
async def test_grep_context_reads_crlf_files_and_skips_binary(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"one\r\nneedle\r\nthree\r\n")
    (tmp_path / "b.bin").write_bytes(b"needle\0binary")
    result = await GrepTool(tmp_path).execute("c", {"pattern": "needle", "context": 1})
    assert text(result) == "a.txt-1- one\na.txt:2: needle\na.txt-3- three"


@needs_rg
async def test_grep_invalid_regex_reports_ripgrep_stderr(tmp_path):
    with pytest.raises(RuntimeError, match="regex parse error"):
        await GrepTool(tmp_path).execute("c", {"pattern": "("})


@needs_fd
async def test_find_nested_repository_boundary_stops_parent_ignore_rules(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("*.py\n")
    (tmp_path / "nested" / ".git").mkdir(parents=True)
    (tmp_path / "nested" / ".gitignore").write_text("local.py\n")
    (tmp_path / "nested" / "kept.py").write_text("")
    (tmp_path / "nested" / "local.py").write_text("")
    result = await FindTool(tmp_path).execute("c", {"path": "nested", "pattern": "*.py"})
    assert text(result) == "kept.py"


@pytest.mark.parametrize(
    "factory,args,message",
    [
        (FindTool, {"pattern": "*"}, "fd is not available and could not be downloaded"),
        (GrepTool, {"pattern": "x"}, r"ripgrep \(rg\) is not available and could not be downloaded"),
    ],
)
async def test_missing_binaries_use_pi_messages(tmp_path, monkeypatch, factory, args, message):
    monkeypatch.setattr(_search.shutil, "which", lambda name: None)
    monkeypatch.setattr("pi_agent.config.DEFAULT_AGENT_DIR", tmp_path / "agent")
    with pytest.raises(FileNotFoundError, match=message):
        await factory(tmp_path).execute("c", args)


async def test_agent_bin_directory_wins_over_path(tmp_path, monkeypatch):
    local = tmp_path / "agent" / "bin" / "fd"
    local.parent.mkdir(parents=True)
    local.write_text("#!/bin/sh\n")
    local.chmod(0o755)
    monkeypatch.setattr("pi_agent.config.DEFAULT_AGENT_DIR", tmp_path / "agent")
    assert _search.get_tool_path("fd") == str(local)


@pytest.mark.parametrize(
    "factory,args", [(FindTool, {"pattern": "*"}), (LsTool, {}), (GrepTool, {"pattern": "needle"})]
)
async def test_cancel_before_search(tmp_path, factory, args):
    token = Cancellation()
    token.abort()
    with pytest.raises(OperationAbortedError, match="Operation aborted"):
        await factory(tmp_path).execute("c", args, token)


async def test_lines_split_like_node_readline(tmp_path):
    script = tmp_path / "emit"
    script.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stdout.buffer.write(b'a\\r\\nb\\rc\\n\\nd\\xff')\n"
    )
    script.chmod(0o700)
    lines: list[str] = []
    outcome = await run_lines(str(script), [], lambda line: lines.append(line), None)
    assert lines == ["a", "b", "c", "", "d�"]
    assert outcome.code == 0


@pytest.mark.parametrize("operation", ["token", "task", "limit", "after_eof", "pipe_limit", "ignore_term"])
async def test_search_process_cleanup_on_abort_and_limit(tmp_path, operation):
    script = tmp_path / "fake-search"
    pid_path = tmp_path / "pid"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time, signal\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if operation == "ignore_term" else "")
        + f"open({str(pid_path)!r}, 'w').write(str(os.getpid()))\n"
        "sys.stderr.write('e' * 100000); sys.stderr.flush()\n"
        "print('record', flush=True)\n"
        + ("os.close(1)\n" if operation == "after_eof" else "")
        + (
            "sys.stdout.write('record\\n' * 200000); sys.stdout.flush()\n"
            if operation == "pipe_limit"
            else ""
        )
        + "time.sleep(60)\n"
    )
    script.chmod(0o700)
    token = Cancellation()
    reached = asyncio.Event()

    def on_line(line):
        reached.set()
        return operation in ("limit", "pipe_limit", "ignore_term")

    before = asyncio.all_tasks()
    task = asyncio.create_task(run_lines(str(script), [], on_line, token))
    if operation in ("token", "task", "after_eof"):
        await asyncio.wait_for(reached.wait(), 1)
        if operation == "after_eof":
            await asyncio.sleep(0.02)
        token.abort() if operation != "task" else task.cancel()
    expected = {
        "token": OperationAbortedError,
        "task": asyncio.CancelledError,
        "after_eof": OperationAbortedError,
    }
    if operation in expected:
        with pytest.raises(expected[operation]):
            await task
    else:
        outcome = await asyncio.wait_for(task, 5)
        assert outcome.code is None  # Ended by a signal, as Node reports it.
    assert pid_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)
    assert asyncio.all_tasks() == before


async def test_cancellation_during_spawn_joins_and_terminates_owned_process(monkeypatch):
    original = asyncio.create_subprocess_exec
    reached, release = asyncio.Event(), asyncio.Event()
    processes = []

    async def spawning(*args, **kwargs):
        reached.set()
        await release.wait()
        process = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawning)
    before = asyncio.all_tasks()
    task = asyncio.create_task(run_lines("unused", [], lambda line: True, None))
    await asyncio.wait_for(reached.wait(), 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(processes) == 1 and processes[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(processes[0].pid, 0)
    assert asyncio.all_tasks() == before
