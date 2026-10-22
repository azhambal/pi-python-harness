import asyncio
import base64
import json
import os
import shutil
import sys

import pytest

from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools import FindTool, GrepTool, LsTool
from pi_agent.tools._search import run_rg
from pi_agent.tools.base import validate_tool_arguments


@pytest.fixture
def repository(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "never.py").write_text("needle")
    (tmp_path / ".gitignore").write_text("ignored/\n*.log\n.env\n")
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "src" / "code.py").write_text("before\nNeedle here\nafter\n")
    (tmp_path / "src" / "nested" / "\u0442\u0435\u0441\u0442.py").write_text("needle nested\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "never.py").write_text("needle")
    (tmp_path / ".env").write_text("SYNTHETIC=needle")
    (tmp_path / "skip.log").write_text("needle")
    (tmp_path / ".hidden.py").write_text("needle hidden")
    return tmp_path


async def test_find_ignore_hidden_rooted_globs_and_no_directory_results(repository):
    tool = FindTool(repository)
    result = await tool.execute("c", {"pattern": "*.py"})
    paths = [item["path"] for item in result.structured_content["items"]]
    assert paths == [".hidden.py", "src/code.py", "src/nested/\u0442\u0435\u0441\u0442.py"]
    rooted = await tool.execute("c", {"pattern": "src/**/*.py"})
    assert [item["path"] for item in rooted.structured_content["items"]] == paths[1:]
    assert not (await tool.execute("c", {"pattern": "src"})).structured_content["items"]


async def test_find_uses_search_root_not_harness_cwd(repository, tmp_path):
    root = repository / "src"
    result = await FindTool(tmp_path).execute("c", {"pattern": "nested/*.py", "path": str(root)})
    assert result.structured_content["items"] == [{"path": "nested/\u0442\u0435\u0441\u0442.py"}]


async def test_grep_match_context_case_glob_and_literal(repository):
    tool = GrepTool(repository)
    result = await tool.execute(
        "c", {"pattern": "needle", "ignoreCase": True, "glob": "src/code.py", "context": 1}
    )
    items = result.structured_content["items"]
    assert [(item["line"], item["match"]) for item in items] == [(1, False), (2, True), (3, False)]
    assert items[1]["path"] == "src/code.py"
    assert "src/code.py:2:" in result.content[0].text
    assert "src/code.py-1-" in result.content[0].text
    assert not (await tool.execute("c", {"pattern": "Needle.*", "literal": True})).structured_content["items"]


async def test_no_match_success_invalid_regex_and_missing_path_errors(repository):
    result = await GrepTool(repository).execute("c", {"pattern": "absent"})
    assert not result.is_error and result.content[0].text == "No matches found"
    with pytest.raises(ValueError, match="regex"):
        await GrepTool(repository).execute("c", {"pattern": "["})
    with pytest.raises(FileNotFoundError):
        await FindTool(repository).execute("c", {"pattern": "*", "path": "absent"})
    with pytest.raises(NotADirectoryError):
        await LsTool(repository).execute("c", {"path": "src/code.py"})


async def test_binary_excluded_and_long_unicode_line_has_read_notice(tmp_path):
    (tmp_path / "binary").write_bytes(b"needle\0secret")
    (tmp_path / "text").write_text("needle " + "\u044f" * 2000)
    result = await GrepTool(tmp_path).execute("c", {"pattern": "needle"})
    assert len(result.structured_content["items"]) == 1
    item = result.structured_content["items"][0]
    assert item["path"] == "text" and item["line_truncated"]
    assert "use read" in item["text"] and "�" not in result.content[0].text


async def test_ls_sorted_hidden_broken_symlink_limit_and_no_contents(tmp_path):
    for name in ("b", "A", "a", ".hidden"):
        (tmp_path / name).write_text("never read these contents")
    (tmp_path / "folder").mkdir()
    (tmp_path / "link").symlink_to("folder")
    (tmp_path / "broken").symlink_to("missing")
    result = await LsTool(tmp_path).execute("c", {})
    assert result.content[0].text.splitlines() == [".hidden", "A", "a", "b", "broken", "folder/", "link/"]
    limited = await LsTool(tmp_path).execute("c", {"limit": 2})
    assert [item["name"] for item in limited.structured_content["items"]] == [".hidden", "A"]
    assert limited.details["reason"] == "entry limit"


@pytest.mark.parametrize(
    "factory,args", [(FindTool, {"pattern": "*"}), (LsTool, {}), (GrepTool, {"pattern": "needle"})]
)
async def test_full_result_envelope_obeys_byte_budget(tmp_path, factory, args):
    for n in range(50):
        (tmp_path / f"{n:03d} \u0434\u043b\u0438\u043d\u043d\u043e\u0435 \u0438\u043c\u044f.txt").write_text(
            "needle " + "\u044f" * 150
        )
    result = await factory(tmp_path, max_bytes=1024).execute("c", args)
    assert len(json.dumps(result.to_json(), ensure_ascii=True).encode()) <= 1024
    assert result.details["truncated"] and "narrow" in result.content[0].text


@pytest.mark.parametrize("value", [True, -1, 0, 1.5, 10001])
async def test_integer_limits_are_strict(tmp_path, value):
    with pytest.raises(ValueError, match="limit"):
        await LsTool(tmp_path).execute("c", {"limit": value})
    with pytest.raises(ValueError, match="Validation failed"):
        validate_tool_arguments(FindTool(tmp_path), {"pattern": "*", "limit": value})


@pytest.mark.parametrize(
    "factory,args", [(FindTool, {"pattern": "*"}), (LsTool, {}), (GrepTool, {"pattern": "needle"})]
)
async def test_cancel_before_search(tmp_path, factory, args):
    token = Cancellation()
    token.abort()
    with pytest.raises(OperationAbortedError):
        await factory(tmp_path).execute("c", args, token)


async def test_names_with_newline_leading_dash_and_spaces_are_unambiguous(tmp_path):
    name = "- weird\nfile.py"
    (tmp_path / name).write_text("-needle")
    result = await FindTool(tmp_path).execute("c", {"pattern": "*.py"})
    assert result.structured_content["items"] == [{"path": name}]
    assert "\\n" in result.content[0].text
    result = await GrepTool(tmp_path).execute("c", {"pattern": "-needle"})
    assert result.structured_content["items"][0]["path"] == name
    assert result.structured_content["items"][0]["text"] == "-needle"


async def test_non_utf8_names_preserve_exact_bytes_and_do_not_break_json(tmp_path):
    raw_name = b"name\xff.py"
    with open(os.fsencode(tmp_path) + b"/" + raw_name, "wb") as file:
        file.write(b"needle")
    (tmp_path / "name\\udcff.py").write_text("needle")
    for tool, args, key in (
        (FindTool(tmp_path), {"pattern": "*.py"}, "path"),
        (GrepTool(tmp_path), {"pattern": "needle"}, "path"),
        (LsTool(tmp_path), {}, "name"),
    ):
        result = await tool.execute("c", args)
        items = result.structured_content["items"]
        assert len(items) == 2
        encoded = [item[f"{key}_bytes_base64"] for item in items if f"{key}_bytes_base64" in item]
        assert len(encoded) == 1 and base64.b64decode(encoded[0]) == raw_name
        json.dumps(result.to_json(), ensure_ascii=False).encode("utf-8")


async def test_missing_rg_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="required.*install"):
        await FindTool(tmp_path).execute("c", {"pattern": "*"})


async def test_line_budget_includes_truncation_notice(tmp_path):
    for number in range(4):
        (tmp_path / f"{number}.py").write_text("needle")
    result = await FindTool(tmp_path, max_lines=2).execute("c", {"pattern": "*.py"})
    assert len(result.content[0].text.splitlines()) == 2
    assert len(result.structured_content["items"]) == 1
    assert result.details["truncated"]


@pytest.mark.parametrize(
    "operation", ["token", "task", "timeout", "limit", "record", "after_eof", "pipe_limit", "ignore_term"]
)
async def test_search_process_cleanup_on_abort_timeout_limit_and_oversize(tmp_path, monkeypatch, operation):
    script = tmp_path / "fake-rg"
    pid_path = tmp_path / "pid"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time, signal\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if operation == "ignore_term" else "")
        + f"open({str(pid_path)!r}, 'w').write(str(os.getpid()))\n"
        "sys.stderr.write('e' * 100000); sys.stderr.flush()\n"
        + (
            "sys.stdout.write('x' * 1100000); sys.stdout.flush()\n"
            if operation == "record"
            else "print('record', flush=True)\n"
        )
        + ("os.close(1)\n" if operation == "after_eof" else "")
        + (
            "sys.stdout.write('record\\n' * 200000); sys.stdout.flush()\n"
            if operation == "pipe_limit"
            else ""
        )
        + "time.sleep(60)\n"
    )
    script.chmod(0o700)
    monkeypatch.setattr(shutil, "which", lambda name: str(script))
    token = Cancellation()
    reached = asyncio.Event()

    def accept(record):
        reached.set()
        return operation not in ("limit", "pipe_limit")

    before = asyncio.all_tasks()
    task = asyncio.create_task(
        run_rg([], accept, token, timeout_s=0.1 if operation in ("timeout", "ignore_term") else 3)
    )
    if operation in ("token", "task", "after_eof"):
        await asyncio.wait_for(reached.wait(), 1)
        if operation == "after_eof":
            await asyncio.sleep(0.02)
        token.abort() if operation != "task" else task.cancel()
    expected = {
        "token": OperationAbortedError,
        "task": asyncio.CancelledError,
        "timeout": TimeoutError,
        "record": ValueError,
        "after_eof": OperationAbortedError,
        "ignore_term": TimeoutError,
    }
    if operation in ("limit", "pipe_limit"):
        await task
    else:
        with pytest.raises(expected[operation]):
            await task
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
    task = asyncio.create_task(run_rg([], lambda record: True, None))
    await asyncio.wait_for(reached.wait(), 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(processes) == 1 and processes[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(processes[0].pid, 0)
    assert asyncio.all_tasks() == before
