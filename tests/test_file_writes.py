"""write/edit modify files in place, as pi's writeFile does."""

import os
import shlex
import stat
import sys
from pathlib import Path

import pytest

from pi_agent import AgentConfig, create_agent_session
from pi_agent.providers import FauxProvider
from pi_agent.tools import EditTool, WriteTool
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage


async def _modify(tool_name: str, cwd: Path, path: str) -> None:
    if tool_name == "edit":
        await EditTool(cwd).execute(
            "c", {"path": path, "edits": [{"oldText": "value = 1", "newText": "value = 2"}]}
        )
    else:
        await WriteTool(cwd).execute("c", {"path": path, "content": "value = 2\n"})


@pytest.mark.parametrize("tool_name", ["edit", "write"])
async def test_tools_keep_inode_permissions_ownership_and_symlink(tmp_path: Path, tool_name: str) -> None:
    path = tmp_path / "script.py"
    path.write_bytes(b"value = 1\n")
    path.chmod(0o751)
    before = path.stat()
    link = tmp_path / "link.py"
    link.symlink_to(path.name)
    link_inode = link.lstat().st_ino
    await _modify(tool_name, tmp_path, link.name)
    after = path.stat()
    assert link.is_symlink() and link.lstat().st_ino == link_inode
    assert link.read_bytes() == b"value = 2\n"
    assert stat.S_IMODE(after.st_mode) == 0o751
    assert (after.st_uid, after.st_gid, after.st_ino) == (before.st_uid, before.st_gid, before.st_ino)
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["link.py", "script.py"]


@pytest.mark.parametrize("tool_name", ["edit", "write"])
async def test_hard_linked_aliases_see_the_new_content(tmp_path: Path, tool_name: str) -> None:
    path = tmp_path / "file.py"
    path.write_bytes(b"value = 1\n")
    alias = tmp_path / "alias.py"
    os.link(path, alias)
    await _modify(tool_name, tmp_path, path.name)
    assert path.read_bytes() == alias.read_bytes() == b"value = 2\n"
    assert path.stat().st_ino == alias.stat().st_ino


async def test_write_creates_parents_and_respects_creation_mode(tmp_path: Path) -> None:
    baseline = tmp_path / "ordinary-file.txt"
    baseline.write_text("baseline")
    expected_mode = stat.S_IMODE(baseline.stat().st_mode)
    path = tmp_path / "nested" / "file.txt"
    await WriteTool(tmp_path).execute(
        "c", {"path": "nested/file.txt", "content": "\u041f\u0440\u0438\u0432\u0435\u0442\r\n"}
    )
    assert path.read_bytes() == "\u041f\u0440\u0438\u0432\u0435\u0442\r\n".encode()
    assert stat.S_IMODE(path.stat().st_mode) == expected_mode


async def test_harness_write_fuzzy_edit_and_test_run_persist_and_resume(config: AgentConfig) -> None:
    def call(call_id: str, name: str, arguments: dict) -> AssistantMessage:
        return AssistantMessage(
            content=[ToolCallBlock(id=call_id, name=name, arguments=arguments)], stop_reason="toolUse"
        )

    provider = FauxProvider(
        [
            call(
                "write_source",
                "write",
                {"path": "calc.py", "content": "def double(x):   \n    return x + 1\n"},
            ),
            call(
                "write_test",
                "write",
                {
                    "path": "test_calc.py",
                    "content": "from calc import double\n\ndef test_double():\n    assert double(3) == 6\n",
                },
            ),
            call(
                "fix_source",
                "edit",
                {
                    "path": "calc.py",
                    "edits": [
                        {
                            "oldText": "def double(x):\n    return x + 1",
                            "newText": "def double(x):\n    return x * 2",
                        }
                    ],
                },
            ),
            call("verify", "bash", {"command": f"{shlex.quote(sys.executable)} -m pytest -q test_calc.py"}),
            AssistantMessage(content=[TextBlock(text="verified")], stop_reason="stop"),
        ]
    )
    session = create_agent_session(provider, config=config, session="new")
    try:
        await session.prompt("Fix the function and run its test")
        results = [message for message in session.messages if isinstance(message, ToolResultMessage)]
        assert len(results) == 4 and not any(message.is_error for message in results)
        assert "1 passed" in results[-1].content[0].text
        assert results[2].details["firstChangedLine"] == 1
        assert (config.cwd / "calc.py").read_text() == "def double(x):\n    return x * 2\n"
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        results = [message for message in resumed.messages if isinstance(message, ToolResultMessage)]
        assert len(results) == 4
        assert results[2].details["firstChangedLine"] == 1
        assert "1 passed" in results[-1].content[0].text
    finally:
        resumed.dispose()
