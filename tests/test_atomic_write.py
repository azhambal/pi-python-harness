import errno
import os
import shlex
import stat
import sys
from pathlib import Path

import pytest

from pi_agent import AgentConfig, create_agent_session
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.providers import FauxProvider
from pi_agent.tools import EditTool, WriteTool, _atomic
from pi_agent.tools._atomic import AtomicWriteCommittedError, ConcurrentFileChangeError, atomic_write
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stage", ["write", "flush", "fsync", "replace"])
def test_failure_before_commit_keeps_original_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    stage: str,
) -> None:
    path = tmp_path / "file.txt"
    if existing:
        path.write_bytes(b"original")
    original_fdopen = os.fdopen
    original_fsync = os.fsync

    class FailingFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def fileno(self):
            return self.file.fileno()

        def write(self, content):
            if stage == "write":
                self.file.write(content[:3])
                self.file.flush()
                raise OSError(errno.ENOSPC, "partial write failed")
            return self.file.write(content)

        def flush(self):
            self.file.flush()
            if stage == "flush":
                raise OSError(errno.ENOSPC, "flush failed")

    def fdopen(fd, mode):
        file = original_fdopen(fd, mode)
        return FailingFile(file) if mode == "wb" and stage in ("write", "flush") else file

    def fsync(fd):
        if stage == "fsync" and stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "file fsync failed")
        original_fsync(fd)

    def replace(*args, **kwargs):
        raise OSError(errno.EACCES, "replace failed")

    monkeypatch.setattr(_atomic.os, "fdopen", fdopen)
    monkeypatch.setattr(_atomic.os, "fsync", fsync)
    if stage == "replace":
        monkeypatch.setattr(_atomic.os, "replace", replace)
    with pytest.raises(OSError):
        atomic_write(path, b"replacement data")
    assert path.read_bytes() == b"original" if existing else not path.exists()
    assert not list(tmp_path.glob(".pi-write-*.tmp"))


def test_directory_fsync_failure_reports_content_already_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"original")
    original_fsync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "directory sync failed")
        original_fsync(fd)

    monkeypatch.setattr(_atomic.os, "fsync", fsync)
    with pytest.raises(AtomicWriteCommittedError, match="installed.*durability") as failure:
        atomic_write(path, b"new")
    assert failure.value.committed
    assert path.read_bytes() == b"new"
    assert not list(tmp_path.glob(".pi-write-*.tmp"))


def test_cancellation_before_commit_keeps_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"original")
    cancellation = Cancellation()
    original_fsync = os.fsync

    def fsync(fd):
        original_fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode):
            cancellation.abort()

    monkeypatch.setattr(_atomic.os, "fsync", fsync)
    with pytest.raises(OperationAbortedError):
        atomic_write(path, b"new", cancellation=cancellation)
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".pi-write-*.tmp"))


def test_cancellation_after_commit_does_not_report_a_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"original")
    cancellation = Cancellation()
    original_replace = os.replace

    def replace(*args, **kwargs):
        original_replace(*args, **kwargs)
        cancellation.abort()

    monkeypatch.setattr(_atomic.os, "replace", replace)
    atomic_write(path, b"new", cancellation=cancellation)
    assert path.read_bytes() == b"new"


@pytest.mark.parametrize("existing", [False, True])
def test_external_change_during_preparation_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    path = tmp_path / "file.txt"
    if existing:
        path.write_bytes(b"original")
    original_fsync = os.fsync

    def fsync(fd):
        original_fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode):
            path.write_bytes(b"external edit")

    monkeypatch.setattr(_atomic.os, "fsync", fsync)
    with pytest.raises(ConcurrentFileChangeError):
        atomic_write(path, b"new")
    assert path.read_bytes() == b"external edit"
    assert not list(tmp_path.glob(".pi-write-*.tmp"))


async def test_edit_refuses_change_after_read_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pi_agent.tools import edit

    path = tmp_path / "file.txt"
    path.write_text("value = 1\n")
    original_apply = edit.apply_edits

    def changed(*args):
        result = original_apply(*args)
        path.write_text("external edit\n")
        return result

    monkeypatch.setattr(edit, "apply_edits", changed)
    with pytest.raises(ConcurrentFileChangeError):
        await EditTool(tmp_path).execute(
            "c", {"path": path.name, "edits": [{"oldText": "value = 1", "newText": "value = 2"}]}
        )
    assert path.read_text() == "external edit\n"


@pytest.mark.parametrize("tool_name", ["edit", "write"])
async def test_tools_keep_permissions_ownership_and_symlink(tmp_path: Path, tool_name: str) -> None:
    path = tmp_path / "script.py"
    path.write_bytes(b"value = 1\n")
    path.chmod(0o751)
    before = path.stat()
    link = tmp_path / "link.py"
    link.symlink_to(path.name)
    link_inode = link.lstat().st_ino
    if tool_name == "edit":
        await EditTool(tmp_path).execute(
            "c", {"path": link.name, "edits": [{"oldText": "value = 1", "newText": "value = 2"}]}
        )
    else:
        await WriteTool(tmp_path).execute("c", {"path": link.name, "content": "value = 2\n"})
    after = path.stat()
    assert link.is_symlink() and link.lstat().st_ino == link_inode
    assert link.read_bytes() == b"value = 2\n"
    assert stat.S_IMODE(after.st_mode) == 0o751
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert after.st_ino != before.st_ino
    assert not list(tmp_path.glob(".pi-write-*.tmp"))


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
    assert not list(path.parent.glob(".pi-write-*.tmp"))


@pytest.mark.parametrize("tool_name", ["edit", "write"])
async def test_hard_links_are_refused_without_breaking_aliases(tmp_path: Path, tool_name: str) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"original")
    alias = tmp_path / "alias.txt"
    os.link(path, alias)
    with pytest.raises(ValueError, match="hard-linked"):
        if tool_name == "edit":
            await EditTool(tmp_path).execute(
                "c", {"path": path.name, "edits": [{"oldText": "original", "newText": "new"}]}
            )
        else:
            await WriteTool(tmp_path).execute("c", {"path": path.name, "content": "new"})
    assert path.read_bytes() == alias.read_bytes() == b"original"
    assert path.stat().st_ino == alias.stat().st_ino


def test_extended_attributes_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"old")
    try:
        os.setxattr(path, "user.pi-review", b"marker")
    except OSError as error:
        if error.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
            pytest.skip("Filesystem does not support user xattrs")
        raise
    atomic_write(path, b"new")
    assert os.getxattr(path, "user.pi-review") == b"marker"


def test_temp_name_collision_does_not_delete_an_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedID:
        hex = "fixed"

    monkeypatch.setattr(_atomic, "uuid4", FixedID)
    path = tmp_path / "file.txt"
    path.write_bytes(b"old")
    foreign = tmp_path / ".pi-write-fixed.tmp"
    foreign.write_bytes(b"belongs to another writer")
    with pytest.raises(FileExistsError):
        atomic_write(path, b"new")
    assert path.read_bytes() == b"old"
    assert foreign.read_bytes() == b"belongs to another writer"


def test_nonregular_target_is_refused_before_open_for_write(tmp_path: Path) -> None:
    path = tmp_path / "pipe"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="not a regular file"):
        atomic_write(path, b"new")
    assert stat.S_ISFIFO(path.stat().st_mode)


def test_open_reader_keeps_old_snapshot_and_new_reader_sees_complete_replacement(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    old = b"old content" * 1000
    new = b"new content" * 1000
    path.write_bytes(old)
    with path.open("rb") as reader:
        atomic_write(path, new)
        assert reader.read() == old
    assert path.read_bytes() == new


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
        assert results[2].details["first_changed_line"] == 1
        assert (config.cwd / "calc.py").read_text() == "def double(x):\n    return x * 2\n"
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        results = [message for message in resumed.messages if isinstance(message, ToolResultMessage)]
        assert len(results) == 4
        assert results[2].details["first_changed_line"] == 1
        assert "1 passed" in results[-1].content[0].text
    finally:
        resumed.dispose()
