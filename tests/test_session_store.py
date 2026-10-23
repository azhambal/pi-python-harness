import json
from pathlib import Path

import pytest

from pi_agent import AgentConfig, FatalPersistenceError, create_agent_session
from pi_agent.providers import FauxProvider
from pi_agent.providers._normalize import transform_messages
from pi_agent.session_store import (
    SessionCorruptError,
    SessionLockError,
    SessionStore,
    SessionUnusableError,
    read_session_file,
)
from pi_agent.types import AssistantMessage, ToolCallBlock, ToolResultMessage, UserMessage


def persisted_store(tmp_path: Path) -> SessionStore:
    store = SessionStore.create(tmp_path, tmp_path / "sessions")
    store.append_message(UserMessage(content="hello"))
    return store


@pytest.mark.parametrize("fragment", [b'{"type":"message",', b"\xf0\x9f"])
def test_torn_tail_recovery_is_idempotent(tmp_path: Path, fragment: bytes) -> None:
    store = persisted_store(tmp_path)
    path = store.session_file
    store.close()
    original = path.read_bytes()
    path.write_bytes(original + fragment)
    with pytest.raises(SessionCorruptError):
        read_session_file(path)  # inspection cannot repair
    for _ in range(2):
        reopened = SessionStore.open(path)
        try:
            assert len(reopened.entries) == 1
            assert path.read_bytes() == original
        finally:
            reopened.close()


def test_complete_tail_is_sealed(tmp_path: Path) -> None:
    store = persisted_store(tmp_path)
    path = store.session_file
    store.close()
    record = {
        "type": "message",
        "id": "new-entry",
        "timestamp": "2026-10-13T00:00:00Z",
        "message": UserMessage(content="tail").to_json(),
    }
    with path.open("ab") as file:
        file.write(json.dumps(record).encode())
    reopened = SessionStore.open(path)
    try:
        assert len(reopened.entries) == 2
        assert path.read_bytes().endswith(b"\n")
    finally:
        reopened.close()


def test_second_writer_cannot_open_locked_session(tmp_path: Path) -> None:
    store = persisted_store(tmp_path)
    try:
        with pytest.raises(SessionLockError):
            SessionStore.open(store.session_file)
    finally:
        store.close()


def test_interior_corruption_is_not_recovered(tmp_path: Path) -> None:
    store = persisted_store(tmp_path)
    path = store.session_file
    store.close()
    data = path.read_bytes() + b"broken\n"
    path.write_bytes(data)
    with pytest.raises(SessionCorruptError):
        SessionStore.open(path)
    assert path.read_bytes() == data


def test_durable_write_failure_never_advances_memory(tmp_path: Path, monkeypatch) -> None:
    store = persisted_store(tmp_path)
    before = store.entries

    def fail_fsync(fd):
        raise OSError("disk failure")

    try:
        monkeypatch.setattr("pi_agent.session_store.os.fsync", fail_fsync)
        with pytest.raises(OSError, match="disk failure"):
            store.append_message(UserMessage(content="uncommitted"))
        assert store.entries == before
        assert not store.usable
        with pytest.raises(SessionUnusableError):
            store.append_message(UserMessage(content="never"))
    finally:
        store.close()


async def test_session_write_failure_is_fatal_not_retryable(config: AgentConfig, monkeypatch) -> None:
    provider = FauxProvider([])
    config.retry.enabled = True
    session = create_agent_session(provider, config=config, session="new")

    def fail_fsync(fd):
        raise OSError("HTTP 503 in disk error should not retry")

    try:
        monkeypatch.setattr("pi_agent.session_store.os.fsync", fail_fsync)
        with pytest.raises(FatalPersistenceError):
            await session.prompt("work")
        assert provider.requests == []
        assert len(session.messages) == 1  # only the initial system prompt
        assert not session.store.usable
    finally:
        session.dispose()


def test_resume_leaves_unanswered_call_for_the_projection_without_replaying(config: AgentConfig) -> None:
    session = create_agent_session(FauxProvider([]), config=config, session="new")
    session.store.append_message(UserMessage(content="work"))
    session.store.append_message(
        AssistantMessage(
            content=[ToolCallBlock(id="c", name="write", arguments={"path": "never", "content": "x"})],
            stop_reason="toolUse",
        )
    )
    path = session.session_file
    session.dispose()
    before = path.read_bytes()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        # As in pi, nothing is written for the unanswered call: the provider projection
        # answers it with "No result provided" on the next request.
        assert isinstance(resumed.messages[-1], AssistantMessage)
        synthetic = transform_messages(resumed.messages, resumed.model)[-1]
        assert isinstance(synthetic, ToolResultMessage) and synthetic.tool_call_id == "c"
        assert synthetic.is_error and synthetic.content[0].text == "No result provided"
        assert not (config.cwd / "never").exists()
    finally:
        resumed.dispose()
    assert path.read_bytes() == before
