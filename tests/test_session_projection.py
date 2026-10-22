import json
from copy import deepcopy
from pathlib import Path

import pytest

from pi_agent import AgentConfig, PermissionPolicy, create_agent_session
from pi_agent._session_projection import _projection_entries
from pi_agent.providers.faux import FauxProvider
from pi_agent.session import build_projection
from pi_agent.session_store import SessionHeader, SessionStore, SessionUnusableError
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage, UserMessage


def user(text: str) -> UserMessage:
    return UserMessage(content=[TextBlock(text=text)])


def call(cid: str) -> AssistantMessage:
    return AssistantMessage(content=[ToolCallBlock(id=cid, name="read", arguments={"path": "x"})])


def visible(store: SessionStore) -> list:
    return [m for entry in build_projection(store) for m in entry.messages]


def test_snapshots_and_append_inputs_do_not_mutate_canonical_journal(tmp_path: Path) -> None:
    model = {"provider": "faux", "id": "a"}
    config = {"model": model}
    store = SessionStore.in_memory(tmp_path, model=model, config=config)
    message = call("original")
    eid = store.append_message(message)
    cached = _projection_entries(store)
    model["id"] = "changed"
    message.content[0].id = "changed"
    store.header.config["model"]["id"] = "changed"
    store.entries[0].message.content[0].id = "changed"
    store.get_entry(eid).message.content[0].id = "changed"
    build_projection(store)[0].messages[0].content[0].id = "changed"
    ids = store.call_ids
    ids.clear()
    unanswered = store.unanswered_calls
    unanswered[0].id = "changed"
    assert _projection_entries(store) is cached
    assert store.call_ids == {"original"}
    assert store.header.model["id"] == "a"
    assert store.unanswered_calls[0].id == "original"
    assert visible(store)[0].content[0].id == "original"


def test_cache_invalidation_and_raw_ids_survive_omission_and_compaction(tmp_path: Path) -> None:
    store = SessionStore.in_memory(tmp_path)
    first = store.append_message(user("old"))
    hidden = store.append_message(call("hidden"))
    kept = store.append_message(user("keep"))
    cached = _projection_entries(store)
    revision = store.revision
    store.append_context_omission(hidden)
    assert store.revision == revision + 1 and _projection_entries(store) is not cached
    store.append_compaction("summary", kept, 100)
    assert store.call_ids == {"hidden"}
    assert all(entry.entry_id != first for entry in build_projection(store))
    cached = _projection_entries(store)
    store.append_config_change({"model": {"provider": "faux", "api": "a", "id": "b"}})
    assert _projection_entries(store) is not cached
    with pytest.raises(ValueError, match="already used"):
        store.append_message(call("hidden"))
    assert store.get_entry(hidden) is not None


def test_omitted_compaction_anchor_does_not_hide_the_remaining_tail(tmp_path: Path) -> None:
    store = SessionStore.in_memory(tmp_path)
    store.append_message(user("summarized"))
    anchor = store.append_message(user("omit anchor"))
    tail = store.append_message(user("tail survives"))
    store.append_compaction("summary", anchor, 100)
    store.append_context_omission(anchor)
    assert [e.entry_id for e in build_projection(store)][-1] == tail
    assert visible(store)[-1] == user("tail survives")


def test_failed_append_does_not_advance_indexes_or_cache(tmp_path: Path, monkeypatch) -> None:
    store = SessionStore.create(tmp_path, tmp_path / "sessions")
    try:
        store.append_message(user("start"))
        cached = _projection_entries(store)
        before = (store.revision, store.entry_count, store.call_ids, store.unanswered_calls)

        def fail(data: bytes) -> None:
            raise OSError("fsync failed")

        monkeypatch.setattr(store, "_append_raw", fail)
        with pytest.raises(OSError, match="fsync failed"):
            store.append_message(call("uncommitted"))
        assert (store.revision, store.entry_count, store.call_ids, store.unanswered_calls) == before
        assert _projection_entries(store) is cached
        with pytest.raises(SessionUnusableError):
            store.append_message(user("never"))
    finally:
        store.close()


def test_resume_and_tail_recovery_restore_all_indexes(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, tmp_path / "sessions")
    store.append_message(user("start"))
    eid = store.append_message(call("existing"))
    path = store.session_file
    store.close()
    record = {"type": "message", "id": "fragment", "timestamp": "now", "message": call("recovered").to_json()}
    with path.open("ab") as file:
        file.write(json.dumps(record).encode())
    resumed = SessionStore.open(path)
    try:
        assert resumed.call_ids == {"existing", "recovered"}
        assert [b.id for b in resumed.unanswered_calls] == ["existing", "recovered"]
        assert resumed.get_entry(eid).message == call("existing")
        assert resumed.get_entry("fragment").message == call("recovered")
        assert resumed.revision == resumed.entry_count == 3
        with pytest.raises(ValueError, match="already used"):
            resumed.append_message(call("recovered"))
        resumed.append_message(ToolResultMessage(tool_call_id="existing", tool_name="read", content=[]))
        assert [b.id for b in resumed.unanswered_calls] == ["recovered"]
    finally:
        resumed.close()


def test_recovered_header_establishes_initial_model_identity(tmp_path: Path) -> None:
    model = {"provider": "faux", "api": "offline", "id": "a"}
    header = SessionHeader(cwd=str(tmp_path), model=model, config={"model": model})
    path = tmp_path / "header.jsonl"
    path.write_text(json.dumps(header.to_json()))
    store = SessionStore.open(path)
    try:
        store.append_config_change({"model": model})
        assert store.last_model_change_source == -1
        store.append_config_change({"model": {**model, "id": "b"}})
        assert store.last_model_change_source == 1
    finally:
        store.close()


async def test_constructor_copies_model_and_nested_config(config: AgentConfig) -> None:
    provider = FauxProvider([AssistantMessage()])
    session = create_agent_session(provider, config=config)
    try:
        config.model.id = "caller mutation"
        config.retry.enabled = True
        config.permissions = PermissionPolicy(read_only=True)
        await session.prompt("original model")
        assert provider.requests[0].model.id == "offline"
        assert not session.config.retry.enabled and not session.config.permissions.read_only
    finally:
        session.dispose()


async def test_listener_mutation_cannot_change_persisted_history_or_next_request(config: AgentConfig) -> None:
    provider = FauxProvider([AssistantMessage(content=[TextBlock(text="original")]) for _ in range(2)])
    session = create_agent_session(provider, config=config, session="new")
    from pi_agent.events import MessageEnd

    def corrupt(event) -> None:
        if isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            event.message.content[0].text = "observer corruption"

    session.subscribe(corrupt)
    try:
        await session.prompt("one")
        copied = session.messages
        copied[-1].content[0].text = "getter corruption"
        before = deepcopy(session.store.entries)
        await session.prompt("two")
        assert provider.requests[1].messages[-2].content[0].text == "original"
        assert session.store.entries[: len(before)] == before
        assert session.messages[-1].content[0].text == "original"
    finally:
        session.dispose()
