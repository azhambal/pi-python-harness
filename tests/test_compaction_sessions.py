from copy import deepcopy

import pytest

from pi_agent import create_agent_session
from pi_agent.compaction import CompactionAbortedError, prepare_compaction
from pi_agent.providers import FauxProvider
from pi_agent.session import build_projection
from pi_agent.session_store import SessionCompactionEntry, SessionStore
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage, Usage, UserMessage


def answer(text, usage=None):
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop", usage=usage or Usage())


def batch(identifier, name, **args):
    return AssistantMessage(
        content=[ToolCallBlock(id=identifier, name=name, arguments=args)], stop_reason="toolUse"
    )


def seed(session, *, prior_history=False):
    if prior_history:
        session.store.append_message(UserMessage(content="earlier question"))
        session.store.append_message(answer("earlier answer"))
    for message in [
        UserMessage(content="single long tool turn"),
        batch("a", "read", path="old\nname.txt"),
        ToolResultMessage(tool_call_id="a", tool_name="read", content=[TextBlock(text="x" * 4000)]),
        batch("b", "write", path="changed.py", content="y" * 1200),
        ToolResultMessage(tool_call_id="b", tool_name="write", content=[TextBlock(text="written")]),
        batch("c", "read", path="changed.py"),
        ToolResultMessage(tool_call_id="c", tool_name="read", content=[TextBlock(text="tail")]),
        answer("done"),
    ]:
        session.store.append_message(message)
    session._refresh_agent_messages()


@pytest.mark.parametrize("prior_history", [False, True])
async def test_split_turn_checkpoint_is_atomic_keeps_pairs_and_accumulates_files(config, prior_history):
    config.compaction.keep_recent_tokens = 70
    summaries = [answer("history", Usage(input=2, output=3, total_tokens=5))] if prior_history else []
    # The second checkpoint cuts at "next answer", splitting that turn too (Pi's findCutPoint).
    summaries += [
        answer("prefix", Usage(input=7, output=11, total_tokens=18)),
        answer("updated"),
        answer("next prefix"),
    ]
    provider = FauxProvider(summaries)
    session = create_agent_session(provider, config=config, session="new")
    try:
        seed(session, prior_history=prior_history)
        preparation = prepare_compaction(
            build_projection(session.store), tokens_before=2000, settings=config.compaction
        )
        assert preparation.is_split_turn and preparation.turn_prefix_messages
        assert bool(preparation.messages_to_summarize) is prior_history
        assert preparation.turn_prefix_messages[0].content == "single long tool turn"
        before = deepcopy(session.store.entries)
        result = await session.compact()
        assert result.is_split_turn and "Turn Context (split turn)" in result.summary
        assert result.usage.total_tokens == (23 if prior_history else 18)
        assert result.read_files == ["old\nname.txt"]
        assert len(provider.requests) == (2 if prior_history else 1)
        checkpoint = session.store.entries[-1]
        assert checkpoint.read_files == ["old\nname.txt"]
        assert session.store.entries[:-1] == before
        calls = [
            block.id
            for m in session.messages
            if isinstance(m, AssistantMessage)
            for block in m.content
            if isinstance(block, ToolCallBlock)
        ]
        results = [m.tool_call_id for m in session.messages if isinstance(m, ToolResultMessage)]
        assert calls == results == ["b", "c"]
        # A later ordinary exchange lets the next checkpoint consume the kept tail.
        session.store.append_message(UserMessage(content="next question"))
        session.store.append_message(answer("next answer"))
        session._refresh_agent_messages()
        session.config.compaction.keep_recent_tokens = 1
        second = await session.compact()
        assert second.read_files == ["old\nname.txt"] and second.modified_files == ["changed.py"]
        path = session.session_file
        projected = deepcopy(session.messages)
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        assert resumed.messages == projected
        assert resumed.store.entries[-1].read_files == ["old\nname.txt"]
        assert resumed.store.entries[-1].modified_files == ["changed.py"]
    finally:
        resumed.dispose()


@pytest.mark.parametrize(
    ("stop", "error"),
    [("error", RuntimeError), ("length", RuntimeError), ("aborted", CompactionAbortedError)],
)
async def test_failed_second_summary_does_not_commit_partial_history_summary(config, stop, error):
    config.compaction.keep_recent_tokens = 70
    provider = FauxProvider(
        [
            answer("completed history"),
            AssistantMessage(
                content=[TextBlock(text="partial prefix")], stop_reason=stop, error_message="summary failed"
            ),
        ]
    )
    session = create_agent_session(provider, config=config, session="new")
    try:
        seed(session, prior_history=True)
        before = deepcopy(session.store.entries)
        with pytest.raises(error):
            await session.compact()
        assert session.store.entries == before
        assert not any(isinstance(e, SessionCompactionEntry) for e in before)
    finally:
        session.dispose()


def test_legacy_checkpoint_file_tags_migrate_and_bad_arrays_reject(config):
    store = SessionStore.in_memory(config.cwd)
    try:
        first = store.append_message(UserMessage(content="first"))
        store.append_message(answer("old"))
        store.append_compaction("old summary\n<read-files>\nlegacy.py\n</read-files>", first, 50)
        for message in [
            UserMessage(content="new"),
            answer("new"),
            UserMessage(content="last"),
            answer("last"),
        ]:
            store.append_message(message)
        config.compaction.keep_recent_tokens = 1
        preparation = prepare_compaction(
            build_projection(store), tokens_before=50, settings=config.compaction
        )
        assert preparation.file_ops.read == {"legacy.py"}
        with pytest.raises(ValueError):
            store.append_compaction("invalid", first, 50, read_files=[1])
    finally:
        store.close()


def test_recorded_empty_file_lists_do_not_import_model_invented_tags(config):
    store = SessionStore.in_memory(config.cwd)
    try:
        first = store.append_message(UserMessage(content="first"))
        store.append_message(answer("old"))
        store.append_compaction(
            "summary\n<read-files>\nfabricated.py\n</read-files>", first, 50, read_files=[], modified_files=[]
        )
        for message in [
            UserMessage(content="new"),
            answer("new"),
            UserMessage(content="last"),
            answer("last"),
        ]:
            store.append_message(message)
        config.compaction.keep_recent_tokens = 1
        preparation = prepare_compaction(
            build_projection(store), tokens_before=50, settings=config.compaction
        )
        assert preparation.file_ops.read == set()
        assert store.entries[2].to_json()["read_files"] == []
    finally:
        store.close()
