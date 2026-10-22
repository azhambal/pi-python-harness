import asyncio
import json
import os
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from pi_agent import (
    CompactionSettings,
    ModelInfo,
    PermissionPolicy,
    ToolHooks,
    create_agent_session,
)
from pi_agent.context import projection_system_tokens
from pi_agent.events import ModelChanged, StreamStart, ThinkingLevelChanged, event_to_json
from pi_agent.providers import AnthropicProvider, FauxProvider, OpenAICompatProvider
from pi_agent.session_store import SessionConfigEntry, SessionStore, SessionUnusableError
from pi_agent.types import AssistantMessage, TextBlock, ThinkingBlock, ToolCallBlock, ToolResultMessage, Usage


def done(text="done", usage=None):
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop", usage=usage or Usage())


async def test_model_and_thinking_commit_together_preserve_state_and_resume(config):
    provider = FauxProvider([done(), done()])
    session = create_agent_session(
        provider, config=config, session="new", permission_policy=PermissionPolicy(read_only=True)
    )
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("first")
        history = deepcopy(session.messages)
        prompt = session.system_prompt
        model = replace(config.model, id="second", reasoning=True, context_window=50000, max_tokens=2000)
        before = len(session.store.entries)
        session.set_model(model, thinking_level="high")
        model.id = "caller-mutation"
        assert len(session.store.entries) == before + 1
        entry = session.store.entries[-1]
        assert isinstance(entry, SessionConfigEntry)
        assert entry.config["model"]["id"] == "second" and entry.config["thinking_level"] == "high"
        assert session.model == session.agent.model == session.config.model
        exposed = session.model
        exposed.id = "getter-mutation"
        assert session.model.id == "second"
        assert session.thinking_level == session.agent.thinking_level == "high"
        assert session.messages == history and session.system_prompt == prompt
        assert session.config.permissions.read_only
        await session.wait_for_idle()
        changes = [event_to_json(e) for e in events if isinstance(e, (ModelChanged, ThinkingLevelChanged))]
        assert [e["type"] for e in changes] == ["model_changed", "thinking_level_changed"]
        assert changes[0]["previous_model"]["id"] == config.model.id
        assert changes[1]["previous_level"] == "off" and changes[1]["level"] == "high"
        session.set_model(session.model)
        session.set_thinking_level("high")
        assert len(session.store.entries) == before + 1
        await session.prompt("second")
        assert provider.requests[-1].model.id == "second"
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        assert resumed.model.id == "second" and resumed.thinking_level == "high"
        assert resumed.config.permissions.read_only and resumed.system_prompt == prompt
    finally:
        resumed.dispose()


async def test_switch_before_first_prompt_and_clamp_non_reasoning_model(config):
    session = create_agent_session(FauxProvider([done()]), config=config, session="new")
    try:
        reasoning = replace(config.model, id="reasoning", reasoning=True)
        session.set_model(reasoning, thinking_level="max")
        assert session.set_thinking_level("low") == "low"
        session.set_model(replace(reasoning, id="plain", reasoning=False))
        assert session.thinking_level == "off"
        before = len(session.store.entries)
        assert session.set_thinking_level("high") == "off"
        assert len(session.store.entries) == before
        await session.prompt("first")
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    assert resumed.model.id == "plain" and resumed.thinking_level == "off"
    resumed.dispose()


@pytest.mark.parametrize("change", ["provider", "api"])
def test_cross_identity_requires_explicit_transport(config, change):
    session = create_agent_session(FauxProvider([]), config=config)
    model = replace(config.model, **{change: "different"})
    try:
        before = session.store.entries
        with pytest.raises(ValueError, match="provider="):
            session.set_model(model)
        assert session.model == config.model and session.store.entries == before
        replacement = FauxProvider([])
        session.set_model(model, provider=replacement)
        assert session.agent.provider is replacement and session._provider is replacement
    finally:
        session.dispose()


@pytest.mark.parametrize("level", ["invalid", None, 1, True])
def test_invalid_thinking_is_rejected_without_writing(config, level):
    session = create_agent_session(FauxProvider([]), config=config)
    try:
        before = session.store.entries
        with pytest.raises(ValueError, match="thinking"):
            session.set_thinking_level(level)
        assert session.thinking_level == "off" and session.store.entries == before and session.is_idle
    finally:
        session.dispose()


@pytest.mark.parametrize(
    "field,value", [("id", ""), ("context_window", -1), ("max_tokens", True), ("reasoning", "yes")]
)
def test_invalid_model_rejected_without_writing(config, field, value):
    session = create_agent_session(FauxProvider([]), config=config)
    try:
        before = session.store.entries
        with pytest.raises(ValueError):
            session.set_model(replace(config.model, **{field: value}))
        assert session.store.entries == before and session.model == config.model and session.is_idle
    finally:
        session.dispose()


@pytest.mark.parametrize("thinking_only", [False, True])
async def test_durable_write_failure_keeps_runtime_state_and_poisoned_store(
    config, monkeypatch, thinking_only
):
    config.model.reasoning = True
    original = FauxProvider([done()])
    replacement = FauxProvider([])
    session = create_agent_session(original, config=config, session="new")
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("first")
        before = session.session_file.read_bytes()
        entries = session.store.entries
        previous_config = session.config
        actual_write = os.write

        def fail(fd, data):
            if fd == session.store._lock_fd:
                raise OSError("injected config write failure")
            return actual_write(fd, data)

        monkeypatch.setattr(os, "write", fail)
        with pytest.raises(OSError, match="config write failure"):
            if thinking_only:
                session.set_thinking_level("high")
            else:
                session.set_model(
                    replace(config.model, id="second"), provider=replacement, thinking_level="high"
                )
        assert session.config is previous_config
        assert session.model == session.agent.model == config.model and session.thinking_level == "off"
        assert session.agent.provider is original and session._provider is original
        assert session.store.entries == entries and session.session_file.read_bytes() == before
        assert not session.store.usable and session.is_idle
        assert not any(isinstance(e, (ModelChanged, ThinkingLevelChanged)) for e in events)
        with pytest.raises(SessionUnusableError):
            session.set_thinking_level("low")
        with pytest.raises(SessionUnusableError):
            session.set_model(session.model, provider=replacement)
    finally:
        session.dispose()


@pytest.mark.parametrize("activity", ["prompt", "compact", "agent"])
async def test_configuration_rejected_during_active_lifecycle(config, activity):
    entered = asyncio.Event()

    class Hanging(FauxProvider):
        async def stream(self, request, options):
            entered.set()
            yield StreamStart()
            await asyncio.Event().wait()

    if activity == "compact":
        config.compaction.keep_recent_tokens = 1
        seed = FauxProvider([done(), done(), done()])
        session = create_agent_session(seed, config=config)
        for index in range(3):
            await session.prompt(str(index) + "x" * 400)
        session.set_model(config.model, provider=Hanging([]))
        task = asyncio.create_task(session.compact())
    else:
        session = create_agent_session(Hanging([]), config=config)
        task = asyncio.create_task(
            session.agent.prompt("direct") if activity == "agent" else session.prompt("run")
        )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        before = session.store.entries
        for mutate in (
            lambda: session.set_model(replace(config.model, id="second")),
            lambda: session.set_thinking_level("high"),
        ):
            with pytest.raises(RuntimeError, match="busy"):
                mutate()
        assert session.store.entries == before and not session.is_idle
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()


async def test_callback_reentry_and_disposed_setters_reject(config):
    rejected = []

    def before(ctx, token):
        for mutate in (lambda: session.set_model(config.model), lambda: session.set_thinking_level("off")):
            with pytest.raises(RuntimeError, match="re-entrantly"):
                mutate()
            rejected.append(True)

    session = create_agent_session(
        FauxProvider(
            [
                AssistantMessage(
                    content=[ToolCallBlock(id="c", name="read", arguments={"path": "missing"})],
                    stop_reason="toolUse",
                ),
                done(),
            ]
        ),
        config=config,
        hooks=ToolHooks(before_tool_call=before),
    )
    try:
        await session.prompt("read")
        assert len(rejected) == 2
    finally:
        session.dispose()
    with pytest.raises(RuntimeError, match="disposed"):
        session.set_model(config.model)
    with pytest.raises(RuntimeError, match="disposed"):
        session.set_thinking_level("off")


def test_provider_validation_is_local_and_failure_does_not_commit(config):
    class Reject(FauxProvider):
        def validate_model(self, model, thinking_level):
            model.id = "validator-mutation"
            raise ValueError("unsupported target")

    session = create_agent_session(FauxProvider([]), config=config)
    try:
        before = session.store.entries
        with pytest.raises(ValueError, match="unsupported"):
            session.set_model(replace(config.model, id="second"), provider=Reject([]))
        assert session.store.entries == before and session.model == config.model
    finally:
        session.dispose()


async def test_usage_invalidated_across_switch_and_return_including_resume(config):
    config.model.reasoning = True
    provider = FauxProvider([done(usage=Usage(input=50000, total_tokens=50000)), done(usage=Usage(input=77))])
    session = create_agent_session(provider, config=config, session="new")
    try:
        await session.prompt("first")
        assert session._projection_tokens() >= 50000
        session.set_thinking_level("high")
        assert session._projection_tokens() >= 50000  # Same tokenizer/identity.
        session.set_model(replace(config.model, id="second"))
        assert session._projection_tokens() == projection_system_tokens(session.messages)
        session.set_model(config.model)
        assert session._projection_tokens() == projection_system_tokens(session.messages)
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(provider, session=path)
    try:
        assert resumed._projection_tokens() == projection_system_tokens(resumed.messages)
        await resumed.prompt("second")
        assert resumed._projection_tokens() >= 77
    finally:
        resumed.dispose()


async def test_new_provider_model_and_limits_used_for_threshold_compaction(config):
    seed = FauxProvider([done(), done(), done()])
    session = create_agent_session(seed, config=config)
    try:
        for index in range(3):
            await session.prompt(str(index) + "x" * 400)
        replacement = FauxProvider([done("## Goal\nSummary"), done("new model answer")])
        # Small new window triggers summary before the next conversation request.
        session.config.compaction = CompactionSettings(enabled=True, reserve_tokens=0, keep_recent_tokens=1)
        new_window = projection_system_tokens(session.messages) - 150
        session.set_model(
            replace(config.model, id="small", context_window=new_window, max_tokens=2048),
            provider=replacement,
        )
        await session.prompt("continue")
        assert seed.requests[-1].model.id == config.model.id
        assert len(replacement.requests) == 2
        assert replacement.requests[0].tools == [] and replacement.requests[1].tools
        assert all(
            request.model.id == "small" and request.model.max_tokens == 2048
            for request in replacement.requests
        )
    finally:
        session.dispose()


def frame(data):
    return f"data: {json.dumps(data)}\n\n".encode()


def anthropic_body(*blocks, stop="end_turn"):
    events = [
        {
            "type": "message_start",
            "message": {"id": "response", "model": "claude-test", "usage": {"input_tokens": 3}},
        }
    ]
    for index, block in enumerate(blocks):
        events += [
            {"type": "content_block_start", "index": index, "content_block": block},
            {"type": "content_block_stop", "index": index},
        ]
    events += [
        {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    return b"".join(frame(event) for event in events)


async def test_cross_provider_http_transition_keeps_raw_signatures_pairing_and_returns_to_native(config):
    (config.cwd / "source.txt").write_text("source")
    native = ModelInfo("claude-test", "Claude", "anthropic", "anthropic-messages", True, 100000, 2048)
    config.model = native
    config.thinking_level = "high"
    wire = []
    first = True

    def handler(request):
        nonlocal first
        payload = json.loads(request.content)
        wire.append((str(request.url), payload))
        if request.url.path == "/v1/messages":
            if first:
                first = False
                body = anthropic_body(
                    {"type": "thinking", "thinking": "native plan", "signature": "signed-original"},
                    {
                        "type": "tool_use",
                        "id": "native_call",
                        "name": "read",
                        "input": {"path": "source.txt"},
                    },
                    stop="tool_use",
                )
            else:
                body = anthropic_body({"type": "text", "text": "native done"})
        else:
            body = (
                frame(
                    {
                        "choices": [
                            {"index": 0, "delta": {"content": "portable done"}, "finish_reason": "stop"}
                        ]
                    }
                )
                + b"data: [DONE]\n\n"
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        anthropic = AnthropicProvider(api_key="synthetic-only", base_url="https://unit.test", client=client)
        deepseek = OpenAICompatProvider(api_key="synthetic-only", base_url="https://unit.test", client=client)
        session = create_agent_session(anthropic, config=config, session="new")
        try:
            await session.prompt("read")
            before = deepcopy(session.store.entries)
            target = ModelInfo(
                "deepseek-flash", "DeepSeek", "deepseek", "openai-completions", True, 20000, 4096
            )
            session.set_model(target, provider=deepseek, thinking_level="low")
            await session.prompt("continue")
            payload = wire[-1][1]
            assert payload["thinking"] == {"type": "enabled"} and payload["reasoning_effort"] == "low"
            assert payload["max_tokens"] == 4096
            old_assistant = next(m for m in payload["messages"] if m.get("tool_calls"))
            assert old_assistant["reasoning_content"] == "" and old_assistant["content"] is None
            tool = next(m for m in payload["messages"] if m["role"] == "tool")
            assert tool["tool_call_id"] == old_assistant["tool_calls"][0]["id"]
            assert session.store.entries[: len(before)] == before
            session.set_model(native, provider=anthropic)
            await session.prompt("return")
            payload = wire[-1][1]
            blocks = [b for m in payload["messages"] for b in m["content"]]
            assert any(b.get("signature") == "signed-original" for b in blocks)
            calls = [b["id"] for b in blocks if b["type"] == "tool_use"]
            results = [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]
            assert calls == results == ["native_call"]
            assert any(
                isinstance(b, ThinkingBlock) and b.thinking_signature == "signed-original"
                for m in session.messages
                if isinstance(m, AssistantMessage)
                for b in m.content
            )
            assert len([m for m in session.messages if isinstance(m, ToolResultMessage)]) == 1
            path = session.session_file
        finally:
            session.dispose()
        resumed = create_agent_session(anthropic, session=path)
        try:
            assert resumed.model == native and resumed.thinking_level == "low"
        finally:
            resumed.dispose()


@pytest.mark.parametrize("provider_name", ["anthropic", "openai"])
def test_builtin_transport_mismatch_rejected_before_switch(config, provider_name):
    session = create_agent_session(FauxProvider([]), config=config)
    provider = (AnthropicProvider if provider_name == "anthropic" else OpenAICompatProvider)(
        api_key="synthetic-only", base_url="https://unit.test"
    )
    model = ModelInfo(
        "target",
        "target",
        "deepseek" if provider_name == "anthropic" else "anthropic",
        "openai-completions" if provider_name == "anthropic" else "anthropic-messages",
    )
    try:
        before = session.store.entries
        with pytest.raises(ValueError, match="requires"):
            session.set_model(model, provider=provider)
        assert session.store.entries == before and session.model == config.model
    finally:
        session.dispose()


def test_anthropic_budget_validation_precedes_persistence(config):
    provider = AnthropicProvider(
        api_key="synthetic-only", base_url="https://unit.test", thinking_mode="budget"
    )
    model = ModelInfo("target", "target", "anthropic", "anthropic-messages", True, 100000, 1024)
    session = create_agent_session(FauxProvider([]), config=config)
    try:
        before = session.store.entries
        with pytest.raises(ValueError, match="max_tokens"):
            session.set_model(model, provider=provider, thinking_level="high")
        assert session.store.entries == before
        session.set_model(model, provider=provider, thinking_level="off")
        with pytest.raises(ValueError, match="max_tokens"):
            session.set_thinking_level("high")
        assert session.thinking_level == "off"
    finally:
        session.dispose()


@pytest.mark.parametrize("level", [None, "invalid", 1])
async def test_malformed_stored_thinking_rejects_resume_and_releases_lock(config, level):
    session = create_agent_session(FauxProvider([done()]), config=config, session="new")
    try:
        await session.prompt("first")
        path = session.session_file
    finally:
        session.dispose()
    store = SessionStore.open(path)
    try:
        snapshot = dict(store.header.config)
        snapshot["thinking_level"] = level
        store.append_config_change(snapshot)
    finally:
        store.close()
    with pytest.raises(ValueError, match="thinking"):
        create_agent_session(FauxProvider([]), session=path)
    SessionStore.open(path).close()


async def test_optional_validator_cannot_be_an_async_callback(config):
    class AsyncValidator(FauxProvider):
        async def validate_model(self, model, thinking_level):
            raise AssertionError("Must not execute an async validator")

    session = create_agent_session(FauxProvider([]), config=config)
    try:
        before = session.store.entries
        with pytest.raises(ValueError, match="synchronous"):
            session.set_model(replace(config.model, id="second"), provider=AsyncValidator([]))
        assert session.is_idle and session.store.entries == before
    finally:
        session.dispose()


async def test_switching_model_within_anthropic_drops_thinking_only_from_new_request(config):
    native = ModelInfo("first", "first", "anthropic", "anthropic-messages", True, 100000, 2048)
    config.model = native
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        body = (
            anthropic_body(
                {"type": "thinking", "thinking": "original plan", "signature": "signed"},
                {"type": "text", "text": "first answer"},
            )
            if len(payloads) == 1
            else anthropic_body({"type": "text", "text": "second answer"})
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(api_key="synthetic-only", base_url="https://unit.test", client=client)
        session = create_agent_session(provider, config=config)
        try:
            await session.prompt("first")
            before = deepcopy(session.messages)
            session.set_model(replace(native, id="second"), thinking_level="high")
            await session.prompt("second")
            blocks = [block for message in payloads[-1]["messages"] for block in message["content"]]
            assert not any(block.get("signature") for block in blocks)
            assert not any(block.get("text") == "original plan" for block in blocks)
            assert any(block.get("text") == "first answer" for block in blocks)
            assert session.messages[: len(before)] == before
            assert payloads[-1]["model"] == "second" and payloads[-1]["output_config"]["effort"] == "high"
        finally:
            session.dispose()
