import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from pi_agent import RetrySettings, create_agent_session
from pi_agent.cancellation import Cancellation
from pi_agent.compaction import (
    CompactionAbortedError,
    CompactionPreparation,
    generate_summary,
    run_compaction,
    serialize_conversation,
)
from pi_agent.context import estimate_tokens, output_token_limit
from pi_agent.error_policy import is_retryable_error
from pi_agent.events import StreamError, StreamStart
from pi_agent.mcp.config import McpConfigError, load_mcp_config
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.providers import AnthropicProvider, FauxProvider, OpenAICompatProvider
from pi_agent.providers._normalize import normalize_messages
from pi_agent.settings import load_settings
from pi_agent.types import (
    AssistantMessage,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
    UserMessage,
)


def answer(text="done"):
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop")


class RecordingProvider(FauxProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.options = []
        self.closed_attempt = asyncio.Event()

    async def stream(self, request, options):
        self.options.append(options)
        try:
            async for event in super().stream(request, options):
                yield event
        finally:
            self.closed_attempt.set()


@pytest.mark.parametrize("provider", ["openai", "deepseek", "sglang", "anthropic"])
def test_switching_models_drops_thinking_without_changing_history(provider):
    model = ModelInfo("same", "Same", provider, api="test")
    original = AssistantMessage(
        provider=provider,
        api="test",
        model="same",
        stop_reason="stop",
        content=[ThinkingBlock(thinking="private", thinking_signature="signed"), TextBlock(text="visible")],
    )
    before = deepcopy(original)
    assert normalize_messages([original], model)[0].content == original.content
    changed = replace(model, id="different")
    assert normalize_messages([original], changed)[0].content == [TextBlock(text="visible")]
    original.content = [ThinkingBlock(thinking="private")]
    assert normalize_messages([original, UserMessage(content="next")], changed) == [
        UserMessage(content="next")
    ]
    original.content = before.content
    assert original == before


@pytest.mark.parametrize("text", ["", " ", "\r\n", "\t\n", "  significant indentation\n"])
def test_empty_tool_output_is_normalized_only_in_provider_payload(text):
    model = ModelInfo("model", "Model", "openai", api="openai-completions")
    messages = [
        AssistantMessage(stop_reason="toolUse", content=[ToolCallBlock(id="call", name="bash")]),
        ToolResultMessage(tool_call_id="call", tool_name="bash", content=[TextBlock(text=text)]),
    ]
    before = deepcopy(messages)
    provider = OpenAICompatProvider(api_key="synthetic", base_url="https://example.invalid/v1")
    payload = provider.payload(ProviderRequest(model=model, messages=messages), ProviderOptions())
    assert payload["messages"][-1]["content"] == (text if text.strip() else "(no tool output)")
    assert messages == before


@pytest.mark.parametrize(
    "error",
    [
        "servers are currently busy",
        "The service is temporarily unavailable",
        "The upstream provider returned an error",
        "Unable to forward upstream stream",
        "upstream_stream_read_error",
        "stream disconnected before completion",
        "upstream request failed",
        "HTTP/2 stream failed",
        "stream was interrupted",
    ],
)
def test_new_gateway_failures_are_retryable(error):
    assert is_retryable_error(AssistantMessage(stop_reason="error", error_message=error))


@pytest.mark.parametrize(
    "error",
    [
        "billing: servers are currently busy",
        "quota exceeded: upstream request failed",
        "invalid id abc503def",
        "invalid byte count 15020",
        "invalid tool arguments",
    ],
)
def test_new_retry_patterns_preserve_terminal_failures(error):
    assert not is_retryable_error(AssistantMessage(stop_reason="error", error_message=error))


def test_summary_serialization_excludes_thinking_and_keeps_actions():
    messages = [
        AssistantMessage(
            stop_reason="toolUse",
            content=[
                ThinkingBlock(thinking="private"),
                TextBlock(text="visible"),
                ToolCallBlock(id="call", name="read", arguments={"path": "work.py"}),
            ],
        ),
        ToolResultMessage(tool_call_id="call", tool_name="read", content=[TextBlock(text="observed")]),
    ]
    before = deepcopy(messages)
    rendered = serialize_conversation(messages)
    assert "private" not in rendered and "thinking" not in rendered
    assert "visible" in rendered and "work.py" in rendered and "observed" in rendered
    assert messages == before


@pytest.mark.parametrize("maximum,expected", [(0, 8192), (1024, 1024), (20000, 20000)])
async def test_summary_uses_model_output_limit(maximum, expected, model):
    provider = RecordingProvider([answer("summary")])
    await generate_summary(
        [UserMessage(content="task")],
        provider=provider,
        model=replace(model, max_tokens=maximum),
        cancellation=Cancellation(),
    )
    assert provider.options[0].max_tokens == expected
    assert provider.options[0].thinking_level == "off"
    assert provider.requests[0].tools == []


async def test_summary_retry_is_bounded_and_does_not_commit_failed_attempts(config):
    config.compaction.keep_recent_tokens = 1
    config.retry = RetrySettings(max_retries=1, base_delay_ms=1, max_delay_ms=1)
    provider = RecordingProvider(
        [
            answer(),
            answer(),
            AssistantMessage(stop_reason="error", error_message="Unable to forward upstream stream"),
            answer("checkpoint"),
        ]
    )
    session = create_agent_session(provider, config=config, session="new")
    try:
        await session.prompt("first " + "x" * 100)
        await session.prompt("second " + "y" * 100)
        before = deepcopy(session.store.entries)
        result = await session.compact()
        assert result.summary == "checkpoint"
        assert session.store.entries[:-1] == before
        assert len(provider.requests) == 4
        assert [message.content for message in provider.requests[-1].messages] == [
            message.content for message in provider.requests[-2].messages
        ]
    finally:
        session.dispose()


async def test_abort_during_summary_retry_closes_stream_and_stops_requests(model):
    provider = RecordingProvider([AssistantMessage(stop_reason="error", error_message="HTTP 503")])
    cancellation = Cancellation()
    task = asyncio.create_task(
        generate_summary(
            [UserMessage(content="task")],
            provider=provider,
            model=model,
            cancellation=cancellation,
            retry=RetrySettings(base_delay_ms=30000, max_delay_ms=30000),
        )
    )
    try:
        await asyncio.wait_for(provider.closed_attempt.wait(), 1)
        cancellation.abort()
        with pytest.raises(CompactionAbortedError):
            await asyncio.wait_for(task, 1)
        assert len(provider.requests) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("error", ["HTTP 503", "billing failure", "input exceeds the context window"])
async def test_failed_summary_obeys_retry_budget(error, model):
    failure = AssistantMessage(stop_reason="error", error_message=error)
    provider = RecordingProvider([failure, failure])
    with pytest.raises(RuntimeError):
        await generate_summary(
            [UserMessage(content="task")],
            provider=provider,
            model=model,
            cancellation=Cancellation(),
            retry=RetrySettings(max_retries=1, base_delay_ms=1, max_delay_ms=1),
        )
    assert len(provider.requests) == (2 if error == "HTTP 503" else 1)


def test_bom_configs_keep_duplicate_key_and_number_checks(tmp_path):
    settings = tmp_path / "settings.json"
    mcp = tmp_path / "mcp.json"
    settings.write_text('{"retry":{"max_retries":1}}', encoding="utf-8-sig")
    mcp.write_text('{"mcpServers":{}}', encoding="utf-8-sig")
    assert load_settings(tmp_path, agent_dir=tmp_path / "absent", explicit=settings)["retry"] == {
        "max_retries": 1
    }
    assert load_mcp_config(mcp).servers == ()
    for text in ('{"mcpServers":{},"mcpServers":{}}', '{"mcpServers":NaN}'):
        mcp.write_text(text, encoding="utf-8-sig")
        with pytest.raises(McpConfigError):
            load_mcp_config(mcp)
    settings.write_text('{"model":"a","model":"b"}', encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Duplicate"):
        load_settings(tmp_path, agent_dir=tmp_path / "absent", explicit=settings)


def test_context_estimate_bounds_output_and_accounts_for_tools(model):
    messages = [UserMessage(content="x" * 350)]
    assert estimate_tokens(messages[0]) == 100
    small = replace(model, context_window=1000, max_tokens=2000)
    assert output_token_limit(messages, small) == 644
    assert output_token_limit(messages, small, 100) == 100
    tools = [ToolDeclaration("read", "read files", {"type": "object"})]
    assert output_token_limit(messages, small, tools=tools) < 644
    assert output_token_limit([SystemMessage(tools_added=tools), *messages], small, tools=tools) == (
        output_token_limit(messages, small, tools=tools)
    )
    with pytest.raises(ValueError, match="context window"):
        output_token_limit([UserMessage(content="x" * 3500)], small)


async def test_reasoning_in_same_chunk_precedes_answer():
    provider = OpenAICompatProvider(api_key="synthetic", base_url="https://example.invalid/v1")
    data = {
        "choices": [
            {"index": 0, "delta": {"content": "answer", "reasoning_content": "plan"}, "finish_reason": "stop"}
        ]
    }
    response = httpx.Response(200, text=f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n")
    message = AssistantMessage()
    async for _ in provider.decode(response, message):
        pass
    assert message.content == [ThinkingBlock(thinking="plan"), TextBlock(text="answer")]


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_provider_output_limits_use_available_context(kind):
    model = ModelInfo(
        "test",
        "Test",
        kind,
        api=f"{kind}-completions" if kind == "openai" else "anthropic-messages",
        max_tokens=8192,
        context_window=1500,
    )
    provider = (
        OpenAICompatProvider(api_key="synthetic", base_url="https://example.invalid/v1")
        if kind == "openai"
        else AnthropicProvider(api_key="synthetic")
    )
    request = ProviderRequest(model=model, messages=[UserMessage(content="x" * 350)])
    assert provider.payload(request, ProviderOptions())["max_tokens"] == 1144


async def test_summary_error_event_cannot_publish_a_successful_message(model):
    class InconsistentProvider(FauxProvider):
        async def stream(self, request, options):
            yield StreamError(reason="error", error=answer("must not become a checkpoint"))

    with pytest.raises(RuntimeError, match="inconsistent"):
        await generate_summary(
            [UserMessage(content="task")],
            provider=InconsistentProvider([]),
            model=model,
            cancellation=Cancellation(),
        )


async def test_summary_retries_unterminated_stream(model):
    class IncompleteProvider(RecordingProvider):
        def __init__(self):
            super().__init__([answer("summary")])
            self.first = True

        async def stream(self, request, options):
            if self.first:
                self.first = False
                yield StreamStart()
                return
            async for event in super().stream(request, options):
                yield event

    summary, _ = await generate_summary(
        [UserMessage(content="task")],
        provider=IncompleteProvider(),
        model=model,
        cancellation=Cancellation(),
        retry=RetrySettings(max_retries=1, base_delay_ms=1, max_delay_ms=1),
    )
    assert summary == "summary"


@pytest.mark.parametrize("reserve", [0, 100, 16384])
async def test_session_summary_output_is_independent_of_reserve(config, reserve):
    config.compaction.reserve_tokens = reserve
    config.compaction.keep_recent_tokens = 1
    config.model.max_tokens = 8192
    provider = RecordingProvider([answer(), answer(), answer("summary")])
    session = create_agent_session(provider, config=config)
    try:
        await session.prompt("first " + "x" * 100)
        await session.prompt("second " + "y" * 100)
        await session.compact()
        assert provider.options[-1].max_tokens == 8192
    finally:
        session.dispose()


async def test_split_turn_summary_retains_focus_and_excludes_thinking(model):
    provider = RecordingProvider([answer("history"), answer("prefix")])
    preparation = CompactionPreparation(
        first_kept_entry_id="retained",
        tokens_before=100,
        messages_to_summarize=[UserMessage(content="first")],
        previous_summary=None,
        turn_prefix_messages=[
            AssistantMessage(
                stop_reason="stop",
                content=[
                    ThinkingBlock(thinking="private"),
                    TextBlock(text="visible work"),
                ],
            )
        ],
        is_split_turn=True,
    )
    await run_compaction(
        preparation,
        provider=provider,
        model=model,
        cancellation=Cancellation(),
        custom_instructions="Preserve ticket ABC-123",
    )
    assert len(provider.requests) == 2
    for request in provider.requests:
        text = request.messages[-1].content[0].text
        assert "ABC-123" in text and "private" not in text


async def test_truncated_tool_recovery_and_resume_keep_wire_pairing(config):
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "cut",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": '{"path":"must-not-exist","content":"partial',
                        },
                    }
                ]
            }
            finish = "length"
        else:
            delta, finish = {"content": "recovered"}, "stop"
        frame = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n",
        )

    config.model = ModelInfo("fixture", "Fixture", "openai", api="openai-completions", max_tokens=8192)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OpenAICompatProvider(
            api_key="synthetic", base_url="https://example.invalid/v1", client=client
        )
        session = create_agent_session(provider, config=config, session="new")
        try:
            await session.prompt("write")
            assert not (config.cwd / "must-not-exist").exists()
            path = session.session_file
        finally:
            session.dispose()
        resumed = create_agent_session(provider, session=path)
        try:
            await resumed.prompt("continue")
        finally:
            resumed.dispose()
    assert len(payloads) == 3
    for payload in payloads[1:]:
        calls = [call["id"] for message in payload["messages"] for call in message.get("tool_calls", [])]
        results = [message["tool_call_id"] for message in payload["messages"] if message["role"] == "tool"]
        assert calls == results == ["cut"]
    assert not (config.cwd / "must-not-exist").exists()
