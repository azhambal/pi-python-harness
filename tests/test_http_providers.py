import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from pi_agent import AgentConfig, Cancellation, RetrySettings, create_agent_session
from pi_agent.error_policy import is_retryable_error
from pi_agent.events import StreamDone, StreamError, TextDelta
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.providers import AnthropicProvider, OpenAICompatProvider
from pi_agent.providers._http import sse_data
from pi_agent.types import (
    AssistantMessage,
    ModelInfo,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
)


def frame(data: object) -> bytes:
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {text}\r\n\r\n".encode()


def completion(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "response_0",
        "model": "deepseek-flash",
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish},
        ],
    }


def openai_text(text: str = "hello", finish: str = "stop") -> bytes:
    return frame(completion({"content": text})) + frame(completion({}, finish)) + frame("[DONE]")


def anthropic_text(text: str = "hello", finish: str = "end_turn") -> bytes:
    return b"".join(
        frame(event)
        for event in [
            {
                "type": "message_start",
                "message": {"id": "msg_0", "model": "claude-test", "usage": {"input_tokens": 2}},
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": finish}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ]
    )


class Chunks(httpx.AsyncByteStream):
    def __init__(self, body: bytes, *, hang: bool = False) -> None:
        self.body = body
        self.hang = hang
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for start in range(0, len(self.body), 7):
            yield self.body[start : start + 7]
        self.started.set()
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def model(provider: str = "deepseek") -> ModelInfo:
    return ModelInfo(
        id="deepseek-flash" if provider == "deepseek" else "claude-test",
        name="test",
        provider=provider,
        api="openai-completions" if provider == "deepseek" else "anthropic-messages",
        reasoning=True,
    )


@pytest.mark.parametrize("provider_name", ["deepseek", "anthropic"])
async def test_chunked_utf8_stream_closes_and_snapshots(provider_name: str) -> None:
    body = (
        openai_text("\u041f\u0440\u0438\u0432\u0435\u0442")
        if provider_name == "deepseek"
        else anthropic_text("\u041f\u0440\u0438\u0432\u0435\u0442")
    )
    stream = Chunks(b": keepalive\r\n\r\n" + body)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cls = OpenAICompatProvider if provider_name == "deepseek" else AnthropicProvider
        provider = cls(api_key="test-only", base_url="https://unit.test", client=client)
        events = [
            event async for event in provider.stream(ProviderRequest(model(provider_name)), ProviderOptions())
        ]
        assert isinstance(events[-1], StreamDone)
        assert events[-1].message.content == [TextBlock(text="\u041f\u0440\u0438\u0432\u0435\u0442")]
        first = events[0].partial
        assert first.content == []
        assert sum(isinstance(event, TextDelta) for event in events) == 1
        assert stream.closed
        assert not client.is_closed  # injected client is caller-owned
    assert str(requests[0].url) == "https://unit.test" + cls.endpoint


async def test_deepseek_interleaved_tools_reasoning_usage_and_replay() -> None:
    body = b"".join(
        frame(data)
        for data in [
            completion({"reasoning_content": "plan"}),
            completion(
                {
                    "tool_calls": [
                        {"index": 1, "id": "call_b", "function": {"name": "read", "arguments": '{"path":'}},
                        {"index": 0, "id": "call_a", "function": {"name": "read", "arguments": '{"path":'}},
                    ]
                }
            ),
            completion(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '"a"}'}},
                        {"index": 1, "function": {"arguments": '"b"}'}},
                    ]
                }
            ),
            completion({}, "tool_calls"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 20,
                    "prompt_cache_hit_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            },
            "[DONE]",
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        )
    ) as client:
        provider = OpenAICompatProvider(api_key="test-only", base_url="https://unit.test", client=client)
        events = [
            event
            async for event in provider.stream(
                ProviderRequest(model()), ProviderOptions(thinking_level="high")
            )
        ]
        final = events[-1].message
        assert final.stop_reason == "toolUse"
        assert final.content[0] == ThinkingBlock(thinking="plan")
        assert [
            (block.id, block.arguments) for block in final.content if isinstance(block, ToolCallBlock)
        ] == [
            ("call_b", {"path": "b"}),
            ("call_a", {"path": "a"}),
        ]
        assert (final.usage.input, final.usage.cache_read, final.usage.output, final.usage.total_tokens) == (
            8,
            12,
            5,
            25,
        )
        payload = provider.payload(
            ProviderRequest(model(), messages=[final]), ProviderOptions(thinking_level="high")
        )
        assert payload["messages"][0]["reasoning_content"] == "plan"
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "high"
        assert final.to_json()["provider"] == "deepseek"


@pytest.mark.parametrize(
    "body",
    [
        frame(completion({"content": "partial"})),
        frame(completion({"content": "partial"}, "stop")),
        frame("[DONE]"),
        frame(completion({}, "content_filter")) + frame("[DONE]"),
        frame(
            completion(
                {"tool_calls": [{"index": 0, "id": "id", "function": {"name": "write", "arguments": "[]"}}]},
                "tool_calls",
            )
        )
        + frame("[DONE]"),
        frame(
            completion(
                {
                    "tool_calls": [
                        {"index": 0, "id": "id", "function": {"name": "write", "arguments": '{"x":NaN}'}}
                    ]
                },
                "tool_calls",
            )
        )
        + frame("[DONE]"),
        b'data: {"choices": []}',
        b"data: not-json\n\n",
    ],
)
async def test_invalid_or_incomplete_stream_is_error(body: bytes) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        )
    ) as client:
        provider = OpenAICompatProvider(api_key="test-only", base_url="https://unit.test", client=client)
        events = [event async for event in provider.stream(ProviderRequest(model()), ProviderOptions())]
        assert isinstance(events[-1], StreamError)
        assert events[-1].error.stop_reason == "error"
        assert not any(isinstance(event, StreamDone) for event in events)


@pytest.mark.parametrize("phase", ["connect", "stream"])
async def test_cancellation_during_connection_or_body(phase: str) -> None:
    reached = asyncio.Event()
    stream = Chunks(frame(completion({"content": "partial"})), hang=True)

    async def handler(_: httpx.Request) -> httpx.Response:
        reached.set()
        if phase == "connect":
            await asyncio.Event().wait()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatProvider(api_key="test-only", base_url="https://unit.test", client=client)
        cancellation = Cancellation()

        async def collect():
            return [
                event
                async for event in provider.stream(
                    ProviderRequest(model()), ProviderOptions(cancellation=cancellation)
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(reached.wait(), 1)
        if phase == "stream":
            await asyncio.wait_for(stream.started.wait(), 1)
        cancellation.abort()
        events = await asyncio.wait_for(task, 1)
        assert events[-1].error.stop_reason == "aborted"
        if phase == "stream":
            assert stream.closed
            assert events[-1].error.content == [TextBlock(text="partial")]


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_http_errors_redact_key_and_do_not_retry(status: int) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text="bad request test-secret")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatProvider(api_key="test-secret", base_url="https://unit.test", client=client)
        events = [event async for event in provider.stream(ProviderRequest(model()), ProviderOptions())]
        assert f"HTTP {status}" in events[-1].error.error_message
        assert "test-secret" not in events[-1].error.error_message
        assert len(requests) == 1


async def test_anthropic_signed_and_redacted_thinking_cache_usage() -> None:
    body = b"".join(
        frame(event)
        for event in [
            {
                "type": "message_start",
                "message": {
                    "id": "msg",
                    "model": "claude-test",
                    "usage": {
                        "input_tokens": 2,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 10,
                        "cache_creation": {"ephemeral_1h_input_tokens": 7},
                    },
                },
            },
            {"type": "ping"},
            {"type": "future_event"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "plan"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "signed"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "redacted_thinking", "data": "opaque"},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "tool_use", "id": "call_0", "name": "read", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"path":"a"}'},
            },
            {"type": "content_block_stop", "index": 2},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
            {"type": "message_stop"},
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        )
    ) as client:
        provider = AnthropicProvider(api_key="test-only", client=client)
        events = [
            event async for event in provider.stream(ProviderRequest(model("anthropic")), ProviderOptions())
        ]
        final = events[-1].message
        assert final.usage.total_tokens == 40
        assert final.usage.cache_write_1h == 7
        payload = provider.payload(
            ProviderRequest(model("anthropic"), messages=[UserMessage(content="start"), final]),
            ProviderOptions(thinking_level="high"),
        )
        assert payload["messages"][1]["content"][0] == {
            "type": "thinking",
            "thinking": "plan",
            "signature": "signed",
        }
        assert payload["messages"][1]["content"][1] == {"type": "redacted_thinking", "data": "opaque"}
        assert payload["messages"][2]["content"][0]["type"] == "tool_result"
        assert payload["messages"][2]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert payload["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("provider_name", ["deepseek", "anthropic"])
async def test_real_tool_loop_over_mock_http(config: AgentConfig, provider_name: str) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            if provider_name == "deepseek":
                body = frame(
                    completion(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_0",
                                    "function": {
                                        "name": "write",
                                        "arguments": '{"path":"result.txt","content":"ok"}',
                                    },
                                }
                            ]
                        },
                        "tool_calls",
                    )
                ) + frame("[DONE]")
            else:
                body = b"".join(
                    frame(event)
                    for event in [
                        {"type": "message_start", "message": {"id": "msg", "model": "claude-test"}},
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "call_0",
                                "name": "write",
                                "input": {},
                            },
                        },
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": '{"path":"result.txt","content":"ok"}',
                            },
                        },
                        {"type": "content_block_stop", "index": 0},
                        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
                        {"type": "message_stop"},
                    ]
                )
        else:
            body = openai_text("done") if provider_name == "deepseek" else anthropic_text("done")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cls = OpenAICompatProvider if provider_name == "deepseek" else AnthropicProvider
        provider = cls(api_key="test-only", base_url="https://unit.test", client=client)
        session = create_agent_session(
            provider, config=replace(config, model=model(provider_name)), session="new"
        )
        try:
            await session.prompt("write a file")
            assert (config.cwd / "result.txt").read_text() == "ok"
            assert session.messages[-1].stop_reason == "stop"
            assert len(requests) == 2
            assert requests[0]["tools"]
            assert "test-only" not in session.session_file.read_text()
        finally:
            session.dispose()


@pytest.mark.parametrize("terminator", [b"\n", b"\r", b"\r\n"])
async def test_sse_multiline_comments_bom_and_line_endings(terminator: bytes) -> None:
    body = b"\xef\xbb\xbf" + terminator.join(
        [
            b": heartbeat",
            b"event: chunk",
            b'data: {"value":',
            b'data: "ok"}',
            b"",
            b"",
        ]
    )
    stream = Chunks(body)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    ) as client:
        async with client.stream("GET", "https://unit.test") as response:
            values = [value async for value in sse_data(response)]
            assert values == ['{"value":\n"ok"}']


async def test_sse_bounds_unterminated_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pi_agent.providers._http.MAX_EVENT_CHARS", 64)
    stream = Chunks(b"data: " + b"x" * 1000, hang=True)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    ) as client:
        async with client.stream("GET", "https://unit.test") as response:
            with pytest.raises(ValueError, match="exceeds"):
                await asyncio.wait_for(anext(sse_data(response)), 1)
    assert stream.closed


async def test_early_generator_close_releases_body() -> None:
    stream = Chunks(frame(completion({"content": "partial"})), hang=True)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        )
    ) as client:
        provider = OpenAICompatProvider(api_key="test-only", base_url="https://unit.test", client=client)
        events = provider.stream(ProviderRequest(model()), ProviderOptions())
        async for event in events:
            if isinstance(event, TextDelta):
                break
        await events.aclose()
        assert stream.closed


@pytest.mark.parametrize("exception", [httpx.ReadTimeout("timeout"), httpx.ConnectError("network")])
async def test_transport_failure_is_terminal_without_hidden_retry(exception: Exception) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise exception

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatProvider(api_key="test-only", base_url="https://unit.test", client=client)
        events = [event async for event in provider.stream(ProviderRequest(model()), ProviderOptions())]
        assert isinstance(events[-1], StreamError)
        assert len(requests) == 1
        assert is_retryable_error(events[-1].error)


@pytest.mark.parametrize(
    "body,expected",
    [
        (b'{"ok":1e999}', "error"),
        (b'{"ok":', "length"),
    ],
)
async def test_truncated_tool_is_length_without_execution(
    config: AgentConfig, body: bytes, expected: str
) -> None:
    native = frame(
        completion(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_0",
                        "function": {
                            "name": "write",
                            "arguments": body.decode(),
                        },
                    }
                ]
            },
            "length" if expected == "length" else "tool_calls",
        )
    ) + frame("[DONE]")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=native if len(requests) == 1 else openai_text("done"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatProvider(api_key="test-only", base_url="https://unit.test", client=client)
        session = create_agent_session(provider, config=replace(config, model=model()))
        events = []
        session.subscribe(events.append)
        try:
            await asyncio.wait_for(session.prompt("write"), 2)
            first = next(message for message in session.messages if isinstance(message, AssistantMessage))
            assert first.stop_reason == expected
            results = [message for message in session.messages if isinstance(message, ToolResultMessage)]
            assert all(message.is_error for message in results)
            assert len(requests) == (2 if expected == "length" else 1)
            if expected == "length":
                assert "not executed" in results[0].content[0].text
            else:
                assert not any(event.type == "tool_execution_start" for event in events)
        finally:
            session.dispose()


@pytest.mark.parametrize("finish,expected", [("max_tokens", "length"), ("tool_use", "error")])
async def test_anthropic_incomplete_tool_obeys_stop_reason(finish: str, expected: str) -> None:
    body = b"".join(
        frame(event)
        for event in [
            {"type": "message_start", "message": {"id": "msg", "model": "claude-test"}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call", "name": "write", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": finish}},
            {"type": "message_stop"},
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        )
    ) as client:
        provider = AnthropicProvider(api_key="test-only", client=client)
        events = [
            event async for event in provider.stream(ProviderRequest(model("anthropic")), ProviderOptions())
        ]
        if expected == "length":
            assert isinstance(events[-1], StreamDone)
            assert events[-1].message.stop_reason == expected
        else:
            assert isinstance(events[-1], StreamError)


@pytest.mark.parametrize("thinking", ["adaptive", "budget"])
def test_anthropic_thinking_payload_modes(thinking: str) -> None:
    provider = AnthropicProvider(api_key="test-only", thinking_mode=thinking)
    request = ProviderRequest(model("anthropic"), messages=[UserMessage(content="start")])
    payload = provider.payload(request, ProviderOptions(thinking_level="high", max_tokens=4096))
    assert payload["max_tokens"] == 4096
    if thinking == "budget":
        assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4095}
    else:
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {"effort": "high"}


async def test_http_retry_keeps_completed_tool_result(config: AgentConfig) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            body = frame(
                completion(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_0",
                                "function": {
                                    "name": "write",
                                    "arguments": '{"path":"result.txt","content":"ok"}',
                                },
                            }
                        ]
                    },
                    "tool_calls",
                )
            ) + frame("[DONE]")
        elif len(requests) == 2:
            return httpx.Response(429, text="rate limit exceeded")
        else:
            body = openai_text("done")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatProvider(api_key="test-only", base_url="https://unit.test", client=client)
        session = create_agent_session(
            provider,
            config=replace(
                config,
                model=model(),
                retry=RetrySettings(max_retries=1, base_delay_ms=1, max_delay_ms=1),
            ),
            session="new",
        )
        events = []
        session.subscribe(events.append)
        try:
            await session.prompt("write a file")
            assert session.messages[-1].stop_reason == "stop"
            assert len(requests) == 3
            assert sum(event.type == "tool_execution_start" for event in events) == 1
            assert requests[2]["messages"][-1] == requests[1]["messages"][-1]
            assert any(entry.type == "context_omission" for entry in session.store.entries)
        finally:
            session.dispose()
