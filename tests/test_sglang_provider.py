import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from pi_agent import AgentConfig, CompactionSettings, cli, create_agent_session
from pi_agent.events import StreamDone
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.providers import OpenAICompatProvider, OpenAIRequestSettings
from pi_agent.runtime_config import _ENV_NAMES, resolve_runtime
from pi_agent.settings import load_settings
from pi_agent.types import AssistantMessage, ModelInfo, TextBlock, ThinkingBlock, ThinkingLevel, UserMessage


def response_body(usage: dict) -> str:
    return (
        'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        + f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
        + "data: [DONE]\n\n"
    )


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({"reasoning_tokens": 104}, 104),
        ({"reasoning_tokens": 0}, 0),
        ({"completion_tokens_details": {"reasoning_tokens": 50}}, 50),
        ({"reasoning_tokens": 104, "completion_tokens_details": {"reasoning_tokens": 0}}, 0),
        ({"reasoning_tokens": 104, "completion_tokens_details": {}}, 104),
        ({"reasoning_tokens": 104, "completion_tokens_details": None}, 104),
        ({}, 0),
    ],
)
async def test_reasoning_usage_formats_and_standard_precedence(extra: dict, expected: int | None) -> None:
    # Representative SGLang usage fields, including flat reasoning counts.
    usage = {
        "prompt_tokens": 709,
        "completion_tokens": 327,
        "total_tokens": 1036,
        "prompt_tokens_details": None,
        **extra,
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=response_body(usage)
            )
        )
    ) as client:
        provider = OpenAICompatProvider(api_key="synthetic", base_url="https://unit.test", client=client)
        model = ModelInfo("test", "test", "sglang", api="openai-completions", reasoning=True)
        events = [event async for event in provider.stream(ProviderRequest(model), ProviderOptions())]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.usage.reasoning == expected
    assert events[-1].message.usage.output == 327
    assert events[-1].message.usage.total_tokens == 1036


@pytest.mark.parametrize("value", [True, 1.5, "104", None])
@pytest.mark.parametrize("nested", [False, True])
async def test_unusable_reasoning_count_reads_as_zero(value: object, nested: bool) -> None:
    # Pi reads usage leniently (``|| 0``); a count that is not an integer is not an error.
    usage = (
        {"completion_tokens_details": {"reasoning_tokens": value}} if nested else {"reasoning_tokens": value}
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=response_body(usage)
            )
        )
    ) as client:
        provider = OpenAICompatProvider(api_key="synthetic", base_url="https://unit.test", client=client)
        events = [
            event
            async for event in provider.stream(
                ProviderRequest(ModelInfo("test", "test", "sglang")), ProviderOptions()
            )
        ]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.usage.reasoning == 0


@pytest.mark.parametrize("level", ["off", "low", "high"])
def test_sglang_thinking_overrides_template_default(level: ThinkingLevel) -> None:
    kwargs = {"enable_thinking": level == "off", "preserve_thinking": True, "nested": {"value": 1}}
    provider = OpenAICompatProvider(
        api_key="synthetic",
        base_url="http://127.0.0.1:30000/v1",
        request_settings=OpenAIRequestSettings(temperature=0, seed=12345, chat_template_kwargs=kwargs),
    )
    kwargs["preserve_thinking"] = False
    request = ProviderRequest(
        ModelInfo("local-model", "Local model", "sglang", api="openai-completions", reasoning=True),
        messages=[UserMessage(content="hello")],
    )
    payload = provider.payload(request, ProviderOptions(thinking_level=level, max_tokens=4096))
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": level != "off",
        "preserve_thinking": True,
        "nested": {"value": 1},
    }
    assert payload["temperature"] == 0 and payload["seed"] == 12345 and payload["max_tokens"] == 4096
    # Pi's chat-template thinking format switches thinking only through the template.
    assert "reasoning_effort" not in payload
    payload["chat_template_kwargs"]["nested"]["value"] = 2
    assert provider.payload(request, ProviderOptions())["chat_template_kwargs"]["nested"]["value"] == 1


@pytest.mark.parametrize("provider_name", ["openai", "deepseek"])
def test_sglang_options_are_not_injected_for_other_providers(provider_name: str) -> None:
    provider = OpenAICompatProvider(api_key="synthetic", base_url="https://unit.test")
    model = ModelInfo("test", "test", provider_name, reasoning=True)
    payload = provider.payload(ProviderRequest(model), ProviderOptions())
    assert "chat_template_kwargs" not in payload
    assert "temperature" not in payload and "seed" not in payload
    if provider_name == "deepseek":
        assert payload["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": True},
        {"seed": True},
        {"seed": 1.5},
        {"chat_template_kwargs": []},
        {"chat_template_kwargs": {"nested": {"value": float("nan")}}},
    ],
)
def test_invalid_request_settings_are_rejected(settings: dict) -> None:
    with pytest.raises(ValueError):
        OpenAIRequestSettings(**settings)


@pytest.mark.parametrize("command", ["run", "chat"])
def test_cli_shares_request_options(command: str) -> None:
    args = cli._build_parser().parse_args(
        [
            command,
            "--provider",
            "sglang",
            "--thinking",
            "off",
            "--temperature",
            "0",
            "--seed",
            "12345",
            "--chat-template-kwargs",
            '{"preserve_thinking":true}',
        ]
    )
    assert args.provider == "sglang" and args.temperature == 0 and args.seed == 12345
    assert args.chat_template_kwargs == {"preserve_thinking": True}


@pytest.mark.parametrize("value", ["null", "[]", "1", "invalid"])
def test_cli_requires_template_json_object(value: str) -> None:
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["run", "--chat-template-kwargs", value])


async def test_cli_settings_override_and_sglang_thinking_survive_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    environment = tmp_path / "fixture.env"
    environment.write_text("OPENAI_API_KEY=synthetic\n")
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "provider": "sglang",
                "model": "local-model",
                "thinking_level": "low",
                "context_window": 32768,
                "max_tokens": 4096,
                "temperature": 1,
                "seed": 99,
                "chat_template_kwargs": {"preserve_thinking": True},
                "compaction": {"enabled": False},
            }
        )
    )
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda cwd, explicit=None: load_settings(cwd, explicit=explicit, agent_dir=tmp_path / "global"),
    )
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=response_body({}))

    argv = [
        "run",
        "hello",
        "--cwd",
        str(tmp_path),
        "--env-file",
        str(environment),
        "--settings",
        str(settings_path),
        "--session-dir",
        str(tmp_path / "sessions"),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        parser = cli._build_parser()
        assert (
            await cli._cmd_run(
                parser.parse_args(argv + ["--thinking", "off", "--temperature", "0", "--seed", "12345"])
            )
            == 0
        )
        assert payloads[-1]["temperature"] == 0 and payloads[-1]["seed"] == 12345
        assert payloads[-1]["max_tokens"] == 4096
        assert payloads[-1]["chat_template_kwargs"] == {"preserve_thinking": True, "enable_thinking": False}
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue"])) == 0
        assert payloads[-1]["chat_template_kwargs"]["enable_thinking"] is False
        # Transport defaults are read on every invocation, not journaled session configuration.
        assert payloads[-1]["temperature"] == 1 and payloads[-1]["seed"] == 99
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue", "--thinking", "high"])) == 0
        assert payloads[-1]["chat_template_kwargs"]["enable_thinking"] is True
    assert not capsys.readouterr().err


@pytest.mark.parametrize("data", [{"temperature": True}, {"seed": True}, {"chat_template_kwargs": []}])
def test_settings_file_rejects_invalid_request_parameters(tmp_path: Path, data: dict) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_settings(tmp_path, explicit=path, agent_dir=tmp_path / "global")


def test_sglang_runtime_defaults_and_unsupported_anthropic_options() -> None:
    settings = resolve_runtime({"OPENAI_API_KEY": "synthetic"}, provider="sglang", model_id="local-model")
    assert settings.base_url == "http://127.0.0.1:30000/v1" and settings.model.reasoning
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        resolve_runtime({"ANTHROPIC_API_KEY": "synthetic"}, provider="anthropic", model_id="claude", seed=1)


async def test_cumulative_reasoning_usage_is_replaced_not_added() -> None:
    first = {
        "choices": [{"index": 0, "delta": {"reasoning_content": "plan"}, "finish_reason": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "reasoning_tokens": 2},
    }
    body = f"data: {json.dumps(first)}\n\n" + response_body(
        {"prompt_tokens": 10, "completion_tokens": 8, "reasoning_tokens": 5}
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        )
    ) as client:
        provider = OpenAICompatProvider(api_key="synthetic", base_url="https://unit.test", client=client)
        events = [
            event
            async for event in provider.stream(
                ProviderRequest(ModelInfo("test", "test", "sglang")), ProviderOptions()
            )
        ]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.usage.reasoning == 5
    assert events[-1].message.usage.output == 8


@pytest.mark.parametrize("level", ["off", "xhigh"])
async def test_sglang_tool_loop_keeps_request_settings(config: AgentConfig, level: ThinkingLevel) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "Use write to create the requested file."
                            if level != "off"
                            else None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "write_1",
                                    "function": {
                                        "name": "write",
                                        "arguments": json.dumps({"path": "result.txt", "content": "written"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
            body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        else:
            body = response_body({"reasoning_tokens": 0})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatProvider(
            api_key="synthetic",
            base_url="https://unit.test",
            client=client,
            request_settings=OpenAIRequestSettings(
                temperature=0,
                seed=12345,
                chat_template_kwargs={"preserve_thinking": True, "reasoning_effort": "xhigh"}
                if level == "xhigh"
                else None,
            ),
        )
        session = create_agent_session(
            provider,
            config=replace(
                config,
                model=ModelInfo(
                    "local-model",
                    "Local model",
                    "sglang",
                    api="openai-completions",
                    reasoning=True,
                    max_tokens=4096,
                ),
                thinking_level=level,
                compaction=CompactionSettings(enabled=False),
            ),
        )
        try:
            await session.prompt("write a file")
            assert (config.cwd / "result.txt").read_text() == "written"
            assert len(requests) == 2
            for payload in requests:
                assert payload["temperature"] == 0 and payload["seed"] == 12345
                expected_kwargs = {"enable_thinking": level != "off"}
                if level == "xhigh":
                    expected_kwargs.update(preserve_thinking=True, reasoning_effort="xhigh")
                assert payload["chat_template_kwargs"] == expected_kwargs
                assert payload["max_tokens"] == 4096
            assert requests[1]["messages"][-1]["role"] == "tool"
            assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
            if level == "off":
                assert "reasoning_content" not in assistant
            else:
                assert assistant["reasoning_content"] == "Use write to create the requested file."
        finally:
            await session.aclose()


@pytest.mark.parametrize("provider_name", ["sglang", "deepseek", "openai"])
@pytest.mark.parametrize("signature", ["reasoning_content", None])
def test_reasoning_replay_follows_the_streamed_field(provider_name: str, signature: str | None) -> None:
    # As in Pi, a thinking block replays into the field it streamed in (its signature).
    model = ModelInfo("fixture", "fixture", provider_name, api="openai-completions", reasoning=True)
    message = AssistantMessage(
        provider=provider_name,
        api=model.api,
        model=model.id,
        content=[
            ThinkingBlock(thinking="Saved plan", thinking_signature=signature),
            TextBlock(text="Result"),
        ],
        stop_reason="stop",
    )
    provider = OpenAICompatProvider(api_key="synthetic", base_url="https://unit.test")
    payload = provider.payload(
        ProviderRequest(model, messages=[UserMessage(content="first"), message, UserMessage(content="next")]),
        ProviderOptions(thinking_level="low"),
    )
    assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
    if signature:
        assert assistant["reasoning_content"] == "Saved plan"
    elif provider_name == "deepseek":
        assert assistant["reasoning_content"] == ""  # requiresReasoningContentOnAssistantMessages
    else:
        assert "reasoning_content" not in assistant
    assert message.content[0].thinking == "Saved plan"
