"""Provider requests and stream decoding must match pi's (see fixtures/pi_version.json).

The fixtures hold what pi's ``streamSimple`` sent through an injected fetch (URL,
selected headers, JSON body) and, for cases with a canned SSE body, the assistant
message it produced. Cases are written by scripts/pi_reference/make_provider_cases.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pi_agent.events import StreamDone, StreamError
from pi_agent.models import find_model, model_from_pi
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.providers import AnthropicProvider, OpenAICompatProvider
from pi_agent.types import AssistantMessage, ModelInfo, message_from_json

HERE = Path(__file__).parent
CASES = json.loads((HERE / "provider_cases.json").read_text(encoding="utf-8"))
EXPECTED = {
    item["name"]: item for item in json.loads((HERE / "fixtures/pi_provider_results.json").read_text("utf-8"))
}
API_KEY = "sk-ant-api03-test"

_FIELDS = {
    "toolCallId": "tool_call_id",
    "toolName": "tool_name",
    "isError": "is_error",
    "stopReason": "stop_reason",
    "mimeType": "mime_type",
    "thinkingSignature": "thinking_signature",
    "textSignature": "text_signature",
    "thoughtSignature": "thought_signature",
    "argumentsError": "arguments_error",
    "cacheRead": "cache_read",
    "cacheWrite": "cache_write",
    "cacheWrite1h": "cache_write_1h",
    "totalTokens": "total_tokens",
    "toolsAdded": "tools_added",
    "toolsRemoved": "tools_removed",
    "providerThinkingLevel": "provider_thinking_level",
    "responseId": "response_id",
    "responseModel": "response_model",
    "errorMessage": "error_message",
    "rawStopReason": "raw_stop_reason",
}
_REVERSE = {snake: camel for camel, snake in _FIELDS.items()}
#: Values inside these keys are user data (tool schemas, arguments), never renamed.
_OPAQUE = {"parameters", "arguments"}


def _rename(value: object, names: dict[str, str]) -> object:
    if isinstance(value, dict):
        return {
            names.get(key, key): item if key in _OPAQUE else _rename(item, names)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename(item, names) for item in value]
    return value


def _model(spec: dict) -> ModelInfo:
    if "catalog" not in spec:
        return model_from_pi(spec)
    model = find_model(spec["provider"], spec["catalog"])
    assert model is not None
    overrides = spec.get("overrides", {})
    return model_from_pi({**_pi_model(model), **overrides})


def _pi_model(model: ModelInfo) -> dict:
    data: dict = {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "api": model.api,
        "reasoning": model.reasoning,
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
        "input": list(model.input),
    }
    for key, value in (
        ("thinkingLevelMap", model.thinking_level_map),
        ("compat", model.compat),
        ("cost", model.cost),
    ):
        if value is not None:
            data[key] = value
    return data


def _base_url(spec: dict) -> str:
    if "catalog" not in spec:
        return spec["baseUrl"]
    groups = json.loads(
        (HERE.parents[1] / f"src/pi_agent/data/models/{spec['provider']}.json").read_text("utf-8")
    )
    return next(
        entries[f"chat:{spec['catalog']}"]
        for entries in groups.values()
        if f"chat:{spec['catalog']}" in entries
    )["baseUrl"]


def _api(case: dict) -> str:
    return _model(case["model"]).api


def _provider(
    case: dict, transport: httpx.MockTransport | None = None
) -> AnthropicProvider | OpenAICompatProvider:
    options = case.get("options", {})
    client = httpx.AsyncClient(transport=transport) if transport else None
    common = {
        "api_key": API_KEY,
        "client": client,
        "cache_retention": options.get("cacheRetention"),
        "thinking_budgets": options.get("thinkingBudgets"),
    }
    if _api(case) == "anthropic-messages":
        return AnthropicProvider(**common)
    return OpenAICompatProvider(base_url=_base_url(case["model"]), **common)


def _request(case: dict) -> tuple[ProviderRequest, ProviderOptions]:
    messages = [message_from_json(_rename(message, _FIELDS)) for message in case["messages"]]
    options = case.get("options", {})
    return (
        ProviderRequest(model=_model(case["model"]), messages=messages),
        ProviderOptions(thinking_level=options.get("reasoning", "off"), session_id=options.get("sessionId")),
    )


def _to_pi(message: AssistantMessage) -> dict:
    data = _rename(message.to_json(), _REVERSE)
    assert isinstance(data, dict)
    for key in ("timestamp", "thinking_level", "duration_ms"):
        data.pop(key, None)
    return data


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
async def test_request_matches_pi(case: dict) -> None:
    provider = _provider(case)
    request, options = _request(case)
    headers, body = provider.prepare(request, options, AssistantMessage())
    expected = EXPECTED[case["name"]]["request"]
    assert provider._url == expected["url"]
    sent = {name.lower(): value for name, value in headers.items()}
    expected_headers = {name: value for name, value in expected["headers"].items() if name != "content-type"}
    assert {name: sent[name] for name in expected_headers if name in sent} == expected_headers
    assert set(sent) == set(expected_headers)
    assert json.loads(json.dumps(body)) == expected["body"]


STREAMS = [case for case in CASES if "sse" in case]


@pytest.mark.parametrize("case", STREAMS, ids=[case["name"] for case in STREAMS])
async def test_stream_matches_pi(case: dict) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=case["sse"].encode()
        )

    provider = _provider(case, httpx.MockTransport(answer))
    request, options = _request(case)
    final: AssistantMessage | None = None
    async for event in provider.stream(request, options):
        if isinstance(event, StreamDone):
            final = event.message
        elif isinstance(event, StreamError):
            final = event.error
    assert final is not None
    expected = dict(EXPECTED[case["name"]]["result"])
    expected.pop("durationMs", None)  # Wall-clock time of the request.
    assert _to_pi(final) == expected
