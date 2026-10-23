"""History projection must match pi's transformMessages (see fixtures/pi_version.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_agent.providers import anthropic, openai_compat
from pi_agent.providers._normalize import transform_messages
from pi_agent.types import ModelInfo, message_from_json

HERE = Path(__file__).parent
CASES = json.loads((HERE / "transform_cases.json").read_text(encoding="utf-8"))
EXPECTED = {
    item["name"]: item["messages"]
    for item in json.loads((HERE / "fixtures/pi_transform_results.json").read_text("utf-8"))
}

#: pi's camelCase message fields and this port's snake_case ones.
_FIELDS = {
    "toolCallId": "tool_call_id",
    "toolName": "tool_name",
    "isError": "is_error",
    "stopReason": "stop_reason",
    "mimeType": "mime_type",
    "thinkingSignature": "thinking_signature",
    "textSignature": "text_signature",
    "thoughtSignature": "thought_signature",
    "cacheRead": "cache_read",
    "cacheWrite": "cache_write",
    "totalTokens": "total_tokens",
}
_REVERSE = {snake: camel for camel, snake in _FIELDS.items()}

NORMALIZERS = {"anthropic": anthropic.normalize_tool_call_id, "openai": openai_compat.normalize_tool_call_id}


def _rename(value: object, names: dict[str, str]) -> object:
    if isinstance(value, dict):
        return {names.get(key, key): _rename(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item, names) for item in value]
    return value


def _to_pi(message) -> dict:
    data = _rename(message.to_json(), _REVERSE)
    data.pop("timestamp", None)
    if data.get("details", "") is None:
        data.pop("details")
    return data


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_transform_matches_pi(case: dict) -> None:
    spec = case["model"]
    model = ModelInfo(
        id=spec["id"], name=spec["id"], provider=spec["provider"], api=spec["api"], input=tuple(spec["input"])
    )
    messages = [message_from_json(_rename(message, _FIELDS)) for message in case["messages"]]
    normalizer = NORMALIZERS.get(case.get("normalizer", "none"))
    actual = [_to_pi(message) for message in transform_messages(messages, model, normalizer)]
    assert actual == EXPECTED[case["name"]]
