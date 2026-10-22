from copy import deepcopy
from dataclasses import replace

import pytest

from pi_agent import create_agent_session
from pi_agent.cancellation import Cancellation
from pi_agent.compaction import generate_summary
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.providers import AnthropicProvider, FauxProvider, OpenAICompatProvider
from pi_agent.runtime_config import resolve_runtime
from pi_agent.types import AssistantMessage, ModelInfo, TextBlock, UserMessage


def test_thinking_capabilities_roundtrip_and_are_copied(model):
    original = replace(
        model, reasoning=True, thinking_level_map={"off": "none", "high": "high", "xhigh": None}
    )
    data = original.to_json()
    restored = ModelInfo.from_json(data)
    assert restored == original
    restored.thinking_level_map["high"] = "changed"
    assert original.thinking_level_map["high"] == "high"
    assert data["thinking_level_map"]["high"] == "high"


@pytest.mark.parametrize(
    "mapping", [{}, {"high": True}, {"high": ""}, {"unknown": "high"}, {"high": None}, []]
)
def test_invalid_capabilities_are_rejected(model, mapping):
    with pytest.raises(ValueError, match="thinking_level_map"):
        ModelInfo.from_json({**model.to_json(), "thinking_level_map": mapping})


@pytest.mark.parametrize("kind", ["openai", "deepseek", "anthropic"])
def test_native_effort_map_is_used_and_unsupported_levels_fail_before_requests(kind):
    model = ModelInfo(
        "test",
        "Test",
        kind,
        api="anthropic-messages" if kind == "anthropic" else "openai-completions",
        reasoning=True,
        max_tokens=8192,
        thinking_level_map={"off": "none", "high": "native-effort"},
    )
    provider = (
        AnthropicProvider(api_key="synthetic")
        if kind == "anthropic"
        else OpenAICompatProvider(api_key="synthetic", base_url="https://example.invalid/v1")
    )
    request = ProviderRequest(model, messages=[UserMessage(content="task")])
    payload = provider.payload(request, ProviderOptions(thinking_level="high"))
    effort = payload["output_config"]["effort"] if kind == "anthropic" else payload["reasoning_effort"]
    assert effort == "native-effort"
    with pytest.raises(ValueError, match="does not support"):
        provider.payload(request, ProviderOptions(thinking_level="xhigh"))
    if kind == "openai":
        assert provider.payload(request, ProviderOptions())["reasoning_effort"] == "none"


async def test_unsupported_thinking_change_does_not_touch_journal_and_resume_keeps_map(config):
    config.model = replace(config.model, reasoning=True, thinking_level_map={"off": "none", "high": "high"})
    provider = FauxProvider([AssistantMessage(stop_reason="stop", content=[TextBlock(text="done")])])
    session = create_agent_session(provider, config=config, session="new")
    try:
        await session.prompt("seed")
        before = deepcopy(session.store.entries)
        with pytest.raises(ValueError, match="does not support"):
            session.set_thinking_level("xhigh")
        assert session.store.entries == before
        session.set_thinking_level("high")
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        assert resumed.model.thinking_level_map == config.model.thinking_level_map
        assert resumed.thinking_level == "high"
    finally:
        resumed.dispose()


def test_new_cli_model_can_declare_reasoning_without_a_catalog():
    runtime = resolve_runtime(
        {"OPENAI_API_KEY": "synthetic", "OPENAI_MODEL": "custom"},
        thinking_level_map={"off": "none", "high": "high"},
    )
    assert runtime.model.reasoning
    assert runtime.model.thinking_level_map == {"off": "none", "high": "high"}


async def test_summary_uses_lowest_declared_level_when_off_is_unsupported(model):
    class Provider(FauxProvider):
        async def stream(self, request, options):
            assert options.thinking_level == "low"
            async for event in super().stream(request, options):
                yield event

    model = replace(model, reasoning=True, thinking_level_map={"off": None, "low": "low", "high": "high"})
    provider = Provider([AssistantMessage(stop_reason="stop", content=[TextBlock(text="summary")])])
    summary, _ = await generate_summary(
        [UserMessage(content="task")], provider=provider, model=model, cancellation=Cancellation()
    )
    assert summary == "summary"
