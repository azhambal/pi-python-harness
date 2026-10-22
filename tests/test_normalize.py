from copy import deepcopy

from pi_agent.providers._normalize import normalize_messages
from pi_agent.types import (
    AssistantMessage,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
)


def test_cross_model_projection_keeps_pairing_without_native_signatures() -> None:
    messages = [
        AssistantMessage(
            provider="source",
            api="source",
            model="source",
            stop_reason="toolUse",
            content=[
                ThinkingBlock(thinking="plan", thinking_signature="sig"),
                ThinkingBlock(thinking="", redacted=True, provider_data={"data": "opaque"}),
                ToolCallBlock(id="native|" + "x" * 200, name="read", thought_signature="sig"),
            ],
        ),
        SystemMessage(content="later instruction"),
        ToolResultMessage(
            tool_call_id="native|" + "x" * 200, tool_name="read", content=[TextBlock(text="result")]
        ),
        UserMessage(content="continue"),
    ]
    before = deepcopy(messages)
    normalized = normalize_messages(messages, ModelInfo(id="new", name="new", provider="anthropic"))
    assert len(normalized[0].content) == 1
    call = normalized[0].content[0]
    assert len(call.id) <= 64 and "|" not in call.id
    assert call.thought_signature is None
    assert normalized[1].tool_call_id == call.id
    assert isinstance(normalized[2], SystemMessage)
    assert messages == before


def test_failed_attempt_and_its_results_are_not_replayed() -> None:
    failed = AssistantMessage(stop_reason="error", content=[ToolCallBlock(id="bad", name="write")])
    messages = [
        failed,
        ToolResultMessage(tool_call_id="bad", tool_name="write"),
        UserMessage(content="retry"),
    ]
    normalized = normalize_messages(messages, ModelInfo(id="new", name="new", provider="deepseek"))
    assert normalized == [UserMessage(content="retry")]


def test_missing_result_becomes_unknown_outcome_and_duplicate_results_are_removed() -> None:
    assistant = AssistantMessage(
        stop_reason="toolUse",
        content=[ToolCallBlock(id="a", name="write"), ToolCallBlock(id="b", name="read")],
    )
    result = ToolResultMessage(tool_call_id="a", tool_name="write")
    normalized = normalize_messages(
        [assistant, result, result, UserMessage(content="next")],
        ModelInfo(id="new", name="new", provider="deepseek"),
    )
    assert len(normalized) == 4
    assert normalized[2].tool_call_id == "b"
    assert normalized[2].interrupted and normalized[2].is_error
