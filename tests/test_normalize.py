"""Port-specific properties of the history projection; pi parity is in tests/parity."""

from copy import deepcopy

from pi_agent.providers import anthropic
from pi_agent.providers._normalize import NO_RESULT_TEXT, transform_messages
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


def test_cross_model_projection_normalizes_ids_without_mutating_history() -> None:
    long_id = "native|" + "x" * 200
    messages = [
        AssistantMessage(
            provider="source",
            api="source",
            model="source",
            stop_reason="toolUse",
            content=[
                ThinkingBlock(thinking="plan", thinking_signature="sig"),
                ToolCallBlock(id=long_id, name="read", thought_signature="sig"),
            ],
        ),
        SystemMessage(content="later instruction"),
        ToolResultMessage(tool_call_id=long_id, tool_name="read", content=[TextBlock(text="result")]),
        UserMessage(content="continue"),
    ]
    before = deepcopy(messages)
    model = ModelInfo(id="new", name="new", provider="anthropic", api="anthropic-messages")
    projected = transform_messages(messages, model, anthropic.normalize_tool_call_id)
    (call,) = projected[0].content
    assert call.id == ("native_" + "x" * 200)[:64] and call.thought_signature is None
    assert projected[1].tool_call_id == call.id
    assert isinstance(projected[2], SystemMessage)  # Held until after the results.
    assert messages == before


def test_in_flight_partial_response_is_not_replayed() -> None:
    partial = AssistantMessage(stop_reason="pending", content=[ToolCallBlock(id="p", name="write")])
    projected = transform_messages(
        [partial, UserMessage(content="retry")], ModelInfo(id="m", name="m", provider="test")
    )
    assert projected == [UserMessage(content="retry")]


def test_missing_result_is_answered_like_pi_without_marking_it_interrupted() -> None:
    assistant = AssistantMessage(stop_reason="toolUse", content=[ToolCallBlock(id="a", name="write")])
    (_, synthetic) = transform_messages([assistant], ModelInfo(id="m", name="m", provider="test"))
    assert synthetic.tool_call_id == "a" and synthetic.is_error and not synthetic.interrupted
    assert synthetic.content == [TextBlock(text=NO_RESULT_TEXT)]
