"""Provider, agent, and session events with explicit JSONL serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .types import (
    AssistantMessage,
    JsonValue,
    Message,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    content_text,
)

# ---------------------------------------------------------------------------
# Provider stream events (one assistant response)
# ---------------------------------------------------------------------------


@dataclass
class StreamStart:
    type: Literal["start"] = "start"
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class TextStart:
    type: Literal["text_start"] = "text_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class TextDelta:
    type: Literal["text_delta"] = "text_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class TextEnd:
    type: Literal["text_end"] = "text_end"
    content_index: int = 0
    content: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ThinkingStart:
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ThinkingDelta:
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ThinkingEnd:
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ToolCallStart:
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = 0
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ToolCallDelta:
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = 0
    delta: str = ""
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class ToolCallEnd:
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = 0
    tool_call: ToolCallBlock = field(default_factory=ToolCallBlock)
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class StreamDone:
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"] = "stop"
    message: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class StreamError:
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"] = "error"
    error: AssistantMessage = field(default_factory=AssistantMessage)


AssistantStreamEvent = (
    StreamStart
    | TextStart
    | TextDelta
    | TextEnd
    | ThinkingStart
    | ThinkingDelta
    | ThinkingEnd
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | StreamDone
    | StreamError
)

# Agent lifecycle events
# Message fields are required: the union has no meaningful default instance.


@dataclass
class AgentStart:
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEnd:
    type: Literal["agent_end"] = "agent_end"
    messages: list[Message] = field(default_factory=list)
    will_retry: bool = False


@dataclass
class TurnStart:
    type: Literal["turn_start"] = "turn_start"


@dataclass
class TurnEnd:
    message: AssistantMessage
    type: Literal["turn_end"] = "turn_end"
    tool_results: list[ToolResultMessage] = field(default_factory=list)


@dataclass
class MessageStart:
    message: Message
    type: Literal["message_start"] = "message_start"


@dataclass
class MessageUpdate:
    message: AssistantMessage
    stream_event: AssistantStreamEvent
    type: Literal["message_update"] = "message_update"


@dataclass
class MessageEnd:
    message: Message
    type: Literal["message_end"] = "message_end"


@dataclass
class ToolExecutionStart:
    tool_call_id: str
    tool_name: str
    args: dict[str, JsonValue]
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass
class ToolExecutionUpdate:
    tool_call_id: str
    tool_name: str
    args: dict[str, JsonValue]
    partial_result: dict[str, JsonValue] | None
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass
class ToolExecutionEnd:
    tool_call_id: str
    tool_name: str
    result: dict[str, JsonValue] | None
    type: Literal["tool_execution_end"] = "tool_execution_end"
    is_error: bool = False
    duration_ms: int | None = None


AgentEvent = (
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageStart
    | MessageUpdate
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionUpdate
    | ToolExecutionEnd
)

# ---------------------------------------------------------------------------
# Session lifecycle events (beyond agent events)
# ---------------------------------------------------------------------------


@dataclass
class AgentSettled:
    type: Literal["agent_settled"] = "agent_settled"
    aborted: bool = False


@dataclass
class QueueUpdate:
    type: Literal["queue_update"] = "queue_update"
    steering: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)


@dataclass
class CompactionStart:
    type: Literal["compaction_start"] = "compaction_start"
    reason: Literal["manual", "threshold", "overflow"] = "manual"


@dataclass
class CompactionEnd:
    type: Literal["compaction_end"] = "compaction_end"
    reason: Literal["manual", "threshold", "overflow"] = "manual"
    result: object | None = None
    aborted: bool = False
    error_message: str | None = None
    #: Pi's ``willRetry``: an overflow recovery re-requests the failed response.
    will_retry: bool = False


@dataclass
class AutoRetryStart:
    type: Literal["auto_retry_start"] = "auto_retry_start"
    attempt: int = 0
    max_attempts: int = 0
    delay_ms: int = 0
    error_message: str = ""


@dataclass
class AutoRetryEnd:
    type: Literal["auto_retry_end"] = "auto_retry_end"
    success: bool = False
    attempt: int = 0
    final_error: str | None = None


@dataclass
class ModelChanged:
    previous_model: ModelInfo
    model: ModelInfo
    type: Literal["model_changed"] = "model_changed"


@dataclass
class ThinkingLevelChanged:
    previous_level: ThinkingLevel
    level: ThinkingLevel
    type: Literal["thinking_level_changed"] = "thinking_level_changed"


SessionEvent = (
    AgentEvent
    | AgentSettled
    | QueueUpdate
    | CompactionStart
    | CompactionEnd
    | AutoRetryStart
    | AutoRetryEnd
    | ModelChanged
    | ThinkingLevelChanged
)

# ---------------------------------------------------------------------------
# Machine-readable serialization (json-event.ts equivalent)
# ---------------------------------------------------------------------------


def _usage_json(usage: Usage | None) -> dict[str, JsonValue]:
    return usage.to_json() if usage is not None else Usage().to_json()


def _block_json(block: object) -> JsonValue:
    if hasattr(block, "to_json"):
        return block.to_json()  # type: ignore[attr-defined, no-any-return]
    if isinstance(block, dict):
        return dict(block)
    if isinstance(block, str):
        return {"type": "text", "text": block}
    return None


def _content_json(content: object) -> list[JsonValue]:
    if not isinstance(content, list):
        return []
    return [_block_json(block) for block in content]


def _tool_result_to_json(result: object) -> dict[str, JsonValue] | None:
    """Serialize a ToolResult or an already-serialized ToolResult dict."""
    if result is None:
        return None
    if hasattr(result, "to_json"):
        serialized = result.to_json()  # type: ignore[attr-defined, no-any-return]
        return serialized if isinstance(serialized, dict) else None
    if not isinstance(result, dict):
        return None
    data: dict[str, JsonValue] = {"content": _content_json(result.get("content"))}
    if "details" in result:
        data["details"] = result.get("details")
    if "structured_content" in result and result.get("structured_content") is not None:
        data["structured_content"] = result.get("structured_content")
    if "usage" in result and result.get("usage") is not None:
        usage = result.get("usage")
        data["usage"] = usage.to_json() if usage is not None and hasattr(usage, "to_json") else usage  # type: ignore[attr-defined]
    data["is_error"] = bool(result.get("is_error", False))
    if result.get("terminate") is not None:
        data["terminate"] = bool(result.get("terminate"))
    return data


def _stream_event_to_json(event: AssistantStreamEvent) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {"type": event.type}
    if isinstance(event, StreamStart | StreamDone | StreamError):
        return data
    data["content_index"] = event.content_index
    if isinstance(event, TextDelta | ThinkingDelta | ToolCallDelta):
        data["delta"] = event.delta
    elif isinstance(event, TextEnd):
        data["content"] = event.content
    elif isinstance(event, ToolCallEnd):
        data["id"] = event.tool_call.id
        data["tool_name"] = event.tool_call.name
    return data


def event_to_json(event: SessionEvent) -> dict[str, JsonValue]:
    """Serialize an event for machine-readable (JSONL) output.

    Streaming updates drop the cumulative assistant snapshot; the terminal
    ``message_end`` event carries the authoritative message.
    """
    data: dict[str, JsonValue] = {"type": event.type}
    if isinstance(event, AgentEnd):
        data["will_retry"] = event.will_retry
        data["messages"] = [message.to_json() for message in event.messages]
    elif isinstance(event, TurnEnd):
        data["message"] = event.message.to_json()
        data["tool_results"] = [result.to_json() for result in event.tool_results]
    elif isinstance(event, MessageStart | MessageEnd):
        data["message"] = event.message.to_json()
    elif isinstance(event, MessageUpdate):
        data["usage"] = _usage_json(event.message.usage)
        data["stream_event"] = _stream_event_to_json(event.stream_event)
    elif isinstance(event, ToolExecutionStart | ToolExecutionUpdate):
        data["tool_call_id"] = event.tool_call_id
        data["tool_name"] = event.tool_name
        data["args"] = dict(event.args)
        if isinstance(event, ToolExecutionUpdate):
            data["partial_result"] = _tool_result_to_json(event.partial_result)
    elif isinstance(event, ToolExecutionEnd):
        data["tool_call_id"] = event.tool_call_id
        data["tool_name"] = event.tool_name
        data["result"] = _tool_result_to_json(event.result)
        data["is_error"] = event.is_error
        if event.duration_ms is not None:
            data["duration_ms"] = event.duration_ms
    elif isinstance(event, AgentSettled):
        data["aborted"] = event.aborted
    elif isinstance(event, ModelChanged):
        data["previous_model"] = event.previous_model.to_json()
        data["model"] = event.model.to_json()
    elif isinstance(event, ThinkingLevelChanged):
        data["previous_level"] = event.previous_level
        data["level"] = event.level
    elif isinstance(event, QueueUpdate):
        data["steering"] = list(event.steering)
        data["follow_up"] = list(event.follow_up)
    elif isinstance(event, CompactionStart):
        data["reason"] = event.reason
    elif isinstance(event, CompactionEnd):
        data["reason"] = event.reason
        data["aborted"] = event.aborted
        data["will_retry"] = event.will_retry
        result = event.result
        data["result"] = result.to_json() if hasattr(result, "to_json") else None  # type: ignore[attr-defined]
        if event.error_message is not None:
            data["error_message"] = event.error_message
    elif isinstance(event, AutoRetryStart):
        data["attempt"] = event.attempt
        data["max_attempts"] = event.max_attempts
        data["delay_ms"] = event.delay_ms
        data["error_message"] = event.error_message
    elif isinstance(event, AutoRetryEnd):
        data["success"] = event.success
        data["attempt"] = event.attempt
        if event.final_error is not None:
            data["final_error"] = event.final_error
    return data


def text_of(message: Message) -> str:
    """Helper for CLI/UI rendering: text content of any message."""
    if isinstance(message, AssistantMessage):
        return "\n".join(block.text for block in message.content if isinstance(block, TextBlock))
    if isinstance(message, ToolResultMessage):
        return content_text(message.content)
    if isinstance(message, SystemMessage):
        return content_text(message.content)
    return content_text(getattr(message, "content", ""))
