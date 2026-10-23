"""Provider-side context estimate, translated from Pi's ``packages/ai/src/utils/estimate.ts``.

Providers use it to cap output tokens. Session compaction uses the separate
chars/4 estimate of ``pi_agent.context`` (Pi's ``compaction.ts``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pi_agent._js import json_stringify, utf16_length
from pi_agent.context import ContextUsageEstimate, assistant_usage, calculate_context_tokens
from pi_agent.types import (
    AssistantMessage,
    ImageBlock,
    Message,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    get_system_message_text,
)

_CHARS_PER_TOKEN = 3.5
_ESTIMATED_IMAGE_CHARS = 4800


def _text_tokens(text: str) -> int:
    """Pi's ``estimateTextTokens``: UTF-16 length / 3.5, rounded up."""
    return math.ceil(utf16_length(text) / _CHARS_PER_TOKEN)


def _content_tokens(content: str | Sequence[TextBlock | ImageBlock]) -> int:
    if isinstance(content, str):
        return _text_tokens(content)
    chars = sum(
        utf16_length(block.text) if isinstance(block, TextBlock) else _ESTIMATED_IMAGE_CHARS
        for block in content
    )
    return math.ceil(chars / _CHARS_PER_TOKEN)


def _tools_tokens(tools: Sequence[object] | None) -> int:
    if not tools:
        return 0
    return _text_tokens(
        json_stringify([tool.to_json() if hasattr(tool, "to_json") else tool for tool in tools])
    )


def estimate_message_tokens(message: Message) -> int:
    """Pi's ``estimateMessageTokens`` (3.5 UTF-16 code units per token)."""
    if isinstance(message, SystemMessage):
        return (
            _text_tokens(get_system_message_text(message))
            + _tools_tokens(message.tools_added)
            + _tools_tokens(message.tools_removed)
        )
    if isinstance(message, (UserMessage, ToolResultMessage)):
        return _content_tokens(message.content)
    chars = 0
    for block in message.content:
        if isinstance(block, TextBlock):
            chars += utf16_length(block.text)
        elif isinstance(block, ThinkingBlock):
            chars += utf16_length(block.thinking)
        elif isinstance(block, ToolCallBlock):
            chars += utf16_length(block.name) + utf16_length(json_stringify(block.arguments))
    return math.ceil(chars / _CHARS_PER_TOKEN)


def _last_assistant_usage(messages: Sequence[Message]) -> tuple[Usage, int] | None:
    """Pi's ``getLastAssistantUsageInfo``: usage only applies to the prefix it was measured on."""
    latest_prefix_timestamp = -math.inf
    found: tuple[Usage, int] | None = None
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage) and message.timestamp >= latest_prefix_timestamp:
            usage = assistant_usage(message)
            if usage is not None:
                found = (usage, index)
        latest_prefix_timestamp = max(latest_prefix_timestamp, message.timestamp)
    return found


def estimate_transcript_tokens(messages: Sequence[Message]) -> ContextUsageEstimate:
    """Pi's ai ``estimateContextTokens``: last valid response's usage, heuristic after it."""
    found = _last_assistant_usage(messages)
    if found is None:
        estimated = sum(estimate_message_tokens(message) for message in messages)
        return ContextUsageEstimate(
            tokens=estimated, usage_tokens=0, trailing_tokens=estimated, last_usage_index=None
        )
    usage, usage_index = found
    usage_tokens = calculate_context_tokens(usage)
    trailing = sum(
        estimate_message_tokens(messages[index]) for index in range(usage_index + 1, len(messages))
    )
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=usage_index,
    )
