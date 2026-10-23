"""Session context estimates, translated from Pi's ``core/compaction/compaction.ts``.

Thresholds and cut points use provider usage where trustworthy and a
conservative chars/4 estimate otherwise. Providers cap output with the
separate chars/3.5 estimate in ``pi_agent.providers._estimate``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._js import json_stringify, utf16_length
from .config import CompactionSettings
from .session_store import SessionCompactionEntry, SessionEntry
from .types import (
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
    get_current_system_message,
)

_CHARS_PER_TOKEN = 4


def calculate_context_tokens(usage: Usage) -> int:
    """Total context tokens from usage, falling back to the component sum."""
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def assistant_usage(message: Message) -> Usage | None:
    if not isinstance(message, AssistantMessage):
        return None
    if message.stop_reason in ("aborted", "error"):
        return None
    if message.usage and calculate_context_tokens(message.usage) > 0:
        return message.usage
    return None


def has_valid_usage(message: Message) -> bool:
    """Whether an assistant message carries trustworthy context usage.

    Error and aborted responses, and responses with zero usage, do not: a
    checkpoint followed by an aborted response must not fall back to the
    pre-checkpoint usage as if it described the current context.
    """
    return assistant_usage(message) is not None


_ESTIMATED_IMAGE_CHARS = 4800


def _content_chars(content: str | Sequence[TextBlock | ImageBlock]) -> int:
    if isinstance(content, str):
        return utf16_length(content)
    return sum(
        utf16_length(block.text) if isinstance(block, TextBlock) else _ESTIMATED_IMAGE_CHARS
        for block in content
    )


def estimate_tokens(message: Message) -> int:
    """Pi's compaction ``estimateTokens``: UTF-16 code units / 4, rounded up (conservative)."""
    chars = 0
    if isinstance(message, SystemMessage):
        chars = _content_chars(message.content)
        chars += sum(utf16_length(section) for section in (message.sections or {}).values() if section)
        if message.tools_added:
            chars += utf16_length(json_stringify([tool.to_json() for tool in message.tools_added]))
    elif isinstance(message, (UserMessage, ToolResultMessage)):
        chars = _content_chars(message.content)
    else:
        for block in message.content:
            if isinstance(block, TextBlock):
                chars += utf16_length(block.text)
            elif isinstance(block, ThinkingBlock):
                chars += utf16_length(block.thinking)
            elif isinstance(block, ToolCallBlock):
                chars += utf16_length(block.name) + utf16_length(json_stringify(block.arguments))
    return math.ceil(chars / _CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: Sequence[Message]) -> int:
    return sum(estimate_tokens(message) for message in messages)


@dataclass
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def estimate_context_tokens(messages: Sequence[Message]) -> ContextUsageEstimate:
    """Pi's compaction ``estimateContextTokens``: last valid usage plus estimates after it."""
    usage_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if assistant_usage(messages[index]) is not None),
        None,
    )
    if usage_index is None:
        estimated = estimate_messages_tokens(messages)
        return ContextUsageEstimate(
            tokens=estimated, usage_tokens=0, trailing_tokens=estimated, last_usage_index=None
        )
    usage = assistant_usage(messages[usage_index])
    assert usage is not None
    usage_tokens = calculate_context_tokens(usage)
    trailing = sum(estimate_tokens(messages[index]) for index in range(usage_index + 1, len(messages)))
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=usage_index,
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    if not settings.enabled:
        return False
    if context_window <= 0:
        return False
    return context_tokens > context_window - settings.reserve_tokens


def projection_system_tokens(projection_messages: Sequence[Message]) -> int:
    """Pure message-size estimate used when provider usage is untrustworthy."""
    system = get_current_system_message(list(projection_messages))
    tokens = estimate_tokens(system) if system is not None else 0
    for message in projection_messages:
        if not isinstance(message, SystemMessage):
            tokens += estimate_tokens(message)
    return tokens


def latest_compaction_entry(entries: Sequence[SessionEntry]) -> SessionCompactionEntry | None:
    for entry in reversed(entries):
        if isinstance(entry, SessionCompactionEntry):
            return entry
    return None
