"""Estimate projected context using provider usage and a chars/3.5 fallback."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from .config import CompactionSettings
from .session_store import SessionCompactionEntry, SessionEntry
from .types import (
    AssistantMessage,
    Message,
    ModelInfo,
    SystemMessage,
    ToolDeclaration,
    ToolResultMessage,
    Usage,
    UserMessage,
    content_text,
    get_current_system_message,
)

_CHARS_PER_TOKEN = 3.5


def calculate_context_tokens(usage: Usage) -> int:
    """Total context tokens from usage, falling back to the component sum."""
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def _assistant_usage(message: Message) -> Usage | None:
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
    return _assistant_usage(message) is not None


def estimate_tokens(message: Message) -> int:
    """Approximate one message at 3.5 characters per token."""
    chars = 0
    if isinstance(message, SystemMessage):
        chars = len(content_text(message.content))
        for section in (message.sections or {}).values():
            if section:
                chars += len(section)
        if message.tools_added:
            chars += len(json.dumps([tool.to_json() for tool in message.tools_added], default=str))
    elif isinstance(message, UserMessage):
        chars = len(content_text(message.content))
    elif isinstance(message, AssistantMessage):
        for block in message.content:
            if hasattr(block, "text"):
                chars += len(block.text)
            elif hasattr(block, "thinking"):
                chars += len(block.thinking)
            elif hasattr(block, "name"):
                chars += len(block.name) + len(json.dumps(getattr(block, "arguments", {}), default=str))
    elif isinstance(message, ToolResultMessage):
        chars = len(content_text(message.content))
    return math.ceil(chars / _CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: Sequence[Message]) -> int:
    return sum(estimate_tokens(message) for message in messages)


def output_token_limit(
    messages: Sequence[Message],
    model: ModelInfo,
    requested: int | None = None,
    tools: Sequence[ToolDeclaration] = (),
) -> int:
    """Bound output by model limits and estimated input, with a finite fallback."""
    limit = requested if requested is not None else (model.max_tokens or 8192)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if model.max_tokens > 0:
        limit = min(limit, model.max_tokens)
    if model.context_window > 0:
        system = get_current_system_message(list(messages))
        if system is not None:
            system.tools_added = None
            system.tools_removed = None
        input_tokens = estimate_tokens(system) if system is not None else 0
        input_tokens += sum(
            estimate_tokens(message) for message in messages if not isinstance(message, SystemMessage)
        )
        if tools:
            input_tokens += estimate_tokens(SystemMessage(tools_added=list(tools)))
        available = model.context_window - input_tokens - 256
        if available <= 0:
            raise ValueError("Estimated input exceeds the context window")
        limit = min(limit, available)
    return limit


@dataclass
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def estimate_context_tokens(messages: Sequence[Message]) -> ContextUsageEstimate:
    """Provider usage up to the last valid response, heuristic after it."""
    usage_index: int | None = None
    usage: Usage | None = None
    for index in range(len(messages) - 1, -1, -1):
        candidate = _assistant_usage(messages[index])
        if candidate is not None:
            usage_index = index
            usage = candidate
            break

    if usage is None or usage_index is None:
        estimated = estimate_messages_tokens(messages)
        return ContextUsageEstimate(
            tokens=estimated,
            usage_tokens=0,
            trailing_tokens=estimated,
            last_usage_index=None,
        )

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
