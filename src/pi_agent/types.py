"""Canonical messages, content blocks, and strict JSON serialization."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

#: JSON-compatible value tree used for details/structured content.
JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

#: Final stop reasons of an assistant response, mirroring Pi's StopReason
#: minus "deferred" (deferred responses are out of scope).
_STOP_REASONS = frozenset({"pending", "stop", "length", "toolUse", "error", "aborted"})
StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted"]

_THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh", "max"})
#: Thinking/reasoning levels understood by the agent core. Providers map these
#: to their native effort levels.
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


class MessageFormatError(ValueError):
    """A serialized message record is structurally invalid."""


def now_ms() -> int:
    """Unix timestamp in milliseconds, the timestamp unit of all messages."""
    return int(time.time() * 1000)


def _require_str(data: object, key: str, *, where: str) -> str:
    value = data.get(key) if isinstance(data, dict) else None  # type: ignore[attr-defined]
    if not isinstance(value, str):
        raise MessageFormatError(f"{where}: field '{key}' must be a string")
    return value


def _optional_str(data: dict[str, object], key: str, *, where: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MessageFormatError(f"{where}: field '{key}' must be a string or null")
    return value


def _require_int(data: dict[str, object], key: str, *, where: str, minimum: int | None = None) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MessageFormatError(f"{where}: field '{key}' must be an integer")
    if not math.isfinite(value):
        raise MessageFormatError(f"{where}: field '{key}' must be finite")
    if minimum is not None and value < minimum:
        raise MessageFormatError(f"{where}: field '{key}' must be >= {minimum}")
    return value


def _optional_int(data: dict[str, object], key: str, *, where: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not math.isfinite(value):
        raise MessageFormatError(f"{where}: field '{key}' must be a finite integer or null")
    return value


def _finite_float(value: object, *, where: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MessageFormatError(f"{where}: field '{key}' must be a finite number")
    return float(value)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


@dataclass
class Cost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "total": self.total,
        }

    @classmethod
    def from_json(cls, data: object) -> Cost:
        if not isinstance(data, dict):
            raise MessageFormatError("usage.cost: must be an object")
        where = "usage.cost"
        return cls(
            input=_finite_float(data.get("input", 0), where=where, key="input"),
            output=_finite_float(data.get("output", 0), where=where, key="output"),
            cache_read=_finite_float(data.get("cache_read", 0), where=where, key="cache_read"),
            cache_write=_finite_float(data.get("cache_write", 0), where=where, key="cache_write"),
            total=_finite_float(data.get("total", 0), where=where, key="total"),
        )


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    total_tokens: int = 0
    cost: Cost = field(default_factory=Cost)

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
        }
        if self.cache_write_1h is not None:
            data["cache_write_1h"] = self.cache_write_1h
        if self.reasoning is not None:
            data["reasoning"] = self.reasoning
        data["total_tokens"] = self.total_tokens
        data["cost"] = self.cost.to_json()
        return data

    @classmethod
    def from_json(cls, data: object) -> Usage:
        if not isinstance(data, dict):
            raise MessageFormatError("usage: must be an object")
        where = "usage"
        cost_data = data.get("cost")
        return cls(
            input=_require_int(data, "input", where=where, minimum=0),
            output=_require_int(data, "output", where=where, minimum=0),
            cache_read=_require_int(data, "cache_read", where=where, minimum=0),
            cache_write=_require_int(data, "cache_write", where=where, minimum=0),
            cache_write_1h=_optional_int(data, "cache_write_1h", where=where),
            reasoning=_optional_int(data, "reasoning", where=where),
            total_tokens=_require_int(data, "total_tokens", where=where, minimum=0),
            cost=Cost.from_json(cost_data) if cost_data is not None else Cost(),
        )


def empty_usage() -> Usage:
    return Usage()


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    type: Literal["text"] = "text"
    text: str = ""
    text_signature: str | None = None

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {"type": "text", "text": self.text}
        if self.text_signature is not None:
            data["text_signature"] = self.text_signature
        return data

    @classmethod
    def from_json(cls, data: object) -> TextBlock:
        if not isinstance(data, dict):
            raise MessageFormatError("text block: must be an object")
        return cls(
            text=_require_str(data, "text", where="text block"),
            text_signature=_optional_str(data, "text_signature", where="text block"),
        )


@dataclass
class ImageBlock:
    """Base64 image content in user messages and tool results (Pi's ImageContent)."""

    type: Literal["image"] = "image"
    data: str = ""
    mime_type: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {"type": "image", "data": self.data, "mime_type": self.mime_type}

    @classmethod
    def from_json(cls, data: object) -> ImageBlock:
        if not isinstance(data, dict):
            raise MessageFormatError("image block: must be an object")
        return cls(
            data=_require_str(data, "data", where="image block"),
            mime_type=_require_str(data, "mime_type", where="image block"),
        )


#: Content of user messages and tool results.
UserContentBlock = TextBlock | ImageBlock


def user_content_block_from_json(data: object, *, where: str) -> UserContentBlock:
    if isinstance(data, dict) and data.get("type") == "image":
        return ImageBlock.from_json(data)
    if isinstance(data, dict) and data.get("type") not in (None, "text"):
        raise MessageFormatError(f"{where}: unknown content type {data.get('type')!r}")
    return TextBlock.from_json(data)


@dataclass
class ThinkingBlock:
    """Reasoning text plus provider-native opaque replay metadata.

    ``thinking_signature`` and ``provider_data`` are stored and replayed
    verbatim; this package never interprets them.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinking_signature: str | None = None
    redacted: bool | None = None
    provider_data: dict[str, JsonValue] | None = None

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {"type": "thinking", "thinking": self.thinking}
        if self.thinking_signature is not None:
            data["thinking_signature"] = self.thinking_signature
        if self.redacted is not None:
            data["redacted"] = self.redacted
        if self.provider_data:
            data["provider_data"] = dict(self.provider_data)
        return data

    @classmethod
    def from_json(cls, data: object) -> ThinkingBlock:
        if not isinstance(data, dict):
            raise MessageFormatError("thinking block: must be an object")
        provider_data = data.get("provider_data")
        if provider_data is not None and not isinstance(provider_data, dict):
            raise MessageFormatError("thinking block: 'provider_data' must be an object")
        redacted = data.get("redacted")
        if redacted is not None and not isinstance(redacted, bool):
            raise MessageFormatError("thinking block: 'redacted' must be a boolean")
        return cls(
            thinking=_require_str(data, "thinking", where="thinking block"),
            thinking_signature=_optional_str(data, "thinking_signature", where="thinking block"),
            redacted=redacted,
            provider_data=dict(provider_data) if provider_data is not None else None,
        )


@dataclass
class ToolCallBlock:
    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, JsonValue] = field(default_factory=dict)
    thought_signature: str | None = None
    namespace: str | None = None
    #: Why the finished arguments text was not a JSON object (Pi's
    #: ``argumentsError``); ``arguments`` is then ``{}`` and validation
    #: reports this text back to the model.
    arguments_error: str | None = None

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "type": "toolCall",
            "id": self.id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.thought_signature is not None:
            data["thought_signature"] = self.thought_signature
        if self.namespace is not None:
            data["namespace"] = self.namespace
        if self.arguments_error is not None:
            data["arguments_error"] = self.arguments_error
        return data

    @classmethod
    def from_json(cls, data: object) -> ToolCallBlock:
        if not isinstance(data, dict):
            raise MessageFormatError("toolCall block: must be an object")
        arguments = data.get("arguments")
        if not isinstance(arguments, dict):
            raise MessageFormatError("toolCall block: 'arguments' must be an object")
        block_id = _require_str(data, "id", where="toolCall block")
        name = _require_str(data, "name", where="toolCall block")
        if not block_id or not name:
            raise MessageFormatError("toolCall block: 'id' and 'name' must be non-empty")
        return cls(
            id=block_id,
            name=name,
            arguments=dict(arguments),
            thought_signature=_optional_str(data, "thought_signature", where="toolCall block"),
            namespace=_optional_str(data, "namespace", where="toolCall block"),
            arguments_error=_optional_str(data, "arguments_error", where="toolCall block"),
        )


ContentBlock = TextBlock | ThinkingBlock | ToolCallBlock


def content_block_from_json(data: object) -> ContentBlock:
    if not isinstance(data, dict):
        raise MessageFormatError("content block: must be an object")
    block_type = data.get("type")
    if block_type == "text":
        return TextBlock.from_json(data)
    if block_type == "thinking":
        return ThinkingBlock.from_json(data)
    if block_type == "toolCall":
        return ToolCallBlock.from_json(data)
    raise MessageFormatError(f"content block: unknown type {block_type!r}")


def content_blocks_from_json(data: object) -> list[ContentBlock]:
    if not isinstance(data, list):
        raise MessageFormatError("content: must be an array of blocks")
    return [content_block_from_json(item) for item in data]


def content_text(content: str | Sequence[ContentBlock | ImageBlock], separator: str = "\n") -> str:
    """Concatenate the text blocks of a content array (or pass a string through)."""
    if isinstance(content, str):
        return content
    return separator.join(block.text for block in content if isinstance(block, TextBlock))


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------


@dataclass
class ToolDeclaration:
    """The model-facing declaration of a tool: name, description, JSON Schema."""

    name: str
    description: str
    parameters: dict[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JsonValue]:
        return {"name": self.name, "description": self.description, "parameters": dict(self.parameters)}

    @classmethod
    def from_json(cls, data: object) -> ToolDeclaration:
        if not isinstance(data, dict):
            raise MessageFormatError("tool declaration: must be an object")
        parameters = data.get("parameters")
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise MessageFormatError("tool declaration: 'parameters' must be an object")
        name = _require_str(data, "name", where="tool declaration")
        if not name:
            raise MessageFormatError("tool declaration: 'name' must be non-empty")
        return cls(
            name=name,
            description=_require_str(data, "description", where="tool declaration"),
            parameters=dict(parameters),
        )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _content_from_json(data: object, *, where: str) -> str | list[UserContentBlock]:
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        if not data:
            return []
        return [user_content_block_from_json(item, where=where) for item in data]
    raise MessageFormatError(f"{where}: 'content' must be a string or an array of content blocks")


@dataclass
class SystemMessage:
    """System instructions and tool declarations at one point in the transcript.

    The leading system message holds the base prompt; later system messages
    add instructions (``content``), patch named ``sections``, or change the
    declared tool set (``tools_added`` / ``tools_removed``). Replaying every
    system message in order yields the current prompt and tools, mirroring Pi.
    """

    role: Literal["system"] = "system"
    content: str | list[TextBlock] = ""
    sections: dict[str, str | None] | None = None
    tools_added: list[ToolDeclaration] | None = None
    tools_removed: list[dict[str, str]] | None = None
    timestamp: int = 0

    #: Host-only resource snapshot; excluded from the rendered provider prompt.
    resource_manifest: str | None = field(default=None, kw_only=True)
    #: Non-secret MCP binding identity; never authorizes a connection by itself.
    mcp_manifest: str | None = field(default=None, kw_only=True)

    def to_json(self) -> dict[str, JsonValue]:
        content: JsonValue = (
            self.content if isinstance(self.content, str) else [block.to_json() for block in self.content]
        )
        data: dict[str, JsonValue] = {"role": "system", "content": content}
        if self.sections:
            data["sections"] = {name: value for name, value in self.sections.items()}
        if self.tools_added:
            data["tools_added"] = [tool.to_json() for tool in self.tools_added]
        if self.tools_removed:
            data["tools_removed"] = [dict(ref) for ref in self.tools_removed]
        data["timestamp"] = self.timestamp
        if self.resource_manifest is not None:
            data["resource_manifest"] = self.resource_manifest
        if self.mcp_manifest is not None:
            data["mcp_manifest"] = self.mcp_manifest
        return data

    @classmethod
    def from_json(cls, data: object) -> SystemMessage:
        if not isinstance(data, dict):
            raise MessageFormatError("system message: must be an object")
        where = "system message"
        manifest = _optional_str(data, "resource_manifest", where=where)
        if manifest is not None:
            from .resources import ResourceSnapshot

            ResourceSnapshot.from_manifest(manifest)
        mcp_manifest = _optional_str(data, "mcp_manifest", where=where)
        if mcp_manifest is not None:
            from .mcp.manifest import McpManifest

            McpManifest.from_manifest(mcp_manifest)
        sections_data = data.get("sections")
        sections: dict[str, str | None] | None = None
        if sections_data is not None:
            if not isinstance(sections_data, dict):
                raise MessageFormatError(f"{where}: 'sections' must be an object")
            sections = {}
            for name, value in sections_data.items():
                if not isinstance(name, str) or (value is not None and not isinstance(value, str)):
                    raise MessageFormatError(f"{where}: 'sections' entries must be string -> string|null")
                sections[name] = value
        tools_added_data = data.get("tools_added")
        tools_removed_data = data.get("tools_removed")
        if tools_added_data is not None and not isinstance(tools_added_data, list):
            raise MessageFormatError(f"{where}: 'tools_added' must be an array")
        if tools_removed_data is not None and not isinstance(tools_removed_data, list):
            raise MessageFormatError(f"{where}: 'tools_removed' must be an array")
        content = _content_from_json(data.get("content"), where=where)
        if not isinstance(content, str) and not all(isinstance(block, TextBlock) for block in content):
            raise MessageFormatError(f"{where}: 'content' must contain only text blocks")
        return cls(
            content=content
            if isinstance(content, str)
            else [block for block in content if isinstance(block, TextBlock)],
            sections=sections,
            resource_manifest=manifest,
            mcp_manifest=mcp_manifest,
            tools_added=(
                [ToolDeclaration.from_json(item) for item in tools_added_data]
                if tools_added_data is not None
                else None
            ),
            tools_removed=(
                [dict(item) for item in tools_removed_data if isinstance(item, dict)]
                if tools_removed_data is not None
                else None
            ),
            timestamp=_require_int(data, "timestamp", where=where, minimum=0),
        )


@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: str | list[UserContentBlock] = ""
    timestamp: int = 0

    def to_json(self) -> dict[str, JsonValue]:
        content: JsonValue = (
            self.content if isinstance(self.content, str) else [block.to_json() for block in self.content]
        )
        return {"role": "user", "content": content, "timestamp": self.timestamp}

    @classmethod
    def from_json(cls, data: object) -> UserMessage:
        if not isinstance(data, dict):
            raise MessageFormatError("user message: must be an object")
        return cls(
            content=_content_from_json(data.get("content"), where="user message"),
            timestamp=_require_int(data, "timestamp", where="user message", minimum=0),
        )


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = field(default_factory=list)
    api: str = "unknown"
    provider: str = "unknown"
    model: str = "unknown"
    response_model: str | None = None
    response_id: str | None = None
    provider_thinking_level: str | None = None
    thinking_level: ThinkingLevel | None = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = "pending"
    error_message: str | None = None
    raw_stop_reason: str | None = None
    provider_data: dict[str, JsonValue] | None = None
    timestamp: int = 0
    duration_ms: int | None = None

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "role": "assistant",
            "content": [block.to_json() for block in self.content],
            "api": self.api,
            "provider": self.provider,
            "model": self.model,
        }
        if self.response_model is not None:
            data["response_model"] = self.response_model
        if self.response_id is not None:
            data["response_id"] = self.response_id
        if self.provider_thinking_level is not None:
            data["provider_thinking_level"] = self.provider_thinking_level
        if self.thinking_level is not None:
            data["thinking_level"] = self.thinking_level
        data["usage"] = self.usage.to_json()
        data["stop_reason"] = self.stop_reason
        if self.error_message is not None:
            data["error_message"] = self.error_message
        if self.raw_stop_reason is not None:
            data["raw_stop_reason"] = self.raw_stop_reason
        if self.provider_data:
            data["provider_data"] = dict(self.provider_data)
        data["timestamp"] = self.timestamp
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        return data

    @classmethod
    def from_json(cls, data: object) -> AssistantMessage:
        if not isinstance(data, dict):
            raise MessageFormatError("assistant message: must be an object")
        where = "assistant message"
        stop_reason = _require_str(data, "stop_reason", where=where)
        if stop_reason not in _STOP_REASONS:
            raise MessageFormatError(f"{where}: invalid stop_reason {stop_reason!r}")
        thinking_level = _optional_str(data, "thinking_level", where=where)
        if thinking_level is not None and thinking_level not in _THINKING_LEVELS:
            raise MessageFormatError(f"{where}: invalid thinking_level {thinking_level!r}")
        provider_data = data.get("provider_data")
        if provider_data is not None and not isinstance(provider_data, dict):
            raise MessageFormatError(f"{where}: 'provider_data' must be an object")
        usage_data = data.get("usage")
        return cls(
            content=content_blocks_from_json(data.get("content")),
            api=_require_str(data, "api", where=where),
            provider=_require_str(data, "provider", where=where),
            model=_require_str(data, "model", where=where),
            response_model=_optional_str(data, "response_model", where=where),
            response_id=_optional_str(data, "response_id", where=where),
            provider_thinking_level=_optional_str(data, "provider_thinking_level", where=where),
            thinking_level=thinking_level,  # type: ignore[arg-type]
            usage=Usage.from_json(usage_data) if usage_data is not None else Usage(),
            stop_reason=stop_reason,  # type: ignore[arg-type]
            error_message=_optional_str(data, "error_message", where=where),
            raw_stop_reason=_optional_str(data, "raw_stop_reason", where=where),
            provider_data=dict(provider_data) if provider_data is not None else None,
            timestamp=_require_int(data, "timestamp", where=where, minimum=0),
            duration_ms=_optional_int(data, "duration_ms", where=where),
        )


@dataclass
class ToolResultMessage:
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[UserContentBlock] = field(default_factory=list)
    details: JsonValue = None
    structured_content: JsonValue = field(default=None, kw_only=True)
    usage: Usage | None = None
    is_error: bool = False
    interrupted: bool = False
    timestamp: int = 0
    duration_ms: int | None = None

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "role": "toolResult",
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "content": [block.to_json() for block in self.content],
            "details": self.details,
        }
        if self.structured_content is not None:
            data["structured_content"] = self.structured_content
        if self.usage is not None:
            data["usage"] = self.usage.to_json()
        data["is_error"] = self.is_error
        if self.interrupted:
            data["interrupted"] = True
        data["timestamp"] = self.timestamp
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        return data

    @classmethod
    def from_json(cls, data: object) -> ToolResultMessage:
        if not isinstance(data, dict):
            raise MessageFormatError("toolResult message: must be an object")
        where = "toolResult message"
        tool_call_id = _require_str(data, "tool_call_id", where=where)
        if not tool_call_id:
            raise MessageFormatError(f"{where}: 'tool_call_id' must be non-empty")
        is_error = data.get("is_error")
        if not isinstance(is_error, bool):
            raise MessageFormatError(f"{where}: 'is_error' must be a boolean")
        interrupted = data.get("interrupted", False)
        if not isinstance(interrupted, bool):
            raise MessageFormatError(f"{where}: 'interrupted' must be a boolean")
        content_data = data.get("content")
        if not isinstance(content_data, list):
            raise MessageFormatError(f"{where}: 'content' must be an array")
        usage_data = data.get("usage")
        return cls(
            tool_call_id=tool_call_id,
            tool_name=_require_str(data, "tool_name", where=where),
            content=[user_content_block_from_json(item, where=where) for item in content_data],
            details=data.get("details"),
            structured_content=data.get("structured_content"),
            usage=Usage.from_json(usage_data) if usage_data is not None else None,
            is_error=is_error,
            interrupted=interrupted,
            timestamp=_require_int(data, "timestamp", where=where, minimum=0),
            duration_ms=_optional_int(data, "duration_ms", where=where),
        )


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage

_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "toolResult"})


def message_from_json(data: object) -> Message:
    if not isinstance(data, dict):
        raise MessageFormatError("message: must be an object")
    role = data.get("role")
    if role not in _MESSAGE_ROLES:
        raise MessageFormatError(f"message: unknown role {role!r}")
    if role == "system":
        return SystemMessage.from_json(data)
    if role == "user":
        return UserMessage.from_json(data)
    if role == "assistant":
        return AssistantMessage.from_json(data)
    return ToolResultMessage.from_json(data)


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------


def _parse_thinking_level_map(value: object) -> dict[str, str | None] | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) - _THINKING_LEVELS
        or any(
            item is not None and (not isinstance(item, str) or not item.strip()) for item in value.values()
        )
    ):
        raise MessageFormatError("Invalid model thinking_level_map")
    return dict(value)


#: Pi's ``EXTENDED_THINKING_LEVELS``, in increasing order.
THINKING_LEVEL_ORDER: tuple[ThinkingLevel, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def _parse_json_object(value: object, *, where: str) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise MessageFormatError(f"{where}: must be an object")
    return deepcopy(value)


@dataclass
class ModelInfo:
    """Identity and limits of the model an agent talks to.

    Mirrors the fields of Pi's ``Model`` that requests and accounting use.
    ``thinking_level_map`` has Pi's meaning: a level mapped to ``None`` is
    unsupported, an absent ``xhigh``/``max`` is unsupported, any other absent
    level is supported and uses the provider's default effort. ``compat`` and
    ``cost`` are Pi's catalog objects (camelCase keys). No secrets, no endpoints.
    """

    id: str
    name: str
    provider: str
    api: str = "unknown"
    reasoning: bool = False
    context_window: int = 0
    max_tokens: int = 0
    thinking_level_map: dict[str, str | None] | None = None
    #: Input modalities, as Pi's ``Model.input``: images reach the model only
    #: when this includes ``"image"``; otherwise they become placeholders.
    input: tuple[str, ...] = ("text",)
    #: Pi's ``Model.compat`` (provider request quirks), or None for defaults.
    compat: dict[str, JsonValue] | None = None
    #: Pi's ``Model.cost`` in USD per million tokens (``input``, ``output``,
    #: ``cacheRead``, ``cacheWrite``, optional ``tiers``), or None when unknown.
    cost: dict[str, JsonValue] | None = None
    #: Pi model.inputLimits, including the model-specific inline image resize profile.
    input_limits: dict[str, JsonValue] | None = None

    @property
    def supports_images(self) -> bool:
        return "image" in self.input

    def supported_thinking_levels(self) -> list[ThinkingLevel]:
        """Pi's ``getSupportedThinkingLevels``."""
        if not self.reasoning:
            return ["off"]
        mapping = self.thinking_level_map or {}
        levels: list[ThinkingLevel] = []
        for level in THINKING_LEVEL_ORDER:
            if level in mapping and mapping[level] is None:
                continue
            if level in ("xhigh", "max") and level not in mapping:
                continue
            levels.append(level)
        return levels

    def clamp_thinking_level(self, level: ThinkingLevel) -> ThinkingLevel:
        """Pi's ``clampThinkingLevel``: the level, else the nearest higher, else lower one."""
        available = self.supported_thinking_levels()
        if level in available:
            return level
        if level not in THINKING_LEVEL_ORDER:
            return available[0] if available else "off"
        index = THINKING_LEVEL_ORDER.index(level)
        for candidate in (*THINKING_LEVEL_ORDER[index:], *reversed(THINKING_LEVEL_ORDER[:index])):
            if candidate in available:
                return candidate
        return available[0] if available else "off"

    def validate_thinking_level(self, level: ThinkingLevel) -> None:
        """Only the name is checked: unsupported levels are clamped, as in Pi."""
        if level not in _THINKING_LEVELS:
            raise ValueError("thinking_level is invalid")

    def thinking_effort(self, level: ThinkingLevel, default: str) -> str:
        """The mapped native effort for ``level``, else ``default``."""
        mapped = (self.thinking_level_map or {}).get(level)
        return mapped if isinstance(mapped, str) else default

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "api": self.api,
            "reasoning": self.reasoning,
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
        }
        if self.thinking_level_map is not None:
            data["thinking_level_map"] = dict(self.thinking_level_map)
        if self.input != ("text",):
            data["input"] = list(self.input)
        if self.compat is not None:
            data["compat"] = deepcopy(self.compat)
        if self.input_limits is not None:
            data["input_limits"] = deepcopy(self.input_limits)
        if self.cost is not None:
            data["cost"] = deepcopy(self.cost)
        return data

    @classmethod
    def from_json(cls, data: object) -> ModelInfo:
        if not isinstance(data, dict):
            raise MessageFormatError("model: must be an object")
        where = "model"
        model_id = _require_str(data, "id", where=where)
        provider = _require_str(data, "provider", where=where)
        if not model_id or not provider:
            raise MessageFormatError(f"{where}: 'id' and 'provider' must be non-empty")
        reasoning = data.get("reasoning", False)
        if not isinstance(reasoning, bool):
            raise MessageFormatError(f"{where}: 'reasoning' must be a boolean")
        name = data.get("name")
        if name is None:
            name = model_id
        if not isinstance(name, str):
            raise MessageFormatError(f"{where}: 'name' must be a string")
        mapping = _parse_thinking_level_map(data.get("thinking_level_map"))
        model_input = data.get("input", ["text"])
        if (
            not isinstance(model_input, list)
            or not model_input
            or any(item not in ("text", "image") for item in model_input)
        ):
            raise MessageFormatError(f"{where}: 'input' must be a list of 'text'/'image'")
        return cls(
            id=model_id,
            name=name,
            provider=provider,
            api=_require_str(data, "api", where=where),
            reasoning=reasoning,
            context_window=_require_int(data, "context_window", where=where, minimum=0),
            max_tokens=_require_int(data, "max_tokens", where=where, minimum=0),
            thinking_level_map=dict(mapping) if mapping is not None else None,
            input=tuple(model_input),
            compat=_parse_json_object(data.get("compat"), where=f"{where}.compat"),
            cost=_parse_json_object(data.get("cost"), where=f"{where}.cost"),
            input_limits=_parse_json_object(data.get("input_limits"), where=f"{where}.input_limits"),
        )


# ---------------------------------------------------------------------------
# Transcript replay helpers (from pi-ai utils/transcript.ts)
# ---------------------------------------------------------------------------


def create_initial_system_message(
    system_prompt: str | None,
    tools: list[ToolDeclaration] | None,
) -> SystemMessage | None:
    """Build the leading system message for a prompt and tool set.

    Returns None when both are empty, so an empty transcript stays empty.
    """
    has_prompt = bool(system_prompt)
    has_tools = bool(tools)
    if not has_prompt and not has_tools:
        return None
    message = SystemMessage(content=system_prompt or "", timestamp=0)
    if tools:
        message.tools_added = list(tools)
    return message


def get_initial_system_message(messages: list[Message]) -> SystemMessage | None:
    first = messages[0] if messages else None
    return first if isinstance(first, SystemMessage) else None


def get_current_tools(messages: list[Message]) -> list[ToolDeclaration]:
    """Resolve the tools available after applying every transcript delta in order."""
    tools: dict[str, ToolDeclaration] = {}
    for message in messages:
        if not isinstance(message, SystemMessage):
            continue
        for ref in message.tools_removed or []:
            tools.pop(ref.get("name", ""), None)
        for tool in message.tools_added or []:
            tools[tool.name] = tool
    return list(tools.values())


def get_system_message_text(message: SystemMessage) -> str:
    """Render a system message as a complete prompt: content, then sections."""
    parts = [content_text(message.content)]
    for value in (message.sections or {}).values():
        if value is not None:
            parts.append(value)
    return "\n\n".join(part for part in parts if part)


def render_system_message_update(message: SystemMessage) -> str:
    """Pi's ``renderSystemMessageUpdate``: a later system message for APIs that accept them."""
    parts: list[str] = []
    text = content_text(message.content)
    if text:
        parts.append(text)
    for name, value in (message.sections or {}).items():
        parts.append(
            f'Removed system prompt section "{name}".'
            if value is None
            else f'Updated system prompt section "{name}":\n\n{value}'
        )
    return "\n\n".join(parts)


def collapse_system_messages(messages: list[Message]) -> list[Message]:
    """Pi's ``collapseSystemMessages``: one replayed leading system message, no later ones."""
    head = get_current_system_message(messages)
    collapsed: list[Message] = [head] if head is not None else []
    collapsed.extend(message for message in messages if not isinstance(message, SystemMessage))
    return collapsed


def resolve_transcript(messages: list[Message], supports_mid_convo_system_messages: bool) -> list[Message]:
    """Pi's ``resolveTranscript``."""
    return list(messages) if supports_mid_convo_system_messages else collapse_system_messages(messages)


def _last_non_additive_tool_change(system_messages: list[SystemMessage]) -> int:
    """Index of the last system message with a removal or same-name redeclaration, or -1."""
    declared: set[str] = set()
    last = -1
    for index, message in enumerate(system_messages):
        if message.tools_removed:
            last = index
        for tool in message.tools_added or []:
            if tool.name in declared:
                last = index
            declared.add(tool.name)
    return last


def resolve_transcript_tools(
    messages: list[Message], support: Literal["none", "additions"]
) -> tuple[list[ToolDeclaration], int]:
    """Pi's ``resolveTranscriptTools``: (request-level tools, number of folded system messages)."""
    system_messages = [message for message in messages if isinstance(message, SystemMessage)]
    if support == "none":
        return get_current_tools(list(system_messages)), len(system_messages)
    folded = max(
        1 if get_initial_system_message(messages) else 0,
        _last_non_additive_tool_change(system_messages) + 1,
    )
    return get_current_tools(list(system_messages[:folded])), folded


def get_current_system_message(messages: list[Message]) -> SystemMessage | None:
    """Replay every system message into one leading system message."""
    content: list[str] = []
    sections: dict[str, str] = {}
    timestamp: int | None = None
    resource_manifest: str | None = None
    mcp_manifest: str | None = None
    for message in messages:
        if not isinstance(message, SystemMessage):
            continue
        if timestamp is None:
            timestamp = message.timestamp
        if message.resource_manifest is not None:
            resource_manifest = message.resource_manifest
        if message.mcp_manifest is not None:
            mcp_manifest = message.mcp_manifest
        text = content_text(message.content)
        if text:
            content.append(text)
        for name, value in (message.sections or {}).items():
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
    tools = get_current_tools(messages)
    if timestamp is None and not tools:
        return None
    return SystemMessage(
        content="\n\n".join(content),
        sections=dict(sections) if sections else None,
        tools_added=tools if tools else None,
        timestamp=timestamp if timestamp is not None else 0,
        resource_manifest=resource_manifest,
        mcp_manifest=mcp_manifest,
    )


def get_current_system_prompt(messages: list[Message]) -> str:
    message = get_current_system_message(messages)
    return get_system_message_text(message) if message else ""
