"""Anthropic Messages API, translated from Pi's ``api/anthropic-messages.ts``.

Request bodies follow Pi's ``streamSimple`` + ``buildParams`` for API-key
authentication (the coding agent passes ``reasoning``, ``sessionId`` and
``thinkingBudgets``); the response stream follows Pi's event handling.
OAuth, federation and GitHub Copilot clients are not supported.
"""

from __future__ import annotations

import codecs
import os
import re
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from typing import Literal

import httpx

from pi_agent._js import sanitize_surrogates
from pi_agent.events import (
    AssistantStreamEvent,
    StreamDone,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.types import (
    AssistantMessage,
    ImageBlock,
    JsonValue,
    Message,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
    Usage,
    UserMessage,
    get_current_tools,
    get_initial_system_message,
    get_system_message_text,
    render_system_message_update,
    resolve_transcript,
)

from ._estimate import estimate_transcript_tokens
from ._http import HTTPProvider
from ._json_parse import parse_json_with_repair, parse_streaming_json, parse_tool_call_arguments
from ._normalize import split_error_tool_result_images, transform_messages

CacheRetention = Literal["none", "short", "long"]

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MID_CONVERSATION_OUTPUT_CONFIG_BETA = "mid-conversation-output-config-2026-07-01"
THINKING_BINDING_CONTROLS_BETA = "thinking-binding-controls-2026-08-01"
INLINE_TOOLS_BETA = "inline-tools-2026-09-15"

#: Declared whenever native tool changes are in use, so the first tool change keeps the
#: cached prefix. Never activated; the model cannot see it.
DEFERRED_TOOL_PLACEHOLDER: dict[str, JsonValue] = {
    "name": "__pi_deferred_placeholder__",
    "description": "Reserved placeholder. Never available. Never call this.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
    "defer_loading": True,
}

#: Pi's ``simple-options.ts``.
CONTEXT_SAFETY_TOKENS = 4096
MIN_ANSWER_TOKENS = 1024
DEFAULT_THINKING_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}

_MESSAGE_EVENTS = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    }
)
_INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

ToolConverter = Callable[[Sequence[ToolDeclaration]], list[dict[str, JsonValue]]]


def normalize_tool_call_id(id: str, model: ModelInfo, source: AssistantMessage) -> str:
    """Pi's anthropic-messages ``normalizeToolCallId``: Anthropic's ID pattern and length."""
    del model, source
    return _INVALID_ID_CHARS.sub("_", id)[:64]


def anthropic_compat(model: ModelInfo, base_url: str = "") -> dict[str, JsonValue]:
    """Pi's ``getAnthropicCompat`` defaults merged with the model's ``compat``."""
    compat = model.compat or {}
    is_open_router = model.provider == "openrouter" or "openrouter.ai" in base_url

    def flag(name: str, default: JsonValue) -> JsonValue:
        value = compat.get(name)
        return default if value is None else value

    return {
        "supportsEagerToolInputStreaming": flag("supportsEagerToolInputStreaming", True),
        "supportsLongCacheRetention": flag("supportsLongCacheRetention", True),
        "sendSessionAffinityHeaders": flag("sendSessionAffinityHeaders", is_open_router),
        "sessionAffinityFormat": flag("sessionAffinityFormat", "openrouter" if is_open_router else None),
        "supportsCacheControlOnTools": flag("supportsCacheControlOnTools", True),
        "supportsTemperature": flag("supportsTemperature", True),
        "allowEmptySignature": flag("allowEmptySignature", False),
        "supportsStrictTools": flag("supportsStrictTools", False),
        "supportsMidConvoSystemMessages": flag("supportsMidConvoSystemMessages", False),
        "supportsMidConvoToolChanges": flag("supportsMidConvoToolChanges", False),
    }


def resolve_cache_retention(retention: CacheRetention | None) -> CacheRetention:
    """Pi's ``resolveCacheRetention``: explicit, else ``PI_CACHE_RETENTION=long``, else short."""
    if retention:
        return retention
    return "long" if os.environ.get("PI_CACHE_RETENTION") == "long" else "short"


def cache_control(
    model: ModelInfo, retention: CacheRetention | None, base_url: str = ""
) -> dict[str, JsonValue] | None:
    """Pi's ``getCacheControl``."""
    resolved = resolve_cache_retention(retention)
    if resolved == "none":
        return None
    control: dict[str, JsonValue] = {"type": "ephemeral"}
    if resolved == "long" and anthropic_compat(model, base_url)["supportsLongCacheRetention"]:
        control["ttl"] = "1h"
    return control


# ----------------------------------------------------------------------
# Token limits (simple-options.ts)
# ----------------------------------------------------------------------


def clamp_max_tokens_to_context(model: ModelInfo, messages: Sequence[Message], max_tokens: int) -> int:
    """Pi's ``clampMaxTokensToContext``."""
    if model.context_window <= 0:
        return max(1, max_tokens)
    available = model.context_window - estimate_transcript_tokens(messages).tokens - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(1, available))


def thinking_budget_for_level(level: ThinkingLevel, budgets: dict[str, int] | None = None) -> int:
    """Pi's ``thinkingBudgetForLevel``; xhigh and max use the high budget."""
    merged = {**DEFAULT_THINKING_BUDGETS, **(budgets or {})}
    return merged["high" if level in ("xhigh", "max") else level]


def adjust_max_tokens_for_thinking(
    base_max_tokens: int, model_max_tokens: int, level: ThinkingLevel, budgets: dict[str, int] | None = None
) -> tuple[int, int]:
    """Pi's ``adjustMaxTokensForThinking``: (max tokens, thinking budget)."""
    thinking_budget = thinking_budget_for_level(level, budgets)
    max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)
    if max_tokens <= thinking_budget:
        thinking_budget = min(thinking_budget, max(0, max_tokens - MIN_ANSWER_TOKENS))
    return max_tokens, thinking_budget


def map_thinking_level_to_effort(model: ModelInfo, level: ThinkingLevel) -> str:
    """Pi's ``mapThinkingLevelToEffort`` for adaptive-thinking models."""
    mapped = (model.thinking_level_map or {}).get(level)
    if isinstance(mapped, str):
        return mapped
    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    return "high"


# ----------------------------------------------------------------------
# Message conversion
# ----------------------------------------------------------------------


def _convert_image(image: ImageBlock) -> dict[str, JsonValue]:
    return {"type": "image", "source": {"type": "base64", "media_type": image.mime_type, "data": image.data}}


def _convert_content_blocks(content: Sequence[TextBlock | ImageBlock]) -> JsonValue:
    """Pi's ``convertContentBlocks``: joined text, or blocks when images are present."""
    if not any(isinstance(block, ImageBlock) for block in content):
        return sanitize_surrogates("\n".join(block.text for block in content if isinstance(block, TextBlock)))
    blocks: list[JsonValue] = [
        {"type": "text", "text": sanitize_surrogates(block.text)}
        if isinstance(block, TextBlock)
        else _convert_image(block)
        for block in content
    ]
    if not any(isinstance(block, TextBlock) for block in content):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


def _convert_tool_result(
    message: ToolResultMessage, content: Sequence[TextBlock | ImageBlock] | None = None
) -> dict[str, JsonValue]:
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": _convert_content_blocks(message.content if content is None else content),
        "is_error": message.is_error,
    }


def _redacted_data(block: ThinkingBlock) -> str:
    if block.thinking_signature:
        return block.thinking_signature
    # Journals written before Pi's representation kept the payload in provider_data.
    data = (block.provider_data or {}).get("data")
    return data if isinstance(data, str) else ""


def _convert_assistant(message: AssistantMessage, allow_empty_signature: bool) -> list[JsonValue]:
    blocks: list[JsonValue] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            if block.text.strip():
                blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        elif isinstance(block, ThinkingBlock):
            if block.redacted:
                blocks.append({"type": "redacted_thinking", "data": _redacted_data(block)})
                continue
            signature = block.thinking_signature
            if not block.thinking.strip() and not (signature and signature.strip()):
                continue
            thinking = sanitize_surrogates(block.thinking)
            if signature and signature.strip():
                blocks.append({"type": "thinking", "thinking": thinking, "signature": signature})
            elif allow_empty_signature:
                blocks.append({"type": "thinking", "thinking": thinking, "signature": ""})
            else:
                # An unsigned block (an aborted stream) is replayed as text.
                blocks.append({"type": "text", "text": thinking})
        elif isinstance(block, ToolCallBlock):
            blocks.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.arguments or {}}
            )
    return blocks


def _convert_system(
    message: SystemMessage, convert_tool_definitions: ToolConverter | None
) -> list[JsonValue]:
    blocks: list[JsonValue] = []
    text = render_system_message_update(message)
    if text:
        blocks.append({"type": "text", "text": sanitize_surrogates(text)})
    if convert_tool_definitions is not None:
        added = message.tools_added or []
        redefined = {tool.name for tool in added}
        for removed in message.tools_removed or []:
            # A new definition under the same name replaces the old one.
            if removed.get("name") not in redefined:
                blocks.append(
                    {"type": "tool_removal", "tool": {"type": "tool_reference", "name": removed["name"]}}
                )
        for definition in convert_tool_definitions(added):
            blocks.append(
                {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition}}
            )
    return blocks


def _convert_user(message: UserMessage) -> JsonValue:
    if isinstance(message.content, str):
        return sanitize_surrogates(message.content) if message.content.strip() else None
    blocks: list[JsonValue] = []
    for item in message.content:
        if isinstance(item, TextBlock):
            if item.text.strip():
                blocks.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            blocks.append(_convert_image(item))
    return blocks or None


def convert_messages(
    messages: Sequence[Message],
    control: dict[str, JsonValue] | None,
    allow_empty_signature: bool = False,
    managed_provider: str | None = None,
    convert_tool_definitions: ToolConverter | None = None,
) -> tuple[list[dict[str, JsonValue]], dict[int, str]]:
    """Pi's ``convertMessages``: (message params, effort of managed assistant messages by index)."""
    params: list[dict[str, JsonValue]] = []
    assistant_levels: dict[int, str] = {}
    # Later system messages wait until the next assistant message: tool_result blocks must
    # directly follow their tool_use.
    pending_system: list[dict[str, JsonValue]] = []

    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, SystemMessage):
            if blocks := _convert_system(message, convert_tool_definitions):
                pending_system.append({"role": "system", "content": blocks})
        elif isinstance(message, UserMessage):
            if (content := _convert_user(message)) is not None:
                params.append({"role": "user", "content": content})
        elif isinstance(message, AssistantMessage):
            params.extend(pending_system)
            pending_system.clear()
            if blocks := _convert_assistant(message, allow_empty_signature):
                if (
                    managed_provider is not None
                    and message.api == "anthropic-messages"
                    and message.provider == managed_provider
                    and message.provider_thinking_level in ("low", "medium", "high", "xhigh", "max")
                ):
                    assistant_levels[len(params)] = message.provider_thinking_level
                params.append({"role": "assistant", "content": blocks})
        elif isinstance(message, ToolResultMessage):
            # Consecutive results share one user message; images of error results follow
            # all of them, since Anthropic rejects images in error results.
            tool_results: list[JsonValue] = []
            error_images: list[JsonValue] = []
            while index < len(messages) and isinstance(result := messages[index], ToolResultMessage):
                split = split_error_tool_result_images(result)
                tool_results.append(_convert_tool_result(result, split[0] if split else None))
                if split:
                    error_images.append({"type": "text", "text": split[1]})
                    error_images.extend(_convert_image(image) for image in split[2])
                index += 1
            index -= 1
            params.append({"role": "user", "content": [*tool_results, *error_images]})
        index += 1
    params.extend(pending_system)

    # Cache the conversation up to the last user or system message.
    if control and params and params[-1]["role"] in ("user", "system"):
        last = params[-1]
        content = last["content"]
        if isinstance(content, list):
            final = content[-1] if content else None
            if isinstance(final, dict) and final.get("type") in (
                "text",
                "image",
                "tool_result",
                "tool_addition",
                "tool_removal",
            ):
                final["cache_control"] = deepcopy(control)
        elif isinstance(content, str):
            last["content"] = [{"type": "text", "text": content, "cache_control": deepcopy(control)}]

    return params, assistant_levels


def insert_thinking_level_messages(
    messages: list[dict[str, JsonValue]], assistant_levels: dict[int, str], active_effort: str
) -> list[dict[str, JsonValue]]:
    """Pi's ``insertThinkingLevelMessages`` for models with mid-conversation effort."""
    result: list[dict[str, JsonValue]] = []
    for index, message in enumerate(messages):
        if index in assistant_levels:
            result.append(
                {"role": "system", "content": [], "output_config": {"effort": assistant_levels[index]}}
            )
        result.append(message)
    result.append({"role": "system", "content": [], "output_config": {"effort": active_effort}})
    return result


def convert_tools(
    tools: Sequence[ToolDeclaration],
    eager_input_streaming: bool,
    control: dict[str, JsonValue] | None = None,
) -> list[dict[str, JsonValue]]:
    """Pi's ``convertTools`` for tools without constrained sampling."""
    converted: list[dict[str, JsonValue]] = []
    for index, tool in enumerate(tools):
        schema = tool.parameters
        definition: dict[str, JsonValue] = {"name": tool.name, "description": tool.description}
        if eager_input_streaming:
            definition["eager_input_streaming"] = True
        definition["input_schema"] = {
            "type": "object",
            "properties": deepcopy(schema.get("properties") or {}),
            "required": deepcopy(schema.get("required") or []),
        }
        if control and index == len(tools) - 1:
            definition["cache_control"] = deepcopy(control)
        converted.append(definition)
    return converted


# ----------------------------------------------------------------------
# Usage and cost
# ----------------------------------------------------------------------


def calculate_cost(cost: JsonValue, usage: Usage) -> None:
    """Pi's ``calculateCost`` (USD per million tokens, input tiers, 1h cache writes at 2x input)."""
    if not isinstance(cost, dict):
        return
    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates: dict[str, JsonValue] = cost
    matched: float = -1
    tiers = cost.get("tiers")
    for tier in tiers if isinstance(tiers, list) else []:
        above = tier.get("inputTokensAbove") if isinstance(tier, dict) else None
        if isinstance(above, (int, float)) and input_tokens > above and above > matched:
            assert isinstance(tier, dict)
            rates, matched = tier, above

    def rate(name: str) -> float:
        value = rates.get(name, 0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write
    usage.cost.input = rate("input") / 1_000_000 * usage.input
    usage.cost.output = rate("output") / 1_000_000 * usage.output
    usage.cost.cache_read = rate("cacheRead") / 1_000_000 * usage.cache_read
    usage.cost.cache_write = (rate("cacheWrite") * short_write + rate("input") * 2 * long_write) / 1_000_000
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write


# ----------------------------------------------------------------------
# Server-sent events (iterateSseMessages)
# ----------------------------------------------------------------------

SseEvent = tuple[str | None, str, list[str]]


def _consume_line(text: str) -> tuple[str, str] | None:
    """Pi's ``consumeLine``: a line ends at the first \\r or \\n; \\r\\n counts once."""
    breaks = [index for index in (text.find("\r"), text.find("\n")) if index != -1]
    if not breaks:
        return None
    end = min(breaks)
    following = end + 1
    if text[end] == "\r" and text[following : following + 1] == "\n":
        following += 1
    return text[:end], text[following:]


class _SseState:
    def __init__(self) -> None:
        self.event: str | None = None
        self.data: list[str] = []
        self.raw: list[str] = []

    def flush(self) -> SseEvent | None:
        if not self.event and not self.data:
            return None
        flushed = (self.event, "\n".join(self.data), list(self.raw))
        self.event, self.data, self.raw = None, [], []
        return flushed

    def line(self, line: str) -> SseEvent | None:
        if line == "":
            return self.flush()
        self.raw.append(line)
        if line.startswith(":"):
            return None
        name, separator, value = line.partition(":")
        if not separator:
            value = ""
        if value.startswith(" "):
            value = value[1:]
        if name == "event":
            self.event = value
        elif name == "data":
            self.data.append(value)
        return None


async def iterate_sse_messages(response: httpx.Response) -> AsyncIterator[SseEvent]:
    """Pi's ``iterateSseMessages`` (TextDecoder: UTF-8, replacement, leading BOM dropped)."""
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
    state = _SseState()
    buffer = ""
    async for chunk in response.aiter_bytes():
        buffer += decoder.decode(chunk)
        while (consumed := _consume_line(buffer)) is not None:
            line, buffer = consumed
            if (event := state.line(line)) is not None:
                yield event
    buffer += decoder.decode(b"", final=True)
    while (consumed := _consume_line(buffer)) is not None:
        line, buffer = consumed
        if (event := state.line(line)) is not None:
            yield event
    if buffer and (event := state.line(buffer)) is not None:
        yield event
    if (event := state.flush()) is not None:
        yield event


async def iterate_anthropic_events(response: httpx.Response) -> AsyncIterator[dict[str, JsonValue]]:
    """Pi's ``iterateAnthropicEvents``."""
    saw_start = saw_stop = False
    async for name, data, raw in iterate_sse_messages(response):
        if name == "error":
            raise ValueError(data)
        if name not in _MESSAGE_EVENTS:
            continue
        try:
            event = parse_json_with_repair(data)
        except ValueError as error:
            raw_text = "\\n".join(raw)
            raise ValueError(
                f"Could not parse Anthropic SSE event {name}: {error}; data={data}; raw={raw_text}"
            ) from None
        if not isinstance(event, dict):
            continue
        if event.get("type") == "message_start":
            saw_start = True
        elif event.get("type") == "message_stop":
            saw_stop = True
        yield event
    if saw_start and not saw_stop:
        raise ValueError("Anthropic stream ended before message_stop")


def map_stop_reason(reason: str, details: object) -> tuple[str, str | None]:
    """Pi's ``mapStopReason``: (stop reason, error message)."""
    if reason in ("end_turn", "pause_turn", "stop_sequence"):
        return "stop", None
    if reason == "max_tokens":
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    if reason == "refusal":
        explanation = details.get("explanation") if isinstance(details, dict) else None
        return "error", str(explanation) if explanation else "The model refused to complete the request"
    if reason == "sensitive":
        return "error", "Provider stopped with: sensitive"
    raise ValueError(f"Unhandled stop reason: {reason}")


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _object(value: object) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


# ----------------------------------------------------------------------
# Provider
# ----------------------------------------------------------------------


class AnthropicProvider(HTTPProvider):
    # The SDK's beta client, which Pi uses, adds ?beta=true.
    endpoint = "/v1/messages?beta=true"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
        thinking_mode: Literal["adaptive", "budget"] | None = None,
        cache_retention: CacheRetention | None = None,
        thinking_budgets: dict[str, int] | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, client=client, timeout=timeout)
        #: None follows the model (``compat.forceAdaptiveThinking``), as Pi does.
        self.thinking_mode = thinking_mode
        self.cache_retention = cache_retention
        self.thinking_budgets = thinking_budgets
        self._base_url = base_url

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        }

    def validate_model(self, model: ModelInfo, thinking_level: ThinkingLevel) -> None:
        model.validate_thinking_level(thinking_level)
        if model.api not in ("unknown", "anthropic-messages") or model.provider in ("openai", "deepseek"):
            raise ValueError("AnthropicProvider requires an Anthropic Messages model API")

    def _adaptive(self, model: ModelInfo) -> bool:
        if self.thinking_mode is not None:
            return self.thinking_mode == "adaptive"
        return (model.compat or {}).get("forceAdaptiveThinking") is True

    def build(
        self, request: ProviderRequest, options: ProviderOptions
    ) -> tuple[dict[str, JsonValue], list[str], str | None]:
        """Pi's ``streamSimple`` + ``buildParams``: (body, beta features, provider thinking level)."""
        model = request.model
        self.validate_model(model, options.thinking_level)
        compat = anthropic_compat(model, self._base_url)
        raw_compat = model.compat or {}
        context = resolve_transcript(list(request.messages), bool(compat["supportsMidConvoSystemMessages"]))
        adaptive = self._adaptive(model)

        # streamSimple maps the reasoning level onto Anthropic options.
        reasoning = None if options.thinking_level == "off" else options.thinking_level
        requested = options.max_tokens if options.max_tokens is not None else model.max_tokens
        # Pi estimates the transcript as given, before system messages are collapsed.
        raw_messages = list(request.messages)
        max_tokens = clamp_max_tokens_to_context(model, raw_messages, requested)
        effort: str | None = None
        budget_tokens = 0
        if reasoning is not None:
            if adaptive:
                effort = map_thinking_level_to_effort(model, reasoning)
            else:
                adjusted, budget = adjust_max_tokens_for_thinking(
                    max_tokens, model.max_tokens, reasoning, self.thinking_budgets
                )
                max_tokens = clamp_max_tokens_to_context(model, raw_messages, adjusted)
                budget_tokens = min(budget, max(0, max_tokens - 1024))

        control = cache_control(model, options.cache_retention or self.cache_retention, self._base_url)
        initial_system = get_initial_system_message(context)
        transformed = transform_messages(context, model, normalize_tool_call_id)
        conversation = transformed[1:] if initial_system else transformed
        eager = bool(compat["supportsEagerToolInputStreaming"])

        def convert_definitions(
            tools: Sequence[ToolDeclaration], tool_control: dict[str, JsonValue] | None = None
        ) -> list[dict[str, JsonValue]]:
            return convert_tools(tools, eager, tool_control)

        # Native tool changes keep the request-level tool list fixed and define later
        # tools in tool_addition blocks; Anthropic needs an initial active tool for that.
        initial_tools = (initial_system.tools_added if initial_system else None) or []
        native_tool_changes = bool(
            compat["supportsMidConvoSystemMessages"]
            and compat["supportsMidConvoToolChanges"]
            and initial_tools
        )
        level_map = model.thinking_level_map or {}
        off_unsupported = "off" in level_map and level_map["off"] is None
        thinking_disabled = model.reasoning and reasoning is None and not off_unsupported
        # Effort cannot change between messages while thinking is disabled.
        effort_messages = raw_compat.get("supportsMidConvoEffort") is True and not thinking_disabled
        converted, assistant_levels = convert_messages(
            conversation,
            control,
            bool(compat["allowEmptySignature"]),
            model.provider if effort_messages else None,
            convert_definitions if native_tool_changes else None,
        )
        active_effort = effort or "high"
        current_tools = get_current_tools(context)
        fallbacks = raw_compat.get("allowedFallbackModels")
        fallbacks = fallbacks if isinstance(fallbacks, list) else []

        betas: list[str] = []
        if current_tools and not eager:
            betas.append(FINE_GRAINED_TOOL_STREAMING_BETA)
        if model.reasoning and reasoning is not None and not adaptive:
            betas.append(INTERLEAVED_THINKING_BETA)
        if fallbacks:
            betas.append(SERVER_SIDE_FALLBACK_BETA)
        if raw_compat.get("supportsMidConvoEffort") is True:
            betas.extend((MID_CONVERSATION_OUTPUT_CONFIG_BETA, THINKING_BINDING_CONTROLS_BETA))
        if native_tool_changes:
            betas.append(INLINE_TOOLS_BETA)

        params: dict[str, JsonValue] = {
            "model": model.id,
            "messages": list(
                insert_thinking_level_messages(converted, assistant_levels, active_effort)
                if effort_messages
                else converted
            ),
            "max_tokens": max_tokens,
            "stream": True,
        }
        system_text = get_system_message_text(initial_system) if initial_system else ""
        if system_text:
            system_block: dict[str, JsonValue] = {"type": "text", "text": sanitize_surrogates(system_text)}
            if control:
                system_block["cache_control"] = deepcopy(control)
            params["system"] = [system_block]

        tool_control = control if compat["supportsCacheControlOnTools"] else None
        if native_tool_changes:
            params["tools"] = [
                *convert_definitions(initial_tools, tool_control),
                deepcopy(DEFERRED_TOOL_PLACEHOLDER),
            ]
        elif current_tools:
            params["tools"] = list(convert_definitions(current_tools, tool_control))

        if thinking_disabled:
            params["thinking"] = {"type": "disabled"}
        elif effort_messages:
            # Managed-effort models drop mismatched thinking prefixes instead of failing.
            params["thinking"] = {
                "type": "adaptive",
                "display": "summarized",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
            params["output_config"] = {"effort": "high"}
        elif model.reasoning and reasoning is not None:
            if adaptive:
                params["thinking"] = {"type": "adaptive", "display": "summarized"}
                if effort:
                    params["output_config"] = {"effort": effort}
            else:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens or 1024,
                    "display": "summarized",
                }

        if fallbacks:
            params["fallbacks"] = [{"model": _object(fallback).get("model")} for fallback in fallbacks]
        return params, list(dict.fromkeys(betas)), active_effort if effort_messages else None

    def payload(self, request: ProviderRequest, options: ProviderOptions) -> dict[str, object]:
        return dict(self.build(request, options)[0])

    def prepare(
        self, request: ProviderRequest, options: ProviderOptions, message: AssistantMessage
    ) -> tuple[dict[str, str], dict[str, object]]:
        params, betas, provider_thinking_level = self.build(request, options)
        if provider_thinking_level:
            message.provider_thinking_level = provider_thinking_level
        headers = self.headers()
        compat = anthropic_compat(request.model, self._base_url)
        if options.session_id and compat["sendSessionAffinityHeaders"]:
            name = "x-session-id" if compat["sessionAffinityFormat"] == "openrouter" else "x-session-affinity"
            headers[name] = options.session_id
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers, dict(params)

    async def decode(
        self, response: httpx.Response, message: AssistantMessage, request: ProviderRequest
    ) -> AsyncIterator[AssistantStreamEvent]:
        """Pi's stream event handling: lenient about unknown blocks and event types."""
        model = request.model
        cost: JsonValue = model.cost
        # Content position of each Anthropic block index; unknown block types get none.
        positions: dict[int, int] = {}
        partial_json: dict[int, str] = {}
        usage = message.usage

        def update_total() -> None:
            usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
            calculate_cost(cost, usage)

        async for event in iterate_anthropic_events(response):
            kind = event.get("type")
            if kind == "message_start":
                start = _object(event.get("message"))
                response_id = start.get("id")
                message.response_id = response_id if isinstance(response_id, str) else None
                response_model = start.get("model")
                if isinstance(response_model, str) and response_model != model.id:
                    message.response_model = response_model
                    allowed = (model.compat or {}).get("allowedFallbackModels")
                    for fallback in allowed if isinstance(allowed, list) else []:
                        entry = _object(fallback)
                        if entry.get("provider") == model.provider and entry.get("model") == response_model:
                            cost = entry.get("cost")
                            break
                start_usage = _object(start.get("usage"))
                usage.input = _int(start_usage.get("input_tokens"))
                usage.output = _int(start_usage.get("output_tokens"))
                usage.cache_read = _int(start_usage.get("cache_read_input_tokens"))
                usage.cache_write = _int(start_usage.get("cache_creation_input_tokens"))
                usage.cache_write_1h = _int(
                    _object(start_usage.get("cache_creation")).get("ephemeral_1h_input_tokens")
                )
                update_total()
            elif kind == "content_block_start":
                native = _object(event.get("content_block"))
                block_type = native.get("type")
                index = _int(event.get("index"))
                position = len(message.content)
                if block_type == "fallback":
                    if message.content:
                        raise ValueError("Anthropic performed an unsupported mid-output model fallback")
                elif block_type == "text":
                    positions[index] = position
                    message.content.append(TextBlock(text=str(native.get("text") or "")))
                    yield TextStart(content_index=position, partial=deepcopy(message))
                elif block_type == "thinking":
                    positions[index] = position
                    message.content.append(
                        ThinkingBlock(
                            thinking=str(native.get("thinking") or ""),
                            thinking_signature=str(native.get("signature") or ""),
                        )
                    )
                    yield ThinkingStart(content_index=position, partial=deepcopy(message))
                elif block_type == "redacted_thinking":
                    positions[index] = position
                    message.content.append(
                        ThinkingBlock(
                            thinking="[Reasoning redacted]",
                            thinking_signature=str(native.get("data") or ""),
                            redacted=True,
                        )
                    )
                    yield ThinkingStart(content_index=position, partial=deepcopy(message))
                elif block_type == "tool_use":
                    positions[index] = position
                    initial = native.get("input")
                    message.content.append(
                        ToolCallBlock(
                            id=str(native.get("id") or ""),
                            name=str(native.get("name") or ""),
                            arguments=deepcopy(initial) if isinstance(initial, dict) else {},
                        )
                    )
                    partial_json[position] = ""
                    yield ToolCallStart(content_index=position, partial=deepcopy(message))
            elif kind in ("content_block_delta", "content_block_stop"):
                found = positions.get(_int(event.get("index")))
                if found is None:
                    continue
                target = message.content[found]
                if kind == "content_block_delta":
                    delta = _object(event.get("delta"))
                    delta_type = delta.get("type")
                    if delta_type == "text_delta" and isinstance(target, TextBlock):
                        text = str(delta.get("text") or "")
                        target.text += text
                        yield TextDelta(content_index=found, delta=text, partial=deepcopy(message))
                    elif delta_type == "thinking_delta" and isinstance(target, ThinkingBlock):
                        text = str(delta.get("thinking") or "")
                        target.thinking += text
                        yield ThinkingDelta(content_index=found, delta=text, partial=deepcopy(message))
                    elif delta_type == "input_json_delta" and isinstance(target, ToolCallBlock):
                        fragment = str(delta.get("partial_json") or "")
                        partial_json[found] += fragment
                        target.arguments = parse_streaming_json(partial_json[found])
                        yield ToolCallDelta(content_index=found, delta=fragment, partial=deepcopy(message))
                    elif delta_type == "signature_delta" and isinstance(target, ThinkingBlock):
                        target.thinking_signature = (target.thinking_signature or "") + str(
                            delta.get("signature") or ""
                        )
                elif isinstance(target, TextBlock):
                    yield TextEnd(content_index=found, content=target.text, partial=deepcopy(message))
                elif isinstance(target, ThinkingBlock):
                    yield ThinkingEnd(content_index=found, partial=deepcopy(message))
                elif isinstance(target, ToolCallBlock):
                    target.arguments, target.arguments_error = parse_tool_call_arguments(
                        partial_json.pop(found, "")
                    )
                    yield ToolCallEnd(
                        content_index=found, tool_call=deepcopy(target), partial=deepcopy(message)
                    )
            elif kind == "message_delta":
                delta = _object(event.get("delta"))
                stop_reason = delta.get("stop_reason")
                if isinstance(stop_reason, str) and stop_reason:
                    message.raw_stop_reason = stop_reason
                    mapped, error_message = map_stop_reason(stop_reason, delta.get("stop_details"))
                    message.stop_reason = mapped  # type: ignore[assignment]
                    if error_message:
                        message.error_message = error_message
                delta_usage = event.get("usage")
                if isinstance(delta_usage, dict):
                    # Keep message_start values when proxies omit fields here.
                    for key, attribute in (
                        ("input_tokens", "input"),
                        ("output_tokens", "output"),
                        ("cache_read_input_tokens", "cache_read"),
                        ("cache_creation_input_tokens", "cache_write"),
                    ):
                        if delta_usage.get(key) is not None:
                            setattr(usage, attribute, _int(delta_usage[key]))
                    creation = _object(delta_usage.get("cache_creation"))
                    if creation.get("ephemeral_1h_input_tokens") is not None:
                        usage.cache_write_1h = _int(creation["ephemeral_1h_input_tokens"])
                    thinking_tokens = _object(delta_usage.get("output_tokens_details")).get("thinking_tokens")
                    if thinking_tokens is not None:
                        usage.reasoning = _int(thinking_tokens)
                update_total()

        if message.stop_reason == "pending":
            raise ValueError("Anthropic stream ended without a stop reason")
        if message.stop_reason not in ("stop", "length", "toolUse"):
            raise ValueError(message.error_message or "An unknown error occurred")
        yield StreamDone(reason=message.stop_reason, message=deepcopy(message))
