"""OpenAI-compatible Chat Completions, translated from Pi's ``api/openai-completions.ts``.

Request bodies follow Pi's ``streamSimple`` + ``buildParams``, including the
compatibility flags Pi detects from the provider and URL or reads from the
model's ``compat``; the response stream follows Pi's chunk handling. Grammar
(custom) tools and strict JSON-schema sampling are not used by this port's tools.

Port extensions: ``OpenAIRequestSettings`` (temperature, seed and literal
``chat_template_kwargs`` from the CLI or settings) and SGLang, whose models
default to ``chat_template_kwargs.enable_thinking`` (Pi's ``chat-template``
thinking format).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from dataclasses import dataclass

import httpx

from pi_agent._js import json_stringify, sanitize_surrogates, short_hash
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
    UserMessage,
    get_system_message_text,
    render_system_message_update,
    resolve_transcript,
    resolve_transcript_tools,
)

from ._http import HTTPProvider
from ._json_parse import parse_streaming_json, parse_tool_call_arguments
from ._normalize import transform_messages
from .anthropic import (
    CacheRetention,
    calculate_cost,
    clamp_max_tokens_to_context,
    iterate_sse_messages,
    resolve_cache_retention,
    thinking_budget_for_level,
)

_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")
_INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_PROMPT_CACHE_KEY_MAX_LENGTH = 64
_MIN_ANSWER_TOKENS = 1024
_BRIDGE_TEXT = "I have processed the tool results."


def token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Provider token count must be a non-negative integer")
    return value


@dataclass(frozen=True)
class OpenAIRequestSettings:
    """Transport defaults, applied to every request including compaction."""

    temperature: float | None = None
    seed: int | None = None
    chat_template_kwargs: dict[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be finite and between 0 and 2")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer")
        if self.chat_template_kwargs is not None:
            if not isinstance(self.chat_template_kwargs, dict) or not all(
                isinstance(key, str) for key in self.chat_template_kwargs
            ):
                raise ValueError("chat_template_kwargs must be a JSON object")
            try:
                json.dumps(self.chat_template_kwargs, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("chat_template_kwargs must contain finite JSON values") from error


def normalize_tool_call_id(id: str, model: ModelInfo, source: AssistantMessage) -> str:
    """Pi's openai-completions ``normalizeToolCallId`` for replaying another model's calls."""
    del source
    # Responses API IDs are "{call_id}|{item_id}"; Chat Completions needs distinct IDs of
    # at most 40 allowed characters.
    if "|" in id:
        separator_index = id.index("|")
        call_id = _INVALID_ID_CHARS.sub("_", id[:separator_index])
        item_id = _INVALID_ID_CHARS.sub("_", id[separator_index + 1 :])
        combined_id = f"{call_id}_{item_id}" if item_id else call_id
        if len(combined_id) <= 40:
            return combined_id
        hash_text = short_hash(id)[:8]
        prefix = call_id[: max(1, 40 - len(hash_text) - 1)]
        return f"{prefix}_{hash_text}"
    if model.provider == "openai":
        return id[:40] if len(id) > 40 else id
    return id


# ----------------------------------------------------------------------
# Compatibility flags (detectCompat / getCompat)
# ----------------------------------------------------------------------


def detect_compat(model: ModelInfo, base_url: str) -> dict[str, JsonValue]:
    """Pi's ``detectCompat``: flags derived from the provider name and base URL."""
    provider = model.provider
    is_zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_open_router = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_cerebras = provider == "cerebras" or "cerebras.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url.lower()
    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_non_standard = (
        is_nvidia
        or is_cerebras
        or is_grok
        or is_together
        or "chutes.ai" in base_url
        or is_deepseek
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers
        or is_cloudflare_gateway
        or is_ant_ling
    )
    use_max_tokens = (
        "chutes.ai" in base_url
        or is_deepseek
        or is_moonshot
        or is_cloudflare_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )
    developer_role_model = is_open_router and model.id.startswith(("anthropic/", "openai/"))
    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_open_router:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"
    compat: dict[str, JsonValue] = {
        "supportsStore": not is_non_standard,
        "supportsDeveloperRole": developer_role_model or (not is_non_standard and not is_open_router),
        "supportsReasoningEffort": not (
            is_grok
            or is_zai
            or is_moonshot
            or is_together
            or is_cloudflare_gateway
            or is_nvidia
            or is_ant_ling
        ),
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deepseek,
        "thinkingFormat": thinking_format,
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "zaiToolStream": False,
        "supportsThinkingTokenBudget": False,
        "thinkingTokenBudgetField": None,
        "supportsStrictMode": False,
        "supportsMidConvoSystemMessages": False,
        "supportsMidConvoToolAdditions": False,
        "cacheControlFormat": "anthropic"
        if provider == "openrouter" and model.id.startswith("anthropic/")
        else None,
        "sendSessionAffinityHeaders": is_open_router,
        "sessionAffinityFormat": "openrouter" if is_open_router else "openai",
        "supportsLongCacheRetention": not (
            is_together or is_cloudflare_workers or is_cloudflare_gateway or is_nvidia or is_ant_ling
        ),
    }
    if provider == "sglang":
        # Port extension: SGLang serves chat-template models; the thinking switch is a template value.
        compat.update(
            supportsStore=False,
            supportsDeveloperRole=False,
            maxTokensField="max_tokens",
            thinkingFormat="chat-template",
            chatTemplateKwargs={"enable_thinking": {"$var": "thinking.enabled"}},
        )
    return compat


def get_compat(model: ModelInfo, base_url: str) -> dict[str, JsonValue]:
    """Pi's ``getCompat``: detected flags overridden by the model's ``compat``."""
    compat = detect_compat(model, base_url)
    for key, value in (model.compat or {}).items():
        if value is not None:
            compat[key] = deepcopy(value)
    return compat


# ----------------------------------------------------------------------
# Reasoning details (structured reasoning replay, as OpenRouter streams it)
# ----------------------------------------------------------------------


def _valid_reasoning_detail(detail: object) -> bool:
    if not isinstance(detail, dict):
        return False
    if not (
        (detail.get("id") is None or isinstance(detail.get("id"), str))
        and ("format" not in detail or isinstance(detail["format"], str))
        and ("index" not in detail or isinstance(detail["index"], (int, float)))
    ):
        return False
    kind = detail.get("type")
    if kind == "reasoning.summary":
        return isinstance(detail.get("summary"), str)
    if kind == "reasoning.encrypted":
        return isinstance(detail.get("data"), str)
    if kind == "reasoning.text":
        return isinstance(detail.get("text"), str) and (
            detail.get("signature") is None or isinstance(detail.get("signature"), str)
        )
    return False


def _parse_reasoning_details(signature: str | None) -> list[JsonValue] | None:
    if not signature:
        return None
    try:
        parsed = json.loads(signature)
    except ValueError:
        return None
    if isinstance(parsed, list) and parsed and all(_valid_reasoning_detail(item) for item in parsed):
        return parsed
    return None


def _parse_legacy_encrypted_detail(signature: str | None) -> dict[str, JsonValue] | None:
    if not signature:
        return None
    try:
        parsed = json.loads(signature)
    except ValueError:
        return None
    if (
        _valid_reasoning_detail(parsed)
        and parsed.get("type") == "reasoning.encrypted"
        and isinstance(parsed.get("id"), str)
        and parsed["id"]
        and parsed["data"]
    ):
        return parsed
    return None


def _append_reasoning_detail(details: list[dict[str, JsonValue]], detail: dict[str, JsonValue]) -> None:
    """Pi's ``appendOpenAIReasoningDetail``: consecutive text/summary deltas merge."""
    last = details[-1] if details else None
    kind = detail["type"]
    if last is not None and last["type"] == kind and kind in ("reasoning.text", "reasoning.summary"):
        field = "text" if kind == "reasoning.text" else "summary"
        last[field] = str(last[field]) + str(detail[field])
        # JavaScript drops undefined values when serializing, so only real values are copied.
        if field == "text" and not last.get("signature") and detail.get("signature"):
            last["signature"] = detail["signature"]
        if last.get("id") is None and detail.get("id") is not None:
            last["id"] = detail["id"]
        if not last.get("format") and detail.get("format"):
            last["format"] = detail["format"]
        if last.get("index") is None and detail.get("index") is not None:
            last["index"] = detail["index"]
        return
    details.append(dict(detail))


# ----------------------------------------------------------------------
# Message and tool conversion
# ----------------------------------------------------------------------


def convert_tools(tools: Sequence[ToolDeclaration], compat: dict[str, JsonValue]) -> list[JsonValue]:
    """Pi's ``convertTools`` for tools without constrained sampling."""
    converted: list[JsonValue] = []
    for tool in tools:
        function: dict[str, JsonValue] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": deepcopy(tool.parameters),
        }
        if compat["supportsStrictMode"] is not False:
            function["strict"] = False
        converted.append({"type": "function", "function": function})
    return converted


def _image_url(block: ImageBlock) -> dict[str, JsonValue]:
    return {"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}}


def _convert_assistant(
    message: AssistantMessage, model: ModelInfo, compat: dict[str, JsonValue]
) -> dict[str, JsonValue] | None:
    result: dict[str, JsonValue] = {
        "role": "assistant",
        "content": "" if compat["requiresAssistantAfterToolResult"] else None,
    }
    texts = [
        sanitize_surrogates(b.text) for b in message.content if isinstance(b, TextBlock) and b.text.strip()
    ]
    assistant_text = "".join(texts)
    thinking_blocks = [block for block in message.content if isinstance(block, ThinkingBlock)]
    tool_calls = [block for block in message.content if isinstance(block, ToolCallBlock)]
    signed_details = next(
        (
            details
            for details in (_parse_reasoning_details(block.thinking_signature) for block in thinking_blocks)
            if details is not None
        ),
        None,
    )
    legacy_details: list[JsonValue] = [
        detail
        for detail in (_parse_legacy_encrypted_detail(call.thought_signature) for call in tool_calls)
        if detail is not None
    ]
    preserved_details = signed_details or legacy_details or None

    non_empty_thinking = [block for block in thinking_blocks if block.thinking.strip()]
    if non_empty_thinking:
        if compat["requiresThinkingAsText"]:
            thinking_text = "\n\n".join(sanitize_surrogates(block.thinking) for block in non_empty_thinking)
            result["content"] = [{"type": "text", "text": text} for text in (thinking_text, *texts)]
        else:
            # Assistant content is a plain string, the Chat Completions standard.
            if assistant_text:
                result["content"] = assistant_text
            if preserved_details is None:
                # The first block's signature names the field it streamed in (llama.cpp, DeepSeek).
                signature = non_empty_thinking[0].thinking_signature
                if model.provider == "opencode-go" and signature == "reasoning":
                    signature = "reasoning_content"
                if signature in _REASONING_FIELDS:
                    result[str(signature)] = "\n".join(block.thinking for block in non_empty_thinking)
    elif assistant_text:
        result["content"] = assistant_text

    if tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json_stringify(call.arguments)},
            }
            for call in tool_calls
        ]
    if preserved_details is not None:
        result["reasoning_details"] = preserved_details
    if (
        compat["requiresReasoningContentOnAssistantMessages"]
        and model.reasoning
        and "reasoning_content" not in result
    ):
        result["reasoning_content"] = ""
    # Providers reject assistant messages with neither content nor tool calls.
    if not result["content"] and "tool_calls" not in result:
        return None
    return result


def convert_messages(
    model: ModelInfo, messages: Sequence[Message], compat: dict[str, JsonValue]
) -> list[dict[str, JsonValue]]:
    """Pi's ``convertMessages`` (messages already resolved for mid-conversation system support)."""
    params: list[dict[str, JsonValue]] = []
    transformed = transform_messages(messages, model, normalize_tool_call_id)
    additions = bool(compat["supportsMidConvoSystemMessages"] and compat["supportsMidConvoToolAdditions"])
    _, folded = resolve_transcript_tools(list(messages), "additions" if additions else "none")
    instruction_role = "developer" if model.reasoning and compat["supportsDeveloperRole"] else "system"
    bridge = compat["requiresAssistantAfterToolResult"]
    last_role: str | None = None
    system_index = 0

    index = 0
    while index < len(transformed):
        message = transformed[index]
        if bridge and last_role == "toolResult" and isinstance(message, UserMessage):
            params.append({"role": "assistant", "content": _BRIDGE_TEXT})
        if isinstance(message, SystemMessage):
            added = (message.tools_added or []) if system_index >= folded else []
            system_index += 1
            if added:
                params.append({"role": "system", "tools": convert_tools(added, compat)})
            text = get_system_message_text(message) if index == 0 else render_system_message_update(message)
            if text:
                params.append({"role": instruction_role, "content": sanitize_surrogates(text)})
        elif isinstance(message, UserMessage):
            if isinstance(message.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(message.content)})
            else:
                parts: list[JsonValue] = [
                    {"type": "text", "text": sanitize_surrogates(item.text)}
                    if isinstance(item, TextBlock)
                    else _image_url(item)
                    for item in message.content
                    if not isinstance(item, TextBlock) or item.text
                ]
                if not parts:
                    index += 1
                    continue
                params.append({"role": "user", "content": parts})
        elif isinstance(message, AssistantMessage):
            converted = _convert_assistant(message, model, compat)
            if converted is None:
                index += 1
                continue
            params.append(converted)
        elif isinstance(message, ToolResultMessage):
            images: list[JsonValue] = []
            while index < len(transformed) and isinstance(result := transformed[index], ToolResultMessage):
                text = "\n".join(block.text for block in result.content if isinstance(block, TextBlock))
                has_images = any(isinstance(block, ImageBlock) for block in result.content)
                content = (
                    text if text.strip() else "(see attached image)" if has_images else "(no tool output)"
                )
                tool_message: dict[str, JsonValue] = {
                    "role": "tool",
                    "content": sanitize_surrogates(content),
                    "tool_call_id": result.tool_call_id,
                }
                if compat["requiresToolResultName"] and result.tool_name:
                    tool_message["name"] = result.tool_name
                params.append(tool_message)
                if has_images and "image" in model.input:
                    images.extend(
                        _image_url(block) for block in result.content if isinstance(block, ImageBlock)
                    )
                index += 1
            if images:
                if bridge:
                    params.append({"role": "assistant", "content": _BRIDGE_TEXT})
                caption: JsonValue = {"type": "text", "text": "Attached image(s) from tool result:"}
                params.append({"role": "user", "content": [caption, *images]})
                last_role = "user"
            else:
                last_role = "toolResult"
            continue
        last_role = message.role
        index += 1
    return params


def _has_tool_history(messages: Sequence[Message]) -> bool:
    return any(
        isinstance(message, ToolResultMessage)
        or (
            isinstance(message, AssistantMessage)
            and any(isinstance(b, ToolCallBlock) for b in message.content)
        )
        for message in messages
    )


def _cache_text(message: dict[str, JsonValue], control: dict[str, JsonValue]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": deepcopy(control)}]
        return True
    if isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "text":
                part["cache_control"] = deepcopy(control)
                return True
    return False


def _apply_anthropic_cache_control(
    messages: list[dict[str, JsonValue]], tools: list[JsonValue] | None, control: dict[str, JsonValue]
) -> None:
    """Pi's ``applyAnthropicCacheControl`` for Anthropic models behind OpenAI-compatible gateways."""
    for message in messages:
        if message.get("role") in ("system", "developer"):
            _cache_text(message, control)
            break
    if tools and isinstance(tools[-1], dict):
        tools[-1]["cache_control"] = deepcopy(control)
    for message in reversed(messages):
        if message.get("role") in ("user", "assistant", "tool") and _cache_text(message, control):
            return


def _chat_template_values(
    model: ModelInfo,
    effort: ThinkingLevel | None,
    values: JsonValue,
    thinking_budget: int | None,
) -> dict[str, JsonValue] | None:
    """Pi's ``buildChatTemplateValues``: resolve ``{"$var": ...}`` placeholders."""
    resolved: dict[str, JsonValue] = {}
    level_map = model.thinking_level_map or {}
    for key, value in (values if isinstance(values, dict) else {}).items():
        if not isinstance(value, dict):
            resolved[key] = value
            continue
        if not effort and value.get("omitWhenOff"):
            continue
        if value.get("$var") == "thinking.enabled":
            resolved[key] = bool(effort)
        elif value.get("$var") == "thinking.budget":
            if thinking_budget is not None:
                resolved[key] = thinking_budget
        else:
            level = effort or "off"
            if level not in level_map:
                if effort is not None:
                    resolved[key] = effort
            elif isinstance(level_map[level], str):
                resolved[key] = level_map[level]
    return resolved or None


def _mapped_effort(model: ModelInfo, level: ThinkingLevel) -> JsonValue:
    """``model.thinkingLevelMap?.[level] ?? level``."""
    mapped = (model.thinking_level_map or {}).get(level)
    return mapped if mapped is not None else level


def map_stop_reason(reason: str) -> tuple[str, str | None]:
    """Pi's ``mapStopReason`` for Chat Completions finish reasons."""
    if reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    return "error", f"Provider finish_reason: {reason}"


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def apply_thinking(
    params: dict[str, JsonValue],
    model: ModelInfo,
    compat: dict[str, JsonValue],
    effort: ThinkingLevel | None,
    thinking_budget: int | None,
) -> None:
    """The ``thinkingFormat`` branches of Pi's ``buildParams``."""
    if not model.reasoning:
        return
    thinking_format = compat["thinkingFormat"]
    level_map = model.thinking_level_map or {}
    off_value_is_null = "off" in level_map and level_map["off"] is None
    supports_effort = bool(compat["supportsReasoningEffort"])
    if thinking_format == "zai":
        params["thinking"] = {"type": "enabled", "clear_thinking": False} if effort else {"type": "disabled"}
        if effort and supports_effort:
            mapped = level_map[effort] if effort in level_map else effort
            if isinstance(mapped, str):
                params["reasoning_effort"] = mapped
    elif thinking_format == "qwen":
        params["enable_thinking"] = bool(effort)
        if effort and supports_effort and isinstance(mapped_effort := _mapped_effort(model, effort), str):
            params["reasoning_effort"] = mapped_effort
    elif thinking_format == "qwen-chat-template":
        params["chat_template_kwargs"] = {"enable_thinking": bool(effort), "preserve_thinking": True}
    elif thinking_format == "chat-template":
        kwargs = _chat_template_values(model, effort, compat.get("chatTemplateKwargs"), thinking_budget)
        if kwargs:
            params["chat_template_kwargs"] = kwargs
    elif thinking_format == "baseten":
        args = _chat_template_values(model, effort, compat.get("chatTemplateArgs"), thinking_budget)
        if args:
            params["chat_template_args"] = args
        if supports_effort:
            key = effort or "off"
            value = level_map[key] if key in level_map else effort
            if isinstance(value, str):
                params["reasoning_effort"] = value
    elif thinking_format == "deepseek":
        if effort:
            params["thinking"] = {"type": "enabled"}
        elif not off_value_is_null:
            params["thinking"] = {"type": "disabled"}
        if effort and supports_effort:
            params["reasoning_effort"] = _mapped_effort(model, effort)
    elif thinking_format == "openrouter":
        if effort:
            params["reasoning"] = {"effort": _mapped_effort(model, effort)}
        elif not off_value_is_null:
            params["reasoning"] = {"effort": level_map.get("off") or "none"}
    elif thinking_format == "ant-ling" and effort:
        # Without an effort Pi's condition fails and the generic branches below apply.
        if isinstance(level_map.get(effort), str):
            params["reasoning"] = {"effort": level_map[effort]}
    elif thinking_format == "together":
        params["reasoning"] = {"enabled": bool(effort)}
        if effort and supports_effort:
            params["reasoning_effort"] = _mapped_effort(model, effort)
    elif thinking_format == "string-thinking":
        if effort:
            params["thinking"] = _mapped_effort(model, effort)
        elif not off_value_is_null:
            params["thinking"] = level_map.get("off") or "none"
    elif effort and supports_effort:
        params["reasoning_effort"] = _mapped_effort(model, effort)
    elif not effort and supports_effort and isinstance(level_map.get("off"), str):
        params["reasoning_effort"] = level_map["off"]


# ----------------------------------------------------------------------
# Provider
# ----------------------------------------------------------------------


class OpenAICompatProvider(HTTPProvider):
    endpoint = "/chat/completions"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
        request_settings: OpenAIRequestSettings | None = None,
        cache_retention: CacheRetention | None = None,
        thinking_budgets: dict[str, int] | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, client=client, timeout=timeout)
        self._request_settings = deepcopy(request_settings or OpenAIRequestSettings())
        self._base_url = base_url
        self.cache_retention = cache_retention
        self.thinking_budgets = thinking_budgets

    def headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self._api_key}", "accept": "application/json"}

    def validate_model(self, model: ModelInfo, thinking_level: ThinkingLevel) -> None:
        model.validate_thinking_level(thinking_level)
        if model.api not in ("unknown", "openai-completions") or model.provider == "anthropic":
            raise ValueError("OpenAICompatProvider requires an OpenAI-compatible model API")

    def payload(self, request: ProviderRequest, options: ProviderOptions) -> dict[str, object]:
        """Pi's ``streamSimple`` + ``buildParams``."""
        model = request.model
        self.validate_model(model, options.thinking_level)
        compat = get_compat(model, self._base_url)
        raw_messages = list(request.messages)
        context = resolve_transcript(raw_messages, bool(compat["supportsMidConvoSystemMessages"]))
        clamped = (
            model.clamp_thinking_level(options.thinking_level) if options.thinking_level != "off" else "off"
        )
        effort: ThinkingLevel | None = None if clamped == "off" else clamped
        settings = self._request_settings
        requested = options.max_tokens if options.max_tokens is not None else model.max_tokens
        # Pi estimates the transcript as given, before system messages are collapsed.
        max_tokens = clamp_max_tokens_to_context(model, raw_messages, requested)
        retention = resolve_cache_retention(options.cache_retention or self.cache_retention)

        additions = bool(compat["supportsMidConvoSystemMessages"] and compat["supportsMidConvoToolAdditions"])
        request_tools, _ = resolve_transcript_tools(context, "additions" if additions else "none")
        messages = convert_messages(model, context, compat)
        params: dict[str, JsonValue] = {"model": model.id, "messages": list(messages), "stream": True}
        long_retention = retention == "long" and bool(compat["supportsLongCacheRetention"])
        if ("api.openai.com" in self._base_url and retention != "none") or long_retention:
            key = options.session_id
            if key is not None:
                params["prompt_cache_key"] = key[:_PROMPT_CACHE_KEY_MAX_LENGTH]
        if long_retention:
            params["prompt_cache_retention"] = "24h"
        if compat["supportsUsageInStreaming"] is not False:
            params["stream_options"] = {"include_usage": True}
        if compat["supportsStore"]:
            params["store"] = False
        if max_tokens:
            params["max_tokens" if compat["maxTokensField"] == "max_tokens" else "max_completion_tokens"] = (
                max_tokens
            )
        if settings.temperature is not None:
            params["temperature"] = settings.temperature

        tools: list[JsonValue] | None = None
        if request_tools:
            tools = convert_tools(request_tools, compat)
            params["tools"] = tools
            if compat["zaiToolStream"]:
                params["tool_stream"] = True
        elif _has_tool_history(context):
            # Anthropic behind LiteLLM-style proxies needs tools when history has tool calls.
            params["tools"] = []
        if compat["cacheControlFormat"] == "anthropic" and retention != "none":
            control: dict[str, JsonValue] = {"type": "ephemeral"}
            if long_retention:
                control["ttl"] = "1h"
            _apply_anthropic_cache_control(messages, tools, control)
        if compat.get("vllmPriority") is not None:
            params["priority"] = compat["vllmPriority"]

        budget_field = compat.get("thinkingTokenBudgetField") or (
            "thinking_token_budget" if compat["supportsThinkingTokenBudget"] else None
        )
        thinking_budget: int | None = None
        if effort is not None and model.reasoning:
            ceiling = max_tokens or model.max_tokens
            budget = min(
                thinking_budget_for_level(effort, self.thinking_budgets), max(0, ceiling - _MIN_ANSWER_TOKENS)
            )
            thinking_budget = budget if budget > 0 else None
        apply_thinking(params, model, compat, effort, thinking_budget)
        if budget_field and thinking_budget is not None:
            params[str(budget_field)] = thinking_budget
        if compat.get("openRouterRouting"):
            params["provider"] = deepcopy(compat["openRouterRouting"])
        routing = compat.get("vercelGatewayRouting")
        if isinstance(routing, dict) and (routing.get("only") or routing.get("order")):
            params["providerOptions"] = {
                "gateway": {key: routing[key] for key in ("only", "order") if routing.get(key)}
            }

        # Port extensions from the CLI/settings; Pi's own template values take precedence.
        if settings.seed is not None:
            params["seed"] = settings.seed
        if settings.chat_template_kwargs is not None:
            current = params.get("chat_template_kwargs")
            params["chat_template_kwargs"] = {
                **deepcopy(settings.chat_template_kwargs),
                **(current if isinstance(current, dict) else {}),
            }
        return dict(params)

    def prepare(
        self, request: ProviderRequest, options: ProviderOptions, message: AssistantMessage
    ) -> tuple[dict[str, str], dict[str, object]]:
        headers = self.headers()
        compat = get_compat(request.model, self._base_url)
        session_id = (
            options.session_id
            if resolve_cache_retention(options.cache_retention or self.cache_retention) != "none"
            else None
        )
        if session_id and compat["sendSessionAffinityHeaders"]:
            if compat["sessionAffinityFormat"] == "openrouter":
                headers["x-session-id"] = session_id
            else:
                if compat["sessionAffinityFormat"] == "openai":
                    headers["session_id"] = session_id
                headers["x-client-request-id"] = session_id
                headers["x-session-affinity"] = session_id
        return headers, self.payload(request, options)

    async def decode(
        self, response: httpx.Response, message: AssistantMessage, request: ProviderRequest
    ) -> AsyncIterator[AssistantStreamEvent]:
        """Pi's chunk handling: one text block, one thinking block, tool calls by index or ID."""
        model = request.model
        compat = get_compat(model, self._base_url)
        text_block: TextBlock | None = None
        thinking_block: ThinkingBlock | None = None
        has_finish_reason = False
        calls_by_index: dict[int, ToolCallBlock] = {}
        calls_by_id: dict[str, ToolCallBlock] = {}
        partial_args: dict[int, str] = {}
        streamed_details: list[dict[str, JsonValue]] | None = None

        def position(block: object) -> int:
            return next(index for index, item in enumerate(message.content) if item is block)

        def usage_from(raw: dict[str, JsonValue]) -> None:
            """Pi's ``parseChunkUsage``: cached tokens are reads; writes are counted separately."""
            details = raw.get("prompt_tokens_details")
            details = details if isinstance(details, dict) else {}
            completion = raw.get("completion_tokens_details")
            cache_read = 0
            for value in (
                details.get("cached_tokens"),
                raw.get("prompt_cache_hit_tokens"),
                raw.get("cached_tokens"),
            ):
                if value is not None:
                    cache_read = _int(value)
                    break
            cache_write = _int(details.get("cache_write_tokens"))
            usage = message.usage
            usage.input = max(0, _int(raw.get("prompt_tokens")) - cache_read - cache_write)
            usage.output = _int(raw.get("completion_tokens"))
            usage.cache_read = cache_read
            usage.cache_write = cache_write
            usage.cache_write_1h = None
            nested = completion.get("reasoning_tokens") if isinstance(completion, dict) else None
            if nested is None and model.provider == "sglang":
                # Port extension: SGLang also reports a flat reasoning_tokens count.
                nested = raw.get("reasoning_tokens")
            usage.reasoning = _int(nested)
            usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
            calculate_cost(model.cost, usage)

        async for chunk in iterate_chunks(response):
            if not message.response_id and isinstance(chunk.get("id"), str) and chunk["id"]:
                message.response_id = str(chunk["id"])
            chunk_model = chunk.get("model")
            if (
                isinstance(chunk_model, str)
                and chunk_model
                and chunk_model != model.id
                and not message.response_model
            ):
                message.response_model = chunk_model
            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, dict):
                usage_from(chunk_usage)
            choices = chunk.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            if not isinstance(choice, dict):
                continue
            choice_usage = choice.get("usage")
            if not isinstance(chunk_usage, dict) and isinstance(choice_usage, dict):
                usage_from(choice_usage)
            finish = choice.get("finish_reason")
            if isinstance(finish, str) and finish:
                message.raw_stop_reason = finish
                mapped, error_message = map_stop_reason(finish)
                message.stop_reason = mapped  # type: ignore[assignment]
                if error_message:
                    message.error_message = error_message
                has_finish_reason = True
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            # Reasoning first: some proxies put reasoning and answer in one delta.
            field = next(
                (name for name in _REASONING_FIELDS if isinstance(delta.get(name), str) and delta[name]), None
            )
            if field is not None:
                if thinking_block is None:
                    signature = (
                        "reasoning_content"
                        if model.provider == "opencode-go" and field == "reasoning"
                        else field
                    )
                    thinking_block = ThinkingBlock(thinking="", thinking_signature=signature)
                    message.content.append(thinking_block)
                    yield ThinkingStart(content_index=position(thinking_block), partial=deepcopy(message))
                text = str(delta[field])
                thinking_block.thinking += text
                yield ThinkingDelta(
                    content_index=position(thinking_block), delta=text, partial=deepcopy(message)
                )
            details = delta.get("reasoning_details")
            for detail in details if isinstance(details, list) else []:
                if not isinstance(detail, dict) or not _valid_reasoning_detail(detail):
                    continue
                if thinking_block is None:
                    thinking_block = ThinkingBlock(thinking="", thinking_signature="")
                    message.content.append(thinking_block)
                    yield ThinkingStart(content_index=position(thinking_block), partial=deepcopy(message))
                if streamed_details is None:
                    streamed_details = []
                _append_reasoning_detail(streamed_details, detail)
            content = delta.get("content")
            if isinstance(content, str) and content:
                if text_block is None:
                    text_block = TextBlock(text="")
                    message.content.append(text_block)
                    yield TextStart(content_index=position(text_block), partial=deepcopy(message))
                text_block.text += content
                yield TextDelta(content_index=position(text_block), delta=content, partial=deepcopy(message))
            tool_calls = delta.get("tool_calls")
            for item in tool_calls if isinstance(tool_calls, list) else []:
                if not isinstance(item, dict):
                    continue
                raw_index = item.get("index")
                stream_index = (
                    raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else None
                )
                function = item.get("function")
                function = function if isinstance(function, dict) else {}
                name = function.get("name") if isinstance(function.get("name"), str) else ""
                call_id = item.get("id") if isinstance(item.get("id"), str) else ""
                call = calls_by_index.get(stream_index) if stream_index is not None else None
                if call is None and call_id:
                    call = calls_by_id.get(str(call_id))
                if call is None:
                    call = ToolCallBlock(id=str(call_id), name=str(name), arguments={})
                    message.content.append(call)
                    partial_args[id(call)] = ""
                    yield ToolCallStart(content_index=position(call), partial=deepcopy(message))
                if stream_index is not None:
                    calls_by_index.setdefault(stream_index, call)
                if call_id:
                    calls_by_id[str(call_id)] = call
                    if not call.id:
                        call.id = str(call_id)
                if not call.name and name:
                    call.name = str(name)
                fragment = ""
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    fragment = arguments
                    partial_args[id(call)] += arguments
                    call.arguments = parse_streaming_json(partial_args[id(call)])
                yield ToolCallDelta(content_index=position(call), delta=fragment, partial=deepcopy(message))

        for index, block in enumerate(message.content):
            if isinstance(block, TextBlock):
                yield TextEnd(content_index=index, content=block.text, partial=deepcopy(message))
            elif isinstance(block, ThinkingBlock):
                if streamed_details is not None:
                    block.thinking_signature = json_stringify(streamed_details)
                yield ThinkingEnd(content_index=index, partial=deepcopy(message))
            elif isinstance(block, ToolCallBlock):
                block.arguments, block.arguments_error = parse_tool_call_arguments(
                    partial_args.get(id(block), "")
                )
                yield ToolCallEnd(content_index=index, tool_call=deepcopy(block), partial=deepcopy(message))

        if not has_finish_reason and not compat["supportsFinishReason"]:
            has_tool = any(isinstance(block, ToolCallBlock) for block in message.content)
            message.stop_reason = "toolUse" if has_tool else "stop"
        if message.stop_reason == "error":
            raise ValueError(message.error_message or "Provider returned an error stop reason")
        if (compat["supportsFinishReason"] and not has_finish_reason) or message.stop_reason not in (
            "stop",
            "length",
            "toolUse",
        ):
            raise ValueError("Stream ended without finish_reason")
        yield StreamDone(reason=message.stop_reason, message=deepcopy(message))


async def iterate_chunks(response: httpx.Response) -> AsyncIterator[dict[str, JsonValue]]:
    """The OpenAI SDK's SSE stream: JSON ``data`` until ``[DONE]``; error payloads raise."""
    async for name, data, _ in iterate_sse_messages(response):
        if data.startswith("[DONE]"):
            return
        if name not in (None, "error"):
            continue
        try:
            chunk = json.loads(data)
        except ValueError as error:
            raise ValueError(f"Could not parse message into JSON: {data}") from error
        if isinstance(chunk, dict) and chunk.get("error") is not None:
            raise ValueError(_format_stream_error(chunk["error"]))
        if name == "error":
            raise ValueError(_format_stream_error(chunk))
        if isinstance(chunk, dict):
            yield chunk


def _format_stream_error(error: object) -> str:
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return str(error["message"])
    return json.dumps(error)
