"""OpenAI-compatible Chat Completions, including DeepSeek and SGLang reasoning replay."""

import json
import math
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass

import httpx

from pi_agent.context import output_token_limit
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
    JsonValue,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    content_text,
    get_current_system_prompt,
)

from ._http import MAX_EVENT_CHARS, HTTPProvider, sse_data, strict_json
from ._normalize import normalize_messages


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
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, client=client, timeout=timeout)
        self._request_settings = deepcopy(request_settings or OpenAIRequestSettings())

    def validate_model(self, model: ModelInfo, thinking_level: ThinkingLevel) -> None:
        model.validate_thinking_level(thinking_level)
        if model.api not in ("unknown", "openai-completions") or model.provider == "anthropic":
            raise ValueError("OpenAICompatProvider requires an OpenAI-compatible model API")

    def payload(self, request: ProviderRequest, options: ProviderOptions) -> dict[str, object]:
        self.validate_model(request.model, options.thinking_level)
        history = normalize_messages(request.messages, request.model)
        messages: list[dict[str, object]] = []
        system = get_current_system_prompt(history)
        if system:
            messages.append({"role": "system", "content": system})
        for message in history:
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, UserMessage):
                messages.append({"role": "user", "content": content_text(message.content)})
            elif isinstance(message, ToolResultMessage):
                text = content_text(message.content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": text if text.strip() else "(no tool output)",
                    }
                )
            elif isinstance(message, AssistantMessage):
                data: dict[str, object] = {
                    "role": "assistant",
                    "content": content_text(message.content) or None,
                }
                calls = [block for block in message.content if isinstance(block, ToolCallBlock)]
                if calls:
                    data["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False, allow_nan=False),
                            },
                        }
                        for call in calls
                    ]
                if request.model.provider in ("deepseek", "sglang"):
                    data["reasoning_content"] = "\n".join(
                        block.thinking for block in message.content if isinstance(block, ThinkingBlock)
                    )
                messages.append(data)
        payload: dict[str, object] = {
            "model": request.model.id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            payload["tools"] = [{"type": "function", "function": tool.to_json()} for tool in request.tools]
        payload["max_tokens"] = output_token_limit(history, request.model, options.max_tokens, request.tools)
        if request.model.provider == "deepseek":
            enabled = options.thinking_level != "off"
            payload["thinking"] = {"type": "enabled" if enabled else "disabled"}
            if enabled:
                payload["reasoning_effort"] = request.model.thinking_effort(
                    options.thinking_level,
                    {
                        "minimal": "low",
                        "low": "low",
                        "medium": "high",
                        "high": "high",
                        "xhigh": "max",
                        "max": "max",
                    }[options.thinking_level],
                )
        elif request.model.reasoning and (
            options.thinking_level != "off" or request.model.thinking_level_map is not None
        ):
            payload["reasoning_effort"] = request.model.thinking_effort(
                options.thinking_level,
                {"xhigh": "high", "max": "high"}.get(options.thinking_level, options.thinking_level),
            )
        settings = self._request_settings
        if settings.temperature is not None:
            payload["temperature"] = settings.temperature
        if settings.seed is not None:
            payload["seed"] = settings.seed
        template_kwargs = deepcopy(settings.chat_template_kwargs or {})
        if request.model.provider == "sglang":
            template_kwargs["enable_thinking"] = options.thinking_level != "off"
        if template_kwargs or settings.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = template_kwargs
        return payload

    async def decode(
        self, response: httpx.Response, message: AssistantMessage
    ) -> AsyncIterator[AssistantStreamEvent]:
        text_index: int | None = None
        thinking_index: int | None = None
        calls: dict[int, tuple[int, ToolCallBlock]] = {}
        arguments: dict[int, str] = {}
        finish: str | None = None
        complete = False
        async for data in sse_data(response):
            if data == "[DONE]":
                complete = True
                break
            chunk = strict_json(data)
            if not isinstance(chunk, dict):
                raise ValueError("Chat completion chunk must be an object")
            if "error" in chunk:
                raise ValueError(f"Provider stream error: {json.dumps(chunk['error'])}")
            if isinstance(chunk.get("id"), str):
                message.response_id = chunk["id"]
            if isinstance(chunk.get("model"), str):
                message.response_model = chunk["model"]
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                input_tokens = token_count(usage.get("prompt_tokens", 0))
                details = usage.get("prompt_tokens_details") or {}
                if not isinstance(details, dict):
                    raise ValueError("Invalid prompt token details")
                cached = token_count(usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0)))
                if cached > input_tokens:
                    raise ValueError("Cached tokens exceed prompt tokens")
                message.usage.input = input_tokens - cached
                message.usage.cache_read = cached
                message.usage.output = token_count(usage.get("completion_tokens", 0))
                message.usage.total_tokens = token_count(
                    usage.get("total_tokens", input_tokens + message.usage.output)
                )
                completion_details = usage.get("completion_tokens_details")
                if completion_details is not None and not isinstance(completion_details, dict):
                    raise ValueError("Invalid completion token details")
                if isinstance(completion_details, dict) and "reasoning_tokens" in completion_details:
                    message.usage.reasoning = token_count(completion_details["reasoning_tokens"])
                elif "reasoning_tokens" in usage:
                    message.usage.reasoning = token_count(usage["reasoning_tokens"])
            choices = chunk.get("choices")
            if not isinstance(choices, list):
                raise ValueError("Chat completion chunk requires a choices array")
            if not choices:
                continue  # usage-only trailer
            if len(choices) != 1 or not isinstance(choices[0], dict) or choices[0].get("index", 0) != 0:
                raise ValueError("Expected exactly one completion choice with index 0")
            choice = choices[0]
            if finish is not None:
                raise ValueError("Completion data after finish_reason")
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise ValueError("Completion delta must be an object")
            reasoning = delta.get("reasoning_content")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise ValueError("Reasoning delta must be text")
                if thinking_index is None:
                    thinking_index = len(message.content)
                    message.content.append(ThinkingBlock())
                    yield ThinkingStart(content_index=thinking_index, partial=deepcopy(message))
                thinking = message.content[thinking_index]
                assert isinstance(thinking, ThinkingBlock)
                thinking.thinking += reasoning
                yield ThinkingDelta(content_index=thinking_index, delta=reasoning, partial=deepcopy(message))
            text = delta.get("content")
            if text is not None:
                if not isinstance(text, str):
                    raise ValueError("Content delta must be text")
                if text_index is None:
                    text_index = len(message.content)
                    message.content.append(TextBlock())
                    yield TextStart(content_index=text_index, partial=deepcopy(message))
                block = message.content[text_index]
                assert isinstance(block, TextBlock)
                block.text += text
                yield TextDelta(content_index=text_index, delta=text, partial=deepcopy(message))
            tool_deltas = delta.get("tool_calls", [])
            if not isinstance(tool_deltas, list):
                raise ValueError("tool_calls delta must be an array")
            for item in tool_deltas:
                if not isinstance(item, dict):
                    raise ValueError("Tool delta must be an object")
                index = token_count(item.get("index"))
                if index not in calls:
                    call = ToolCallBlock()
                    calls[index] = (len(message.content), call)
                    arguments[index] = ""
                    message.content.append(call)
                    yield ToolCallStart(content_index=calls[index][0], partial=deepcopy(message))
                content_index, call = calls[index]
                function = item.get("function", {})
                if not isinstance(function, dict):
                    raise ValueError("Tool function must be an object")
                value = item.get("id")
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError("Tool ID must be text")
                    if value != call.id:
                        call.id += value
                name = function.get("name")
                if name is not None:
                    if not isinstance(name, str):
                        raise ValueError("Tool name must be text")
                    if name != call.name:
                        call.name += name
                fragment = function.get("arguments", "")
                if not isinstance(fragment, str):
                    raise ValueError("Tool arguments delta must be text")
                arguments[index] += fragment
                if len(arguments[index]) > MAX_EVENT_CHARS:
                    raise ValueError("Tool arguments exceed 4 MiB")
                yield ToolCallDelta(content_index=content_index, delta=fragment, partial=deepcopy(message))
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
                if finish not in ("stop", "length", "tool_calls"):
                    raise ValueError(f"Provider finish_reason: {finish}")
                message.raw_stop_reason = finish
        if not complete or finish is None:
            raise ValueError("Provider stream ended without a terminal event")
        for index, (content_index, call) in calls.items():
            try:
                parsed = strict_json(arguments[index] or "{}")
            except ValueError:
                if finish != "length":
                    raise
                parsed = {}
                if message.provider_data is None:
                    message.provider_data = {}
                message.provider_data[f"incomplete_tool_arguments_{content_index}"] = arguments[index]
            if not isinstance(parsed, dict) or not call.id or not call.name:
                raise ValueError("Tool call requires an ID, name and JSON object arguments")
            call.arguments = parsed
            yield ToolCallEnd(
                content_index=content_index, tool_call=deepcopy(call), partial=deepcopy(message)
            )
        if text_index is not None:
            block = message.content[text_index]
            assert isinstance(block, TextBlock)
            yield TextEnd(content_index=text_index, content=block.text, partial=deepcopy(message))
        if thinking_index is not None:
            yield ThinkingEnd(content_index=thinking_index, partial=deepcopy(message))
        if finish == "tool_calls":
            if not calls:
                raise ValueError("tool_calls finish_reason without any tool calls")
            message.stop_reason = "toolUse"
        elif finish == "length":
            message.stop_reason = "length"
        else:
            if calls:
                raise ValueError("Tool calls require a tool_calls finish_reason")
            message.stop_reason = "stop"
        yield StreamDone(reason=message.stop_reason, message=deepcopy(message))
