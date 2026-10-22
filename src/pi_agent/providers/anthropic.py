"""Anthropic Messages streaming with signed thinking and cache accounting."""

from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Literal

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

from ._http import MAX_EVENT_CHARS, HTTPProvider, sse_data, strict_json, text_field
from ._normalize import normalize_messages
from .openai_compat import token_count


class AnthropicProvider(HTTPProvider):
    endpoint = "/v1/messages"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
        thinking_mode: Literal["adaptive", "budget"] = "adaptive",
        cache: bool = True,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, client=client, timeout=timeout)
        self.thinking_mode = thinking_mode
        self.cache = cache

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Accept": "text/event-stream",
        }

    def validate_model(self, model: ModelInfo, thinking_level: ThinkingLevel) -> None:
        model.validate_thinking_level(thinking_level)
        if model.api not in ("unknown", "anthropic-messages") or model.provider in ("openai", "deepseek"):
            raise ValueError("AnthropicProvider requires an Anthropic Messages model API")
        if self.thinking_mode == "budget" and thinking_level != "off" and (model.max_tokens or 8192) <= 1024:
            raise ValueError("Budget thinking requires max_tokens > 1024")

    def payload(self, request: ProviderRequest, options: ProviderOptions) -> dict[str, object]:
        self.validate_model(request.model, options.thinking_level)
        history = normalize_messages(request.messages, request.model)
        messages: list[dict[str, object]] = []
        for message in history:
            blocks: list[dict[str, object]] = []
            role = "user"
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, UserMessage):
                blocks.append({"type": "text", "text": content_text(message.content)})
            elif isinstance(message, ToolResultMessage):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": content_text(message.content),
                        "is_error": message.is_error,
                    }
                )
            elif isinstance(message, AssistantMessage):
                role = "assistant"
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if block.text:
                            blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolCallBlock):
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.arguments,
                            }
                        )
                    elif isinstance(block, ThinkingBlock):
                        if block.redacted:
                            data = (block.provider_data or {}).get("data")
                            if not isinstance(data, str):
                                raise ValueError("Redacted thinking lacks replay data")
                            blocks.append({"type": "redacted_thinking", "data": data})
                        elif block.thinking_signature:
                            blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": block.thinking,
                                    "signature": block.thinking_signature,
                                }
                            )
                        elif block.thinking.strip():
                            blocks.append({"type": "text", "text": block.thinking})
            if not blocks:
                continue
            if messages and messages[-1]["role"] == role:
                previous = messages[-1]["content"]
                assert isinstance(previous, list)
                previous.extend(blocks)
            else:
                messages.append({"role": role, "content": blocks})
        cap = output_token_limit(history, request.model, options.max_tokens, request.tools)
        payload: dict[str, object] = {
            "model": request.model.id,
            "messages": messages,
            "max_tokens": cap,
            "stream": True,
        }
        if self.cache:
            for item in reversed(messages):
                if item["role"] == "user":
                    content = item["content"]
                    assert isinstance(content, list)
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                    break
        system = get_current_system_prompt(history)
        if system:
            system_block: dict[str, object] = {"type": "text", "text": system}
            if self.cache:
                system_block["cache_control"] = {"type": "ephemeral"}
            payload["system"] = [system_block]
        if request.tools:
            tools: list[dict[str, object]] = [
                {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
                for tool in request.tools
            ]
            if self.cache:
                tools[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = tools
        if options.thinking_level != "off":
            if self.thinking_mode == "adaptive":
                payload["thinking"] = {"type": "adaptive"}
                payload["output_config"] = {
                    "effort": request.model.thinking_effort(
                        options.thinking_level,
                        {
                            "minimal": "low",
                            "low": "low",
                            "medium": "medium",
                            "high": "high",
                            "xhigh": "xhigh",
                            "max": "max",
                        }[options.thinking_level],
                    )
                }
            else:
                budget = {
                    "minimal": 1024,
                    "low": 2048,
                    "medium": 8192,
                    "high": 16384,
                    "xhigh": 32768,
                    "max": 32768,
                }[options.thinking_level]
                budget = min(budget, cap - 1)
                if budget < 1024:
                    raise ValueError("Budget thinking requires max_tokens > 1024")
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            payload["thinking"] = {"type": "disabled"}
        return payload

    def _usage(self, message: AssistantMessage, data: object) -> None:
        if not isinstance(data, dict):
            return
        for key, target in (
            ("input_tokens", "input"),
            ("output_tokens", "output"),
            ("cache_read_input_tokens", "cache_read"),
            ("cache_creation_input_tokens", "cache_write"),
        ):
            if key in data:
                setattr(message.usage, target, token_count(data[key]))
        creation = data.get("cache_creation")
        if isinstance(creation, dict) and "ephemeral_1h_input_tokens" in creation:
            message.usage.cache_write_1h = token_count(creation["ephemeral_1h_input_tokens"])
        message.usage.total_tokens = (
            message.usage.input + message.usage.output + message.usage.cache_read + message.usage.cache_write
        )

    async def decode(
        self, response: httpx.Response, message: AssistantMessage
    ) -> AsyncIterator[AssistantStreamEvent]:
        started = False
        stopped = False
        open_blocks: set[int] = set()
        arguments: dict[int, str] = {}
        invalid_arguments: set[int] = set()
        finish: str | None = None
        async for frame in sse_data(response):
            event = strict_json(frame)
            if not isinstance(event, dict):
                raise ValueError("Anthropic event must be an object")
            kind = event.get("type")
            if kind == "error":
                raise ValueError(f"Provider stream error: {event.get('error')}")
            if kind == "message_start":
                if started:
                    raise ValueError("Duplicate message_start")
                started = True
                initial = event.get("message")
                if not isinstance(initial, dict):
                    raise ValueError("message_start requires a message")
                message.response_id = text_field(initial, "id") or None
                message.response_model = text_field(initial, "model") or None
                self._usage(message, initial.get("usage"))
            elif kind in ("content_block_start", "content_block_delta", "content_block_stop"):
                if not started:
                    raise ValueError("Content before message_start")
                index = token_count(event.get("index"))
                if kind == "content_block_start":
                    if index != len(message.content):
                        raise ValueError("Content block indexes must be consecutive")
                    native = event.get("content_block")
                    if not isinstance(native, dict):
                        raise ValueError("Invalid content block")
                    block_type = native.get("type")
                    if block_type == "text":
                        message.content.append(TextBlock(text=text_field(native, "text")))
                        yield TextStart(content_index=index, partial=deepcopy(message))
                    elif block_type in ("thinking", "redacted_thinking"):
                        message.content.append(
                            ThinkingBlock(
                                thinking=text_field(native, "thinking"),
                                thinking_signature=text_field(native, "signature") or None,
                                redacted=block_type == "redacted_thinking",
                                provider_data={"data": text_field(native, "data")}
                                if block_type == "redacted_thinking"
                                else None,
                            )
                        )
                        yield ThinkingStart(content_index=index, partial=deepcopy(message))
                    elif block_type == "tool_use":
                        initial_input = native.get("input", {})
                        if not isinstance(initial_input, dict):
                            raise ValueError("Tool input must be an object")
                        message.content.append(
                            ToolCallBlock(
                                id=text_field(native, "id"),
                                name=text_field(native, "name"),
                                arguments=initial_input,
                            )
                        )
                        arguments[index] = ""
                        yield ToolCallStart(content_index=index, partial=deepcopy(message))
                    else:
                        raise ValueError(f"Unsupported Anthropic content block: {block_type}")
                    open_blocks.add(index)
                else:
                    if index not in open_blocks:
                        raise ValueError("Delta/stop refers to a closed or missing block")
                    block = message.content[index]
                    if kind == "content_block_delta":
                        delta = event.get("delta")
                        if not isinstance(delta, dict):
                            raise ValueError("Invalid content delta")
                        delta_type = delta.get("type")
                        if delta_type == "text_delta" and isinstance(block, TextBlock):
                            text = delta["text"]
                            if not isinstance(text, str):
                                raise ValueError("Text delta must be text")
                            block.text += text
                            yield TextDelta(content_index=index, delta=text, partial=deepcopy(message))
                        elif delta_type == "thinking_delta" and isinstance(block, ThinkingBlock):
                            text = delta["thinking"]
                            if not isinstance(text, str):
                                raise ValueError("Thinking delta must be text")
                            block.thinking += text
                            yield ThinkingDelta(content_index=index, delta=text, partial=deepcopy(message))
                        elif delta_type == "signature_delta" and isinstance(block, ThinkingBlock):
                            signature = delta["signature"]
                            if not isinstance(signature, str):
                                raise ValueError("Thinking signature must be text")
                            block.thinking_signature = (block.thinking_signature or "") + signature
                        elif delta_type == "input_json_delta" and isinstance(block, ToolCallBlock):
                            fragment = delta["partial_json"]
                            if not isinstance(fragment, str):
                                raise ValueError("Tool input delta must be text")
                            arguments[index] += fragment
                            if len(arguments[index]) > MAX_EVENT_CHARS:
                                raise ValueError("Tool arguments exceed 4 MiB")
                            yield ToolCallDelta(
                                content_index=index, delta=fragment, partial=deepcopy(message)
                            )
                        else:
                            raise ValueError(f"Unexpected content delta: {delta_type}")
                    else:
                        open_blocks.remove(index)
                        if isinstance(block, TextBlock):
                            yield TextEnd(content_index=index, content=block.text, partial=deepcopy(message))
                        elif isinstance(block, ThinkingBlock):
                            if not block.redacted and not block.thinking_signature:
                                raise ValueError("Thinking block requires a signature")
                            yield ThinkingEnd(content_index=index, partial=deepcopy(message))
                        else:
                            if arguments[index]:
                                try:
                                    parsed = strict_json(arguments[index])
                                    if not isinstance(parsed, dict):
                                        raise ValueError("Tool input must be a JSON object")
                                    block.arguments = parsed
                                except ValueError:
                                    invalid_arguments.add(index)
                            if not block.id or not block.name:
                                raise ValueError("Tool call requires ID and name")
                            yield ToolCallEnd(
                                content_index=index, tool_call=deepcopy(block), partial=deepcopy(message)
                            )
            elif kind == "message_delta":
                if not started:
                    raise ValueError("message_delta before message_start")
                delta = event.get("delta")
                if not isinstance(delta, dict):
                    raise ValueError("Invalid message delta")
                if delta.get("stop_reason") is not None:
                    finish = delta["stop_reason"]
                self._usage(message, event.get("usage"))
            elif kind == "message_stop":
                stopped = True
                break
            # Ignore ping and future event types as required by the API's versioning policy.
        if not started or not stopped or finish is None or open_blocks:
            raise ValueError("Provider stream ended without a terminal event")
        if invalid_arguments:
            if finish != "max_tokens":
                raise ValueError("Tool input must be a complete JSON object")
            message.provider_data = {
                f"incomplete_tool_arguments_{index}": arguments[index] for index in invalid_arguments
            }
        message.raw_stop_reason = finish
        if finish in ("end_turn", "stop_sequence"):
            if any(isinstance(block, ToolCallBlock) for block in message.content):
                raise ValueError("Tool calls require a tool_use stop reason")
            message.stop_reason = "stop"
        elif finish == "max_tokens":
            message.stop_reason = "length"
        elif finish == "tool_use":
            if not any(isinstance(block, ToolCallBlock) for block in message.content):
                raise ValueError("tool_use stop reason without calls")
            message.stop_reason = "toolUse"
        else:
            raise ValueError(f"Provider stop_reason: {finish}")
        yield StreamDone(reason=message.stop_reason, message=deepcopy(message))
