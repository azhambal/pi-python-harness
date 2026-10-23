"""Overflow and retry classification, translated from Pi's ``ai/src/utils/overflow.ts`` and ``retry.ts``."""

import re

from .types import AssistantMessage

#: Pi's ``OVERFLOW_PATTERNS``: input over the context window or the request size limit.
_OVERFLOW_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt (?:is )?too long",  # Anthropic and z.ai token overflow
        r"prompt exceeds max length",  # z.ai CN endpoint token overflow
        r"request_too_large",  # Anthropic request byte-size overflow (HTTP 413)
        r"input is too long for requested model",  # Amazon Bedrock
        r"exceeds the context window",  # OpenAI (Completions & Responses API)
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",  # LiteLLM
        r"input token count.*exceeds the maximum",  # Google (Gemini)
        r"maximum prompt length is \d+",  # xAI (Grok)
        r"reduce the length of the messages",  # Groq
        r"maximum context length is \d+ tokens",  # OpenRouter (most backends)
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",  # OpenRouter/Poolside
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",  # Together AI
        r"exceeds the limit of \d+",  # GitHub Copilot
        r"exceeds the available context size",  # llama.cpp server
        r"greater than the context length",  # LM Studio
        r"context window exceeds limit",  # MiniMax
        r"exceeded model token limit",  # Kimi For Coding
        r"too large for model with \d+ maximum context length",  # Mistral
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",  # DS4 server
        r"model_context_window_exceeded",  # z.ai non-standard finish_reason surfaced as error text
        r"prompt too long; exceeded (?:max )?context length",  # Ollama explicit overflow error
        r"range of input length should be",  # DashScope / Qwen Token Plan
        r"context[_ ]length[_ ]exceeded",  # Generic fallback
        r"too many tokens",  # Generic fallback
        r"token limit exceeded",  # Generic fallback
        # HTTP 413 means request-size overflow regardless of provider or body text.
        r"^(?:[\w -]+ API error \(413\):|413(?:\s|:|$))",
    )
)
_CEREBRAS_BODYLESS_OVERFLOW = re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE)
_NON_OVERFLOW_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (r"^(Throttling error|Service unavailable):", r"rate limit", r"too many requests")
)

#: Pi's ``NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN``: account limits, not transient throttles.
_NON_RETRYABLE = re.compile(
    "|".join(
        (
            "GoUsageLimitError",
            "FreeUsageLimitError",
            "Monthly usage limit reached",
            "available balance",
            "insufficient_quota",
            "out of budget",
            "quota exceeded",
            "billing",
            "subscription_sharing_usage_limit_exceeded",
            "subscription_sharing_workplace_individual_usage_limit_exceeded",
        )
    ),
    re.IGNORECASE,
)
#: Pi's ``RETRYABLE_PROVIDER_ERROR_PATTERN``: transient provider and transport failures.
_RETRYABLE = re.compile(
    "|".join(
        (
            "overloaded",
            "server_busy",
            "servers are currently busy",
            "currently experiencing high demand",
            "model is at capacity",
            "rate.?limit",
            "too many requests",
            "(?<![a-z0-9])(?:429|500|502|503|504|520|524)(?![a-z0-9])",
            "service.?(?:is.?)?(?:temporarily.?)?unavailable",
            "server.?error",
            "internal.?error",
            "provider.?returned.?(?:an.?)?error",
            "exceeded request buffer limit while retrying upstream",
            "unable to forward upstream",
            "upstream.?request.?failed",
            "upstream_stream_read_error",
            "stream was interrupted",
            "http/?2 stream failed",
            "network.?error",
            "connection.?error",
            "connection.?refused",
            "connection.?lost",
            "other side closed",
            "fetch failed",
            r"request to \S+ failed, reason:",
            "getaddrinfo",
            "ENOTFOUND",
            "EAI_AGAIN",
            "upstream.?connect",
            "reset before headers",
            "socket hang up",
            "socket connection was closed",
            "timed? out",
            "timeout",
            "terminated",
            "websocket.?closed",
            "websocket.?error",
            "ended without",
            "stream ended before message_stop",
            "stream ended before a terminal response event",
            "http2 request did not get a response",
            "stream disconnected before completion",
            "pending stream has been canceled",
            "retry delay",
            "you can retry your request",
            "try your request again",
            "please retry your request",
            "ResourceExhausted",
            "subscription_sharing_usage_unavailable",
            "subscription_sharing_user_unavailable",
        )
    ),
    re.IGNORECASE,
)


def is_context_overflow(message: AssistantMessage, context_window: int = 0) -> bool:
    """Pi's ``isContextOverflow``: an overflow error, or silent overflow shown by usage."""
    text = message.error_message
    if message.stop_reason == "error" and text:
        if not any(pattern.search(text) for pattern in _NON_OVERFLOW_PATTERNS):
            if any(pattern.search(text) for pattern in _OVERFLOW_PATTERNS):
                return True
            if message.provider == "cerebras" and _CEREBRAS_BODYLESS_OVERFLOW.search(text):
                return True
    input_tokens = message.usage.input + message.usage.cache_read
    # Silent overflow (z.ai style): a successful response whose input exceeds the window.
    if context_window and message.stop_reason == "stop" and input_tokens > context_window:
        return True
    # Length-stop overflow (Xiaomi MiMo style): truncated input fills the window, no output.
    return bool(
        context_window
        and message.stop_reason == "length"
        and message.usage.output == 0
        and input_tokens >= context_window * 0.99
    )


def is_recoverable_length(message: AssistantMessage, desired_max_output: int) -> bool:
    """Pi's ``isRecoverableLength``: a length stop below the model's own output limit."""
    return (
        message.stop_reason == "length"
        and desired_max_output > 0
        and message.usage.output < desired_max_output
    )


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """Pi's ``isRetryableAssistantError``: a transient provider or transport error."""
    if message.stop_reason != "error" or not message.error_message:
        return False
    if _NON_RETRYABLE.search(message.error_message):
        return False
    return bool(_RETRYABLE.search(message.error_message))


def is_retryable_error(message: AssistantMessage, context_window: int = 0) -> bool:
    """Pi's session ``_isRetryableError``: context overflow is compaction's, not retry's."""
    if is_context_overflow(message, context_window):
        return False
    return is_retryable_assistant_error(message)
