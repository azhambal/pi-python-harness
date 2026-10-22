"""Pure retry/overflow classification, based on Pi ai/src/utils/overflow.ts.

Only failed conversational requests are recovered here. Completed answers and
ordinary output-length stops are left to the caller, preserving this port's policy.
"""

import re

from .types import AssistantMessage

_NON_RETRYABLE = re.compile(
    r"insufficient_quota|out of budget|quota exceeded|billing|usage limit|subscription_sharing_usage_limit",
    re.IGNORECASE,
)
_RETRYABLE = re.compile(
    r"overloaded|server_busy|servers are currently busy|currently experiencing high demand|"
    r"model is at capacity|rate.?limit|too many requests|\b(?:429|50[0234]|52[04])\b|"
    r"service.?(?:is.?)?(?:temporarily.?)?unavailable|server.?error|internal.?error|"
    r"provider.?returned.?(?:an.?)?error|unable to forward upstream|upstream.?request.?failed|"
    r"upstream_stream_read_error|stream was interrupted|http/?2 stream failed|"
    r"stream disconnected before completion|network.?error|connection.?error|"
    r"connection.?refused|connection.?lost|fetch failed|timed? out|timeout|terminated|"
    r"socket hang up|please retry|you can retry",
    re.IGNORECASE,
)
_OVERFLOW = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt (?:is )?too long",
        r"prompt exceeds max length",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds the context window",
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        r"input token count.*exceeds the maximum",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"too large for model with \d+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"model_context_window_exceeded",
        r"prompt too long; exceeded (?:max )?context length",
        r"range of input length should be",
        r"context[_ ]length[_ ]exceeded",
        r"too many tokens",
        r"token limit exceeded",
        # Existing port formulations not covered by upstream's templates.
        r"input too large",
    )
)
_NON_OVERFLOW = re.compile(
    r"^(?:Throttling error|Service unavailable):|rate.?limit|too many requests", re.IGNORECASE
)
_BODYLESS = re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE)


def is_context_overflow(message: AssistantMessage, context_window: int = 0) -> bool:
    if message.stop_reason != "error":
        return False
    text = message.error_message or ""
    if _NON_OVERFLOW.search(text):
        return False
    if any(pattern.search(text) for pattern in _OVERFLOW):
        return True
    if message.provider == "cerebras" and _BODYLESS.search(text):
        return True
    # Input usage is the relevant limit; output tokens alone must not trigger compaction.
    return context_window > 0 and message.usage.input + message.usage.cache_read > context_window


def is_retryable_error(message: AssistantMessage, context_window: int = 0) -> bool:
    if message.stop_reason != "error" or not message.error_message:
        return False
    if is_context_overflow(message, context_window) or _NON_RETRYABLE.search(message.error_message):
        return False
    # A prematurely exhausted provider iterator is an explicit transport retry case.
    return message.error_message == "Provider stream ended without a terminal event" or bool(
        _RETRYABLE.search(message.error_message)
    )
