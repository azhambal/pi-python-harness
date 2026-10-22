import asyncio

import pytest

from pi_agent import AgentConfig, create_agent_session
from pi_agent.error_policy import is_context_overflow as _is_context_overflow
from pi_agent.error_policy import is_retryable_error as _is_retryable_error
from pi_agent.events import AutoRetryStart, CompactionStart
from pi_agent.providers import FauxProvider
from pi_agent.types import AssistantMessage, TextBlock, Usage

OVERFLOW_ERRORS = [
    "prompt is too long: 213462 tokens > 200000 maximum",
    '413 {"type":"request_too_large"}',
    "Your input exceeds the context window of this model",
    "Requested 15000 tokens exceeds the model's maximum context length of 8192 tokens",
    "Input length (265330) exceeds model's maximum context length (262144)",
    "The input token count exceeds the maximum number of tokens allowed",
    "This model's maximum prompt length is 131072",
    "Please reduce the length of the messages or completion",
    "maximum context length is 8192 tokens",
    "Input length 15000 exceeds the maximum allowed input length of 8192 tokens",
    "The input (15000 tokens) is longer than the model's context length (8192 tokens)",
    "prompt token count of 15000 exceeds the limit of 8192",
    "the request exceeds the available context size, try increasing it",
    "tokens to keep from the initial prompt is greater than the context length",
    "invalid params, context window exceeds limit",
    "Your request exceeded model token limit: 8192 (requested: 15000)",
    "Prompt contains 15000 tokens: too large for model with 8192 maximum context length",
    "Prompt has 15000 tokens, but the configured context size is 8192 tokens",
    "model_context_window_exceeded",
    "prompt too long; exceeded max context length",
    "Range of input length should be [1, 8192]",
    "context_length_exceeded",
    "too many tokens",
    "token limit exceeded",
    "Prompt exceeds max length",
    "input is too long for requested model",
]


@pytest.mark.parametrize("error", OVERFLOW_ERRORS)
def test_overflow_never_enters_transient_retry(error: str) -> None:
    # CODE_REVIEW 4.6; examples from Pi ai/src/utils/overflow.ts.
    message = AssistantMessage(stop_reason="error", error_message=error)
    assert _is_context_overflow(message, 8192)
    assert not _is_retryable_error(message)


@pytest.mark.parametrize("error", ["Invalid model 15000", "Unrecognized ID 4290", "Bad argument 15020"])
def test_http_status_substrings_are_not_retryable(error: str) -> None:
    assert not _is_retryable_error(AssistantMessage(stop_reason="error", error_message=error))


@pytest.mark.parametrize("error", ["HTTP 500", "429 Too many requests", "connection refused", "timeout"])
def test_transient_errors_remain_retryable(error: str) -> None:
    assert _is_retryable_error(AssistantMessage(stop_reason="error", error_message=error))


@pytest.mark.parametrize(
    "error",
    [
        "Rate limit: too many tokens",
        "Throttling error: Too many tokens",
        "Service unavailable: too many tokens",
    ],
)
def test_throttling_is_not_overflow(error: str) -> None:
    assert not _is_context_overflow(AssistantMessage(stop_reason="error", error_message=error), 8192)


async def test_overflow_compaction_precedes_retry(config: AgentConfig) -> None:
    provider = FauxProvider([AssistantMessage(stop_reason="error", error_message=OVERFLOW_ERRORS[17])])
    config.retry.enabled = True
    config.retry.base_delay_ms = 1
    session = create_agent_session(provider, config=config)
    events = []
    session.subscribe(events.append)
    try:
        await session.prompt("work")
        assert any(isinstance(event, CompactionStart) and event.reason == "overflow" for event in events)
        assert not any(isinstance(event, AutoRetryStart) for event in events)
        assert len(provider.requests) == 1  # insufficient history: compaction fails, no blind retry
    finally:
        session.dispose()


async def test_retry_omits_failed_attempt_and_reports_success(config: AgentConfig) -> None:
    config.retry.enabled = True
    config.retry.base_delay_ms = 1
    provider = FauxProvider(
        [
            AssistantMessage(stop_reason="error", error_message="HTTP 503"),
            AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop"),
        ]
    )
    session = create_agent_session(provider, config=config, session="new")
    try:
        await session.prompt("work")
        assert len(provider.requests) == 2
        assert provider.requests[-1].messages[-1].role == "user"
        assert len([entry for entry in session.store.entries if entry.type == "context_omission"]) == 1
    finally:
        session.dispose()


async def test_abort_during_retry_backoff(config: AgentConfig) -> None:
    config.retry.enabled = True
    config.retry.base_delay_ms = 30_000
    provider = FauxProvider([AssistantMessage(stop_reason="error", error_message="HTTP 503")])
    session = create_agent_session(provider, config=config)
    started = asyncio.Event()
    session.subscribe(lambda event: started.set() if isinstance(event, AutoRetryStart) else None)
    task = asyncio.create_task(session.prompt("work"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(session.abort(), 1)
        await task
        assert len(provider.requests) == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()


def test_usage_overflow_is_excluded_from_transient_retry() -> None:
    message = AssistantMessage(stop_reason="error", error_message="HTTP 503", usage=Usage(input=9000))
    assert _is_context_overflow(message, 8192)
    assert not _is_retryable_error(message, 8192)


def test_success_and_output_tokens_do_not_trigger_overflow() -> None:
    assert not _is_context_overflow(AssistantMessage(stop_reason="stop", usage=Usage(input=9000)), 8192)
    assert not _is_context_overflow(AssistantMessage(stop_reason="error", usage=Usage(output=9000)), 8192)


@pytest.mark.parametrize("provider, expected", [("cerebras", True), ("other", False)])
def test_bodyless_overflow_is_provider_specific(provider: str, expected: bool) -> None:
    message = AssistantMessage(
        stop_reason="error", provider=provider, error_message="413 status code (no body)"
    )
    assert _is_context_overflow(message, 8192) == expected
