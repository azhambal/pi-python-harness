"""Pure post-request recovery decisions; execution remains in the facade."""

from dataclasses import dataclass
from typing import Literal

from .config import RetrySettings
from .error_policy import is_context_overflow, is_retryable_error
from .types import AssistantMessage


@dataclass(frozen=True)
class RecoveryDecision:
    action: Literal["none", "retry", "overflow", "overflow_exhausted"] = "none"
    attempt: int = 0
    delay_ms: int = 0


def recovery_decision(
    message: AssistantMessage | None,
    settings: RetrySettings,
    *,
    context_window: int,
    attempt: int,
    aborted: bool,
    overflow_attempted: bool,
) -> RecoveryDecision:
    if aborted or message is None:
        return RecoveryDecision()
    if is_context_overflow(message, context_window):
        return RecoveryDecision(action="overflow_exhausted" if overflow_attempted else "overflow")
    if settings.enabled and attempt < settings.max_retries and is_retryable_error(message, context_window):
        next_attempt = attempt + 1
        return RecoveryDecision(
            action="retry", attempt=next_attempt, delay_ms=settings.delay_ms(next_attempt)
        )
    return RecoveryDecision()
