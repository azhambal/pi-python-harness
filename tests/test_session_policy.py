import pytest

from pi_agent._session_policy import recovery_decision
from pi_agent.config import RetrySettings
from pi_agent.types import AssistantMessage


@pytest.mark.parametrize(
    "reason,error,attempt,enabled,aborted,overflow_attempted,action,delay",
    [
        ("error", "HTTP 503", 0, True, False, False, "retry", 10),
        ("error", "HTTP 503", 1, True, False, False, "retry", 15),
        ("error", "HTTP 503", 2, True, False, False, "none", 0),
        ("error", "HTTP 503", 0, False, False, False, "none", 0),
        ("error", "HTTP 503", 0, True, True, False, "none", 0),
        ("error", "quota exceeded", 0, True, False, False, "none", 0),
        ("error", "context_length_exceeded", 0, True, False, False, "overflow", 0),
        ("error", "context_length_exceeded", 0, True, False, True, "overflow_exhausted", 0),
        ("error", "context_length_exceeded", 0, False, False, False, "overflow", 0),
        ("length", "context_length_exceeded", 0, True, False, False, "none", 0),
        ("aborted", "HTTP 503", 0, True, False, False, "none", 0),
        ("stop", "HTTP 503", 0, True, False, False, "none", 0),
    ],
)
def test_recovery_decisions(
    reason, error, attempt, enabled, aborted, overflow_attempted, action, delay
) -> None:
    settings = RetrySettings(enabled=enabled, max_retries=2, base_delay_ms=10, max_delay_ms=15)
    message = AssistantMessage(stop_reason=reason, error_message=error)
    before = message.to_json()
    decision = recovery_decision(
        message,
        settings,
        context_window=1000,
        attempt=attempt,
        aborted=aborted,
        overflow_attempted=overflow_attempted,
    )
    assert decision.action == action and decision.delay_ms == delay
    assert decision.attempt == (attempt + 1 if action == "retry" else 0)
    assert message.to_json() == before
