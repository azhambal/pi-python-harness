"""Record unknown outcomes without replaying interrupted tools."""

from __future__ import annotations

from .session_store import (
    SessionStore,
)
from .types import (
    TextBlock,
    ToolResultMessage,
    now_ms,
)

INTERRUPTION_RESULT_TEXT = (
    "The session was interrupted before this tool call's outcome was recorded. The outcome is unknown: "
    "the call may or may not have run to completion. Inspect the affected files on disk before "
    "re-issuing this tool call."
)


def record_interruptions(store: SessionStore) -> bool:
    appended = False
    for block in store.unanswered_calls:
        store.append_message(
            ToolResultMessage(
                tool_call_id=block.id,
                tool_name=block.name,
                content=[TextBlock(text=INTERRUPTION_RESULT_TEXT)],
                is_error=True,
                interrupted=True,
                timestamp=now_ms(),
            )
        )
        appended = True
    return appended
