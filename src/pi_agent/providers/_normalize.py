"""Provider-boundary projection, adapted from Pi's transform-messages.ts."""

import hashlib
import re
from collections.abc import Sequence
from copy import deepcopy

from pi_agent.types import (
    AssistantMessage,
    ContentBlock,
    Message,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
)


def normalize_messages(messages: Sequence[Message], model: ModelInfo) -> list[Message]:
    """Copy history; pair portable IDs and repair missing results without executing tools.

    Failed partial responses and their results never enter a new API request.
    Native thinking signatures are only replayable to the same model identity.
    """
    result: list[Message] = []
    pending: dict[str, ToolCallBlock] = {}
    answered: set[str] = set()
    id_map: dict[str, str] = {}
    used: set[str] = set()
    held_system: list[SystemMessage] = []

    def close_pending() -> None:
        for call_id, call in pending.items():
            if call_id not in answered:
                result.append(
                    ToolResultMessage(
                        tool_call_id=call_id,
                        tool_name=call.name,
                        content=[TextBlock(text="No result provided; outcome unknown")],
                        is_error=True,
                        interrupted=True,
                    )
                )
        pending.clear()
        answered.clear()
        result.extend(held_system)
        held_system.clear()

    for original in messages:
        message = deepcopy(original)
        if isinstance(message, AssistantMessage):
            close_pending()
            if message.stop_reason in ("error", "aborted", "pending"):
                continue
            same = (message.provider, message.api, message.model) == (model.provider, model.api, model.id)
            blocks: list[ContentBlock] = []
            for block in message.content:
                if isinstance(block, ThinkingBlock):
                    if same:
                        blocks.append(block)
                elif isinstance(block, ToolCallBlock):
                    old_id = block.id
                    candidate = old_id
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", candidate) or candidate in used:
                        digest = hashlib.sha256(old_id.encode()).hexdigest()[:32]
                        candidate = f"call_{digest}"
                        suffix = 0
                        while candidate in used:
                            suffix += 1
                            candidate = f"call_{digest}_{suffix}"
                    block.id = candidate
                    used.add(candidate)
                    id_map[old_id] = candidate
                    if not same:
                        block.thought_signature = None
                    pending[candidate] = block
                    blocks.append(block)
                else:
                    blocks.append(block)
            message.content = blocks
            if blocks:
                result.append(message)
        elif isinstance(message, ToolResultMessage):
            message.tool_call_id = id_map.get(message.tool_call_id, message.tool_call_id)
            if message.tool_call_id in pending and message.tool_call_id not in answered:
                answered.add(message.tool_call_id)
                result.append(message)
        elif isinstance(message, SystemMessage) and pending:
            held_system.append(message)
        else:
            if not isinstance(message, SystemMessage):
                close_pending()
            result.append(message)
    close_pending()
    return result
