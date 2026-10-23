"""Provider-boundary history projection, translated from Pi's ``transform-messages.ts``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import replace

from pi_agent.types import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    Message,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    now_ms,
)

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"
#: Pi's synthetic answer for a tool call that has no result in the history.
NO_RESULT_TEXT = "No result provided"
ERROR_TOOL_IMAGE_PLACEHOLDER = "(image attached after the tool results)"

ToolCallIdNormalizer = Callable[[str, ModelInfo, AssistantMessage], str]


def _replace_images_with_placeholder(
    content: Sequence[TextBlock | ImageBlock], placeholder: str
) -> list[TextBlock | ImageBlock]:
    result: list[TextBlock | ImageBlock] = []
    previous_was_placeholder = False
    for block in content:
        if isinstance(block, ImageBlock):
            if not previous_was_placeholder:
                result.append(TextBlock(text=placeholder))
            previous_was_placeholder = True
            continue
        result.append(block)
        previous_was_placeholder = block.text == placeholder
    return result


def _downgrade_unsupported_images(messages: list[Message], model: ModelInfo) -> list[Message]:
    if "image" in model.input:
        return messages
    downgraded: list[Message] = []
    for message in messages:
        if isinstance(message, UserMessage) and not isinstance(message.content, str):
            message = replace(
                message,
                content=_replace_images_with_placeholder(message.content, NON_VISION_USER_IMAGE_PLACEHOLDER),
            )
        elif isinstance(message, ToolResultMessage):
            message = replace(
                message,
                content=_replace_images_with_placeholder(message.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER),
            )
        downgraded.append(message)
    return downgraded


def _transform_assistant(
    message: AssistantMessage,
    model: ModelInfo,
    normalize_tool_call_id: ToolCallIdNormalizer | None,
    tool_call_id_map: dict[str, str],
) -> AssistantMessage:
    is_same_model = (message.provider, message.api, message.model) == (model.provider, model.api, model.id)
    content: list[ContentBlock] = []
    for block in message.content:
        if isinstance(block, ThinkingBlock):
            # Thinking from another model is dropped: signatures are only valid for the
            # model that produced them, and readable thinking would read as visible reply.
            if not is_same_model:
                continue
            # Redacted and signed blocks are kept even when empty: they are needed for replay.
            if block.redacted or block.thinking_signature:
                content.append(block)
            elif block.thinking and block.thinking.strip():
                content.append(block)
        elif isinstance(block, TextBlock):
            content.append(block if is_same_model else TextBlock(text=block.text))
        elif isinstance(block, ToolCallBlock):
            tool_call = block
            if not is_same_model and tool_call.thought_signature:
                tool_call = replace(tool_call, thought_signature=None)
            if not is_same_model and normalize_tool_call_id is not None:
                normalized_id = normalize_tool_call_id(tool_call.id, model, message)
                if normalized_id != tool_call.id:
                    tool_call_id_map[tool_call.id] = normalized_id
                    tool_call = replace(tool_call, id=normalized_id)
            content.append(tool_call)
        else:
            content.append(block)
    return replace(message, content=content)


def transform_messages(
    messages: Sequence[Message],
    model: ModelInfo,
    normalize_tool_call_id: ToolCallIdNormalizer | None = None,
) -> list[Message]:
    """Pi's ``transformMessages``: what a provider may replay of the history.

    Unsupported images become placeholders, other models' thinking and
    signatures are dropped, tool call IDs are normalized for the target
    provider, failed or aborted responses are skipped, and every tool call
    without a result is answered with ``"No result provided"``.
    """
    tool_call_id_map: dict[str, str] = {}
    image_aware = _downgrade_unsupported_images([deepcopy(message) for message in messages], model)

    transformed: list[Message] = []
    for message in image_aware:
        if isinstance(message, ToolResultMessage):
            normalized_id = tool_call_id_map.get(message.tool_call_id)
            if normalized_id and normalized_id != message.tool_call_id:
                message = replace(message, tool_call_id=normalized_id)
        elif isinstance(message, AssistantMessage):
            message = _transform_assistant(message, model, normalize_tool_call_id, tool_call_id_map)
        transformed.append(message)

    result: list[Message] = []
    pending_tool_calls: list[ToolCallBlock] = []
    existing_tool_result_ids: set[str] = set()
    # System messages are transparent to tool-call accounting: one between a call and
    # its results is emitted after the results, synthetic ones included.
    held_system_messages: list[Message] = []

    def close_pending_tool_calls() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        if pending_tool_calls:
            for tool_call in pending_tool_calls:
                if tool_call.id not in existing_tool_result_ids:
                    result.append(
                        ToolResultMessage(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            content=[TextBlock(text=NO_RESULT_TEXT)],
                            is_error=True,
                            timestamp=now_ms(),
                        )
                    )
            pending_tool_calls = []
            existing_tool_result_ids = set()
        result.extend(held_system_messages)
        held_system_messages.clear()

    for message in transformed:
        if isinstance(message, AssistantMessage):
            close_pending_tool_calls()
            # Incomplete turns are not replayed; "pending" is this port's in-flight partial.
            if message.stop_reason in ("error", "aborted", "pending"):
                continue
            tool_calls = [block for block in message.content if isinstance(block, ToolCallBlock)]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()
            result.append(message)
        elif isinstance(message, ToolResultMessage):
            existing_tool_result_ids.add(message.tool_call_id)
            result.append(message)
        elif isinstance(message, SystemMessage):
            if pending_tool_calls:
                held_system_messages.append(message)
            else:
                result.append(message)
        elif isinstance(message, UserMessage):
            close_pending_tool_calls()
            result.append(message)
        else:
            result.append(message)

    close_pending_tool_calls()
    return result


def split_error_tool_result_images(
    message: ToolResultMessage,
) -> tuple[list[TextBlock | ImageBlock], str, list[ImageBlock]] | None:
    """Pi's ``splitErrorToolResultImages``: (text-only content, caption, images), for APIs
    that reject images in error results; None when no split is needed."""
    if not message.is_error:
        return None
    images = [block for block in message.content if isinstance(block, ImageBlock)]
    if not images:
        return None
    return (
        _replace_images_with_placeholder(message.content, ERROR_TOOL_IMAGE_PLACEHOLDER),
        f"Images from the error result of tool call {message.tool_call_id} ({message.tool_name}):",
        images,
    )
