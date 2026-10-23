"""Runtime tool hooks, copied contexts and cancellation-aware callback lifetime."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .cancellation import Cancellation
from .tools.base import ToolResult
from .types import (
    AssistantMessage,
    ImageBlock,
    JsonValue,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolDeclaration,
    Usage,
)

_IN_TOOL_HOOK: contextvars.ContextVar[bool] = contextvars.ContextVar("pi_agent_in_tool_hook", default=False)


@dataclass(frozen=True)
class ToolProvenance:
    """Runtime identity for host policy; contains no connection configuration."""

    kind: str
    server: str
    tool: str
    schema_digest: str
    annotations_json: str = "{}"


@dataclass(frozen=True)
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCallBlock
    args: dict[str, JsonValue]
    messages: list[Message] = field(default_factory=list)
    tools: list[ToolDeclaration] = field(default_factory=list)
    cwd: Path | None = None
    #: Set by dispatch from a verified builtin implementation, not its declared name.
    builtin_search: bool = field(default=False, kw_only=True)
    tool_provenance: ToolProvenance | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class AfterToolCallContext(BeforeToolCallContext):
    result: ToolResult = field(default_factory=ToolResult)

    @property
    def is_error(self) -> bool:
        return self.result.is_error


@dataclass(frozen=True)
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None
    terminate: bool = False


class _Unset(Enum):
    VALUE = "unset"


_UNSET = _Unset.VALUE


@dataclass(frozen=True)
class AfterToolCallResult:
    """Partial replacement. Explicit None clears nullable fields; omission keeps them."""

    content: list[TextBlock | ImageBlock] | None = None
    details: JsonValue | _Unset = _UNSET
    structured_content: JsonValue | _Unset = _UNSET
    usage: Usage | None | _Unset = _UNSET
    is_error: bool | None = None
    terminate: bool | None = None


BeforeToolCallHook = Callable[
    [BeforeToolCallContext, Cancellation],
    BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
]
AfterToolCallHook = Callable[
    [AfterToolCallContext, Cancellation], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]
]


@dataclass(frozen=True)
class ToolHooks:
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None


async def invoke_hook[Context, Result](
    callback: Callable[[Context, Cancellation], Result | Awaitable[Result]],
    context: Context,
    cancellation: Cancellation,
) -> Result:
    """Race a callback with cancellation, cancel/join it and guard lifecycle reentry.

    Sync callbacks may be used for cheap checks. Blocking Python and callbacks
    that suppress task cancellation remain outside the cancellation guarantee.
    """

    async def invoke() -> Result:
        marker = _IN_TOOL_HOOK.set(True)
        try:
            result = callback(context, cancellation)
            if inspect.isawaitable(result):
                return await result
            return result
        finally:
            _IN_TOOL_HOOK.reset(marker)

    cancellation.raise_if_aborted()
    task = asyncio.create_task(invoke())
    aborted = asyncio.create_task(cancellation.wait())
    try:
        await asyncio.wait((task, aborted), return_when=asyncio.FIRST_COMPLETED)
        cancellation.raise_if_aborted()
        return await task
    finally:
        task.cancel()
        aborted.cancel()
        await asyncio.gather(task, aborted, return_exceptions=True)


def validate_before_result(value: object) -> BeforeToolCallResult | None:
    if value is None:
        return None
    if not isinstance(value, BeforeToolCallResult):
        raise ValueError("Before-tool hook must return BeforeToolCallResult or None")
    if not isinstance(value.block, bool) or not isinstance(value.terminate, bool):
        raise ValueError("Before-tool hook block/terminate must be boolean")
    if value.reason is not None and not isinstance(value.reason, str):
        raise ValueError("Before-tool hook reason must be text")
    return value


def merge_after_result(original: ToolResult, value: object) -> ToolResult:
    if value is None:
        return original
    if not isinstance(value, AfterToolCallResult):
        raise ValueError("After-tool hook must return AfterToolCallResult or None")
    result = deepcopy(original)
    if value.content is not None:
        result.content = deepcopy(value.content)
        if value.structured_content is _UNSET:
            result.structured_content = None
    if value.details is not _UNSET:
        result.details = deepcopy(value.details)
    if value.structured_content is not _UNSET:
        result.structured_content = deepcopy(value.structured_content)
    if value.usage is not _UNSET:
        result.usage = deepcopy(value.usage)
    if value.is_error is not None:
        result.is_error = value.is_error
    if value.terminate is not None:
        result.terminate = value.terminate
    validate_tool_result(result)
    return result


def validate_tool_result(result: ToolResult) -> None:
    if not isinstance(result, ToolResult):
        raise ValueError("Tool must return ToolResult")
    if not isinstance(result.content, list) or any(
        not (
            (isinstance(block, TextBlock) and isinstance(block.text, str))
            or (
                isinstance(block, ImageBlock)
                and isinstance(block.data, str)
                and isinstance(block.mime_type, str)
            )
        )
        for block in result.content
    ):
        raise ValueError("Tool result content must contain text or image blocks")
    if not isinstance(result.is_error, bool) or not isinstance(result.terminate, bool):
        raise ValueError("Tool result is_error/terminate must be boolean")
    if result.usage is not None:
        Usage.from_json(result.usage.to_json())
    json.dumps(result.to_json(), allow_nan=False)
