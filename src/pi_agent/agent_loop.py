"""Stream assistant responses, validate calls, and execute bounded tool batches."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from .cancellation import Cancellation, OperationAbortedError, iterate_with_cancellation
from .events import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    StreamDone,
    StreamError,
    StreamStart,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    TurnEnd,
    TurnStart,
)
from .permissions import PermissionPolicy, PermissionPrompt, check_permission
from .provider import Provider, ProviderOptions, ProviderRequest
from .tool_hooks import (
    AfterToolCallContext,
    BeforeToolCallContext,
    ToolHooks,
    invoke_hook,
    merge_after_result,
    validate_before_result,
    validate_tool_result,
)
from .tools import is_builtin_search
from .tools._mutation import file_lock
from .tools.base import Tool, ToolResult, ToolValidationError, error_tool_result, validate_tool_arguments
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    ModelInfo,
    TextBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolDeclaration,
    ToolResultMessage,
    Usage,
    now_ms,
)

#: Sink used by the loop to publish lifecycle events.
EventSink = Callable[[AgentEvent], Awaitable[None]]
SteeringSource = Callable[[], Awaitable[list[Message]]]
FollowUpSource = Callable[[], Awaitable[list[Message]]]


@dataclass
class PrepareRequestContext:
    """Runtime state immediately before a conversational provider request."""

    messages: list[Message]
    tools: list[Tool]


@dataclass
class PrepareRequestUpdate:
    """Replacement context for the request being prepared."""

    messages: list[Message] | None = None


PrepareRequestHook = Callable[[PrepareRequestContext], Awaitable[PrepareRequestUpdate | None]]
#: Supplies the set of tool-call IDs already used across RAW session history
#: (including omitted attempts and pre-compaction entries). Called per
#: response so omissions and checkpoints recorded mid-session are reflected;
#: None for a bare Agent, which checks only its own transcript.
HistoricalCallIds = Callable[[], set[str]]

_PARTIAL_EVENT_TYPES = (
    TextStart,
    TextDelta,
    TextEnd,
    ThinkingStart,
    ThinkingDelta,
    ThinkingEnd,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
)


@dataclass
class AgentLoopConfig:
    """What the loop needs besides the provider."""

    model: ModelInfo = field(
        default_factory=lambda: ModelInfo(id="unknown", name="unknown", provider="unknown")
    )
    model_declarations: list[ToolDeclaration] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    thinking_level: ThinkingLevel = "off"
    get_steering_messages: SteeringSource | None = None
    get_follow_up_messages: FollowUpSource | None = None
    prepare_request: PrepareRequestHook | None = None
    historical_call_ids: HistoricalCallIds | None = None
    session_id: str | None = None
    hooks: ToolHooks = field(default_factory=ToolHooks)
    permissions: PermissionPolicy = field(default_factory=PermissionPolicy)
    confirm_permission: PermissionPrompt | None = None
    cwd: Path | None = None
    tool_execution: Literal["sequential", "parallel"] = "sequential"
    max_parallel_tools: int = 4
    admission_lock: asyncio.Lock | None = None


@dataclass
class ExecutedToolBatch:
    messages: list[ToolResultMessage] = field(default_factory=list)
    terminate: bool = False


async def _close_stream(stream: object) -> None:
    """Close a provider async iterator on early termination (best effort)."""
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        result = close()
        if isinstance(result, Awaitable):
            await result
    except Exception:
        return


def _tool_call_is_usable(tool_call: ToolCallBlock) -> bool:
    return bool(tool_call.id) and bool(tool_call.name) and isinstance(tool_call.arguments, dict)


async def run_agent_loop(
    prompts: Sequence[Message],
    context_messages: list[Message],
    config: AgentLoopConfig,
    emit: EventSink,
    cancellation: Cancellation,
    provider: Provider,
) -> list[Message]:
    """Start a loop with new prompt messages (runAgentLoop in Pi)."""
    initial = list(prompts)
    new_messages: list[Message] = list(initial)
    current: list[Message] = list(context_messages) + initial

    await emit(AgentStart())
    await emit(TurnStart())
    for message in initial:
        await emit(MessageStart(message=message))
        await emit(MessageEnd(message=message))

    await _run_loop(current, new_messages, config, emit, cancellation, provider)
    return new_messages


async def run_agent_loop_continue(
    context_messages: list[Message],
    config: AgentLoopConfig,
    emit: EventSink,
    cancellation: Cancellation,
    provider: Provider,
) -> list[Message]:
    """Continue from the current transcript without new messages.

    The last message must be a user or tool-result message; an assistant
    tail cannot be continued (the provider would reject the replay).
    """
    if not context_messages:
        raise ValueError("Cannot continue: no messages in context")
    if isinstance(context_messages[-1], AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[Message] = []
    current = list(context_messages)
    await emit(AgentStart())
    await emit(TurnStart())
    await _run_loop(current, new_messages, config, emit, cancellation, provider)
    return new_messages


async def _run_loop(
    current: list[Message],
    new_messages: list[Message],
    config: AgentLoopConfig,
    emit: EventSink,
    cancellation: Cancellation,
    provider: Provider,
) -> None:
    pending: list[Message] = (await config.get_steering_messages()) if config.get_steering_messages else []
    first_iteration = True

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending:
            # Honor cancellation before any additional model request or tool
            # call; the batch that finished is already fully paired.
            if cancellation.aborted:
                await emit(AgentEnd(messages=new_messages))
                return

            if not first_iteration:
                # Every additional tool/follow-up turn starts like the first
                # one did in the prelude (Pi emits turn_start per iteration).
                await emit(TurnStart())
            first_iteration = False

            for message in pending:
                await emit(MessageStart(message=message))
                await emit(MessageEnd(message=message))
                current.append(message)
                new_messages.append(message)
            pending = []

            update = (
                await config.prepare_request(PrepareRequestContext(messages=current, tools=config.tools))
                if config.prepare_request
                else None
            )
            if update is not None and update.messages is not None:
                current = list(update.messages)

            message = await _stream_assistant_response(current, config, emit, cancellation, provider)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await emit(TurnEnd(message=message, tool_results=[]))
                await emit(AgentEnd(messages=new_messages))
                return

            tool_calls = [block for block in message.content if isinstance(block, ToolCallBlock)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                batch = (
                    await _fail_tool_calls_from_truncated_message(tool_calls, emit)
                    if message.stop_reason == "length"
                    else await (
                        _execute_tool_calls_parallel
                        if config.tool_execution == "parallel"
                        else _execute_tool_calls_sequential
                    )(config, tool_calls, current, message, emit, cancellation)
                )
                tool_results = batch.messages
                has_more_tool_calls = not batch.terminate
                for result in tool_results:
                    current.append(result)
                    new_messages.append(result)

            await emit(TurnEnd(message=message, tool_results=tool_results))
            pending = (await config.get_steering_messages()) if config.get_steering_messages else []

        follow_ups = (await config.get_follow_up_messages()) if config.get_follow_up_messages else []
        if follow_ups:
            pending = follow_ups
            continue
        break

    await emit(AgentEnd(messages=new_messages))


_DONE_REASON_TO_STOP_REASON = {
    "stop": "stop",
    "length": "length",
    "toolUse": "toolUse",
}


def _transcript_call_ids(messages: Sequence[Message]) -> set[str]:
    """Every tool-call ID used by assistant messages in the transcript.

    For session runs this is the visible projection — the caller layers the
    raw historical ID set on top so compaction or omission cannot hide a
    collision; a bare Agent passes only its own transcript.
    """
    ids: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolCallBlock):
                    ids.add(block.id)
    return ids


def _sanitize_final_message(message: AssistantMessage, known_ids: set[str]) -> AssistantMessage:
    """Assign journal-unique IDs; malformed calls become explicit protocol failures.

    Original IDs are retained in harness metadata for future provider adapters.
    Never replace malformed arguments with {}: that could execute an unintended call.
    """
    seen_ids = set(known_ids)
    sanitized: list[ContentBlock] = []
    original_ids: dict[str, str] = {}
    malformed = False
    for block in message.content:
        if not isinstance(block, ToolCallBlock):
            sanitized.append(block)
            continue
        if not block.name or not isinstance(block.arguments, dict):
            sanitized.append(TextBlock(text="Provider returned a tool call with invalid name or arguments."))
            malformed = True
            continue
        if not block.id or block.id in seen_ids:
            original_id = block.id
            while not block.id or block.id in seen_ids:
                block.id = f"call_{uuid.uuid4().hex}"
            original_ids[block.id] = original_id
        seen_ids.add(block.id)
        sanitized.append(block)
    message.content = sanitized
    if original_ids:
        message.provider_data = {
            **(message.provider_data or {}),
            "pi_original_tool_call_ids": dict(original_ids),
        }
    if malformed and message.stop_reason not in ("error", "aborted"):
        message.stop_reason = "error"
        message.error_message = "Provider returned a tool call with invalid name or arguments"
    if not message.content or (
        message.stop_reason == "toolUse" and not any(isinstance(b, ToolCallBlock) for b in message.content)
    ):
        if message.stop_reason not in ("error", "aborted"):
            message.stop_reason = "error"
            message.error_message = "Provider returned an empty assistant response or toolUse without calls"
        message.content = message.content or [
            TextBlock(text=message.error_message or "Provider stream failed")
        ]
    return message


async def _stream_assistant_response(
    current: list[Message],
    config: AgentLoopConfig,
    emit: EventSink,
    cancellation: Cancellation,
    provider: Provider,
) -> AssistantMessage:
    """Stream one assistant response, updating the transcript in place.

    Terminal events are validated against the final message's stop reason,
    unusable tool-call blocks are sanitized before the message commits, a
    successful completion is downgraded to an abort when cancellation fired
    (even if the provider ignored the token), partial content is preserved
    when the stream fails, and the stream is closed on early return.
    """
    started_at = time.monotonic()
    request = ProviderRequest(
        model=config.model,
        messages=list(current),
        tools=list(config.model_declarations),
    )
    options = ProviderOptions(
        cancellation=cancellation,
        thinking_level=config.thinking_level,
        session_id=config.session_id,
    )

    partial: AssistantMessage | None = None
    added_partial = False
    # Sessions include hidden raw-history IDs; bare Agents use their transcript.
    # Reassign conflicting provider IDs before commit/execution.
    known_call_ids = _transcript_call_ids(current)
    if config.historical_call_ids is not None:
        known_call_ids |= config.historical_call_ids()

    def finalize(message: AssistantMessage) -> AssistantMessage:
        if message.duration_ms is None:
            message.duration_ms = max(0, round((time.monotonic() - started_at) * 1000))
        return message

    async def commit_and_end(final: AssistantMessage) -> AssistantMessage:
        final = deepcopy(final)
        _sanitize_final_message(final, known_call_ids)
        if added_partial:
            current[-1] = final
        else:
            current.append(final)
            await emit(MessageStart(message=final))
        await emit(MessageEnd(message=final))
        return final

    async def fail(
        reason: str, stop_reason: str, content: list[ContentBlock] | None = None
    ) -> AssistantMessage:
        failure = AssistantMessage(
            content=list(content) if content is not None else [],
            model=request.model.id,
            provider=request.model.provider,
            api=request.model.api,
            usage=Usage(),
            stop_reason=stop_reason,  # type: ignore[arg-type]
            error_message=reason,
            timestamp=now_ms(),
        )
        finalize(failure)
        return await commit_and_end(failure)

    if cancellation.aborted:
        return await fail("Operation aborted", "aborted")

    stream = provider.stream(request, options)
    stream_events = iterate_with_cancellation(stream, cancellation)
    try:
        async for event in stream_events:
            if isinstance(event, StreamDone):
                final = event.message
                # The event reason must match the message's stop reason; the
                # message is authoritative for content, the pair for intent.
                expected = _DONE_REASON_TO_STOP_REASON.get(event.reason)
                if expected is None or final.stop_reason != expected:
                    return await fail(
                        f"provider stream ended with reason {event.reason!r} but stop_reason "
                        f"{final.stop_reason!r}",
                        "error",
                        content=final.content or (partial.content if partial is not None else None),
                    )
                # A provider that ignored the cancellation token and emitted a
                # successful completion is still aborted: the run was
                # cancelled and nothing of it may execute.
                if cancellation.aborted:
                    return await fail("Operation aborted", "aborted", content=final.content)
                return await commit_and_end(finalize(final))
            if isinstance(event, StreamError):
                final = event.error
                # Keep useful partial output when the error carries none.
                if not final.content and partial is not None and partial.content:
                    final.content = list(partial.content)
                if event.reason == "aborted" and final.stop_reason != "aborted":
                    return await fail(
                        f"provider stream aborted with stop_reason {final.stop_reason!r}",
                        "aborted",
                        content=final.content,
                    )
                if event.reason == "error" and final.stop_reason not in ("error",):
                    return await fail(
                        f"provider stream error with stop_reason {final.stop_reason!r}",
                        "error",
                        content=final.content,
                    )
                if final.stop_reason not in ("error", "aborted"):
                    final.stop_reason = "error"
                if not final.error_message:
                    final.error_message = "Provider stream failed"
                return await commit_and_end(finalize(final))
            if isinstance(event, _PARTIAL_EVENT_TYPES) or isinstance(event, StreamStart):
                if getattr(event, "partial", None) is not None:
                    snapshot = deepcopy(event.partial)
                    if not added_partial:
                        partial = snapshot
                        current.append(partial)
                        added_partial = True
                        await emit(MessageStart(message=partial))
                    else:
                        partial = snapshot
                        current[-1] = partial
                    await emit(MessageUpdate(message=partial, stream_event=deepcopy(event)))
    except OperationAbortedError:
        return await fail("Operation aborted", "aborted", content=partial.content if partial else None)
    finally:
        try:
            await stream_events.aclose()
        finally:
            await _close_stream(stream)

    return await fail(
        "Provider stream ended without a terminal event",
        "error",
        content=partial.content if partial is not None else None,
    )


async def _fail_tool_calls_from_truncated_message(
    tool_calls: Sequence[ToolCallBlock],
    emit: EventSink,
) -> ExecutedToolBatch:
    """Fail every tool call of a length-truncated response (Pi semantics)."""
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await emit(
            ToolExecutionStart(
                tool_call_id=tool_call.id or "<missing-id>",
                tool_name=tool_call.name,
                args=dict(tool_call.arguments) if isinstance(tool_call.arguments, dict) else {},
            )
        )
        result = error_tool_result(
            f'Tool call "{tool_call.name}" was not executed: the response hit the output token '
            "limit, so its arguments may be truncated. Re-issue the tool call with complete arguments."
        )
        await _emit_tool_execution_end(tool_call, result, emit, is_error=True, duration_ms=None)
        tool_result = _tool_result_message(tool_call, result, is_error=True, duration_ms=None)
        await emit(MessageStart(message=tool_result))
        await emit(MessageEnd(message=tool_result))
        messages.append(tool_result)
    return ExecutedToolBatch(messages=messages, terminate=False)


async def _emit_tool_execution_end(
    tool_call: ToolCallBlock,
    result: ToolResult,
    emit: EventSink,
    *,
    is_error: bool,
    duration_ms: int | None,
) -> None:
    await emit(
        ToolExecutionEnd(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=deepcopy(result.to_json()),
            is_error=is_error,
            duration_ms=duration_ms,
        )
    )


def _make_update_sink(
    emit: EventSink,
    tool_call: ToolCallBlock,
) -> Callable[[ToolResult], Awaitable[None]]:
    """Build the per-call on_update callback (avoids binding loop variables)."""

    async def on_update(partial_result: ToolResult) -> None:
        await emit(
            ToolExecutionUpdate(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=dict(tool_call.arguments) if isinstance(tool_call.arguments, dict) else {},
                partial_result=deepcopy(partial_result.to_json()),
            )
        )

    return on_update


async def _execute_tool_calls_sequential(
    config: AgentLoopConfig,
    tool_calls: Sequence[ToolCallBlock],
    current: Sequence[Message],
    assistant_message: AssistantMessage,
    emit: EventSink,
    cancellation: Cancellation,
) -> ExecutedToolBatch:
    tools = {tool.name: tool for tool in config.tools}
    messages: list[ToolResultMessage] = []
    terminate = True
    cancelled_caller = False

    for index, original_call in enumerate(tool_calls):
        tool_call = deepcopy(original_call)
        # Pair every requested call even when the batch is interrupted:
        # unstarted calls get an aborted error result in this same batch.
        if cancellation.aborted:
            for remaining in tool_calls[index:]:
                result = error_tool_result("Operation aborted")
                await _emit_tool_execution_end(remaining, result, emit, is_error=True, duration_ms=None)
                tool_result = _tool_result_message(remaining, result, is_error=True, duration_ms=None)
                await emit(MessageStart(message=tool_result))
                await emit(MessageEnd(message=tool_result))
                messages.append(tool_result)
            break

        await emit(
            ToolExecutionStart(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=deepcopy(tool_call.arguments) if isinstance(tool_call.arguments, dict) else {},
            )
        )

        outcome = await _run_tool_call(
            config,
            tools.get(tool_call.name),
            tool_call,
            assistant_message,
            [*current, *messages],
            emit,
            cancellation,
        )
        result = outcome.result
        cancelled_caller |= outcome.cancelled_caller
        if outcome.cancelled_caller:
            cancellation.abort()
        await _emit_tool_execution_end(
            tool_call, result, emit, is_error=result.is_error, duration_ms=outcome.duration_ms
        )
        tool_result = _tool_result_message(
            tool_call, result, is_error=result.is_error, duration_ms=outcome.duration_ms
        )
        tool_result.interrupted = outcome.interrupted
        await emit(MessageStart(message=tool_result))
        await emit(MessageEnd(message=tool_result))
        messages.append(tool_result)
        terminate = terminate and result.terminate is True

    if cancelled_caller:
        raise asyncio.CancelledError
    return ExecutedToolBatch(messages=messages, terminate=terminate and bool(messages))


@dataclass
class _ToolOutcome:
    result: ToolResult
    duration_ms: int | None = None
    cancelled_caller: bool = False
    interrupted: bool = False


async def _execute_tool_calls_parallel(
    config: AgentLoopConfig,
    tool_calls: Sequence[ToolCallBlock],
    current: Sequence[Message],
    assistant_message: AssistantMessage,
    emit: EventSink,
    cancellation: Cancellation,
) -> ExecutedToolBatch:
    """Bounded concurrency; resolved-path FIFO queues; ordered final commits.

    Non-opted-in custom tools are barriers. Admission/confirmation is serialized,
    execution updates may interleave, and final results follow source order.
    """
    tools = {tool.name: tool for tool in config.tools}
    semaphore = asyncio.Semaphore(config.max_parallel_tools)
    guarded = replace(config, admission_lock=asyncio.Lock())
    messages: list[ToolResultMessage] = []
    terminate = True
    caller_cancelled = False

    async def execute(call: ToolCallBlock, context: list[Message]) -> _ToolOutcome:
        tool = tools.get(call.name)
        path = call.arguments.get("path") if isinstance(call.arguments, dict) else None
        lock = None
        if tool is not None and call.name in ("read", "write", "edit") and isinstance(path, str):
            from .tools.base import resolve_path

            cwd = getattr(tool, "cwd", None) or config.cwd
            if cwd is not None:
                lock = file_lock(resolve_path(path, cwd))
        try:
            # Path queues precede the capacity gate so queued same-file calls don't
            # monopolize capacity needed by independent calls.
            async with lock if lock is not None else nullcontext():
                async with semaphore:
                    if cancellation.aborted:
                        return _ToolOutcome(error_tool_result("Operation aborted; tool was not executed"))
                    await emit(
                        ToolExecutionStart(
                            tool_call_id=call.id, tool_name=call.name, args=deepcopy(call.arguments)
                        )
                    )
                    return await _run_tool_call(
                        guarded, tool, call, assistant_message, context, emit, cancellation
                    )
        except asyncio.CancelledError:
            return _ToolOutcome(
                error_tool_result("Operation aborted; queued tool was not executed"), cancelled_caller=True
            )

    index = 0
    while index < len(tool_calls):
        end = index + 1
        tool = tools.get(tool_calls[index].name)
        if tool is not None and tool.parallel_safe:
            while end < len(tool_calls):
                following = tools.get(tool_calls[end].name)
                if following is None or not following.parallel_safe:
                    break
                end += 1
        calls = deepcopy(list(tool_calls[index:end]))
        context = deepcopy([*current, *messages])
        tasks = [asyncio.create_task(execute(call, deepcopy(context))) for call in calls]
        joined = asyncio.gather(*tasks, return_exceptions=True)
        try:
            outcomes = await asyncio.shield(joined)
        except asyncio.CancelledError:
            caller_cancelled = True
            cancellation.abort()
            for task in tasks:
                task.cancel()
            outcomes = await joined
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for call, outcome in zip(calls, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                raise outcome
            caller_cancelled |= outcome.cancelled_caller
            if outcome.cancelled_caller:
                cancellation.abort()
            result = outcome.result
            await _emit_tool_execution_end(
                call, result, emit, is_error=result.is_error, duration_ms=outcome.duration_ms
            )
            message = _tool_result_message(call, result, result.is_error, outcome.duration_ms)
            message.interrupted = outcome.interrupted
            await emit(MessageStart(message=message))
            await emit(MessageEnd(message=message))
            messages.append(message)
            terminate = terminate and result.terminate
        index = end
    if caller_cancelled:
        raise asyncio.CancelledError
    return ExecutedToolBatch(messages, terminate and bool(messages))


def _caller_is_cancelled() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _after_hook_failure(original: ToolResult, reason: str, *, cancelled: bool = False) -> ToolResult:
    result = deepcopy(original)
    result.content.append(
        TextBlock(
            text=f"After-tool hook {'cancelled' if cancelled else 'failed'}: {reason}. "
            "The tool already executed; its original result is preserved. Inspect effects before retrying."
        )
    )
    result.details = {
        "original_result": original.to_json(),
        "hook_stage": "after_tool_call",
        "hook_error": reason,
        "executed": True,
    }
    result.structured_content = None
    result.is_error = original.is_error if cancelled else True
    return result


async def _run_tool_call(
    config: AgentLoopConfig,
    tool: Tool | None,
    call: ToolCallBlock,
    assistant_message: AssistantMessage,
    messages: Sequence[Message],
    emit: EventSink,
    cancellation: Cancellation,
) -> _ToolOutcome:
    if not _tool_call_is_usable(call):
        return _ToolOutcome(error_tool_result(f'Tool call "{call.name}" is unusable and was not executed.'))
    if tool is None:
        return _ToolOutcome(error_tool_result(f"Tool {call.name} not found"))
    try:
        validated = validate_tool_arguments(tool, deepcopy(call.arguments))
    except ToolValidationError as error:
        return _ToolOutcome(error_tool_result(str(error)))
    tool_cwd = getattr(tool, "cwd", None)
    cwd = Path(tool_cwd).resolve() if isinstance(tool_cwd, (str, Path)) else config.cwd
    context = BeforeToolCallContext(
        assistant_message=deepcopy(assistant_message),
        tool_call=deepcopy(call),
        args=deepcopy(validated),
        messages=deepcopy(list(messages)),
        tools=deepcopy(config.model_declarations),
        cwd=cwd,
        builtin_search=is_builtin_search(tool),
        tool_provenance=deepcopy(tool.provenance),
    )
    try:
        async with config.admission_lock if config.admission_lock is not None else nullcontext():
            if config.hooks.before_tool_call is not None:
                before = validate_before_result(
                    await invoke_hook(config.hooks.before_tool_call, deepcopy(context), cancellation)
                )
                if before is not None and before.block:
                    result = error_tool_result(before.reason or "Tool execution was blocked")
                    result.terminate = before.terminate
                    return _ToolOutcome(result)

            async def permission_check(ctx: BeforeToolCallContext, token: Cancellation):
                return await check_permission(
                    ctx, token, policy=config.permissions, confirm=config.confirm_permission
                )

            decision = validate_before_result(
                await invoke_hook(permission_check, deepcopy(context), cancellation)
            )
            if decision is not None and decision.block:
                return _ToolOutcome(error_tool_result(decision.reason or "Tool execution denied"))
            cancellation.raise_if_aborted()
    except OperationAbortedError:
        return _ToolOutcome(error_tool_result("Operation aborted; tool was not executed"))
    except asyncio.CancelledError:
        return _ToolOutcome(
            error_tool_result("Tool admission was cancelled; tool was not executed"),
            cancelled_caller=_caller_is_cancelled(),
        )
    except Exception as error:
        return _ToolOutcome(error_tool_result(f"Tool admission failed; tool was not executed: {error}"))

    started = time.monotonic()
    try:
        result = await tool.execute(call.id, deepcopy(validated), cancellation, _make_update_sink(emit, call))
        validate_tool_result(result)
        result = deepcopy(result)
    except asyncio.CancelledError:
        return _ToolOutcome(
            error_tool_result(
                "Tool execution interrupted; outcome unknown. Inspect effects before retrying."
            ),
            max(0, round((time.monotonic() - started) * 1000)),
            True,
            True,
        )
    except Exception as error:
        result = error_tool_result(str(error))
    duration = max(0, round((time.monotonic() - started) * 1000))
    after = config.hooks.after_tool_call
    if after is None:
        return _ToolOutcome(result, duration)
    after_context = AfterToolCallContext(
        assistant_message=context.assistant_message,
        tool_call=context.tool_call,
        args=context.args,
        messages=context.messages,
        tools=context.tools,
        cwd=context.cwd,
        builtin_search=context.builtin_search,
        tool_provenance=context.tool_provenance,
        result=deepcopy(result),
    )
    try:
        override = await invoke_hook(after, after_context, cancellation)
        return _ToolOutcome(merge_after_result(result, override), duration)
    except OperationAbortedError:
        return _ToolOutcome(_after_hook_failure(result, "Operation aborted", cancelled=True), duration)
    except asyncio.CancelledError:
        caller = _caller_is_cancelled()
        return _ToolOutcome(
            _after_hook_failure(result, "Callback cancelled", cancelled=True), duration, caller
        )
    except Exception as error:
        return _ToolOutcome(_after_hook_failure(result, str(error)), duration)


def _tool_result_message(
    tool_call: ToolCallBlock,
    result: ToolResult,
    is_error: bool,
    duration_ms: int | None,
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=deepcopy(result.content) or [TextBlock(text="")],
        details=deepcopy(result.details),
        structured_content=deepcopy(result.structured_content),
        usage=deepcopy(result.usage) if isinstance(result.usage, Usage) else None,
        is_error=is_error,
        timestamp=now_ms(),
        duration_ms=duration_ms,
    )
