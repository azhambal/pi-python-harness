"""Stateful agent loop with ordered events, cancellation, and input queues."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

from .agent_loop import (
    AgentLoopConfig,
    HistoricalCallIds,
    PrepareNextTurnHook,
    PrepareRequestContext,
    PrepareRequestHook,
    PrepareRequestUpdate,
    run_agent_loop,
    run_agent_loop_continue,
)
from .cancellation import Cancellation
from .events import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
)
from .permissions import PermissionPolicy, PermissionPrompt
from .provider import Provider
from .tool_hooks import _IN_TOOL_HOOK, ToolHooks
from .tools.base import Tool
from .types import (
    AssistantMessage,
    ContentBlock,
    Message,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ThinkingLevel,
    ToolResultMessage,
    Usage,
    UserMessage,
    create_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
    now_ms,
)

QueueMode = str  # "all" | "one-at-a-time"


class FatalPersistenceError(RuntimeError):
    """A durable write failed; the run stops without synthesizing a model error.

    Raised out of the internal (persistence) listener path. Unlike provider
    failures it is never converted into an error assistant message, never
    retried, and every later emission re-raises it until the session ends.
    """


#: Listener return values that are awaitable are awaited; sync listeners are
#: fine too. The internal listener follows the same shape.
Listener = Callable[[AgentEvent, Cancellation], object]
InternalListener = Callable[[AgentEvent, Cancellation], object]
#: Post-reduce dispatch hook: invoked after state reduction, before observer
#: listeners, with each event awaited before the run advances. Failures are
#: isolated like observer failures (never FatalPersistenceError).
PostReduceHook = Callable[[AgentEvent, Cancellation], Awaitable[None]]

_EMPTY_USAGE = Usage()


class _PendingMessageQueue:
    """Steering/follow-up queue with Pi's two drain modes."""

    def __init__(self, mode: QueueMode = "one-at-a-time") -> None:
        self.mode = mode
        self._messages: list[Message] = []

    def enqueue(self, message: Message) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return bool(self._messages)

    def peek(self) -> list[Message]:
        if self.mode == "all":
            return list(self._messages)
        return self._messages[:1]

    def drain(self) -> list[Message]:
        drained = self.peek()
        self._messages = self._messages[len(drained) :]
        return drained

    def clear(self) -> None:
        self._messages = []


async def _call_listener(
    listener: Listener | InternalListener, event: AgentEvent, signal: Cancellation
) -> None:
    result = listener(event, signal)
    if inspect.isawaitable(result):
        await result


def _caller_cancelled() -> bool:
    """Whether the *current task* has a pending cancellation request.

    Used to distinguish cancellation of the actual caller (propagates) from
    an independently cancelled observer Task/Future that was awaited
    (isolated): on Python 3.11+ ``Task.cancelling()`` counts un-cancelled
    requests targeting this task itself.
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


#: Task-local marker for "running inside an agent listener" (hook or
#: observer delivery). ContextVars are inherited by tasks created inside a
#: listener, so listener-owned tasks are guarded too, while unrelated
#: external tasks are not.
_IN_AGENT_LISTENER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pi_agent_in_agent_listener", default=False
)


class Agent:
    """Owns the transcript and drives the loop. One active run per agent."""

    def __init__(
        self,
        provider: Provider,
        *,
        model: ModelInfo,
        tools: Iterable[Tool] = (),
        initial_messages: Sequence[Message] | None = None,
        thinking_level: ThinkingLevel = "off",
        session_id: str | None = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        prepare_request: PrepareRequestHook | None = None,
        prepare_next_turn: PrepareNextTurnHook | None = None,
        historical_call_ids: HistoricalCallIds | None = None,
        hooks: ToolHooks | None = None,
        permission_policy: PermissionPolicy | None = None,
        confirm_permission: PermissionPrompt | None = None,
        cwd: Path | str | None = None,
        tool_execution: Literal["sequential", "parallel"] = "parallel",
    ) -> None:
        self.provider = provider
        self._tools: list[Tool] = list(tools)
        self._messages: list[Message] = list(initial_messages) if initial_messages else []
        initial_system = create_initial_system_message(
            get_current_system_prompt(self._messages) or None,
            [tool.declaration() for tool in self._tools],
        )
        if not (self._messages and isinstance(self._messages[0], SystemMessage)) and initial_system:
            self._messages.insert(0, initial_system)
        self.model = model
        self.thinking_level: ThinkingLevel = thinking_level
        self.session_id = session_id
        self._listeners: list[Listener] = []
        self._internal_listener: InternalListener | None = None
        self._internal_failed = False
        self._post_reduce_hook: PostReduceHook | None = None
        self._steering_queue = _PendingMessageQueue(steering_mode)
        self._follow_up_queue = _PendingMessageQueue(follow_up_mode)
        self._prepare_request_hook = prepare_request
        self._prepare_next_turn_hook = prepare_next_turn
        self._historical_call_ids = historical_call_ids
        self._hooks = hooks or ToolHooks()
        self._permissions = permission_policy or PermissionPolicy()
        self._permissions.validate()
        self._confirm_permission = confirm_permission
        self._cwd = Path(cwd).resolve() if cwd is not None else None
        if tool_execution not in ("sequential", "parallel"):
            raise ValueError("tool_execution must be sequential or parallel")
        self.tool_execution = tool_execution
        self._active_run: asyncio.Future[None] | None = None
        self._active_cancellation: Cancellation | None = None
        self.is_streaming = False
        self.streaming_message: Message | None = None
        self.pending_tool_calls: set[str] = set()
        self.error_message: str | None = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[Message]:
        """Conversation transcript (leading system message carries the prompt)."""
        return self._messages

    @messages.setter
    def messages(self, value: Sequence[Message]) -> None:
        self._messages = list(value)

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    @tools.setter
    def tools(self, value: Iterable[Tool]) -> None:
        self._tools = list(value)

    @property
    def system_prompt(self) -> str:
        return get_current_system_prompt(self._messages)

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def set_internal_listener(self, listener: InternalListener | None) -> None:
        """Install the single internal (persistence) listener.

        Unlike subscribed observers, this listener's exceptions propagate and
        stop the run. The session uses it for journal writes.
        """
        self._internal_listener = listener

    def set_post_reduce_hook(self, hook: PostReduceHook | None) -> None:
        """Install the post-reduce dispatch hook (the session uses it).

        The hook runs after the internal listener (persistence) and after
        state reduction, and is *awaited* before the run advances — this is
        what gives session observers bounded, ordered, up-to-date-state
        delivery. Hook failures are isolated like observer failures, never
        wrapped as FatalPersistenceError.
        """
        self._post_reduce_hook = hook

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Subscribe to lifecycle events as an observer.

        Listeners may be sync or async; awaitable results (coroutines, Tasks,
        Futures) are awaited in subscription order and are part of the
        current run's settlement. Observer failures are isolated.
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    # ------------------------------------------------------------------
    # Queues
    # ------------------------------------------------------------------

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def steer(self, text_or_message: str | Message) -> None:
        """Queue a message injected after the current assistant turn's tools."""
        self._steering_queue.enqueue(_coerce_user_message(text_or_message))

    def follow_up(self, text_or_message: str | Message) -> None:
        """Queue a message processed only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(_coerce_user_message(text_or_message))

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    def peek_queued_messages(self) -> list[Message]:
        steering = self._steering_queue.peek()
        return steering if steering else self._follow_up_queue.peek()

    def drain_queued_messages(self) -> list[Message]:
        """Drain and return queued input, steering first (no re-queue).

        Used by the session to deliver input that arrived after a run's final
        answer (e.g. queued by observers during settlement) through a fresh
        prompt run instead of discarding it.
        """
        drained = self._steering_queue.drain()
        if not drained:
            drained = self._follow_up_queue.drain()
        return drained

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    @property
    def is_idle(self) -> bool:
        return self._active_run is None

    @property
    def signal(self) -> Cancellation | None:
        return self._active_cancellation

    def abort(self) -> None:
        if self._active_cancellation is not None:
            self._active_cancellation.abort()

    async def wait_for_idle(self) -> None:
        if _IN_AGENT_LISTENER.get() or _IN_TOOL_HOOK.get():
            raise RuntimeError(
                "agent.wait_for_idle() must not be awaited from inside an agent listener, "
                "post-reduce hook or tool hook; the run cannot finish while the callback is active"
            )
        if self._active_run is not None:
            await asyncio.shield(self._active_run)

    def reset(self) -> None:
        """Clear the conversation while keeping the replayed prompt/tool baseline."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before resetting.")
        baseline = get_current_system_message(self._messages)
        self._messages = [baseline] if baseline else []
        self.is_streaming = False
        self.streaming_message = None
        self.pending_tool_calls = set()
        self.error_message = None
        self.clear_all_queues()

    async def prompt(self, text_or_messages: str | Message | Sequence[Message]) -> None:
        """Start a new prompt from text, one message, or a batch of messages."""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, "
                "or wait for completion."
            )
        messages = _coerce_prompt_input(text_or_messages)
        await self._run_with_lifecycle(
            lambda cancellation: run_agent_loop(
                messages,
                self._messages,
                self._loop_config(),
                self._emit,
                cancellation,
                self.provider,
            )
        )

    async def continue_(self) -> None:
        """Continue from the current transcript (retries, queued input)."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")
        last = self._messages[-1] if self._messages else None
        if last is None or all(isinstance(message, SystemMessage) for message in self._messages):
            raise RuntimeError("No messages to continue from")
        if isinstance(last, AssistantMessage):
            raise RuntimeError("Cannot continue from message role: assistant")
        await self._run_with_lifecycle(
            lambda cancellation: run_agent_loop_continue(
                self._messages,
                self._loop_config(),
                self._emit,
                cancellation,
                self.provider,
            )
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _loop_config(self) -> AgentLoopConfig:
        async def prepare_request(context: PrepareRequestContext) -> PrepareRequestUpdate | None:
            if self._prepare_request_hook is not None:
                return await self._prepare_request_hook(context)
            return None

        async def drain(queue: _PendingMessageQueue) -> list[Message]:
            return queue.drain()

        return AgentLoopConfig(
            model=self.model,
            model_declarations=[tool.declaration() for tool in self._tools],
            tools=list(self._tools),
            thinking_level=self.thinking_level,
            get_steering_messages=(lambda: drain(self._steering_queue)),
            get_follow_up_messages=(lambda: drain(self._follow_up_queue)),
            prepare_request=prepare_request,
            prepare_next_turn=self._prepare_next_turn_hook,
            historical_call_ids=self._historical_call_ids,
            session_id=self.session_id,
            hooks=self._hooks,
            permissions=self._permissions,
            confirm_permission=self._confirm_permission,
            cwd=self._cwd,
            tool_execution=self.tool_execution,
        )

    async def _run_with_lifecycle(self, executor: Callable[[Cancellation], Awaitable[object]]) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")
        loop = asyncio.get_running_loop()
        run_future: asyncio.Future[None] = loop.create_future()
        cancellation = Cancellation()
        self._active_run = run_future
        self._active_cancellation = cancellation
        self.is_streaming = True
        self.streaming_message = None
        self.error_message = None
        try:
            await executor(cancellation)
        except asyncio.CancelledError:
            cancellation.abort()
            await self._handle_run_failure(RuntimeError("Run cancelled"), aborted=True)
            raise
        except FatalPersistenceError:
            # A durable write failed. Never synthesize a model error for it,
            # never retry it; the lifecycle still finishes in finally.
            cancellation.abort()
            raise
        except Exception as error:  # noqa: BLE001 - surfaced as an error assistant message
            await self._handle_run_failure(error, cancellation.aborted)
        finally:
            self.is_streaming = False
            self.streaming_message = None
            self.pending_tool_calls = set()
            self._active_cancellation = None
            self._active_run = None
            if not run_future.done():
                run_future.set_result(None)

    async def _handle_run_failure(self, error: Exception, aborted: bool) -> None:
        # Preserve useful partial output: the streamed content so far (text
        # and thinking blocks) is kept in the failure message when present.
        partial_content: list[ContentBlock] = []
        streaming = self.streaming_message
        if isinstance(streaming, AssistantMessage):
            partial_content = [
                block for block in streaming.content if isinstance(block, TextBlock | ThinkingBlock)
            ]
        failure = AssistantMessage(
            content=partial_content or [TextBlock(text="")],
            model=self.model.id,
            provider=self.model.provider,
            api=self.model.api,
            usage=_EMPTY_USAGE,
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=now_ms(),
        )
        await self._emit(MessageStart(message=failure))
        await self._emit(MessageEnd(message=failure))
        await self._emit(TurnEnd(message=failure, tool_results=[]))
        await self._emit(AgentEnd(messages=[failure]))

    async def _emit(self, event: AgentEvent) -> None:
        cancellation = self._active_cancellation
        if cancellation is None:
            raise RuntimeError("Agent event emitted outside active run")
        internal = self._internal_listener
        if internal is not None:
            if self._internal_failed:
                raise FatalPersistenceError("persistence already failed")
            # Stage 1 — persistence: a failed durable write propagates and
            # stops the run instead of being retried as a provider failure.
            try:
                await _call_listener(internal, event, cancellation)
            except Exception as error:
                self._internal_failed = True
                raise FatalPersistenceError(f"persistent write failed: {error}") from error
        # Stage 2 — state reduction: agent state advances only after the
        # event is durable, so a failed write never leaves a message in
        # memory that is not on disk.
        self._reduce_state(event)
        # Await observers after state reduction to bound and order delivery.
        # Observer failures are isolated from persistence failures.
        token = _IN_AGENT_LISTENER.set(True)
        try:
            hook = self._post_reduce_hook
            if hook is not None:
                try:
                    await hook(event, cancellation)
                except asyncio.CancelledError:
                    if _caller_cancelled():
                        raise
                except Exception:
                    pass
            for listener in list(self._listeners):
                try:
                    await _call_listener(listener, event, cancellation)
                except asyncio.CancelledError:
                    # An independently cancelled observer (e.g. a listener that
                    # returned an already-cancelled Task or Future) is isolated;
                    # cancellation of the actual caller propagates.
                    if _caller_cancelled():
                        raise
                    continue
                except Exception:
                    # Observers are isolated: one failing listener must not
                    # erase finalized messages or stop the run.
                    continue
        finally:
            _IN_AGENT_LISTENER.reset(token)

    def _reduce_state(self, event: AgentEvent) -> None:
        if isinstance(event, MessageStart | MessageUpdate):
            self.streaming_message = event.message
        elif isinstance(event, MessageEnd):
            self.streaming_message = None
            self._messages.append(event.message)
        elif isinstance(event, AgentStart | TurnStart):
            pass
        elif isinstance(event, AgentEnd):
            self.streaming_message = None
        elif isinstance(event, TurnEnd):
            if event.message.stop_reason in ("error", "aborted") and event.message.error_message:
                self.error_message = event.message.error_message
        elif isinstance(event, ToolExecutionStart):
            self.pending_tool_calls.add(event.tool_call_id)
        elif isinstance(event, ToolExecutionEnd):
            self.pending_tool_calls.discard(event.tool_call_id)


def _is_message(value: object) -> bool:
    return isinstance(value, SystemMessage | UserMessage | AssistantMessage | ToolResultMessage)


def _coerce_user_message(text_or_message: str | Message) -> Message:
    if isinstance(text_or_message, str):
        return UserMessage(content=[TextBlock(text=text_or_message)], timestamp=now_ms())
    if not _is_message(text_or_message):
        raise TypeError("steer()/follow_up() accept a string or a Message instance")
    return text_or_message


def _coerce_prompt_input(text_or_messages: str | Message | Sequence[Message]) -> list[Message]:
    if isinstance(text_or_messages, str):
        return [UserMessage(content=[TextBlock(text=text_or_messages)], timestamp=now_ms())]
    if isinstance(text_or_messages, SystemMessage | UserMessage | AssistantMessage | ToolResultMessage):
        return [text_or_messages]
    return list(text_or_messages)
