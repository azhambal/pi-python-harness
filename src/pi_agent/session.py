"""AgentSession: the public session facade wiring agent, store, and compaction.

Contracts and failure semantics: docs/invariants.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ._async_resources import _CLEANING, ResourceOwner, reject_cleanup_reentry, settle_cleanup
from ._session_construction import (
    _config_json,
    _effective_thinking_level,
    _validate_runtime_model,
    construct_session,
)
from ._session_dispatch import _IN_LISTENER, ObserverDispatcher, _BusyState
from ._session_projection import _projection_entries, build_projection, estimate_projected_context_tokens
from .agent import Agent
from .agent_loop import PrepareRequestContext, PrepareRequestUpdate
from .cancellation import Cancellation
from .compaction import (
    CompactionAbortedError,
    CompactionPreparation,
    CompactionResult,
    _entry_timestamp,
    prepare_compaction,
    run_compaction,
)
from .config import AgentConfig
from .context import (
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_messages_tokens,
    latest_compaction_entry,
    should_compact,
)
from .error_policy import is_context_overflow, is_recoverable_length, is_retryable_error
from .events import (
    AgentEnd,
    AgentEvent,
    AgentSettled,
    AutoRetryEnd,
    AutoRetryStart,
    CompactionEnd,
    CompactionStart,
    MessageEnd,
    MessageStart,
    ModelChanged,
    QueueUpdate,
    SessionEvent,
    ThinkingLevelChanged,
)
from .permissions import PermissionPolicy, PermissionPrompt
from .provider import Provider
from .resources import ResourceSnapshot
from .session_store import (
    SessionCompactionEntry,
    SessionMessageEntry,
    SessionOmissionEntry,
    SessionStore,
    SessionUnusableError,
)
from .skills import ResourceDiagnostic, SkillInfo, SkillSettings
from .tool_hooks import _IN_TOOL_HOOK, ToolHooks
from .tools.base import Tool
from .tools.bash import BashTool
from .types import (
    AssistantMessage,
    Message,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    content_text,
    get_current_system_message,
    now_ms,
)

SessionListener = Callable[[SessionEvent], object]
_ABORT_GRACE_SECONDS = 2.0

if TYPE_CHECKING:
    from .mcp.manifest import McpBinding
    from .mcp.runtime import McpRuntime


class AgentSession:
    """One conversation: agent + journal + compaction + retry policy."""

    def __init__(
        self,
        *,
        agent: Agent,
        store: SessionStore,
        config: AgentConfig,
        provider: Provider,
        tools: Sequence[Tool],
        resources: ResourceSnapshot | None = None,
    ) -> None:
        self.agent = agent
        self.store = store
        self.config = config
        self._provider = provider
        self._tools = list(tools)
        self._resources = resources or ResourceSnapshot()
        self._observers = ObserverDispatcher()
        #: Pi's ``_entryIdsByMessage``: journal entry of each live message, by identity.
        self._entry_ids: dict[int, tuple[Message, str]] = {}
        self._pending_retry_success: int | None = None
        self._steering_texts: list[str] = []
        self._follow_up_texts: list[str] = []
        self._busy = _BusyState()
        self._active_task: asyncio.Task[None] | None = None
        self._abort_requested = False
        self._run_cancellation = Cancellation()
        self._compaction_cancellation: Cancellation | None = None
        self._retry_attempt = 0
        self._overflow_attempted = False
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._disposed = False
        self._resource_owner: ResourceOwner | None = None
        self._mcp_runtime: McpRuntime | None = None
        self._close_task: asyncio.Task[None] | None = None
        agent.set_internal_listener(self._on_agent_event)
        agent.set_post_reduce_hook(self._deliver_agent_event)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _create(
        cls,
        provider: Provider,
        *,
        model: ModelInfo | None,
        cwd: Path | str | None,
        session: Path | str | Literal["new"] | None,
        tools: Iterable[Tool] | None,
        tool_names: Sequence[str] | None,
        config: AgentConfig | None,
        refresh_instructions: bool,
        hooks: ToolHooks | None,
        permission_policy: PermissionPolicy | None,
        confirm_permission: PermissionPrompt | None,
    ) -> AgentSession:
        return construct_session(
            cls,
            provider,
            model=model,
            cwd=cwd,
            session=session,
            tools=tools,
            tool_names=tool_names,
            config=config,
            refresh_instructions=refresh_instructions,
            hooks=hooks,
            permission_policy=permission_policy,
            confirm_permission=confirm_permission,
        )

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self.store.session_id

    @property
    def session_file(self) -> Path | None:
        return self.store.session_file

    @property
    def cwd(self) -> Path:
        return self.store.cwd

    @property
    def messages(self) -> list[Message]:
        return deepcopy(self.agent.messages)

    @property
    def is_streaming(self) -> bool:
        return self._busy.activity == "prompt"

    @property
    def is_compacting(self) -> bool:
        return self._compaction_cancellation is not None

    @property
    def is_idle(self) -> bool:
        return self._busy.activity is None and not self.is_compacting and self.agent.is_idle

    @property
    def model(self) -> ModelInfo:
        return deepcopy(self.config.model)

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self.config.thinking_level

    def set_model(
        self,
        model: ModelInfo,
        *,
        provider: Provider | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> None:
        """Durably change an idle session's model and optional provider/thinking.

        A provider/API identity change requires an explicit provider instance.
        Transport functions and credentials remain runtime-only. On ordinary
        validation/write failure, runtime state stays unchanged. Thinking is
        retained unless supplied, and clamped to off for a non-reasoning model.
        This synchronous method queues observer events after the durable commit.
        """
        self._begin_configuration("set_model")
        try:
            if not isinstance(model, ModelInfo):
                raise ValueError("model must be ModelInfo")
            owned_model = ModelInfo.from_json(model.to_json())
            current = self.config.model
            if provider is None and (owned_model.provider, owned_model.api) != (
                current.provider,
                current.api,
            ):
                raise ValueError("Changing provider or API requires provider= explicitly")
            selected_provider = provider if provider is not None else self._provider
            level = _effective_thinking_level(
                owned_model, self.config.thinking_level if thinking_level is None else thinking_level
            )
            new_config = replace(self.config, model=owned_model, thinking_level=level)
            new_config.validate()
            _validate_runtime_model(selected_provider, owned_model, level)
            self._commit_configuration(new_config, selected_provider)
        finally:
            self._busy.release()
            self._idle_event.set()

    def set_thinking_level(self, level: ThinkingLevel) -> ThinkingLevel:
        """Persist a thinking level while idle, clamped to the model's supported levels as in Pi."""
        self._begin_configuration("set_thinking_level")
        try:
            effective = _effective_thinking_level(self.config.model, level)
            new_config = replace(self.config, thinking_level=effective)
            new_config.validate()
            _validate_runtime_model(self._provider, new_config.model, effective)
            self._commit_configuration(new_config, self._provider)
            return effective
        finally:
            self._busy.release()
            self._idle_event.set()

    def set_tool_execution(self, mode: Literal["sequential", "parallel"]) -> None:
        """Durably select tool scheduling while idle."""
        self._begin_configuration("set_tool_execution")
        try:
            config = replace(
                self.config,
                tool_execution=mode,
            )
            config.validate()
            self._commit_configuration(config, self._provider)
        finally:
            self._busy.release()
            self._idle_event.set()

    def _begin_configuration(self, method: str) -> None:
        self._reject_reentrant(method)
        if self._disposed:
            raise RuntimeError("Session was disposed.")
        if not self.store.usable:
            raise SessionUnusableError("Cannot configure a session after a failed durable write")
        if not self.agent.is_idle or self.is_compacting:
            raise RuntimeError("Session is busy; wait_for_idle() or abort() before configuration changes")
        self._busy.claim("configure")
        self._idle_event.clear()

    def _commit_configuration(self, config: AgentConfig, provider: Provider) -> None:
        previous = self.config
        if config != previous:
            self.store.append_config_change(_config_json(config, config.model, self.cwd))
        # No await/callback between durable append and application of the complete state.
        self.config = config
        self._provider = provider
        self.agent.provider = provider
        self.agent.model = deepcopy(config.model)
        self.agent.thinking_level = config.thinking_level
        self.agent.tool_execution = config.tool_execution
        if config.model != previous.model:
            self._observers.emit_nowait(ModelChanged(deepcopy(previous.model), deepcopy(config.model)))
        if config.thinking_level != previous.thinking_level:
            self._observers.emit_nowait(ThinkingLevelChanged(previous.thinking_level, config.thinking_level))

    @property
    def system_prompt(self) -> str:
        return self.agent.system_prompt

    @property
    def skills(self) -> tuple[SkillInfo, ...]:
        return self._resources.skills

    @property
    def resource_diagnostics(self) -> tuple[ResourceDiagnostic, ...]:
        return self._resources.diagnostics

    def reload_resources(self, settings: SkillSettings | None = None) -> tuple[ResourceDiagnostic, ...]:
        """Idle-only reload; catalog and manifest commit before runtime publication."""
        self._begin_configuration("reload_resources")
        try:
            resources = ResourceSnapshot.discover(
                settings or self.config.skills, self.cwd, self.config.agent_dir
            )
            section = resources.prompt_section({tool.name for tool in self._tools})
            self.store.append_message(
                SystemMessage(
                    content="",
                    sections={"skills": section},
                    resource_manifest=resources.to_manifest(),
                    timestamp=now_ms(),
                )
            )
            self._resources = resources
            self.config = replace(self.config, skills=resources.settings)
            self._refresh_agent_messages()
            return resources.diagnostics
        finally:
            self._busy.release()
            self._idle_event.set()

    @property
    def mcp_server_states(self) -> dict[str, str]:
        return dict(self._mcp_runtime.server_states) if self._mcp_runtime is not None else {}

    @property
    def mcp_bindings(self) -> tuple[McpBinding, ...]:
        from .mcp.manifest import McpManifest

        current = get_current_system_message(self.agent.messages)
        return (
            McpManifest.from_manifest(current.mcp_manifest).bindings
            if current is not None and current.mcp_manifest is not None
            else ()
        )

    async def reconnect_mcp(self, server: str) -> None:
        """Idle-only explicit reconnection; recorded identities/schemas must match."""
        if type(server) is not str or not server:
            raise ValueError("MCP reconnect requires a server name")
        from .mcp.session import update_mcp_tools

        await update_mcp_tools(self, server=server)

    async def reload_mcp(self, mcp_tools: Sequence[str] | None = None) -> None:
        """Explicitly refresh current bindings, select replacements, or ['*'] for all."""
        from .mcp.session import update_mcp_tools

        await update_mcp_tools(self, selectors=mcp_tools)

    @property
    def stats(self) -> dict[str, object]:
        entries = self.store.entries
        user_messages = assistant_messages = tool_results = tool_calls = 0
        for entry in entries:
            if not isinstance(entry, SessionMessageEntry):
                continue
            message = entry.message
            if isinstance(message, UserMessage):
                user_messages += 1
            elif isinstance(message, AssistantMessage):
                assistant_messages += 1
                tool_calls += sum(1 for block in message.content if isinstance(block, ToolCallBlock))
            elif isinstance(message, ToolResultMessage):
                tool_results += 1
        return {
            "session_id": self.session_id,
            "session_file": str(self.session_file) if self.session_file else None,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "entries": len(entries),
        }

    # ------------------------------------------------------------------
    # Subscription and event plumbing
    # ------------------------------------------------------------------

    def subscribe(self, listener: SessionListener) -> Callable[[], None]:
        """Subscribe to ordered observations; see docs/invariants.md#lifecycle-and-event-delivery."""
        return self._observers.subscribe(listener)

    # ------------------------------------------------------------------
    # Ordered observer dispatch
    # ------------------------------------------------------------------

    def _emit_queue_update(self) -> None:
        self._observers.emit_nowait(
            QueueUpdate(steering=list(self._steering_texts), follow_up=list(self._follow_up_texts))
        )

    def _reject_reentrant(self, method: str) -> None:
        reject_cleanup_reentry(self, method)
        # Task-local: only calls originating inside listener/hook delivery
        # (or tasks a callback created, which inherit the context)
        # are rejected; unrelated external tasks are unaffected.
        if _IN_LISTENER.get() or _IN_TOOL_HOOK.get():
            raise RuntimeError(
                f"{method}() must not be called re-entrantly from inside an event listener or tool hook; "
                "the active operation cannot finish while the callback is awaiting it"
            )

    async def _deliver_agent_event(self, event: AgentEvent, cancellation: Cancellation) -> None:
        """Post-reduce dispatch hook: awaited, ordered observer delivery.

        Runs after persistence and state reduction, awaited before the run
        advances — this bounds queued stream updates (a slow listener
        throttles the run instead of accumulating snapshots) and guarantees
        that input an observer enqueues during ``agent_end`` delivery is
        visible before the post-run handling decides whether to continue.
        """
        del cancellation
        if isinstance(event, AgentEnd):
            await self._observers.emit(
                AgentEnd(messages=event.messages, will_retry=self._will_retry_after_agent_end())
            )
            return
        await self._observers.emit(event)
        retry_success = self._pending_retry_success
        if (
            retry_success is not None
            and isinstance(event, MessageEnd)
            and isinstance(event.message, AssistantMessage)
        ):
            self._pending_retry_success = None
            await self._observers.emit(AutoRetryEnd(success=True, attempt=retry_success))

    async def _on_agent_event(self, event: AgentEvent, cancellation: Cancellation) -> None:
        """Persistence and queue bookkeeping only (stage 1).

        The durable write happens first (fatal on failure, wrapped by the
        Agent as FatalPersistenceError). Observer delivery does NOT happen
        here: the agent reduces state next and then awaits the post-reduce
        hook, which delivers to session observers — so observers always see
        reduced state, delivery is awaited (bounded), and an observer
        failure can never masquerade as a persistence failure.
        """
        del cancellation
        if isinstance(event, MessageStart) and isinstance(event.message, UserMessage):
            self._overflow_attempted = False
            text = content_text(event.message.content)
            if text and text in self._steering_texts:
                self._steering_texts.remove(text)
                self._emit_queue_update()
            elif text and text in self._follow_up_texts:
                self._follow_up_texts.remove(text)
                self._emit_queue_update()

        if isinstance(event, MessageEnd):
            entry_id = self.store.append_message(event.message)  # fatal on write failure
            self._entry_ids[id(event.message)] = (event.message, entry_id)
            if isinstance(event.message, AssistantMessage):
                assistant = event.message
                if assistant.stop_reason not in ("error", "length"):
                    self._overflow_attempted = False
                if assistant.stop_reason not in ("error", "aborted") and self._retry_attempt > 0:
                    # The retry-success event is emitted by the post-reduce
                    # hook after the assistant message itself is delivered.
                    self._pending_retry_success = self._retry_attempt
                    self._retry_attempt = 0

    def _will_retry_after_agent_end(self) -> bool:
        """Pi's ``_willRetryAfterAgentEnd``."""
        if self._abort_requested:
            return False
        retry = self.config.retry
        if not retry.enabled or self._retry_attempt >= retry.max_retries:
            return False
        for message in reversed(self.agent.messages):
            if isinstance(message, AssistantMessage):
                return self._is_retryable_error(message)
        return False

    def _model_for_message(self, message: AssistantMessage) -> ModelInfo | None:
        """Pi's ``_modelForMessage``: the current model when it produced ``message``."""
        model = self.config.model
        return model if (model.provider, model.id) == (message.provider, message.model) else None

    def _is_retryable_error(self, message: AssistantMessage) -> bool:
        model = self._model_for_message(message) or self.config.model
        return is_retryable_error(message, model.context_window)

    # ------------------------------------------------------------------
    # Steering / follow-up
    # ------------------------------------------------------------------

    def steer(self, text: str) -> None:
        """Queue steering input; injected after the current complete tool batch.

        Steering never cancels the in-flight batch; the loop delivers it
        between turns, so tool call/result pairing stays valid.
        """
        if self._disposed:
            raise RuntimeError("Session was disposed.")
        text = self._resources.expand(text)
        self._steering_texts.append(text)
        self.agent.steer(text)
        self._emit_queue_update()

    def follow_up(self, text: str) -> None:
        """Queue follow-up input; processed once the agent would otherwise stop."""
        if self._disposed:
            raise RuntimeError("Session was disposed.")
        text = self._resources.expand(text)
        self._follow_up_texts.append(text)
        self.agent.follow_up(text)
        self._emit_queue_update()

    def clear_queue(self) -> dict[str, list[str]]:
        if self._disposed:
            raise RuntimeError("Session was disposed.")
        steering = list(self._steering_texts)
        follow_up = list(self._follow_up_texts)
        self._steering_texts.clear()
        self._follow_up_texts.clear()
        self.agent.clear_all_queues()
        self._emit_queue_update()
        return {"steering": steering, "follow_up": follow_up}

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    async def prompt(self, text: str, streaming_behavior: Literal["steer", "followUp"] | None = None) -> None:
        """Send a prompt; while streaming, queue it per ``streaming_behavior``."""
        self._reject_reentrant("prompt")
        # Claim synchronously before any await so concurrent prompts see it.
        if self._busy.activity is not None:
            if self._busy.activity == "prompt" and streaming_behavior is not None:
                if streaming_behavior == "followUp":
                    self.follow_up(text)
                else:
                    self.steer(text)
                return
            raise RuntimeError(
                f"Session is busy with an ongoing {self._busy.activity}. Use wait_for_idle() or abort()."
            )
        if self._disposed:
            raise RuntimeError("Session was disposed.")
        text = self._resources.expand(text)
        self._busy.claim("prompt")
        self._active_task = asyncio.current_task()
        try:
            self._abort_requested = False
            self._run_cancellation = Cancellation()
            self._overflow_attempted = False
            self._idle_event.clear()
            # Pi's _prepareRun: check compaction before sending (catches aborted
            # responses too) inside the claim; an abort during it must not start
            # a new model run. The prompt itself is sent below, so no retry here.
            last_assistant = self._last_assistant_message()
            if last_assistant is not None:
                await self._check_compaction(last_assistant, skip_aborted_check=False)
            if self._abort_requested:
                return
            user_message = UserMessage(content=[TextBlock(text=text)], timestamp=now_ms())
            await self._run_agent_prompt([user_message])
        finally:
            try:
                self._finish_cancelled_retry()
                self._refresh_agent_messages()
                # Settled is queued before settling so the final callback is
                # included; wait_for_idle only returns once it has run.
                self._emit_settled()
                await self._observers.settle()
            finally:
                # Release and settle unconditionally: a failing recovery
                # write must not leave the session permanently busy.
                self._busy.release()
                self._active_task = None
                self._idle_event.set()

    async def _run_agent_prompt(self, messages: Sequence[Message]) -> None:
        try:
            await self.agent.prompt(messages)
            while not self._abort_requested:
                if not await self._handle_post_run():
                    break
                if self._abort_requested:
                    break
                last = self.agent.messages[-1] if self.agent.messages else None
                if isinstance(last, AssistantMessage):
                    # Input queued after the final answer (e.g. by observers
                    # during settlement): deliver it through a fresh prompt
                    # run instead of discarding it.
                    queued = self.agent.drain_queued_messages()
                    if not queued:
                        break
                    await self.agent.prompt(queued)
                    continue
                await self.agent.continue_()
        except asyncio.CancelledError:
            self._abort_requested = True
            raise

    def _last_assistant_message(self) -> AssistantMessage | None:
        for message in reversed(self.agent.messages):
            if isinstance(message, AssistantMessage):
                return message
        return None

    async def _handle_post_run(self) -> bool:
        """Pi's ``_handlePostAgentRun``: retry, then overflow/threshold compaction.

        Continuation happens only for a retry, an overflow recovery, or
        genuinely queued input — never merely because a threshold compaction
        succeeded after a completed final answer.
        """
        if self._abort_requested:
            return False
        message = self._last_assistant_message()
        if message is None:
            return self.agent.has_queued_messages()

        if self._is_retryable_error(message) and await self._prepare_retry(message):
            return not self._abort_requested
        if self._abort_requested:
            return False

        if message.stop_reason == "error" and self._retry_attempt > 0:
            await self._observers.emit(
                AutoRetryEnd(success=False, attempt=self._retry_attempt, final_error=message.error_message)
            )
            self._retry_attempt = 0

        if await self._check_compaction(message):
            return not self._abort_requested
        return not self._abort_requested and self.agent.has_queued_messages()

    async def _prepare_retry(self, failed: AssistantMessage) -> bool:
        """Pi's ``_prepareRetry``: bounded attempts with cancellable backoff."""
        retry = self.config.retry
        if not retry.enabled:
            return False
        self._retry_attempt += 1
        if self._retry_attempt > retry.max_retries:
            # Keep the completed attempt count so post-run handling reports the failure.
            self._retry_attempt -= 1
            return False
        delay_ms = retry.delay_ms(self._retry_attempt)
        await self._observers.emit(
            AutoRetryStart(
                attempt=self._retry_attempt,
                max_attempts=retry.max_retries,
                delay_ms=delay_ms,
                error_message=failed.error_message or "Unknown error",
            )
        )
        self._omit_recovery_attempt(failed)
        try:
            await asyncio.wait_for(self._run_cancellation.wait(), delay_ms / 1000)
            self._finish_cancelled_retry()
            return False
        except TimeoutError:
            return not self._abort_requested
        except asyncio.CancelledError:
            self._abort_requested = True
            self._finish_cancelled_retry()
            raise

    def _omit_recovery_attempt(self, failed: AssistantMessage) -> None:
        """Pi's ``_omitRecoveryAttempt``: durably omit a failed attempt and its tool results.

        Raw history keeps them. Fatal on persistence failure.
        """
        call_ids = {block.id for block in failed.content if isinstance(block, ToolCallBlock)}
        messages = self.agent.messages
        index = next((i for i, message in enumerate(messages) if message is failed), -1)
        targets: list[Message] = [failed]
        for message in [] if index < 0 else messages[index + 1 :]:
            if isinstance(message, AssistantMessage):
                break
            if isinstance(message, ToolResultMessage) and message.tool_call_id in call_ids:
                targets.append(message)
        target_ids = [self._entry_id_for_message(target) for target in targets]
        if any(
            target_id is None and any(target is message for message in messages)
            for target, target_id in zip(targets, target_ids, strict=True)
        ):
            raise RuntimeError(
                "Cannot persist recovery omission because a projected message has no source entry"
            )
        for target_id in target_ids:
            if target_id is not None:
                self.store.append_context_omission(target_id)
        self._refresh_agent_messages()

    def _entry_id_for_message(self, message: Message) -> str | None:
        known = self._entry_ids.get(id(message))
        return known[1] if known is not None and known[0] is message else None

    def _finish_cancelled_retry(self) -> None:
        if self._retry_attempt == 0:
            return
        attempt = self._retry_attempt
        self._retry_attempt = 0
        self._observers.emit_nowait(
            AutoRetryEnd(success=False, attempt=attempt, final_error="Retry cancelled")
        )

    # ------------------------------------------------------------------
    # Abort / idle / disposal
    # ------------------------------------------------------------------

    async def abort(self) -> None:
        """Abort the current run (and compaction), then wait for idle."""
        self._reject_reentrant("abort")
        if self._busy.activity is not None:
            self._abort_requested = True
            self._run_cancellation.abort()
        if self._compaction_cancellation is not None:
            self._compaction_cancellation.abort()
        self.agent.abort()
        try:
            async with asyncio.timeout(_ABORT_GRACE_SECONDS):
                await self.wait_for_idle()
        except TimeoutError:
            task = self._active_task
            if task is not None and task is not asyncio.current_task() and not task.cancelling():
                task.cancel()
            await self.wait_for_idle()

    async def wait_for_idle(self) -> None:
        self._reject_reentrant("wait_for_idle")
        await self._idle_event.wait()
        await self.agent.wait_for_idle()
        await self._observers.settle()

    def dispose(self) -> None:
        """Synchronous disposal for core sessions; async owners require aclose()."""
        self._reject_reentrant("dispose")
        if not self.is_idle:
            raise RuntimeError(
                f"Cannot dispose while a {self._busy.activity or 'agent run/compaction'} is active; abort() first."
            )
        if self._resource_owner is not None and self._resource_owner.pending:
            raise RuntimeError("Session owns asynchronous resources; use await session.aclose()")
        self._detach()
        errors = self._close_sync_resources()
        if errors:
            raise ExceptionGroup("Session cleanup failed; dispose() can retry cleanup", errors)

    def _detach(self) -> None:
        self._disposed = True
        self.agent.set_internal_listener(None)
        self.agent.set_post_reduce_hook(None)
        self._observers.listeners.clear()
        self._observers.close()
        self._entry_ids.clear()

    def _close_sync_resources(self) -> list[Exception]:
        errors: list[Exception] = []
        for tool in self._tools:
            if isinstance(tool, BashTool):
                try:
                    tool.close()
                except Exception as error:
                    errors.append(error)
        try:
            self.store.close()
        except Exception as error:
            errors.append(error)
        return errors

    async def aclose(self) -> None:
        """Close an idle session, join tasks, and retry unfinished cleanup.

        Call abort() first for active work. Concurrent close waiters share one
        cleanup; cancellation of a waiter propagates only after cleanup settles.
        Borrowed providers and custom tools are never closed implicitly.
        """
        self._reject_reentrant("aclose")
        task = self._close_task
        if task is asyncio.current_task():
            raise RuntimeError("Session cleanup cannot await itself")
        if task is None or task.done():
            if not self.is_idle:
                raise RuntimeError("Cannot close while session work is active; abort() first")
            self._busy.claim("close")
            self._idle_event.clear()
            self._detach()
            task = asyncio.create_task(self._close_async_resources())
            self._close_task = task
        await settle_cleanup(task)

    async def _close_async_resources(self) -> None:
        token = _CLEANING.set((*_CLEANING.get(), self))
        errors: list[Exception] = []
        try:
            await self._observers.aclose()
            if self._resource_owner is not None:
                try:
                    await self._resource_owner.aclose()
                except Exception as error:
                    errors.append(error)
        finally:
            errors.extend(self._close_sync_resources())
            self._busy.release()
            self._idle_event.set()
            _CLEANING.reset(token)
        if errors:
            raise ExceptionGroup("Session cleanup failed; aclose() can retry cleanup", errors)

    async def __aenter__(self) -> AgentSession:
        if self._disposed:
            raise RuntimeError("Session was disposed.")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if not self.is_idle:
            await self.abort()
        await self.aclose()

    def _emit_settled(self) -> None:
        self._observers.emit_nowait(AgentSettled(aborted=self._abort_requested))

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        """Manual compaction. Aborts the current run first; never retries it."""
        self._reject_reentrant("compact")
        if self._busy.activity is not None:
            if self._busy.activity == "prompt":
                await self.abort()
            else:
                raise RuntimeError("A compaction is already in progress.")
        if self._disposed:
            raise RuntimeError("Session was disposed.")
        self._busy.claim("compact")
        self._active_task = asyncio.current_task()
        self._abort_requested = False
        self._run_cancellation = Cancellation()
        self._compaction_cancellation = Cancellation()
        self._idle_event.clear()
        try:
            await self._observers.emit(CompactionStart(reason="manual"))
            preparation = self._prepare_compaction()
            if preparation is None:
                if isinstance(self.store.last_entry, SessionCompactionEntry):
                    raise RuntimeError("Already compacted")
                raise RuntimeError("Nothing to compact (session too small)")
            result = await self._run_compaction_core(
                preparation,
                custom_instructions=custom_instructions,
                cancellation=self._compaction_cancellation,
            )
            await self._observers.emit(CompactionEnd(reason="manual", result=result, aborted=False))
            return result
        except asyncio.CancelledError:
            self._abort_requested = True
            await self._observers.emit(CompactionEnd(reason="manual", result=None, aborted=True))
            raise
        except Exception as error:  # noqa: BLE001 - reported through the event, then raised
            aborted = isinstance(error, CompactionAbortedError) or (
                self._compaction_cancellation is not None and self._compaction_cancellation.aborted
            )
            message = None if aborted else f"Compaction failed: {error}"
            await self._observers.emit(
                CompactionEnd(reason="manual", result=None, aborted=aborted, error_message=message)
            )
            raise
        finally:
            try:
                self._compaction_cancellation = None
                self._refresh_agent_messages()
                self._emit_settled()
                await self._observers.settle()
            finally:
                self._busy.release()
                self._active_task = None
                self._idle_event.set()

    async def _run_auto_compaction(self, reason: Literal["threshold", "overflow"], will_retry: bool) -> bool:
        """Pi's ``_runAutoCompaction``; reports whether the run should continue.

        Nothing to compact is silent. Contracts and failure semantics:
        docs/invariants.md.
        """
        preparation = self._prepare_compaction()
        if preparation is None:
            return False
        cancellation = Cancellation()
        self._compaction_cancellation = cancellation
        try:
            await self._observers.emit(CompactionStart(reason=reason))
            result = await self._run_compaction_core(
                preparation, custom_instructions=None, cancellation=cancellation
            )
            await self._observers.emit(
                CompactionEnd(reason=reason, result=result, aborted=False, will_retry=will_retry)
            )
        except (SessionUnusableError, OSError) as error:
            # Durable write failed: the store is poisoned. Report, then let
            # the fatal error propagate out of the run.
            await self._observers.emit(
                CompactionEnd(
                    reason=reason,
                    result=None,
                    aborted=False,
                    error_message=f"Compaction persistence failure: {error}",
                )
            )
            raise
        except Exception as error:  # noqa: BLE001 - auto compaction reports, not raises
            aborted = isinstance(error, CompactionAbortedError) or cancellation.aborted
            prefix = "Context overflow recovery" if reason == "overflow" else "Auto-compaction"
            message = None if aborted else f"{prefix} failed: {error}"
            await self._observers.emit(
                CompactionEnd(reason=reason, result=None, aborted=aborted, error_message=message)
            )
            return False
        finally:
            self._compaction_cancellation = None
        if will_retry:
            return True
        # Queued steering/follow-up input is delivered by one more run.
        return self.agent.has_queued_messages()

    def _prepare_compaction(self) -> CompactionPreparation | None:
        """Pi's ``prepareCompaction`` over the current journal."""
        if isinstance(self.store.last_entry, SessionCompactionEntry):
            return None
        return prepare_compaction(
            build_projection(self.store),
            tokens_before=self._projection_tokens(),
            settings=self.config.compaction,
        )

    async def _run_compaction_core(
        self,
        preparation: CompactionPreparation,
        *,
        custom_instructions: str | None,
        cancellation: Cancellation,
    ) -> CompactionResult:
        if cancellation.aborted:
            raise CompactionAbortedError()
        result = await run_compaction(
            preparation,
            provider=self._provider,
            model=self.config.model,
            cancellation=cancellation,
            thinking_level=self.config.thinking_level,
            custom_instructions=custom_instructions,
            retry=self.config.retry,
        )
        if cancellation.aborted:
            raise CompactionAbortedError()

        system_message = get_current_system_message(self.agent.messages)
        self.store.append_compaction(
            result.summary,
            result.first_kept_entry_id,
            result.tokens_before,
            usage=result.usage,
            system_message=system_message,
            read_files=result.read_files,
            modified_files=result.modified_files,
        )
        self._refresh_agent_messages()
        result.estimated_tokens_after = estimate_messages_tokens(self._projection_messages())
        return result

    async def _prepare_request(self, context: PrepareRequestContext) -> PrepareRequestUpdate | None:
        """Pi's request projection: every provider request leaves from the journal projection."""
        del context
        projection_messages = self._projection_messages()
        self.agent.messages = projection_messages
        return PrepareRequestUpdate(messages=projection_messages)

    async def _prepare_next_turn(self, context: PrepareRequestContext) -> PrepareRequestUpdate | None:
        """Pi's ``_compactBeforeNextAssistantResponse``: threshold check between turns."""
        del context
        if self._exceeds_compaction_threshold():
            await self._run_auto_compaction("threshold", False)
        projection_messages = self._projection_messages()
        self.agent.messages = projection_messages
        return PrepareRequestUpdate(messages=projection_messages)

    def _projection_has_omissions(self) -> bool:
        """Whether an omission falls in the projected journal range (Pi's projected context edits)."""
        journal = self.store._entries
        compaction = latest_compaction_entry(journal)
        start = self.store._entry_positions[compaction.first_kept_entry_id] if compaction is not None else 0
        return any(isinstance(entry, SessionOmissionEntry) for entry in journal[start:])

    def _exceeds_compaction_threshold(self) -> bool:
        window = self.config.model.context_window
        if window <= 0:
            return False
        return should_compact(self._projection_tokens(), window, self.config.compaction)

    async def _check_compaction(self, message: AssistantMessage, skip_aborted_check: bool = True) -> bool:
        """Pi's ``_checkCompaction``: overflow recovery, then threshold compaction.

        Returns whether the post-run loop should continue (an overflow
        retry, or queued input after a compaction).
        """
        settings = self.config.compaction
        if not settings.enabled:
            return False
        if skip_aborted_check and message.stop_reason == "aborted":
            return False

        # An overflow from another model must not compact for the current one.
        message_model = self._model_for_message(message)
        same_model = message_model is not None
        context_window = (message_model or self.config.model).context_window

        # Stale pre-compaction usage or errors must not retrigger compaction.
        compaction = latest_compaction_entry(self.store._entries)
        compaction_time = _entry_timestamp(compaction) if compaction is not None else None
        if compaction_time is not None and message.timestamp <= compaction_time:
            return False

        entry_id = self._entry_id_for_message(message)
        projection = _projection_entries(self.store)
        is_projected = entry_id is None or any(
            entry.entry_id == entry_id and any(isinstance(m, AssistantMessage) for m in entry.messages)
            for entry in projection
        )
        position = self.store._entry_positions.get(entry_id) if entry_id is not None else None
        later = self.store._entries[position + 1 :] if position is not None else []
        has_later_omission = any(isinstance(entry, SessionOmissionEntry) for entry in later)
        retained_for_explicit_recovery = entry_id is None or (
            not any(isinstance(entry, SessionCompactionEntry) for entry in later)
            and not any(
                isinstance(entry, SessionOmissionEntry) and entry.target_entry_id == entry_id
                for entry in later
            )
        )
        usage_matches_projection = is_projected and not has_later_omission
        explicit_overflow = message.stop_reason == "error" and is_context_overflow(message)
        context_overflow = same_model and (
            (explicit_overflow and retained_for_explicit_recovery)
            or (usage_matches_projection and is_context_overflow(message, context_window))
        )
        recoverable_length = (
            message_model is not None
            and is_projected
            and is_recoverable_length(message, message_model.max_tokens)
        )
        if context_overflow or recoverable_length:
            will_retry = message.stop_reason != "stop"
            # A completed response is compacted but not retried: there is nothing to continue.
            if not will_retry:
                return await self._run_auto_compaction("overflow", False)
            if self._overflow_attempted:
                error_message = (
                    "Context overflow recovery failed after one compact-and-retry attempt. Try reducing "
                    "context or switching to a larger-context model."
                    if context_overflow
                    else "Truncated response recovery failed after one compact-and-retry attempt."
                )
                await self._observers.emit(
                    CompactionEnd(reason="overflow", result=None, aborted=False, error_message=error_message)
                )
                return False
            # Durably omit the failed attempt so the retry leaves from the last
            # user/tool-result message, then compact and re-request.
            self._overflow_attempted = True
            self._omit_recovery_attempt(message)
            return await self._run_auto_compaction("overflow", True)

        # Threshold compaction without retry. Error and zero-usage responses
        # are sized from the last valid response instead.
        direct_tokens = calculate_context_tokens(message.usage) if message.usage else 0
        if self._projection_has_omissions():
            context_tokens = self._projection_tokens()
        elif message.stop_reason == "error" or direct_tokens == 0:
            messages = self.agent.messages
            estimate = estimate_context_tokens(messages)
            if estimate.last_usage_index is not None and compaction_time is not None:
                usage_message = messages[estimate.last_usage_index]
                if isinstance(usage_message, AssistantMessage) and usage_message.timestamp <= compaction_time:
                    return False
            context_tokens = estimate.tokens
        else:
            context_tokens = direct_tokens
        if should_compact(context_tokens, context_window, settings):
            return await self._run_auto_compaction("threshold", False)
        return False

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _projection_messages(self) -> list[Message]:
        messages: list[Message] = []
        for entry in build_projection(self.store):
            for message in entry.messages:
                self._entry_ids[id(message)] = (message, entry.entry_id)
                messages.append(message)
        return messages

    def _projection_tokens(self) -> int:
        """Pi's ``estimateProjectedContextTokens``; see ``_session_projection``."""
        return estimate_projected_context_tokens(self.store)

    def _refresh_agent_messages(self) -> None:
        """Between runs the projection is authoritative; during an active run
        the loop owns the transcript and ``_prepare_request`` refreshes it
        before each provider request."""
        if not self.agent.is_streaming:
            self.agent.messages = self._projection_messages()


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def create_agent_session(
    provider: Provider,
    *,
    model: ModelInfo | None = None,
    cwd: Path | str | None = None,
    session: Path | str | Literal["new"] | None = None,
    tools: Iterable[Tool] | None = None,
    tool_names: Sequence[str] | None = None,
    config: AgentConfig | None = None,
    refresh_instructions: bool = False,
    hooks: ToolHooks | None = None,
    permission_policy: PermissionPolicy | None = None,
    confirm_permission: PermissionPrompt | None = None,
) -> AgentSession:
    """Public entry point (Pi's ``createAgentSession``-style facade).

    Contracts and failure semantics: docs/invariants.md.
    """
    return AgentSession._create(
        provider,
        model=model,
        cwd=cwd,
        session=session,
        tools=tools,
        tool_names=tool_names,
        config=config,
        refresh_instructions=refresh_instructions,
        hooks=hooks,
        permission_policy=permission_policy,
        confirm_permission=confirm_permission,
    )


__all__ = [
    "AgentSession",
    "SessionStore",
    "build_projection",
    "create_agent_session",
]
