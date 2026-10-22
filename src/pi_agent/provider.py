"""Injected streaming provider contract shared by conversation and compaction."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .cancellation import Cancellation
from .events import AssistantStreamEvent
from .types import Message, ModelInfo, ThinkingLevel, ToolDeclaration


@dataclass
class ProviderOptions:
    """Per-request options passed alongside the transcript."""

    cancellation: Cancellation | None = None
    thinking_level: ThinkingLevel = "off"
    max_tokens: int | None = None
    session_id: str | None = None


@dataclass
class ProviderRequest:
    """One model request: identity, projected transcript, tool declarations."""

    model: ModelInfo
    messages: list[Message] = field(default_factory=list)
    tools: list[ToolDeclaration] = field(default_factory=list)


@runtime_checkable
class Provider(Protocol):
    """Stream one response as start, snapshot updates, then done or error.

    Done reasons match the final message. Request failures yield StreamError
    with partial content and error_message; cancellation uses aborted.
    Implement an async generator so early closure releases resources.
    Read the system prompt from request.messages and declarations from tools.
    """

    def stream(
        self,
        request: ProviderRequest,
        options: ProviderOptions,
    ) -> AsyncIterator[AssistantStreamEvent]: ...


@runtime_checkable
class ModelValidator(Protocol):
    """Optional synchronous local check before a durable model/thinking change.

    Must not perform network I/O or mutate the session. Custom providers may
    omit this interface; the host then owns model/transport compatibility.
    """

    def validate_model(self, model: ModelInfo, thinking_level: ThinkingLevel) -> None: ...
