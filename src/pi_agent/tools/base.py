"""Tool contract shared by the agent loop and the built-in tools.

Translated from ``packages/agent/src/types.ts`` (AgentTool / AgentToolResult)
and ``packages/ai/src/utils/validation.ts`` (argument validation).

Argument validation follows Pi's ``validateToolArguments`` (see
``_validation``): optional nulls are dropped, primitives are coerced like
TypeBox ``Value.Convert`` (built-in tools) or ``coerceWithJsonSchema``
(plain JSON schemas), and failures use TypeBox's messages.
"""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pi_agent.cancellation import Cancellation
from pi_agent.tools._validation import ArgumentValidationError, ConversionMode, validate_arguments
from pi_agent.types import ImageBlock, JsonValue, TextBlock, ToolDeclaration, Usage

if TYPE_CHECKING:
    from pi_agent.tool_hooks import ToolProvenance


def error_tool_result(message: str) -> ToolResult:
    """The error result shape used for unknown tools, invalid arguments, and exceptions."""
    return ToolResult(content=[TextBlock(text=message)], details={}, is_error=True)


@dataclass
class ToolResult:
    """Final or partial result of one tool execution."""

    content: list[TextBlock | ImageBlock] = field(default_factory=list)
    details: JsonValue = None
    structured_content: JsonValue = None
    usage: Usage | None = None
    is_error: bool = False
    terminate: bool = False

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "content": [block.to_json() for block in self.content],
            "details": self.details,
            "is_error": self.is_error,
        }
        if self.structured_content is not None:
            data["structured_content"] = self.structured_content
        if self.usage is not None:
            data["usage"] = self.usage.to_json()
        if self.terminate:
            data["terminate"] = True
        return data


#: Partial-result callback tools invoke while streaming execution updates.
#: Async on purpose: the agent loop awaits it so updates interleave with the
#: tool's own awaits in emission order.
ToolUpdateCallback = Callable[["ToolResult"], Awaitable[None]]


class Tool(abc.ABC):
    """Base class of every executable tool.

    Class attributes describe the model-facing declaration; ``execute`` runs a
    validated call. Tools must never execute before the loop hands them final,
    schema-validated arguments — the loop guarantees that ordering.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, JsonValue] | None = None
    #: One-line snippet for the system prompt's tool list.
    prompt_snippet: str | None = None
    #: Guideline bullets contributed to the system prompt's rules section.
    guidelines: tuple[str, ...] = ()
    #: Pi's ``executionMode``: a call to a "sequential" tool runs its whole
    #: batch sequentially; None follows the agent's tool_execution setting.
    execution_mode: Literal["sequential", "parallel"] | None = None
    #: Host-only identity for hooks/permissions; never part of the declaration.
    provenance: ToolProvenance | None = None
    #: How arguments are coerced before validation: Pi's built-in tools use
    #: TypeBox schemas (``Value.Convert``); other tools use plain JSON schemas.
    argument_conversion: ConversionMode = "json_schema"

    def prepare_arguments(self, args: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Repair raw model arguments before validation (Pi's ``prepareArguments``)."""
        return args

    def schema(self) -> dict[str, JsonValue]:
        return self.parameters or {"type": "object"}

    def declaration(self) -> ToolDeclaration:
        return ToolDeclaration(name=self.name, description=self.description, parameters=self.schema())

    def validate_arguments(self, args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Host adapters may impose stronger admission budgets and diagnostics."""
        return _validate_tool_arguments(self, args)

    @abc.abstractmethod
    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        """Execute one call. Raise on failure; the loop converts that into an
        ``is_error`` tool result."""


class ToolValidationError(ValueError):
    """Raised when tool-call arguments do not match the tool's schema."""


def validate_tool_arguments(tool: Tool, args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Validate (and return) a copy of the arguments against the tool schema.

    Raises :class:`ToolValidationError` with a Pi-shaped message on failure.
    """
    return tool.validate_arguments(args)


def _validate_tool_arguments(tool: Tool, args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    try:
        value = validate_arguments(tool.name, tool.schema(), dict(args), mode=tool.argument_conversion)
    except ArgumentValidationError as error:
        raise ToolValidationError(str(error)) from None
    return value  # type: ignore[return-value]


def resolve_path(path: str, cwd: Path | str) -> Path:
    """Resolve a tool path: ``~`` expands, absolute paths win, anything else is
    relative to ``cwd``.

    The cwd is a base directory, not a sandbox: resolved paths may escape it.
    This function never changes the process working directory.
    """
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return Path(cwd).joinpath(expanded).resolve()
