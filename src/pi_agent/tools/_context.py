"""Per-execution tool context (Pi's ``ctx.model``), isolated across concurrent calls."""

from contextvars import ContextVar

from pi_agent.types import ModelInfo

current_model: ContextVar[ModelInfo | None] = ContextVar("tool_current_model", default=None)
