"""Validated SDK results with the original server envelope for durable output."""

from copy import deepcopy

from mcp.types import CallToolResult
from pydantic import PrivateAttr

from pi_agent.types import JsonValue


class WireCallToolResult(CallToolResult):
    _wire_result: dict[str, JsonValue] = PrivateAttr()

    @classmethod
    def from_wire(cls, value: dict[str, JsonValue]) -> "WireCallToolResult":
        result = cls.model_validate(value, strict=True, by_name=False)
        result._wire_result = deepcopy(value)
        return result

    @property
    def wire_result(self) -> dict[str, JsonValue]:
        return self._wire_result
