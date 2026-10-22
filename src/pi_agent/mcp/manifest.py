"""Host-only MCP identities for journals; never serialize runtime configuration.

The parent stores this JSON string separately from the model-facing prompt and
checks restored bindings against the current adapters before exposing tools.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from ._contracts import binding_name

if TYPE_CHECKING:
    from pi_agent.tools.base import Tool
    from pi_agent.types import ToolDeclaration

__all__ = ["McpBinding", "McpManifest", "declaration_digest"]

_MAX_BYTES = 1024 * 1024
_MAX_BINDINGS = 1024
_MAX_FIELD_NAME = 64
_SERVER_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_FIELDS = frozenset({"name", "server", "tool", "schema_digest", "declaration_digest"})


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        raise ValueError("Invalid MCP manifest JSON value") from None


def declaration_digest(tool: Tool) -> str:
    """SHA-256 of the complete canonical model-facing declaration, including prose."""
    text = _canonical_json(tool.declaration().to_json())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class McpBinding:
    name: str
    server: str
    tool: str
    schema_digest: str
    declaration_digest: str

    def __post_init__(self) -> None:
        if type(self.server) is not str or not _SERVER_NAME.fullmatch(self.server):
            raise ValueError("Invalid MCP binding server name")
        if type(self.tool) is not str or not self.tool or len(self.tool) > 256:
            raise ValueError("Invalid MCP binding tool name")
        try:
            tool_bytes = len(self.tool.encode("utf-8"))
        except UnicodeError:
            raise ValueError("Invalid MCP binding tool name") from None
        if tool_bytes > 256:
            raise ValueError("Invalid MCP binding tool name")
        if (
            type(self.name) is not str
            or len(self.name) > 64
            or self.name != binding_name(self.server, self.tool)
        ):
            raise ValueError("Invalid MCP binding name")
        for digest in (self.schema_digest, self.declaration_digest):
            if type(digest) is not str or not _DIGEST.fullmatch(digest):
                raise ValueError("Invalid MCP binding digest")


@dataclass(frozen=True, slots=True)
class McpManifest:
    bindings: tuple[McpBinding, ...] = ()

    def __post_init__(self) -> None:
        if type(self.bindings) is not tuple or len(self.bindings) > _MAX_BINDINGS:
            raise ValueError("Invalid MCP bindings; maximum is 1024")
        names: set[str] = set()
        identities: set[tuple[str, str]] = set()
        for binding in self.bindings:
            if type(binding) is not McpBinding:
                raise ValueError("Invalid MCP binding")
            identity = (binding.server, binding.tool)
            if binding.name in names or identity in identities:
                raise ValueError("Duplicate MCP binding name or server/tool identity")
            names.add(binding.name)
            identities.add(identity)
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings, key=lambda binding: binding.name)))
        self.to_manifest()

    def to_manifest(self) -> str:
        """Encode the strict version 1 envelope with deterministic JSON keys."""
        text = _canonical_json({"version": 1, "bindings": [asdict(binding) for binding in self.bindings]})
        if len(text.encode("utf-8")) > _MAX_BYTES:
            raise ValueError("MCP manifest exceeds 1 MiB")
        return text

    def validate_declarations(self, declarations: Sequence[ToolDeclaration]) -> None:
        """Check that identity metadata and the active journal declarations agree."""
        by_name = {declaration.name: declaration for declaration in declarations}
        for binding in self.bindings:
            declaration = by_name.get(binding.name)
            if (
                declaration is None
                or hashlib.sha256(_canonical_json(declaration.to_json()).encode("utf-8")).hexdigest()
                != binding.declaration_digest
            ):
                raise ValueError("MCP manifest does not match the active tool declarations")

    @classmethod
    def from_manifest(cls, text: str) -> McpManifest:
        """Decode only the bounded, duplicate-free version 1 identity format."""
        if type(text) is not str:
            raise ValueError("MCP manifest must be a JSON string")
        if len(text) > _MAX_BYTES:
            raise ValueError("MCP manifest exceeds 1 MiB")
        try:
            size = len(text.encode("utf-8"))
        except UnicodeError:
            raise ValueError("Invalid MCP manifest encoding") from None
        if size > _MAX_BYTES:
            raise ValueError("MCP manifest exceeds 1 MiB")

        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if len(key) > _MAX_FIELD_NAME:
                    raise ValueError("MCP manifest field name exceeds host budget")
                if key in result:
                    raise ValueError("Duplicate MCP manifest field")
                result[key] = value
            return result

        def invalid_constant(value: str) -> object:
            raise ValueError("Non-finite MCP manifest value")

        try:
            data = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
        except (ValueError, RecursionError):
            raise ValueError("Invalid MCP manifest JSON") from None
        if (
            type(data) is not dict
            or set(data) != {"version", "bindings"}
            or type(data["version"]) is not int
            or data["version"] != 1
        ):
            raise ValueError("Unsupported MCP manifest")
        items = data["bindings"]
        if type(items) is not list or len(items) > _MAX_BINDINGS:
            raise ValueError("Invalid MCP bindings; maximum is 1024")
        bindings = []
        for item in items:
            if type(item) is not dict or set(item) != _BINDING_FIELDS:
                raise ValueError("Invalid MCP binding fields")
            bindings.append(McpBinding(**item))
        return cls(tuple(bindings))

    @classmethod
    def from_tools(cls, tools: Sequence[Tool]) -> McpManifest:
        """Copy host provenance only from the actual adapter, without SDK imports."""
        from .tools import McpTool

        bindings = []
        for tool in tools:
            if type(tool) is not McpTool:
                continue
            provenance = tool.provenance
            if provenance is None:
                raise ValueError("MCP adapter lacks host provenance")
            bindings.append(
                McpBinding(
                    name=tool.name,
                    server=provenance.server,
                    tool=provenance.tool,
                    schema_digest=provenance.schema_digest,
                    declaration_digest=declaration_digest(tool),
                )
            )
            if len(bindings) > _MAX_BINDINGS:
                raise ValueError("Invalid MCP bindings; maximum is 1024")
        return cls(tuple(bindings))
