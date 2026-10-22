"""MCP descriptors/results adapted to the existing Tool and hook pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

import jsonschema

from pi_agent._async_resources import _IN_RESOURCE_CALLBACK
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tool_hooks import ToolProvenance
from pi_agent.tools.base import (
    Tool,
    ToolResult,
    ToolUpdateCallback,
    ToolValidationError,
    validate_tool_arguments,
)
from pi_agent.types import JsonValue, TextBlock

from ._contracts import McpError, binding_name, bounded_json, checked_schema, validate_schema_instance

if TYPE_CHECKING:
    from mcp.types import CallToolResult
    from mcp.types import Tool as McpDescriptor

    from .runtime import McpConnection


class McpTool(Tool):
    """An external tool is a scheduling barrier; annotations grant no permission."""

    def __init__(
        self,
        connection: McpConnection,
        original_name: str,
        description: str,
        parameters: dict[str, JsonValue],
        output_schema: dict[str, JsonValue] | None,
        annotations: dict[str, JsonValue],
        *,
        resource_method: str | None = None,
    ) -> None:
        from .resources import RESOURCE_METHODS, RESOURCE_PREFIX

        if resource_method is not None:
            if (
                resource_method not in RESOURCE_METHODS
                or original_name != RESOURCE_PREFIX + resource_method.removeprefix("resources/")
            ):
                raise McpError("Invalid host MCP resource identity")
        elif original_name.startswith(RESOURCE_PREFIX):
            raise McpError("MCP server tool uses a reserved host resource identity")
        self._connection = connection
        self._original_name = original_name
        self._resource_method = resource_method
        self._parameters = deepcopy(parameters)
        self._output_schema = deepcopy(output_schema)
        self.name = binding_name(connection.config.name, original_name)
        self.description = description
        self.prompt_snippet = "External MCP tool; host permissions apply."
        schema_json = json.dumps(
            [parameters, output_schema], ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        self.provenance = ToolProvenance(
            kind="mcp",
            server=connection.config.name,
            tool=original_name,
            schema_digest=hashlib.sha256(schema_json.encode()).hexdigest(),
            annotations_json=bounded_json(annotations, 4096),
        )

    def schema(self) -> dict[str, JsonValue]:
        return deepcopy(self._parameters)

    def validate_arguments(self, args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        value = dict(args)
        try:
            bounded_json(value, self._connection.limits.argument_bytes)
            validate_schema_instance(self._parameters, value)
            if self._resource_method is not None:
                field = "uri" if self._resource_method == "resources/read" else "cursor"
                text = value.get(field)
                if isinstance(text, str) and len(text.encode("utf-8")) > (8192 if field == "uri" else 1024):
                    raise ValueError
                if field == "uri":
                    from .resources import validate_resource_uri

                    validate_resource_uri(text)
        except (McpError, jsonschema.ValidationError, RecursionError, ValueError):
            raise ToolValidationError("MCP arguments failed validation or exceed the host budget") from None
        return value

    @classmethod
    def from_descriptor(cls, connection: McpConnection, descriptor: McpDescriptor) -> McpTool:
        from .resources import RESOURCE_PREFIX

        if descriptor.name.startswith(RESOURCE_PREFIX):
            raise McpError("MCP server tool uses a reserved host resource identity")
        if not descriptor.name or len(descriptor.name.encode("utf-8")) > 256:
            raise McpError("Invalid MCP tool name")
        description = descriptor.description or "External MCP tool"
        if len(description.encode("utf-8")) > 8192:
            raise McpError("MCP tool description limit exceeded")
        annotations: dict[str, JsonValue] = {}
        if descriptor.annotations is not None:
            # Ignore unknown annotations/_meta and bound the known non-authoritative hints.
            raw = descriptor.annotations.model_dump(mode="json", by_alias=True, exclude_none=True)
            for key in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
                if key in raw and type(raw[key]) is bool:
                    annotations[key] = raw[key]
        return cls(
            connection,
            descriptor.name,
            description,
            checked_schema(descriptor.input_schema, connection.limits.schema_bytes, input_schema=True),
            checked_schema(descriptor.output_schema, connection.limits.schema_bytes, input_schema=False)
            if descriptor.output_schema is not None
            else None,
            annotations,
        )

    async def execute(
        self,
        tool_call_id: str,
        args: Mapping[str, JsonValue],
        cancellation: Cancellation | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        if cancellation is not None:
            cancellation.raise_if_aborted()
        bounded_json(dict(args), self._connection.limits.argument_bytes)
        validated = validate_tool_arguments(self, args)
        updates = 0
        update_error = False

        async def emit_progress(progress: float, total: float | None, message: str | None) -> None:
            nonlocal updates, update_error
            if on_update is None or updates >= self._connection.limits.progress_updates:
                return
            updates += 1
            if not math.isfinite(progress) or (total is not None and not math.isfinite(total)):
                update_error = True
                return
            # Progress messages are untrusted content, not diagnostics or config.
            text = (message or "MCP progress").encode("utf-8")[:1024].decode("utf-8", errors="ignore")
            partial = ToolResult(
                content=[TextBlock(text=text)], details={"progress": progress, "total": total}
            )
            try:
                bounded_json(partial.to_json(), self._connection.limits.result_bytes)
            except McpError:
                partial.content = [TextBlock(text="MCP progress (message exceeds output budget)")]
            marker = _IN_RESOURCE_CALLBACK.set(True)
            try:
                await on_update(partial)
            except Exception:
                # SDK logs callback exceptions. Keep raw host/server exception text
                # out of that logger and fail the final result with a constant notice.
                update_error = True
            finally:
                _IN_RESOURCE_CALLBACK.reset(marker)

        try:
            if self._resource_method is not None:
                from .resources import request_resource

                result = await request_resource(
                    self._connection, self._resource_method, validated, cancellation
                )
            else:
                result = await self._connection.call(
                    self._original_name,
                    validated,
                    cancellation,
                    emit_progress if on_update is not None else None,
                )
        except OperationAbortedError:
            return ToolResult(
                content=[
                    TextBlock(text="MCP call interrupted; outcome unknown. Inspect effects before retrying.")
                ],
                details={"outcome": "unknown"},
                is_error=True,
            )
        if not result.is_error and self._output_schema is not None:
            try:
                if "structured_content" not in result.model_fields_set:
                    raise ValueError
                bounded_json(result.structured_content, self._connection.stdio_limits.frame_bytes)
                validate_schema_instance(self._output_schema, result.structured_content)
            except (ValueError, jsonschema.ValidationError, RecursionError, McpError):
                return ToolResult(
                    content=[TextBlock(text="MCP result does not match the declared output schema.")],
                    details={"invalid_output_schema": True},
                    is_error=True,
                )
        converted = self._convert_result(result)
        if update_error:
            converted.content = [TextBlock(text="MCP progress delivery failed; the tool call completed.")]
            converted.details = {"progress_delivery_failed": True, "server_is_error": result.is_error}
            converted.is_error = True
        return converted

    def _convert_result(self, result: CallToolResult) -> ToolResult:
        parts: list[str] = []
        unsupported: list[JsonValue] = []
        for block in result.content:
            if block.type == "text":
                parts.append(block.text)
            elif block.type == "resource" and hasattr(block.resource, "text"):
                parts.append(f"[MCP resource: {block.resource.uri}]\n{block.resource.text}")
            elif block.type == "resource_link":
                parts.append(f"[MCP resource link: {block.uri}; read explicitly with the resource read tool]")
            else:
                if block.type not in unsupported:
                    unsupported.append(block.type)
                mime = getattr(block, "mime_type", None)
                parts.append(f"[Unsupported MCP content: {block.type}{'; MIME: ' + mime if mime else ''}]")
        if not result.content and "structured_content" in result.model_fields_set:
            bounded_json(result.structured_content, self._connection.stdio_limits.frame_bytes)
            parts.append(json.dumps(result.structured_content, ensure_ascii=False, allow_nan=False, indent=2))
        text = "\n".join(parts)
        if result.is_error and not text:
            text = f"MCP tool {self._connection.config.name}/{self._original_name} returned an error"
        # The envelope preserves absent versus explicit null, including error payloads.
        from ._results import WireCallToolResult

        envelope = (
            deepcopy(result.wire_result)
            if isinstance(result, WireCallToolResult)
            else result.model_dump(mode="json", by_alias=True, exclude_unset=True)
        )
        envelope.pop("_meta", None)
        bounded_json(envelope, self._connection.stdio_limits.frame_bytes)
        details: dict[str, JsonValue] = {"unsupported_content": unsupported}
        raw = text.encode("utf-8")
        limit = min(20 * 1024, self._connection.limits.result_bytes)
        if len(raw) > limit:
            details["truncated"] = True
            half = limit // 2
            head = raw[:half].decode("utf-8", errors="ignore")
            tail = raw[-half:].decode("utf-8", errors="ignore")
            try:
                with tempfile.NamedTemporaryFile(prefix="pi-mcp-", suffix=".txt", delete=False) as output:
                    self._connection.spills.register(output.name, output.fileno())
                    output.write(raw)
                    path = output.name
                details["full_output_path"] = path
                where = f"[Full output: {path} (read it with offset/limit)]"
            except OSError:
                where = "[Could not save the full MCP output]"
            text = f"{head}\n\n[... output truncated ...]\n\n{tail}\n\n{where}"
        return ToolResult(
            content=[TextBlock(text=text)] if text else [],
            structured_content=envelope,
            details=details,
            is_error=result.is_error,
        )
