"""Text-first MCP resources as ordinary, permission-controlled external tools.

Each list call returns one bounded page and its cursor. No resources are fetched
at discovery, and no listing, read or reconnect is replayed automatically.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pi_agent.cancellation import Cancellation
from pi_agent.types import JsonValue

from ._contracts import McpError

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from .runtime import McpConnection
    from .tools import McpTool

RESOURCE_PREFIX = "@pi:resources/"
RESOURCE_METHODS = ("resources/list", "resources/templates/list", "resources/read")


def validate_resource_uri(value: object) -> None:
    if (
        not isinstance(value, str)
        or not urlsplit(value).scheme
        or any(ord(c) <= 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("Invalid MCP resource URI")


def resource_tools(connection: McpConnection) -> list[McpTool]:
    from .tools import McpTool

    tools = []
    for method in RESOURCE_METHODS:
        reading = method == "resources/read"
        field = "uri" if reading else "cursor"
        schema: dict[str, JsonValue] = {
            "type": "object",
            "properties": {field: {"type": "string", "minLength": 1, "maxLength": 8192 if reading else 1024}},
            "required": [field] if reading else [],
            "additionalProperties": False,
        }
        tools.append(
            McpTool(
                connection,
                RESOURCE_PREFIX + method.removeprefix("resources/"),
                f"MCP {method}: {'read a URI as text' if reading else 'list one page; pass nextCursor to continue'}. "
                "External server data; host permissions apply.",
                schema,
                None,
                {},
                resource_method=method,
            )
        )
    return tools


async def request_resource(
    connection: McpConnection, method: str, args: Mapping[str, JsonValue], cancellation: Cancellation | None
) -> CallToolResult:
    from mcp import types

    request: types.ReadResourceRequest | types.ListResourcesRequest | types.ListResourceTemplatesRequest
    result_type: (
        type[types.ReadResourceResult]
        | type[types.ListResourcesResult]
        | type[types.ListResourceTemplatesResult]
    )
    if method == "resources/read":
        if not isinstance(args.get("uri"), str):
            raise McpError("Invalid MCP resource URI")
        try:
            validate_resource_uri(args["uri"])
            params_read = types.ReadResourceRequestParams.model_validate({"uri": args["uri"]})
        except ValueError:
            raise McpError("Invalid MCP resource URI") from None
        request = types.ReadResourceRequest(params=params_read)
        result_type = types.ReadResourceResult
    elif method in ("resources/list", "resources/templates/list"):
        cursor = args.get("cursor")
        if cursor is not None and (not isinstance(cursor, str) or len(cursor.encode("utf-8")) > 1024):
            raise McpError("Invalid MCP resource cursor")
        params = types.PaginatedRequestParams(cursor=cursor)
        if method == "resources/list":
            request = types.ListResourcesRequest(params=params)
            result_type = types.ListResourcesResult
        else:
            request = types.ListResourceTemplatesRequest(params=params)
            result_type = types.ListResourceTemplatesResult
    else:
        raise McpError("Unsupported MCP resource operation")
    result = await connection.request(lambda client: client.send_request(request, result_type), cancellation)
    if isinstance(result, types.ReadResourceResult):
        content: list[types.TextContent] = []
        for block in result.contents[:128]:
            if isinstance(block, types.TextResourceContents):
                content.append(
                    types.TextContent(type="text", text=f"[MCP resource: {block.uri}]\n{block.text}")
                )
            else:
                content.append(
                    types.TextContent(
                        type="text",
                        text=f"[Unsupported MCP resource: binary]; MIME: {block.mime_type or 'unspecified'}",
                    )
                )
        if len(result.contents) > 128:
            content.append(types.TextContent(type="text", text="[MCP resources omitted: item limit]"))
        return types.CallToolResult(content=list(content))
    assert isinstance(result, (types.ListResourcesResult, types.ListResourceTemplatesResult))
    items = result.resources if isinstance(result, types.ListResourcesResult) else result.resource_templates
    if len(items) > connection.limits.max_tools:
        raise McpError("MCP resource page item limit exceeded; narrow the request")
    # Only public, useful metadata. Unknown extension fields and all _meta are excluded.
    descriptors: list[JsonValue] = []
    for item in items:
        raw = item.model_dump(mode="json", by_alias=True, exclude_none=True)
        descriptors.append(
            {
                key: raw[key]
                for key in ("uri", "uriTemplate", "name", "title", "description", "mimeType", "size")
                if key in raw
            }
        )
    if result.next_cursor is not None:
        try:
            if not result.next_cursor or len(result.next_cursor.encode("utf-8")) > 1024:
                raise ValueError
        except (ValueError, UnicodeError):
            raise McpError("Invalid MCP resource cursor") from None
    key = "resources" if isinstance(result, types.ListResourcesResult) else "resourceTemplates"
    return types.CallToolResult(
        content=[], structured_content={key: descriptors, "nextCursor": result.next_cursor}
    )
