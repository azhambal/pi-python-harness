"""Optional MCP integration; importing these host APIs does not import the SDK."""

from ._contracts import McpError, McpLimits, binding_name
from ._http import HttpLimits
from ._stdio import StdioLimits
from .config import (
    HttpServerConfig,
    McpConfig,
    McpConfigError,
    StdioServerConfig,
    load_mcp_config,
    parse_mcp_config,
)
from .manifest import McpBinding, McpManifest
from .runtime import McpConnection, McpRuntime
from .tools import McpTool

__all__ = [
    "McpConfig",
    "McpBinding",
    "McpManifest",
    "McpConfigError",
    "McpConnection",
    "McpError",
    "McpLimits",
    "McpRuntime",
    "McpTool",
    "StdioLimits",
    "StdioServerConfig",
    "HttpServerConfig",
    "HttpLimits",
    "binding_name",
    "load_mcp_config",
    "parse_mcp_config",
]
