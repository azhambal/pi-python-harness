"""Asynchronous preparation around the unchanged synchronous core factory."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ._async_resources import AsyncResource, ResourceOwner
from .config import AgentConfig
from .permissions import PermissionPolicy, PermissionPrompt
from .provider import Provider
from .session import AgentSession, create_agent_session
from .tool_hooks import ToolHooks
from .tools.base import Tool
from .types import ModelInfo

PrepareTools = Callable[[ResourceOwner], Awaitable[Iterable[Tool]]]

if TYPE_CHECKING:
    from .mcp.config import McpConfig


async def create_harness(
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
    resource_owner: ResourceOwner | None = None,
    prepare_tools: PrepareTools | None = None,
    mcp_config: McpConfig | None = None,
    mcp_tools: Sequence[str] | None = None,
    refresh_mcp: bool = False,
    mcp_environment: Mapping[str, str] | None = None,
) -> AgentSession:
    """Prepare host tools and transfer their owner to a core session.

    The optional callback returns the complete injected toolset, just like
    tools=. Register a resource before its first connection await so failed or
    cancelled preparation can close it. No process starts merely from a journal.
    An explicit owner transfers on entry and must not be reused by another host.
    Supply resource_owner if cleanup may need a retry after construction fails.
    """
    if prepare_tools is not None and (tools is not None or tool_names is not None):
        raise ValueError("prepare_tools cannot be combined with tools/tool_names")
    if type(refresh_mcp) is not bool:
        raise ValueError("refresh_mcp must be boolean")
    if mcp_tools is not None and (
        isinstance(mcp_tools, (str, bytes))
        or not isinstance(mcp_tools, Sequence)
        or len(mcp_tools) > 1024
        or any(not isinstance(name, str) or not name for name in mcp_tools)
    ):
        raise ValueError("mcp_tools must be a bounded sequence of selectors")
    managed_mcp = mcp_config is not None or mcp_tools is not None or refresh_mcp
    if managed_mcp and (tools is not None or prepare_tools is not None):
        raise ValueError("Managed MCP cannot be combined with tools/prepare_tools")
    if mcp_environment is not None and not managed_mcp:
        raise ValueError("mcp_environment requires managed MCP")
    owner = resource_owner if resource_owner is not None else ResourceOwner()
    owner.claim()
    try:
        if managed_mcp:
            from .mcp.session import construct_mcp_harness

            result = await construct_mcp_harness(
                provider,
                model=model,
                cwd=cwd,
                session=session,
                tool_names=tool_names,
                config=config,
                refresh_instructions=refresh_instructions,
                hooks=hooks,
                permission_policy=permission_policy,
                confirm_permission=confirm_permission,
                owner=owner,
                mcp_config=mcp_config,
                mcp_tools=mcp_tools,
                refresh_mcp=refresh_mcp,
                environment=mcp_environment,
            )
            result._resource_owner = owner
            return result
        prepared_tools = await prepare_tools(owner) if prepare_tools is not None else tools
        result = create_agent_session(
            provider,
            model=model,
            cwd=cwd,
            session=session,
            tools=prepared_tools,
            tool_names=tool_names,
            config=config,
            refresh_instructions=refresh_instructions,
            hooks=hooks,
            permission_policy=permission_policy,
            confirm_permission=confirm_permission,
        )
        result._resource_owner = owner
        if prepare_tools is not None:
            from .mcp.runtime import McpRuntime

            runtime = getattr(prepare_tools, "__self__", None)
            if isinstance(runtime, McpRuntime):
                result._mcp_runtime = runtime
        return result
    except BaseException as error:
        try:
            await owner.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as cleanup_error:
            raise BaseExceptionGroup(
                "Harness construction and cleanup failed", [error, cleanup_error]
            ) from None
        raise


__all__ = ["AsyncResource", "ResourceOwner", "create_harness"]
