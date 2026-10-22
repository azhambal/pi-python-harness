"""Host-authorized MCP construction and journal-first idle tool transactions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pi_agent._async_resources import ResourceOwner
from pi_agent._session_construction import _align_resumed_prompt, _config_from_snapshot, construct_session
from pi_agent._session_projection import build_projection
from pi_agent.cancellation import OperationAbortedError
from pi_agent.config import AgentConfig
from pi_agent.permissions import PermissionPolicy, PermissionPrompt
from pi_agent.provider import Provider
from pi_agent.session import AgentSession
from pi_agent.session_store import SessionStore, latest_config_change
from pi_agent.system_prompt import SystemPromptInput, build_system_prompt_sections
from pi_agent.tool_hooks import ToolHooks
from pi_agent.tools import CODING_TOOL_NAMES, TOOL_NAMES, create_builtin_tools, validate_builtin_names
from pi_agent.types import ModelInfo, get_current_system_message, get_current_tools

from ._contracts import McpError
from .config import McpConfig
from .manifest import McpManifest
from .runtime import McpRuntime
from .tools import McpTool


async def construct_mcp_harness(
    provider: Provider,
    *,
    model: ModelInfo | None,
    cwd: Path | str | None,
    session: Path | str | Literal["new"] | None,
    tool_names: Sequence[str] | None,
    config: AgentConfig | None,
    refresh_instructions: bool,
    hooks: ToolHooks | None,
    permission_policy: PermissionPolicy | None,
    confirm_permission: PermissionPrompt | None,
    owner: ResourceOwner,
    mcp_config: McpConfig | None,
    mcp_tools: Sequence[str] | None,
    refresh_mcp: bool,
    environment: Mapping[str, str] | None,
) -> AgentSession:
    """Keep the resumed journal writer lock throughout asynchronous preparation."""
    resumed = session is not None and session != "new"
    opened: SessionStore | None = None
    try:
        declarations = []
        recorded = McpManifest()
        if resumed:
            assert session is not None
            opened = SessionStore.open(session, cwd_override=cwd)
            messages = [message for entry in build_projection(opened) for message in entry.messages]
            system = get_current_system_message(messages)
            declarations = get_current_tools(messages)
            if system is not None and system.mcp_manifest is not None:
                recorded = McpManifest.from_manifest(system.mcp_manifest)
                recorded.validate_declarations(declarations)
            snapshot = latest_config_change(opened.entries) or opened.header.config
            if cwd is not None:
                effective_cwd = Path(cwd).resolve()
            else:
                effective_cwd = Path(str(snapshot.get("cwd") or opened.cwd)).resolve()
            opened.cwd = effective_cwd
            stored_model = snapshot.get("model") or opened.header.model
            active_model = model or (ModelInfo.from_json(stored_model) if stored_model is not None else None)
            if active_model is None:
                raise ValueError("The resumed session records no model identity")
            effective_config = config or _config_from_snapshot(snapshot, effective_cwd, active_model)
            declared_names = {tool.name for tool in declarations}
            known_mcp = {binding.name for binding in recorded.bindings}
            unknown = declared_names - set(TOOL_NAMES) - known_mcp
            allow_legacy = (refresh_mcp or mcp_tools is not None and len(mcp_tools) == 0) and all(
                name.startswith("mcp__") for name in unknown
            )
            if unknown and not allow_legacy:
                raise McpError("The saved session requires custom tools unavailable to managed MCP")
            builtin_names = (
                validate_builtin_names(tool_names)
                if tool_names is not None
                else tuple(name for name in TOOL_NAMES if name in declared_names)
            )
        else:
            effective_cwd = Path(cwd or (config.cwd if config is not None else Path.cwd())).resolve()
            active_model = model or (config.model if config is not None else None)
            if active_model is None:
                raise ValueError("model is required for a new harness")
            effective_config = config or AgentConfig(cwd=effective_cwd, model=active_model)
            builtin_names = (
                validate_builtin_names(tool_names) if tool_names is not None else CODING_TOOL_NAMES
            )

        # A default resume connects only recorded servers/tools. Empty selection is
        # an explicit offline removal; discovery extras never activate implicitly.
        selectors = mcp_tools
        servers: tuple[str, ...] | None = None
        if resumed and selectors is not None:
            known = {binding.name: binding.server for binding in recorded.bindings}
            if all(selector in known or "/" in selector for selector in selectors):
                servers = tuple(
                    sorted(
                        {
                            known[selector] if selector in known else selector.partition("/")[0]
                            for selector in selectors
                        }
                    )
                )
        if resumed and selectors is None:
            if recorded.bindings:
                selectors = tuple(binding.name for binding in recorded.bindings)
                servers = tuple(sorted({binding.server for binding in recorded.bindings}))
            elif refresh_mcp:
                selectors = tuple(tool.name for tool in declarations if tool.name.startswith("mcp__"))
            else:
                selectors, servers = (), ()
        if selectors is not None and len(selectors) == 0:
            servers = ()
        if selectors is not None and tuple(selectors) == ("*",):
            selectors, servers = None, None
        if mcp_config is None:
            if selectors:
                raise McpError("Saved MCP tools require explicitly supplied trusted configuration")
            mcp_config = McpConfig(())
        builtins = tuple(
            create_builtin_tools(
                effective_cwd,
                builtin_names,
                bash_default_timeout_s=effective_config.bash_default_timeout_s,
            )
        )
        runtime = McpRuntime(
            mcp_config, builtins=builtins, environment=environment, selectors=selectors, servers=servers
        )
        prepared = await runtime.prepare_tools(owner)
        result = construct_session(
            AgentSession,
            provider,
            model=model,
            cwd=cwd,
            session=session,
            tools=prepared,
            tool_names=None,
            config=config,
            refresh_instructions=refresh_instructions,
            hooks=hooks,
            permission_policy=permission_policy,
            confirm_permission=confirm_permission,
            opened_store=opened,
            mcp_update=refresh_mcp or mcp_tools is not None,
            mcp_schema_update=refresh_mcp,
            allow_tool_removal=mcp_tools is not None or tool_names is not None or refresh_mcp,
        )
        result._mcp_runtime = runtime
        return result
    except BaseException:
        if opened is not None:
            opened.close()
        raise


async def update_mcp_tools(
    session: AgentSession,
    *,
    server: str | None = None,
    selectors: Sequence[str] | None = None,
) -> None:
    """Stage replacements, commit one system record, publish synchronously, retire."""
    if selectors is not None:
        selectors = McpRuntime._selection(selectors)
    session._begin_configuration("reconnect_mcp" if server is not None else "reload_mcp")
    runtime = session._mcp_runtime
    session._active_task = asyncio.current_task()
    session._abort_requested = False
    stage = None
    committed = False
    try:
        if runtime is None:
            raise McpError("MCP configuration is not attached; use create_harness with trusted mcp_config")
        recorded = McpManifest.from_tools(session._tools)
        servers: tuple[str, ...] | None = None
        if server is not None:
            wanted = tuple(binding.name for binding in recorded.bindings if binding.server == server)
            if not wanted:
                raise McpError("Cannot reconnect an inactive MCP binding")
            selectors, servers = wanted, (server,)
        elif selectors is None:
            selectors = tuple(binding.name for binding in recorded.bindings)
            servers = tuple(sorted({binding.server for binding in recorded.bindings}))
        elif tuple(selectors) == ("*",):
            selectors = None
        stage = await runtime.stage(selectors, servers=servers)
        if session._abort_requested:
            raise OperationAbortedError("MCP configuration update aborted")
        if server is not None:
            replacement = {tool.name: tool for tool in stage.tools}
            if set(replacement) != set(wanted):
                raise McpError("MCP reconnect did not restore every required binding")
            tools = [replacement.get(tool.name, tool) for tool in session._tools]
            if McpManifest.from_tools(tools) != recorded:
                raise McpError("MCP binding or schema changed; explicitly reload before reconnecting")
        else:
            tools = [tool for tool in session._tools if type(tool) is not McpTool] + list(stage.tools)
        if len({tool.name for tool in tools}) != len(tools):
            raise McpError("MCP active tool name collision")
        manifest = McpManifest.from_tools(tools)
        runtime.validate_stage(stage, tools)
        # Keep recorded instructions. Refresh only sections derived from active tools.
        current = get_current_system_message(session.agent.messages)
        sections = (
            {name: text for name, text in (current.sections or {}).items() if text is not None}
            if current
            else {}
        )
        rendered = build_system_prompt_sections(SystemPromptInput(cwd=session.cwd, tools=tools))
        for name in ("tools", "rules", "cwd"):
            if name in rendered:
                sections[name] = rendered[name]
        sections.pop("skills", None)
        skill_section = session._resources.prompt_section({tool.name for tool in tools})
        if skill_section is not None:
            sections["skills"] = skill_section
        _align_resumed_prompt(
            session.store,
            session.agent.messages,
            sections,
            [tool.declaration() for tool in tools],
            session.cwd,
            mcp_manifest=manifest.to_manifest(),
        )
        # No await/callback between durable append and activation of every binding.
        runtime.publish(stage, tools)
        session._tools = list(tools)
        session.agent.tools = tools
        session._refresh_agent_messages()
        committed = True
        try:
            await runtime.retire_previous(stage)
        except Exception:
            raise McpError(
                "MCP loadout committed; cleanup pending. Resources remain owned for close/retry."
            ) from None
    finally:
        try:
            if stage is not None and not committed:
                await stage.aclose()
        finally:
            session._active_task = None
            session._busy.release()
            session._idle_event.set()
