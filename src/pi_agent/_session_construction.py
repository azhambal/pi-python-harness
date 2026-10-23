"""Create/resume construction and durable configuration serialization."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from ._session_projection import _raw_call_ids, build_projection
from .agent import Agent
from .agent_loop import PrepareRequestContext, PrepareRequestUpdate
from .compaction import (
    ProjectedEntry,
)
from .config import DEFAULT_AGENT_DIR, AgentConfig, CompactionSettings, RetrySettings
from .mcp.manifest import McpManifest
from .permissions import PermissionPolicy, PermissionPrompt
from .provider import ModelValidator, Provider
from .resources import ResourceSnapshot
from .session_store import (
    SessionStore,
    latest_config_change,
)
from .skills import SkillSettings
from .system_prompt import (
    ContextFile,
    SystemPromptInput,
    build_system_prompt_sections,
    diff_system_prompt_sections,
    load_project_context_files,
)
from .tool_hooks import ToolHooks
from .tools import TOOL_NAMES, create_builtin_tools, create_coding_tools, validate_builtin_names
from .tools.base import Tool
from .types import (
    JsonValue,
    Message,
    ModelInfo,
    SystemMessage,
    ThinkingLevel,
    ToolDeclaration,
    get_current_system_message,
    get_current_tools,
    now_ms,
)

if TYPE_CHECKING:
    from .session import AgentSession


def _resolve_cwd(value: Path | str) -> Path:
    return Path(value).resolve()


def _effective_thinking_level(model: ModelInfo, level: ThinkingLevel) -> ThinkingLevel:
    if not isinstance(level, str) or level not in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("thinking_level is invalid")
    # Pi clamps to the nearest supported level (clampThinkingLevel).
    return model.clamp_thinking_level(level)


def _validate_runtime_model(provider: Provider, model: ModelInfo, level: ThinkingLevel) -> None:
    if not callable(getattr(provider, "stream", None)):
        raise ValueError("provider must implement stream()")
    if isinstance(provider, ModelValidator):
        validate = cast(Callable[[ModelInfo, ThinkingLevel], object], provider.validate_model)
        result = validate(deepcopy(model), level)
        if result is not None:
            if inspect.iscoroutine(result):
                result.close()
            raise ValueError("validate_model must be synchronous and return None")


def _config_json(config: AgentConfig, model: ModelInfo, cwd: Path | str) -> dict[str, JsonValue]:
    """Durable configuration snapshot (no secrets; cwd and model included)."""
    return {
        "cwd": str(cwd),
        "model": model.to_json(),
        "thinking_level": config.thinking_level,
        "agent_dir": str(config.agent_dir),
        "session_dir": str(config.session_dir) if config.session_dir is not None else None,
        "compaction": {
            "enabled": config.compaction.enabled,
            "reserve_tokens": config.compaction.reserve_tokens,
            "keep_recent_tokens": config.compaction.keep_recent_tokens,
        },
        "retry": {
            "enabled": config.retry.enabled,
            "max_retries": config.retry.max_retries,
            "base_delay_ms": config.retry.base_delay_ms,
            "max_delay_ms": config.retry.max_delay_ms,
        },
        "bash_default_timeout_s": config.bash_default_timeout_s,
        "instruction_files": (
            [str(path) for path in config.instruction_files] if config.instruction_files is not None else None
        ),
        "permissions": config.permissions.to_json(),
        "tool_execution": config.tool_execution,
        "skills": config.skills.to_json(),
    }


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return Path(value)
    return None


def _config_from_snapshot(
    snapshot: Mapping[str, object],
    effective_cwd: Path,
    model: ModelInfo,
) -> AgentConfig:
    """Rebuild a validated AgentConfig from a stored snapshot.

    Missing keys (older snapshots) fall back to defaults; invalid values are
    rejected by the settings parsers and ``validate``.
    """
    instruction_files = snapshot.get("instruction_files")
    parsed_instructions: list[Path] | None = None
    if isinstance(instruction_files, list) and all(isinstance(item, str) for item in instruction_files):
        parsed_instructions = [Path(item) for item in instruction_files]
    bash_timeout = snapshot.get("bash_default_timeout_s")
    if bash_timeout is not None and (
        isinstance(bash_timeout, bool)
        or not isinstance(bash_timeout, (int, float))
        or not math.isfinite(bash_timeout)
    ):
        raise ValueError("stored bash_default_timeout_s is invalid")
    config = AgentConfig(
        cwd=effective_cwd,
        model=model,
        thinking_level=_stored_thinking_level(snapshot),
        agent_dir=_optional_path(snapshot.get("agent_dir")) or DEFAULT_AGENT_DIR,
        session_dir=_optional_path(snapshot.get("session_dir")),
        compaction=_stored_compaction(snapshot),
        retry=_stored_retry(snapshot),
        bash_default_timeout_s=bash_timeout,
        instruction_files=parsed_instructions,
        permissions=PermissionPolicy.from_json(snapshot["permissions"])
        if "permissions" in snapshot
        else PermissionPolicy(),
        tool_execution=cast(Literal["sequential", "parallel"], snapshot.get("tool_execution", "parallel")),
        skills=SkillSettings.from_json(snapshot.get("skills", {})),
    )
    config.validate()
    return config


def _stored_thinking_level(stored: Mapping[str, object]) -> ThinkingLevel:
    value = stored.get("thinking_level")
    if isinstance(value, str) and value in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
        return cast(ThinkingLevel, value)
    if "thinking_level" not in stored:
        return "off"
    raise ValueError("stored thinking_level is invalid")


def _stored_compaction(stored: Mapping[str, object]) -> CompactionSettings:
    return CompactionSettings.from_json(stored.get("compaction", {}))


def _stored_retry(stored: Mapping[str, object]) -> RetrySettings:
    return RetrySettings.from_json(stored.get("retry", {}))


def _context_files_for(config: AgentConfig, effective_cwd: Path) -> list[ContextFile]:
    if config.instruction_files is not None:
        files: list[ContextFile] = []
        for path in config.instruction_files:
            if path.is_file():
                try:
                    files.append(ContextFile(path=path, content=path.read_text(encoding="utf-8")))
                except OSError:
                    continue
        return files
    return load_project_context_files(effective_cwd, config.agent_dir)


def _declared_tool_names(projected: list[ProjectedEntry]) -> set[str]:
    return {tool.name for tool in get_current_tools([m for entry in projected for m in entry.messages])}


def _align_resumed_prompt(
    store: SessionStore,
    agent_messages: list[Message],
    sections: dict[str, str] | None,
    declarations: list[ToolDeclaration],
    effective_cwd: Path,
    resource_manifest: str | None = None,
    mcp_manifest: str | None = None,
) -> list[Message]:
    """Align a resumed prompt with the effective cwd and the active tool set.

    Contracts and failure semantics: docs/invariants.md.
    """
    messages = list(agent_messages)
    patch: dict[str, str | None] | None = None
    existing_system = get_current_system_message(messages)
    if sections is not None:
        if existing_system is not None and existing_system.sections:
            patch = diff_system_prompt_sections(existing_system.sections, sections)
    else:
        if existing_system is not None and existing_system.sections:
            expected_cwd_section = f"<cwd>\n{effective_cwd}\n</cwd>"
            if existing_system.sections.get("cwd") != expected_cwd_section:
                patch = {"cwd": expected_cwd_section}
    previous_tools = get_current_tools(messages)
    previous_by_name = {tool.name: tool for tool in previous_tools}
    current_by_name = {tool.name: tool for tool in declarations}
    removed = [{"name": name} for name in previous_by_name if name not in current_by_name]
    added = [
        tool
        for tool in declarations
        if tool.name not in previous_by_name or previous_by_name[tool.name].to_json() != tool.to_json()
    ]
    manifest_changed = resource_manifest is not None and (
        existing_system is None or existing_system.resource_manifest != resource_manifest
    )
    mcp_changed = mcp_manifest is not None and (
        existing_system is None or existing_system.mcp_manifest != mcp_manifest
    )
    if patch or added or removed or manifest_changed or mcp_changed:
        update = SystemMessage(
            content="",
            sections=patch,
            resource_manifest=resource_manifest if manifest_changed else None,
            mcp_manifest=mcp_manifest if mcp_changed else None,
            tools_added=added or None,
            tools_removed=removed or None,
            timestamp=now_ms(),
        )
        store.append_message(update)
        messages.append(update)
    return messages


def _resolve_session_config(
    config: AgentConfig | None,
    stored_snapshot: Mapping[str, object],
    effective_cwd: Path,
    active_model: ModelInfo,
    *,
    resumed: bool,
    permission_policy: PermissionPolicy | None,
) -> AgentConfig:
    if config is not None:
        cfg = AgentConfig(
            cwd=effective_cwd,
            model=active_model,
            thinking_level=config.thinking_level,
            agent_dir=config.agent_dir,
            session_dir=config.session_dir,
            compaction=config.compaction,
            retry=config.retry,
            bash_default_timeout_s=config.bash_default_timeout_s,
            instruction_files=config.instruction_files,
            permissions=config.permissions,
            tool_execution=config.tool_execution,
            skills=config.skills,
        )
    elif resumed and stored_snapshot:
        cfg = _config_from_snapshot(stored_snapshot, effective_cwd, active_model)
    else:
        cfg = AgentConfig(cwd=effective_cwd, model=active_model)
    if permission_policy is not None:
        cfg = replace(cfg, permissions=permission_policy)
    cfg.validate()
    return deepcopy(cfg)


def _resolve_session_tools(
    cfg: AgentConfig,
    projected: list[ProjectedEntry],
    tools: Iterable[Tool] | None,
    selected: Sequence[str] | None,
    *,
    allow_tool_removal: bool,
) -> list[Tool]:
    effective_cwd = cfg.cwd
    declared = _declared_tool_names(projected)
    if selected is not None:
        active_tools = create_builtin_tools(
            effective_cwd, selected, bash_default_timeout_s=cfg.bash_default_timeout_s
        )
    elif tools is not None:
        active_tools = list(tools)
        missing = sorted(declared - {tool.name for tool in active_tools})
        if missing and not allow_tool_removal:
            raise ValueError(
                f"the resumed session declares tools {missing} that the injected toolset "
                "does not provide; pass tools= including them"
            )
    elif not projected:
        # Fresh session: the default four built-ins.
        active_tools = create_coding_tools(effective_cwd, bash_default_timeout_s=cfg.bash_default_timeout_s)
    elif declared - set(TOOL_NAMES):
        unknown = sorted(declared - set(TOOL_NAMES))
        raise ValueError(
            f"the resumed session declares custom tools {unknown} that cannot be rebuilt "
            "from the built-ins; pass tools= providing implementations for them"
        )
    else:
        # Resumed session: exactly the declared set — an explicitly
        # empty declaration stays empty instead of silently expanding
        # to the four built-ins.
        active_tools = create_builtin_tools(
            effective_cwd,
            [name for name in TOOL_NAMES if name in declared],
            bash_default_timeout_s=cfg.bash_default_timeout_s,
        )

    return active_tools


def _validate_session_bindings(
    active_tools: list[Tool],
    projected: list[ProjectedEntry],
    existing_system: SystemMessage | None,
    *,
    mcp_update: bool,
    mcp_schema_update: bool,
    selected: Sequence[str] | None,
) -> McpManifest:
    bindings = McpManifest.from_tools(active_tools)
    if projected and existing_system is not None:
        recorded = (
            McpManifest.from_manifest(existing_system.mcp_manifest)
            if existing_system.mcp_manifest is not None
            else McpManifest()
        )
        recorded.validate_declarations(
            get_current_tools([message for entry in projected for message in entry.messages])
        )
        recorded_by_name = {binding.name: binding for binding in recorded.bindings}
        if not mcp_schema_update and any(
            binding.name in recorded_by_name and binding != recorded_by_name[binding.name]
            for binding in bindings.bindings
        ):
            raise ValueError("MCP binding schema or declaration changed; explicitly reload MCP")
        legacy_injection = existing_system.mcp_manifest is None and bool(bindings.bindings)
        if legacy_injection and not mcp_update:
            # Connected host injections may adopt only declarations already
            # present in legacy history; discovery extras remain inactive.
            bindings.validate_declarations(
                get_current_tools([message for entry in projected for message in entry.messages])
            )
        if recorded != bindings and not (mcp_update or selected is not None or legacy_injection):
            raise ValueError(
                "MCP bindings or schemas changed; supply trusted matching tools or explicitly reload MCP"
            )
    return bindings


def _resolve_session_resources(
    cfg: AgentConfig,
    config: AgentConfig | None,
    projected: list[ProjectedEntry],
    existing_system: SystemMessage | None,
) -> ResourceSnapshot:
    effective_cwd = cfg.cwd
    if projected:
        resources = (
            ResourceSnapshot.from_manifest(existing_system.resource_manifest).check_availability()
            if existing_system is not None and existing_system.resource_manifest is not None
            else ResourceSnapshot()
        )
        if config is not None and config.skills != resources.settings:
            raise ValueError("Use reload_resources() to change skills on a resumed session")
    else:
        resources = ResourceSnapshot.discover(cfg.skills, effective_cwd, cfg.agent_dir)
    return resources


def _prepare_session_prompt(
    store: SessionStore,
    cfg: AgentConfig,
    projected: list[ProjectedEntry],
    active_tools: list[Tool],
    resources: ResourceSnapshot,
    bindings: McpManifest,
    existing_system: SystemMessage | None,
    *,
    refresh_instructions: bool,
    refresh_tools: bool,
) -> list[Message]:
    effective_cwd = cfg.cwd
    skills_section = resources.prompt_section({tool.name for tool in active_tools})
    declarations = [tool.declaration() for tool in active_tools]
    if projected:
        agent_messages = [message for entry in projected for message in entry.messages]
        if refresh_instructions:
            context_files = _context_files_for(cfg, effective_cwd)
            sections = build_system_prompt_sections(
                SystemPromptInput(
                    cwd=effective_cwd,
                    tools=active_tools,
                    context_files=context_files,
                    skills_section=skills_section,
                )
            )
        else:
            sections = None  # keep the persisted prompt untouched
        if refresh_tools and sections is None:
            # Refresh only tool-derived sections, preserving recorded project instructions.
            existing = get_current_system_message(agent_messages)
            if existing is not None and existing.sections:
                kept = {name: text for name, text in existing.sections.items() if text is not None}
                rendered = build_system_prompt_sections(
                    SystemPromptInput(cwd=effective_cwd, tools=active_tools, skills_section=skills_section)
                )
                for name in ("tools", "rules", "cwd"):
                    if name in rendered:
                        kept[name] = rendered[name]
                kept.pop("skills", None)
                # Skills sit right before cwd, as Pi orders them.
                sections = {}
                for name, text in kept.items():
                    if name == "cwd" and "skills" in rendered:
                        sections["skills"] = rendered["skills"]
                    sections[name] = text
                if "skills" in rendered and "skills" not in sections:
                    sections["skills"] = rendered["skills"]
        agent_messages = _align_resumed_prompt(
            store,
            agent_messages,
            sections,
            declarations,
            effective_cwd,
            mcp_manifest=bindings.to_manifest()
            if bindings.bindings or (existing_system is not None and existing_system.mcp_manifest is not None)
            else None,
        )
    else:
        context_files = _context_files_for(cfg, effective_cwd)
        sections = build_system_prompt_sections(
            SystemPromptInput(
                cwd=effective_cwd,
                tools=active_tools,
                context_files=context_files,
                skills_section=skills_section,
            )
        )
        system_message = SystemMessage(
            content="",
            sections=dict(sections),
            tools_added=declarations,
            timestamp=0,
            resource_manifest=resources.to_manifest() if cfg.skills.enabled else None,
            mcp_manifest=bindings.to_manifest() if bindings.bindings else None,
        )
        store.append_message(system_message)
        agent_messages = [system_message]

    return agent_messages


def _restore_session_identity(
    store: SessionStore, cwd: Path | str | None, model: ModelInfo | None
) -> tuple[Path, Mapping[str, object], ModelInfo | None]:
    stored_snapshot = dict(latest_config_change(store.entries) or store.header.config)
    # A persisted cwd override (config_change record) replays as
    # the effective cwd unless the caller passes one explicitly.
    stored_cwd = stored_snapshot.get("cwd")
    if cwd is None and isinstance(stored_cwd, str) and stored_cwd:
        store.cwd = Path(stored_cwd).resolve()
    effective_cwd = store.cwd
    stored_model = ModelInfo.from_json(store.header.model) if isinstance(store.header.model, dict) else None
    stored_model_json = stored_snapshot.get("model")
    if isinstance(stored_model_json, dict):
        stored_model = ModelInfo.from_json(stored_model_json)
    if model is None:
        if stored_model is None:
            raise ValueError("the resumed session records no model identity; pass model= explicitly")
    elif stored_model is not None and (model.provider, model.id) != (
        stored_model.provider,
        stored_model.id,
    ):
        raise ValueError(
            f"the resumed session is bound to model {stored_model.provider}/{stored_model.id}; "
            f"pass that model (or none to use the stored identity), not "
            f"{model.provider}/{model.id}"
        )
    return effective_cwd, stored_snapshot, stored_model


def construct_session(
    session_cls: type[AgentSession],
    provider: Provider,
    *,
    model: ModelInfo | None,
    cwd: Path | str | None,
    session: Path | str | Literal["new"] | None,
    tools: Iterable[Tool] | None,
    tool_names: Sequence[str] | None,
    config: AgentConfig | None,
    refresh_instructions: bool,
    hooks: ToolHooks | None,
    permission_policy: PermissionPolicy | None,
    confirm_permission: PermissionPrompt | None,
    opened_store: SessionStore | None = None,
    mcp_update: bool = False,
    mcp_schema_update: bool = False,
    allow_tool_removal: bool = False,
) -> AgentSession:
    store: SessionStore | None = opened_store
    try:
        if tools is not None and tool_names is not None:
            raise ValueError("Pass tools= implementations or tool_names= builtin selection, not both")
        selected = validate_builtin_names(tool_names) if tool_names is not None else None
        resumed = session is not None and not (isinstance(session, str) and session == "new")
        stored_snapshot: Mapping[str, object] = {}
        stored_model: ModelInfo | None = None

        # A config object may also supply the model and cwd; explicit
        # parameters win, and contradictions between the two are errors
        # rather than silent field overrides.
        if model is None and config is not None:
            model = config.model
        if cwd is None and config is not None and not resumed:
            cwd = config.cwd
        if cwd is not None and config is not None and config.cwd.resolve() != _resolve_cwd(cwd):
            raise ValueError(f"config.cwd ({config.cwd}) contradicts the cwd parameter ({cwd}); align them")

        # -- Open the journal and resolve defaults vs explicit overrides --
        if not resumed:
            effective_cwd = _resolve_cwd(cwd or Path.cwd())
            if model is None:
                raise ValueError("model is required for new and in-memory sessions")
            base_config = config or AgentConfig(cwd=effective_cwd, model=model)
            if permission_policy is not None:
                base_config = replace(base_config, permissions=permission_policy)
            # The session directory always follows the *effective* cwd,
            # never a conflicting config.cwd (rejected above).
            session_dir = replace(base_config, cwd=effective_cwd).resolved_session_dir()
            snapshot = _config_json(base_config, model, effective_cwd)
            if session is None:
                store = SessionStore.in_memory(effective_cwd, model=model.to_json(), config=snapshot)
            else:
                store = SessionStore.create(
                    effective_cwd,
                    session_dir,
                    model=model.to_json(),
                    config=snapshot,
                )
            stored_snapshot = snapshot
        else:
            assert session is not None
            if store is None:
                store = SessionStore.open(session, cwd_override=cwd)
            elif cwd is not None:
                store.cwd = _resolve_cwd(cwd)
            effective_cwd, stored_snapshot, stored_model = _restore_session_identity(store, cwd, model)

        active_model = model if model is not None else stored_model
        assert active_model is not None
        active_model = deepcopy(active_model)
        if config is not None and config.model != active_model:
            raise ValueError(
                f"config.model ({config.model.provider}/{config.model.id}) contradicts the "
                f"effective model ({active_model.provider}/{active_model.id}); align them"
            )

        explicit_override = (
            config is not None or model is not None or cwd is not None or permission_policy is not None
        )
        cfg = _resolve_session_config(
            config,
            stored_snapshot,
            effective_cwd,
            active_model,
            resumed=resumed,
            permission_policy=permission_policy,
        )

        projected = build_projection(store)
        active_tools = _resolve_session_tools(
            cfg, projected, tools, selected, allow_tool_removal=allow_tool_removal
        )

        existing_system = get_current_system_message(
            [message for entry in projected for message in entry.messages]
        )
        bindings = _validate_session_bindings(
            active_tools,
            projected,
            existing_system,
            mcp_update=mcp_update,
            mcp_schema_update=mcp_schema_update,
            selected=selected,
        )
        resources = _resolve_session_resources(cfg, config, projected, existing_system)
        if projected:
            # Resource-only reloads supersede the original configuration snapshot.
            cfg = replace(cfg, skills=resources.settings)

        agent_messages = _prepare_session_prompt(
            store,
            cfg,
            projected,
            active_tools,
            resources,
            bindings,
            existing_system,
            refresh_instructions=refresh_instructions,
            refresh_tools=selected is not None or allow_tool_removal,
        )

        # Persist changed overrides only after configuration and tool validation.
        # Failed construction and identical snapshots must not change the journal.
        if resumed and explicit_override:
            new_snapshot = _config_json(cfg, active_model, effective_cwd)
            if new_snapshot != stored_snapshot:
                store.append_config_change(new_snapshot)

        # Late binding: the hook needs the session instance built below.
        holder: list[AgentSession] = []

        async def prepare_request(context: PrepareRequestContext) -> PrepareRequestUpdate | None:
            return await holder[0]._prepare_request(context)

        async def prepare_next_turn(context: PrepareRequestContext) -> PrepareRequestUpdate | None:
            return await holder[0]._prepare_next_turn(context)

        agent = Agent(
            provider,
            model=active_model,
            tools=active_tools,
            initial_messages=agent_messages,
            thinking_level=cfg.thinking_level,
            session_id=store.session_id,
            prepare_request=prepare_request,
            prepare_next_turn=prepare_next_turn,
            # Check raw-history IDs, including hidden entries, before commit/execution.
            # This collision set is never sent to the model.
            historical_call_ids=lambda: _raw_call_ids(store),
            hooks=hooks,
            permission_policy=cfg.permissions,
            confirm_permission=confirm_permission,
            cwd=effective_cwd,
            tool_execution=cfg.tool_execution,
        )
        instance = session_cls(
            agent=agent, store=store, config=cfg, provider=provider, tools=active_tools, resources=resources
        )
        holder.append(instance)
        return instance
    except BaseException:
        if store is not None:
            store.close()
        raise
