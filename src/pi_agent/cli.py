"""Live print/JSONL runs and read-only session inspection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from . import __version__
from ._session_projection import build_projection
from .cancellation import Cancellation
from .config import DEFAULT_AGENT_DIR, default_session_dir
from .events import AgentSettled, MessageEnd, MessageUpdate, SessionEvent, TextDelta, event_to_json
from .harness import create_harness
from .mcp.config import McpConfig, McpConfigError, load_mcp_config
from .permissions import PermissionPolicy, PermissionRequest
from .provider import Provider
from .providers import AnthropicProvider, OpenAICompatProvider
from .resources import ResourceSnapshot
from .runtime_config import RuntimeSettings, read_environment, resolve_runtime
from .session import AgentSession, create_agent_session
from .session_store import (
    SessionCompactionEntry,
    SessionConfigEntry,
    SessionCorruptError,
    SessionLockError,
    SessionOmissionEntry,
    SessionStore,
    load_session_records,
    read_session_file,
)
from .settings import config_from_settings, load_settings, settings_environment
from .skills import ResourceDiagnostic, SkillSettings
from .tools import TOOL_NAMES, resolve_builtin_selection
from .types import (
    AssistantMessage,
    JsonValue,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolResultMessage,
    get_current_tools,
)


def _diagnostics() -> TextIO:
    return sys.stderr


def _add_skill_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skill", action="append", help="Explicit skill Markdown file/directory (repeatable)"
    )
    enable = parser.add_mutually_exclusive_group()
    enable.add_argument(
        "--skills",
        dest="skills_enabled",
        action="store_true",
        default=None,
        help="Enable skills; explicit on resume reloads the snapshot",
    )
    enable.add_argument(
        "--no-skills",
        dest="skills_enabled",
        action="store_false",
        help="Disable skills; explicit on resume updates the snapshot",
    )
    parser.add_argument(
        "--no-user-skills", action="store_true", default=None, help="Skip automatic user skill roots"
    )
    parser.add_argument(
        "--trust-project-resources",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow project Markdown skills (no script execution)",
    )
    parser.add_argument(
        "--pi-skills",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Discover Pi-specific skill directories too",
    )


def _add_mcp_options(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--mcp-config", metavar="PATH", help="Explicit trusted MCP configuration (stdio/Streamable HTTP)"
    )
    source.add_argument(
        "--mcp-project-config", metavar="PATH", help="Project MCP configuration; requires --trust-project-mcp"
    )
    parser.add_argument(
        "--trust-project-mcp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly trust the selected project MCP configuration",
    )
    parser.add_argument(
        "--mcp-tools",
        type=_mcp_tool_names,
        help="Comma-separated canonical MCP names or server/tool selectors; empty disables all MCP tools",
    )
    parser.add_argument(
        "--mcp-reload",
        action="store_true",
        help="Explicitly accept changed recorded MCP schemas/declarations when resuming",
    )


def _skill_settings(args: argparse.Namespace, base: SkillSettings) -> SkillSettings:
    return replace(
        base,
        enabled=args.skills_enabled
        if args.skills_enabled is not None
        else True
        if args.skill is not None or args.trust_project_resources is True or args.pi_skills is True
        else base.enabled,
        include_user=False if args.no_user_skills else base.include_user,
        project_trusted=args.trust_project_resources
        if args.trust_project_resources is not None
        else base.project_trusted,
        pi_compat=args.pi_skills if args.pi_skills is not None else base.pi_compat,
        paths=tuple(Path(path) for path in args.skill) if args.skill is not None else base.paths,
    )


def _print_resource_diagnostics(diagnostics: tuple[ResourceDiagnostic, ...]) -> None:
    for diagnostic in diagnostics:
        print(f"skills: {diagnostic.path}: {diagnostic.message}", file=_diagnostics())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-agent",
        description="Python coding-agent harness with OpenAI-compatible and Anthropic HTTP providers.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a prompt with the coding tools")
    run_parser.add_argument("prompt", nargs="*", help="Prompt text (or read stdin when omitted)")
    run_parser.add_argument("--json", action="store_true", help="Write lifecycle events as JSONL")
    run_parser.add_argument("--cwd", default=None, help="Tool working directory")
    run_parser.add_argument(
        "--env-file", default=".env", help="Environment file relative to launch directory"
    )
    run_parser.add_argument("--provider", choices=("openai", "deepseek", "anthropic", "sglang"))
    run_parser.add_argument("--model", help="Model ID; also durably switches a resumed session")
    run_parser.add_argument("--base-url", help="HTTP API root")
    run_parser.add_argument("--http-timeout", type=float, help="HTTP inactivity timeout in seconds")
    run_parser.add_argument(
        "--context-window", type=_positive_int, help="Known context limit for auto-compaction"
    )
    run_parser.add_argument(
        "--max-tokens", type=_positive_int, help="Per-response output limit (default 8192)"
    )
    run_parser.add_argument("--thinking", choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"))
    run_parser.add_argument("--temperature", type=float, help="OpenAI-compatible sampling temperature (0–2)")
    run_parser.add_argument("--seed", type=int, help="OpenAI-compatible sampling seed")
    run_parser.add_argument(
        "--chat-template-kwargs", type=_json_object, help="OpenAI-compatible chat template JSON object"
    )
    run_parser.add_argument("--session-dir", help="Directory for new/continued sessions")
    run_parser.add_argument(
        "--settings",
        help="JSON session defaults and runtime request settings (global/project files also load)",
    )
    run_parser.add_argument("--tool-execution", choices=("sequential", "parallel"))
    run_parser.add_argument("--max-parallel-tools", type=_positive_int)
    _add_skill_options(run_parser)
    _add_mcp_options(run_parser)
    run_parser.add_argument(
        "--tools",
        type=_tool_names,
        help="Builtin names or +name/-name edits; empty disables all. Resume edits its recorded set.",
    )
    run_parser.add_argument(
        "--read-only", action="store_true", help="Permit read/grep/find/ls; persisted in the session"
    )
    run_parser.add_argument(
        "--permissions-file", help="JSON tool permission policy; overrides the stored policy"
    )
    run_parser.add_argument(
        "--confirm-permissions",
        action="store_true",
        help="Confirm ask rules interactively (stderr prompts)",
    )
    session_choice = run_parser.add_mutually_exclusive_group()
    session_choice.add_argument(
        "--continue", dest="continue_session", action="store_true", help="Resume latest session for cwd"
    )
    session_choice.add_argument("--resume", help="Resume a session .jsonl file")
    session_choice.add_argument("--no-session", action="store_true", help="Use an in-memory session")
    subparsers.add_parser(
        "chat", parents=[run_parser], add_help=False, help="Interactive REPL with steer/abort"
    )

    skills_parser = subparsers.add_parser("skills", help="List/validate skills without model credentials")
    actions = skills_parser.add_subparsers(dest="skills_command", required=True)
    for action in ("list", "validate"):
        command = actions.add_parser(action)
        command.add_argument("--cwd", default=".")
        command.add_argument("--agent-dir", default=str(DEFAULT_AGENT_DIR))
        command.add_argument("--settings")
        command.add_argument("--json", action="store_true")
        _add_skill_options(command)

    list_parser = subparsers.add_parser("list", help="List sessions for a working directory")
    list_parser.add_argument("--cwd", default=".", help="Working directory (default: current)")
    list_parser.add_argument("--session-dir", default=None, help="Override the session directory")
    list_parser.add_argument(
        "--limit", type=_positive_int, default=20, help="Maximum sessions to print (default: 20)"
    )

    show_parser = subparsers.add_parser("show", help="Inspect one session file (read-only)")
    show_parser.add_argument("path", help="Path to a session .jsonl file")
    show_parser.add_argument(
        "--entries",
        type=_positive_int,
        default=40,
        help="Number of recent entries to summarize (default: 40)",
    )
    return parser


def _json_object(value: str) -> dict[str, JsonValue]:
    try:
        data = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a JSON object") from error
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return data


def _tool_names(value: str) -> tuple[str, ...]:
    try:
        names = tuple(name.strip() for name in value.split(",")) if value else ()
        resolve_builtin_selection(names, ())
        return names
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _mcp_tool_names(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    names = tuple(name.strip() for name in value.split(","))
    if any(not name for name in names):
        raise argparse.ArgumentTypeError("MCP selectors must not contain empty items")
    return names


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


async def _cmd_run(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt)
    if not prompt:
        if sys.stdin.isatty():
            raise ValueError("Provide a prompt or pipe it through stdin")
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise ValueError("Prompt must not be empty")
    session, settings, switch_model, resumed = await _open_session_async(args)

    printed = False
    run_aborted = False

    def output(event: SessionEvent) -> None:
        nonlocal printed, run_aborted
        if isinstance(event, AgentSettled):
            run_aborted = event.aborted
        if args.json:
            print(json.dumps(event_to_json(event), ensure_ascii=False, allow_nan=False), flush=True)
        elif isinstance(event, MessageUpdate) and isinstance(event.stream_event, TextDelta):
            print(event.stream_event.delta, end="", flush=True)
            printed = True
        elif isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            if not printed:
                text = "\n".join(
                    block.text for block in event.message.content if isinstance(block, TextBlock)
                )
                if text:
                    print(text, end="", flush=True)
                    printed = True
            if printed:
                print(flush=True)
            printed = False

    unsubscribe = session.subscribe(output)
    loop = asyncio.get_running_loop()
    abort_tasks: list[asyncio.Task[None]] = []
    run_task = asyncio.current_task()
    installed: list[signal.Signals] = []

    def interrupt() -> None:
        if abort_tasks and run_task is not None:
            run_task.cancel()
        else:
            abort_tasks.append(asyncio.create_task(session.abort()))

    try:
        _apply_cli_changes(session, args, settings, switch_model, resumed)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, interrupt)
                installed.append(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        await session.prompt(prompt)
        if run_aborted:
            print("Operation aborted", file=_diagnostics())
            return 130
        final = next(
            (message for message in reversed(session.messages) if isinstance(message, AssistantMessage)), None
        )
        if final is None:
            raise ValueError("Run produced no assistant response")
        if final.stop_reason in ("error", "aborted", "length"):
            print(final.error_message or f"Run stopped: {final.stop_reason}", file=_diagnostics())
            return 130 if final.stop_reason == "aborted" else 1
        return 0
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)
        if not session.is_idle:
            await session.abort()
        if abort_tasks:
            await asyncio.gather(*abort_tasks, return_exceptions=True)
        unsubscribe()
        await session.aclose()


def _apply_cli_changes(
    session: AgentSession,
    args: argparse.Namespace,
    settings: RuntimeSettings,
    switch_model: bool,
    resumed: bool,
) -> None:
    if resumed:
        if switch_model:
            session.set_model(
                settings.model,
                provider=session.agent.provider,
                thinking_level=cast(ThinkingLevel | None, args.thinking),
            )
        elif args.thinking is not None:
            session.set_thinking_level(cast(ThinkingLevel, args.thinking))
        if args.tool_execution is not None or args.max_parallel_tools is not None:
            session.set_tool_execution(
                args.tool_execution or session.config.tool_execution,
                max_parallel_tools=args.max_parallel_tools,
            )
        if any(
            value is not None
            for value in (
                args.skill,
                args.skills_enabled,
                args.no_user_skills,
                args.trust_project_resources,
                args.pi_skills,
            )
        ):
            _print_resource_diagnostics(
                session.reload_resources(_skill_settings(args, session.config.skills))
            )


def _runtime_provider(settings: RuntimeSettings) -> Provider:
    provider: Provider
    if settings.model.provider == "anthropic":
        provider = AnthropicProvider(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            thinking_mode=cast(Literal["adaptive", "budget"], settings.anthropic_thinking_mode),
        )
    else:
        provider = OpenAICompatProvider(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            request_settings=settings.openai_request,
        )
    return provider


def _session_path(args: argparse.Namespace, cwd: Path) -> Path | None:
    session_path = Path(args.resume).resolve() if args.resume else None
    if args.continue_session:
        directory = Path(args.session_dir) if args.session_dir else default_session_dir(cwd)
        candidates = SessionStore.list_sessions(directory, match_cwd=cwd)
        if not candidates:
            raise ValueError(f"No session to continue for {cwd}")
        session_path = candidates[0].path
    return session_path


def _load_cli_mcp_config(args: argparse.Namespace, cwd: Path, session_path: Path | None) -> McpConfig | None:
    project_path = args.mcp_project_config
    if project_path and not args.trust_project_mcp:
        raise McpConfigError("Project MCP configuration requires --trust-project-mcp")
    path = project_path or args.mcp_config
    if path is None:
        return None
    if session_path is not None and not args.cwd:
        # Inspect committed metadata only. Parsing must precede writer recovery,
        # which can truncate or seal the trailing journal fragment.
        loaded = load_session_records(session_path)
        if loaded.header is not None:
            latest = next(
                (entry.config for entry in reversed(loaded.entries) if isinstance(entry, SessionConfigEntry)),
                loaded.header.config,
            )
            cwd = Path(cast(str, latest.get("cwd") or loaded.header.cwd)).resolve()
    return load_mcp_config(
        Path(path),
        cwd=cwd,
        source="project" if project_path else "explicit",
        project_trusted=args.trust_project_mcp,
    )


def _session_options(
    args: argparse.Namespace, *, with_mcp: bool = False
) -> tuple[RuntimeSettings, bool, dict[str, Any]]:
    cwd = Path(args.cwd or ".").resolve()
    session_path = _session_path(args, cwd)
    mcp_config = _load_cli_mcp_config(args, cwd, session_path) if with_mcp else None
    restored: ModelInfo | None = None
    restored_tools: tuple[str, ...] = ()
    settings_cwd = cwd
    stored_permissions: object = {"default": "allow"}
    if session_path is not None:
        # Resume owns recovery; the read-only show parser deliberately rejects torn tails.
        store = SessionStore.open(session_path)
        try:
            declared = get_current_tools(
                [message for entry in build_projection(store) for message in entry.messages]
            )
            restored_tools = tuple(tool.name for tool in declared if tool.name in TOOL_NAMES)
            if not with_mcp and any(tool.name.startswith("mcp__") for tool in declared):
                raise McpConfigError(
                    "Saved MCP tools require explicitly supplied trusted configuration; "
                    "use --mcp-tools '' to clear them"
                )
            model_data = store.header.model
            stored_permissions = store.header.config.get("permissions", {"default": "allow"})
            for entry in store.entries:
                if isinstance(entry, SessionConfigEntry):
                    stored_permissions = entry.config.get("permissions", {"default": "allow"})
                if isinstance(entry, SessionConfigEntry) and isinstance(entry.config.get("model"), dict):
                    model_data = cast(dict, entry.config["model"])
            if model_data is not None:
                restored = ModelInfo.from_json(model_data)
            if not args.cwd:
                latest = next(
                    (
                        entry.config
                        for entry in reversed(store.entries)
                        if isinstance(entry, SessionConfigEntry)
                    ),
                    store.header.config,
                )
                settings_cwd = Path(cast(str, latest.get("cwd") or store.cwd)).resolve()
        finally:
            store.close()
    defaults = load_settings(settings_cwd, explicit=Path(args.settings).resolve() if args.settings else None)
    if session_path is None:
        stored_permissions = defaults.get("permissions", {"default": "allow"})
    switch_model = restored is not None and any(
        value is not None for value in (args.provider, args.model, args.context_window, args.max_tokens)
    )
    target_provider = args.provider or (restored.provider if restored is not None else None)
    target_id = args.model
    if switch_model and restored is not None and target_id is None and target_provider == restored.provider:
        target_id = restored.id
    settings = resolve_runtime(
        {**settings_environment(defaults), **read_environment(Path(args.env_file).resolve())},
        provider=target_provider if switch_model else args.provider,
        model_id=target_id if switch_model else args.model,
        base_url=args.base_url,
        timeout=args.http_timeout,
        context_window=args.context_window
        if args.context_window is not None
        else (defaults.get("context_window") if session_path is None else None),
        max_tokens=args.max_tokens
        if args.max_tokens is not None
        else (defaults.get("max_tokens") if session_path is None else None),
        restored_model=None if switch_model else restored,
        temperature=args.temperature if args.temperature is not None else defaults.get("temperature"),
        seed=args.seed if args.seed is not None else defaults.get("seed"),
        chat_template_kwargs=args.chat_template_kwargs
        if args.chat_template_kwargs is not None
        else defaults.get("chat_template_kwargs"),
        thinking_level_map=defaults.get("thinking_level_map") if session_path is None else None,
    )
    if (
        switch_model
        and restored is not None
        and (settings.model.provider, settings.model.id) == (restored.provider, restored.id)
    ):
        # Limit-only changes keep the declared identity/capabilities, never shell defaults.
        settings = replace(
            settings,
            model=replace(
                restored,
                context_window=args.context_window
                if args.context_window is not None
                else restored.context_window,
                max_tokens=args.max_tokens if args.max_tokens is not None else restored.max_tokens,
            ),
        )
    selected_policy: PermissionPolicy | None = None
    if args.permissions_file:
        selected_policy = _read_permission_policy(Path(args.permissions_file))
    if args.read_only:
        selected_policy = replace(
            selected_policy or PermissionPolicy.from_json(stored_permissions), read_only=True
        )
    confirm = _confirm_on_tty if args.confirm_permissions else None
    selected_tools = (
        resolve_builtin_selection(
            args.tools, restored_tools if session_path is not None else defaults.get("tools", TOOL_NAMES)
        )
        if args.tools is not None
        else None
    )
    options: dict[str, Any] = {
        "permission_policy": selected_policy,
        "confirm_permission": confirm,
    }
    if session_path is not None:
        options.update(
            session=session_path,
            cwd=cwd if args.cwd else None,
            tool_names=selected_tools,
        )
    else:
        config = config_from_settings(cwd, settings.model, defaults)
        config = replace(
            config,
            skills=_skill_settings(
                args, SkillSettings.from_json({"enabled": True, **defaults.get("skills", {})})
            ),
            thinking_level=cast(ThinkingLevel, args.thinking or config.thinking_level),
            session_dir=Path(args.session_dir).resolve() if args.session_dir else None,
            tool_execution=cast(
                Literal["sequential", "parallel"], args.tool_execution or config.tool_execution
            ),
            max_parallel_tools=args.max_parallel_tools
            if args.max_parallel_tools is not None
            else config.max_parallel_tools,
        )
        options.update(
            config=config,
            session=None if args.no_session else "new",
            tool_names=selected_tools if selected_tools is not None else defaults.get("tools", TOOL_NAMES),
        )
    if with_mcp:
        options.update(mcp_config=mcp_config, mcp_tools=args.mcp_tools, refresh_mcp=args.mcp_reload)
    return settings, switch_model, options


def _open_session(args: argparse.Namespace) -> tuple[AgentSession, RuntimeSettings, bool, bool]:
    settings, switch_model, options = _session_options(args)
    session = create_agent_session(_runtime_provider(settings), **options)
    _print_resource_diagnostics(session.resource_diagnostics)
    return session, settings, switch_model, isinstance(options["session"], Path)


async def _open_session_async(args: argparse.Namespace) -> tuple[AgentSession, RuntimeSettings, bool, bool]:
    if not (
        args.mcp_config is not None
        or args.mcp_project_config is not None
        or args.mcp_tools is not None
        or args.mcp_reload
    ):
        return _open_session(args)
    settings, switch_model, options = _session_options(args, with_mcp=True)
    session = await create_harness(_runtime_provider(settings), **options)
    try:
        _print_resource_diagnostics(session.resource_diagnostics)
    except BaseException:
        await session.aclose()
        raise
    return session, settings, switch_model, isinstance(options["session"], Path)


def _cmd_skills(args: argparse.Namespace) -> int:
    try:
        cwd, agent_dir = Path(args.cwd).resolve(), Path(args.agent_dir).expanduser().resolve()
        defaults = load_settings(
            cwd, explicit=Path(args.settings).resolve() if args.settings else None, agent_dir=agent_dir
        )
        selected = _skill_settings(
            args, SkillSettings.from_json({"enabled": True, **defaults.get("skills", {})})
        )
        snapshot = ResourceSnapshot.discover(selected, cwd, agent_dir)
        if args.json:
            print(
                json.dumps(
                    {
                        "skills": [skill.to_json() for skill in snapshot.skills],
                        "diagnostics": [diagnostic.to_json() for diagnostic in snapshot.diagnostics],
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        else:
            for skill in snapshot.skills:
                print(f"{skill.name}: {skill.description}\n  {skill.file_path}")
            _print_resource_diagnostics(snapshot.diagnostics)
        return 1 if args.skills_command == "validate" and snapshot.diagnostics else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=_diagnostics())
        return 2


def _read_permission_policy(path: Path) -> PermissionPolicy:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key in permission policy")
            result[key] = value
        return result

    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    return PermissionPolicy.from_json(data)


async def _confirm_on_tty(request: PermissionRequest, cancellation: Cancellation) -> bool:
    """Read terminal input via add_reader, so cancellation never leaves a blocked thread."""
    cancellation.raise_if_aborted()
    try:
        fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
    except OSError:
        print("Permission denied: no terminal is available", file=_diagnostics())
        return False
    loop = asyncio.get_running_loop()
    answer: asyncio.Future[bool] = loop.create_future()
    buffer = bytearray()

    def readable() -> None:
        if answer.done():
            return
        try:
            chunk = os.read(fd, 1024)
        except BlockingIOError:
            return
        except OSError:
            answer.set_result(False)
            return
        buffer.extend(chunk)
        if b"\n" in buffer:
            first_line = bytes(buffer).partition(b"\n")[0].strip().lower()
            answer.set_result(first_line in (b"y", b"yes"))
        elif not chunk or len(buffer) >= 4096:
            answer.set_result(False)

    try:
        print(
            "Permission required: "
            + json.dumps(
                {
                    "tool": request.context.tool_call.name,
                    "args": request.context.args,
                    "reason": request.reason,
                    "resolved_path": request.resolved_path,
                },
                ensure_ascii=False,
            ),
            file=_diagnostics(),
        )
        print("Allow this call once? [y/N]: ", end="", flush=True, file=_diagnostics())
        loop.add_reader(fd, readable)
        return await answer
    finally:
        loop.remove_reader(fd)
        os.close(fd)
        print(file=_diagnostics())


def _cmd_list(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir) if args.session_dir else default_session_dir(args.cwd)
    sessions = SessionStore.list_sessions(session_dir, match_cwd=Path(args.cwd))
    if not sessions:
        print(f"No sessions found in {session_dir}", file=_diagnostics())
        return 0
    for info in sessions[: args.limit]:
        created = info.created.split("T")[0] if "T" in info.created else info.created
        preview = info.first_message[:72]
        print(f"{created}  {info.message_count:>4} msgs  {info.id[:8]}  {preview}")
        print(f"    {info.path}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        header, entries = read_session_file(path)
    except FileNotFoundError:
        print(f"Session file not found: {path}", file=_diagnostics())
        return 2
    except SessionCorruptError as error:
        print(str(error), file=_diagnostics())
        return 2
    print(f"session      {header.id}")
    print(f"version      {header.version}")
    print(f"created      {header.timestamp}")
    print(f"cwd          {header.cwd}")
    if header.model:
        print(f"model        {header.model.get('provider', '?')}/{header.model.get('id', '?')}")
    print(f"entries      {len(entries)}")
    if header.config:
        print(f"config       {json.dumps(header.config, ensure_ascii=False)}")
    print()
    for entry in entries[-args.entries :]:
        if isinstance(entry, SessionCompactionEntry):
            print(f"[compaction] {entry.timestamp} kept-from={entry.first_kept_entry_id[:8]}")
            for line in entry.summary.splitlines()[:4]:
                print(f"    {line}")
            continue
        if isinstance(entry, SessionOmissionEntry):
            print(f"[omission  ] {entry.timestamp} target={entry.target_entry_id[:8]}")
            continue
        if isinstance(entry, SessionConfigEntry):
            print(f"[config    ] {entry.timestamp} {json.dumps(entry.config, ensure_ascii=False)}")
            continue
        message = entry.message
        role = getattr(message, "role", "?")
        if isinstance(message, AssistantMessage):
            stop = getattr(message, "stop_reason", "?")
            print(f"[{role:<11}] {entry.timestamp} stop={stop}")
            for block in message.content:
                if isinstance(block, TextBlock):
                    text = block.text.strip()
                    if text:
                        for line in text.splitlines()[:3]:
                            print(f"    {line}")
                elif isinstance(block, ToolCallBlock):
                    print(f"    tool-call {block.name}({json.dumps(block.arguments, ensure_ascii=False)})")
        elif isinstance(message, ToolResultMessage):
            status = "interrupted" if message.interrupted else ("error" if message.is_error else "ok")
            text = (message.content[0].text if message.content else "").strip()
            print(f"[{role:<11}] {entry.timestamp} {message.tool_name} {status}")
            for line in text.splitlines()[:3]:
                print(f"    {line}")
        elif isinstance(message, SystemMessage):
            sections = list((message.sections or {}).keys())
            print(f"[{role:<11}] {entry.timestamp} sections={sections}")
        else:
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                content = " ".join(getattr(block, "text", "") for block in content)
            preview = content.strip().splitlines()[0] if content.strip() else ""
            print(f"[{role:<11}] {entry.timestamp} {preview[:100]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command in ("run", "chat"):
        try:
            if args.command == "chat":
                from .chat import run_chat

                return asyncio.run(run_chat(args))
            return asyncio.run(_cmd_run(args))
        except (KeyboardInterrupt, asyncio.CancelledError):
            return 130
        except (OSError, ValueError, RuntimeError, SessionCorruptError, SessionLockError) as error:
            print(str(error), file=_diagnostics())
            return 2
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "show":
        return _cmd_show(args)
    if args.command == "skills":
        return _cmd_skills(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
