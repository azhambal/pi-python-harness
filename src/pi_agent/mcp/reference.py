"""Opt-in example profile for the pinned reference Git/Fetch servers.

The caller must pass both the selection and guard to its harness. This is host
admission policy, not an OS sandbox. Fetch always requires a permission callback;
approving an initial URL does not constrain subsequent redirects or DNS changes.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from pi_agent.cancellation import Cancellation
from pi_agent.permissions import PermissionPolicy, PermissionRule
from pi_agent.tool_hooks import BeforeToolCallContext, BeforeToolCallResult, ToolHooks

from ._contracts import binding_name
from .config import McpConfig, parse_mcp_config

GIT_READ_TOOLS = (
    "git_status",
    "git_diff_unstaged",
    "git_diff_staged",
    "git_diff",
    "git_log",
    "git_show",
    "git_branch",
)
REFERENCE_SELECTORS = tuple(f"git/{name}" for name in GIT_READ_TOOLS) + ("fetch/fetch",)
REFERENCE_VERSIONS = {"mcp": "2.3.0", "mcp-server-git": "2026.10.19", "mcp-server-fetch": "2026.10.19"}


def reference_config(
    repository: Path, python_executable: Path, *, allow_private_fetch: bool = False
) -> McpConfig:
    """Explicit host paths; no uvx/runtime downloads or environment discovery."""
    if type(allow_private_fetch) is not bool:
        raise ValueError("allow_private_fetch must be boolean")
    if not isinstance(repository, Path) or not repository.is_absolute() or not repository.is_dir():
        raise ValueError("Reference profile requires an existing absolute repository directory")
    # Keep the venv entry point, rather than resolving its symlink to base Python.
    if (
        not isinstance(python_executable, Path)
        or not python_executable.is_absolute()
        or not python_executable.is_file()
    ):
        raise ValueError("Reference profile requires an absolute Python executable")
    root = repository.resolve()
    return parse_mcp_config(
        {
            "mcpServers": {
                "git": {
                    "command": str(python_executable),
                    "args": ["-m", "mcp_server_git", "--repository", str(root)],
                    "cwd": str(root),
                    "timeout": 15,
                    "exposure": "direct",
                },
                "fetch": {
                    "command": str(python_executable),
                    "args": [
                        "-m",
                        "mcp_server_fetch",
                        *(["--allow-private-ips"] if allow_private_fetch else []),
                    ],
                    "cwd": str(root),
                    "timeout": 15,
                    "exposure": "direct",
                },
            }
        },
        cwd=root,
    )


def reference_hooks(repository: Path) -> ToolHooks:
    """Bind every selected Git call to exactly this root, including symlinks.

    The reference server allows a subtree; this guard rejects nested repositories
    as well. Arguments cannot substitute an unrelated repository or Git mutation.
    Fetch admits only HTTP(S) URLs without inline credentials, then asks permission.
    """
    root = repository.resolve()

    def before(context: BeforeToolCallContext, cancellation: Cancellation) -> BeforeToolCallResult | None:
        provenance = context.tool_provenance
        if provenance is None:
            if context.tool_call.name.startswith("mcp__"):
                return BeforeToolCallResult(block=True, reason="Call lacks reference MCP provenance")
            return None
        allowed = False
        if provenance.kind == "mcp" and context.tool_call.name == binding_name(
            provenance.server, provenance.tool
        ):
            if provenance.server == "git" and provenance.tool in GIT_READ_TOOLS:
                value = context.args.get("repo_path")
                if isinstance(value, str) and value:
                    try:
                        allowed = (root / value).resolve() == root
                    except (OSError, RuntimeError, ValueError):
                        pass
            elif provenance.server == "fetch" and provenance.tool == "fetch":
                value = context.args.get("url")
                if isinstance(value, str):
                    try:
                        url = urlsplit(value)
                        allowed = (
                            url.scheme in ("http", "https")
                            and bool(url.hostname)
                            and url.username is None
                            and url.password is None
                        )
                    except ValueError:
                        pass
        if not allowed:
            return BeforeToolCallResult(block=True, reason="Call is outside the reference MCP profile")
        return None

    return ToolHooks(before_tool_call=before)


def reference_permissions() -> PermissionPolicy:
    """Builtins keep ordinary permissions; selected Git reads allow, Fetch asks."""
    return PermissionPolicy(
        rules=(
            *(PermissionRule(binding_name("git", name), "allow") for name in GIT_READ_TOOLS),
            PermissionRule(
                binding_name("fetch", "fetch"),
                "ask",
                reason="Approve the destination and redirect/network scope explicitly",
            ),
            PermissionRule("mcp__*", "deny"),
        )
    )
