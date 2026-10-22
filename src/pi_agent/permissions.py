"""Serializable tool-dispatch permissions, not an OS sandbox or a shell parser."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal, cast

from .cancellation import Cancellation
from .tool_hooks import BeforeToolCallContext, BeforeToolCallResult
from .tools.base import resolve_path
from .types import JsonValue

PermissionAction = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class PermissionRule:
    tool: str
    action: PermissionAction
    command: str | None = None
    path: str | None = None
    reason: str | None = None

    def validate(self) -> None:
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError("Permission rule tool must be a nonempty glob")
        if self.action not in ("allow", "ask", "deny"):
            raise ValueError("Permission rule action must be allow, ask or deny")
        for field in (self.command, self.path, self.reason):
            if field is not None and (not isinstance(field, str) or not field):
                raise ValueError("Permission command/path/reason must be nonempty text or null")
        if self.path is not None and ".." in Path(self.path).parts:
            raise ValueError("Permission path patterns must not contain '..'; use an absolute path")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "tool": self.tool,
            "action": self.action,
            "command": self.command,
            "path": self.path,
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, data: object) -> PermissionRule:
        if not isinstance(data, dict) or set(data) - {"tool", "action", "command", "path", "reason"}:
            raise ValueError("Invalid permission rule")
        tool, action = data.get("tool"), data.get("action")
        if not isinstance(tool, str) or action not in ("allow", "ask", "deny"):
            raise ValueError("Invalid permission rule tool or action")
        rule = cls(
            tool, cast(PermissionAction, action), data.get("command"), data.get("path"), data.get("reason")
        )
        rule.validate()
        return rule


@dataclass(frozen=True)
class PermissionPolicy:
    default: PermissionAction = "allow"
    read_only: bool = False
    rules: tuple[PermissionRule, ...] = ()

    def validate(self) -> None:
        if self.default not in ("allow", "ask", "deny") or not isinstance(self.read_only, bool):
            raise ValueError("Invalid permission default or read_only")
        if not isinstance(self.rules, tuple) or any(
            not isinstance(rule, PermissionRule) for rule in self.rules
        ):
            raise ValueError("Permission rules must be a tuple of PermissionRule")
        for rule in self.rules:
            rule.validate()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "default": self.default,
            "read_only": self.read_only,
            "rules": [rule.to_json() for rule in self.rules],
        }

    @classmethod
    def from_json(cls, data: object) -> PermissionPolicy:
        if not isinstance(data, dict) or set(data) - {"default", "read_only", "rules"}:
            raise ValueError("Invalid permissions configuration")
        rules = data.get("rules", [])
        if not isinstance(rules, list):
            raise ValueError("Permission rules must be an array")
        policy = cls(
            data.get("default", "allow"),
            data.get("read_only", False),
            tuple(PermissionRule.from_json(rule) for rule in rules),
        )
        policy.validate()
        return policy


@dataclass(frozen=True)
class PermissionRequest:
    context: BeforeToolCallContext
    reason: str
    resolved_path: str | None = None


PermissionPrompt = Callable[[PermissionRequest, Cancellation], bool | Awaitable[bool]]


def _resolved_pattern(pattern: str, cwd: Path | None) -> str:
    path = Path(pattern).expanduser()
    if not path.is_absolute():
        if cwd is None:
            raise ValueError("Relative permission path requires a tool cwd")
        path = cwd / path
    parts = path.parts
    first_glob = next(
        (index for index, part in enumerate(parts) if any(char in part for char in "*?[")), len(parts)
    )
    prefix = Path(*parts[:first_glob]).resolve()
    return (prefix.joinpath(*parts[first_glob:])).as_posix()


def _target_path(context: BeforeToolCallContext) -> str | None:
    value = context.args.get("path", "." if context.builtin_search else None)
    if not isinstance(value, str):
        return None
    if context.cwd is None and not Path(value).expanduser().is_absolute():
        raise ValueError("Relative tool path requires a cwd for permissions")
    return resolve_path(value, context.cwd or Path("/")).as_posix()


def permission_decision(
    policy: PermissionPolicy, context: BeforeToolCallContext
) -> tuple[PermissionAction, str]:
    """Read-only takes priority; otherwise the first matching rule wins."""
    if policy.read_only and context.tool_call.name != "read" and not context.builtin_search:
        return "deny", "Read-only policy permits only read and verified builtin search tools"
    for rule in policy.rules:
        if not fnmatchcase(context.tool_call.name, rule.tool):
            continue
        if rule.command is not None:
            command = context.args.get("command")
            if (
                context.tool_call.name != "bash"
                or not isinstance(command, str)
                or not fnmatchcase(command, rule.command)
            ):
                continue
        if rule.path is not None:
            path = _target_path(context)
            if path is None or not fnmatchcase(path, _resolved_pattern(rule.path, context.cwd)):
                continue
        return rule.action, rule.reason or f"Permission rule: {rule.action} {rule.tool}"
    return policy.default, f"Default permission: {policy.default}"


async def check_permission(
    context: BeforeToolCallContext,
    cancellation: Cancellation,
    *,
    policy: PermissionPolicy,
    confirm: PermissionPrompt | None,
) -> BeforeToolCallResult | None:
    action, reason = permission_decision(policy, context)
    if action == "allow":
        return None
    if action == "deny":
        return BeforeToolCallResult(block=True, reason=f"Tool execution denied: {reason}")
    if confirm is None:
        return BeforeToolCallResult(
            block=True,
            reason=f"Tool execution requires confirmation: {reason}; no confirmation callback is available",
        )
    original_path = _target_path(context)
    request = PermissionRequest(context=deepcopy(context), reason=reason, resolved_path=original_path)
    approved = confirm(request, cancellation)
    if inspect.isawaitable(approved):
        approved = await approved
    if not isinstance(approved, bool):
        raise ValueError("Permission confirmation must return a boolean")
    cancellation.raise_if_aborted()
    if not approved:
        return BeforeToolCallResult(block=True, reason=f"Tool execution denied by confirmation: {reason}")
    if _target_path(context) != original_path:
        return BeforeToolCallResult(
            block=True, reason="Tool target changed while awaiting confirmation; request permission again"
        )
    return None
