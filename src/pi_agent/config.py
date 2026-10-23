"""Session configuration and validation; provider credentials remain runtime-only."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .permissions import PermissionPolicy
from .skills import SkillSettings
from .types import _THINKING_LEVELS, ModelInfo, ThinkingLevel

#: Default agent state directory: ~/.pi-agent (sessions live under
#: sessions/<hash>/, one directory per exact cwd).
DEFAULT_AGENT_DIR = Path.home() / ".pi-agent"


@dataclass
class CompactionSettings:
    """Compaction thresholds (defaults mirror Pi's DEFAULT_COMPACTION_SETTINGS)."""

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000

    def validate(self) -> None:
        _bool_value(self.enabled, "compaction.enabled")
        _int_value(self.reserve_tokens, "compaction.reserve_tokens", minimum=0)
        _int_value(self.keep_recent_tokens, "compaction.keep_recent_tokens", minimum=0)

    @classmethod
    def from_json(cls, data: object) -> CompactionSettings:
        if not isinstance(data, dict):
            raise ValueError("compaction settings must be an object")
        settings = cls(
            enabled=_bool_field(data, "enabled", default=True),
            reserve_tokens=_int_field(data, "reserve_tokens", minimum=0, default=16384),
            keep_recent_tokens=_int_field(data, "keep_recent_tokens", minimum=0, default=20000),
        )
        settings.validate()
        return settings


@dataclass
class RetrySettings:
    """Bounded retry policy for transient provider errors only.

    Defaults mirror Pi's settings-manager defaults: exponential backoff of
    ``base_delay_ms * 2^(attempt-1)`` (2s, 4s, 8s), capped at 60s, with at
    most 3 retries after the initial failed attempt. Retries re-issue the
    provider request from the last user/tool-result message; completed tools
    are never re-executed as a side effect.
    """

    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 2000
    max_delay_ms: int = 60000

    def validate(self) -> None:
        _bool_value(self.enabled, "retry.enabled")
        _int_value(self.max_retries, "retry.max_retries", minimum=0)
        _int_value(self.base_delay_ms, "retry.base_delay_ms", minimum=1)
        _int_value(self.max_delay_ms, "retry.max_delay_ms", minimum=1)
        if self.max_delay_ms < self.base_delay_ms:
            raise ValueError("retry.max_delay_ms must be >= retry.base_delay_ms")

    def delay_ms(self, attempt: int) -> int:
        """Backoff for a 1-indexed attempt, capped at ``max_delay_ms``."""
        return min(self.base_delay_ms * 2 ** max(0, attempt - 1), self.max_delay_ms)

    @classmethod
    def from_json(cls, data: object) -> RetrySettings:
        if not isinstance(data, dict):
            raise ValueError("retry settings must be an object")
        settings = cls(
            enabled=_bool_field(data, "enabled", default=True),
            max_retries=_int_field(data, "max_retries", minimum=0, default=3),
            base_delay_ms=_int_field(data, "base_delay_ms", minimum=1, default=2000),
            max_delay_ms=_int_field(data, "max_delay_ms", minimum=1, default=60000),
        )
        settings.validate()
        return settings


def default_session_dir(cwd: Path | str, agent_dir: Path | None = None) -> Path:
    """Session directory for one exact cwd: ``<agent_dir>/sessions/<sha256[:16]>``.

    The directory name hashes the resolved cwd to avoid collisions caused by
    lossy path encoding. Hash collisions remain possible; discovery also
    filters each session header by exact cwd. Pi encodes the path into the
    directory name; this port uses the header to recover the original path.
    """
    base = agent_dir if agent_dir is not None else DEFAULT_AGENT_DIR
    resolved = str(Path(cwd).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return base / "sessions" / digest


@dataclass
class AgentConfig:
    """Everything the session needs besides the provider and tools."""

    cwd: Path
    model: ModelInfo
    thinking_level: ThinkingLevel = "off"
    agent_dir: Path = field(default_factory=lambda: DEFAULT_AGENT_DIR)
    session_dir: Path | None = None
    compaction: CompactionSettings = field(default_factory=CompactionSettings)
    retry: RetrySettings = field(default_factory=RetrySettings)
    #: Default bash timeout in seconds; None means no default timeout (the
    #: model may still pass a per-call ``timeout`` argument).
    bash_default_timeout_s: float | None = None
    #: Explicit instruction files. When None, discovered deterministically
    #: (global <agent_dir>/AGENTS.md, then every AGENTS.md from cwd up to /).
    instruction_files: list[Path] | None = None
    permissions: PermissionPolicy = field(default_factory=PermissionPolicy)
    #: Pi's ``toolExecution``: "parallel" (default) prepares calls in order and
    #: then runs them concurrently; "sequential" runs one call at a time.
    tool_execution: Literal["sequential", "parallel"] = "parallel"
    skills: SkillSettings = field(default_factory=SkillSettings)

    def validate(self) -> None:
        if not isinstance(self.cwd, Path):
            raise ValueError("config.cwd must be a Path")
        ModelInfo.from_json(self.model.to_json())
        if not isinstance(self.thinking_level, str) or self.thinking_level not in _THINKING_LEVELS:
            raise ValueError("thinking_level is invalid")
        self.model.validate_thinking_level(self.thinking_level if self.model.reasoning else "off")
        self.compaction.validate()
        self.retry.validate()
        self.permissions.validate()
        self.skills.validate()
        if self.tool_execution not in ("sequential", "parallel"):
            raise ValueError("tool_execution must be sequential or parallel")
        if self.bash_default_timeout_s is not None:
            timeout = self.bash_default_timeout_s
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise ValueError("config.bash_default_timeout_s must be a number or None")
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("config.bash_default_timeout_s must be finite and > 0")

    def resolved_session_dir(self) -> Path:
        return (
            self.session_dir
            if self.session_dir is not None
            else default_session_dir(self.cwd, self.agent_dir)
        )


def _bool_field(data: dict[str, object], key: str, *, default: bool) -> bool:
    return _bool_value(data.get(key, default), key)


def _bool_value(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"settings field '{key}' must be a boolean")
    return value


def _int_field(data: dict[str, object], key: str, *, minimum: int, default: int) -> int:
    return _int_value(data.get(key, default), key, minimum=minimum)


def _int_value(value: object, key: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"settings field '{key}' must be an integer")
    if value < minimum:
        raise ValueError(f"settings field '{key}' must be >= {minimum}")
    return value
