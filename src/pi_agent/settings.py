"""Strict global/project/explicit defaults; credentials remain runtime-only."""

import json
import math
from pathlib import Path

from .config import DEFAULT_AGENT_DIR, AgentConfig, CompactionSettings, RetrySettings
from .permissions import PermissionPolicy
from .providers.openai_compat import OpenAIRequestSettings
from .skills import SkillSettings
from .tools import validate_builtin_names
from .types import ModelInfo

_FIELDS = {
    "$schema",
    "provider",
    "model",
    "base_url",
    "http_timeout",
    "thinking_level",
    "thinking_level_map",
    "context_window",
    "max_tokens",
    "compaction",
    "retry",
    "bash_default_timeout_s",
    "tool_execution",
    "permissions",
    "tools",
    "skills",
    "temperature",
    "seed",
    "chat_template_kwargs",
    "thinking_budgets",
}

#: Pi's ``thinkingBudgets``: token budgets per thinking level for budget-based providers.
THINKING_BUDGET_LEVELS = ("minimal", "low", "medium", "high")


def validate_thinking_budgets(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) - set(THINKING_BUDGET_LEVELS):
        raise ValueError("Settings thinking_budgets may only set minimal, low, medium and high")
    for level, budget in value.items():
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError(f"Settings thinking_budgets.{level} must be a non-negative integer")
    return dict(value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate settings key: {key}")
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError("Non-finite number in settings")


def _validate(data: object) -> dict:
    if not isinstance(data, dict) or set(data) - _FIELDS:
        raise ValueError("Settings must be an object containing only supported non-secret fields")
    if "$schema" in data:
        if not isinstance(data["$schema"], str) or not data["$schema"].strip():
            raise ValueError("Settings $schema must be nonempty text")
        data = {key: value for key, value in data.items() if key != "$schema"}
    for key in ("model", "base_url"):
        if key in data and (not isinstance(data[key], str) or not data[key]):
            raise ValueError(f"Settings {key} must be nonempty text")
    if "provider" in data and data["provider"] not in ("openai", "deepseek", "anthropic", "sglang"):
        raise ValueError("Settings provider must be openai, deepseek, anthropic or sglang")
    OpenAIRequestSettings(data.get("temperature"), data.get("seed"), data.get("chat_template_kwargs"))
    if "thinking_level_map" in data:
        ModelInfo.from_json(
            {
                **ModelInfo("settings", "settings", "settings").to_json(),
                "thinking_level_map": data["thinking_level_map"],
            }
        )
    for key in ("context_window", "max_tokens"):
        if key in data and (isinstance(data[key], bool) or not isinstance(data[key], int) or data[key] <= 0):
            raise ValueError(f"Settings {key} must be a positive integer")
    if "http_timeout" in data and (
        isinstance(data["http_timeout"], bool)
        or not isinstance(data["http_timeout"], (int, float))
        or not math.isfinite(data["http_timeout"])
        or data["http_timeout"] <= 0
    ):
        raise ValueError("Settings http_timeout must be finite and positive")
    for key, fields, parser in (
        ("compaction", {"enabled", "reserve_tokens", "keep_recent_tokens"}, CompactionSettings.from_json),
        ("retry", {"enabled", "max_retries", "base_delay_ms", "max_delay_ms"}, RetrySettings.from_json),
    ):
        if key in data:
            value = data[key]
            if not isinstance(value, dict) or set(value) - fields:
                raise ValueError(f"Invalid settings {key} fields")
            parser(value)
    if "thinking_budgets" in data:
        validate_thinking_budgets(data["thinking_budgets"])
    if "permissions" in data:
        PermissionPolicy.from_json(data["permissions"])
    if "skills" in data:
        SkillSettings.from_json(data["skills"])
    if "tools" in data:
        if not isinstance(data["tools"], list):
            raise ValueError("Settings tools must be a list of builtin names")
        validate_builtin_names(data["tools"])
    # Validate enum/budgets with the same configuration parser as the harness.
    config_from_settings(Path("."), ModelInfo("settings", "settings", "settings"), data).validate()
    return data


def _merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if key in ("compaction", "retry", "skills", "thinking_budgets") and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def load_settings(cwd: Path, *, explicit: Path | None = None, agent_dir: Path = DEFAULT_AGENT_DIR) -> dict:
    """Global -> project -> explicit; nested budgets merge, policy replaces.

    Automatic missing files are ignored; explicit missing files are errors.
    Session defaults apply to new sessions; resume keeps journaled configuration.
    Runtime transport settings are reloaded by the CLI on each invocation.
    """
    result: dict = {}
    automatic = [agent_dir / "settings.json", cwd / ".pi-agent" / "settings.json"]
    for index, path in enumerate([*automatic, *([explicit] if explicit is not None else [])]):
        if index < len(automatic) and not path.exists():
            continue
        data = json.loads(
            path.read_text(encoding="utf-8-sig"), object_pairs_hook=_object, parse_constant=_invalid_number
        )
        result = _merge(result, _validate(data))
    return _validate(result)


def settings_environment(settings: dict) -> dict[str, str]:
    """Lowest-priority non-secret defaults under dotenv/shell/CLI values."""
    result = {}
    prefix = "ANTHROPIC" if settings.get("provider") == "anthropic" else "OPENAI"
    for key, names in {
        "provider": ("PI_PROVIDER",),
        "model": (f"{prefix}_MODEL",),
        "base_url": (f"{prefix}_BASE_URL",),
        "http_timeout": ("PI_HTTP_TIMEOUT",),
    }.items():
        if key in settings:
            for name in names:
                result[name] = str(settings[key])
    return result


def config_from_settings(cwd: Path, model: ModelInfo, settings: dict) -> AgentConfig:
    return AgentConfig(
        cwd=cwd,
        model=model,
        thinking_level=settings.get("thinking_level", "off"),
        compaction=CompactionSettings.from_json(settings.get("compaction", {})),
        retry=RetrySettings.from_json(settings.get("retry", {})),
        bash_default_timeout_s=settings.get("bash_default_timeout_s"),
        permissions=PermissionPolicy.from_json(settings.get("permissions", {})),
        tool_execution=settings.get("tool_execution", "parallel"),
        skills=SkillSettings.from_json(settings.get("skills", {})),
    )
