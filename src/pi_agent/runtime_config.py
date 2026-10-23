"""CLI-only environment settings; credentials never enter AgentConfig or the journal."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import dotenv_values

from .models import find_model
from .providers.openai_compat import OpenAIRequestSettings
from .types import JsonValue, ModelInfo

_ENV_NAMES = (
    "PI_PROVIDER",
    "PI_HTTP_TIMEOUT",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_THINKING_MODE",
)


def read_environment(path: Path | None) -> dict[str, str]:
    """Load only the named file; shell environment wins. Never export keys to tools.

    No upward .env discovery, interpolation, or logging of values is performed.
    """
    values = dotenv_values(path, interpolate=False) if path is not None and path.is_file() else {}
    return {
        name: os.environ.get(name, values.get(name) or "")
        for name in _ENV_NAMES
        if name in os.environ or name in values
    }


@dataclass(frozen=True)
class RuntimeSettings:
    model: ModelInfo
    base_url: str
    api_key: str = field(repr=False)
    timeout: float = 300.0
    #: None follows the model catalog (adaptive for models that force it, as in Pi).
    anthropic_thinking_mode: str | None = None
    openai_request: OpenAIRequestSettings = field(default_factory=OpenAIRequestSettings)
    #: Pi's ``thinkingBudgets`` overrides of the per-level token budgets.
    thinking_budgets: dict[str, int] | None = None


def resolve_runtime(
    environment: Mapping[str, str],
    *,
    provider: str | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    context_window: int | None = None,
    max_tokens: int | None = None,
    restored_model: ModelInfo | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    chat_template_kwargs: dict[str, JsonValue] | None = None,
    thinking_level_map: dict[str, str | None] | None = None,
    thinking_budgets: dict[str, int] | None = None,
) -> RuntimeSettings:
    provider = (
        provider
        or (restored_model.provider if restored_model else environment.get("PI_PROVIDER"))
        or "openai"
    )
    if provider not in ("openai", "deepseek", "anthropic", "sglang"):
        raise ValueError("PI_PROVIDER must be openai, deepseek, anthropic or sglang")
    request_settings = OpenAIRequestSettings(temperature, seed, chat_template_kwargs)
    if provider == "anthropic" and any(
        value is not None for value in (temperature, seed, chat_template_kwargs)
    ):
        raise ValueError("temperature, seed and chat_template_kwargs require an OpenAI-compatible provider")
    prefix = "ANTHROPIC" if provider == "anthropic" else "OPENAI"
    api_key = environment.get(f"{prefix}_API_KEY", "")
    if not api_key.strip():
        raise ValueError(f"{prefix}_API_KEY is required")
    model_id = model_id or (restored_model.id if restored_model else environment.get(f"{prefix}_MODEL"))
    if not model_id:
        raise ValueError(f"Set {prefix}_MODEL or pass --model")
    url = (
        base_url
        or environment.get(f"{prefix}_BASE_URL")
        or {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com",
            "anthropic": "https://api.anthropic.com",
            "sglang": "http://127.0.0.1:30000/v1",
        }[provider]
    )
    if restored_model is not None:
        if (provider, model_id) != (restored_model.provider, restored_model.id):
            raise ValueError(
                "Resumed sessions retain their stored provider/model; start a new session to change it"
            )
        if context_window is not None and context_window != restored_model.context_window:
            raise ValueError("--context-window differs from the resumed session")
        if max_tokens is not None and max_tokens != restored_model.max_tokens:
            raise ValueError("--max-tokens differs from the resumed session")
        model = restored_model
    elif (known := find_model(provider, model_id)) is not None:
        # Pi's catalog entry; explicit CLI/settings values override it.
        model = replace(
            known,
            context_window=context_window if context_window is not None else known.context_window,
            max_tokens=max_tokens if max_tokens is not None else known.max_tokens,
            reasoning=known.reasoning or thinking_level_map is not None,
            thinking_level_map=dict(thinking_level_map)
            if thinking_level_map is not None
            else known.thinking_level_map,
        )
    else:
        model = ModelInfo(
            id=model_id,
            name=model_id,
            provider=provider,
            api="anthropic-messages" if provider == "anthropic" else "openai-completions",
            reasoning=provider in ("deepseek", "anthropic", "sglang") or thinking_level_map is not None,
            context_window=context_window or 0,
            max_tokens=max_tokens or 8192,
            thinking_level_map=dict(thinking_level_map) if thinking_level_map is not None else None,
        )
    seconds = timeout if timeout is not None else float(environment.get("PI_HTTP_TIMEOUT") or "300")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("PI_HTTP_TIMEOUT must be finite and positive")
    thinking_mode = environment.get("ANTHROPIC_THINKING_MODE") or None
    if thinking_mode not in (None, "adaptive", "budget"):
        raise ValueError("ANTHROPIC_THINKING_MODE must be adaptive or budget")
    return RuntimeSettings(
        model,
        url,
        api_key,
        seconds,
        thinking_mode,
        request_settings,
        dict(thinking_budgets) if thinking_budgets is not None else None,
    )
