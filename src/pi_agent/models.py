"""Pi's model catalog for the providers this port speaks to.

``data/models/<provider>.json`` are copies of pi's generated
``packages/ai/src/providers/data`` files (refreshed by
``scripts/pi_reference/generate.sh``). Only chat models whose API this port
implements are exposed.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files

from .types import JsonValue, ModelInfo

#: Model APIs this port has request/response adapters for.
SUPPORTED_APIS = ("anthropic-messages", "openai-completions")


@cache
def _catalog(provider: str) -> dict[str, dict[str, JsonValue]]:
    resource = files("pi_agent").joinpath(f"data/models/{provider}.json")
    if not resource.is_file():
        return {}
    groups = json.loads(resource.read_text("utf-8"))
    models: dict[str, dict[str, JsonValue]] = {}
    for api, entries in groups.items():
        if api not in SUPPORTED_APIS:
            continue
        for entry in entries.values():
            if entry.get("type", "chat") == "chat":
                models[entry["id"]] = entry
    return models


def _int(value: JsonValue) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def catalog_model_ids(provider: str) -> list[str]:
    return sorted(_catalog(provider))


def find_model(provider: str, model_id: str) -> ModelInfo | None:
    """Pi's catalog entry for ``provider``/``model_id`` as a ModelInfo, or None."""
    entry = _catalog(provider).get(model_id)
    return model_from_pi(entry) if entry is not None else None


def model_from_pi(entry: dict[str, JsonValue]) -> ModelInfo:
    """A ModelInfo from a model object in Pi's JSON format."""
    thinking_level_map = entry.get("thinkingLevelMap")
    compat = entry.get("compat")
    cost = entry.get("cost")
    input_limits = entry.get("inputLimits")
    model_input = entry.get("input") or ["text"]
    return ModelInfo(
        id=str(entry["id"]),
        name=str(entry.get("name") or entry["id"]),
        provider=str(entry["provider"]),
        api=str(entry["api"]),
        reasoning=bool(entry.get("reasoning", False)),
        context_window=_int(entry.get("contextWindow")),
        max_tokens=_int(entry.get("maxTokens")),
        thinking_level_map=dict(thinking_level_map) if isinstance(thinking_level_map, dict) else None,  # type: ignore[arg-type]
        input=tuple(str(item) for item in model_input),  # type: ignore[union-attr]
        compat=json.loads(json.dumps(compat)) if isinstance(compat, dict) else None,
        input_limits=json.loads(json.dumps(input_limits)) if isinstance(input_limits, dict) else None,
        cost=json.loads(json.dumps(cost)) if isinstance(cost, dict) else None,
    )
