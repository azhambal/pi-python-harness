"""Host limits and non-secret MCP tool identity, independent of the optional SDK."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import jsonschema

from pi_agent.types import JsonValue


class McpError(RuntimeError):
    """A bounded host error; never embed transport/configuration exception text."""


@dataclass(frozen=True)
class McpLimits:
    max_pages: int = 16
    max_tools: int = 256
    schema_bytes: int = 64 * 1024
    catalog_bytes: int = 1024 * 1024
    result_bytes: int = 50 * 1024
    argument_bytes: int = 64 * 1024
    progress_updates: int = 32

    def __post_init__(self) -> None:
        for value, ceiling in (
            (self.max_pages, 128),
            (self.max_tools, 1024),
            (self.schema_bytes, 256 * 1024),
            (self.catalog_bytes, 4 * 1024 * 1024),
            (self.result_bytes, 1024 * 1024),
            (self.argument_bytes, 1024 * 1024),
            (self.progress_updates, 256),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("Invalid MCP host budget")
        if self.result_bytes < 1024:
            raise ValueError("MCP result budget must be at least 1024 bytes")


def bounded_json(value: object, limit: int) -> str:
    """Check depth/nodes before encoding; reject non-JSON and non-finite values."""
    pending = [(value, 0)]
    nodes = 0
    minimum_bytes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > 32 or nodes > 20_000:
            raise McpError("MCP JSON complexity limit exceeded")
        if isinstance(item, dict):
            if len(item) + len(pending) + nodes > 20_000:
                raise McpError("MCP JSON complexity limit exceeded")
            if any(not isinstance(key, str) for key in item):
                raise McpError("Invalid MCP JSON object")
            minimum_bytes += sum(len(key) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            if len(item) + len(pending) + nodes > 20_000:
                raise McpError("MCP JSON complexity limit exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            minimum_bytes += len(item)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise McpError("Invalid MCP JSON value")
        if minimum_bytes > limit:
            raise McpError("MCP JSON byte limit exceeded")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        size = len(encoded.encode("utf-8"))
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise McpError("Invalid MCP JSON value") from None
    if size > limit:
        raise McpError("MCP JSON byte limit exceeded")
    return encoded


def _schema_nodes(value: object) -> Iterator[dict[str, JsonValue]]:
    """Walk assertion schemas, leaving property names and literal data alone."""
    pending = [value]
    while pending:
        item = pending.pop()
        if not isinstance(item, dict):
            continue
        yield cast(dict[str, JsonValue], item)
        for key in ("properties", "$defs", "definitions", "dependentSchemas"):
            mapping = item.get(key)
            if isinstance(mapping, dict):
                pending.extend(mapping.values())
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            schemas = item.get(key)
            if isinstance(schemas, list):
                pending.extend(schemas)
        for key in (
            "additionalProperties",
            "unevaluatedProperties",
            "items",
            "unevaluatedItems",
            "contains",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
        ):
            child = item.get(key)
            if isinstance(child, dict):
                pending.append(child)


def checked_schema(value: object, limit: int, *, input_schema: bool) -> dict[str, JsonValue]:
    """Accept bounded schemas without reference or regex evaluation."""
    bounded_json(value, limit)
    if not isinstance(value, dict):
        raise McpError("MCP schema must be an object")
    value = deepcopy(value)
    if input_schema:
        value.setdefault("type", "object")
        value.setdefault("properties", {})
        if value["type"] != "object":
            raise McpError("MCP input schema must have object type")
    for item in _schema_nodes(value):
        if {"$ref", "$dynamicRef", "$recursiveRef"}.intersection(item):
            raise McpError("MCP schema references are unsupported in the stdio subset")
        if "pattern" in item or "patternProperties" in item or item.get("uniqueItems") is True:
            raise McpError("MCP regex and uniqueItems schemas are unsupported in the stdio subset")
        if "$schema" in item and item["$schema"] != "https://json-schema.org/draft/2020-12/schema":
            raise McpError("MCP schemas must use JSON Schema draft 2020-12")
    try:
        jsonschema.Draft202012Validator.check_schema(value)
    except (jsonschema.SchemaError, RecursionError):
        raise McpError("Invalid MCP tool schema") from None
    # JSON roundtrip detaches the mutable server/SDK schema.
    return json.loads(bounded_json(value, limit))


_VALIDATION_STEPS: ContextVar[int] = ContextVar("mcp_validation_steps", default=0)


def _validation_guard(keyword, validate):
    def guarded(validator, rule, instance, schema):
        cost = 1
        if keyword in ("enum", "const"):
            # These validators compare whole trees internally without descend().
            pending = [instance]
            while pending:
                item = pending.pop()
                cost += 1
                if isinstance(item, dict):
                    pending.extend(item.values())
                elif isinstance(item, list):
                    pending.extend(item)
                if cost > 4096:
                    raise McpError("MCP schema validation work limit exceeded")
            if keyword == "enum" and isinstance(rule, list):
                cost *= len(rule)
        remaining = _VALIDATION_STEPS.get() - cost
        if remaining < 0:
            raise McpError("MCP schema validation work limit exceeded")
        _VALIDATION_STEPS.set(remaining)
        yield from validate(validator, rule, instance, schema)

    return guarded


_BudgetValidator = jsonschema.validators.extend(
    jsonschema.Draft202012Validator,
    {
        keyword: _validation_guard(keyword, validate)
        for keyword, validate in jsonschema.Draft202012Validator.VALIDATORS.items()
    },
)


def validate_schema_instance(schema: dict[str, JsonValue], value: object) -> None:
    # jsonschema.evolve() selects the registered base validator for an explicit
    # $schema in a subschema. Its dialect was already checked; removing only that
    # annotation from this private copy keeps every descend on our budget class.
    validation_schema = deepcopy(schema)
    for item in _schema_nodes(validation_schema):
        item.pop("$schema", None)
    token = _VALIDATION_STEPS.set(4096)
    try:
        _BudgetValidator(validation_schema).validate(value)
    finally:
        _VALIDATION_STEPS.reset(token)


def binding_name(server: str, tool: str) -> str:
    """Always suffix a readable identifier so sanitization cannot merge bindings."""
    clean_server = re.sub(r"[^a-zA-Z0-9_]", "_", server)[:16] or "server"
    clean_tool = re.sub(r"[^a-zA-Z0-9_]", "_", tool)[:19] or "tool"
    digest = hashlib.sha256(json.dumps([server, tool], ensure_ascii=True).encode()).hexdigest()[:16]
    return f"mcp__{clean_server}__{clean_tool}_{digest}"
