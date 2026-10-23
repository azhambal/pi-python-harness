"""Tool-argument validation translated from Pi.

Mirrors ``packages/ai/src/utils/validation.ts`` together with the TypeBox
behaviour it relies on (``Value.Convert`` and the ``Errors`` engine with the
en_US messages and the default capacity of eight errors).

Two conversion modes exist, as in Pi:

* ``"typebox"`` — schemas Pi declares with TypeBox (the built-in tools).
  Optional ``null`` values are dropped and ``Value.Convert`` coerces
  primitives ("2" -> 2, 5 -> "5", "true" -> True, x -> [x] for arrays).
* ``"json_schema"`` — plain JSON schemas (MCP and other custom tools).
  Optional ``null`` values are dropped and ``coerceWithJsonSchema`` applies.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Literal

from pi_agent._js import JS_WHITESPACE as _JS_WHITESPACE
from pi_agent._js import is_js_number, json_stringify
from pi_agent._js import number_to_string as js_number_to_string
from pi_agent._js import to_number as js_to_number
from pi_agent.types import JsonValue

ConversionMode = Literal["typebox", "json_schema"]

#: TypeBox ``Settings.maxErrors``.
MAX_ERRORS = 8


class ArgumentValidationError(ValueError):
    """Raised with Pi's exact message when tool arguments are invalid."""


def _is_finite_number(value: object) -> bool:
    return is_js_number(value) and math.isfinite(value)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# TypeBox Value.Convert (try_*.mjs / from_*.mjs)
# ----------------------------------------------------------------------

_FAIL = object()


def _try_number(value: object) -> object:
    if isinstance(value, bool):
        return 1 if value else 0
    if is_js_number(value):
        return value
    if value is None:
        return 0
    if isinstance(value, str):
        coerced = js_to_number(value)
        if coerced is not None and math.isfinite(coerced):
            return coerced
        lowered = value.lower()
        if lowered == "false":
            return 0
        if lowered == "true":
            return 1
        try:
            big = int(value.strip(_JS_WHITESPACE), 0) if value.strip(_JS_WHITESPACE) else 0
        except ValueError:
            return _FAIL
        return big if abs(big) <= 2**53 - 1 else _FAIL
    return _FAIL


def _try_string(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if is_js_number(value):
        return js_number_to_string(value)  # type: ignore[arg-type]
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return _FAIL


def _try_boolean(value: object) -> object:
    if isinstance(value, bool):
        return value
    if is_js_number(value):
        if value == 0:
            return False
        if value == 1:
            return True
        return _FAIL
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "false" or value == "0":
            return False
        if lowered == "true" or value == "1":
            return True
    return _FAIL


def _schema_type(schema: Mapping[str, JsonValue]) -> str | None:
    kind = schema.get("type")
    return kind if isinstance(kind, str) else None


def typebox_convert(schema: Mapping[str, JsonValue], value: JsonValue) -> JsonValue:
    """``Value.Convert`` for the TypeBox kinds Pi's tool schemas use."""
    kind = _schema_type(schema)
    if kind == "array":
        items: list[JsonValue] = value if isinstance(value, list) else [value]
        item_schema = schema.get("items")
        if not isinstance(item_schema, Mapping):
            return items
        return [typebox_convert(item_schema, item) for item in items]
    if kind == "boolean":
        result = _try_boolean(value)
        return value if result is _FAIL else result  # type: ignore[return-value]
    if kind == "integer":
        result = _try_number(value)
        return value if result is _FAIL else math.trunc(result)  # type: ignore[arg-type]
    if kind == "number":
        result = _try_number(value)
        return value if result is _FAIL else result  # type: ignore[return-value]
    if kind == "string":
        result = _try_string(value)
        return value if result is _FAIL else result  # type: ignore[return-value]
    if kind == "object":
        if not isinstance(value, dict):
            return value
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for key in list(value):
                property_schema = properties.get(key)
                if isinstance(property_schema, Mapping):
                    value[key] = typebox_convert(property_schema, value[key])
        additional = schema.get("additionalProperties")
        if isinstance(additional, Mapping):
            known = set(properties) if isinstance(properties, Mapping) else set()
            for key in list(value):
                if key not in known:
                    value[key] = typebox_convert(additional, value[key])
        return value
    return value


# ----------------------------------------------------------------------
# coerceWithJsonSchema (plain JSON schemas)
# ----------------------------------------------------------------------


def _schema_types(schema: Mapping[str, JsonValue]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    if isinstance(kind, list):
        return [item for item in kind if isinstance(item, str)]
    return []


def _matches_json_type(value: object, kind: str) -> bool:
    if kind == "number":
        return is_js_number(value)
    if kind == "integer":
        return is_js_number(value) and float(value).is_integer()  # type: ignore[arg-type]
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "null":
        return value is None
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return False


def _coerce_primitive(value: JsonValue, kind: str) -> JsonValue:
    if kind == "number":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip(_JS_WHITESPACE) != "":
            parsed = js_to_number(value)
            if parsed is not None and math.isfinite(parsed):
                return parsed
        if isinstance(value, bool):
            return 1 if value else 0
        return value
    if kind == "integer":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip(_JS_WHITESPACE) != "":
            parsed = js_to_number(value)
            if parsed is not None and math.isfinite(parsed) and float(parsed).is_integer():
                return int(parsed)
        if isinstance(value, bool):
            return 1 if value else 0
        return value
    if kind == "boolean":
        if value is None:
            return False
        if value == "true":
            return True
        if value == "false":
            return False
        if is_js_number(value):
            if value == 1:
                return True
            if value == 0:
                return False
        return value
    if kind == "string":
        if value is None:
            return ""
        if isinstance(value, bool) or is_js_number(value):
            return _try_string(value)  # type: ignore[return-value]
        return value
    if kind == "null":
        if value == "" or (is_js_number(value) and value == 0) or value is False:
            return None
        return value
    return value


def _differs(a: JsonValue, b: JsonValue) -> bool:
    """JavaScript ``!==`` for the primitive candidates produced above."""
    return type(a) is not type(b) or a != b


def _coerce_union(value: JsonValue, schemas: list[JsonValue]) -> JsonValue:
    for schema in schemas:
        if isinstance(schema, Mapping) and not collect_errors(schema, value):
            return value
    for schema in schemas:
        if not isinstance(schema, Mapping):
            continue
        coerced = coerce_with_json_schema(copy.deepcopy(value), schema)
        if not collect_errors(schema, coerced):
            return coerced
    return value


def coerce_with_json_schema(value: JsonValue, schema: Mapping[str, JsonValue]) -> JsonValue:
    next_value = value
    for key in ("allOf",):
        nested = schema.get(key)
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, Mapping):
                    next_value = coerce_with_json_schema(next_value, item)
    for key in ("anyOf", "oneOf"):
        nested = schema.get(key)
        if isinstance(nested, list):
            next_value = _coerce_union(next_value, nested)

    kinds = _schema_types(schema)
    matches_union_member = len(kinds) > 1 and any(_matches_json_type(next_value, kind) for kind in kinds)
    if kinds and not matches_union_member:
        for kind in kinds:
            candidate = _coerce_primitive(next_value, kind)
            if _differs(candidate, next_value):
                next_value = candidate
                break

    if "object" in kinds and isinstance(next_value, dict):
        properties = schema.get("properties")
        defined = set(properties) if isinstance(properties, Mapping) else set()
        if isinstance(properties, Mapping):
            for key, property_schema in properties.items():
                if key in next_value and isinstance(property_schema, Mapping):
                    next_value[key] = coerce_with_json_schema(next_value[key], property_schema)
        additional = schema.get("additionalProperties")
        if isinstance(additional, Mapping):
            for key in list(next_value):
                if key not in defined:
                    next_value[key] = coerce_with_json_schema(next_value[key], additional)

    if "array" in kinds and isinstance(next_value, list):
        items = schema.get("items")
        if isinstance(items, list):
            for index, item_schema in enumerate(items[: len(next_value)]):
                if isinstance(item_schema, Mapping):
                    next_value[index] = coerce_with_json_schema(next_value[index], item_schema)
        elif isinstance(items, Mapping):
            for index in range(len(next_value)):
                next_value[index] = coerce_with_json_schema(next_value[index], items)

    return next_value


def normalize_optional_nulls(value: JsonValue, schema: Mapping[str, JsonValue]) -> None:
    """Drop ``null`` from optional properties whose schema rejects null."""
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, list):
            for index, item_schema in enumerate(items[: len(value)]):
                if isinstance(item_schema, Mapping):
                    normalize_optional_nulls(value[index], item_schema)
        elif isinstance(items, Mapping):
            for item in value:
                normalize_optional_nulls(item, items)
        return
    properties = schema.get("properties")
    if not isinstance(value, dict) or not isinstance(properties, Mapping):
        return
    required = schema.get("required")
    required_set = set(required) if isinstance(required, list) else set()
    for key, property_schema in properties.items():
        if key not in value or not isinstance(property_schema, Mapping):
            continue
        if (
            value[key] is None
            and key not in required_set
            and not isinstance(property_schema.get("$ref"), str)
            and collect_errors(property_schema, None)
        ):
            del value[key]
        else:
            normalize_optional_nulls(value[key], property_schema)


# ----------------------------------------------------------------------
# TypeBox Errors engine (schema/engine/*.mjs) with en_US messages
# ----------------------------------------------------------------------


class _Error:
    __slots__ = ("instance_path", "keyword", "params")

    def __init__(self, keyword: str, instance_path: str, params: dict[str, object]) -> None:
        self.keyword, self.instance_path, self.params = keyword, instance_path, params


def _format_limit(limit: object) -> str:
    return js_number_to_string(limit) if is_js_number(limit) else str(limit)  # type: ignore[arg-type]


def _message(error: _Error) -> str:
    params, keyword = error.params, error.keyword
    if keyword == "additionalProperties":
        return "must not have additional properties"
    if keyword == "anyOf":
        return "must match a schema in anyOf"
    if keyword == "boolean":
        return "schema is false"
    if keyword == "const":
        return "must be equal to constant"
    if keyword == "enum":
        return "must be equal to one of the allowed values"
    if keyword in ("exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum"):
        return f"must be {params['comparison']} {_format_limit(params['limit'])}"
    if keyword == "format":
        return f'must match format "{params["format"]}"'
    if keyword == "maxItems":
        return f"must not have more than {_format_limit(params['limit'])} items"
    if keyword == "maxLength":
        return f"must not have more than {_format_limit(params['limit'])} characters"
    if keyword == "maxProperties":
        return f"must not have more than {_format_limit(params['limit'])} properties"
    if keyword == "minItems":
        return f"must not have fewer than {_format_limit(params['limit'])} items"
    if keyword == "minLength":
        return f"must not have fewer than {_format_limit(params['limit'])} characters"
    if keyword == "minProperties":
        return f"must not have fewer than {_format_limit(params['limit'])} properties"
    if keyword == "multipleOf":
        return f"must be multiple of {_format_limit(params['multipleOf'])}"
    if keyword == "not":
        return "must not be valid"
    if keyword == "oneOf":
        return "must match exactly one schema in oneOf"
    if keyword == "pattern":
        return f'must match pattern "{params["pattern"]}"'
    if keyword == "required":
        return f"must have required properties {', '.join(params['requiredProperties'])}"  # type: ignore[arg-type]
    if keyword == "type":
        kind = params["type"]
        return f"must be {kind}" if isinstance(kind, str) else f"must be either {' or '.join(kind)}"  # type: ignore[arg-type]
    if keyword == "uniqueItems":
        return "must not have duplicate items"
    return "an unknown validation error occurred"


def _type_matches(kind: str, value: object) -> bool:
    if kind == "number":
        return _is_finite_number(value)
    if kind == "integer":
        return _is_finite_number(value) and float(value).is_integer()  # type: ignore[arg-type]
    return _matches_json_type(value, kind)


def _js_length(text: str) -> int:
    """String length in UTF-16 code units, as JavaScript measures it."""
    return len(text.encode("utf-16-le")) // 2


def _deep_equal(a: object, b: object) -> bool:
    if is_js_number(a) and is_js_number(b):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    return a == b


class _Context:
    def __init__(self) -> None:
        self.errors: list[_Error] = []

    def at_capacity(self) -> bool:
        return len(self.errors) >= MAX_ERRORS

    def add(self, error: _Error) -> bool:
        if not self.at_capacity():
            self.errors.append(error)
        return False


def _check(schema: object, value: object) -> bool:
    return not _errors(schema, value, "", _Context())


def _errors(schema: object, value: object, path: str, context: _Context) -> list[_Error]:
    before = len(context.errors)
    _error_schema(schema, value, path, context)
    return context.errors[before:]


def _error_schema(schema: object, value: object, path: str, context: _Context) -> bool:
    if context.at_capacity():
        return False
    if isinstance(schema, bool):
        return schema or context.add(_Error("boolean", path, {}))
    if not isinstance(schema, Mapping):
        return True
    ok = True

    kind = schema.get("type")
    if isinstance(kind, str):
        if not _type_matches(kind, value):
            ok = context.add(_Error("type", path, {"type": kind}))
    elif isinstance(kind, list):
        if not any(isinstance(item, str) and _type_matches(item, value) for item in kind):
            ok = context.add(_Error("type", path, {"type": kind}))

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            missing = [key for key in required if isinstance(key, str) and key not in value]
            if missing:
                ok = context.add(_Error("required", path, {"requiredProperties": missing}))
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        pattern_properties = schema.get("patternProperties")
        pattern_properties = pattern_properties if isinstance(pattern_properties, Mapping) else {}
        additional = schema.get("additionalProperties")
        if additional is not None and not (isinstance(additional, bool) and additional):
            extra = [
                key
                for key in value
                if key not in properties and not any(re.search(p, key) for p in pattern_properties)
            ]
            failed = [
                key for key in extra if not _error_schema(additional, value[key], f"{path}/{key}", context)
            ]
            if failed:
                ok = context.add(_Error("additionalProperties", path, {"additionalProperties": failed}))
        for pattern, property_schema in pattern_properties.items():
            for key in value:
                if re.search(pattern, key) and not _error_schema(
                    property_schema, value[key], f"{path}/{key}", context
                ):
                    ok = False
        for key, property_schema in properties.items():
            if key in value and not _error_schema(property_schema, value[key], f"{path}/{key}", context):
                ok = False
        min_properties = schema.get("minProperties")
        if is_js_number(min_properties) and len(value) < min_properties:  # type: ignore[operator]
            ok = context.add(_Error("minProperties", path, {"limit": min_properties}))
        max_properties = schema.get("maxProperties")
        if is_js_number(max_properties) and len(value) > max_properties:  # type: ignore[operator]
            ok = context.add(_Error("maxProperties", path, {"limit": max_properties}))

    if isinstance(value, list):
        prefix = schema.get("prefixItems")
        offset = len(prefix) if isinstance(prefix, list) else 0
        items = schema.get("items")
        if isinstance(items, list):
            for index, item_schema in enumerate(items):
                if index < len(value) and not _error_schema(
                    item_schema, value[index], f"{path}/{index}", context
                ):
                    ok = False
        elif isinstance(items, (Mapping, bool)):
            for index in range(offset, len(value)):
                if not _error_schema(items, value[index], f"{path}/{index}", context):
                    ok = False
        max_items = schema.get("maxItems")
        if is_js_number(max_items) and len(value) > max_items:  # type: ignore[operator]
            ok = context.add(_Error("maxItems", path, {"limit": max_items}))
        min_items = schema.get("minItems")
        if is_js_number(min_items) and len(value) < min_items:  # type: ignore[operator]
            ok = context.add(_Error("minItems", path, {"limit": min_items}))
        if isinstance(prefix, list):
            for index, item_schema in enumerate(prefix):
                if index < len(value) and not _error_schema(
                    item_schema, value[index], f"{path}/{index}", context
                ):
                    ok = False
        if schema.get("uniqueItems") is True:
            seen: list[object] = []
            for item in value:
                if any(_deep_equal(item, other) for other in seen):
                    ok = context.add(_Error("uniqueItems", path, {}))
                    break
                seen.append(item)

    if isinstance(value, str):
        length = _js_length(value)
        max_length = schema.get("maxLength")
        if is_js_number(max_length) and length > max_length:  # type: ignore[operator]
            ok = context.add(_Error("maxLength", path, {"limit": max_length}))
        min_length = schema.get("minLength")
        if is_js_number(min_length) and length < min_length:  # type: ignore[operator]
            ok = context.add(_Error("minLength", path, {"limit": min_length}))
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                matched = re.search(pattern, value) is not None
            except re.error:
                matched = True
            if not matched:
                ok = context.add(_Error("pattern", path, {"pattern": pattern}))

    if is_js_number(value):
        number = value
        for keyword, comparison, passes in (
            ("exclusiveMaximum", "<", lambda limit: number < limit),
            ("exclusiveMinimum", ">", lambda limit: number > limit),
            ("maximum", "<=", lambda limit: number <= limit),
            ("minimum", ">=", lambda limit: number >= limit),
        ):
            limit = schema.get(keyword)
            if is_js_number(limit) and not passes(limit):
                ok = context.add(_Error(keyword, path, {"comparison": comparison, "limit": limit}))
        multiple = schema.get("multipleOf")
        if is_js_number(multiple) and multiple and not float(number / multiple).is_integer():  # type: ignore[operator]
            ok = context.add(_Error("multipleOf", path, {"multipleOf": multiple}))

    if "const" in schema and not _deep_equal(value, schema["const"]):
        ok = context.add(_Error("const", path, {"allowedValue": schema["const"]}))
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(_deep_equal(value, item) for item in enum):
        ok = context.add(_Error("enum", path, {"allowedValues": enum}))
    negated = schema.get("not")
    if negated is not None and _check(negated, value):
        ok = context.add(_Error("not", path, {}))
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for item in all_of:
            if not _error_schema(item, value, path, context):
                ok = False
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        failures: list[_Context] = []
        passed = False
        for item in any_of:
            nested = _Context()
            if _error_schema(item, value, path, nested):
                passed = True
            else:
                failures.append(nested)
        if not passed:
            for failure in failures:
                for error in failure.errors:
                    context.add(error)
            ok = context.add(_Error("anyOf", path, {}))
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        failures = []
        passing = 0
        for item in one_of:
            nested = _Context()
            if _error_schema(item, value, path, nested):
                passing += 1
            else:
                failures.append(nested)
        if passing != 1:
            if passing == 0:
                for failure in failures:
                    for error in failure.errors:
                        context.add(error)
            ok = context.add(_Error("oneOf", path, {}))
    return ok


def collect_errors(schema: Mapping[str, JsonValue], value: JsonValue) -> list[str]:
    """``validator.Errors(value)`` formatted as Pi's ``  - path: message`` rows."""
    rows = []
    for error in _errors(schema, value, "", _Context()):
        path = error.instance_path.lstrip("/").replace("/", ".")
        if error.keyword == "required":
            missing = error.params["requiredProperties"]
            first = missing[0] if isinstance(missing, list) and missing else None
            if first:
                path = f"{path}.{first}" if path else first
        rows.append(f"  - {path or 'root'}: {_message(error)}")
    return rows


def validate_arguments(
    tool_name: str,
    schema: Mapping[str, JsonValue],
    arguments: JsonValue,
    *,
    mode: ConversionMode = "json_schema",
) -> JsonValue:
    """Pi's ``validateToolArguments``: normalize, coerce, check, or raise."""
    args = copy.deepcopy(arguments)
    normalize_optional_nulls(args, schema)
    if mode == "typebox":
        args = typebox_convert(schema, args)
    else:
        coerced = coerce_with_json_schema(args, schema)
        if coerced is not args and not (isinstance(args, dict) and isinstance(coerced, dict)):
            return coerced if not collect_errors(schema, coerced) else args
        args = coerced
    errors = collect_errors(schema, args)
    if not errors:
        return args
    received = json_stringify(arguments, 2)
    raise ArgumentValidationError(
        f'Validation failed for tool "{tool_name}":\n'
        + ("\n".join(errors) or "Unknown validation error")
        + f"\n\nReceived arguments:\n{received}"
    )
