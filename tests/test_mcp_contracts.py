from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from pi_agent.mcp._contracts import (
    McpError,
    binding_name,
    bounded_json,
    checked_schema,
    validate_schema_instance,
)
from pi_agent.types import TextBlock, ToolResultMessage


@pytest.mark.parametrize("value", [float("nan"), {1: "bad"}, {"key": object()}, "\ud800"])
def test_invalid_json_is_sanitized(value):
    with pytest.raises(McpError, match="Invalid MCP JSON"):
        bounded_json(value, 1024)


def test_json_byte_depth_and_node_limits():
    with pytest.raises(McpError, match="byte limit"):
        bounded_json("x" * 100_000, 1024)
    with pytest.raises(McpError, match="complexity"):
        bounded_json([None] * 20_001, 100_000)
    value = None
    for _ in range(34):
        value = [value]
    with pytest.raises(McpError, match="complexity"):
        bounded_json(value, 100_000)


def test_schema_reference_words_in_properties_and_data_are_literal():
    schema = {
        "type": "object",
        "properties": {"$ref": {"type": "string"}, "pattern": {"type": "string"}},
        "default": {"$ref": "literal", "pattern": "literal"},
        "enum": [{"$ref": "literal"}],
    }
    checked = checked_schema(schema, 64 * 1024, input_schema=True)
    assert checked == schema and checked is not schema
    validate_schema_instance(checked, {"$ref": "literal"})


@pytest.mark.parametrize(
    "field",
    [
        {"pattern": "^(a+)+$"},
        {"patternProperties": {".*": {"type": "string"}}},
        {"uniqueItems": True},
        {"$dynamicRef": "#x"},
        {"$schema": "https://invalid.test/schema"},
    ],
)
def test_unsafe_or_unsupported_schema_features_rejected(field):
    with pytest.raises(McpError):
        checked_schema({"type": "object", "properties": {"value": field}}, 64 * 1024, input_schema=True)


def test_schema_validation_visit_budget():
    branch = {"type": "integer"}
    for _ in range(11):
        branch = {"anyOf": [deepcopy(branch), deepcopy(branch)]}
    schema = {"type": "object", "properties": {"value": branch}}
    checked = checked_schema(schema, 256 * 1024, input_schema=True)
    with pytest.raises(McpError, match="work limit"):
        validate_schema_instance(checked, {"value": "invalid"})
    # The ContextVar budget resets after both failure and successful validation.
    validate_schema_instance({"type": "object", "properties": {"x": {"type": "integer"}}}, {"x": 1})


def test_nullable_combinators_supported():
    schema = checked_schema(
        {
            "type": "object",
            "properties": {
                "x": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "null"},
                    ]
                }
            },
        },
        64 * 1024,
        input_schema=True,
    )
    validate_schema_instance(schema, {"x": None})
    validate_schema_instance(schema, {"x": 1})


def test_explicit_subschema_dialect_cannot_escape_validation_budget():
    schema = checked_schema(
        {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "string", "$schema": "https://json-schema.org/draft/2020-12/schema"},
                }
            },
        },
        64 * 1024,
        input_schema=True,
    )
    with pytest.raises(McpError, match="work limit"):
        validate_schema_instance(schema, {"values": ["valid"] * 5000})
    assert "$schema" in schema["properties"]["values"]["items"]


def test_binding_names_are_stable_order_independent_and_bounded():
    pairs = [("server", "a-b"), ("server", "a_b"), ("server", "猫" * 100), ("server-2", "a-b")]
    forward = {pair: binding_name(*pair) for pair in pairs}
    reverse = {pair: binding_name(*pair) for pair in reversed(pairs)}
    assert forward == reverse and len(set(forward.values())) == len(pairs)
    assert all(len(value) <= 64 for value in forward.values())


def test_structured_tool_message_roundtrip_and_legacy_compatibility():
    legacy = ToolResultMessage(
        tool_call_id="call", tool_name="tool", content=[TextBlock(text="old")]
    ).to_json()
    assert "structured_content" not in legacy
    assert ToolResultMessage.from_json(legacy).structured_content is None
    message = ToolResultMessage(
        tool_call_id="call",
        tool_name="tool",
        content=[TextBlock(text="answer")],
        structured_content={"answer": 42},
    )
    assert ToolResultMessage.from_json(json.loads(json.dumps(message.to_json()))) == message


def test_core_and_empty_runtime_work_when_optional_sdk_import_is_blocked(tmp_path):
    script = """
import asyncio, builtins, sys
from pathlib import Path
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    level = args[3] if len(args) > 3 else kwargs.get('level', 0)
    if level == 0 and name.split('.')[0] == 'mcp':
        raise ImportError('synthetic missing optional dependency')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from pi_agent import AgentConfig, ModelInfo, create_harness
from pi_agent.mcp import McpRuntime, McpError, parse_mcp_config
from pi_agent.providers import FauxProvider
directory = Path(sys.argv[1])
config = AgentConfig(cwd=directory, agent_dir=directory, instruction_files=[],
                     model=ModelInfo(id='offline', name='Offline', provider='faux'))
async def main():
    runtime = McpRuntime(parse_mcp_config({}, cwd=directory))
    session = await create_harness(FauxProvider([]), config=config, prepare_tools=runtime.prepare_tools)
    await session.aclose()
    unavailable = McpRuntime(parse_mcp_config({'mcpServers': {'missing': {
        'command': '/must/not/start', 'exposure': 'direct'}}}, cwd=directory))
    try:
        await create_harness(FauxProvider([]), config=config, prepare_tools=unavailable.prepare_tools)
    except McpError as error:
        assert str(error) == 'Install the optional MCP extra: pi-agent[mcp]'
    else:
        raise AssertionError('missing extra was not reported')
asyncio.run(main())
assert 'mcp' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], env={}, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and result.stderr == ""
