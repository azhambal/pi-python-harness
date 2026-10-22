import json
from importlib.resources import files

import jsonschema
import pytest

from pi_agent.mcp.config import parse_mcp_config
from pi_agent.settings import _FIELDS, load_settings
from pi_agent.tools import TOOL_NAMES


def schema(name):
    return json.loads(files("pi_agent").joinpath(f"schemas/{name}.schema.json").read_text())


def test_settings_schema_fields_match_runtime():
    document = schema("settings")
    jsonschema.Draft202012Validator.check_schema(document)
    assert set(document["properties"]) == _FIELDS
    assert document["properties"]["tools"]["items"]["enum"] == list(TOOL_NAMES)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"max_tokens": 1, "compaction": {"reserve_tokens": 0}},
        {"temperature": None, "seed": -1, "bash_default_timeout_s": None},
        {"thinking_level_map": {"off": "none", "high": "high", "xhigh": None}},
        {"permissions": {"rules": [{"tool": "bash", "action": "ask", "reason": "confirm"}]}},
        {"skills": {"paths": ["skills/demo"], "prompt_max_bytes": 512}},
    ],
)
def test_settings_schema_and_parser_accept_supported_data(tmp_path, data):
    data = {"$schema": "https://example.invalid/schema.json", **data}
    jsonschema.validate(data, schema("settings"))
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    assert load_settings(tmp_path, agent_dir=tmp_path / "absent", explicit=path) == {
        key: value for key, value in data.items() if key != "$schema"
    }


@pytest.mark.parametrize(
    "data",
    [
        {"max_tokens": True},
        {"max_tokens": 0},
        {"http_timeout": 0},
        {"retry": {"enabled": "false"}},
        {"retry": {"max_retries": 1.5}},
        {"compaction": {"unknown": 1}},
        {"tools": ["+read"]},
        {"tools": ["read", "read"]},
        {"skills": {"paths": [""]}},
        {"thinking_level_map": {"invalid": "value"}},
        {"OPENAI_API_KEY": "synthetic"},
    ],
)
def test_settings_schema_and_parser_reject_invalid_data(tmp_path, data):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema("settings"))
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_settings(tmp_path, agent_dir=tmp_path / "absent", explicit=path)


@pytest.mark.parametrize(
    "server",
    [
        {"command": "fixture", "exposure": "direct"},
        {"command": "fixture", "enabled": False},
        {"type": "stdio", "command": "fixture", "exposure": "direct", "env": {"TOKEN": "${TOKEN}"}},
        {
            "type": "http",
            "url": "https://example.invalid/mcp",
            "exposure": "direct",
            "headers": {"Authorization": "${TOKEN}"},
        },
        {"url": "http://127.0.0.1:8000/mcp", "enabled": False},
    ],
)
def test_mcp_schema_and_parser_accept_supported_data(tmp_path, server):
    data = {"$schema": "https://example.invalid/schema.json", "mcpServers": {"fixture": server}}
    jsonschema.Draft202012Validator.check_schema(schema("mcp"))
    jsonschema.validate(data, schema("mcp"))
    assert parse_mcp_config(data, cwd=tmp_path).servers


@pytest.mark.parametrize(
    "server",
    [
        {"command": "fixture"},
        {"command": "fixture", "exposure": "deferred"},
        {"command": "fixture", "exposure": "direct", "url": "https://example.invalid"},
        {"url": "https://example.invalid", "exposure": "direct", "headers": {"Authorization": "literal"}},
        {"command": "fixture", "exposure": "direct", "env": {"TOKEN": "literal"}},
        {"command": "fixture", "exposure": "direct", "timeout": 0},
    ],
)
def test_mcp_schema_and_parser_reject_unsupported_data(tmp_path, server):
    data = {"mcpServers": {"fixture": server}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema("mcp"))
    with pytest.raises(ValueError):
        parse_mcp_config(data, cwd=tmp_path)
