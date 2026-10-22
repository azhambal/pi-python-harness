from __future__ import annotations

from collections.abc import Mapping

import pytest

from pi_agent.mcp import HttpServerConfig, McpConfigError, parse_mcp_config
from pi_agent.mcp.config import resolve_server_headers


def document(**changes):
    return {
        "mcpServers": {
            "remote": {
                "type": "http",
                "url": "https://synthetic.invalid/mcp",
                "exposure": "direct",
                **changes,
            }
        }
    }


@pytest.mark.parametrize("transport", ["http", "streamable-http", None])
def test_supported_endpoint_and_explicit_references(tmp_path, transport):
    data = document(headers={"Authorization": "${MCP_AUTH}"})
    if transport is None:
        del data["mcpServers"]["remote"]["type"]
    else:
        data["mcpServers"]["remote"]["type"] = transport
    server = parse_mcp_config(data, cwd=tmp_path).servers[0]
    assert isinstance(server, HttpServerConfig)
    assert "synthetic" not in repr(server)
    assert "Authorization" not in repr(server)
    with pytest.raises(TypeError):
        server.headers["Other"] = "${OTHER}"
    assert resolve_server_headers(
        server, {"MCP_AUTH": "Bearer synthetic-token", "OPENAI_API_KEY": "synthetic-provider"}
    ) == {"Authorization": "Bearer synthetic-token"}


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:1234/mcp", "https://example.invalid/mcp", "http://[::1]:8080/mcp"]
)
def test_valid_url_shapes(tmp_path, url):
    assert parse_mcp_config(document(url=url), cwd=tmp_path).servers[0].url == url


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "synthetic-secret",
        "ftp://example.invalid",
        "https://",
        "https://user:synthetic-secret@example.invalid/mcp",
        "https://example.invalid/mcp?token=synthetic-secret",
        "https://example.invalid/mcp#synthetic-secret",
        "https://example.invalid:0/mcp",
        "https://example.invalid:99999/mcp",
        "https://example.invalid:bad/mcp",
        "https://example.invalid/mcp\n",
        "https://example.invalid/a b",
    ],
)
def test_invalid_urls_are_redacted(tmp_path, url):
    with pytest.raises(McpConfigError) as error:
        parse_mcp_config(document(url=url), cwd=tmp_path)
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize(
    "headers",
    [
        None,
        [],
        {"Authorization": "Bearer synthetic-secret"},
        {"Authorization": "Bearer ${TOKEN}"},
        {"Authorization": "${TOKEN}suffix"},
        {"Bad\nHeader": "${TOKEN}"},
        {"Authorization": "${TOKEN}", "authorization": "${OTHER}"},
        {"": "${TOKEN}"},
        {"Mcp-Session-Id": "${TOKEN}"},
        {"Mcp-Protocol-Version": "${TOKEN}"},
        {"Accept": "${TOKEN}"},
        {"Accept-Encoding": "${TOKEN}"},
        {"Content-Length": "${TOKEN}"},
        {"Last-Event-ID": "${TOKEN}"},
        {"Cookie": "${TOKEN}"},
        {"Host": "${TOKEN}"},
        {"Transfer-Encoding": "${TOKEN}"},
        {"Connection": "${TOKEN}"},
    ],
)
def test_invalid_or_protocol_headers_rejected_without_echo(tmp_path, headers):
    with pytest.raises(McpConfigError) as error:
        parse_mcp_config(document(headers=headers), cwd=tmp_path)
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("command", "secret"),
        ("args", []),
        ("cwd", "secret"),
        ("env", {}),
        ("oauth", {}),
        ("auth", {}),
        ("type", "sse"),
        ("type", "stdio"),
        ("exposure", "codemode"),
    ],
)
def test_mixed_or_unsupported_options_rejected(tmp_path, field, value):
    with pytest.raises(McpConfigError):
        parse_mcp_config(document(**{field: value}), cwd=tmp_path)


class Exact(Mapping):
    def __init__(self, values):
        self.values = values
        self.lookups = []

    def __getitem__(self, key):
        self.lookups.append(key)
        return self.values[key]

    def __len__(self):
        pytest.fail("Environment must not be scanned")

    def __iter__(self):
        pytest.fail("Environment must not be scanned")


def test_only_declared_headers_are_resolved_disabled_looks_up_nothing(tmp_path):
    env = Exact({"TOKEN": "synthetic-token", "OPENAI_API_KEY": "synthetic-provider"})
    server = parse_mcp_config(document(headers={"Authorization": "${TOKEN}"}), cwd=tmp_path).servers[0]
    assert resolve_server_headers(server, env) == {"Authorization": "synthetic-token"}
    assert env.lookups == ["TOKEN"]
    server = parse_mcp_config(
        document(enabled=False, headers={"Authorization": "${MISSING}"}), cwd=tmp_path
    ).servers[0]
    assert resolve_server_headers(server, env) == {}
    assert env.lookups == ["TOKEN"]


@pytest.mark.parametrize(
    "value", [None, "", "a\r\nInjected: secret", "\x00", "\u0442\u043e\u043a\u0435\u043d", "x" * 8193]
)
def test_missing_invalid_header_values_are_constant(tmp_path, value):
    server = parse_mcp_config(document(headers={"Authorization": "${TOKEN}"}), cwd=tmp_path).servers[0]
    with pytest.raises(McpConfigError):
        resolve_server_headers(server, {} if value is None else {"TOKEN": value})


def test_http_project_trust_precedes_endpoint_parsing(tmp_path):
    with pytest.raises(McpConfigError, match="project trust"):
        parse_mcp_config(document(url="bad-secret"), cwd=tmp_path, source="project")
