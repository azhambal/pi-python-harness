import builtins
import importlib.util
import json
import os
import stat
import sys
import traceback
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pi_agent.mcp import config as config_module
from pi_agent.mcp.config import (
    HOST_ENVIRONMENT_KEYS,
    MAX_CONFIG_BYTES,
    McpConfig,
    McpConfigError,
    StdioServerConfig,
    load_mcp_config,
    parse_mcp_config,
    resolve_server_environment,
)


@pytest.fixture(autouse=True)
def synthetic_environment(monkeypatch):
    # No test in this module resolves credentials from the real host environment.
    monkeypatch.setattr(config_module.os, "environ", {"OPENAI_API_KEY": "synthetic-provider-secret"})


def document(**changes):
    return {"mcpServers": {"example": {"command": "synthetic-command", "exposure": "direct", **changes}}}


def server(tmp_path, **changes):
    return parse_mcp_config(document(**changes), cwd=tmp_path).servers[0]


def test_public_contract_defaults_and_order(tmp_path):
    data = document(type="stdio")
    data["mcpServers"]["second"] = {"command": "another", "exposure": "direct"}
    config = parse_mcp_config(data, cwd=tmp_path)
    assert isinstance(config, McpConfig)
    assert type(config.servers) is tuple
    one, two = config.servers
    assert isinstance(one, StdioServerConfig)
    assert (one.name, two.name) == ("example", "second")
    assert one.command == "synthetic-command"
    assert one.args == () and one.env == {}
    assert one.cwd == tmp_path.resolve()
    assert one.enabled is True and one.exposure == "direct"
    assert one.timeout == 60.0 and type(one.timeout) is float
    assert parse_mcp_config({}, cwd=tmp_path).servers == ()


def test_relative_and_absolute_cwd_use_host_not_file_directory(tmp_path):
    host = tmp_path / "host"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document(args=["", "--flag", "space here"], cwd="child/../work", timeout=2)))
    one = load_mcp_config(path, cwd=host).servers[0]
    assert one.cwd == host / "work" and one.args == ("", "--flag", "space here")
    assert one.timeout == 2.0
    assert server(tmp_path, cwd=str(host)).cwd == host


def test_load_default_cwd_is_process_cwd(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document()))
    monkeypatch.chdir(tmp_path)
    assert load_mcp_config(path).servers[0].cwd == tmp_path


@pytest.mark.parametrize("source", ["explicit", "user", "project"])
def test_trusted_sources(tmp_path, source):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document()))
    assert load_mcp_config(path, source=source, project_trusted=True).servers
    assert parse_mcp_config(document(), cwd=tmp_path, source=source, project_trusted=True).servers


def test_untrusted_project_is_rejected_before_any_path_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Untrusted project touched the filesystem")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "cwd", forbidden)
    monkeypatch.setattr(config_module.os, "open", forbidden)
    with pytest.raises(McpConfigError, match="project trust"):
        load_mcp_config(tmp_path / "missing.json", source="project")
    with pytest.raises(McpConfigError, match="project trust"):
        parse_mcp_config(object(), cwd=tmp_path, source="project")


@pytest.mark.parametrize("source,trust", [("unknown-secret", False), ("explicit", 1), (None, False)])
def test_invalid_source_or_trust_is_rejected(tmp_path, source, trust):
    with pytest.raises(McpConfigError, match="source or trust"):
        parse_mcp_config(document(), cwd=tmp_path, source=source, project_trusted=trust)
    with pytest.raises(McpConfigError, match="source or trust"):
        load_mcp_config(tmp_path / "absent", source=source, project_trusted=trust)


def test_enabled_requires_explicit_direct_for_every_source(tmp_path):
    for source in ("explicit", "user", "project"):
        for enabled in (True, None):
            entry = {"command": "synthetic-command"}
            if enabled is not None:
                entry["enabled"] = enabled
            with pytest.raises(McpConfigError, match='explicit exposure "direct"'):
                parse_mcp_config(
                    {"mcpServers": {"example": entry}}, cwd=tmp_path, source=source, project_trusted=True
                )


def test_disabled_entries_are_retained_without_resolving_references(tmp_path):
    data = document(enabled=False, env={"TOKEN": "${MISSING}"})
    del data["mcpServers"]["example"]["exposure"]
    one = parse_mcp_config(data, cwd=tmp_path).servers[0]
    assert not one.enabled and one.exposure == "direct"
    assert one.env == {"TOKEN": "${MISSING}"}
    assert resolve_server_environment(one, ExactLookups({})) == {}


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "exposure", ["codemode", "codemode-deferred", "deferred", "hidden", "secret", None, []]
)
def test_unsupported_exposure_never_activates(tmp_path, enabled, exposure):
    with pytest.raises(McpConfigError, match="MCP exposure is unsupported"):
        server(tmp_path, enabled=enabled, exposure=exposure)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "synthetic-url-secret"),
        ("headers", {"Authorization": "synthetic-secret"}),
        ("oauth", {}),
        ("auth", {"provider": "synthetic-secret"}),
        ("type", "http"),
        ("type", "streamable-http"),
        ("type", "sse"),
    ],
)
def test_mixed_http_stdio_and_oauth_fields_have_constant_diagnostic(tmp_path, enabled, field, value):
    with pytest.raises(McpConfigError, match="Unsupported MCP HTTP configuration") as exc:
        server(tmp_path, enabled=enabled, **{field: value})
    assert "synthetic" not in str(exc.value)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "field",
    [
        "toolExposure",
        "description",
        "permissions",
        "readOnly",
        "parallel_safe",
        "autoEnableCodemode",
        "synthetic-secret-field",
    ],
)
def test_unknown_fields_cannot_enable_permission_or_exposure_models(tmp_path, enabled, field):
    with pytest.raises(McpConfigError, match="Unsupported MCP server configuration fields"):
        server(tmp_path, enabled=enabled, **{field: "synthetic-secret"})


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        "synthetic-secret",
        {"mcpServers": []},
        {"mcpServers": None},
        {"autoEnableCodemode": True},
        {"mcpServers": {"example": []}},
        {"mcpServers": {"example": {"enabled": False}}},
    ],
)
def test_invalid_top_level_and_partial_overrides_fail(tmp_path, data):
    with pytest.raises(McpConfigError):
        parse_mcp_config(data, cwd=tmp_path)


@pytest.mark.parametrize("name", ["", "space name", "../server", "x" * 129, "name\x00", "a\nb", 1])
def test_invalid_server_names(tmp_path, name):
    data = {"mcpServers": {name: {"command": "x", "exposure": "direct"}}}
    with pytest.raises(McpConfigError, match="server name"):
        parse_mcp_config(data, cwd=tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("command", ""),
        ("command", " \t"),
        ("command", 1),
        ("command", "secret\x00"),
        ("command", "\ud800"),
        ("args", "--secret"),
        ("args", ("one",)),
        ("args", [True]),
        ("args", [None]),
        ("args", ["secret\x00"]),
        ("cwd", None),
        ("cwd", ""),
        ("cwd", "\ud800"),
        ("cwd", "secret\x00"),
        ("enabled", 1),
        ("enabled", "false"),
        ("timeout", True),
        ("timeout", "60"),
        ("timeout", 0),
        ("timeout", -1),
        ("timeout", float("nan")),
        ("timeout", float("inf")),
        ("timeout", 10**400),
        ("env", []),
        ("env", None),
        ("type", None),
        ("type", "synthetic-secret"),
    ],
)
def test_strict_field_types_and_safe_values(tmp_path, field, value):
    with pytest.raises(McpConfigError):
        server(tmp_path, **{field: value})


@pytest.mark.parametrize(
    "reference",
    [
        "literal-secret",
        "$TOKEN",
        "!synthetic-command",
        "Bearer ${TOKEN}",
        "${TOKEN:-fallback}",
        "${}",
        "${9TOKEN}",
        "${TOKEN}${OTHER}",
        "${TOKEN",
        "${TOKEN}\n",
        1,
        None,
    ],
)
def test_environment_reference_subset(tmp_path, reference):
    with pytest.raises(McpConfigError, match=r"only whole-value \$\{NAME\}"):
        server(tmp_path, env={"TOKEN": reference})


@pytest.mark.parametrize("target", ["", "9TOKEN", "TOKEN=bad", "TOKEN\x00", "A" * 129, 1])
def test_environment_target_names(tmp_path, target):
    with pytest.raises(McpConfigError, match="environment variable name"):
        server(tmp_path, env={target: "${SOURCE}"})


def test_configuration_is_deeply_immutable_and_detached(tmp_path):
    data = document(args=["one"], env={"TOKEN": "${FIRST}"})
    config = parse_mcp_config(data, cwd=tmp_path)
    one = config.servers[0]
    data["mcpServers"]["example"]["args"].append("changed")
    data["mcpServers"]["example"]["env"]["TOKEN"] = "${SECOND}"
    assert one.args == ("one",) and one.env["TOKEN"] == "${FIRST}"
    with pytest.raises(FrozenInstanceError):
        one.command = "changed"
    with pytest.raises(FrozenInstanceError):
        config.servers = ()
    with pytest.raises(TypeError):
        one.env["TOKEN"] = "${SECOND}"


def test_direct_construction_cannot_bypass_reference_or_exposure_validation(tmp_path):
    one = server(tmp_path)
    for changes in (
        {"env": {"TOKEN": "literal-secret"}},
        {"exposure": "hidden"},
        {"args": ["one"]},
        {"cwd": Path("relative")},
        {"enabled": 1},
    ):
        with pytest.raises(McpConfigError):
            replace(one, **changes)
    with pytest.raises(McpConfigError):
        McpConfig(servers=[one])
    with pytest.raises(McpConfigError, match="Duplicate"):
        McpConfig(servers=(one, one))


def test_repr_redacts_all_user_configuration_and_has_no_serializer(tmp_path):
    config = parse_mcp_config(
        {
            "mcpServers": {
                "synthetic-name-secret": {
                    "command": "synthetic-command-secret",
                    "args": ["synthetic-arg-secret"],
                    "cwd": "synthetic-path-secret",
                    "env": {"SYNTHETIC_TARGET": "${SYNTHETIC_SOURCE}"},
                    "exposure": "direct",
                }
            }
        },
        cwd=tmp_path,
    )
    one = config.servers[0]
    for text in (repr(config), str(config), repr(one), str(one)):
        assert "synthetic" not in text.lower() and "redacted" in text
    assert not hasattr(config, "to_json") and not hasattr(one, "to_json")
    with pytest.raises(TypeError):
        json.dumps(config)


class ExactLookups(Mapping):
    """A synthetic environment that fails on enumeration or undeclared lookups."""

    def __init__(self, values, allowed=None):
        self.values = values
        self.allowed = set(HOST_ENVIRONMENT_KEYS) if allowed is None else set(allowed)
        self.lookups = []

    def __getitem__(self, key):
        assert key in self.allowed, "Unexpected environment lookup"
        self.lookups.append(key)
        return self.values[key]

    def __iter__(self):
        pytest.fail("Environment must never be enumerated")

    def __len__(self):
        pytest.fail("Environment must never be copied")


def test_minimal_host_environment_uses_exact_lookups_and_no_provider_key(tmp_path, monkeypatch):
    values = {key: f"synthetic-{key}" for key in HOST_ENVIRONMENT_KEYS}
    values.update(
        OPENAI_API_KEY="synthetic-provider-secret",
        LD_PRELOAD="synthetic-injection",
        PYTHONPATH="synthetic-injection",
        HTTPS_PROXY="synthetic-proxy",
    )
    environment = ExactLookups(values)
    with monkeypatch.context() as patch:
        patch.setattr(config_module.os, "environ", environment)
        assert resolve_server_environment(server(tmp_path)) == {
            key: values[key] for key in HOST_ENVIRONMENT_KEYS
        }
    assert environment.lookups == list(HOST_ENVIRONMENT_KEYS)


def test_parse_and_load_never_read_environment(tmp_path, monkeypatch):
    environment = ExactLookups({}, allowed=[])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document(env={"TOKEN": "${OPENAI_API_KEY}"})))
    with monkeypatch.context() as patch:
        patch.setattr(config_module.os, "environ", environment)
        one = load_mcp_config(path, cwd=tmp_path).servers[0]
        parse_mcp_config(document(env={"TOKEN": "${OPENAI_API_KEY}"}), cwd=tmp_path)
        assert one.env["TOKEN"] == "${OPENAI_API_KEY}"
    assert environment.lookups == []


def test_explicit_reference_can_forward_provider_key_without_other_inheritance(tmp_path):
    one = server(tmp_path, env={"TOKEN": "${OPENAI_API_KEY}", "OPENAI_API_KEY": "${OPENAI_API_KEY}"})
    environment = ExactLookups(
        {"OPENAI_API_KEY": "synthetic-provider-secret"}, allowed=(*HOST_ENVIRONMENT_KEYS, "OPENAI_API_KEY")
    )
    assert resolve_server_environment(one, environment) == {
        "TOKEN": "synthetic-provider-secret",
        "OPENAI_API_KEY": "synthetic-provider-secret",
    }
    assert "synthetic-provider-secret" not in repr(one)


def test_supplied_mapping_has_no_host_fallback_and_resolution_is_not_cached(tmp_path):
    one = server(tmp_path, env={"PATH": "${SERVER_PATH}", "TOKEN": "${SERVER_TOKEN}"})
    environment = {"PATH": "/host", "SERVER_PATH": "/server", "SERVER_TOKEN": "synthetic-first"}
    first = resolve_server_environment(one, environment)
    assert first == {"PATH": "/server", "TOKEN": "synthetic-first"}
    environment["SERVER_TOKEN"] = ""
    assert resolve_server_environment(one, environment) == {"PATH": "/server", "TOKEN": ""}
    assert first["TOKEN"] == "synthetic-first"
    with pytest.raises(McpConfigError, match="variable is missing"):
        resolve_server_environment(server(tmp_path, env={"TOKEN": "${OPENAI_API_KEY}"}), {})


@pytest.mark.parametrize("value", ["synthetic-secret\x00", "\ud800", "x" * 8193, 1])
def test_malformed_runtime_environment_values_are_generic(tmp_path, value):
    one = server(tmp_path, env={"TOKEN": "${SYNTHETIC_SOURCE}"})
    with pytest.raises(McpConfigError) as exc:
        resolve_server_environment(one, {"SYNTHETIC_SOURCE": value})
    assert "synthetic" not in str(exc.value).lower() and len(str(exc.value)) < 160
    with pytest.raises(McpConfigError):
        resolve_server_environment(one, {"PATH": value})


@pytest.mark.parametrize(
    "raw",
    [
        b'{"mcpServers":{},"mcpServers":{}}',
        b'{"mcpServers":{"one":{"command":"a","command":"b","exposure":"direct"}}}',
        b'{"mcpServers":{"one":{},"one":{}}}',
        b'{"mcpServers":{"one":{"env":{"TOKEN":"${A}","TOKEN":"${B}"}}}}',
        b'{"synthetic-secret":NaN}',
        b'{"synthetic-secret":Infinity}',
        b'{"synthetic-secret":-Infinity}',
        b'{"mcpServers":{"one":{"command":"a","exposure":"direct","timeout":1e999}}}',
        b'{"synthetic-secret":',
        b"\xff",
        b"[]",
        b'{"mcpServers":null}',
        b"[" * 1500 + b"0" + b"]" * 1500,
    ],
)
def test_strict_json_errors_are_bounded_and_do_not_echo_input(tmp_path, raw):
    path = tmp_path / "synthetic-path-secret.json"
    path.write_bytes(raw)
    with pytest.raises(McpConfigError) as exc:
        load_mcp_config(path, cwd=tmp_path)
    formatted = "".join(traceback.format_exception_only(exc.value))
    assert "synthetic" not in formatted and len(str(exc.value)) < 160
    assert exc.value.__cause__ is None


def test_file_limit_allows_exact_boundary_and_rejects_one_extra_byte(tmp_path):
    path = tmp_path / "config.json"
    prefix = b'{"mcpServers":{}}'
    path.write_bytes(prefix + b" " * (MAX_CONFIG_BYTES - len(prefix)))
    assert load_mcp_config(path, cwd=tmp_path).servers == ()
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(McpConfigError, match="1 MiB"):
        load_mcp_config(path, cwd=tmp_path)


def test_read_is_bounded_even_when_file_grows_after_stat(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    monkeypatch.setattr(
        config_module.os, "fstat", lambda fd: SimpleNamespace(st_mode=stat.S_IFREG, st_size=0)
    )
    original = config_module.os.fdopen
    reads = []

    class RecordingStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

    def record_stream(*args, **kwargs):
        recording = RecordingStream()
        recording.stream = original(*args, **kwargs)
        return recording

    monkeypatch.setattr(config_module.os, "fdopen", record_stream)
    with pytest.raises(McpConfigError, match="1 MiB"):
        load_mcp_config(path, cwd=tmp_path)
    assert reads == [MAX_CONFIG_BYTES + 1]


def test_nonregular_missing_and_unreadable_files_are_generic(tmp_path, monkeypatch):
    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    for path in (fifo, tmp_path, tmp_path / "synthetic-missing-secret.json"):
        with pytest.raises(McpConfigError) as exc:
            load_mcp_config(path, cwd=tmp_path)
        assert "synthetic" not in str(exc.value)

    def fail_open(*args, **kwargs):
        raise PermissionError("synthetic-permission-secret")

    monkeypatch.setattr(config_module.os, "open", fail_open)
    with pytest.raises(McpConfigError) as exc:
        load_mcp_config(tmp_path / "config.json", cwd=tmp_path)
    assert "synthetic" not in str(exc.value) and exc.value.__cause__ is None


def test_environment_file_and_symlink_alias_are_never_opened(tmp_path, monkeypatch):
    # Synthetic path only; no actual environment file is created or read.
    target = tmp_path / ".env"
    alias = tmp_path / "config.json"
    alias.symlink_to(target)

    def forbidden(*args, **kwargs):
        pytest.fail("Environment file was opened")

    monkeypatch.setattr(config_module.os, "open", forbidden)
    for path in (target, tmp_path / ".env.local", alias):
        with pytest.raises(McpConfigError, match="Environment files"):
            load_mcp_config(path, cwd=tmp_path)


def test_symlink_replacement_after_resolution_is_not_followed(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text("{}")
    original = Path.resolve

    def replace_with_alias(candidate, *args, **kwargs):
        resolved = original(candidate, *args, **kwargs)
        if candidate == path:
            path.unlink()
            path.symlink_to(tmp_path / ".env")
        return resolved

    monkeypatch.setattr(Path, "resolve", replace_with_alias)
    with pytest.raises(McpConfigError, match="could not be read"):
        load_mcp_config(path, cwd=tmp_path)


def test_collection_and_utf8_byte_budgets(tmp_path):
    cases = [
        document(args=["a"] * 257),
        document(env={f"V{i}": "${X}" for i in range(129)}),
        document(command="é" * 4097),
        document(args=["a" * 8193]),
        {"mcpServers": {f"s{i}": {"command": "a", "exposure": "direct"} for i in range(129)}},
    ]
    for data in cases:
        with pytest.raises(McpConfigError):
            parse_mcp_config(data, cwd=tmp_path)


def test_schema_errors_do_not_echo_user_keys_names_or_values(tmp_path):
    cases = [
        document(command="synthetic-secret\x00"),
        document(env={"TOKEN": "synthetic-secret"}),
        {"synthetic-secret-field": {}},
        {"mcpServers": {"synthetic-secret/name": {"command": "a", "exposure": "direct"}}},
    ]
    for data in cases:
        with pytest.raises(McpConfigError) as exc:
            parse_mcp_config(data, cwd=tmp_path)
        assert "synthetic" not in str(exc.value) and len(str(exc.value)) < 160


def test_module_has_no_sdk_or_dotenv_import_dependency(monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "mcp" and not name.startswith("mcp.")
        assert name != "dotenv" and not name.startswith("dotenv.")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded_import)
        # Load a separate module so production dataclass/error identities remain
        # stable for other tests and runtime modules already imported by pytest.
        name = "_mcp_config_without_optional_dependencies"
        spec = importlib.util.spec_from_file_location(name, config_module.__file__)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
