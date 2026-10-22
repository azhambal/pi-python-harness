from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import subprocess
import sys
import traceback
from dataclasses import FrozenInstanceError, asdict, fields, replace
from types import SimpleNamespace

import pytest

from pi_agent.mcp import manifest as manifest_module
from pi_agent.mcp._contracts import binding_name
from pi_agent.mcp.manifest import McpBinding, McpManifest, declaration_digest
from pi_agent.mcp.tools import McpTool
from pi_agent.tool_hooks import ToolProvenance
from pi_agent.tools.base import Tool

MAX_BYTES = 1024 * 1024


def binding(server="fixture", tool="echo", **changes):
    values = {
        "name": binding_name(server, tool),
        "server": server,
        "tool": tool,
        "schema_digest": "a" * 64,
        "declaration_digest": "b" * 64,
    }
    values.update(changes)
    return McpBinding(**values)


def document(**changes):
    data = {"version": 1, "bindings": [asdict(binding())]}
    data.update(changes)
    return data


def adapter(server="fixture", tool="echo", description="Echo café", parameters=None, output_schema=None):
    # No connection, SDK descriptor, process, or environment lookup is needed.
    connection = SimpleNamespace(
        config=SimpleNamespace(
            name=server,
            command="synthetic-command-private",
            args=("synthetic-argument-private",),
            env={"SYNTHETIC_TARGET": "synthetic-environment-private"},
            url="https://synthetic-private.invalid",
        )
    )
    return McpTool(
        connection,
        tool,
        description,
        {"type": "object"} if parameters is None else parameters,
        output_schema,
        {"readOnlyHint": True},
    )


class OrdinaryTool(Tool):
    name = "ordinary"

    async def execute(self, *args, **kwargs):
        pytest.fail("Manifest construction must not execute tools")


def test_exact_api_and_legacy_empty_catalog():
    assert manifest_module.__all__ == ["McpBinding", "McpManifest", "declaration_digest"]
    assert [field.name for field in fields(McpBinding)] == [
        "name",
        "server",
        "tool",
        "schema_digest",
        "declaration_digest",
    ]
    assert [field.name for field in fields(McpManifest)] == ["bindings"]
    empty = McpManifest()
    assert empty.bindings == ()
    assert empty.to_manifest() == '{"bindings":[],"version":1}'
    assert McpManifest.from_manifest(empty.to_manifest()) == empty
    assert McpManifest.from_tools([]) == empty
    assert McpManifest.from_tools([OrdinaryTool()]) == empty


def test_canonical_roundtrip_sorts_identity_and_preserves_ascii_encoding():
    one = binding(tool="检索 café 🐍")
    two = binding(server="other", tool="a-b")
    manifest = McpManifest((two, one))
    expected = json.dumps(
        {"version": 1, "bindings": [asdict(one), asdict(two)]},
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    assert manifest.to_manifest() == expected
    assert McpManifest((one, two)) == manifest
    assert expected.isascii()
    assert McpManifest.from_manifest(expected) == manifest
    pretty = json.dumps(json.loads(expected), ensure_ascii=False, indent=2)
    assert McpManifest.from_manifest(pretty).to_manifest() == expected


@pytest.mark.parametrize("server", ["a", "A_09-z", "s" * 128])
def test_valid_server_name_boundaries(server):
    assert binding(server=server).server == server


@pytest.mark.parametrize("tool", ["x", "t" * 256, "é" * 128, "🐍" * 64])
def test_valid_tool_utf8_boundaries(tool):
    assert binding(tool=tool).tool == tool


@pytest.mark.parametrize("server", ["", "s" * 129, "é", "a.b", "a/b", "a b", "a\n", None, 1, True])
def test_invalid_server_names_in_direct_and_json_construction(server):
    with pytest.raises(ValueError):
        replace(binding(), server=server)
    data = document()
    data["bindings"][0]["server"] = server
    with pytest.raises(ValueError):
        McpManifest.from_manifest(json.dumps(data))


@pytest.mark.parametrize("tool", ["", "t" * 257, "é" * 129, "🐍" * 65, "\ud800", None, 1, True])
def test_invalid_tool_names_in_direct_and_json_construction(tool):
    with pytest.raises(ValueError):
        replace(binding(), tool=tool)
    data = document()
    data["bindings"][0]["tool"] = tool
    with pytest.raises(ValueError):
        McpManifest.from_manifest(json.dumps(data))


@pytest.mark.parametrize("name", ["", "ordinary", "mcp__fake", "x" * 65, "é", None, 1, True])
def test_binding_name_must_match_host_identity(name):
    with pytest.raises(ValueError):
        replace(binding(), name=name)
    data = document()
    data["bindings"][0]["name"] = name
    with pytest.raises(ValueError):
        McpManifest.from_manifest(json.dumps(data))


@pytest.mark.parametrize("field", ["schema_digest", "declaration_digest"])
@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "é" * 64, None, 1, True])
def test_hashes_are_exact_lower_hex(field, digest):
    with pytest.raises(ValueError):
        replace(binding(), **{field: digest})
    data = document()
    data["bindings"][0][field] = digest
    with pytest.raises(ValueError):
        McpManifest.from_manifest(json.dumps(data))


def test_frozen_hashable_and_strict_direct_collection():
    one = binding()
    manifest = McpManifest((one,))
    assert hash(manifest) == hash(McpManifest((one,)))
    with pytest.raises(FrozenInstanceError):
        one.server = "other"
    with pytest.raises(FrozenInstanceError):
        manifest.bindings = ()
    for invalid in ([one], None, (asdict(one),), (None,)):
        with pytest.raises(ValueError):
            McpManifest(invalid)


def test_duplicate_names_and_server_tool_identities_rejected():
    one = binding()
    different_hashes = replace(one, schema_digest="c" * 64, declaration_digest="d" * 64)
    for other in (one, different_hashes):
        with pytest.raises(ValueError, match="Duplicate"):
            McpManifest((one, other))
        with pytest.raises(ValueError, match="Duplicate"):
            McpManifest.from_manifest(json.dumps(document(bindings=[asdict(one), asdict(other)])))
    assert len(McpManifest((binding(tool="a-b"), binding(tool="a_b"))).bindings) == 2
    assert len(McpManifest((binding(), binding(server="other"))).bindings) == 2


@pytest.mark.parametrize("version", [True, False, None, "1", 1.0, 0, 2])
def test_version_is_exact_integer_one(version):
    with pytest.raises(ValueError):
        McpManifest.from_manifest(json.dumps(document(version=version)))


@pytest.mark.parametrize("data", [None, [], 1, "text", {}, {"version": 1}, {"bindings": []}])
def test_strict_top_level_shape(data):
    with pytest.raises(ValueError):
        McpManifest.from_manifest(json.dumps(data))


@pytest.mark.parametrize("items", [None, {}, "text", 1, [None], [[]], ["text"], [1]])
def test_strict_binding_array_shape(items):
    with pytest.raises(ValueError):
        McpManifest.from_manifest(json.dumps(document(bindings=items)))


@pytest.mark.parametrize("field", ["name", "server", "tool", "schema_digest", "declaration_digest"])
def test_all_binding_fields_required(field):
    data = document()
    del data["bindings"][0][field]
    with pytest.raises(ValueError):
        McpManifest.from_manifest(json.dumps(data))


@pytest.mark.parametrize("field", ["config", "command", "args", "env", "url", "credentials", "x" * 65])
@pytest.mark.parametrize("location", ["root", "binding"])
def test_unknown_fields_never_admitted(field, location):
    data = document()
    target = data if location == "root" else data["bindings"][0]
    target[field] = "synthetic-private"
    with pytest.raises(ValueError) as exc:
        McpManifest.from_manifest(json.dumps(data))
    assert "synthetic" not in str(exc.value)


@pytest.mark.parametrize(
    "raw",
    [
        '{"version":1,"version":1,"bindings":[]}',
        '{"version":1,"bindings":[],"bindings":[]}',
        '{"version":1,"bindings":[{"name":"x","name":"x"}]}',
        '{"version":1,"bindings":[{"name":"x","na\\u006de":"x"}]}',
        '{"version":NaN,"bindings":[]}',
        '{"version":Infinity,"bindings":[]}',
        '{"version":-Infinity,"bindings":[]}',
        '{"version":1e999,"bindings":[]}',
        '{"version":1,"bindings":[],"extra":{"nested":NaN}}',
        "synthetic-private",
        '{"synthetic-private":',
        '{"version":1,"bindings":[]} trailing',
        "[" * 1500 + "0" + "]" * 1500,
        '{"version":' + "9" * 5000 + ',"bindings":[]}',
        "\ud800",
    ],
)
def test_strict_json_errors_are_bounded_and_do_not_echo_input(raw):
    with pytest.raises(ValueError) as exc:
        McpManifest.from_manifest(raw)
    formatted = "".join(traceback.format_exception_only(exc.value))
    assert "synthetic-private" not in formatted
    assert len(str(exc.value)) < 160
    assert exc.value.__cause__ is None


@pytest.mark.parametrize("text", [None, b'{"version":1,"bindings":[]}', {}, 1])
def test_codec_requires_string_input(text):
    with pytest.raises(ValueError):
        McpManifest.from_manifest(text)


def test_manifest_utf8_byte_limit_inclusive():
    one = binding(tool="é")
    raw = json.dumps(document(bindings=[asdict(one)]), ensure_ascii=False)
    padded = raw + " " * (MAX_BYTES - len(raw.encode("utf-8")))
    assert McpManifest.from_manifest(padded) == McpManifest((one,))
    with pytest.raises(ValueError, match="1 MiB"):
        McpManifest.from_manifest(padded + " ")


def test_binding_count_limit_inclusive_in_codec_and_constructor():
    bindings = tuple(binding(server=f"s{i}") for i in range(1024))
    manifest = McpManifest(bindings)
    assert McpManifest.from_manifest(manifest.to_manifest()) == manifest
    with pytest.raises(ValueError, match="1024"):
        McpManifest((*bindings, binding(server="extra")))
    raw = json.dumps(document(bindings=[asdict(one) for one in (*bindings, binding(server="extra"))]))
    with pytest.raises(ValueError, match="1024"):
        McpManifest.from_manifest(raw)


def test_canonical_output_byte_limit_applies_to_direct_and_decoded_construction():
    # A short valid UTF-8 tool name can expand sixfold in canonical escaped JSON.
    bindings = tuple(binding(server=f"s{i}", tool="\x01" * 256) for i in range(1024))
    with pytest.raises(ValueError, match="1 MiB"):
        McpManifest(bindings)
    # Literal DEL is legal JSON and expands in ensure_ascii canonical output.
    bindings = tuple(binding(server=f"s{i}", tool="\x7f" * 256) for i in range(1024))
    raw = json.dumps(document(bindings=[asdict(one) for one in bindings]), ensure_ascii=False)
    assert len(raw.encode("utf-8")) < MAX_BYTES
    with pytest.raises(ValueError, match="1 MiB"):
        McpManifest.from_manifest(raw)


def test_declaration_digest_hashes_complete_canonical_declaration():
    tool = adapter(parameters={"properties": {"é": {"type": "string"}}, "type": "object"})
    encoded = json.dumps(
        tool.declaration().to_json(),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    assert declaration_digest(tool) == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    same = adapter(parameters={"type": "object", "properties": {"é": {"type": "string"}}})
    assert declaration_digest(same) == declaration_digest(tool)
    original = declaration_digest(tool)
    tool.description += " changed"
    assert declaration_digest(tool) != original
    assert declaration_digest(adapter(description="new")) != original
    assert declaration_digest(adapter(tool="other")) != original
    assert declaration_digest(adapter(parameters={"type": "object"})) != original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object()])
def test_declaration_digest_rejects_non_json_or_non_finite_values(value):
    tool = OrdinaryTool()
    tool.parameters = {"default": value}
    with pytest.raises(ValueError):
        declaration_digest(tool)


def test_from_tools_copies_only_actual_adapter_provenance():
    actual = adapter()
    fake = OrdinaryTool()
    fake.name = actual.name
    fake.provenance = actual.provenance

    class DerivedAdapter(McpTool):
        pass

    derived = DerivedAdapter(actual._connection, "derived", "Derived", {"type": "object"}, None, {})
    fake.declaration = lambda: pytest.fail("Fake MCP provenance must be ignored")
    derived.declaration = lambda: pytest.fail("Subclasses must be ignored")
    manifest = McpManifest.from_tools((fake, actual, derived))
    assert manifest.bindings == (
        McpBinding(
            actual.name, "fixture", "echo", actual.provenance.schema_digest, declaration_digest(actual)
        ),
    )
    before = manifest.to_manifest()
    actual.name = "changed"
    actual.description = "changed"
    actual.provenance = ToolProvenance("mcp", "other", "other", "c" * 64)
    assert manifest.to_manifest() == before


def test_schema_digest_retains_output_contract_and_declaration_digest_retains_prose():
    first = adapter()
    output_changed = adapter(output_schema={"type": "object"})
    prose_changed = adapter(description="Changed prose")
    one, two, three = (
        McpManifest.from_tools([tool]).bindings[0] for tool in (first, output_changed, prose_changed)
    )
    assert one.schema_digest != two.schema_digest
    assert one.declaration_digest == two.declaration_digest
    assert one.schema_digest == three.schema_digest
    assert one.declaration_digest != three.declaration_digest


def test_from_tools_validates_identity_missing_provenance_and_duplicates():
    tool = adapter()
    with pytest.raises(ValueError, match="Duplicate"):
        McpManifest.from_tools([tool, tool])
    tool.name = "forged"
    with pytest.raises(ValueError, match="binding name"):
        McpManifest.from_tools([tool])
    tool.provenance = None
    with pytest.raises(ValueError, match="provenance"):
        McpManifest.from_tools([tool])


def test_from_tools_binding_limit():
    tools = [adapter(server=f"s{i}") for i in range(1024)]
    assert len(McpManifest.from_tools(tools).bindings) == 1024
    with pytest.raises(ValueError, match="1024"):
        McpManifest.from_tools([*tools, adapter(server="extra")])


def test_repr_and_json_contain_identity_and_no_runtime_config():
    manifest = McpManifest.from_tools([adapter()])
    one = manifest.bindings[0]
    for text in (repr(one), repr(manifest), manifest.to_manifest()):
        assert one.name in text and "fixture" in text and "echo" in text
        assert one.schema_digest in text and one.declaration_digest in text
        assert "synthetic" not in text and "https://" not in text
    assert set(json.loads(manifest.to_manifest())["bindings"][0]) == {
        "name",
        "server",
        "tool",
        "schema_digest",
        "declaration_digest",
    }


def test_module_import_defers_own_adapter_import(monkeypatch):
    original = builtins.__import__
    imports = []

    def guarded(name, *args, **kwargs):
        assert name != "mcp" and not name.startswith("mcp.")
        if name == "tools" and args[2:] == (("McpTool",), 1):
            imports.append(name)
        return original(name, *args, **kwargs)

    name = "pi_agent.mcp._isolated_manifest"
    spec = importlib.util.spec_from_file_location(name, manifest_module.__file__)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded)
        patch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        assert imports == []
        module.McpManifest.from_tools([])
        assert imports == ["tools"]


def test_fresh_process_codec_and_adapter_collection_need_no_optional_sdk():
    code = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    level = args[3] if len(args) > 3 else kwargs.get('level', 0)
    if level == 0 and (name == 'mcp' or name.startswith('mcp.')):
        raise AssertionError('Optional SDK imported')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from types import SimpleNamespace
from pi_agent.mcp.manifest import McpManifest
from pi_agent.mcp.tools import McpTool
tool = McpTool(SimpleNamespace(config=SimpleNamespace(name='fixture')), 'echo',
               'Echo', {'type': 'object'}, None, {})
manifest = McpManifest.from_tools([tool])
assert McpManifest.from_manifest(manifest.to_manifest()) == manifest
assert not any(name == 'mcp' or name.startswith('mcp.') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
