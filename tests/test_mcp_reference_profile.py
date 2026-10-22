from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from pi_agent import Cancellation, ToolProvenance
from pi_agent.mcp import binding_name, parse_mcp_config
from pi_agent.mcp.reference import (
    GIT_READ_TOOLS,
    REFERENCE_SELECTORS,
    reference_config,
    reference_hooks,
    reference_permissions,
)
from pi_agent.permissions import permission_decision
from pi_agent.tool_hooks import BeforeToolCallContext
from pi_agent.types import AssistantMessage, ToolCallBlock


def context(root, name="git_status", args=None, server="git"):
    block = ToolCallBlock(
        id="id", name=binding_name(server, name), arguments=args or {"repo_path": str(root)}
    )
    return BeforeToolCallContext(
        AssistantMessage(content=[block]),
        block,
        block.arguments,
        cwd=root,
        tool_provenance=ToolProvenance("mcp", server, name, "0" * 64),
    )


@pytest.mark.parametrize("name", GIT_READ_TOOLS)
def test_selected_git_reads_require_exact_root_and_then_allow(tmp_path, name):
    ctx = context(tmp_path, name)
    assert reference_hooks(tmp_path).before_tool_call(ctx, Cancellation()) is None
    assert permission_decision(reference_permissions(), ctx)[0] == "allow"
    assert f"git/{name}" in REFERENCE_SELECTORS


@pytest.mark.parametrize("path", [".", "child/.."])
def test_git_relative_paths_are_bound_to_server_root(tmp_path, path):
    (tmp_path / "child").mkdir()
    ctx = context(tmp_path, args={"repo_path": path})
    assert reference_hooks(tmp_path).before_tool_call(ctx, Cancellation()) is None


@pytest.mark.parametrize("target", ["outside", "nested", "symlink", "missing", "empty", "boolean"])
def test_repository_substitution_is_blocked_before_permissions(tmp_path, target):
    nested = tmp_path / "nested"
    nested.mkdir()
    external = tmp_path.parent / (tmp_path.name + "-outside")
    external.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(external, target_is_directory=True)
    value = {
        "outside": str(external),
        "nested": str(nested),
        "symlink": str(link),
        "missing": None,
        "empty": "",
        "boolean": True,
    }[target]
    ctx = context(tmp_path, args={"repo_path": value})
    result = reference_hooks(tmp_path).before_tool_call(ctx, Cancellation())
    assert result.block and str(external) not in result.reason


@pytest.mark.parametrize("name", ["git_commit", "git_add", "git_reset", "git_checkout", "git_create_branch"])
def test_mutations_are_never_admitted_by_profile(tmp_path, name):
    ctx = context(tmp_path, name)
    assert reference_hooks(tmp_path).before_tool_call(ctx, Cancellation()).block
    assert permission_decision(reference_permissions(), ctx)[0] == "deny"
    assert f"git/{name}" not in REFERENCE_SELECTORS


@pytest.mark.parametrize("url", ["https://example.invalid/doc", "http://127.0.0.1:8000/doc"])
def test_fetch_requires_manual_confirmation_even_for_admitted_url(tmp_path, url):
    ctx = context(tmp_path, "fetch", {"url": url}, "fetch")
    assert reference_hooks(tmp_path).before_tool_call(ctx, Cancellation()) is None
    assert permission_decision(reference_permissions(), ctx)[0] == "ask"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:synthetic-secret@example.invalid",
        "ftp://example.invalid",
        "http://",
        None,
    ],
)
def test_fetch_bad_destination_is_blocked_without_echo(tmp_path, url):
    ctx = context(tmp_path, "fetch", {"url": url}, "fetch")
    result = reference_hooks(tmp_path).before_tool_call(ctx, Cancellation())
    assert result.block and "synthetic-secret" not in result.reason


def test_provenance_and_binding_must_agree(tmp_path):
    ctx = context(tmp_path)
    changed = replace(ctx.tool_call, name=binding_name("other", "git_status"))
    assert reference_hooks(tmp_path).before_tool_call(replace(ctx, tool_call=changed), Cancellation()).block
    assert reference_hooks(tmp_path).before_tool_call(context(tmp_path, server="other"), Cancellation()).block
    assert (
        reference_hooks(tmp_path).before_tool_call(replace(ctx, tool_provenance=None), Cancellation()).block
    )


def test_profile_preserves_venv_executable_and_private_fetch_is_opt_in(tmp_path):
    python = Path(sys.executable)
    default = reference_config(tmp_path, python)
    assert default.servers[0].command == str(python)
    assert default.servers[0].args[-1] == str(tmp_path)
    assert "--allow-private-ips" not in default.servers[1].args
    assert (
        "--allow-private-ips" in reference_config(tmp_path, python, allow_private_fetch=True).servers[1].args
    )
    with pytest.raises(ValueError):
        reference_config(tmp_path, python, allow_private_fetch=1)


def test_example_config_loads_without_env_or_launch(tmp_path):
    import json

    path = Path(__file__).parents[1] / "examples/mcp.example.json"
    cfg = parse_mcp_config(json.loads(path.read_text()), cwd=tmp_path)
    assert [server.name for server in cfg.servers] == ["git", "fetch"]
    assert all(server.env == {} and server.command == ".venv/bin/python" for server in cfg.servers)
