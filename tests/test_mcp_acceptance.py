from __future__ import annotations

import asyncio
import io
import json
import runpy
import sys
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from pi_agent import SkillSettings, cli, create_agent_session
from pi_agent.chat import run_chat
from pi_agent.mcp import McpConnection, binding_name
from pi_agent.mcp.reference import REFERENCE_VERSIONS, reference_config
from pi_agent.providers import FauxProvider
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, UserMessage, content_text


async def test_unmodified_pi_skill_fixture_loads_and_expands(config):
    # Copied byte-for-byte from Pi packages/coding-agent/test/fixtures/skills/valid-skill/SKILL.md.
    root = Path(__file__).parent / "fixtures/pi_skills"
    fixture = root / "valid-skill/SKILL.md"
    configured = replace(config, skills=SkillSettings(enabled=True, include_user=False, paths=(root,)))
    session = create_agent_session(
        FauxProvider([AssistantMessage(content=[TextBlock(text="done")])]), config=configured
    )
    try:
        assert "valid-skill" in session.system_prompt
        assert "This is a valid skill" not in session.system_prompt
        await session.prompt("/skill:valid-skill check compatibility")
        user = next(message for message in session.messages if isinstance(message, UserMessage))
        body = fixture.read_text().split("---", 2)[2].strip()
        assert body in content_text(user.content)
    finally:
        await session.aclose()


async def test_real_git_fetch_combined_acceptance(tmp_path):
    for name, expected in REFERENCE_VERSIONS.items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            pytest.skip("Optional reference servers missing; install .[dev,mcp-reference]")
        assert actual == expected
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/mcp_acceptance.py"))
    async with asyncio.timeout(45):
        report = await module["acceptance"](tmp_path)
    assert report["status"] == "passed"
    assert report["active_mcp_bindings"] == 8 and report["model_requests"] == 0
    assert report["largest_mcp_result_message_bytes"] < 50 * 1024
    assert "abort-no-replay" in report["checks"] and "child-task-lock-cleanup" in report["checks"]


async def test_pinned_fetch_refuses_loopback_without_test_opt_in(tmp_path):
    pytest.importorskip("mcp_server_fetch")
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/mcp_acceptance.py"))
    async with module["DocumentServer"]() as documents:
        client = McpConnection(reference_config(tmp_path, Path(sys.executable)).servers[1], environment={})
        try:
            tools = await client.discover()
            result = await tools[0].execute("id", {"url": documents.base + "/spec"})
            assert result.is_error and "not a public address" in content_text(result.content)
            assert documents.paths == []
        finally:
            await client.aclose()


async def test_large_skill_tree_and_paginated_catalog_are_bounded(tmp_path):
    pytest.importorskip("mcp")
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/benchmark_startup.py"))
    report = await module["workload"](tmp_path)
    assert report["generated_skills"] == 1000 and report["admitted_skills"] == 256
    assert report["skill_catalog_bytes"] <= 4096
    assert report["mcp_tools"] == 256
    assert report["model_text_bytes"] < 22 * 1024
    assert report["original_text_bytes"] == 60000
    assert report["structured_envelope_bytes"] < 1024 * 1024
    assert report["result_json_bytes"] > report["structured_envelope_bytes"]
    assert report["result_truncated"] and not report["timings_are_ci_thresholds"]


@pytest.mark.parametrize("command", ["run", "chat"])
async def test_cli_with_real_git_selection_json_and_cleanup(tmp_path, monkeypatch, capsys, command):
    pytest.importorskip("mcp_server_git")
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/mcp_acceptance.py"))
    repository = tmp_path / "repository"
    repository.mkdir()
    module["git"](repository, "init", "-q")
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "git": {
                        "command": sys.executable,
                        "args": ["-m", "mcp_server_git", "--repository", str(repository)],
                        "exposure": "direct",
                        "timeout": 15,
                    }
                }
            }
        )
    )
    provider = FauxProvider(
        [
            AssistantMessage(
                content=[
                    ToolCallBlock(
                        id="status",
                        name=binding_name("git", "git_status"),
                        arguments={"repo_path": str(repository)},
                    )
                ],
                stop_reason="toolUse",
            ),
            AssistantMessage(content=[TextBlock(text="Git check complete")], stop_reason="stop"),
        ]
    )
    # Replace credential/settings boundaries before the actual CLI async factory.
    monkeypatch.setattr(
        cli,
        "read_environment",
        lambda path: {"OPENAI_API_KEY": "synthetic-cli-secret", "OPENAI_MODEL": "offline"},
    )
    monkeypatch.setattr(cli, "load_settings", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "_runtime_provider", lambda settings: provider)
    monkeypatch.setattr("pi_agent.config.DEFAULT_AGENT_DIR", tmp_path / "agent")
    monkeypatch.setattr("pi_agent.mcp.config.os.environ", {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    original = cli._open_session_async
    sessions = []
    pids = []

    async def capture(args):
        result = await original(args)
        sessions.append(result[0])
        pids.extend(module["child_pids"](result[0]))
        return result

    monkeypatch.setattr(cli, "_open_session_async", capture)
    args = cli._build_parser().parse_args(
        [
            command,
            *(["Check Git status"] if command == "run" else []),
            "--cwd",
            str(repository),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--mcp-config",
            str(cfg),
            "--mcp-tools",
            "git/git_status",
            "--no-skills",
            "--json",
        ]
    )
    if command == "chat":
        monkeypatch.setattr("sys.stdin", io.StringIO("Check Git status\n/wait\n"))
    assert await (cli._cmd_run(args) if command == "run" else run_chat(args)) == 0
    module["reaped"](pids)
    assert sessions[0]._disposed
    assert [binding.tool for binding in sessions[0].mcp_bindings] == ["git_status"]
    output = capsys.readouterr()
    events = [json.loads(line) for line in output.out.splitlines()]
    assert any(event["type"] == "tool_execution_end" for event in events)
    assert "synthetic-cli-secret" not in output.out + output.err
