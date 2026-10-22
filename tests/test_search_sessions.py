import json
import sys
from dataclasses import replace

import pytest

from pi_agent import PermissionPolicy, PermissionRule, cli, create_agent_session
from pi_agent.providers import FauxProvider
from pi_agent.session_store import SessionStore, read_session_file
from pi_agent.settings import load_settings
from pi_agent.tools import CODING_TOOL_NAMES, TOOL_NAMES, GrepTool, create_builtin_tools, create_coding_tools
from pi_agent.tools.base import Tool
from pi_agent.types import (
    AssistantMessage,
    SystemMessage,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    get_current_tools,
)


def response(*calls):
    return AssistantMessage(content=list(calls), stop_reason="toolUse")


def call(name, number, **arguments):
    return ToolCallBlock(id=f"call-{number}", name=name, arguments=arguments)


def done():
    return AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop")


async def test_search_read_edit_pytest_parallel_turns_and_resume(config):
    (config.cwd / "value.py").write_text("def value():\n    return 1\n")
    (config.cwd / "test_value.py").write_text(
        "from value import value\n\ndef test_value():\n    assert value() == 2\n"
    )
    provider = FauxProvider(
        [
            response(call("grep", 1, pattern="return"), call("find", 2, pattern="*.py"), call("ls", 3)),
            response(call("read", 4, path="value.py")),
            response(
                call("edit", 5, path="value.py", edits=[{"oldText": "return 1", "newText": "return 2"}])
            ),
            response(call("bash", 6, command=f"{sys.executable} -m pytest -q test_value.py")),
            done(),
        ]
    )
    events = []
    session = create_agent_session(
        provider, config=replace(config, tool_execution="parallel"), session="new", tool_names=TOOL_NAMES
    )
    session.subscribe(lambda event: events.append(event.type))
    path = session.session_file
    try:
        await session.prompt("find and fix the value, run tests")
        results = [message for message in session.messages if isinstance(message, ToolResultMessage)]
        assert [result.tool_name for result in results] == ["grep", "find", "ls", "read", "edit", "bash"]
        assert all(not result.is_error for result in results)
        assert "1 passed" in results[-1].content[0].text
        assert "return 2" in (config.cwd / "value.py").read_text()
        assert events[-1] == "agent_settled"
    finally:
        session.dispose()
    restored = create_agent_session(FauxProvider([]), session=path)
    try:
        assert {tool.name for tool in restored.agent.tools} == set(TOOL_NAMES)
    finally:
        restored.dispose()


@pytest.mark.parametrize("names", [CODING_TOOL_NAMES, ("grep", "find", "ls"), ()])
async def test_resume_preserves_declared_set_and_defaults_do_not_expand(config, names):
    session = create_agent_session(FauxProvider([done()]), config=config, session="new", tool_names=names)
    await session.prompt("hello")
    path = session.session_file
    session.dispose()
    restored = create_agent_session(FauxProvider([]), session=path)
    try:
        assert {tool.name for tool in restored.agent.tools} == set(names)
        assert {tool.name for tool in get_current_tools(restored.messages)} == set(names)
    finally:
        restored.dispose()


def test_sdk_default_and_factories(config):
    assert [tool.name for tool in create_coding_tools(config.cwd)] == list(CODING_TOOL_NAMES)
    assert [tool.name for tool in create_builtin_tools(config.cwd)] == list(TOOL_NAMES)
    session = create_agent_session(FauxProvider([]), config=config)
    try:
        assert [tool.name for tool in session.agent.tools] == list(CODING_TOOL_NAMES)
    finally:
        session.dispose()


async def test_explicit_resume_selection_patches_prompt_and_tools_in_one_record(config):
    session = create_agent_session(FauxProvider([done()]), config=config, session="new")
    await session.prompt("hello")
    path = session.session_file
    count = session.store.entry_count
    prompt = session.system_prompt
    session.dispose()
    restored = create_agent_session(FauxProvider([]), session=path, tool_names=["read", "grep", "find", "ls"])
    try:
        assert restored.store.entry_count == count + 1
        entry = restored.store.last_entry
        assert isinstance(entry.message, SystemMessage)
        assert entry.message.sections and entry.message.tools_added and entry.message.tools_removed
        assert "Use bash for file operations" in prompt
        assert "Use bash for file operations" not in restored.system_prompt
        assert "Use grep" in restored.system_prompt
    finally:
        restored.dispose()
    again = create_agent_session(FauxProvider([]), session=path)
    try:
        assert {tool.name for tool in again.agent.tools} == {"read", "grep", "find", "ls"}
    finally:
        again.dispose()


@pytest.mark.parametrize("names", [["read", "read"], ["fake"], "read", [True]])
async def test_invalid_selection_never_changes_journal(config, names):
    session = create_agent_session(FauxProvider([done()]), config=config, session="new")
    await session.prompt("hello")
    path = session.session_file
    session.dispose()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="tool_names"):
        create_agent_session(FauxProvider([]), session=path, tool_names=names)
    assert path.read_bytes() == before


async def test_failed_loadout_fsync_has_one_coherent_record_on_recovery(config, monkeypatch):
    session = create_agent_session(FauxProvider([done()]), config=config, session="new")
    await session.prompt("hello")
    path = session.session_file
    session.dispose()

    def fail(descriptor):
        raise OSError("fsync failed")

    with monkeypatch.context() as patch:
        patch.setattr("pi_agent.session_store.os.fsync", fail)
        with pytest.raises(OSError, match="fsync failed"):
            create_agent_session(FauxProvider([]), session=path, tool_names=["grep"])
    restored = create_agent_session(FauxProvider([]), session=path)
    try:
        assert [tool.name for tool in restored.agent.tools] == ["grep"]
        assert "Use grep" in restored.system_prompt and "Use bash" not in restored.system_prompt
    finally:
        restored.dispose()


@pytest.mark.parametrize(
    "name,args", [("grep", {"pattern": "needle"}), ("find", {"pattern": "*.txt"}), ("ls", {})]
)
async def test_read_only_builtin_search_and_default_path_permission(config, name, args):
    (config.cwd / "source.txt").write_text("needle")
    requests = []
    policy = PermissionPolicy(
        read_only=True, default="deny", rules=(PermissionRule(name, "ask", path=str(config.cwd)),)
    )

    def confirm(request, cancellation):
        requests.append(request)
        return True

    session = create_agent_session(
        FauxProvider([response(call(name, 1, **args)), done()]),
        config=config,
        tool_names=[name],
        permission_policy=policy,
        confirm_permission=confirm,
    )
    try:
        await session.prompt("search")
        assert len(requests) == 1 and requests[0].resolved_path == str(config.cwd)
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert not result.is_error
    finally:
        session.dispose()


async def test_default_search_path_does_not_bypass_deny_rule(config):
    policy = PermissionPolicy(rules=(PermissionRule("grep", "deny", path=str(config.cwd)),))
    session = create_agent_session(
        FauxProvider([response(call("grep", 1, pattern="x")), done()]),
        config=config,
        tool_names=["grep"],
        permission_policy=policy,
    )
    try:
        await session.prompt("search")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and "denied" in result.content[0].text
    finally:
        session.dispose()


async def test_custom_tool_cannot_impersonate_read_only_search(config):
    class Impostor(Tool):
        name = "grep"
        parameters = GrepTool.parameters

        async def execute(self, *args, **kwargs):
            pytest.fail("untrusted search tool executed")

    session = create_agent_session(
        FauxProvider([response(call("grep", 1, pattern="x")), done()]),
        config=config,
        tools=[Impostor()],
        permission_policy=PermissionPolicy(read_only=True),
    )
    try:
        await session.prompt("search")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and "Read-only" in result.content[0].text
    finally:
        session.dispose()


async def test_cli_tools_defaults_settings_override_and_resume(config, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "read_environment", lambda path: {"OPENAI_API_KEY": "synthetic-only", "OPENAI_MODEL": "offline"}
    )
    defaults = {}
    monkeypatch.setattr(cli, "load_settings", lambda *args, **kwargs: defaults)
    monkeypatch.setattr(cli, "_runtime_provider", lambda settings: FauxProvider([done()]))
    parser = cli._build_parser()
    argv = ["run", "hello", "--cwd", str(config.cwd), "--session-dir", str(config.session_dir)]
    assert await cli._cmd_run(parser.parse_args(argv)) == 0
    path = SessionStore.list_sessions(config.session_dir, match_cwd=config.cwd)[0].path
    assert {
        tool.name
        for tool in get_current_tools(
            [entry.message for entry in read_session_file(path)[1] if hasattr(entry, "message")]
        )
    } == set(TOOL_NAMES)
    defaults["tools"] = ["read"]
    session, _, _, _ = cli._open_session(parser.parse_args(["run", "hello", "--resume", str(path)]))
    try:
        assert {tool.name for tool in session.agent.tools} == set(TOOL_NAMES)
    finally:
        session.dispose()
    session, _, _, _ = cli._open_session(parser.parse_args([*argv, "--no-session", "--tools", ""]))
    try:
        assert session.agent.tools == []
    finally:
        session.dispose()
    assert "synthetic-only" not in capsys.readouterr().out


def test_settings_tools_are_strict_and_not_environment_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"tools": ["read", "grep"]}))
    assert load_settings(tmp_path, explicit=path, agent_dir=tmp_path / "global")["tools"] == ["read", "grep"]
    path.write_text(json.dumps({"tools": ["grep", "grep"]}))
    with pytest.raises(ValueError, match="unique"):
        load_settings(tmp_path, explicit=path, agent_dir=tmp_path / "global")
