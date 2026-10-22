import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from pi_agent import (
    AgentConfig,
    BeforeToolCallContext,
    PermissionPolicy,
    PermissionRule,
    ToolHooks,
    create_agent_session,
)
from pi_agent.permissions import permission_decision
from pi_agent.providers import FauxProvider
from pi_agent.session_store import SessionStore
from pi_agent.tools import WriteTool
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage


def call(name: str = "write", call_id: str = "c", **args) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCallBlock(id=call_id, name=name, arguments=args)], stop_reason="toolUse"
    )


def done() -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop")


def context(cwd: Path, name: str, **args) -> BeforeToolCallContext:
    block = ToolCallBlock(id="c", name=name, arguments=args)
    return BeforeToolCallContext(assistant_message=call(name, **args), tool_call=block, args=args, cwd=cwd)


@pytest.mark.parametrize(
    "tool,args",
    [
        ("write", {"path": "never.txt", "content": "new"}),
        ("edit", {"path": "source.txt", "edits": [{"oldText": "old", "newText": "new"}]}),
        ("bash", {"command": "touch never.txt"}),
    ],
)
async def test_read_only_blocks_mutating_dispatch_even_with_allow_rule(
    config: AgentConfig, tool: str, args: dict
) -> None:
    (config.cwd / "source.txt").write_text("old")
    after_calls = []
    session = create_agent_session(
        FauxProvider([call(tool, **args), done()]),
        config=config,
        permission_policy=PermissionPolicy(read_only=True, rules=(PermissionRule("*", "allow"),)),
        hooks=ToolHooks(after_tool_call=lambda ctx, token: after_calls.append(ctx)),
    )
    try:
        await session.prompt("run")
        assert (config.cwd / "source.txt").read_text() == "old"
        assert not (config.cwd / "never.txt").exists()
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and "Read-only" in result.content[0].text
        assert after_calls == []
    finally:
        session.dispose()


async def test_read_only_allows_read_and_survives_resume(config: AgentConfig) -> None:
    (config.cwd / "source.txt").write_text("source")
    session = create_agent_session(
        FauxProvider([call("read", path="source.txt"), done()]),
        config=replace(config, permissions=PermissionPolicy(read_only=True)),
        session="new",
    )
    try:
        await session.prompt("read")
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert not result.is_error and result.content[0].text == "source"
        assert session.store.header.config["permissions"]["read_only"]
        path = session.session_file
    finally:
        session.dispose()
    resumed = create_agent_session(
        FauxProvider([call("write", call_id="second", path="never.txt", content="new"), done()]), session=path
    )
    try:
        assert resumed.config.permissions.read_only
        await resumed.prompt("write")
        assert not (config.cwd / "never.txt").exists()
        assert next(
            message
            for message in resumed.messages
            if isinstance(message, ToolResultMessage) and message.tool_call_id == "second"
        ).is_error
    finally:
        resumed.dispose()


async def test_explicit_policy_override_is_durable(config: AgentConfig) -> None:
    session = create_agent_session(FauxProvider([done()]), config=config, session="new")
    try:
        await session.prompt("hello")
        path = session.session_file
    finally:
        session.dispose()
    policy = PermissionPolicy(default="deny", rules=(PermissionRule("read", "allow"),))
    resumed = create_agent_session(FauxProvider([]), session=path, permission_policy=policy)
    resumed.dispose()
    reopened = create_agent_session(FauxProvider([]), session=path)
    try:
        assert reopened.config.permissions == policy
        assert reopened.store.entries[-1].type == "config_change"
    finally:
        reopened.dispose()


@pytest.mark.parametrize("approved", [True, False, None, "yes"])
async def test_ask_requires_explicit_boolean_and_never_changes_execution_args(
    config: AgentConfig, approved
) -> None:
    requests = []

    def confirm(request, token):
        requests.append(request)
        assert request.resolved_path == str(config.cwd / "result.txt")
        request.context.args["path"] = "wrong.txt"
        return approved

    session = create_agent_session(
        FauxProvider([call(path="result.txt", content="new"), done()]),
        config=config,
        permission_policy=PermissionPolicy(default="ask"),
        confirm_permission=confirm,
    )
    try:
        await session.prompt("write")
        assert len(requests) == 1
        assert (config.cwd / "result.txt").exists() is (approved is True)
        assert not (config.cwd / "wrong.txt").exists()
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error is (approved is not True)
    finally:
        session.dispose()


async def test_ask_without_callback_denies_and_after_hook_cannot_override_it(config: AgentConfig) -> None:
    after_calls = []
    session = create_agent_session(
        FauxProvider([call(path="never.txt", content="new"), done()]),
        config=config,
        permission_policy=PermissionPolicy(default="ask"),
        hooks=ToolHooks(after_tool_call=lambda ctx, token: after_calls.append(ctx)),
    )
    try:
        await session.prompt("write")
        assert not (config.cwd / "never.txt").exists()
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert "no confirmation callback" in result.content[0].text
        assert not after_calls
    finally:
        session.dispose()


async def test_confirmation_can_be_aborted_without_leaking_waiters(config: AgentConfig) -> None:
    reached = asyncio.Event()
    closed = asyncio.Event()
    before_tasks = asyncio.all_tasks()

    async def confirm(request, token):
        reached.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    session = create_agent_session(
        FauxProvider([call(path="never.txt", content="new")]),
        config=config,
        permission_policy=PermissionPolicy(default="ask"),
        confirm_permission=confirm,
    )
    task = asyncio.create_task(session.prompt("write"))
    try:
        await asyncio.wait_for(reached.wait(), 1)
        await asyncio.wait_for(session.abort(), 1)
        await task
        assert closed.is_set() and not (config.cwd / "never.txt").exists()
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and not result.interrupted
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.dispose()
    await asyncio.sleep(0)
    assert not (asyncio.all_tasks() - before_tasks)


def test_command_rules_use_first_match_and_do_not_parse_shell(tmp_path: Path) -> None:
    policy = PermissionPolicy(
        default="deny",
        rules=(
            PermissionRule("bash", "deny", command="rm *"),
            PermissionRule("bash", "ask", command="pytest*"),
            PermissionRule("read", "allow"),
        ),
    )
    assert permission_decision(policy, context(tmp_path, "bash", command="rm file"))[0] == "deny"
    assert permission_decision(policy, context(tmp_path, "bash", command="pytest -q"))[0] == "ask"
    assert permission_decision(policy, context(tmp_path, "bash", command="echo hello"))[0] == "deny"
    assert permission_decision(policy, context(tmp_path, "read", path="source.txt"))[0] == "allow"


def test_path_rules_resolve_symlinks_and_parent_traversal(tmp_path: Path) -> None:
    inside = tmp_path / "src"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    (inside / "link").symlink_to(outside, target_is_directory=True)
    policy = PermissionPolicy(default="deny", rules=(PermissionRule("write", "allow", path="src/**"),))
    assert permission_decision(policy, context(tmp_path, "write", path="src/file.py"))[0] == "allow"
    assert permission_decision(policy, context(tmp_path, "write", path="src/../outside/file.py"))[0] == "deny"
    assert permission_decision(policy, context(tmp_path, "write", path="src/link/file.py"))[0] == "deny"
    policy_link = PermissionPolicy(
        default="deny", rules=(PermissionRule("write", "allow", path="src/link/**"),)
    )
    assert permission_decision(policy_link, context(tmp_path, "write", path="outside/file.py"))[0] == "allow"


async def test_approval_does_not_cover_a_changed_symlink_target(config: AgentConfig) -> None:
    first, second = config.cwd / "first", config.cwd / "second"
    first.write_text("old first")
    second.write_text("old second")
    link = config.cwd / "link"
    link.symlink_to(first.name)

    def confirm(request, token):
        link.unlink()
        link.symlink_to(second.name)
        return True

    session = create_agent_session(
        FauxProvider([call(path="link", content="new"), done()]),
        config=config,
        permission_policy=PermissionPolicy(default="ask"),
        confirm_permission=confirm,
    )
    try:
        await session.prompt("write")
        assert first.read_text() == "old first" and second.read_text() == "old second"
        result = next(message for message in session.messages if isinstance(message, ToolResultMessage))
        assert result.is_error and "target changed" in result.content[0].text
    finally:
        session.dispose()


async def test_policy_uses_actual_injected_tool_cwd(config: AgentConfig) -> None:
    other = config.cwd / "other"
    other.mkdir()
    allowed = config.cwd / "allowed"
    allowed.mkdir()
    policy = PermissionPolicy(
        default="deny", rules=(PermissionRule("write", "allow", path=str(allowed / "*")),)
    )
    session = create_agent_session(
        FauxProvider([call(path="result.txt", content="new"), done()]),
        config=config,
        tools=[WriteTool(other)],
        permission_policy=policy,
    )
    try:
        await session.prompt("write")
        assert not (other / "result.txt").exists()
        assert next(
            message for message in session.messages if isinstance(message, ToolResultMessage)
        ).is_error
    finally:
        session.dispose()


@pytest.mark.parametrize(
    "data",
    [
        None,
        {"default": "invalid"},
        {"read_only": "false"},
        {"rules": "invalid"},
        {"typo": "allow"},
        {"rules": [{}]},
        {"rules": [{"tool": "write", "action": "allow", "typo": True}]},
        {"rules": [{"tool": "write", "action": "allow", "path": "../outside/*"}]},
    ],
)
def test_invalid_serialized_policy_is_not_permissive(data) -> None:
    with pytest.raises(ValueError):
        PermissionPolicy.from_json(data)


@pytest.mark.parametrize("permissions", [None, {"default": "typo"}, {"read_only": "false"}])
async def test_resume_rejects_malformed_present_policy_and_releases_lock(
    config: AgentConfig, permissions
) -> None:
    session = create_agent_session(FauxProvider([done()]), config=config, session="new")
    try:
        await session.prompt("hello")
        path = session.session_file
    finally:
        session.dispose()
    store = SessionStore.open(path)
    try:
        snapshot = dict(store.header.config)
        snapshot["permissions"] = permissions
        store.append_config_change(snapshot)
    finally:
        store.close()
    with pytest.raises(ValueError):
        create_agent_session(FauxProvider([]), session=path)
    # A rejected resume must not retain the writer lock.
    SessionStore.open(path).close()
