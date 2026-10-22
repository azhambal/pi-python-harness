import asyncio
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pi_agent import PermissionPolicy, SkillSettings, cli, create_agent_session
from pi_agent.chat import run_chat
from pi_agent.providers import FauxProvider
from pi_agent.resources import ResourceSnapshot
from pi_agent.settings import load_settings
from pi_agent.types import (
    AssistantMessage,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    content_text,
    get_current_system_message,
)


def make_skill(config, *, disabled=False, body="INSTRUCTIONS_ONLY", extra=""):
    path = config.cwd / "skills" / "review-python" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: review-python\ndescription: Review Python modules\ndisable-model-invocation: {str(disabled).lower()}\n{extra}---\n\n{body}\n"
    )
    settings = SkillSettings(enabled=True, include_user=False, paths=(path.parent.parent,))
    return path, replace(config, skills=settings)


def answer(text="done"):
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop")


async def test_catalog_body_expansion_and_host_manifest_exclusion(config):
    path, config = make_skill(config)
    provider = FauxProvider([answer()])
    session = create_agent_session(provider, config=config)
    try:
        assert "review-python" in session.system_prompt and "INSTRUCTIONS_ONLY" not in session.system_prompt
        assert "resource_manifest" not in session.system_prompt
        await session.prompt("/skill:review-python inspect files")
        user = next(message for message in session.messages if isinstance(message, UserMessage))
        text = content_text(user.content)
        assert "INSTRUCTIONS_ONLY" in text and text.endswith("inspect files")
        assert f"References are relative to {path.parent}" in text
        system = get_current_system_message(provider.requests[0].messages)
        assert system.resource_manifest and "INSTRUCTIONS_ONLY" not in system.resource_manifest
    finally:
        session.dispose()


async def test_disabled_model_invocation_remains_explicitly_available_without_reader(config):
    _, config = make_skill(config, disabled=True)
    session = create_agent_session(FauxProvider([answer()]), config=config, tool_names=[])
    try:
        assert "<skills>" not in session.system_prompt and len(session.skills) == 1
        await session.prompt("/skill:review-python")
        assert any(
            "INSTRUCTIONS_ONLY" in content_text(message.content)
            for message in session.messages
            if isinstance(message, UserMessage)
        )
    finally:
        session.dispose()


async def test_unknown_or_changed_skill_does_not_request_model_or_append_user(config):
    path, config = make_skill(config)
    provider = FauxProvider([])
    session = create_agent_session(provider, config=config)
    try:
        before = session.store.revision
        with pytest.raises(ValueError, match="Unknown skill"):
            await session.prompt("/skill:missing")
        path.write_text("changed")
        with pytest.raises(ValueError, match="changed"):
            await session.prompt("/skill:review-python")
        assert session.store.revision == before and session.is_idle and not provider.requests
    finally:
        session.dispose()


async def test_automatic_read_uses_permissions_and_allowed_tools_grants_nothing(config):
    path, config = make_skill(config, extra="allowed-tools: bash\n")
    responses = [
        AssistantMessage(
            content=[ToolCallBlock(id="read", name="read", arguments={"path": str(path)})],
            stop_reason="toolUse",
        ),
        AssistantMessage(
            content=[ToolCallBlock(id="bash", name="bash", arguments={"command": "touch forbidden"})],
            stop_reason="toolUse",
        ),
        answer(),
    ]
    session = create_agent_session(
        FauxProvider(responses), config=config, permission_policy=PermissionPolicy(read_only=True)
    )
    try:
        await session.prompt("Review Python code")
        results = [message for message in session.messages if isinstance(message, ToolResultMessage)]
        assert not results[0].is_error and "INSTRUCTIONS_ONLY" in content_text(results[0].content)
        assert results[1].is_error and "Read-only" in content_text(results[1].content)
        assert not (config.cwd / "forbidden").exists()
    finally:
        session.dispose()


@pytest.mark.parametrize("queue", ["steer", "follow_up"])
async def test_queued_skill_expands_at_admission_and_not_again_at_dequeue(config, queue):
    path, config = make_skill(config)

    class PausingProvider(FauxProvider):
        started, release = asyncio.Event(), asyncio.Event()

        async def stream(self, request, options):
            if not self.requests:
                self.started.set()
                await self.release.wait()
            async for event in super().stream(request, options):
                yield event

    provider = PausingProvider([answer(), answer()])
    session = create_agent_session(provider, config=config)
    task = asyncio.create_task(session.prompt("initial"))
    try:
        await asyncio.wait_for(provider.started.wait(), 1)
        getattr(session, queue)("/skill:review-python /skill:unknown")
        path.write_text("modified after admission")
        provider.release.set()
        await task
        assert len(provider.requests) == 2
        users = [
            content_text(message.content)
            for message in provider.requests[-1].messages
            if isinstance(message, UserMessage)
        ]
        assert "INSTRUCTIONS_ONLY" in users[-1] and users[-1].endswith("/skill:unknown")
    finally:
        provider.release.set()
        await task
        session.dispose()


async def test_resume_keeps_catalog_and_expanded_body_and_requires_reload(config):
    path, config = make_skill(config)
    session = create_agent_session(FauxProvider([answer()]), config=config, session="new")
    await session.prompt("/skill:review-python")
    journal = session.session_file
    original_prompt = session.system_prompt
    session.dispose()
    path.write_text(
        path.read_text()
        .replace("INSTRUCTIONS_ONLY", "NEW_BODY")
        .replace("Review Python modules", "New description")
    )
    restored = create_agent_session(FauxProvider([answer()]), session=journal)
    try:
        assert restored.system_prompt == original_prompt
        assert restored.skills[0].description == "Review Python modules"
        assert restored.resource_diagnostics[0].code == "changed"
        assert any(
            "INSTRUCTIONS_ONLY" in content_text(message.content)
            for message in restored.messages
            if isinstance(message, UserMessage)
        )
        before = restored.store.entry_count
        restored.reload_resources()
        assert restored.store.entry_count == before + 1
        update = restored.store.last_entry.message
        assert update.resource_manifest and update.sections["skills"]
        assert "New description" in restored.system_prompt and restored.config.skills.enabled
        await restored.prompt("/skill:review-python")
        assert "NEW_BODY" in content_text(
            [message for message in restored.messages if isinstance(message, UserMessage)][-1].content
        )
    finally:
        restored.dispose()
    again = create_agent_session(FauxProvider([]), session=journal)
    try:
        assert again.skills[0].description == "New description"
    finally:
        again.dispose()


async def test_missing_file_on_resume_preserves_snapshot_and_can_disable(config):
    path, config = make_skill(config)
    session = create_agent_session(FauxProvider([answer()]), config=config, session="new")
    await session.prompt("hello")
    journal = session.session_file
    session.dispose()
    path.unlink()
    restored = create_agent_session(FauxProvider([]), session=journal)
    try:
        assert len(restored.skills) == 1 and restored.resource_diagnostics[0].code == "missing"
        restored.reload_resources(replace(restored.config.skills, enabled=False))
        assert not restored.skills and "<skills>" not in restored.system_prompt
    finally:
        restored.dispose()
    again = create_agent_session(FauxProvider([]), session=journal)
    try:
        assert not again.config.skills.enabled and not again.skills
    finally:
        again.dispose()


async def test_reload_fsync_failure_does_not_publish_but_reopen_recovers_atomically(config, monkeypatch):
    path, config = make_skill(config)
    session = create_agent_session(FauxProvider([answer()]), config=config, session="new")
    await session.prompt("hello")
    journal = session.session_file
    old_skills, old_prompt = session.skills, session.system_prompt
    path.write_text(path.read_text().replace("Review Python modules", "New description"))

    def fail(descriptor):
        raise OSError("fsync failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr("pi_agent.session_store.os.fsync", fail)
            with pytest.raises(OSError, match="fsync failure"):
                session.reload_resources()
        assert session.skills == old_skills and session.system_prompt == old_prompt
        assert not session.store.usable
    finally:
        session.dispose()
    restored = create_agent_session(FauxProvider([]), session=journal)
    try:
        assert restored.skills[0].description == "New description"
        assert "New description" in restored.system_prompt
    finally:
        restored.dispose()


async def test_reload_busy_and_callback_reentry_reject(config):
    _, config = make_skill(config)
    errors = []
    session = create_agent_session(FauxProvider([answer()]), config=config)

    def listener(event):
        if event.type == "message_end":
            with pytest.raises(RuntimeError):
                session.reload_resources()
            errors.append(event.type)

    session.subscribe(listener)
    try:
        await session.prompt("hello")
        assert errors
    finally:
        session.dispose()


async def test_compaction_checkpoint_and_model_change_preserve_resources(config):
    _, config = make_skill(config)
    config.compaction.keep_recent_tokens = 10
    provider = FauxProvider([answer() for _ in range(3)] + [answer("## Goal\nContinue")])
    session = create_agent_session(provider, config=config, session="new")
    try:
        for _ in range(3):
            await session.prompt("/skill:review-python " + "x" * 400)
        await session.compact()
        snapshot = get_current_system_message(session.messages).resource_manifest
        assert snapshot and ResourceSnapshot.from_manifest(snapshot).skills == session.skills
        session.set_model(replace(session.model, id="second", name="Second"))
        journal = session.session_file
    finally:
        session.dispose()
    restored = create_agent_session(FauxProvider([]), session=journal)
    try:
        assert restored.skills == ResourceSnapshot.from_manifest(snapshot).skills
        assert restored.model.id == "second"
    finally:
        restored.dispose()


def test_offline_list_validate_does_not_load_env_or_require_key(config, monkeypatch, capsys):
    path, _ = make_skill(config)
    monkeypatch.setattr(cli, "read_environment", lambda *args: pytest.fail("environment read"))
    argv = ["skills", "list", "--json", "--no-user-skills", "--cwd", str(config.cwd), "--skill", str(path)]
    assert cli.main(argv) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["skills"][0]["name"] == "review-python" and "INSTRUCTIONS_ONLY" not in str(data)
    path.write_text(path.read_text().replace("name: review-python", "name: BAD_NAME"))
    assert cli.main([*argv[:1], "validate", *argv[2:]]) == 1
    assert json.loads(capsys.readouterr().out)["diagnostics"]


async def test_cli_run_and_explicit_resume_enable_disable(config, monkeypatch, capsys):
    path, _ = make_skill(config)
    providers = []

    def provider(settings):
        result = FauxProvider([answer()])
        providers.append(result)
        return result

    monkeypatch.setattr(
        cli, "read_environment", lambda path: {"OPENAI_API_KEY": "synthetic-only", "OPENAI_MODEL": "offline"}
    )
    monkeypatch.setattr(cli, "load_settings", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "_runtime_provider", provider)
    parser = cli._build_parser()
    argv = [
        "run",
        "/skill:review-python inspect",
        "--cwd",
        str(config.cwd),
        "--skill",
        str(path),
        "--no-user-skills",
        "--session-dir",
        str(config.session_dir),
    ]
    assert await cli._cmd_run(parser.parse_args(argv)) == 0
    assert "INSTRUCTIONS_ONLY" in content_text(
        [m for m in providers[0].requests[0].messages if isinstance(m, UserMessage)][-1].content
    )
    from pi_agent.session_store import SessionStore

    journal = SessionStore.list_sessions(config.session_dir, match_cwd=config.cwd)[0].path
    assert (
        await cli._cmd_run(parser.parse_args(["run", "hello", "--resume", str(journal), "--no-skills"])) == 0
    )
    restored = create_agent_session(FauxProvider([]), session=journal)
    try:
        assert not restored.skills and not restored.config.skills.enabled
    finally:
        restored.dispose()
    assert "synthetic-only" not in capsys.readouterr().out


@pytest.mark.parametrize("json_mode", [False, True])
async def test_chat_skill_command_list_reload_and_clean_json(config, monkeypatch, capsys, json_mode):
    _, config = make_skill(config)
    provider = FauxProvider([answer()])
    session = create_agent_session(provider, config=config)
    monkeypatch.setattr(
        cli, "_open_session", lambda args: (session, SimpleNamespace(model=session.model), False, False)
    )
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("/skills\n/skill:review-python inspect\n/wait\n/reload\n/quit\n")
    )
    args = cli._build_parser().parse_args(["chat", *(["--json"] if json_mode else [])])
    assert await run_chat(args) == 0
    captured = capsys.readouterr()
    assert len(provider.requests) == 1
    assert "INSTRUCTIONS_ONLY" in content_text(
        [message for message in provider.requests[0].messages if isinstance(message, UserMessage)][-1].content
    )
    if json_mode:
        assert all(json.loads(line)["type"] for line in captured.out.splitlines())
        assert "Resources reloaded" in captured.err
    else:
        assert "Resources reloaded" in captured.out


def test_skill_settings_merge_and_invalid_fields_are_strict(tmp_path):
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "settings.json").write_text(
        json.dumps({"skills": {"enabled": True, "include_user": False}})
    )
    project = tmp_path / ".pi-agent"
    project.mkdir()
    (project / "settings.json").write_text(json.dumps({"skills": {"paths": ["skills"]}}))
    settings = load_settings(tmp_path, agent_dir=global_dir)
    assert settings["skills"] == {"enabled": True, "include_user": False, "paths": ["skills"]}
    for invalid in ({"enabled": "yes"}, {"unknown": True}, {"paths": [1]}, {"prompt_max_bytes": True}):
        (project / "settings.json").write_text(json.dumps({"skills": invalid}))
        with pytest.raises(ValueError):
            load_settings(tmp_path, agent_dir=global_dir)


def test_conflicting_resource_config_on_resume_is_not_silently_applied(config):
    _, enabled = make_skill(config)
    session = create_agent_session(FauxProvider([]), config=enabled, session="new")
    session.store.append_message(UserMessage(content="persist"))
    journal = session.session_file
    session.dispose()
    before = journal.read_bytes()
    with pytest.raises(ValueError, match="reload_resources"):
        create_agent_session(FauxProvider([]), session=journal, config=config)
    assert journal.read_bytes() == before


async def test_reload_manifest_survives_later_configuration_records(config):
    _, enabled = make_skill(config)
    session = create_agent_session(FauxProvider([answer()]), config=config, session="new")
    await session.prompt("hello")
    session.reload_resources(enabled.skills)
    session.set_tool_execution("parallel")
    journal = session.session_file
    session.dispose()
    restored = create_agent_session(FauxProvider([]), session=journal)
    try:
        assert restored.config.skills.enabled and restored.config.tool_execution == "parallel"
        assert len(restored.skills) == 1
    finally:
        restored.dispose()
