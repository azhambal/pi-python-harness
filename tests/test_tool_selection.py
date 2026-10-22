import json

import pytest

from pi_agent import cli
from pi_agent.providers import FauxProvider
from pi_agent.runtime_config import _ENV_NAMES
from pi_agent.settings import load_settings
from pi_agent.tools import resolve_builtin_selection
from pi_agent.types import AssistantMessage, TextBlock


@pytest.mark.parametrize(
    "selection,base,expected",
    [
        (("+grep", "-write"), ("read", "write"), ("read", "grep")),
        (("+read",), ("read",), ("read",)),
        (("-write",), (), ()),
        ((), ("read",), ()),
        (("write",), ("read",), ("write",)),
    ],
)
def test_relative_or_absolute_selection(selection, base, expected):
    assert resolve_builtin_selection(selection, base) == expected


@pytest.mark.parametrize("value", ["read,+grep", "+read,write", "+unknown", "+", "+read,-read", "read,read"])
def test_invalid_cli_selection_is_rejected_during_argument_parsing(value):
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["run", "task", f"--tools={value}"])


async def test_relative_cli_selection_uses_recorded_set_on_resume(tmp_path, monkeypatch):
    for key in _ENV_NAMES:
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / "fixture.env"
    env.write_text("OPENAI_API_KEY=synthetic\nOPENAI_MODEL=fixture\n")
    defaults = tmp_path / "settings.json"
    defaults.write_text(json.dumps({"tools": ["read", "write"], "skills": {"enabled": False}}))
    monkeypatch.setattr(
        cli,
        "_runtime_provider",
        lambda settings: FauxProvider(
            [AssistantMessage(stop_reason="stop", content=[TextBlock(text="done")])]
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda cwd, explicit=None: load_settings(cwd, explicit=explicit, agent_dir=tmp_path / "absent"),
    )
    parser = cli._build_parser()
    argv = [
        "run",
        "task",
        "--cwd",
        str(tmp_path),
        "--env-file",
        str(env),
        "--settings",
        str(defaults),
        "--session-dir",
        str(tmp_path / "sessions"),
    ]
    session, _, _, _ = cli._open_session(parser.parse_args(argv + ["--tools=+grep,-write"]))
    try:
        assert [tool.name for tool in session.agent.tools] == ["read", "grep"]
        await session.prompt("seed")
        path = session.session_file
    finally:
        session.dispose()
    defaults.write_text(json.dumps({"tools": ["bash", "write"]}))
    resumed, _, _, _ = cli._open_session(
        parser.parse_args(argv + ["--resume", str(path), "--tools=+ls,-read"])
    )
    try:
        assert [tool.name for tool in resumed.agent.tools] == ["grep", "ls"]
        await resumed.prompt("continue")
    finally:
        resumed.dispose()
    again, _, _, _ = cli._open_session(parser.parse_args(argv + ["--resume", str(path)]))
    try:
        assert [tool.name for tool in again.agent.tools] == ["grep", "ls"]
    finally:
        again.dispose()


async def test_relative_selection_preserves_explicit_empty_resumed_set(tmp_path, monkeypatch):
    for key in _ENV_NAMES:
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / "fixture.env"
    env.write_text("OPENAI_API_KEY=synthetic\nOPENAI_MODEL=fixture\n")
    monkeypatch.setattr(
        cli,
        "_runtime_provider",
        lambda settings: FauxProvider(
            [AssistantMessage(stop_reason="stop", content=[TextBlock(text="done")])]
        ),
    )
    monkeypatch.setattr(cli, "load_settings", lambda cwd, explicit=None: {})
    parser = cli._build_parser()
    argv = [
        "run",
        "task",
        "--cwd",
        str(tmp_path),
        "--env-file",
        str(env),
        "--session-dir",
        str(tmp_path / "sessions"),
    ]
    session, _, _, _ = cli._open_session(parser.parse_args(argv + ["--tools="]))
    try:
        await session.prompt("seed")
        path = session.session_file
    finally:
        session.dispose()
    resumed, _, _, _ = cli._open_session(parser.parse_args(argv + ["--resume", str(path), "--tools=+grep"]))
    try:
        assert [tool.name for tool in resumed.agent.tools] == ["grep"]
    finally:
        resumed.dispose()
