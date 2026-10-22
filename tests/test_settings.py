import json

import httpx
import pytest

from pi_agent import cli
from pi_agent.providers import OpenAICompatProvider
from pi_agent.runtime_config import _ENV_NAMES
from pi_agent.settings import config_from_settings, load_settings, settings_environment


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_settings_order_nested_merge_and_policy_replacement(tmp_path, model):
    global_dir = tmp_path / "global"
    project = tmp_path / "project"
    explicit = tmp_path / "explicit.json"
    save(
        global_dir / "settings.json",
        {
            "provider": "deepseek",
            "model": "global",
            "retry": {"max_retries": 1},
            "permissions": {"default": "deny", "rules": [{"tool": "read", "action": "allow"}]},
        },
    )
    save(
        project / ".pi-agent/settings.json",
        {
            "model": "project",
            "retry": {"base_delay_ms": 3},
            "tool_execution": "parallel",
            "max_parallel_tools": 2,
        },
    )
    save(explicit, {"model": "explicit", "permissions": {"read_only": True}})
    settings = load_settings(project, agent_dir=global_dir, explicit=explicit)
    assert settings["model"] == "explicit" and settings["retry"] == {"max_retries": 1, "base_delay_ms": 3}
    config = config_from_settings(project, model, settings)
    config.validate()
    assert config.permissions.read_only and config.permissions.rules == ()
    assert config.tool_execution == "parallel" and config.max_parallel_tools == 2
    assert not any("KEY" in name for name in settings_environment(settings))


@pytest.mark.parametrize(
    "source",
    [
        '{"OPENAI_API_KEY":"synthetic-secret"}',
        '{"unknown":true}',
        '{"model":"one","model":"two"}',
        '{"retry":{"typo":1}}',
        '{"compaction":{"enabled":"false"}}',
        '{"http_timeout":NaN}',
        '{"max_parallel_tools":true}',
        '{"tool_execution":"invalid"}',
        '{"thinking_level":"invalid"}',
        '{"retry":{"base_delay_ms":2,"max_delay_ms":1}}',
        '{"permissions":{"default":"invalid"}}',
    ],
)
def test_invalid_or_secret_settings_reject_without_printing_values(tmp_path, source):
    explicit = tmp_path / "settings.json"
    explicit.write_text(source)
    with pytest.raises(ValueError) as error:
        load_settings(tmp_path, agent_dir=tmp_path / "global", explicit=explicit)
    assert "synthetic-secret" not in str(error.value)


def test_missing_explicit_file_is_error_even_at_automatic_path(tmp_path):
    path = tmp_path / ".pi-agent/settings.json"
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path, agent_dir=tmp_path / "global", explicit=path)


def test_provider_specific_settings_do_not_leak_old_endpoint_into_new_transport():
    environment = settings_environment(
        {"provider": "deepseek", "model": "first", "base_url": "https://unit.test"}
    )
    assert "ANTHROPIC_MODEL" not in environment and "ANTHROPIC_BASE_URL" not in environment


async def test_cli_file_defaults_environment_cli_precedence_and_resume(tmp_path, monkeypatch, capsys):
    for key in _ENV_NAMES:
        monkeypatch.delenv(key, raising=False)
    environment = tmp_path / "fixture.env"
    environment.write_text(
        "OPENAI_API_KEY=synthetic-only\nOPENAI_MODEL=env-model\nOPENAI_BASE_URL=https://unit.test\n"
    )
    save(
        tmp_path / ".pi-agent/settings.json",
        {
            "provider": "deepseek",
            "model": "file-model",
            "thinking_level": "high",
            "tool_execution": "parallel",
            "max_parallel_tools": 2,
            "permissions": {"read_only": True},
            "compaction": {"enabled": False},
        },
    )
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda cwd, explicit=None: load_settings(cwd, explicit=explicit, agent_dir=tmp_path / "global"),
    )
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        chunk = {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    argv = [
        "run",
        "hello",
        "--cwd",
        str(tmp_path),
        "--env-file",
        str(environment),
        "--session-dir",
        str(tmp_path / "sessions"),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            cli, "OpenAICompatProvider", lambda **kwargs: OpenAICompatProvider(client=client, **kwargs)
        )
        parser = cli._build_parser()
        assert await cli._cmd_run(parser.parse_args(argv + ["--model", "cli-model"])) == 0
        assert payloads[-1]["model"] == "cli-model" and payloads[-1]["thinking"] == {"type": "enabled"}
        save(
            tmp_path / ".pi-agent/settings.json",
            {"model": "changed", "thinking_level": "off", "tool_execution": "sequential"},
        )
        assert await cli._cmd_run(parser.parse_args(argv + ["--continue"])) == 0
        assert payloads[-1]["model"] == "cli-model" and payloads[-1]["thinking"] == {"type": "enabled"}
        session, _, _, _ = cli._open_session(parser.parse_args(argv + ["--continue"]))
        try:
            assert session.config.tool_execution == "parallel" and session.config.permissions.read_only
        finally:
            session.dispose()
    assert "synthetic-only" not in capsys.readouterr().out
