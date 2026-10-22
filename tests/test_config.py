from dataclasses import asdict

import pytest

from pi_agent import CompactionSettings, RetrySettings, create_agent_session
from pi_agent.providers import FauxProvider
from pi_agent.types import AssistantMessage, TextBlock


@pytest.mark.parametrize(
    ("settings_type", "field"),
    [
        (CompactionSettings, "reserve_tokens"),
        (CompactionSettings, "keep_recent_tokens"),
        (RetrySettings, "max_retries"),
        (RetrySettings, "base_delay_ms"),
        (RetrySettings, "max_delay_ms"),
    ],
)
@pytest.mark.parametrize("value", [True, False, 1.5, float("nan"), float("inf"), "3", None, -1])
def test_numeric_settings_reject_invalid_values_in_python_and_json(settings_type, field, value):
    settings = settings_type()
    setattr(settings, field, value)
    with pytest.raises(ValueError):
        settings.validate()
    with pytest.raises(ValueError):
        settings_type.from_json(asdict(settings))


@pytest.mark.parametrize("settings_type", [CompactionSettings, RetrySettings])
@pytest.mark.parametrize("value", [0, 1, "false", None, float("nan")])
def test_enabled_requires_boolean_in_python_and_json(settings_type, value):
    settings = settings_type(enabled=value)
    with pytest.raises(ValueError):
        settings.validate()
    with pytest.raises(ValueError):
        settings_type.from_json(asdict(settings))


def test_invalid_sdk_settings_are_rejected_before_persisting_configuration(config):
    config.retry.max_retries = 1.5
    with pytest.raises(ValueError):
        create_agent_session(FauxProvider([]), config=config, session="new")
    assert all(not path.read_bytes() for path in config.session_dir.glob("*.jsonl"))


@pytest.mark.parametrize("enabled", [False, True])
async def test_valid_sdk_settings_survive_session_resume(config, enabled):
    config.compaction = CompactionSettings(enabled=enabled, reserve_tokens=0, keep_recent_tokens=0)
    config.retry = RetrySettings(enabled=enabled, max_retries=0, base_delay_ms=1, max_delay_ms=1)
    session = create_agent_session(
        FauxProvider([AssistantMessage(content=[TextBlock(text="done")], stop_reason="stop")]),
        config=config,
        session="new",
    )
    try:
        await session.prompt("work")
        path = session.session_file
        assert path is not None
    finally:
        session.dispose()
    resumed = create_agent_session(FauxProvider([]), session=path)
    try:
        assert resumed.config.retry == config.retry
        assert resumed.config.compaction == config.compaction
    finally:
        resumed.dispose()
