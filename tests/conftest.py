from pathlib import Path

import pytest

from pi_agent import AgentConfig, CompactionSettings, ModelInfo, RetrySettings


@pytest.fixture(autouse=True)
def isolated_user_skills(tmp_path, monkeypatch):
    # New CLI defaults must never discover the operator's actual installed skills in tests.
    monkeypatch.setattr("pi_agent.skills.default_user_roots", lambda agent_dir: [tmp_path / "user-skills"])


@pytest.fixture
def model() -> ModelInfo:
    return ModelInfo(id="offline", name="Offline", provider="faux", context_window=100_000)


@pytest.fixture
def config(tmp_path: Path, model: ModelInfo) -> AgentConfig:
    return AgentConfig(
        cwd=tmp_path,
        model=model,
        agent_dir=tmp_path / ".agent",
        session_dir=tmp_path / "sessions",
        instruction_files=[],
        compaction=CompactionSettings(enabled=False),
        retry=RetrySettings(enabled=False),
    )
