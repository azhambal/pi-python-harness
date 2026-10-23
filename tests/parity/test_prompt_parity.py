"""System prompt sections must match pi's (see fixtures/pi_version.json).

pi's <docs> section points at pi's own documentation, which the port does not
ship, so it is the one section left out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_agent.resources import ResourceSnapshot
from pi_agent.skills import SkillInfo, SkillSettings
from pi_agent.system_prompt import ContextFile, SystemPromptInput, build_system_prompt_sections
from pi_agent.tools import create_builtin_tools

HERE = Path(__file__).parent
PROMPTS = json.loads((HERE / "fixtures/pi_declarations.json").read_text("utf-8"))["prompts"]
SELECTIONS = {
    "default": ["read", "bash", "edit", "write"],
    "all": ["read", "bash", "edit", "write", "grep", "find", "ls"],
    "readonly": ["read", "grep", "find", "ls"],
}


@pytest.mark.parametrize("name", sorted(SELECTIONS))
def test_prompt_sections_match_pi(name: str) -> None:
    tools = create_builtin_tools(Path("/work"), SELECTIONS[name])
    sections = build_system_prompt_sections(SystemPromptInput(cwd=Path("/work"), tools=tools))
    expected = {key: value for key, value in PROMPTS[name].items() if key != "docs"}
    assert list(sections) == list(expected)
    assert sections == expected


def _skill(name: str, description: str, disabled: bool = False) -> SkillInfo:
    return SkillInfo(
        name=name,
        description=description,
        file_path=f"/skills/{name}/SKILL.md",
        base_dir=f"/skills/{name}",
        source="explicit",
        digest="0" * 64,
        disable_model_invocation=disabled,
    )


SKILLS = ResourceSnapshot(
    SkillSettings(enabled=True),
    (_skill("demo", "Use <tags> & 'quotes' \"here\""), _skill("hidden", "Never listed", True)),
)


@pytest.mark.parametrize(
    "name,selection,resources",
    [("resources", SELECTIONS["default"], True), ("bashSkills", ["bash"], False)],
)
def test_prompt_with_context_files_addendum_and_skills_matches_pi(
    name: str, selection: list[str], resources: bool
) -> None:
    tools = create_builtin_tools(Path("/work"), selection)
    prompt_input = SystemPromptInput(
        cwd=Path("/work"), tools=tools, skills_section=SKILLS.prompt_section(set(selection))
    )
    if resources:
        prompt_input.append_prompt = "Extra instructions."
        prompt_input.context_files = [
            ContextFile(Path("/home/u/.pi/agent/AGENTS.md"), "Global rules."),
            ContextFile(Path("/work/AGENTS.md"), "Project rules."),
        ]
    sections = build_system_prompt_sections(prompt_input)
    expected = {key: value for key, value in PROMPTS[name].items() if key != "docs"}
    assert list(sections) == list(expected)
    assert sections == expected
