"""Model-facing tool declarations must match pi's (see fixtures/pi_version.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_agent.tools import TOOL_NAMES, create_builtin_tools

HERE = Path(__file__).parent
DECLARATIONS = {
    item["name"]: item
    for item in json.loads((HERE / "fixtures/pi_declarations.json").read_text("utf-8"))["declarations"]
}


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_declaration_matches_pi(name: str, tmp_path: Path) -> None:
    (tool,) = create_builtin_tools(tmp_path, [name])
    expected = DECLARATIONS[name]
    assert {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "promptSnippet": tool.prompt_snippet,
        "promptGuidelines": list(tool.guidelines),
    } == expected
