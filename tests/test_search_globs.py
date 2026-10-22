from pathlib import Path

import pytest

from pi_agent.tools import FindTool, GrepTool
from pi_agent.tools._search import matches_glob


async def test_glob_filters_cannot_reinclude_gitignored_files(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "ignored.log").write_text("needle")
    found = await FindTool(tmp_path).execute("c", {"pattern": "*.log"})
    assert found.structured_content["items"] == []
    matches = await GrepTool(tmp_path).execute("c", {"pattern": "needle", "glob": "*.log"})
    assert matches.structured_content["items"] == []


@pytest.mark.parametrize(
    "path,pattern,matched",
    [
        ("src/a.py", "src/**/*.py", True),
        ("src/nested/a.py", "src/**/*.py", True),
        ("src/nested/a.py", "src/*.py", False),
        ("src/a.py", "*.py", True),
        ("a.py", "**/*.py", True),
        ("src/a.txt", "!*.py", True),
        ("src/a.py", "!*.py", False),
        ("src/a.py", "src/[ab].py", True),
        ("src/z.py", "src/[!ab].py", True),
    ],
)
def test_glob_contract(path, pattern, matched):
    assert matches_glob(path, pattern) is matched


async def test_nested_repository_boundary_stops_parent_ignore_rules(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("*.py\n")
    (tmp_path / "nested" / ".git").mkdir(parents=True)
    (tmp_path / "nested" / ".gitignore").write_text("local.py\n")
    (tmp_path / "nested" / "kept.py").write_text("needle")
    (tmp_path / "nested" / "local.py").write_text("needle")
    result = await FindTool(tmp_path).execute("c", {"path": "nested", "pattern": "*.py"})
    assert result.structured_content["items"] == [{"path": "kept.py"}]
