"""Differential parity: built-in tools must answer exactly like pi.

The fixtures hold what pi (see fixtures/pi_version.json) returned for every
case in tool_cases.json when run through its own agent-loop tool path. They
are regenerated with scripts/pi_reference/generate.sh.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from pi_agent.agent_loop import AgentLoopConfig, _run_tool_call
from pi_agent.cancellation import Cancellation
from pi_agent.tools import create_builtin_tools
from pi_agent.types import AssistantMessage, ImageBlock, TextBlock, ToolCallBlock

HERE = Path(__file__).parent
CASES = json.loads((HERE / "tool_cases.json").read_text(encoding="utf-8"))
EXPECTED = {
    item["name"]: item for item in json.loads((HERE / "fixtures/pi_tool_results.json").read_text("utf-8"))
}

#: Tools whose output order follows ripgrep/fd's parallel directory walk.
UNORDERED_TOOLS = {"grep", "find"}
_SPILL = re.compile(r"/[^\s\]]*/pi-bash-[0-9a-f]+\.log")


def _needs(tool: str) -> str | None:
    if tool == "grep" and shutil.which("rg") is None:
        return "rg"
    if tool == "find" and shutil.which("fd") is None and shutil.which("fdfind") is None:
        return "fd"
    return None


def _setup(root: Path, case: dict) -> None:
    for directory in case.get("dirs", []):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name, content in case.get("files", {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content.encode("utf-8"))
    for name, data in case.get("binaryFiles", {}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(base64.b64decode(data))
    for name, target in case.get("symlinks", {}).items():
        os.symlink(target, root / name)
    if case.get("gitInit"):
        (root / ".git").mkdir(exist_ok=True)


def _normalize(case: dict, item: dict) -> dict:
    content = []
    for block in item["content"]:
        if block["type"] != "text":
            content.append(block)
            continue
        text = _SPILL.sub("<SPILL>", block["text"])
        if case["tool"] in UNORDERED_TOOLS and not item["isError"]:
            body, sep, notice = text.partition("\n\n[")
            rows = sorted(body.split("\n"))
            # Which rows fill a limit depends on the parallel walk; only their count is stable.
            text = (f"<{len(rows)} rows>" if "limit reached" in notice else "\n".join(rows)) + sep + notice
        content.append({"type": "text", "text": text})
    return {"isError": item["isError"], "content": content, "files": item.get("files", {})}


async def _run_port(case: dict) -> dict:
    root = Path(tempfile.mkdtemp(prefix="pi-parity-")).resolve()
    try:
        _setup(root, case)
        tools = create_builtin_tools(root, expose_session_environment=False)
        call = ToolCallBlock(id="call_1", name=case["tool"], arguments=case["args"])
        tool = next((candidate for candidate in tools if candidate.name == call.name), None)
        config = AgentLoopConfig(tools=tools, cwd=root)
        outcome = await _run_tool_call(
            config, tool, call, AssistantMessage(content=[call]), [], _noop_emit, Cancellation()
        )
        content: list[dict] = []
        for block in outcome.result.content:
            if isinstance(block, TextBlock):
                content.append({"type": "text", "text": block.text.replace(str(root), "<ROOT>")})
            elif isinstance(block, ImageBlock):
                content.append({"type": "image", "mimeType": block.mime_type})
        files = {}
        for name in case.get("readBack", []):
            try:
                files[name] = (root / name).read_bytes().decode("utf-8", errors="replace")
            except OSError:
                files[name] = None
        for tool in tools:
            close = getattr(tool, "close", None)
            if close is not None:
                close()
        return {"isError": outcome.result.is_error, "content": content, "files": files}
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def _noop_emit(_event: object) -> None:
    return None


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
async def test_tool_matches_pi(case: dict) -> None:
    missing = _needs(case["tool"])
    if missing:
        pytest.skip(f"{missing} is not installed")
    expected = _normalize(case, EXPECTED[case["name"]])
    actual = _normalize(case, await _run_port(case))
    assert actual == expected
