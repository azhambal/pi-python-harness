"""Pi's complete image payload, hints and model-aware read output, byte for byte."""

import asyncio
import base64
import json
from pathlib import Path

import pytest

from pi_agent.agent_loop import AgentLoopConfig, _run_tool_call
from pi_agent.cancellation import Cancellation
from pi_agent.models import find_model
from pi_agent.tools import ReadTool
from pi_agent.tools._context import current_model
from pi_agent.tools._images import process_image
from pi_agent.types import AssistantMessage, ModelInfo, ToolCallBlock

HERE = Path(__file__).parent
CASES = json.loads((HERE / "image_cases.json").read_text("utf-8"))
EXPECTED = {c["name"]: c for c in json.loads((HERE / "fixtures/pi_image_results.json").read_text("utf-8"))}


async def _emit(event):
    pass


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
async def test_image_bytes_and_read_context_match_pi(case, tmp_path):
    data = base64.b64decode(case["data"])
    options = case.get("options", {})
    result = process_image(
        data,
        case["mimeType"],
        auto_resize=options.get("autoResizeImages", True),
        resize_options=options.get("resizeOptions"),
    )
    actual = (
        {"ok": True, "data": result.data, "mimeType": result.mime_type, "hints": result.hints}
        if result.ok
        else {"ok": False, "message": result.message}
    )
    assert actual == EXPECTED[case["name"]]["result"]
    (tmp_path / "image").write_bytes(data)
    model = ModelInfo(
        "image-model",
        "Images",
        "test",
        input=("text", "image"),
        input_limits={"images": {"resize": options.get("resizeOptions", {})}},
    )
    tool = ReadTool(tmp_path, auto_resize_images=options.get("autoResizeImages", True))
    call = ToolCallBlock(id="c", name="read", arguments={"path": "image"})
    outcome = await _run_tool_call(
        AgentLoopConfig(model=model, tools=[tool], cwd=tmp_path),
        tool,
        call,
        AssistantMessage(content=[call]),
        [],
        _emit,
        Cancellation(),
    )
    assert not outcome.result.is_error
    blocks = []
    for block in outcome.result.content:
        item = block.to_json()
        if "mime_type" in item:
            item["mimeType"] = item.pop("mime_type")
        blocks.append(item)
    assert blocks == EXPECTED[case["name"]]["readContent"]
    assert current_model.get() is None  # The per-call context must not leak.


async def test_shared_read_tool_uses_each_concurrent_models_profile(tmp_path):
    case = next(c for c in CASES if c["name"] == "PNG model profile")
    (tmp_path / "image").write_bytes(base64.b64decode(case["data"]))
    tool = ReadTool(tmp_path)

    async def read(width, vision):
        model = ModelInfo(
            str(width),
            "m",
            "test",
            input=("text", "image") if vision else ("text",),
            input_limits={"images": {"resize": {"maxWidth": width, "maxHeight": 6}}},
        )
        call = ToolCallBlock(id=str(width), name="read", arguments={"path": "image"})
        outcome = await _run_tool_call(
            AgentLoopConfig(model=model, tools=[tool]),
            tool,
            call,
            AssistantMessage(content=[call]),
            [],
            _emit,
            Cancellation(),
        )
        return outcome.result.content[0].text

    first, second = await asyncio.gather(read(9, True), read(4, False))
    assert "displayed at 9x5" in first and "does not support images" not in first
    assert "displayed at 4x2" in second and "does not support images" in second
    assert current_model.get() is None


def test_catalog_image_limits_survive_journal_serialization():
    model = find_model("anthropic", "claude-opus-4-8")
    assert model is not None and model.input_limits["images"]["resize"]["maxWidth"] == 2000
    original = model.to_json()
    restored = ModelInfo.from_json(original)
    restored.input_limits["images"]["resize"]["maxWidth"] = 17
    assert model.input_limits["images"]["resize"]["maxWidth"] == 2000
    assert original["input_limits"]["images"]["resize"]["maxWidth"] == 2000
