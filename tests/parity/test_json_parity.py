"""Tool-call argument parsing must match pi's parseToolCallArguments, error text included."""

from __future__ import annotations

import json
from pathlib import Path

from pi_agent._js import json_stringify
from pi_agent.providers._json_parse import parse_streaming_json, parse_tool_call_arguments

HERE = Path(__file__).parent
CASES = json.loads((HERE / "json_cases.json").read_text(encoding="utf-8"))
EXPECTED = json.loads((HERE / "fixtures/pi_json_results.json").read_text(encoding="utf-8"))


def test_arguments_and_errors_match_pi() -> None:
    mismatches = []
    for text, expected in zip(CASES, EXPECTED, strict=True):
        arguments, error = parse_tool_call_arguments(text)
        actual = {"arguments": arguments, **({"argumentsError": error} if error is not None else {})}
        if actual != expected:
            mismatches.append((text, actual, expected))
    assert not mismatches, mismatches[:5]


def test_partial_arguments_match_pi_at_every_prefix() -> None:
    texts = json.loads((HERE / "streaming_json_cases.json").read_text("utf-8"))
    expected = json.loads((HERE / "fixtures/pi_streaming_json_results.json").read_text("utf-8"))
    for text, want in zip(texts, expected, strict=True):
        # JSON.stringify turns partial-json's non-finite values into null, as in Pi's runner.
        assert json.loads(json_stringify(parse_streaming_json(text))) == want, repr(text)
