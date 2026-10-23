"""Tool-call argument parsing must match pi's parseToolCallArguments, error text included."""

from __future__ import annotations

import json
from pathlib import Path

from pi_agent.providers._json_parse import parse_tool_call_arguments

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
