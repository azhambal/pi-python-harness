"""Compaction must match pi's prepareCompaction/compact (see fixtures/pi_version.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_agent._session_projection import build_projection, estimate_projected_context_tokens
from pi_agent.cancellation import Cancellation
from pi_agent.compaction import prepare_compaction, run_compaction
from pi_agent.config import CompactionSettings
from pi_agent.events import StreamDone
from pi_agent.session_store import SessionStore
from pi_agent.types import AssistantMessage, ModelInfo, TextBlock, Usage, message_from_json

HERE = Path(__file__).parent
CASES = json.loads((HERE / "compaction_cases.json").read_text(encoding="utf-8"))
EXPECTED = {
    item["name"]: item
    for item in json.loads((HERE / "fixtures/pi_compaction_results.json").read_text("utf-8"))
}

_FIELDS = {
    "toolCallId": "tool_call_id",
    "toolName": "tool_name",
    "isError": "is_error",
    "stopReason": "stop_reason",
    "mimeType": "mime_type",
    "thinkingSignature": "thinking_signature",
    "cacheRead": "cache_read",
    "cacheWrite": "cache_write",
    "totalTokens": "total_tokens",
}


def _rename(value: object) -> object:
    if isinstance(value, dict):
        return {_FIELDS.get(key, key): _rename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item) for item in value]
    return value


class RecordingProvider:
    """Answers every summarization request with ``summary <n>``, like the pi runner."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def stream(self, request, options):
        self.calls.append((request, options))
        count = len(self.calls)
        message = AssistantMessage(
            content=[TextBlock(text=f"summary {count}")],
            stop_reason="stop",
            usage=Usage(input=count, total_tokens=count),
        )
        yield StreamDone(reason="stop", message=message)


def _store(case: dict) -> tuple[SessionStore, dict[str, str]]:
    store = SessionStore.in_memory(".")
    pi_ids: dict[str, str] = {}
    ids: list[str] = []
    for index, item in enumerate(case["entries"]):
        if "message" in item:
            entry_id = store.append_message(message_from_json(_rename(item["message"])))
        elif "omit" in item:
            entry_id = store.append_context_omission(ids[item["omit"]])
        else:
            spec = item["compaction"]
            details = spec.get("details") or {}
            entry_id = store.append_compaction(
                spec["summary"],
                ids[spec["firstKept"]],
                spec["tokensBefore"],
                read_files=details.get("readFiles", []),
                modified_files=details.get("modifiedFiles", []),
            )
        ids.append(entry_id)
        pi_ids[entry_id] = f"e{index}"
    return store, pi_ids


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
async def test_compaction_matches_pi(case: dict) -> None:
    expected = EXPECTED[case["name"]]
    store, pi_ids = _store(case)
    try:
        projection = build_projection(store)
        settings = CompactionSettings(keep_recent_tokens=case["keepRecentTokens"])
        preparation = prepare_compaction(
            projection, tokens_before=estimate_projected_context_tokens(store), settings=settings
        )
        if expected["preparation"] is None:
            assert preparation is None
            return
        assert preparation is not None
        assert {
            "firstKeptEntryId": pi_ids[preparation.first_kept_entry_id],
            "isSplitTurn": preparation.is_split_turn,
            "messagesToSummarize": len(preparation.messages_to_summarize),
            "turnPrefixMessages": len(preparation.turn_prefix_messages),
            "tokensBefore": preparation.tokens_before,
            "previousSummary": preparation.previous_summary,
        } == expected["preparation"]

        provider = RecordingProvider()
        model = ModelInfo(
            "test", "test", "anthropic", api="anthropic-messages", reasoning=case.get("reasoning", False)
        )
        result = await run_compaction(
            preparation,
            provider=provider,
            model=model,
            cancellation=Cancellation(),
            thinking_level=case.get("thinkingLevel", "off"),
            custom_instructions=case.get("customInstructions"),
        )
        requests = [
            {
                "system": request.messages[0].content,
                "roles": [message.role for message in request.messages],
                "prompt": "".join(block.text for block in request.messages[1].content),
                "reasoning": None if options.thinking_level == "off" else options.thinking_level,
                "cacheRetention": options.cache_retention,
                "maxTokens": options.max_tokens,
                "hasSessionId": bool(options.session_id),
            }
            for request, options in provider.calls
        ]
        assert requests == expected["requests"]
        assert result.summary == expected["summary"]
        assert {"readFiles": result.read_files, "modifiedFiles": result.modified_files} == expected["details"]
        assert result.usage is not None and result.usage.total_tokens == expected["usageTotal"]
    finally:
        store.close()
