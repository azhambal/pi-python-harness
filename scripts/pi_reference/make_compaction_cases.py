"""Write tests/parity/compaction_cases.json (pi-shaped session entries).

Run: .venv/bin/python scripts/pi_reference/make_compaction_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "tests/parity/compaction_cases.json"


def usage(total: int = 0) -> dict:
    return {
        "input": total,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": total,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def user(text, timestamp=1) -> dict:
    return {"message": {"role": "user", "content": text, "timestamp": timestamp}}


def assistant(*blocks, stop="stop", total=0) -> dict:
    content = [{"type": "text", "text": block} if isinstance(block, str) else block for block in blocks]
    return {
        "message": {
            "role": "assistant",
            "content": content,
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "test",
            "usage": usage(total),
            "stopReason": stop,
            "timestamp": 2,
        }
    }


def call(identifier, name, **arguments) -> dict:
    return {"type": "toolCall", "id": identifier, "name": name, "arguments": arguments}


def result(identifier, name, text) -> dict:
    return {
        "message": {
            "role": "toolResult",
            "toolCallId": identifier,
            "toolName": name,
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "timestamp": 3,
        }
    }


def tool_turn(prefix: str) -> list[dict]:
    return [
        user(f"{prefix} single long tool turn"),
        assistant(call(f"{prefix}a", "read", path="src/app.py"), stop="toolUse"),
        result(f"{prefix}a", "read", "x" * 4000),
        assistant(
            "Writing the fix.",
            call(f"{prefix}b", "write", path="src/fix.py", content="y" * 1200),
            stop="toolUse",
        ),
        result(f"{prefix}b", "write", "written"),
        assistant(call(f"{prefix}c", "edit", path="src/app.py", edits=[]), stop="toolUse"),
        result(f"{prefix}c", "edit", "tail"),
        assistant("done"),
    ]


def exchanges(count: int, size: int = 400) -> list[dict]:
    entries = []
    for index in range(count):
        entries += [user(f"question {index} " + "q" * size), assistant(f"answer {index} " + "a" * size)]
    return entries


CASES = [
    {
        "name": "UTF16 file sort and truncation through a surrogate pair",
        "keepRecentTokens": 5,
        "entries": [
            user("inspect"),
            assistant(
                call("utf1", "read", path="😀.txt"), call("utf2", "read", path="\ue000.txt"), stop="toolUse"
            ),
            result("utf1", "read", "x" * 1999 + "😀tail"),
            result("utf2", "read", "ok"),
            assistant("inspected"),
            *exchanges(2, 80),
        ],
    },
    {
        "name": "multiple user and tool text blocks concatenate without separators",
        "keepRecentTokens": 5,
        "entries": [
            user([{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]),
            assistant(call("multi", "read", path="parts.txt"), stop="toolUse"),
            {
                "message": {
                    "role": "toolResult",
                    "toolCallId": "multi",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "😀" * 999}, {"type": "text", "text": "tail"}],
                    "isError": False,
                    "timestamp": 3,
                }
            },
            assistant("first", "second"),
            *exchanges(2, 80),
        ],
    },
    {"name": "simple exchanges cut at turn start", "keepRecentTokens": 150, "entries": exchanges(4)},
    {"name": "small budget splits the last exchange", "keepRecentTokens": 10, "entries": exchanges(3)},
    {"name": "long tool turn without prior history", "keepRecentTokens": 70, "entries": tool_turn("t")},
    {
        "name": "long tool turn with prior history",
        "keepRecentTokens": 70,
        "entries": exchanges(1, 20) + tool_turn("t"),
    },
    {
        "name": "trailing tool results keep their call",
        "keepRecentTokens": 100,
        "entries": [
            *exchanges(1, 20),
            user("inspect"),
            assistant(call("big", "read", path="big.txt"), stop="toolUse"),
            result("big", "read", "z" * 3000),
        ],
    },
    {"name": "nothing to compact", "keepRecentTokens": 100000, "entries": exchanges(2)},
    {
        "name": "previous compaction is updated with file lists",
        "keepRecentTokens": 50,
        "entries": [
            *exchanges(2, 50),
            {
                "compaction": {
                    "summary": "## Goal\nOld goal",
                    "firstKept": 2,
                    "tokensBefore": 999,
                    "details": {"readFiles": ["old/read.py"], "modifiedFiles": ["old/changed.py"]},
                }
            },
            user("next task " + "n" * 300),
            assistant(call("r1", "read", path="new/read.py"), stop="toolUse"),
            result("r1", "read", "content"),
            assistant("read it " + "r" * 300, total=500),
            user("final " + "f" * 300),
            assistant("final answer " + "g" * 300, total=900),
        ],
    },
    {
        "name": "custom instructions and reasoning model",
        "keepRecentTokens": 150,
        "reasoning": True,
        "thinkingLevel": "high",
        "customInstructions": "Preserve ticket ABC-123",
        "entries": exchanges(4),
    },
    {
        "name": "non-reasoning model ignores thinking level",
        "keepRecentTokens": 150,
        "thinkingLevel": "high",
        "entries": exchanges(4),
    },
    {
        "name": "serialization is JavaScript-exact",
        "keepRecentTokens": 5,
        "entries": [
            user("Ünïcødé 😀 request"),
            assistant(
                {"type": "thinking", "thinking": "private reasoning", "thinkingSignature": "sig"},
                "Visible 😀 text",
                call("u1", "bash", command="echo 'ü' \"q\"", timeout=1.5, nested={"a": [1, None, True]}),
                stop="toolUse",
            ),
            result("u1", "bash", "😀" * 1500 + "é" * 10),
            assistant("second text", "third text"),
            user("follow up"),
            assistant("closing"),
        ],
    },
    {
        "name": "omitted length attempt after the only prompt",
        "keepRecentTokens": 1,
        "entries": [user("x" * 5000), assistant("partial", stop="length"), {"omit": 1}],
    },
    {
        "name": "omitted error attempt between exchanges",
        "keepRecentTokens": 30,
        "entries": [
            *exchanges(2, 100),
            user("retry me " + "r" * 100),
            assistant(stop="error"),
            {"omit": 5},
            assistant("recovered " + "s" * 100),
        ],
    },
    {
        "name": "omitted tool attempt keeps the suffix rule closed",
        "keepRecentTokens": 1,
        "entries": [
            *exchanges(1, 50),
            user("inspect"),
            assistant(call("t1", "read", path="a.txt"), stop="length"),
            result("t1", "read", "Tool call was not executed: truncated"),
            {"omit": 3},
            {"omit": 4},
        ],
    },
    {
        "name": "usage before an omission is not trusted",
        "keepRecentTokens": 30,
        "entries": [
            *exchanges(1, 100),
            user("next " + "n" * 100),
            assistant("answer " + "a" * 100, total=5000),
            user("retry me"),
            assistant(stop="error"),
            {"omit": 5},
        ],
    },
    {
        "name": "usage after an omission is trusted",
        "keepRecentTokens": 30,
        "entries": [
            *exchanges(1, 100),
            user("retry me"),
            assistant(stop="error"),
            {"omit": 3},
            assistant("recovered " + "s" * 100, total=7000),
        ],
    },
    {
        "name": "images count toward the cut",
        "keepRecentTokens": 1202,
        "entries": [
            *exchanges(2, 100),
            {
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                    ],
                    "timestamp": 4,
                }
            },
            assistant("seen"),
        ],
    },
]


def main() -> None:
    OUT.write_text(json.dumps(CASES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
