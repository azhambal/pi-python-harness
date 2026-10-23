"""Write tests/parity/provider_cases.json: provider requests and streams to compare with pi.

Run: .venv/bin/python scripts/pi_reference/make_provider_cases.py, then generate.sh.
Messages are in pi's JSON format; tool declarations and the system prompt come from
tests/parity/fixtures/pi_declarations.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
decl = json.loads((ROOT / "tests/parity/fixtures/pi_declarations.json").read_text())
tools = {
    d["name"]: {"name": d["name"], "description": d["description"], "parameters": d["parameters"]}
    for d in decl["declarations"]
}
prompt = {key: value for key, value in decl["prompts"]["default"].items() if key != "docs"}


def system(names=("read", "bash", "edit", "write"), sections=prompt, ts=0, **extra):
    message = {"role": "system", "content": "", "sections": sections, "timestamp": ts}
    if sections is None:
        del message["sections"]
    if names:
        message["toolsAdded"] = [tools[name] for name in names]
    message.update(extra)
    return message


def usage(**values):
    base = {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }
    base.update(values)
    return base


def asst(
    content,
    model="claude-opus-4-8",
    provider="anthropic",
    api="anthropic-messages",
    stop="toolUse",
    ts=2,
    **extra,
):
    message = {
        "role": "assistant",
        "content": content,
        "api": api,
        "provider": provider,
        "model": model,
        "usage": usage(),
        "stopReason": stop,
        "timestamp": ts,
    }
    message.update(extra)
    return message


def user(content, ts=1):
    return {"role": "user", "content": content, "timestamp": ts}


def result(id, content, name="read", error=False, ts=3):
    return {
        "role": "toolResult",
        "toolCallId": id,
        "toolName": name,
        "content": content,
        "isError": error,
        "timestamp": ts,
    }


def call(id, name="read", args=None, **extra):
    return {
        "type": "toolCall",
        "id": id,
        "name": name,
        "arguments": args if args is not None else {"path": "README.md"},
        **extra,
    }


def cat(id, provider="anthropic", **overrides):
    return {"provider": provider, "catalog": id, **({"overrides": overrides} if overrides else {})}


img = {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}
history = [
    system(),
    user("Fix the bug in src/a.py"),
    asst(
        [
            {"type": "thinking", "thinking": "Look first.", "thinkingSignature": "sig-1"},
            {"type": "text", "text": "Reading."},
            call("toolu_01"),
        ]
    ),
    result("toolu_01", [{"type": "text", "text": "def f():\n    return 1\n"}]),
    asst(
        [
            call("toolu_02", "edit", {"path": "src/a.py", "edits": [{"oldText": "1", "newText": "2"}]}),
            call("toolu_03", "bash", {"command": "pytest"}),
        ],
        ts=4,
    ),
    result(
        "toolu_02", [{"type": "text", "text": "Successfully replaced 1 block(s) in src/a.py."}], "edit", ts=5
    ),
    result("toolu_03", [{"type": "text", "text": "1 passed"}], "bash", ts=5),
    asst([{"type": "text", "text": "Fixed."}], stop="stop", ts=6),
    user("Thanks, now explain it", ts=7),
]
cases = []


def case(name, model, messages, options=None, sse=None, **extra):
    entry = {"name": name, "model": model, "messages": messages}
    if options:
        entry["options"] = options
    if sse is not None:
        entry["sse"] = sse
    entry.update(extra)
    cases.append(entry)


# --- Anthropic requests -------------------------------------------------
for mid in [
    "claude-opus-4-8",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]:
    case(f"{mid} first prompt off", cat(mid), [system(), user("hello")])
    case(f"{mid} history high", cat(mid), history, {"reasoning": "high"})
for level in ["minimal", "low", "medium", "xhigh", "max"]:
    case(f"sonnet-4-5 budget {level}", cat("claude-sonnet-4-5"), history, {"reasoning": level})
    case(f"opus-4-8 adaptive {level}", cat("claude-opus-4-8"), history, {"reasoning": level})
    case(f"opus-5-5 managed {level}", cat("claude-opus-5-5"), history, {"reasoning": level})
case(
    "custom budgets",
    cat("claude-sonnet-4-5"),
    history,
    {"reasoning": "medium", "thinkingBudgets": {"medium": 4000}},
)
case(
    "custom model unknown to catalog",
    {
        "id": "claude-custom",
        "name": "Custom",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.anthropic.com",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 200000,
        "maxTokens": 8192,
    },
    history,
    {"reasoning": "low"},
)
case(
    "small context clamps max tokens",
    cat("claude-sonnet-4-5", contextWindow=12000),
    history + [user("x" * 20000, ts=8)],
    {"reasoning": "high"},
)
case("tiny context leaves one token", cat("claude-haiku-4-5", contextWindow=5000), history)
case("long cache retention", cat("claude-opus-4-8"), history, {"cacheRetention": "long"})
case("no cache", cat("claude-opus-4-8"), history, {"cacheRetention": "none"})
case("session id without affinity", cat("claude-opus-4-8"), history, {"sessionId": "sess-1"})
delta_add = system(None, sections=None, ts=8, toolsAdded=[tools["grep"], tools["find"]])
delta_section = system(
    None, sections={"cwd": "<cwd>\n/other\n</cwd>", "rules": None}, ts=9, content="Extra note."
)
delta_remove = system(None, sections=None, ts=9, toolsRemoved=[{"name": "write"}])
mid_history = history[:4] + [delta_add, delta_section] + history[4:] + [delta_remove, user("go on", ts=10)]
for mid in ["claude-opus-4-8", "claude-sonnet-4-5", "claude-opus-5", "claude-fable-5-1"]:
    case(f"{mid} mid-conversation system and tool changes", cat(mid), mid_history, {"reasoning": "medium"})
case(
    "collapsed transcript still estimates every system message",
    cat("claude-sonnet-4-5", contextWindow=12000),
    mid_history,
    {"reasoning": "low"},
)
managed = [
    system(),
    user("a"),
    asst([{"type": "text", "text": "b"}], model="claude-opus-5", stop="stop", providerThinkingLevel="low"),
    user("c", ts=3),
    asst(
        [{"type": "text", "text": "d"}], model="claude-opus-5", stop="stop", ts=4, providerThinkingLevel="max"
    ),
    user("e", ts=5),
]
case("managed effort replays historical levels", cat("claude-opus-5"), managed, {"reasoning": "xhigh"})
case("managed effort off keeps no markers", cat("claude-opus-5"), managed)
thinking = [
    system(),
    user("q"),
    asst(
        [
            {"type": "thinking", "thinking": "signed", "thinkingSignature": "s"},
            {
                "type": "thinking",
                "thinking": "[Reasoning redacted]",
                "thinkingSignature": "opaque",
                "redacted": True,
            },
            {"type": "thinking", "thinking": "unsigned"},
            {"type": "thinking", "thinking": "  "},
            {"type": "text", "text": "  "},
            {"type": "text", "text": "answer"},
        ],
        stop="stop",
    ),
    user("again", ts=3),
]
case("thinking blocks same model", cat("claude-opus-4-8"), thinking, {"reasoning": "high"})
case("thinking blocks other model", cat("claude-sonnet-4-5"), thinking, {"reasoning": "high"})
case(
    "images and error tool image",
    cat("claude-opus-4-8"),
    [
        system(),
        user([{"type": "text", "text": "see"}, img]),
        asst([call("t1"), call("t2")]),
        result("t1", [img], error=True),
        result("t2", [img, {"type": "text", "text": "ok"}]),
        user([img], ts=4),
    ],
)
case(
    "non-vision model gets placeholders",
    cat("claude-opus-4-8", input=["text"]),
    [system(), user([{"type": "text", "text": "see"}, img]), asst([call("t1")]), result("t1", [img])],
)
case(
    "cross-provider ids are normalized",
    cat("claude-opus-4-8"),
    [
        system(),
        user("q"),
        asst(
            [call("call_ab|fc+1/2=", thoughtSignature="x")],
            model="gpt-5",
            provider="openai",
            api="openai-responses",
        ),
        result("call_ab|fc+1/2=", [{"type": "text", "text": "r"}]),
        user("next", ts=4),
    ],
)
case(
    "unanswered call before user",
    cat("claude-opus-4-8"),
    [
        system(),
        user("q"),
        asst([call("t1"), call("t2")]),
        result("t1", [{"type": "text", "text": "r"}]),
        user("stop", ts=4),
    ],
)
case("no tools and no prompt", cat("claude-haiku-4-5"), [user("plain")])
case(
    "empty user text skipped",
    cat("claude-haiku-4-5"),
    [system(), user("  "), user([{"type": "text", "text": " "}]), user("real")],
)
case("unicode and surrogates", cat("claude-haiku-4-5"), [system(), user("привет 😀 \ud83d tail")])


# --- Anthropic streams --------------------------------------------------
def sse(*events, crlf=False):
    parts = []
    for event in events:
        if isinstance(event, str):
            parts.append(event)
        else:
            name, data = event
            parts.append(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")
    text = "".join(parts)
    return text.replace("\n", "\r\n") if crlf else text


def start(model="claude-opus-4-8", **values):
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 2000,
                    "cache_creation_input_tokens": 300,
                    **values,
                },
            },
        },
    )


def bstart(index, block):
    return ("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})


def delta(index, value):
    return ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": value})


def bstop(index):
    return ("content_block_stop", {"type": "content_block_stop", "index": index})


def mdelta(reason, **values):
    return (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None},
            "usage": {"output_tokens": 50, **values},
        },
    )


stop = ("message_stop", {"type": "message_stop"})
ping = 'event: ping\ndata: {"type": "ping"}\n\n'
base = [system(), user("hello")]
case(
    "stream thinking text and tool call",
    cat("claude-opus-4-8"),
    base,
    {"reasoning": "high"},
    sse(
        start(),
        ping,
        bstart(0, {"type": "thinking", "thinking": "", "signature": ""}),
        delta(0, {"type": "thinking_delta", "thinking": "Let me "}),
        delta(0, {"type": "thinking_delta", "thinking": "see."}),
        delta(0, {"type": "signature_delta", "signature": "EqQB"}),
        bstop(0),
        bstart(1, {"type": "text", "text": ""}),
        delta(1, {"type": "text_delta", "text": "Reading the file."}),
        bstop(1),
        bstart(2, {"type": "tool_use", "id": "toolu_9", "name": "read", "input": {}}),
        delta(2, {"type": "input_json_delta", "partial_json": '{"path": "REA'}),
        delta(2, {"type": "input_json_delta", "partial_json": 'DME.md", "limit": 5}'}),
        bstop(2),
        mdelta(
            "tool_use",
            cache_creation={"ephemeral_1h_input_tokens": 100},
            output_tokens_details={"thinking_tokens": 20},
        ),
        stop,
    ),
)
case(
    "stream redacted thinking",
    cat("claude-sonnet-4-5"),
    base,
    {"reasoning": "low"},
    sse(
        start("claude-sonnet-4-5"),
        bstart(0, {"type": "redacted_thinking", "data": "ENCRYPTED"}),
        bstop(0),
        bstart(1, {"type": "text", "text": "ok"}),
        bstop(1),
        mdelta("end_turn"),
        stop,
    ),
)
case(
    "stream malformed tool arguments",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(
        start(),
        bstart(0, {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}}),
        delta(0, {"type": "input_json_delta", "partial_json": '{"command": "ls" "x"}'}),
        bstop(0),
        mdelta("tool_use"),
        stop,
    ),
)
case(
    "stream repaired tool arguments",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(
        start(),
        bstart(0, {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}}),
        delta(0, {"type": "input_json_delta", "partial_json": '{"command": "echo \\q\ttab"}'}),
        bstop(0),
        mdelta("tool_use"),
        stop,
    ),
)
case(
    "stream refusal",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(
        start(),
        bstart(0, {"type": "text", "text": "I"}),
        bstop(0),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "refusal", "stop_details": {"explanation": "Not allowed."}},
                "usage": {"output_tokens": 3},
            },
        ),
        stop,
    ),
)
case(
    "stream max tokens",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(start(), bstart(0, {"type": "text", "text": "partial"}), bstop(0), mdelta("max_tokens"), stop),
)
case(
    "stream unknown block ignored",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(
        start(),
        bstart(0, {"type": "server_tool_use", "id": "srv", "name": "web_search", "input": {}}),
        delta(0, {"type": "input_json_delta", "partial_json": "{}"}),
        bstop(0),
        bstart(1, {"type": "text", "text": "done"}),
        bstop(1),
        mdelta("end_turn"),
        stop,
    ),
)
case(
    "stream error event",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(
        start(),
        'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n',
    ),
)
case(
    "stream without message_stop",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(start(), bstart(0, {"type": "text", "text": "x"}), bstop(0), mdelta("end_turn")),
)
case(
    "stream without stop reason",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(start(), bstart(0, {"type": "text", "text": "x"}), bstop(0), stop),
)
case(
    "stream fallback model cost",
    cat("claude-fable-5"),
    base,
    {"reasoning": "high"},
    sse(
        start("claude-opus-4-8"),
        bstart(0, {"type": "text", "text": "hi"}),
        bstop(0),
        mdelta("end_turn"),
        stop,
    ),
)
case(
    "stream crlf framing",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(
        start(),
        bstart(0, {"type": "text", "text": ""}),
        delta(0, {"type": "text_delta", "text": "a\nb"}),
        bstop(0),
        mdelta("pause_turn"),
        stop,
        crlf=True,
    ),
)
case(
    "stream tool input in start block",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(
        start(),
        bstart(0, {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {"path": "."}}),
        bstop(0),
        mdelta("tool_use"),
        stop,
    ),
)


# --- OpenAI-compatible requests -----------------------------------------
def custom(id, provider, base_url, reasoning=True, compat=None, **extra):
    model = {
        "id": id,
        "name": id,
        "api": "openai-completions",
        "provider": provider,
        "baseUrl": base_url,
        "reasoning": reasoning,
        "input": ["text", "image"],
        "cost": {"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 16384,
    }
    if compat:
        model["compat"] = compat
    model.update(extra)
    return model


def oa(content, model="deepseek-flash", provider="deepseek", stop="toolUse", ts=2, **extra):
    return asst(content, model=model, provider=provider, api="openai-completions", stop=stop, ts=ts, **extra)


ds_history = [
    system(),
    user("Fix it"),
    oa(
        [
            {"type": "thinking", "thinking": "plan", "thinkingSignature": "reasoning_content"},
            {"type": "text", "text": "ok"},
            call("call_1"),
        ]
    ),
    result("call_1", [{"type": "text", "text": "data"}]),
    oa([{"type": "text", "text": "done"}], stop="stop", ts=4),
    user("more", ts=5),
]
for level in ["off", "minimal", "low", "medium", "high", "max"]:
    options = {} if level == "off" else {"reasoning": level}
    case(f"deepseek-flash {level}", cat("deepseek-flash", provider="deepseek"), ds_history, options)
case(
    "deepseek-v4-pro mid-conversation",
    cat("deepseek-v4-pro", provider="deepseek"),
    mid_history,
    {"reasoning": "high"},
)
case(
    "deepseek cross-model history", cat("deepseek-flash", provider="deepseek"), history, {"reasoning": "high"}
)
case(
    "deepseek empty tool output and images",
    cat("deepseek-flash", provider="deepseek"),
    [
        system(),
        user([{"type": "text", "text": "see"}, img]),
        oa([call("c1"), call("c2")]),
        result("c1", [{"type": "text", "text": "  "}]),
        result("c2", [img]),
        user("next", ts=4),
    ],
)
openai_model = custom("gpt-test", "openai", "https://api.openai.com/v1")
case(
    "openai reasoning model with session cache key",
    openai_model,
    history,
    {"reasoning": "high", "sessionId": "s" * 70},
)
case(
    "openai long cache retention",
    openai_model,
    history,
    {"reasoning": "low", "sessionId": "abc", "cacheRetention": "long"},
)
case(
    "openai non-reasoning",
    custom("gpt-plain", "openai", "https://api.openai.com/v1", reasoning=False),
    history,
)
case(
    "local server defaults",
    custom("local", "local", "http://127.0.0.1:8080/v1"),
    history,
    {"reasoning": "medium"},
)
case(
    "local server off with off effort",
    custom("local", "local", "http://127.0.0.1:8080/v1", thinkingLevelMap={"off": "none"}),
    history,
)
router = custom("anthropic/claude-sonnet-4.5", "openrouter", "https://openrouter.ai/api/v1")
case("openrouter anthropic model", router, history, {"reasoning": "high", "sessionId": "sess"})
case("openrouter off", router, history, {"sessionId": "sess"})
sglang = {
    "thinkingFormat": "chat-template",
    "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}},
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "maxTokensField": "max_tokens",
}
for level in ["off", "high"]:
    case(
        f"chat-template {level}",
        custom("qwen", "sglang", "http://127.0.0.1:30000/v1", compat=sglang),
        history,
        {} if level == "off" else {"reasoning": level},
    )
budget = {
    "thinkingFormat": "chat-template",
    "supportsThinkingTokenBudget": True,
    "chatTemplateKwargs": {
        "enable_thinking": {"$var": "thinking.enabled"},
        "budget": {"$var": "thinking.budget"},
        "effort": {"$var": "thinking.effort", "omitWhenOff": True},
        "fixed": "x",
    },
}
case(
    "chat-template budget variables",
    custom("qwen", "vllm", "http://127.0.0.1:8000/v1", compat=budget),
    history,
    {"reasoning": "medium", "thinkingBudgets": {"medium": 3000}},
)
case(
    "chat-template budget variables off",
    custom("qwen", "vllm", "http://127.0.0.1:8000/v1", compat=budget),
    history,
)
for fmt in ["qwen", "qwen-chat-template", "zai", "together", "string-thinking", "baseten", "ant-ling"]:
    compat = {
        "thinkingFormat": fmt,
        "supportsReasoningEffort": True,
        "chatTemplateArgs": {"thinking": {"$var": "thinking.enabled"}},
    }
    model = custom(
        "m",
        "custom",
        "http://127.0.0.1:9000/v1",
        compat=compat,
        thinkingLevelMap={"low": "lite", "off": "none"},
    )
    case(f"thinking format {fmt} low", model, [system(), user("q")], {"reasoning": "low"})
    case(f"thinking format {fmt} off", model, [system(), user("q")])
quirks = {
    "requiresAssistantAfterToolResult": True,
    "requiresToolResultName": True,
    "requiresThinkingAsText": True,
    "supportsStrictMode": True,
    "supportsUsageInStreaming": False,
    "zaiToolStream": True,
    "vllmPriority": 5,
}
case(
    "compat quirks",
    custom("q", "custom", "http://127.0.0.1:9000/v1", compat=quirks),
    ds_history + [result("x", [img])],
    {"reasoning": "high"},
)
case(
    "mid-conversation tool additions",
    custom(
        "kimi",
        "custom",
        "http://127.0.0.1:9000/v1",
        compat={"supportsMidConvoSystemMessages": True, "supportsMidConvoToolAdditions": True},
    ),
    mid_history,
    {"reasoning": "high"},
)
case(
    "mid-conversation system messages only",
    custom("m", "custom", "http://127.0.0.1:9000/v1", compat={"supportsMidConvoSystemMessages": True}),
    mid_history,
)
details = json.dumps(
    [
        {
            "type": "reasoning.text",
            "text": "plan",
            "signature": "sig",
            "format": "anthropic-claude-v1",
            "index": 0,
        }
    ]
)
case(
    "reasoning details replay",
    router,
    [
        system(),
        user("q"),
        oa(
            [
                {"type": "thinking", "thinking": "plan", "thinkingSignature": details},
                {"type": "text", "text": "a"},
            ],
            model=router["id"],
            provider="openrouter",
            stop="stop",
        ),
        user("again", ts=3),
    ],
    {"reasoning": "high"},
)
case(
    "legacy encrypted reasoning on tool call",
    router,
    [
        system(),
        user("q"),
        oa(
            [
                {"type": "thinking", "thinking": ""},
                call(
                    "c1",
                    thoughtSignature=json.dumps({"type": "reasoning.encrypted", "id": "r1", "data": "enc"}),
                ),
            ],
            model=router["id"],
            provider="openrouter",
        ),
        result("c1", [{"type": "text", "text": "r"}]),
        user("again", ts=4),
    ],
    {"reasoning": "high"},
)
case(
    "non-vision tool images",
    custom("t", "custom", "http://127.0.0.1:9000/v1", input=["text"]),
    [system(), user("q"), oa([call("c1")]), result("c1", [img])],
)
case(
    "no tools but tool history",
    custom("t", "custom", "http://127.0.0.1:9000/v1"),
    [user("q"), oa([call("c1")]), result("c1", [{"type": "text", "text": "r"}]), user("n", ts=4)],
)


# --- OpenAI-compatible streams ------------------------------------------
def chunks(*items, done=True):
    text = "".join(f"data: {json.dumps(item, ensure_ascii=False)}\n\n" for item in items)
    return text + ("data: [DONE]\n\n" if done else "")


def ch(delta=None, finish=None, **extra):
    choice = {"index": 0, "delta": delta or {}, "finish_reason": finish}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": extra.pop("model", "deepseek-flash"),
        "choices": [choice],
        **extra,
    }


usage_chunk = {
    "id": "chatcmpl-1",
    "model": "deepseek-flash",
    "choices": [],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 80,
        "prompt_cache_hit_tokens": 600,
        "completion_tokens_details": {"reasoning_tokens": 30},
    },
}
ds = cat("deepseek-flash", provider="deepseek")
case(
    "openai stream reasoning text and tools",
    ds,
    base,
    {"reasoning": "high"},
    chunks(
        ch({"role": "assistant", "reasoning_content": "Let "}),
        ch({"reasoning_content": "me think"}),
        ch({"content": "Reading"}),
        ch({"content": " now"}),
        ch(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "read", "arguments": ""},
                    }
                ]
            }
        ),
        ch({"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]}),
        ch({"tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}]}),
        ch(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls" x}'},
                    }
                ]
            }
        ),
        ch({}, "tool_calls"),
        usage_chunk,
    ),
)
case(
    "openai stream reasoning and answer in one delta",
    ds,
    base,
    {"reasoning": "high"},
    chunks(ch({"content": "answer", "reasoning_content": "plan"}, "stop")),
)
case(
    "openai stream reasoning details",
    router,
    base,
    {"reasoning": "high"},
    chunks(
        ch(
            {
                "reasoning": "a",
                "reasoning_details": [{"type": "reasoning.text", "text": "a", "format": "f", "index": 0}],
            },
            model=router["id"],
        ),
        ch(
            {
                "reasoning": "b",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "b", "signature": "sig", "index": 0}
                ],
            },
            model=router["id"],
        ),
        ch(
            {"reasoning_details": [{"type": "reasoning.encrypted", "data": "enc", "id": "r1"}]},
            model=router["id"],
        ),
        ch(
            {"content": "ok"},
            "stop",
            model=router["id"],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
            },
        ),
    ),
)
case("openai stream content filter", ds, base, None, chunks(ch({"content": "x"}), ch({}, "content_filter")))
case("openai stream without finish reason", ds, base, None, chunks(ch({"content": "x"})))
case("openai stream without done marker", ds, base, None, chunks(ch({"content": "x"}, "stop"), done=False))
case(
    "openai stream tool call by id without index",
    ds,
    base,
    None,
    chunks(
        ch({"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]}),
        ch({}, "tool_calls"),
    ),
)
case(
    "openai stream error payload",
    ds,
    base,
    None,
    chunks(ch({"content": "x"}), {"error": {"message": "Rate limit", "type": "rate_limit"}}),
)
case(
    "openai stream other model name and choice usage",
    ds,
    base,
    None,
    chunks(ch({"content": "x"}, model="deepseek-other"), ch({}, "length", model="deepseek-other")),
)
case(
    "openai stream length with partial tool json",
    ds,
    base,
    None,
    chunks(
        ch(
            {
                "tool_calls": [
                    {"index": 0, "id": "c1", "function": {"name": "write", "arguments": '{"path": "a'}}
                ]
            }
        ),
        ch({}, "length"),
    ),
)

# Arguments cross incomplete boolean/escape boundaries in the actual stream callbacks.
fragments = ['{"a":1,"b":tru', 'e,"c":fal', 'se,"d":"abc\\', '\\"}']
case(
    "anthropic partial boolean and escape snapshots",
    cat("claude-opus-4-8"),
    base,
    None,
    sse(
        start(),
        bstart(0, {"type": "tool_use", "id": "c1", "name": "read", "input": {}}),
        *(delta(0, {"type": "input_json_delta", "partial_json": fragment}) for fragment in fragments),
        bstop(0),
        mdelta("tool_use"),
        stop,
    ),
)
case(
    "openai partial boolean and escape snapshots",
    ds,
    base,
    None,
    chunks(
        ch({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": ""}}]}),
        *(ch({"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]}) for fragment in fragments),
        ch({}, "tool_calls"),
    ),
)

(ROOT / "tests/parity/provider_cases.json").write_text(json.dumps(cases, indent=1) + "\n")
print(f"{len(cases)} cases")
