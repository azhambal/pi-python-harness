# Providers and credentials

[Documentation](README.md) · [Project](../README.md)

For DeepSeek, retain your private `OPENAI_API_KEY` and configure:

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
PI_PROVIDER=deepseek
```

`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` are used by the
OpenAI-compatible adapter. `PI_PROVIDER` accepts `deepseek`, `openai`, `anthropic`, or `sglang`; generic OpenAI defaults to `https://api.openai.com/v1`.
Anthropic uses `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (default
`https://api.anthropic.com`), and `ANTHROPIC_MODEL`. Optional
`PI_HTTP_TIMEOUT` defaults to 300 seconds.

## Model catalog

New sessions look the model up in pi's catalog, shipped as
`pi_agent/data/models/{anthropic,deepseek}.json` (copies of pi's generated
provider data). A catalog entry supplies the context window, output cap, input
modalities, prices, thinking levels and pi's `compat` request flags;
`--context-window`, `--max-tokens` and a settings `thinking_level_map` override
it. Unknown model IDs still work with conservative defaults (8192 output tokens,
no context window, no prices). Usage costs are computed with pi's
`calculateCost`, including input tiers and 1-hour cache writes.

## Anthropic requests

Requests are translations of pi's `streamSimple` and `buildParams` and are
compared with pi's real request bodies in `tests/parity/test_provider_parity.py`:
system prompt and tool caching, `eager_input_streaming` tool definitions,
mid-conversation system messages and native tool changes (`tool_addition` /
`tool_removal`) for models that support them, managed effort markers, beta
headers, fallback models, thinking replay and error-result images.

Thinking follows the model: models with `compat.forceAdaptiveThinking` use
adaptive thinking with an effort level, others a token budget (pi's defaults:
minimal 1024, low 2048, medium 8192, high/xhigh/max 16384) added to the answer
room. `ANTHROPIC_THINKING_MODE=adaptive|budget` overrides the model's choice.
Prompt caching uses 5-minute breakpoints; `PI_CACHE_RETENTION=long` asks for one
hour, as in pi. The output cap is the model cap, reduced to the context window
minus the estimated context and 4096 safety tokens. OAuth tokens, workload
identity federation and GitHub Copilot are not supported.

## OpenAI-compatible requests

Chat Completions requests are translations of pi's `openai-completions` adapter
and are compared with pi in the same parity test. Request quirks come from pi's
`detectCompat` (provider name and base URL: DeepSeek, OpenRouter, Together, Z.ai,
xAI and others) overridden by the catalog's `compat`: the instruction role
(`developer` for reasoning models on standard endpoints), `store: false`,
`max_tokens` or `max_completion_tokens`, the thinking parameter format
(`reasoning_effort`, DeepSeek's `thinking`, OpenRouter's `reasoning`,
`chat_template_kwargs`, ...), reasoning replay in the field it streamed in
(`reasoning_content`, `reasoning`, or OpenRouter `reasoning_details`), prompt
cache keys and Anthropic-style cache control behind OpenRouter. pi's OpenAI
catalog uses the Responses API, which this port does not implement, so
`--provider openai` models are treated as custom Chat Completions models.

`--temperature`, `--seed` and `--chat-template-kwargs` are port extensions added
to every request; template values that pi computes (such as `enable_thinking`)
take precedence over the literal ones.

Explicit CLI options take precedence over shell environment, then the named
environment file. Resume instead uses its stored model identity. Files are
not searched in ancestor directories; dotenv interpolation is disabled.
Credentials are omitted from settings repr and journal configuration. File
values are not exported to `os.environ` or Bash tools. Existing shell variables
retain normal subprocess inheritance. HTTP errors redact the configured key;
request headers are never written to the journal.

Providers project a copy of history with pi's `transformMessages`: thinking from
a different provider/API/model is dropped, tool IDs are normalized per provider,
failed partial turns are omitted, images become placeholders for text-only
models, and missing results are answered with `"No result provided"`. The
original journal is unchanged.

Same-model thinking is replayed natively. OpenAI-compatible whitespace-only tool
results use `(no tool output)` in the request, as in pi; canonical journal text
remains unchanged. Reasoning in a combined stream chunk is processed before its
visible answer. Malformed tool-call arguments become the call's
`arguments_error` with pi's `JSON.parse` message, so the model can retry.

Context estimates follow pi's `estimateContextTokens`: the last valid response's
usage plus 3.5 UTF-16 code units per token after it (images count as 4800
characters). Both adapters cap output at the model's limit (8192 when unknown),
reduced to the context window minus that estimate and 4096 safety tokens.

`ModelInfo.thinking_level_map` has pi's meaning: a level mapped to `null` is
unsupported, `xhigh` and `max` are supported only when listed, other levels are
supported and use the provider's default effort, and a string is the native
effort sent. A requested unsupported level is clamped to the nearest supported
one, as pi's `clampThinkingLevel` does. For example, `{"off": null, "xhigh":
"xhigh"}` makes thinking mandatory and enables `xhigh`.

Wire-format references: [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/),
[DeepSeek thinking replay](https://api-docs.deepseek.com/guides/thinking_mode/),
[Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming),
[Anthropic thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking).

## Local SGLang

See [SGLang](sglang.md) for explicit thinking controls, request settings and reasoning usage.
