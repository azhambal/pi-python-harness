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
`PI_HTTP_TIMEOUT` defaults to 300 seconds. `ANTHROPIC_THINKING_MODE` selects
`adaptive` (default, newer models) or `budget` (legacy models); supported
effort levels and whether thinking can be disabled depend on the chosen model.

Explicit CLI options take precedence over shell environment, then the named
environment file. Resume instead uses its stored model identity. Files are
not searched in ancestor directories; dotenv interpolation is disabled.
Credentials are omitted from settings repr and journal configuration. File
values are not exported to `os.environ` or Bash tools. Existing shell variables
retain normal subprocess inheritance. HTTP errors redact the configured key;
request headers are never written to the journal.

Providers normalize a copy of history: thinking from a different provider/API/model
is dropped, tool IDs/results remain paired, failed partial turns are omitted,
and missing results become explicit unknown-outcome placeholders. The original
journal is unchanged. Costs remain zero without a pricing catalog; usage counts
are real provider counts, and cached input is accounted separately.

Same-model thinking remains available for native replay, including DeepSeek and
SGLang. OpenAI-compatible whitespace-only tool results use `(no tool output)` in
the request; canonical journal text remains unchanged. Reasoning in a combined
stream chunk is processed before its visible answer.

Output limits use the model cap (8192 when unknown), the requested limit and the
estimated remaining context, whichever is smaller. Known context windows reserve
256 tokens for request framing; message and tool-declaration estimates use 3.5
characters per token. Estimates are approximate and may trigger overflow recovery.
Set accurate model limits for your backend.

An optional `ModelInfo.thinking_level_map` declares supported levels and native
effort values; unsupported levels fail before a request or durable setter change.
For example, `{"off": "none", "high": "high", "xhigh": "xhigh"}` preserves
`xhigh` on a backend that supports it. Existing provider defaults apply when no
map is supplied. For DeepSeek and Anthropic, `off` still uses their explicit
disable flag; budget-mode Anthropic still uses numeric budgets. The map declares
support in those modes but does not replace their wire protocol.

Wire-format references: [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/),
[DeepSeek thinking replay](https://api-docs.deepseek.com/guides/thinking_mode/),
[Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming),
[Anthropic thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking).

## Local SGLang

See [SGLang](sglang.md) for explicit thinking controls, request settings and reasoning usage.
