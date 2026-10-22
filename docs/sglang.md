# SGLang

[Documentation](README.md) · [Project](../README.md)

SGLang uses the OpenAI-compatible Chat Completions transport. A server must
already be running with tool calling configured for the model it serves.
pi-agent connects to the endpoint; it does not install or patch the serving runtime.

## Connect

Set `model` in [the example settings](../examples/sglang/settings.json) to the
served model ID, and make the context/output limits match the server:

```bash
OPENAI_API_KEY=local .venv/bin/pi-agent chat \
  --settings examples/sglang/settings.json
```

For an unauthenticated localhost endpoint the client accepts a placeholder key.
Use the server's real authentication token when authentication is enabled.
Select a different URL with `--base-url`; the default SGLang URL is
`http://127.0.0.1:30000/v1`.

## Thinking and request settings

`--provider sglang` maps the session's thinking level to
`chat_template_kwargs.enable_thinking`. In particular, `--thinking off` sends
`false` even if the server defaults to thinking enabled. This behavior depends
on provider identity, not on guessing a URL or model name. The session setting
takes precedence over a conflicting template `enable_thinking` value.

Both `run` and `chat` accept `--temperature`, `--seed` and
`--chat-template-kwargs` with a JSON object. The backend must support the supplied
template arguments; they are model-specific.

```python
from pi_agent.providers import OpenAICompatProvider, OpenAIRequestSettings

provider = OpenAICompatProvider(
    api_key="local",
    base_url="http://127.0.0.1:30000/v1",
    request_settings=OpenAIRequestSettings(temperature=0, seed=12345),
)
```

Use `ModelInfo(provider="sglang", ...)` for the SGLang-specific mapping and
reasoning replay. Request settings also work with ordinary OpenAI-compatible
providers, without automatic SGLang fields.

Transport settings apply to every request, including compaction. They are loaded
on each CLI invocation and are not restored from the journal: pass the same
settings when using `--continue` or `--resume`. Model identity and session thinking
level are journaled. Request payloads are copied so mutation does not change later
requests. A seed requests backend repeatability; it does not guarantee identical
results across hardware or serving configurations.

## Reasoning usage

The decoder accepts `usage.completion_tokens_details.reasoning_tokens` and the
flat `usage.reasoning_tokens`. The standard nested field takes precedence,
including an explicit zero. Missing counts remain unknown. Cumulative updates
replace previous values; invalid counts produce a protocol error.
