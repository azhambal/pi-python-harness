# Configuration

[Documentation](README.md) · [Project](../README.md)

`AgentConfig` is plain data and contains no secrets — no API keys, tokens,
or endpoints (providers own their credentials). It is validated before use:

- `cwd`, `model` (a `ModelInfo`: identity + context window + output cap),
  `thinking_level` (default `"off"`).
- `agent_dir` (default `~/.pi-agent`) and `session_dir` (default
  `~/.pi-agent/sessions/<sha256-of-cwd[:16]>`, always derived from the
  *effective* cwd). Hashing avoids lossy path encoding; hash collisions
  remain possible, so discovery also filters session headers by exact cwd.
- `compaction`: `enabled`, `reserve_tokens` (16384), `keep_recent_tokens`
  (20000) — Pi's defaults.
- `retry`: `enabled`, `max_retries` (3), `base_delay_ms` (2000),
  `max_delay_ms` (60000) — exponential backoff `base * 2^(n-1)`, capped;
  transient provider errors only.
- `bash_default_timeout_s` (default: none) and `instruction_files`
  (explicit AGENTS.md paths; when unset, deterministic discovery runs).
- `permissions`: a `PermissionPolicy` (default `allow`), serialized with the
  configuration. Callback functions are passed separately at construction.
- `tool_execution`: `"sequential"` by default, or `"parallel"`;
  `max_parallel_tools`: positive integer, default 4. Both persist on resume.

## Settings files

CLI `run` and `chat` load defaults in order:
`~/.pi-agent/settings.json`, `<cwd>/.pi-agent/settings.json`, then an explicit
`--settings PATH`. Automatic missing files are ignored; a missing explicit file
is an error. Nested `compaction` and `retry` fields merge; a `permissions` object
replaces the whole policy. Unknown fields, duplicate keys, invalid budgets and
credential fields are rejected. Files are read without writing global defaults.

Settings and MCP JSON accept an optional UTF-8 BOM. Add `$schema` for editor
validation; the runtime ignores that reference and never fetches it. Packaged
schemas are [settings.schema.json](../src/pi_agent/schemas/settings.schema.json)
and [mcp.schema.json](../src/pi_agent/schemas/mcp.schema.json). For example, a
project `.pi-agent/settings.json` can reference
`"$schema": "../src/pi_agent/schemas/settings.schema.json"` in this checkout.
See the generated [field reference](configuration-reference.md) for defaults,
descriptions and the additional checks performed at runtime.

```bash
.venv/bin/python -m pi_agent chat --settings examples/settings.example.json
.venv/bin/python -m pi_agent run "Inspect the project" --tool-execution parallel --max-parallel-tools 2
```

The public [examples/settings.example.json](../examples/settings.example.json) shows supported fields.
Model/provider/base_url/http_timeout are non-secret runtime defaults; credentials
stay in the named environment file or shell environment. Runtime precedence is
CLI → shell → named environment file → settings files → built-in defaults.
Provider-specific model/URL settings apply to their own transport prefix.
Explicit CLI thinking/scheduling/limit flags override file defaults. Programmatic
embeddings can use `pi_agent.settings.load_settings` and `config_from_settings`.

Settings are defaults for new sessions. Resume preserves the stored agent config;
files can still provide non-secret runtime transport defaults. Explicit CLI flags
and the public idle setters are the way to change an existing session. No hidden
file edit weakens a resumed permission policy or changes its model. The global
and project locations are read-only; no automatic credentials lookup outside the
named environment file is added. Set a known `context_window` for automatic
threshold compaction; the example intentionally leaves it unknown.

OpenAI-compatible runtime request settings also include `temperature`, `seed`,
and `chat_template_kwargs`. These are loaded on each CLI invocation, including
resume, and applied to conversation and compaction requests. They are not restored
from the session journal. See [SGLang](sglang.md) for precedence details.

`thinking_level_map` is an optional model capability map, for example
`{"off": "none", "low": "low", "high": "high", "xhigh": null}`. Missing or null
entries reject that level; string values select native effort for OpenAI-compatible
requests and Anthropic adaptive thinking. At least one level must be supported.
This enables reasoning for a generic OpenAI-compatible model without a catalog.
The map is copied into `ModelInfo` and persisted; resume retains it. Models and
capabilities must match the backend; the harness does not discover them remotely.
