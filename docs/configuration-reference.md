# Configuration field reference

[Documentation](README.md) · [Configuration](configuration.md)

Generated from the packaged JSON Schemas. Regenerate with
`python scripts/generate_config_reference.py`; use `--check` to detect drift.

Schemas provide editor validation. Runtime checks additionally enforce
finite numbers, cross-field constraints, UTF-8 byte limits, trusted sources,
URL authority, reserved headers and filesystem rules. Duplicate JSON keys
are always rejected. A `$schema` reference is never fetched by the runtime.

## pi-agent settings

Non-secret CLI defaults. Session defaults apply to new sessions; resume retains journaled configuration. Runtime validation additionally checks cross-field and filesystem constraints.

- `$schema` (string): Editor schema reference; ignored by the runtime and never fetched.
- `provider` (enum): HTTP provider family. Default: `"openai"`. Values: `"openai"`, `"deepseek"`, `"anthropic"`, `"sglang"`.
- `model` (string): Model identifier sent to the provider.
- `base_url` (string): Provider API root. Credentials, query strings and fragments are not supported.
- `http_timeout` (number): HTTP inactivity timeout in seconds, not a whole-run deadline. Default: `300`.
- `thinking_level` (enum): Requested reasoning level, checked against model capabilities. Default: `"off"`. Values: `"off"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`.
- `thinking_level_map` (object | null): Optional explicit native effort map. Missing or null levels are unsupported; at least one level must be supported. Persisted with model identity.
- `context_window` (integer): Known context window in tokens; omit when unknown.
- `max_tokens` (integer): Maximum output tokens per response, also bounded by estimated available context. Default: `8192`.
- `compaction` (object): History compaction thresholds; these do not cap summary output.
- `retry` (object): Bounded transient-error retries for conversation and summary requests.
- `bash_default_timeout_s` (number | null): Default Bash timeout in seconds; null leaves it unset. Default: `null`.
- `tool_execution` (enum): Tool batch scheduling mode. Default: `"sequential"`. Values: `"sequential"`, `"parallel"`.
- `max_parallel_tools` (integer): Maximum admitted concurrent tool calls in parallel mode. Default: `4`.
- `permissions` (object): Admission policy, not an OS sandbox. Replaces the whole lower-priority policy.
- `tools` (array): Absolute builtin selection for new sessions. An empty list disables builtins; relative edits belong in CLI --tools.
- `skills` (object): Trusted Markdown skill discovery and prompt budget.
- `temperature` (number | null): OpenAI-compatible sampling temperature; runtime-only.
- `seed` (integer | null): OpenAI-compatible sampling seed; runtime-only.
- `chat_template_kwargs` (object | null): OpenAI-compatible template arguments. SGLang enable_thinking follows the selected level.

### compaction

- `enabled` (boolean): Enable threshold and overflow compaction. Default: `true`.
- `reserve_tokens` (integer): Start compaction when context exceeds window minus this reserve. Default: `16384`.
- `keep_recent_tokens` (integer): Approximate recent history retained after compaction. Default: `20000`.

### retry

- `enabled` (boolean): Retry transient provider failures. Default: `true`.
- `max_retries` (integer): Retries after the initial request. Default: `3`.
- `base_delay_ms` (integer): Initial exponential backoff in milliseconds. Default: `2000`.
- `max_delay_ms` (integer): Backoff cap in milliseconds; must be at least base_delay_ms. Default: `60000`.

### permissions

- `default` (enum): Action when no rule matches. Default: `"allow"`. Values: `"allow"`, `"ask"`, `"deny"`.
- `read_only` (boolean): Allow only read and verified builtin search tools. Default: `false`.
- `rules` (array): First matching rule wins, subject to read_only. Default: `[]`.

### rule

- `tool` (string): Tool-name glob.
- `action` (enum): Admission decision. Values: `"allow"`, `"ask"`, `"deny"`.
- `command` (string | null): Optional Bash command glob.
- `path` (string | null): Optional path glob; parent traversal components are rejected.
- `reason` (string | null): Optional explanation shown to the caller.

### skills

- `enabled` (boolean): Enable skills in new CLI sessions; the core SDK defaults to false. Default: `true`.
- `include_user` (boolean): Include automatic user skill roots. Default: `true`.
- `project_trusted` (boolean): Trust project Markdown skill discovery. Default: `false`.
- `pi_compat` (boolean): Include Pi-compatible skill directories. Default: `false`.
- `paths` (array): Explicit skill paths. Default: `[]`.
- `prompt_max_bytes` (integer): Maximum UTF-8 bytes for the skills prompt section. Default: `20480`.

## pi-agent MCP configuration

Explicit trusted direct MCP servers. Runtime validation additionally checks UTF-8 byte budgets, URL authority, reserved headers and project trust.

- `$schema` (string): Editor schema reference; ignored by the runtime and never fetched.
- `mcpServers` (object): Named servers; no process or connection is restored from journal authority.

### common

- `enabled` (boolean): Connect the server when explicitly selected by the host. Default: `true`.
- `timeout` (number): Request timeout in seconds. Default: `60`.
- `exposure` (enum): Required for enabled servers. Deferred and codemode exposure are unsupported. Values: `"direct"`.

### stdio

- `type` (enum): Subprocess transport; the default when command is present. Values: `"stdio"`.
- `command` (string): Executable from trusted configuration.
- `args` (array): Literal process arguments. Default: `[]`.
- `cwd` (string): Working directory relative to host cwd, not the configuration file.
- `env` (object): Explicit whole-value environment references; literal secrets are rejected.

### http

- `type` (enum): Streamable HTTP transport; OAuth and legacy SSE are unsupported. Values: `"http"`, `"streamable-http"`.
- `url` (string): Endpoint without credentials, query or fragment.
- `headers` (object): Distinct non-protocol header names mapped to whole-value environment references.
