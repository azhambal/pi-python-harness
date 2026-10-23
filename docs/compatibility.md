# Compatibility and scope

[Documentation](README.md) · [Project](../README.md)

The goal is that a model sees the same thing it would see in pi: tool
declarations, the system prompt, tool results and errors, and the history sent to
the provider. `tests/parity` compares these against fixtures produced by running
pi's own code (`scripts/pi_reference/generate.sh`; the pi version is in
`tests/parity/fixtures/pi_version.json`).

Matches pi:

- Tool declarations: names, descriptions, parameter schemas, prompt snippets and
  guidelines of all seven builtins.
- Tool arguments are converted and validated like pi's TypeBox schemas
  (`"2"` becomes `2`), with pi's error text. Malformed tool-call JSON becomes a
  tool error the model can retry.
- `read` (text and images, using pi's exact Photon WASM and model resize limits), `write`, `edit` (fuzzy matching, legacy argument
  shapes, per-file mutation queue, in-place writes), `bash` (`PI_*` session
  variables, shell lookup, timeout text, `pi-bash-*.log` full output),
  `grep`/`find`/`ls` (rg/fd, Node's sort order). Edit results carry pi's
  `details`: `diff`, `patch`, `firstChangedLine`.
- Tool batches follow pi's `executeToolCalls`: parallel by default, preparation in
  call order, no concurrency cap, `execution_mode = "sequential"` per tool, hook
  errors as error results.
- The system prompt: preamble, `<tools>`, `<rules>`, `<addendum>`,
  `<project_context>`, `<skills>` and `<cwd>`, in pi's order and wording. Context
  files are `AGENTS.override.md`, `AGENTS.md`, `AGENTS.MD`, `CLAUDE.md` or
  `CLAUDE.MD` (first per directory), global `~/.pi-agent/` first, then from `/`
  down to the cwd, skipping a main checkout's file shadowed by a nested worktree.
- Provider requests and stream decoding for Anthropic Messages and OpenAI-compatible
  Chat Completions: pi's `buildParams`, `convertMessages`, compat detection, beta
  headers, caching, thinking parameters, usage and cost, stop reasons, and
  `argumentsError` with V8's `JSON.parse` messages. Intermediate arguments follow
  partial-json 0.1.7; tool-call delta snapshots are compared with pi too. The model
  catalog is pi's.
- The provider history follows pi's `transformMessages`: other models' thinking and
  signatures are dropped, tool-call IDs are normalized per provider, failed or
  aborted responses are skipped, images become placeholders for text-only models,
  and calls without a result are answered with `"No result provided"`. The
  journal records no synthetic results.
- Compaction follows pi's `prepareCompaction` and `compact`: the cut point (chars/4
  estimate, cuts at user or assistant messages, split turns), conversation
  serialization, history, update and turn-prefix prompts, and the summary request
  (session thinking level, no cache writes, fresh routing session ID, no output
  cap), failure text and the appended file lists.
- Automatic compaction and retry follow pi's `_checkCompaction`,
  `_handlePostAgentRun` and `prepareNextTurn`: overflow and retry classification
  (`isContextOverflow`, `isRetryableAssistantError`), silent and length-stop
  overflow, recovery of truncated responses, one compact-and-retry per request,
  and usage invalidation by omissions and checkpoints.
- `thinking_budgets` is pi's `thinkingBudgets` (snake_case key, integer values).

Differs from pi:

- pi's `<docs>` prompt section points at pi's own documentation, which this
  package does not ship, so it is left out.
- No download of `rg`/`fd`; a container image must provide them.
- `ls` needs the ICU runtime. As with Node, Unicode order depends on the ICU
  version and process locale: use matching versions/locales for exact reproduction
  (`PI_ICU_LIBRARY` can select a particular library). CI pins ICU 77.1, matching
  Node 24.13.0 used for the fixtures. There is no approximate table
  fallback. Image processing ships pi's Photon binary and uses Wasmtime.
- Python task cancellation (not an abort) marks in-flight tool calls with an
  unknown-outcome result; pi has no equivalent.
- An after hook cancelled by an abort keeps the executed result.
- The skills catalog in the prompt is cut at `skills.prompt_max_bytes`.
- `details` of `read`, `bash` and the truncation records use snake_case keys.
- No OpenAI Responses, Google, Bedrock or other pi adapters; no OAuth. Request
  headers other than authentication, version, beta and session affinity (for
  example the user agent) are this port's own.
- SGLang thinking defaults, flat SGLang reasoning counts, and the
  temperature/seed/chat-template request settings are port extensions.
- An `aborted` summary response cancels the compaction (pi returns one only for
  an aborted signal, which then cancels). Checkpoints written by older versions
  of this port import file lists from their summary tags.
- No dynamic tool-loadout protocol: the leading system message declares the
  run's tool set once; resumed sessions announce a delta when the active set
  changed.
- Before/after hooks are injected Python callbacks; no extension system or custom
  message roles.
- The session journal is this port's own linear format (version 1, snake_case
  records, `context_omission` and `config_change` entries), not pi's version 3
  session tree; neither can open the other's files. Retries keep the failed
  attempt in the raw journal and omit it from the projection, where pi writes
  context-edit entries.
- Settings files are this port's own (`~/.pi-agent/settings.json`, snake_case
  keys); pi's `compaction.modelOverrides` is not supported.

## Scope

This is an independent Python implementation of pi's coding harness, meant to run
where Node.js is not installed. It uses its own journal format and settings files
and does not promise pi's API or configuration compatibility. TUI/web UI,
executable extensions, subagents, OAuth, session trees, RPC servers, OpenAI
Responses and MCP codemode/deferred exposure are outside the supported scope.
