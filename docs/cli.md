# Command-line interface

[Documentation](README.md) · [Project](../README.md)

```text
pi-agent --help
pi-agent --version
pi-agent list [--cwd DIR] [--session-dir DIR] [--limit N]
pi-agent show PATH [--entries N]
python -m pi_agent ...        # same entry point
```

`list` and `show` inspect session files read-only: no lock, no file
creation, no repair writes; any uncommitted trailing bytes are reported as
corruption (only the writer path recovers them). Inspection never calls a
model. Diagnostics go to stderr; inspection output to stdout.
Live runs:

```bash
.venv/bin/python -m pi_agent run "Fix the failing test and rerun pytest"
.venv/bin/python -m pi_agent run "Continue the task" --continue
.venv/bin/python -m pi_agent run "Check the changes" --resume /path/session.jsonl
.venv/bin/python -m pi_agent run "Inspect the project" --json --no-session
.venv/bin/python -m pi_agent chat --continue
printf 'Explain the failing tests' | .venv/bin/python -m pi_agent run
```

`--provider`, `--model`, `--base-url`, `--max-tokens`, `--http-timeout`,
`--context-window`, and `--thinking` configure runs. OpenAI-compatible providers
also accept `--temperature`, `--seed`, and `--chat-template-kwargs`. `--session-dir` chooses
where new sessions are stored or where `--continue` discovers them. New runs
persist by default; `--no-session` keeps the journal in memory. Resume retains
the stored provider/model/limits/thinking by default. Explicit model/provider,
thinking or limit flags durably update the resumed session before its next prompt;
`--cwd` overrides cwd separately. Discovery
counts committed lines, and resume recovers a crashed tail under the writer
lock. `show` remains strictly read-only.

Print mode streams assistant text. `--json` emits `event_to_json` JSONL on
stdout, including tool lifecycle events; diagnostics remain on stderr. Exit
codes: 0 for completion, 1 for terminal provider failure or a final truncated
answer, 2 for setup/persistence failures, 130 for interruption. A truncated
tool call is never executed; it receives an error result, after which the model
may re-issue a complete call, matching the core Pi behavior.

HTTP read timeout is an inactivity limit, not a whole-run or spend budget.
There is no model catalog: auto-compaction by token threshold requires a known
`--context-window`; without it the context size defaults to 0 (unknown).
`--max-tokens` defaults to 8192 per request and does not bound a whole tool loop.

Use `--tools=+grep,-write` to edit defaults, or the recorded builtin set on resume.
An absolute list replaces the builtin set; `--tools=''` clears it. MCP tools use
their separate selectors. See [tool selection](tools.md#repository-search-and-tool-selection).

## Chat REPL

```bash
.venv/bin/python -m pi_agent chat
.venv/bin/python -m pi_agent chat --continue --json
```

Commands: `/steer TEXT`, `/follow TEXT`, `/abort`, `/wait`, `/compact [FOCUS]`,
`/model [provider/]ID`, `/thinking LEVEL`, `/parallel sequential|parallel [LIMIT]`,
`/status`, `/help`, `/quit` (or `/exit`). Plain text entered during a run queues
a follow-up; while idle it starts a prompt. Model/thinking/scheduling changes
require idle state. `/model` resets the context limit to unknown for a new ID;
cross-provider changes configure an explicit target transport from runtime env.

Input remains live while the model, tools or compaction wait. Ctrl-C aborts active
work; Ctrl-C while idle, a repeated interrupt, or SIGTERM exits with cleanup.
`/quit` aborts pending work; EOF waits for the current work to settle. Input readers
and model waiters are joined; the caller-owned stdin descriptor is not closed.
Piped input and regular files are supported, including split UTF-8 chunks.

With `--confirm-permissions`, chat routes answers through its single REPL reader
and prompts on stderr; y/yes approves one call, blank/no/EOF denies. `/abort` and
`/quit` remain available during confirmation. `/wait` yields back to input when
an approval is needed, so it cannot hide the confirmation behind a blocked wait.
The print-mode `run` continues to use `/dev/tty`. In chat JSON mode, help/status
and permission prompts also go to stderr; stdout contains JSONL lifecycle events.

See [providers](providers.md) for credentials and [configuration](configuration.md) for settings files.
