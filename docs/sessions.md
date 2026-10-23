# Sessions, cancellation and compaction

[Documentation](README.md) · [Project](../README.md)

## Cancellation

`Cancellation.wait()` wakes every waiter when `abort()` is called. Model streams
and compaction race their next event against cancellation, so even a provider
that ignores the token is interrupted while waiting for a network chunk.
Retry backoff waits on the same event. `session.abort()` gives the active run
two seconds to settle, then cancels its task and awaits lifecycle cleanup.
Forced task cancellation propagates `CancelledError` to the prompt caller.
Providers and custom tools must allow normal asyncio cancellation to propagate;
synchronous blocking code cannot be interrupted by this mechanism.

The bash tool observes leader exit independently of pipe EOF and drains output
until the pipe closes or stays idle for one second. Each chunk re-arms the idle
deadline. Abort and timeout remain active during draining and kill the process
group; a normal exit keeps background descendants alive for later tool calls.
`session.dispose()` terminates the groups owned by its BashTool instances.
Full-output files (`pi-bash-*.log` in the temporary directory) stay, as in pi.
Standalone BashTool users call `tool.close()` themselves. Processes that start
a new session (`setsid`) are outside this ownership. All auxiliary tasks are
joined, and full-output files are closed even when finishing fails. File tools
check cancellation before and after their synchronous filesystem operations.

A tool call left without a result (a call never started, a crash) gets none in
the journal; each provider request answers it with `"No result provided"`, as pi
does. A tool interrupted by Python task cancellation while running is recorded
with an error result saying its outcome is unknown.

## Sessions

Core contracts are maintained in [core invariants](invariants.md).
`session.py` keeps the public facade and pi's retry/compaction decisions;
construction, projection and observer delivery live in `_session_*` modules.
`SessionStore.header`, `.entries`, `.get_entry()`, `session.messages` and
`build_projection()` return independent snapshots. Mutating them does not
change committed history. Session observers also receive independent events;
tool hooks remain the mechanism for controlling execution.

Projection caches and journal indexes advance only after a successful append.
Compaction and omissions preserve raw tool-call ID reservations. A checkpoint's
source position remains the retained boundary even if its anchor is omitted.
For an offline timing fixture, run
`.venv/bin/python scripts/benchmark_projection.py` (no provider or dotenv access).

`dispose()` rejects active work, including direct agent runs. Cleanup attempts
every owned resource and releases the journal lock even if one fails; errors are
grouped and another `dispose()` can retry failed cleanup.

One JSON object per line; first line is a versioned header (`version: 1`,
UUID, timestamp, cwd, model identity, non-secret config), then `message`,
`compaction`, `context_omission`, and `config_change` entries in event
order. The user prompt is journaled before its response is requested. The
raw journal is immutable and append-only; the model only ever sees a
projection.

- Durability: every append validates its record (same rules as load),
  then writes + flushes + fsyncs **before** the in-memory journal advances;
  a failed write poisons the store (`SessionUnusableError` on further
  appends) and propagates out of the agent as a fatal
  `FatalPersistenceError` — it is never synthesized or retried as a model
  error, including from compaction. `os.write` making no progress is an
  error, not a loop. A persisted session's file is created (empty and
  locked) at `create()`; the header and entries are written once the first
  user/assistant message makes the conversation real.
- Validation: strict at load and at append — valid header first with the
  exact version, valid message shapes (roles, stop reasons, blocks,
  tool-call IDs, finite numbers), unique entry IDs, compaction checkpoints
  pointing at an eligible earlier user/assistant message (never at
  themselves, another checkpoint, an omission, a config record, or a tool
  result), and omission targets that are message entries. Violations raise
  `SessionCorruptError` with the line number.
- Crash recovery: uncommitted bytes after the last newline (invalid JSON or
  a torn multi-byte character) are truncated away **under the writer lock,
  exactly once**; a trailing fragment that parses as a complete record is
  validated and sealed with its missing newline — the durable write happens
  before any in-memory state advances. An unterminated header line is
  adopted when the file has no header yet. Recovery never turns corrupt
  bytes into interior corruption, and repeated opens are stable.
- Contention: both `create` and `open` hold an exclusive `flock` for the
  store's lifetime, taken *before* the file is read; a second writer gets
  `SessionLockError`, and locks are released when construction fails.
- Interrupted tool batches: never replayed. Unanswered tool calls get no
  journal record; the provider projection answers them with
  `"No result provided"` (pi's `transformMessages`). Journals written by
  earlier versions may hold recorded interruption results; the projection
  still places them, with any real results, in one contiguous batch per
  assistant in call order. Every finalized tool-call ID is unique across **raw session
  history**, including omitted attempts and pre-compaction entries. The loop
  reassigns conflicting IDs before persistence and execution, preserving the
  original IDs in `AssistantMessage.provider_data.pi_original_tool_call_ids`.
  Journal validation still rejects duplicate canonical IDs. Malformed names or
  non-object arguments produce an explicit assistant error and never execute.
- Failed-attempt retries: before a transient error, context overflow or
  recoverable length stop is retried, the failed assistant attempt and its tool
  results are durably omitted from the projection (`context_omission` records;
  the raw history keeps them), so the retry request leaves from the last
  user/tool-result message. Reused provider
  IDs are reassigned, including IDs reserved by omitted attempts.
  Completed tools are never re-executed, and an unrecordable omission stops
  the session instead of risking a blind retry.
- Config changes: explicit model/config/cwd overrides on a resumed session
  are persisted as `config_change` records (the latest one wins over the
  header on reopen), after the constructor fully validated.
- No tree/fork/bookmark functionality — the journal is linear.

## Compaction

Automatic compaction follows pi's `_checkCompaction`. It runs after each agent
run and before each new prompt (on the last response), and as a threshold check
before every further turn inside a run. A context overflow (error text, or a
completed response whose input exceeds the window), or a length stop below the
model's own output limit, omits the failed attempt, compacts and re-requests
once; a completed overflowing answer is compacted without a retry. Responses
from another model, or older than the latest checkpoint, trigger nothing. When
projected context exceeds `context_window - reserve_tokens`, the older history is
summarized through the same injected provider — with **no executable tools**
— and only a *completed* checkpoint is persisted: the summary, the first
kept entry, the replayed system message (prompt + tool declarations), and
the summarizer usage. Errors, length stops, aborts, tool-call attempts and
unterminated streams fail the compaction without touching prior context, with
pi's messages (`Summarization failed: ...`, `... hit the token cap and the
summary is incomplete`). The summary format
(Goal / Constraints / Progress / Key Decisions / Next Steps / Critical
Context) is preserved on update so task goals are not weakened; file paths,
names, and error messages are required to survive.

Summary input excludes thinking while retaining visible messages, tool calls and
results. The raw journal keeps original thinking. `reserve_tokens` controls when
compaction starts, not its output cap: summaries send no `max_tokens`, so the
provider applies the model's limit. As in pi, they use the session's thinking
level on reasoning models (none otherwise), disable prompt-cache writes, and get
a fresh routing session ID per request.

Summary requests use the session's bounded transient-error retry policy and
cancellable backoff. They never execute tools or record failed summary attempts.
Length stops, invalid summaries, billing errors and context overflow are not blindly
retried. Compaction start/end events cover the entire operation; conversational
auto-retry events are not emitted for individual summary attempts.

The cut point is pi's: walking back from the newest entry with a chars/4
estimate until `keep_recent_tokens`, it lands on the nearest user or assistant
message, never a tool result. A cut at an assistant message splits its turn: the
older history and the turn prefix receive separate summaries (the prefix with
pi's turn-prefix prompt), combined with a Turn Context section and summed usage.
A failure in either request prevents the checkpoint. Tool calls and their results
remain together in the retained tail. File-operation lists accumulate across
checkpoints as typed `read_files` / `modified_files`, including empty arrays,
independent of model-written tags. Legacy checkpoints import their existing
summary tags; new ones use their recorded lists. These lists track requested
file operations, as Pi does, and are not proof that a denied operation changed
the filesystem. The journal version remains 1 with backward-compatible optional
fields; no tool-result checkpoint pointer is introduced.

Overflow recovery performs at most one compact-and-retry per failing request;
there are no compaction/retry cycles, and the single retry layer (in the
session) never re-runs completed tools. With compaction disabled, overflows are
not recovered. When there is nothing to compact, automatic compaction does
nothing and emits no events. The request projection is re-read after any
checkpoint; a threshold compaction after a completed final answer only starts
another run for queued input. Token estimates follow pi's
`estimateProjectedContextTokens`: provider usage counts only when recorded after
the latest omission or checkpoint (a model switch keeps it), otherwise pi's
chars/4 approximation applies — sizes are approximate, not exact token
accounting.
