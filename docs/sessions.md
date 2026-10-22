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
`session.dispose()` terminates the groups owned by its BashTool instances
and removes their spill files. Spill paths are temporary: copy output needed
for later inspection before disposal. Journal records retain the paths and
recorded output, but reopening a disposed session does not recreate those files.
Standalone BashTool users call `tool.close()` themselves. Processes that start
a new session (`setsid`) are outside this ownership. All auxiliary tasks are
joined, and spill files are closed even when finishing fails. File tools check
cancellation before and after their synchronous filesystem operations.

After a run settles, any tool call that ended up unanswered is answered with
a recorded interruption result (see below), so the transcript stays
provider-valid.

## Sessions

Core contracts are maintained in [core invariants](invariants.md).
`session.py` keeps the public facade; construction, projection, recovery,
recovery decisions and observer delivery live in `_session_*` modules.
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
all owned BashTools and releases the journal lock even if deletion fails;
errors are grouped and another `dispose()` can retry failed cleanup. Only
files registered by the tool during creation are removed, with device/inode
checks; journal-supplied paths are never ownership evidence. A process crash
without disposal leaves spill cleanup to the temporary-directory lifecycle.
Standalone `OutputAccumulator` without a registry is caller-owned.

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
- Interrupted tool batches: never replayed. Unanswered tool calls of any
  assistant message receive `is_error` results that state the outcome is
  **unknown** ("the call may or may not have run to completion; inspect the
  affected files before re-issuing"). The results are recorded at the end of
  the raw journal and the projection builds one contiguous result batch per
  assistant in call order — real and interruption results together — so
  pairing is adjacent, ordered, and each call ID has exactly one projected
  result. Every finalized tool-call ID is unique across **raw session
  history**, including omitted attempts and pre-compaction entries. The loop
  reassigns conflicting IDs before persistence and execution, preserving the
  original IDs in `AssistantMessage.provider_data.pi_original_tool_call_ids`.
  Journal validation still rejects duplicate canonical IDs. Malformed names or
  non-object arguments produce an explicit assistant error and never execute.
- Failed-attempt retries: before a transient error or context-overflow error
  is retried, the failed assistant attempt is durably omitted from the
  projection (`context_omission` record; the raw history keeps it), so the
  retry request leaves from the last user/tool-result message. Reused provider
  IDs are reassigned, including IDs reserved by omitted attempts.
  Completed tools are never re-executed, and an unrecordable omission stops
  the session instead of risking a blind retry.
- Config changes: explicit model/config/cwd overrides on a resumed session
  are persisted as `config_change` records (the latest one wins over the
  header on reopen), after the constructor fully validated.
- No tree/fork/bookmark functionality — the journal is linear.

## Compaction

When projected context exceeds `context_window - reserve_tokens` (checked
before every provider request and before each new prompt), or when a
response fails with a context-overflow error, the older history is
summarized through the same injected provider — with **no executable tools**
— and only a *completed* checkpoint is persisted: the summary, the first
kept entry, the replayed system message (prompt + tool declarations), and
the summarizer usage. Only a stream that terminates with `done` and a final
`stop_reason` of exactly `"stop"` may become a checkpoint; errors, length
stops, aborts, tool-call attempts, empty summaries, and unterminated streams
all fail the compaction without touching prior context. The summary format
(Goal / Constraints / Progress / Key Decisions / Next Steps / Critical
Context) is preserved on update so task goals are not weakened; file paths,
names, and error messages are required to survive.

Summary input excludes thinking while retaining visible messages, tool calls and
results. The raw journal keeps original thinking. `reserve_tokens` controls when
compaction starts, not its output cap: summaries use the model's output limit
(8192 when unknown), reduced for estimated available context. They request thinking
off, or the lowest explicitly supported level if the model cannot disable it.

Summary requests use the session's bounded transient-error retry policy and
cancellable backoff. They never execute tools or record failed summary attempts.
Length stops, invalid summaries, billing errors and context overflow are not blindly
retried. Compaction start/end events cover the entire operation; conversational
auto-retry events are not emitted for individual summary attempts.

Long tool turns can be split at a retained assistant message: the older history
and the turn prefix receive separate summaries, combined with a Turn Context
section and summed usage. A failure in either request prevents the checkpoint.
Tool calls and their results remain together in the retained tail. Simple
completed exchanges are kept whole. File-operation lists accumulate across
checkpoints as typed `read_files` / `modified_files`, including empty arrays,
independent of model-written tags. Legacy checkpoints import their existing
summary tags; new ones use their recorded lists. These lists track requested
file operations, as Pi does, and are not proof that a denied operation changed
the filesystem. The journal version remains 1 with backward-compatible optional
fields; no tool-result checkpoint pointer is introduced.

Overflow recovery performs at most one compact-and-retry per failing request
(completed answers and generic output-length stops are not overflow
retries); there are no compaction/retry cycles, and the single retry layer
(in the session) never re-runs completed tools. The request projection is
re-read after any checkpoint; a threshold compaction after a completed final
answer never triggers another request on its own. Token estimates use
provider-reported usage only from the last *valid* response recorded after
the latest checkpoint (pre-checkpoint usage is discarded even when its
message is retained), falling back to an explicit chars/3.5 approximation —
sizes are approximate, not exact token accounting.
