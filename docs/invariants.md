# Core invariants

[Documentation](README.md) · [Project](../README.md)

These are implementation contracts for maintainers. Changes to persistence,
resource ownership, cancellation or admission require corresponding regression
checks. See [SDK](sdk.md), [sessions](sessions.md), [tools](tools.md) and
[MCP](mcp.md) for the detailed public behavior.

## Durable state before runtime publication

- The linear append-only JSONL journal uses the Python harness's own version 1
  format. It is not the TypeScript Pi session format.
- Persisted records are validated, fully written and fsynced before in-memory
  indexes or runtime configuration advance. Failed persistence poisons the store
  and stops execution; it is not converted into a provider error or retried.
- Before the first conversation message, system/config records may remain pending
  in memory. The first user/assistant append commits them before execution.
- Input objects and public snapshots are copied. Mutating returned messages,
  entries, declarations or events must not change committed state.

## Recovery and identity

- One writer holds an advisory `flock` throughout a persisted session's lifetime.
  Resume requires an existing file and acquires the lock before reading it.
- Only an uncommitted trailing fragment may be recovered. Interior corruption,
  invalid records, duplicate IDs or broken references are errors.
- Canonical tool-call IDs are unique across raw history, including omissions and
  pre-compaction messages. Conflicting provider IDs are reassigned before commit.
- A tool interrupted by task cancellation while running receives one recorded
  unknown-outcome result. A call left without any result gets no record; each
  request answers it with `"No result provided"`. Recovery never executes a
  call again.

## Projection, retries and compaction

- Providers receive a projection; raw journal history stays append-only.
- Cross-model thinking and summary-only thinking removal affect requests, never
  rewrite historical messages. Same-model native replay remains intact.
- Omissions and checkpoints retain raw ID reservations. Tool calls/results remain
  adjacent and ordered in the provider projection.
- Projection indexes/caches advance only after successful appends. An omitted
  checkpoint anchor does not move the original retained boundary.
- Retry omits a failed attempt durably before requesting another response.
  Completed tools are never replayed. Overflow allows one compact-and-retry cycle
  per failing request, with no recursive recovery loop.
- An error, aborted, tool-calling or length-limited summary cannot become a
  compaction checkpoint (pi's `getSummarizationFailure`). Both summaries of a
  split turn must succeed.
- Summary retries are bounded, cancellable provider requests with no tools.
  Reserve thresholds do not cap summary output; the model's limit does.
- Provider usage becomes stale after a compaction or an omission. Until valid
  new usage arrives, context sizing uses pi's chars/4 approximation.
  Per-request limits are not whole-run budgets.

## Lifecycle and event delivery

- Prompt, compaction and configuration operations have one active-session guard.
  Steering/follow-up queues do not interrupt a tool batch.
- Final event order is persistence, state reduction, then awaited observer
  delivery. Observers receive independent snapshots; failures are isolated.
- Lifecycle calls from observers, tool callbacks or their child tasks are rejected
  to prevent reentry deadlocks. Unrelated caller tasks remain independent.
- Cancellation joins owned work. Synchronous filesystem calls cannot be
  interrupted mid-syscall; callbacks must cooperate with asyncio cancellation.

## Tools and filesystem effects

- Schema validation and admission precede execution. Before/confirmation callbacks
  are serialized; an after hook may replace a result but cannot undo an operation.
  A raising hook turns the result into an error with the exception text.
- Read-only allows only verified builtin readers/search tools, subject to policy.
  Tool names or MCP annotations alone never grant admission.
- Parallel batches follow pi: preparation in call order, then unbounded concurrent
  execution; a tool with `execution_mode="sequential"` makes the batch sequential.
  `write`/`edit` on the same resolved path use a FIFO queue.
- `write`/`edit` write in place, as pi does; there is no cross-process locking.
- Bash abort/timeout terminates its owned process group. Normal leader exit may
  leave background descendants alive until tool/session close. Descendants that
  escape the process group are outside ownership.
- Bash full-output files stay after close, as in pi. MCP spill cleanup uses
  registered file identity, never journal-supplied paths.

## Resource ownership and integrations

- Injected providers/custom resources are borrowed unless explicitly registered.
  Owned resources are registered before awaited initialization.
- Async close joins cleanup in reverse ownership order. Successful resources are
  released; failures remain retryable. Caller cancellation waits for settlement.
  Registered resources must provide bounded, idempotent close.
- Skills load from explicit/discovered trusted scope. Metadata grants no tool
  permissions. Reload commits catalog and prompt before publication; ordinary
  resume preserves the recorded snapshot.
- MCP configuration/credentials remain runtime-only. Journals store binding
  identity and declaration/schema digests, never connection authority.
- MCP resume connects recorded bindings from trusted configuration. Changed
  contracts require explicit refresh; reconnect never replays interrupted calls.
- MCP result envelopes preserve absent/null and error payloads; model text limits
  never truncate the structured envelope. Only top-level result `_meta` is removed.
- MCP catalogs, framing, output and work are bounded. Staged replacements validate
  before one durable commit; postcommit cleanup failure does not roll it back.

## Execution boundary

Admission policies are application-level controls, not OS, network or filesystem
sandboxes. The embedding host owns isolation, external side effects, total time
and spending limits. Offline tests establish their tested contracts, not model
quality or live-account compatibility.
