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
- An unanswered call receives one recorded unknown-outcome result. Recovery never
  automatically executes the call again.

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
- Only a nonempty summary ending in successful `stop` may become a compaction
  checkpoint. A partial, aborted, tool-calling or length-limited summary cannot
  replace context. Both summaries of a split tool turn must succeed.
- Summary retries are bounded, cancellable provider requests with no tools.
  Reserve thresholds do not cap summary output; model and context limits do.
- Provider usage becomes stale after compaction or provider/API/model identity
  change. Until valid new usage arrives, context sizing uses a documented
  approximation. Per-request limits are not whole-run budgets.

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
- Read-only allows only verified builtin readers/search tools, subject to policy.
  Tool names or MCP annotations alone never grant admission.
- Parallel execution is bounded. Custom tools are barriers unless explicitly safe;
  builtin file operations on the same resolved path use FIFO queues.
- Atomic write/edit use same-directory temporary files, version checks, replace
  and fsync. Postcommit durability errors report that content may already be
  installed. These checks are not a cross-process transaction.
- Bash abort/timeout terminates its owned process group. Normal leader exit may
  leave background descendants alive until tool/session close. Descendants that
  escape the process group are outside ownership.
- Spill cleanup uses tool-registered file identity, never journal-supplied paths.
  Spill files are temporary and are not recreated by resume.

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
