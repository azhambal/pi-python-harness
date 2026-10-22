# Python SDK

[Documentation](README.md) · [Project](../README.md)

Use the SDK to embed a session in your own application. The example below assumes
you provide `MyProvider`; the [project quick start](../README.md#python-api)
shows the bundled HTTP adapter. The SDK does not load dotenv automatically.

```python
import asyncio
from pi_agent import ModelInfo, create_agent_session

from my_provider import MyProvider  # your implementation of the Provider protocol


async def main() -> None:
    session = create_agent_session(
        MyProvider(),
        model=ModelInfo(
            id="example-model",
            name="Example Model",
            provider="example",
            api="example",
            context_window=128_000,
            max_tokens=8_192,
        ),
        cwd="/tmp/project",
        session="new",  # None = in-memory, a path = resume that session file
    )

    unsubscribe = session.subscribe(lambda event: print(event.type))
    await session.prompt("List the Python files in this directory and summarize each one.")
    unsubscribe()

    await session.aclose()  # rejects active work — abort() first if needed


asyncio.run(main())
```

`MyProvider` implements `pi_agent.Provider`:

```python
from pi_agent import Provider, ProviderOptions, ProviderRequest
from pi_agent.events import StreamDone


class MyProvider:
    async def stream(self, request: ProviderRequest, options: ProviderOptions):
        # Build the provider-native request from request.model,
        # request.messages (system prompt and tool declarations live in the
        # leading system message), and request.tools; then yield
        # AssistantStreamEvent objects and finish with a terminal event:
        ...
        yield StreamDone(reason="stop", message=final_message)
```

The contract: `stream()` must not raise for request/model/runtime failures —
encode them as a terminal `StreamError` carrying a final `AssistantMessage`
with `stop_reason` `"error"` (or `"aborted"` if `options.cancellation` fired)
and an `error_message`, preserving any partial content. Every non-terminal
event carries the partial response snapshot. The loop may close the iterator
early (`aclose`); implement `stream` as an async generator and this is
automatic.

Key entry points:

| API | Purpose |
| --- | --- |
| `create_agent_session(provider, *, model=None, cwd=None, session=None, tools=None, tool_names=None, config=None, refresh_instructions=False)` | Build a session (`session=None` in-memory, `"new"` persisted, a path resumes) |
| `await session.prompt(text, streaming_behavior=...)` | Run a prompt; queue as `steer`/`followUp` while streaming |
| `session.steer(text)` / `session.follow_up(text)` | Queue input mid-run / for after the run |
| `session.subscribe(listener)` | Agent + session events (sync or async listeners) |
| `await session.abort()` / `await session.wait_for_idle()` | Cancel and settle |
| `await session.compact(instructions)` | Manual compaction through the provider |
| `session.messages` / `session.stats` | Current projected transcript / counters |
| `session.dispose()` | Synchronous teardown; rejects active work |
| `Agent(provider, model=..., tools=...)` | The loop without session persistence |

### Event delivery and listener contract

Agent events are staged: the durable journal write happens first (fatal on
failure), agent state is reduced second, and only then are observers
notified — session observers through one ordered dispatcher, with each
event's delivery **awaited before the run advances**. Events reach listeners
in emission order, listeners run in subscription order, async listener
results are awaited, and slow listeners throttle the run instead of
accumulating unbounded queued snapshots. Observer failures — and
independently cancelled observer tasks — are isolated and never invalidate
persisted messages; cancellation of the actual caller propagates. Input an
observer enqueues during `agent_end` delivery (e.g. a follow-up) is visible
before the run's post-run handling decides whether to continue, and the
`agent_settled` event is delivered before `wait_for_idle()` returns.

Listeners must not call session lifecycle methods (`prompt`/`compact`/
`abort`/`wait_for_idle`/`dispose`/`aclose`) — or `Agent.wait_for_idle()` —
re-entrantly from inside delivery; that would deadlock the run and is
rejected with a clear error. The guard is task-local (a `ContextVar`
inherited by tasks a listener creates), so unrelated external tasks calling
these methods while a listener awaits are unaffected.

### Async resource ownership

`create_agent_session()` remains the synchronous core factory. `await
create_harness()` adds optional asynchronous tool preparation and transfers a
`ResourceOwner` to the resulting session. Injected providers/custom tools remain
borrowed unless registered explicitly; existing BashTool cleanup is retained.

```python
from pi_agent import ResourceOwner, create_harness

owner = ResourceOwner()

async def prepare_tools(resources):
    resources.add(connection)  # register before the first connection await
    await connection.connect()
    return host_tools  # complete injected set; includes any desired builtins

session = await create_harness(provider, config=config,
                               resource_owner=owner, prepare_tools=prepare_tools)
try:
    await session.prompt("Review the project")
finally:
    if not session.is_idle:
        await session.abort()
    await session.aclose()
```

The host supplies `connection`, `host_tools`, `provider`, and `config`; MCP configuration is explicit; see the [MCP guide](mcp.md). `prepare_tools` cannot
be combined with `tools`/`tool_names`. Construction failure or cancellation closes
registered resources. Supply an explicit owner to retry cleanup if construction
and cleanup both fail; otherwise the callback must retain it for that purpose.
After transfer, do not reuse an owner for another harness or close it independently.

`await session.aclose()` requires idle (abort first); `async with session` aborts
active work on exit and then closes. Concurrent close callers share one operation.
Cancellation of a waiter propagates only after cleanup has settled; registered
resources must implement bounded, idempotent `aclose()`. Cleanup runs in reverse
registration order and attempts all resources. Successful resources are removed;
failures remain retryable. The journal lock and Bash spills are cleaned even when
an async resource fails. Observer tasks are joined, including those stopped by a
prior sync dispose. Session lifecycle reentry from cleanup callbacks is rejected,
including child tasks inheriting that callback context.

Sync `dispose()` remains available for core sessions. Pending async resources
make it fail before teardown and direct the caller to `aclose()`. Run/chat exit
paths now use async close. Neither path infers resource ownership from journal
paths, recorded declarations, or injected tool names.

## Resuming a session

When `session` is a path, the stored cwd, model identity, configuration,
prompt, and tool declarations are the defaults:

- Tools are rebuilt against the *stored* cwd; a declared custom toolset
  requires `tools=` providing matching implementations (unknown declared
  tools are an error, and an explicitly empty stored declaration stays
  empty rather than expanding to current defaults). All seven builtins can be
  restored without injection. Explicit `tool_names=` replaces the active builtin
  set, committing tool-derived prompt sections and declarations in one record;
  other stored instructions remain intact. `tools=` and `tool_names=` are exclusive.
- The stored prompt is used as-is; AGENTS.md is only reread (and the prompt
  sections re-derived) with `refresh_instructions=True`.
- `model=None` adopts the stored identity; passing `model` requires it to
  match the stored one. `config.model`/`config.cwd` contradicting the
  effective model/cwd is an error, not a silent override.
  To change models, open the session and call `set_model` while idle.
- Passing `cwd` or `config` overrides the runtime values; on a resumed
  session the resulting effective configuration is persisted as a durable
  `config_change` record (after the whole constructor validated) and
  replayed on the next open. An explicit cwd override also patches the
  persisted prompt's `cwd` section without rereading AGENTS.md.
- `SessionStore.open` requires the file to exist — resuming a mistyped path
  is an error, not a silent new session.

## Model and thinking changes

`session.set_model(model, *, provider=None, thinking_level=None)` and
`session.set_thinking_level(level)` are synchronous methods for idle sessions.
They reject active prompts, compaction, direct agent runs, disposed sessions,
callback reentry and unusable journals. `session.model` returns a copy;
`session.thinking_level` reports the effective level.

```python
from pi_agent import ModelInfo

# After await session.wait_for_idle():
session.set_model(
    ModelInfo(
        id="another-model",
        name="Another model",
        provider="deepseek",
        api="openai-completions",
        reasoning=True,
        context_window=128_000,
        max_tokens=8_192,
    ),
    thinking_level="high",
)
session.set_thinking_level("low")
# A provider/API change requires the new configured transport explicitly:
# session.set_model(anthropic_model, provider=anthropic_provider, thinking_level="high")
```

Within the same provider/API, the current transport is reused. A provider/API
change requires `provider=`; it replaces the transport used for conversation
and compaction. A provider-only replacement for the same model is runtime-only.
The harness does not close replaced providers or caller-owned HTTP clients.
Credentials/endpoints/functions are not stored: resume requires an injected
provider configured for the latest model. Constructor `model=` still verifies
the recorded identity; use the setter for an intentional change.

Model fields and thinking enum are validated before writing. Setters retain
thinking unless explicitly supplied; `reasoning=False` clamps it to `off`,
matching Pi's non-reasoning behavior. `set_thinking_level` returns the effective
level. No model catalog or per-model thinking defaults exist; hosts declare
`ModelInfo.reasoning`, and the adapters map accepted effort names. Model-specific
server restrictions may still produce request errors.

Providers may implement the optional `ModelValidator` protocol:
`validate_model(model, thinking_level) -> None`. It is a synchronous, local check
used by setters; it must not perform network I/O or mutate the session. Omit it
for a custom provider whose compatibility the host manages. Bundled adapters
validate API compatibility; Anthropic budget mode also requires an output cap
above 1024 when thinking is enabled. These checks do not verify account access
or model availability. HTTP payload construction checks transport compatibility
too; session construction preserves stored settings and performs data validation.

A change writes one complete `config_change` before updating config, agent and
provider. On a validation/append failure runtime state stays unchanged; a failed
write poisons the journal, so reopen/recover before further work. After a write
or fsync failure, disk recovery may find a fully written record even though
durability was not confirmed to the caller. Exact no-ops create no records or
change events. Model/limit and thinking changes emit `model_changed` and
`thinking_level_changed` with previous/new values. Sync setters queue these for
ordered observer delivery; `await session.wait_for_idle()` settles pending events.

History, tool declarations, instructions, permissions and queued input are
preserved. Provider normalization removes foreign thinking signatures only from
the request copy; returning to a previous identity can replay its original signed
blocks. Prior token usage becomes untrustworthy after a provider/API/model
identity change, including a return to an earlier model. Context estimation uses
the heuristic until the new model produces valid usage. Same-identity limit or
thinking changes retain trustworthy usage; compaction uses the new context limit
and transport, with its existing thinking-off summarization policy.

CLI example:

```bash
.venv/bin/python -m pi_agent run "Continue the task" --continue --model deepseek-flash --thinking high
.venv/bin/python -m pi_agent run "Continue the task" --resume /tmp/session.jsonl --max-tokens 4096
```

Explicit `--provider` can switch transports, using the target provider's runtime
credentials and `--model` (or its configured model). Without explicit flags,
resume keeps stored settings even when environment defaults change. A limit-only
update preserves the recorded model/capabilities and unspecified limit. A new
identity gets a context window of 0 unless `--context-window` is supplied and
the usual output cap default; the old model's window is not assumed for it.
No API call is issued to probe a switch; the next prompt uses the new settings.
