# Tools and permissions

[Documentation](README.md) · [Project](../README.md)

## Repository search and tool selection

New `run` and `chat` sessions enable `read/bash/edit/write/grep/find/ls`.
`--tools read,grep,find,ls` selects an explicit set; `--tools ''` disables all tools.
The JSON settings field `tools` takes a list of builtin names, including an empty
list. CLI selection overrides file defaults; resume keeps its declared set unless
`--tools` is explicitly supplied. Duplicate/unknown names are errors.

Relative CLI entries edit that base instead: `--tools=+grep,-write` adds grep
and removes write. On resume, edits start from the recorded builtin set,
including an explicitly empty set. Entries must be either all absolute or all
signed, and each name may appear once. Use `--tools=-write` for a value beginning
with `-`. MCP selection remains separate through `--mcp-tools`; builtin edits do
not activate remote tools or bypass schema checks. SDK `tool_names` and settings
`tools` continue to take absolute builtin names.

The SDK retains its four-tool default. Use
`create_agent_session(provider, ..., tool_names=["read", "grep", "find", "ls"])`
or inject `create_builtin_tools(cwd)` to enable search. `create_coding_tools(cwd)`
still returns four tools. Selection can be made without `rg`; execution of
grep/find requires an installed `rg` on PATH, with an actionable error if absent.
The harness never downloads a search binary. `ls` needs no external executable.

`grep` supports regex/literal, `ignoreCase`, `glob`, context rows and a match
limit (default 100). `find` returns files matching a glob (default limit 1000);
`ls` returns a sorted directory listing including dotfiles and broken symlinks
(default limit 500). Both search tools discover unignored hidden files. Directory
search respects ignore rules even outside Git; nested Git roots stop inherited
Git ignore rules. Explicit grep file targets follow rg's explicit-file semantics.
`ls` does not filter `.gitignore`.

Globs without `/` match basenames; path globs are relative to the search root,
with `**` matching zero or more segments and `*` staying within one segment.
Character classes work; grep also accepts a negated glob. Brace expansion is not
supported. Unlike passing a positive rg `--glob` directly, these filters never
re-include files excluded by ignore rules. Grep streams context from rg, without
rereading matched files. Filtering follows ignore discovery; a narrow glob does
not necessarily narrow the underlying content scan.

Host caps: limit <=10000, context <=100, 30-second grep/find timeout,
1 MiB maximum rg record, 4 KiB retained stderr. Tool constructors expose output
budgets (defaults 2000 lines / 50 KiB; byte budget 512 bytes to 1 MiB) and search
timeout. Text, structured items and notices share the byte budget; notices count
against the line budget. Grep lines over 500 characters have a read notice.
Reaching a requested count conservatively reports the limit, including when it
equals the exact result count. A truncated final match may omit its trailing
context. Search does not spill full output; refine the query or use read.

Structured items preserve names with newlines independently of escaped display.
Non-UTF-8 filenames include `path_bytes_base64` (or `name_bytes_base64` for ls)
so hosts can recover exact bytes. Cancellation/timeout/output limits terminate
and join the owned subprocess, drain pipes and escalate to SIGKILL if needed.
Search child environments contain PATH/locale only; provider credentials are not
forwarded. No-match is successful; invalid regex and filesystem errors are explicit.
Concurrent external edits can still change files while search is running.

## Parallel tool scheduling

Use `AgentConfig(tool_execution="parallel", max_parallel_tools=4)` or the CLI
flags above. `session.set_tool_execution(mode, max_parallel_tools=...)` makes a
durable idle-session change. A bare `Agent` accepts the same construction options.

Built-in tools opt into parallel execution. Custom tools default to a barrier:
preceding work completes before the custom call, and following work waits for it.
Set `Tool.parallel_safe=True` only when concurrent execution is supported by the
implementation. Admission/before hooks and permission confirmation are serialized
within each parallel batch; after hooks follow each tool's actual completion.
Context snapshots contain the batch's starting history and prior barrier groups;
they do not contain concurrently completing results in the same group.

Read/write/edit calls to the same resolved path use FIFO locks shared between
harness batches on one event loop, including separate sessions and symlink aliases.
File waiters acquire the path queue before capacity, so they do not monopolize
slots needed for independent files. The queue covers admission, execution and
after processing. Locks are weakly retained after use. Bash commands and custom
tools with other names can modify arbitrary paths; the host must manage those
dependencies. These queues are not cross-process locks or protection against
external writers. Synchronous filesystem syscalls still block the event loop.

Start/update events may interleave, while final execution events, canonical
messages and persisted results follow the assistant's source order. A completed
tool keeps its known result during batch cancellation; a cancelled in-flight
execution gets an explicit unknown-outcome interruption result; a queued,
unstarted call gets a known unexecuted error. Every requested call remains paired.
Tasks are cancelled and joined before the batch settles; a fatal journal error
still stops the session, and unrecorded outcomes require inspection on recovery.

## Tool hooks and permissions

Both `Agent` and `create_agent_session` accept `hooks=ToolHooks(...)`,
`permission_policy=PermissionPolicy(...)`, and `confirm_permission=callback`.
For example, an embedding can restrict writes to `src` and require approval
for shell calls:

```python
from pi_agent import (
    AfterToolCallResult,
    PermissionPolicy,
    PermissionRule,
    ToolHooks,
)
from pi_agent.types import TextBlock


def before(context, cancellation):
    # Return BeforeToolCallResult(block=True, reason="...") to block a call.
    # Returning None continues admission through the permission policy.
    return None


def after(context, cancellation):
    if context.tool_call.name == "write" and not context.is_error:
        return AfterToolCallResult(content=[TextBlock(text="Write completed")])
    return None


policy = PermissionPolicy(
    default="deny",
    rules=(
        PermissionRule("read", "allow"),
        PermissionRule("write", "allow", path="src/**"),
        PermissionRule("edit", "allow", path="src/**"),
        PermissionRule("bash", "ask"),
    ),
)
# create_agent_session(provider, model=model, cwd=project, hooks=ToolHooks(before, after),
#                      permission_policy=policy, confirm_permission=confirm)
```

Callbacks receive `(context, cancellation)` and may return directly or await.
Before context contains the final assistant message, tool call, validated args,
projected messages (including earlier results in this batch), tool declarations
and the actual tool cwd when available. These are copied snapshots: mutating
args does not change execution. Before may return `BeforeToolCallResult` with
`block`, `reason`, and `terminate`; it cannot replace arguments or override a
permission denial. After context additionally exposes the executed `ToolResult`
and `is_error`; after also runs when execution raises an ordinary exception.
Unknown tools, invalid calls, truncated responses and denied calls skip after.

After may return `AfterToolCallResult`: omitted fields stay unchanged; explicit
`None` clears nullable `details`, `structured_content`, or `usage`. Use `content=[]`
to clear text. Replacing content clears structured content unless explicitly
supplied, matching Pi. `is_error` and `terminate` may also be replaced.
The merged result drives the final event and canonical tool-result message;
structured content and termination remain execution-event metadata in this
text-only transcript. Termination stops the next model request only when every
result in the batch requests it. No automatic raw-result audit accompanies a
successful override. Already emitted streaming updates cannot be redacted by after.

Before failure denies execution. After failure preserves the original executed
result in `details.original_result` and reports an error explaining that the tool
already executed. Cancellation during after retains the known result and its
original error flag. Pending callbacks are cancelled and joined; synchronous
blocking code or callbacks suppressing task cancellation are outside this guarantee.
`ToolExecutionStart` announces an attempted call before admission; denied calls
still get a paired result and `duration_ms=None`. Execution duration excludes
hooks and confirmation. Session lifecycle calls from callbacks or their child
tasks are rejected to prevent deadlock; use `cancellation.abort()` to stop the run.

Policy rules use **first match**, then `default`; glob comparisons are case
sensitive. Tool names are globs, `command` matches the raw Bash command without
shell parsing, and `path` matches the resolved tool target. Relative path rules
are based on the actual tool cwd (or agent cwd), including after a cwd override.
Literal path prefixes and targets resolve symlinks. Glob `*` crosses path
separators (`fnmatch` semantics); `**` is not a separate recursive-glob operator.
Patterns with a `..` path segment are refused. Read-only permits `read` and verified
builtin `grep/find/ls` implementations before applying rules; an explicit
rule/default may still deny them. A custom tool named grep/find/ls is not admitted
on the basis of its name. Omitted search paths resolve to the tool cwd during
admission, so path rules and confirmation also apply to default-directory searches.
It blocks Bash even for inspection commands. Custom tools and hooks remain trusted.
This is a dispatch policy, not process isolation: broad Bash patterns can authorize
arbitrary shell expressions, and concurrent filesystem changes after admission
remain possible. A changed symlink target during confirmation is refused.

Ask invokes `confirm_permission(PermissionRequest, cancellation)` and requires
an actual boolean; only `True` allows that call once. Requests include copied
context, reason and resolved path. Missing confirmation denies execution.
Policies persist on resume; explicit `permission_policy=` overrides are durable
`config_change` records. Older journals without a policy keep legacy `allow`;
malformed present policies are rejected. Hooks and confirmation functions must
be passed again on resume. Stored decisions do not grant future approvals.

CLI admission controls:

```bash
.venv/bin/python -m pi_agent run "Inspect source" --read-only
.venv/bin/python -m pi_agent run "Fix the bug" --permissions-file permissions.json --confirm-permissions
```

Example `permissions.json` (unknown fields and duplicate keys are rejected):

```json
{
  "default": "deny",
  "rules": [
    {"tool": "read", "action": "allow"},
    {"tool": "write", "action": "ask", "path": "src/**"},
    {"tool": "edit", "action": "ask", "path": "src/**"},
    {"tool": "bash", "action": "ask"}
  ]
}
```

`--permissions-file` replaces the stored policy; `--read-only` adds its restriction
to the chosen policy and persists it. `run --confirm-permissions` reads `/dev/tty`
with a cancellable event-loop reader and writes prompts to stderr, leaving JSONL
stdout clean and piped prompt input independent. Only `y`/`yes` approves;
blank input, refusal, EOF or absence of a terminal denies. SIGINT/SIGTERM during
confirmation settles the run with exit 130. A completed conversation may exit 0
after a denied tool call; JSONL tool results record the denial.

## Dependencies and shell behavior

- Runtime: Python >= 3.12 standard library plus pinned jsonschema, httpx, python-dotenv
  (versions pinned in `pyproject.toml`).
- The `bash` tool is Linux-first: `/bin/bash` (fallback `/bin/sh`), the
  child runs in its own session/process group (`start_new_session=True`),
  stderr is merged into stdout at the subprocess level (one pipe, one
  decoder — concurrent decoders could split multi-byte UTF-8). Timeout and
  cancellation kill the whole group (SIGTERM, then SIGKILL after a grace
  period) and reap it; descendants holding the inherited pipe after the
  leader exited get an idle drain window, extended by incoming data. Normal
  exit preserves background jobs until tool/session closure. Reader failures surface with the collected
  output. Tool paths resolve against the session cwd (`~` and absolute
  paths supported); the cwd is a base directory, not a sandbox, and the
  process working directory is never changed.
