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
still returns four tools.

`grep`, `find` and `ls` are line-by-line translations of pi's tools: the same
descriptions, parameter schemas, output text and error messages, checked against
pi in `tests/parity`. `grep` runs ripgrep (`rg --json`), `find` runs `fd --glob`,
and `ls` sorts entries the way Node's `localeCompare` does. Like pi, the harness
looks for `rg` and `fd` (or Debian's `fdfind`) in `~/.pi-agent/bin`, then on
`PATH`, but never downloads them: a container image must ship both. Selection
works without them; a call reports pi's "not available" error.

- `grep`: regex or `literal`, `ignoreCase`, `glob`, `context` lines and `limit`
  (default 100 matches). Rows are `path:line: text`, context rows `path-line- text`;
  lines over 500 characters are cut with `... [truncated]`.
- `find`: glob over files and directories (directories end in `/`), default
  limit 1000. A pattern containing `/` matches the full path.
- `ls`: one directory, dotfiles included, directories marked `/`, entries that
  cannot be stat'ed (broken symlinks) skipped, default limit 500.

Outside a Git repository fd keeps honoring `.gitignore` (`--no-require-git`);
ripgrep, as in pi, does not. Output is cut at 50 KiB with a notice, and a reached
limit adds a notice suggesting a larger `limit`. The order of grep and find rows
follows the parallel directory walk and is not stable between runs. Search
processes inherit the environment, as in pi. Cancellation stops the process group
(SIGTERM, then SIGKILL after one second) and reaps it.

## Parallel tool scheduling

`tool_execution` is `"parallel"` (default, as in pi) or `"sequential"`; set it with
`AgentConfig(tool_execution=...)`, `--tool-execution`, or
`session.set_tool_execution(mode)` on an idle session. A bare `Agent` accepts the
same option.

The batch follows pi's `executeToolCalls`. In parallel mode every call is
prepared in source order (argument validation, before hook, permission check),
then all prepared calls run at once with no concurrency cap. A tool with
`execution_mode = "sequential"` makes the whole batch sequential. Before hooks
see the history up to the assistant message; results of the current batch reach
the context only after the whole batch, as in pi.

`write` and `edit` serialize mutations of one file through a queue keyed by the
resolved real path, shared across sessions on one event loop; the wait happens
inside the tool, as pi's `withFileMutationQueue` does. Bash commands and custom
tools can modify arbitrary paths; the host must manage those dependencies. The
queue is not a cross-process lock.

`ToolExecutionEnd` events arrive as calls finish; result messages and the
journal follow the assistant's call order. On abort, a sequential batch stops
after the running call and a parallel batch stops preparing further calls; calls
left without a result get none in the journal, and the provider request answers
them with `"No result provided"`, as pi's `transformMessages` does. Cancelling
the Python task (not an abort) marks in-flight calls with an unknown-outcome
result.

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
projected messages (without results of the current batch), tool declarations
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

A before or after hook that raises turns the result into an error whose text is
the exception message, as in pi (the after case replaces the executed result).
Cancellation during after retains the known result and its original error flag. Pending callbacks are cancelled and joined; synchronous
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

- Runtime: Python >= 3.12 standard library plus pinned jsonschema, httpx, python-dotenv,
  PyYAML, pathspec and Pillow (versions pinned in `pyproject.toml`); `rg` and `fd`
  for grep/find.
- The `bash` tool is Linux-first: the shell is pi's choice (`/bin/bash`, then
  `bash` on PATH, then `sh`), the
  child runs in its own session/process group (`start_new_session=True`),
  stderr is merged into stdout at the subprocess level (one pipe, one
  decoder — concurrent decoders could split multi-byte UTF-8). Timeout and
  cancellation kill the whole group (SIGTERM, then SIGKILL after a grace
  period) and reap it; descendants holding the inherited pipe after the
  leader exited get an idle drain window, extended by incoming data. Normal
  exit preserves background jobs until tool/session closure. Reader failures surface with the collected
  output. Truncated output is saved to `pi-bash-<hex>.log` in the temporary
  directory and, as in pi, is not deleted when the session closes. Tool paths resolve against the session cwd (`~` and absolute
  paths supported); the cwd is a base directory, not a sandbox, and the
  process working directory is never changed.
