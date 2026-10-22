# Model Context Protocol

[Documentation](README.md) · [Project](../README.md)

Install the optional client dependencies before configuring a server:

```bash
.venv/bin/python -m pip install -e '.[mcp]'
```

## MCP tools through the SDK

An explicitly selected JSON file uses the `mcpServers` shape. Enabled servers
must select `"exposure": "direct"`; Pi's default codemode is not silently enabled.
No project/user configuration is discovered automatically. A host loading a
project source must pass `source="project", project_trusted=True` explicitly.

```json
{
  "mcpServers": {
    "local": {
      "command": "/absolute/path/to/server",
      "args": [],
      "cwd": "/absolute/path/to/project",
      "exposure": "direct",
      "timeout": 60,
      "env": { "SERVER_TOKEN": "${MY_SERVER_TOKEN}" }
    }
  }
}
```

Environment values support whole-value `${NAME}` references only. Parsing never
resolves them or loads dotenv. Launch inherits only PATH/HOME/TMPDIR/LANG/LC_ALL/
LC_CTYPE/TZ/USER plus explicit references; provider credentials are not inherited
automatically. Runtime configuration has a redacted repr and no journal serializer.
Commands execute as argv without a shell. Use absolute executable/argument paths;
relative cwd is resolved against the host cwd. Pi shell substitutions, literal or
embedded env values, partial project overrides, toolExposure, deferred/codemode,
OAuth, legacy SSE and unknown fields are rejected. Invalid configuration fails as a whole.
Streamable HTTP uses the separate configuration below.

Configuration accepts UTF-8 BOM and an editor-only `$schema` reference. See the
[packaged schema](../src/pi_agent/schemas/mcp.schema.json) and
[field reference](configuration-reference.md). Runtime trust and byte-budget
checks remain authoritative; schema references are never fetched.

```python
from pi_agent import ResourceOwner, create_builtin_tools, create_harness
from pi_agent.mcp import McpRuntime, load_mcp_config

owner = ResourceOwner()
mcp = McpRuntime(
    load_mcp_config(config_path, cwd=config.cwd),
    builtins=tuple(create_builtin_tools(config.cwd)),
)
session = await create_harness(
    provider, config=config, session="new",
    resource_owner=owner, prepare_tools=mcp.prepare_tools,
)
async with session:
    await session.prompt("Inspect the repository with the available tools")
```

The host supplies `config_path` (a Path), `config` and `provider`. Preparation
registers ownership before connecting and returns the complete toolset. Every
enabled server is required for this preparation; disabled servers never start.
A builtin-only harness needs no MCP configuration or SDK installation. Missing
extra reports `Install the optional MCP extra: pi-agent[mcp]` before process launch.

Tools have bounded, deterministic `mcp__server__tool_<hash>` names; the hash is
always present, including collision-free names. `context.tool_provenance` in
before/after hooks and permission confirmation includes the original server/tool,
schema digest and non-authoritative annotations, without transport configuration.
Canonical declarations contain input schemas, not credentials. MCP tools are
scheduling barriers, and read-only policy denies them even with readOnlyHint.
Use explicit permission rules/host hooks for permitted external operations.

The stdio transport bounds frames (1 MiB), queued frames/outstanding requests (16),
total stdout (16 MiB per connection), and stderr (64 KiB, discarded). Violations
disconnect; cleanup joins owned tasks and TERM→KILL process groups. Python 3.12
pipe cleanup uses asyncio's private subprocess transport and is POSIX-specific;
descendants escaping their process group are outside ownership.

Tool catalogs cap pages/tools/schema bytes. Input schemas must describe objects;
omitted `type` and `properties` default to `"object"` and `{}`, as in Pi v1.4.
Output schemas may describe any JSON type, including arrays, scalars, nullable
unions and unconstrained `{}`. JSON Schema draft 2020-12 schemas are validated.
References, regex/pattern schemas and uniqueItems=true are explicitly unsupported. Validation also has a
work budget, including nested explicit dialects. Progress is capped at 32 updates;
callbacks cannot reenter connection requests or resource/session lifecycle.
Host callbacks must remain cooperative with cancellation.

MCP results follow Pi v1.4's `convertMcpResult` contract. The harness's
`structured_content` contains the complete server result envelope (`content`,
optional `structuredContent`, optional `isError`, and extension fields), excluding
only its top-level `_meta`. The server's `structuredContent` may be any JSON value.
Explicit `null` remains distinct from an absent field, including in errors,
events, journals and resumed sessions. To read it in a hook or SDK consumer, use
`result.structured_content["structuredContent"]`; older Python versions exposed
the inner value directly. Historical records are preserved without conversion.
Input-schema normalization can change recorded digests; accepting that change
on an older session requires explicit `refresh_mcp=True` or CLI `--mcp-reload`.

Content blocks take precedence for the model. With no blocks, a present
`structuredContent` becomes indented JSON text, including `null`, `false`, `0`
and empty strings/arrays. An otherwise empty error receives a server/tool error
notice. Declared output schemas validate successful structured results; errors
retain their payload without applying the success schema.

Model-facing text keeps its beginning/end after 20 KiB (or a lower
`McpLimits.result_bytes`), plus an omission notice and a path to the full 0600
UTF-8 temporary file. Files belong to the connection and are removed on close;
resume does not recreate them. The structured envelope is never shortened to
fit this text budget. It remains subject to transport frame and JSON
complexity limits (1 MiB, 32 nested levels, 20,000 nodes). Progress updates use
the separate serialized result budget. Host after-hooks own their replacements.

Embedded text resources and resource links have text projections. The harness
remains text-first: image/audio/binary blocks produce model-facing unsupported
notices, while the original blocks remain in the structured envelope. Nested
content metadata is retained there, as in Pi; it is not projected into model text.

Request timeout is absolute and does not reset on progress. Cancel/timeout closes
the connection and joins cleanup; a begun operation can have an unknown outcome
and is never automatically replayed. Each failed connection is terminal; explicit
reconnect creates a new connection for future calls. The pinned reference Git/Fetch
profile and offline acceptance are described below.

## Streamable HTTP and text resources

The same explicit config, managed factory, CLI flags, permissions, staging and
strict restore work with HTTP endpoints. Install the optional MCP extra as above.
`"http"` and `"streamable-http"` select Streamable HTTP; a URL without `type` also
selects it. Stdio fields cannot be mixed into an HTTP entry.

```json
{
  "mcpServers": {
    "remote": {
      "type": "streamable-http",
      "url": "https://example.invalid/mcp",
      "headers": { "Authorization": "${MCP_AUTHORIZATION}" },
      "timeout": 60,
      "exposure": "direct"
    }
  }
}
```

`MCP_AUTHORIZATION` contains the complete header value, including any required
`Bearer ` prefix. References resolve from an explicitly supplied `mcp_environment`
mapping or exact `os.environ` lookups. The MCP resolver does not load dotenv or
inherit provider credentials, proxies or netrc. HTTP header values must be ASCII;
literal/interpolated values and overrides of transport headers are rejected.
Endpoints support HTTP/HTTPS without userinfo, query or fragment. Use a header
reference for credentials. URL/headers/session ID are runtime-only and never
become manifest or journal configuration.

POST responses support bounded JSON and SSE, with an optional GET notification
stream (405 means unsupported). Frames/SSE events and outgoing messages cap at
1 MiB, lifetime response-body bytes at 16 MiB, queues/outstanding requests at 16.
`HttpLimits` can reduce these caps when constructing `McpRuntime` or
`McpConnection`. SSE includes CR/LF/CRLF, multiline data and UTF-8 BOM; small
events are delivered without waiting for a full buffer or EOF. Redirects,
compression, cookies, stream resumption and automatic retries are disabled.
HTTP session expiry/disconnect/cancellation is terminal; explicit reconnect is
for future calls. Cleanup joins request/GET tasks and closes the client; session
DELETE is best effort with a 0.5-second default cap. Client close failures remain
owned and retryable. Transport logs redact the calling connection's traffic.

Servers advertising resources expose three normal external bindings:
`remote/@pi:resources/list`, `remote/@pi:resources/templates/list`, and
`remote/@pi:resources/read`. Select these aliases with `mcp_tools` or `--mcp-tools`;
their canonical names still use `binding_name(server, original_identity)`.
`list` and `templates/list` accept optional `cursor`; `read` requires `uri`.
Each list call returns one page with `nextCursor`; no eager crawl or read occurs
at startup. URI/cursor inputs are bounded before hooks/permission admission.
All three bindings are scheduling barriers and denied by read-only policy.

Listings retain public metadata and the complete page/cursor in the structured
envelope. Long model-facing text spills to a file without shortening that page.
Reads retain text/JSON with the same model-facing text limit. Unsupported
binary resources return a MIME notice without decoding/saving their blob.
Resource capability removal rejects strict resume of required bindings. The
reserved identity namespace cannot be claimed by server tools. Existing manifest
v1/journal contracts remain unchanged. Server instructions are not automatically
adopted into the prompt. OAuth, sampling, tasks, MCP Apps and multimodal execution
remain outside this subset.

## Pinned Git/Fetch and offline acceptance

```bash
.venv/bin/python -m pip install -e '.[dev,mcp-reference]'
.venv/bin/python scripts/mcp_acceptance.py
.venv/bin/python scripts/benchmark_startup.py
```

The optional reference extra pins Git/Fetch 2026.10.19 alongside SDK 2.3.0.
Acceptance uses real server processes, a temporary Git repository, a localhost
document and FauxProvider. It verifies search/skills, Fetch confirmation, edit/test,
all seven selected Git reads, compaction/resume/abort/reconnect and cleanup without
loading dotenv or making live model requests. A normal runner must permit local
sockets and SDK server subprocess operation. CI installs this extra to execute
the integration tests, while base installs remain independent of MCP.

`pi_agent.mcp.reference` supplies explicit config/selection/hooks/permissions:
Git reads require exactly the bound repository root, mutations are excluded, and
Fetch always asks. Builtin coding tools keep ordinary permissions. The normal
Fetch profile refuses private IPs; only the local fixture opts in. URL approval
is not a redirect/DNS isolation guarantee. Configuration template, SDK usage,
scope limits and fixture provenance are in [examples/README.md](../examples/README.md).
No live DeepSeek quality claim follows from the scripted acceptance.

SDK 2.3.0 owns request dispatch, progress and cancellation over our bounded
stdio/HTTP channels. Like stable Pi v1.4.0, initialization negotiates MCP
2025-11-25 while tool results already accept arbitrary structured JSON from the
2026-07-28 revision. This is not a full implementation of that newer protocol's
`server/discover` lifecycle. Tool listing/calling uses the SDK's public dispatcher
with host schema/result checks, avoiding its legacy object-only result check and
implicit schema refresh. Cancellation may notify the peer, then the host closes
the connection; notification never proves that remote effects were rolled back.

The implementation was checked against stable Pi v1.4.0, commit
`4a080252f06caf63a525d7873c5d94d3c23887e0`:
[protocol content](https://github.com/earendil-works/pi/blob/v1.4.0/packages/mcp/src/protocol/content.ts),
[client result validation](https://github.com/earendil-works/pi/blob/v1.4.0/packages/mcp/src/client.ts),
and [agent result conversion](https://github.com/earendil-works/pi/blob/v1.4.0/packages/coding-agent/src/extensions/mcp/tools.ts).
See also [SDK 2.3.0 release notes](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.3.0).

## MCP restore, reload and CLI

The managed factory selects the four SDK builtins by default (CLI selects seven),
plus MCP tools from the trusted configuration. `tool_names` selects builtins;
`mcp_tools` selects remote bindings by `server/tool` alias or canonical hashed name.
An explicit empty sequence removes MCP without connecting; `("*",)` selects all
configured tools. Managed MCP options are exclusive with `tools`/`prepare_tools`.

```python
from pi_agent import create_harness
from pi_agent.mcp import load_mcp_config

mcp_config = load_mcp_config(config_path, cwd=config.cwd)
session = await create_harness(
    provider, config=config, session="new", mcp_config=mcp_config,
    tool_names=("read", "grep", "find", "ls", "edit", "write", "bash"),
)
async with session:
    await session.prompt("Inspect the repository")

# A resumed session selects only recorded tools; supply the trusted config again.
resumed = await create_harness(provider, session=session_path, mcp_config=mcp_config)
async with resumed:
    await resumed.reconnect_mcp("local")  # only if local has active bindings
    await resumed.reload_mcp()           # explicitly refresh current bindings
    # await resumed.reload_mcp(("*",))    # explicitly select all configured tools
```

The host supplies provider/config/config_path/session_path. SDK resume holds the
journal writer lock throughout asynchronous preparation. Only recorded servers
and bindings connect by default; extra tools/servers remain inactive. Aliases and
saved canonical names select their exact server scope. An unknown fresh canonical
hash cannot be reversed, so its lookup may require the full enabled catalog;
prefer aliases for new selections.

Host-only `SystemMessage.mcp_manifest` stores versioned binding identity and input/
output-schema and full-declaration digests. It shares the authoritative journal
record with tool/prompt changes and survives replay/compaction. It contains no
commands, environment values, URLs or auth headers. Ordinary resume rejects a
missing binding or changed schema/description before a provider request. Even an
explicit selection must pass matching retained digests; `refresh_mcp=True` accepts
those changes deliberately. Without a trusted config, use `mcp_tools=()` to remove
MCP and record the removal for offline resume.

Legacy builtin journals do not acquire MCP merely because a configuration was
supplied. Legacy MCP journals without a manifest require explicit managed adoption
(`refresh_mcp=True`) or complete already-connected host injection matching the
recorded declarations. Injection cannot activate MCP extras or change input/prose.
Historical output schemas were not recorded; adoption captures the trusted current
output contract. An explicitly empty manifest differs from an absent legacy field.
The synchronous core factory keeps its complete builtin replacement behavior when
`tool_names` is supplied; managed CLI changes only builtins and requires explicit
MCP selection/removal.

`reconnect_mcp(server)` is idle-only and strict: missing/changed bindings reject the
candidate and preserve the current registry. `reload_mcp()` refreshes the current
set; selectors replace the remote set, and `("*",)` discovers all enabled servers.
Both use the immutable trusted config already attached to the harness; editing its
JSON file requires a new load/startup. Environment references resolve anew on each
connection. No API retries an interrupted tool call.

Replacement connections are staged before activation. Validation precedes one
system-record append; publication follows without an await, then retired/unused
connections close. Before the first conversation, this respects SessionStore's
existing pending-record rule; the first user message flushes history before any
model/tool execution. A failed durable write never publishes candidates and
poisons the session store. Cancellation before commit closes candidates and leaves
current tools intact. Failure/cancellation after commit does not roll back it;
retirement failure reports `loadout committed; cleanup pending`, and resources
remain owned for close/retry. Chat reports this without exiting. Callbacks must
remain cooperative with cancellation.

Run and chat accept explicit configurations without automatic discovery:

```bash
.venv/bin/python -m pi_agent run --mcp-config mcp.json "Inspect the project"
.venv/bin/python -m pi_agent chat --mcp-config mcp.json
.venv/bin/python -m pi_agent run --resume SESSION.jsonl --mcp-config mcp.json "Continue"
.venv/bin/python -m pi_agent run --resume SESSION.jsonl --mcp-tools '' "Continue offline"
```

`--mcp-tools` takes comma-separated aliases/names (empty clears MCP).
`--mcp-reload` accepts changed recorded schemas/declarations; selecting new tools
requires an explicit selection. Project config uses `--mcp-project-config PATH
--trust-project-mcp`, separate from trust in Markdown skills. Invalid project
trust/config is rejected before reading model credentials. CLI's ordinary provider
dotenv behavior is unchanged; MCP does not load credentials/config from history.

Chat commands: `/mcp` displays non-secret states and active mapping;
`/mcp reconnect SERVER`, `/mcp reload`, `/mcp reload SELECTORS`, `/mcp reload all`
perform explicit idle configuration. `/wait` waits for the operation. Permissions,
queues, resource cleanup and model behavior use the same session as ordinary tools.
