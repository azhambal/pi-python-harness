# Examples

- [Settings template](settings.example.json): common settings fields.
- [Local SGLang settings](sglang/settings.json): connect to a configured endpoint.
- [Agent Skill](skills/review-python/SKILL.md): explicit skill discovery and loading.
- [MCP configuration](mcp.example.json): pinned Git/Fetch reference servers.

## Git/Fetch MCP profile

The reference integration uses real MCP servers with a scripted provider.
See the [MCP guide](../docs/mcp.md) for configuration and connection contracts.

Use Python 3.12 and the project's local environment:

```bash
.venv/bin/python -m pip install -e '.[dev,mcp-reference]'
.venv/bin/python -m pip check
.venv/bin/python scripts/mcp_acceptance.py
.venv/bin/python scripts/benchmark_startup.py
```

The reference extra pins `mcp==2.3.0`, `mcp-server-git==2026.10.19` and
`mcp-server-fetch==2026.10.19`. It is optional; the base harness and plain MCP extra
do not install the reference servers. Git and ripgrep must be installed separately.
The acceptance also uses pytest from the dev extra. No runtime `uvx` download.

Acceptance creates a temporary Git repository, copies the coding skill unchanged,
and serves a document/robots.txt on 127.0.0.1. A scripted FauxProvider drives
ls/find/grep, explicit skill loading, denied repository/URL substitution, Fetch
HTML conversion, a failing test, edit, a passing test and all seven selected Git
reads. Compaction, strict resume, interrupted Fetch, explicit reconnect and
child/task/spill/journal-lock cleanup complete the scenario. It loads no dotenv,
uses no real API key and makes no external model or document requests. The script
prints a JSON report and removes its temporary files. Local sockets and normal
SDK server subprocess operation must be allowed by the runner.

The local fixture explicitly enables Fetch's `--allow-private-ips`; the normal
profile leaves it disabled. An executable regression checks that the default
server refuses loopback before sending an HTTP request. Fetch may use Node for
HTML readability when available, with the package's Python fallback otherwise.
Conversion correctness is checked, without requiring an exact Markdown rendering.

## Host profile

Given a host-supplied `provider`, `AgentConfig` named `config`, and a permission
callback `confirm_fetch`, create the session as follows:

```python
import sys
from pathlib import Path

from pi_agent import create_harness
from pi_agent.mcp.reference import (
    REFERENCE_SELECTORS,
    reference_config,
    reference_hooks,
    reference_permissions,
)
from pi_agent.tools import TOOL_NAMES

root = config.cwd.resolve()
session = await create_harness(
    provider,
    config=config,
    session="new",
    tool_names=TOOL_NAMES,
    mcp_config=reference_config(root, Path(sys.executable)),
    mcp_tools=REFERENCE_SELECTORS,
    hooks=reference_hooks(root),
    permission_policy=reference_permissions(),
    confirm_permission=confirm_fetch,
)
async with session:
    await session.prompt("/skill:review-python Inspect the repository changes")
```

Configure `config.skills` with explicit access to `examples/skills` to use that
command. The callback receives the copied permission request/cancellation token;
it must return an explicit boolean approval. With no callback, Fetch is denied.
Supply the same trusted config/hooks/permission callback when resuming the session;
the journal restores bindings, not connection configuration or callback authority.

Selection exposes `git_status`, `git_diff_unstaged`, `git_diff_staged`, `git_diff`,
`git_log`, `git_show`, `git_branch` and `fetch`. Git add/commit/reset/checkout/branch
creation are excluded. The hook checks host provenance/canonical binding and
resolves each `repo_path` to exactly `root`, rejecting unrelated or nested
repositories and symlink substitution. The server independently checks its
`--repository` subtree. The hook, selection and policy must be supplied together.
Preserve the `.venv/bin/python` entry point: resolving its symlink to base Python
would lose the virtual environment's installed packages.

This policy controls MCP admission. Builtin coding tools retain their ordinary
permissions; it is not a read-only session or an OS/network sandbox. Fetch asks
for destination approval, including redirect/network scope. An initial hostname
allowlist alone does not constrain DNS rebinding or redirects. The server checks
private destinations by default, but this profile does not promise a network
isolation boundary or a strict public-document-host allowlist. Such isolation
belongs in the server/network layer when required.

`mcp.example.json` demonstrates the CLI config shape for a normal repository with
`.venv` at its root. Relative command paths resolve in the server cwd. If the target
repo differs from the Python project, replace the command with the absolute
`.venv/bin/python` path and the repository argument with the target absolute path.
Always use explicit `--mcp-tools` selectors. The JSON template alone does not
install the SDK host hook; CLI users get the server's subtree restriction, whereas
the SDK profile adds exact-root admission. No tokens or executable config are
recovered from a journal.

## Compatibility evidence

The supported subset is search, Agent Skills format, direct MCP tools over
stdio/Streamable HTTP and text resources. Full Pi codemode/deferred exposure,
OAuth, legacy SSE, sampling/tasks/MCP Apps, executable extensions and multimodal
execution remain outside this subset. A Pi skill test fixture is copied byte for
byte into `tests/fixtures/pi_skills/valid-skill/SKILL.md` from
`packages/coding-agent/test/fixtures/skills/valid-skill/SKILL.md`; its SHA-256 is
`db6af8bf9364290c5376d39e97c622f0152b6cd64f379bc03855d79ea058c0ba`.
Tests check discovery and unchanged body expansion. This fixture proves a format
contract; it does not represent every community skill or execute supporting scripts.

The benchmark creates 1000 skill directories and a paginated 256-tool catalog.
It checks skill/manifest/result budgets and reports elapsed time. Timings are
observations of one workload, not CI thresholds or end-to-end LLM latency.
`scripts/benchmark_projection.py` separately measures the existing projection
workload. Remote CI and live DeepSeek quality/latency require separate execution.
