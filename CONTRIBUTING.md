# Development

Use Linux and Python 3.12. All Python commands run through the project's `.venv`.
Keep package, import and CLI names as `pi-agent`, `pi_agent` and `pi-agent`;
the repository name is `pi-python-harness`.

## Environment

Install Git and ripgrep (`rg`) separately, then:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,mcp-reference]'
.venv/bin/python -m pip check
```

The reference extra installs the pinned MCP SDK and Git/Fetch servers. Base
installations do not require MCP. Preserve `.venv/bin/python` when launching
servers: resolving its symlink to the system interpreter loses venv packages.

## Checks

```bash
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format --check src tests scripts
.venv/bin/python -m mypy
.venv/bin/python scripts/generate_config_reference.py --check
.venv/bin/python -m pytest -q
.venv/bin/python scripts/mcp_acceptance.py
```

The tests use synthetic credentials and providers, temporary repositories and
local HTTP servers. The runner must permit subprocesses and loopback sockets.
Offline checks must not require a provider account or a paid model request.
Optional benchmarks report observations, not CI timing thresholds:

```bash
.venv/bin/python scripts/benchmark_projection.py
.venv/bin/python scripts/benchmark_startup.py
```

## Credentials and artifacts

Configure provider variables locally using the public `.env.example` as a
variable-name guide. Do not commit `.env`, keys, authorization headers, credential
stores, local session journals or raw evaluation transcripts. `.env` and `.env.*`
are ignored except `.env.example`; `.secrets/` and `evaluation-results/` are also
ignored. Git ignore is a staging aid, not a guarantee against `git add --force`.

Dotenv loading does not export file values into tool environments. Environment
variables already exported by the caller may be inherited by ordinary Bash;
hosts running live evaluation must explicitly isolate child environments.
Tool admission policies are not an OS sandbox.

## Changes and documentation

Read [docs/invariants.md](docs/invariants.md) before changing journal, lifecycle,
permissions or MCP contracts. Update the relevant API guide and invariant with
behavioral changes. Add focused regression checks for meaningful contract changes.
Preserve the copied Pi skill fixture byte for byte; it is compatibility test data.

Configuration schemas ship in `src/pi_agent/schemas`. When changing a settings
field, update its schema and parser tests, then regenerate the field reference:
`.venv/bin/python scripts/generate_config_reference.py`.

Record validation with its actual scope. Offline provider fixtures do not
establish live account access, model quality or compatibility with every backend.
