# Getting started

[Documentation](README.md) · [Project](../README.md)

## Install

Use Linux and Python 3.12 or newer. Install Git and ripgrep (`rg`) with your
operating system's package manager, then:

```bash
git clone https://github.com/azhambal/pi-python-harness.git
cd pi-python-harness
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

## Configure a model

Edit `.env` with your credentials, endpoint and model. The supplied example uses
the OpenAI-compatible DeepSeek transport. For another Chat Completions endpoint,
set `PI_PROVIDER=openai`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `OPENAI_MODEL`.
Anthropic and SGLang configuration is covered in [providers](providers.md).
Do not commit credentials or session transcripts.

The CLI reads `.env` from the launch directory. Use `--env-file PATH` to select
another file and `--cwd PATH` to select the tools' working directory. These are
independent settings. Exported environment variables take precedence over dotenv.

## Run a task

Start with read-only tools:

```bash
.venv/bin/pi-agent run --read-only "Explain the main modules in this project"
```

For a coding task, run with your chosen permissions:

```bash
.venv/bin/pi-agent run "Fix the failing test and rerun it"
.venv/bin/pi-agent run --continue "Review the changes"
.venv/bin/pi-agent chat --continue
```

Read-only mode denies Bash as well as mutation tools. Ordinary runs use the
configured admission policy and execute as your operating system user. Review
[tools and permissions](tools.md) before running unfamiliar tasks.

## Inspect sessions

```bash
.venv/bin/pi-agent list
.venv/bin/pi-agent show /path/to/session.jsonl
.venv/bin/pi-agent run --resume /path/to/session.jsonl "Continue the task"
```

Sessions are persisted by default. `--no-session` keeps a run in memory.
Use `--json` for streaming JSONL events. Inspection commands do not call a model.

## Optional MCP integration

```bash
.venv/bin/python -m pip install -e '.[mcp]'
```

MCP configuration is explicit. Follow the [MCP guide](mcp.md) or the
[Git/Fetch example](../examples/README.md) to select servers, tools and permissions.
