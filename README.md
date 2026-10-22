# pi-agent

[![Python checks](https://github.com/azhambal/pi-python-harness/actions/workflows/check.yml/badge.svg)](https://github.com/azhambal/pi-python-harness/actions/workflows/check.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Python coding-agent harness for the terminal and your applications.

Connect a model, give it tools, and run a coding task with durable sessions,
streaming events, configurable permissions and context compaction.

Built from the core ideas of [Pi](https://github.com/earendil-works/pi).
Independent Python implementation · Python 3.12+ · Linux-first · MIT licensed · Alpha

## Quick start

```bash
git clone https://github.com/azhambal/pi-python-harness.git
cd pi-python-harness
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

Set your provider credentials and model in `.env`, then run:

```bash
.venv/bin/pi-agent run --read-only "Explain the structure of this repository"
.venv/bin/pi-agent chat
```

Install Git and [ripgrep](https://github.com/BurntSushi/ripgrep) separately for
repository workflows and the `grep` / `find` tools. The Python package, import and
command names are `pi-agent`, `pi_agent` and `pi-agent`.

## What it provides

- **Coding tools:** read, write, fuzzy edit, Bash and bounded repository search.
- **Sessions:** append-only JSONL journals, resume, interrupted-call recovery and
  model switching.
- **Control:** streaming events, cancellation, steering, follow-up queues,
  tool hooks and allow/ask/deny permissions.
- **Context:** compaction, bounded retries and configurable parallel tool execution.
- **Providers:** OpenAI-compatible Chat Completions, DeepSeek, Anthropic Messages
  and local SGLang endpoints; custom providers are injectable.
- **Extensions through data and tools:** Agent Skills and optional MCP tools/text
  resources over stdio and Streamable HTTP.

## Python API

```python
import asyncio
import os

from pi_agent import ModelInfo, create_harness
from pi_agent.providers import OpenAICompatProvider


async def main():
    provider = OpenAICompatProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    session = await create_harness(
        provider,
        model=ModelInfo(
            id=os.environ["OPENAI_MODEL"],
            name="Coding model",
            provider="openai",
            api="openai-completions",
            context_window=128_000,
            max_tokens=8_192,
        ),
        cwd=".",
        session="new",
    )
    async with session:
        await session.prompt("Read the README and summarize the project")


asyncio.run(main())
```

The SDK uses the environment you supply; it does not load `.env` automatically.
Set model limits to match your backend. Providers and clients injected by the host
remain host-owned. See the [SDK guide](docs/sdk.md) for lifecycle and provider
contracts.

## Documentation

- [Getting started](docs/getting-started.md) — installation and first task.
- [CLI](docs/cli.md) — run, chat, resume and JSONL events.
- [Python SDK](docs/sdk.md) — embedding, providers and resource ownership.
- [Configuration](docs/configuration.md) — settings, precedence and runtime limits.
- [Tools and permissions](docs/tools.md) — tool selection, hooks and execution policy.
- [Sessions](docs/sessions.md) — persistence, recovery and compaction.
- [Agent Skills](docs/skills.md) — discovery, trust and explicit invocation.
- [MCP](docs/mcp.md) — direct tools, resources and connection management.
- [Providers](docs/providers.md) and [SGLang](docs/sglang.md) — backend configuration.

See the [documentation index](docs/README.md), [examples](examples/README.md),
[contributor guide](CONTRIBUTING.md) and [changelog](CHANGELOG.md).

## Execution boundaries

Tools operate with the permissions of the host process. Tool admission policies
are not an OS sandbox. Use process or container isolation when running untrusted
tasks. HTTP timeouts and per-response token limits do not bound an entire run.

This is a text-first alpha project with its own session format. See
[compatibility and scope](docs/compatibility.md) for supported features and
differences from Pi.
