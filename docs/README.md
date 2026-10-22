# Documentation

pi-agent is a Python coding-agent harness with a terminal CLI and an embeddable SDK.
Start with a short task, then add the tools and integrations your application needs.

## Start here

- [Getting started](getting-started.md): installation, credentials and your first task.
- [Command-line interface](cli.md): run, chat, JSONL output and resume.
- [Python SDK](sdk.md): sessions, events, custom providers and resource ownership.
- [Examples](../examples/README.md): a pinned Git/Fetch MCP profile and local settings.

## Configure a harness

- [Configuration](configuration.md): settings files, precedence and execution defaults.
- [Configuration reference](configuration-reference.md): schema-derived fields and defaults.
- [Providers](providers.md): OpenAI-compatible, DeepSeek and Anthropic backends.
- [SGLang](sglang.md): local endpoints, request options and thinking controls.
- [Tools and permissions](tools.md): coding/search tools, hooks and parallel execution.
- [Agent Skills](skills.md): discovery, project trust and explicit invocation.
- [Model Context Protocol](mcp.md): stdio/HTTP tools, resources and strict restore.

## Understand the contracts

- [Sessions](sessions.md): durable journals, cancellation, recovery and compaction.
- [Core invariants](invariants.md): persistence, lifecycle and execution guarantees.
- [Compatibility and scope](compatibility.md): differences from the original Pi.
- [Contributing](../CONTRIBUTING.md): development environment and offline checks.
- [Changelog](../CHANGELOG.md): project changes.

Examples assume commands are run from the repository root using `.venv`.
The SDK is provider-independent; CLI credentials and transport configuration are
loaded separately from persisted session state.
