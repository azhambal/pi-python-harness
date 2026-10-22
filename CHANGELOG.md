# Changelog

## Unreleased

- Drop cross-model thinking from provider requests and thinking from summary input.
- Decouple summary output limits from compaction reserve; retry transient summary failures.
- Bound provider output by model limits and estimated remaining context.
- Handle additional transient gateway errors and whitespace-only OpenAI tool results.
- Accept UTF-8 BOM and editor schema references in settings and MCP configuration.
- Ship settings/MCP schemas and a generated configuration field reference.
- Add relative CLI builtin selection and persisted native thinking-level maps.
- Upgrade the optional MCP SDK to 2.3.0 and Git/Fetch reference servers to 2026.10.19.
- Match Pi v1.4 MCP result envelopes, arbitrary structured JSON and explicit null.
- Preserve full structured results while spilling long model-facing text to owned temporary files.
- Reorganize documentation into focused public guides.
- Add SGLang request settings, thinking controls and reasoning-token accounting.

## 0.1.0 — initial repository snapshot, 2026-10-14

Initial release of the Python coding-agent harness. It is an alpha package, not a full Pi feature-parity release.

- Async provider-independent agent loop; DeepSeek/OpenAI-compatible and Anthropic adapters.
- Coding tools, bounded repository search, fuzzy edit and atomic file mutations.
- Durable linear sessions, resume, compaction, model switching, retries and cancellation.
- Tool hooks, allow/ask/deny policies, bounded parallel execution and live run/chat CLI.
- Compatible Agent Skills discovery, explicit invocation and durable resource snapshots.
- Optional bounded MCP stdio/Streamable HTTP, direct tools/text resources,
  strict binding restore, staged reload, explicit reconnect and owned cleanup.
- Pinned Git/Fetch examples and reproducible combined offline acceptance.
- User guides, contributor documentation and implementation contracts.

The package implements a text-first subset of Pi; see
[compatibility and scope](docs/compatibility.md).
