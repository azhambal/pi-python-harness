# Compatibility and scope

[Documentation](README.md) · [Project](../README.md)

- Tool execution defaults to sequential; parallel mode is opt-in with bounded
  capacity, resolved-path queues and barriers for custom tools that do not opt in.
  Pi defaults to parallel. See [parallel scheduling](tools.md#parallel-tool-scheduling).
- **Edit prefers exact text and supports Pi's fuzzy fallback.** NFKC,
  trailing whitespace, smart quotes, Unicode dashes and special spaces are
  normalized for matching. Multiple replacements use one original matching
  view, reject ambiguous normalized duplicates and overlapping ranges, then
  rewrite only touched lines. Unchanged lines keep their original characters
  and trailing spaces. An unchanged final whitespace-only line is retained.
  Exact-only edits retain all surrounding text on touched lines too.
  Line endings are LF-normalized for matching and restored with the original
  ending convention/BOM; mixed endings become uniform. Invalid UTF-8 is refused.
- `edit` and `write` use same-directory atomic replacement: prepare a private
  temp file, preserve owner/group/ordinary mode/xattrs, flush/fsync and close,
  verify the original version, replace, fsync the directory. Existing symlinks
  resolve to their target; the symlink itself is preserved. Hard links and
  special files are refused. Set-ID bits/file capabilities are cleared.
  New files follow umask/default ACL, and write creates missing parents.
  Rename needs a writable parent; metadata that cannot be preserved causes
  refusal before replacement. Harness file tools serialize each resolved path
  on one event loop; version checks do not provide compare-and-swap against
  arbitrary external writers.
  Synchronous I/O cannot be interrupted mid-syscall by asyncio cancellation.
  Cancellation before commit leaves the original; after commit it is not
  reported as an uncommitted write. A directory-fsync failure explicitly says
  the new content is installed but durability is unconfirmed. Created parent
  directories and temp files left by process death are not rolled back;
  ordinary failure paths clean up their temp file.
- Edit results expose an applicable unified `patch` and `first_changed_line`
  (one-based, snake_case). `details.diff` retains the unified patch for current
  consumers. Patches use the LF/BOM-free matching view and support missing EOF
  newlines. Legacy edit arguments and display-oriented line-numbered diffs
  remain deferred.
- No dynamic tool-loadout protocol: the leading system message declares the
  run's tool set once; resumed sessions announce a delta when the active set
  changed.
- Before/after hooks are injected Python callbacks; no extension system or custom message
  roles (hence no convert-to-LLM step — the transcript is the LLM message
  list).
- Retries keep the failed assistant attempt in the raw journal but durably
  omit it from the provider projection (Pi uses context-edit entries).
- No `AGENTS.override.md` / `CLAUDE.md` aliases; only `AGENTS.md` is loaded
  (global `~/.pi-agent/AGENTS.md` first, then one per directory from `/`
  down to the cwd).
- No PI_* environment variables exported to bash; no image support in
  `read`.

## Scope

This is an independent Python implementation of a text-first coding harness.
It uses its own journal format and does not promise full Pi API or configuration
compatibility. TUI/web UI, executable extensions, subagents, OAuth, multimodal
content, session trees, RPC servers, OpenAI Responses and MCP codemode/deferred
exposure are outside the supported scope.
