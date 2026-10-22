# Agent Skills

[Documentation](README.md) · [Project](../README.md)

New CLI sessions load user skills from `~/.pi-agent/skills` and `~/.agents/skills`.
The core SDK remains opt-in via `AgentConfig(skills=SkillSettings(enabled=True))`.
`--skill PATH` adds an explicit Markdown file or directory (repeatable).
Project `.agents/skills` roots from cwd to the repository boundary require
`--trust-project-resources`. `--pi-skills` additionally enables `.pi/skills`
and `~/.pi/agent/skills`. Explicit paths precede nearest project, ancestor project,
then user roots. Within a project, .agents precedes .pi; native Python user roots
precede shared .agents and legacy Pi roots. Collisions report the retained winner;
duplicates of the same physical file are skipped.

```bash
.venv/bin/python -m pi_agent skills list --no-user-skills --skill examples/skills
.venv/bin/python -m pi_agent skills validate --no-user-skills --skill examples/skills --json
.venv/bin/python -m pi_agent run '/skill:review-python review src' --skill examples/skills --read-only
```

`skills list/validate` does not load dotenv, construct a provider, or require an
API key. Validate returns 1 for diagnostics; list returns the usable catalog.
`--skills` / `--no-skills` enable/disable discovery; `--no-user-skills` skips user
roots. Negative project/Pi flags revoke their discovery scope. These flags are
explicit reloads on resume; ordinary resume keeps the journaled snapshot and
ignores changed file defaults. Settings support a nested `skills` object with
enabled/include_user/project_trusted/pi_compat/paths/prompt_max_bytes.

Portable skills use `directory/SKILL.md`, YAML frontmatter with name/description,
and Markdown body. Multiline metadata, BOM/CRLF and relative supporting references
work. Runtime warns on portable naming/metadata violations, skips invalid YAML
or missing descriptions, and supports explicit standalone .md files. Root SKILL.md
stops nested discovery. Hidden/dependency directories are skipped; .gitignore,
.ignore and .piignore rules are applied within each discovered root. Symlink
cycles and duplicate physical directories are bounded.

The system catalog contains metadata/location, not the full instructions. A model
can load the file via read (or bash when that is its reader); normal tool hooks and
permissions apply. `disable-model-invocation: true` hides automatic routing while
keeping explicit invocation. Without a reader tool, no automatic catalog is shown.
`/skill:name args` expands at session-level input admission for SDK/run/chat and
queues; the body and reference directory become user-message content. This explicit
resource load uses the configured resource scope, not a synthetic tool call.
Allowed-tools metadata never grants permission or executes scripts; supporting
files retain their runtime/dependency requirements.

In chat, `/skills` lists the catalog and `/reload` updates it while idle. SDK uses
`session.skills`, `session.resource_diagnostics`, and
`session.reload_resources(settings=None)`. Unknown/unreadable/changed skills fail
explicit invocation before a model request. Queued expansions retain the admitted
body if the file changes later. Ordinary model reads still use the live filesystem;
skills are not a frozen archive of supporting files.

An optional host-only `resource_manifest` on SystemMessage records the catalog
independently of its display section; it is excluded from the rendered provider
prompt. Reload writes manifest and skills section in one journal record before
publishing runtime state. Compaction checkpoints and configuration changes retain
it. Resume diagnoses changed/missing files without replacing the stored catalog;
already expanded/read bodies remain in raw history, subject to normal compaction.
Change resource settings through reload, rather than a conflicting config override
when constructing a resumed session. Legacy journals without a manifest restore
an empty catalog until explicitly enabled/reloaded.

Bounds: 1 MiB regular Markdown files, 64 KiB frontmatter/ignore files, 16 MiB
aggregate discovery reads (including invalid files), 256 skills/diagnostics,
20000 directory entries, depth 32, 2048 YAML nodes. YAML aliases, duplicate mapping
keys and Python object tags are rejected. Only true/false are implicit booleans;
on/off/date values stay strings. Metadata is finite, bounded JSON; non-UTF-8 skill
paths/content are refused. Catalog display defaults to 20 KiB, retains all skills
for explicit invocation when truncated, and never omits truncation notices.
The manifest has a 1 MiB hard cap. These bounds do not make skills or tools an
OS sandbox; filesystem changes can race ordinary reads.
