"""System prompt sections and project context files, translated from Pi.

Mirrors ``core/system-prompt.ts`` (section order and wording) and
``loadProjectContextFiles`` from ``core/resource-loader.ts``. Pi's ``<docs>``
section points at Pi's own documentation, which this port does not ship, so it
is left out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_AGENT_DIR
from .tools.base import Tool

#: Per directory, the first of these that is a file wins (Pi's ``loadContextFileFromDir``).
CONTEXT_FILE_NAMES = ("AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")

_DEFAULT_PREAMBLE = (
    "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by "
    "reading files, executing commands, editing code, and writing new files."
)

_BASE_RULES = ("Be concise in your responses", "Show file paths clearly when working with files")


@dataclass
class ContextFile:
    path: Path
    content: str


@dataclass
class SystemPromptInput:
    cwd: Path
    tools: list[Tool] = field(default_factory=list)
    context_files: list[ContextFile] = field(default_factory=list)
    custom_preamble: str | None = None
    append_prompt: str | None = None
    skills_section: str | None = None


_UTF8_BOM = "\ufeff"


def _strip_bom(text: str) -> str:
    return text[1:] if text.startswith(_UTF8_BOM) else text


def _load_context_file_from_dir(directory: Path) -> ContextFile | None:
    for name in CONTEXT_FILE_NAMES:
        candidate = directory / name
        if not candidate.exists():
            continue
        try:
            if not candidate.is_file():
                continue
            content = candidate.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue
        return ContextFile(path=candidate, content=_strip_bom(content))
    return None


def _canonical(path: Path) -> Path:
    try:
        return Path(os.path.realpath(path, strict=True))
    except OSError:
        return path


def _find_git_paths(cwd: Path) -> tuple[Path, Path] | None:
    """Pi's ``findGitPaths``: (repository dir, common git dir) for the nearest ``.git``."""
    directory = cwd
    while True:
        git_path = directory / ".git"
        if git_path.exists():
            try:
                if git_path.is_file():
                    content = git_path.read_text("utf-8").strip()
                    if content.startswith("gitdir: "):
                        git_dir = Path(os.path.normpath(directory / content[8:].strip()))
                        if not (git_dir / "HEAD").exists():
                            return None
                        common = git_dir / "commondir"
                        if common.exists():
                            return directory, Path(
                                os.path.normpath(git_dir / common.read_text("utf-8").strip())
                            )
                        return directory, git_dir
                elif git_path.is_dir():
                    return (directory, git_path) if (git_path / "HEAD").exists() else None
            except OSError:
                return None
        if directory.parent == directory:
            return None
        directory = directory.parent


def _find_shadowed_context_file(cwd: Path) -> Path | None:
    """The main checkout's context file that a nested linked worktree's own copy shadows."""
    git_paths = _find_git_paths(cwd)
    if git_paths is None:
        return None
    common_git_dir = _canonical(git_paths[1])
    worktree_root = _canonical(git_paths[0])
    main_repo_root = common_git_dir.parent
    if not str(worktree_root).startswith(f"{main_repo_root}{os.sep}"):
        return None
    if _canonical(main_repo_root / ".git") != common_git_dir:
        return None
    worktree_context = _load_context_file_from_dir(worktree_root)
    return main_repo_root / worktree_context.path.name if worktree_context is not None else None


def load_project_context_files(cwd: Path, agent_dir: Path | None = None) -> list[ContextFile]:
    """Global context file first, then ancestors from the filesystem root to cwd."""
    resolved_cwd = Path(os.path.abspath(cwd))
    resolved_agent_dir = Path(os.path.abspath(agent_dir if agent_dir is not None else DEFAULT_AGENT_DIR))
    context_files: list[ContextFile] = []
    seen: set[Path] = set()

    global_file = _load_context_file_from_dir(resolved_agent_dir)
    if global_file is not None:
        context_files.append(global_file)
        seen.add(global_file.path)

    ancestors: list[ContextFile] = []
    shadowed = _find_shadowed_context_file(resolved_cwd)
    current = resolved_cwd
    while True:
        context_file = _load_context_file_from_dir(current)
        # Compare the directory's real path: a worktree file that symlinks to the main
        # checkout's file resolves to the shadowed path itself.
        is_shadowed = (
            shadowed is not None
            and context_file is not None
            and _canonical(current) / context_file.path.name == shadowed
        )
        if context_file is not None and not is_shadowed and context_file.path not in seen:
            ancestors.insert(0, context_file)
            seen.add(context_file.path)
        if current.parent == current:
            break
        current = current.parent
    context_files.extend(ancestors)
    return context_files


def build_rules(tools: list[Tool]) -> str:
    rules: list[str] = []
    seen: set[str] = set()

    def add(rule: str) -> None:
        normalized = rule.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            rules.append(normalized)

    has_bash = any(tool.name == "bash" for tool in tools)
    if has_bash and not any(tool.name in ("grep", "find", "ls") for tool in tools):
        add("Use bash for file operations like ls, rg, find")
    for tool in tools:
        for guideline in tool.guidelines:
            add(guideline)
    for rule in _BASE_RULES:
        add(rule)
    return "\n".join(f"- {rule}" for rule in rules)


def build_system_prompt_sections(prompt_input: SystemPromptInput) -> dict[str, str]:
    """The ordered section map: raw text for ``preamble``, tagged for the rest."""
    tools = prompt_input.tools
    sections: dict[str, str] = {}
    if prompt_input.custom_preamble:
        sections["preamble"] = prompt_input.custom_preamble
    else:
        sections["preamble"] = _DEFAULT_PREAMBLE
        visible = [tool for tool in tools if tool.prompt_snippet]
        if visible:
            tool_lines = "\n".join(f"- {tool.name}: {tool.prompt_snippet}" for tool in visible)
        else:
            tool_lines = "(none)"
        sections["tools"] = (
            f"{tool_lines}\n\nIn addition to the tools above, you may have access to other custom tools "
            "depending on the project."
        )
        sections["rules"] = build_rules(tools)
    if prompt_input.append_prompt:
        sections["addendum"] = prompt_input.append_prompt
    if prompt_input.context_files:
        rendered = ["Project-specific instructions and guidelines:"]
        for context_file in prompt_input.context_files:
            rendered.append(
                f'<project_instructions path="{context_file.path}">\n{context_file.content}\n</project_instructions>'
            )
        sections["project_context"] = "\n\n".join(rendered)
    if prompt_input.skills_section:
        sections["skills"] = prompt_input.skills_section
    sections["cwd"] = str(prompt_input.cwd).replace("\\", "/")
    # Wrap every non-preamble section in a tag of its section name.
    wrapped: dict[str, str] = {"preamble": sections["preamble"]}
    for name, content in sections.items():
        if name != "preamble":
            wrapped[name] = wrap_section(name, content)
    return wrapped


def wrap_section(name: str, content: str) -> str:
    return f"<{name}>\n{content}\n</{name}>"


def diff_system_prompt_sections(
    previous: dict[str, str | None],
    current: dict[str, str],
) -> dict[str, str | None] | None:
    """Patch transforming ``previous`` into ``current``; None when identical."""
    patch: dict[str, str | None] = {}
    for name, text in current.items():
        if previous.get(name) != text:
            patch[name] = text
    for name in previous:
        if name not in current:
            patch[name] = None
    return patch or None
