"""Build tagged prompt sections and load AGENTS.md from global to local scope."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_AGENT_DIR
from .tools.base import Tool

INSTRUCTION_FILE_NAME = "AGENTS.md"

_DEFAULT_PREAMBLE = (
    "You are an expert coding assistant operating inside a coding agent harness. You help users by "
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
    candidate = directory / INSTRUCTION_FILE_NAME
    if not candidate.is_file():
        return None
    try:
        return ContextFile(path=candidate, content=_strip_bom(candidate.read_text(encoding="utf-8")))
    except OSError:
        return None


def load_project_context_files(cwd: Path, agent_dir: Path | None = None) -> list[ContextFile]:
    """Global AGENTS.md first, then ancestors from the filesystem root to cwd."""
    base = agent_dir if agent_dir is not None else DEFAULT_AGENT_DIR
    context_files: list[ContextFile] = []
    seen: set[Path] = set()

    global_file = _load_context_file_from_dir(base)
    if global_file is not None:
        context_files.append(global_file)
        seen.add(global_file.path)

    resolved = Path(cwd).resolve()
    ancestors: list[ContextFile] = []
    current = resolved
    while True:
        context_file = _load_context_file_from_dir(current)
        if context_file is not None and context_file.path not in seen:
            ancestors.append(context_file)
            seen.add(context_file.path)
        parent = current.parent
        if parent == current:
            break
        current = parent
    ancestors.reverse()
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
    sections["cwd"] = str(prompt_input.cwd)
    # Wrap every non-preamble section in a tag of its section name.
    wrapped: dict[str, str] = {"preamble": sections["preamble"]}
    for name, content in sections.items():
        if name != "preamble":
            wrapped[name] = f"<{name}>\n{content}\n</{name}>"
    if prompt_input.skills_section:
        wrapped["skills"] = prompt_input.skills_section
    return wrapped


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
