"""Bounded Agent Skills parsing, discovery and routing metadata. Never executes scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pathspec import GitIgnoreSpec

from .types import JsonValue

MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * MAX_FILE_BYTES
MAX_SKILLS = 256
MAX_ENTRIES = 20_000
MAX_DEPTH = 32
MAX_MANIFEST_BYTES = MAX_FILE_BYTES


def encode_metadata(value: dict) -> str:
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > MAX_DEPTH or count > 2048:
            raise ValueError("Metadata exceeds structural limits")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Metadata keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif (
            not isinstance(item, (str, int, float, bool, type(None)))
            or isinstance(item, float)
            and not math.isfinite(item)
        ):
            raise ValueError("Metadata must be finite JSON data")
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    encoded.encode("utf-8")
    return encoded


@dataclass(frozen=True)
class SkillSettings:
    enabled: bool = False  # Core SDK is opt-in; CLI defaults new sessions to enabled.
    include_user: bool = True
    project_trusted: bool = False
    pi_compat: bool = False
    paths: tuple[Path, ...] = ()
    prompt_max_bytes: int = 20 * 1024

    def validate(self) -> None:
        for name in ("enabled", "include_user", "project_trusted", "pi_compat"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"skills.{name} must be boolean")
        if (
            not isinstance(self.paths, tuple)
            or len(self.paths) > 64
            or any(not isinstance(path, Path) for path in self.paths)
        ):
            raise ValueError("skills.paths must contain at most 64 Paths")
        if (
            isinstance(self.prompt_max_bytes, bool)
            or not isinstance(self.prompt_max_bytes, int)
            or not 512 <= self.prompt_max_bytes <= 65536
        ):
            raise ValueError("skills.prompt_max_bytes must be between 512 and 65536")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "enabled": self.enabled,
            "include_user": self.include_user,
            "project_trusted": self.project_trusted,
            "pi_compat": self.pi_compat,
            "paths": [str(path) for path in self.paths],
            "prompt_max_bytes": self.prompt_max_bytes,
        }

    @classmethod
    def from_json(cls, data: object) -> SkillSettings:
        fields = {"enabled", "include_user", "project_trusted", "pi_compat", "paths", "prompt_max_bytes"}
        if not isinstance(data, dict) or set(data) - fields:
            raise ValueError("Invalid skills settings fields")
        paths = data.get("paths", [])
        if not isinstance(paths, list) or any(not isinstance(path, str) or not path for path in paths):
            raise ValueError("skills.paths must be an array of nonempty paths")
        options = cls(
            **{key: value for key, value in data.items() if key != "paths"},
            paths=tuple(Path(path) for path in paths),
        )
        options.validate()
        return options


@dataclass(frozen=True)
class ResourceDiagnostic:
    code: str
    path: str
    message: str

    def to_json(self) -> dict[str, JsonValue]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    file_path: str
    base_dir: str
    source: str
    digest: str
    disable_model_invocation: bool = False
    metadata_json: str = "{}"

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "base_dir": self.base_dir,
            "source": self.source,
            "digest": self.digest,
            "disable_model_invocation": self.disable_model_invocation,
            "metadata": json.loads(self.metadata_json),
        }

    @classmethod
    def from_json(cls, data: object) -> SkillInfo:
        if not isinstance(data, dict) or set(data) != {
            "name",
            "description",
            "file_path",
            "base_dir",
            "source",
            "digest",
            "disable_model_invocation",
            "metadata",
        }:
            raise ValueError("Invalid recorded skill")
        for key in ("name", "description", "file_path", "base_dir", "source", "digest"):
            if not isinstance(data[key], str) or not data[key] or len(data[key]) > 4096:
                raise ValueError(f"Invalid recorded skill {key}")
            data[key].encode("utf-8")
        if data["source"] not in ("explicit", "project", "user") or not re.fullmatch(
            r"[0-9a-f]{64}", data["digest"]
        ):
            raise ValueError("Invalid recorded skill provenance/digest")
        if (
            not Path(data["file_path"]).is_absolute()
            or not Path(data["base_dir"]).is_absolute()
            or Path(data["file_path"]).suffix.lower() != ".md"
        ):
            raise ValueError("Recorded skills require absolute Markdown paths")
        if not isinstance(data["disable_model_invocation"], bool) or not isinstance(data["metadata"], dict):
            raise ValueError("Invalid recorded skill metadata")
        metadata = encode_metadata(data["metadata"])
        return cls(**{key: value for key, value in data.items() if key != "metadata"}, metadata_json=metadata)


class _FrontmatterLoader(yaml.SafeLoader):
    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.depth = self.nodes = 0

    def compose_node(self, parent: Any, index: Any) -> yaml.Node | None:
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported in skills")
        self.depth += 1
        self.nodes += 1
        try:
            if self.depth > MAX_DEPTH or self.nodes > 2048:
                raise ValueError("YAML frontmatter exceeds structural limits")
            return super().compose_node(parent, index)
        finally:
            self.depth -= 1

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("YAML mapping keys must be unique strings")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# YAML 1.2-like metadata: 'on', 'off', dates remain strings; only true/false are booleans.
_FrontmatterLoader.yaml_implicit_resolvers = {
    key: [
        (tag, regex)
        for tag, regex in values
        if tag not in ("tag:yaml.org,2002:bool", "tag:yaml.org,2002:timestamp")
    ]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_FrontmatterLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.IGNORECASE), list("tTfF")
)


def read_skill_file(path: Path) -> bytes:
    """Read regular Markdown only, with a size cap; reject aliases to environment files."""
    try:
        resolved = path.resolve()
    except RuntimeError as error:
        raise ValueError("Skill path has a symlink cycle") from error
    if path.suffix.lower() != ".md" or resolved.suffix.lower() != ".md":
        raise ValueError("Skill files must be Markdown")
    descriptor = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError("Skill must be a regular Markdown file within the size limit")
        raw = file.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Skill file exceeds size limit")
    return raw


def parse_document(raw: bytes) -> tuple[dict, str]:
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    end = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if end is None:
        raise ValueError("Unterminated skill frontmatter")
    header = "\n".join(lines[1:end])
    if len(header.encode("utf-8")) > 65536:
        raise ValueError("Skill frontmatter exceeds size limit")
    loader = _FrontmatterLoader(header)
    try:
        data = loader.get_single_data()
    except yaml.YAMLError as error:
        raise ValueError("Invalid skill YAML frontmatter") from error
    finally:
        loader.dispose()
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Skill frontmatter must be a mapping")
    return data, "\n".join(lines[end + 1 :]).strip()


def load_skill(path: Path, source: str) -> tuple[SkillInfo | None, list[ResourceDiagnostic], int]:
    diagnostics: list[ResourceDiagnostic] = []
    raw = b""
    try:
        raw = read_skill_file(path)
        header, _ = parse_document(raw)
    except (OSError, ValueError, yaml.YAMLError):
        return (
            None,
            [ResourceDiagnostic("invalid", str(path), "Cannot read/parse bounded UTF-8 Markdown skill")],
            len(raw),
        )
    declared_name = header.get("name")
    name = declared_name if isinstance(declared_name, str) and declared_name else path.parent.name
    if not isinstance(declared_name, str) or not declared_name:
        diagnostics.append(
            ResourceDiagnostic("validation", str(path), "Missing/non-text name; using directory name")
        )
    description = header.get("description")
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 4096
        or not isinstance(description, str)
        or not description.strip()
        or len(description) > 4096
    ):
        return (
            None,
            [ResourceDiagnostic("invalid", str(path), "Skill needs a text name and nonempty description")],
            len(raw),
        )
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
        diagnostics.append(
            ResourceDiagnostic(
                "validation",
                str(path),
                "Name must use lowercase words with single hyphens, at most 64 characters",
            )
        )
    if name != path.parent.name:
        diagnostics.append(
            ResourceDiagnostic("validation", str(path), "Name differs from its parent directory")
        )
    if len(description) > 1024:
        diagnostics.append(ResourceDiagnostic("validation", str(path), "Description exceeds 1024 characters"))
    disabled = header.get("disable-model-invocation", False)
    if not isinstance(disabled, bool):
        diagnostics.append(
            ResourceDiagnostic("validation", str(path), "disable-model-invocation must be boolean")
        )
    for key in ("license", "compatibility", "allowed-tools"):
        if key in header and (
            not isinstance(header[key], str) or key == "compatibility" and len(header[key]) > 500
        ):
            diagnostics.append(
                ResourceDiagnostic("validation", str(path), f"Invalid portable {key} metadata")
            )
    if "metadata" in header and (
        not isinstance(header["metadata"], dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in header["metadata"].items()
        )
    ):
        diagnostics.append(
            ResourceDiagnostic("validation", str(path), "Portable metadata must map strings to strings")
        )
        disabled = False
    metadata = {
        key: header[key] for key in ("license", "compatibility", "metadata", "allowed-tools") if key in header
    }
    try:
        encoded = encode_metadata(metadata)
        (name + description + str(path.absolute())).encode("utf-8")
    except (TypeError, ValueError):
        return (
            None,
            [ResourceDiagnostic("invalid", str(path), "Skill metadata must be finite JSON data")],
            len(raw),
        )
    skill = SkillInfo(
        name,
        description.strip(),
        str(path.absolute()),
        str(path.parent.absolute()),
        source,
        hashlib.sha256(raw).hexdigest(),
        disabled,
        encoded,
    )
    return skill, diagnostics, len(raw)


def default_user_roots(agent_dir: Path) -> list[Path]:
    return [agent_dir / "skills", Path.home() / ".agents" / "skills"]


def skill_roots(settings: SkillSettings, cwd: Path, agent_dir: Path) -> list[tuple[Path, str]]:
    roots = [
        (path.expanduser() if path.is_absolute() else cwd / path.expanduser(), "explicit")
        for path in settings.paths
    ]
    if settings.project_trusted:
        directory = cwd.resolve()
        while True:
            roots.append((directory / ".agents" / "skills", "project"))
            if settings.pi_compat:
                roots.append((directory / ".pi" / "skills", "project"))
            if (directory / ".git").exists() or directory.parent == directory:
                break
            directory = directory.parent
    if settings.include_user:
        roots.extend((path, "user") for path in default_user_roots(agent_dir))
        if settings.pi_compat:
            roots.append((Path.home() / ".pi" / "agent" / "skills", "user"))
    return roots


@dataclass
class _Discovery:
    skills: dict[str, SkillInfo] = field(default_factory=dict)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)
    files: set[Path] = field(default_factory=set)
    directories: set[Path] = field(default_factory=set)
    entries: int = 0
    bytes: int = 0

    def diagnostic(self, code: str, path: Path, message: str) -> None:
        if len(self.diagnostics) < MAX_SKILLS:
            self.diagnostics.append(ResourceDiagnostic(code, str(path), message))

    def add(self, path: Path, source: str) -> None:
        canonical = path.resolve()
        if canonical in self.files:
            return
        self.files.add(canonical)
        info, diagnostics, size = load_skill(path, source)
        self.bytes += size
        self.diagnostics.extend(diagnostics[: max(0, MAX_SKILLS - len(self.diagnostics))])
        if self.bytes > MAX_TOTAL_BYTES:
            raise ValueError("Skill discovery byte budget reached")
        if info is not None:
            if info.name in self.skills:
                self.diagnostic(
                    "collision", path, f"Name collision; winner: {self.skills[info.name].file_path}"
                )
            elif len(self.skills) < MAX_SKILLS:
                self.skills[info.name] = info
            else:
                raise ValueError("Skill count limit reached")

    def walk(
        self, directory: Path, source: str, rules: tuple[tuple[Path, GitIgnoreSpec], ...] = (), depth: int = 0
    ) -> None:
        canonical = directory.resolve()
        if canonical in self.directories:
            return
        if depth > MAX_DEPTH:
            self.diagnostic("limit", directory, "Discovery depth limit reached")
            return
        self.directories.add(canonical)
        for name in (".gitignore", ".ignore", ".piignore"):
            candidate = directory / name
            if candidate.is_file():
                with candidate.open("rb") as file:
                    raw = file.read(65537)
                self.bytes += len(raw)
                if self.bytes > MAX_TOTAL_BYTES:
                    raise ValueError("Skill discovery byte budget reached")
                if len(raw) > 65536:
                    raise ValueError("Skill ignore file exceeds size limit")
                rules += (
                    (
                        directory,
                        GitIgnoreSpec.from_lines(raw.decode("utf-8-sig").splitlines(), backend="simple"),
                    ),
                )

        def ignored(path: Path, is_directory: bool = False) -> bool:
            excluded = False
            for base, spec in rules:
                match = spec.check_file(path.relative_to(base).as_posix() + ("/" if is_directory else ""))
                if match.include is not None:
                    excluded = match.include
            return excluded

        declared = directory / "SKILL.md"
        if declared.is_file() and not ignored(declared):
            self.add(declared, source)
            return
        entries = []
        with os.scandir(directory) as scan:
            for entry in scan:
                self.entries += 1
                if self.entries > MAX_ENTRIES:
                    raise ValueError("Skill discovery entry limit reached")
                entries.append(entry)
        for entry in sorted(entries, key=lambda value: value.name):
            if entry.name.startswith(".") or entry.name in ("node_modules", "__pycache__", "venv"):
                continue
            path = directory / entry.name
            try:
                is_directory = entry.is_dir()
                if ignored(path, is_directory):
                    continue
                if is_directory:
                    self.walk(path, source, rules, depth + 1)
                elif depth == 0 and entry.is_file() and path.suffix.lower() == ".md":
                    self.add(path, source)
            except (OSError, ValueError, RuntimeError):
                self.diagnostic("invalid", path, "Resource discovery failed or reached a budget")
                if (
                    self.entries > MAX_ENTRIES
                    or self.bytes > MAX_TOTAL_BYTES
                    or len(self.skills) >= MAX_SKILLS
                ):
                    raise


def discover_skills(
    settings: SkillSettings, cwd: Path, agent_dir: Path
) -> tuple[tuple[SkillInfo, ...], tuple[ResourceDiagnostic, ...]]:
    settings.validate()
    state = _Discovery()
    if settings.enabled:
        for path, source in skill_roots(settings, cwd, agent_dir):
            try:
                if path.is_dir():
                    state.walk(path, source)
                elif path.is_file() and path.suffix.lower() == ".md":
                    state.add(path, source)
                elif source == "explicit":
                    state.diagnostic("invalid", path, "Explicit skill path is missing or is not Markdown")
            except (OSError, ValueError, RuntimeError):
                state.diagnostic("invalid", path, "Resource discovery failed or reached a budget")
            if (
                state.entries > MAX_ENTRIES
                or state.bytes > MAX_TOTAL_BYTES
                or len(state.skills) >= MAX_SKILLS
            ):
                break
    return tuple(state.skills.values()), tuple(state.diagnostics)
