"""Immutable skill snapshots, independent of the human-readable system prompt."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

from .skills import (
    MAX_MANIFEST_BYTES,
    MAX_SKILLS,
    MAX_TOTAL_BYTES,
    ResourceDiagnostic,
    SkillInfo,
    SkillSettings,
    discover_skills,
    parse_document,
    read_skill_file,
)


@dataclass(frozen=True)
class ResourceSnapshot:
    settings: SkillSettings = SkillSettings()
    skills: tuple[SkillInfo, ...] = ()
    diagnostics: tuple[ResourceDiagnostic, ...] = ()

    @classmethod
    def discover(cls, settings: SkillSettings, cwd: Path, agent_dir: Path) -> ResourceSnapshot:
        skills, diagnostics = discover_skills(settings, cwd, agent_dir)
        snapshot = cls(settings, skills, diagnostics)
        snapshot.to_manifest()  # Enforce the durable envelope budget before any commit.
        return snapshot

    def to_manifest(self) -> str:
        text = json.dumps(
            {
                "version": 1,
                "settings": self.settings.to_json(),
                "skills": [skill.to_json() for skill in self.skills],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(text.encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise ValueError("Resource manifest exceeds host budget")
        return text

    @classmethod
    def from_manifest(cls, text: str) -> ResourceSnapshot:
        if len(text.encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise ValueError("Resource manifest exceeds host budget")

        def unique(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("Duplicate resource manifest field")
                result[name] = value
            return result

        def invalid_constant(value):
            raise ValueError("Non-finite resource manifest value")

        try:
            data = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
        except RecursionError as error:
            raise ValueError("Resource manifest exceeds structural limits") from error
        if (
            not isinstance(data, dict)
            or set(data) != {"version", "settings", "skills"}
            or type(data["version"]) is not int
            or data["version"] != 1
        ):
            raise ValueError("Unsupported resource manifest")
        settings = SkillSettings.from_json(data["settings"])
        if not isinstance(data["skills"], list) or len(data["skills"]) > MAX_SKILLS:
            raise ValueError("Invalid recorded skills list")
        skills = tuple(SkillInfo.from_json(item) for item in data["skills"])
        if (
            len({skill.name for skill in skills}) != len(skills)
            or (not settings.enabled and skills)
            or (not settings.project_trusted and any(skill.source == "project" for skill in skills))
        ):
            raise ValueError("Invalid recorded skill selection/trust")
        return cls(settings, skills)

    def check_availability(self) -> ResourceSnapshot:
        diagnostics = list(self.diagnostics)
        total = 0
        for skill in self.skills:
            try:
                raw = read_skill_file(Path(skill.file_path))
                total += len(raw)
                if total > MAX_TOTAL_BYTES:
                    diagnostics.append(
                        ResourceDiagnostic("limit", skill.file_path, "Availability check byte budget reached")
                    )
                    break
                if hashlib.sha256(raw).hexdigest() != skill.digest:
                    diagnostics.append(
                        ResourceDiagnostic(
                            "changed", skill.file_path, "Skill changed; reload before explicit invocation"
                        )
                    )
            except (OSError, ValueError):
                diagnostics.append(
                    ResourceDiagnostic(
                        "missing", skill.file_path, "Skill unavailable; recorded catalog is retained"
                    )
                )
        return ResourceSnapshot(self.settings, self.skills, tuple(diagnostics[:MAX_SKILLS]))

    def prompt_section(self, tool_names: set[str]) -> str | None:
        reader = "read" if "read" in tool_names else "bash" if "bash" in tool_names else None
        visible = [skill for skill in self.skills if not skill.disable_model_invocation]
        if not reader or not visible:
            return None
        start = f"<skills>\nUse {reader} to load SKILL.md when its description matches the task. Resolve references against the skill directory.\n<available_skills>\n"
        end = "</available_skills>\n</skills>"
        notice = "Catalog truncated; explicit skill commands and the skill list still include every discovered skill.\n"
        budget = self.settings.prompt_max_bytes
        rendered: list[str] = []
        for skill in visible:
            row = f"<skill><name>{escape(skill.name)}</name><description>{escape(skill.description)}</description><location>{escape(skill.file_path)}</location></skill>\n"
            if len((start + "".join(rendered) + row + end + notice).encode("utf-8")) > budget:
                return start + "".join(rendered) + end + notice
            rendered.append(row)
        return start + "".join(rendered) + end

    def expand(self, text: str) -> str:
        if not text.startswith("/skill:"):
            return text
        command, _, arguments = text.partition(" ")
        name = command[7:]
        skill = next((item for item in self.skills if item.name == name), None)
        if skill is None:
            raise ValueError(f"Unknown skill: {name}")
        try:
            raw = read_skill_file(Path(skill.file_path))
            if hashlib.sha256(raw).hexdigest() != skill.digest:
                raise ValueError("Skill changed; reload resources before invoking it")
            _, body = parse_document(raw)
        except OSError as error:
            raise ValueError("Skill file is unavailable; reload resources") from error
        block = f'<skill name="{escape(skill.name, quote=True)}" location="{escape(skill.file_path, quote=True)}">\nReferences are relative to {skill.base_dir}.\n\n{body}\n</skill>'
        return block + ("\n\n" + arguments.strip() if arguments.strip() else "")
