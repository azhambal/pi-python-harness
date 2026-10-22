import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from pi_agent.resources import ResourceSnapshot
from pi_agent.skills import SkillSettings, discover_skills, load_skill, parse_document


def skill(root: Path, name="sample", description="Review Python code", body="BODY_ONLY", extra=""):
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{body}\n")
    return path


def options(*paths, **kwargs):
    return SkillSettings(enabled=True, include_user=False, paths=tuple(paths), **kwargs)


def test_multiline_yaml_bom_crlf_and_metadata(tmp_path):
    path = skill(
        tmp_path,
        description="|\n  First line\n  Second line",
        extra="disable-model-invocation: true\nmetadata:\n  mode: off\n  date: 2026-10-13\nallowed-tools: bash\n",
    )
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes().replace(b"\n", b"\r\n"))
    info, diagnostics, _ = load_skill(path, "explicit")
    assert not diagnostics and info.description == "First line\nSecond line"
    assert info.disable_model_invocation
    assert json.loads(info.metadata_json)["metadata"] == {"mode": "off", "date": "2026-10-13"}
    assert parse_document(path.read_bytes())[1] == "BODY_ONLY"


@pytest.mark.parametrize(
    "header",
    [
        "name: one\nname: two\ndescription: test",
        "name: test\ndescription: [invalid",
        "name: test\ndescription: !python/object:os.system {}",
        "name: test\ndescription: &x [*x]",
        "name: test\ndescription: test\nmetadata: " + "[" * 40 + "1" + "]" * 40,
        "name: test\ndescription: test\nmetadata: {a: 1, a: 2}",
    ],
)
def test_unsafe_duplicate_alias_or_excessively_nested_yaml_rejected(tmp_path, header):
    path = tmp_path / "test" / "SKILL.md"
    path.parent.mkdir()
    path.write_text(f"---\n{header}\n---\nbody")
    info, diagnostics, _ = load_skill(path, "explicit")
    assert info is None and diagnostics[0].code == "invalid"
    assert "invalid" not in diagnostics[0].message  # YAML values/snippets are not diagnostics.


def test_runtime_tolerates_name_violations_strict_diagnostics_report_them(tmp_path):
    path = skill(tmp_path, name="BAD--name", description="x" * 1025, extra="unknown-field: ignored\n")
    info, diagnostics, _ = load_skill(path, "explicit")
    assert info.name == "BAD--name" and len(diagnostics) == 2
    assert "unknown-field" not in info.to_json()
    path.write_text("---\ndescription: present\n---\nbody")
    info, diagnostics, _ = load_skill(path, "explicit")
    assert info.name == "BAD--name" and any("Missing/non-text" in item.message for item in diagnostics)


def test_missing_description_and_invalid_utf8_are_not_loaded(tmp_path):
    path = skill(tmp_path)
    path.write_text("---\nname: sample\n---\nbody")
    assert load_skill(path, "explicit")[0] is None
    path.write_bytes(b"\xff")
    assert load_skill(path, "explicit")[0] is None


def test_root_skill_stops_recursion_and_ignore_rules_apply(tmp_path):
    root = tmp_path / "skills"
    top = skill(root)
    skill(top.parent, name="nested")
    skill(root, name="ignored")
    (root / ".gitignore").write_text("ignored/\n")
    (root / "sample" / "cycle").symlink_to(root, target_is_directory=True)
    skills, diagnostics = discover_skills(options(root), tmp_path, tmp_path / "agent")
    assert [item.name for item in skills] == ["sample"] and not diagnostics


def test_symlink_duplicates_cycles_and_broken_links_do_not_hang(tmp_path):
    root = tmp_path / "skills"
    path = skill(root)
    (root / "alias").symlink_to(path.parent, target_is_directory=True)
    (root / "cycle").symlink_to(root, target_is_directory=True)
    (root / "broken").symlink_to("absent")
    skills, _ = discover_skills(options(root, path), tmp_path, tmp_path / "agent")
    assert len(skills) == 1


def test_explicit_project_user_precedence_and_git_boundary(tmp_path, monkeypatch):
    project = tmp_path / "repository"
    cwd = project / "subdir"
    cwd.mkdir(parents=True)
    (project / ".git").mkdir()
    user = tmp_path / "user"
    explicit = tmp_path / "explicit"
    skill(user, description="user")
    skill(project / ".agents/skills", description="outer project")
    nearest = skill(cwd / ".agents/skills", description="near project")
    skill(tmp_path / ".agents/skills", description="outside repository")
    monkeypatch.setattr("pi_agent.skills.default_user_roots", lambda agent: [user])
    snapshot = ResourceSnapshot.discover(
        SkillSettings(enabled=True, project_trusted=True), cwd, tmp_path / "agent"
    )
    assert snapshot.skills[0].file_path == str(nearest) and len(snapshot.diagnostics) == 2
    selected = skill(explicit, description="explicit")
    snapshot = ResourceSnapshot.discover(
        replace(snapshot.settings, paths=(explicit,)), cwd, tmp_path / "agent"
    )
    assert snapshot.skills[0].file_path == str(selected)
    untrusted = ResourceSnapshot.discover(SkillSettings(enabled=True), cwd, tmp_path / "agent")
    assert untrusted.skills[0].source == "user"


def test_ignore_negation_and_nested_rules(tmp_path):
    root = tmp_path / "skills"
    one, two = skill(root, name="one"), skill(root, name="two")
    (root / ".ignore").write_text("*/SKILL.md\n!one/SKILL.md\n")
    (two.parent / ".piignore").write_text("!SKILL.md\n")
    found, _ = discover_skills(options(root), tmp_path, tmp_path / "agent")
    assert {item.file_path for item in found} == {str(one), str(two)}


def test_file_entry_byte_and_prompt_budgets(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    path = skill(root, name="one", description="d" * 800)
    skill(root, name="two", description="d" * 800)
    snapshot = ResourceSnapshot.discover(options(root, prompt_max_bytes=512), tmp_path, tmp_path / "agent")
    section = snapshot.prompt_section({"read"})
    assert len(section.encode()) <= 512 and "Catalog truncated" in section
    assert len(snapshot.skills) == 2 and "BODY_ONLY" not in section
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    assert load_skill(path, "explicit")[0] is None
    monkeypatch.setattr("pi_agent.skills.MAX_ENTRIES", 1)
    _, diagnostics = discover_skills(options(root), tmp_path, tmp_path / "agent")
    assert diagnostics


def test_explicit_non_markdown_and_environment_symlinks_are_never_read(tmp_path):
    environment = tmp_path / ".env"
    environment.write_text("SYNTHETIC_KEY=do-not-load")
    root = tmp_path / "skills"
    root.mkdir()
    (root / "SKILL.md").symlink_to(environment)
    snapshot = ResourceSnapshot.discover(options(root, environment), tmp_path, tmp_path / "agent")
    assert snapshot.skills == () and "do-not-load" not in snapshot.to_manifest()
    assert all("do-not-load" not in item.message for item in snapshot.diagnostics)


def test_fifo_skill_is_not_read(tmp_path):
    path = tmp_path / "SKILL.md"
    os.mkfifo(path)
    info, diagnostics, _ = load_skill(path, "explicit")
    assert info is None and diagnostics


def test_manifest_roundtrip_is_immutable_and_checks_availability(tmp_path):
    path = skill(tmp_path)
    original = ResourceSnapshot.discover(options(tmp_path), tmp_path, tmp_path / "agent")
    restored = ResourceSnapshot.from_manifest(original.to_manifest())
    assert restored.skills == original.skills
    copy = restored.skills[0].to_json()
    copy["name"] = "mutated"
    assert restored.skills[0].name == "sample"
    path.write_text("changed")
    assert restored.check_availability().diagnostics[0].code == "changed"
    with pytest.raises(ValueError, match="changed"):
        restored.expand("/skill:sample request")
    path.unlink()
    assert restored.check_availability().diagnostics[0].code == "missing"


@pytest.mark.parametrize("corruption", ["version", "duplicate", "path", "trust", "nan"])
def test_manifest_corruption_is_rejected(tmp_path, corruption):
    skill(tmp_path)
    snapshot = ResourceSnapshot.discover(options(tmp_path), tmp_path, tmp_path / "agent")
    data = json.loads(snapshot.to_manifest())
    if corruption == "version":
        data["version"] = True
    elif corruption == "duplicate":
        data["skills"].append(data["skills"][0])
    elif corruption == "path":
        data["skills"][0]["file_path"] = str(tmp_path / ".env")
    elif corruption == "trust":
        data["skills"][0]["source"] = "project"
    else:
        data["skills"][0]["metadata"] = {"bad": float("nan")}
    with pytest.raises(ValueError):
        ResourceSnapshot.from_manifest(json.dumps(data))


def test_invalid_files_still_count_against_total_read_budget(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    for name in ("one", "two", "three"):
        path = skill(root, name=name)
        path.write_bytes(b"\xff" * 200)
    monkeypatch.setattr("pi_agent.skills.MAX_TOTAL_BYTES", 300)
    skills, diagnostics = discover_skills(options(root), tmp_path, tmp_path / "agent")
    assert not skills and len(diagnostics) < 5


def test_unpaired_unicode_and_manifest_recursion_reject_cleanly(tmp_path):
    path = skill(tmp_path)
    path.write_text('---\nname: sample\ndescription: "\\ud800"\n---\nbody')
    assert load_skill(path, "explicit")[0] is None
    with pytest.raises(ValueError):
        ResourceSnapshot.from_manifest("[" * 2000 + "1" + "]" * 2000)
    path = skill(tmp_path)
    snapshot = ResourceSnapshot.discover(options(tmp_path), tmp_path, tmp_path / "agent")
    data = json.loads(snapshot.to_manifest())
    nested = [1]
    for _ in range(48):
        nested = [nested]
    data["skills"][0]["metadata"] = {"nested": nested}
    with pytest.raises(ValueError, match="structural"):
        ResourceSnapshot.from_manifest(json.dumps(data))


def test_pi_roots_are_opt_in_and_project_resources_need_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    cwd = tmp_path / "repo"
    cwd.mkdir()
    (cwd / ".git").mkdir()
    skill(cwd / ".pi/skills", name="project-pi")
    skill(tmp_path / ".pi/agent/skills", name="user-pi")
    initial = ResourceSnapshot.discover(SkillSettings(enabled=True), cwd, tmp_path / "agent")
    assert not initial.skills
    user_only = ResourceSnapshot.discover(
        SkillSettings(enabled=True, pi_compat=True), cwd, tmp_path / "agent"
    )
    assert [item.name for item in user_only.skills] == ["user-pi"]
    trusted = ResourceSnapshot.discover(
        SkillSettings(enabled=True, pi_compat=True, project_trusted=True), cwd, tmp_path / "agent"
    )
    assert [item.name for item in trusted.skills] == ["project-pi", "user-pi"]


def test_catalog_xml_escaping_and_optional_metadata_validation(tmp_path):
    path = skill(
        tmp_path / "a&b<z>",
        description='"use a & b < c > d"',
        extra="compatibility: " + "x" * 501 + "\nmetadata: {number: 1}\n",
    )
    snapshot = ResourceSnapshot.discover(options(path), tmp_path, tmp_path / "agent")
    assert len(snapshot.diagnostics) == 2
    section = snapshot.prompt_section({"read"})
    assert "&amp;" in section and "&lt;" in section and "a&b<z>" not in section
