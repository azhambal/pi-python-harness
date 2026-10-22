from pathlib import Path

import pytest

from pi_agent.tools import EditTool
from pi_agent.tools.edit import Edit, apply_edits, first_changed_line, generate_unified_patch


async def test_edit_rejects_invalid_utf8_without_changing_file(tmp_path: Path) -> None:
    # CODE_REVIEW 4.8: decoding with replacement silently corrupted unrelated bytes.
    path = tmp_path / "latin1.py"
    original = b"caf\xe9 = 1\nx = 2\n"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="UTF-8"):
        await EditTool(tmp_path).execute(
            "c", {"path": path.name, "edits": [{"oldText": "x = 2", "newText": "x = 3"}]}
        )
    assert path.read_bytes() == original


async def test_edit_preserves_bom_crlf_and_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "file.py"
    path.write_bytes(b"\xef\xbb\xbfa = 1\r\nb = 2\r\n")
    tool = EditTool(tmp_path)
    await tool.execute("c", {"path": path.name, "edits": [{"oldText": "a = 1", "newText": "a = 3"}]})
    expected = b"\xef\xbb\xbfa = 3\r\nb = 2\r\n"
    assert path.read_bytes() == expected
    with pytest.raises(ValueError):
        await tool.execute(
            "c",
            {
                "path": path.name,
                "edits": [{"oldText": "a = 3", "newText": "a"}, {"oldText": "= 3", "newText": "b"}],
            },
        )
    assert path.read_bytes() == expected


async def test_fuzzy_edit_ignores_trailing_spaces(tmp_path: Path) -> None:
    path = tmp_path / "file.py"
    path.write_text("def f():   \n    return 1\n")
    await EditTool(tmp_path).execute(
        "c",
        {
            "path": path.name,
            "edits": [{"oldText": "def f():\n    return 1", "newText": "def f():\n    return 2"}],
        },
    )
    assert "return 2" in path.read_text()


@pytest.mark.parametrize(
    "source,old,new,expected",
    [
        (
            "line one   \nline two  \nline three\n",
            "line one\nline two\n",
            "replaced\n",
            "replaced\nline three\n",
        ),
        (
            "你好，世界\n你好（世界）\n",
            "你好,世界\n你好(世界)\n",
            "你好，pi\n你好(pi)\n",
            "你好，pi\n你好(pi)\n",
        ),
        ("ＡＢＣ１２３\ncafe\u0301\n", "ABC123\ncafé\n", "XYZ789\ncoffee\n", "XYZ789\ncoffee\n"),
        (
            "console.log(‘hello’);\n",
            "console.log('hello');",
            "console.log('world');",
            "console.log('world');\n",
        ),
        ("const msg = “Hello”;\n", 'const msg = "Hello";', 'const msg = "Bye";', 'const msg = "Bye";\n'),
        (
            "range: 1–5\nbreak—here\n",
            "range: 1-5\nbreak-here",
            "range: 10-50\nbreak--here",
            "range: 10-50\nbreak--here\n",
        ),
        ("hello\u00a0world\n", "hello world", "hello universe", "hello universe\n"),
        ("replace me   \nafter   \n", "replace me\n", "after\n", "after\nafter   \n"),
    ],
)
def test_fuzzy_cases_from_pi_tools_spec(source: str, old: str, new: str, expected: str) -> None:
    # ../pi/packages/coding-agent/test/tools.test.ts: edit tool fuzzy matching.
    result = apply_edits(source, [Edit(old, new)], "file.txt")
    assert result.base_content == source
    assert result.new_content == expected


def test_fuzzy_preserves_untouched_unicode_and_duplicate_normalized_lines() -> None:
    source = 'keep = “ﬀ”   \ntarget = “x”   \nkeep = "ff"\t\n'
    result = apply_edits(source, [Edit('target = "x"', 'target = "y"')], "file.txt")
    assert result.new_content == 'keep = “ﬀ”   \ntarget = "y"\nkeep = "ff"\t\n'


def test_mixed_exact_and_fuzzy_multi_edit_uses_shared_coordinates() -> None:
    source = "keep = “ﬀ”   \nalpha = 1   \nmid = “hello”   \nomega = 3 \n"
    result = apply_edits(
        source, [Edit("alpha = 1", "alpha = 22"), Edit('mid = "hello"', "mid = “bye”")], "file.txt"
    )
    assert result.new_content == "keep = “ﬀ”   \nalpha = 22\nmid = “bye”\nomega = 3 \n"


def test_two_disjoint_edits_on_one_fuzzy_line() -> None:
    source = "a = 1; b = “x”; keep = “z”  \n"
    result = apply_edits(source, [Edit("a = 1", "a = 22"), Edit('b = "x"', 'b = "y"')], "file.txt")
    assert result.new_content == 'a = 22; b = "y"; keep = "z"\n'


def test_fuzzy_preserves_last_whitespace_only_line() -> None:
    source = "a = “x”\n   "
    result = apply_edits(source, [Edit('a = "x"', 'a = "y"')], "file.txt")
    assert result.new_content == 'a = "y"\n   '


@pytest.mark.parametrize(
    "source,edits,error",
    [
        ('x = "same"\nx = “same”\n', [Edit('x = "same"', 'x = "new"')], "Found 2"),
        ("hello world   \nhello world\n", [Edit("hello world\n", "new\n")], "Found 2"),
        ("aaa\n", [Edit("aa", "new")], "Found 2"),
        ("alpha = “x”\n", [Edit('alpha = "x"', "a"), Edit('= "x"', "b")], "overlap"),
        ("alpha = “x”   \n", [Edit("   ", "\t"), Edit('alpha = "x"', 'alpha = "y"')], "empty after fuzzy"),
        ("x = 1\n", [Edit("x = 1", "x = 1")], "No changes"),
        ("x = 1\n", [Edit("missing", "new")], "Could not find"),
        ("x = 1\n", [Edit("", "new")], "must not be empty"),
    ],
)
def test_unsafe_or_noop_edits_are_rejected(source: str, edits: list[Edit], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        apply_edits(source, edits, "file.txt")


async def test_fuzzy_bom_crlf_and_patch_apply(tmp_path: Path) -> None:
    import subprocess

    source = "keep = “ﬀ”   \r\na = “x”   \r\nb = 2\r\n"
    path = tmp_path / "file.txt"
    path.write_bytes(b"\xef\xbb\xbf" + source.encode())
    result = await EditTool(tmp_path).execute(
        "c",
        {
            "path": path.name,
            "edits": [{"oldText": 'a = "x"\nb = 2', "newText": 'a = "y"\nb = 3'}],
        },
    )
    expected = 'keep = “ﬀ”   \r\na = "y"\r\nb = 3\r\n'
    assert path.read_bytes() == b"\xef\xbb\xbf" + expected.encode()
    # The patch describes the documented LF/BOM-free matching representation.
    patch_root = tmp_path / "patch-check"
    patch_root.mkdir()
    (patch_root / path.name).write_text(source.replace("\r\n", "\n"), encoding="utf-8")
    patch = result.structured_content["patch"]
    applied = subprocess.run(
        ["git", "apply", "--no-index", "-p0", "-"],
        cwd=patch_root,
        input=patch,
        text=True,
        capture_output=True,
    )
    assert applied.returncode == 0, applied.stderr
    assert (patch_root / path.name).read_text() == expected.replace("\r\n", "\n")
    assert result.details["first_changed_line"] == 2


@pytest.mark.parametrize(
    "source,new",
    [
        ("value = 1", "value = 2"),
        ("value = 1", "value = 2\n"),
        ("value = 1\n", "value = 2"),
        ("keep\nx\ny", "keep\nx\nchanged"),
        ("keep\u2028middle\nx\n", "keep\u2028middle\ny\n"),
    ],
)
def test_patch_without_final_newline_is_applicable(tmp_path: Path, source: str, new: str) -> None:
    import subprocess

    path = tmp_path / "file.txt"
    path.write_text(source, encoding="utf-8")
    patch = generate_unified_patch(path.name, source, new)
    result = subprocess.run(
        ["git", "apply", "--no-index", "-p0", "-"], cwd=tmp_path, input=patch, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert path.read_text() == new


@pytest.mark.parametrize(
    "old,new,line",
    [
        ("unchanged\nold\n", "unchanged\nnew\n", 2),
        ("old", "old\n", 1),
        ("old\n", "", 1),
        ("same\n", "same\n", None),
    ],
)
def test_first_changed_line(old: str, new: str, line: int | None) -> None:
    assert first_changed_line(old, new) == line
