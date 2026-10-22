"""Generate the configuration field reference from packaged editor schemas."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "src" / "pi_agent" / "schemas"
OUTPUT = ROOT / "docs" / "configuration-reference.md"


def fields(properties: dict, prefix: str = "") -> list[str]:
    lines = []
    for name, field in properties.items():
        label = prefix + name
        kind = field.get("type", "enum" if "enum" in field or "const" in field else "object")
        if isinstance(kind, list):
            kind = " | ".join(kind)
        suffix = f" Default: `{json.dumps(field['default'])}`." if "default" in field else ""
        choices = field.get("enum", [field["const"]] if "const" in field else None)
        if choices is not None:
            suffix += " Values: " + ", ".join(f"`{json.dumps(value)}`" for value in choices) + "."
        lines.append(f"- `{label}` ({kind}): {field['description']}{suffix}")
    return lines


def render() -> str:
    lines = [
        "# Configuration field reference",
        "",
        "[Documentation](README.md) · [Configuration](configuration.md)",
        "",
        "Generated from the packaged JSON Schemas. Regenerate with",
        "`python scripts/generate_config_reference.py`; use `--check` to detect drift.",
        "",
        "Schemas provide editor validation. Runtime checks additionally enforce",
        "finite numbers, cross-field constraints, UTF-8 byte limits, trusted sources,",
        "URL authority, reserved headers and filesystem rules. Duplicate JSON keys",
        "are always rejected. A `$schema` reference is never fetched by the runtime.",
        "",
    ]
    for name in ("settings", "mcp"):
        schema = json.loads((SCHEMAS / f"{name}.schema.json").read_text())
        lines += [f"## {schema['title']}", "", schema["description"], ""]
        lines += fields(schema["properties"]) + [""]
        for section, definition in schema["$defs"].items():
            if "properties" in definition:
                lines += [f"### {section}", ""]
                lines += fields(definition["properties"]) + [""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != expected:
            parser.exit(1, "Configuration reference is stale; regenerate it.\n")
    else:
        OUTPUT.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
