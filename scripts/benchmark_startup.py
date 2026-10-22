"""Offline startup/budget workload, with no latency threshold or live provider.

Run .venv/bin/python scripts/benchmark_startup.py. Creates 1000 skill directories,
discovers the capped snapshot, prepares a 256-tool paginated stdio catalog and
checks model-text truncation while retaining the complete structured envelope.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from pi_agent import ResourceOwner, SkillSettings
from pi_agent.mcp import McpManifest, McpRuntime, parse_mcp_config
from pi_agent.resources import ResourceSnapshot

PROJECT = Path(__file__).resolve().parents[1]


async def workload(directory: Path) -> dict:
    skills = directory / "skills"
    for number in range(1000):
        path = skills / f"skill-{number:04d}" / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            f"---\nname: skill-{number:04d}\ndescription: "
            + "Search review test documentation " * 8
            + "\n---\n\n"
            + "Not eagerly injected. " * 200
        )
    settings = SkillSettings(enabled=True, include_user=False, paths=(skills,), prompt_max_bytes=4096)
    start = perf_counter()
    snapshot = ResourceSnapshot.discover(settings, directory, directory / "agent")
    skills_ms = (perf_counter() - start) * 1000
    catalog = snapshot.prompt_section({"read"}) or ""
    config = parse_mcp_config(
        {
            "mcpServers": {
                "load": {
                    "command": sys.executable,
                    "args": [str(PROJECT / "tests/fixtures/mcp_catalog_server.py"), "256"],
                    "timeout": 15,
                    "exposure": "direct",
                }
            }
        },
        cwd=directory,
    )
    runtime = McpRuntime(config, environment={})
    owner = ResourceOwner()
    owner.claim()
    try:
        start = perf_counter()
        tools = await runtime.prepare_tools(owner)
        mcp_ms = (perf_counter() - start) * 1000
        result = await tools[0].execute("load", {})
        result_bytes = len(json.dumps(result.to_json(), ensure_ascii=False, separators=(",", ":")).encode())
        if len(snapshot.skills) != 256 or len(tools) != 256 or len(catalog.encode()) > 4096:
            raise RuntimeError("Startup budget regression")
        if (
            not result.details["truncated"]
            or len(result.content[0].text.encode()) > 22 * 1024
            or Path(result.details["full_output_path"]).read_text()
            != result.structured_content["content"][0]["text"]
        ):
            raise RuntimeError("Result projection budget regression")
        return {
            "generated_skills": 1000,
            "admitted_skills": len(snapshot.skills),
            "skills_ms": round(skills_ms, 2),
            "skill_catalog_bytes": len(catalog.encode()),
            "skill_manifest_bytes": len(snapshot.to_manifest().encode()),
            "skill_diagnostics": len(snapshot.diagnostics),
            "mcp_tools": len(tools),
            "mcp_prepare_ms": round(mcp_ms, 2),
            "mcp_manifest_bytes": len(McpManifest.from_tools(tools).to_manifest().encode()),
            "tool_declarations_bytes": len(
                json.dumps([tool.declaration().to_json() for tool in tools]).encode()
            ),
            "result_json_bytes": result_bytes,
            "structured_envelope_bytes": len(json.dumps(result.structured_content).encode()),
            "model_text_bytes": len(result.content[0].text.encode()),
            "original_text_bytes": len(result.structured_content["content"][0]["text"].encode()),
            "result_truncated": True,
            "timings_are_ci_thresholds": False,
        }
    finally:
        await owner.aclose()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pi-startup-benchmark-") as directory:
        print(json.dumps(asyncio.run(workload(Path(directory))), indent=2))


if __name__ == "__main__":
    main()
