"""Offline acceptance with pinned real MCP servers and a scripted provider.

Run with .venv/bin/python scripts/mcp_acceptance.py. Uses a temporary Git
repository and localhost HTTP only. Never loads dotenv or provider credentials.
The scripted provider verifies routing/behaviour, not autonomous LLM quality.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from pi_agent import (
    AgentConfig,
    CompactionSettings,
    ModelInfo,
    RetrySettings,
    SessionStore,
    SkillSettings,
    create_harness,
    event_to_json,
)
from pi_agent.mcp import binding_name
from pi_agent.mcp.config import resolve_server_environment
from pi_agent.mcp.reference import (
    REFERENCE_SELECTORS,
    REFERENCE_VERSIONS,
    reference_config,
    reference_hooks,
    reference_permissions,
)
from pi_agent.providers import FauxProvider
from pi_agent.tools import TOOL_NAMES
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage, content_text

PROJECT = Path(__file__).resolve().parents[1]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def done(text: str = "done") -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="stop")


def call(identifier: str, name: str, args: dict) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCallBlock(id=identifier, name=name, arguments=args)], stop_reason="toolUse"
    )


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        env={
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_AUTHOR_NAME": "Offline fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Offline fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        },
        check=True,
        capture_output=True,
        timeout=10,
    )


class DocumentServer:
    """Local document/robots fixture. Every handler/writer is owned and joined."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.tasks: set[asyncio.Task] = set()
        self.slow_started = asyncio.Event()
        self.server: asyncio.Server | None = None

    async def __aenter__(self) -> DocumentServer:
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.base = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *args: object) -> None:
        assert self.server is not None
        self.server.close()
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Python 3.12 Server.wait_closed also waits for accepted connections;
        # handlers must close their writers before awaiting this barrier.
        await self.server.wait_closed()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.tasks.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            path = header.split(b" ", 2)[1].decode("ascii")
            self.paths.append(path)
            if path == "/slow":
                self.slow_started.set()
                await asyncio.Event().wait()
            if path == "/robots.txt":
                body = b"User-agent: *\nAllow: /\n"
                mime = "text/plain"
            else:
                body = b"<html><head><title>Arithmetic specification</title></head><body><article><h1>Addition</h1><p>The add function returns a + b. For add(2, 3), the correct answer is 5.</p><p>Change the implementation, then run the existing tests before checking the Git diff.</p></article></body></html>"
                mime = "text/html"
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            self.tasks.discard(task)


def child_pids(session) -> list[int]:
    # Acceptance instrumentation only: process handles are intentionally not a
    # public session API and must not become model-facing status/configuration.
    return [connection._transport._connection.process.pid for connection in session._mcp_runtime._connections]


def reaped(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError("An owned MCP process survived close")


async def acceptance(directory: Path) -> dict:
    installed = {name: version(name) for name in REFERENCE_VERSIONS}
    require(installed == REFERENCE_VERSIONS, "Install the exact mcp-reference extra versions")
    require(sys.version_info[:2] == (3, 12), "Acceptance requires Python 3.12")
    root = directory / "repository"
    root.mkdir()
    (directory / "home").mkdir()
    (root / "arithmetic.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_arithmetic.py").write_text(
        "from arithmetic import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\nignored.txt\n")
    (root / "ignored.txt").write_text("IGNORED_FIXTURE_CONTENT")
    git(root, "init", "-q")
    git(root, "add", "arithmetic.py", "test_arithmetic.py", ".gitignore")
    git(root, "commit", "-qm", "Seed the offline arithmetic fixture")
    skill = directory / "skills" / "review-python" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT / "examples/skills/review-python/SKILL.md", skill)
    require(
        skill.read_bytes() == (PROJECT / "examples/skills/review-python/SKILL.md").read_bytes(),
        "Skill was rewritten",
    )
    cfg = AgentConfig(
        cwd=root,
        model=ModelInfo(id="offline", name="Offline", provider="faux", context_window=100_000),
        agent_dir=directory / "agent",
        session_dir=directory / "sessions",
        instruction_files=[],
        skills=SkillSettings(enabled=True, include_user=False, paths=(skill.parent.parent,)),
        compaction=CompactionSettings(enabled=False, keep_recent_tokens=0),
        retry=RetrySettings(enabled=False),
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(directory / "home"),
        "TMPDIR": str(directory),
        "OPENAI_API_KEY": "synthetic-provider-secret",
    }
    settings = reference_config(root, Path(sys.executable), allow_private_fetch=True)
    require(
        all(
            "OPENAI_API_KEY" not in resolve_server_environment(server, environment)
            for server in settings.servers
        ),
        "Provider key inherited by MCP",
    )

    def git_name(name: str) -> str:
        return binding_name("git", name)

    fetch_name = binding_name("fetch", "fetch")
    pytest_command = (
        "env -i PATH=/usr/bin:/bin " + shlex.quote(sys.executable) + " -m pytest -q test_arithmetic.py"
    )
    spill_command = (
        "env -i PATH=/usr/bin:/bin "
        + shlex.quote(sys.executable)
        + " -c "
        + shlex.quote("print('x' * 70000)")
    )
    started = perf_counter()
    async with DocumentServer() as documents:
        confirmations = []

        def confirm(request, token):
            confirmations.append(request.context.args.get("url"))
            return request.context.args.get("url") in (documents.base + "/spec", documents.base + "/slow")

        responses = [
            call("outside", git_name("git_status"), {"repo_path": str(directory)}),
            call("denied-url", fetch_name, {"url": documents.base + "/denied"}),
            call("list", "ls", {"path": "."}),
            call("files", "find", {"pattern": "*.py"}),
            call("ignored", "grep", {"pattern": "IGNORED_FIXTURE_CONTENT"}),
            call("search", "grep", {"pattern": "return a", "glob": "arithmetic.py"}),
            call("source", "read", {"path": "arithmetic.py"}),
            call("before", "bash", {"command": pytest_command}),
            call("spec", fetch_name, {"url": documents.base + "/spec", "max_length": 1000}),
            call(
                "edit",
                "edit",
                {"path": "arithmetic.py", "edits": [{"oldText": "return a - b", "newText": "return a + b"}]},
            ),
            call("after", "bash", {"command": pytest_command}),
            call("status", git_name("git_status"), {"repo_path": str(root)}),
            call("diff", git_name("git_diff_unstaged"), {"repo_path": str(root)}),
            call("staged", git_name("git_diff_staged"), {"repo_path": str(root)}),
            call("compare", git_name("git_diff"), {"repo_path": str(root), "target": "HEAD"}),
            call("log", git_name("git_log"), {"repo_path": str(root), "max_count": 1}),
            call("show", git_name("git_show"), {"repo_path": str(root), "revision": "HEAD"}),
            call("branch", git_name("git_branch"), {"repo_path": str(root), "branch_type": "local"}),
            call("spill", "bash", {"command": spill_command}),
            done(),
            done("Second turn"),
            done("Summary: addition fixed and tested; MCP/skill loadout preserved."),
            # keep_recent_tokens=0 cuts at the last answer, so its turn prefix is summarized too.
            done("Turn prefix: keep this result for continuation."),
        ]
        provider = FauxProvider(responses)
        events = []
        preparation = perf_counter()
        session = await create_harness(
            provider,
            config=cfg,
            session="new",
            tool_names=TOOL_NAMES,
            mcp_config=settings,
            mcp_tools=REFERENCE_SELECTORS,
            mcp_environment=environment,
            hooks=reference_hooks(root),
            permission_policy=reference_permissions(),
            confirm_permission=confirm,
        )
        preparation_ms = (perf_counter() - preparation) * 1000
        pids = child_pids(session)
        session.subscribe(events.append)
        try:
            require(
                "Read the relevant source and callers" not in session.system_prompt,
                "Skill body eagerly injected",
            )
            await session.prompt(
                "/skill:review-python Fix addition using the local specification and check the result."
            )
            results = {
                message.tool_call_id: message
                for message in session.messages
                if isinstance(message, ToolResultMessage)
            }
            require(
                results["outside"].is_error and results["denied-url"].is_error, "Profile admission failed"
            )
            require(
                results["before"].is_error and not results["after"].is_error,
                "Behavioural test did not fail before/pass after edit",
            )
            for identifier in (
                "list",
                "files",
                "ignored",
                "search",
                "source",
                "spec",
                "edit",
                "status",
                "diff",
                "staged",
                "compare",
                "log",
                "show",
                "branch",
                "spill",
            ):
                require(not results[identifier].is_error, f"Acceptance tool failed: {identifier}")
            require(
                "a + b" in content_text(results["spec"].content), "Fetch HTML conversion lost specification"
            )
            require(
                "+    return a + b" in content_text(results["diff"].content), "Git diff did not observe edit"
            )
            require(
                "Seed the offline" in content_text(results["log"].content),
                "Git log did not observe initial commit",
            )
            require(
                "/denied" not in documents.paths and documents.paths.count("/spec") == 1,
                "Fetch admission/replay failed",
            )
            require(
                "No matches found" in content_text(results["ignored"].content),
                "Ignore rules failed",
            )
            spill = Path(results["spill"].details["full_output_path"])
            require(spill.is_file(), "Bash spill was not created")
            bindings = session.mcp_bindings
            await session.prompt("Keep this result for continuation")
            await session.compact()
            require(session.mcp_bindings == bindings, "Compaction changed MCP bindings")
            path = session.session_file
            require(path is not None, "Acceptance journal missing")
            require(
                "synthetic-provider-secret" not in path.read_text(), "Synthetic credential leaked to journal"
            )
            require(
                "synthetic-provider-secret" not in json.dumps([event_to_json(event) for event in events]),
                "Synthetic credential leaked to events",
            )
            mcp_result_bytes = max(
                len(json.dumps(result.to_json(), ensure_ascii=False).encode())
                for result in results.values()
                if result.tool_name.startswith("mcp__")
            )
        finally:
            await session.aclose()
        reaped(pids)
        # Bash full-output files stay after close, as in pi.
        require(spill.is_file(), "Bash full-output file vanished on close")
        spill.unlink()
        SessionStore.open(path).close()
        before_resume = path.read_bytes()
        continuation = FauxProvider(
            [call("slow-fetch", fetch_name, {"url": documents.base + "/slow"}), done()]
        )
        restored = await create_harness(
            continuation,
            session=path,
            mcp_config=settings,
            mcp_environment=environment,
            hooks=reference_hooks(root),
            permission_policy=reference_permissions(),
            confirm_permission=confirm,
        )
        pids = child_pids(restored)
        task = None
        try:
            require(restored.mcp_bindings == bindings, "Resume changed MCP selection")
            require(path.read_bytes() == before_resume, "Resume silently rewrote journal")
            task = asyncio.create_task(restored.prompt("Fetch the delayed local document"))
            async with asyncio.timeout(10):
                await documents.slow_started.wait()
            await restored.abort()
            await task
            require(documents.paths.count("/slow") == 1, "Aborted Fetch replayed")
            require(restored.mcp_server_states["fetch"] != "ready", "Abort retained live Fetch connection")
            await restored.reconnect_mcp("fetch")
            require(restored.mcp_server_states["fetch"] == "ready", "Explicit reconnect failed")
            require(documents.paths.count("/slow") == 1, "Reconnect replayed old Fetch")
            pids += child_pids(restored)
        finally:
            await restored.abort()
            if task is not None:
                await task
            await restored.aclose()
        reaped(pids)
        SessionStore.open(path).close()
        require(
            not any(
                task.get_name().startswith(("pi-mcp-", "mcp-stdio-"))
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task()
            ),
            "MCP worker survived cleanup",
        )
    return {
        "status": "passed",
        "python": ".".join(map(str, sys.version_info[:3])),
        "versions": installed,
        "active_mcp_bindings": len(bindings),
        "prepare_ms": round(preparation_ms, 2),
        "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        "largest_mcp_result_message_bytes": mcp_result_bytes,
        "model_requests": 0,
        "checks": [
            "search",
            "unchanged-skill",
            "fetch-html",
            "repo-admission",
            "fetch-confirmation",
            "test-fails-before-passes-after",
            "git-status-diff-log",
            "compaction",
            "strict-resume",
            "abort-no-replay",
            "explicit-reconnect",
            "child-task-lock-cleanup",
            "synthetic-credentials-excluded",
        ],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pi-mcp-acceptance-") as directory:
        print(json.dumps(asyncio.run(acceptance(Path(directory))), indent=2))


if __name__ == "__main__":
    main()
