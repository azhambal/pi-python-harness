"""Trusted, runtime-only MCP stdio configuration, independent of the optional SDK.

Only whole-value ``${NAME}`` environment references are supported. References are
resolved immediately before launch, never during parsing. Do not put these runtime
objects in a journal: commands, arguments and paths may themselves contain secrets.
There is deliberately no config serializer, dotenv loading, discovery, or launch.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

MAX_CONFIG_BYTES = 1024 * 1024
MAX_SERVERS = 128
MAX_ARGS = 256
MAX_ENV = 128
MAX_TEXT_BYTES = 8192
DEFAULT_TIMEOUT = 60.0

# Exact lookups only: no os.environ copy, iteration, provider keys, loader hooks,
# proxy settings, Python/Node injection variables, or dynamically selected LC_*.
HOST_ENVIRONMENT_KEYS = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER")

_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]{0,127})\}\Z")
_SERVER_FIELDS = frozenset({"type", "command", "args", "cwd", "env", "enabled", "timeout", "exposure"})
_HTTP_FIELDS = frozenset({"url", "headers", "oauth", "auth"})
_HTTP_SERVER_FIELDS = frozenset({"type", "url", "headers", "enabled", "timeout", "exposure"})
_HTTP_ERROR = "Unsupported MCP HTTP configuration; OAuth and legacy SSE are not supported"
_EXPOSURE_ERROR = 'MCP exposure is unsupported; enabled servers require explicit exposure "direct"'


class McpConfigError(ValueError):
    """A bounded diagnostic containing no config paths, names, values, or secrets."""


@dataclass(frozen=True, slots=True, repr=False)
class StdioServerConfig:
    """Validated host configuration; env maps targets to unresolved references.

    Defaults belong to the parser. Direct construction must supply every field;
    validation also runs here so the resolver cannot accidentally admit literals.
    Disabled entries remain in McpConfig and must be skipped by the runtime.
    """

    name: str
    command: str
    args: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    enabled: bool
    timeout: float
    exposure: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise McpConfigError("Invalid MCP server name")
        _text(self.command, nonempty=True)
        if type(self.args) is not tuple or len(self.args) > MAX_ARGS:
            raise McpConfigError("Invalid MCP args; expected a bounded array of strings")
        for arg in self.args:
            _text(arg)
        if not isinstance(self.cwd, Path) or not self.cwd.is_absolute():
            raise McpConfigError("Invalid MCP cwd; expected an absolute host path")
        _text(str(self.cwd), nonempty=True)
        if type(self.enabled) is not bool:
            raise McpConfigError("Invalid MCP enabled; expected a boolean")
        if self.exposure != "direct":
            raise McpConfigError(_EXPOSURE_ERROR)
        if type(self.timeout) not in (int, float):
            raise McpConfigError("Invalid MCP timeout; expected finite positive seconds")
        try:
            timeout = float(self.timeout)
        except OverflowError:
            raise McpConfigError("Invalid MCP timeout; expected finite positive seconds") from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise McpConfigError("Invalid MCP timeout; expected finite positive seconds")
        object.__setattr__(self, "timeout", timeout)
        if not isinstance(self.env, Mapping) or len(self.env) > MAX_ENV:
            raise McpConfigError("Invalid MCP env; expected a bounded reference mapping")
        references: dict[str, str] = {}
        for target, reference in self.env.items():
            if not isinstance(target, str) or not _ENV_NAME.fullmatch(target):
                raise McpConfigError("Invalid MCP environment variable name")
            if not isinstance(reference, str) or not _ENV_REFERENCE.fullmatch(reference):
                raise McpConfigError("Invalid MCP env; only whole-value ${NAME} references are supported")
            references[target] = reference
        object.__setattr__(self, "env", MappingProxyType(references))

    def __repr__(self) -> str:
        return "StdioServerConfig(<redacted runtime configuration>)"


@dataclass(frozen=True, slots=True, repr=False)
class HttpServerConfig:
    """Streamable HTTP endpoint with explicitly referenced header values only.

    Header references resolve to the complete value (including any Bearer prefix).
    Protocol headers are owned by the transport, never by configuration.
    """

    name: str
    url: str
    headers: Mapping[str, str]
    enabled: bool
    timeout: float
    exposure: str

    def __post_init__(self) -> None:
        # Share validation of identity, timeout, trust/exposure and env reference grammar.
        validated = StdioServerConfig(
            self.name, "http", (), Path("/"), {}, self.enabled, self.timeout, self.exposure
        )
        object.__setattr__(self, "timeout", validated.timeout)
        _text(self.url, nonempty=True)
        try:
            endpoint = urlsplit(self.url)
            port = endpoint.port
            if (
                endpoint.scheme not in ("http", "https")
                or not endpoint.hostname
                or endpoint.username is not None
                or endpoint.password is not None
                or endpoint.query
                or endpoint.fragment
                or any(char.isspace() or ord(char) < 32 for char in self.url)
                or (port is not None and port == 0)
            ):
                raise ValueError
        except ValueError:
            raise McpConfigError(
                "Invalid MCP HTTP endpoint; credentials and query/fragment are unsupported"
            ) from None
        if not isinstance(self.headers, Mapping) or len(self.headers) > MAX_ENV:
            raise McpConfigError("Invalid MCP HTTP headers")
        headers: dict[str, str] = {}
        seen: set[str] = set()
        reserved = {
            "accept",
            "accept-encoding",
            "cookie",
            "content-type",
            "content-length",
            "host",
            "connection",
            "transfer-encoding",
        }
        for name, reference in self.headers.items():
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9!#$%&'*+.^_`|~-]{1,128}", name)
                or name.lower() in seen | reserved
                or name.lower().startswith("mcp-")
                or name.lower() == "last-event-id"
                or not isinstance(reference, str)
                or not _ENV_REFERENCE.fullmatch(reference)
            ):
                raise McpConfigError(
                    "Invalid MCP HTTP headers; use distinct non-protocol headers with ${NAME} references"
                )
            seen.add(name.lower())
            headers[name] = reference
        object.__setattr__(self, "headers", MappingProxyType(headers))

    def __repr__(self) -> str:
        return "HttpServerConfig(<redacted runtime configuration>)"


type ServerConfig = StdioServerConfig | HttpServerConfig


@dataclass(frozen=True, slots=True, repr=False)
class McpConfig:
    servers: tuple[ServerConfig, ...]

    def __post_init__(self) -> None:
        if (
            type(self.servers) is not tuple
            or len(self.servers) > MAX_SERVERS
            or any(not isinstance(server, (StdioServerConfig, HttpServerConfig)) for server in self.servers)
        ):
            raise McpConfigError("Invalid MCP servers")
        if len({server.name for server in self.servers}) != len(self.servers):
            raise McpConfigError("Duplicate MCP server names")

    def __repr__(self) -> str:
        return "McpConfig(<redacted runtime configuration>)"


def _text(value: object, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > MAX_TEXT_BYTES:
        raise McpConfigError("Invalid MCP text value")
    if "\x00" in value or (nonempty and not value.strip()):
        raise McpConfigError("Invalid MCP text value")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise McpConfigError("Invalid MCP text value") from None
    if size > MAX_TEXT_BYTES:
        raise McpConfigError("MCP text value exceeds size limit")
    return value


def _check_source(source: str, project_trusted: bool) -> None:
    if source not in ("explicit", "user", "project") or type(project_trusted) is not bool:
        raise McpConfigError("Invalid MCP configuration source or trust setting")
    if source == "project" and not project_trusted:
        raise McpConfigError("Project MCP configuration requires explicit project trust")


def parse_mcp_config(
    data: object, *, cwd: Path, source: str = "explicit", project_trusted: bool = False
) -> McpConfig:
    """Parse the strict Pi mcpServers stdio subset without reading the environment.

    Missing mcpServers means an empty configuration. There is no source merging or
    project override syntax. Disabled servers still undergo all schema checks;
    only their missing exposure may default to direct, without enabling them.
    Relative server paths use host cwd, never the configuration file's directory.
    """
    _check_source(source, project_trusted)
    if type(data) is not dict or set(data) - {"mcpServers", "$schema"}:
        raise McpConfigError("Invalid MCP configuration; only mcpServers is supported")
    if "$schema" in data:
        _text(data["$schema"], nonempty=True)
    entries = data.get("mcpServers", {})
    if type(entries) is not dict or len(entries) > MAX_SERVERS:
        raise McpConfigError("Invalid MCP mcpServers; expected a bounded object")
    if not isinstance(cwd, Path):
        raise McpConfigError("Invalid MCP host cwd")
    try:
        base = cwd.resolve()
    except (OSError, RuntimeError, ValueError):
        raise McpConfigError("Invalid MCP host cwd") from None
    servers: list[ServerConfig] = []
    for name, entry in entries.items():
        if type(entry) is not dict:
            raise McpConfigError("Invalid MCP server; expected an object")
        enabled = entry.get("enabled", True)
        if type(enabled) is not bool:
            raise McpConfigError("Invalid MCP enabled; expected a boolean")
        if (enabled and "exposure" not in entry) or entry.get("exposure", "direct") != "direct":
            raise McpConfigError(_EXPOSURE_ERROR)
        if entry.get("type") in ("http", "streamable-http") or "url" in entry:
            if set(entry) - _HTTP_SERVER_FIELDS or entry.get("type", "http") not in (
                "http",
                "streamable-http",
            ):
                raise McpConfigError(_HTTP_ERROR)
            servers.append(
                HttpServerConfig(
                    name=name,
                    url=_text(entry.get("url"), nonempty=True),
                    headers=entry.get("headers", {}),
                    enabled=enabled,
                    timeout=entry.get("timeout", DEFAULT_TIMEOUT),
                    exposure="direct",
                )
            )
            continue
        if _HTTP_FIELDS.intersection(entry) or entry.get("type") == "sse":
            raise McpConfigError(_HTTP_ERROR)
        if set(entry) - _SERVER_FIELDS:
            raise McpConfigError("Unsupported MCP server configuration fields")
        if "type" in entry and entry["type"] != "stdio":
            raise McpConfigError("Unsupported MCP transport")
        command = _text(entry.get("command"), nonempty=True)
        args = entry.get("args", [])
        if type(args) is not list or len(args) > MAX_ARGS:
            raise McpConfigError("Invalid MCP args; expected a bounded array of strings")
        server_cwd = _text(entry.get("cwd", str(base)), nonempty=True)
        try:
            resolved_cwd = (base / server_cwd).resolve()
        except (OSError, RuntimeError, ValueError):
            raise McpConfigError("Invalid MCP server cwd") from None
        env = entry.get("env", {})
        if type(env) is not dict:
            raise McpConfigError("Invalid MCP env; expected a bounded reference mapping")
        servers.append(
            StdioServerConfig(
                name=name,
                command=command,
                args=tuple(args),
                cwd=resolved_cwd,
                env=env,
                enabled=enabled,
                timeout=entry.get("timeout", DEFAULT_TIMEOUT),
                exposure="direct",
            )
        )
    return McpConfig(servers=tuple(servers))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise McpConfigError("Duplicate JSON configuration keys are not allowed")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise McpConfigError("Nonfinite JSON numbers are not allowed")


def load_mcp_config(
    path: Path,
    *,
    project_trusted: bool = False,
    source: Literal["explicit", "user", "project"] = "explicit",
    cwd: Path | None = None,
) -> McpConfig:
    """Read at most 1 MiB of UTF-8 JSON from a regular explicitly selected file.

    Trust is checked before any filesystem access. Missing/unreadable files are
    errors; environment files (including symlink aliases) are never opened.
    """
    _check_source(source, project_trusted)
    if not isinstance(path, Path):
        raise McpConfigError("Invalid MCP configuration path")
    try:
        resolved = path.resolve()
        if any(
            candidate.name == ".env" or candidate.name.startswith(".env.") for candidate in (path, resolved)
        ):
            raise McpConfigError("Environment files cannot be used as MCP configuration")
        descriptor = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CONFIG_BYTES:
                raise McpConfigError("MCP configuration must be a regular file of at most 1 MiB")
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise McpConfigError("MCP configuration exceeds the 1 MiB limit")
        data = json.loads(
            raw.decode("utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except McpConfigError:
        raise
    except (OSError, ValueError, RuntimeError, RecursionError):
        raise McpConfigError("MCP configuration could not be read as valid UTF-8 JSON") from None
    return parse_mcp_config(
        data, cwd=cwd if cwd is not None else Path.cwd(), source=source, project_trusted=project_trusted
    )


def resolve_server_environment(
    config: StdioServerConfig, environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build a fresh minimal child environment using exact key lookups only.

    Inherit only HOST_ENVIRONMENT_KEYS, then overlay explicitly declared env refs.
    A supplied mapping is the complete source, with no host fallback. Missing
    explicit references fail; empty values are valid. No resolution is cached.
    Disabled servers return an empty environment without looking up any keys.
    """
    if not config.enabled:
        return {}
    available = os.environ if environment is None else environment
    result: dict[str, str] = {}
    for key in HOST_ENVIRONMENT_KEYS:
        value = available.get(key)
        if value is not None:
            result[key] = _text(value)
    for target, reference in config.env.items():
        # Constructor validation guarantees the exact ${NAME} grammar.
        value = available.get(reference[2:-1])
        if value is None:
            raise McpConfigError("An explicitly referenced MCP environment variable is missing")
        result[target] = _text(value)
    return result


def resolve_server_headers(
    config: HttpServerConfig, environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Exact declared lookups only; no provider credential/proxy/dotenv inheritance."""
    if not config.enabled:
        return {}
    available = os.environ if environment is None else environment
    resolved: dict[str, str] = {}
    for name, reference in config.headers.items():
        value = available.get(reference[2:-1])
        if value is None:
            raise McpConfigError("An explicitly referenced MCP header variable is missing")
        value = _text(value, nonempty=True)
        if any(ord(char) < 32 or ord(char) > 126 for char in value):
            raise McpConfigError("Invalid MCP HTTP header value")
        resolved[name] = value
    return resolved
