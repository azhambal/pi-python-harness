"""One-task SDK lifetime over owned bounded transports; no tool-call replay."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from pi_agent._async_resources import (
    _IN_RESOURCE_CALLBACK,
    ResourceOwner,
    reject_cleanup_reentry,
    settle_cleanup,
)
from pi_agent.cancellation import Cancellation, OperationAbortedError
from pi_agent.tools._spill import SpillRegistry
from pi_agent.tools.base import Tool
from pi_agent.types import JsonValue

from ._contracts import McpError, McpLimits, bounded_json
from ._http import DEFAULT_HTTP_LIMITS, HttpLimits, HttpTransport, bounded_http
from ._stdio import StdioLimits, Transport, bounded_stdio
from .config import (
    HttpServerConfig,
    McpConfig,
    ServerConfig,
    resolve_server_environment,
    resolve_server_headers,
)

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.shared.dispatcher import ProgressFnT
    from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher
    from mcp.types import CallToolResult, InitializeResult

DEFAULT_MCP_LIMITS = McpLimits()
DEFAULT_STDIO_LIMITS = StdioLimits()


class McpConnection:
    """A connection is single-flight and terminal after cancel, timeout or failure.

    Every entered SDK context exits in its owner task. A cancelled request closes
    the connection and waits for child cleanup; the runtime stages explicit reconnects.
    """

    def __init__(
        self,
        config: ServerConfig,
        *,
        limits: McpLimits = DEFAULT_MCP_LIMITS,
        stdio_limits: StdioLimits = DEFAULT_STDIO_LIMITS,
        http_limits: HttpLimits = DEFAULT_HTTP_LIMITS,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.limits = limits
        self.stdio_limits = stdio_limits
        self.http_limits = http_limits
        self._environment = environment
        self._ready: asyncio.Future[InitializeResult] | None = None
        self._stop = asyncio.Event()
        self._owner: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._client: ClientSession | None = None
        self._dispatcher: JSONRPCDispatcher | None = None
        self._transport: Transport | HttpTransport | None = None
        self._requests: set[asyncio.Task[Any]] = set()
        self.spills = SpillRegistry()
        self.state = "new"

    async def connect(self) -> InitializeResult:
        reject_cleanup_reentry(self, "MCP connect")
        if self._stop.is_set():
            raise McpError("MCP connection is closed; explicit reconnection is required")
        if self._owner is None:
            self.state = "connecting"
            self._ready = asyncio.get_running_loop().create_future()
            self._ready.add_done_callback(lambda value: None if value.cancelled() else value.exception())
            self._owner = asyncio.create_task(self._run(), name="pi-mcp-owner")
        assert self._ready is not None
        try:
            return await asyncio.shield(self._ready)
        except asyncio.CancelledError:
            await self.aclose()
            raise

    async def _run(self) -> None:
        assert self._ready is not None
        initializing: asyncio.Task[InitializeResult] | None = None
        stopping: asyncio.Task[bool] | None = None
        failed: asyncio.Task[bool] | None = None
        error: McpError | None = None
        try:
            try:
                from mcp import ClientSession
                from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher
            except ImportError:
                error = McpError("Install the optional MCP extra: pi-agent[mcp]")
                return

            async def incoming(message: object) -> None:
                # Unsolicited messages/errors carry untrusted server data. SDK
                # protocol handlers still handle ping and progress independently.
                return None

            context: AbstractAsyncContextManager[Transport | HttpTransport]
            if isinstance(self.config, HttpServerConfig):
                context = bounded_http(
                    self.config.url,
                    resolve_server_headers(self.config, self._environment),
                    self.config.timeout,
                    self.http_limits,
                )
            else:
                context = bounded_stdio(
                    self.config.command,
                    self.config.args,
                    self.config.cwd,
                    resolve_server_environment(self.config, self._environment),
                    self.stdio_limits,
                )
            async with context as transport:
                self._transport = transport
                dispatcher = JSONRPCDispatcher(transport.read_stream, transport.write_stream)
                async with ClientSession(
                    dispatcher=dispatcher,
                    read_timeout_seconds=self.config.timeout,
                    message_handler=incoming,
                ) as client:
                    self._client = client
                    self._dispatcher = dispatcher
                    initializing = asyncio.create_task(client.initialize())
                    stopping = asyncio.create_task(self._stop.wait())
                    failed = asyncio.create_task(transport.failed.wait())
                    try:
                        async with asyncio.timeout(self.config.timeout):
                            await asyncio.wait(
                                (initializing, stopping, failed), return_when=asyncio.FIRST_COMPLETED
                            )
                            if self._stop.is_set() or transport.failed.is_set():
                                raise McpError("MCP connection closed during initialize")
                            initialized = initializing.result()
                        self.state = "ready"
                        self._ready.set_result(initialized)
                        await asyncio.wait((stopping, failed), return_when=asyncio.FIRST_COMPLETED)
                        if transport.failed.is_set() and not self._stop.is_set():
                            error = McpError("MCP transport failed; connection closed")
                    finally:
                        # Signal before cancelling requests so their cancellation
                        # handlers do not try to join this owner from themselves.
                        self._stop.set()
                        self._client = None
                        self._dispatcher = None
                        tasks = [initializing, stopping, failed, *self._requests]
                        for task in tasks:
                            if task is not None:
                                task.cancel()
                        await asyncio.gather(
                            *(task for task in tasks if task is not None), return_exceptions=True
                        )
        except Exception:
            error = McpError("MCP connection failed; check trusted configuration and server availability")
        finally:
            self._stop.set()
            self._client = None
            self._dispatcher = None
            self.state = "failed" if error is not None else "closed"
            if not self._ready.done():
                self._ready.set_exception(error or McpError("MCP connection closed during initialize"))

    async def request[T](
        self,
        operation: Callable[[ClientSession], Awaitable[T]],
        cancellation: Cancellation | None = None,
    ) -> T:
        reject_cleanup_reentry(self, "MCP request")
        if cancellation is not None:
            cancellation.raise_if_aborted()
        if self.state != "ready" or self._client is None or self._stop.is_set():
            raise McpError("MCP connection is unavailable; explicit reconnection is required")
        client = self._client

        async def invoke() -> T:
            return await operation(client)

        task = asyncio.create_task(invoke())
        self._requests.add(task)
        aborted = asyncio.create_task(cancellation.wait()) if cancellation is not None else None
        try:
            async with asyncio.timeout(self.config.timeout):
                await asyncio.wait(
                    (task, aborted) if aborted is not None else (task,),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancellation is not None:
                    cancellation.raise_if_aborted()
                return task.result()
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            caller_cancelled = caller is not None and caller.cancelling() > 0
            await self.aclose()
            if caller_cancelled:
                raise
            raise McpError(
                "MCP connection closed during request; outcome unknown. Inspect effects before retrying."
            ) from None
        except OperationAbortedError:
            # Close before propagating interruption. Side effects cannot be rolled
            # back; this request must never be retried automatically.
            await self.aclose()
            raise
        except Exception:
            await self.aclose()
            raise McpError(
                "MCP request failed or timed out; outcome unknown. Inspect effects before retrying."
            ) from None
        finally:
            task.cancel()
            if aborted is not None:
                aborted.cancel()
            await asyncio.gather(task, *([aborted] if aborted is not None else []), return_exceptions=True)
            self._requests.discard(task)

    async def aclose(self) -> None:
        reject_cleanup_reentry(self, "MCP close")
        self._stop.set()
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close())
        await settle_cleanup(self._close_task)

    async def _close(self) -> None:
        if self._owner is not None:
            await settle_cleanup(self._owner)
        else:
            self.state = "closed"
        if self._transport is not None:
            # Retain the process cleanup handle even after context-exit failure.
            # ResourceOwner only releases this connection after verified cleanup.
            try:
                await self._transport.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                raise McpError("MCP transport cleanup failed; close may be retried") from None

        try:
            self.spills.close()
        except Exception:
            raise McpError("MCP output cleanup failed; close may be retried") from None

    async def discover(self) -> list[Tool]:
        try:
            from mcp import types
        except ImportError:
            raise McpError("Install the optional MCP extra: pi-agent[mcp]") from None

        from .resources import resource_tools
        from .tools import McpTool

        initialized = await self.connect()
        resources: list[Tool] = (
            list(resource_tools(self)) if initialized.capabilities.resources is not None else []
        )
        if len(resources) > self.limits.max_tools:
            await self.aclose()
            raise McpError("MCP resource tool count limit exceeded")
        if initialized.capabilities.tools is None:
            return resources
        tools: list[Tool] = []
        seen_names: set[str] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None
        catalog_bytes = 0
        try:
            # One deadline for the whole catalog, not a fresh timeout per page.
            async with asyncio.timeout(self.config.timeout):
                for _ in range(self.limits.max_pages):

                    async def fetch_page(
                        client: ClientSession, page_cursor: str | None = cursor
                    ) -> types.ListToolsResult:
                        raw = await self._request_raw(
                            client, "tools/list", {"cursor": page_cursor} if page_cursor is not None else {}
                        )
                        return types.ListToolsResult.model_validate(raw, strict=True, by_name=False)

                    page = await self.request(fetch_page)
                    catalog_bytes += len(
                        bounded_json(
                            page.model_dump(mode="json", by_alias=True, exclude_none=True),
                            self.limits.catalog_bytes,
                        ).encode("utf-8")
                    )
                    if catalog_bytes > self.limits.catalog_bytes:
                        raise McpError("MCP catalog byte limit exceeded")
                    for descriptor in page.tools:
                        if descriptor.name in seen_names or len(tools) >= self.limits.max_tools:
                            raise McpError("MCP duplicate tool or tool count limit exceeded")
                        seen_names.add(descriptor.name)
                        tools.append(McpTool.from_descriptor(self, descriptor))
                    cursor = page.next_cursor
                    if cursor is None:
                        if len(tools) + len(resources) > self.limits.max_tools:
                            raise McpError("MCP tool count limit exceeded")
                        return [*tools, *resources]
                    if not cursor or len(cursor.encode("utf-8")) > 1024 or cursor in seen_cursors:
                        raise McpError("Invalid or repeated MCP catalog cursor")
                    seen_cursors.add(cursor)
                raise McpError("MCP catalog page limit exceeded")
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception:
            await self.aclose()
            raise McpError(
                "MCP tool discovery failed; catalog or schema is invalid or exceeds limits"
            ) from None

    async def _request_raw(
        self,
        client: ClientSession,
        method: str,
        params: dict[str, JsonValue],
        progress: ProgressFnT | None = None,
    ) -> dict[str, Any]:
        from mcp.shared.dispatcher import CallOptions

        assert self._dispatcher is not None and client.protocol_version is not None
        options: CallOptions = {
            "timeout": self.config.timeout,
            "headers": {"MCP-Protocol-Version": client.protocol_version},
        }
        if progress is not None:
            options["on_progress"] = progress
        # Pi accepts omitted input types and arbitrary legacy structuredContent.
        # Host validation replaces only the SDK's version-specific tool checks.
        return await self._dispatcher.send_raw_request(method, params, options)

    async def call(
        self,
        name: str,
        args: dict[str, JsonValue],
        cancellation: Cancellation | None,
        progress: ProgressFnT | None,
    ) -> CallToolResult:
        from mcp import types

        callback = progress

        async def guarded_progress(progress: float, total: float | None, message: str | None) -> None:
            marker = _IN_RESOURCE_CALLBACK.set(True)
            try:
                assert callback is not None
                await callback(progress, total, message)
            except Exception:
                # SDK logs callback exceptions independently of message_handler.
                raise McpError("MCP progress callback failed") from None
            finally:
                _IN_RESOURCE_CALLBACK.reset(marker)

        async def invoke(client: ClientSession) -> types.CallToolResult:
            from ._results import WireCallToolResult

            raw = await self._request_raw(
                client,
                "tools/call",
                {"name": name, "arguments": args},
                guarded_progress if callback is not None else None,
            )
            return WireCallToolResult.from_wire(raw)

        return await self.request(invoke, cancellation)


class McpStage:
    """Runtime-owned, unpublished remote bindings; never includes builtins.

    Only the creating runtime can publish this stage. Discarding candidates and
    retiring replaced connections are separate, cancellation-safe operations.
    """

    def __init__(
        self,
        runtime: McpRuntime,
        connections: tuple[McpConnection, ...],
        replace_servers: frozenset[str] | None,
    ) -> None:
        self._runtime = runtime
        self.connections = connections
        self.tools: tuple[Tool, ...] = ()
        self._replace_servers = replace_servers
        self._generation = runtime._generation
        self._ready = False
        self._discarded = False
        self._committed = False
        self._pending = list(connections)
        self._previous: list[McpConnection] = []
        self._close_task: asyncio.Task[None] | None = None
        self._retire_task: asyncio.Task[None] | None = None

    @property
    def committed(self) -> bool:
        return self._committed

    async def aclose(self) -> None:
        reject_cleanup_reentry(self._runtime, "MCP stage close")
        if self._committed:
            raise McpError("Cannot discard a committed MCP stage")
        self._discarded = True
        if self._close_task is None or self._close_task.done():
            self._close_task = self._runtime._cleanup_task(self._runtime._discard_serialized(self))
        await settle_cleanup(self._close_task)


class McpRuntime:
    """Owned MCP preparation, candidate staging and synchronous publication.

    The parent validates schemas/manifest and appends its journal before calling
    publish without an intervening await. validate_stage is available before that
    append. A partial stage exposes only the replacement servers' remote tools;
    complete_tools supplied by the parent includes retained bindings and builtins.
    """

    def __init__(
        self,
        config: McpConfig,
        *,
        builtins: tuple[Tool, ...] = (),
        limits: McpLimits = DEFAULT_MCP_LIMITS,
        stdio_limits: StdioLimits = DEFAULT_STDIO_LIMITS,
        http_limits: HttpLimits = DEFAULT_HTTP_LIMITS,
        environment: Mapping[str, str] | None = None,
        selectors: Sequence[str] | None = None,
        servers: Sequence[str] | None = None,
    ) -> None:
        self._config = config  # Trusted runtime-only config; its repr is redacted.
        self._builtins = builtins
        self._limits = limits
        self._stdio_limits = stdio_limits
        self._http_limits = http_limits
        self._environment = environment
        self._initial_selectors = self._selection(selectors)
        self._initial_servers = self._server_selection(servers)
        self._connections: tuple[McpConnection, ...] = ()
        # Registration precedes every discovery/spawn await. Failed cleanup stays
        # in this pool even if the caller never received a stage handle.
        self._owned_connections: list[McpConnection] = []
        self._stages: list[McpStage] = []
        self._setup_tasks: set[asyncio.Task[McpStage]] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._stage_lock = asyncio.Lock()
        self._generation = 0
        self._known_bindings: dict[str, str] = {}
        self._prepared: tuple[Tool, ...] | None = None
        self._prepare_task: asyncio.Task[tuple[Tool, ...]] | None = None
        self._owner: ResourceOwner | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @staticmethod
    def _selection(values: Sequence[str] | None) -> tuple[str, ...] | None:
        if values is None:
            return None
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise McpError("Invalid MCP selection")
        selected = tuple(values)
        if any(not isinstance(value, str) or not value for value in selected):
            raise McpError("Invalid MCP selection")
        if len(set(selected)) != len(selected):
            raise McpError("Duplicate MCP selection")
        return selected

    def _server_selection(self, servers: Sequence[str] | None) -> tuple[str, ...] | None:
        selected = self._selection(servers)
        if selected is not None:
            enabled = {server.name for server in self._config.servers if server.enabled}
            if not set(selected) <= enabled:
                raise McpError("Selected MCP server is unknown or disabled")
        return selected

    @property
    def server_states(self) -> dict[str, str]:
        """Non-secret configured status; candidates never replace active status."""
        active = {connection.config.name: connection.state for connection in self._connections}
        return {
            server.name: active.get(server.name, "inactive") if server.enabled else "disabled"
            for server in self._config.servers
        }

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The currently published complete tool registry."""
        return self._prepared or ()

    async def prepare_tools(self, owner: ResourceOwner) -> tuple[Tool, ...]:
        reject_cleanup_reentry(self, "MCP preparation")
        if self._closed:
            raise McpError("MCP runtime is closed")
        if self._owner is not None and self._owner is not owner:
            raise McpError("MCP runtime is already owned by another harness")
        self._owner = owner
        owner.add(self)  # Before starting any subprocess or connection await.
        if self._prepare_task is None:
            self._prepare_task = asyncio.create_task(self._prepare())
        try:
            return await asyncio.shield(self._prepare_task)
        except asyncio.CancelledError:
            await self.aclose()
            raise

    async def _prepare(self) -> tuple[Tool, ...]:
        stage = await self.stage(self._initial_selectors, servers=self._initial_servers)
        try:
            tools = (*self._builtins, *stage.tools)
            if len({tool.name for tool in tools}) != len(tools):
                raise McpError("MCP tool binding collision")
            self.validate_stage(stage, tools)
            self.publish(stage, tools)
            await self.retire_previous(stage)
            return tools
        except BaseException:
            if not stage.committed:
                await stage.aclose()
            raise

    async def stage(
        self, selectors: Sequence[str] | None = None, *, servers: Sequence[str] | None = None
    ) -> McpStage:
        """Discover fresh remote candidates without changing active bindings.

        None selects all enabled servers/tools; an empty selection starts none.
        Aliases and known canonical names infer the required server scope. Fresh
        unknown hashes require full catalog discovery unless servers is supplied.
        Explicit servers is an exact enabled scope, never augmented by selectors.
        """
        reject_cleanup_reentry(self, "MCP staging")
        if self._closed:
            raise McpError("MCP runtime is closed")
        selected = self._selection(selectors)
        selected_servers = self._server_selection(servers)
        if selected_servers is None and selected is not None:
            selected_servers = self._selector_servers(selected)
        task = asyncio.create_task(self._setup(selected, selected_servers), name="pi-mcp-stage")
        self._setup_tasks.add(task)
        task.add_done_callback(self._setup_tasks.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Cancel the dedicated setup and join its candidate cleanup even if
            # this waiter is cancelled repeatedly, or setup already completed.
            await settle_cleanup(self._cleanup_task(self._cancel_setup(task)))
            raise

    def _selector_servers(self, selectors: tuple[str, ...]) -> tuple[str, ...] | None:
        """Avoid unrelated launches for aliases and previously discovered hashes.

        An unknown canonical hash cannot be reversed. Only that case falls back
        to the entire enabled catalog; resume callers provide an exact servers
        scope to avoid this fallback. Explicit servers always take precedence.
        """
        required: set[str] = set()
        unknown = False
        for selector in selectors:
            if "/" in selector:
                server, _, tool = selector.partition("/")
                if not tool:
                    raise McpError("Invalid MCP tool selection")
                required.add(server)
            elif selector in self._known_bindings:
                required.add(self._known_bindings[selector])
            else:
                unknown = True
        # Validate known/alias scopes even when another hash requires discovery.
        self._server_selection(tuple(required))
        if unknown:
            return None
        return tuple(server.name for server in self._config.servers if server.name in required)

    async def _cancel_setup(self, task: asyncio.Task[McpStage]) -> None:
        task.cancel()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        if isinstance(result, McpStage) and not result.committed:
            await result.aclose()

    async def _setup(self, selectors: tuple[str, ...] | None, servers: tuple[str, ...] | None) -> McpStage:
        async with self._stage_lock:
            if self._closed:
                raise McpError("MCP runtime is closed")
            connections = tuple(
                McpConnection(
                    server,
                    limits=self._limits,
                    stdio_limits=self._stdio_limits,
                    http_limits=self._http_limits,
                    environment=self._environment,
                )
                for server in self._config.servers
                if server.enabled and selectors != () and (servers is None or server.name in servers)
            )
            stage = McpStage(self, connections, None if servers is None else frozenset(servers))
            self._stages.append(stage)
            self._owned_connections.extend(connections)
            try:
                tools: list[Tool] = []
                for connection in connections:
                    tools.extend(await connection.discover())
                if len({tool.name for tool in tools}) != len(tools):
                    raise McpError("MCP tool binding collision")
                for tool in tools:
                    provenance = tool.provenance
                    assert provenance is not None and provenance.server is not None
                    self._known_bindings[tool.name] = provenance.server
                if selectors is not None:
                    aliases: dict[str, Tool] = {}
                    for tool in tools:
                        aliases[tool.name] = tool
                        provenance = tool.provenance
                        assert provenance is not None
                        aliases[f"{provenance.server}/{provenance.tool}"] = tool
                    chosen: set[str] = set()
                    for selector in selectors:
                        selected_tool = aliases.get(selector)
                        if selected_tool is None:
                            raise McpError("Selected MCP tool is unavailable")
                        if selected_tool.name in chosen:
                            raise McpError("Duplicate MCP tool selection")
                        chosen.add(selected_tool.name)
                    # Catalog/config order remains deterministic for both aliases.
                    tools = [tool for tool in tools if tool.name in chosen]
                stage.tools = tuple(tools)
                stage._ready = True
                return stage
            except BaseException:
                stage._discarded = True
                # Cleanup gets its own shielded task, while setup retains the lock.
                # An error never changes active bindings; failed cleanup stays owned.
                await settle_cleanup(self._cleanup_task(self._discard(stage)))
                raise

    def validate_stage(self, stage: McpStage, complete_tools: Sequence[Tool] | None = None) -> None:
        """Check publication ownership/lifecycle before the parent's durable append.

        Publication repeats only these programmer-misuse checks; tool/schema and
        journal validation belong to the parent, never to this commit operation.
        """
        reject_cleanup_reentry(self, "MCP publication")
        if not isinstance(stage, McpStage) or stage._runtime is not self:
            raise McpError("MCP stage belongs to another runtime")
        if self._closed or stage._discarded or stage._committed or not stage._ready:
            raise McpError("MCP stage cannot be published")
        if stage._generation != self._generation:
            raise McpError("MCP stage is stale")
        if complete_tools is not None:
            self._publication_connections(stage, tuple(complete_tools))

    def _publication_connections(self, stage: McpStage, tools: tuple[Tool, ...]) -> tuple[McpConnection, ...]:
        from .tools import McpTool

        available = (*self._connections, *stage.connections)
        active: dict[str, McpConnection] = {}
        for tool in tools:
            if not isinstance(tool, McpTool):
                continue
            connection = tool._connection
            if connection not in available:
                raise McpError("MCP tool belongs to an unavailable connection")
            name = connection.config.name
            if name in active and active[name] is not connection:
                raise McpError("MCP loadout mixes connections for one server")
            active[name] = connection
        # No requirement to retain a current binding: the parent's final loadout
        # is authoritative, including removal of servers outside stage targets.
        return tuple(active.values())

    def publish(self, stage: McpStage, complete_tools: Sequence[Tool]) -> None:
        """Commit synchronously, without callbacks, awaits or schema validation."""
        tools = tuple(complete_tools)
        self.validate_stage(stage, tools)
        connections = self._publication_connections(stage, tools)
        previous = [
            connection
            for connection in (*self._connections, *stage.connections)
            if connection not in connections
        ]
        self._connections = connections
        self._prepared = tools
        stage._previous = previous
        stage._pending = []
        stage._committed = True
        self._generation += 1

    async def retire_previous(self, stage: McpStage) -> None:
        """Close replaced connections; failure keeps the committed bindings active."""
        reject_cleanup_reentry(self, "MCP retirement")
        if not isinstance(stage, McpStage) or stage._runtime is not self or not stage.committed:
            raise McpError("MCP retirement requires this runtime's committed stage")
        if stage._retire_task is None or stage._retire_task.done():
            stage._retire_task = self._cleanup_task(self._retire_serialized(stage))
        await settle_cleanup(stage._retire_task)

    def _cleanup_task(self, operation: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(operation, name="pi-mcp-cleanup")
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        return task

    async def _discard_serialized(self, stage: McpStage) -> None:
        async with self._stage_lock:
            await self._discard(stage)

    async def _discard(self, stage: McpStage) -> None:
        await self._close_connections(stage._pending)
        if stage in self._stages:
            self._stages.remove(stage)

    async def _retire_serialized(self, stage: McpStage) -> None:
        async with self._stage_lock:
            await self._close_connections(stage._previous)
            if stage in self._stages:
                self._stages.remove(stage)

    async def _close_connections(self, pending: list[McpConnection]) -> None:
        for connection in pending:
            connection._stop.set()
        errors: list[Exception] = []
        for connection in reversed(tuple(pending)):
            try:
                await connection.aclose()
            except (Exception, asyncio.CancelledError):
                errors.append(McpError("MCP connection cleanup failed; close may be retried"))
            else:
                pending.remove(connection)
                if connection in self._owned_connections:
                    self._owned_connections.remove(connection)
        if errors:
            raise ExceptionGroup("MCP cleanup failed", errors)

    async def aclose(self) -> None:
        reject_cleanup_reentry(self, "MCP runtime close")
        self._closed = True
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close())
        await settle_cleanup(self._close_task)

    async def _close(self) -> None:
        # Signal active, retired and candidate connections, including owners that
        # are still spawning. Cancel/join setup before taking the serialization
        # lock so candidate cleanup can finish without a close/setup deadlock.
        for connection in self._owned_connections:
            connection._stop.set()
        tasks: list[asyncio.Task[Any]] = list(self._setup_tasks)
        if self._prepare_task is not None:
            tasks.append(self._prepare_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        while self._cleanup_tasks:
            await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
        async with self._stage_lock:
            await self._close_connections(self._owned_connections)
