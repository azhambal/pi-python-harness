"""Shared HTTP lifetime and SSE framing; no retries below the session layer."""

import asyncio
import math
from collections.abc import AsyncGenerator, AsyncIterator
from copy import deepcopy

import httpx

from pi_agent.cancellation import Cancellation, OperationAbortedError, iterate_with_cancellation
from pi_agent.events import AssistantStreamEvent, StreamError, StreamStart
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.types import AssistantMessage, now_ms


class HTTPProvider:
    """Client injection is caller-owned; otherwise each request owns its client."""

    endpoint = ""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
    ) -> None:
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        if not api_key.strip():
            raise ValueError("API key is required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("HTTP timeout must be finite and positive")
        self._api_key = api_key
        self._url = str(url).rstrip("/") + self.endpoint
        self._client = client
        self._timeout = timeout

    def payload(self, request: ProviderRequest, options: ProviderOptions) -> dict[str, object]:
        raise NotImplementedError

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "text/event-stream"}

    def prepare(
        self, request: ProviderRequest, options: ProviderOptions, message: AssistantMessage
    ) -> tuple[dict[str, str], dict[str, object]]:
        """Headers and JSON body of one request; may annotate the reply ``message``."""
        return self.headers(), self.payload(request, options)

    def decode(
        self, response: httpx.Response, message: AssistantMessage, request: ProviderRequest
    ) -> AsyncIterator[AssistantStreamEvent]:
        raise NotImplementedError

    async def _events(
        self, request: ProviderRequest, options: ProviderOptions, message: AssistantMessage
    ) -> AsyncGenerator[AssistantStreamEvent, None]:
        headers, payload = self.prepare(request, options, message)
        client = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=15.0), follow_redirects=False
        )
        try:
            async with client.stream("POST", self._url, headers=headers, json=payload) as response:
                if response.status_code >= 300:
                    # Read bounded error data; never put the request/headers in the journal.
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk[: 8192 - len(body)])
                        if len(body) >= 8192:
                            break
                    detail = body.decode("utf-8", errors="replace").replace(self._api_key, "[REDACTED]")
                    raise ValueError(f"HTTP {response.status_code}: {detail}")
                if "text/event-stream" not in response.headers.get("content-type", "").lower():
                    raise ValueError("Expected a text/event-stream response")
                async for event in self.decode(response, message, request):
                    yield event
        finally:
            if self._client is None:
                await client.aclose()

    async def stream(
        self, request: ProviderRequest, options: ProviderOptions
    ) -> AsyncIterator[AssistantStreamEvent]:
        message = AssistantMessage(
            api=request.model.api,
            provider=request.model.provider,
            model=request.model.id,
            thinking_level=options.thinking_level,
            timestamp=now_ms(),
        )
        yield StreamStart(partial=deepcopy(message))
        raw = self._events(request, options, message)
        wrapped = iterate_with_cancellation(raw, options.cancellation or Cancellation())
        try:
            async for event in wrapped:
                yield event
        except OperationAbortedError:
            message.stop_reason = "aborted"
            message.error_message = "Operation aborted"
            yield StreamError(reason="aborted", error=deepcopy(message))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            message.stop_reason = "error"
            if isinstance(error, httpx.TimeoutException):
                detail = "HTTP request timed out"
            elif isinstance(error, httpx.TransportError):
                detail = "HTTP network error: connection failed"
            else:
                detail = str(error) or "Provider protocol error"
            message.error_message = detail.replace(self._api_key, "[REDACTED]")
            yield StreamError(reason="error", error=deepcopy(message))
        finally:
            await wrapped.aclose()
            await raw.aclose()
