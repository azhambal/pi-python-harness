"""Shared HTTP lifetime and SSE framing; no retries below the session layer."""

import asyncio
import codecs
import json
import math
import re
from collections.abc import AsyncGenerator, AsyncIterator
from copy import deepcopy

import httpx

from pi_agent.cancellation import Cancellation, OperationAbortedError, iterate_with_cancellation
from pi_agent.events import AssistantStreamEvent, StreamError, StreamStart
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.types import AssistantMessage, now_ms

MAX_EVENT_CHARS = 4 * 1024 * 1024


def text_field(data: dict, key: str, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Provider field {key} must be text")
    return value


def strict_json(value: str) -> object:
    def invalid_constant(_: str) -> None:
        raise ValueError("Non-finite JSON number")

    def finite_float(text: str) -> float:
        number = float(text)
        if not math.isfinite(number):
            raise ValueError("Non-finite JSON number")
        return number

    return json.loads(value, parse_constant=invalid_constant, parse_float=finite_float)


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Frame data across TCP chunks, CRLF, comments and multiline events."""
    data: list[str] = []
    size = 0
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    async for chunk in response.aiter_bytes():
        buffer += decoder.decode(chunk)
        while match := re.search(r"\r\n|\r|\n", buffer):
            if match.group() == "\r" and match.end() == len(buffer):
                break  # CRLF may be split across TCP chunks
            line = buffer[: match.start()]
            buffer = buffer[match.end() :]
            size += len(line)
            if size > MAX_EVENT_CHARS:
                raise ValueError("SSE event exceeds 4 MiB")
            if not line:
                if data:
                    yield "\n".join(data)
                data.clear()
                size = 0
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        if size + len(buffer) > MAX_EVENT_CHARS:
            raise ValueError("SSE event exceeds 4 MiB")
    buffer += decoder.decode(b"", final=True)
    if buffer == "\r":
        if data:
            yield "\n".join(data)
        return
    if data or buffer.strip():
        raise ValueError("Incomplete SSE frame at end of stream")


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

    def decode(
        self, response: httpx.Response, message: AssistantMessage
    ) -> AsyncIterator[AssistantStreamEvent]:
        raise NotImplementedError

    async def _events(
        self, request: ProviderRequest, options: ProviderOptions, message: AssistantMessage
    ) -> AsyncGenerator[AssistantStreamEvent, None]:
        payload = self.payload(request, options)
        client = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=15.0), follow_redirects=False
        )
        try:
            async with client.stream("POST", self._url, headers=self.headers(), json=payload) as response:
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
                async for event in self.decode(response, message):
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
