"""Scripted responses for offline integration tests and harness development."""

from collections.abc import AsyncIterator, Iterable
from copy import deepcopy

from pi_agent.events import AssistantStreamEvent, StreamDone, StreamError, StreamStart, TextDelta
from pi_agent.provider import ProviderOptions, ProviderRequest
from pi_agent.types import AssistantMessage, TextBlock, now_ms


class FauxProvider:
    """Consume one scripted assistant per request; record independent request snapshots.

    Exhausting the script produces a terminal error, never a fabricated answer.
    Messages and events are copied so core mutations cannot alter the script.
    """

    def __init__(self, responses: Iterable[AssistantMessage]) -> None:
        self._responses = iter(deepcopy(list(responses)))
        self.requests: list[ProviderRequest] = []

    async def stream(
        self, request: ProviderRequest, options: ProviderOptions
    ) -> AsyncIterator[AssistantStreamEvent]:
        self.requests.append(deepcopy(request))
        final = next(self._responses, None)
        if final is None:
            final = AssistantMessage(
                content=[TextBlock(text="Faux provider script exhausted")],
                stop_reason="error",
                error_message="Faux provider script exhausted",
            )
        final.model = request.model.id
        final.provider = request.model.provider
        final.api = request.model.api
        if not final.timestamp:
            # Like a real response, stamped when the request starts.
            final.timestamp = now_ms()
        partial = deepcopy(final)
        partial.content = []
        partial.stop_reason = "pending"
        yield StreamStart(partial=deepcopy(partial))
        for index, block in enumerate(final.content):
            partial.content.append(deepcopy(block))
            if isinstance(block, TextBlock):
                yield TextDelta(content_index=index, delta=block.text, partial=deepcopy(partial))
        if options.cancellation is not None and options.cancellation.aborted:
            final.stop_reason = "aborted"
            final.error_message = "Operation aborted"
        if final.stop_reason in ("error", "aborted"):
            yield StreamError(reason=final.stop_reason, error=final)
        elif final.stop_reason in ("stop", "length", "toolUse"):
            yield StreamDone(reason=final.stop_reason, message=final)
        else:
            final.stop_reason = "error"
            final.error_message = "Faux response must have a terminal stop reason"
            yield StreamError(reason="error", error=final)
