"""Summarize projected history into validated, durable checkpoints."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial

from ._js import json_stringify, utf16_length, uuidv7
from .cancellation import Cancellation, OperationAbortedError, iterate_with_cancellation
from .config import CompactionSettings, RetrySettings
from .context import estimate_tokens
from .error_policy import is_retryable_assistant_error
from .events import StreamDone, StreamError
from .provider import Provider, ProviderOptions, ProviderRequest
from .session_store import SessionCompactionEntry
from .types import (
    AssistantMessage,
    Message,
    ModelInfo,
    SystemMessage,
    TextBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    content_text,
    now_ms,
)

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between a user and "
    "an AI assistant, then produce a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output "
    "the structured summary."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """The messages above are earlier context from an ongoing conversation. Later messages are stored separately and do not need to be reconstructed.

Create a concise checkpoint of the user's request and the progress shown above. This checkpoint will be placed before the later messages so the conversation can continue with the necessary context.

## Original Request
[What did the user ask for?]

## Progress So Far
- [Key decisions and work completed in these messages]

## Context Needed to Continue
- [Information from these messages needed to understand the later work]

Only summarize information explicitly present above. Do not infer or recreate later messages."""

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

TOOL_RESULT_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# File operation tracking (compaction/utils.ts)
# ---------------------------------------------------------------------------


@dataclass
class FileOperations:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def extract_file_ops_from_message(message: Message, ops: FileOperations) -> None:
    if not isinstance(message, AssistantMessage):
        return
    for block in message.content:
        if not isinstance(block, ToolCallBlock):
            continue
        path = block.arguments.get("path")
        if not isinstance(path, str):
            continue
        if block.name == "read":
            ops.read.add(path)
        elif block.name == "write":
            ops.written.add(path)
        elif block.name == "edit":
            ops.edited.add(path)


def compute_file_lists(ops: FileOperations) -> tuple[list[str], list[str]]:
    modified = ops.edited | ops.written

    # JS Array.sort compares UTF-16 code units, not Python Unicode code points.
    def key(path: str) -> bytes:
        return path.encode("utf-16-be", errors="surrogatepass")

    read_only = sorted((path for path in ops.read if path not in modified), key=key)
    return read_only, sorted(modified, key=key)


def format_file_operations(read_files: Sequence[str], modified_files: Sequence[str]) -> str:
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    return "\n\n" + "\n\n".join(sections) if sections else ""


def _truncate_for_summary(text: str, max_chars: int) -> str:
    length = utf16_length(text)
    if length <= max_chars:
        return text
    # JS slice may leave a lone surrogate at the boundary. Keep it until the
    # provider sanitizes the prompt, exactly as Pi does.
    prefix = text.encode("utf-16-le", errors="surrogatepass")[: max_chars * 2].decode(
        "utf-16-le", errors="surrogatepass"
    )
    return f"{prefix}\n\n[... {length - max_chars} more characters truncated]"


def serialize_conversation(messages: Sequence[Message]) -> str:
    """Render messages as tagged text so the summarizer cannot continue them."""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            content = content_text(message.content, "")
            if content:
                parts.append(f"[User]: {content}")
        elif isinstance(message, AssistantMessage):
            calls = []
            for block in message.content:
                if isinstance(block, ToolCallBlock):
                    args = ", ".join(
                        f"{key}={json_stringify(value)}" for key, value in block.arguments.items()
                    )
                    calls.append(f"{block.name}({args})")
            if any(isinstance(block, TextBlock) for block in message.content):
                parts.append(f"[Assistant]: {content_text(message.content)}")
            if calls:
                parts.append("[Assistant tool calls]: " + "; ".join(calls))
        elif isinstance(message, ToolResultMessage):
            content = content_text(message.content, "")
            if content:
                parts.append(f"[Tool result]: {_truncate_for_summary(content, TOOL_RESULT_MAX_CHARS)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Projection building blocks
# ---------------------------------------------------------------------------


@dataclass
class ProjectedEntry:
    """One journal entry's contribution to the model-visible projection.

    ``source_index`` is the entry's position in the raw journal; the session
    uses it to invalidate provider usage recorded before a compaction
    checkpoint even when that message is retained after the checkpoint in the
    projection.
    """

    entry_id: str
    source_index: int
    messages: list[Message] = field(default_factory=list)
    #: Role of a message omitted from the model context; Pi projects such entries without messages.
    omitted_role: str | None = None

    @property
    def omitted(self) -> bool:
        return self.omitted_role is not None


class _CompactionMarker(ProjectedEntry):
    """Projected contribution of a compaction entry (system message + summary)."""

    def __init__(
        self,
        entry_id: str,
        source_index: int,
        messages: list[Message],
        summary: str,
        read_files: Sequence[str] = (),
        modified_files: Sequence[str] = (),
        file_lists_recorded: bool = False,
    ) -> None:
        super().__init__(entry_id=entry_id, source_index=source_index, messages=messages)
        self.summary = summary
        self.read_files = list(read_files)
        self.modified_files = list(modified_files)
        self.file_lists_recorded = file_lists_recorded


def compaction_projected_entry(entry: SessionCompactionEntry, source_index: int) -> _CompactionMarker:
    messages: list[Message] = []
    if entry.system_message is not None:
        messages.append(entry.system_message)
    messages.append(
        UserMessage(
            content=[TextBlock(text=COMPACTION_SUMMARY_PREFIX + entry.summary + COMPACTION_SUMMARY_SUFFIX)],
            timestamp=_entry_timestamp(entry),
        )
    )
    return _CompactionMarker(
        entry_id=entry.id,
        source_index=source_index,
        messages=messages,
        summary=entry.summary,
        read_files=entry.read_files,
        modified_files=entry.modified_files,
        file_lists_recorded=entry.file_lists_recorded,
    )


def _entry_timestamp(entry: SessionCompactionEntry) -> int:
    try:
        parsed = datetime.fromisoformat(entry.timestamp)
        return int(parsed.timestamp() * 1000)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Cut points
# ---------------------------------------------------------------------------


def _is_cut_point_message(message: Message) -> bool:
    return isinstance(message, (UserMessage, AssistantMessage))


def _is_turn_start_message(message: Message) -> bool:
    return isinstance(message, UserMessage)


def _is_turn_start_entry(entry: ProjectedEntry) -> bool:
    return not isinstance(entry, _CompactionMarker) and any(
        _is_turn_start_message(message) for message in entry.messages
    )


def find_turn_start_index(entries: Sequence[ProjectedEntry], entry_index: int, start_index: int) -> int:
    """The user message starting the turn that contains ``entry_index``, or -1."""
    for index in range(entry_index, start_index - 1, -1):
        if _is_turn_start_entry(entries[index]):
            return index
    return -1


@dataclass
class CutPointResult:
    first_kept_entry_index: int
    #: User message starting the turn being split, or -1 when not splitting.
    turn_start_index: int
    is_split_turn: bool


def find_cut_point(
    entries: Sequence[ProjectedEntry],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """Pi's ``findProjectedCutPoint``: walk back until ``keep_recent_tokens``.

    Cut candidates are entries containing a user or assistant message; a cut
    never lands on a tool result, so a kept assistant tool-call message keeps
    its results with it. Cutting at an assistant message splits its turn.
    """
    cut_points = [
        index
        for index in range(start_index, end_index)
        if not isinstance(entries[index], _CompactionMarker)
        and any(_is_cut_point_message(message) for message in entries[index].messages)
    ]
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)

    accumulated = 0
    exceeded_budget = False
    cut_index = cut_points[0]
    for index in range(end_index - 1, start_index - 1, -1):
        entry_tokens = sum(estimate_tokens(message) for message in entries[index].messages)
        if entry_tokens == 0:
            continue
        accumulated += entry_tokens
        if accumulated >= keep_recent_tokens:
            exceeded_budget = True
            # Trailing tool results over budget keep their assistant tool call.
            cut_index = next((candidate for candidate in cut_points if candidate >= index), cut_points[-1])
            break

    # A recovery attempt omitted after the last visible input: advance only for a
    # closed suffix of omitted entries that contains an omitted assistant attempt.
    suffix = entries[cut_index + 1 : end_index]
    if (
        exceeded_budget
        and any(entry.omitted_role == "assistant" for entry in suffix)
        and all(entry.omitted for entry in suffix)
    ):
        cut_index += 1

    # Include adjacent entries that do not affect context.
    while cut_index > start_index:
        previous = entries[cut_index - 1]
        if isinstance(previous, _CompactionMarker) or previous.messages:
            break
        cut_index -= 1

    starts_turn = _is_turn_start_entry(entries[cut_index])
    turn_start = -1 if starts_turn else find_turn_start_index(entries, cut_index, start_index)
    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=not starts_turn and turn_start != -1,
    )


@dataclass
class CompactionPreparation:
    first_kept_entry_id: str
    messages_to_summarize: list[Message]
    tokens_before: int
    previous_summary: str | None
    file_ops: FileOperations = field(default_factory=FileOperations)
    settings: CompactionSettings = field(default_factory=CompactionSettings)
    turn_prefix_messages: list[Message] = field(default_factory=list)
    is_split_turn: bool = False


@dataclass
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    estimated_tokens_after: int | None = None
    usage: Usage | None = None
    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    is_split_turn: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "first_kept_entry_id": self.first_kept_entry_id,
            "tokens_before": self.tokens_before,
            "estimated_tokens_after": self.estimated_tokens_after,
            "usage": self.usage.to_json() if self.usage is not None else None,
            "read_files": list(self.read_files),
            "modified_files": list(self.modified_files),
            "is_split_turn": self.is_split_turn,
        }


def _messages_for_compaction(entry: ProjectedEntry) -> list[Message]:
    if isinstance(entry, _CompactionMarker):
        return []
    # System messages are prompt state, not conversation; the checkpoint carries their replay.
    return [message for message in entry.messages if not isinstance(message, SystemMessage)]


def prepare_compaction(
    projected: Sequence[ProjectedEntry],
    *,
    tokens_before: int,
    settings: CompactionSettings,
) -> CompactionPreparation | None:
    """Pi's ``prepareCompaction``: pick the cut point and collect what will be summarized.

    Returns None when there is nothing older than the kept window to
    summarize. The caller rejects a journal whose newest entry already is a
    compaction.
    """
    previous_summary: str | None = None
    file_ops = FileOperations()
    boundary_start = 0
    # The newest checkpoint is projected first.
    previous = next(
        (
            index
            for index, entry in enumerate(projected)
            if isinstance(entry, _CompactionMarker) and entry.messages
        ),
        -1,
    )
    if previous >= 0:
        marker = projected[previous]
        assert isinstance(marker, _CompactionMarker)
        previous_summary = marker.summary
        file_ops.read.update(marker.read_files)
        file_ops.edited.update(marker.modified_files)
        # Legacy checkpoints stored file sets only in summary tags.
        if not marker.file_lists_recorded:
            for tag, destination in (("read-files", file_ops.read), ("modified-files", file_ops.edited)):
                for match in re.finditer(rf"<{tag}>\n(.*?)\n</{tag}>", marker.summary, re.DOTALL):
                    destination.update(line for line in match.group(1).splitlines() if line)
        boundary_start = previous + 1

    cut = find_cut_point(projected, boundary_start, len(projected), settings.keep_recent_tokens)
    if cut.first_kept_entry_index >= len(projected):
        return None
    first_kept_entry_id = projected[cut.first_kept_entry_index].entry_id
    if not first_kept_entry_id:
        return None
    history_end = cut.turn_start_index if cut.is_split_turn else cut.first_kept_entry_index

    messages_to_summarize = [
        message
        for entry in projected[boundary_start:history_end]
        for message in _messages_for_compaction(entry)
    ]
    turn_prefix_messages = (
        [
            message
            for entry in projected[cut.turn_start_index : cut.first_kept_entry_index]
            for message in _messages_for_compaction(entry)
        ]
        if cut.is_split_turn
        else []
    )
    if not messages_to_summarize and not turn_prefix_messages:
        return None

    for message in [*messages_to_summarize, *turn_prefix_messages]:
        extract_file_ops_from_message(message, file_ops)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=messages_to_summarize,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut.is_split_turn,
    )


# ---------------------------------------------------------------------------
# Summarization through the injected provider
# ---------------------------------------------------------------------------


class CompactionAbortedError(Exception):
    """Compaction observed its cancellation token."""


def get_summarization_failure(response: AssistantMessage, label: str) -> str | None:
    """A response that cannot safely be persisted; a length stop holds partial text."""
    if response.stop_reason == "error":
        return f"{label} failed: {response.error_message or 'Unknown error'}"
    if response.stop_reason == "length":
        return f"{label} failed: generation hit the token cap and the summary is incomplete"
    return None


async def _close_provider_stream(stream: object) -> None:
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        result = close()
        if isinstance(result, Awaitable):
            await result
    except Exception:
        return


async def _stream_summarization(
    provider: Provider,
    request: ProviderRequest,
    options: ProviderOptions,
    cancellation: Cancellation,
) -> AssistantMessage:
    """Run one request and return its terminal response."""
    model = request.model
    final: AssistantMessage | None = None
    stream = provider.stream(request, options)
    stream_events = iterate_with_cancellation(stream, cancellation)
    try:
        async for event in stream_events:
            if isinstance(event, StreamDone):
                if event.reason != event.message.stop_reason:
                    raise RuntimeError("Summarization failed: inconsistent provider done event")
                final = event.message
                break
            if isinstance(event, StreamError):
                if event.error.stop_reason != event.reason:
                    raise RuntimeError("Summarization failed: inconsistent provider error event")
                final = event.error
                break
            if cancellation.aborted:
                raise CompactionAbortedError()
    except OperationAbortedError as error:
        raise CompactionAbortedError() from error
    finally:
        try:
            await stream_events.aclose()
        finally:
            await _close_provider_stream(stream)
    if cancellation.aborted:
        raise CompactionAbortedError()
    if final is None:
        return AssistantMessage(
            provider=model.provider,
            api=model.api,
            model=model.id,
            stop_reason="error",
            error_message="Provider stream ended without a terminal event",
        )
    return final


async def complete_summarization(
    provider: Provider,
    model: ModelInfo,
    prompt_text: str,
    *,
    cancellation: Cancellation,
    thinking_level: ThinkingLevel = "off",
    session_id: str | None = None,
    retry: RetrySettings | None = None,
) -> AssistantMessage:
    """Pi's ``completeSummarization``: one uncached request under the retry policy.

    Reuses a caller-supplied routing ``session_id``; otherwise each request
    gets a fresh one. The model's own output limit applies (no max_tokens).
    """
    if cancellation.aborted:
        raise CompactionAbortedError()
    request = ProviderRequest(
        model=model,
        messages=[
            SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT, timestamp=0),
            UserMessage(content=[TextBlock(text=prompt_text)], timestamp=now_ms()),
        ],
        tools=[],
    )
    options = ProviderOptions(
        cancellation=cancellation,
        # Pi's createSummarizationOptions: reasoning only for reasoning models above "off".
        thinking_level=thinking_level if model.reasoning and thinking_level != "off" else "off",
        session_id=session_id if session_id is not None else uuidv7(),
        cache_retention="none",
    )
    policy = retry or RetrySettings(enabled=False)
    policy.validate()
    max_attempts = policy.max_retries if policy.enabled else 0
    attempt = 0
    while True:
        response = await _stream_summarization(provider, request, options, cancellation)
        if response.stop_reason != "error":
            return response
        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            return response
        attempt += 1
        try:
            await asyncio.wait_for(cancellation.wait(), policy.delay_ms(attempt) / 1000)
        except TimeoutError:
            pass
        if cancellation.aborted:
            raise CompactionAbortedError()


def _summary_text(response: AssistantMessage, label: str) -> str:
    # Pi only returns "aborted" for an aborted signal, which then cancels the compaction.
    if response.stop_reason == "aborted":
        raise CompactionAbortedError()
    failure = get_summarization_failure(response, label)
    if failure:
        raise RuntimeError(failure)
    if any(isinstance(block, ToolCallBlock) for block in response.content):
        raise RuntimeError(f"{label} attempted to call a tool")
    return content_text(response.content)


async def generate_summary(
    messages_to_summarize: Sequence[Message],
    *,
    provider: Provider,
    model: ModelInfo,
    cancellation: Cancellation,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel = "off",
    session_id: str | None = None,
    retry: RetrySettings | None = None,
) -> tuple[str, Usage]:
    """Pi's ``generateSummaryWithUsage``: generate or update the checkpoint summary."""
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    conversation_text = serialize_conversation(messages_to_summarize)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    response = await complete_summarization(
        provider,
        model,
        prompt_text,
        cancellation=cancellation,
        thinking_level=thinking_level,
        session_id=session_id,
        retry=retry,
    )
    return _summary_text(response, "Summarization"), response.usage


async def generate_turn_prefix_summary(
    messages: Sequence[Message],
    *,
    provider: Provider,
    model: ModelInfo,
    cancellation: Cancellation,
    custom_instructions: str | None = None,
    thinking_level: ThinkingLevel = "off",
    session_id: str | None = None,
    retry: RetrySettings | None = None,
) -> tuple[str, Usage]:
    """Pi's ``generateTurnPrefixSummary``: the early part of a split turn."""
    conversation_text = serialize_conversation(messages)
    prompt_text = f"# Conversation\n{conversation_text}\n\n# Instructions\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    if custom_instructions:
        prompt_text = f"{prompt_text}\n\nAdditional focus: {custom_instructions}"
    response = await complete_summarization(
        provider,
        model,
        prompt_text,
        cancellation=cancellation,
        thinking_level=thinking_level,
        session_id=session_id,
        retry=retry,
    )
    return _summary_text(response, "Turn prefix summarization"), response.usage


def combine_usage(first: Usage, second: Usage) -> Usage:
    """Pi's ``combineUsage``."""
    combined = Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        total_tokens=first.total_tokens + second.total_tokens,
    )
    if first.cache_write_1h is not None or second.cache_write_1h is not None:
        combined.cache_write_1h = (first.cache_write_1h or 0) + (second.cache_write_1h or 0)
    if first.reasoning is not None or second.reasoning is not None:
        combined.reasoning = (first.reasoning or 0) + (second.reasoning or 0)
    for key in ("input", "output", "cache_read", "cache_write", "total"):
        setattr(combined.cost, key, getattr(first.cost, key) + getattr(second.cost, key))
    return combined


async def run_compaction(
    preparation: CompactionPreparation,
    *,
    provider: Provider,
    model: ModelInfo,
    cancellation: Cancellation,
    thinking_level: ThinkingLevel = "off",
    session_id: str | None = None,
    custom_instructions: str | None = None,
    retry: RetrySettings | None = None,
) -> CompactionResult:
    """Pi's ``compact``. Persisting the checkpoint is the caller's job."""
    if cancellation.aborted:
        raise CompactionAbortedError()
    summarize = partial(
        generate_summary,
        provider=provider,
        model=model,
        cancellation=cancellation,
        custom_instructions=custom_instructions,
        previous_summary=preparation.previous_summary,
        thinking_level=thinking_level,
        session_id=session_id,
        retry=retry,
    )
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = preparation.previous_summary or "No prior history."
        history_usage: Usage | None = None
        if preparation.messages_to_summarize:
            history_text, history_usage = await summarize(preparation.messages_to_summarize)
        prefix_text, prefix_usage = await generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            provider=provider,
            model=model,
            cancellation=cancellation,
            custom_instructions=custom_instructions,
            thinking_level=thinking_level,
            session_id=session_id,
            retry=retry,
        )
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}"
        usage = combine_usage(history_usage, prefix_usage) if history_usage is not None else prefix_usage
    else:
        summary, usage = await summarize(preparation.messages_to_summarize)
    if cancellation.aborted:
        raise CompactionAbortedError()
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)
    if not preparation.first_kept_entry_id:
        raise RuntimeError("First kept entry has no UUID - session may need migration")
    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        usage=usage,
        read_files=read_files,
        modified_files=modified_files,
        is_split_turn=preparation.is_split_turn,
    )
