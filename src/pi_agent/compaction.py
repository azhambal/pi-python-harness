"""Summarize projected history into validated, durable checkpoints."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from .cancellation import Cancellation, OperationAbortedError, iterate_with_cancellation
from .config import CompactionSettings, RetrySettings
from .context import estimate_tokens, output_token_limit
from .error_policy import is_retryable_error
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
    read_only = sorted(path for path in ops.read if path not in modified)
    return read_only, sorted(modified)


def format_file_operations(read_files: Sequence[str], modified_files: Sequence[str]) -> str:
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    return "\n\n" + "\n\n".join(sections) if sections else ""


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[... {len(text) - max_chars} more characters truncated]"


def serialize_conversation(messages: Sequence[Message]) -> str:
    """Render messages as tagged text so the summarizer cannot continue them."""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            content = content_text(message.content)
            if content:
                parts.append(f"[User]: {content}")
        elif isinstance(message, AssistantMessage):
            calls = []
            for block in message.content:
                if isinstance(block, ToolCallBlock):
                    args = ", ".join(f"{key}={json.dumps(value)}" for key, value in block.arguments.items())
                    calls.append(f"{block.name}({args})")
            if any(isinstance(block, TextBlock) for block in message.content):
                parts.append(f"[Assistant]: {content_text(message.content)}")
            if calls:
                parts.append("[Assistant tool calls]: " + "; ".join(calls))
        elif isinstance(message, ToolResultMessage):
            content = content_text(message.content)
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


@dataclass
class CutPointResult:
    first_kept_entry_index: int
    is_split_turn: bool


def find_cut_point(
    entries: Sequence[ProjectedEntry],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """Walk backwards accumulating sizes until ``keep_recent_tokens``.

    Cut candidates are entries containing a user or assistant message; a cut
    never lands on a tool result, so a kept assistant tool-call message keeps
    its results with it.
    """
    cut_points = [
        index
        for index in range(start_index, end_index)
        if any(_is_cut_point_message(message) for message in entries[index].messages)
    ]
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, is_split_turn=False)

    accumulated = 0
    cut_index = cut_points[0]
    for index in range(end_index - 1, start_index - 1, -1):
        entry_tokens = sum(estimate_tokens(message) for message in entries[index].messages)
        if entry_tokens == 0:
            continue
        accumulated += entry_tokens
        if accumulated >= keep_recent_tokens:
            candidates = [candidate for candidate in cut_points if candidate >= index]
            cut_index = candidates[0] if candidates else cut_points[-1]
            break

    starts_turn = any(_is_turn_start_message(message) for message in entries[cut_index].messages)
    return CutPointResult(first_kept_entry_index=cut_index, is_split_turn=not starts_turn)


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


def prepare_compaction(
    projected: Sequence[ProjectedEntry],
    *,
    tokens_before: int,
    settings: CompactionSettings,
) -> CompactionPreparation | None:
    """Pick the cut point and collect what will be summarized.

    Returns None when the newest journal entry already is a compaction, or
    when there is nothing older than the kept window to summarize.
    """
    if not projected:
        return None

    previous_summary: str | None = None
    file_ops = FileOperations()
    boundary_start = 0
    for index, entry in enumerate(projected):
        if isinstance(entry, _CompactionMarker) and entry.summary:
            previous_summary = entry.summary
            file_ops.read.update(entry.read_files)
            file_ops.written.update(entry.modified_files)
            # Legacy checkpoints stored file sets only in summary tags.
            if not entry.file_lists_recorded:
                for tag, destination in (("read-files", file_ops.read), ("modified-files", file_ops.written)):
                    for match in re.finditer(rf"<{tag}>\n(.*?)\n</{tag}>", entry.summary, re.DOTALL):
                        destination.update(line for line in match.group(1).splitlines() if line)
            boundary_start = index + 1
            break

    if boundary_start >= len(projected):
        return None

    cut = find_cut_point(projected, boundary_start, len(projected), settings.keep_recent_tokens)
    history_end = cut.first_kept_entry_index
    turn_start = next(
        (
            index
            for index in range(history_end, boundary_start - 1, -1)
            if any(isinstance(message, UserMessage) for message in projected[index].messages)
        ),
        boundary_start,
    )
    # Keep simple completed exchanges whole; split a long tool turn at an assistant boundary.
    tool_turn = any(
        isinstance(message, AssistantMessage)
        and any(isinstance(block, ToolCallBlock) for block in message.content)
        for entry in projected[turn_start : history_end + 1]
        for message in entry.messages
    )
    split = cut.is_split_turn and history_end > turn_start and tool_turn
    if cut.is_split_turn and not split:
        history_end = turn_start
        cut.first_kept_entry_index = turn_start
    prefix: list[Message] = (
        [
            message
            for entry in projected[turn_start:history_end]
            for message in entry.messages
            if not isinstance(message, SystemMessage)
        ]
        if split
        else []
    )
    if split:
        history_end = turn_start
    messages_to_summarize: list[Message] = []
    for entry in projected[boundary_start:history_end]:
        messages_to_summarize.extend(
            message for message in entry.messages if not isinstance(message, SystemMessage)
        )
    if not messages_to_summarize and not prefix:
        return None

    for message in [*messages_to_summarize, *prefix]:
        extract_file_ops_from_message(message, file_ops)

    return CompactionPreparation(
        first_kept_entry_id=projected[cut.first_kept_entry_index].entry_id,
        messages_to_summarize=messages_to_summarize,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
        turn_prefix_messages=prefix,
        is_split_turn=split,
    )


# ---------------------------------------------------------------------------
# Summarization through the injected provider
# ---------------------------------------------------------------------------


class CompactionAbortedError(Exception):
    """Compaction observed its cancellation token."""


def get_summarization_failure(response: AssistantMessage, label: str) -> str | None:
    """Only a fully completed stop response may become a checkpoint."""
    if response.stop_reason != "stop":
        return f"{label} failed: response did not complete (stop_reason {response.stop_reason!r})"
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


async def _complete_summarization(
    provider: Provider,
    model: ModelInfo,
    prompt_text: str,
    *,
    session_id: str | None,
    cancellation: Cancellation,
) -> AssistantMessage:
    """Run one request and return its validated terminal response."""
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
    level: ThinkingLevel = "off"
    if model.thinking_level_map is not None:
        level = next(
            candidate
            for candidate in ("off", "minimal", "low", "medium", "high", "xhigh", "max")
            if model.thinking_level_map.get(candidate) is not None
        )
    options = ProviderOptions(
        cancellation=cancellation,
        thinking_level=level,
        max_tokens=output_token_limit(request.messages, model),
        session_id=session_id,
    )
    final: AssistantMessage | None = None
    stream = provider.stream(request, options)
    stream_events = iterate_with_cancellation(stream, cancellation)
    try:
        async for event in stream_events:
            if isinstance(event, StreamDone):
                if event.reason != "stop" or event.message.stop_reason != "stop":
                    raise RuntimeError(
                        "Summarization failed: terminal event and message must both complete "
                        f"with stop (reason={event.reason!r}, stop_reason={event.message.stop_reason!r})"
                    )
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


async def generate_summary(
    messages_to_summarize: Sequence[Message],
    *,
    provider: Provider,
    model: ModelInfo,
    cancellation: Cancellation,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    session_id: str | None = None,
    retry: RetrySettings | None = None,
) -> tuple[str, Usage]:
    """Generate (or update) the checkpoint summary; raises on unusable output."""
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    conversation_text = serialize_conversation(messages_to_summarize)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    policy = retry or RetrySettings(enabled=False)
    policy.validate()
    attempt = 0
    while True:
        response = await _complete_summarization(
            provider, model, prompt_text, session_id=session_id, cancellation=cancellation
        )
        if not (
            policy.enabled
            and attempt < policy.max_retries
            and is_retryable_error(response, model.context_window)
        ):
            break
        attempt += 1
        try:
            await asyncio.wait_for(cancellation.wait(), policy.delay_ms(attempt) / 1000)
        except TimeoutError:
            pass
        if cancellation.aborted:
            raise CompactionAbortedError()

    failure = get_summarization_failure(response, "Summarization")
    if failure:
        raise RuntimeError(f"{failure}: {response.error_message or 'unusable response'}")
    if any(isinstance(block, ToolCallBlock) for block in response.content):
        raise RuntimeError("Summarization attempted to call a tool")
    text = content_text(response.content).strip()
    if not text:
        raise RuntimeError("Summarization produced an empty summary")
    return text, response.usage


async def run_compaction(
    preparation: CompactionPreparation,
    *,
    provider: Provider,
    model: ModelInfo,
    cancellation: Cancellation,
    session_id: str | None = None,
    custom_instructions: str | None = None,
    retry: RetrySettings | None = None,
) -> CompactionResult:
    """Produce a compaction result. Persisting the checkpoint is the caller's job."""
    if cancellation.aborted:
        raise CompactionAbortedError()
    summary = preparation.previous_summary or "No prior history."
    usage = Usage()
    if preparation.messages_to_summarize:
        summary, usage = await generate_summary(
            preparation.messages_to_summarize,
            provider=provider,
            model=model,
            cancellation=cancellation,
            custom_instructions=custom_instructions,
            previous_summary=preparation.previous_summary,
            session_id=session_id,
            retry=retry,
        )
    if preparation.is_split_turn:
        prefix_summary, prefix_usage = await generate_summary(
            preparation.turn_prefix_messages,
            provider=provider,
            model=model,
            cancellation=cancellation,
            custom_instructions="Summarize the initial request, completed work and context needed to understand "
            "the retained continuation of this single turn. " + (custom_instructions or ""),
            session_id=session_id,
            retry=retry,
        )
        summary += "\n\n---\n\n**Turn Context (split turn):**\n\n" + prefix_summary
        for key in ("input", "output", "cache_read", "cache_write", "total_tokens"):
            setattr(usage, key, getattr(usage, key) + getattr(prefix_usage, key))
        for key in ("cache_write_1h", "reasoning"):
            if getattr(prefix_usage, key) is not None:
                setattr(usage, key, (getattr(usage, key) or 0) + getattr(prefix_usage, key))
        for key in ("input", "output", "cache_read", "cache_write", "total"):
            setattr(usage.cost, key, getattr(usage.cost, key) + getattr(prefix_usage.cost, key))
    if cancellation.aborted:
        raise CompactionAbortedError()
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)
    if not preparation.first_kept_entry_id:
        raise RuntimeError("First kept entry has no id")
    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        usage=usage,
        read_files=read_files,
        modified_files=modified_files,
        is_split_turn=preparation.is_split_turn,
    )
