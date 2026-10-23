"""Journal projection; canonical rules live in docs/invariants.md."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from .compaction import (
    ProjectedEntry,
    _CompactionMarker,
    compaction_projected_entry,
)
from .context import estimate_context_tokens, projection_system_tokens
from .session_store import (
    SessionCompactionEntry,
    SessionMessageEntry,
    SessionOmissionEntry,
    SessionStore,
)
from .types import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCallBlock,
    ToolResultMessage,
)


def build_projection(store: SessionStore) -> list[ProjectedEntry]:
    """Independent model-context snapshot, cached until the next committed append."""
    return deepcopy(_projection_entries(store))


def _projection_entries(store: SessionStore) -> list[ProjectedEntry]:
    """Private read-only view; never pass canonical messages to callbacks/providers."""
    if store._projection_cache is None:
        store._projection_cache = _build_projection(store)
    return store._projection_cache


def _build_projection(store: SessionStore) -> list[ProjectedEntry]:
    """Model-visible projection of the raw journal.

    Contracts and failure semantics: docs/invariants.md.
    """
    entries = store._entries
    omitted_ids = store._omitted_ids
    compaction_index = store._latest_compaction_source
    projected: list[ProjectedEntry] = []

    def project(entry: SessionMessageEntry, index: int) -> ProjectedEntry:
        if entry.id in omitted_ids:
            return ProjectedEntry(entry_id=entry.id, source_index=index, omitted_role=entry.message.role)
        return ProjectedEntry(entry_id=entry.id, source_index=index, messages=[entry.message])

    if compaction_index is None:
        for index, entry in enumerate(entries):
            if isinstance(entry, SessionMessageEntry):
                projected.append(project(entry, index))
    else:
        latest_compaction = cast(SessionCompactionEntry, entries[compaction_index])
        projected.append(compaction_projected_entry(latest_compaction, compaction_index))
        first_kept = store._entry_positions[latest_compaction.first_kept_entry_id]
        for index in range(first_kept, compaction_index):
            entry = entries[index]
            if isinstance(entry, SessionMessageEntry) and not isinstance(entry.message, SystemMessage):
                projected.append(project(entry, index))
        for index, entry in enumerate(entries[compaction_index + 1 :], start=compaction_index + 1):
            if isinstance(entry, SessionMessageEntry):
                projected.append(project(entry, index))

    return _relocate_interruption_results(projected)


def estimate_projected_context_tokens(store: SessionStore) -> int:
    """Pi's ``estimateProjectedContextTokens``.

    Provider usage counts only when recorded after the latest omission or
    compaction; otherwise the chars/4 estimate of the projection applies.
    """
    entries = _projection_entries(store)
    messages = [message for entry in entries for message in entry.messages]
    estimate = estimate_context_tokens(messages)
    if estimate.last_usage_index is not None:
        index = 0
        usage_source = -1
        for entry in entries:
            if estimate.last_usage_index < index + len(entry.messages):
                usage_source = entry.source_index
                break
            index += len(entry.messages)
        journal = store._entries
        latest_invalidating = next(
            (
                position
                for position in range(len(journal) - 1, -1, -1)
                if isinstance(journal[position], (SessionOmissionEntry, SessionCompactionEntry))
            ),
            -1,
        )
        if usage_source > latest_invalidating:
            return estimate.tokens
    return projection_system_tokens(messages)


def _raw_call_ids(store: SessionStore) -> set[str]:
    """Every tool-call ID used across RAW session history.

    Including omitted attempts and pre-compaction entries. Used only for
    core validation (the loop reassigns reused IDs before anything commits
    or executes); never sent to the model.
    """
    return store.call_ids


def _relocate_interruption_results(projected: list[ProjectedEntry]) -> list[ProjectedEntry]:
    """Build one contiguous, correctly ordered result batch per assistant.

    Contracts and failure semantics: docs/invariants.md.
    """
    real_results: dict[str, ToolResultMessage] = {}
    interruptions: dict[str, ToolResultMessage] = {}
    # Entry positions (projection order) of projected assistants claiming
    # each call ID, and of each real result. A legitimate real result always
    # follows its own assistant; a result that no *earlier* projected
    # assistant claims belongs to an omitted earlier attempt — an orphan.
    claim_positions: dict[str, list[int]] = {}
    result_positions: dict[str, int] = {}
    for position, entry in enumerate(projected):
        for message in entry.messages:
            if isinstance(message, ToolResultMessage):
                if message.interrupted:
                    interruptions.setdefault(message.tool_call_id, message)
                else:
                    real_results.setdefault(message.tool_call_id, message)
                    result_positions.setdefault(message.tool_call_id, position)
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolCallBlock):
                        claim_positions.setdefault(block.id, []).append(position)
    orphan_real = {
        call_id
        for call_id, position in result_positions.items()
        if not any(claim < position for claim in claim_positions.get(call_id, []))
    }
    if not interruptions and not orphan_real:
        return projected
    for call_id in orphan_real:
        # Orphaned results cannot be claimed by any rebuilt batch; under
        # unique IDs this drops exactly the orphan, never a legitimate result.
        real_results.pop(call_id, None)

    # An assistant needs a rebuilt batch when any of its calls has an
    # interruption record (pull its real results into the same batch) —
    # plain fully-real batches already sit adjacent and ordered.
    needs_batch: set[str] = set()  # entry ids of assistants to rebuild
    for entry in projected:
        for message in entry.messages:
            if isinstance(message, AssistantMessage):
                if any(
                    isinstance(block, ToolCallBlock) and block.id in interruptions
                    for block in message.content
                ):
                    needs_batch.add(entry.entry_id)
                    break

    result: list[ProjectedEntry] = []
    for entry in projected:
        if entry.entry_id in needs_batch:
            messages: list[Message] = []
            for message in entry.messages:
                messages.append(message)
                if isinstance(message, AssistantMessage):
                    # One contiguous batch in the assistant's call order.
                    for block in message.content:
                        if not isinstance(block, ToolCallBlock):
                            continue
                        if block.id in real_results:
                            messages.append(real_results.pop(block.id))
                        elif block.id in interruptions:
                            messages.append(interruptions.pop(block.id))
            if isinstance(entry, _CompactionMarker):
                result.append(
                    _CompactionMarker(
                        entry.entry_id,
                        entry.source_index,
                        messages,
                        entry.summary,
                        entry.read_files,
                        entry.modified_files,
                        entry.file_lists_recorded,
                    )
                )
            else:
                result.append(
                    ProjectedEntry(
                        entry_id=entry.entry_id, source_index=entry.source_index, messages=messages
                    )
                )
            continue
        # Drop consumed result entries (moved into a rebuilt batch) and
        # orphaned ones; keep everything else as the original instance.
        if len(entry.messages) == 1 and isinstance(entry.messages[0], ToolResultMessage):
            single = entry.messages[0]
            if single.interrupted or single.tool_call_id in orphan_real:
                continue
            if single.tool_call_id not in real_results:
                # Consumed by a rebuilt batch above.
                continue
        result.append(entry)
    return result
