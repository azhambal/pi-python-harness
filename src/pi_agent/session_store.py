"""Linear append-only JSONL session store.

Contracts and failure semantics: docs/invariants.md.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import (
    AssistantMessage,
    JsonValue,
    Message,
    SystemMessage,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    message_from_json,
)

SESSION_VERSION = 1

if TYPE_CHECKING:
    from .compaction import ProjectedEntry


def _model_identity(data: object) -> tuple[object, object, object]:
    if not isinstance(data, dict):
        return (None, None, None)
    return data.get("provider"), data.get("api", "unknown"), data.get("id")


class SessionCorruptError(Exception):
    """The session file contains an unrecoverable interior record."""

    def __init__(self, path: Path, line_no: int, reason: str) -> None:
        super().__init__(f"Corrupt session record at {path}:{line_no}: {reason}")
        self.path = path
        self.line_no = line_no
        self.reason = reason


class SessionLockError(Exception):
    """Another writer holds the session file (advisory flock, Linux)."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Session file is locked by another writer: {path}")
        self.path = path


class SessionUnusableError(RuntimeError):
    """A previous durable write failed; the store refuses further appends.

    Raised for every append attempt until the store is closed (and, on disk,
    the file recovered by a later ``open``). Sessions treat this as fatal: it
    must not be synthesized or retried as a model error.
    """


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _reject_constant(name: str) -> float:
    raise ValueError(f"non-finite JSON constant {name!r} is not allowed in session records")


def _dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _parse_line(line: str, path: Path, line_no: int) -> dict[str, Any]:
    try:
        record = json.loads(line, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise SessionCorruptError(path, line_no, f"invalid JSON: {error}") from error
    except ValueError as error:
        raise SessionCorruptError(path, line_no, str(error)) from error
    if not isinstance(record, dict):
        raise SessionCorruptError(path, line_no, "record is not a JSON object")
    if not isinstance(record.get("type"), str):
        raise SessionCorruptError(path, line_no, "record has no 'type' field")
    return record


@dataclass
class SessionHeader:
    type: str = "session"
    version: int = SESSION_VERSION
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_utc_now_iso)
    cwd: str = ""
    model: dict[str, JsonValue] | None = None
    config: dict[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "type": "session",
            "version": self.version,
            "id": self.id,
            "timestamp": self.timestamp,
            "cwd": self.cwd,
            "model": self.model,
            "config": dict(self.config),
        }


@dataclass
class SessionMessageEntry:
    type: str = "message"
    id: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)
    message: Message = field(default_factory=lambda: UserMessage())

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "type": "message",
            "id": self.id,
            "timestamp": self.timestamp,
            "message": self.message.to_json(),
        }


@dataclass
class SessionCompactionEntry:
    type: str = "compaction"
    id: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    usage: Usage | None = None
    system_message: SystemMessage | None = None
    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    file_lists_recorded: bool = field(default=False, compare=False, repr=False)

    def to_json(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "type": "compaction",
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "first_kept_entry_id": self.first_kept_entry_id,
            "tokens_before": self.tokens_before,
        }
        if self.usage is not None:
            data["usage"] = self.usage.to_json()
        if self.system_message is not None:
            data["system_message"] = self.system_message.to_json()
        if self.file_lists_recorded or self.read_files or self.modified_files:
            data["read_files"] = list(self.read_files)
            data["modified_files"] = list(self.modified_files)
        return data


@dataclass
class SessionOmissionEntry:
    """Durable context omission: target stays raw, leaves the projection."""

    type: str = "context_omission"
    id: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)
    target_entry_id: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "type": "context_omission",
            "id": self.id,
            "timestamp": self.timestamp,
            "target_entry_id": self.target_entry_id,
        }


@dataclass
class SessionConfigEntry:
    """Durable configuration snapshot; the latest one wins over the header."""

    type: str = "config_change"
    id: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)
    config: dict[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "type": "config_change",
            "id": self.id,
            "timestamp": self.timestamp,
            "config": dict(self.config),
        }


SessionEntry = SessionMessageEntry | SessionCompactionEntry | SessionOmissionEntry | SessionConfigEntry


@dataclass
class SessionInfo:
    path: Path
    id: str
    cwd: str
    created: str
    modified: float
    message_count: int
    first_message: str


@dataclass
class _LoadedSession:
    header: SessionHeader | None
    entries: list[SessionEntry]
    #: line number of each entry, parallel to ``entries`` (header excluded)
    entry_lines: list[int]
    #: bytes of a trailing fragment after the last newline (b"" when the file
    #: ends with a newline).
    trailing_fragment: bytes


def _header_from_record(record: dict[str, Any], path: Path, line_no: int) -> SessionHeader:
    version = record.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SessionCorruptError(path, line_no, "'version' must be an integer")
    if version != SESSION_VERSION:
        raise SessionCorruptError(
            path,
            line_no,
            f"unsupported session version {version} (expected {SESSION_VERSION}); "
            "this format has no migrations",
        )
    header_id = record.get("id")
    if not isinstance(header_id, str) or not header_id:
        raise SessionCorruptError(path, line_no, "'id' must be a non-empty string")
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise SessionCorruptError(path, line_no, "'timestamp' must be a non-empty string")
    cwd = record.get("cwd")
    if not isinstance(cwd, str):
        raise SessionCorruptError(path, line_no, "'cwd' must be a string")
    model = record.get("model")
    if model is not None and not isinstance(model, dict):
        raise SessionCorruptError(path, line_no, "'model' must be an object or null")
    config = record.get("config")
    if config is not None and not isinstance(config, dict):
        raise SessionCorruptError(path, line_no, "'config' must be an object")
    return SessionHeader(
        version=version,
        id=header_id,
        timestamp=timestamp,
        cwd=cwd,
        model=dict(model) if model is not None else None,
        config=dict(config) if config is not None else {},
    )


def _entry_from_record(
    record: dict[str, Any],
    path: Path,
    line_no: int,
    entries_by_id: dict[str, SessionEntry],
) -> SessionEntry:
    entry_id = record.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise SessionCorruptError(path, line_no, "entry record has no 'id'")
    if entry_id in entries_by_id:
        raise SessionCorruptError(path, line_no, f"duplicate entry id {entry_id!r}")
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise SessionCorruptError(path, line_no, "entry 'timestamp' must be a non-empty string")
    entry_type = record.get("type")

    if entry_type == "message":
        message_data = record.get("message")
        if not isinstance(message_data, dict):
            raise SessionCorruptError(path, line_no, "message entry has no 'message' object")
        try:
            message = message_from_json(message_data)
        except ValueError as error:
            raise SessionCorruptError(path, line_no, f"invalid message: {error}") from error
        return SessionMessageEntry(id=entry_id, timestamp=timestamp, message=message)

    if entry_type == "compaction":
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise SessionCorruptError(path, line_no, "compaction entry needs a non-empty 'summary'")
        first_kept = record.get("first_kept_entry_id")
        if not isinstance(first_kept, str) or not first_kept:
            raise SessionCorruptError(path, line_no, "compaction entry needs a 'first_kept_entry_id'")
        target = entries_by_id.get(first_kept)
        # A self-referencing pointer cannot resolve (its own id is not yet in
        # the map), and the target must be an earlier user/assistant message —
        # never a checkpoint, omission, config record, or tool result.
        if not isinstance(target, SessionMessageEntry) or not isinstance(
            target.message, (UserMessage, AssistantMessage)
        ):
            raise SessionCorruptError(
                path,
                line_no,
                f"compaction first_kept_entry_id {first_kept!r} must reference an earlier "
                "user or assistant message entry",
            )
        tokens_before = record.get("tokens_before")
        if isinstance(tokens_before, bool) or not isinstance(tokens_before, int) or tokens_before < 0:
            raise SessionCorruptError(path, line_no, "'tokens_before' must be a non-negative integer")
        usage_data = record.get("usage")
        usage: Usage | None = None
        if usage_data is not None:
            try:
                usage = Usage.from_json(usage_data)
            except ValueError as error:
                raise SessionCorruptError(path, line_no, f"invalid compaction usage: {error}") from error
        system_data = record.get("system_message")
        system_message: SystemMessage | None = None
        if system_data is not None:
            try:
                system_message = SystemMessage.from_json(system_data)
            except ValueError as error:
                raise SessionCorruptError(
                    path, line_no, f"invalid compaction system message: {error}"
                ) from error
        file_lists = {}
        for key in ("read_files", "modified_files"):
            value = record.get(key, [])
            if not isinstance(value, list) or any(not isinstance(path, str) for path in value):
                raise SessionCorruptError(path, line_no, f"compaction {key} must be a string array")
            file_lists[key] = list(value)
        return SessionCompactionEntry(
            id=entry_id,
            timestamp=timestamp,
            summary=summary,
            first_kept_entry_id=first_kept,
            tokens_before=tokens_before,
            usage=usage,
            system_message=system_message,
            read_files=file_lists["read_files"],
            modified_files=file_lists["modified_files"],
            file_lists_recorded="read_files" in record or "modified_files" in record,
        )

    if entry_type == "context_omission":
        target_id = record.get("target_entry_id")
        if not isinstance(target_id, str) or not target_id:
            raise SessionCorruptError(path, line_no, "omission entry needs a 'target_entry_id'")
        if not isinstance(entries_by_id.get(target_id), SessionMessageEntry):
            raise SessionCorruptError(path, line_no, f"omission target {target_id!r} is not a message entry")
        return SessionOmissionEntry(id=entry_id, timestamp=timestamp, target_entry_id=target_id)

    if entry_type == "config_change":
        config = record.get("config")
        if not isinstance(config, dict):
            raise SessionCorruptError(path, line_no, "config_change entry needs a 'config' object")
        return SessionConfigEntry(id=entry_id, timestamp=timestamp, config=dict(config))

    raise SessionCorruptError(path, line_no, f"unknown entry type {entry_type!r}")


def _decode_committed(committed: bytes, path: Path) -> str:
    try:
        return committed.decode("utf-8")
    except UnicodeDecodeError as error:
        line_no = committed.count(b"\n", 0, error.start) + 1
        raise SessionCorruptError(path, line_no, f"invalid UTF-8: {error}") from error


def load_session_records(path: Path) -> _LoadedSession:
    """Parse a session file strictly. Pure: no lock, no write, no repair.

    Interior corruption raises :class:`SessionCorruptError`; duplicate
    tool-call IDs across raw assistant batches are rejected the same
    way. A trailing fragment after the last newline is reported via
    ``trailing_fragment`` for the caller's recovery policy.
    """
    raw = path.read_bytes()
    last_newline = raw.rfind(b"\n")
    if last_newline == -1:
        committed_bytes = b""
        trailing_fragment = raw
    else:
        committed_bytes = raw[: last_newline + 1]
        trailing_fragment = raw[last_newline + 1 :]

    text = _decode_committed(committed_bytes, path)
    header: SessionHeader | None = None
    entries: list[SessionEntry] = []
    entry_lines: list[int] = []
    entries_by_id: dict[str, SessionEntry] = {}
    for line_no, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        record = _parse_line(line, path, line_no)
        if header is None:
            if record.get("type") != "session":
                raise SessionCorruptError(path, line_no, "first record is not a session header")
            header = _header_from_record(record, path, line_no)
            continue
        entry = _entry_from_record(record, path, line_no, entries_by_id)
        entries.append(entry)
        entry_lines.append(line_no)
        entries_by_id[entry.id] = entry

    if header is None and committed_bytes:
        raise SessionCorruptError(path, 1, "file has records but no session header")
    _reject_duplicate_call_ids(entries, entry_lines, path)
    return _LoadedSession(
        header=header, entries=entries, entry_lines=entry_lines, trailing_fragment=trailing_fragment
    )


def _reject_duplicate_call_ids(entries: list[SessionEntry], entry_lines: list[int], path: Path) -> None:
    """Reject tool-call IDs reused anywhere in raw session history.

    Every finalized tool-call ID is unique across the raw journal —
    including omitted attempts and pre-compaction history. Anything else
    can never be paired unambiguously, and refusing to load is better than
    silently mispairing.
    """
    owners: dict[str, tuple[str, int]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, SessionMessageEntry):
            continue
        message = entry.message
        if not isinstance(message, AssistantMessage):
            continue
        line_no = entry_lines[index] if index < len(entry_lines) else 0
        for block in message.content:
            if not isinstance(block, ToolCallBlock) or not block.id:
                continue
            previous = owners.get(block.id)
            if previous is not None:
                raise SessionCorruptError(
                    path,
                    line_no,
                    f"tool call id {block.id!r} is reused across assistant batches "
                    f"(first used on line {previous[1]}); tool-call ids must be unique "
                    "across raw session history",
                )
            owners[block.id] = (entry.id, line_no)


class SessionStore:
    """Append-only journal for one session; single writer per file."""

    def __init__(
        self,
        cwd: Path | str,
        *,
        persist: bool,
        header: SessionHeader,
        entries: list[SessionEntry],
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.persist = persist
        self._header = deepcopy(header)
        self._entries = deepcopy(entries)
        self._entry_index: dict[str, SessionEntry] = {}
        self._entry_positions: dict[str, int] = {}
        self._omitted_ids: set[str] = set()
        self._latest_compaction_source: int | None = None
        self._call_ids: set[str] = set()
        self._unanswered_calls: dict[str, ToolCallBlock] = {}
        self._answered_ids: set[str] = set()
        self._revision = 0
        self._projection_cache: list[ProjectedEntry] | None = None
        self._last_model_change_source = -1
        self._previous_model_identity = _model_identity(
            self._header.config.get("model") or self._header.model
        )
        self._has_user_or_assistant = False
        self._reindex()
        self._session_file: Path | None = None
        self._lock_fd: int | None = None
        self._file_has_header = False
        self._unusable: BaseException | None = None
        self._closed = False

    # -- construction ---------------------------------------------------

    @staticmethod
    def create(
        cwd: Path | str,
        session_dir: Path | str,
        *,
        model: dict[str, JsonValue] | None = None,
        config: dict[str, JsonValue] | None = None,
        session_id: str | None = None,
    ) -> SessionStore:
        """Create a fresh persisted session (unique file, writer-locked)."""
        header = SessionHeader(
            cwd=str(Path(cwd).resolve()),
            model=model,
            config=dict(config) if config else {},
        )
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("session_id must be a non-empty string")
            header.id = session_id
        directory = Path(session_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_timestamp = header.timestamp.replace(":", "-").replace(".", "-")
        file_path = directory / f"{file_timestamp}_{header.id}.jsonl"
        store = SessionStore(cwd, persist=True, header=header, entries=[])
        try:
            store._open_and_lock(file_path, create=True)
        except BaseException:
            store.close()
            raise
        return store

    @staticmethod
    def open(path: Path | str, *, cwd_override: Path | str | None = None) -> SessionStore:
        """Open (resume) an existing session file: lock, load, then recover.

        The file must exist — resuming a mistyped path is an error, not a
        silent new session. An existing empty file starts a fresh session at
        that exact path.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Session file does not exist: {file_path}")
        store: SessionStore | None = None
        try:
            store = SessionStore(Path.cwd(), persist=True, header=SessionHeader(), entries=[])
            store._open_and_lock(file_path, create=False)
            loaded = load_session_records(file_path)
            store._file_has_header = loaded.header is not None
            if loaded.header is not None:
                store._header = loaded.header
                store._entries = loaded.entries
                store._reindex()
            # Recovery runs before the no-header rejection so an unterminated
            # header line can be adopted.
            adopted_header = store._recover_trailing_fragment(
                loaded.trailing_fragment, allow_header=loaded.header is None
            )
            if loaded.header is None and not adopted_header:
                if file_path.stat().st_size == 0:
                    # Empty file: start a fresh session at this exact path.
                    store._header = SessionHeader(
                        cwd=str((Path(cwd_override) if cwd_override else Path.cwd()).resolve())
                    )
                else:
                    raise SessionCorruptError(file_path, 1, "file has no session header")
            if cwd_override is not None:
                # Explicit runtime working-directory override: the header on
                # disk stays untouched (it is the session's recorded origin);
                # only the effective cwd of this store changes.
                store.cwd = Path(cwd_override).resolve()
            else:
                store.cwd = Path(store._header.cwd).resolve()
            return store
        except BaseException:
            if store is not None:
                store.close()
            raise

    @staticmethod
    def in_memory(
        cwd: Path | str = ".",
        *,
        model: dict[str, JsonValue] | None = None,
        config: dict[str, JsonValue] | None = None,
    ) -> SessionStore:
        header = SessionHeader(
            cwd=str(Path(cwd).resolve()), model=model, config=dict(config) if config else {}
        )
        return SessionStore(cwd, persist=False, header=header, entries=[])

    def _open_and_lock(self, file_path: Path, *, create: bool) -> None:
        """Open and exclusively lock the session file (creating it if asked)."""
        self._session_file = file_path
        if not self.persist:
            return
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        fd = os.open(file_path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(fd)
            raise SessionLockError(file_path) from error
        self._lock_fd = fd

    def _recover_trailing_fragment(self, fragment: bytes, *, allow_header: bool) -> bool:
        """Recover uncommitted final bytes under the writer lock.

        Contracts and failure semantics: docs/invariants.md.
        """
        if not self.persist or self._lock_fd is None or not fragment:
            return False
        try:
            text = fragment.decode("utf-8")
        except UnicodeDecodeError:
            text = None  # torn multi-byte character at the cut: uncommitted
        record: object = None
        if text is not None and text.strip():
            try:
                record = json.loads(text, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError):
                record = None
        if isinstance(record, dict) and isinstance(record.get("type"), str):
            line_no = _committed_line_count(self._entries)
            path = self._session_file if self._session_file is not None else Path("<session>")
            if self._file_has_header:
                try:
                    entry = _entry_from_record(record, path, line_no, self._entry_index)
                except SessionCorruptError:
                    entry = None  # unparseable-as-record fragment: truncate below
                if entry is not None:
                    try:
                        # Same raw-history call-ID invariant as a normal
                        # append; a violating fragment is not adoptable.
                        self._validate_call_ids(entry)
                    except ValueError:
                        self._truncate_fragment(fragment)
                        return False
                    # Seal first; only then adopt into memory.
                    self._append_raw(b"\n")
                    self._commit_entry(entry)
                else:
                    self._truncate_fragment(fragment)
            elif allow_header and record.get("type") == "session":
                try:
                    header = _header_from_record(record, path, 1)
                except SessionCorruptError:
                    self._truncate_fragment(fragment)
                    return False
                self._append_raw(b"\n")
                self._header = header
                self._reindex()
                self._file_has_header = True
                return True
            else:
                # An entry fragment with no header on disk cannot be adopted.
                self._truncate_fragment(fragment)
        else:
            self._truncate_fragment(fragment)
        return False

    def _truncate_fragment(self, fragment: bytes) -> None:
        assert self._session_file is not None and self._lock_fd is not None
        committed_len = self._session_file.stat().st_size - len(fragment)
        os.ftruncate(self._lock_fd, committed_len)
        os.fsync(self._lock_fd)

    # -- attributes -----------------------------------------------------

    @property
    def header(self) -> SessionHeader:
        return deepcopy(self._header)

    @property
    def session_id(self) -> str:
        return self._header.id

    @property
    def session_file(self) -> Path | None:
        return self._session_file

    @property
    def entries(self) -> list[SessionEntry]:
        """Independent journal snapshots; mutations cannot change committed records."""
        return deepcopy(self._entries)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def call_ids(self) -> set[str]:
        """All raw-history call IDs, including hidden and compacted attempts."""
        return set(self._call_ids)

    @property
    def last_model_change_source(self) -> int:
        return self._last_model_change_source

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def last_entry(self) -> SessionEntry | None:
        return deepcopy(self._entries[-1]) if self._entries else None

    @property
    def unanswered_calls(self) -> list[ToolCallBlock]:
        return deepcopy(list(self._unanswered_calls.values()))

    def _reindex(self) -> None:
        self._entry_index.clear()
        self._entry_positions.clear()
        self._omitted_ids.clear()
        self._latest_compaction_source = None
        self._call_ids.clear()
        self._unanswered_calls.clear()
        self._answered_ids.clear()
        self._last_model_change_source = -1
        self._previous_model_identity = _model_identity(
            self._header.config.get("model") or self._header.model
        )
        self._has_user_or_assistant = False
        for source, entry in enumerate(self._entries):
            self._index_entry(entry, source)
        self._revision = len(self._entries)
        self._projection_cache = None

    def _index_entry(self, entry: SessionEntry, source: int) -> None:
        self._entry_index[entry.id] = entry
        self._entry_positions[entry.id] = source
        if isinstance(entry, SessionCompactionEntry):
            self._latest_compaction_source = source
        elif isinstance(entry, SessionOmissionEntry):
            self._omitted_ids.add(entry.target_entry_id)
        if isinstance(entry, SessionMessageEntry):
            if isinstance(entry.message, (UserMessage, AssistantMessage)):
                self._has_user_or_assistant = True
            if isinstance(entry.message, AssistantMessage):
                for block in entry.message.content:
                    if isinstance(block, ToolCallBlock):
                        self._call_ids.add(block.id)
                        if block.id not in self._answered_ids:
                            self._unanswered_calls[block.id] = block
            elif isinstance(entry.message, ToolResultMessage):
                self._answered_ids.add(entry.message.tool_call_id)
                self._unanswered_calls.pop(entry.message.tool_call_id, None)
        elif isinstance(entry, SessionConfigEntry) and isinstance(entry.config.get("model"), dict):
            identity = _model_identity(entry.config["model"])
            if identity != self._previous_model_identity:
                self._last_model_change_source = source
            self._previous_model_identity = identity

    def _commit_entry(self, entry: SessionEntry) -> None:
        self._index_entry(entry, len(self._entries))
        self._entries.append(entry)
        self._revision += 1
        self._projection_cache = None

    @property
    def usable(self) -> bool:
        """False once a durable write has failed; appends then raise."""
        return self._unusable is None

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return deepcopy(self._entry_index.get(entry_id))

    # -- appends --------------------------------------------------------

    def append_message(self, message: Message) -> str:
        entry = SessionMessageEntry(id=self._next_id(), message=message)
        self._append(entry)
        return entry.id

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        *,
        usage: Usage | None = None,
        system_message: SystemMessage | None = None,
        read_files: list[str] | None = None,
        modified_files: list[str] | None = None,
    ) -> str:
        entry = SessionCompactionEntry(
            id=self._next_id(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            usage=usage,
            system_message=system_message,
            read_files=list(read_files or []),
            modified_files=list(modified_files or []),
            file_lists_recorded=read_files is not None or modified_files is not None,
        )
        self._append(entry)
        return entry.id

    def append_context_omission(self, target_entry_id: str) -> str:
        """Durably omit one message entry from the provider projection.

        The raw record stays in the journal. Raises if the target does not
        exist or is not a message entry.
        """
        target = self.get_entry(target_entry_id)
        if not isinstance(target, SessionMessageEntry):
            raise ValueError(f"Cannot omit unknown or non-message entry {target_entry_id!r}")
        entry = SessionOmissionEntry(id=self._next_id(), target_entry_id=target_entry_id)
        self._append(entry)
        return entry.id

    def append_config_change(self, config: dict[str, JsonValue]) -> str:
        """Durably snapshot the effective configuration.

        The latest config_change record overrides the header's config on
        resume, so explicit overrides survive reopening.
        """
        entry = SessionConfigEntry(id=self._next_id(), config=dict(config))
        self._append(entry)
        return entry.id

    def _next_id(self) -> str:
        for _ in range(100):
            candidate = uuid.uuid4().hex[:8]
            if candidate not in self._entry_index:
                return candidate
        return uuid.uuid4().hex

    def _has_conversation(self, extra_entry: SessionEntry) -> bool:
        """A file appears once a user or assistant message exists (Pi rule).

        ``extra_entry`` is the entry about to be appended; earlier pending
        entries already live in ``self._entries``.
        """
        return self._has_user_or_assistant or (
            isinstance(extra_entry, SessionMessageEntry)
            and isinstance(extra_entry.message, (UserMessage, AssistantMessage))
        )

    def _append_raw(self, data: bytes) -> None:
        """Write raw bytes at end-of-file and fsync. Raises on failure."""
        if self._lock_fd is None:
            raise RuntimeError("Session store has no writer lock")
        os.lseek(self._lock_fd, 0, os.SEEK_END)
        view = memoryview(data)
        while view:
            written = os.write(self._lock_fd, view)
            if written <= 0:
                raise OSError("os.write made no progress; refusing to loop")
            view = view[written:]
        os.fsync(self._lock_fd)

    def _append(self, entry: SessionEntry) -> None:
        """Durably append one entry; memory advances only after a full write.

        The record is validated with the same rules as load before anything
        touches the file, so an invalid provider object can never be written
        successfully only to be rejected on the next open. A durability
        failure (write/fsync) marks the store unusable; a validation failure
        rejects the record without poisoning anything, since no bytes were
        written.
        """
        if self._closed:
            raise RuntimeError("Session store is closed")
        if self._unusable is not None:
            raise SessionUnusableError(f"session store is unusable after a failed write: {self._unusable}")
        entry = deepcopy(entry)
        self._validate_record(entry)
        self._validate_call_ids(entry)
        if not self.persist or self._session_file is None:
            self._commit_entry(entry)
            return

        if not self._file_has_header:
            # Lazy file creation: nothing is durable until the conversation
            # starts; then the header plus every pending entry go in one
            # write so the file always begins with exactly one header.
            if not self._has_conversation(entry):
                self._commit_entry(entry)
                return
            self._validate_header()
            payload = "".join(
                _dumps(record) + "\n"
                for record in [self._header.to_json(), *(item.to_json() for item in [*self._entries, entry])]
            )
            try:
                self._append_raw(payload.encode("utf-8"))
            except OSError as error:
                self._mark_unusable(error)
                raise
            self._file_has_header = True
            self._commit_entry(entry)
            return

        try:
            self._append_raw((_dumps(entry.to_json()) + "\n").encode("utf-8"))
        except OSError as error:
            self._mark_unusable(error)
            raise
        self._commit_entry(entry)

    def _mark_unusable(self, error: BaseException) -> None:
        """Poison the store after a durability failure."""
        if self._unusable is None:
            self._unusable = error

    def _validate_record(self, entry: SessionEntry) -> None:
        """Run the load-time validators on a record about to be written."""
        path = self._session_file if self._session_file is not None else Path("<session>")
        line_no = _committed_line_count(self._entries)
        try:
            serialized = _dumps(entry.to_json())
        except ValueError as error:
            raise ValueError(f"record is not JSON-serializable: {error}") from error
        try:
            record = json.loads(serialized, parse_constant=_reject_constant)
            _entry_from_record(record, path, line_no, self._entry_index)
        except SessionCorruptError as error:
            raise ValueError(f"record rejected by session validation: {error.reason}") from error

    def _validate_call_ids(self, entry: SessionEntry) -> None:
        """Reject an assistant message that reuses any earlier raw call ID.

        The same invariant ``load_session_records`` enforces for loaded
        journals, applied incrementally so a live append can never create a
        journal the next open would reject: every finalized tool-call ID is
        unique across raw session history — including omitted attempts and
        pre-compaction history. Retries must use fresh call IDs.
        """
        if not isinstance(entry, SessionMessageEntry) or not isinstance(entry.message, AssistantMessage):
            return
        entry_ids = {
            block.id for block in entry.message.content if isinstance(block, ToolCallBlock) and block.id
        }
        if len(entry_ids) != sum(
            1 for block in entry.message.content if isinstance(block, ToolCallBlock) and block.id
        ):
            raise ValueError("tool call ids duplicated within one assistant message cannot be paired")
        clash = sorted(entry_ids & self._call_ids)
        if clash:
            raise ValueError(
                f"tool call id(s) {clash} already used by an earlier assistant message "
                "in raw session history; tool-call ids must be unique"
            )

    def _validate_header(self) -> None:
        """Validate the header record before it is first written to disk."""
        path = self._session_file if self._session_file is not None else Path("<session>")
        try:
            serialized = _dumps(self._header.to_json())
        except ValueError as error:
            raise ValueError(f"session header is not JSON-serializable: {error}") from error
        try:
            _header_from_record(json.loads(serialized, parse_constant=_reject_constant), path, 1)
        except SessionCorruptError as error:
            raise ValueError(f"session header rejected by validation: {error.reason}") from error

    # -- locking / closing ----------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._lock_fd)
            self._lock_fd = None

    # -- discovery --------------------------------------------------------

    @staticmethod
    def list_sessions(session_dir: Path | str, *, match_cwd: Path | str | None = None) -> list[SessionInfo]:
        """Best-effort discovery: unparseable files are skipped, not fatal.

        Only newline-committed records are counted. An uncommitted tail is
        left for writer recovery, so a crashed session remains discoverable.

        When ``match_cwd`` is given, only sessions whose header cwd resolves
        to the same directory are returned (exact-cwd filtering; the
        per-cwd directory name is a hash and carries no path information).
        """
        directory = Path(session_dir)
        if not directory.is_dir():
            return []
        infos: list[SessionInfo] = []
        for file_path in sorted(directory.glob("*.jsonl")):
            try:
                infos.append(_scan_session_file(file_path))
            except Exception:
                continue
        if match_cwd is not None:
            resolved = str(Path(match_cwd).resolve())
            infos = [info for info in infos if info.cwd == resolved]
        infos.sort(key=lambda info: info.modified, reverse=True)
        return infos


def _committed_line_count(entries: list[SessionEntry]) -> int:
    return len(entries) + 1  # entries plus header; fragments start after


def _scan_session_file(file_path: Path) -> SessionInfo:
    stat = file_path.stat()
    header: SessionHeader | None = None
    message_count = 0
    first_message = ""
    with open(file_path, "rb") as file:
        for raw_line in file:
            if not raw_line.endswith(b"\n"):
                break
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise SessionCorruptError(file_path, message_count + 1, f"invalid UTF-8: {error}") from error
            if not line.strip():
                continue
            record = _parse_line(line, file_path, message_count + 1)
            if header is None:
                if record.get("type") != "session":
                    raise SessionCorruptError(file_path, 1, "first record is not a session header")
                header = _header_from_record(record, file_path, 1)
                continue
            if record.get("type") == "message":
                message_count += 1
                message = record.get("message")
                if not isinstance(message, dict):
                    continue
                if not first_message and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        first_message = content
                    elif isinstance(content, list):
                        first_message = "\n".join(
                            str(block.get("text", ""))
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
    if header is None:
        raise SessionCorruptError(file_path, 1, "file has no session header")
    return SessionInfo(
        path=file_path,
        id=header.id,
        cwd=header.cwd,
        created=header.timestamp,
        modified=stat.st_mtime,
        message_count=message_count,
        first_message=first_message.splitlines()[0] if first_message else "(no messages)",
    )


def read_session_file(path: Path | str) -> tuple[SessionHeader, list[SessionEntry]]:
    """Read-only inspection: parse strictly without locking, creating, or repair.

    Any uncommitted trailing bytes after the last newline (a crashed writer)
    are rejected here — inspection never guesses whether a fragment is a
    complete record; the writer path (:meth:`SessionStore.open`) is the one
    that seals or truncates it under the lock.
    """
    file_path = Path(path)
    loaded = load_session_records(file_path)
    if loaded.header is None:
        raise SessionCorruptError(file_path, 1, "file has no session header")
    if loaded.trailing_fragment:
        raise SessionCorruptError(
            file_path,
            len(loaded.entries) + 2,
            "uncommitted trailing bytes (a crashed writer); open with SessionStore.open to recover",
        )
    return loaded.header, loaded.entries


def latest_config_change(entries: list[SessionEntry]) -> dict[str, JsonValue] | None:
    """The newest durable config snapshot, or None when none was recorded."""
    for entry in reversed(entries):
        if isinstance(entry, SessionConfigEntry):
            return entry.config
    return None
