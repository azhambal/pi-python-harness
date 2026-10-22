"""Offline microbenchmark; run with .venv/bin/python scripts/benchmark_projection.py."""

import argparse
from pathlib import Path
from timeit import repeat

from pi_agent.session import build_projection
from pi_agent.session_store import SessionCompactionEntry, SessionHeader, SessionMessageEntry, SessionStore
from pi_agent.types import AssistantMessage, TextBlock, ToolCallBlock, ToolResultMessage, UserMessage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=4000)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    if args.turns < 10 or args.iterations < 1:
        parser.error("turns must be >= 10; iterations must be >= 1")
    entries = []
    for n in range(args.turns):
        cid = f"call-{n}"
        for message in (
            UserMessage(content=[TextBlock(text="task")]),
            AssistantMessage(content=[ToolCallBlock(id=cid, name="read", arguments={"path": "x"})]),
            ToolResultMessage(tool_call_id=cid, tool_name="read", content=[TextBlock(text="ok")]),
        ):
            entries.append(SessionMessageEntry(id=str(len(entries)), message=message))
    last_id = entries[-1].id
    entries.append(
        SessionCompactionEntry(id="checkpoint", summary="Summary", first_kept_entry_id=str(len(entries) - 30))
    )
    store = SessionStore(Path("/tmp"), persist=False, header=SessionHeader(), entries=entries)
    build_projection(store)

    def cold_projection():
        # Benchmark cache rebuilding without changing the fixture or timing an append.
        store._projection_cache = None
        return build_projection(store)

    operations = {
        "projection_warm": lambda: build_projection(store),
        "projection_cold": cold_projection,
        "raw_ids": lambda: store.call_ids,
        "entry_lookup": lambda: store.get_entry(last_id),
        "unanswered_calls": lambda: store.unanswered_calls,
    }
    for label, operation in operations.items():
        elapsed = min(repeat(operation, number=args.iterations, repeat=3)) / args.iterations
        print(f"{label}: {elapsed * 1000:.3f} ms/call")
    print(
        f"journal_entries: {store.entry_count}; visible_messages: {sum(len(e.messages) for e in build_projection(store))}"
    )
    store.close()


if __name__ == "__main__":
    main()
