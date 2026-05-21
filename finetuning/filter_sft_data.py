from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, write_jsonl


ACTION_TASK_TYPES = {
    "question_decomposition",
    "query_rewrite",
    "doc_selection",
    "document_find",
    "candidate_extraction",
    "candidate_verification",
    "finish_decision",
    "action_decision",
}


def rejection_reason(row: dict[str, Any], *, max_user_chars: int = 8000, max_assistant_chars: int = 1000) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return "missing_messages"
    if any(
        message.get("role") == "assistant" and "<think>" in str(message.get("content", "")).lower()
        for message in messages
    ):
        return "contains_think"
    assistant = messages[-1]
    if assistant.get("role") != "assistant":
        return "last_message_not_assistant"
    if len(str(assistant.get("content", ""))) > max_assistant_chars:
        return "assistant_too_long"
    if sum(len(str(message.get("content", ""))) for message in messages if message.get("role") == "user") > max_user_chars:
        return "user_too_long"
    task_type = str(row.get("task_type", ""))
    if task_type in ACTION_TASK_TYPES:
        try:
            action = json.loads(str(assistant.get("content", "")))
        except json.JSONDecodeError:
            return "invalid_action_json"
        if not isinstance(action, dict) or not action.get("action"):
            return "missing_action"
        if str(action.get("action")) == "search" and not str(action.get("query", "")).strip():
            return "missing_search_query"
        if str(action.get("action")) == "get_document" and not str(action.get("docid", "")).strip():
            return "missing_docid"
    elif task_type == "final_answer":
        if not str(assistant.get("content", "")).strip():
            return "empty_final_answer"
    else:
        return "unknown_task_type"
    return None


def filter_records(
    rows: Iterable[dict[str, Any]],
    *,
    max_user_chars: int = 8000,
    max_assistant_chars: int = 1000,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[tuple[dict[str, Any], str]] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id and row_id in seen_ids:
            rejected.append((row, "duplicate_id"))
            continue
        reason = rejection_reason(row, max_user_chars=max_user_chars, max_assistant_chars=max_assistant_chars)
        if reason is None:
            kept.append(row)
            if row_id:
                seen_ids.add(row_id)
        else:
            rejected.append((row, reason))
    return kept, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SFT JSONL records for format and safety.")
    parser.add_argument("--input", required=True, help="Input SFT JSONL path")
    parser.add_argument("--output", required=True, help="Filtered output JSONL path")
    parser.add_argument("--rejected-output", default=None, help="Optional rejected records JSONL path")
    parser.add_argument("--max-user-chars", type=int, default=8000)
    parser.add_argument("--max-assistant-chars", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kept, rejected = filter_records(
        read_jsonl(args.input),
        max_user_chars=args.max_user_chars,
        max_assistant_chars=args.max_assistant_chars,
    )
    write_jsonl(args.output, kept)
    if args.rejected_output:
        write_jsonl(
            args.rejected_output,
            ({**row, "reject_reason": reason} for row, reason in rejected),
        )
    print(f"Kept {len(kept)} records; rejected {len(rejected)} records")


if __name__ == "__main__":
    main()
