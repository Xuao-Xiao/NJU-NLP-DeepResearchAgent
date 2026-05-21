from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from .common import compact_text, normalize_judgment, read_jsonl, safe_json_loads, stable_id, write_jsonl


ACTION_TASK_TYPES = {
    "decompose_question": "question_decomposition",
    "search": "query_rewrite",
    "get_document": "doc_selection",
    "find_in_document": "document_find",
    "extract_answer_candidates": "candidate_extraction",
    "verify_claim": "candidate_verification",
    "finish": "finish_decision",
}


def collect_success_ids(eval_path: str | Path, correct_label: str = "CORRECT") -> set[str]:
    success: set[str] = set()
    for row in read_jsonl(eval_path):
        query_id = row.get("query_id")
        if query_id is None:
            continue
        if normalize_judgment(row.get("eval_judgment")) == correct_label:
            success.add(str(query_id))
    return success


def load_eval_judgments(eval_path: str | Path | None) -> dict[str, str]:
    if eval_path is None:
        return {}
    judgments: dict[str, str] = {}
    for row in read_jsonl(eval_path):
        query_id = row.get("query_id")
        if query_id is not None:
            judgments[str(query_id)] = normalize_judgment(row.get("eval_judgment"))
    return judgments


def _first_system_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "system":
            return str(message.get("content", "")).strip()
    return "You are a Deep Research Agent working over a fixed offline document corpus."


def _first_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def _previous_tool_observation(messages: list[dict[str, Any]], index: int, max_chars: int) -> str:
    for previous in reversed(messages[:index]):
        if previous.get("role") == "tool":
            return compact_text(previous.get("content", ""), max_chars=max_chars)
    return "No tool has been used immediately before this decision."


def _action_from_tool_call(tool_call: dict[str, Any], assistant_content: str) -> dict[str, Any] | None:
    function = tool_call.get("function", {})
    name = str(function.get("name", "")).strip()
    if not name:
        return None
    args = safe_json_loads(function.get("arguments", "{}"))
    action: dict[str, Any] = {"action": name}
    action.update(args)
    reason = compact_text(assistant_content, max_chars=240)
    if reason and "<think>" not in reason.lower():
        action.setdefault("reason", reason)
    else:
        action.setdefault("reason", f"Select {name} as the next evidence-gathering action.")
    return action


def _build_user_content(
    question: str,
    state_summary: str,
    observation: str,
    task_type: str,
    max_chars: int,
) -> str:
    return compact_text(
        "\n".join(
            [
                f"Task type: {task_type}",
                f"Question:\n{question}",
                f"State summary:\n{state_summary or 'No state summary was recorded.'}",
                f"Recent observation:\n{observation}",
                "Return exactly one valid action JSON object.",
            ]
        ),
        max_chars=max_chars,
    )


def extract_sft_records(
    submission_path: str | Path,
    eval_path: str | Path | None = None,
    *,
    only_correct: bool = True,
    include_final_answer: bool = False,
    max_user_chars: int = 6000,
    max_observation_chars: int = 2000,
) -> Iterator[dict[str, Any]]:
    judgments = load_eval_judgments(eval_path)
    for row in read_jsonl(submission_path):
        query_id = str(row.get("query_id", ""))
        if only_correct and judgments and judgments.get(query_id) != "CORRECT":
            continue
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        question = _first_user_message(messages)
        system = _first_system_message(messages)
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list) or not tool_calls:
                continue
            action = _action_from_tool_call(tool_calls[0], str(message.get("content", "")))
            if action is None:
                continue
            task_type = ACTION_TASK_TYPES.get(str(action.get("action")), "action_decision")
            state_summary = str(message.get("state_summary", "")).strip()
            observation = _previous_tool_observation(messages, index, max_chars=max_observation_chars)
            assistant_content = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            record_id = stable_id(submission_path, query_id, index, assistant_content, prefix="sft")
            yield {
                "id": record_id,
                "task_type": task_type,
                "source": str(submission_path),
                "source_query_id": query_id,
                "quality": {
                    "eval_judgment": judgments.get(query_id, ""),
                    "from_correct_trajectory": judgments.get(query_id) == "CORRECT" if judgments else None,
                },
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": _build_user_content(
                            question=question,
                            state_summary=state_summary,
                            observation=observation,
                            task_type=task_type,
                            max_chars=max_user_chars,
                        ),
                    },
                    {"role": "assistant", "content": assistant_content},
                ],
            }
        if include_final_answer:
            predicted = compact_text(row.get("predicted_answer", ""), max_chars=300)
            if predicted and "<think>" not in predicted.lower():
                yield {
                    "id": stable_id(submission_path, query_id, "final", predicted, prefix="sft"),
                    "task_type": "final_answer",
                    "source": str(submission_path),
                    "source_query_id": query_id,
                    "quality": {
                        "eval_judgment": judgments.get(query_id, ""),
                        "from_correct_trajectory": judgments.get(query_id) == "CORRECT" if judgments else None,
                    },
                    "messages": [
                        {"role": "system", "content": "Return only the final short answer supported by evidence."},
                        {"role": "user", "content": f"Question:\n{question}\n\nReturn the final short answer only."},
                        {"role": "assistant", "content": predicted},
                    ],
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SFT records from agent trajectory JSONL files.")
    parser.add_argument("--submission", required=True, help="Path to multistep_submission.jsonl")
    parser.add_argument("--eval", default=None, help="Optional path to multistep_eval_results.jsonl")
    parser.add_argument("--output", required=True, help="Output SFT JSONL path")
    parser.add_argument("--include-incorrect", action="store_true", help="Include trajectories not judged CORRECT")
    parser.add_argument("--include-final-answer", action="store_true", help="Also emit final-answer format samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = extract_sft_records(
        args.submission,
        eval_path=args.eval,
        only_correct=not args.include_incorrect,
        include_final_answer=args.include_final_answer,
    )
    count = write_jsonl(args.output, records)
    print(f"Wrote {count} SFT records to {args.output}")


if __name__ == "__main__":
    main()

