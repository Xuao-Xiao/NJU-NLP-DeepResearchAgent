from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from agent.multistep_agent import FINAL_ANSWER_SYSTEM_PROMPT, _build_final_user_prompt

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


def _action_from_finish_message(message: dict[str, Any]) -> dict[str, Any] | None:
    action_plan = message.get("action_plan")
    if isinstance(action_plan, dict):
        action = dict(action_plan)
    else:
        action = safe_json_loads(message.get("content", ""))
    if str(action.get("action", "")).strip() != "finish":
        return None
    reason = compact_text(action.get("reason") or message.get("content", ""), max_chars=240)
    return {
        "action": "finish",
        "answer_hint": compact_text(action.get("answer_hint", ""), max_chars=200),
        "reason": reason or "Stop because opened evidence supports the current candidate answer.",
    }


def _final_answer_content(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            raw_content = str(message.get("content", ""))
            if "<think>" in raw_content.lower():
                continue
            parsed = safe_json_loads(raw_content)
            exact = compact_text(parsed.get("exact_answer", ""), max_chars=300)
            if exact:
                payload = {
                    "exact_answer": exact,
                    "confidence": parsed.get("confidence", 0),
                    "support": compact_text(parsed.get("support", "Verified by trajectory evidence."), max_chars=300),
                }
                return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    predicted = compact_text(row.get("predicted_answer", ""), max_chars=300)
    if predicted and "<think>" not in predicted.lower():
        payload = {
            "exact_answer": predicted,
            "confidence": 0,
            "support": "Trajectory top-level predicted answer.",
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return ""


def _build_final_answer_user_content(question: str, row: dict[str, Any], max_chars: int) -> str:
    state = row.get("agent_state")
    if isinstance(state, dict):
        try:
            return compact_text(_build_final_user_prompt(question=question, state=state), max_chars=max_chars)
        except Exception:
            pass
    return compact_text(
        "\n".join([f"Question:\n{question}", "", "Give the final answer now."]),
        max_chars=max_chars,
    )


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
            if isinstance(tool_calls, list) and tool_calls:
                action = _action_from_tool_call(tool_calls[0], str(message.get("content", "")))
            else:
                action = _action_from_finish_message(message)
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
            final_content = _final_answer_content(row)
            if final_content:
                yield {
                    "id": stable_id(submission_path, query_id, "final", final_content, prefix="sft"),
                    "task_type": "final_answer",
                    "source": str(submission_path),
                    "source_query_id": query_id,
                    "quality": {
                        "eval_judgment": judgments.get(query_id, ""),
                        "from_correct_trajectory": judgments.get(query_id) == "CORRECT" if judgments else None,
                    },
                    "messages": [
                        {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _build_final_answer_user_content(
                                question=question,
                                row=row,
                                max_chars=max_user_chars,
                            ),
                        },
                        {"role": "assistant", "content": final_content},
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
