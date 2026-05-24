from __future__ import annotations

import argparse
import json
import re
from typing import Any, Iterable, Iterator

from agent.multistep_agent import ACTION_DECISION_SYSTEM_PROMPT, FINAL_ANSWER_SYSTEM_PROMPT

from .common import compact_text, read_jsonl, stable_id, write_jsonl


def _keywords(text: str, *, limit: int = 10) -> list[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "from",
        "into",
        "listed",
        "same",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", text)
    seen: set[str] = set()
    kept: list[str] = []
    for word in words:
        lowered = word.lower().strip("'")
        if lowered in stopwords or lowered in seen:
            continue
        seen.add(lowered)
        kept.append(word.strip("'"))
        if len(kept) >= limit:
            break
    return kept


def _answer_type(answer: str) -> str:
    answer = answer.strip()
    if re.fullmatch(r"\d{4}(?:-\d{2}-\d{2})?", answer):
        return "date"
    if re.fullmatch(r"\d+(?:\.\d+)?%?", answer):
        return "number"
    if re.fullmatch(r"(?:Dr\.?\s+|Prof\.?\s+|Professor\s+)?[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4}", answer):
        return "person"
    if len(answer.split()) <= 6:
        return "short answer"
    return "title or phrase"


def _distractors(tasks: list[dict[str, Any]], index: int, *, limit: int = 2) -> list[dict[str, Any]]:
    current = tasks[index]
    same_type = [
        task
        for offset, task in enumerate(tasks)
        if offset != index and task.get("task_type") == current.get("task_type") and task.get("answer")
    ]
    fallback = [task for offset, task in enumerate(tasks) if offset != index and task.get("answer")]
    picked = same_type[:limit] or fallback[:limit]
    return picked[:limit]


def _search_results(task: dict[str, Any], distractors: list[dict[str, Any]]) -> str:
    rows = [
        {
            "docid": str(task.get("source_docid", "")),
            "score": 19.2,
            "snippet": compact_text(task.get("evidence", ""), max_chars=700),
        }
    ]
    for rank, distractor in enumerate(distractors, start=1):
        rows.append(
            {
                "docid": str(distractor.get("source_docid", f"distractor-{rank}")),
                "score": round(18.4 - rank * 0.7, 2),
                "snippet": compact_text(distractor.get("evidence", ""), max_chars=550),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def _state_block(
    task: dict[str, Any],
    *,
    last_action: str,
    verifier: str,
    candidates: str,
    opened: str,
) -> str:
    question = str(task.get("question", "")).strip()
    keywords = " | ".join(_keywords(question))
    return "\n".join(
        [
            f"Question: {question}",
            f"Expected answer type: {_answer_type(str(task.get('answer', '')))}",
            f"Planned clue queries: {keywords}",
            "Known facts:",
            compact_text(opened, max_chars=1200) if opened else "None",
            "Open questions:",
            "1. Verify that the chosen answer satisfies the relation asked in the question.",
            "Searches already tried:",
            keywords or "None",
            "Opened documents:",
            str(task.get("source_docid", "")) if opened else "None",
            "Current candidate answers:",
            candidates or "None",
            "Verifier results:",
            verifier or "None",
            f"Last action: {last_action}",
            "Stall count: 0",
        ]
    )


def _action_user_content(task: dict[str, Any], *, task_type: str, state: str, observation: str) -> str:
    return compact_text(
        "\n".join(
            [
                f"Task type: {task_type}",
                f"Question:\n{task.get('question', '')}",
                f"State summary:\n{state}",
                f"Recent observation:\n{observation}",
                "Return exactly one valid action JSON object.",
            ]
        ),
        max_chars=7000,
    )


def _record(
    task: dict[str, Any],
    *,
    variant: str,
    task_type: str,
    system: str,
    user_content: str,
    assistant_payload: dict[str, Any],
) -> dict[str, Any]:
    assistant_content = json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": stable_id("v3_contrast_sft", task.get("id", ""), variant, assistant_content, prefix="sft"),
        "task_type": task_type,
        "source": "v3_contrast_sft",
        "source_query_id": str(task.get("id", "")),
        "source_docid": str(task.get("source_docid", "")),
        "quality": {
            "synthetic_pattern": variant,
            "from_correct_trajectory": True,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def _query_from_question(question: str) -> str:
    keywords = _keywords(question, limit=12)
    return " ".join(keywords) or question[:120]


def build_contrast_records(
    tasks: Iterable[dict[str, Any]],
    *,
    max_source_tasks: int | None = None,
) -> Iterator[dict[str, Any]]:
    task_list = [task for task in tasks if task.get("question") and task.get("answer") and task.get("evidence")]
    if max_source_tasks is not None:
        task_list = task_list[:max_source_tasks]
    for index, task in enumerate(task_list):
        answer = compact_text(task.get("answer", ""), max_chars=180)
        distractors = _distractors(task_list, index)
        distractor_answer = compact_text(distractors[0].get("answer", "nearby distractor"), max_chars=180) if distractors else "nearby distractor"
        evidence = compact_text(task.get("evidence", ""), max_chars=1400)
        search_observation = _search_results(task, distractors)

        state = _state_block(task, last_action=f"search:{_query_from_question(str(task.get('question', '')))}", verifier="", candidates="", opened="")
        yield _record(
            task,
            variant="doc_selection_true_doc",
            task_type="doc_selection",
            system=ACTION_DECISION_SYSTEM_PROMPT,
            user_content=_action_user_content(
                task,
                task_type="doc_selection",
                state=state,
                observation=search_observation,
            ),
            assistant_payload={
                "action": "get_document",
                "docid": str(task.get("source_docid", "")),
                "reason": "Open the result whose snippet directly contains the relation needed by the question.",
            },
        )

        weak_state = _state_block(
            task,
            last_action="verify_claim",
            verifier=f"{distractor_answer}: supported=True score=0.70 note=Candidate appears in one weak document but the relation is not fully verified.",
            candidates=distractor_answer,
            opened=f"Opened weak evidence mentions {distractor_answer}, but it does not prove the full question relation.",
        )
        yield _record(
            task,
            variant="weak_support_continue_search",
            task_type="query_rewrite",
            system=ACTION_DECISION_SYSTEM_PROMPT,
            user_content=_action_user_content(
                task,
                task_type="query_rewrite",
                state=weak_state,
                observation="A single weakly supported candidate is not enough; continue searching for relation-complete evidence.",
            ),
            assistant_payload={
                "action": "search",
                "query": _query_from_question(str(task.get("question", ""))),
                "reason": "Do not finish on a single weak candidate; search again for evidence that proves the requested relation.",
            },
        )

        opened_state = _state_block(
            task,
            last_action=f"get_document:{task.get('source_docid', '')}",
            verifier="None",
            candidates=f"{distractor_answer}; {answer}",
            opened=evidence,
        )
        yield _record(
            task,
            variant="opened_evidence_find_relation",
            task_type="document_find",
            system=ACTION_DECISION_SYSTEM_PROMPT,
            user_content=_action_user_content(
                task,
                task_type="document_find",
                state=opened_state,
                observation="The opened document contains several plausible values; search within it for the exact relation asked by the question.",
            ),
            assistant_payload={
                "action": "find_in_document",
                "docid": str(task.get("source_docid", "")),
                "query": _query_from_question(str(task.get("question", ""))),
                "reason": "Focus the opened document on the relation in the question before choosing a final answer.",
            },
        )

        candidate_state = _state_block(
            task,
            last_action="extract_answer_candidates",
            verifier="None",
            candidates=f"{distractor_answer}; {answer}",
            opened=evidence,
        )
        yield _record(
            task,
            variant="verify_true_candidate_from_opened_evidence",
            task_type="candidate_verification",
            system=ACTION_DECISION_SYSTEM_PROMPT,
            user_content=_action_user_content(
                task,
                task_type="candidate_verification",
                state=candidate_state,
                observation="Candidate extraction produced both a distractor and the relation-supported answer from the opened evidence.",
            ),
            assistant_payload={
                "action": "verify_claim",
                "candidate_answer": answer,
                "reason": "Verify the candidate that matches the asked relation in the opened evidence before considering finish.",
            },
        )

        strong_state = _state_block(
            task,
            last_action="verify_claim",
            verifier=f"{answer}: supported=True score=0.90 note=Opened evidence proves the requested relation.",
            candidates=answer,
            opened=evidence,
        )
        yield _record(
            task,
            variant="strong_support_finish",
            task_type="finish_decision",
            system=ACTION_DECISION_SYSTEM_PROMPT,
            user_content=_action_user_content(
                task,
                task_type="finish_decision",
                state=strong_state,
                observation="The candidate is supported by opened evidence with the exact relation requested.",
            ),
            assistant_payload={
                "action": "finish",
                "answer_hint": answer,
                "reason": "Opened evidence supports the exact answer and the requested relation is verified.",
            },
        )

        final_user = compact_text(
            "\n".join(
                [
                    f"Question:\n{task.get('question', '')}",
                    "",
                    "Retrieved evidence:",
                    f"- docid {task.get('source_docid', '')}: {evidence}",
                    "",
                    "Candidate answers:",
                    f"- {distractor_answer}: weak or incomplete support",
                    f"- {answer}: supported by opened evidence",
                    "",
                    "Return a JSON object with exact_answer, confidence, and support.",
                ]
            ),
            max_chars=7000,
        )
        yield _record(
            task,
            variant="final_select_supported_answer",
            task_type="final_answer",
            system=FINAL_ANSWER_SYSTEM_PROMPT,
            user_content=final_user,
            assistant_payload={
                "exact_answer": answer,
                "confidence": 0.9,
                "support": "Opened evidence directly supports the selected answer.",
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 contrast SFT records from synthetic tasks.")
    parser.add_argument("--tasks", required=True, help="Input synthetic_tasks_v3.jsonl")
    parser.add_argument("--output", required=True, help="Output SFT JSONL")
    parser.add_argument("--max-source-tasks", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_jsonl(
        args.output,
        build_contrast_records(
            read_jsonl(args.tasks),
            max_source_tasks=args.max_source_tasks,
        ),
    )
    print(f"Wrote {count} v3 contrast SFT records to {args.output}")


if __name__ == "__main__":
    main()
