from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl


def normalize_answer(text: object) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\"'`]+|[\"'`]+$", "", value)
    value = re.sub(r"[.。]+$", "", value)
    return value


def evaluate_predictions(
    tasks: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_by_id = {str(task.get("id")): task for task in tasks}
    rows: list[dict[str, Any]] = []
    correct = 0
    for pred in predictions:
        query_id = str(pred.get("query_id", ""))
        task = task_by_id.get(query_id)
        if task is None:
            continue
        gold = str(task.get("answer", ""))
        predicted = str(pred.get("predicted_answer", ""))
        is_correct = normalize_answer(gold) == normalize_answer(predicted)
        correct += int(is_correct)
        rows.append(
            {
                "query_id": query_id,
                "question": task.get("question", ""),
                "gold_answer": gold,
                "predicted_answer": predicted,
                "eval_judgment": "CORRECT" if is_correct else "INCORRECT",
                "eval_reasoning": "deterministic exact match over synthetic oracle answer",
                "status": pred.get("status", ""),
            }
        )
    summary = {
        "type": "summary",
        "total": len(rows),
        "correct": correct,
        "incorrect": len(rows) - correct,
        "accuracy": (correct / len(rows)) if rows else 0.0,
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic teacher trajectories against synthetic oracle answers.")
    parser.add_argument("--tasks", required=True, help="Synthetic task JSONL path")
    parser.add_argument("--predictions", required=True, help="Teacher submission JSONL path")
    parser.add_argument("--output", required=True, help="Output eval JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, rows = evaluate_predictions(list(read_jsonl(args.tasks)), list(read_jsonl(args.predictions)))
    write_jsonl(args.output, [summary, *rows])
    print(f"accuracy={summary['accuracy']:.4f} correct={summary['correct']} total={summary['total']}")


if __name__ == "__main__":
    main()

