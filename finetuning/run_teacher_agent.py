from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_jsonl, write_jsonl


def tasks_to_agent_rows(tasks_path: str | Path, *, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for task in read_jsonl(tasks_path):
        rows.append(
            {
                "query_id": str(task.get("id")),
                "query": str(task.get("question", "")),
                "synthetic_answer": str(task.get("answer", "")),
                "source_docid": str(task.get("source_docid", "")),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the existing multistep agent over synthetic tasks to produce teacher trajectories.")
    parser.add_argument("--tasks", required=True, help="Synthetic task JSONL path")
    parser.add_argument("--index-path", required=True, help="BM25 sqlite index path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", default="qwen_auto")
    parser.add_argument("--output", required=True, help="Output teacher submission JSONL path")
    parser.add_argument("--agent-input-output", default=None, help="Optional converted agent input JSONL path")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=7)
    parser.add_argument("--decision-max-tokens", type=int, default=512)
    parser.add_argument("--answer-max-tokens", type=int, default=768)
    parser.add_argument("--search-snippet-max-chars", type=int, default=1200)
    parser.add_argument("--tool-content-max-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    from agent.multistep_agent import generate_submission

    args = parse_args()
    rows = tasks_to_agent_rows(args.tasks, limit=args.limit)
    if args.agent_input_output:
        write_jsonl(args.agent_input_output, rows)
    generate_submission(
        dataset_rows=rows,
        index_path=args.index_path,
        base_url=args.base_url,
        model_name=args.model_name,
        output_path=args.output,
        api_key=args.api_key,
        top_k=args.top_k,
        search_snippet_max_chars=args.search_snippet_max_chars,
        tool_content_max_chars=args.tool_content_max_chars,
        max_rounds=args.max_rounds,
        decision_max_tokens=args.decision_max_tokens,
        answer_max_tokens=args.answer_max_tokens,
    )


if __name__ == "__main__":
    main()

