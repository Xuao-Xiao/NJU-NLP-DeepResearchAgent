from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Iterator

from .common import read_jsonl, stable_id, write_jsonl


def parse_boosts(items: list[str]) -> dict[str, int]:
    boosts: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Boost must use pattern=value format: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = int(raw_value)
        if not key or value < 1:
            raise ValueError(f"Invalid boost: {item}")
        boosts[key] = value
    return boosts


def _copy_with_id(row: dict[str, Any], copy_index: int) -> dict[str, Any]:
    if copy_index == 0:
        return dict(row)
    copied = dict(row)
    copied["id"] = stable_id(row.get("id", ""), copy_index, prefix=str(row.get("id", "sft")))
    copied["reweight_copy"] = copy_index
    return copied


def _multiplier(
    row: dict[str, Any],
    source_boosts: dict[str, int],
    task_boosts: dict[str, int],
    query_id_boosts: dict[str, int],
) -> int:
    multiplier = 1
    source = str(row.get("source", ""))
    task_type = str(row.get("task_type", ""))
    source_query_id = str(row.get("source_query_id", ""))
    for needle, value in source_boosts.items():
        if needle in source:
            multiplier = max(multiplier, value)
    if task_type in task_boosts:
        multiplier = max(multiplier, task_boosts[task_type])
    if source_query_id in query_id_boosts:
        multiplier = max(multiplier, query_id_boosts[source_query_id])
    return multiplier


def reweight_records(
    rows: Iterable[dict[str, Any]],
    *,
    source_boosts: dict[str, int] | None = None,
    task_boosts: dict[str, int] | None = None,
    query_id_boosts: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    source_boosts = source_boosts or {}
    task_boosts = task_boosts or {}
    query_id_boosts = query_id_boosts or {}
    for row in rows:
        for copy_index in range(_multiplier(row, source_boosts, task_boosts, query_id_boosts)):
            yield _copy_with_id(row, copy_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reweight SFT JSONL records by duplicating high-value sources/tasks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--boost-source",
        action="append",
        default=[],
        help="Substring multiplier, for example OTEasyRun0.11_22=4",
    )
    parser.add_argument(
        "--boost-task",
        action="append",
        default=[],
        help="Task multiplier, for example final_answer=2",
    )
    parser.add_argument(
        "--boost-query-id",
        action="append",
        default=[],
        help="source_query_id multiplier, for example 556=8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_jsonl(
        args.output,
        reweight_records(
            read_jsonl(args.input),
            source_boosts=parse_boosts(args.boost_source),
            task_boosts=parse_boosts(args.boost_task),
            query_id_boosts=parse_boosts(args.boost_query_id),
        ),
    )
    print(f"Wrote {count} reweighted records to {args.output}")


if __name__ == "__main__":
    main()
