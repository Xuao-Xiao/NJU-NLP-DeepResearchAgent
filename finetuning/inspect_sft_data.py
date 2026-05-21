from __future__ import annotations

import argparse
from collections import Counter

from .common import read_jsonl
from .filter_sft_data import rejection_reason


def inspect_file(path: str, *, sample_count: int = 3) -> dict:
    task_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    total = 0
    examples = []
    for row in read_jsonl(path):
        total += 1
        task_counts[str(row.get("task_type", ""))] += 1
        reason = rejection_reason(row)
        if reason:
            rejection_counts[reason] += 1
        elif len(examples) < sample_count:
            examples.append(row)
    return {
        "total": total,
        "task_counts": dict(task_counts),
        "rejection_counts": dict(rejection_counts),
        "sample_ids": [row.get("id") for row in examples],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SFT JSONL quality and task distribution.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_file(args.input, sample_count=args.sample_count)
    print(f"Total records: {report['total']}")
    print(f"Task counts: {report['task_counts']}")
    print(f"Rejection counts: {report['rejection_counts']}")
    print(f"Sample ids: {report['sample_ids']}")


if __name__ == "__main__":
    main()

