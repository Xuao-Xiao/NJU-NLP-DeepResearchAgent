from __future__ import annotations

import argparse
import random
from pathlib import Path

from .common import read_jsonl, write_jsonl


def split_rows(rows: list[dict], *, seed: int = 13, train_ratio: float = 0.8, dev_ratio: float = 0.1) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get("source_docid") or row.get("source_query_id") or row.get("id"))
        grouped.setdefault(key, []).append(row)
    keys = list(grouped)
    random.Random(seed).shuffle(keys)
    train_cut = int(len(keys) * train_ratio)
    dev_cut = train_cut + int(len(keys) * dev_ratio)
    train_keys = set(keys[:train_cut])
    dev_keys = set(keys[train_cut:dev_cut])
    train: list[dict] = []
    dev: list[dict] = []
    heldout: list[dict] = []
    for key, group in grouped.items():
        if key in train_keys:
            train.extend(group)
        elif key in dev_keys:
            dev.extend(group)
        else:
            heldout.extend(group)
    return train, dev, heldout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split JSONL records into train/dev/heldout by source group.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="sft")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    train, dev, heldout = split_rows(rows, seed=args.seed, train_ratio=args.train_ratio, dev_ratio=args.dev_ratio)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / f"{args.prefix}_train.jsonl", train)
    write_jsonl(output_dir / f"{args.prefix}_dev.jsonl", dev)
    write_jsonl(output_dir / f"{args.prefix}_heldout.jsonl", heldout)
    print(f"train={len(train)} dev={len(dev)} heldout={len(heldout)}")


if __name__ == "__main__":
    main()

