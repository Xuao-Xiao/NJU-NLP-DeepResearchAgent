from __future__ import annotations

import argparse

from .common import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple JSONL files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_jsonl(args.output, (row for path in args.inputs for row in read_jsonl(path)))
    print(f"Wrote {count} merged records to {args.output}")


if __name__ == "__main__":
    main()
