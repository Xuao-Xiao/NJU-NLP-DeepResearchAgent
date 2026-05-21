from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any, Iterator

from .common import compact_text, stable_id, write_jsonl


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


def parse_front_matter(text: str) -> dict[str, str]:
    match = FRONT_MATTER_RE.search(text)
    if not match:
        return {}
    metadata: dict[str, str] = {}
    for line in match.group("body").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip().strip('"')
    return metadata


def extract_headings(text: str, limit: int = 8) -> list[str]:
    headings: list[str] = []
    for match in HEADING_RE.finditer(text):
        heading = match.group(2).strip()
        if 4 <= len(heading) <= 120 and heading.lower() not in {"contents", "table of contents"}:
            headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def _task(
    *,
    docid: str,
    url: str,
    task_type: str,
    question: str,
    answer: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "id": stable_id(docid, task_type, question, answer, prefix="synthetic"),
        "task_type": task_type,
        "source_docid": str(docid),
        "source_url": str(url or ""),
        "question": question,
        "answer": answer,
        "evidence": compact_text(evidence, max_chars=1200),
        "messages": [
            {
                "role": "system",
                "content": "Answer using only evidence from the fixed offline corpus. Return a short answer.",
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def build_tasks_from_document(docid: str, url: str, text: str, *, max_tasks_per_doc: int = 4) -> list[dict[str, Any]]:
    metadata = parse_front_matter(text)
    title = metadata.get("title", "").strip()
    author = metadata.get("author", "").strip()
    date = metadata.get("date", "").strip()
    tasks: list[dict[str, Any]] = []
    evidence = compact_text(text, max_chars=1200)
    if title:
        tasks.append(
            _task(
                docid=docid,
                url=url,
                task_type="metadata_title",
                question=f"What is the title of the document with docid {docid}?",
                answer=title,
                evidence=evidence,
            )
        )
    if author:
        clue = f" titled {title}" if title else f" with docid {docid}"
        tasks.append(
            _task(
                docid=docid,
                url=url,
                task_type="metadata_author",
                question=f"Who is listed as the author of the document{clue}?",
                answer=author,
                evidence=evidence,
            )
        )
    if date:
        clue = title or docid
        tasks.append(
            _task(
                docid=docid,
                url=url,
                task_type="metadata_date",
                question=f"What date is listed in the metadata for {clue}?",
                answer=date,
                evidence=evidence,
            )
        )
    for heading in extract_headings(text):
        clue = title or f"docid {docid}"
        tasks.append(
            _task(
                docid=docid,
                url=url,
                task_type="heading_title",
                question=f"In the document {clue}, what is one section heading shown in the text?",
                answer=heading,
                evidence=evidence,
            )
        )
        if len(tasks) >= max_tasks_per_doc:
            break
    return tasks[:max_tasks_per_doc]


def iter_corpus_documents(corpus_path: str | Path, *, limit_docs: int | None = None) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to stream BrowseComp-Plus parquet files.") from exc

    base = Path(corpus_path)
    data_dir = base / "data" if (base / "data").exists() else base
    seen = 0
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=64, columns=["docid", "text", "url"]):
            for row in batch.to_pylist():
                yield row
                seen += 1
                if limit_docs is not None and seen >= limit_docs:
                    return


def generate_synthetic_tasks(
    corpus_path: str | Path,
    *,
    limit_docs: int | None = None,
    max_tasks: int | None = None,
    max_tasks_per_doc: int = 4,
    seed: int = 13,
) -> Iterator[dict[str, Any]]:
    random.seed(seed)
    emitted = 0
    for doc in iter_corpus_documents(corpus_path, limit_docs=limit_docs):
        for task in build_tasks_from_document(
            docid=str(doc.get("docid", "")),
            url=str(doc.get("url", "")),
            text=str(doc.get("text", "")),
            max_tasks_per_doc=max_tasks_per_doc,
        ):
            yield task
            emitted += 1
            if max_tasks is not None and emitted >= max_tasks:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic non-test tasks from BrowseComp-Plus corpus parquet files.")
    parser.add_argument("--corpus-path", required=True, help="Path to browsecomp-plus-corpus or its data directory")
    parser.add_argument("--output", required=True, help="Output synthetic task JSONL path")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-tasks-per-doc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_jsonl(
        args.output,
        generate_synthetic_tasks(
            args.corpus_path,
            limit_docs=args.limit_docs,
            max_tasks=args.max_tasks,
            max_tasks_per_doc=args.max_tasks_per_doc,
            seed=args.seed,
        ),
    )
    print(f"Wrote {count} synthetic tasks to {args.output}")


if __name__ == "__main__":
    main()

