from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any, Iterator

from .common import compact_text, stable_id, write_jsonl


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
KEY_VALUE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_ /-]{1,40}):\s*(.+?)\s*$", re.MULTILINE)
PERSON_HEADING_RE = re.compile(
    r"(?m)^(?P<name>(?:(?:Dr|Prof|Professor)\.?\s+)?[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,3})\s*$"
)

FIELD_PRIORITY = [
    "country",
    "type",
    "established",
    "chancellor",
    "vice_chancellor",
    "city",
    "state",
    "campus_size",
    "budget",
    "students",
    "faculty",
    "birth_date",
    "nationality",
    "occupation",
    "known_for",
]
SKIP_FIELD_KEYS = {
    "title",
    "author",
    "date",
    "image",
    "image_upright",
    "logo",
    "logo_size",
    "website",
}
COUNTRY_NAMES = [
    "Australia",
    "United States",
    "United Kingdom",
    "Canada",
    "Germany",
    "France",
    "Japan",
    "China",
    "India",
    "Italy",
    "Spain",
    "Brazil",
    "South Africa",
    "New Zealand",
    "Netherlands",
    "Switzerland",
]
NON_PERSON_NAME_TERMS = {
    "about",
    "author",
    "books",
    "collaboration",
    "contents",
    "department",
    "editorial",
    "educational",
    "international",
    "journal",
    "national",
    "patterns",
    "references",
    "resources",
    "reviews",
    "school",
    "science",
    "student",
    "university",
}
BAD_PROFILE_ROLE_MARKERS = {"|", "?", "review by", "student page", "about the author", "resources"}


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


def _clean_field_value(key: str, value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.split(r"\b(?:live|retrieved|accessed|pdf)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if key == "country":
        for country in COUNTRY_NAMES:
            if value.startswith(country):
                return country
    if key in {"chancellor", "vice_chancellor", "president", "dean", "director"}:
        value = re.split(
            r"(?=Chancellors\b|Vice-Chancellors\b|Biography\b|Office\b|Profile\b)",
            value,
            maxsplit=1,
        )[0].strip()
        match = re.match(
            r"((?:Dr\.?\s+|Prof\.?\s+|Professor\s+)?[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,3})",
            value,
        )
        if match:
            return match.group(1).strip()
    return value.strip(" .,:;\"'")


def extract_key_value_fields(text: str, limit: int = 8) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in KEY_VALUE_RE.finditer(text):
        key = match.group(1).strip().lower().replace(" ", "_")
        value = _clean_field_value(key, match.group(2))
        if key in seen or key in SKIP_FIELD_KEYS:
            continue
        if not (2 <= len(key) <= 40 and 2 <= len(value) <= 160):
            continue
        if value.startswith(("http://", "https://", "[[", "{{")):
            continue
        fields.append((key, value))
        seen.add(key)
    fields.sort(key=lambda item: FIELD_PRIORITY.index(item[0]) if item[0] in FIELD_PRIORITY else len(FIELD_PRIORITY))
    return fields[:limit]


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").replace("-", " ").strip()


def _nonempty_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]


def _looks_like_person_heading(name: str) -> bool:
    if not re.match(r"^(?:Dr|Prof|Professor)\.?\s+", name):
        return False
    lowered_parts = {part.lower().strip(".,:-") for part in name.split()}
    if lowered_parts & NON_PERSON_NAME_TERMS:
        return False
    if name.isupper():
        return False
    parts = [part for part in name.replace(".", " ").split() if part.lower() not in {"dr", "prof", "professor"}]
    return 2 <= len(parts) <= 5 and all(part[:1].isupper() for part in parts)


def _looks_like_profile_role(role: str) -> bool:
    lowered = role.lower()
    if any(marker in lowered for marker in BAD_PROFILE_ROLE_MARKERS):
        return False
    if role.startswith(("-", "|")):
        return False
    return 4 <= len(role) <= 90


def extract_profile_person_tasks(docid: str, url: str, text: str, title: str, *, limit: int = 3) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    matches = list(PERSON_HEADING_RE.finditer(text))
    for idx, match in enumerate(matches):
        name = match.group("name").strip()
        if not _looks_like_person_heading(name):
            continue
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(text), match.end() + 900)
        block = text[match.end():block_end]
        lines = _nonempty_lines(block)
        if not lines:
            continue
        role = next((line for line in lines if _looks_like_profile_role(line) and not line.startswith("(")), "")
        if not role:
            continue
        clue = ""
        for line in lines[1:]:
            lowered = line.lower()
            if any(term in lowered for term in ["earned", "received", "worked", "research", "teaches", "prior work", "degree"]):
                clue = line
                break
        question = f"In the document {title or 'docid ' + docid}, which person is described as {role}?"
        if clue:
            question = f"{question} The same profile mentions: {compact_text(clue, max_chars=160)}"
        tasks.append(
            _task(
                docid=docid,
                url=url,
                task_type="profile_person",
                question=question,
                answer=name,
                evidence=compact_text(f"{name}\n{block}", max_chars=1200),
            )
        )
        if len(tasks) >= limit:
            break
    return tasks


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
    for key, value in extract_key_value_fields(text):
        clue = title or f"docid {docid}"
        tasks.append(
            _task(
                docid=docid,
                url=url,
                task_type="infobox_field",
                question=f"In the document {clue}, what is listed as the {_humanize_key(key)}?",
                answer=value,
                evidence=evidence,
            )
        )
        if len(tasks) >= max_tasks_per_doc:
            break
    if len(tasks) < max_tasks_per_doc:
        tasks.extend(
            extract_profile_person_tasks(
                docid=docid,
                url=url,
                text=text,
                title=title,
                limit=max_tasks_per_doc - len(tasks),
            )
        )
    for heading in extract_headings(text):
        if len(tasks) >= max_tasks_per_doc:
            break
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
