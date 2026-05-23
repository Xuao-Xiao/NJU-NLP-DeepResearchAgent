import json
import tempfile
import unittest
from pathlib import Path

from finetuning.filter_sft_data import filter_records
from finetuning.evaluate_synthetic import evaluate_predictions
from finetuning.synthetic_tasks import build_tasks_from_document
from finetuning.train import TrainConfig, load_config_file
from finetuning.trajectory_sft import collect_success_ids, extract_sft_records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


class FineTuningPipelineTests(unittest.TestCase):
    def test_collect_success_ids_skips_summary_and_keeps_correct_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = Path(tmpdir) / "eval.jsonl"
            write_jsonl(
                eval_path,
                [
                    {"type": "summary", "correct": 1},
                    {"query_id": "5", "eval_judgment": "CORRECT"},
                    {"query_id": "6", "eval_judgment": "INCORRECT"},
                ],
            )

            self.assertEqual(collect_success_ids(eval_path), {"5"})

    def test_extract_sft_records_turns_tool_call_into_action_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            submission_path = Path(tmpdir) / "submission.jsonl"
            eval_path = Path(tmpdir) / "eval.jsonl"
            write_jsonl(eval_path, [{"query_id": "5", "eval_judgment": "CORRECT"}])
            write_jsonl(
                submission_path,
                [
                    {
                        "query_id": "5",
                        "predicted_answer": "Spero Therapeutics",
                        "messages": [
                            {"role": "system", "content": "System prompt"},
                            {"role": "user", "content": "Who is the company?"},
                            {
                                "role": "assistant",
                                "content": "Planning: search for company evidence.",
                                "state_summary": "Question: Who is the company?\nKnown facts:\nNone",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": "{\"query\":\"Spero Therapeutics FDA\"}",
                                        },
                                    }
                                ],
                            },
                            {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
                        ],
                    }
                ],
            )

            records = list(extract_sft_records(submission_path, eval_path=eval_path))

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["task_type"], "query_rewrite")
            self.assertEqual(record["source_query_id"], "5")
            self.assertEqual(record["messages"][-1]["role"], "assistant")
            self.assertEqual(
                json.loads(record["messages"][-1]["content"]),
                {
                    "action": "search",
                    "query": "Spero Therapeutics FDA",
                    "reason": "Planning: search for company evidence.",
                },
            )

    def test_extract_sft_records_emits_finish_and_final_answer_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            submission_path = Path(tmpdir) / "submission.jsonl"
            eval_path = Path(tmpdir) / "eval.jsonl"
            write_jsonl(eval_path, [{"query_id": "5", "eval_judgment": "CORRECT"}])
            state = {
                "question": "Who is the cover designer?",
                "question_plan": {
                    "answer_type": "person",
                    "keywords": ["cover designer"],
                },
                "confirmed_facts": ["Cristina Ortiz is the cover designer."],
                "candidate_answers": ["Cristina Ortiz"],
                "opened_passages": ["Cristina Ortiz Graphic Designer"],
                "search_evidence": [],
                "verification_results": [
                    {
                        "candidate_answer": "Cristina Ortiz",
                        "supported": True,
                        "support_score": 0.9,
                    }
                ],
                "candidate_records": [],
            }
            write_jsonl(
                submission_path,
                [
                    {
                        "query_id": "5",
                        "predicted_answer": "Cristina Ortiz",
                        "agent_state": state,
                        "messages": [
                            {"role": "system", "content": "System prompt"},
                            {"role": "user", "content": "Who is the cover designer?"},
                            {
                                "role": "assistant",
                                "content": "Heuristic policy selected finish after candidate verification.",
                                "state_summary": "Question: Who is the cover designer?\nVerifier results:\nCristina Ortiz supported=True",
                                "action_plan": {
                                    "action": "finish",
                                    "answer_hint": "Cristina Ortiz",
                                    "reason": "Opened evidence supports the candidate.",
                                },
                            },
                            {
                                "role": "assistant",
                                "content": "{\"exact_answer\":\"Cristina Ortiz\",\"confidence\":0.9,\"support\":\"Verified by opened evidence.\"}",
                                "state_summary": "Question: Who is the cover designer?",
                                "finish_reason": "model_or_fallback_finish",
                            },
                        ],
                    }
                ],
            )

            records = list(
                extract_sft_records(
                    submission_path,
                    eval_path=eval_path,
                    include_final_answer=True,
                )
            )

            self.assertEqual([record["task_type"] for record in records], ["finish_decision", "final_answer"])
            self.assertEqual(
                json.loads(records[0]["messages"][-1]["content"]),
                {
                    "action": "finish",
                    "answer_hint": "Cristina Ortiz",
                    "reason": "Opened evidence supports the candidate.",
                },
            )
            self.assertEqual(
                json.loads(records[1]["messages"][-1]["content"])["exact_answer"],
                "Cristina Ortiz",
            )

    def test_filter_records_rejects_invalid_json_action_and_think_output(self) -> None:
        rows = [
            {
                "id": "ok",
                "task_type": "query_rewrite",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "{\"action\":\"search\",\"query\":\"abc\"}"},
                ],
            },
            {
                "id": "bad-json",
                "task_type": "query_rewrite",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "search abc"},
                ],
            },
            {
                "id": "think",
                "task_type": "final_answer",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "<think>hidden</think>Answer"},
                ],
            },
        ]

        kept, rejected = filter_records(rows)

        self.assertEqual([row["id"] for row in kept], ["ok"])
        self.assertEqual({row["id"] for row, _ in rejected}, {"bad-json", "think"})

    def test_build_tasks_from_document_creates_metadata_and_heading_tasks(self) -> None:
        text = """---
title: Example Book
author: Ada Lovelace
date: 1901-01-01
---
# Introduction

The first chapter describes analytical engines.
"""

        tasks = build_tasks_from_document(docid="doc-1", url="https://example.test", text=text)

        task_types = {task["task_type"] for task in tasks}
        self.assertIn("metadata_title", task_types)
        self.assertIn("metadata_author", task_types)
        self.assertIn("heading_title", task_types)
        self.assertTrue(all(task["source_docid"] == "doc-1" for task in tasks))

    def test_evaluate_predictions_matches_synthetic_answers_by_query_id(self) -> None:
        tasks = [
            {"id": "task-1", "answer": "Example Book"},
            {"id": "task-2", "answer": "Ada Lovelace"},
        ]
        predictions = [
            {"query_id": "task-1", "predicted_answer": "example book"},
            {"query_id": "task-2", "predicted_answer": "Charles Babbage"},
        ]

        summary, rows = evaluate_predictions(tasks, predictions)

        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(rows[0]["eval_judgment"], "CORRECT")
        self.assertEqual(rows[1]["eval_judgment"], "INCORRECT")

    def test_load_config_file_keeps_train_defaults_and_overrides_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "./Qwen3-8B",
                        "train_file": "train.jsonl",
                        "output_dir": "out",
                        "learning_rate": 5e-5,
                    }
                ),
                encoding="utf-8",
            )

            config = load_config_file(config_path)

            self.assertIsInstance(config, TrainConfig)
            self.assertEqual(config.learning_rate, 5e-5)
            self.assertEqual(config.lora_rank, 16)


if __name__ == "__main__":
    unittest.main()
