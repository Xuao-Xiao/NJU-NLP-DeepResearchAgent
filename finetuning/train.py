from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    model_path: str
    train_file: str
    output_dir: str
    dev_file: str | None = None
    max_length: int = 4096
    learning_rate: float = 1e-4
    epochs: float = 2.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    merge_output_dir: str | None = None


def load_config_file(path: str | Path) -> TrainConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = set(TrainConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown train config keys: {', '.join(unknown)}")
    return TrainConfig(**raw)


class ChatSFTDataset:
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int) -> None:
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        messages = row["messages"]
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Row {row.get('id', index)} must end with an assistant message.")
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_text = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        prefix = self.tokenizer(prefix_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        input_ids = list(full["input_ids"])
        labels = list(input_ids)
        prefix_len = min(len(prefix["input_ids"]), len(labels))
        labels[:prefix_len] = [-100] * prefix_len
        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


class DataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="LoRA SFT training entry for Qwen3-8B OpenTrack fine-tuning.")
    parser.add_argument("--config", default=None, help="Optional JSON config file. CLI flags override config values.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--merge-output-dir", default=None)
    # Parse once with relaxed required fields so --config can provide required values.
    for action in parser._actions:
        if action.dest in {"model_path", "train_file", "output_dir"}:
            action.required = False
    args = parser.parse_args()
    config = load_config_file(args.config) if args.config else None
    values = config.__dict__.copy() if config else {}
    cli_values = {
        "model_path": args.model_path,
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "bf16": not args.no_bf16,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "merge_output_dir": args.merge_output_dir,
    }
    defaults = TrainConfig(model_path="", train_file="", output_dir="")
    for key, value in cli_values.items():
        if value is None:
            continue
        if getattr(defaults, key) != value or key not in values:
            values[key] = value
    missing = [key for key in ("model_path", "train_file", "output_dir") if not values.get(key)]
    if missing:
        parser.error(f"missing required arguments: {', '.join('--' + key.replace('_', '-') for key in missing)}")
    return TrainConfig(**values)


def _training_arguments(config: TrainConfig, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "bf16": config.bf16,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = config.eval_steps
    return TrainingArguments(**kwargs)


def main() -> None:
    config = parse_args()
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
    except ImportError as exc:
        raise SystemExit(
            "Training requires torch, transformers, peft, and their Ascend-compatible runtime on the cloud server."
        ) from exc

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if config.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[module.strip() for module in config.target_modules.split(",") if module.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatSFTDataset(config.train_file, tokenizer=tokenizer, max_length=config.max_length)
    eval_dataset = ChatSFTDataset(config.dev_file, tokenizer=tokenizer, max_length=config.max_length) if config.dev_file else None
    trainer = Trainer(
        model=model,
        args=_training_arguments(config, has_eval=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if config.merge_output_dir:
        merged = model.merge_and_unload()
        merged.save_pretrained(config.merge_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(config.merge_output_dir)
        print(f"Merged model saved to {config.merge_output_dir}")


if __name__ == "__main__":
    main()
