# OpenTrack 微调流水线

本目录包含 OpenTrack 模型训练阶段的微调链路。它与 `agent/multistep_agent.py` 有意保持分离；训练完成后，原有的 notebook/eval 入口保持不变。

大型 JSONL 数据集、LoRA 适配器、合并后的检查点以及日志应存放在云服务器上的 `open_track_finetune/` 目录下。这些文件被 Git 忽略。

## 1. 从已有轨迹中提取 SFT 样本

仅用于模式校验、冒烟测试和回归保护。目前硬编码的50条成功轨迹数量太少，不足以构成真正的训练集。

```bash
python -m finetuning.trajectory_sft \
  --submission OTEasyRun0.11_22/multistep_submission.jsonl \
  --eval OTEasyRun0.11_22/multistep_eval_results.jsonl \
  --output open_track_finetune/datasets/oteasy_success_sft.jsonl

python -m finetuning.filter_sft_data \
  --input open_track_finetune/datasets/oteasy_success_sft.jsonl \
  --output open_track_finetune/datasets/oteasy_success_sft.filtered.jsonl \
  --rejected-output open_track_finetune/datasets/oteasy_success_sft.rejected.jsonl
```

## 2. 生成合成的非测试任务

在云服务器上运行，这样生成的数据集无需推送到 GitHub。

```bash
python -m finetuning.synthetic_tasks \
  --corpus-path browsecomp-plus-corpus \
  --output open_track_finetune/datasets/synthetic_tasks.jsonl \
  --max-tasks 1500 \
  --max-tasks-per-doc 3
```

## 3. 运行教师智能体轨迹

先启动基础 `Qwen3-8B` vLLM 服务，然后运行：

```bash
python -m finetuning.run_teacher_agent \
  --tasks open_track_finetune/datasets/synthetic_tasks.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --base-url http://127.0.0.1:8000/v1 \
  --model-name qwen_auto \
  --output open_track_finetune/raw_runs/synthetic_teacher_submission.jsonl
```

用合成的标准答案评估合成预测结果：

```bash
python -m finetuning.evaluate_synthetic \
  --tasks open_track_finetune/datasets/synthetic_tasks.jsonl \
  --predictions open_track_finetune/raw_runs/synthetic_teacher_submission.jsonl \
  --output open_track_finetune/raw_runs/synthetic_teacher_eval.jsonl
```

## 4. 构建过滤后的 SFT 数据

```bash
python -m finetuning.trajectory_sft \
  --submission open_track_finetune/raw_runs/synthetic_teacher_submission.jsonl \
  --eval open_track_finetune/raw_runs/synthetic_teacher_eval.jsonl \
  --output open_track_finetune/datasets/synthetic_teacher_sft.jsonl

python -m finetuning.merge_jsonl \
  --inputs open_track_finetune/datasets/synthetic_teacher_sft.jsonl open_track_finetune/datasets/oteasy_success_sft.filtered.jsonl \
  --output open_track_finetune/datasets/sft_all.raw.jsonl

python -m finetuning.filter_sft_data \
  --input open_track_finetune/datasets/sft_all.raw.jsonl \
  --output open_track_finetune/datasets/sft_all.filtered.jsonl \
  --rejected-output open_track_finetune/datasets/sft_all.rejected.jsonl

python -m finetuning.split_jsonl \
  --input open_track_finetune/datasets/sft_all.filtered.jsonl \
  --output-dir open_track_finetune/datasets \
  --prefix sft

python -m finetuning.inspect_sft_data \
  --input open_track_finetune/datasets/sft_train.jsonl
```

在第一次正式训练运行之前，将 `sft_all.filtered.jsonl` 或至少 `inspect_sft_data` 的输出发回进行质量审查。

## 5. LoRA SFT

Python 训练器是一个轻量级的 Hugging Face PEFT 入口。在华为昇腾上，需要先安装匹配的 CANN / `torch_npu` / PyTorch 技术栈。如果云镜像上已经配置了 `ms-swift` 或 `LLaMA-Factory`，也可以使用它们的 CLI 配合相同的 JSONL 数据；保留此 `train.py` 作为项目自有的回退方案和参考实现。

```bash
python -m finetuning.train \
  --model-path ./Qwen3-8B \
  --train-file open_track_finetune/datasets/sft_train.jsonl \
  --dev-file open_track_finetune/datasets/sft_dev.jsonl \
  --output-dir open_track_finetune/outputs/qwen3_8b_lora_sft_v1 \
  --max-length 4096 \
  --learning-rate 1e-4 \
  --epochs 2 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

可选的合并检查点导出：

```bash
python -m finetuning.train \
  --model-path ./Qwen3-8B \
  --train-file open_track_finetune/datasets/sft_train.jsonl \
  --dev-file open_track_finetune/datasets/sft_dev.jsonl \
  --output-dir open_track_finetune/outputs/qwen3_8b_lora_sft_v1 \
  --merge-output-dir open_track_finetune/outputs/qwen3_8b_lora_sft_v1_merged
```

训练完成后，使用适配器或合并后的检查点重启模型服务，然后按原有方式运行 `agent_vllm_multistep.ipynb` 和 `agent/eval.py`。
