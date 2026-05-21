# OpenTrack 微调全流程跟做指南

本文档是云端执行手册。目标是从“已有轨迹 + BrowseComp-Plus 离线语料”构造 SFT 数据，跑 teacher trajectories，过滤数据，最后用 `Qwen3-8B` 做 LoRA SFT。

先说结论：你不是只 push `finetuning/` 就够了。建议至少同步这些文件：

```bash
git add .gitignore finetuning tests Documents/OpenTrack-FineTuning.md Documents/Menu.md
git commit -m "add OpenTrack fine-tuning pipeline"
git push origin main
```

其中：

- `finetuning/` 是微调代码和说明。
- `.gitignore` 防止大数据、checkpoint、训练日志误传 GitHub。
- `tests/` 用于在云端快速确认脚本没坏。
- `Documents/*.md` 是更新后的设计和执行说明，可选但建议一起同步。

不要提交这些内容：

- `open_track_finetune/`
- `finetuning/data/`
- `finetuning/outputs/`
- `finetuning/runs/`
- 任何大 `.jsonl` 数据集
- 任何 LoRA adapter、merged checkpoint、训练日志

这些大文件只保存在云端。

## 0. 云端同步代码

在云端项目目录执行：

```bash
git pull origin main
```

确认 `finetuning/` 已存在：

```bash
ls finetuning
```

建议先跑本地单元测试：

```bash
python -m unittest tests.test_finetuning_pipeline -v
```

预期看到：

```text
Ran 6 tests
OK
```

如果这里失败，先不要继续生成大数据。

## 1. 建立云端工作目录

所有大文件都放在云端 `open_track_finetune/`，不要放进 Git。

```bash
mkdir -p open_track_finetune/datasets
mkdir -p open_track_finetune/raw_runs
mkdir -p open_track_finetune/outputs
mkdir -p open_track_finetune/eval_runs
```

确认 `.gitignore` 生效：

```bash
git check-ignore open_track_finetune/datasets/test.jsonl
```

预期输出：

```text
open_track_finetune/datasets/test.jsonl
```

## 2. 先从现有成功轨迹抽取小样本

这一步只用于验证 schema、过滤器和训练格式，不代表正式训练数据足够。

```bash
python -m finetuning.trajectory_sft \
  --submission OTEasyRun0.11_22/multistep_submission.jsonl \
  --eval OTEasyRun0.11_22/multistep_eval_results.jsonl \
  --output open_track_finetune/datasets/oteasy_success_sft.jsonl
```

过滤：

```bash
python -m finetuning.filter_sft_data \
  --input open_track_finetune/datasets/oteasy_success_sft.jsonl \
  --output open_track_finetune/datasets/oteasy_success_sft.filtered.jsonl \
  --rejected-output open_track_finetune/datasets/oteasy_success_sft.rejected.jsonl
```

检查：

```bash
python -m finetuning.inspect_sft_data \
  --input open_track_finetune/datasets/oteasy_success_sft.filtered.jsonl
```

本地 smoke 结果曾经是：抽取 `115` 条，过滤后保留 `107` 条。云端结果允许略有差异，但如果保留数是 `0`，不要继续训练。

## 3. 生成少量合成任务做 smoke test

先小规模跑通，不要一上来生成上千条。

```bash
python -m finetuning.synthetic_tasks \
  --corpus-path browsecomp-plus-corpus \
  --output open_track_finetune/datasets/synthetic_tasks_smoke.jsonl \
  --limit-docs 50 \
  --max-tasks 30 \
  --max-tasks-per-doc 3
```

看前两条：

```bash
head -n 2 open_track_finetune/datasets/synthetic_tasks_smoke.jsonl
```

每条应包含：

- `id`
- `task_type`
- `source_docid`
- `question`
- `answer`
- `evidence`
- `messages`

## 4. 启动 base 模型服务

另开一个终端启动原始 `Qwen3-8B` 服务。示例：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

如果你在 Ascend 环境中使用的是 vLLM Ascend、MindIE 或平台自带启动方式，以你当前已经跑通 `agent_vllm_multistep.ipynb` 的方式为准。关键是保持：

```text
base-url: http://127.0.0.1:8000/v1
model-name: qwen_auto
```

## 5. 跑少量 teacher trajectories

先用第 3 步的小任务跑 teacher，确认 agent 入口正常。

```bash
python -m finetuning.run_teacher_agent \
  --tasks open_track_finetune/datasets/synthetic_tasks_smoke.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --base-url http://127.0.0.1:8000/v1 \
  --model-name qwen_auto \
  --output open_track_finetune/raw_runs/synthetic_teacher_smoke_submission.jsonl \
  --limit 10
```

如果这里报 index 不存在，先构建 BM25：

```bash
python -m agent.build_bm25_index \
  --corpus-path ./browsecomp-plus-corpus \
  --index-path ./indexes/browsecomp_plus_bm25.sqlite \
  --overwrite
```

## 6. 评测 teacher smoke 轨迹

```bash
python -m finetuning.evaluate_synthetic \
  --tasks open_track_finetune/datasets/synthetic_tasks_smoke.jsonl \
  --predictions open_track_finetune/raw_runs/synthetic_teacher_smoke_submission.jsonl \
  --output open_track_finetune/raw_runs/synthetic_teacher_smoke_eval.jsonl
```

看结果：

```bash
head -n 1 open_track_finetune/raw_runs/synthetic_teacher_smoke_eval.jsonl
```

这一步的准确率不一定很高，因为合成问题比较直接但 agent 仍按完整检索流程跑。重点是确认：

- 文件正常生成。
- `query_id` 能和 synthetic task 的 `id` 对上。
- eval 文件第一行有 summary。

## 7. 从 teacher smoke 轨迹抽 SFT 并过滤

```bash
python -m finetuning.trajectory_sft \
  --submission open_track_finetune/raw_runs/synthetic_teacher_smoke_submission.jsonl \
  --eval open_track_finetune/raw_runs/synthetic_teacher_smoke_eval.jsonl \
  --output open_track_finetune/datasets/synthetic_teacher_smoke_sft.jsonl
```

过滤：

```bash
python -m finetuning.filter_sft_data \
  --input open_track_finetune/datasets/synthetic_teacher_smoke_sft.jsonl \
  --output open_track_finetune/datasets/synthetic_teacher_smoke_sft.filtered.jsonl \
  --rejected-output open_track_finetune/datasets/synthetic_teacher_smoke_sft.rejected.jsonl
```

检查：

```bash
python -m finetuning.inspect_sft_data \
  --input open_track_finetune/datasets/synthetic_teacher_smoke_sft.filtered.jsonl
```

如果这里保留样本数正常，说明数据链路跑通。

## 8. 生成正式合成任务

确认 smoke 没问题后，再生成正式数据。第一版建议不要太大，先 `500-1500` 条任务。

```bash
python -m finetuning.synthetic_tasks \
  --corpus-path browsecomp-plus-corpus \
  --output open_track_finetune/datasets/synthetic_tasks.jsonl \
  --max-tasks 1500 \
  --max-tasks-per-doc 3
```

如果时间不够，先用：

```bash
python -m finetuning.synthetic_tasks \
  --corpus-path browsecomp-plus-corpus \
  --output open_track_finetune/datasets/synthetic_tasks.jsonl \
  --max-tasks 500 \
  --max-tasks-per-doc 3
```

## 9. 跑正式 teacher trajectories

这一步会调用模型和 BM25，耗时会比较长。

```bash
python -m finetuning.run_teacher_agent \
  --tasks open_track_finetune/datasets/synthetic_tasks.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --base-url http://127.0.0.1:8000/v1 \
  --model-name qwen_auto \
  --output open_track_finetune/raw_runs/synthetic_teacher_submission.jsonl
```

如果你想先跑前 100 条确认速度：

```bash
python -m finetuning.run_teacher_agent \
  --tasks open_track_finetune/datasets/synthetic_tasks.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --base-url http://127.0.0.1:8000/v1 \
  --model-name qwen_auto \
  --output open_track_finetune/raw_runs/synthetic_teacher_100_submission.jsonl \
  --limit 100
```

## 10. 评测正式 teacher trajectories

```bash
python -m finetuning.evaluate_synthetic \
  --tasks open_track_finetune/datasets/synthetic_tasks.jsonl \
  --predictions open_track_finetune/raw_runs/synthetic_teacher_submission.jsonl \
  --output open_track_finetune/raw_runs/synthetic_teacher_eval.jsonl
```

查看 summary：

```bash
head -n 1 open_track_finetune/raw_runs/synthetic_teacher_eval.jsonl
```

## 11. 构造正式 SFT 数据

从 teacher trajectories 抽取 SFT：

```bash
python -m finetuning.trajectory_sft \
  --submission open_track_finetune/raw_runs/synthetic_teacher_submission.jsonl \
  --eval open_track_finetune/raw_runs/synthetic_teacher_eval.jsonl \
  --output open_track_finetune/datasets/synthetic_teacher_sft.jsonl
```

合并 synthetic SFT 和现有成功轨迹小样本：

```bash
python -m finetuning.merge_jsonl \
  --inputs \
    open_track_finetune/datasets/synthetic_teacher_sft.jsonl \
    open_track_finetune/datasets/oteasy_success_sft.filtered.jsonl \
  --output open_track_finetune/datasets/sft_all.raw.jsonl
```

过滤：

```bash
python -m finetuning.filter_sft_data \
  --input open_track_finetune/datasets/sft_all.raw.jsonl \
  --output open_track_finetune/datasets/sft_all.filtered.jsonl \
  --rejected-output open_track_finetune/datasets/sft_all.rejected.jsonl
```

切分：

```bash
python -m finetuning.split_jsonl \
  --input open_track_finetune/datasets/sft_all.filtered.jsonl \
  --output-dir open_track_finetune/datasets \
  --prefix sft
```

检查：

```bash
python -m finetuning.inspect_sft_data \
  --input open_track_finetune/datasets/sft_train.jsonl
```

## 12. 第一次训练前必须做的数据质量确认

第一次真实训练前，建议先把以下输出发回本地让我检查：

```bash
python -m finetuning.inspect_sft_data \
  --input open_track_finetune/datasets/sft_all.filtered.jsonl
```

同时回传下面两个文件之一：

1. 首选：`open_track_finetune/datasets/sft_all.filtered.jsonl`
2. 如果文件太大：随机抽样 100 条

抽样命令：

```bash
python - <<'PY'
import json, random
from pathlib import Path

src = Path("open_track_finetune/datasets/sft_all.filtered.jsonl")
dst = Path("open_track_finetune/datasets/sft_all.sample100.jsonl")
rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
random.seed(13)
sample = random.sample(rows, min(100, len(rows)))
dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
print(f"wrote {len(sample)} rows to {dst}")
PY
```

然后把 `sft_all.sample100.jsonl` 和 `inspect_sft_data` 输出发回来。

## 13. 安装训练依赖

`train.py` 使用 Hugging Face Transformers + PEFT。云端 Ascend 环境必须先有匹配的 CANN / PyTorch / `torch_npu`。

最低 Python 包：

```bash
pip install -r finetuning/requirements-train.txt
```

注意：

- 如果平台镜像已经预装 Ascend 版 PyTorch，不要随便用 pip 覆盖成普通 CUDA/CPU 版。
- 如果 `torch_npu` 环境和普通 `torch` 冲突，优先按华为云镜像说明修环境。
- 如果你已经跑通 `ms-swift` 或 `LLaMA-Factory` 的 Ascend 训练，也可以用它们训练同一份 `sft_train/dev/heldout.jsonl`。

## 14. 跑 LoRA SFT smoke test

先跑 1 epoch 小训练，确认训练链路不崩。

```bash
python -m finetuning.train \
  --model-path ./Qwen3-8B \
  --train-file open_track_finetune/datasets/sft_train.jsonl \
  --dev-file open_track_finetune/datasets/sft_dev.jsonl \
  --output-dir open_track_finetune/outputs/qwen3_8b_lora_sft_smoke \
  --max-length 4096 \
  --learning-rate 5e-5 \
  --epochs 1 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --target-modules q_proj,v_proj,o_proj
```

如果 smoke test 正常，再跑第一版正式配置。

## 15. 跑第一版 LoRA SFT

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

也可以用配置文件：

```bash
python -m finetuning.train \
  --config finetuning/configs/qwen3_8b_lora_sft.json
```

## 16. 可选：导出 merged checkpoint

如果 vLLM 不方便直接加载 LoRA adapter，可以导出合并模型：

```bash
python -m finetuning.train \
  --model-path ./Qwen3-8B \
  --train-file open_track_finetune/datasets/sft_train.jsonl \
  --dev-file open_track_finetune/datasets/sft_dev.jsonl \
  --output-dir open_track_finetune/outputs/qwen3_8b_lora_sft_v1 \
  --merge-output-dir open_track_finetune/outputs/qwen3_8b_lora_sft_v1_merged
```

注意：这会重新进入一次训练流程。若只想 merge 已训练好的 adapter，后续可以再单独补一个只 merge 的脚本。

## 17. 重新部署并评测

训练后不要改 `agent_vllm_multistep.ipynb` 和 `agent/eval.py`。只切换模型服务。

两种方式：

1. 用 LoRA adapter 启动服务。
2. 用 merged checkpoint 启动服务。

尽量保持：

```text
base-url: http://127.0.0.1:8000/v1
served-model-name: qwen_auto
```

然后按原流程跑：

1. `agent_vllm_multistep.ipynb`
2. `agent/eval.py`
3. 对比 `OTEasyRun0.11_22`

重点检查：

- hard50 正确数是否提升。
- 11 道稳定正确题是否回退。
- 平均 tool calls 是否下降。
- action JSON 是否更稳定。
- final answer 是否仍然干净无 `<think>`。

## 18. 出问题时怎么停

如果出现以下情况，先回滚，不要继续加训：

- SFT 数据过滤后几乎为空。
- 训练 loss 异常为 NaN。
- 部署后模型不会输出合法 action JSON。
- hard50 11 道稳定正确题回退超过 2 道。
- 模型明显更倾向直接猜答案而不是调用工具。

最简单回滚方式：

```bash
# 停止微调模型服务后，重新用原始 Qwen3-8B 启动
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

不要覆盖原始 `Qwen3-8B` 目录。

## 19. 最小执行顺序

如果你只想按最短路径执行，顺序是：

1. `git pull origin main`
2. `python -m unittest tests.test_finetuning_pipeline -v`
3. 建立 `open_track_finetune/` 目录
4. 抽取并过滤 `oteasy_success_sft`
5. 生成 `synthetic_tasks_smoke.jsonl`
6. 启动 base `Qwen3-8B` 服务
7. 跑 smoke teacher
8. 评测 smoke teacher
9. 抽取并过滤 smoke SFT
10. 生成正式 `synthetic_tasks.jsonl`
11. 跑正式 teacher
12. 评测正式 teacher
13. 抽取、合并、过滤、切分 SFT
14. 回传数据质量信息让我检查
15. 跑 LoRA SFT smoke
16. 跑 LoRA SFT v1
17. 重新部署并用原评测入口评测
