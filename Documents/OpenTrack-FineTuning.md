# OpenTrack-FineTuning 详细设计方案

本文档对应 Open Track 中最复杂的部分：在 `Qwen3-8B` 基础上做微调，让模型更适应本项目的离线 Deep Research Agent 行为。

当前设计结论先写在前面：

1. **不要做全量微调。** 当前云端资源和项目周期都不适合全参 SFT。
2. **第一版只做 LoRA SFT。** 不建议第一版上 RL、QLoRA 或 AdaLoRA。
3. **现有 11 条成功轨迹太少。** 它们只能作为 seed、格式样例、回归保护集或极小 smoke test，不能支撑一轮可信微调。
4. **真正的数据来源应是合规的人造任务和自生成轨迹。** 只从 BrowseComp-Plus 离线语料中构造问题、证据、答案和轨迹，不接入外部知识库，不使用 public/private test 的 gold answer。
5. **微调代码必须和 `agent/multistep_agent.py` 解耦。** 新建 `finetuning/` 目录，训练数据和 checkpoint 持久化保存在云端；评测入口原则上不改，只切换部署模型。

## 1. 项目约束与目标

### 1.1 必须遵守的约束

微调仍然属于课程项目的一部分，不因为进入 Open Track 就放松约束。

必须遵守：

- 不替换主赛道检索器，继续使用课程统一 BM25。
- 不接入 Google、Bing 或任何外部实时搜索。
- 不引入 BrowseComp-Plus 语料库以外的知识库。
- 不把 public-test/private-test 作为训练样本。
- 不把标准答案硬编码进 prompt、agent 或训练数据。
- 不破坏 `submission.jsonl` 的完整轨迹可复原性。

合规数据的核心判断标准是：训练样本中的事实、答案和证据必须来自允许使用的离线语料或自己运行 agent 得到的轨迹，而不是来自测试集 gold answer。

### 1.2 微调目标

本项目微调不是为了让 `Qwen3-8B` 变成更强的通用模型，而是让它在当前 agent 框架中更稳定地产生有效行为。

优先学习的行为：

- 把复杂问题拆成可检索的子目标。
- 生成更短、更准、更少重复的 BM25 query。
- 在搜索结果中选择更可能包含证据的 docid。
- 打开文档后用 `find_in_document` 找局部证据。
- 在候选不足或验证不足时继续补证据。
- 在证据足够时输出干净短答案，不输出 `<think>` 或解释。

不优先学习的行为：

- 背答案。
- 直接从问题猜测最终答案。
- 生成很长的自然语言推理过程。
- 让模型替代已有候选打分和最终答案 guard。

## 2. 当前轨迹数量判断

### 2.1 现有数据规模

当前 OpenTrack-Easy 定版为 `OTEasyRun0.11_22`：

- 正确率：`11/50 = 22%`
- 正确题：`5, 314, 380, 397, 556, 651, 684, 761, 815, 1082, 1095`

本地已有历史 run 的正确题并集为 12 道：

- `5, 241, 314, 380, 397, 556, 651, 684, 761, 815, 1082, 1095`

因此，用户判断是正确的：即便把旧版本不相交成功集合也算上，可用成功题仍然远低于一个正常微调数据集所需规模。

### 2.2 11 条轨迹能不能做微调

可以做，但只能做三类低风险用途：

1. **数据格式验证**
   - 检查能否从 `messages` 中抽取 action decision、query rewrite、doc selection、final answer 样本。
   - 检查训练脚本、chat template、LoRA 保存、合并和部署是否能跑通。

2. **极小 LoRA smoke test**
   - 训练几十到一百多条拆分出来的行为样本。
   - 只看模型是否学会输出合法 JSON action、是否能减少格式错误。
   - 不把这类结果当作真实性能提升。

3. **回归保护集**
   - 微调后重新评测 11 道稳定正确题，检查是否回退。
   - 若回退明显，说明数据、学习率、adapter 或部署方式有问题。

不能做的事：

- 不能宣称 11 条成功轨迹足以支撑稳定泛化。
- 不能只训练“问题 -> gold answer”。
- 不能为了提升 hard50 分数，把 hard50 gold answer 做成监督标签。

### 2.3 为什么拆轨迹后仍然不够

`OTEasyRun0.11_22` 的平均工具调用约为 `12.72` 次/题。11 条成功轨迹最多能拆出约 100-150 条行为样本，看上去比 11 条多，但这些样本高度相关：

- 来自同一批问题。
- 查询词、文档、候选答案之间有强相关。
- 错误模式覆盖很窄。
- 容易让 LoRA 记住局部格式，而不是学习搜索策略。

所以它适合做训练流程打样，不适合做最终 Open Track 报告里的主训练数据。

## 3. 数据设计

### 3.1 数据总体路线

第一版训练数据分三层：

1. **真实轨迹 seed**
   - 来源：已有 `OTEasyRun*` / `BaseRun*` 的 `multistep_submission.jsonl`。
   - 用途：定义数据 schema、抽取样本、做回归保护。
   - 注意：如果这些轨迹来自最终评测题，则默认只用于分析和回归；是否允许作为 Open Track 训练样本，需要以课程/助教口径为准。

2. **非测试合成人造任务**
   - 来源：BrowseComp-Plus 离线语料。
   - 方法：从文档中抽取实体、标题、日期、作者、组织、章节名、数值等，自动构造可检索问题和标准证据。
   - 用途：构造几百到几千条合规训练任务。

3. **教师轨迹蒸馏**
   - 来源：当前 agent 或更高预算的同一 agent 在合成任务上的运行轨迹。
   - 方法：只保留答案可由证据自动验证、轨迹不重复、工具调用合理的样本。
   - 用途：把“问题 -> 答案”扩展为“状态 -> 下一步动作”的 SFT 数据。

### 3.2 允许使用的样本

推荐保留：

- 自己 agent 成功完成的非测试任务轨迹。
- 当前 agent 接近成功、且人工只根据轨迹证据修正动作的样本。
- 合成任务中答案可在打开文档中精确匹配的样本。
- 证据链包含明确 docid、片段、候选和验证结果的样本。

谨慎使用：

- 答案正确但轨迹很混乱的样本。
- 搜索次数过多、重复 query 过多的样本。
- 最终答案正确但中间证据无法支撑的样本。

禁止使用：

- public/private test gold answer。
- 为 hard50 错题手工填入 gold answer 后构造的 SFT 样本。
- 外部网页或搜索引擎补充的事实。
- 无证据链的“问题 -> 答案”样本。

### 3.3 人造任务构造方法

合成任务必须从离线语料内部生成，不能引入外部知识。

可构造的任务类型：

1. **标题/作者/日期类**
   - 从文档 metadata 抽取 `title`、`author`、`date`。
   - 问题形式：根据标题片段、日期、作者描述询问完整标题或作者。
   - 优点：自动验证容易。

2. **章节/目录类**
   - 从 `Contents`、`Table of Contents`、Markdown 标题中抽取章节名。
   - 问题形式：询问某文档的第一章、某章节前后的标题。
   - 优点：贴近 hard50 中“标题/章节名”类题。

3. **实体属性类**
   - 从同一段落中抽取人物、组织、地点、年份、奖项、职位。
   - 问题形式：给出若干上下文线索，询问对应实体。
   - 优点：能训练候选抽取和类型匹配。

4. **数值/日期类**
   - 抽取百分比、金额、年份、尺寸、页码。
   - 问题形式：询问某事件或对象对应的数值。
   - 优点：最终答案格式容易校验。

5. **桥接类**
   - 从同一文档或相邻段落构造两跳线索。
   - 例如：先定位文档，再问该文档内另一个字段。
   - 优点：更接近 BrowseComp-Plus 的复杂问法。

不建议第一版构造过难的多跳问题。第一版应优先覆盖“能稳定检索、能自动验真、能产生高质量轨迹”的任务。

### 3.4 轨迹生成策略

对每个人造任务，运行当前 agent 生成轨迹：

1. 使用与主评测相同的 BM25、工具、prompt、`max_rounds`。
2. 保存完整 `messages`、`agent_state`、工具调用和最终答案。
3. 用合成任务自带的 oracle answer 做自动验真。
4. 只保留正确轨迹或可人工修正的近成功轨迹。

如果希望更快获得高质量教师轨迹，可以使用更高预算配置：

- 更大的 `max_rounds`。
- 更大的 `tool_content_max_chars`。
- 更保守的停止条件。
- 更多候选验证。

这些高预算轨迹只用于训练模型学行为；最终评测仍用正常预算，避免报告中混淆推理预算和微调收益。

### 3.5 SFT 样本类型

不要直接训练完整长轨迹。应拆成以下行为样本。

#### 3.5.1 动作决策样本

目标：让模型根据当前状态输出下一步 action JSON。

输入：

- system prompt：当前 `ACTION_DECISION_SYSTEM_PROMPT` 的精简版。
- user prompt：原问题、state summary、最近 observation、已尝试 query、已打开 docid、当前候选。

输出：

```json
{"action":"search","query":"...","reason":"..."}
```

或：

```json
{"action":"get_document","docid":"...","reason":"..."}
```

训练注意：

- 只对 assistant 输出计算 loss。
- 输出必须是严格 JSON。
- `reason` 保持短句，不写长推理。

#### 3.5.2 Query 改写样本

目标：减少废 query 和重复 query。

输入：

- 原问题。
- 已失败 query 列表。
- 当前缺失证据。
- 当前 planned clue keywords。

输出：

```json
{"action":"search","query":"concise evidence-oriented query","reason":"missing evidence"}
```

筛选标准：

- 新 query 不应与历史 query 高度重复。
- 该 query 的搜索结果中应包含目标 docid 或明显更相关 docid。

#### 3.5.3 文档选择样本

目标：在搜索结果中选择更可能包含证据的 docid。

输入：

- 原问题。
- 搜索 query。
- top-k 搜索结果摘要。
- 已打开 docid。

输出：

```json
{"action":"get_document","docid":"30148","reason":"snippet matches book and chapter clues"}
```

筛选标准：

- 所选 docid 后续确实提供关键证据。
- 不选已打开文档。
- 不选明显广告、索引页、无关百科页。

#### 3.5.4 文档内定位样本

目标：让模型在打开长文档后生成局部查找 query。

输入：

- 原问题。
- docid。
- 文档标题或片段。
- 当前缺失证据。

输出：

```json
{"action":"find_in_document","docid":"30148","query":"first chapter title Australian colonisation","reason":"locate table of contents"}
```

筛选标准：

- `find_in_document` 结果应命中答案附近或关键证据附近。

#### 3.5.5 候选验证样本

目标：让模型知道什么时候验证候选，而不是继续盲搜。

输入：

- 原问题。
- 当前候选列表。
- 候选证据摘要。
- expected answer type。

输出：

```json
{"action":"verify_claim","candidate_answer":"Spero Therapeutics","reason":"candidate matches company clue and appears in evidence"}
```

#### 3.5.6 最终答案样本

目标：让模型输出干净短答案。

输入：

- 原问题。
- 最佳候选。
- 证据摘要。
- verifier result。

输出：

```text
Spero Therapeutics
```

训练注意：

- 最终答案样本占比不要太高。
- 不训练长解释。
- 不训练 `<think>`。

### 3.6 数据配比建议

第一版数据集建议规模：

- 最小 smoke：`100-300` 条样本。
- 第一版可报告：`500-1500` 条样本。
- 时间允许：`2000-5000` 条样本。

推荐配比：

- 动作决策：30%
- Query 改写：25%
- 文档选择：20%
- 文档内定位：15%
- 候选验证：5%
- 最终答案：5%

如果数据很少，宁可减少任务类型，也不要塞低质量样本。第一版优先保留动作决策、query 改写、文档选择三类。

### 3.7 数据划分

合成数据必须划分：

- `train`：80%
- `dev`：10%
- `heldout`：10%

划分时按 source docid 或问题模板分组，避免同一文档的几乎相同问题同时出现在 train 和 dev。

hard50 不作为训练集。hard50 用于最终对比评测和错误分析。

## 4. 理论训练方案

### 4.1 第一阶段：LoRA SFT

第一阶段采用标准监督微调。

训练目标：

给定输入 token 序列 `x` 和目标 assistant 输出 `y`，最小化 assistant 输出 token 的负对数似然：

```text
L_SFT = - sum_t log p_theta(y_t | x, y_<t)
```

关键点：

- 只对 assistant action / answer 部分计算 loss。
- system、user、工具 observation 只作为上下文，不参与 loss。
- 对于 action JSON，loss 目标是合法、短、可解析的动作。

为什么不用“问题 -> 答案”：

- 容易鼓励模型跳过工具。
- 容易学习测试答案模式。
- 对当前 agent 的主要瓶颈帮助较小。

### 4.2 样本权重

第一版不建议手写复杂 loss。可以用样本重复近似加权：

- 高质量成功轨迹动作：权重 1.0
- query 改写后明显找到证据：权重 1.5
- 最终答案格式样本：权重 0.5
- 人工修正样本：权重 0.5-1.0，取决于证据清晰度

如果使用训练库支持 `loss_weight`，再显式加权；否则用重复采样实现。

### 4.3 优化器与学习率

推荐默认：

- optimizer：`AdamW`
- learning rate：`1e-4`
- scheduler：cosine 或 linear decay
- warmup ratio：`0.03-0.05`
- max grad norm：`1.0`
- weight decay：`0.0-0.01`
- precision：`bf16`

数据很少时更保守：

- learning rate：`5e-5`
- epoch：`1-2`
- LoRA rank：`8`
- dropout：`0.05-0.1`

数据达到 1000 条以上时：

- learning rate：`1e-4`
- epoch：`2-3`
- LoRA rank：`16`
- dropout：`0.05`

### 4.4 序列长度

建议：

- `max_length = 4096` 作为第一版默认。
- 如果 state summary 和搜索结果较长，可尝试 `8192`。
- 不建议把完整长文档塞进训练输入；训练样本应使用压缩状态和片段。

原因：

- 训练目标是动作选择，不是全文阅读能力。
- 长上下文显著拖慢训练。
- 长输入会让少量数据更容易过拟合。

### 4.5 训练轮数

建议：

- smoke test：`1 epoch`
- 第一版训练：`2 epochs`
- 最多：`3 epochs`

如果 train loss 持续下降但 dev action format accuracy 下降，说明过拟合，应降低 epoch 或学习率。

### 4.6 可选第二阶段：偏好学习

如果 SFT 后已有稳定收益，可以考虑 DPO/ORPO，而不是直接上 RL。

偏好对来源：

- 同一状态下，成功 action 优于失败 action。
- 能找到证据的 query 优于重复/泛化 query。
- 正确 docid 优于无关 docid。
- 有证据再 finish 优于过早 finish。

偏好学习目标比 RL 更轻量，不需要在线 rollout，适合课程项目。

### 4.7 RL 的最低可行设计

不建议第一版做 RL。只有满足以下条件才考虑：

- SFT 已经带来稳定提升。
- 有至少数百条可自动验真的合成任务。
- 已能自动计算轨迹级 reward。
- 有时间承受多轮训练和评测。

最小 reward：

```text
R = 1.0 * answer_correct
  + 0.2 * evidence_supported
  + 0.1 * valid_action_format
  - 0.05 * repeated_query_count
  - 0.03 * unnecessary_tool_calls
  - 0.2 * premature_finish
```

说明：

- `answer_correct` 只对合成任务或允许训练任务计算，不能用测试集 gold 做训练 reward。
- `evidence_supported` 根据打开文档中是否包含答案和关键线索计算。
- 工具惩罚必须轻，不能让模型为了省工具而过早停止。

可选算法：

- GRPO/PPO 都可以作为报告中的后续方向。
- 当前项目第一版不实施 RL，除非 SFT 已经可靠。

## 5. PEFT 方法选择

### 5.1 推荐：LoRA

第一版选择 LoRA。

理由：

- 训练快。
- 显存/显存等价占用低。
- 容易回滚，只需切换 adapter 或模型路径。
- 对 8B 模型和小数据最稳。
- 当前目标是行为风格和工具决策，不需要大规模更新全参知识。

推荐配置：

```yaml
lora_rank: 8        # smoke；数据足够后可用 16
lora_alpha: 16      # rank=8 时
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

如果训练太慢或不稳定，先只训：

```yaml
target_modules:
  - q_proj
  - v_proj
  - o_proj
```

### 5.2 不推荐第一版：QLoRA

QLoRA 的优势是进一步节省显存，但它通常依赖 CUDA 生态中的量化支持。在华为 Ascend NPU 上，bitsandbytes 这类 CUDA 路线并不是默认可用方案。

本项目当前云端是 `NPU 910B`，不是 NVIDIA GPU，因此第一版不要把 QLoRA 作为主路线。除非确认所选训练库在 Ascend 上完整支持 4-bit QLoRA，否则会把主要时间浪费在环境适配上。

### 5.3 不推荐第一版：AdaLoRA

AdaLoRA 会动态分配 rank，理论上更灵活，但引入额外超参数和调试复杂度。当前数据规模小，主要风险是数据质量和过拟合，不是 rank 分配不够精细。

第一版不使用 AdaLoRA。

### 5.4 不推荐：全量 SFT

全量 SFT 不适合当前项目：

- 训练慢。
- checkpoint 大。
- 回滚成本高。
- 小数据下灾难性遗忘风险更高。
- 微调后还要跑 hard50，整体周期过长。

## 6. 训练库与 Ascend 环境

### 6.1 首选训练库

首选：`ms-swift`。

理由：

- 与 Qwen / ModelScope 生态结合紧。
- 支持 SFT、LoRA、DPO、GRPO 等训练路线。
- 对 Qwen 系列模型和 chat template 支持较直接。

备选：`LLaMA-Factory`。

理由：

- 配置化程度高。
- 有 NPU training 文档。
- 适合快速做 LoRA SFT 和导出 adapter。

不建议第一版手写 Hugging Face `Trainer`：

- 可控性强，但环境、chat template、assistant-only loss、LoRA 保存、Ascend 适配都要自己处理。
- 对第一次做微调的大三课程项目来说，工程风险高。

### 6.2 Ascend 与 CUDA 的区别

华为 `NPU 910B` 不使用 NVIDIA CUDA。常见关键词是：

- CANN
- Ascend Driver/Firmware
- `torch_npu`
- `ASCEND_RT_VISIBLE_DEVICES`
- vLLM Ascend / MindIE / MindSpore 相关运行时

因此不能默认照搬 CUDA 命令，例如：

- `CUDA_VISIBLE_DEVICES=0`
- bitsandbytes 4-bit QLoRA
- FlashAttention CUDA wheel

在 Ascend 上应优先使用训练库官方 NPU 路线，并确认 PyTorch、`torch_npu`、CANN 版本匹配。

### 6.3 Qwen3 注意事项

Qwen3 使用专门的 chat template，并且模型有 thinking / non-thinking 相关配置。当前 agent 要求：

- 不输出 `<think>`。
- action decision 输出严格 JSON。
- final answer 输出短答案。

因此训练样本中 assistant 内容必须是无 thinking 的目标输出。部署和评测时也应保持与当前 agent 一致的非思考输出约束。

## 7. 工程结构设计

### 7.1 新增目录

微调应和 `agent/multistep_agent.py` 解耦，新增目录：

```text
finetuning/
  README.md
  data_schema.md
  configs/
    qwen3_8b_lora_sft.yaml
  scripts/
    collect_success_ids.py
    build_sft_from_trajectories.py
    generate_synthetic_tasks.py
    run_teacher_agent.py
    filter_sft_data.py
    inspect_sft_data.py
  train/
    train_lora_swift.sh
    train_lora_llamafactory.sh
  deploy/
    merge_lora_adapter.sh
    serve_lora_vllm.sh
```

本地仓库保存：

- 数据抽取脚本。
- 配置文件。
- 小样例数据。
- 文档。

云端持久化保存：

- 大规模训练数据。
- 训练日志。
- LoRA adapter。
- 合并后的模型 checkpoint。
- 评测 run 目录。

### 7.2 数据文件布局

云端建议：

```text
open_track_finetune/
  raw_runs/
    OTEasyRun0.11_22/
    synthetic_teacher_run_001/
  datasets/
    sft_train.jsonl
    sft_dev.jsonl
    sft_heldout.jsonl
    synthetic_tasks.jsonl
  outputs/
    qwen3_8b_lora_sft_v1/
    qwen3_8b_lora_sft_v2/
  eval_runs/
    base_oteasy/
    lora_sft_v1/
```

不要把大 checkpoint 提交进 Git。

### 7.3 SFT JSONL 格式

建议采用 chat messages 格式：

```json
{
  "id": "synthetic_000123_step_04",
  "task_type": "query_rewrite",
  "source": "synthetic_teacher_run_001",
  "quality": {
    "answer_correct": true,
    "evidence_supported": true,
    "dedup_ok": true
  },
  "messages": [
    {"role": "system", "content": "You are a Deep Research Agent..."},
    {"role": "user", "content": "Question: ...\nState summary: ..."},
    {"role": "assistant", "content": "{\"action\":\"search\",\"query\":\"...\",\"reason\":\"...\"}"}
  ]
}
```

训练时只对最后一条 assistant content 计算 loss。

### 7.4 数据抽取逻辑

从 `multistep_submission.jsonl` 中抽取：

- assistant message 中的 `state_summary`
- assistant tool_calls 中的 tool name 和 arguments
- tool message 的结果摘要
- final predicted_answer
- 如果有 eval 文件，则连接 `eval_judgment`

抽取样本时要过滤：

- 无法解析的 action。
- 重复 query。
- 重复打开 docid。
- 过长输出。
- 带 `<think>` 的输出。
- 无证据支持的 final answer。

### 7.5 是否修改云端评测入口

原则上不改。

当前评测入口仍然是：

- [agent_vllm_multistep.ipynb](/D:/nju-nlp-deep-research/agent_vllm_multistep.ipynb:113)
- [agent/eval.py](/D:/nju-nlp-deep-research/agent/eval.py:150)

微调后只改变模型服务：

1. 停止原始 `Qwen3-8B` vLLM 服务。
2. 启动 LoRA 合并后模型或 LoRA adapter 版本。
3. 保持 OpenAI-compatible endpoint 不变。
4. 尽量保持 `served-model-name = qwen_auto` 不变。
5. 用同一 notebook 生成 submission。
6. 用同一 eval 脚本评测。

这样可以保证对照实验只改变模型，不改变 agent 代码和评测流程。

## 8. 第一版实施计划

### 8.1 V0：只做数据抽取，不训练

目标：

- 从 `OTEasyRun0.11_22` 抽取 SFT 样本。
- 输出 `sft_sample.jsonl`。
- 人工检查 20 条样本。

成功标准：

- action JSON 可解析率 100%。
- 不包含 `<think>`。
- 不包含 gold answer 泄漏。
- 每条样本能追溯到原始 trajectory。

### 8.2 V1：LoRA smoke test

目标：

- 使用 100-300 条样本跑通 LoRA SFT。
- 保存 adapter。
- 合并或加载 adapter 部署。
- 跑 5-10 道非正式小评测，确认模型服务可用。

成功标准：

- 训练不报错。
- loss 正常下降但不过拟合。
- 输出仍能被 agent 解析。
- 不影响 notebook 和 eval 入口。

### 8.3 V2：合成人造任务扩充

目标：

- 从 BrowseComp-Plus 构造 500-1500 条合成任务。
- 当前 agent 跑 teacher trajectories。
- 过滤得到高质量 SFT 数据。
- 重新训练 LoRA。

成功标准：

- dev action format accuracy 高。
- heldout synthetic answer accuracy 不低于 base。
- hard50 至少不明显回退，理想情况提升 1-3 题。

### 8.4 V3：偏好学习或 RL 预研

只有 V2 有收益才做。

目标：

- 用成功/失败 action 构造偏好对。
- 优先尝试 DPO/ORPO。
- RL 仅作为报告中的扩展方向，除非时间充足。

## 9. 评测设计

### 9.1 对照组

必须对比：

1. 单步 baseline。
2. BaseDemand 多轮版本。
3. OpenTrack-Easy 定版 `OTEasyRun0.11_22`。
4. LoRA SFT 版本。

### 9.2 指标

主指标：

- hard50 accuracy。

行为指标：

- 平均 tool calls。
- 平均 retrieved docs。
- 重复 query 数。
- 重复 opened docid 数。
- action JSON 解析失败率。
- final answer 中 `<think>` 出现率。
- 已正确 11 题回退数量。

数据指标：

- train/dev loss。
- dev action format accuracy。
- dev exact action type accuracy。
- query 去重率。
- doc selection 命中率。

### 9.3 快速评测策略

因为完整 hard50 一次接近一小时，训练后先跑小规模 smoke：

1. 选择 5 道历史稳定正确题。
2. 选择 5 道历史接近成功错题。
3. 跑完后检查 submission 轨迹。
4. 没有明显格式崩坏，再跑完整 hard50。

### 9.4 成功与失败判断

可以认为有效：

- hard50 正确数提升。
- 正确数不变但工具调用明显减少，且没有回退稳定题。
- action 格式错误明显下降。
- 最终答案格式更干净。

应回滚：

- 11 道稳定正确题回退超过 2 道。
- action JSON 大量解析失败。
- 模型频繁提前 finish。
- 模型更倾向直接猜答案而不是调工具。
- hard50 总正确数下降且行为指标无改善。

## 10. 问题定位与改进

### 10.1 数据问题

症状：

- train loss 低，dev 表现差。
- 模型复读固定 action。
- 模型输出特定题目的答案或实体。

处理：

- 降低重复轨迹比例。
- 按 docid/template 分组划分数据。
- 减少最终答案样本占比。
- 增加 query/doc selection 多样性。

### 10.2 训练配置问题

症状：

- 输出格式变差。
- 原本会调用工具，现在提前 finish。
- 少量训练后 hard50 大幅回退。

处理：

- 降低 learning rate。
- 降低 epoch。
- 降低 LoRA rank。
- 增加 dropout。
- 只训练 attention 层 LoRA。

### 10.3 部署问题

症状：

- 训练 loss 正常，但部署后行为和 base 完全一样。
- 或服务启动成功但输出乱码/模板错乱。

处理：

- 确认 vLLM 加载的是 adapter/merged checkpoint。
- 确认 tokenizer 和 chat template 来自同一模型目录。
- 确认 `served-model-name` 与 notebook 中 `MODEL_NAME` 一致。
- 用固定 prompt 对 base 和 LoRA 模型做单样本对比。

### 10.4 Agent 接口问题

症状：

- 微调模型输出 action，但 agent 解析失败。
- action 字段名和现有 `_action_to_tool_call` 不兼容。

处理：

- 训练目标严格复用现有 action schema。
- 增加数据检查脚本，训练前验证每条 assistant 输出可被 `_extract_action_from_raw_text` 解析。
- 不为了微调大改 `multistep_agent.py`。

## 11. 回滚与兜底

必须保留以下可回滚点：

1. 原始 `Qwen3-8B` 权重目录。
2. LoRA adapter 输出目录。
3. merged checkpoint 输出目录。
4. 训练配置 yaml。
5. 训练数据版本号。
6. 对应 eval run 目录。

回滚方式：

- 最简单：重新用原始 `Qwen3-8B` 启动 vLLM。
- 如果 adapter 坏了：删除 adapter 服务配置，回到 base。
- 如果 merged 模型坏了：不要覆盖 base 权重，直接切换模型路径。

报告中可以诚实写：

- 第一版 SFT 未超过 OpenTrack-Easy，但验证了数据抽取、LoRA 训练、部署、评测闭环。
- 或 SFT 只改善格式/工具效率，准确率无显著提升。

这比强行调参到不可复现更安全。

## 12. 推荐超参数表

### 12.1 Smoke 配置

```yaml
model: Qwen3-8B
method: LoRA SFT
train_samples: 100-300
max_length: 4096
epochs: 1
learning_rate: 5e-5
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
batch_size_per_device: 1
gradient_accumulation_steps: 8
precision: bf16
warmup_ratio: 0.03
max_grad_norm: 1.0
target_modules: q_proj,v_proj,o_proj
```

### 12.2 第一版可报告配置

```yaml
model: Qwen3-8B
method: LoRA SFT
train_samples: 500-1500
max_length: 4096
epochs: 2
learning_rate: 1e-4
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
batch_size_per_device: 1
gradient_accumulation_steps: 16
precision: bf16
warmup_ratio: 0.05
max_grad_norm: 1.0
weight_decay: 0.01
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### 12.3 若过拟合

```yaml
epochs: 1
learning_rate: 5e-5
lora_rank: 8
lora_dropout: 0.1
target_modules: q_proj,v_proj,o_proj
```

## 13. 报告写法建议

Open Track 微调章节建议结构：

1. 任务动机：当前 agent 主要瓶颈是搜索策略和工具决策。
2. 数据构造：从成功轨迹和合成任务中构造行为 SFT 样本。
3. 方法：LoRA SFT，assistant-only cross entropy。
4. 工程环境：Qwen3-8B，Ascend 910B，LoRA adapter，vLLM 重新部署。
5. 实验设置：数据规模、超参数、对照组、评测入口。
6. 结果：accuracy、tool calls、重复 query、稳定题回退。
7. 分析：哪些题改善，哪些题回退，失败原因。
8. 局限：数据规模、合成任务分布、未做 RL。
9. 后续：DPO/ORPO 或 RL reward 设计。

## 14. 外部参考

以下资料用于确认 Qwen3、LoRA/SFT、Ascend NPU 路线，后续实施时应优先查官方最新版本：

- [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [Qwen3-8B Hugging Face model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen documentation](https://qwen.readthedocs.io/)
- [Qwen MS-SWIFT training documentation](https://qwen.readthedocs.io/en/v3.0/training/ms_swift.html)
- [ModelScope ms-swift GitHub](https://github.com/modelscope/ms-swift)
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Hugging Face PEFT LoRA documentation](https://huggingface.co/docs/peft/)
- [Ascend Extension for PyTorch](https://github.com/Ascend/pytorch)
- [LLaMA-Factory NPU training documentation](https://llamafactory.readthedocs.io/en/latest/advanced/npu_training.html)
- [vLLM Ascend documentation](https://docs.vllm.ai/projects/ascend/)

## 15. 最终执行结论

本项目微调的最优先路线是：

1. 新建 `finetuning/`，先写数据抽取和检查脚本。
2. 用现有 11 条成功轨迹验证 schema，但不把它们当作充分训练集。
3. 从 BrowseComp-Plus 语料构造合规合成任务。
4. 用当前 agent 生成 teacher trajectories。
5. 过滤出高质量行为样本。
6. 在华为云 Ascend 910B 上用 `ms-swift` 或 `LLaMA-Factory` 做 LoRA SFT。
7. 合并或加载 LoRA adapter 后重新启动 vLLM。
8. 不改 notebook 和 eval 入口，重新跑 hard50。
9. 若收益不稳定，回滚到 OpenTrack-Easy 定版，并把微调作为可复现探索写入报告。

这一方案能最大限度降低第一次做微调的工程风险，同时让 Open Track 报告具备清晰的数据、方法、实验和失败分析。

## 16. 已落地代码单元

当前仓库已经按本方案新增独立 `finetuning/` 模块，微调链路与 [agent/multistep_agent.py](/D:/nju-nlp-deep-research/agent/multistep_agent.py:1) 解耦。

已落地文件：

- [finetuning/trajectory_sft.py](/D:/nju-nlp-deep-research/finetuning/trajectory_sft.py)：从 `multistep_submission.jsonl` 和可选 eval 结果中抽取行为 SFT 样本。
- [finetuning/synthetic_tasks.py](/D:/nju-nlp-deep-research/finetuning/synthetic_tasks.py)：从 BrowseComp-Plus parquet 语料流式生成合规合成任务。
- [finetuning/run_teacher_agent.py](/D:/nju-nlp-deep-research/finetuning/run_teacher_agent.py)：在云端用当前 agent 跑 synthetic teacher trajectories。
- [finetuning/evaluate_synthetic.py](/D:/nju-nlp-deep-research/finetuning/evaluate_synthetic.py)：用合成任务 oracle answer 做确定性评测，产出 teacher eval。
- [finetuning/filter_sft_data.py](/D:/nju-nlp-deep-research/finetuning/filter_sft_data.py)：过滤非法 JSON action、`<think>` 输出、过长样本和重复样本。
- [finetuning/merge_jsonl.py](/D:/nju-nlp-deep-research/finetuning/merge_jsonl.py)：合并多个 SFT JSONL 来源。
- [finetuning/split_jsonl.py](/D:/nju-nlp-deep-research/finetuning/split_jsonl.py)：按 source 分组切分 `train/dev/heldout`，降低同源泄漏。
- [finetuning/inspect_sft_data.py](/D:/nju-nlp-deep-research/finetuning/inspect_sft_data.py)：检查样本数量、任务类型分布和过滤风险。
- [finetuning/train.py](/D:/nju-nlp-deep-research/finetuning/train.py)：项目自有 LoRA SFT 训练入口，支持 config、assistant-only loss、LoRA adapter 保存和可选 merge。
- [finetuning/configs/qwen3_8b_lora_sft.json](/D:/nju-nlp-deep-research/finetuning/configs/qwen3_8b_lora_sft.json)：第一版 Qwen3-8B LoRA SFT 推荐配置。
- [finetuning/README.md](/D:/nju-nlp-deep-research/finetuning/README.md)：云端执行全流程命令。

本地 smoke 结果：

- 从 `OTEasyRun0.11_22` 抽取 `115` 条行为样本，过滤后保留 `107` 条。
- 保留样本包含 `question_decomposition / query_rewrite / doc_selection / document_find / candidate_extraction / candidate_verification`。
- 从本地 BrowseComp-Plus parquet 语料流式生成小批合成任务验证通过。
- `train.py --help` 可在未安装训练依赖的本地环境正常查看；真正训练仍应在云端 Ascend 环境执行。
