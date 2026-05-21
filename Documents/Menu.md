# Deep Research Agent 项目执行文书

本文档用于作为本项目的统一执行菜单，目标是：

- 明确课程项目必须完成的内容与禁止事项
- 固化本地开发、云端运行、评测与提交的标准工作流
- 为后续详细设计预留统一位置，便于逐步补全、交接和复查

本文档当前依据以下材料整理：

- [README.md](/D:/nju-nlp-deep-research/README.md)
- [agent/README.md](/D:/nju-nlp-deep-research/agent/README.md)
- [agent_vllm.ipynb](/D:/nju-nlp-deep-research/agent_vllm.ipynb)
- [deep-research-agent.html](/D:/nju-nlp-deep-research/deep-research-agent.html)

后续如果课程要求或仓库代码发生变化，应优先更新本文档。

## 1. 项目要求总览

### 1.1 主赛道必须完成的基础部分

根据课程说明，主赛道至少需要在 baseline 基础上完成以下核心能力：

1. 多轮检索 loop
2. 明确的停止条件
3. 历史与上下文管理
4. Prompt 设计
5. 基于证据的最终回答

其中，课程文档中反复强调，本项目目标不是单次 `search -> answer`，而是构建一个能够在固定离线语料环境中逐步推进、保留证据、输出可检查答案的 Deep Research Agent。

### 1.2 Open Track 可选加分部分

根据 html 中的 Open Track 说明，额外加分方向主要包括三部分：

1. 模型训练
在 `Qwen3` 基础上进行微调，可包含：
收集 agent 成功搜索轨迹做 SFT；
用答案是否正确作为 reward 信号做 RL；
让模型逐渐学会更高效的搜索策略。

2. 工具增强
在统一主框架下尝试补充更细粒度、更有效的工具能力，但不能违反课程的统一检索约束。

3. Agent 架构增强
例如更复杂的规划、分工、验证、协作机制，如多 Agent 或更强的状态管理与反思机制。

说明：

- Open Track 是加分项，不是主赛道必做项。
- 不做 Open Track 时，仍然需要完成主赛道 agent、轨迹输出和自动评测。

### 1.3 统一提交与结果要求

最终提交的核心结果是统一格式的 `submission.jsonl`，每条记录应至少包含：

- `query_id`
- `status`
- `predicted_answer`
- `messages`

其中 `messages` 应尽可能完整记录整个对话与工具调用轨迹，包括：

- `system`
- `user`
- `assistant`
- `tool`

课程说明还要求：

- 最终答案与完整轨迹同时保留
- 评测脚本可从 `messages` 中还原搜索与决策过程
- 主赛道和 Open Track 都应保留足够的实验结果以供复查

### 1.4 不要做的事：约束清单

这一部分必须严格遵守，后续实现、调参和补工具时都要逐项对照。

#### A. 检索与数据约束

1. 不允许替换检索器
主赛道统一使用课程提供的 BM25 检索器。

2. 不允许使用额外外部搜索服务
不得接入 `Google`、`Bing` 或其他外部网络搜索服务。

3. 不允许引入 benchmark 外部知识库
不能通过额外知识库、额外语料、外挂数据库等方式绕开 BrowseComp-Plus 的固定离线语料设定。

#### B. 训练与测试约束

4. 禁止使用测试数据作为训练样本
html 中明确写明：禁止使用 `public-test` / `private-test` 数据作为训练样本。

5. Open Track 的训练数据若过大，可保存在云服务器
这不是鼓励额外扩展训练数据来源，而是在合规前提下说明大文件无需作为作业一并上传。

#### C. 任务目标约束

6. 不要把项目做成“换检索器比赛”
课程统一固定检索器，就是为了将重点放在 agent 的推理、状态管理、工具使用和证据整合上。

7. 不要把项目做成“一次搜索猜答案”
课程明确强调，目标不是单轮搜索碰运气，而是多轮推进、逐步收集证据并给出可追溯答案。

8. 不要忽略轨迹记录
最终提交不是只交答案；必须保留能够复原 agent 行为过程的轨迹。

## 2. 项目详细工作流

本项目推荐采用“本地开发，云端运行”的模式。

### 2.1 基本原则

1. 本地环境负责：
- 阅读文档
- 编写和修改代码
- 查看评测结果
- 分析错误案例
- 整理报告与提交材料

2. 华为云 Linux 云机负责：
- 安装运行依赖
- 下载和托管模型权重
- 启动 `vLLM` 服务
- 构建 BM25 索引
- 运行 notebook 或脚本生成 `submission.jsonl`
- 运行自动评测
- 如参加 Open Track，则负责模型训练与训练产物存储

3. 原则上不建议为了本作业专门在 Windows 本机部署完整 `WSL + vLLM + Qwen3-8B`
除非后续确实需要本地完整复现，否则优先将模型服务放在云端 Linux 环境。

### 2.2 仓库与同步信息

当前远程仓库地址：

- `origin`: [https://github.com/Xuao-Xiao/NJU-NLP-DeepResearchAgent.git](https://github.com/Xuao-Xiao/NJU-NLP-DeepResearchAgent.git)

推荐同步方式：

1. 本地修改代码后执行：

```bash
git add .
git commit -m "your message"
git push origin main
```

2. 云端进入项目目录后执行：

```bash
git pull origin main
```

说明：

- 若后续改为分支开发，应将本文档中的 `main` 替换为实际分支名。
- 如果云端不能直接访问 GitHub，再考虑使用压缩包上传、SFTP 或平台文件同步。
- 首选方式始终应是 `git push` / `git pull`。

### 2.3 环境部署工作流

#### 第 0 步：明确项目运行结构

本项目不是“clone 仓库后直接自带 8B 模型”。

实际结构是：

1. 本仓库提供：
- agent 框架
- BM25 检索代码
- notebook 示例
- 自动评测脚本

2. 模型权重需要额外下载：
- `Qwen3-8B` 或课程允许的兼容模型

3. notebook 与脚本默认依赖一个已经启动好的 `vLLM` OpenAI-compatible endpoint

#### 第 1 步：云端准备运行环境

在华为云 Linux 云机上完成：

1. 克隆项目仓库
2. 安装 Python 依赖
3. 下载模型权重
4. 构建 BM25 索引
5. 启动 `vLLM`

课程材料中的最小依赖包括：

- `pyarrow`
- `python-dotenv`

安装命令：

```bash
pip install -r agent/requirements.txt
```

#### 第 2 步：云端下载模型

根据 [README.md](/D:/nju-nlp-deep-research/README.md)：

```bash
git clone https://atomgit.com/hf_mirrors/MindSpore-Lab/Qwen3-8B.git
```

说明：

- `Qwen3-8B` 不在本仓库中
- 模型权重通常应保存在云端，而不是 Windows 本地
- 不做 Open Track 时，也通常仍然需要该模型来运行 agent 和评测

#### 第 3 步：云端构建 BM25 索引

首次执行一次即可：

```bash
python -m agent.build_bm25_index \
  --corpus-path ./browsecomp-plus-corpus \
  --index-path ./indexes/browsecomp_plus_bm25.sqlite \
  --overwrite
```

说明：

- 索引通常只需构建一次
- 后续 notebook 或脚本直接复用 `index_path`

#### 第 4 步：云端启动 vLLM 服务

Qwen 路线示例：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

说明：

- `vLLM` 需要在终端中单独启动
- 启动后必须保持运行
- notebook 只负责调用已启动的服务

标准访问地址通常为：

```text
http://127.0.0.1:8000/v1
```

如果 notebook 和 `vLLM` 在同一台云机上运行，保持上述地址即可。

### 2.4 开发与评测循环

这是后续整个项目最核心的闭环。

#### 阶段 A：本地开发

1. 本地阅读并理解：
- [README.md](/D:/nju-nlp-deep-research/README.md)
- [agent/README.md](/D:/nju-nlp-deep-research/agent/README.md)
- [agent_vllm.ipynb](/D:/nju-nlp-deep-research/agent_vllm.ipynb)
- [agent_vllm_weather.ipynb](/D:/nju-nlp-deep-research/agent_vllm_weather.ipynb)

2. 本地实现或修改：
- agent loop
- 停止条件
- 上下文管理
- prompt
- 轨迹记录逻辑

3. 本地提交代码：

```bash
git add .
git commit -m "implement/update agent logic"
git push origin main
```

#### 阶段 B：云端同步代码

```bash
git pull origin main
```

#### 阶段 C：云端运行生成结果

在云端运行 notebook 或脚本，生成：

- `runs/submission.jsonl`

此文件应包含：

- 最终答案
- 工具调用轨迹
- 完整 `messages`

#### 阶段 D：云端自动评测

评测脚本会再次调用模型服务，因此即便不做 Open Track，也需要 `vLLM` 正在运行。

示例命令：

```bash
python -m agent.eval \
  --submission runs/submission.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model Qwen3-8B \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/eval_results.jsonl
```

输出通常包括：

- 准确率
- 各题评测结果
- 轨迹统计信息

#### 阶段 E：本地分析与迭代

将以下内容带回本地查看和分析：

- `submission.jsonl`
- `eval_results.jsonl`
- 错误案例
- 轨迹中的工具调用顺序与停止位置

然后继续修改代码，重新进入下一轮：

`本地改代码 -> push -> 云端 pull -> 云端运行 -> 云端评测 -> 本地分析`

### 2.5 Open Track 工作流

仅在参与 Open Track 时执行。

#### 阶段 F：轨迹收集

使用当前 agent 运行足够多样的任务，收集：

- 成功轨迹
- 失败轨迹
- 可用于训练或分析的对话与工具调用数据

当前结论：

- `OTEasyRun0.11_22` 只有 `11/50` 正确。
- 历史 run 的正确题并集也只有 `12` 道。
- 这些轨迹不足以支撑正式微调，只能作为数据格式 seed、smoke test 和回归保护集。
- 真正训练应优先从 BrowseComp-Plus 离线语料构造合规人造任务，再用当前 agent 生成 teacher trajectories。

#### 阶段 G：云端训练

在云端进行：

- LoRA SFT，作为第一优先路线
- 可选 DPO/ORPO，只有 SFT 稳定有效后再考虑
- RL 仅作为后续方向，不作为第一版目标
- LoRA adapter、merged checkpoint 和训练日志保存

说明：

- 训练应在华为云 Linux 云机上完成
- 大规模训练数据和 checkpoint 可以持久化保存在云端
- 华为 `NPU 910B` 不使用 NVIDIA CUDA；训练应优先走 Ascend/CANN/`torch_npu` 适配路线
- 第一版不建议 QLoRA、AdaLoRA 或全量 SFT

#### 阶段 H：训练后重新部署与评测

1. 用 LoRA adapter 或合并后的模型重新启动 `vLLM`
2. 重新生成 `submission.jsonl`
3. 重新运行自动评测
4. 对比 baseline 与训练后模型结果

原则上不改 [agent_vllm_multistep.ipynb](/D:/nju-nlp-deep-research/agent_vllm_multistep.ipynb:113) 和 [agent/eval.py](/D:/nju-nlp-deep-research/agent/eval.py:150)，只切换模型服务。最好保持 OpenAI-compatible endpoint 和 `served-model-name` 不变，使对照实验只改变模型权重。

#### 阶段 I：Open Track 补充材料

按课程说明，额外整理：

- 消融实验轨迹
- Open Track 章节报告
- 算法说明
- 实验设置
- 实验分析

## 3. 项目具体设计

本节只保留当前阶段最简单、最准确的设计概况。详细实施细节见：

- [BaseDemand.md](/D:/nju-nlp-deep-research/BaseDemand.md)
- [OpenTrack-Easy.md](/D:/nju-nlp-deep-research/OpenTrack-Easy.md)
- [OpenTrack-FineTuning.md](/D:/nju-nlp-deep-research/OpenTrack-FineTuning.md)

当前基线事实：

- 首轮单步 RAG 结果位于 `first_runs/`
- 当前基线准确率为 `3/50 = 6%`
- 每题固定只有 `1` 次 `search`
- 每题固定只看 `5` 个 snippet
- 不做全文展开
- 不做查询改写
- 不做状态维护
- 不做停止判断
- 输出中混入了 `<think>` 推理内容

因此，后续设计的核心原则是：

1. 先把主赛道三项必做任务做扎实并跑通
2. 每次只引入一类改动，便于做对照评测
3. 所有增强都必须保持统一轨迹格式与课程约束

### 3.1 完整循环设计

设计概况：

- 从单步 `search -> answer` 升级为最多 `N` 轮的 agent loop
- 每轮由“规划 -> 调工具 -> 解析结果 -> 更新状态 -> 判断是否停止”构成
- 最小工具集先使用 `search` + `get_document`
- 停止条件采用“已拿到足够证据”与“达到最大轮数/无新增信息”双重机制
- 第一阶段优先实现单 agent、多轮工具调用，不引入复杂协作

### 3.2 上下文管理设计

设计概况：

- 显式维护 agent state，而不是把所有历史都塞回对话
- state 至少包含：原问题、已执行查询、已看文档、已确认事实、待确认子问题、候选答案、停止原因
- 对模型只回填压缩后的状态摘要与当前必要证据，避免 token 爆炸
- 通过查询去重、文档去重、事实摘要，降低重复搜索与重复阅读

### 3.3 提示词设计

设计概况：

- 将 prompt 从“直接回答”改为“先决定下一步动作，再在证据充分时回答”
- 强制要求答案由证据支持，不允许仅凭常识猜测
- 明确要求模型在信息不足时继续搜索或展开文档，而不是过早作答
- 最终答案格式必须去掉 `<think>`，并统一成可解析结构

### 3.4 其他工具补充设计

设计概况：

- 主赛道先不扩展过多工具，优先用好现有 BM25 与全文读取能力
- Open Track 初期只补“局部查找”一类轻量工具，如 `find_in_document`
- 所有新工具都必须建立在本地离线语料与现有检索器之上
- 工具输出必须继续兼容现有 `messages/tool_calls/tool` 轨迹结构

### 3.5 多 Agent 架构设计

设计概况：

- 多 Agent 不进入主赛道第一阶段
- Open Track 初期优先考虑“两角色”结构：规划/执行，或执行/验证
- 多 agent 的目标不是炫技，而是减少无效搜索、增加答案校验
- 若多 agent 不能稳定提升正确率，则应回退到强化单 agent

### 3.6 微调设计

设计概况：

- 微调属于最后阶段，当前详细方案见 [Documents/OpenTrack-FineTuning.md](/D:/nju-nlp-deep-research/Documents/OpenTrack-FineTuning.md)
- 现有 11 条成功轨迹太少，只能用于 schema 验证、smoke test 和稳定题回归
- 正式训练数据应通过 BrowseComp-Plus 离线语料构造合规人造任务，再由当前 agent 生成 teacher trajectories
- 第一版采用 LoRA SFT；不做全量 SFT，不把 QLoRA/AdaLoRA/RL 作为第一选择
- 训练目标是 agent 行为：动作决策、query 改写、文档选择、文档内定位、候选验证和最终短答案格式
- 微调代码与 `agent/multistep_agent.py` 解耦，建议新建 `finetuning/` 目录管理数据抽取、训练配置、adapter 部署和实验日志
- 云端评测入口原则上不改，只通过重新部署 base/adapter/merged model 做对照

## 4. 当前阶段行动清单

当前微调工程骨架已经落地，相关入口集中在 [finetuning/README.md](/D:/nju-nlp-deep-research/finetuning/README.md)。

已经完成：

1. 新建 `finetuning/`，实现轨迹到 SFT 样本的数据抽取和检查脚本。
2. 从 `OTEasyRun0.11_22` 抽取小样例，验证 schema 和过滤规则。
3. 实现 BrowseComp-Plus 合成人造任务生成器，不使用测试集 gold answer。
4. 实现 teacher trajectories 运行入口、合成 oracle 评测、SFT 合并/过滤/切分入口。
5. 实现项目自有 LoRA SFT 训练入口 [finetuning/train.py](/D:/nju-nlp-deep-research/finetuning/train.py) 和推荐配置。

下一步在云端执行：

1. 保持 OpenTrack-Easy 定版 `OTEasyRun0.11_22` 作为当前稳定对照。
2. 在云端生成较大 `synthetic_tasks.jsonl`，不要提交到 GitHub。
3. 在云端用当前 agent 跑 synthetic teacher trajectories。
4. 过滤得到第一版 `sft_train/dev/heldout.jsonl`。
5. 第一次真实训练前，回传过滤后数据或至少回传 `inspect_sft_data` 输出做质量确认。
6. 在华为云 Ascend 910B 上用 [finetuning/train.py](/D:/nju-nlp-deep-research/finetuning/train.py) 或已配置好的 `ms-swift` / `LLaMA-Factory` 跑 LoRA SFT smoke test。
7. 重新部署 adapter 或 merged model，用同一 notebook 和 eval 入口评测。
8. 对比 base、OpenTrack-Easy、LoRA SFT 三组结果，并检查 11 道稳定正确题是否回退。

## 5. 文档维护规则

后续更新本文档时，建议遵守以下规则：

1. 已确认的流程写成确定语句，不写成猜测
2. 未确认的内容统一放在“待补充”或“待验证”位置
3. 每次设计变更后，优先更新本文档，再继续实现
4. 本文档应始终能够让接手者快速回答两个问题：
- 当前项目必须做什么
- 下一步具体应该怎么做
