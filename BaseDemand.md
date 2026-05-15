# BaseDemand 设计与实施文档

本文档对应主赛道三项必做部分的详细设计：

1. 完整循环
2. 上下文管理
3. 提示词设计

目标不是做一个“看起来复杂”的 agent，而是以当前仓库为基础，做出一版可直接施工、可稳定评测、能明显超过单步 RAG 的最小可行方案。

## 1. 当前基线与问题定位

当前可确认事实来自 `first_runs/`：

- 单步 RAG 已跑通
- `submission.jsonl` 轨迹结构是合规的
- `eval_results.jsonl` 显示准确率为 `3/50 = 6%`
- 平均每题 `1.0` 次工具调用
- 平均每题 `5.0` 个检索结果

当前基线的核心问题：

1. 只有一次 `search`
无法对复杂问题进行分解、修正和追踪。

2. 只看 snippet，不看全文
很多题目需要交叉验证人物、时间、章节、作者关系，仅靠片段通常不够。

3. 没有显式状态
模型每轮看不到结构化的“已知/未知/待查”。

4. 没有停止逻辑
当前不是“判断后停止”，而是“单轮结束后直接作答”。

5. 输出格式不稳定
最终回答直接包含 `<think>`，既影响提交质量，也不利于后续解析。

## 2. 总体实施目标

第一阶段只做一件事：

把当前单步 RAG 升级为一个单 Agent、多轮、证据驱动、可停止的最小 Deep Research Agent。

该版本必须满足：

1. 使用课程允许的本地 BM25 检索器
2. 不接入任何外部搜索服务
3. 保持 `submission.jsonl` 轨迹格式兼容
4. 比单步 RAG 更强，且能通过错误分析解释改进来源

## 3. 实施范围

本阶段建议只在现有代码能力上做最小扩展，不引入过多新组件。

### 3.1 直接复用的现有能力

- [agent/browsecomp_searcher.py](/D:/nju-nlp-deep-research/agent/browsecomp_searcher.py:135)
- [agent/tools.py](/D:/nju-nlp-deep-research/agent/tools.py:75)
- [agent/vllm_client.py](/D:/nju-nlp-deep-research/agent/vllm_client.py:6)
- [agent/eval.py](/D:/nju-nlp-deep-research/agent/eval.py:132)

当前仓库已经有：

- `search(query)`
- `get_document(docid)`
- OpenAI-compatible `tools`
- 轨迹记录格式
- 自动评测脚本

### 3.2 本阶段不做的事

- 不做多 Agent
- 不做微调
- 不大量扩工具
- 不改评测脚本逻辑
- 不改 BM25 检索器本身

## 4. 完整循环设计

### 4.1 循环目标

每轮只做一个清晰动作：

- 搜索新线索
- 展开关键文档
- 或在证据充分时给出最终答案

### 4.2 最小回合结构

建议每题最多 `4-6` 轮，推荐默认 `5` 轮。

每一轮流程如下：

1. 读取当前状态摘要
2. 让模型决定下一步动作
3. 若产生工具调用，则执行工具
4. 将工具结果写入轨迹
5. 更新状态
6. 判断是否停止
7. 若未停止，进入下一轮

### 4.3 工具集合

本阶段使用两类工具即可：

1. `search(query)`
用途：
- 找候选文档
- 做查询改写后的再次检索
- 在信息不足时继续扩展证据面

2. `get_document(docid)`
用途：
- 查看完整文档
- 核实 snippet 中的关键关系
- 对候选答案做最终确认

这套最小组合已经足够覆盖大部分复杂题：

- `search` 负责召回
- `get_document` 负责核验

### 4.4 动作策略

建议模型每轮只允许以下三种动作之一：

1. `search`
当还没有找到高相关候选文档时使用。

2. `get_document`
当某个 docid 很可能包含关键答案，但 snippet 不够时使用。

3. `finish`
当已有证据足够支持单一答案时使用。

实现方式上可以有两种：

1. 纯工具调用式
让模型通过 `tool_calls` 决定 `search` / `get_document`，最后不再调用工具时视为 `finish`。

2. 混合状态式
让模型先输出结构化动作决策，再决定是否调用工具。

本阶段建议采用第 1 种，原因是更接近现有 notebook 与 `vllm` 调用方式，施工成本最低。

### 4.5 停止条件

必须显式实现，不能再默认单轮结束。

建议停止条件如下：

1. 正常停止
满足任意一个即可：

- 模型不给出工具调用，直接输出最终答案
- 已找到唯一高置信候选答案，且至少有 2 条相互支持的证据
- 已展开关键文档并完成答案核验

2. 强制停止
满足任意一个即可：

- 达到最大轮数
- 连续两轮没有新增有效信息
- 新一轮查询与历史查询高度重复
- 工具返回空结果过多

3. 强制收尾
若被强制停止但仍未完全确定答案：

- 允许输出当前最优答案
- 但最终提示词必须要求说明证据基础
- 不允许输出纯思维过程替代答案

### 4.6 查询改写策略

本阶段不需要做复杂 query planner，但必须支持基本改写。

建议规则：

1. 初始轮
先用原问题检索。

2. 后续轮
根据当前状态，只保留最关键的实体、年份、关系词重新检索。

3. 若问题过长
将其拆成 1 个主目标 + 1 个待验证关系。

例如：

- 原问题很长时，不再直接整句搜索
- 改为“人物 A + 作品 B + 时间 C”
- 或“作者 + spouse + year”

### 4.7 推荐轮次模板

推荐最小模板：

1. 第 1 轮：原问题搜索
2. 第 2 轮：针对最相关候选文档做全文展开
3. 第 3 轮：围绕未确认关系做改写搜索
4. 第 4 轮：再次展开关键文档
5. 第 5 轮：结束或强制收尾

## 5. 上下文管理设计

### 5.1 设计目标

不要把所有历史原样塞回模型。

要做的是：

- 保存完整轨迹用于提交
- 给模型只提供压缩后的状态摘要

### 5.2 推荐状态结构

建议在实现中引入一个显式 `state` 字典，至少包含以下字段：

```python
state = {
    "question": str,
    "search_history": list,
    "seen_docids": list,
    "opened_docids": list,
    "evidence_notes": list,
    "confirmed_facts": list,
    "pending_subquestions": list,
    "candidate_answers": list,
    "last_action": str,
    "stall_count": int,
    "finish_reason": str,
}
```

字段含义：

- `question`: 原始问题
- `search_history`: 已执行过的 query 列表
- `seen_docids`: 搜索结果中出现过的文档
- `opened_docids`: 已全文展开过的文档
- `evidence_notes`: 从工具结果提炼出的简短证据
- `confirmed_facts`: 已确认的关键事实
- `pending_subquestions`: 仍待验证的问题
- `candidate_answers`: 当前候选答案及其证据
- `last_action`: 上一轮动作
- `stall_count`: 连续无新信息的轮数
- `finish_reason`: 最终停止原因

### 5.3 状态更新规则

每轮工具执行后都要更新。

#### 搜索后更新

- 把 query 写入 `search_history`
- 把返回 docid 合并到 `seen_docids`
- 从 top 结果中提炼 1-3 条简短线索进入 `evidence_notes`
- 若出现明显新实体或新关系，写入 `pending_subquestions`

#### 全文展开后更新

- 把 docid 写入 `opened_docids`
- 提取该文档可直接支持答案的句级事实
- 若文档只支持部分线索，也要标注为“部分支持”

#### 结束前更新

- 将最终答案写入 `candidate_answers`
- 将停止原因写入 `finish_reason`

### 5.4 去重与压缩

必须做三类去重：

1. 查询去重
若新 query 与历史 query 完全相同，禁止再次调用。

2. 文档去重
已展开的 docid 原则上不再重复展开，除非后续实现确有必要。

3. 事实去重
`confirmed_facts` 中不保留重复表述，只留简洁版本。

### 5.5 提供给模型的状态摘要

每轮不要把完整 `state` 直接原样传给模型。

建议构造成一段短摘要，结构固定：

```text
Question: ...

Known facts:
1. ...
2. ...

Open questions:
1. ...
2. ...

Searches already tried:
- ...
- ...

Opened documents:
- ...

Current best candidate:
- ...
```

摘要原则：

- 只保留必要信息
- 不超过 10-15 条短句
- 避免原文大段复制

### 5.6 证据存储粒度

本阶段不要求复杂引用系统，但至少要做到：

- 每条关键证据能追溯到 `docid`
- 证据内容使用简洁摘要，不贴整段原文
- 最终答案前至少能列出若干“支持答案的已确认事实”

## 6. 提示词设计

### 6.1 设计目标

prompt 的职责不是让模型“更会聊天”，而是强约束它：

1. 先想下一步动作
2. 证据不足就继续查
3. 证据足够再回答
4. 最终答案格式稳定
5. 不输出 `<think>`

### 6.2 系统提示词的核心要求

系统提示词至少应包含以下约束：

1. 你是一个 Deep Research Agent
2. 你必须通过工具收集证据，不能只靠常识猜测
3. 如果信息不足，优先继续搜索或展开文档
4. 避免重复搜索与重复读文档
5. 只有在证据足够时才给最终答案
6. 最终答案必须使用统一格式，且不要输出思维链标签

### 6.3 建议的双阶段提示结构

本阶段建议把提示拆成两个用途：

1. 循环阶段提示
用于决定下一步是 `search`、`get_document` 还是停止。

2. 最终回答提示
用于在最后一轮输出整洁答案。

这样做的原因：

- 决策 prompt 和答案 prompt 目标不同
- 分开后更容易调试和替换

### 6.4 循环阶段提示要点

循环阶段的系统提示应明确：

- 你的目标是逐步回答复杂问题
- 每轮优先决定一个最有效的下一步动作
- 不要重复使用已尝试的查询
- 如果某个 snippet 看起来重要，优先展开全文
- 如果仍有关键关系未验证，不要直接结束

传给模型的用户内容建议包含：

- 原问题
- 状态摘要
- 最近一轮工具结果摘要

### 6.5 最终回答提示要点

最终回答提示应明确：

- 基于已确认事实输出答案
- 不要输出 `<think>`、`<analysis>` 或自由推理痕迹
- 使用统一结构

建议格式：

```text
Explanation: <2-4 句，简要说明证据链>
Exact Answer: <最终答案>
Confidence: <0-100%>
```

### 6.6 输出清洗

即使 prompt 已经约束，仍建议在代码层增加最终清洗逻辑。

最少应做：

1. 去掉 `<think> ...` 段
2. 去掉多余前缀和空白
3. 优先提取 `Exact Answer:` 后的文本作为 `predicted_answer`

说明：

当前 `eval.py` 在没有专门抽取时，会直接使用 `predicted_answer` 或最后一条 `assistant.content`。因此如果最终消息混有长推理文本，会降低评测稳定性。

## 7. 轨迹设计要求

保持与课程格式兼容。

每题输出仍使用：

- `system`
- `user`
- `assistant`
- `tool`

与当前基线不同的是：

- 会出现多轮 `assistant(tool_calls)` 和 `tool`
- 最后一条 `assistant` 必须是干净的最终答案

建议增加但不强依赖的字段：

- `state_summary`
- `current_subgoal`
- `next_action_plan`

这些字段可以放在 `assistant` message 的扩展字段中，方便后续分析，但不要破坏主结构。

## 8. 建议的代码改造位置

本阶段建议新增一个独立脚本或 notebook 版本，不要直接覆盖单步基线。

推荐思路：

1. 保留当前 `agent_vllm.ipynb` 作为单步 baseline
2. 新增一个多轮版本，例如：
- `agent_vllm_multistep.ipynb`
- 或 `agent/run_multistep_agent.py`

3. 工具层复用 [agent/tools.py](/D:/nju-nlp-deep-research/agent/tools.py:75)
4. 需要时在 `agent/tools.py` 中补充轻量工具

建议最先修改或新增的代码位置：

- `agent/tools.py`
- 新的 agent loop 实现文件
- 新的 submission 生成逻辑

## 9. 推荐实施顺序

严格按这个顺序做，避免一次性改太多。

1. 先做多轮 loop
目标：
- 支持 `search` 多轮调用
- 支持 `get_document`
- 支持最大轮数

2. 再做显式状态管理
目标：
- 记录已查 query / docid
- 增加 state summary
- 防重复

3. 最后做 prompt 稳定化
目标：
- 强化动作决策
- 规范最终答案
- 清洗 `<think>`

4. 跑一轮评测并和 `first_runs` 对照
重点看：
- 准确率是否超过 6%
- 平均工具调用次数是否合理
- 错题是否从“完全没查到”变成“查到了但推错了”

## 10. 验收标准

BaseDemand 阶段完成后，应至少满足：

1. 每题不再固定只有 1 次工具调用
2. 至少有一部分题目会调用 `get_document`
3. 不再出现明显的 `<think>` 污染最终答案
4. `submission.jsonl` 结构仍可直接被 `agent.eval` 评测
5. 正确率应显著高于当前 `6%`

## 11. 当前不确定项

本阶段仍有两点需要在实现时边做边验证：

1. `qwen_auto` 在当前云端配置下是否稳定产生标准 `tool_calls`
若不稳定，需要进一步调整 tool parser 或 prompt。

2. `get_document` 返回全文后 token 压力是否过大
若过大，下一步再考虑加入局部查找工具，而不是现在提前复杂化。

## 12. 下一步

按本文档施工时，下一步建议是：

1. 新建多轮 agent 实现文件
2. 先接上 `search + get_document`
3. 加入最大轮数与状态结构
4. 先在 `hard50` 的少量样本上试跑
5. 确认轨迹格式后再批量评测
