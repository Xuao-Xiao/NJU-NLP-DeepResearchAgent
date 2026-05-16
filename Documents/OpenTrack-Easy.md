# OpenTrack-Easy

本文档面向 OpenTrack 前两个加分方向：
- 工具增强
- 多 Agent 架构

目标不是立即施工所有点，而是给下一位接手者一份能直接开始实施的设计草案，并且与当前 BaseDemand 定版自然衔接。

当前 BaseDemand 基线：
- [agent/multistep_agent.py](/D:/nju-nlp-deep-research/agent/multistep_agent.py:1)

## 1. 设计原则

- 不替换课程统一 BM25 检索器。
- 不接外部搜索引擎。
- 不把 OpenTrack 做成“重写整套系统”。
- 优先补“当前 BaseDemand 已暴露的结构性短板”：
  - 候选答案验证不足
  - 多跳问题拆解不足
  - 错误文档簇清理不足
  - 终答前缺少专门的验证角色

## 2. 工具设计

### 2.1 工具 1：`verify_claim`

目标：
- 判断某个候选答案是否真的被当前证据支持。

建议输入：
```json
{
  "question": "...",
  "candidate_answer": "...",
  "evidence_docids": ["..."],
  "evidence_snippets": ["..."]
}
```

建议输出：
```json
{
  "supported": true,
  "support_score": 0.0,
  "missing_piece": "...",
  "contradictions": ["..."],
  "verdict_note": "..."
}
```

作用：
- 解决“有候选答案但不敢停”。
- 解决“错误答案被草率收尾”。
- 非常适合接在当前 BaseDemand 的 `candidate shortlist` 之后。

最适合接入的位置：
- 在 final answer 前
- 或者在 `_decide_next_action(...)` 中作为停止前验证

### 2.2 工具 2：`decompose_question`

目标：
- 对复杂多跳题给出更稳定的子任务分解，而不是只产出几个 query。

建议输入：
```json
{
  "question": "..."
}
```

建议输出：
```json
{
  "answer_type": "...",
  "subgoals": [
    "...",
    "..."
  ],
  "entities_to_identify": [
    "...",
    "..."
  ],
  "verification_targets": [
    "...",
    "..."
  ]
}
```

与当前 BaseDemand 的区别：
- 当前 `_plan_question(...)` 只做到 query 级拆解。
- `decompose_question` 应升级成“任务图级拆解”。

最适合的收益场景：
- spouse / partner / then-husband
- annual report 双年份交叉定位
- 书名 / 章节 / 作者 / 出版社 / 婚姻关系串联

### 2.3 工具 3：`find_in_document`

目标：
- 在已知 `docid` 的前提下，定点找证据，而不是重复打开整篇。

建议输入：
```json
{
  "docid": "...",
  "query": "..."
}
```

建议输出：
```json
{
  "docid": "...",
  "matches": [
    {
      "score": 0.0,
      "snippet": "..."
    }
  ]
}
```

作用：
- 降低长文档阅读成本
- 尤其适合：
  - 10-K / annual report
  - dissertation / thesis
  - acknowledgments
  - chapter contents

### 2.4 工具 4：`extract_answer_candidates`

目标：
- 把“候选答案抽取”从 agent 内部启发式逻辑抽成独立工具。

建议输入：
```json
{
  "answer_type": "...",
  "evidence_blocks": ["..."]
}
```

建议输出：
```json
{
  "candidates": [
    {
      "text": "...",
      "source": "...",
      "score": 0.0
    }
  ]
}
```

作用：
- 降低 final answer 阶段自由生成的占比
- 让“抽取”和“裁决”解耦

### 2.5 工具优先级

推荐实施顺序：
1. `verify_claim`
2. `find_in_document`
3. `decompose_question`
4. `extract_answer_candidates`

原因：
- `verify_claim` 最直接提升停搜质量。
- `find_in_document` 最直接提升长文证据利用率。
- `decompose_question` 和 `extract_answer_candidates` 更偏架构升级。

## 3. 多 Agent 架构设计

### 3.1 最小可行方案：Planner + Executor + Verifier

推荐角色划分：

`Planner Agent`
- 输入：原问题
- 输出：
  - `answer_type`
  - `subgoals`
  - `search plan`
  - `verification targets`

`Executor Agent`
- 输入：planner 的分解结果
- 职责：
  - 调用 `search / get_document / find_in_document`
  - 维护 evidence state
  - 生成 candidate shortlist

`Verifier Agent`
- 输入：
  - question
  - candidate answers
  - evidence
- 职责：
  - 判断当前证据是否足够
  - 找出缺失证据点
  - 决定：
    - `finish`
    - 或“返回 executor 再查一轮”

### 3.2 为什么不建议一开始就做并行多 agent

不推荐：
- 多个搜索 agent 并行乱搜
- 多个 planner 同时出方案
- 长记忆协作网络

原因：
- 你当前 BaseDemand 的主要问题不是“算力不够并行”，而是“证据闭环不够稳”。
- 先做串行三角色结构，更容易归因。

### 3.3 推荐的消息流

1. `Planner Agent` 先输出问题分解。
2. `Executor Agent` 根据分解结果做多轮检索与证据收集。
3. `Verifier Agent` 对 candidate shortlist 做验证。
4. 若验证失败，返回：
   - 缺失的实体
   - 缺失的关系
   - 建议下一步查询
5. `Executor Agent` 再补 1-2 轮。
6. 验证通过后统一输出最终答案。

### 3.4 和当前 BaseDemand 的衔接方式

当前最自然的迁移路径不是推翻 [agent/multistep_agent.py](/D:/nju-nlp-deep-research/agent/multistep_agent.py:1)，而是：

1. 保留现有 search / open / final answer 主循环。
2. 把 `_plan_question(...)` 升级为 `Planner Agent`。
3. 把 `_collect_answer_candidates(...)` 升级为独立候选抽取工具或轻量 agent。
4. 在 final answer 前加一个 `Verifier Agent`。

这样做的优点：
- BaseDemand 轨迹格式仍然兼容。
- 可以逐步比较：
  - 单 agent
  - 单 agent + verifier
  - planner/executor/verifier

## 4. 对 BaseDemand 的回溯优化建议

如果 OpenTrack 开始后又想顺手回头优化 BaseDemand，最值得沿着 OpenTrack 设计反哺的点是：

- 把 `verify_claim` 先做成单 agent 内部步骤，再决定是否拆成 verifier agent。
- 把 `find_in_document` 先做成工具，再决定是否让 executor agent 专门使用。
- 把 `decompose_question` 先替换 `_plan_question(...)`，再考虑拆 planner agent。

也就是说：
- 先做“能力模块化”
- 再做“角色拆分”

## 5. 交接结论

OpenTrack-Easy 最合理的起步顺序：

1. 先补 `verify_claim`
2. 再补 `find_in_document`
3. 然后引入 `Planner + Executor + Verifier` 三角色最小结构

不建议的顺序：
- 一上来做并行多 agent
- 一上来做复杂工具森林
- 一上来推翻当前 BaseDemand 主循环

推荐原则：
- 每次只引入一个新能力
- 必须保留可与 BaseDemand 对照的评测链路
- 所有 OpenTrack 增强都应能解释“为什么比当前 BaseDemand 更强”

