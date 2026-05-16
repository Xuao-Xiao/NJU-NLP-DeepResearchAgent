# BaseDemand

本文档总结当前 BaseDemand 定版实现、几轮迭代过程、当前结论，以及后续若要回头继续优化 BaseDemand 时最值得优先处理的方向。

核心代码入口：
- [agent/multistep_agent.py](/D:/nju-nlp-deep-research/agent/multistep_agent.py:1)
- [agent_vllm_multistep.ipynb](/D:/nju-nlp-deep-research/agent_vllm_multistep.ipynb)

## 1. 当前定版总结

### 1.1 多轮检索 Loop + 停止条件

是否实现：
- 已实现。

实现方式：
- 不是纯文字版 ReAct，也不是完全依赖原生 `tool_calls`。
- 当前是“混合式多轮 agent”：
  1. 先用 `QUESTION_DECOMPOSITION_SYSTEM_PROMPT` 做问题拆解，产出 `answer_type + primary/bridge/verification query`。
  2. 第 1 轮固定执行一次 `search(primary_query)`。
  3. 后续轮次由 `_decide_next_action(...)` 决定 `search / get_document / finish`。
  4. 动作执行由代码侧手动落成 `tool_calls` 并写回轨迹，而不是把可靠性完全交给模型的原生工具调用。

当前 loop 的实际控制特点：
- `search` 后优先 `get_document`，避免连续空搜。
- 已打开至少两篇文档且已有短候选答案时，允许提前 `finish`。
- 若达到最大轮数则强制收尾。

停止策略：
- 显式停止，不再是单轮 `search -> answer`。
- 当前实现中的主要停止条件：
  - `round_id >= max_rounds`
  - `stall_count >= 2`
  - 已有候选答案且已检查至少两篇文档时，提前 `finish`
  - planner 或 fallback 明确选择 `finish`

当前结论：
- “有 loop” 这件事已经成立。
- 但停止策略仍然偏保守，真实评测里仍常见 `max_rounds_reached`。
- 下一位接手者如果要继续回头优化 BaseDemand，首要不是再加轮数，而是提高“证据足够时提前停”的判断质量。

### 1.2 上下文管理

是否实现：
- 已实现。

实现方式：
- 没有把每轮原始结果无脑拼到上下文。
- 当前使用结构化 `state` 管理上下文，主要字段包括：
  - `question`
  - `question_plan`
  - `search_history`
  - `seen_docids`
  - `opened_docids`
  - `last_search_results`
  - `search_evidence`
  - `opened_passages`
  - `confirmed_facts`
  - `candidate_answers`
  - `pending_subquestions`
  - `last_action`
  - `stall_count`
  - `finish_reason`

压缩策略：
- `search_history` 只保留最近几条用于摘要展示。
- `opened_docids` 只在摘要中展示最近几条，但完整状态保留。
- `search_evidence` 只存搜索命中的短摘要，不把搜索结果全文当作已确认事实。
- `get_document` 后会做相关片段提取，只把与问题相关的 passages 放入 `opened_passages`。
- `confirmed_facts` 主要由已打开文档贡献，而不是原始 search hit 直接灌入。

摘要策略：
- 每轮给模型的是 `_build_state_summary(...)` 生成的结构化摘要，而不是完整历史。
- 最终回答阶段使用：
  - `Key search evidence`
  - `Opened document evidence`
  - `Confirmed evidence`
  - `Candidate shortlist extracted from evidence`

当前结论：
- 上下文管理已经从“直接堆历史消息”升级为“结构化 state + 证据摘要”。
- 当前最有价值的部分不是“保留最近几轮”，而是“把 opened passages 和 extracted candidates 从 search 噪声里分离出来”。

### 1.3 Prompt 设计

是否实现：
- 已实现，且是分阶段 prompt，而不是单 prompt 包办一切。

当前 prompt 结构：
- `ACTION_DECISION_SYSTEM_PROMPT`
  - 告知当前目标：根据 state 和 recent observation 决定下一步。
  - 告知可用动作：`search / get_document / finish`
  - 告知每步输出格式：严格 JSON。
  - 告知何时停止：证据足够或没有更优下一步时 `finish`。
- `QUESTION_DECOMPOSITION_SYSTEM_PROMPT`
  - 用于把长问题先拆成 `answer_type + primary/bridge/verification query + keywords`。
- `FINAL_ANSWER_SYSTEM_PROMPT`
  - 当前版本已改为严格 JSON：
    `{"exact_answer":"...","confidence":0,"support":"..."}`
  - 明确禁止：
    - chain-of-thought
    - placeholder answer
    - 长句式解释替代答案
  - 明确要求按题型输出：
    - `person`: 只输出人名
    - `company`: 优先输出常用公司名，去掉 `Inc./Corp.` 等后缀
    - `year`: 只输出 4 位年份
    - `percentage`: 只输出带 `%` 的数值
    - `title`: 只输出标题
- `FINAL_ANSWER_REPAIR_SYSTEM_PROMPT`
  - 用于修复脏答案或格式泄漏。

当前结论：
- prompt 设计已经覆盖了课程要求里“当前目标、已知信息、可用工具、调用格式、何时停止”的关键点。
- 当前 prompt 体系最核心的贡献不是“文案更丰富”，而是把任务拆成：
  - 问题拆解
  - 动作决策
  - 最终答案选择
  - 脏输出修复

## 2. 几轮迭代复盘

### 阶段 0：单步 baseline

输入形态：
- 单步 `search -> answer`

已知结果：
- `first_runs` 中单步 RAG 约 `3/50 = 6%`

主要问题：
- 每题只搜一次
- 只看 snippet
- 不打开全文
- 无显式状态
- 无停止逻辑
- 最终答案会混入 `<think>`

### 阶段 1：第一版多轮 agent

主要设计：
- 新增 [agent/multistep_agent.py](/D:/nju-nlp-deep-research/agent/multistep_agent.py:1)
- 引入 `search + get_document + finish`
- 引入结构化 `state`
- 先做一个多轮循环骨架

评测表现：
- `BaseRun0`
- 约 `1/50 = 2%`

暴露出的主问题：
- 绝大多数题几乎没有真正发起工具调用
- 依赖模型原生 `tool_calls` 不稳定
- loop 写出来了，但 agent 实际没工作起来

### 阶段 2：从原生 tool-calling 改为“动作规划 + 手动执行”

主要设计：
- 不再依赖模型稳定地产生 OpenAI 原生 `tool_calls`
- 改成先让模型输出动作 JSON，再由代码落成 `tool_calls`
- 首轮搜索改成确定性执行

评测表现：
- `BaseRun0.1`
- `0/5`

正向变化：
- 工具调用和文档展开明显增多

新暴露问题：
- planner 经常输出非 JSON
- fallback 过多
- 虽然“在工作”，但动作质量很差

### 阶段 3：加强 query rewrite、passage extraction、终答修复

主要设计：
- query bundle：一次 search 扩成多个改写 query
- 文档不再只截开头，而是抽相关片段
- 加入 final answer repair

评测表现：
- `BaseRun0.1.1`
- `2/20 = 10%`

正向变化：
- 在“强命中、少跳推理”的题型上开始能答对

新暴露问题：
- search query 会被 planner 废话污染
- 复杂题仍缺少稳定的实体链定位
- final answer 仍会输出长草稿

### 阶段 4：问题拆解 + 半确定性 search/open 流程

主要设计：
- 增加 question decomposition
- `search -> get_document` 更确定化
- 状态摘要中加入 `answer_type` 和计划 query
- recent observation 中加入真实结果预览

评测表现：
- `BaseRun0.1.2`
- `1/10 = 10%`

正向变化：
- 明显比上一版更愿意打开文档
- 搜索 query 更短、更可控

新暴露问题：
- 大量题仍然跑满轮数
- 终答经常泄漏为 `First... / Looking at... / Alternatively...`
- 说明瓶颈已从“会不会查”转成“会不会把证据变成短答案”

### 阶段 5：当前定版

主要设计：
- final answer 改成严格 JSON 选择器
- 题型识别规则前移，优先判定 `company/person/title`
- 引入证据候选抽取层 `_collect_answer_candidates(...)`
- 对公司名做后缀归一化，如 `FormFactor Inc.` -> `FormFactor`
- 检索重排加入“源类型匹配”偏好：
  - `annual report / 10-K`
  - `dissertation / thesis`
  - `acknowledgments`
  - `chapter / contents`
- 停止策略改为“已有短候选 + 已开至少两篇文档时允许 finish”

已知结果：
- 还没有在你当前线程里补充这版新的正式云端跑分
- 上一轮对比停在 `BaseRun0.1.3` 和 `BaseRun0.1.3_50`

当前结论：
- 当前定版是“BaseDemand 可交接版本”，不是“性能已经足够高的最终最优版本”。
- 它解决的是：
  - loop 存在性
  - state 管理
  - prompt 分阶段
  - 终答草稿泄漏
  - 公司后缀归一化
  - 题型相关的候选抽取
- 它没有完全解决的是：
  - 复杂多跳实体链定位
  - 高置信提前停止
  - 错误文档簇的早期排除

## 3. BaseDemand 交接结论

### 3.1 已完成的课程三项要求映射

`Component 1: 多轮检索 LOOP + 停止条件`
- 已完成。
- 当前是“确定性起搜 + 规划式多轮动作决策 + 显式停止条件”，不是最原教旨的文本 ReAct，但本质上仍是 `Reason + Act`。

`Component 2: 上下文管理`
- 已完成。
- 当前不是拼接原始历史，而是 `state + summary + evidence passages + candidate shortlist`。

`Component 3: PROMPT 设计`
- 已完成。
- 当前 prompt 已明确：
  - 当前目标
  - 已知信息
  - 可用动作
  - 输出格式
  - 什么时候继续搜
  - 什么时候停止

### 3.2 如果下一位接手者要回头继续优化 BaseDemand

最值得继续做的不是：
- 单纯增加 `max_rounds`
- 再往 prompt 里堆更多说明
- 再加更多 fallback 文案

最值得优先做的是：
- 做更稳的“字段抽取式终答”，而不是继续让模型自由生成
- 做更强的文档源类型识别与二次排序
- 做更明确的“候选答案验证后再 finish”
- 针对高频题型做小型专用策略：
  - annual report / 10-K
  - dissertation / acknowledgments
  - chapter / contents
  - spouse / partner / husband / wife

