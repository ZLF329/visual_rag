# Active Clue Graph：可编辑页级记忆与主动回看（方法提案）

日期：2026-11-01。本文是待实现、待验证的方法设计，不代表已有实验结果。

## 1. 核心设定

在扁平页图语料库中，单个 VLM agent 交替执行读图、更新外部记忆与视觉访问。每轮视觉输入至多为一张整页图，或同一页的整页图与一个放大区域。长期记忆保存问题结构、页级事实及其定位和来源；历史图像保存在外部页库，只有模型主动访问时才重新进入上下文。

取消自动 final replay。搜索、回看和裁剪均由同一策略决定；回答直接作为正常 agent loop 的终止动作。

研究假设：页面事实可以持续修订，且模型能按当前推理需求回看来源图像时，有机会在较低的视觉上下文成本下，改善文本记忆的信息遗漏和错误传播。是否优于 replay 必须通过受控实验确认。

## 2. 记忆：问题树 + 共享页节点 + 引用关系

记忆形式为 M=(T,P,E)：

- T：原问题和有序子问题。初始只有原问题 N0，active=N0；先搜索、读图，再由已经获得的部分答案触发 expand。第一版延续根节点加有序子节点，不要求任意深度的嵌套树；任一时刻只有一个 active 问题。环境选择顺序上第一个未完成子问题；没有子问题或子问题全部完成时回到根。重新打开较早的子问题后，active 自动回到它。
- P：已访问页面的唯一页节点。一个真实页面对应一个局部别名 P1、P2……；每页拥有自己的 facts。
- E：问题节点到证据页面的引用，以及由这些引用确定的反向链接。问题节点只引用 P1、P2 等页面，不引用具体 fact。

严格说，T 是树，页面共享后整体是带类型的图。页面被多个子问题使用时，只保存一份事实，避免复制后出现不同版本。

### 2.1 问题节点

字段：id、parent_id、question、status、answer、pages。

- status 采用 open / partial / resolved；证据修订后可重新变为 open 或 partial。
- answer 是由证据推出的工作结论，不是原始视觉事实。
- parent_id 表示这个问题属于哪个总问题。根 N0 的 parent_id 为 null；第一版所有子问题的 parent_id 都是 N0。
- pages 只保存证据页 ID，例如 ["P2"]。fact 和 bbox 始终保存在对应页面节点内部；fact_id 仅供编辑事实或定位裁剪使用。
- 子问题按创建顺序排列；后续子问题直接写入已确定的年份、实体等条件。不再额外维护 depends_on。
- 环境从 parent_id 自动建立 children 并按树缩进展示，模型不重复输出 children 列表。
- 根节点的原问题保持固定，子问题分解和求解状态可以修改。
- 子问题只由 expand 创建：已经有证据回答一部分时，记下该部分，并具体化剩余问题。不在开始时预先创建一套开放子问题。
- 已经存在部分事实、但还不能作答时，可以继续积累，不必把问题强制拆成“本页已答部分 + 剩余部分”。

模型看到的树示意：

```text
N0：增长率第三高的那一年，利润是多少？
├─ N1 [resolved]：增长率第三高的是哪一年？
│  answer: 2011
│  pages: [P2]
└─ N2 [open, ACTIVE]：2011 年的利润是多少？
   answer: null
   pages: []

页面记忆 P2：十年增长率图
  F1: 增长率数据…… bbox=[...]
```

### 2.2 页节点

字段建议：page_id、visual_pointer、descriptor、facts、visit_records、linked_questions。

- visual_pointer 为环境管理的真实文件或缓存地址；模型看到稳定的局部 page_id。
- descriptor 是看图后记录的简短页面描述，用于以后辨认该页；不从真实文件名泄露答案信息。
- facts 为该页的事实集合；每条有稳定 fact_id、text、bbox，以及环境内部维护的 revision。模型通过更新原条目来纠正事实。
- visit_records 至少记录访问时对应的 active 问题和访问顺序；用于解释该页当时为何被查看。
- linked_questions 由问题节点的 pages 反向生成，不要求模型同时维护两份一致的链接。
- 没有事实、没有当前用途的页面仍保留可回看的地址和简短描述；不把“对当前子问题没用”等同于永久删除。

### 2.3 原始事实与推导结论

页级事实记录图像能支持的内容。例如“2011 年 Organic Growth 为 7.5%”。
“2011 年排第三”属于比较推理，保存在问题节点的 answer 中，并通过 pages 关联包含比较所需事实的页面。
若一条事实依赖多个分散区域，可拆成多个事实；不把一个无关小框当作整个推导结论的定位。

## 3. 每轮：先提交记忆补丁，再执行一个动作

（2026-11-01 起协议改为 §11.8 的页中心格式：`page` = {desc, facts, update: resolve | expand}，本节的 op 表和
JSON 示例已被取代；记忆结构和"先编辑、后动作"的原则不变。）

输入：原问题、问题树、页级文字记忆、当前页及可选 zoom、当前坐标帧说明。

模型输出固定结构：memory_edit 与 action。memory_edit 可以为空；每轮至多一个外部动作。

**记忆编辑使用 4 种 op；其中问题操作只有 expand 和 update：**

| op | 放在哪 | 含义 |
|---|---|---|
| `add_fact` | page_ops | 在当前页新增一条事实（新 fact_id、text、bbox） |
| `update_fact` | page_ops | 修改当前页已有事实的 text 和 bbox，保留原 fact_id |
| `expand` | task_ops | 将 active 问题拆成已有证据支持的已答部分和具体的剩余问题；环境创建节点并管理树关系 |
| `update` | task_ops | 按 node_id 修改已有问题的 question、status、answer、pages；不能创建新节点 |

`add_fact` 和 `update_fact` 都提交完整的 text 和 bbox；`update` 提交完整的 question、status、answer 和 pages。不引入局部字段补丁语法。回看或放大后发现旧事实有误，直接 update_fact，不需要先删除再新增；读到另一条新信息才 add_fact。节点的完成与重新打开用 `status` 和 `answer` 表达；增加或解除证据关联用新的 `pages` 列表表达。它们都属于 `update`，不另设 resolve、reopen、link、unlink 等 op。parent_id 与 children 由环境维护，不要求模型在 update 中重复填写。

页面创建、页地址、访问记录、反向链接、内部修订记录及 active 选择由环境维护。第一版不提供删除问题节点的操作。op 只编辑记忆；search / revisit / crop / answer 是另外 4 种 action。

```json
{
  "memory_edit": {
    "page_ops": [
      {"op": "update_fact", "fact_id": "P1.F1", "text": "Trading operating profit = CHF 12.5 bn", "bbox": [40, 60, 520, 140]}
    ],
    "task_ops": [
      {"op": "update", "node_id": "N2", "question": "What was the trading operating profit in 2011?", "status": "resolved", "answer": "CHF 12.5 bn", "pages": ["P1"]}
    ]
  },
  "action": {
    "type": "answer",
    "text": "CHF 12.5 bn",
    "support_nodes": ["N1", "N2"]
  }
}
```

以上假设 P1.F1 已存在，回看或放大后修正其内容。坐标是假设当前展示帧上的坐标，不是实际标注。可保留现有短分析前缀；状态更新只依据结构化补丁。

### 3.1 编辑边界

- 当前页可添加或修订事实；环境绑定页 ID，模型不能给未显示的页面凭空写入新的视觉事实。
- 所有问题节点都可关联已看过的证据页面，也可解除不再适用的页面关联。
- 要修改旧页面的事实内容或 bbox，先 revisit 该页；仅调整任务关联无须重新看图。
- fact_id 稳定，环境保留内部修订历史，模型只看到当前版本。没有单独的删除事实操作。
- 已有关联页面的事实被修改时，环境将引用该页的已完成子问题及其后续已完成子问题保守地重新打开，并清空旧答案。新增事实不触发这个处理；当前轮 update 可以重新填写经复核的结论。这样无需额外维护事实级依赖图。
- 补丁和动作先完整校验，再原子提交。无效输出不能留下部分更新；有效记忆提交后，即使检索没有返回新页，也保留已提交的记忆。

### 3.2 bbox 规则

第一版延续 Qwen2.5-VL 的展示像素坐标：整页轮使用整页帧，zoom 轮的事实框使用 zoom 帧，由环境确定坐标帧。入库统一转换到原页规范坐标，并记录 resize/crop 变换。

历史记忆可以展示 fact_id、文字与规范 bbox，但必须注明历史 bbox 的坐标帧，不能直接当成当前显示尺寸上的 crop 参数。crop 可接收当前页的 fact_id，由环境解析其位置；这不增加新的动作类型。

## 4. 四种视觉访问与结束动作

| 动作 | 参数 | 效果 |
|---|---|---|
| search | query | 从全库返回最高排名的未见页；新建唯一页节点 |
| revisit | page_id | 按稳定指针载入已经看过的整页；复用原页节点 |
| crop | bbox 或当前页 fact_id | 展示当前整页与其一个放大区域 |
| answer | text、support_nodes | 从当前记忆和当前视觉观察回答原问题，结束；支持节点明确包含所用的各步子问题 |

- revisit 不依赖搜索召回，不能指向从未看过的页面。
- 第一版 revisit 整页，需要细读时再 crop；暂不加入直接打开任意旧区域的第五种动作。
- 一次连续页面访问内最多 crop 一次；crop 后可以更新事实，再 search / revisit 另一页 / answer。
- 同一页日后因新的问题需求被 revisit，可以查看另一个区域；一次访问结束后旧图像从模型上下文移出。
- 禁止用 revisit 当前页刷新 crop 配额；总动作数、访问次数和视觉 token 预算共同约束循环。
- 已访问页目录始终提供稳定地址；大量历史事实可按 active 关联展开，但不能无提示截断全部页面地址。

## 5. 边搜索边 expand，已有节点用 update

保留旧 AGv3 的证据驱动拆分方式，同时允许记忆持续更新。移除 staged facts 和“离页才提交一次决定”的约束；crop 前后均可编辑记忆。

### expand：已经解决一部分，继续追踪剩余问题

初始树只有 N0：“增长率第三高的那一年，利润是多少？”；先围绕原问题搜索。读到 P2 的增长率数据，确认目标年份为 2011 后，输出：

```json
{
  "op": "expand",
  "answered": "增长率第三高的是哪一年？",
  "answer": "2011年",
  "pages": ["P2"],
  "next": "2011年的利润是多少？"
}
```

环境将树更新为：

```text
N0：增长率第三高的那一年，利润是多少？
├─ N1 [resolved]：增长率第三高的是哪一年？ answer=2011年，pages=[P2]
└─ N2 [open, ACTIVE]：2011年的利润是多少？
```

expand 始终作用于 active，模型不填写新 node_id 或 parent_id。active 是根时，保留原问题并创建两个子节点；active 是子问题时，将该节点改为已答部分，在其后插入新的剩余子问题。第一版均挂在 N0 下，沿用星形树。每轮最多一次 expand；若当前只读到片段、尚未解决任何部分，只保存事实或 update 为 partial，继续搜索即可。

### update：补充、完成或纠正已有问题

之后回看 P1，读到利润，可以用 update 将 N2 设为 resolved 并填写答案和 pages。若新证据推翻旧结论，也用 update 修正问题、答案或重新打开节点。update 保留节点 ID 和树关系，不产生新节点。

单跳问题直接 update N0 后 answer，无须 expand。页面当前没用时，仅保留访问记录并继续搜索，无须输出 reject。

## 6. 直接回答与主动回看

单页、多页使用相同的 answer 协议。answer 不触发重载全部历史图像，不自动拼接子节点答案，也不把最后一个子问题的答案直接作为原问题答案。

模型可以从事实与工作结论直接完成原问题；若发现遗漏、矛盾或需要重新辨认图表，必须在 answer 前主动 revisit，必要时 crop，再修订记忆。

最终支持页 E_final 是 answer.support_nodes 中各节点 pages 的去重并集，不再计算事实级依赖闭包。仅被浏览、没有用于答案的页面不算最终证据。引用页面本身只能证明来源地址有效，不能证明该页的事实正确或足以支持答案；事实质量单独核验。

达到预算时只允许一次明确的 answer 输出，不能重新启用多页 replay；同时记录 budget_exhausted，与自主完成分别报告。

## 7. SFT

用新协议重新采样教师轨迹，覆盖：

1. 直接检索并作答，不需要回看。
2. 后续问题使早先看过但未使用的页面重新相关。
3. 回看后从同一页读出新的事实。
4. 回看与裁剪纠正已有事实，并重新求解相关子问题。

不要求每条轨迹都 revisit，避免把主动选择学成固定结束程序。

保留条件：协议有效、最终答案由评测器判对、gold 页包含在最终支持页集合 E_final 中。仅访问过 gold 页或仅把页 ID 挂上树不构成事实正确性的验证；有事实标注或可用核验器时，还应核对事实与图像的一致性。

按学生实际看到的每轮重建上下文构造训练调用；目标为教师的 memory_edit 与 action。该轮输入只能含历史已提交的记忆，不能含教师本轮或未来写入的事实。取消独立 final replay 样本。

## 8. RL：奖励证据质量，控制访问成本

终局奖励建议：

R = R_answer + λ_cov R_coverage + λ_cmp 1[full coverage] R_compactness
    + λ_ground R_grounding - λ_cost C_access。

- R_answer：沿用固定的二值答案评测协议；记录评测器及版本。
- R_coverage / R_compactness：基于 E_final 的不同来源页计算，不基于 visited pages；明确 gold 不穷尽所有有效证据时，非 gold 页惩罚只是近似。
- R_grounding：对语义匹配的事实评估其对应框。框在 gold 区域内不能单独证明事实正确；只对可靠标注计算。teacher 框作为弱监督时须明确区分。
- C_access：归一化的搜索、回看、裁剪和视觉 token 成本；权重由验证集校准，避免让模型完全放弃回看。

移除旧的 gold 决定类型匹配奖励。新的 expand 取决于当前问题已有哪一部分被实际证据解决，update 则还承担纠错和补充；不能按“是否为 gold 页、是否已检齐 gold”来指定唯一正确的操作类型。

不为 revisit、更新次数、重复看到 gold 页本身发正奖励。第一版可用终局奖励与小幅访问成本学习该策略；若增加步级 shaping，应基于经过核验的有效证据状态变化，并采用折扣一致的 potential shaping 或独立一次性计分，避免反复改写同一事实或反复访问刷分。未标注的语义进展不能冒充可计算的逐步正确性。

## 9. 与当前 AGv3 的主要改动

| 当前 | 新提案 |
|---|---|
| facts 归属于已完成问题并冻结 | facts 归属于唯一页节点，可以修订 |
| reject 丢弃 staged facts | 保留页面地址；是否使用由支持引用表达 |
| active 固定向前推进 | 结论可重新打开，active 自动选择首个未完成子问题 |
| 证据驱动 expand，完成后不可变 | 保留证据驱动 expand，增加已有问题的 update |
| 离页一次性提交 | 每轮原子 memory patch |
| search / crop / answer | 增加 revisit |
| 多页 final replay | 同一循环直接 answer |
| done_pages 作为证据集合 | 最终支持节点的 pages 去重并集 |
| gold 决定类型奖励 | 去除；使用证据质量和访问成本 |

主要涉及 clue_graph.py（状态和引用）、protocol.py（补丁和动作）、prompts.py（每轮上下文）、agent.py 与 rl/envs.py（统一状态流转和结束路径）、SFT 生成与过滤。

## 10. 必要验证

固定 backbone、检索器、训练样本规模、答案评测器及总视觉访问预算，比较：

- 原 AGv3 自动 replay。
- 页级可编辑记忆，无 replay、无 revisit。
- 页级可编辑记忆与主动 revisit，完整方案。
- 页面文本和可回看地址保留，但去掉 bbox 定位或冻结事实更新，分别验证定位和修订的贡献。

报告整体/单跳/多跳答案准确率、gold 覆盖、可靠标注上的事实定位质量、搜索/回看/裁剪次数、总视觉 token、延迟。回看之后变对只是一种相关性；回看的因果收益需要禁用回看的配对消融支持。

## 11. 实现决策（2026-11-01，第一版代码）

代码在本目录（`src/ rl/ scripts/ tests/`）。草案未定或需要落成具体规则的地方如下；以后的改动在本节后追加，不改写。

**格式**
- 每轮 `<think>…</think>` + `<act>{"memory_edit": {"page_desc", "page_ops", "task_ops"}, "action": {…}}</act>`。memory_edit 内三个键可省（默认 null / []），教师目标经规范化后三个键总是写全。page / fact / node id 大小写不敏感（`p1.f2` → `P1.F2`）。
- 字段名沿用草案：`add_fact {text, bbox}`、`update_fact {fact_id, text, bbox}`、`expand {answered, answer, pages, next}`、`update {node_id, question, status, answer, pages}`。update 五个字段都必须给（answer 可为 null）；resolved 必须有 answer；expand 的 pages 至少一页；每轮最多一个 expand。
- **descriptor**：通过 `memory_edit.page_desc` 写当前页的描述（null 保持不变）。它是字段不是 op，4 种 op 不变。没写过描述的页渲染为 `(no description)`，同时总显示首次检索它的 query。
- **根问题固定**：update N0 时 question 字段被忽略。

**一轮内的执行顺序**：page_desc → page_ops → 事实被修订的页触发重开 → task_ops（按给出顺序）→ 校验 action → 原子提交 → 执行 action。所有校验在副本上完成，失败时记忆不变。

**重开规则**：只有 update_fact 触发。找到第一个引用该页的 resolved 节点，把它和之后所有 resolved 节点重开为 open（顺序是子问题依次排列，root 在最后）；answer 清空，pages 保留。同一轮的 update 可以把结论填回去。add_fact 不触发。

**active**：第一个未 resolved 的子问题；没有子问题或全部 resolved 时是 N0。

**上下文渲染**：问题树用 AGv2 的 `Graph nodes:` 布局，节点上显示 `pages:`（不显示 facts）。另有 `Pages seen:` 目录：页 id、描述、首次 query、访问次数（大于 1 时）、关联问题，以及每条事实 `P1.F2: text`。**不渲染 bbox**，定位靠 `crop {fact_id}`；**不出现真实页名**。第一版渲染全部页和全部事实，不截断。

**坐标帧**：整页轮用整页像素，zoom 轮的 page_ops 用 zoom 像素（由轮次决定，模型不必声明）。observation 和每轮提示都写明当前图的宽×高，并注明不是 0-1000（实测 qwen3.8-27b 只有写明尺寸才输出像素）。入库统一为页帧，并记录 `src: page|zoom` 和修订历史。

**动作**
- search：取 top-k 中第一张没看过的页，建新页节点；没有新页时不展示页，并给出提示。
- revisit：可以回看任何看过的页，但不能是当前正在看的那页；展示整页，清掉 zoom，crop 配额重置。
- crop：每次访问一次。参数是 `bbox`（整页像素）或 `fact_id`（取该事实入库的页帧框）；加 28px pad，放大到至少 500k 像素。
- answer：参数 `text` 和 `support_nodes`（可以为空）。E_final = 各支持节点 pages 的去重并集。

**错误车道**
- 硬错（ProtocolError）：无法解析、schema 不符、结构规则违反（无页时写 page_desc / page_ops 或 crop、同一次访问内第二次 crop、revisit 当前页、预算轮不是 answer）。eval 允许一次格式重试，仍失败则 `format_error` 结束；RL 直接 −1 并终止。
- 软错（EditError）：引用不存在的 page / fact / node、对非当前页的事实做 update_fact、框在图上不可用、crop 的 fact_id 不属于当前页。整轮不生效，下一轮带提示重试，计入轮数；RL 记步级惩罚 `AG_SOFT_ERROR_REWARD`（默认 −1）。

**预算**：`max_turns` 个正常轮之后，追加一个强制作答轮。这一轮的提示写明预算用尽，仍可以编辑记忆，但 action 必须是 answer。结束状态分为 `answer` / `budget_exhausted` / `format_error`；强制轮出现软错时，以无答案结束。视觉 token 预算暂未实现，目前靠每轮最多 2 张图兜底。

**奖励（rl/envs.py）**
- R = 1[correct] + COVERAGE_W·cov + CMP_W·1[cov=1]·cmp + GROUNDING_W·R_grounding − ACCESS_COST_W·C_access，默认 0.5 / 0.2 / 0.5 / 0.1。
- cov = |gold∩E_final| / |gold|（取比例，同 SCORE）；cmp = |gold∩E_final| / |E_final|。只看过但没被支持节点引用的页不计入。
- C_access = (search + revisit + crop) / max_steps。
- R_grounding：每个 gold key-fact 框，与 E_final 页上 `src == page` 的事实框按同页取最大 IoU，再取平均。zoom 帧的框不计，没有 keyfacts 时为 0。第一版不做语义匹配。
- 去掉所有正向步级奖励：旧的 decision 类型匹配、search rank-1 新 gold 页都删了，多跳 gate 也删了。budget_exhausted 的答案照常打分，只额外标记。
- judge 的支持事实取 E_final 各页的事实。

**SFT**：保留条件是 terminated_by == answer（预算耗尽的不要）、无错误轮、答案非空、judge 判对、gold ⊆ E_final。每个生效的轮次一条样本。每条轨迹打上标签（revisit / crop / update_fact / reopened / expand 次数），用于按 §7 的四类轨迹配比。教师默认 qwen3.8-27b + 像素帧；保留 `bbox_frame: norm1000` 适配器，给 Qwen3-VL 系教师用。

**未实现 / 待验证**：视觉 token 预算；页目录过长时的截断策略；R_grounding 的语义匹配；SFT 中事实与图像一致性的核验。RL 侧的 step group 名是 action 类型（search / revisit / crop / answer），trainer 侧需要相应配置，第一版建议 `step_advantage_w=0`。

### 11.1 渲染与事实编号（2026-11-01 追加，替换 §11 中"上下文渲染"一条）
- 问题树改为树形：`Question tree:` 下面先是 `N0 [status]: 问题`，再按顺序列出 `├─ / └─ Nk [status]: 问题`，节点下写 `answer:`、`pages:`（为空就不写），当前问题标 `ACTIVE`（如 `[open, ACTIVE]`），节点之间用 `│` 隔开。
- 页面记忆改为每页一块：`Page memory:` 下是 `P1: 描述   (found by: "query")`，再用 `├─ / └─ F1: text` 列出该页的事实。依旧不渲染 bbox，也不出现真实页名；去掉了 linked 和访问次数，因为这些信息在问题树里已经有了。
- **事实编号改为页内编号**：模型看到和引用的都是 `F1`、`F2`。update_fact 和 crop 的 fact_id 指的是当前在看的那页上的事实，所以写 `F2` 就够了，写 `P1.F2` 也接受。内部存储仍用 `P1.F2` 保证全局唯一。

### 11.2 只依据检索结果作答（2026-11-01 追加）

- `POLICY_SYSTEM` 的 Rules 第一条：答案只能来自检索到的页面（页面记忆里的 facts + 当前在看的页）；
  不用外部知识、不猜；页面还撑不起答案时继续 search / revisit / crop。
- 预算耗尽的强制回答轮提示同步改为"只陈述页面记忆支持的内容"。
- 动机：teacher 轨迹要教学生"证据驱动"的回答，而不是靠模型先验蒙对；SFT 过滤里的
  judge 正确 + gold ⊆ E_final 仍然保留，两者互补（规则约束生成，过滤剔除漏网的）。
- 改了 prompt，之前生成的 SFT 轨迹（如有）作废重生成。


### 11.3 crop 与坐标帧：对齐 SCORE / VISOR，并按模型原生尺度（2026-11-01 追加，替换 §11 的"坐标帧"和 crop 两条）

来源：SCORE 论文附录 B（未开源）；VISOR 代码 `vrag_agent/generation.py` 的 crop 分支与
`scripts/data_construct_pipeline_v3.py`（与 SCORE 同一作者；commit 4140198）。

- **图片预算**（VISOR `process_image(512*28*28, 256*28*28)`，SCORE "fixed max_pixels cap" / teacher
  256–512×28×28）：模型看到的每张图（整页、crop）都缩放到 [256·28², 512·28²] 像素。缩放用 Qwen 的
  smart_resize，边长对齐到该模型的 patch 网格（Qwen2.5-VL 28，Qwen3 系 32），所以模型自己的
  processor 不会再改尺寸，"展示的图"就是模型真正的坐标帧。原图（整页、crop 区域）全分辨率保存。
- **坐标帧 = 模型原生尺度**：
  - `qwen2.5-vl`：展示图上的绝对像素（Qwen2.5-VL grounding 的原生约定），每轮提示写明宽×高。
  - `qwen3`（Qwen3-VL / Qwen3.5 / Qwen3.8）：0–1000 相对坐标（Qwen3 原生约定，也是 VISOR teacher
    的约定）。提示里不写像素尺寸，避免诱导它输出像素。
- **规范存储**：所有 bbox（fact、crop 区域）一律存原图像素。模型框 → 单位坐标（px 除以展示尺寸，
  norm1000 除以 1000）→ 原图像素；zoom 轮的框先映射到 crop 区域再到原图。R_grounding 的 gold
  （keyfacts bbox_2d）也是原图像素，口径不变。
- **crop**（SCORE 附录 B）：当前页上的框 → 线性映射回原图 → 四周各加 28 个**原图像素** → 裁到页面
  边界 → 从原图裁出 → 按图片预算缩放后展示。展示后小于 4 像素宽/高的框是软错误，提示重发
  （SCORE："error message that asks the model to reissue"）。`crop {fact_id}` 用该 fact 的原图框走
  同样流程。每次访问只能 crop 一次、不能 crop 一个 crop（= VISOR "cannot use bbox on a cropped image"）。
  - 与 VISOR 代码的差异：VISOR 只在验证时、在展示帧里加 28px；我们按 SCORE 论文，映射后在原图上、
    始终加。原来"放大到 ≥ 500k 像素"的规则取消，改用统一图片预算。
- **跨帧蒸馏**：teacher 用自己的原生帧跑（qwen3：0–1000）；每条 SFT 样本按学生帧（qwen2.5-vl：
  展示像素）重新渲染：图片按学生帧缩放后单独落盘、每轮提示写学生帧、上一轮输出和 target 里的每个框
  都经原图坐标换算到学生帧。VISOR 把 teacher 的 0–1000 框原样抄进 SFT，而它的 RL 环境按像素解读，
  两边不一致；我们换算。
- RL 环境：策略帧 = 学生帧（默认 qwen2.5-vl）。
- 配置：`agent.frame`（跑的模型）、`agent.sft_frame`（SFT 目标，缺省同 frame）；RL 用
  `slidevqa.frame`。旧的 `bbox_frame` 去掉。

### 11.4 teacher 的推理 = 原生思考（2026-11-01 追加）

- Qwen3.8 关闭思考时，对话模板会在回复前插入空的 `<think></think>`，模型随即跳过协议要求的
  `<think>`，冒烟测试 8/8 格式错误。改为打开原生思考：`reasoning_content` 由 `src/vlm.py` 包成
  `<think>…</think>`，正文就是 `<act>`。实测每轮推理 20–150 token，SFT 目标长度可控。
- 同日追加：answer 的 text 只写答案本身（一个词、数字或短语，不加解释），对齐 VISOR prompt 的
  "without detailed illustrations"。冒烟测试里 teacher 常给整句答案，学生会照学。

### 11.5 检索实验开关 R1 / R2a（2026-11-01 追加；val 上选定后删掉落选分支）

依据：test 错误分析（380 错，73% 检索失败；167 例 gold 在模型自己查询的第 2–5 名；首个查询几乎从不等于
原题，而原题检索在 54 个单跳错例里把 gold 排第 1），并与 Codex（gpt-6-astra）讨论定稿。

- `agent.first_search: model | question`（RL：`slidevqa.first_search`）。`question` = 第 0 轮由环境用原题
  搜索（SCORE §3.2："the initial search is constrained to the original user query"），模型从第 1 轮开始，
  直接看到 P1。固定回复 `first_search_response()` 作为"上一轮输出"放进上下文；这一步不调模型、不产生
  SFT 行，计一次 search 访问，占一轮预算。
- `agent.search_guidance: false | true`。`true` 时 system prompt 的 search 说明改为：返回该查询下"没看过的
  最高排名页"，语料混着很多无关文档，同一查询再搜一次看下一名；Rules 加一条：只有确定这一页包含当前问题的
  答案（对得上问题里的名称、标题、日期、数字、图中物体；"this presentation / the ... shown" 指某一份文档而
  可能有多份符合；同一文档的页有相同的标题样式、页脚或 logo）才 resolve 或回答，不确定就用同一查询看下一名，
  或用更独特的线索再搜。配套示例：搜到别家公司的同类页 → 同一查询再搜（`false` 时同位置是原来的"无关页 →
  换查询"示例）。
- 原计划在 val 上分臂比较（R0 / R1 / R2a）；用户决定不做分臂，直接在 test 上跑合并后的最新版本：
  `config/latest.yaml` = 两个开关都开；§11.7 的 validate 动作是协议的一部分，始终启用。

### 11.6 system prompt 精简（2026-11-01 追加）

R0/R1/R2a 都用精简后的 prompt（字符数 8.1k → 6.5k，R2a 8.8k → 7.1k）。删掉的只有重复内容，协议不变：
- Rules 里"无页在视时 page_desc/page_ops 为空""zoom 在视时不能再 crop"：每轮的 turn hint 都会说，违反时
  协议直接拒绝并给错误提示。
- "不能 revisit 当前页""单跳先 resolve N0 再回答""不要凭猜测 expand"：并进对应动作/操作的说明里。
- 答案说明里"有缺失/矛盾就先 revisit 或 crop"：与第一条 Rule 重复。
- 第一个示例（无页时搜索）：R1 下第 0 轮由环境完成，后面的示例也展示了 search 格式。
- 记忆说明里 "Page in view" 一行（状态块里直接写着）、`next` 的示例（第一个示例里有）、坐标规则的重复措辞。
保留：全部操作/动作的格式、短答案和"只依据检索页回答"两条约束、其余 6 个示例（expand、crop、zoom 后作答、
revisit + 改 fact + 恢复答案、无关页、单跳）。


### 11.7 validate 动作（2026-11-01 试过，已取消）

同日先后试了两版"多页证据再看一遍"：答完后自动检查（<validate> 标签，可改 fact）、模型主动的 validate 动作
（先带参数，后不带参数）。冒烟测试都能跑，但用户最终决定不要 validate，改回主动 revisit（§11.8）。
相关代码已全部删除。

### 11.8 页中心编辑格式：resolve / expand，页自动挂到问题上（2026-11-01 追加；用户设计，当前协议）

动机：原格式里页和问题的连线有两处（page_ops 的读数、task_ops 的 `pages` 列表），模型会出现"引用了没读过的页"
或"读了页却没挂到问题上"。新格式一轮只写一个 `page` 对象，说明眼前这页对当前问题做了什么，页由环境自动挂上。

```
<think>...</think>
<act>{"page": {"desc": ..., "facts": [...], "update": ...} | null, "action": {...}}</act>
```

- `page`：眼前这页的贡献；没有页在视时必须是 null（或全空）。
  - `desc`：页描述，null 不改。
  - `facts`：这页（或其 zoom）上的读数：`add_fact` / `update_fact`（只能改眼前这页的 fact，页内编号 F2）。
  - `update`：这页对 ACTIVE 问题的结论，或 null：
    - `{"op": "resolve", "answer": ...}`：这页回答了当前问题 → 问题 resolved，眼前这页挂上（并集）。
      可选 `"node_id": "N1"` 改为回答/修正另一个问题（revisit 后修正旧结论用）。
    - `{"op": "expand", "answered": ..., "answer": ..., "next": ...}`：这页回答了一部分 → 已答子问题（resolved，
      眼前这页挂上）+ 未答子问题（open）。语义同原 expand，去掉 `pages`。
  - 没有 `pages` / `status` / `question` 字段；状态只有 open / resolved（partial 取消）。
- 动作：search / revisit / crop / answer。`answer` 只带 `text`，不再有 `support_nodes`；answer 把 N0 置为
  resolved、answer = text（不挂页）。
- **最终证据页 E_final = 问题树所有节点挂的页的并集**（节点顺序去重）。coverage / cmp / grounding 都在它上面算，
  reward 公式不变。模型没 resolve 过任何问题 → E_final 为空（同原来 support_nodes 为空）。
- 改 fact 不再触发重开（§3.1 的级联重开取消，实测 reopen ≈ 0）：要改旧结论就用 `resolve` + `node_id`。
- 硬规则（格式错）：无页在视时写了 desc / facts / update 或 crop；zoom 在视时再 crop；revisit 当前页；预算轮不是
  answer。软错误（不应用、重试）：框不可用、未知 fact / node_id、改别页的 fact。
- 渲染：问题树不变；页记忆每页标出它挂到的问题（`P1: ...   → N1`）；turn hint 相应改为 `"page": null`。
- SFT：一轮一行不变。RL 环境共用同一套协议代码。

### 11.9 先读过的页靠 expand + revisit 挂上（2026-11-02 追加；用户方案，替代"uses"字段提议）

问题：test_page_v1（85.06%）的"金标准页都在最终证据里"只有 73.9%（旧版 76.2）。单跳 88.0 正常；多跳看到全部
金标准页 55.7%，挂上的只有 32.8%。145 题"看到却没挂上"里 137 题是单节点树：模型先在 P1 记下读数（当时不确定，
`update` 为 null），后来在 P2 上直接 resolve N0，答案用了 P1 的读数，但只有 P2 挂上（152 个漏挂页里 142 个有读数）。

决定：不加 `uses` 字段，不改协议，只改 prompt。
- 规则：答案用到哪页的读数，哪页就要挂到问题上；页只能在眼前时由 `update` 挂上。当前问题还要用到之前某页的读数时，
  先在眼前这页 expand（answered = 这页答的部分，next = 剩下的部分），再 revisit 那页并在那页上 resolve。
- 例子：原来"revisit 修正 fact"的两段换成这个模式（同一张图，先 expand，再 revisit 先前读过的 KPI 页 resolve）。
  update_fact 和"修正"的用法仍在格式说明和 revisit 描述里。
- 理由：保持"页只在眼前时挂上"；多跳的问题树真的拆开；revisit 成为多跳的自然一步，SFT 天然有回看数据；
  RL 里挂页要花一轮访问成本，没有零成本刷 coverage 的口子（`uses` 有）。
- 代价：这类题多一轮（全量约 +0.07 轮/题）。只靠 prompt，teacher 是否照做要实测：先在这 145 题上跑，再跑全量。

### 11.10 试验开关 earlier_pages: revisit | uses（2026-11-02 追加；用户要求和 §11.9 同题对比，选定后删掉落选分支）

- `uses` 版：resolve / expand 可带 `"uses": ["P1", ...]`，列出答案还依据的、之前看过的页；这些页和眼前这页一起挂到
  被 resolve 的问题上（expand 时挂到已答的子问题）。只能列看过且有 fact 的页，否则是软错误；列了眼前这页等于没列。
- prompt：uses 版把 §11.9 的规则和"expand + revisit"两段例子换成一条规则加一段例子（图表页直接 resolve，带 uses P1），
  revisit 动作的说明回到"re-read a value or correct a fact or an answer"；其余逐字相同。revisit 版不提 uses。
- 配置 `agent.earlier_pages`，默认 revisit；`config/exp_uses.yaml` 切到 uses。协议两种模式都接受 uses 字段。
- reward 公式不变。注意：uses 挂页不花访问成本，RL 若选 uses 版，要靠非 gold 挂页计数（已在报）或惩罚防刷 coverage。
- 对比：同一组 145 题（`test_gap145.jsonl`，§11.9 那批），看挂全率、准确率、回合数、非 gold 挂页数。
- 结果（145 题，同配置；改前 = test_page_v1 的同一批题）：

  | | 改前 | revisit 版 | uses 版 |
  |---|---|---|---|
  | 金标准页全部挂上 | 0% | 43.4% | **66.2%** |
  | 准确率 | 82.8 | 80.0 | 81.4 |
  | 平均回合 | 4.23 | 4.61 | 4.07 |
  | 非 gold 挂页 / 题 | 0.11 | 0.16 | 0.20 |

  uses 版 64% 的题用了 uses，列出的 98 页里 88 页是 gold；软错误 1 次。准确率差异在 145 题的噪声内（±3 点）。
- **决定（2026-11-02，用户）：用 uses 版，删掉 revisit 版。** 开关 `agent.earlier_pages` 和 `config/exp_uses.yaml` 删除，
  uses 成为协议本身（§11.9 的"expand + revisit"规则和例子作废；revisit 动作保留，用于重读和修正）。下一步：全量 test。
- **全量 test 结果（2026-11-02，uses 版，2215 题）：Qwen3.8-27B 85.19%**（单跳 87.44 / 多跳 78.66；page_v1 为 85.06，持平），
  金标准页都在最终证据里 73.9 → **79.1%**（多跳 32.8 → 53.1），非 gold 挂页 0.13/题（0.12），回合 2.62 / 搜索 1.51 不变，
  11.2% 的题用了 uses、列出的页 86% 是 gold，格式错 8。同一协议换 Qwen3.5-27B（`test_uses_qwen35`）只有 81.90%
  （SCORE 提示法 82.53）：格式错 57 题（32 题把 <act> 写进思考段、6 题空 <think>）约损 2 个点（剔除后 84.06），
  预算用完 68 题，crop 仅 3.2%、搜索更多（1.68）。Qwen3.5 若要可比，需先处理它的输出格式（think 内 <act> 的解析或重试）。

### 11.11 prompt 精简 + 容错解析 + 必答轮（2026-11-02 追加；用户要求，动机是 Qwen3.5-27B 的 57 次格式错）

Qwen3.5-27B 跑 uses 版 81.90%，57 题因格式错提前结束（Qwen3.8 只有 8）。原始输出分类：32 次在 `</act>` 后多写一个
`</think>`；11 次 content 里又带了一个 `<think>…</think>`（和折叠进来的 reasoning 合起来两个 think 块）；6 次 `<think>`
为空（有时后面再跟一个真正的 think 块）；6 次 JSON 后多一个 `}`；2 次真坏（缺逗号、action 写进 page 里）。重试一次也
同样失败，因为问题出在模型的原生 thinking 输出习惯上，提示再怎么改也压不住。决定三件事：

1. **解析容错（protocol.parse_turn）**：think = 所有 `<think>…</think>` 块内容拼接（允许为空）；act = 最后一个
   `<act>…</act>` 块，JSON 用 `raw_decode` 取第一个完整对象、忽略尾部多余字符；块外的文字和多余的 think 标签忽略。
   仍是格式错的只有：没有 act 块、JSON 本身坏、schema 错、违反视图规则。SFT 目标不再用原文，统一用规范渲染
   `render_turn(think, act)` = `<think>{think}</think>\n<act>{canonical json}</act>`（`rewrite_act` 删除）。
2. **必答轮（agent + env）**：预算轮（turn == max_turns）照旧可改记忆、动作必须 answer。若预算轮失败（格式错、
   软错误、非 answer），不再直接以空答案结束，而是再给一轮 answer-only：提示"上一条无法使用，现在只输出答案"，
   `page` 必须为空、动作必须 answer；成功 → terminated_by = budget_exhausted；再失败 → 空答案、format_error（同原来）。
   env 同步：预算轮失败给一步软错误奖励（不终止），answer-only 轮成功按正常终局计分，再失败 −1 终止。
   有错误步的轨迹仍不进 SFT（ERROR_STEPS 规则不变）。
3. **prompt 精简**：例子全部保留，但每段 `<think>` 缩成一句；规则只留三条（只依据页面、用到的页必须挂上、
   确定才 resolve / 不确定看下一页）；题目脚注缩短。目标从 1943 token 降到约 1500（SCORE 约 716）。
   改完 Qwen3.5 全量重跑验证格式错归零，Qwen3.8 全量重跑验证不掉点（基线 85.19）。
- **结果（2026-11-02，Qwen3.5-27B 全量 `test_v2_qwen35`）：83.93%**（单跳 86.89 / 多跳 75.31；改前 81.90 / 85.13 / 72.49，
  逐题净 +45），格式错结束 57 → **3**（剩下的是 action 嵌进 page、JSON 缺引号、预算轮只输出纯文本各一类），软错误 0，
  金标准页都在最终证据里 77.6 → 79.0，回合 / 搜索不变。用户决定**不跑 Qwen3.8 的回归轮**（接力脚本按 PID 停掉），
  所以精简 prompt 对 Qwen3.8 的影响未验证；85.19 仍是旧 prompt 的数。
  Qwen3.5 剩余 356 错题的原因：检索 235（66%；单跳 139 题里 104 题是搜 1–2 次就拿错页作答，122 题 Qwen3.8 同样没检到；
  多跳 96 题里 68 题没 expand、看完第一跳就作答），读图/推理 100（28%；61 题读错元素、30 题数值读错，其中 38 题
  Qwen3.8 答对），挂页 18，格式 3。预算用完 73 题（答错 55，平均搜 9.4 次，必答轮没触发）。

### 11.12 文件名可见 + 桥接题先拆 + 例子再精简（2026-11-02 追加；用户决定，针对 Qwen3.5 的检索类错误）

Qwen3.5 的 356 道错题里检索环节占 235：单跳 104 题搜 1–2 次就拿错页（多为另一份相像的文档）作答；多跳 68 题没 expand、
看完第一跳就答。用户否决了"限定同文档搜索"（把数据集先验塞进检索器，SCORE 的 agent 没有这个能力，不公平）。决定：
1. **页的文件名对模型可见**（页记忆头和 observation 行都显示 `P1 [deck/page_05]`）。SCORE 的证据账本每条就是
   `File: {image_basename}`，rerank / 最终作答 prompt 也带文件名，所以是同一口径；§2 原先"不出现真实页名"的顾虑
   （slug 可能带标题信息）接受，与 SCORE 持平。作用：判断眼前这页和之前的页是否同一份文档（"this presentation"），
   第二跳可以自己把 slug 里的关键词写进 query（不是硬限定范围）。
2. **桥接规则**并入规则一：问题里间接指代的东西（the company that…、the year when…）必须先从页面 resolve（expand）
   再找剩下的部分，不能用常识补。
3. **例子再精简**：字符串值用省略号缩短（desc、长 fact、子问题文本），框坐标保留真实值（避免模型照抄 `[...]`）。
   规则仍三条。验证：先在 Qwen3.5 的 235 道检索错题子集（从 test_v2_qwen35 导出）上跑，再全量。
- **235 题试跑（2026-11-02，Qwen3.5，`miss235_v3_qwen35`；这批题改前全错）：答对 38（单跳 11/128，多跳 27/107）**；
  多跳 expand 16 → 30，金标准页看全 0 → 41，看到金标准文档 137 → 143，"搜 ≤2 次就作答" 168 → 158（基本没变），
  平均搜索 3.0 → 3.1，格式错 3。结论：桥接规则对多跳有效；文件名可见对"拿错文档"帮助很小；单跳早答的行为
  prompt 压不住。prompt 1763 → 1715 token。全量验证待跑。
- **决定（2026-11-02，用户）：去掉文件名可见（试跑里几乎没帮助，且只对题目点名文档的题有用），保留桥接规则和例子精简；
  页记忆和 observation 恢复只显示 P1/P2 别名（§2 的"不出现真实页名"继续有效）。用这版跑 Qwen3.5 全量 `test_v3_qwen35`。**
- **全量结果（2026-11-02，Qwen3.5-27B，`test_v3_qwen35`，桥接规则 + 例子精简、无文件名）：84.97%**（单跳 87.32 / 多跳
  **78.13**；上一版 83.93 / 86.89 / 75.31，逐题净 +23，多跳 +16），金标准页都在最终证据里 79.0 → 80.0（多跳 54.5 → 57.8），
  格式错 3 → 6，预算用完 73 → 84，回合 2.73 → 2.78，搜索 1.67 → 1.70。错题 333：检索 213（单跳 102 题搜 ≤2 次拿错页，
  120 题 Qwen3.8 也没检到；多跳 51 题仍没 expand），读图/推理 93，挂页 21，格式 6。Qwen3.5 多跳已与 Qwen3.8（78.66）
  持平，总体差 0.2；同骨干对比 SCORE 提示法 82.53 → 84.97。

### 11.13 去掉 update_fact（2026-11-02 追加；用户决定）

全量轨迹里 update_fact 几乎不出现（Qwen3.8 uses 版 4 次 / 2215 题，Qwen3.5 两版 0 次和 1 次；79 次 revisit 里 0 次），
teacher 不产这种数据，学生也学不到。决定：facts 只剩 `add_fact`，页上的读数只增不改；Fact 的 revision / history 一并删除。
读错了的处理 = 回看后加一条新读数，再 resolve（可带 node_id 改旧答案）；旧读数留着。"可编辑页记忆"里可改的只剩
问题树的结论。revisit 动作保留（回看后 resolve+uses、回去 crop 看细节这两种用法真实存在）。prompt 删一行格式说明和
revisit 描述里的 "correct a fact"；SFT 标签里的 update_fact 删掉；事件字典不再有 updated_facts。
- **撤销（2026-11-02，用户）：update_fact 保留。** 上面的删除已原样恢复（协议、记忆、prompt、SFT 标签、测试），
  prompt 回到 1690 token 左右。§11.13 只作为记录留着。

### 11.14 被筛掉的多跳训练题用文档内检索重新生成（2026-11-02 追加；用户决定，只影响 SFT 数据）

train_v1（全库检索，Qwen3.8 teacher）留下 7,763/10,617，多跳只留下 574/1,236（46%），在数据里只占 7.4%。被筛掉的 662 道多跳题
（金标准页没检到 350、答错 246、看到没挂上 63、其他 3）用 `config/deck_scope.yaml`（retrieval_scope: deck，检索只在本题文档内）
重新生成，其余配置不变；输出 `outputs/new_sft/train_multi_deck_v1_gpu{0,1}`，不覆盖 train_v1。评测仍是全库检索；
这批轨迹里搜索总落在对的文档，学生学不到"搜到别的文档要换词再搜"，占比小（≤ 662 条），接受。

### 11.15 只给 teacher 的多跳提示（2026-11-02 追加；用户决定，只影响 SFT 数据生成）

§11.14 的文档内检索重做，662 道被筛多跳题只留下约 19%。被筛的主因是"答对了但只看了 2 页里的 1 页"：桥接那一跳 teacher 用常识补
（Dassler Shoes→Adidas、lafarge.com→Lafarge），一次搜索就作答。检索范围帮不上，是 teacher 行为。决定：
- `agent.teacher_hint`（字符串，默认空）：非空时追加在 teacher 看到的题目后面（system 里 `=== QUESTION ===` 之后一行 `Note: ...`），
  **SFT 行里的 messages 用不带提示的版本重新渲染**，学生永远看不到；评测和 RL 不设它。
- overlay `config/multi_hint.yaml` 给出提示文本：这题要两页证据，先 expand 解析间接指代，再找剩下的；两部分都不许用常识补；
  不要在推理里提这条 note。
- 生成后检查 think 里是否提到 note/hint，提到的轨迹在合并时剔除（不进 SFT）。
- 只对 §11.14 之后仍被筛掉的多跳题跑，仍用文档内检索；输出 `outputs/new_sft/train_multi_hint_v1_gpu{0,1}`。
- **结果（2026-11-02）**：第一轮（文档内检索）662 题留下 124；第二轮（文档内检索 + teacher 提示）538 题留下 223，其中 154 条
  推理里提到了提示（"The note says this needs evidence from two different pages…"，提示里"不要提"无效）。这 154 条由 4 个 agent
  人工清理（删掉/改写提到提示的句子，必要时换成基于题目或页面的理由，动作不动），独立复核后丢弃 2 条质量差的
  （6090 推理绕圈、5785 依赖提示做选择），下一轮输入里的"上一条回复"同步替换；残留 0。
  **最终数据 `outputs/new_sft/train_merged_v1/`：8,108 条轨迹，10,728 行 SFT，多跳 919 条（11.3%，覆盖 1,236 道多跳题的 74%），
  有 expand 的 627 条。**

### 11.16 SFT 数据 v1 的挑选（2026-11-02 追加；用户定量：1,800 单跳 + 600 多跳）

从 `train_merged_v1`（8,108 条）挑，脚本与特征在 H800 `outputs/new_sft/{sft_features.py, select_v1.py}`，结果 `outputs/new_sft/sft_v1/`
（2,400 条轨迹、3,902 行 SFT，含 summary.json 和每条的 selection 理由）。
- 硬过滤：答案与金标准精确/近似一致、答案能在金标准页的某条读数里找到、框大小合理、推理无重复绕圈、没挂非金标准页、
  推理长度在范围内、答案 ≤ 12 词。过后单跳 5,471、多跳 643。
- 质量分：精确匹配 > 近似、短答案、推理长度适中、金标准页 1–4 条读数；多跳另加来源分（全库检索 > 文档内 > 加提示清理）。
- 多样性（单跳）：有非平凡行为的全收（crop 232、再搜一次 180、回看/改读数 4），其余 1,384 条平凡一轮轨迹按
  证据类型 × 答案类型分格，配额 ∝ √格大小，每份文档最多 1 条（不够再放宽到 2）；1,800 条覆盖 1,408 份文档。
  多跳（用户要求同样兼顾质量与多样性，重选）：比较题 / 计算题的答案是推出来的，对它们放宽"答案在读数里"这条（池子 643 → 695）；
  按题型（桥接 / 比较 / 计算 / 其他）× 做法（expand / 只用 uses）分 8 格，注水法定配额 min(格大小, λ√格大小) 合计 600，
  格内按分取，每份文档先限 1 条（不够放宽到 2、3）。结果：桥接 247、比较 157、计算 132、其他 64；expand 402、只用 uses 198；
  来源 train_v1 395 / 加提示 129 / 文档内 76；417 份文档；精确匹配 563。v1 合计 2,400 条轨迹、3,887 行 SFT。

### 11.17 每轮消息改成严格交替（2026-11-02 追加；用户决定）+ 学生 SFT v1 的训练设置

**消息结构。** 旧：system（策略 + 问题）→ assistant（上一轮输出）→ user（观察）→ user（记忆板）。推理没问题（Qwen 模板按原样渲染），
但 ms-swift / LLaMA-Factory 都要求 user/assistant 严格交替：会合并相邻的 user、在 system 后插一条空 user，训练 token 就和推理对不上。
新（`build_turn_messages`）：system（只有策略）→ user（问题，`question_text`）→ assistant（上一轮输出）→ user（观察 + `TURN_SEP` + 记忆板）；
还没有上一轮输出时，问题和记忆板合成一条 user。teacher 提示（§11.15）仍接在问题后面。
- sft_v1 不重生成：内容一字不变，只是上下文重新排列，目标（teacher 输出）不变。`outputs/student_sft/relayout_sft_v1.py` 逐条核对
  旧 system 文本 == 现在代码的 `policy_system(qwen2.5-vl)` + `question_text(query)` 后重排，3,887 行全部通过；旧文件留作
  `sft_v1/sft_layout_v0.jsonl`。teacher 的评测（§11.10–11.12 的 85.19 / 84.97）是旧结构下跑的，没在新结构下重测。

**学生训练（Qwen2.5-VL-3B，LoRA，1 epoch）。** 用 LLaMA-Factory v0.9.5（SCORE Table 8 的参数名就是 LLaMA-Factory 的），环境 H800
`envs/lf`：Python 3.11（uv）、torch 2.8.0 cu128、transformers 4.57.1、peft 0.18.1、flash-attn 2.8.3.post2（官方 wheel）。
配置 `outputs/student_sft/qwen25vl3b_lora_v1.yaml`：SCORE 的 cold-start 设置改成 LoRA——冻结 ViT 和 projector；LoRA r 64 / α 128 挂语言模型全部线性层；
lr 1e-4（SCORE 全参 1e-5 的 10 倍）、cosine、warmup 0.1；全局 batch 16（2 卡 × 每卡 2 × 累积 4；不开 gradient checkpointing，
每卡 4 条最长样本会超 80 GB）；cutoff 16384；`mask_history`（只训最后一轮 = target）；
FA2、bf16；2 卡 DDP（LLaMA-Factory 自带的 LoRA 例子就不用 DeepSpeed）。v1 训练时 `envs/lf` 里还没有 deepspeed（见下面"DeepSpeed"条）。
- DeepSpeed（2026-11-02 装好）：0.18.4（LLaMA-Factory 0.9.5 的上限，`check_version` 强制）每次 import 都要检查 CUDA 工具链里 nvcc 的版本，
  pip 的 CUDA 12 包不带 nvcc，机器上原来也没有 → 装了 NVIDIA 官方 runfile 的 CUDA 12.8.1 工具链（只装 toolkit、不装驱动，md5 已核对）
  到数据盘 `/root/rivermind-data/cuda-12.8`（容器重启不丢），`run_student_sft.sh` 设 `CUDA_HOME`。ds_report 全部常用 op 可用，
  FusedAdam JIT 编译 30 s 并跑通。LLaMA-Factory 冒烟：LoRA + ZeRO-2 的 4 步 loss 与 DDP 冒烟到小数点后三位一致；全参（LM 3.09B 可训，
  ViT/projector 冻结）+ ZeRO-3（官方 `qwen3vl_full_sft.yaml` 的做法）跑通，最长样本每卡 1 条 + gradient checkpointing，约 39 GB/卡，
  22 s/步（全局 batch 16）。卡间带宽实测（`outputs/student_sft/nccl_bw.py`）：两张是 700 W 的 SXM H800，但 NVLink 未启用、CUDA P2P 不支持，
  NCCL 走主机共享内存，只有 3.8 GB/s。ZeRO-3 每个微批都要跨卡收集参数和规约梯度，约 85 GB/步 ÷ 3.8 GB/s ≈ 22 s，正好是实测步时；
  LoRA/DDP 每步只有一次 ~0.5 GB 的 all-reduce，不受影响。3B 全参在这台机器上该用 ZeRO-1/2（单卡放得下，通信降到约 10–16 GB/步）；
  7B 全参只能 ZeRO-3，需要 NVLink 可用的机器。
图像像素上下限 [256, 512] × 28²：存下来的 3,851 张图全在 28 像素网格上且在这个范围内，训练和推理都不会缩放，bbox 像素坐标系和 teacher 一致。
- 对齐检查：LLaMA-Factory 自己的预处理（`get_dataset`）出的 token，和 vLLM 环境里 HF chat template 渲染推理消息出的 token 逐条比，
  3,887 行全部一致（含图像 token 数），loss 只落在最后一轮回复 + `<|im_end|>`；共 10.86M token。
- 冒烟：最长的 64 行、2 卡，4 步约 3.1 s/步，loss ≈ 0.83，LoRA 保存正常。容器显示 100 个 CPU 但配额 44 核，预处理要设
  `OMP_NUM_THREADS=1`，否则 16 个 worker 各开 100 线程卡死（`run_student_sft.sh` 已设）。
- 正式训练（2026-11-02）：243 步约 8.5 分钟（2 × H800）。第 241 步一个两条约 4.7k token 的微批在 backward 时 OOM（fp32 logits 梯度 5.4 GB），
  从 checkpoint-240（lr 已降到 8e-8）开 gradient checkpointing 续完最后 3 步。loss 每 50 步均值 0.59 → 0.48 → 0.45 → 0.43 → 0.42。
  LoRA 只挂在语言模型上（q/k/v/o + `layers.N.mlp.*`，视觉塔的 mlp 没挂）。产物：H800 `outputs/student_sft/qwen25vl3b_lora_v1/`
  （adapter 457 MB，另有 checkpoint-120/240/243）。以后同配置要么开 checkpointing，要么每卡 1 条。
- **测试集结果（2026-11-02，`test_student_v1`，全量 2,215，评测设置与 teacher 相同：全库检索、10 轮、DeepSeek v4-flash 判分）：
  68.44%（单跳 74.33 / 多跳 51.32）。** 对照：SCORE 3B（SFT + RL）70.47（76.33 / 53.44），VISOR 3B 68.49，EVisRAG 3B 68.35；
  teacher Qwen3.5 84.97、Qwen3.8 85.19。合并后的模型 `models/student_qwen25vl3b_lora_v1`（LLaMA-Factory export，chat template 与底座逐字一致），
  vLLM 用 `--generation-config vllm`（纯贪心，不带模型自带的 repetition_penalty 1.05），每卡一个副本 + 每卡 3 个评测分片，10 分钟跑完。
  - 格式错结束 148（teacher 6–8）：105 是第 11 轮（预算用完的只答轮）还在 search——teacher 预算用完的轨迹全被过滤掉，学生没见过"强制作答"这一轮；
    25 是贪心重复到 2,048 token 上限、写不出 `<act>`；其余是结构规则。去掉这 148 条是 73.34。
  - 错 699（teacher 333）：单跳读错 224 vs 70（看到了金标准页仍答错，最大差距）、不作答 153 vs 63、多跳检索 132 vs 70（不 expand）、
    多跳读错 64 vs 26、单跳检索 126 vs 104。multi 全部金标准页进最终证据 32.6%（teacher 53–58）。

### 11.18 SFT 数据 v2：think 质量过滤（2026-11-02 追加；用户决定先只做这一项，不补强制作答轮）

v1 的审查（`outputs/new_sft/audit_sft_v1.py`）：答案、fact 文字干净，题型分布与 test 一致；随机抽 32 个 bbox 约 12% 偏一行（未处理）。
这次只处理 think：任一行命中下列规则，整条轨迹删掉（`outputs/new_sft/build_sft_v2.py`）——
- 动作不一致：think 最后一句说了打算（let me / I'll / I should + crop / zoom / search / revisit），act 却是别的动作
  （v1 粗筛 44 行逐条看过：27 行是真的，其余是 "The zoom clearly shows…" 这类误报，规则据此收紧）；
- 重复：同一个 think 里一句 ≥ 25 字符的话出现两次；
- 摇摆：一个 think 里 ≥ 5 个 maybe / perhaps / might be / probably / I guess / not sure / Actually / Wait / Hmm / alternatively。
删掉的从 `train_merged_v1` 池子按同类补（同样的硬过滤、也不能命中上述规则）：单跳先同行为同格，多跳先同题型×同做法；先守每份文档的上限，
不够再放开。结果：删 131（单跳 42、多跳 89；摇摆 94、重复 25、不一致 27）。单跳补满 1,800；多跳干净余量只有 86（整个池子多跳 19% 被标），
所以 597。**v2 = 2,397 条轨迹、3,811 行**，`outputs/new_sft/sft_v2/`（`changes.json` 逐条记录删了谁、补了谁、为什么）。
- 前后对比：think 字符 p90 699 → 556、最长 3,965 → 2,442；含 Wait/Actually 的行 349 → 208；重复句 25 → 0。
  多跳 expand 402 → 440、只用 uses 198 → 157；比较题 157 → 131、计算题 132 → 113（这两类 think 最爱来回改口），补进来的多是桥接题（247 → 300）。
- token 对齐检查 3,811/3,811 通过。训练配置 `outputs/student_sft/qwen25vl3b_lora_v2.yaml`：和 v1 只差数据，另开 gradient checkpointing（不改数学）。尚未训练。
- v2 只训到第 30 步就被用户停掉（先捞多跳），没有结果。

### 11.19 SFT 数据 v3：捞回多跳 + 去掉超长 + 扩到 3,000 条（2026-11-02 追加；用户决定）

**捞回多跳。** teacher 保留的 919 条多跳里，v2 只用了 597 条。剩下 322 条（`outputs/new_sft/unused_multi.py`、`recheck_unused.py`）中：
think 质量差 171 条（要改写才能用，暂不处理）；**101 条只是被 `sft_features.py` 的字符串规则误杀**——
- 千分位、单位算不一致：`293,000` 对 `293000`、`66,065 MW` 对 `66065`；
- 算出来的答案（差值、总数、计数、增减）不会原样出现在 fact 里，被"答案在金标准页读数里"这条卡掉（其中计算题 41 条）。
抽查 8 条最可疑的（桥接题且答案不在 fact 里）都是由 fact 推出来的，不是凭常识。另外 12 条修了匹配仍不一致，但人工看是对的
（Feb-14 对 February 2014、One 对 1、8,6 对 8.6、gold 拼错的 Malayasia 等），也收进来。teacher 写成长句的 9 条、
只因框面积越界的 16 条、跨轮重复"再搜一次"的 7 条不收。
**去掉超长（单跳、多跳都适用）。** 一个 think 超过 1,500 字符，或者 ≥ 7 个模型轮：前者在原地反复确认同一结论（没到 5 个摇摆词），
后者多是同一个 query 反复搜、碰巧撞上。v2 + 新增里共 44 条（单跳 14、多跳 30）。
**扩到 3,000 条。** 单跳从池子补：v1 的硬过滤 + §11.18 的 think 规则 + 不超长，非平凡行为优先（已经全收了），其余平凡的按
证据×答案类型 √ 配额、每份文档 ≤ 2（不够放到 3）。
**结果（`outputs/new_sft/sft_v3/`，`build_sft_v3.py`）：3,000 条轨迹、4,377 行；单跳 2,320 + 多跳 680（23%）。**
多跳：桥接 331 / 计算 151 / 比较 128 / 其他 70，expand 496 / 只用 uses 184。单跳覆盖 1,485 份文档。最长 think 1,491 字符，最多 6 个模型轮。
token 对齐 4,377/4,377 通过（共 11.97M token）。训练配置 `outputs/student_sft/qwen25vl3b_lora_v3.yaml`（只换数据，开 gradient checkpointing）。
- **结果（2026-11-02，`test_student_v3`，全量 2,215，评测设置同 v1）：68.49%（单跳 74.09 / 多跳 52.20）**，v1 是 68.44（74.33 / 51.32）——
  总分持平（v1→v3 错改对 130、对改错 129，正好是噪声），多跳 +0.9；多跳全部金标准页进最终证据 32.6 → 38.4（expand 多了起作用）。
  格式错结束 148 → 136：第 11 轮仍不作答 116 → 99、复读到上限 25 → 21——think 清理和去超长只把
  它降了一点，根因（训练里几乎没有强制作答轮）没动。结论：在这个规模上多 600 条数据、清 think 都不改变总分；剩下的差距在感知和
  强制作答轮，需要 RL 或补那类数据。

### 11.20 ViDoSeek 评测接入 + 学生 v3 的 ViDoSeek 分数（2026-11-02 追加；只是测量，不改方法）

**数据。** 用 UniDoc-RL 发布的那份（HF `DeepGlint-AI/UniDoc-RL` `test_data/`：`test_vidoseek.parquet` + `images.tar.gz` 里的
`images/vidoseek/<sha1>_<页>.jpg`，页码从 1 起，和 `reference_page` 一致）。`scripts/prepare_vidoseek.py` 转成和 SlideVQA 同样的布局：
`data/vidoseek/pages/<doc>/page_NN.jpg`（硬链接）+ `test.jsonl`。1,142 题（single_hop 645 → hop=single，multi_hop 497 → hop=multi；
每题只有 1 个金标准页，所以这里的"多跳"是推理多步、不是多页），292 份文档、5,385 页（每份 ≤ 25 页）。
旧报告里 ViDoSeek 的 Extraction / Logic 就是这里的 single / multi（3B 旧方法 60.31×645 + 71.43×497 = 744/1142 = 65.15%）。
**索引。** ViDoSeek 是高分辨率文档页（常见 2667×1500、3308×2339），ColQwen 处理器默认上限 12.8 MP 时一页近 1 万 token，
batch 8 就把 80 GB 撑爆。`build_colqwen_index.py` 加了 `--max-visual-tokens`（colpali-engine 官方的 `max_num_visual_tokens`；
默认 0 = 不限，SlideVQA 索引行为不变）；ViDoSeek 用 2048（≈1.6 MP），实际均值 1,997、最小 983，和 SlideVQA 索引的 788–1,935 同一量级。
另外 /dev/shm 只有 16 GB，batch 16 / 4 个 DataLoader worker 才不溢出。
**评测设置同 SlideVQA**（扁平语料、10 轮、第一次搜索 = 问题、DeepSeek v4-flash 判分；judge prompt 写的是 SlideVQA，但判分规则是通用的）。
单卡：vLLM GPU_UTIL 0.5 + 3 个评测分片（`/root/rivermind-data/run_vidoseek_test.sh`），1,142 题 13 分钟。
- **结果（`outputs/new_eval/vidoseek_student_v3`）：65.67%（single 62.33 / multi 70.02）。** 对照（旧检索器 Qwen3-VL-Embedding，口径不完全一样）：
  旧方法 3B 65.15（60.31 / 71.43）、UniDoc-3B 72.85（71.16 / 75.05）、UniDoc-3B 每次只给 1 张候选 66.99、旧方法 7B v9 71.54、UniDoc-7B 76.27。
- 错 392 题：读错（金标准页看过、答错）180、格式错结束 132（强制轮仍不作答 77、think 绕圈/截断没给 act 55）、金标准页从没看到 80。
  金标准页看到过 86.1%（SlideVQA 78.1），排除格式错结束后准确率 74.26。和 SlideVQA 一样，主要差距在读图（single 尤其弱）和强制作答轮。

### 11.21 crop 判据写进共用 system prompt，和 SCORE 一致（2026-11-03 追加；用户决定）

**来源（SCORE 2609.15800）。** agent prompt 只写"必要时 zoom"（只对整页、每张一次）；造数据时 teacher 在"相关页藏着答案关键的
局部细节（表格单元格、图表数值、坐标轴、小字、名字、日期、数字）"时发 bbox；附录 B Fig. 4：脚注里的 2796 在整页下被读成 2798，
放大后读对。
**改动（`src/prompts.py`，测试 `test_system_prompt_crop_rule_follows_score`）：**
- crop 动作说明从"(small print, chart values, table cells)"改成 SCORE 的判据：页相关但藏着答案关键的局部细节（表格单元格、
  图表数值、坐标轴、小字、名字、日期、数字）时放大，从放大图里读。teacher 和学生看到同一条规则，不再用只给 teacher 的提示。
- 最后一个例子原来直接从整页读出脚注数字 2796 并作答——正是 SCORE 的反例——改成对脚注 crop，不从整页记这个数。
- 删掉 Examples 前多余的空行（`_UNSURE` 末尾的换行）。prompt 5,616 → 5,554 字符（qwen3 guided）。
**为什么。** SFT v3 里只有 8.1% 的轨迹 crop，学生在 ViDoSeek 上只 crop 2.9%。VisR-Bench 上试过只给 teacher 的提示
（`config/visr.yaml` 的旧版本，smoke `outputs/new_sft/visr_smoke30*`）：弱提示（看不清才 crop）crop 25%；强提示 82–93%，但
think 里约 7% 会说出"the note says to crop"，要另加禁令再过滤。规则写进共用 prompt 就没有这个问题。
VisR 本身（`data/visr/`：2,591 题、6,913 页、索引已建）暂不用于 SFT（用户决定）；smoke 里被拒的题约 1/5 是金标准错
（图明明上升却标 Decreasing、取错系列、OCR 乱码），其余是 teacher 放大后仍读错、答不全、扁平语料下检索不到泛问题。
**影响。** 旧 SFT 数据（v1–v3）都是旧 prompt，按规矩重生成：Qwen3.8-27B 全量重跑 SlideVQA train（`outputs/new_sft/train_v2_gpu{0,1}`）。
teacher 在新 prompt + §11.17 严格交替结构下重测：Qwen3.5-27B 全量 test（`outputs/new_eval/test_crop_qwen35`），基线 84.97
（旧 prompt、旧结构）；低于 83 就先停下复查，不生成。流水线 `/root/rivermind-data/run_crop_prompt.sh`。
- **排版按 SCORE 重排（同日，用户决定；测试 `test_system_prompt_layout_follows_score`）**：一句角色；每轮固定两块写成
  Part 1 `<think>`（借 SCORE 的 observe 要求：读图后自己推出中间结论——排序、比较、计算——并明确写出）和 Part 2 `<act>`
  （page 记录 + 恰好一个动作）；每个动作后面写清什么时候用：search 加 SCORE 的 query 建议（先用问题本身搜，只有拿到具体
  中间目标才改写）、crop 用 §11.21 判据、answer 要求"每一部分都有已挂接的页面支撑"；第三条规则末尾加"宁可再搜一次也不放弃"。
  记忆说明、page 三字段和所有 op、坐标规则、桥接规则、挂接规则、"确定才答 + 看下一名"、6 个例子原样保留。
  长度 5,554 → 6,075 字符（qwen3 guided），紧凑上限从 5,700 放宽到 6,200。
- **crop 判据改成按答案类型（规则 A，同日，用户决定；测试 `test_system_prompt_crop_rule_by_answer_type`）**：SCORE 的
  "看不清才 crop"在 SlideVQA 上对 27B teacher 几乎不触发（50 题 smoke `smoke50_crop_qwen35`：新旧 prompt 都只 crop 1 题，
  准确率 86 vs 88 属噪声）——幻灯片 0.79 MP 压到 0.4 MP 只缩 1.4 倍，teacher 读得清就不放大，而 SCORE 训好的模型从 Table 4
  反推每题约 0.9 次 bbox（3.65 轮 − 1.76 次检索 − 1 轮 consolidate），我们的学生 0.12 次。改成客观触发：答案或中间值是
  数字、日期、名字、标签、表格单元格时，先 crop、从 zoom 里读，再 resolve；标题、页级内容直接读。例子改成先 crop 图表再
  从 zoom 读数并 expand（多一个例子块，新例子框 `<<EX_CHART_ZOOM>>`），KPI 表、脚注的 crop 理由也改成答案类型。
  长度 6,344 字符（qwen3 guided），上限放到 6,600。备选方案 B（只把 teacher 看到的图缩到 0.2 MP，规则不变）未采用。
- **例子改成"先读、再 crop 确认"（同日，用户决定）**：规则 A 第一版的例子是"先 crop、从 zoom 读"，50 题 smoke
  （`smoke50_cropA_qwen35`）crop 27/50，但准确率 82（旧 prompt 88），28 次 crop 里 23 次 teacher 仍先在整页读出值再 crop 确认；
  用户认为读完确认本身没问题。例子改成同一流程：页面轮读数、记 fact、crop 确认 → zoom 轮确认（或 update_fact 修正）后直接
  expand / resolve 往下走，不再 crop；expand 例子带全字段。第二版 smoke（`smoke50_cropA2_qwen35`，同 50 题）：
  **crop 37/50（74%），准确率 90（旧 88），每题 3.56 轮（旧 2.90），格式错 1**（expand 漏 "answer"，三次新 prompt 都在同一题上出现，
  旧 prompt 没有）。按这版跑 Qwen3.5 全量 test + Qwen3.8 全量 train（`run_crop_prompt.sh`）。
- **结果（2026-11-02 晚）。Qwen3.5-27B 全量 test（`test_crop_qwen35`，新 prompt + 严格交替结构）：83.70%**（单跳 86.17 / 多跳 76.54），
  旧 prompt 84.97（87.32 / 78.13）；crop 3.0% → 61.9%，每题 2.78 → 3.51 轮，格式错结束 6 → 16，轮数用完 84 → 116；
  排除格式错后 85.20 → 84.31。掉的 1.3 分来自格式错和多跳轮数更紧，不是读图变差。
  **Qwen3.8-27B 全量 train 重生成（`train_v2_gpu{0,1}`，2 卡 2 小时 42 分）：10,617 题留下 7,811（73.6%），单跳 7,200（77%）、
  多跳 611（49%）；留下的轨迹 89.2% 带 crop（train_v1 6.0%），update_fact 464（v1 71），expand 460，revisit 97。**
  拒绝原因和 v1 几乎一样（金标准页没进最终证据 2,384、判错 1,651）。下一步：在 train_v2 上重做特征 / 筛选，组 SFT v4。

### 11.22 掉分归因：不是 crop，是借来的两句话；prompt 改回，SFT v4（2026-11-03 追加；用户决定）

**逐题对比（`outputs/new_eval/flip_analysis.py test_v3_qwen35 test_crop_qwen35`，同 2,215 题）。** 净 −28 题 = 两边都正常作答的
2,059 题 **+5**（单跳 −6、多跳 +11）、轮数用完相关 **−22**、格式错相关 **−11**。读图没有变差。
- 轮数用完：旧版答对、新版用完的 50 题里，新版每题 11.0 轮、7.7 次搜索、4.8 个不同 query（旧 7.0 / 4.4 / 2.9），crop 只
  0.58 次/题，32 题一次没 crop；强制作答时拒答（"Cannot be determined"）60/116（旧 35/84），拒答全错。来源是 §11.21 排版时
  从 SCORE 抄的两句：answer 动作 "finish, only when every part of the question is supported by attached pages" 和规则末尾
  "Prefer another search over giving up"（旧版就是 `finish.`）。teacher 反复搜、不肯下结论、被迫作答时拒答。
- 格式错 16：7 把 action 塞进 page（都在 resolve 轮，排版引起）、4 同一页想第二次 crop（唯一和 crop 有关）、2 expand 漏
  answer、3 JSON 手滑。
- crop 本身：1,414 次，98% 先在整页记 fact 再 crop，zoom 改 fact 26 次（1.8%）；翻转题里 zoom 修对 4、改错 1。对 27B 在
  SlideVQA（0.79 MP 压 0.4 MP，边长缩 1.4 倍）是中性，和 §11.21 的判断一致。所以 crop 比例不是损失来源，不调。
- train_v2 留下的轨迹只比 v1 多一轮 crop（每条 1.28 → 2.12 轮，crop 0.06 → 0.90），搜索次数不变（0.21 / 0.20），≥6 轮
  1.1% / 1.3%：留下的都是成功轨迹，"多搜不下结论"没进 SFT 数据。
- 顺带：学生 v3 的格式错大头是强制作答轮不出 answer 动作（test 183、ViDoSeek 157），因为 SFT 里没有强制作答行。train_v2
  里只有 30 条 teacher 强制作答且判对的轨迹；**不加**（用户决定）。
**改动（`src/prompts.py`；测试 `test_system_prompt_layout_follows_score` 改断言）：** answer 动作回到 `finish.`（与 "text"
说明并一行），删掉 "Prefer another search over giving up."。协议不变，所以 SFT 行按新 prompt 重渲染，不重生成。prompt 6,344 →
6,236 字符（qwen3 guided）。Qwen3.5-27B 全量 test 重测 `outputs/new_eval/test_fix_qwen35`（单卡，`/root/rivermind-data/run_fix_test.sh`）：
顺便把 §11.17 交替结构的影响和两句话分开。结果待填。
**SFT v4（`outputs/new_sft/build_sft_v4.py` → `sft_v4/`）。** 池 = `train_v2_merged`（`merge_v2.py`，7,811 条；特征
`sft_features_any.py`、多跳问题类 `multi_qclass_any.py`，同 v1 脚本只把路径参数化）。过滤同 v1/v3：答案 exact/close、落在
金标准页 fact 上（comparison/arithmetic 且 exact 时豁免）、框合理、无循环、无多余页、think 长度、答案 ≤12 词；§11.18 think
标记；超长 = think > 1,500 或 **非 crop 轮 ≥ 7**（crop 轮每访一页加一轮，剔除后再数，等价于 v3 的 ≥7）。
- 多跳 611 → 硬过滤 147（不落地 64、loose 54、think>4000 24、框 20、循环 12）、think 标记 45（摇摆 33、重复 14、动作不符 4）、
  超长 9 → **410，全取**。单跳 7,200 → 5,370 合格 → **1,990**：先取少数行为（二次搜索、revisit/update_fact）≤2/deck，再按
  证据×答案类型 sqrt 配额 ≤2/deck（缺则 3）。crop 不作为筛选条件（池里 90% 都 crop）。system 行换成 11.22 的 prompt（断言
  旧文替换两句后与新文逐字相同）。
- 组成：2,400 条、5,392 行、每行 1.41 张图。单跳 crop 93%（行为：crop 75%、revisit/update_fact 12%、plain 7%、二次搜索 6%），
  2 轮 86%；多跳 crop 87%，bridge 49 / comparison 23 / arithmetic 21 / other 7，expand 79%，3 轮 66%。对比 v3：3,000 条
  4,377 行，单跳 crop 10%，多跳 680（23%）。
- 多跳不足 500：曾对 625 条被拒的多跳题用 deck 范围补跑（`run_v2_multi_deck.sh` → `train_v2_multi_deck`），跑到 237 条时
  停掉（用户决定，GPU 1 让给 teacher 重测），结果不用。
- **上面这版 v4 作废，改为按 11.22 的 prompt 重新生成（同日，用户决定）**：teacher 的 think 和行为要和学生看到的 prompt
  一致，不只换 system 行。单跳不全跑，4,000 道够用：sft_v4 选中的 1,990 道（已知过滤和配额都能过）+ 其余单跳 2,010 道
  （种子打乱、每轮每 deck 一道，含 train_v2 里被拒的，不偏）；多跳 1,236 道全跑。多跳、单跳交错排，两卡各 618 道多跳
  （`make_train_v3_subset.py` → `data/slidevqa/train_v3_subset.jsonl`，5,236 道）。teacher 重测改两卡：单卡跑到 890 题时停，
  GPU 0 续跑 [0, 1552)、GPU 1 另起副本跑 [1552, 2215)（`run_fix_test2.sh`），合并到 `test_fix_qwen35_merged`。接力
  `/root/rivermind-data/run_train_v3.sh`：等重测合并 → 门槛 ≥ 84%、测试通过 → Qwen3.8 两卡 flat 生成 `train_v3_gpu{0,1}` →
  train_v3 被拒的多跳用 deck 范围补跑（`train_v3_multi_deck`，同 §11.14 先例）→ 停副本。v4 再从 train_v3 + train_v3_multi_deck
  组，目标仍是 1,900 单跳 + 500 多跳，强制作答行不加。结果待填。
- **重测结果（`test_fix_qwen35_merged`，两卡）：83.97%**（单跳 87.32 = 旧 prompt，多跳 74.25；旧 78.13、v2 76.54），
  轮数用完 116、格式错 13。低于门槛 84%，生成接力已停（用户决定"先停下查多跳"），train_v3 没有生成。多跳旧版答对、这版答错 41 题，
  单跳 26 题。revisit 分析（131 次）：回去什么也没记又走 45、只用已有 fact 下结论 29、回去补放大 22、记下新 fact 35、改正 fact 0；
  有 revisit 的 101 题里 47 题轮数用完，同页回去两次以上 18 题。

### 11.23 page / tree / action 三键协议（2026-11-03 追加；用户决定）

**为什么。** §11.8 起树的修改（resolve / expand）挂在 page 里，结论只能落在正在看的页上，早先的页靠 `uses`；想用 P1 下结论就得先
revisit P1，多花一轮（§11.22 重测里 revisit 后 resolve 27 次），回去还常重新放大、接着搜。用户提出把 page 记录和问题树编辑分开。
**协议（`src/protocol.py`、`src/memory.py`；测试先改，89 项通过）：**
- `<act>{"page": {"desc", "facts"} | null, "tree": {resolve | expand} | null, "action": {...}}</act>`，三个键平级、每轮都写
  （规范渲染总是三键、按此顺序）。page 只记正在看的页；tree 从页记忆下结论，可以在任何一页、甚至没有页在看时做。
- resolve / expand 必须带 `"pages"`（至少一页，去重）：结论只挂它列出的页，正在看的页不再隐式挂上；`uses` 取消。列出的页必须
  看过且有 fact（本轮先应用 page 编辑，所以本轮刚记的 fact 算数），否则软错误重试。expand 仍只作用于 ACTIVE，resolve 可带 node_id。
- 宽松解析：tree / action 写进 page 里、而顶层没有时上提（§11.22 格式错里 7/16 是 action 塞进 page）；两层都有则不猜、报错。
- 没有页在看时 page 必须为 null，tree 可以写；answer-only 轮 page 必须为 null，tree 可以写。RL 环境 `rl/envs.py` 同步。
**Prompt（`src/prompts.py`，6,577 字符 qwen3 guided；原 6,236）：** 桥接规则、"确定才下结论"、搜索说明、crop 触发条件逐字保留。
改动：desc 写"这页涵盖什么（标题、表格的行、图的系列）"——facts 只记相关内容后，desc 是决定回不回去的索引；facts 只记和树里任一
问题相关的（含后一步要搜的名字、年份），无关页 `[]`；crop 时机从"resolve 之前"改为"离开这页之前"（tree 可在别处下结论，读到值
当下就从 zoom 记进记忆）；revisit 改为"回到之前的页（不是正在看的）去读 ACTIVE 问题需要、但 fact 里还没有的东西，或改正 fact；
每页每个问题一次"（用户要求简短）；挂页规则合成一条"结论在 pages 里列出用到 fact 的页，且只列这些；列的页必须有 fact"。例子 7 个
场景保留，第 5 个改成"从记忆下结论，pages 列 P1、P2，不用 revisit"。
**Smoke（Qwen3.5-27B，100 题，`outputs/new_eval/smoke100_tree_qwen35`；题组 `smoke100_tree_groups.json`）：**

| 题组 | 旧 prompt | 去掉两句（§11.22） | tree |
|---|---|---|---|
| 多跳·刚丢的 30 | 100 | 0 | 50 |
| 多跳·随机 20 | 70 | 85 | 90 |
| 单跳·刚丢的 10 | 100 | 0 | 50 |
| 单跳·随机 40 | 85 | 85 | 82.5 |
| 轮数用完（100 题） | 12 | 22 | 19 |

随机 60 题上 tree 与 §11.22 打平（51/60）；丢的题救回一半（含回归均值，不能全算协议的功劳）。tree 编辑 109 次，列 2 页以上 35 次、
没有页在看时下结论 5 次；每页 fact 0.78（§11.22 1.21），无 fact 的页 41%（21%）。每题轮数 5.23（5.20）、revisit 0.15（0.14）
没降。剩下的 19 题轮数用完几乎全是一路搜到底（S S S …），8 题金标准页都看过：teacher 嫌证据不够精确（要"Q3 2014"、看到的是
12 月；要"1999 年以来"的完整数据），同一 query 往下翻。这一点三个版本的 prompt 都有，与协议无关，待查。

**全量 test（Qwen3.5-27B，2,215 题，`outputs/new_eval/test_tree_qwen35`，2 卡 26 分钟）：83.51%**（judge 出错 1 题；逐题口径 83.48）。

| | 准确率 | 单跳 | 多跳 | 轮数用完 | 格式错 | 每题轮数 | 搜索 | crop | 最终证据覆盖 | 多跳覆盖 | 非金标准页 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 旧 prompt（旧结构，`test_v3_qwen35`） | 84.97 | 87.32 | 78.13 | 84 | 6 | 2.78 | 1.70 | 3.0% | 79.95 | 75.60 | 0.129 |
| 去掉两句（§11.22，`test_fix_qwen35_merged`） | 83.97 | 87.32 | 74.25 | 116 | 13 | 3.48 | 1.79 | 60.7% | 79.73 | 74.34 | 0.122 |
| tree（本节） | 83.51 | 86.52 | 74.78 | 123 | 15 | 3.62 | 1.83 | 68.8% | 80.50 | 76.43 | 0.129 |

- 逐题对比旧 prompt（`flip_analysis.py`）：净 −33 = 两边都正常作答的 2,052 题 −1（87.72 vs 87.67，读图和推理一样）、
  轮数用完相关 −22、格式错相关 −10。对比 §11.22：净 −10 = 多跳 +3（正常作答 +19、轮数用完 −12、格式错 −4）、单跳 −14（格式错 −8、轮数用完 −6）。
- 格式错 15 里 9 次是同一个手滑：`"facts":[...]}}, "tree": ..., "action": ...}`——page 后面多关一个括号，JSON 在 page 处就结束，
  tree / action 丢掉报 "action: Field required"；删掉那个括号就是完整的三键对象。8 题，其中 7 题旧 prompt 答对。prompt 里没有这种写法
  （仅有的 `]}}` 是 crop 动作 bbox 的结尾），是三键平级后嵌套深度写错。其余：同页第二次 crop 2、revisit 正在看的页 1、
  pages 列了没 fact 的页 2、最后一轮没 answer 1。
- 轮数用完 123（judge 判对 32，强制作答仍会判分）：每题约 8 次搜索、0.55 次 crop、一半 query 是重复的；旧 prompt 答对、tree 用完的
  60 题里 43 题金标准页都看过。不是"已有答案还继续确认"：verify 式 expand（next 以 confirm / verify 开头）全量只 7–8 题，三版一样；
  答案先进树、之后又多走 ≥ 3 轮的只 5 题。是一直没下结论、换说法反复搜。
- 混杂未拆：84.97 是 §11.17 之前的旧消息结构（问题在 system 里），§11.21 起所有重测都是新结构，轮数用完 84 → 116 → 123 一直没回到旧水平；
  旧 prompt 在新结构下从没跑过（§11.17 已注明）。
**结论。** tree 协议本身中性（对 §11.22 −0.46，多跳 +0.53、最终证据覆盖 +0.8），掉分仍全在"轮数用完 + 格式错"，两边都作答的题准确率不变。
- **crop 占不占轮数（`crop_turn_cost.py`、`after_zoom.py`、`prompt_audit.py`）。** 平均占：每题轮数 1.77 → 2.61，其中 crop +0.72；
  放大后改 fact 的只有 54 / 1,584 次（3.4%），基本是确认。但不是跑满 10 轮的主因：旧版答完、tree 跑满的 60 题里 32 题一次没 crop，
  比旧版多用 5.9 轮、其中 crop 只 0.70 轮，crop 能单独解释的 3 题。跑满的题里每题 7.8 次搜索；看过金标准页后的搜索，think 多是
  "要更具体 / 更精确的数据"，重复 query 占 53%（全量重复率 旧 31% → §11.22 35% → tree 40%）。放大后转去搜索的 41 题里，一部分是
  zoom 让 teacher 起疑（"3.9 是用户评分，不是官方星级"）。
- **prompt 里推着继续搜的句子（旧 prompt 也有）：** 问题消息 "Answer only when every part of it is supported."；规则 "only when you are sure …
  If not sure, search the same query again"；search 说明 "The same query again shows the next-ranked page" 和 "refine the query only for a
  specific intermediate target"；Part 1 "name the year, item or page you need next"。没有一句说什么时候停；模型也看不到剩几轮
  （只有第 10 轮被告知"最后一轮"）。

### 11.24 四块输出 + 每轮显示轮数 + 旧 prompt（2026-11-03 追加；用户决定）

**为什么。** §11.23 全量 test 83.51：两边都作答的题准确率不变，掉分全在轮数用完 + 格式错（审视见 §11.23 末）。用户逐条定了改法：
显示轮数（支持）、问题后的 "Answer only when every part of it is supported." 留着、crop 按需约一半（支持）、修好例 5 不加新例子、
"If not sure, search the same query again" 不改；格式上用户问能不能不要三层括号，选了拆成多个标签。
**协议（`src/protocol.py`；测试先改，91 项通过）：** 每轮四块：
`<think>…</think>` `<page>{"desc", "facts"} | null</page>` `<tree>{resolve | expand} | null</tree>` `<act>{"type": …}</act>`。
每块一个 JSON 值、各自解码到值结束为止；§11.23 的手滑（facts 后多关一个括号）现在只结束 page 这一块，tree / action 在自己的块里不受影响。
`<page>` / `<tree>` 可省略或写 null；`<act>` 必须有、必须是对象；块写了开头没结尾报错（不悄悄丢掉一块编辑）；think 里出现的块不读；
同类块取最后一个。报错以块名开头（"<page> schema: facts.0.bbox: …"），重试时知道改哪块。旧的单对象格式（`<act>{"page", "tree", "action"}`）
直接拒绝，§11.23 的 `_lift_nested` 删除。内部仍是 `Act(page, tree, action)`，trace 里的 `act` 记录形状不变；规范渲染 `render_turn` /
`Act.to_text()` 输出四块紧凑 JSON（SFT target 和上一轮回复）。`first_search_response`、`FORMAT_RETRY_SUFFIX`、`ANSWER_ONLY_HINT` 同步。
**轮数（`src/prompts.py::turn_of`，eval 的 `src/agent.py` 和 RL 的 `rl/envs.py` 共用）：** 每轮提示以 "Turn k of n:" 开头（原 "This turn:"），
k 从 1 数模型自己的轮次；first_search="question" 时第 0 轮的固定搜索不算，n = max_turns（turns10 下 "Turn 1 of 10" … "Turn 10 of 10"，
第 10 轮仍附 "Budget exhausted"）；模型先搜时 n = max_turns + 1。answer-only 轮只显示它自己的提示。
**Prompt 第一版（按需 crop，已弃用）：** crop 改成 "when what you need is small or dense: small print, a cell of a dense table, a chart's
tick labels …; what you can read clearly … needs no zoom"，图表例子直接读、例 5 改写。Smoke（`smoke100_blocks_qwen35`，§11.23 同 100 题）：
74 / 100（tree 71、旧 prompt 88，含 40 道按挑法偏向旧版的题），随机 60 题 49（tree 51），轮数用完 11（tree 19），每题 4.21 轮（5.23），
**但只有 8% 的题 crop**（tree 67%）——"看不清才放大"对 27B 太弱，和 SCORE 原句当初 1–3% 一样。用户不接受，全量跑到 334 题时停掉。
**Prompt 第二版（§11.22 的 prompt，已弃用）：** 用户："就是 83.9 那版 prompt，按现在 page / tree 分开的 <> 以及 turn 提醒"。§11.22 的 prompt
（`test_fix_qwen35_merged` 83.97 那次实际发出的，`outputs/new_eval/prompt_fix_qwen3.txt`）只改四块必需处，crop 用其原句 "Do this before resolving
whenever …"，6,312 字符。Smoke（`smoke100_fixblocks_qwen35`）：68 / 100（随机 60 题 48），轮数用完 14，格式错 4（都是强制轮 tree 的 pages 列了
没 fact 的页、answer-only 轮再犯，答案丢掉），crop 55% 的题（§11.22 在这 100 题上 52%）。不 crop 的 45 题：文字答案 26（规则允许标题、页级内容
直接读）、没找到页而轮数用完 11、上述格式错 4、数字答案没放大 4。全量跑到 245 题时用户停掉，改用下面的定稿。
**Prompt 定稿（用户决定："按照 old prompt 这版，别的都别动；crop 不用必须放大，针对 extraction 类型说要放大的场景；schema 换成我们这个样子，
针对性调整例子"）：** 底稿是 84.97 那次（`test_v3_qwen35`）实际发出的 system prompt（`outputs/new_eval/prompt_old_qwen3.txt`，已核对一字不差；
§11.10–11.12 的版本，没有 §11.21 的 SCORE 排版：无 Part 1 / Part 2、无 observe 句、无 query 建议）。只改：
- 格式：`<act>{"page", "action"}` 两块 → `<think>` `<page>` `<tree>` `<act>` 四块；"page" → `<page>`；`"update"` → `<tree>`（resolve / expand 带
  "pages"，`uses` 那句换成 "pages" 的说明，"the part this page answers" → "the part answered"）；"action" → `<act>`；
- 挂页规则：→ "listed in its "pages" (the page in view and earlier pages alike); a listed page must have facts"；
- crop：旧句 "once per visit (small print, chart values, table cells)" → "once per visit, when the question asks for a value printed on the page
  (a number, name, date or label) and it is small or dense there: small print or a footnote, a cell of a dense table, a chart's tick labels or
  data labels. Read that value from the zoom."（不要求必须放大，只说提取类问题什么时候需要）；
- 例子：全部换成四块写法；例 4 "Resolve with uses P1" → "Resolve with pages P1 and P2"；脚注例子原来从整页直接读出 2796 并作答（SCORE Fig. 4
  的反例），与新 crop 句矛盾，改成"问题要的是脚注小字里的数，放大去读"→ crop。图表例子（直接读、expand）、KPI 表"太小看不清"→ crop、zoom 轮
  读出 CHF 12.5 bn → resolve、Unilever 例子不变。
其余（角色句、MEMORY、desc、facts、box 规则、search、revisit、answer、规则 1 和 3、例子前的空行）逐字不动。5,654 字符（qwen3 guided；旧 5,486）；
unguided 的 search 句按 §11.21 之前的写法还原（只有 "a new, unseen page for the active question."；评测和生成都用 guided，`config/latest.yaml`）。
代码渲染与草稿逐字相同；测试 `test_system_prompt_is_the_old_prompt_in_four_blocks`、`test_system_prompt_crop_for_small_or_dense_values` 等，90 项通过。
每轮 "Turn k of n" 和问题后的 "Answer only when every part of it is supported." 保留；消息结构仍是 §11.17 的严格交替（训练需要）。
**影响。** 协议变了，SFT 数据必须重生成（§11.23 也还没生成）。先给用户看定稿，再 Qwen3.5 smoke + 全量 test，对比旧 prompt（84.97）、§11.22（83.97）、§11.23（83.51）。
- **全量跑到 832 题时用户停掉（改试 §11.25）。** 同 832 题：旧 prompt 85.46、§11.22 85.34、§11.23 83.89、本版 83.77（单跳 87.71、多跳 69.61，
  多跳最低）；轮数用完 17（最少）、格式错 6；crop 3% 的题。smoke 100 题 75（随机 60 题 50），crop 5%：新 crop 句几乎不触发，
  数字答案没放大 32 题（`test_225` 整页读成"约 21%"，标准答案 20）。

### 11.25 只让 teacher 看小图（2026-11-03 追加；用户决定）

**为什么。** 同一个 27B，crop 比例只看触发方式：按答案类型强制（§11.21 规则 A + "先读、再放大确认"的例子）55–69%，按"看起来小、密"
3–8%。SlideVQA 页约 0.79 MP，给 teacher 只缩 1.4 倍，它几乎都觉得看得清。用户不想强制放大，也不换模型，选 §11.21 的备选方案 B：
teacher 看的图缩小（"0.1 吧，反正就是尽量小"），让小字真的看不清，"小或密才放大"自然触发。
**实现（测试先改，92 项通过）：** `Frame` 加 `min_pixels` / `max_pixels`（默认仍是 SCORE 的 256–512×28²）和 `with_max_pixels()`；
`Agent(image_max_pixels=...)` 只改运行模型的 frame（页和 zoom 都按它缩），`sft_frame` 不变，SFT 行里的图仍是学生尺寸；配置
`agent.image_max_pixels`（默认 None），overlay `config/teacher_lowres.yaml` = 100,352（128×28²，约 0.1 MP；800×600 的页 → 352×256，约 88 个
32×32 视觉块）。teacher 用 0–1000 相对坐标，框与分辨率无关；服务端 Qwen3.5 处理器最小 65,536 像素，0.1 MP 原样进模型。zoom 也按 0.1 MP，
但只框页面的一小块（中位数 8% 面积），细节相当于整页 1.2 MP，比原图还清楚。
**注意。** SFT 行里 teacher 的 think 会写"太小看不清，放大"，而学生看到的是 0.2–0.4 MP 的图；学生本来读图就更弱，可以接受。评测分数和
其他版本不可直接比（teacher 看得更差）；这一步看的是 crop 比例和准确率代价。
**结果（smoke 100 题，`smoke100_lowres_qwen35`，`outputs/new_eval/lowres_why.py`）：不成立。** 56 / 100（同 prompt 原尺寸 75），多跳 40%（74%）、
单跳 72%（76%）；crop 仍只有 5% 的题（5 次，原尺寸也是 5 次），think 里对清晰度含糊其辞的比例也没变（14% vs 13%）。原尺寸对、小图错 23 题
（多跳 19），反过来 4 题；丢的 23 题里 18 题正常作答但读错，而且很笃定："The chart clearly shows 80%"（金标准 92）、"@gijsbrecht clearly visible"
（gijsbregt）、76 → 78、23 → 16。图变小时 27B 是过度自信而不是犹豫，"看不清才放大"照样不触发。结论：crop 比例只能靠客观触发（答案类型 +
"先读、再放大确认"的例子）。`image_max_pixels` 与 overlay 留着（默认 None，不影响其他运行），不用于生成。

### 11.26 三个硬要求下的定稿：83.97 的措辞 + 四块 + 轮数，answer-only 轮兜住（2026-11-03 追加；用户决定）

**要求（用户）：** ① 准确率在 83.9 这一档或更高；② 四块分开 `<page>` `<tree>` `<act>`；③ crop 的题不低于 30%。
**证据回看。** §11.24 第二版（83.97 的 prompt 原文 + 四块 + 轮数，`smoke100_fixblocks_qwen35` / `test_fixblocks_qwen35_s{0,1}`）当时按
smoke 68 判为差而停掉，重新按同题对比并不差：
- smoke 那 100 题里 40 题是按"旧 prompt 对、83.97 错"挑的（§11.23），83.97 自己在这 100 题上只有 51、四块版 68；随机 60 题
  四块版 48、83.97 版 51、旧 prompt 48、旧 prompt + 四块 50，都在噪声内。
- 全量停在 245 题。同 245 题：**四块版 89.80**（单跳 93.58、多跳 77.59）= 83.97 版 89.80（逐项相同），旧 prompt 88.98，tree 版 87.35；
  轮数用完 2（83.97 版 3、旧 5）、格式错 2（1 / 1）；crop 61% 的题。
- 另两版：旧 prompt + 四块（§11.24 定稿）832 题 83.77、crop 3%，不满足 ③；§11.23 tree 版全量 83.51、crop 69%，措辞改动多，差 ①。
  三条同时满足的，在已有证据上只有 83.97 措辞 + 四块这一版。
**唯一的失败模式（smoke 4 题、245 题里 1 题，全部以格式错结束）：** 第 10 轮（强制作答）模型想答"没找到"，`<tree>` resolve 把看过的
页全列进 pages，其中有没 fact 的页 → 编辑错误；answer-only 轮的提示写明 `<tree>null</tree>`，模型照样再列一遍 → 再错 → format_error，
答案丢掉。这 5 题的答案都是拒答（"Not found"），本来就判错，对准确率没影响；但 answer-only 是兜底路，该兜住。
**改动（测试先改，92 项通过）：**
- prompt（`src/prompts.py`）：模板换回 §11.24 第二版——`outputs/new_eval/prompt_fix_qwen3.txt`（83.97 实际发出的）只做四块必需的改动
  （"EVERY TURN output exactly these four blocks"、Part 2 一行、`<page>` / `<tree>` / `<act>` 标签、"pages" 说明、挂页规则、例子四块写法；
  §11.24 已列）；qwen3 guided 6,312 字符，与当时发出的逐字相同（存档 `outputs/new_eval/prompt_fixblocks_qwen3.txt`；测试锁定长度）。crop 句是 83.97 的客观触发 "Do this before resolving
  whenever the answer or an intermediate value is a number, date, name, label or table cell"，例子"先读、再放大确认"（§11.25 的结论：
  crop 比例只能靠这种触发）。unguided 的 search 句按 §11.21 写法。每轮 "Turn k of n"（§11.24）保留。
- 协议（`src/protocol.py::parse_turn`）：answer-only 轮只读 `<act>`（必须是 answer）；`<page>` / `<tree>` 写了也丢掉（没闭合、JSON 坏的
  也不报错）。正常轮和强制轮规则不变（列的页必须有 fact）。RL 环境走同一个 parse_turn。trace 里 answer-only 轮的 act.page / act.tree
  为 null，记忆不动。
**验证：** 同 100 题 smoke（`smoke100_fixblocks2_qwen35`；门槛格式错 ≤ 5、crop ≥ 30%）→ 全量 `test_fixblocks2_qwen35`，对比 84.97 /
83.97 / 83.51（`/root/rivermind-data/run_fixblocks2_test.sh`；分析脚本 `/root/rivermind-data/analysis/`）。
- smoke：71 / 100（随机 60 题 50：多跳随机 17/20、单跳随机 33/40；旧 48、83.97 版 51、旧+四块 50），crop 58% 的题，轮数用完 15，
  格式错 2（answer-only 轮仍要搜 1、同页二次 crop 1）。强制轮 tree 列无 fact 页 3 题，其中 2 题 answer-only 轮兜住（拒答，仍判错）。
- **全量（2 卡 26 分钟）：83.97%**（单跳 86.71、多跳 76.01），轮数用完 93、格式错 11、每题 3.47 轮、**crop 63% 的题**。三条要求：
  ① 与 83.97 原版同分（逐题净 0）；② 四块；③ 63%。

| | 准确率 | 单跳 | 多跳 | 轮数用完 | 格式错 | 每题轮数 | crop 的题 |
|---|---|---|---|---|---|---|---|
| 旧 prompt（旧结构，84.97） | 84.97 | 87.32 | 78.13 | 84 | 6 | 2.78 | 3% |
| §11.22（83.97） | 83.97 | 87.32 | 74.25 | 116 | 13 | 3.48 | 61% |
| §11.23 tree（83.51） | 83.48 | 86.47 | 74.78 | 123 | 15 | 3.62 | 69% |
| **本节** | **83.97** | 86.71 | 76.01 | 93 | 11 | 3.47 | 63% |

- 逐题对 83.97 原版（`flip_analysis.py`）：净 0 = 两边都作答 2,051 题 −9、轮数用完相关 +7（用完 116 → 93）、格式错相关 +2；多跳 +10、
  单跳 −10。逐题对 84.97：净 −22 = 两边都作答 2,079 题 −3（87.64 vs 87.49，多跳持平 82.04）、轮数用完相关 −13（多跳 −17）、格式错相关 −6
  （旧版答对、本版格式错结束 7 题，6 题是同页第二次 crop）。读图不变，差距仍全在多跳的轮数用完 + 格式错。
- 编辑错 / 格式错明细（全量）：框太小 10、同页二次 crop 6、pages 列无 fact 页 8（其中 1 题以格式错结束，其余 answer-only 兜住或重试修好）、
  resolve 的 pages 为空 3、revisit 正在看的页 3、缺 <act> 2、answer-only 轮不作答 1。
- 不 crop 的 822 题：文字答案 675、数字答案 147（其中 29 题轮数用完）；crop 的题准确率 88%、不 crop 78%（crop 的题本身更难不是原因，是文字题
  直接读）。
**结论。** 定稿。SFT 数据按本协议重生成（§11.23 起旧数据不兼容），待用户决定。

### 11.27 SFT 数据 v5 的生成：5,000 单跳 + 全部多跳，按 §11.26 协议（2026-11-03 追加；用户决定）

**用户决定：** "5000条单跳 全部多跳"。§11.23 起协议变了，旧 SFT 数据（train_v1 / v2、sft_v1–v4）全部不兼容。
**子集（`outputs/new_sft/make_train_v4_subset.py` → `data/slidevqa/train_v4_subset.jsonl`，6,236 道）：** 多跳 1,236 道全取；单跳 5,000 =
sft_v4 当时选中的 1,990 道（已知过滤和配额都能过）+ 其余单跳 3,010 道（种子打乱、每轮每 deck 一道；含 train_v2 里被拒的，不偏；
1,912 个 deck，每 deck ≤ 5）。多跳、单跳均匀交错，两张卡各 618 道多跳。
**轮数用 10（`config/turns10.yaml`），不是 teacher.yaml 的 12：** §11.24 起每轮显示 "Turn k of n"，SFT 行里的 n 必须和学生评测的 10 一致，
否则学生训练只见过 "of 12"。代价是多跳轮数用完会略多（train_v1 在 12 轮下轮数用完 189 / 10,617）。
**接力（`/root/rivermind-data/run_train_v4.sh`，日志 `run_train_v4.log`）：** 先按 PID 停掉两个 Qwen3.5 副本（675、3202）；
① 测试 92 项、代码是 §11.26 版、子集 6,236 行；② `run_generation_multi_gpu.sh` 起两张卡的 Qwen3.8-27B 副本，跑 30 题 smoke
（`smoke_gen_v5`，子集前 30 道）；③ 门槛（`analysis/check_gen.py`）：留下 ≥ 8 条、每条 SFT target 都是四块、每行最后一条 user
消息有 "Turn k of 10"；④ 同一对副本上跑全量两分片 → `train_v4_gpu{0,1}`（过滤同前：judge 判对 + 金标准页全看过）；⑤ 被拒的多跳
按 §11.14 先例用 deck 范围补跑（GPU 0 副本，`train_v4_multi_deck`）；⑥ 停副本。之后再从 train_v4 + multi_deck 组 sft_v5（过滤与
配额按 §11.22 的 build_sft_v4 思路另写）。结果待填。
- **smoke 过了**（30 题：留下 22，单跳 21/24、多跳 1/6，44 行，target 全是四块、全有 "Turn k of 10"，留下的 91% 带 crop）。全量 09:08
  起跑，约 62 道/分钟。
- **09:53 盒子时间中断：容器被重建**（根目录全新、`authorized_keys` 清空、`nvidia-smi` 只剩 GPU 0；数据盘 `/root/rivermind-data` 完好）。
  两个副本收到关停信号后退出，生成因 `--stop-on-api-error` 停在 938 + 972 = 1,910 道（留下 1,422：单跳 1,232/1,532、多跳 190/378；3,180 行），
  deck 补跑停在 71 / 188（留下 12）。用户重装密钥后，单卡续跑（`run_train_v4_resume.sh`，日志 `run_train_v4_resume.log`）：一个副本、两分片各 8 个
  worker（同 OUT 跳过已完成的题）、再把所有被拒多跳重算一遍做 deck 补跑（同 OUT 续跑）。单卡约 31 道/分钟，剩 4,326 道约 2 小时 20 分。
  实际：单卡放不下副本（64 GB）+ 两个 ColQwen 查询编码器（各 7 GB），分片 0 启动即 CUDA OOM；改成分片 1 先跑（8 worker，21–30 道/分钟）、
  再单跑分片 0（`run_train_v4_resume2.sh`）。分片 1 跑到 2,991 时用户把第二张卡加了回来（容器再次重建、密钥再清一次），改两卡续跑
  `run_train_v4_resume3.sh`：两副本、两分片各 16 worker、deck 补跑在 GPU 0 副本上、按 PID 停副本。分片 1 先完，GPU 1 空出来后先把当时
  已知被拒的多跳（435 道）提前做 deck 补跑；再把分片 0 剩下的尾段 758 行切给 GPU 1（`run_train_v4_resume4.sh`，OUT `train_v4_gpu0b`）。
  教训：停进程时 `pgrep -f "bash run_xxx.sh"` 会把自己的 ssh 会话匹配进去杀掉，按已知 PID 停。
**结果（盒子时间 15:23 全部完成，`train_v4_gpu0` + `train_v4_gpu0b` + `train_v4_gpu1` + `train_v4_multi_deck`）：**
- flat：6,236 道全部处理，留下 **4,680（75.0%）**：单跳 4,070 / 5,000（81%）、多跳 610 / 1,236（49%）；SFT 行 10,360；留下的 91% 带 crop，
  expand 433、update_fact 320、revisit 82。拒绝：金标准页没看全 1,326、判错 852、轮数用完 108（10 轮预算的代价，1.7%）、编辑错 25、
  空答案 12、judge 缺失 15、API 错 6（中断那次）。target 全是四块、每行都是 "Turn k of 10"。
- deck 补跑（被拒多跳 626 道）：留下 121（19%），431 行。多跳合计 **731 / 1,236（59%）**。
- 下一步：从这四个目录组 sft_v5（过滤与配额待定），训学生。

### 11.28 SFT 数据 v5 的组成：crop 批 70% + 不 crop 批 30%，单跳多跳都有；学生 v5（2026-11-03 追加；用户决定）

**用户决定：** "extraction 类型（crop）的留一批、不用 crop 的留一批，单跳多跳，顾好质量多样性，训一轮跑一下"；
随后改为 **"带 crop 的占 60–70%，70 吧；单跳不够就拿之前版本可用的转换一下"**（训练尚未开始时改的）。
**池。** 当前池 `train_v4_merged`（`merge_v4.py`，4,801 条；特征 `sft_features_v5.py`——按 §11.23 协议改：`uses` 没有了，
多页结论 = resolve 的 pages 列了多页、或列的不是正在看的页；多跳问题类 `multi_qclass_any.py`）里 crop / 不 crop：单跳 3,764 / 306，
多跳 591 / 140——不 crop 的远不够 30%。
**旧轨迹转换（`convert_old_nocrop.py` → `old_nocrop_v5/`）。** 把 train_v2（prompt v2，同一条 crop 判据）和 train_v1（旧 crop 判据）
里**没 crop** 的轨迹用当前代码重放：旧的 page 编辑 → `<page>`；旧的 update（resolve / expand 挂在正在看的页上，早先的页靠 `uses`）
→ `<tree>`，`pages` = [正在看的页] + uses；action 不变；think 原文保留。每轮都走真的解析器、记忆编辑和动作检查
（`parse_turn` / `apply_edit` / `prepare_action`），上下文用当前 prompt、当前渲染器和 "Turn k of 10" 重建；框从 teacher 的
qwen3 系经原图像素转到学生的 qwen2.5-vl 系（`reframe_act`），观察图复用旧的 `samples/*/observations/*.qwen2.5-vl.png`
（断言尺寸 = 当前 frame 的 shown_size）。重放后的记忆必须和旧记录的 fact、证据页完全一致，否则丢弃。候选：没 crop、页级答案、
v1/v3/v4 硬过滤、≤ 9 个模型轮（12 轮预算下的后段在 10 轮下会变成强制轮）、think 不提 uses / update / crop / zoom（旧 think 里
"I can resolve with uses P1"、"Let me crop to confirm" 这种会误导）。结果：候选 3,820 → 转换 3,257（单跳 2,967 = v2 346 + v1 2,621；
多跳 290），零协议/编辑错误；丢弃 = 同一题重复 358、think 提旧协议词 159、fact 不一致 41、搜索没拿到新页 5。54 秒。
**过滤同 v1/v3/v4（`build_sft_v5.py`）：** 答案 exact/close、落在金标准页 fact 上（comparison/arithmetic 且 exact 时豁免）、框合理、
无循环、无多余页、think 长度、答案 ≤ 12 词；§11.18 think 标记（`<act>` 里直接读 `type`）；超长 = think > 1,500 或非 crop 轮 ≥ 7。
**不 crop 批只收页级答案（text / list / long text / yes-no）：** prompt 规定答案是数字、日期、名字、标签、表格单元格时必须 zoom，
没 crop 却答数字的轨迹（当前池单跳 36、多跳 37）违反规则，不收。
**配额。** crop 批只来自当前池：单跳 2,000（少数行为优先 ≤ 2/deck，再按证据×答案类型 sqrt 配额 ≤ 2/deck，缺则 3）、多跳合格的
全取 ≤ 3/deck。不 crop 批 = 该跳 crop 数 × 3/7，分三层取：① 当前池、② train_v2（同一条 crop 判据）全取 ≤ 3/deck，③ train_v1
补齐（单跳：少数行为优先，再证据×答案类型 sqrt 配额 ≤ 2/deck；多跳：问题类×模式水填 sqrt 配额，每 deck 1→2→3）。整个数据集
一题只出现一次（同一题不会既有 crop 版又有不 crop 版）。
**结果（`sft_v5/`）：3,377 条、6,777 行、每行 1.36 张图。crop 批 2,397（71%；单跳 2,000 + 多跳 397），不 crop 批 980（29%；单跳 857 =
当前 180 + v2 180 + v1 497，多跳 123 = 当前 31 + v2 7 + v1 系 85，多跳目标 170 没到：剩下的旧多跳不 crop 轨迹和 crop 批同题）。**
按行算 crop 批占 82%（crop 轨迹多一轮）。单跳 2,857 / 多跳 520（15%）。单跳：crop 70%，行为 crop 60% / plain 28% /
revisit·update_fact 6% / 二次搜索 6%，2 轮 67%、1 轮 28%；证据 text 32 / chart 30 / table 17 / diagram 10 / image 7 / map 4；
答案 text 56 / number 19 / list 8；1,501 份文档。多跳：crop 76%，bridge 50 / comparison 23 / arithmetic 18 / other 9，expand 77% /
多页结论 23%。不 crop 批答案 text 89 / list 8，证据 text 41 / chart 25 / table 13。最长 think 1,480 字符，最多 7 个模型轮。
来源：单跳 train_v4 76% / train_v1 17% / train_v2 6%；多跳 train_v4(+deck) 82% / 旧 18%。
**训练与评测（`/root/rivermind-data/run_student_v5.sh`，日志 `run_student_v5.log`）：** 转 LLaMA-Factory 格式 + token 对齐检查
（每行 ok 才往下）→ `outputs/student_sft/qwen25vl3b_lora_v5.yaml`（配方同 v3，只换数据）→ 合并 LoRA →
`models/student_qwen25vl3b_lora_v5` → 两卡各一副本 → `run_student_test_v5.sh` 全量 2,215（口径同 v1/v3：flat、10 轮、首搜 = 问题、
DeepSeek v4-flash judge）→ `outputs/new_eval/test_student_v5`。
- 第一次起跑（2,608 条的旧组成）卡在 token 对齐的预处理：`dump_lf_tokens.py` 的 16 个 worker 没限线程，负载 221（容器配额 22 核）
  空转 14 分钟 0 进度（v3 同一步 17 秒）；按 PID 停掉，这两步加 `OMP_NUM_THREADS=1`（`run_student_sft.sh` 早就这么做）后 1 分钟完成。
- 对齐 6,777/6,777 通过（token p50 3,008 / max 4,749，共 20.4M）。训练 21 分钟（424 步，train_loss 0.345；v3 0.432）。
- 第二次接力评测分片 14 秒就退：脚本开头全局 `export HF_HUB_OFFLINE=1`（给转换/对齐/训练的）被评测继承，ColQwen 的 flash-attn3
  Hub kernel 启动时要经 hf-mirror 解析一次版本，离线就失败（缓存其实在）。改成只在那几步设离线，单独重跑评测
  （`run_student_v5_test.sh`）。
**结果（`test_student_v5`，全量 2,215，口径同 v1/v3）：68.71%（单跳 74.27 / 多跳 52.56）**；v3 68.49（74.09 / 52.20）。

| | v3（crop 10% 数据） | **v5（crop 71% 数据）** | teacher Qwen3.5 |
|---|---|---|---|
| accuracy | 68.49 | **68.71** | 83.97 |
| crop 率（题） | 4.8% | **83.2%**（单 86 / 多 74） | 62.9% |
| 平均轮数 | 2.83 | 3.54 | 3.47 |
| 格式错结束 | 136 | 106（第 10 轮不作答 86、缺 `<act>` 9、…） | 11 |
| 轮数用完（强制轮答了） | 3 | 14 | 93 |
| final_coverage / 非 gold 页 | 73.5 / 0.13 | 73.4 / 0.13 | 80.2 / 0.17 |

- 逐题：单跳 v3 对 v5 错 76、v3 错 v5 对 79；多跳 58 / 60——纯噪声，**crop 从 5% 到 83% 对 3B 学生的总分没影响**（和 §11.19 一致：
  这个规模上改数据组成不动总分）。crop 要求（≥ 30%）大幅满足：学生按 **行** 的 crop 占比（82%）学，不是按轨迹（71%）。
- v5 有 crop 的题 76.5 / 没 crop 59.8（单跳），teacher 89.8 / 80.9：没 crop 的题本来就难（检索到的页不对），不是 crop 的因果。
- 剩下的可避免损失还是老的：第 10 轮不作答 86 题（训练里没有强制作答行，§11.19 已指出），多跳 65 / 单跳 41。
- 学生 v5 的瓶颈和 v3 一样在感知与多跳推理（多跳 52.6 vs teacher 76.0），SFT 数据组成已不是杠杆；下一步是 RL 或补强制作答行。

### 11.29 RL 前置：学生 v5 的 ViDoSeek + RL 候选池 pass@5 / crop 统计（2026-11-04 追加；用户决定）

**用户决定：** 转 RL；第一步跑一遍 ViDoSeek，然后做数据筛选：在 SFT 没用过的数据里，取全部多跳（含 val）+ 2,000 道单跳，
每题推理 5 遍，统计 pass@5 并计算 crop 比率。
**候选池（`/root/rivermind-data/analysis/make_rl_pool_v1.py` → `data/slidevqa/rl_pool_v1/`）：** "SFT 没用过" = 不在 sft_v5 的
3,377 道里（学生 v5 只在 sft_v5 上训过）。多跳：train 1,236 道里没用过的 716 + val 全部 215；单跳：train 没用过的 6,524 道里抽
2,000（种子打乱、每轮每 deck 一道 → 1,843 份文档、每份 ≤ 2，其余不偏）。每题标上 teacher 在 train_v4 的结果
（train 多跳 kept 267 / rejected 449；train 单跳 kept 649 / rejected 348 / 没跑过 1,003；val 全没跑过）。共 2,931 题、14,655 次 rollout。
`*_x5.jsonl` 每题复制 5 份（sample_id 加 `__r<k>`、`base_sample_id` 记原 id），打乱后分片。
**采样：** 学生 v5，`config/sample_t1.yaml`（temperature 1.0；副本带 `--generation-config vllm`，即 top_p 1.0、top_k -1、无重复惩罚，
和 RL rollout 一致），其余同评测（扁平语料、10 轮、首搜 = 问题、DeepSeek v4-flash 判分）。train 题在 train 语料里检索（
`index/train_colqwen`），val 题在 val 语料里（`index/val_colqwen`）。
**统计（`analysis/passk_v1.py` → `outputs/new_eval/rlpool_v1/`）：** 按 train 多跳 / val 多跳 / train 单跳，再拆数字答案与页级答案：
pass@5、pass@1、k/5 分布（k=0 / 1–4 / 5）、crop 率（按 rollout；答对与答错分开）、5 次都 crop / 都不 crop 的题数、金标准页进最终证据
的比例、格式错、平均轮数。接力 `/root/rivermind-data/run_rl_pool_v1.sh`（日志 `run_rl_pool_v1.log`）：两卡各一个学生副本 →
ViDoSeek 全量（贪心，`vidoseek_student_v5`）→ val 多跳 ×5 → train ×5 → 统计 → 按 PID 停副本。
**ViDoSeek（学生 v5，全量 1,142，口径同 v3）：0.4 MP（默认）64.54（single 60.62 / multi 69.62）**，v3 65.67（62.33 / 70.02）。
single 掉在大页：ViDoSeek 原图中位数 2167×1500，按 0.4 MP 预算给模型看 756×504（每边缩 2.9 倍；SlideVQA 1.4 倍）；看到金标准页且
作答时的准确率按原图大小：1–3 MP 84.7、3–6 MP 76.0、≥ 6 MP 74.4（v3 84.2 / 79.3 / 80.0）——v5 在大页上 crop 82–84% 反而低 3–6 分，
可能是在缩小后的模糊整页上选的 crop 框偏了。**用户决定试 0.6 MP**（`config/image_max_0p6mp.yaml`，`agent.image_max_pixels` 602,112，
只改运行模型看到的图；模型写的框坐标最大 1,155 vs 0.4 MP 的 840，确认生效）：**66.20（61.24 / 72.64）**，+1.7。逐题：single 丢 47 / 新对 51
（净 +4，噪声内），multi 丢 27 / 新对 42（净 +15）；single 3–6 MP 76.0 → 78.3，≥ 6 MP 不变 74.4。学生在 0.4 MP 上训，0.6 MP 推理有分布偏移。
**采样结果（`outputs/new_eval/rlpool_v1/`；每题 5 次全齐，2,931 题 / 14,655 次）：**

| 组 | 题数 | pass@5 | pass@1 | k=0 | k=1–4 | k=5 | crop 率 | crop 率｜对 / 错 | 格式错 |
|---|---|---|---|---|---|---|---|---|---|
| train 单跳 | 2,000 | 78.4 | 63.3 | 432 | 675 | 893 | 72.9% | 74.7 / 69.9 | 289 |
| train 多跳 | 716 | 67.5 | 39.8 | 233 | 389 | 94 | 59.3% | 66.0 / 54.9 | 388 |
| val 多跳 | 215 | 79.1 | 49.2 | 45 | 121 | 49 | 61.6% | 65.2 / 58.1 | 119 |
| 合计 | 2,931 | 75.8 | 56.6 | 710 | 1,185 | 1,036 | 68.8% | 72.6 / 63.8 | 796 |

- crop 率按答案类型：单跳数字 80.4 / 页级 66.0；train 多跳 66.5 / 51.3；val 多跳 70.5 / 54.2。5 次里都 crop 的题：单跳 592、train 多跳 106；
  都不 crop：26 / 32（crop 是逐次采样的选择，不是题目固定的）。
- **同一题内 crop 对不 crop**（5 次里 1–4 次 crop 的题）：单跳数字 64.2 vs 63.6（无差别）、单跳页级 64.2 vs 57.9、train 多跳数字 44.6 vs 37.0、
  页级 42.9 vs 31.0、val 多跳 53.2 vs 47.1 / 52.9 vs 46.2。crop 的那几次更常答对，但不全是 crop 的因果（crop 的那次往往也找到了对的页）。
- 按 teacher 在 train_v4 的结果：单跳 kept 90.8 / 没跑过 81.5 / rejected 46.6（pass@5）；多跳 kept 74.9 / rejected 63.0——teacher 拒掉的题
  学生也难，但 rejected 里仍有 k=1–4 的 106 + 220 道。
- **k=1–4（有学习信号的"normal"题）共 1,185：单跳 675（数字 322，657 份文档）+ train 多跳 389 + val 多跳 121 = 多跳 510（43%）**，
  符合 [[active-graph-7b-rl-v7_1]] 的教训（多跳占 ~40%）。val 题要在 val 语料里检索（`index/val_colqwen`）。
- 采样时格式错 796 次（5.4%；train 多跳 10.8%），temperature 1.0 比贪心多（贪心测试 106 / 2,215 = 4.8%）。
**基础设施：** train 语料扁平检索时，CPU 页向量（38,380 页、~9.5 GB fp16）每次查询都要整体经 PCIe 搬到 GPU，6 个分片同时搬时
只有 ~57 次/分钟；按 PID 停掉、保留已完成的 402 次，改成页向量常驻 GPU（`config/bank_gpu.yaml`，分数相同）+ 副本 GPU_UTIL 0.3 +
每卡 2 分片 × 24 worker（`run_rl_pool_v1_rest.sh`），~390 次/分钟（7 倍），剩下 13,178 次 34 分钟。
- **第二批单跳（用户决定"再推 1,000 道单跳 ×5"，`analysis/make_rl_pool_v1b.py` → `rl_pool_v1/train_single2*.jsonl`，接力
  `run_rl_pool_v1c.sh`）：** train 单跳里 sft_v5 没用过、第一批也没抽过的 4,524 道中再抽 1,000（种子 2，每轮从被抽得最少的 deck 各取一道
  → 1,000 份文档、每份 1 道；teacher kept 293 / rejected 145 / 没跑过 562），设置同第一批（temperature 1.0、0.4 MP、页向量常驻 GPU），
  5,000 次 11 分钟。第二批 pass@5 79.2 / pass@1 64.8 / crop 73.2%，和第一批（78.4 / 63.3 / 72.9%）一致。
  **全池（3,931 题）：pass@5 76.65 / pass@1 58.65 / crop 69.9%；k=1–4 共 1,499 = 单跳 989（数字 473，854 份文档）+ 多跳 510（34%）。**
  单跳 3,000 道的 k 分布 0: 640 / 1: 213 / 2: 194 / 3: 212 / 4: 370 / 5: 1,371。同题内 crop 对不 crop（单跳）：数字 64.6 vs 62.7、页级 64.8 vs 60.1。
  第一批的统计另存为 `rlpool_v1/passk_batch1.{jsonl,log}`。

### 11.30 RL 数据集 v1：多跳全收 + 单跳按 crop / 不 crop 分好、按质量过滤（2026-11-04 追加；用户决定）

**用户决定：** "多跳全要；单跳分好应该 crop 和不用 crop 的，并且保证最好质量"。理解为：多跳收 k=1–4 的全部 510 道（k=0 / 5 在只看答案的
奖励下组内无信号）；单跳在 k=1–4 的 989 道里过质量关，再按每题自己的 5 次 rollout 判断 crop 要不要。
**逐次 rollout 特征（`analysis/rl_features_v1.py` → `outputs/new_eval/rlpool_v1/features.jsonl`）：** 每次记对错、是否看到 / 挂上金标准页、
**是否 crop 了金标准页**（不是任意页）、答案；每题在"看到过金标准页"的 rollout 上建 2×2 表（crop 金标准页与否 × 对错）。
**单跳质量过滤（`analysis/build_rl_set_v1.py`），989 → 689：**
- 有据：≥ 1 次答对且挂上了金标准页，且答对的里至少一半挂上了（丢 175 + 8）。抽查：答对却从没看到金标准页的多是常识或蒙的
  （"What does PaaS stand for?"、"When was Wright founded?"），RL 里会奖励不找证据就答。
- 不可猜：二选一 "X or Y?" 且答案是其中之一（丢 84；抽查 8 道全是真二选一），yes/no（0）。
- 判分一致：同一答案在一次里判对、另一次判错（丢 19）。金标准答案 ≤ 8 词（丢 14）。teacher 状态不作条件（rejected 但有据的 61 道保留）。
**crop 标签（看到过金标准页的 rollout）：** CROP = 不 crop 的全错且 crop 至少对 1 次 / crop 正确率更高 / 只试过 crop 且答案是数字；
NO_CROP = 不 crop 至少对 1 次且正确率不低于 crop / 只试过不 crop 且答案是页级。只试过 crop 的页级答案（89）、只试过不 crop 的数字答案（6）
证据不足，单放 `ambiguous.jsonl`，不进集合。**`crop_confidence`**：strong = 更严的规则也成立（CROP：crop ≥ 2 次对且比不 crop 高 ≥ 0.5，
或只试 crop、数字、≥ 2 次对；NO_CROP：不 crop ≥ 2 次对且正确率不低于 crop，或只试不 crop、页级、≥ 2 次对），否则 weak。抽查发现
宽松规则里 NO_CROP 常只靠 1 次不 crop 的成功（crop 3/4 对、不 crop 1/1 对 → 判 NO_CROP），所以加了这一档；5 次 rollout 证据薄，k=1 的题全是 weak。
**结果（`data/slidevqa/rl_set_v1/`）：1,104 道 = 多跳 510（46%；train 389 + val 121）+ 单跳 594。**

| 单跳 | strong | weak | 合计 | 数字答案 | pass@1 | 汇总：crop 时对 / 不 crop 时对 |
|---|---|---|---|---|---|---|
| CROP | 229 | 161 | 390 | 268 | 0.524 | strong：696/942（74%）/ 31/176（18%） |
| NO_CROP | 69 | 135 | 204 | 102 | 0.603 | strong：74/151（49%）/ 157/179（88%） |

- NO_CROP 里 102 道是数字答案（大字号的数字不 crop 也能读对），和 prompt 的"数字要 zoom"规则冲突；若以后按标签给 crop 相关奖励，
  要先决定这部分怎么处理。单跳 541 份文档、每份 ≤ 2。
- 多跳全收，质量标记只记录不应用：答对的 rollout 没把**全部**金标准页挂上 234（其中 220 道至少看到 1 个金标准页——学生平均只挂 1.1 页；
  14 道一个都没看到）、二选一 78、判分冲突 4、答案 > 8 词 9。
- 每行 = 数据集原行 + `corpus`（train / val，决定检索哪个语料：`index/train_colqwen` / `index/val_colqwen`）+ `crop_label` + `crop_confidence`
  + `crop_evidence` + `crop_table` + `k_correct` / `pass1` + `quality_flags`。被丢的单跳在 `dropped_single.jsonl`（带原因）。

### 11.31 RL 数据 1,200 + RL v1 环境：沿用旧 Active-Graph RL 的算法与参数（2026-11-04 追加；用户决定）

**用户决定：** "要 1,200 条；不够就再推 1,000×5，最后保留 1,200；同时准备之前的 RL 同款算法 GSPO 以及各个参数"。
**第三批单跳（`make_rl_pool_v1b.py`，BATCH=3，种子 3）：** 再抽 1,000 道没用过的 train 单跳（848 份文档、每份 ≤ 2），×5：pass@1 63.1、crop 74.4%，
和前两批一致。全池 4,931 题、k=1–4 共 1,819（单跳 1,309 + 多跳 510）。单跳过 §11.30 的质量关后 801 道有标签（crop 527 / 不 crop 274）、
125 道证据不足。
**选 1,200（`analysis/build_rl_set_1200.py` → `data/slidevqa/rl_set_1200/`）：** 多跳 510 全收；单跳 690 = strong 标签全收（crop 304 +
不 crop 87），其余用 weak 标签按 crop : 不 crop 原比例补（162 + 137），优先"答对的里挂上金标准页的比例"高的，每份文档 ≤ 2。
= 75 步 × 16。单跳 crop 68%；数字答案 439；k 分布 单跳 142/133/148/267、多跳 139/129/125/117（k=1..4）。parquet 按 verl-agent
`preprocess_slidevqa_active_graph_ra.py` 的布局，env_kwargs 带 `sample_id` 和 `corpus`。
**算法与参数 = 旧 3B v10.x（AutoDL `autodl_run_rl_3b_v10_3.sh` + 实际运行的 base，快照 `~/Desktop/research/slidevqa_rl_data/active_graph_rl_method_20260625.md`）：**
RA-GiGPO（`gigpo_ra`，gamma 0，step_advantage_w 0 → 只有组内归一化的结局优势），GSPO（`policy_loss.loss_mode=gspo`、
`loss_agg_mode=seq-mean-token-mean`），KL 0.001（low_var_kl，不进 reward），entropy 0，clip 0.2 / 0.28，lr 1e-6 cosine、warmup 0.2，
batch 16 × group 8、mini 32、micro 1，75 步、save 25，temperature / top_p 1.0，10 轮，cursor top-5，filter_groups 关，cudagraph 开
（enforce_eager False、free_cache_engine False），env.seed 0，全参（lora_rank 0），ref 模型 offload。改动只有硬件相关：2 张 H800（旧 4 张 96 GB），
vLLM gmem 0.3（旧 0.4），检索器在 GPU 1。
**奖励 = v9 / v10 的形式：** 1[答对] + 0.5 · 1[全部金标准页都在 E_final]（`COVERAGE_MODE=all`，即旧 R_hit_all_gt）+ 0.5 · R_grounding；
格式错 −1 并结束。**R_grounding 现在恒为 0：新池没有金标准 key-fact 框**（旧流程用 qwen3.7-plus 标注，这里还没做），所以实际是
1[答对] + 0.5 · 1[全金标准页]。CMP_W = ACCESS_COST_W = 0（v9 没有）。**答对的判分 = 评测判分器**（`rl/reward_judge.py` → `src/judge.py`，
v4-flash、同一 prompt），不再用旧 `visor_answer_reward.py`（默认 v4-pro，且把 facts 一起判）。
**环境与对齐（`rl/envs.py` + `code/verl-agent`，测试先改，96 项通过）：**
- `rl/envs.py`：`COVERAGE_MODE=fraction|all`；每个语料一个检索器（`index_paths` "train=…,val=…"，共用一个 ColQwen 编码器），题目按 `corpus`
  检索（val 多跳在 val 语料里，和测 pass@5 时一致；未知语料直接报错）；上一轮回复放规范重写（`render_turn`），和 `src/agent.py` 一致。
- verl-agent（H800 上 `code/verl-agent`，本地 `~/Desktop/research/verl-agent` 不动）：`env_package/slidevqa/envs.py` 只转发到
  `rl/envs.py`（旧 Active-Graph 环境留在 `old_agv/`）；projection 原样透传（新协议由环境自己解析）；env manager 把环境给的整段上下文
  （system / 问题 / 上一轮 / 观察 + 记忆）和全部图片（页 + zoom）传给 rollout；rollout 循环按这段上下文生成（旧版只给一条 user 消息 + 1 张图），
  `mode=new` 时多跑一轮（强制轮失败后的只答轮，§11.11）；`limit_images=2`；`loss_granularity=step`（一轮一条训练样本，上下文 = SFT 行）。
  图片不会被再缩放（rollout 只缩 > 4.2 MP 或 < 0.07 MP 的图），分辨率同 SFT（0.4 MP 框架）。
- 新环境 `envs/rl`：Python 3.11、torch 2.8.0、vllm 0.11.0、transformers 4.57.1、flash-attn 2.8.3、ray 2.50、verl 0.3.1（verl-agent）、
  colpali-engine 0.3.13（`setup_rl_env.sh`）。启动脚本 `/root/rivermind-data/run_rl_new_v1.sh`（`SMOKE=1` 跑 2 步）。

### 11.32 RL v1 定参：GSPO + v9 奖励，其余数字跟 SCORE；冒烟测试修的三处；金标准框补标（2026-11-04 追加；用户决定）

**用户决定：** "跟着 SCORE 对具体参数""肯定是 GSPO + v9 reward""用百炼的 deepseek-v4.1-flash-1017 补标注，先标几个看准不准"。
**SCORE（2609.15800 Table 9 / Eq. 4）：** GRPO（整条轨迹一个奖励，所有轮的 token 共用）；每题采样 5、每步 8 题、mini / micro 每卡 1（8 卡）、
lr 1e-6、warmup 0.285、KL 0.01、TP 2、prompt 8192 / response 2048、10 轮、100 步、vLLM gmem 0.4；奖励
r = cov + 0.2·1[cov=1]·cmp + 1[cov=1]·r_ans − 1[cov<1]·1。
**定下来的（`run_rl_new_v1.sh`）：** 算法 GSPO（不是 SCORE 的 GRPO，用户决定）；奖励 v9 形式 1[答对] + 0.5·1[全部金标准页在 E_final] + 0.5·R_grounding
（不是 SCORE Eq. 4，用户决定）；每题 5、每步 8 题、lr 1e-6、warmup 0.285、KL 0.01、8192 / 2048、10 轮（跟 SCORE）；150 步 = 1,200 题过一遍
（SCORE 100 步只用得上 800 题）。SCORE 没写、沿用旧值：RA-GiGPO 且 step 权重 0（= 组内归一化的结局优势）、GSPO seq-mean-token-mean、
clip 0.2 / 0.28、entropy 0、low_var_kl、temperature / top_p 1.0、cursor top-5、filter_groups 关、cudagraph 开。学习率曲线 SCORE 没写，
用 verl 默认的 constant（旧方案是 cosine）。mini batch 32 轮（verl-agent 按"轮"计；每步约 145 轮 → 4–5 次更新；凑整用随机复制，平均约 1 成）；
SCORE 的"每卡 1"最自然的读法是每步更新 1 次，按"轮"计做不到固定大小，所以沿用旧值。gmem 0.3（检索器和训练共用 GPU 1）。
**奖励模拟（本集 1,200 题已有的 5 次 rollout，`analysis/reward_sim.py`）：** 多跳里答对但奖励 ≤ 0 的比例：v9 形式 0%，SCORE Eq. 4 57%；
5 次奖励全同（无梯度）的组：0% / 15%。原因是学生平均只挂 1.1 页，Eq. 4 把答对分也门控在全覆盖上。
**冒烟测试（`SMOKE=1`：2 步、每步 2 题）修的三处：**
1. 某一步没有任何轨迹在搜索时，`ra_retrieved_page_ids` 整步缺失，按步拼批时长度对不上（58 ≠ 62）→ 每个 info 带同一套字段
   （`rl/envs.py::STEP_DEFAULTS`；测试 `test_every_info_has_the_same_keys`）。
2. 最后一步之后的验证用另一个环境对象，又加载了一份检索器，GPU 1 溢出 → 同一配置的环境共用一份检索器（进程级缓存
   `_RETRIEVERS`；测试 `test_envs_with_the_same_config_share_one_retriever`）。页向量放内存（`bank_device=cpu`，每次检索多约 0.5 s，省约 11 GB 显存）。
3. 参数更新每条样本 2.5 s（micro 1 + 跨卡只有 3.8 GB/s，FSDP 每个 micro-batch 都要 all-gather）→ `use_dynamic_bsz`，按 token 打包
   （actor 12,288、log-prob 16,384 / 卡）。
第 1 步验证过：RL 第 1 轮的输入和评测第 1 轮逐字相同（7,269 字符）；输出是四块格式；格式错都是策略自己的（revisit 正在看的页、JSON 写坏、
zoom 上再 crop）。
**TensorBoard：** 盒子上 `envs/rl/bin/tensorboard --logdir outputs/rl --host 127.0.0.1 --port 6006`（PID 在 `tb_rl_pid.txt`，按 PID 停），Mac 上
`ssh -N -L 6006:127.0.0.1:6006 h800` → http://localhost:6006。新增 episode 指标（env manager + `metric_utils.py`）：`reward/answer`、
`reward/hit_all_gt`、`reward/grounding`、`reward/terminal`、`end/format_error`、`end/budget_exhausted`、`end/unfinished`、`behaviour/crop_rate`、
`behaviour/n_search|n_crop|n_revisit`、`behaviour/final_pages`。
**金标准框：** 1,200 题里旧流程（qwen3.7-plus 标注、部分手修，`~/Desktop/research/slidevqa_rl_data/fresh_multi_20260702/`）已标全的：多跳 335 / 510、
单跳 99 / 690；其余约 766 题要补。脚本 `analysis/annotate_keyfacts.py`（百炼 OpenAI 兼容接口，默认 deepseek-v4.1-flash-1017，每题把金标准页
一起送，按页要 1–3 条支撑事实 + 0–1000 框，存成旧格式 + 原图像素框）。先标 14 题（10 题有旧标注可比 IoU，4 题没有）看准不准。结果见 §11.35
（最后用的是 qwen3.7-plus 逐页标 + Codex / Claude 智能体质检，不是 deepseek）。

### 11.33 RL v1 改回旧超参（除 group 5）；冒烟测试又修的四处（2026-11-04 追加；用户决定）

**用户决定：** 每步规模、步数、KL、学习率曲线"group 5 除此之外用以前"，即回到上一套 Active-Graph RL（7B v9 / 3B-v10）的值：
每步 16 题 × 每题 5 = 80 条轨迹（SCORE 8 × 5）、75 步 = 1,200 题过一遍（SCORE 式 150 步）、KL 0.001（0.01）、warmup 0.2 后 cosine
（warmup 0.285 后 constant）。其余照 §11.32 不变（GSPO + v9 奖励、lr 1e-6、mini 32 轮、8192 / 2048、10 轮）。每步约 310 轮（冒烟测试平均 3.9 轮 / 轨迹）→ mini 32 约 10 次更新。
`run_rl_new_v1.sh` 已改（SCORE 版留在 `run_rl_new_v1.sh.bak_score`）。
**冒烟测试修的四处（都出在某一轮看不到页面、没有图：例如一次搜索没有新页）：** 这样的轮和有图的轮混在同一批里。
1. rollout 行：无图轮也按 Qwen-VL 的行格式（4 行 mrope position_ids、`multi_modal_data={"image": []}`、`multi_modal_inputs={}`），
   否则拼批时 [4, 8192] 和 [8192] 对不上（`rollout_loop.py::preprocess_single_sample`）。
2. vLLM：空图列表按纯文本 prompt 送（`vllm_rollout_spmd.py`）。
3. actor：拼图片输入时跳过 `{}` 的行（`dp_actor.py::_forward_micro_batch`）；模型侧对无图 micro-batch 已有 0 权重的假图前向，FSDP 不会卡。
4. reward manager：`{}` 时不读 `pixel_values`（`agent_system/reward_manager/episode.py`）。
**§11.32 第 3 条其实没生效：** 这份 verl 在有图输入时不走 `use_dynamic_bsz`，固定每 micro-batch 1 条；而且两条路径同时开时 log-prob 会用到未定义的
`indices`，update 的 loss 缩放用的是 `len(dict)`。现在有图的行也按 token 打包（`dp_actor.py::_dynamic_micro_batches`，图片输入按同一组下标跟着走），
缩放改成按行数。
**这份 verl-agent（来自本地旧副本）比 AutoDL 版少两项，尚未补：** 跳过已结束轨迹的生成（active-subset；现在每轮都给全部 env 生成）和
最后一步先存 ckpt 再验证（现在先验证，验证 OOM 会丢最后的 ckpt）。

### 11.34 补回 AutoDL 版的两项：只给还在跑的轨迹生成；先存 ckpt 再验证（2026-11-04 追加；用户决定）

**用户决定：** "补这两项"（§11.33 末尾列的两项）。
1. **active-subset**（`rollout_loop.py::_generate_active`，开关 `ROLLOUT_ACTIVE_SUBSET`，默认 1，launcher 显式写上；0 = 原来每轮全量生成）：
   每轮只把还没结束的轨迹送 vLLM；已结束的轨迹填空回复（全 pad、回复部分 attention 0、位置接着 prompt 往后排，和 vLLM 输出同一布局，
   `_scatter_active_output`）。env 对已结束状态直接返回、`gather_rollout_data` 只收 `active_masks=True` 的行，所以收集到的训练数据和全量生成
   相同，只是少了白算的生成。每次 rollout 打一行 `[ROLLOUT_ACTIVE] ... generated per turn: ...`。
   单测（盒子上 CPU，假 vLLM）：活跃行的每个输出张量和全量生成逐项相同；已结束行是空回复；生成请求从 8 行降到 4 行。
   冒烟测试（第 6 次）：验证那次 rollout 每轮生成 16,15,11,3,1,1,1,1 条，共 49 条，全量生成是 16 × 8 = 128 条（少 62%）。墙钟时间在
   冒烟的小批量上量不出来（一条拖尾轨迹也要走完一轮），要看正式训练的 `timing_s/gen`。
2. **先存 ckpt 再验证**（`ray_trainer.py`）：存 ckpt 挪到验证前面；冒烟测试里日志顺序已是先存后验。
**同时改的 launcher：** `max_actor_ckpt_to_keep` / `max_critic_ckpt_to_keep` 3 → 2（和以前一样；一个 ckpt 59 GB，盘 541 GB，保留 3 个放不下）；
`trainer.resume_mode=$RESUME`：正式训练 auto（重启接着最新 ckpt 跑），`SMOKE=1` 时 disable（第 6 次冒烟误从第 5 次的 ckpt 续训，只跑了第 3 步）。
**冒烟第 5 次（全量生成、按 token 打包之后）：** 每步 288 s = 生成 191 + update 61 + ref 22 + old_log_prob 14；打包前 update 158 s。

### 11.35 金标准框（R_grounding）补标完成：qwen3.7-plus 逐页标 + 质检修正；多跳每个金标准页至少留 1 条（2026-11-04 追加；用户决定）

**用户决定：** "deepseek-v4.1-flash-1017 补标注，先标几个看准不准" → "qwen3.8-max-0902 试试" → "一页一页标，gt 一张张传" → "还是 qwen3.7-plus 吧"
→ "每标好 100 个拿去给 codex cli gpt-6.1-sol 检查质量然后修" → Codex 额度用完后 "开最多 5 个 agent 修，尽量别删成单页"。
**选标注模型（同 14 题、23 个金标准页，逐页看渲染图）：** deepseek-v4.1-flash 框系统性偏上；qwen3.8-max 逐页约 12/21 准，错的多是框落在文字
下方（4:3 页上 y 被拉伸约 1.3 倍；改要像素坐标只修好了一部分，模型本身定位不稳）；qwen3.7-plus 逐页 22/23 准（错的是密集表格差一行）。
**标注：** `analysis/annotate_keyfacts.py --model qwen3.7-plus --mode page`（每个金标准页单独一次请求，一张图 + 问题 + 答案，0–1000 框，
存原图像素框），48 并发，缺的 766 题全标完（JSON 解析改成只取第一个对象后 2 题重跑成功）。
**质检修正（`~/Desktop/research/slidevqa_rl_data/keyfact_qc/`）：** 每个金标准页一张复核图（编号框 + 每个框的放大裁图），逐条判 ok / fixed / drop，
错的在原图上裁图核对后给紧框；需要的事实缺了就补（答案值 / 答案实体 / 比较值 / 桥接实体 / 计数题的被计数项）；只用来定位的表头、图例、
坐标轴、标题删掉；每页 1–3 条，计数题少则逐个框、多则框整块区域。Codex gpt-6.1-sol 做了 91 题（每题约 2.3 万 token、3 分钟），撞额度后
其余 675 题由 Claude 智能体做（23 块 × 30 题，同时 5 个）。**多跳硬规则（用户）：** 每个金标准页至少留 1 条——把这一页和问题连起来的元素
（桥接实体、被问的公司 / 人 / 物、标识它的标题）；不许删掉某页的最后一条；真无关的页留最相关的元素并标 `WEAK_PAGE`（7 页）。原因：coverage
项要求所有金标准页都在 E_final，若某页没有金标准框，模型在那页记的事实拿不到 grounding 分，两项奖励互相拉扯；Codex 曾清空 8 页（其中有
真正的桥接页，如 3721 的公司名页），已按此规则重查。
**结果：** 766 题质检后 verdict：ok 368 / fixed 960 / drop 402 / added 111（agent 部分）；多跳题没有空金标准页。合并（新 766 题用质检版，其余 434 题
用旧 qwen3.7-plus 标注）→ `data/slidevqa/rl_set_1200_kf/{train,val}.parquet`，env_kwargs.keyfacts = JSON `[{page, bbox_2d(原图像素)}]`，
1,200 题全部有框且覆盖所有金标准页，共 3,023 个框；`rl.envs.normalize_keyfacts` 读出无误；坐标系与 `Fact.bbox`（原图像素）一致。
**qwen 框的常见错（质检里反复出现）：** 小数字 / 短标签偏 1–15 px、切掉 "%" "$" 末位数字或下伸部；落在相邻行 / 列 / 刻度；框住整个格子 / 节点 /
色块而不是文字；多行短语截短；加很多定位用的标签；比较题漏掉答案实体或被比较的值；计数题被计数项不全或被标题替代。
**可疑金标准答案（18 题，`keyfact_qc/answer_flags_all.jsonl`）：** 框按存的答案标，是否修正 / 剔除待用户定。
**未做：** 旧 434 题的框没有经过质检（抽查时 4236 的旧框也差一行）。

### 11.36 混合覆盖奖励与 GSPO clip（2026-11-05 追加；用户决定）

用户要求将讨论中的 0.3 / 0.2 对调，并一起调整 clip low / high；grounding 保持 0.5，不做 pruning，不引入 GiGPO。
下一实验的正常终止奖励为 `A + 0.2*C + 0.3*H + 0.5*G`：
`A` 是答案正确性；`C = |gold ∩ E_final| / |gold|`；`H = 1[gold 非空且 gold ⊆ E_final]`。
没有 gold 页时 C、H 都为 0；只访问但未纳入 E_final 的页不计覆盖；额外非 gold 页不取消全覆盖奖励。
`G` 沿用当前 page-frame facts 与金标准框的同页 best-IoU 均值，不改 scale 或 zoom 规则。
`CMP_W=0`、`ACCESS_COST_W=0`；格式错误及软错误处理维持现状。满覆盖的覆盖奖励仍是 0.5；覆盖一半时得 0.1。

GSPO clip 采用前面讨论的论文值：`clip_ratio_low=0.0003`、`clip_ratio_high=0.0004`（此次实现选择），
保持 `policy_loss.loss_mode=gspo`。本次不改变 advantage normalization、cosine + 20% warmup，也不启动训练。
新增独立 `full_coverage_w` 项；显式 `env.slidevqa` 奖励配置优先于同名大写环境变量，防止旧 launcher 的 export 覆盖新实验。
新实验参数单独保存在 `rl/experiments/hybrid_coverage_gspo.args`，作为最终追加的 Hydra overrides；历史实验启动快照不改。
该文件仅定义奖励和 clip，不是完整启动配置；实际开训前仍需核对当前机器的 GPU 数量、基础配置、模型路径和新实验输出目录。

### 11.37 Runpod 环境准备与单卡到四卡迁移（2026-11-05 追加）

用户计划后续使用单机四张 H100 / H200。实际准备实例 `sfnn12xqq60p57` 是 H100 80GB 单卡，
50 GB 系统盘、AP-IN-1 的 500 GB 区域 Network Volume（`/workspace`、nfs4），没有挂 Global Volume。
同机房替换 Pod 可复用这块盘；跨机房需要先复制并核验数据。准备过程不创建四卡实例、不启动正式训练。

托管 Python 和独立环境都放进网络盘（`/workspace/python`、`/workspace/envs/rl`）；代码在 `/workspace/code`。
部署当前 new 代码和 §11.36 的奖励 / clip overrides；verl 使用实际 H800 上的自定义代码。
数据、索引和检索器 adapter 的旧绝对路径通过 `/root/rivermind-data -> /workspace` 保留，不批量重写元数据。
`scripts/activate_runpod.sh` 在换 Pod 后恢复路径和环境；新镜像需兼容 CUDA 12.8 和现有 Linux Python 环境。

准备依赖为 torch 2.8/cu128、transformers 4.57.1、vLLM 0.11.0、colpali-engine 0.3.13、peft 0.17.1；
FlashAttention 使用匹配实际 torch C++ ABI 的 2.8.3 公开 wheel（原 H800 是 2.8.3.post2），先以真实 GPU
计算对比 PyTorch SDPA，再跑代理代码测试。SlideVQA 图片任务不安装可选 Decord wheel，视频解码依赖使用 PyAV。
这些检查不替代目标四卡 NCCL 测试、完整训练步和最终评测。

迁移资源清单覆盖当前训练数据、所有 train/val/test 页面与索引、检索器及自定义 verl；生成逐文件 SHA-256。
公开检索器权重只有在官方 SHA-256 与源文件一致时才直接在海外下载。大训练索引可分块并行传输，完整 SHA-256
匹配后才原子替换目标文件。安装生成的 `.egg-info` 元数据不参加源代码一致性核验。
服务器间传输使用临时受限 rsync 身份（仅写指定目录、禁止删除、无交互和转发），完成后撤销。

当前 H800 的 models 目录没有 7B 起训权重；正式 7B 实验仍需用户提供起训 checkpoint。
四卡实例实际租好后才调整和核对并行参数，先做 NCCL 和完整训练冒烟测试；不引入 GiGPO step advantage 或 pruning。

**用户追加决定：** 暂不处理 7B 权重；当前只完成通用环境、数据、索引和检索器迁移及校验，7B 不作为本次迁移的阻塞项。

**替换实例验证（2026-11-05）：** 用户提供新 Pod `5fb2fqupdetrg4`，实测仍是单张 H100 80GB，
挂载同一个 AP-IN-1 区域网络盘。前次完整迁移校验为 55,175 / 55,175 文件 SHA-256 一致、零错误。
继续在新容器验证保存的 Python / CUDA 环境和 GPU 检索；不启动训练，不自动租用或切换 GPU。

新 Pod 的 FlashAttention 与 SDPA 数值比对、真实 GPU 检索及图片读取、trainer 的 Hydra 配置解析均通过。
完整代理测试 104 通过、0 失败。测试目录新增显式包声明，避免 editable 安装 verl 后导入其同名 `tests` 包。
这些只验证运行环境和资源可用性，尚未验证四卡通信、完整策略训练步或新的答案准确率。

**3B SFT 起训模型补迁（2026-11-05，用户要求）：** 7B 的暂缓决定不包含 3B。
本次使用原正式训练解析配置指定的 `models/student_qwen25vl3b_lora_v5`，即合并后的 SFT v5，
不替换成 RL step50 / step75 checkpoint。共 15 个文件、7,525,286,697 bytes，源端逐文件 SHA-256
记录在 `preparation/source_sft3b_manifest.json`。使用原机现有密码由 RunPod 反向分块拉取；
密码只经内存管道用于 SSH，不写入文件、命令参数或环境变量，不新增 SSH 公钥。
目标目录为 `/workspace/models/student_qwen25vl3b_lora_v5`；文件完整性与本地 GPU 加载 / 图片前向
分别以 `sft3b_files_verified.json`、`sft3b_runtime_validation.json` 为准，不据此声称完成训练或提升准确率。

### 11.38 RL 训练集从 1,200 扩展到 1,600（2026-11-05，用户决定）

用户要求开始补齐 400 道新题，并明确不调整原 val 来源。现有 `rl_set_1200_kf` 的 1,200 道题、
答案和 key-fact 框保留原样，生成独立的 `rl_set_1600_kf`，不覆盖旧集，不重训 SFT，不启动正式 RL。
优先复用已有候选池、5 次采样记录与有效旧框；不足时用当前 SFT v5 补跑候选并按原质量标准选样，
需要的新框沿用逐页标注流程。所有新增题均检查去重、原始 gold 页、图片尺寸与框坐标，并记录标注来源。
不把结构校验、旧模型难度记录或自动标注当成当前模型的独立测试准确率。新增题不使用测试集。
奖励保持 §11.36 的 0.2 / 0.3 / 0.5；若后续保持全局 batch 16、group 5、单 epoch，则 1,600 题对应
100 个 rollout step、cosine + 20% warmup 对应 20 step，完整四卡配置另行验证。

**补齐完成（2026-11-05）：** 独立目录 `/workspace/data/slidevqa/rl_set_1600_kf` 含 1,600 题，
新增 400 题为 58 multi-hop + 342 single-hop；总计 567 multi-hop + 1,033 single-hop、3,980 个 key-fact 框。
新候选保持原来的五次采样 mixed-correctness 过滤及 single-hop 每 deck 上限 2；补充采样的 1,600 个候选
完成 8,000 次 rollout，已有候选记录同时保留使用。DeepSeek 判分用于训练选样，不属于独立评测结果。
用户明确授权百炼官方 API 的 `qwen3.7-plus` 处理新增题的 gold 页、问题和答案；所有最终新增题经过
逐页原图加候选框近图的自动复核。实际目视抽查 11 个候选，其中 10 个进入最终集，修正偏移、截字和
非必要证据，并排除 1 个标准答案有疑点的候选；此抽查不是全量人工核验，也不是统计准确率估计。

最终校验包括：原 1,200 条基线 parquet SHA-256 未变、导出前缀逐字段一致、val 文件字节未变、
新增去重、gold 页及其图片尺寸/框边界检查、正确 corpus 索引覆盖，以及实际 RL 环境加载全部 1,600 条。
训练文件 SHA-256 为 `5ec65000638e21c024ad618405f8b00207640f0482eb19e0fa7a7eaaa1b84547`。
`validation.json` 和 `annotation_qc_added400.jsonl` 保存实际检查结果及自动/目视复核范围。
本次不启动正式 RL；目标四卡通信和完整训练步仍需换卡后验证。
