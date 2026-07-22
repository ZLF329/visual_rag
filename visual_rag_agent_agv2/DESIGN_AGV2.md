# AGv2 — merged-action protocol refactor (2026-07-06)

用户指令:动作空间对齐 UniDoc-RL(search/bbox/answer),update_graph 内联到动作之前,
crop 只跟在 accept/expand 后(prompt 引导),crop 证据写进源图对应的 node,facts 不带 bbox。
目标:提速(去掉独立 update_graph / answer 轮)+ 加回 crop 并适配 tree memory。

## 1. 响应格式(每轮)

```
<think>...</think>
<update_graph>{...}</update_graph>     # 有 pending observation 时必须出现;没有时禁止出现
<search>query</search> | <bbox>[x1,y1,x2,y2]</bbox> | <answer>final</answer>   # 恰好一个
```

- 第一轮(无 observation):`<think>` + 动作(通常 search)。
- search 返回页面图 → 下一轮必须先 `<update_graph>` 提交该页(accept/expand/reject),再给动作。
- bbox 返回 zoom 图 → 下一轮必须先 `<update_graph>` 提交 crop 证据,再给动作。

## 2. 关键设计决策(用户未明说处,按最合理取)

**D1. `<answer>` 隐式 finalize root。** 多跳收尾 = 同一轮里 update_graph accept 最后一个 leaf
+ `<answer>`。root 不再需要显式 accept:answer 到来时,env/agent 把 root 置为 sufficient、
root.answer = answer 内容。省掉旧协议的"root accept 轮"和"纯 ANSWER 轮"。

**D2. crop 目标 node 的确定。** 页面被 accept/expand 提交时,记录
`crop_source_image = 该页原图` 和 `crop_target_node = 事实写入的 node`
(accept → active node 本身;expand → answered_child)。
`<bbox>` 永远裁剪 crop_source_image(不裁剪 crop 的 crop,和 UniDoc 一致);
crop-commit 的 facts 追加到 crop_target_node。
reject 不设置 crop 目标 → reject 后 bbox = 结构违规。

**D3. crop-commit 的语义。** crop 观察后的 update_graph:
- accept 或 expand → facts 追加到 crop_target_node(不改变图结构、不改 active、不改 status;
  用户原话"把新的visual evidence 加入到这张图对应的那个node 的visual fact里面")。
- reject → 丢弃(zoom 没提供新信息),不写任何东西。
crop-commit 后动作仍自由:可 search / 再 bbox(同一源页)/ answer。

**D4. facts 不带 bbox。** SupportingFact 退化为纯字符串;所有 bbox 校验/coerce 代码删除。
=> R_grounding(基于 fact bbox 的 IoU)自然死亡:terminal reward 里 grounding 项恒 0(留钩子)。
episode reward 变为 1[ans] + W_r·R_hit_all_gt(+ 未来可加 crop-box grounding)。

**D5. 结构违规 = format_error(train==eval 一致,两侧都终止)。**
- 响应解析失败(无 think / 动作 tag ≠ 1 / update 块位置错)
- 有 pending obs 却没有 update_graph;没有 obs 却给了 update_graph
- bbox 无 crop 目标(reject 后 / 从未 accept)
- answer 空
RL:episode -1 + terminate(现有 format_error lane);eval:terminated_by=policy_error。
坐标格式错误的 bbox(4 数、x2>x1 等不满足)= 温和错误:RL step -1 不终止,
observation 返回错误文本(沿用旧 bbox_error lane 精神);eval 同样继续。

**D6. bbox 坐标 = 展示图绝对像素(UniDoc 式,放弃 0-1000 归一)。**
displayed→raw 线性映射,±28px(raw)padding,clamp,crop 后 smart_resize_crop
(≥ACTIVE_GRAPH_MIN_PIXELS,32 对齐)。

**D7. Valid-actions 行删除。** 合并协议下约束是结构性的(见 D5),per-turn 动作 mask 无意义。
prompt 尾部加一行状态提示:"Pending observation: commit it with <update_graph> first." /
"No pending observation: do NOT emit <update_graph>."。

**D8. 多跳 gate 重写。** answer 时:root 有 children 且存在仍 open 的非 root node → -1
(扩展了却没解决完就抢答)。从未 expand 的多跳题不罚(同现状)。

**D9. 一个解析器、一套实现。** 新建 `src/protocol.py`(parser + crop 映射 + crop-commit +
finalize_root + 提示行),eval 的 agent.py 和 RL 的 envs.py 都从它 import
(延续 rl-eval-alignment recipe)。envs.py 里自己的 parse_slidevqa_action 删除。

## 3. 报酬(RL)变化摘要

| 通道 | 旧 | 新 |
|---|---|---|
| episode | 1[ans]+0.5·R_grounding+0.5·R_hit | 1[ans]+W_r·R_hit(grounding 项=0 留钩子) |
| search step | top1-new-GT {0,1} | 不变 |
| update_graph step | type-match {0,1} + bbox_problem -1 | page-commit type-match {0,1};crop-commit 0(防刷);bbox_problem lane 删除 |
| bbox step | (不存在) | 0;坐标格式错 -1 不终止 |
| format_error | -1 + terminate | 不变,覆盖面按 D5 |
| 多跳 gate | expand 后未 accept root → -1 | expand 后仍有 open 非 root node 时 answer → -1 |

## 4. 删除清单
- eval:UPDATE_GRAPH/ANSWER 独立轮、valid_actions mask、CROP cells/image_id 老参数、
  POLICY_JSON_* 与 UPDATE_GRAPH_* prompt、vlm.update_clue_graph 回退、Memory 镜像逻辑、
  确定性 fallback 动作。
- schemas:SupportingFact.bbox_2d 及全部别名 coerce;PolicyActionResult 的 CROP cells。
- envs.py:_coerce_supporting_facts_bbox、accepted_page_boxes(fact 框)、
  parse_slidevqa_action(active_graph 路径)、bbox_problem lane。
- 兼容性:不留旧格式开关。旧 checkpoint 用旧代码(box 上 .bak / git)评。SFT 数据需按新格式重生成。

## 5. 冒烟计划
parser 全组合单测;graph 转移(单跳 accept+answer / 多跳 expand→accept→accept+answer /
reject 重搜 / crop 链);crop 坐标映射数值;违规矩阵(D5 每一条);
脚本化假 VLM 全 episode;box 上 3 样本真 vLLM 干跑(旧模型会格式违规——只验 harness 不崩、
终止路径正确)。

## 6. 设计迭代补充(2026-07-07,与用户讨论定案)

**D10. crop 期间 active 钉住(pin/resume)。** accept/expand + 同轮 bbox 时,active 不立即迁移,
钉在 crop_target 上(crop 是对该子问题的补充);被推迟的迁移(accept→parent / expand→remaining)
存在 CropContext.resume_active_node_id,finish_crop_chain 在链结束轮执行。crop 轮 hint 明示:
"Pending ZOOM observation of <page> ... adds its facts to node [Nx]"。

**D11. commit-only 轮合法。** <update_graph> 后可省略动作,下轮看到更新后的渲染图再行动。
语法保留,不作为教学默认。

**D12. 多跳/单跳收尾 = MERGED(final commit + answer 同轮)——定案。** 调研三个 baseline
(UniDoc-RL template step6、VISOR Algorithm1+VerificationHint、VRAG-RL)全部是"看到最后一张图
直接答",无一有独立 answer 轮;VISOR(结构化记忆+滑窗,与我们最像)即 merged。separate 的代价:
单跳(74% 数据)2→3 轮 = rollout 生成 +50%,3B 多一轮格式风险;收益(统一渲染下 synthesis)无直接
证据。SFT 重生成时可做 A/B(多跳 final 两种风格各教一半,500-eval 定量),env 无需改动。

**D13. update_graph 字段顺序:supporting_facts 在 answer 之前**(先证据后结论的自回归顺序),
prompt 全部示例已统一。

**D14. teacher 坐标帧转换(norm1000 → displayed px),2026-07-09。** Qwen3-VL 系 teacher
(235B-Thinking)的 grounding 原生输出 0-1000 归一化坐标,无视 prompt 的"绝对像素"指令——
smoke4 的"暗 crop"实锤:框 [280,410,400,650] 按 px 读是 559 高页面的底部黑桌面,按 0-1000 读
恰是它声称要 zoom 的头像带(y 229-363px)。协议 canon 不动(displayed-px,匹配学生
Qwen2.5-VL 的绝对像素先验 + UniDoc 口径);生成侧 Agent(bbox_frame="norm1000") 在解析后立即
把 0-1000 框换算成 displayed px(源页 prompt_image 尺寸;zoom 重试链用 crop_ctx.displayed_size),
**并把 <bbox> payload 重写进 raw_text**,故落盘 SFT target / 后续轮上下文全部是像素帧,学生
永远只见 canonical 帧。守卫:任一坐标 >1000 视为已是像素、不转换。teacher yaml agent.bbox_frame:
norm1000;学生 SFT/eval/RL 默认 displayed_px 不受影响。受污染数据处置:frame 修复前生成的带
bbox 轨迹(run2 的 136/527)全部弃用重跑。
