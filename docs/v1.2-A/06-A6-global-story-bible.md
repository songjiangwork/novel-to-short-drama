# v1.2-A6 — Global Story Bible（冻结）

## 职责

将 A1–A5 的 canonical evidence 提升为整本小说的全局故事理解。

核心边界：

```text
A1–A5：建立可靠证据
A6：基于证据形成全局解释
v1.2-B：做改编决策
```

A6 不决定：

- 删除哪些剧情
- 合并哪些事件
- 改成多少集
- 每集 cliffhanger
- screenplay
- shot planning

---

## 输入

正常情况下消费：

```text
Canonical Characters
Canonical Locations
Canonical Facts
Canonical Events
Canonical Relationships
Conflicts
Unresolved Entities
```

必要时通过 provenance 回查原始 source paragraphs。

---

## 输出

建议：

```text
global/
├── character_analysis.yaml
├── global_event_analysis.json
├── arc_analysis.yaml
├── global_structure.yaml
├── story_bible.yaml
└── analysis_manifest.json
```

---

## 冻结原则

1. A6 正常情况下消费 A1–A5 canonical evidence，不重新解析全文。
2. A6 做 global interpretation，不做 adaptation decision。
3. Character / plot / relationship arcs 在 A6 首次正式产生。
4. 重要结论必须引用 canonical event/fact IDs。
5. inferred interpretation 必须与 explicit fact 区分。
6. A6 不得发明新的 canonical event/fact/entity。
7. global event importance 是 overlay，不修改 A5 artifact。
8. foreshadow / reveal / theme 只在有充分全局 evidence 时生成。
9. unresolved entity/conflict/ambiguity 必须保留。
10. StoryBible 是 high-level synthesis，不复制整个 evidence database。
11. A6 使用 hierarchical global analysis，不把最终判断建立在互相隔离的局部 1/4 数据上。
12. 所有 StoryBible cross-reference 必须由 Python fail-closed validation。

---

# A6 如何处理超大 A5 数据

## 不是这样

错误理解：

```text
A5 的 1/4 → Qwen → final analysis A
A5 的 1/4 → Qwen → final analysis B
A5 的 1/4 → Qwen → final analysis C
A5 的 1/4 → Qwen → final analysis D
```

然后直接拼接。

这样确实无法形成完整的全局分析。

---

## 正确设计：Hierarchical Synthesis

A6 分为多个 pass，但**最终 synthesis 必须看到整个故事的压缩表示**。

### Pass 1 — Character dossiers

Python 对每个 character 聚合：

```text
relevant facts
relevant events
relationships
state transitions
conflicts
```

Qwen 输出 character analysis。

每个 character 可以单独处理，但结果成为后续 global context。

### Pass 2 — Chronological / plot windows

将完整 canonical event stream 按 narrative order 分窗。

例如：

```text
Window 1: evt_000001–evt_000080
Window 2: evt_000061–evt_000140
Window 3: evt_000121–evt_000200
...
```

窗口有 overlap，避免跨边界剧情断裂。

每个窗口输出：

- major developments
- unresolved threads
- relationship changes
- candidate turning points
- candidate reveals
- arc continuation markers

这些不是最终 StoryBible，只是压缩 representation。

### Pass 3 — Global skeleton

Python 构建一个覆盖整本小说的紧凑 global index，例如：

```text
所有 canonical character IDs + short descriptors
所有 relationships + state summaries
所有 core/major events
所有 window summaries
所有 character dossiers
所有 unresolved/conflicts
候选 arcs / turning points
```

此时数据量远小于 raw A5，但**覆盖全书**。

### Pass 4 — Global synthesis

Qwen 最终一次或少数几次看到：

```text
complete global skeleton
+
character dossiers
+
window summaries
+
core/major event details
```

然后生成：

- premise
- synopsis
- main conflict
- main/sub plots
- character arcs
- relationship arcs
- turning points
- reveals
- foreshadow/payoff
- themes
- global event importance

因此最终 Qwen 并不是只看到 1/4 或 1/3。

它看到的是：

> 整本小说经过结构化压缩后的完整表示。

---

## Whole-story packet ceiling authority is staged

A6 的 character/window packet 可以在 A6B 通过 exact A5 snapshot 直接测量；但 global-skeleton 与 StoryBible 的完整 request 依赖前序 semantic outputs，因此它们的 ceiling 不能在 A6B 被诚实测量。

冻结顺序：

```text
A6B
→ freeze character evidence budget
→ freeze plot-window budget / ownership / overlap
→ global-skeleton ceiling = DEFERRED
→ story-bible ceiling = DEFERRED

A6C + A6D semantic outputs complete
→ construct exact complete A6E input
→ measure
→ freeze global-skeleton ceiling

A6E semantic output complete
→ construct exact complete A6F input
→ measure
→ freeze story-bible ceiling
```

`DEFERRED` 是正式 authority state，不代表 unlimited，也不得用 A5-only placeholder 数字冒充完整 whole-story ceiling。

任何 semantic pass 在自己的 ceiling 尚未 freeze 时必须 fail closed，并且不得调用 provider。

这也意味着是否需要 recursive compression 必须基于 actual complete compressed representation 的 measured size 决定，而不能仅根据 A6B 的 A5-derived base 推断。

### 修订：global-skeleton ceiling 已在 A6E gate 冻结为 100000

在 A6E-1 / A6E-2 测量后，global-skeleton ceiling 已按 staged authority 在 pre-A6E
gate 冻结；story-bible ceiling 仍为 DEFERRED = null（在 A6E output 存在后于
pre-A6F gate 冻结）。修订规则区分五层：

```text
1. 实测完整 semantic-input 基线
   73929 (A6E-1) 与 74404 (A6E-2 真实 provider)，观测变异 475。
   对 exact Alice A5 CURRENT 上的完整 A6E 输入包（A5-derived global index
   + 全部 A6C CharacterAnalysis + 全部 A6D PlotWindowAnalysis）。
2. 有界变异余量（针对非确定性 LLM 输出）
   不是一次测量向上取整到 1000 步，而是刻意选择的保守运行包络。
3. 冻结的最大 packet ceiling
   global_skeleton_packet_max_estimated_tokens = 100000。
   是上限（upper bound），不是目标输入长度；不授权截断 / 采样。
4. 实际 model-context 可行性（估算器口径，非权威 tokenizer 计数）
   n_ctx = 163840；冻结输出保留 max_output_tokens = 16384；
   约 83 estimated tokens 的 prompt framing：
   100000 + 83 + 16384 = 116467；context 余量 163840 - 116467 = 47373。
5. Fail-closed 强制
   ceiling 为 null 或 packet 超限 → 零 provider calls、拒绝运行（不截断）。
```

这是刻意保守的运行余量：headroom 25596 estimated tokens（100000 - 74404），
约 34.4% 的 margin。它不是统计意义上建立的 worst-case 上界，也不代表任意小说的
通用 100K 预算，也不引入通用自适应运行 ceiling。物理 GPU / slots / NP /
endpoint / provider model 不进入 semantic identity。完整测量依据见
docs/v1.2-A6E-1-global-skeleton-measurement-report.md。

---
## 如果 262K 足够怎么办？

如果压缩后的 A5 canonical evidence 可以安全放进 262K，并仍保留足够输出空间，则 A6 final synthesis 可以直接读取整个 compressed evidence package。

不要为了“必须分块”而分块。

原则是：

```text
raw full evidence 太大 / 噪声太多
→ hierarchical compression

compressed complete representation 能放下
→ final global synthesis 一次读取完整 representation
```

---

## 如果连 compressed representation 也超过 context

继续做递归层级，而不是丢掉全局信息：

```text
events
→ window summaries
→ act/section summaries
→ global skeleton
→ final synthesis
```

每一级都必须保存 stable IDs，允许 final synthesis 回查：

```text
arc candidate
→ window
→ event IDs
→ fact IDs
→ source refs
```

---

## Retrieval / targeted expansion

Final synthesis 如果某个 arc 需要更详细 evidence，可由 Python针对：

```text
char_0003
arc candidate
evt_000142
```

构造 targeted evidence package，再让 Qwen refinement。

这不是 RAG 必需条件；第一版可以直接根据 canonical IDs 做 deterministic retrieval。

---

# A6 内部 Pass

推荐：

```text
A6.1 Character Analysis
A6.2 Plot-window Analysis
A6.3 Global Skeleton / Arc Candidate Analysis
A6.4 Story Bible Synthesis
```

---

## Character Analysis

对单角色输出：

- role
- goals
- motivations
- traits
- key events
- relationships
- character arc

重要判断引用 supporting event/fact IDs。

---

## Plot / Arc Analysis

输出 candidate：

```text
plot arcs
character arcs
relationship arcs
turning points
reveals
```

Arc IDs 最终由 Python deterministic 分配。

---

## Global Structure

最终输出：

```text
premise_zh
synopsis_zh
genre
tone
setting
main_conflict
secondary_conflicts
main_characters
main_plot
subplots
ending_state
themes
```

---

## StoryBible

`story_bible.yaml` 是高层 canonical synthesis。

不复制全部 facts/events，只引用 IDs。

保留：

```text
unresolved entities
source conflicts
story ambiguities
coverage metadata
```

---

## Completion gate

Python必须验证：

- A1–A5 COMPLETE
- 所有 char_id 存在
- 所有 loc_id 存在
- 所有 event_id 存在
- 所有 fact_id 存在
- 所有 relationship_id 存在
- 所有 conflict_id 存在
- no hallucinated cross-reference
- coverage complete

存在显式 unresolved 不阻止 `STORY_BIBLE_READY`。

structural inconsistency 必须 FAILED。
