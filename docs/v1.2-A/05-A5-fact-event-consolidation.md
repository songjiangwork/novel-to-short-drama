# v1.2-A5 — Fact / Event Consolidation（冻结）

## 职责

在 A4 已解决“谁是谁”之后，整理：

- canonical facts
- canonical events
- canonical relationships
- state transitions
- source conflicts

A5 保持 interpretation-light，不做 adaptation 或 screenplay。

---

## 输入

```text
A3 immutable extractions
+
A4 entity_map
+
canonical characters / locations
```

---

## 输出

建议：

```text
consolidation/
├── canonical_facts.json
├── canonical_events.yaml
├── canonical_relationships.yaml
├── story_conflicts.yaml
└── consolidation_manifest.json
```

---

## 冻结原则

1. A5 消费 immutable A3 + A4 artifacts。
2. Fact 与 Event 是两个独立 canonical 概念。
3. 每个 canonical fact/event 保持 source-backed。
4. deterministic blocking/dedupe 优先，Qwen 只做语义 ambiguity。
5. 后来的 facts 不得静默覆盖早期 facts。
6. Fact pair 分类：
   - same
   - compatible
   - state_change
   - conflict
   - unrelated
   - uncertain
7. conflicts 必须显式保留。
8. relationship identity 与 relationship state 分开。
9. Event duplicate consolidation 使用 graph-based approach。
10. canonical event IDs 按 narrative/source order deterministic 分配。
11. semantic uncertainty 合法；structural inconsistency fail closed。
12. A5 不做 adaptation、screenplay、global story-arc invention。

---

## Reference resolution

A5 首先将：

```text
local candidate ref
```

通过 A4 entity_map 转换为：

```text
char_XXXX
loc_XXXX
```

仍 unresolved 的参与者继续保留 unresolved identity，不丢 event。

---

## Fact

Fact 表达角色/世界状态。

第一版可支持：

```text
identity
appearance
possession
knowledge
relationship_state
location_state
world_fact
continuity_relevant
other
```

不要求所有自然语言都强行 attribute/value 化；允许 narrative fact。

---

## Fact consolidation

Python 先处理明显 duplicate。

语义 ambiguous pair 交给 Qwen 分类：

```text
same_fact
compatible_fact
state_change
conflict
unrelated
uncertain
```

任何 conflict 都不得 last-write-wins。

---

## Relationship

Relationship 是一等对象。

必须支持：

- relationship type
- direction
- state history
- transitions

例如：

```text
同学
+
戒备 → 初步信任 → 合作
```

关系可以非对称。

---

## Event

A3 candidate events 通过：

- source overlap
- participant overlap
- location
- narrative proximity

构建 possible duplicate groups。

必要时 Qwen 判断：

```text
same_event
different_event
uncertain
```

uncertain 时宁可保留两个 events。

---

## Canonical Event

应保存：

- event_id
- narrative_order
- summary_zh
- participants
- location
- candidate event refs
- source refs

canonical summary 可由 Qwen综合，但 IDs / participants / source refs 由 Python 控制。

---

## Narrative order vs chronology

必须分开。

允许：

```text
normal
flashback
flashforward
dream
memory
unknown
```

chronology 无法确定时不得猜。

---

## Continuity-relevant facts

A5 开始标记对后续 production continuity 有价值的 facts，例如：

- 受伤
- 得到/失去物品
- 换装
- 地点状态变化
- 身份揭露

但不做 production asset 决策。
