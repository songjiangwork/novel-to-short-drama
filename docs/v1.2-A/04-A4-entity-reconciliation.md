# v1.2-A4 — Entity Reconciliation（冻结）

## 职责

把 A3 各 chunk 的局部 character/location candidates 保守地合并成全局 canonical entities。

核心原则：

> 错误 merge 比暂时 unresolved 更危险。

因此采用 precision-first reconciliation。

---

## 输入

A3 immutable chunk extractions。

Candidate 的全局引用使用：

```text
CH003_C002:cand_char_001
CH003_C002:cand_loc_001
```

---

## 输出

建议：

```text
reconciliation/
├── candidate_index.json
├── reconciliation_decisions.json
├── entity_map.json
├── canonical_characters.yaml
├── canonical_locations.yaml
└── unresolved_entities.yaml
```

---

## Python / Qwen 分工

### Python 主导

- name normalization
- candidate indexing
- deterministic exact matches
- must-not-merge constraints
- candidate blocking
- graph construction
- conflict detection
- stable canonical ID assignment
- coverage audit

### Qwen 只处理 ambiguity

固定分类：

```text
same_entity
different_entity
uncertain
```

不得直接修改 canonical registry。

---

## 冻结原则

1. A3 extraction artifacts immutable。
2. Canonical entities 是 candidates + reconciliation decisions 的派生产物。
3. Auto-merge 只允许高置信 deterministic identity case。
4. Ambiguous identity 由 Qwen 返回 same/different/uncertain。
5. `uncertain` 是合法结果。
6. must-not-merge constraints 高于字符串相似性。
7. graph contradiction fail closed。
8. 每个 candidate 必须 resolved 或 explicit unresolved。
9. canonical IDs deterministic，按 earliest source appearance 分配。
10. A4 不做故事解释、人物总结或 production decision。

---

## Safe deterministic merge

第一版仅建议自动处理：

- normalized full name 完全相同且无 hard conflict
- 原文明确 alias / identity reveal

其它如：

```text
John
Mr. Smith
Captain
the teacher
Miss Carter
```

默认进入 ambiguity handling。

---

## Must-not-merge

强 negative evidence 包括：

- 同一事件明确作为两个不同参与者出现
- 原文明确表示是不同人
- explicit incompatible identity evidence

年龄、职业、外貌推断通常只能作为 soft evidence。

---

## Candidate blocking

不做全组合 pairwise comparison。

Python 根据：

- name overlap
- surname overlap
- explicit aliases
- descriptors
- source proximity
- shared relationships

构建小的 ambiguity blocks。

Block 仅表示“值得比较”，不表示 merge。

---

## Reconciliation decision artifact

每个 semantic decision 应保留：

```text
decision_id
candidate refs
decision
method: deterministic | llm | manual
prompt_version
reason_zh
evidence_refs
```

`reason_zh` 只用于 debug，程序逻辑依赖结构化 decision。

---

## Graph canonicalization

`same_entity` decision 形成 graph edges。

connected component → one canonical entity。

如果出现：

```text
A same B
B same C
A different C
```

则：

```text
RECONCILIATION_CONFLICT
```

fail closed。

---

## Unresolved

Story pipeline 允许 unresolved entity。

成功条件是：

```text
resolved + unresolved == total candidates
unaccounted == 0
```

不是强制全部 resolved。
