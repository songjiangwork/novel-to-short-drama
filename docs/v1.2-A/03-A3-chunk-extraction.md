# v1.2-A3 — Chunk Extraction（冻结）

## 职责

Python 调用 Qwen，对单个 chunk 做局部故事语义抽取。

A3 只产生 candidate，不生成 canonical Story Bible。

---

## 输入

- SourceDocument
- Chunk artifact
- Story extraction profile
- versioned extraction prompt

---

## 输出

每个 chunk 一个 immutable extraction JSON：

```text
extractions/CH003_C002.json
```

六类输出：

```text
characters
locations
facts
events
relationships
unresolved_mentions
```

---

## 核心规则

### 1. Candidate only

A3 使用局部 candidate ID：

```text
cand_char_001
cand_loc_001
cand_fact_001
cand_evt_001
```

不得生成：

```text
char_0001
loc_0001
evt_000001
```

canonical identity 在 A4/A5 生成。

### 2. 中文为工作语言

- source text 保留原文
- summaries / descriptions / semantic fields 尽量中文
- 专有名词可保留原名

### 3. Context 可用于理解，但不能拥有 evidence

Qwen 输入必须明确：

```text
LEFT CONTEXT
OWNERSHIP SPAN
RIGHT CONTEXT
```

每个 extraction object 至少有一个 primary source reference 位于 ownership span。

### 4. 所有 claims 必须 source-backed

Python验证：

- source_ref 存在
- primary evidence 属于 ownership
- local candidate ref 有效

### 5. 不做 global interpretation

A3 不输出：

- story arc
- global story function
- adaptation
- cliffhanger
- screenplay
- production decisions

原则：

```text
extract first, interpret later
```

### 6. Unresolved 是合法产物

例如：

```text
他
the Captain
the woman
```

无法确认身份时，应输出 `unresolved_mentions`，不得强行猜测。

### 7. Evidence strength 使用枚举

推荐：

```text
explicit
implied
uncertain
```

不使用伪精确 0.83 confidence。

---

## Qwen / Python 分工

### Qwen

负责：

- 人物识别
- 地点识别
- 事件理解
- facts
- relationships
- pronoun / alias ambiguity
- 中文 semantic summaries

### Python

负责：

- prompt construction
- API call
- JSON parsing
- schema validation
- semantic validation
- state
- retry
- persistence
- cache / resume

---

## Retry

区分：

### Technical / invalid-output retry

例如：

- timeout
- invalid JSON
- schema violation
- invalid source refs
- ownership invariant violation

允许有限自动 retry。

### Semantic unresolved

例如：

```text
the Captain 到底是谁？
```

如果 Qwen 判断 uncertain，这是成功结果，不得 retry 到模型强行选一个答案。

---

## Resume / cache

Extraction identity 至少依赖：

- source revision SHA
- chunk definition
- prompt version
- LLM profile

只有全部一致，validated extraction 才可复用。

---

## 结构化 LLM capability

业务层不写死 llama.cpp URL。

应抽象为类似：

```text
LLMClient.generate_structured(...)
```

未来可以替换：

- local Qwen
- GPT
- 其它 OpenAI-compatible provider
