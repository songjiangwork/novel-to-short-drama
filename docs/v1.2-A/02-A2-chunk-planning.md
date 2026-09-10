# v1.2-A2 — Chunk Planning（冻结）

## 职责

将 `SourceDocument` deterministic 地规划成适合 LLM extraction 的 chunks。

A2 不调用 Qwen。

---

## 核心设计

采用：

```text
context span
+
ownership span
```

而不是简单 overlap chunk。

例如：

```text
Chunk 1
context:   P001 ───────────── P035
ownership: P001 ─────── P030

Chunk 2
context:          P026 ───────────── P065
ownership:              P031 ─────── P060
```

`P026-P030` 在 Chunk 2 里只能作为上下文，不能产生 canonical evidence ownership。

---

## 冻结 invariants

1. 每个 source paragraph 必须恰好属于一个 ownership span。
2. 一个 paragraph 可以作为 context 出现在多个 chunks。
3. ownership 不得遗漏。
4. ownership 不得重复。
5. chunk boundary 必须尊重 paragraph boundary。
6. token budget / context budget 属于 profile，不写死在 schema。
7. chunk planning deterministic。
8. coverage audit fail closed。

---

## Chunk artifact

建议：

```text
runs/<project>/story/chunks/CH003_C002.json
```

概念：

```json
{
  "schema_version": 1,
  "chunk_id": "CH003_C002",
  "document_id": "src_001",
  "chapter_id": "CH003",
  "context_span": {
    "start": "CH003_P0021",
    "end": "CH003_P0060"
  },
  "ownership_span": {
    "start": "CH003_P0026",
    "end": "CH003_P0055"
  },
  "paragraph_ids": [
    "CH003_P0021",
    "...",
    "CH003_P0060"
  ],
  "token_count": 7621
}
```

Chunk artifact 不复制全文；通过 paragraph IDs 回读 SourceDocument。

---

## Chunk Manifest

```text
chunk_manifest.json
```

必须包含 coverage：

```json
{
  "chunk_count": 147,
  "coverage": {
    "paragraphs_total": 4821,
    "owned_once": 4821,
    "unowned": 0,
    "multiply_owned": 0
  },
  "state": "CHUNKING_COMPLETE"
}
```

只有：

```text
owned_once == paragraphs_total
unowned == 0
multiply_owned == 0
```

才能进入 A3。
