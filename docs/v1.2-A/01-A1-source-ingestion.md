# v1.2-A1 — Source Ingestion（冻结）

## 职责

将 TXT / text-based PDF 可靠地转换为 immutable `SourceDocument`。

A1 不做故事理解，不调用 Qwen。

---

## 输入

```text
novel.txt
或
text-based novel.pdf
```

---

## 输出

推荐 canonical artifact：

```text
source_document.json
```

机器生成、原文量大，因此 JSON 优先。

---

## 冻结原则

### 1. SourceDocument 是 immutable evidence snapshot

摄取时记录：

- source format
- original path
- SHA256
- encoding / normalization metadata
- detected source language

原文件内容变化时，不静默覆盖旧 revision；必须视为新的 source revision。

### 2. 原文永远保留

`text_original` 只保存原语言文本。

英文小说不预先全文翻译成中文。

中文工作语言从后续 semantic extraction 开始使用。

### 3. Paragraph 是 canonical provenance 最小单位

推荐：

```text
CH001_P0001
CH001_P0002
```

PDF page number可以保存为辅助 metadata，但不是 canonical identity。

### 4. Chapter detection 先 deterministic

支持常见：

- Chapter 1 / Chapter One
- Part II
- Prologue / Epilogue
- 第一章 / 第十二章
- 第一卷 / 序章 / 楔子 / 尾声

没有可靠章节时，建立 deterministic synthetic sections。

不由 LLM 判断章节边界。

### 5. Stable IDs 不依赖 LLM

所有 chapter / paragraph IDs 必须由 Python deterministic 生成。

---

## 概念结构

```json
{
  "schema_version": 1,
  "project_id": "my_novel",
  "document_id": "src_001",
  "source": {
    "type": "txt",
    "path": "inputs/novel.txt",
    "sha256": "...",
    "detected_language": "en"
  },
  "normalization": {
    "encoding": "utf-8",
    "newline": "LF"
  },
  "structure": {
    "chapters": [
      {
        "chapter_id": "CH001",
        "title_original": "Chapter One",
        "paragraphs": [
          {
            "paragraph_id": "CH001_P0001",
            "text_original": "..."
          }
        ]
      }
    ]
  }
}
```

PDF paragraph 可附带：

```json
{
  "source_pages": [12, 13]
}
```

---

## Fail-closed

如果：

- 文档读取失败
- normalization 失败
- paragraph ID 重复
- source artifact 不完整

则不能进入 A2。
