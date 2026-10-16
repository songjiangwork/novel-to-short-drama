# v1.2-B / B1 — Implementation Plan

状态：**IMPLEMENTATION PLAN（非架构变更）**  
基线：`main f2d874088400d0c6d93a308943c068e34ed1a5e8`（Phase A / Alice A6 rev 2 accepted）  
维护原则：此文档是任务顺序与验收计划，不替代既有 FROZEN architecture；若有冲突，以冻结文档为准。

## 1. Authoritative references

- `docs/v1.2-B/00-v1.2-B-B1-overview.md` 和 `01-` 至 `07-` 的 B1.1–B1.7 冻结 contracts。
- `docs/v1.2-user-direction-revision-guidance.md`（用户指导、ownership、review/revision、effective direction identity 的冻结设计）。
- `docs/v1.2-implementation-task-map.md`（整体 roadmap；B1 在此文档细化）。
- `docs/v1.2-B/B1.1-implementation-plan.md`（第一实施 Slice）。
- `docs/v1.2-user-direction-implementation-plan.md`（G1 共享 primitives）。
- `docs/v1.2-B/B1-alice-acceptance-plan.md`（真实验收）。

跟踪总 Issue：[B1 #108](https://github.com/songjiangwork/novel-to-short-drama/issues/108)。

## 2. B1 的工程目标和不变量

输入是**已验证、精确绑定的 A6 CURRENT**（`StoryAnalysisManifest` + `GlobalStoryBible` + A6 global analysis）和它所引用的 authoritative A5 canonical evidence；不是“任意最新文件组合”，也**不要求重新跑 A1–A6**。A6 的 validation/CURRENT 与 B1 的 formal human approval 是不同概念，禁止把 A6 称为已经取得 B1 ApprovalRecord。

输出是：

```text
Validated A6 StoryAnalysisManifest (exact ref)
       + its authoritative A5 canonical evidence
       + B1.1 ResolvedAdaptationTarget
       + Effective B1 UserDirectionSet (possibly empty)
                     ↓
B1.2 strategy → B1.3 disposition/mapping
              → B1.4 story/presentation skeleton
              → B1.5 episode partition
                     ↓
B1.6 immutable integrated ValidationReport + ReviewManifest projection
                     ↓
B1.7 immutable ApprovalRecord + CURRENT_APPROVED
                     ↓
exact APPROVED AdaptationPlan revision → B2
```

- A 层 `CanonicalEntity/Fact/Event/Relationship`、GlobalStoryBible 不变；B1 决策、编造、压缩或有意偏离只能记录在 adaptation-owned artifacts。
- Source Event、AdaptedStoryBeat、PresentationEntry、Episode 与未来的 Screenplay Scene/Shot 都是不同 identity。
- Python 负责 ID、排序、算术、引用、hash、revision、publication、validation；Qwen 负责有界 semantic proposal/review。
- B1.6 **read-only verify+diagnose+route**，不能自动修复。
- B1.7 是 B1 唯一正式 Human Approval Gate；无有效 ApprovalRecord 的 revision 不能成为 B2 authoritative input。
- 默认无用户指导可以自动规划、验证和生成 review material；**自动运行不等于自动代表用户审批**。生产 authoritative B2 handoff 仍需 human approval。
- 不实现 B2–B7、H3/ComfyUI、asset authoring、UI、跨项目 stale engine 或数据库。

## 3. 实施 Slice 与 Issue 顺序

| 顺序 | Issue | 内容 | 依赖 / 交付边界 |
|---|---|---|---|
| 1 | [#109 B1.1](https://github.com/songjiangwork/novel-to-short-drama/issues/109) | Target contracts + deterministic resolution | 零 Qwen，无 EpisodePlan |
| 2 | [#110 G1](https://github.com/songjiangwork/novel-to-short-drama/issues/110) | Shared UserDirection/RevisionRequest + effective projection | 跨 B1–B6 可复用；不实现 UI / 下游 adapters |
| 3 | [#111 B1.2](https://github.com/songjiangwork/novel-to-short-drama/issues/111) | Whole-story AdaptationStrategy | #109、#110；首个 Qwen pass |
| 4 | [#112 B1.3-A](https://github.com/songjiangwork/novel-to-short-drama/issues/112) | Adapted identities/disposition/mapping contracts | #111；不跑语义窗口 |
| 5 | [#113 B1.3-B](https://github.com/songjiangwork/novel-to-short-drama/issues/113) | Windowed disposition + reconciliation | #112；exact-one source event |
| 6 | [#114 B1.4-A](https://github.com/songjiangwork/novel-to-short-drama/issues/114) | Skeleton, PresentationEntry, state, reveal contracts | #113 |
| 7 | [#115 B1.4-B](https://github.com/songjiangwork/novel-to-short-drama/issues/115) | Semantic global skeleton + bounded bridge repair | #114；不可修改 B1.3 decisions |
| 8 | [#116 B1.5-A](https://github.com/songjiangwork/novel-to-short-drama/issues/116) | Deterministic partition feasibility | #115；不创造 episode drama |
| 9 | [#117 B1.5-B](https://github.com/songjiangwork/novel-to-short-drama/issues/117) | Episode dramatic planning + continuity checkpoints | #116；不可改 PE 顺序 |
| 10 | [#118 B1.6-A](https://github.com/songjiangwork/novel-to-short-drama/issues/118) | Whole-plan deterministic audit | #117；read-only |
| 11 | [#119 B1.6-B](https://github.com/songjiangwork/novel-to-short-drama/issues/119) | Whole-season semantic audit / review registers | #118；read-only |
| 12 | [#120 B1.7](https://github.com/songjiangwork/novel-to-short-drama/issues/120) | Whole-plan review, approval and publication | #119；复用 Foundation ApprovalRecord |
| 13 | [#121 Acceptance](https://github.com/songjiangwork/novel-to-short-drama/issues/121) | Alice real-provider + zero-provider/reuse/approval verification | #120；不可自动代表 reviewer 批准 |

G1 是共享底座开发 Slice，**不是新增 B1.1.5 语义阶段或第二个人工 gate**。前两项可以独立实现，但 B1.2 消费 user directions 前必须具备有效的 G1 projection contract。

Issue 先创建跟踪，**不等于要求一次创建 13 个 PR**。每项默认独立 Draft PR；后续如实现发现确属同一小边界，应先说明再调整拆分，不为遵守 Issue 数量牺牲 Implementation Minimalism。

## 4. Foundation / A6 reuse matrix

| 已有实现 | B1 处理 |
|---|---|
| `short_drama.artifacts.models.ArtifactRef / ImmutableArtifactEnvelope` | 直接复用，不能创造第二套 ref/hash/envelope |
| `short_drama.artifacts.store.FileArtifactStore` | 直接复用 immutable get/put；验证已存在目标的 integrity |
| `short_drama.foundation.lineage.LineageRef` | 精确上游 refs、B1 constituent refs |
| `short_drama.foundation.pointers.FilePointerStore` | publication CAS；approval 使用 `CURRENT_APPROVED` |
| `short_drama.foundation.validation.ValidationReport / ValidationFinding` | B1.6 直接复用，补充 B1-domain checks |
| `short_drama.foundation.approval.ApprovalRecord` | B1.7 直接复用；**另外核查完整 constituent compatibility**，不能假定 generic approval 已完成所有 B1 专属校验 |
| `short_drama.story.story_analysis_persistence` | 参考 A6 exact binding、whole-validation、publication、reuse 模式；不复制整个模块或其 A6-specific identity |
| `short_drama.llm`、PromptRegistry、versioned profiles | 复用 semantic call/provenance/transport/retry 基础；不得重新造 LLM client |

建议最小增加 B1 专属 typed models、services、semantic preparation、validators、persistence adapter。模块命名/粒度由实现审查决定；**不得为未来 B2–B7 预造巨型 domain framework**。

B1 的语义 revision 应 pin 已消费的 exact A6/A5 refs、ResolvedTarget、stage-relevant direction set、profile/prompt/output-schema identity 及实际上游 semantic pass outputs。Reuse 必须先完整验证现有 CURRENT，而不是因新输入不同就绕过历史 corruption。不得历史遍历式找可复用 revision，不得 hidden latest。

## 5. Per-stage input / output ownership

### B1.1 Target

Raw structured intent + profile defaults + explicit overrides → ResolvedTarget。绝不做剧情保留、删改或分集。Duration 和 episode count 可为 fixed/target/range/derived；hard/soft 是正交约束；B1.5 最终确定 resolved count。自然语言不可由 Python 关键词 heuristics 擅自解释。见 `B1.1-implementation-plan.md`。

### B1.2 Strategy

以**全书 compressed global representation**、A6 key indexes、target、有效 B1 guidance 为输入。输出 thesis、dramatic promise、primary question、plot/character/relationship/reveal/ending strategy 和 priorities。重大 source divergence、target-source conflict 明示；不能直接生成 final event dispositions。首次 provider 前必须 zero-provider 计量完整 packet；任何 token/bytes ceiling 必须来源于实测或已冻结的明确策略，不允许先猜数填生产 profile。

### B1.3 Source disposition

全部 canonical events 恰好一次 disposition：`keep | compress | merge_member | omit`。Fact/Relationship adaptation selectively materialized；Source↔Adapted mapping 是一等多对多 graph。每条保留的 source event 有合法 active mapping；omit 不能有 active mapping。AdaptedCharacter 与 A4 canonical identity 分离，composite 是 narrative-function transfer。窗口冲突显式 reconciliation；Python 不使用文本长度推断事件应删除。

### B1.4 Skeleton

在 B1.3 的 surviving adapted identities 上建立 **adapted chronology（可 partial order）** 与 **primary presentation sequence（确定性可分集）**。World truth、character knowledge、audience knowledge 明确分离。Reveal/withholding、causal prerequisites、setup/payoff、arc checkpoints、minimum continuity overlay，重要 gaps/intentional ambiguity 都需结构化。只有注册过 purpose/provenance 的 invented beat 可作为 Qwen bridge；改变 disposition/strategy 必须向上游返回，而非静默 patch。

### B1.5 Partition

只对已冻结的 primary PresentationEntry 顺序做 contiguous partition；每个 primary entry 恰好属于一集。不得 reorder、split PE、造新 beat。Hard constraints 和 no-cut 先通过确定性可行性检查；Qwen 才决定 dramatic goals、boundary preference、climax、hook。Estimated duration 应保留区间；弱但合法 boundary 可以 review，不用 filler 凑时长。Hook 和 climax 必须锚定现有 skeleton。产出 sparse inter-episode continuity checkpoint。

### B1.6 Audit

严格只读。绑定 complete exact B1.1–B1.5 refs/revisions/hashes；检查 identity、mapping coverage、continuity、reveal、setup/payoff、arc、episode coverage、target/budget。Findings 分 `BLOCKING | REVIEW_REQUIRED | WARNING`，结果 `FAIL | PASS_WITH_REVIEW_ITEMS | PASS`，并标注 earliest owner。ReviewManifestDraft 只是 authoritative B1 artifacts 的 index/projection；不复制 source/adapted truth。

### B1.7 Review / Approval

只对 exact immutable AdaptationPlan revision 作出 `APPROVE | REVISION_REQUESTED | ABANDON`。每个 REVIEW_REQUIRED 必须 `ACCEPT_AS_DESIGNED` 或请求修订；不能默许。RevisionRequest pin exact revision/hash，返回最早 owner 重新生成受影响 downstream，再验证；不在审批阶段修改剧情。Approve 复用 independent immutable ApprovalRecord 和 `CURRENT_APPROVED` CAS。B2 只消费 exact approved revision + approval ref。Formal approval 全剧一次，不允许部分 episode approval。

## 6. User direction integration and identity

冻结入口：`UserDirection / UserDirectionSet / RevisionRequest`；raw chat **不是** production authority。类型/生命周期/strength/owner 参见独立 G1 文档。B1 stage 只消费 relevant EffectiveUserDirectionSet；例如 B4 camera direction 变化不改变 B1 semantic request identity。UserDirection 合规与 source provenance 都可回溯。Unresolved hard direction conflict 不能被 Qwen last-write-wins。

角色故事/戏份改编 → B1；对白/动作 realization → B2；外观/服装/舞台空间 → B3；画面和 camera → B4；实际图片 reference 选用 → B6。**人物视觉资产优先于 shot guidance 实现**，但均不应进入 B1 target/skeleton authority。

## 7. Testing gates

每个实施 PR 最低要求：

1. Domain/schema round-trip、stable serialization/hash、empty/edge/error cases；
2. Exact ArtifactRef / missing/tampered/stale/cross-revision mismatch 测试；
3. Critical invariant deterministic negative tests（未满足返回 fail closed）；
4. Provider-facing stage 的 mock/fake tests 先通过，实际 local Qwen 运行放在对应 acceptance gate；
5. Runtime-only config 改变不应污染 semantic identity；有效 user direction 改变按 owner 精确 invalidates；
6. `git diff --check`、targeted tests、必要的 full tests 与真实退出码；
7. 不应更新 A1–A6 的 source artifacts 或 CURRENT。

First provider gates：先 real-corpus **zero-provider, read-only** input audit（exact packet shape、cost/budget、canonical ID coverage），再明确启用 semantic execution。杜绝 A6 曾出现的“CLI exit 0 但没有真的执行任务”，所以验收报告必须有 semantic-call 统计和输出 refs。

## 8. Milestones / readiness

- M-B1T：#109 + #110 → deterministic target/guidance primitives。**无 Qwen**。
- M-B1S：#111–#115 → 全局 adaptation semantic materialization、mapping/skeleton。需要 zero-provider measurement + bounded provider passes。
- M-B1E：#116–#117 → 有效 EpisodePlan、hard partition feasibility。
- M-B1V：#118–#120 → exact whole-plan validation + formal approval。
- M-B1A：#121 → Alice real-provider / human narrative QA / exact reuse；满足用户批准后的完整 B1 acceptance。

## 9. Definition of Done

```text
exact validated A6/A5 inputs
+ complete B1.1–B1.5 artifacts
+ current exact B1.6 ValidationReport (no BLOCKING)
+ all REVIEW_REQUIRED explicitly accepted
+ valid immutable B1.7 ApprovalRecord
+ approved AdaptationPlan ref and authority/current-approved pointer
= B1 COMPLETE
```

Technical PASS / zero warnings 不等于 human approved，更不等于 commercial-grade adaptation quality。测试小说 Alice 的人工可读审查必须单独记录。

## 10. Development policy

工作语言 zh-CN；Python orchestrates / Qwen semantic；small Issues + Draft PR，用户自己 merge（除非明确授权）；Implementation Minimalism；先诊断再改代码/冻结文档。文档提交不会自动授权实现 PR merge 或改变冻结 architecture。
