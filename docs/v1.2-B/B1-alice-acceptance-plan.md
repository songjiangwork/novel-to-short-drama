# v1.2-B1 — Alice Real-Provider Acceptance Plan

状态：IMPLEMENTATION ACCEPTANCE PLAN（尚未执行）。Tracking: [#121](https://github.com/songjiangwork/novel-to-short-drama/issues/121)，[umbrella #108](https://github.com/songjiangwork/novel-to-short-drama/issues/108)。
权威：B1.1–B1.7 frozen contracts、B1 implementation plan、User Direction frozen guidance。此计划不是立即再次运行 Alice / Qwen 的授权；必须先由用户合并所有相应的 Draft implementation PR。

## 1. Immutable accepted Phase A source

| Item | Accepted value |
|---|---|
| Corpus | Alice's Adventures in Wonderland |
| project_id / document_id | a3e-real-novel / src_001 |
| A5 ConsolidationManifest | revision 1; 02973505f6a499e075d003120ad83305a42f315e5fe1b313a786e8182581903a |
| A6 StoryAnalysisManifest CURRENT | revision 2; bfdf63509d66a2872d3d7f3d9f43797f2e51e6509eba91193c7515a482f7f79d |
| A6 ValidationReport | revision 2; 6ec95609afd3e13e6fcef888d8bbaa9e0830d1d6913b0daf03f4f53a0c29c671 |
| A6 CURRENT pointer | revision 2; 932e21885c215221f6896f2499e7df8776cdf55ac90d7473efc67761df336c68 |
| A6 coverage | 47 canonical characters / 47 analyses; 166 canonical events / 166 owned; 14 windows; 2 arcs; 4 turning points; 2 reveals; 2 payoffs |
| A5 conflicts / unresolved | 0 source conflicts / 13 unresolved entities |

These values are cross-checks only: FileArtifactStore and FilePointerStore must resolve and verify real ArtifactRefs and lineage. Do not treat printed hashes as substitute for load/verify. A6 rev 2 is accepted and immutable; B1 must not modify A5/A6 source truth.

## 2. Gates in execution order

### Gate 0 — Read-only zero-provider input binding

- Resolve exact validated A6 CURRENT/manifest, leaves, validation and authoritative A5 manifest/IDs; fail closed on missing or mismatched refs, corrupted current or stale lineage.
- Establish exact source universe and 166 canonical events. Read-only, zero Qwen calls, zero A5/A6 writes or pointer movement.
- Ensure a fresh independent AdaptationPlan branch ID is reserved for new B1 acceptance. Never overwrite any existing approved branch.
- Existing A6 verification uses scripts/a6g_alice_acceptance.py --verification-only with actual arguments; avoid fresh A6 acceptance and incorrect python -m short_drama.cli invocation.

### Gate 1 — B1.1 target + G1 directions

- Explicit versioned Alice adaptation test profile: all hard/soft, fixed/target/range/derived settings identifiable; avoid untracked profile mutations.
- Test empty directions and selected stage-relevant B1 user guidance. Keep B3 visuals, B4 camera and B6 asset selections outside B1 semantic identity.
- Exercise negative test: 20 hard-fixed episodes × 90 hard-fixed seconds != hard-fixed 1200 total => fail closed before Qwen.
- Exercise hard contradictory directions with no explicit exception => fail closed before Qwen.

### Gate 2 — Zero-provider B1.2 shape/budget measurement

- Build actual full compressed Alice whole-story packet from exact accepted A6 outputs; targeted evidence expansion only where relevant.
- Measure serialized bytes, estimated tokens, source coverage, per-window constraints and actual LLM preparation envelopes. Do not invent numeric ceilings.
- Freeze stage-relevant semantic policy/profile ceilings only after measurement and before enabling that pass. Missing/deferred/oversized ceilings => zero provider calls.
- Use versioned prompt and output schema identities; separate operator runtime settings from semantic identity.

### Gate 3 — B1.2–B1.5 real-provider execution

- Run via existing local Qwen/llama.cpp production LLM adapter, with bounded technical retry and semantic regeneration; log actual semantic calls, request identities and exits.
- B1.2 uses a whole-story compressed context; produces thesis, dramatic promise, primary question and explicit strategy, not per-event final disposition.
- B1.3 gives **every one of 166 canonical events exactly one** keep/compress/merge_member/omit disposition. Non-omitted events require legal active mapping; omitted events have none; composites retain narrative-function provenance.
- B1.4 has coherent adapted chronology (partial order permitted), deterministic presentation sequence, critical continuity states, knowledge/reveal and setup/payoff constraints. Invented bridge beats have registered source anchors, purpose and divergence.
- B1.5 partitions all primary PresentationEntries exactly once into contiguous episodes; no reordering, splitting, new beats, filler or fake cliffhanger. Every episode has independent climax and ending-hook refs.
- Stage-owner repair routing must be explicit. No silent upstream modifications and no unbounded retries.
- Verify process exit 0 means actual provider work and output refs, not a no-op command.

### Gate 4 — B1.6 read-only integrated audit

- Verify exact B1.1–B1.5 constituent refs/hashes, A6/A5 lineage, hard target constraints, mapping graph, beat reachability, reveal/knowledge, episodes and duration ranges.
- Output immutable ValidationReport with FAIL/PASS_WITH_REVIEW_ITEMS/PASS, BLOCKING/REVIEW_REQUIRED/WARNING findings, earliest-owner routes.
- Emit authoritative-ID-backed deliberate-divergence, major-omission, invention and uncertainty registers plus ReviewManifestDraft.
- Audit can persist its report/projection but must not mutate upstream B1 artifacts or A-layer artifacts/pointers.

### Gate 5 — B1.7 actual human review

- Human reviews target, strategy, critical character/relationship/plot decisions, ending/reveals, deliberate divergences, important omissions/inventions, episode dramatic goals/climaxes/hooks and duration risks.
- Each REVIEW_REQUIRED must be explicitly ACCEPT_AS_DESIGNED or REVISION_REQUESTED; warnings remain visible. FAIL/BLOCKING/stale validation forbids APPROVE.
- The user may request revision or abandon even if technical validation passes. Revision request pins exact immutable artifact ID/revision/hash and returns to earliest owner; then downstream revalidation.
- **Never generate an APPROVE decision or human reviewer provenance on behalf of an unconsulted user.** Only after actual explicit reviewer approval persist ApprovalRecord and advance CURRENT_APPROVED.
- Until then verdict is TECHNICAL PASS / HUMAN APPROVAL PENDING. Whole-plan approval only.

### Gate 6 — Exact reuse + negative integrity verification

- Verification-only and exact reuse against already published B1 revision must have zero provider calls and zero new writes after validating its full input/manifest/report/approval authority.
- Runtime-only endpoint or slot changes must not invalidate unchanged semantic inputs; changed effective B1 direction must invalidate affected B1 request identity.
- Adding only B4 camera guidance must not change B1 effective direction hash/reuse.
- Reject malformed refs, tampered hashes, stale constituent/report mismatch, cross-revision composition and pointer corruption; no historical-scan fallback.
- Preserve A5/A6 CURRENT pointer and immutable A6 revisions.

## 3. Human narrative quality checklist

- Original Alice premise, important character identities, main arc and ending: changed only with tracked deliberate divergence.
- Important source turning points/reveals, omitted plots and composite characters: clear provenance and recognizable narrative-function transfer.
- Adapted causal/motivation/continuity chains, world truth vs character/audience knowledge and setup/payoff closure.
- Episode boundaries: hooks from existing skeleton, not fabricated excitement; climax distinct from hook; plausible estimated duration and dramatic density.
- Source unresolved entities are not silently guessed; A5 no-source-conflicts is never converted into narrative-conflict-as-source-evidence.
- Record critical artistic defects separately from purely technical validation, and route revision to earliest stage owner.

## 4. Acceptance report required

Record actual BASE/HEAD, source ArtifactRefs/revisions/hashes, B1 branch/revision/hash, target/profile/direction identities, constituent refs, all 166 event dispositions, adapted characters/beats/presentation entries/episode count and duration risk, semantic calls/retries, packet measured sizes/ceilings, ValidationReport findings, human dispositions, ApprovalRecord (or PENDING), current pointer status, fresh vs reuse call/write counts, command exit codes and narrative-quality findings. Never fabricate counts.

## 5. Command safety / final verdict

Acceptance CLI has not yet been implemented; do not document an imaginary working B1 CLI command. Once implemented, verify actual executable, args and paths. For shell logs create destination directories first, capture the real command exit using PIPESTATUS[0], and avoid assuming tee exit success means generation success. Ordinary WSL terminal is sufficient; tmux not mandatory.

**B1 ACCEPTED only when:** exact A-layer input + complete B1 plan + current B1.6 report with zero BLOCKING + all REVIEW_REQUIRED dispositions + genuine immutable human ApprovalRecord + approved exact AdaptationPlan revision + reproducible real-provider/zero-provider/reuse evidence + narrative review. Technical pass alone is not final B1 completion.
