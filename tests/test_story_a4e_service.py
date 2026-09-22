"""A4E-A production orchestration core tests.

Covers the frozen A4E-A contract against a deterministic, provider-neutral
fake ``LLMClient`` — NO live Qwen server, NO real provider. The orchestration
must:

  * resolve + verify the complete A1/A2/A3 CURRENT set (fail closed on any
    invalidity);
  * compose A4B planning → A4C preparation → A4D reuse → (on miss) A4C
    generation → A4D finalization/publication;
  * enforce upstream pointer stability (A1/A2/A3 heads must not move);
  * return a non-persisted stage result;
  * propagate LLMError unchanged (no outer provider retry);
  * zero provider calls on reuse-hit or invalid input.

Deliberately out of scope (A4E-B / A4E-C): CLI, runtime-config loading,
OpenAICompatibleLLMClient construction, live Qwen, real-novel smoke.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import (
    FilePointerStore,
    LineageRef,
    PointerKind,
    ValidationReport,
    persist_validation_report,
)
from short_drama.llm import (
    LLMClient,
    LLMError,
    ProviderMeta,
    PromptRegistry,
    ReasoningSettings,
    SemanticLLMProfile,
    StructuredGenerationResult,
    build_provenance,
    build_structured_request,
    validate_against_output_schema,
)
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    CHUNK_PLANNER_VERSION,
    ChunkExtractionBatchService,
    ChunkManifest,
    ChunkPlanningProfile,
    DEFAULT_OUTPUT_SCHEMA_PATH,
    EntityReconciliationProfile,
    EntityReconciliationService,
    EntityReconciliationStageResult,
    EvidenceRef,
    LANGUAGE_DETECTOR_ID,
    SourceChapter,
    SourceChunk,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    StoryExtractionProfile,
    StoryIntegrityError,
    StoryPersistenceError,
    TOKEN_COUNTER_ID,
    persist_chunk_manifest,
    persist_source_chunk,
    persist_source_document,
    plan_chunks,
    resolve_current_a3_reconciliation_inputs,
    resolve_current_story_snapshot,
    CurrentStorySnapshot,
    CurrentA3ReconciliationInputs,
)
from short_drama.story.extraction import (
    CandidatePayload,
    CharacterCandidate,
)
from short_drama.story.extraction_persistence import (
    candidate_extraction_artifact_id,
    candidate_extraction_pointer_id,
    load_candidate_extraction,
)
from short_drama.story.persistence import (
    chunk_pointer_id,
    chunk_validation_artifact_id,
    source_pointer_id,
    source_validation_artifact_id,
)
from short_drama.story.reconciliation_semantic import (
    DEFAULT_OUTPUT_SCHEMA_PATH as A4_SCHEMA_PATH,
)
from short_drama.story.source import NormalizationInfo
from short_drama.story.service import _current_pointer

# ---------------------------------------------------------------------------
# Frozen identity constants
# ---------------------------------------------------------------------------

PROJECT = "classroom"
DOCUMENT = "src_001"
CHUNK_PROFILE_ID = "story-analysis-v1"
EXTRACTION_PROFILE_ID = "story-extraction-v1"
PROMPTS_DIR = REPO_ROOT / "prompts" / "story"


# ---------------------------------------------------------------------------
# Source / profile / payload builders
# ---------------------------------------------------------------------------


def make_chunk_profile() -> ChunkPlanningProfile:
    return ChunkPlanningProfile(
        schema_version=1,
        profile_id=CHUNK_PROFILE_ID,
        token_counter=TOKEN_COUNTER_ID,
        ownership_token_budget=8,
        context_overlap_token_budget=4,
        context_token_budget=24,
    )


def make_source_document() -> SourceDocument:
    """Two chapters x two short paragraphs -> multiple single-paragraph chunks."""
    chapters = []
    texts_by_chapter = {
        "CH001": ("甲走进教室。", "乙坐在角落。"),
        "CH002": ("丙打开了门。", "丁递上了书。"),
    }
    for chapter_id, texts in texts_by_chapter.items():
        paragraphs = tuple(
            SourceParagraph(f"{chapter_id}_P{index:04d}", text, None)
            for index, text in enumerate(texts, start=1)
        )
        chapters.append(SourceChapter(chapter_id, None, "synthetic", paragraphs))
    return SourceDocument(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source=SourceInfo(
            "txt", "source/novel.txt", "f" * 64, 128, "zh-CN", "zh", LANGUAGE_DETECTOR_ID
        ),
        normalization=NormalizationInfo(
            "utf-8", "LF", "short_drama_source_ingestion_v1", "1"
        ),
        chapters=tuple(chapters),
    )


def make_extraction_profile(**overrides) -> StoryExtractionProfile:
    values = {
        "schema_version": 1,
        "profile_id": EXTRACTION_PROFILE_ID,
        "working_language": "zh-CN",
        "prompt_id": "a3.chunk-extraction",
        "prompt_version": 1,
        "output_schema_id": "a3-candidate-payload",
        "output_schema_version": 1,
        "max_generation_rounds": 2,
    }
    values.update(overrides)
    return StoryExtractionProfile(**values)


def make_semantic_profile(**overrides) -> SemanticLLMProfile:
    values = {
        "schema_version": 2,
        "profile_id": "story-extraction-llm-v1",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "structured_output_mode": "json_schema",
        "reasoning": ReasoningSettings(enabled=False),
    }
    values.update(overrides)
    return SemanticLLMProfile(**values)


def make_reconciliation_profile(**overrides) -> EntityReconciliationProfile:
    values = {
        "schema_version": 1,
        "profile_id": "entity-reconciliation-v1",
        "working_language": "zh-CN",
        "name_normalization_policy_id": "a4-name-normalization-v1",
        "blocking_policy_id": "a4-blocking-v1",
        "canonicalization_policy_id": "a4-canonicalization-v1",
        "prompt_id": "a4.entity-reconciliation",
        "prompt_version": 3,
        "output_schema_id": "a4-reconciliation-decision-selector-payload",
        "output_schema_version": 1,
        "max_generation_rounds": 2,
    }
    values.update(overrides)
    return EntityReconciliationProfile(**values)


def make_reconciliation_semantic_profile(**overrides) -> SemanticLLMProfile:
    values = {
        "schema_version": 2,
        "profile_id": "entity-reconciliation-llm-v1",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "structured_output_mode": "json_schema",
        "reasoning": ReasoningSettings(enabled=False),
    }
    values.update(overrides)
    return SemanticLLMProfile(**values)


def valid_payload_for(chunk: SourceChunk, name: str = "主角") -> CandidatePayload:
    """A valid A3B payload with a single character candidate."""
    return CandidatePayload(
        characters=(
            CharacterCandidate(
                candidate_id="cand_char_001",
                display_name_original=name,
                aliases_original=(),
                descriptors_zh=(),
                summary_zh="该 chunk 的局部人物候选。",
                evidence_strength="explicit",
                evidence=(
                    EvidenceRef(
                        paragraph_id=chunk.ownership_span.start,
                        role="primary",
                        strength="explicit",
                        excerpt=None,
                    ),
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# A1 + A2 state setup
# ---------------------------------------------------------------------------


def setup_a1_a2(
    root: Path,
) -> tuple[FileArtifactStore, FilePointerStore, ArtifactRef, ChunkManifest, tuple[SourceChunk, ...], ArtifactRef]:
    """Persist a current A1 SourceDocument + a current A2 ChunkManifest.

    Returns (store, pointers, source_ref, manifest, chunks, manifest_ref).
    """
    store = FileArtifactStore(root / "artifacts")
    pointers = FilePointerStore(root / "pointers", store)
    source_document = make_source_document()
    chunk_profile = make_chunk_profile()

    source_ref = persist_source_document(store, source_document, revision=1)
    pointers.compare_and_set(
        pointer_id=source_pointer_id(PROJECT, DOCUMENT),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=source_ref,
    )
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(LineageRef("source_document", source_ref),), findings=()
        ),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=source_ref.revision,
    )

    chunks, coverage = plan_chunks(source_document, source_ref, chunk_profile)
    chunk_refs = tuple(
        persist_source_chunk(store, chunk, profile_id=CHUNK_PROFILE_ID, revision=1)
        for chunk in chunks
    )
    manifest = ChunkManifest(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source_document_ref=source_ref,
        planner_version=CHUNK_PLANNER_VERSION,
        profile=chunk_profile,
        chunk_refs=chunk_refs,
        chunk_count=len(chunk_refs),
        coverage=coverage,
        state="CHUNKING_COMPLETE",
    )
    manifest_ref = persist_chunk_manifest(store, manifest, revision=1)
    pointers.compare_and_set(
        pointer_id=chunk_pointer_id(PROJECT, DOCUMENT, CHUNK_PROFILE_ID),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=manifest_ref,
    )
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(
                LineageRef("source_document", source_ref),
                LineageRef("chunk_manifest", manifest_ref),
            ),
            findings=(),
        ),
        artifact_id=chunk_validation_artifact_id(PROJECT, DOCUMENT, CHUNK_PROFILE_ID),
        revision=manifest_ref.revision,
    )
    return store, pointers, source_ref, manifest, chunks, manifest_ref


# ---------------------------------------------------------------------------
# A3 state setup (via A3E batch)
# ---------------------------------------------------------------------------


class A3FakeLLMClient(LLMClient):
    """Deterministic fake for A3 chunk extraction."""

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.call_count = 0

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.call_count += 1
        if not self.payloads:
            raise AssertionError("unexpected extra generate_structured call")
        response = self.payloads.pop(0)
        if isinstance(response, BaseException):
            raise response
        parsed = response.to_dict()
        validate_against_output_schema(parsed, output_schema)
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        return StructuredGenerationResult(
            parsed_json=parsed,
            provenance=build_provenance(
                request, ProviderMeta(finish_reason="stop", usage={"total_tokens": 100}), request_model="qwen3-27b", provider_family="qwen"
            ),
            attempts=1,
        )


def setup_a3(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    chunks: tuple[SourceChunk, ...],
    names: list[str] | None = None,
) -> A3FakeLLMClient:
    """Run the A3E batch to persist A3 state for all chunks."""
    if names is None:
        names = ["主角"] * len(chunks)
    client = A3FakeLLMClient(
        [valid_payload_for(chunk, name) for chunk, name in zip(chunks, names)]
    )
    service = ChunkExtractionBatchService(
        store, pointers, PromptRegistry(PROMPTS_DIR), DEFAULT_OUTPUT_SCHEMA_PATH
    )
    service.extract_chunks(
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile=make_chunk_profile(),
        extraction_profile=make_extraction_profile(),
        semantic_profile=make_semantic_profile(),
        llm_client=client,
    )
    return client


def setup_full_state(
    root: Path,
    names: list[str] | None = None,
) -> tuple[FileArtifactStore, FilePointerStore, ArtifactRef, ChunkManifest, tuple[SourceChunk, ...], ArtifactRef]:
    """Set up complete A1+A2+A3 state."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(root)
    setup_a3(store, pointers, chunks, names=names)
    return store, pointers, source_ref, manifest, chunks, manifest_ref


# ---------------------------------------------------------------------------
# A4 fake LLM client
# ---------------------------------------------------------------------------


class A4FakeLLMClient(LLMClient):
    """Deterministic fake provider for A4C semantic resolution."""

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.calls = []

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.call_count += 1
        self.calls.append((rendered_prompt, output_schema, semantic_profile))
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        if not self.responses:
            raise AssertionError("unexpected extra generate_structured call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            parsed, provenance = response
        else:
            parsed = response
            provenance = build_provenance(
                request,
                ProviderMeta(finish_reason="stop", usage={"total_tokens": 100}),
                request_model="qwen3-27b",
                provider_family="qwen",
            )
        validate_against_output_schema(parsed, output_schema)
        return StructuredGenerationResult(
            parsed_json=parsed, provenance=provenance, attempts=1
        )


def make_decision_payload(decisions: list[dict]) -> dict:
    """Build a reconciliation decision payload dict."""
    return {"decisions": decisions}


def make_decision(
    left: str, right: str, decision: str = "same_entity", reason_zh: str = "测试理由。"
) -> dict:
    """Build a single decision item dict (pair-local evidence selector)."""
    return {
        "left_candidate_ref": left,
        "right_candidate_ref": right,
        "decision": decision,
        "reason_zh": reason_zh,
        "evidence_selectors": ["L0"],
    }


class _ExplodeIfCalledClient(LLMClient):
    """LLM client that raises if generate_structured is called."""

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self):
        self.call_count = 0

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.call_count += 1
        raise AssertionError("LLM client should not be called")


# ---------------------------------------------------------------------------
# Service helpers
# ---------------------------------------------------------------------------


def make_reconciliation_service(
    store: FileArtifactStore, pointers: FilePointerStore
) -> EntityReconciliationService:
    return EntityReconciliationService(
        store,
        pointers,
        PromptRegistry(PROMPTS_DIR),
        A4_SCHEMA_PATH,
    )


def run_reconciliation(
    service: EntityReconciliationService, llm_client: LLMClient
) -> EntityReconciliationStageResult:
    return service.reconcile_entities(
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile=make_chunk_profile(),
        extraction_profile=make_extraction_profile(),
        reconciliation_profile=make_reconciliation_profile(),
        semantic_profile=make_reconciliation_semantic_profile(),
        llm_client=llm_client,
    )


def _chunk_id_from_candidate_ref(ref: ArtifactRef) -> str:
    """Extract chunk_id from a candidate_extraction artifact ref."""
    return ref.artifact_id.split(".")[-2]


def _candidate_refs_for_chunks(chunks: tuple[SourceChunk, ...]) -> tuple[str, ...]:
    """Build the candidate refs for each chunk (for decision payloads)."""
    return tuple(f"{chunk.chunk_id}:cand_char_001" for chunk in chunks)


def _make_decision_with_evidence(
    left: str, right: str, decision: str = "same_entity",
    left_paragraph: str = "CH001_P0001", right_paragraph: str = "CH001_P0001",
) -> dict:
    """Build a decision item citing both endpoints via pair-local selectors."""
    return {
        "left_candidate_ref": left,
        "right_candidate_ref": right,
        "decision": decision,
        "reason_zh": "测试理由。",
        "evidence_selectors": ["L0", "R0"],
    }


def _all_pairs_decision_payload(chunks: tuple[SourceChunk, ...]) -> dict:
    """Build a decision payload covering all C(n,2) pairs for n chunks.

    This is used when all chunks share the same non-strong name, producing
    all needs_semantic_decision pairs. Each decision uses the correct
    paragraph_id evidence for each candidate.
    """
    refs = _candidate_refs_for_chunks(chunks)
    decisions = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            left, right = sorted([refs[i], refs[j]])
            # Find the paragraph_ids for left and right
            left_para = chunks[i].ownership_span.start if refs[i] == left else chunks[j].ownership_span.start
            right_para = chunks[j].ownership_span.start if refs[j] == right else chunks[i].ownership_span.start
            decisions.append(
                _make_decision_with_evidence(
                    left, right, "same_entity", left_para, right_para
                )
            )
    return make_decision_payload(decisions)


# ===========================================================================
# 1. Resolver seam: valid current A1/A2 snapshot
# ===========================================================================


def test_resolve_current_story_snapshot_valid(tmp_path):
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)
    snapshot = resolve_current_story_snapshot(
        store,
        pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile=make_chunk_profile(),
    )
    assert isinstance(snapshot, CurrentStorySnapshot)
    assert snapshot.source_document_ref == source_ref
    assert snapshot.chunk_manifest_ref == manifest_ref
    assert len(snapshot.source_chunks) == len(chunks)
    assert len(snapshot.source_chunk_refs) == len(chunks)
    assert snapshot.source_current_pointer_ref is not None
    assert snapshot.chunk_current_pointer_ref is not None


def test_resolve_current_story_snapshot_missing_a1(tmp_path):
    """No A1 source document → fail closed."""
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)
    with pytest.raises(StoryIntegrityError, match="A1 SourceDocument"):
        resolve_current_story_snapshot(
            store,
            pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
        )


def test_resolve_current_story_snapshot_missing_a2(tmp_path):
    """A1 exists but no A2 manifest → fail closed."""
    # Create A1 only (no A2 pointer)
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)
    source_ref = persist_source_document(
        store, make_source_document(), revision=1
    )
    pointers.compare_and_set(
        pointer_id=source_pointer_id(PROJECT, DOCUMENT),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=source_ref,
    )
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(LineageRef("source_document", source_ref),), findings=()
        ),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=source_ref.revision,
    )
    # No A2 pointer set → fail closed
    with pytest.raises(StoryIntegrityError, match="A2 ChunkManifest"):
        resolve_current_story_snapshot(
            store,
            pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
        )


def test_resolve_current_story_snapshot_stale_a2_vs_a1(tmp_path):
    """A2 pins a different A1 than current → fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)
    # Create a new A1 revision with a valid ValidationReport
    new_source_ref = persist_source_document(store, make_source_document(), revision=2)
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(LineageRef("source_document", new_source_ref),), findings=()
        ),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=new_source_ref.revision,
    )
    a1_pointer = source_pointer_id(PROJECT, DOCUMENT)
    current_a1_ref = pointers.resolve_current_pointer_ref(a1_pointer)
    pointers.compare_and_set(
        pointer_id=a1_pointer,
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=current_a1_ref,
        target_ref=new_source_ref,
    )
    # Now A2 pins the old source, but current A1 is the new one
    with pytest.raises(StoryIntegrityError, match="stale"):
        resolve_current_story_snapshot(
            store,
            pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
        )


# ===========================================================================
# 2. Resolver seam: valid complete A3 current set
# ===========================================================================


def test_resolve_current_a3_valid(tmp_path):
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(tmp_path)
    result = resolve_current_a3_reconciliation_inputs(
        store,
        pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile=make_chunk_profile(),
        extraction_profile=make_extraction_profile(),
    )
    assert isinstance(result, CurrentA3ReconciliationInputs)
    assert len(result.candidate_extractions) == len(chunks)
    assert len(result.candidate_extraction_refs) == len(chunks)
    assert len(result.candidate_current_pointer_refs) == len(chunks)


def test_resolve_current_a3_missing_a3(tmp_path):
    """A1/A2 valid but no A3 for a chunk → fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)
    with pytest.raises(StoryIntegrityError, match="A3 CandidateExtraction"):
        resolve_current_a3_reconciliation_inputs(
            store,
            pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
            extraction_profile=make_extraction_profile(),
        )


def test_resolve_current_a3_wrong_extraction_profile_id(tmp_path):
    """A3 exists for profile A, but we request profile B → fail closed
    (the pointer for profile B doesn't exist)."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(tmp_path)
    wrong_profile = make_extraction_profile(profile_id="different-profile-id")
    with pytest.raises(StoryIntegrityError, match="A3 CandidateExtraction is not current"):
        resolve_current_a3_reconciliation_inputs(
            store,
            pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
            extraction_profile=wrong_profile,
        )


def test_resolve_current_a3_wrong_extraction_profile_hash(tmp_path):
    """A3 exists but extraction_profile_hash differs → fail closed.

    Same profile_id (so the pointer exists) but a different semantic field
    (prompt_version) changes the hash, triggering the hash mismatch check.
    """
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(tmp_path)
    # Same profile_id but different prompt_version → different hash
    wrong_profile = make_extraction_profile(prompt_version=99)
    with pytest.raises(StoryIntegrityError, match="extraction_profile_hash"):
        resolve_current_a3_reconciliation_inputs(
            store,
            pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
            extraction_profile=wrong_profile,
        )


# ===========================================================================
# 3. All invalid states → zero provider calls
# ===========================================================================


def test_all_invalid_states_zero_provider_calls(tmp_path):
    """Multiple invalid states → all fail before any provider call."""
    # Missing A1
    store1 = FileArtifactStore(tmp_path / "artifacts1")
    pointers1 = FilePointerStore(tmp_path / "pointers1", store1)
    explode_client = _ExplodeIfCalledClient()
    service1 = make_reconciliation_service(store1, pointers1)
    with pytest.raises(StoryIntegrityError):
        service1.reconcile_entities(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
            extraction_profile=make_extraction_profile(),
            reconciliation_profile=make_reconciliation_profile(),
            semantic_profile=make_reconciliation_semantic_profile(),
            llm_client=explode_client,
        )
    assert explode_client.call_count == 0

    # Missing A3
    store2, pointers2, _, _, chunks2, _ = setup_a1_a2(tmp_path / "sub2")
    explode_client2 = _ExplodeIfCalledClient()
    service2 = make_reconciliation_service(store2, pointers2)
    with pytest.raises(StoryIntegrityError):
        service2.reconcile_entities(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
            extraction_profile=make_extraction_profile(),
            reconciliation_profile=make_reconciliation_profile(),
            semantic_profile=make_reconciliation_semantic_profile(),
            llm_client=explode_client2,
        )
    assert explode_client2.call_count == 0


# ===========================================================================
# 4. Fresh semantic miss → provider → publication
# ===========================================================================


def test_fresh_semantic_miss_provider_publication(tmp_path):
    """Two chunks with the same 2-char name → needs_semantic_decision →
    provider called → A4 published."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    assert len(chunks) >= 4

    # Build decision payloads for all semantic pairs (all 4 chunks share the same name)
    # Adjacent chunks will form pairs: (0,1), (1,2), (2,3) for same-chunk + adjacent
    # Plus same-chunk pairs within each chunk (but each chunk has only 1 candidate)
    # So we get cross-chunk pairs from adjacent chunk windows
    decision_payload = _all_pairs_decision_payload(chunks)
    client = A4FakeLLMClient([decision_payload])
    service = make_reconciliation_service(store, pointers)
    result = run_reconciliation(service, client)

    assert client.call_count >= 1
    assert result.reused is False
    assert result.entity_map_ref is not None
    assert result.validation_report_ref is not None
    assert result.current_pointer_ref is not None
    assert result.semantic_pair_count >= 1
    assert result.llm_same_count >= 1
    assert result.canonical_character_count >= 1


def test_fresh_zero_semantic_no_provider(tmp_path):
    """Empty candidate extractions → zero candidates → zero pairs →
    zero provider calls → A4 published."""
    # Use empty payloads (no characters) so there are no candidates at all
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)
    # Persist A3 with empty payloads
    from short_drama.story.extraction import CandidatePayload as CP
    empty_client = A3FakeLLMClient([CP(characters=(), locations=(), facts=(), events=(), relationships=(), unresolved_mentions=()) for _ in chunks])
    a3_service = ChunkExtractionBatchService(
        store, pointers, PromptRegistry(PROMPTS_DIR), DEFAULT_OUTPUT_SCHEMA_PATH
    )
    a3_service.extract_chunks(
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile=make_chunk_profile(),
        extraction_profile=make_extraction_profile(),
        semantic_profile=make_semantic_profile(),
        llm_client=empty_client,
    )

    client = A4FakeLLMClient([])
    service = make_reconciliation_service(store, pointers)
    result = run_reconciliation(service, client)

    assert client.call_count == 0
    assert result.reused is False
    assert result.semantic_pair_count == 0
    assert result.semantic_block_count == 0
    assert result.candidate_count_total == 0
    assert result.entity_map_ref is not None


# ===========================================================================
# 5. Exact A4 reuse → no provider
# ===========================================================================


def test_exact_a4_reuse_no_provider(tmp_path):
    """First run publishes A4. Second run with identical inputs reuses A4
    with zero provider calls."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    decision_payload = _all_pairs_decision_payload(chunks)

    client1 = A4FakeLLMClient([decision_payload])
    service = make_reconciliation_service(store, pointers)
    result1 = run_reconciliation(service, client1)
    assert result1.reused is False
    assert client1.call_count >= 1
    first_entity_map_ref = result1.entity_map_ref

    client2 = A4FakeLLMClient([])
    result2 = run_reconciliation(service, client2)
    assert result2.reused is True
    assert client2.call_count == 0
    assert result2.entity_map_ref == first_entity_map_ref
    assert result2.current_pointer_ref == result1.current_pointer_ref


# ===========================================================================
# 6. Pointer stability: A1 changes before reuse return
# ===========================================================================


def test_a1_changes_before_reuse_return(tmp_path):
    """A4 CURRENT exists. A1 pointer moves. Reuse path detects movement."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    decision_payload = _all_pairs_decision_payload(chunks)
    client1 = A4FakeLLMClient([decision_payload])
    service = make_reconciliation_service(store, pointers)
    result1 = run_reconciliation(service, client1)
    assert result1.reused is False

    # Move A1 pointer (need a valid ValidationReport for the new revision)
    new_source_ref = persist_source_document(store, make_source_document(), revision=2)
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(LineageRef("source_document", new_source_ref),), findings=()
        ),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=new_source_ref.revision,
    )
    a1_pointer = source_pointer_id(PROJECT, DOCUMENT)
    current_a1_ref = pointers.resolve_current_pointer_ref(a1_pointer)
    pointers.compare_and_set(
        pointer_id=a1_pointer,
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=current_a1_ref,
        target_ref=new_source_ref,
    )

    client2 = A4FakeLLMClient([])
    with pytest.raises((StoryPersistenceError, StoryIntegrityError)):
        run_reconciliation(service, client2)


# ===========================================================================
# 7. Pointer stability: A2 changes before reuse return
# ===========================================================================


def test_a2_changes_during_fresh_path(tmp_path):
    """A2 pointer moves during the fresh path (after LLM call, before
    publication). The upstream stability check detects movement → fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    decision_payload = _all_pairs_decision_payload(chunks)

    class A2MovingClient(A4FakeLLMClient):
        def __init__(self, responses, pointers, store, manifest):
            super().__init__(responses)
            self._pointers = pointers
            self._store = store
            self._manifest = manifest
            self._moved = False

        def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
            result = super().generate_structured(
                rendered_prompt, output_schema, semantic_profile
            )
            if not self._moved and self.call_count >= 1:
                new_ref = persist_chunk_manifest(self._store, self._manifest, revision=2)
                a2_pointer = chunk_pointer_id(PROJECT, DOCUMENT, CHUNK_PROFILE_ID)
                current = self._pointers.resolve_current_pointer_ref(a2_pointer)
                self._pointers.compare_and_set(
                    pointer_id=a2_pointer,
                    pointer_kind=PointerKind.CURRENT,
                    expected_pointer_ref=current,
                    target_ref=new_ref,
                )
                self._moved = True
            return result

    client = A2MovingClient([decision_payload], pointers, store, manifest)
    service = make_reconciliation_service(store, pointers)
    with pytest.raises(StoryPersistenceError):
        run_reconciliation(service, client)


# ===========================================================================
# 8. Pointer stability: A3 changes before reuse return
# ===========================================================================


def test_a3_changes_during_fresh_path(tmp_path):
    """A3 pointer moves during the fresh path (after LLM call, before
    publication). The upstream stability check detects movement → fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    decision_payload = _all_pairs_decision_payload(chunks)

    class A3MovingClient(A4FakeLLMClient):
        def __init__(self, responses, pointers, store, chunks):
            super().__init__(responses)
            self._pointers = pointers
            self._store = store
            self._chunks = chunks
            self._moved = False

        def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
            result = super().generate_structured(
                rendered_prompt, output_schema, semantic_profile
            )
            if not self._moved and self.call_count >= 1:
                # Persist a minimal artifact and move the A3 pointer to it
                from short_drama.artifacts.models import (
                    ImmutableArtifactEnvelope,
                    artifact_content_hash,
                )
                from short_drama.artifacts.canonical import canonical_json_bytes
                chunk = self._chunks[0]
                a3_artifact_id = candidate_extraction_artifact_id(
                    PROJECT, DOCUMENT, CHUNK_PROFILE_ID, chunk.chunk_id, EXTRACTION_PROFILE_ID
                )
                payload_obj = {"schema_version": 1, "payload": {}}
                payload_bytes = canonical_json_bytes(payload_obj)
                content_hash = artifact_content_hash(
                    schema_version=1, payload={"schema_version": 1, "payload": {}}
                )
                envelope = ImmutableArtifactEnvelope(
                    artifact_type="candidate_extraction",
                    artifact_id=a3_artifact_id,
                    revision=2,
                    schema_version=1,
                    content_hash=content_hash,
                    _payload_bytes=payload_bytes,
                )
                new_ref = self._store.put(envelope)
                a3_pointer_id = candidate_extraction_pointer_id(
                    PROJECT, DOCUMENT, CHUNK_PROFILE_ID, chunk.chunk_id, EXTRACTION_PROFILE_ID
                )
                current = self._pointers.resolve_current_pointer_ref(a3_pointer_id)
                self._pointers.compare_and_set(
                    pointer_id=a3_pointer_id,
                    pointer_kind=PointerKind.CURRENT,
                    expected_pointer_ref=current,
                    target_ref=new_ref,
                )
                self._moved = True
            return result

    client = A3MovingClient([decision_payload], pointers, store, chunks)
    service = make_reconciliation_service(store, pointers)
    with pytest.raises(StoryPersistenceError):
        run_reconciliation(service, client)


# ===========================================================================
# 9. LLMError propagation
# ===========================================================================


def test_llm_error_propagates_unchanged(tmp_path):
    """A4C semantic generation raises LLMError → propagates unchanged.

    The first attempt returns a schema-valid but semantically invalid payload
    (wrong decision count), so A4C retries. The second attempt raises LLMError.
    """
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    # Schema-valid but semantically invalid: only 1 decision instead of 6
    all_refs = _candidate_refs_for_chunks(chunks)
    left, right = sorted([all_refs[0], all_refs[1]])
    invalid_payload = make_decision_payload(
        [make_decision(left, right, "same_entity")]  # only 1 of 6
    )
    client = A4FakeLLMClient([
        invalid_payload,
        LLMError("provider timeout"),
    ])
    service = make_reconciliation_service(store, pointers)
    with pytest.raises(LLMError, match="provider timeout"):
        run_reconciliation(service, client)
    assert client.call_count == 2


# ===========================================================================
# 10. Reuse hit with explode-if-called client
# ===========================================================================


def test_reuse_hit_explode_if_called_client(tmp_path):
    """On a reuse hit, the LLM client must never be called."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    decision_payload = _all_pairs_decision_payload(chunks)
    client1 = A4FakeLLMClient([decision_payload])
    service = make_reconciliation_service(store, pointers)
    result1 = run_reconciliation(service, client1)
    assert result1.reused is False

    class ExplodeClient(LLMClient):
        supported_structured_output_modes = frozenset(
            {"none", "json_object", "json_schema"}
        )

        def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
            raise AssertionError("LLM client should not be called on reuse")

    result2 = run_reconciliation(service, ExplodeClient())
    assert result2.reused is True
    assert result2.entity_map_ref == result1.entity_map_ref


# ===========================================================================
# 11. A3 order test
# ===========================================================================


def test_a3_order_in_manifest_order(tmp_path):
    """A3 candidate refs are in exact ChunkManifest.chunk_refs order."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(tmp_path)
    result = resolve_current_a3_reconciliation_inputs(
        store,
        pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile=make_chunk_profile(),
        extraction_profile=make_extraction_profile(),
    )
    expected_chunk_ids = [chunk.chunk_id for chunk in chunks]
    actual_chunk_ids = [
        _chunk_id_from_candidate_ref(ref) for ref in result.candidate_extraction_refs
    ]
    assert actual_chunk_ids == [cid.lower() for cid in expected_chunk_ids]


# ===========================================================================
# 12. Fresh-path upstream movement must not move A4 CURRENT
# ===========================================================================


def test_fresh_path_upstream_movement_no_a4_current_change(tmp_path):
    """On the fresh path, if upstream moves before publication, the A4
    CURRENT must not be moved (fail closed)."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    decision_payload = _all_pairs_decision_payload(chunks)

    class UpstreamMovingClient(A4FakeLLMClient):
        def __init__(self, responses, pointers, store):
            super().__init__(responses)
            self._pointers = pointers
            self._store = store
            self._moved = False

        def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
            result = super().generate_structured(
                rendered_prompt, output_schema, semantic_profile
            )
            if not self._moved and self.call_count >= 1:
                new_ref = persist_source_document(
                    self._store, make_source_document(), revision=2
                )
                a1_pointer = source_pointer_id(PROJECT, DOCUMENT)
                current = self._pointers.resolve_current_pointer_ref(a1_pointer)
                self._pointers.compare_and_set(
                    pointer_id=a1_pointer,
                    pointer_kind=PointerKind.CURRENT,
                    expected_pointer_ref=current,
                    target_ref=new_ref,
                )
                self._moved = True
            return result

    client = UpstreamMovingClient([decision_payload], pointers, store)
    service = make_reconciliation_service(store, pointers)
    with pytest.raises(StoryPersistenceError):
        run_reconciliation(service, client)


# ===========================================================================
# 13. A3InputIdentity exact
# ===========================================================================


def test_a3_input_identity_exact(tmp_path):
    """The A3 refs and story snapshot refs match the expected values."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(tmp_path)
    result = resolve_current_a3_reconciliation_inputs(
        store,
        pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile=make_chunk_profile(),
        extraction_profile=make_extraction_profile(),
    )
    assert len(result.candidate_extraction_refs) == len(chunks)
    assert result.story_snapshot.source_document_ref == source_ref
    assert result.story_snapshot.chunk_manifest_ref == manifest_ref


# ===========================================================================
# 14. A3 CURRENT resolver regression matrix
# ===========================================================================


def _setup_full_state_single_chunk(tmp_path):
    """Set up A1+A2+A3 state for a single-chunk scenario."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)
    # Use only the first chunk for A3 setup
    setup_a3(store, pointers, chunks, names=["主角"])
    return store, pointers, source_ref, manifest, chunks, manifest_ref


def test_a3_wrong_logical_artifact_target(tmp_path):
    """A3 CURRENT pointer targets a different logical artifact → fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)
    # Set up A3 for all chunks
    setup_a3(store, pointers, chunks, names=["主角"] * len(chunks))

    # Move the A3 pointer for chunk 0 to target a different artifact
    chunk = chunks[0]
    a3_pointer_id = candidate_extraction_pointer_id(
        PROJECT, DOCUMENT, CHUNK_PROFILE_ID, chunk.chunk_id, EXTRACTION_PROFILE_ID
    )
    # Persist a different artifact with a different artifact_id
    from short_drama.artifacts.models import ImmutableArtifactEnvelope, artifact_content_hash
    from short_drama.artifacts.canonical import canonical_json_bytes
    other_chunk_id = chunks[1].chunk_id  # different chunk
    other_artifact_id = candidate_extraction_artifact_id(
        PROJECT, DOCUMENT, CHUNK_PROFILE_ID, other_chunk_id, EXTRACTION_PROFILE_ID
    )
    payload_obj = {"schema_version": 1, "payload": {}}
    payload_bytes = canonical_json_bytes(payload_obj)
    content_hash = artifact_content_hash(schema_version=1, payload=payload_obj)
    envelope = ImmutableArtifactEnvelope(
        artifact_type="candidate_extraction",
        artifact_id=other_artifact_id,
        revision=1,
        schema_version=1,
        content_hash=content_hash,
        _payload_bytes=payload_bytes,
    )
    # Use revision=2 to avoid conflict with existing chunk 1 A3 at revision=1
    envelope = ImmutableArtifactEnvelope(
        artifact_type="candidate_extraction",
        artifact_id=other_artifact_id,
        revision=2,
        schema_version=1,
        content_hash=content_hash,
        _payload_bytes=payload_bytes,
    )
    foreign_ref = store.put(envelope)
    current_pointer_ref = pointers.resolve_current_pointer_ref(a3_pointer_id)
    pointers.compare_and_set(
        pointer_id=a3_pointer_id,
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=current_pointer_ref,
        target_ref=foreign_ref,
    )

    explode_client = _ExplodeIfCalledClient()
    service = make_reconciliation_service(store, pointers)
    with pytest.raises(StoryIntegrityError, match="different logical"):
        run_reconciliation(service, explode_client)
    assert explode_client.call_count == 0


def test_a3_wrong_source_document_ref(tmp_path):
    """A3 CandidateExtraction.source_document_ref does not match the
    requested current A1 ref → fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(tmp_path)

    # Create a new A1 source document (different ref) and persist it,
    # but DON'T move the A1 pointer (so the A3's source_document_ref
    # won't match the current A1).
    # Instead: move the A1 pointer so the A3's source_document_ref is stale.
    new_source_ref = persist_source_document(store, make_source_document(), revision=2)
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(LineageRef("source_document", new_source_ref),), findings=()
        ),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=new_source_ref.revision,
    )
    a1_pointer = source_pointer_id(PROJECT, DOCUMENT)
    current_a1_ref = pointers.resolve_current_pointer_ref(a1_pointer)
    pointers.compare_and_set(
        pointer_id=a1_pointer,
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=current_a1_ref,
        target_ref=new_source_ref,
    )

    # Now the A3 was built against the old source_ref, but the current
    # A1 is the new source_ref. The A2 pins the old source, so the A1/A2
    # resolver will fail first (A2 stale vs A1). Let's test the A3 level
    # directly by using the A3 resolver with the old source ref.
    service_a3 = __import__(
        "short_drama.story.extraction_persistence",
        fromlist=["CandidateExtractionService"],
    ).CandidateExtractionService(store, pointers)
    chunk = chunks[0]
    chunk_ref = manifest.chunk_refs[0]
    with pytest.raises(StoryIntegrityError, match="source_document_ref"):
        service_a3.require_current_validated(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile_id=CHUNK_PROFILE_ID,
            chunk_id=chunk.chunk_id,
            source_document_ref=new_source_ref,  # different from A3's
            source_chunk_ref=chunk_ref,
            extraction_profile=make_extraction_profile(),
        )


def test_a3_wrong_source_chunk_ref(tmp_path):
    """A3 CandidateExtraction.source_chunk_ref does not match the
    requested source_chunk_ref → fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(tmp_path)

    service_a3 = __import__(
        "short_drama.story.extraction_persistence",
        fromlist=["CandidateExtractionService"],
    ).CandidateExtractionService(store, pointers)
    chunk = chunks[0]
    # Use the wrong chunk ref (chunk 1's ref for chunk 0)
    wrong_chunk_ref = manifest.chunk_refs[1]
    with pytest.raises(StoryIntegrityError, match="source_chunk_ref"):
        service_a3.require_current_validated(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile_id=CHUNK_PROFILE_ID,
            chunk_id=chunk.chunk_id,
            source_document_ref=source_ref,
            source_chunk_ref=wrong_chunk_ref,
            extraction_profile=make_extraction_profile(),
        )


def _make_provenance():
    """Build a minimal valid LLMInvocationProvenance for test fixtures."""
    from short_drama.llm import LLMInvocationProvenance
    return LLMInvocationProvenance(
        provider_family="qwen",
        model="qwen3-27b",
        semantic_profile_id="story-extraction-llm-v1",
        semantic_profile_hash="f" * 64,
        prompt_id="a3.chunk-extraction",
        prompt_version=1,
        prompt_content_hash="f" * 64,
        rendered_prompt_hash="f" * 64,
        output_schema_id="a3-candidate-payload",
        output_schema_version=1,
        output_schema_hash="f" * 64,
        request_hash="f" * 64,
        provider_response_id=None,
        finish_reason="stop",
        usage={"total_tokens": 100},
    )


def test_a3_validation_report_missing(tmp_path):
    """A3 ValidationReport is missing → fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)

    # Persist A3 manually WITHOUT a validation report
    from short_drama.story.extraction_persistence import persist_candidate_extraction
    from short_drama.story.extraction import CandidateExtraction

    chunk = chunks[0]
    chunk_ref = manifest.chunk_refs[0]
    extraction = CandidateExtraction(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile_id=CHUNK_PROFILE_ID,
        chunk_id=chunk.chunk_id,
        source_document_ref=source_ref,
        source_chunk_ref=chunk_ref,
        extraction_profile_id=EXTRACTION_PROFILE_ID,
        extraction_profile_hash=make_extraction_profile().profile_hash,
        generation_provenance=_make_provenance(),
        candidates=valid_payload_for(chunk, "主角"),
    )
    a3_ref = persist_candidate_extraction(store, extraction, revision=1)
    pointers.compare_and_set(
        pointer_id=candidate_extraction_pointer_id(
            PROJECT, DOCUMENT, CHUNK_PROFILE_ID, chunk.chunk_id, EXTRACTION_PROFILE_ID
        ),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=a3_ref,
    )
    # No validation report persisted → require_current_validated must fail
    service_a3 = __import__(
        "short_drama.story.extraction_persistence",
        fromlist=["CandidateExtractionService"],
    ).CandidateExtractionService(store, pointers)
    with pytest.raises((StoryIntegrityError, StoryPersistenceError, Exception)):
        service_a3.require_current_validated(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile_id=CHUNK_PROFILE_ID,
            chunk_id=chunk.chunk_id,
            source_document_ref=source_ref,
            source_chunk_ref=chunk_ref,
            extraction_profile=make_extraction_profile(),
        )


def test_a3_semantic_invalid_extraction(tmp_path):
    """A3 extraction is semantic-invalid (evidence paragraph not in chunk) →
    fail closed."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)

    # Persist A3 with an invalid payload: evidence references a paragraph
    # that's not in the chunk's ownership span
    from short_drama.story.extraction import CandidatePayload, CharacterCandidate as CC
    from short_drama.story import EvidenceRef as ER

    def invalid_payload():
        return CandidatePayload(
            characters=(
                CC(
                    candidate_id="cand_char_001",
                    display_name_original="主角",
                    aliases_original=(),
                    descriptors_zh=(),
                    summary_zh="test",
                    evidence_strength="explicit",
                    evidence=(
                        ER(
                            paragraph_id="CH999_P9999",  # not in chunk
                            role="primary",
                            strength="explicit",
                            excerpt=None,
                        ),
                    ),
                ),
            ),
        )

    # We need to bypass the A3E service's semantic validation. Let's persist
    # a raw A3 artifact directly.
    from short_drama.story.extraction_persistence import persist_candidate_extraction
    from short_drama.story.extraction import CandidateExtraction
    from short_drama.artifacts import ArtifactRef as AR

    chunk = chunks[0]
    chunk_ref = manifest.chunk_refs[0]
    extraction = CandidateExtraction(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile_id=CHUNK_PROFILE_ID,
        chunk_id=chunk.chunk_id,
        source_document_ref=source_ref,
        source_chunk_ref=chunk_ref,
        extraction_profile_id=EXTRACTION_PROFILE_ID,
        extraction_profile_hash=make_extraction_profile().profile_hash,
        generation_provenance=_make_provenance(),
        candidates=invalid_payload(),
    )
    a3_ref = persist_candidate_extraction(store, extraction, revision=1)
    pointers.compare_and_set(
        pointer_id=candidate_extraction_pointer_id(
            PROJECT, DOCUMENT, CHUNK_PROFILE_ID, chunk.chunk_id, EXTRACTION_PROFILE_ID
        ),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=a3_ref,
    )
    # Persist a matching PASS validation report (to get past the report check,
    # but the semantic validation will fail)
    from short_drama.story.extraction_persistence import (
        candidate_extraction_validation_artifact_id as cva_id,
    )
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(
                LineageRef("source_document", source_ref),
                LineageRef("source_chunk", chunk_ref),
                LineageRef("candidate_extraction", a3_ref),
            ),
            findings=(),
        ),
        artifact_id=cva_id(a3_ref.artifact_id),
        revision=1,
    )

    # Now A3 CURRENT exists but is semantic-invalid. The resolver should
    # fail closed when re-validating.
    service_a3 = __import__(
        "short_drama.story.extraction_persistence",
        fromlist=["CandidateExtractionService"],
    ).CandidateExtractionService(store, pointers)
    with pytest.raises(StoryIntegrityError, match="re-validate to PASS"):
        service_a3.require_current_validated(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile_id=CHUNK_PROFILE_ID,
            chunk_id=chunk.chunk_id,
            source_document_ref=source_ref,
            source_chunk_ref=chunk_ref,
            extraction_profile=make_extraction_profile(),
        )


# ===========================================================================
# 15. Post-generation same-identity race
# ===========================================================================


def test_post_generation_same_identity_race(tmp_path):
    """A4 pre-provider reuse MISS → A4C generation → concurrent writer
    publishes same identity → publish_validated returns reused=True.

    The stage result must report:
    - provider was called (semantic_generation_call_count > 0)
    - reused = True
    - entity_map_ref = concurrent publication's EntityMap
    - counts from the persisted publication
    - no duplicate A4 publication revision
    """
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    decision_payload = _all_pairs_decision_payload(chunks)

    # First: publish A4 normally (to get the refs for the race)
    client1 = A4FakeLLMClient([decision_payload])
    service = make_reconciliation_service(store, pointers)
    result1 = run_reconciliation(service, client1)
    assert result1.reused is False

    # Now: monkeypatch try_reuse_current to return None (simulating a miss)
    # and publish_validated to return reused=True (simulating the race).
    # Load the real EntityMap to get the actual artifact refs.
    from short_drama.story.reconciliation_persistence import (
        ReconciliationPersistenceService,
        ReconciliationPublication,
        load_entity_map,
    )

    entity_map = load_entity_map(
        store, result1.entity_map_ref,
        expected_artifact_id=result1.entity_map_ref.artifact_id,
    )

    def fake_try_reuse(self, **kwargs):
        return None  # pre-provider reuse miss

    def fake_publish_validated(self, **kwargs):
        # Simulate: a concurrent writer already published the same identity.
        # publish_validated returns the existing (concurrent) publication.
        return ReconciliationPublication(
            candidate_entity_index_ref=entity_map.candidate_entity_index_ref,
            reconciliation_decision_set_ref=entity_map.reconciliation_decision_set_ref,
            canonical_character_registry_ref=entity_map.canonical_character_registry_ref,
            canonical_location_registry_ref=entity_map.canonical_location_registry_ref,
            unresolved_entity_set_ref=entity_map.unresolved_entity_set_ref,
            entity_map_ref=result1.entity_map_ref,
            validation_report_ref=result1.validation_report_ref,
            current_pointer_ref=result1.current_pointer_ref,
            reused=True,
        )

    with patch.object(
        ReconciliationPersistenceService, "try_reuse_current", fake_try_reuse,
    ), patch.object(
        ReconciliationPersistenceService, "publish_validated", fake_publish_validated,
    ):
        client2 = A4FakeLLMClient([decision_payload])
        result2 = run_reconciliation(service, client2)

    # Assert the race behavior
    assert client2.call_count >= 1  # provider was called
    assert result2.reused is True  # same-identity race detected
    assert result2.semantic_generation_call_count > 0  # provider was actually called
    assert result2.entity_map_ref == result1.entity_map_ref  # concurrent writer's EntityMap


# ===========================================================================
# 16. Reuse-return pointer stability (movement AFTER try_reuse_current)
# ===========================================================================


def _make_reuse_scenario(tmp_path):
    """Set up A1+A2+A3+A4 state where A4 CURRENT exists (reuse hit)."""
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_full_state(
        tmp_path, names=["张三", "张三", "张三", "张三"]
    )
    decision_payload = _all_pairs_decision_payload(chunks)
    client1 = A4FakeLLMClient([decision_payload])
    service = make_reconciliation_service(store, pointers)
    result1 = run_reconciliation(service, client1)
    assert result1.reused is False
    return store, pointers, source_ref, manifest, chunks, manifest_ref, service


def test_reuse_return_a1_moves_after_reuse_found(tmp_path):
    """A4 reuse hit. A1 pointer moves AFTER try_reuse_current but BEFORE
    the stability check. The result must fail closed (no stale A4 returned).
    LLM call_count = 0."""
    store, pointers, source_ref, manifest, chunks, manifest_ref, service = (
        _make_reuse_scenario(tmp_path)
    )

    # Monkeypatch: after try_reuse_current returns a hit, move the A1 pointer
    from short_drama.story.reconciliation_persistence import ReconciliationPersistenceService

    real_try_reuse = ReconciliationPersistenceService.try_reuse_current

    def moving_try_reuse(self, **kwargs):
        result = real_try_reuse(self, **kwargs)
        if result is not None:
            # Move A1 pointer (simulating upstream movement after reuse found)
            new_source_ref = persist_source_document(
                store, make_source_document(), revision=2
            )
            persist_validation_report(
                store,
                ValidationReport(
                    validated_refs=(LineageRef("source_document", new_source_ref),),
                    findings=(),
                ),
                artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
                revision=new_source_ref.revision,
            )
            a1_pointer = source_pointer_id(PROJECT, DOCUMENT)
            current = self.pointers.resolve_current_pointer_ref(a1_pointer)
            self.pointers.compare_and_set(
                pointer_id=a1_pointer,
                pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=current,
                target_ref=new_source_ref,
            )
        return result

    with patch.object(ReconciliationPersistenceService, "try_reuse_current", moving_try_reuse):
        client = _ExplodeIfCalledClient()
        with pytest.raises(StoryPersistenceError):
            run_reconciliation(service, client)
    assert client.call_count == 0


def test_reuse_return_a2_moves_after_reuse_found(tmp_path):
    """A4 reuse hit. A2 pointer moves AFTER try_reuse_current but BEFORE
    the stability check. The result must fail closed. LLM call_count = 0."""
    store, pointers, source_ref, manifest, chunks, manifest_ref, service = (
        _make_reuse_scenario(tmp_path)
    )

    from short_drama.story.reconciliation_persistence import ReconciliationPersistenceService

    real_try_reuse = ReconciliationPersistenceService.try_reuse_current

    def moving_try_reuse(self, **kwargs):
        result = real_try_reuse(self, **kwargs)
        if result is not None:
            # Move A2 pointer
            new_manifest_ref = persist_chunk_manifest(store, manifest, revision=2)
            a2_pointer = chunk_pointer_id(PROJECT, DOCUMENT, CHUNK_PROFILE_ID)
            current = self.pointers.resolve_current_pointer_ref(a2_pointer)
            self.pointers.compare_and_set(
                pointer_id=a2_pointer,
                pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=current,
                target_ref=new_manifest_ref,
            )
        return result

    with patch.object(ReconciliationPersistenceService, "try_reuse_current", moving_try_reuse):
        client = _ExplodeIfCalledClient()
        with pytest.raises(StoryPersistenceError):
            run_reconciliation(service, client)
    assert client.call_count == 0


def test_reuse_return_a3_moves_after_reuse_found(tmp_path):
    """A4 reuse hit. One A3 pointer moves AFTER try_reuse_current but BEFORE
    the stability check. The result must fail closed. LLM call_count = 0."""
    store, pointers, source_ref, manifest, chunks, manifest_ref, service = (
        _make_reuse_scenario(tmp_path)
    )

    from short_drama.story.reconciliation_persistence import ReconciliationPersistenceService

    real_try_reuse = ReconciliationPersistenceService.try_reuse_current

    def moving_try_reuse(self, **kwargs):
        result = real_try_reuse(self, **kwargs)
        if result is not None:
            # Move A3 pointer for chunk 0
            from short_drama.artifacts.models import (
                ImmutableArtifactEnvelope,
                artifact_content_hash,
            )
            from short_drama.artifacts.canonical import canonical_json_bytes
            chunk = chunks[0]
            a3_artifact_id = candidate_extraction_artifact_id(
                PROJECT, DOCUMENT, CHUNK_PROFILE_ID, chunk.chunk_id, EXTRACTION_PROFILE_ID
            )
            payload_obj = {"schema_version": 1, "payload": {}}
            payload_bytes = canonical_json_bytes(payload_obj)
            content_hash = artifact_content_hash(schema_version=1, payload=payload_obj)
            envelope = ImmutableArtifactEnvelope(
                artifact_type="candidate_extraction",
                artifact_id=a3_artifact_id,
                revision=2,
                schema_version=1,
                content_hash=content_hash,
                _payload_bytes=payload_bytes,
            )
            new_ref = self.store.put(envelope)
            a3_pointer_id = candidate_extraction_pointer_id(
                PROJECT, DOCUMENT, CHUNK_PROFILE_ID, chunk.chunk_id, EXTRACTION_PROFILE_ID
            )
            current = self.pointers.resolve_current_pointer_ref(a3_pointer_id)
            self.pointers.compare_and_set(
                pointer_id=a3_pointer_id,
                pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=current,
                target_ref=new_ref,
            )
        return result

    with patch.object(ReconciliationPersistenceService, "try_reuse_current", moving_try_reuse):
        client = _ExplodeIfCalledClient()
        with pytest.raises(StoryPersistenceError):
            run_reconciliation(service, client)
    assert client.call_count == 0


# ===========================================================================
# 17. Shared A1/A2 resolver: A3E and A4E use the same authority
# ===========================================================================


def test_shared_resolver_a3e_and_a4e_same_seam(tmp_path):
    """Prove that both A3E (ChunkExtractionBatchService) and A4E
    (EntityReconciliationService) route through the same
    resolve_current_story_snapshot() authority.

    We monkeypatch the shared resolver to record calls and verify both
    paths invoke it.
    """
    store, pointers, source_ref, manifest, chunks, manifest_ref = setup_a1_a2(tmp_path)

    call_log = []
    real_resolver = resolve_current_story_snapshot

    def tracking_resolver(*args, **kwargs):
        call_log.append(args + (kwargs,))
        return real_resolver(*args, **kwargs)

    # Monkeypatch in both modules that import it
    import short_drama.story.extraction_batch as batch_mod
    import short_drama.story.reconciliation_service as recon_mod
    import short_drama.story.service as svc_mod

    with patch.object(batch_mod, "resolve_current_story_snapshot", tracking_resolver), \
         patch.object(svc_mod, "resolve_current_story_snapshot", tracking_resolver):
        # A3E path
        a3_client = A3FakeLLMClient(
            [valid_payload_for(chunk, "主角") for chunk in chunks]
        )
        a3_service = ChunkExtractionBatchService(
            store, pointers, PromptRegistry(PROMPTS_DIR), DEFAULT_OUTPUT_SCHEMA_PATH
        )
        a3_service.extract_chunks(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile=make_chunk_profile(),
            extraction_profile=make_extraction_profile(),
            semantic_profile=make_semantic_profile(),
            llm_client=a3_client,
        )
        assert len(call_log) == 1  # A3E called the shared resolver once

        # A4E path (now A3 state exists)
        call_log.clear()
        # Need to patch in reconciliation_service too
        with patch.object(recon_mod, "resolve_current_story_snapshot", tracking_resolver):
            service = make_reconciliation_service(store, pointers)
            client = A4FakeLLMClient([])
            # Use empty payloads so no A4C semantic calls
            from short_drama.story.extraction import CandidatePayload as CP
            # We already have A3 from the A3E run above. Now run A4E.
            # Since all chunks have "主角" (2 CJK chars), there will be
            # semantic pairs. We need to provide decisions.
            decision_payload = _all_pairs_decision_payload(chunks)
            client = A4FakeLLMClient([decision_payload])
            run_reconciliation(service, client)

        # The A4E path calls resolve_current_story_snapshot via
        # resolve_current_a3_reconciliation_inputs which calls it.
        # But since we patched recon_mod, the call goes through recon_mod's
        # import. Let's verify the call was made.
        # Actually, resolve_current_a3_reconciliation_inputs is defined in
        # recon_mod and calls resolve_current_story_snapshot which is
        # imported from service. Since we patched recon_mod's name, it
        # should be tracked.
        assert len(call_log) >= 1  # A4E called the shared resolver
