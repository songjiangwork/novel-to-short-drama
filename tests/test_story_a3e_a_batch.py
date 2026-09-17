"""A3E-A deterministic PROJECT/BATCH chunk-extraction orchestration tests.

Covers the frozen A3E-A contract against a deterministic, provider-neutral fake
``LLMClient`` — NO live Qwen server, NO real provider. The batch layer must:

  * resolve the authoritative current A1 ``SourceDocument`` + A2 ``ChunkManifest``
    and verify their lineage / ValidationReports (reusing the A1/A2 project /
    persistence primitives);
  * iterate manifest chunks in exact deterministic manifest order (sequential);
  * invoke the A3D single-chunk service exactly once per chunk and let A3D/A3C
    decide reuse vs generation;
  * return a non-persisted batch summary that is NOT a new canonical authority;
  * fail closed (zero provider calls) on an upstream A1/A2 integrity failure and
    on a failing chunk (no successful summary).

Deliberately out of scope: the ``extract-chunks`` CLI (A3E-B), real-Qwen smoke,
real-novel / semantic-profile invalidation smoke (A3E-C), concurrency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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
    ProviderMeta,
    PromptRegistry,
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
    ChunkExtractionBatchSummary,
    ChunkExtractionService,
    ChunkManifest,
    ChunkPlanningProfile,
    DEFAULT_OUTPUT_SCHEMA_PATH,
    EvidenceRef,
    ExtractionSemanticGenerationError,
    LANGUAGE_DETECTOR_ID,
    SourceChapter,
    SourceChunk,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    StoryExtractionProfile,
    StoryIntegrityError,
    TOKEN_COUNTER_ID,
    load_source_document,
    persist_chunk_manifest,
    persist_source_chunk,
    persist_source_document,
    plan_chunks,
)
from short_drama.story.extraction import (
    CandidatePayload,
    CharacterCandidate,
    FactCandidate,
)
from short_drama.story.persistence import (
    chunk_pointer_id,
    chunk_validation_artifact_id,
    source_pointer_id,
    source_validation_artifact_id,
)
from short_drama.story.source import NormalizationInfo

# ---------------------------------------------------------------------------
# Frozen identity constants
# ---------------------------------------------------------------------------

PROJECT = "classroom"
DOCUMENT = "src_001"
CHUNK_PROFILE_ID = "story-analysis-v1"
EXTRACTION_PROFILE_ID = "story-extraction-v1"
PROMPTS_DIR = REPO_ROOT / "prompts" / "story"

_CHUNK_ID_RE = re.compile(r"chunk_id = (\S+)")


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
    from short_drama.llm import ReasoningSettings

    values = {
        "schema_version": 1,
        "profile_id": "story-extraction-llm-v1",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "structured_output_mode": "json_schema",
        "reasoning": ReasoningSettings(enabled=False),
    }
    values.update(overrides)
    return SemanticLLMProfile(**values)


def valid_payload_for(chunk: SourceChunk) -> CandidatePayload:
    """A valid A3B payload for ``chunk`` (primary evidence in its ownership)."""
    return CandidatePayload(
        characters=(
            CharacterCandidate(
                candidate_id="cand_char_001",
                display_name_original="主角",
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


def invalid_payload_for(chunk: SourceChunk) -> CandidatePayload:
    """A JSON-Schema-valid but A3B-invalid payload (dangling local cross-ref)."""
    return CandidatePayload(
        facts=(
            FactCandidate(
                candidate_id="cand_fact_001",
                fact_type="identity",
                statement_zh="测试事实。",
                subject_refs=("cand_char_999",),  # dangling local ref
                object_refs=(),
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
# A1 + A2 state setup (reuses the exact persistence primitives)
# ---------------------------------------------------------------------------


def setup_a1_a2(
    root: Path,
) -> tuple[FileArtifactStore, FilePointerStore, ArtifactRef, ChunkManifest, tuple[SourceChunk, ...]]:
    """Persist a current A1 ``SourceDocument`` + a current A2 ``ChunkManifest``
    (with their exact A1/A2 ValidationReports + CURRENT pointers)."""
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
    return store, pointers, source_ref, manifest, chunks


def make_batch_service(
    store: FileArtifactStore, pointers: FilePointerStore
) -> ChunkExtractionBatchService:
    return ChunkExtractionBatchService(
        store, pointers, PromptRegistry(PROMPTS_DIR), DEFAULT_OUTPUT_SCHEMA_PATH
    )


def make_single_service(
    store: FileArtifactStore, pointers: FilePointerStore
) -> ChunkExtractionService:
    return ChunkExtractionService(
        store, pointers, PromptRegistry(PROMPTS_DIR), DEFAULT_OUTPUT_SCHEMA_PATH
    )


def run_batch(
    service: ChunkExtractionBatchService, client: "BatchFakeLLMClient"
):
    return service.extract_chunks(
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile=make_chunk_profile(),
        extraction_profile=make_extraction_profile(),
        semantic_profile=make_semantic_profile(),
        llm_client=client,
    )


def run_single(
    service: ChunkExtractionService,
    source_ref: ArtifactRef,
    chunk_ref: ArtifactRef,
    client: "BatchFakeLLMClient",
):
    return service.extract_chunk(
        source_document_ref=source_ref,
        source_chunk_ref=chunk_ref,
        chunk_profile_id=CHUNK_PROFILE_ID,
        extraction_profile=make_extraction_profile(),
        semantic_profile=make_semantic_profile(),
        llm_client=client,
    )


def chunk_id_from_ref(ref: ArtifactRef) -> str:
    # <project>.<document>.<chunk_profile>.<chunk_id-lower>.<extraction_profile>
    return ref.artifact_id.split(".")[-2]


# ---------------------------------------------------------------------------
# Provider-neutral fake LLM client (batch-aware)
# ---------------------------------------------------------------------------


class BatchFakeLLMClient(LLMClient):
    """Deterministic fake provider returning a scripted sequence of payloads.

    Each scripted element is a ``CandidatePayload`` (returned as a validated
    success) or an ``Exception`` (raised). It honors the A-I3 trust boundary: a
    returned payload is first validated against the request's output schema. It
    records the ``chunk_id`` of each rendered prompt so tests can assert the
    exact execution order.
    """

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.call_count = 0
        self.chunk_ids = []

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.call_count += 1
        match = _CHUNK_ID_RE.search(rendered_prompt.user_text)
        if match is not None:
            self.chunk_ids.append(match.group(1))
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
                request, ProviderMeta(), request_model="qwen3-27b", provider_family="qwen"
            ),
            attempts=1,
        )


# ===========================================================================
# 1. Deterministic manifest-order execution
# ===========================================================================


def test_manifest_chunks_execute_in_exact_deterministic_order(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    assert len(chunks) >= 2
    expected_order = [chunk.chunk_id for chunk in chunks]

    client = BatchFakeLLMClient([valid_payload_for(chunk) for chunk in chunks])
    summary = run_batch(make_batch_service(store, pointers), client)

    # The provider was invoked once per chunk, in the exact manifest order.
    assert client.call_count == len(chunks)
    assert client.chunk_ids == expected_order
    # The summary refs are in the same manifest order.
    assert [chunk_id_from_ref(ref) for ref in summary.candidate_extraction_refs] == [
        cid.lower() for cid in expected_order
    ]


# ===========================================================================
# 2. First batch run generates all missing chunks
# ===========================================================================


def test_first_batch_run_generates_all_missing_chunks(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    client = BatchFakeLLMClient([valid_payload_for(chunk) for chunk in chunks])
    summary = run_batch(make_batch_service(store, pointers), client)

    assert summary.chunks_total == len(chunks)
    assert summary.chunks_generated == len(chunks)
    assert summary.chunks_reused == 0
    assert summary.chunks_failed == 0
    assert client.call_count == len(chunks)
    # every chunk now has an immutable revision-1 CandidateExtraction.
    assert all(
        ref.revision == 1 for ref in summary.candidate_extraction_refs
    )


# ===========================================================================
# 3. Exact second batch run reuses every eligible chunk
# ===========================================================================


def test_exact_second_run_reuses_every_eligible_chunk(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    first = run_batch(
        make_batch_service(store, pointers),
        BatchFakeLLMClient([valid_payload_for(chunk) for chunk in chunks]),
    )
    assert first.chunks_generated == len(chunks)

    second_client = BatchFakeLLMClient(
        [valid_payload_for(chunk) for chunk in chunks]
    )
    second = run_batch(make_batch_service(store, pointers), second_client)

    assert second_client.call_count == 0  # zero provider calls on full reuse
    assert second.chunks_reused == len(chunks)
    assert second.chunks_generated == 0
    assert second.chunks_failed == 0
    assert second.candidate_extraction_refs == first.candidate_extraction_refs
    assert second.validation_report_refs == first.validation_report_refs


# ===========================================================================
# 4. Mixed state: only missing/stale chunks generate
# ===========================================================================


def test_mixed_state_only_missing_chunks_generate(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    single = make_single_service(store, pointers)

    # Pre-generate all but the last two chunks -> they become reusable. The
    # last two remain missing.
    pre_refs = []
    for chunk, ref in list(zip(chunks, manifest.chunk_refs))[:-2]:
        publication = run_single(
            single, source_ref, ref, BatchFakeLLMClient([valid_payload_for(chunk)])
        )
        pre_refs.append(publication.candidate_extraction_ref)

    # Only the two missing chunks (in manifest order) need a generated payload.
    last_two = chunks[-2:]
    client = BatchFakeLLMClient([valid_payload_for(chunk) for chunk in last_two])
    summary = run_batch(make_batch_service(store, pointers), client)

    assert summary.chunks_total == len(chunks)
    assert summary.chunks_reused == len(chunks) - 2
    assert summary.chunks_generated == 2
    assert summary.chunks_failed == 0
    assert client.call_count == 2
    # Reused chunks keep their exact pre-published refs; order is preserved.
    assert summary.candidate_extraction_refs[: len(pre_refs)] == tuple(pre_refs)


# ===========================================================================
# 5. Summary counts generated vs reused correctly
# ===========================================================================


def test_summary_counts_generated_vs_reused(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    single = make_single_service(store, pointers)
    # Pre-generate exactly the first half -> reusable; the second half missing.
    split = len(chunks) // 2
    for chunk, ref in list(zip(chunks, manifest.chunk_refs))[:split]:
        run_single(
            single, source_ref, ref, BatchFakeLLMClient([valid_payload_for(chunk)])
        )

    client = BatchFakeLLMClient(
        [valid_payload_for(chunk) for chunk in chunks[split:]]
    )
    summary = run_batch(make_batch_service(store, pointers), client)

    assert summary.chunks_total == len(chunks)
    assert summary.chunks_reused == split
    assert summary.chunks_generated == len(chunks) - split
    assert summary.chunks_failed == 0
    # accounting invariant for a successful run
    assert summary.chunks_total == summary.chunks_reused + summary.chunks_generated
    # exactly one provider call per generated (non-reused) chunk
    assert client.call_count == summary.chunks_generated


# ===========================================================================
# 6. Returned extraction/report refs preserve manifest order
# ===========================================================================


def test_extraction_and_report_refs_preserve_manifest_order(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    client = BatchFakeLLMClient([valid_payload_for(chunk) for chunk in chunks])
    summary = run_batch(make_batch_service(store, pointers), client)

    expected = [chunk.chunk_id.lower() for chunk in chunks]
    extraction_order = [chunk_id_from_ref(ref) for ref in summary.candidate_extraction_refs]
    assert extraction_order == expected
    # validation report refs carry the same logical order (their artifact id is
    # the extraction artifact id + ".a3-validation").
    report_order = [
        ref.artifact_id[: -len(".a3-validation")]
        for ref in summary.validation_report_refs
    ]
    assert [part.split(".")[-2] for part in report_order] == expected
    # the two ref lists are aligned element-wise (same chunk at each index).
    for extraction_ref, report_ref in zip(
        summary.candidate_extraction_refs, summary.validation_report_refs
    ):
        assert report_ref.artifact_id.startswith(extraction_ref.artifact_id)


# ===========================================================================
# 7. Upstream A1/A2 integrity failure -> zero provider calls
# ===========================================================================


def test_upstream_a1_integrity_failure_causes_zero_provider_calls(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    # Destroy the A1 ValidationReport backing the current SourceDocument.
    a1_report_id = source_validation_artifact_id(PROJECT, DOCUMENT)
    (
        store.root
        / "validation_report"
        / a1_report_id
        / f"r{source_ref.revision:08d}.json"
    ).unlink()

    client = BatchFakeLLMClient([valid_payload_for(chunk) for chunk in chunks])
    with pytest.raises(StoryIntegrityError):
        run_batch(make_batch_service(store, pointers), client)
    assert client.call_count == 0


def test_stale_a2_manifest_not_pinning_current_source_fails_closed(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    # Publish a NEWER A1 SourceDocument (revision 2) AND its exact valid A1
    # ValidationReport, then move the A1 CURRENT pointer to revision 2, leaving
    # the A2 manifest pinning the (now stale) source revision 1. This proves the
    # stale-A2 guard fires (not an earlier A1 validation failure):
    #   valid current A1 rev2 + valid-but-stale A2 manifest pinning rev1 -> fail.
    current_source = load_source_document(store, source_ref)
    newer = SourceDocument(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source=SourceInfo(
            "txt", "source/novel.txt", "e" * 64, 128, "zh-CN", "zh", LANGUAGE_DETECTOR_ID
        ),
        normalization=NormalizationInfo(
            "utf-8", "LF", "short_drama_source_ingestion_v1", "1"
        ),
        chapters=current_source.chapters,
    )
    newer_ref = persist_source_document(store, newer, revision=2)
    persist_validation_report(
        store,
        ValidationReport(
            validated_refs=(LineageRef("source_document", newer_ref),),
            findings=(),
        ),
        artifact_id=source_validation_artifact_id(PROJECT, DOCUMENT),
        revision=newer_ref.revision,
    )
    pointers.compare_and_set(
        pointer_id=source_pointer_id(PROJECT, DOCUMENT),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=pointers.resolve_current_pointer_ref(
            source_pointer_id(PROJECT, DOCUMENT)
        ),
        target_ref=newer_ref,
    )

    client = BatchFakeLLMClient([valid_payload_for(chunk) for chunk in chunks])
    with pytest.raises(StoryIntegrityError) as exc_info:
        run_batch(make_batch_service(store, pointers), client)
    # The failure is the stale-A2 guard (manifest pins the old source), reached
    # only after A1 rev2 resolves as current and valid.
    assert "stale relative to the current source" in str(exc_info.value)
    assert client.call_count == 0


# ===========================================================================
# 8. A failing chunk fails the batch cleanly (no successful summary)
# ===========================================================================


def test_failing_chunk_fails_batch_cleanly(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    single = make_single_service(store, pointers)
    # Pre-generate all but the last chunk (they become reusable + persisted).
    for chunk, ref in list(zip(chunks, manifest.chunk_refs))[:-1]:
        run_single(
            single, source_ref, ref, BatchFakeLLMClient([valid_payload_for(chunk)])
        )

    # The last chunk is missing and fails both semantic rounds (dangling ref).
    last = chunks[-1]
    client = BatchFakeLLMClient([invalid_payload_for(last), invalid_payload_for(last)])
    service = make_batch_service(store, pointers)
    with pytest.raises(ExtractionSemanticGenerationError):
        run_batch(service, client)
    # exactly the two failed semantic rounds hit the provider
    assert client.call_count == 2
    # no successful summary was manufactured: the failing chunk has no current,
    # while the earlier chunks remain persisted (resume is possible on rerun).
    from short_drama.story import candidate_extraction_pointer_id

    last_pointer = candidate_extraction_pointer_id(
        PROJECT, DOCUMENT, CHUNK_PROFILE_ID, last.chunk_id, EXTRACTION_PROFILE_ID
    )
    try:
        pointers.resolve_current(last_pointer)
        raised = False
    except Exception:  # noqa: BLE001
        raised = True
    assert raised, "failing chunk must not publish a CURRENT CandidateExtraction"


# ===========================================================================
# 9. Batch layer creates no new canonical artifact / pointer authority
# ===========================================================================


def test_batch_creates_no_new_canonical_authority(tmp_path):
    store, pointers, source_ref, manifest, chunks = setup_a1_a2(tmp_path)
    client = BatchFakeLLMClient([valid_payload_for(chunk) for chunk in chunks])
    summary = run_batch(make_batch_service(store, pointers), client)

    # The summary is an in-memory dataclass, not a persisted artifact/pointer.
    assert isinstance(summary, ChunkExtractionBatchSummary)
    assert not isinstance(summary, ArtifactRef)

    # No new "batch" artifact type or pointer was introduced: the store holds
    # only the pre-existing A1/A2/A3 artifact types.
    artifact_types = {p.name for p in store.root.iterdir() if p.is_dir()}
    known = {
        "source_document",
        "source_chunk",
        "chunk_manifest",
        "candidate_extraction",
        "validation_report",
        "current_pointer",
        "supersession_record",
    }
    assert artifact_types <= known
    assert "candidate_extraction" in artifact_types

    # Every summary ref resolves to an existing per-chunk artifact.
    for ref in summary.candidate_extraction_refs:
        store.get_ref(ref)
    for ref in summary.validation_report_refs:
        store.get_ref(ref)
