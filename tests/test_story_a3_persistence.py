"""A3C CandidateExtraction persistence / ValidationReport / CURRENT / reuse tests.

Covers the frozen A3C contract (parent A-I4 sections 20-24, 28):

  * deterministic artifact / ValidationReport / CURRENT-pointer identity;
  * immutable persistence + a fail-closed typed loader;
  * exact A3 ValidationReport lineage + deterministic reuse verification;
  * current-only semantic reuse (and every frozen identity-invalidation field);
  * supersession (new revision under the same logical artifact ID), historical
    retention, and no historical auto-resurrection;
  * failed candidate validation never replaces a valid CURRENT.

Deliberately does NOT require: a running Qwen server, an LLM client, a stage
CLI, a real-novel fixture, or any A3D/A3E behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from short_drama.artifacts import (
    ArtifactRef,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
    canonical_json_bytes,
)
from short_drama.foundation import (
    FilePointerStore,
    LineageRef,
    PointerKind,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
    ValidationResult,
    load_validation_report,
    persist_validation_report,
)
from short_drama.llm import LLMInvocationProvenance
from short_drama.story import (
    CandidateExtraction,
    CandidateExtractionPublication,
    CandidateExtractionService,
    CandidatePayload,
    CharacterCandidate,
    EvidenceRef,
    FactCandidate,
    LANGUAGE_DETECTOR_ID,
    ParagraphSpan,
    SOURCE_PARSER_ID,
    SourceChapter,
    SourceChunk,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    StoryExtractionProfile,
    StoryIntegrityError,
    StoryPersistenceError,
    TOKEN_COUNTER_ID,
    candidate_extraction_artifact_id,
    candidate_extraction_pointer_id,
    candidate_extraction_validation_artifact_id,
    load_candidate_extraction,
    persist_candidate_extraction,
    persist_source_chunk,
    persist_source_document,
)
from short_drama.story.source import NormalizationInfo


# ---------------------------------------------------------------------------
# Frozen identity constants
# ---------------------------------------------------------------------------

PROJECT = "classroom"
DOCUMENT = "src_001"
CHUNK_PROFILE_ID = "story-analysis-v1"
CHUNK_ID = "CH001_C001"
CHAPTER_ID = "CH001"
EXTRACTION_PROFILE_ID = "story-extraction-v1"

PARAGRAPHS = {
    "CH001_P0001": "左上下文第一段。",
    "CH001_P0002": "左上下文第二段。",
    "CH001_P0003": "林晚走进教室。",
    "CH001_P0004": "老师正在板书。",
    "CH001_P0005": "右上下文第一段。",
    "CH001_P0006": "右上下文第二段。",
}
OWNERSHIP = ("CH001_P0003", "CH001_P0004")


# ---------------------------------------------------------------------------
# Builders / fixtures
# ---------------------------------------------------------------------------


def make_evidence(
    paragraph_id: str = "CH001_P0003",
    role: str = "primary",
    strength: str = "explicit",
    excerpt: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role=role, strength=strength, excerpt=excerpt
    )


def make_character(candidate_id: str, display_name: str, evidence) -> CharacterCandidate:
    return CharacterCandidate(
        candidate_id=candidate_id,
        display_name_original=display_name,
        aliases_original=(),
        descriptors_zh=(),
        summary_zh=f"{display_name}的简要描述。",
        evidence_strength="explicit",
        evidence=tuple(evidence),
    )


def canonical_payload() -> CandidatePayload:
    """A valid, already-canonical candidate payload (primary evidence in
    ownership)."""
    return CandidatePayload(
        characters=(
            make_character("cand_char_001", "林晚", (make_evidence("CH001_P0003"),)),
            make_character("cand_char_002", "老师", (make_evidence("CH001_P0004"),)),
        ),
    )


def noncanonical_payload() -> CandidatePayload:
    """Valid but NOT in deterministic canonical form (002 ordered before 001)."""
    return CandidatePayload(
        characters=(
            make_character("cand_char_002", "老师", (make_evidence("CH001_P0004"),)),
            make_character("cand_char_001", "林晚", (make_evidence("CH001_P0003"),)),
        ),
    )


def make_provenance(**overrides) -> LLMInvocationProvenance:
    values = {
        "provider_family": "qwen",
        "model": "qwen3-27b",
        "semantic_profile_id": "story-llm-qwen-v1",
        "semantic_profile_hash": "a" * 64,
        "prompt_id": "a3.chunk-extraction",
        "prompt_version": 1,
        "prompt_content_hash": "b" * 64,
        "rendered_prompt_hash": "c" * 64,
        "output_schema_id": "a3-candidate-payload",
        "output_schema_version": 1,
        "output_schema_hash": "d" * 64,
        "request_hash": "e" * 64,
        "provider_response_id": None,
        "finish_reason": None,
        "usage": None,
    }
    values.update(overrides)
    return LLMInvocationProvenance(**values)


def make_profile(**overrides) -> StoryExtractionProfile:
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


def build_source_document() -> SourceDocument:
    chapter = SourceChapter(
        CHAPTER_ID,
        None,
        "synthetic",
        tuple(
            SourceParagraph(pid, PARAGRAPHS[pid], None) for pid in sorted(PARAGRAPHS)
        ),
    )
    return SourceDocument(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        source=SourceInfo(
            "txt", "source/novel.txt", "f" * 64, 128, "zh-CN", "zh", LANGUAGE_DETECTOR_ID
        ),
        normalization=NormalizationInfo("utf-8", "LF", SOURCE_PARSER_ID, "1"),
        chapters=(chapter,),
    )


def build_source_chunk(source_document_ref: ArtifactRef) -> SourceChunk:
    pids = tuple(sorted(PARAGRAPHS))
    return SourceChunk(
        schema_version=1,
        chunk_id=CHUNK_ID,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chapter_id=CHAPTER_ID,
        source_document_ref=source_document_ref,
        context_span=ParagraphSpan(pids[0], pids[-1]),
        ownership_span=ParagraphSpan(OWNERSHIP[0], OWNERSHIP[-1]),
        paragraph_ids=pids,
        token_count_method=TOKEN_COUNTER_ID,
        context_token_count=10,
        ownership_token_count=5,
    )


@dataclass
class Harness:
    store: FileArtifactStore
    pointers: FilePointerStore
    service: CandidateExtractionService
    source_document: SourceDocument
    source_document_ref: ArtifactRef
    source_chunk: SourceChunk
    source_chunk_ref: ArtifactRef
    profile: StoryExtractionProfile
    provenance: LLMInvocationProvenance
    payload: CandidatePayload


def make_harness(tmp_path: Path, **overrides) -> Harness:
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)

    source_document = build_source_document()
    source_document_ref = persist_source_document(
        store, source_document, revision=1
    )
    source_chunk = build_source_chunk(source_document_ref)
    source_chunk_ref = persist_source_chunk(
        store, source_chunk, profile_id=CHUNK_PROFILE_ID, revision=1
    )

    profile = overrides.get("profile", make_profile())
    provenance = overrides.get("provenance", make_provenance())
    payload = overrides.get("payload", canonical_payload())
    service = CandidateExtractionService(store, pointers)
    return Harness(
        store=store,
        pointers=pointers,
        service=service,
        source_document=source_document,
        source_document_ref=source_document_ref,
        source_chunk=source_chunk,
        source_chunk_ref=source_chunk_ref,
        profile=profile,
        provenance=provenance,
        payload=payload,
    )


def extract_kwargs(h: Harness, **overrides) -> dict:
    kw = dict(
        source_document=h.source_document,
        source_document_ref=h.source_document_ref,
        source_chunk=h.source_chunk,
        source_chunk_ref=h.source_chunk_ref,
        chunk_profile_id=CHUNK_PROFILE_ID,
        extraction_profile=h.profile,
        generation_provenance=h.provenance,
        payload=h.payload,
    )
    kw.update(overrides)
    return kw


def make_extraction(
    h: Harness,
    *,
    payload: CandidatePayload | None = None,
    provenance: LLMInvocationProvenance | None = None,
    profile: StoryExtractionProfile | None = None,
    source_document_ref: ArtifactRef | None = None,
    source_chunk_ref: ArtifactRef | None = None,
) -> CandidateExtraction:
    profile = profile or h.profile
    return CandidateExtraction(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile_id=CHUNK_PROFILE_ID,
        chunk_id=CHUNK_ID,
        source_document_ref=source_document_ref or h.source_document_ref,
        source_chunk_ref=source_chunk_ref or h.source_chunk_ref,
        extraction_profile_id=profile.profile_id,
        extraction_profile_hash=profile.profile_hash,
        generation_provenance=provenance or h.provenance,
        candidates=payload or canonical_payload(),
    )


def pointer_id() -> str:
    return candidate_extraction_pointer_id(
        PROJECT, DOCUMENT, CHUNK_PROFILE_ID, CHUNK_ID, EXTRACTION_PROFILE_ID
    )


def extraction_artifact_id() -> str:
    return candidate_extraction_artifact_id(
        PROJECT, DOCUMENT, CHUNK_PROFILE_ID, CHUNK_ID, EXTRACTION_PROFILE_ID
    )


def validation_artifact_id() -> str:
    return candidate_extraction_validation_artifact_id(extraction_artifact_id())


def store_path(store: FileArtifactStore, artifact_type: str, artifact_id: str, revision: int) -> Path:
    return store.root / artifact_type / artifact_id / f"r{revision:08d}.json"


def _finding(severity: ValidationSeverity, code: str) -> ValidationFinding:
    return ValidationFinding(
        finding_id=code,
        code=code,
        severity=severity,
        owner_stage="A3",
        repair_route="A3_REGENERATE_CANDIDATE_PAYLOAD",
        message=code,
    )


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def test_artifact_identity_matches_frozen_contract():
    assert extraction_artifact_id() == (
        "classroom.src_001.story-analysis-v1.ch001_c001.story-extraction-v1"
    )
    assert validation_artifact_id() == (
        "classroom.src_001.story-analysis-v1.ch001_c001.story-extraction-v1"
        ".a3-validation"
    )
    assert pointer_id() == (
        "classroom.a3.src_001.story-analysis-v1.ch001_c001.story-extraction-v1"
    )


# ---------------------------------------------------------------------------
# 1-5. Persistence
# ---------------------------------------------------------------------------


def test_first_valid_publish(tmp_path):
    h = make_harness(tmp_path)
    result = h.service.extract_chunk(**extract_kwargs(h))

    assert isinstance(result, CandidateExtractionPublication)
    assert result.reused is False
    # CandidateExtraction + ValidationReport persisted at matching revisions.
    assert result.candidate_extraction_ref.revision == 1
    assert result.validation_report_ref.revision == 1
    loaded = load_candidate_extraction(h.store, result.candidate_extraction_ref)
    assert loaded.candidates == h.payload
    # ValidationReport lineage + PASS result.
    report = load_validation_report(h.store, result.validation_report_ref)
    assert {(ref.role, ref.artifact_ref) for ref in report.validated_refs} == {
        ("source_document", h.source_document_ref),
        ("source_chunk", h.source_chunk_ref),
        ("candidate_extraction", result.candidate_extraction_ref),
    }
    assert report.summary.result is ValidationResult.PASS
    # CURRENT created and points at the extraction.
    pointer = h.pointers.resolve_current(pointer_id())
    assert pointer.pointer_kind is PointerKind.CURRENT
    assert pointer.target_ref == result.candidate_extraction_ref


def test_typed_loader_round_trip(tmp_path):
    h = make_harness(tmp_path)
    result = h.service.extract_chunk(**extract_kwargs(h))
    loaded = load_candidate_extraction(h.store, result.candidate_extraction_ref)
    assert loaded == make_extraction(h, payload=canonical_payload())
    assert loaded.candidates == h.payload


def test_bad_artifact_type_fails(tmp_path):
    h = make_harness(tmp_path)
    result = h.service.extract_chunk(**extract_kwargs(h))
    wrong_type = ArtifactRef(
        artifact_type="not_candidate_extraction",
        artifact_id=result.candidate_extraction_ref.artifact_id,
        revision=result.candidate_extraction_ref.revision,
        content_hash=result.candidate_extraction_ref.content_hash,
    )
    with pytest.raises(StoryIntegrityError, match="wrong artifact_type"):
        load_candidate_extraction(h.store, wrong_type)


def test_artifact_id_payload_identity_mismatch_fails(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h)
    # Persist under a deliberately wrong artifact_id (bypass the helper).
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type="candidate_extraction",
        artifact_id="wrong.artifact.id",
        revision=1,
        schema_version=1,
        payload=extraction.to_dict(),
    )
    ref = h.store.put(envelope)
    with pytest.raises(
        StoryIntegrityError, match="artifact_id does not match payload identity"
    ):
        load_candidate_extraction(h.store, ref)


def test_tampered_payload_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    result = h.service.extract_chunk(**extract_kwargs(h))
    path = store_path(
        h.store, "candidate_extraction", extraction_artifact_id(), 1
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["candidates"]["characters"][0]["summary_zh"] = "被篡改的摘要。"
    path.write_bytes(canonical_json_bytes(data))
    with pytest.raises(StoryIntegrityError):
        load_candidate_extraction(h.store, result.candidate_extraction_ref)


# ---------------------------------------------------------------------------
# 6-13. Exact current-only reuse
# ---------------------------------------------------------------------------


def test_identical_semantic_identity_reuses_same_ref(tmp_path):
    h = make_harness(tmp_path)
    first = h.service.extract_chunk(**extract_kwargs(h))
    second = h.service.extract_chunk(**extract_kwargs(h))
    assert second.reused is True
    assert second.candidate_extraction_ref == first.candidate_extraction_ref
    assert second.validation_report_ref == first.validation_report_ref
    assert second.current_pointer_ref == first.current_pointer_ref


def test_identical_rerun_does_not_allocate_new_revision(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    second = h.service.extract_chunk(**extract_kwargs(h))
    assert second.reused is True
    assert second.candidate_extraction_ref.revision == 1
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_exact_matching_pass_report_required(tmp_path):
    h = make_harness(tmp_path)
    first = h.service.extract_chunk(**extract_kwargs(h))
    # A second identical run re-verifies and reuses the exact PASS report.
    second = h.service.extract_chunk(**extract_kwargs(h))
    assert second.reused is True
    report = load_validation_report(h.store, second.validation_report_ref)
    assert report == load_validation_report(h.store, first.validation_report_ref)
    assert report.summary.result is ValidationResult.PASS


def test_missing_report_prevents_reuse_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    # Persist extraction + CURRENT with NO validation report.
    extraction = make_extraction(h)
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=ref,
    )
    with pytest.raises(StoryIntegrityError, match="ValidationReport is missing"):
        h.service.extract_chunk(**extract_kwargs(h))


def test_mismatched_report_lineage_prevents_reuse(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h)
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    # Report with the wrong source_chunk lineage ref.
    wrong_chunk_ref = ArtifactRef(
        "source_chunk", extraction.source_chunk_ref.artifact_id, 1, "9" * 64
    )
    bad_report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", wrong_chunk_ref),
            LineageRef("candidate_extraction", ref),
        ),
        findings=(),
    )
    persist_validation_report(
        h.store,
        bad_report,
        artifact_id=validation_artifact_id(),
        revision=1,
    )
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=ref,
    )
    with pytest.raises(StoryIntegrityError, match="ValidationReport"):
        h.service.extract_chunk(**extract_kwargs(h))


def test_fail_report_cannot_back_current(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h)
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    bad_report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", h.source_chunk_ref),
            LineageRef("candidate_extraction", ref),
        ),
        findings=(_finding(ValidationSeverity.BLOCKING, "A3_FAIL"),),
    )
    assert bad_report.summary.result is ValidationResult.FAIL
    persist_validation_report(
        h.store, bad_report, artifact_id=validation_artifact_id(), revision=1
    )
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=ref,
    )
    with pytest.raises(StoryIntegrityError):
        h.service.extract_chunk(**extract_kwargs(h))


def test_pass_with_review_items_cannot_be_current_eligible(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h)
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    bad_report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", h.source_chunk_ref),
            LineageRef("candidate_extraction", ref),
        ),
        findings=(_finding(ValidationSeverity.REVIEW_REQUIRED, "A3_REVIEW"),),
    )
    assert bad_report.summary.result is ValidationResult.PASS_WITH_REVIEW_ITEMS
    persist_validation_report(
        h.store, bad_report, artifact_id=validation_artifact_id(), revision=1
    )
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=ref,
    )
    with pytest.raises(StoryIntegrityError):
        h.service.extract_chunk(**extract_kwargs(h))


def test_current_pointer_change_during_reuse_fails_closed(tmp_path, monkeypatch):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))

    original = h.pointers.resolve_current_pointer_ref
    calls = {"n": 0}

    def flaky_resolve(pointer_id):
        calls["n"] += 1
        if calls["n"] >= 3:  # the re-check call after initial resolution
            return ArtifactRef("current_pointer", pointer_id, 999, "f" * 64)
        return original(pointer_id)

    monkeypatch.setattr(h.pointers, "resolve_current_pointer_ref", flaky_resolve)
    with pytest.raises(StoryPersistenceError, match="changed during reuse verification"):
        h.service.extract_chunk(**extract_kwargs(h))


def test_reuse_rejects_noncanonical_persisted_payload(tmp_path):
    h = make_harness(tmp_path)
    # Persist a VALID but non-canonical payload + a correct PASS report + CURRENT.
    extraction = make_extraction(h, payload=noncanonical_payload())
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", h.source_chunk_ref),
            LineageRef("candidate_extraction", ref),
        ),
        findings=(),
    )
    persist_validation_report(
        h.store, report, artifact_id=validation_artifact_id(), revision=1
    )
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=ref,
    )
    # Reuse must fail closed: the stored payload is not the canonical form.
    with pytest.raises(StoryIntegrityError, match="canonical form"):
        h.service.extract_chunk(**extract_kwargs(h))


# ---------------------------------------------------------------------------
# 14-21. Identity invalidation
# ---------------------------------------------------------------------------


def test_source_document_ref_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    new_doc_ref = ArtifactRef(
        "source_document", h.source_document_ref.artifact_id, 1, "9" * 64
    )
    result = h.service.extract_chunk(
        **extract_kwargs(h, source_document_ref=new_doc_ref)
    )
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 2


def test_source_chunk_ref_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    new_chunk_ref = ArtifactRef(
        "source_chunk", h.source_chunk_ref.artifact_id, 1, "9" * 64
    )
    result = h.service.extract_chunk(
        **extract_kwargs(h, source_chunk_ref=new_chunk_ref)
    )
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 2


def test_extraction_profile_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    alt_profile = make_profile(working_language="en-US")
    assert alt_profile.profile_hash != h.profile.profile_hash
    result = h.service.extract_chunk(
        **extract_kwargs(h, extraction_profile=alt_profile)
    )
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 2


def test_semantic_profile_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    result = h.service.extract_chunk(
        **extract_kwargs(
            h,
            generation_provenance=replace(
                h.provenance, semantic_profile_hash="f" * 64
            ),
        )
    )
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 2


def test_prompt_content_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    result = h.service.extract_chunk(
        **extract_kwargs(
            h,
            generation_provenance=replace(h.provenance, prompt_content_hash="f" * 64),
        )
    )
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 2


def test_rendered_prompt_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    result = h.service.extract_chunk(
        **extract_kwargs(
            h,
            generation_provenance=replace(
                h.provenance, rendered_prompt_hash="f" * 64
            ),
        )
    )
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 2


def test_output_schema_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    result = h.service.extract_chunk(
        **extract_kwargs(
            h,
            generation_provenance=replace(h.provenance, output_schema_hash="f" * 64),
        )
    )
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 2


def test_request_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    h.service.extract_chunk(**extract_kwargs(h))
    result = h.service.extract_chunk(
        **extract_kwargs(
            h, generation_provenance=replace(h.provenance, request_hash="f" * 64)
        )
    )
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 2


def test_non_semantic_provider_metadata_change_does_not_invalidate(tmp_path):
    h = make_harness(tmp_path)
    first = h.service.extract_chunk(**extract_kwargs(h))
    # provider_family / model / provider_response_id / finish_reason / usage are
    # NOT part of the A3 semantic reuse identity.
    alt_provenance = replace(
        h.provenance,
        provider_family="qwen",
        model="qwen3-27b-v2",
        provider_response_id="resp-123",
        finish_reason="stop",
        usage={"prompt_tokens": 5, "completion_tokens": 7},
    )
    result = h.service.extract_chunk(
        **extract_kwargs(h, generation_provenance=alt_provenance)
    )
    assert result.reused is True
    assert result.candidate_extraction_ref == first.candidate_extraction_ref


# ---------------------------------------------------------------------------
# 22-25. Supersession / history
# ---------------------------------------------------------------------------


def test_stale_identity_creates_new_revision_same_artifact_id(tmp_path):
    h = make_harness(tmp_path)
    first = h.service.extract_chunk(**extract_kwargs(h))
    second = h.service.extract_chunk(
        **extract_kwargs(
            h, generation_provenance=replace(h.provenance, request_hash="f" * 64)
        )
    )
    assert first.candidate_extraction_ref.revision == 1
    assert second.candidate_extraction_ref.revision == 2
    assert (
        first.candidate_extraction_ref.artifact_id
        == second.candidate_extraction_ref.artifact_id
        == extraction_artifact_id()
    )
    # CURRENT moved to the new revision by CAS.
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        second.candidate_extraction_ref
    )
    # The old revision remains exactly resolvable.
    assert load_candidate_extraction(h.store, first.candidate_extraction_ref) == make_extraction(
        h
    )


def test_no_historical_auto_resurrection(tmp_path):
    h = make_harness(tmp_path)
    first = h.service.extract_chunk(**extract_kwargs(h))
    h.service.extract_chunk(
        **extract_kwargs(
            h, generation_provenance=replace(h.provenance, request_hash="f" * 64)
        )
    )
    # Request the OLD (historical) identity again: it must publish a NEW revision,
    # not resurrect the historical revision 1.
    result = h.service.extract_chunk(**extract_kwargs(h))
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 3
    assert result.candidate_extraction_ref != first.candidate_extraction_ref
    # Historical revision 1 is still resolvable but is not current.
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        result.candidate_extraction_ref
    )
    assert load_candidate_extraction(h.store, first.candidate_extraction_ref) == make_extraction(
        h
    )


# ---------------------------------------------------------------------------
# 26-27. Failed candidate validation
# ---------------------------------------------------------------------------


def _invalid_payload() -> CandidatePayload:
    # A dangling local reference: structurally valid, semantically invalid.
    return CandidatePayload(
        characters=(make_character("cand_char_001", "林晚", (make_evidence(),)),),
        facts=(
            FactCandidate(
                candidate_id="cand_fact_001",
                fact_type="identity",
                statement_zh="x",
                subject_refs=("cand_char_999",),
                object_refs=(),
                evidence_strength="explicit",
                evidence=(make_evidence(),),
            ),
        ),
    )


def test_invalid_candidate_does_not_replace_valid_current(tmp_path):
    h = make_harness(tmp_path)
    first = h.service.extract_chunk(**extract_kwargs(h))
    # A new candidate with a different identity and an invalid payload must not
    # be published; the valid CURRENT remains revision 1.
    with pytest.raises(StoryIntegrityError, match="cannot become current"):
        h.service.extract_chunk(
            **extract_kwargs(
                h,
                payload=_invalid_payload(),
                generation_provenance=replace(h.provenance, request_hash="f" * 64),
            )
        )
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_invalid_candidate_same_identity_reuses_not_republishes(tmp_path):
    h = make_harness(tmp_path)
    first = h.service.extract_chunk(**extract_kwargs(h))
    # Same identity but an invalid payload: current is reused, nothing new is
    # persisted.
    result = h.service.extract_chunk(**extract_kwargs(h, payload=_invalid_payload()))
    assert result.reused is True
    assert result.candidate_extraction_ref == first.candidate_extraction_ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_invalid_payload_not_persisted_as_canonical(tmp_path):
    # A fresh store + an invalid candidate must not persist anything at all.
    h = make_harness(tmp_path)
    with pytest.raises(StoryIntegrityError, match="cannot become current"):
        h.service.extract_chunk(**extract_kwargs(h, payload=_invalid_payload()))
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)
