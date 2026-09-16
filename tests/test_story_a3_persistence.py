"""A3C CandidateExtraction persistence / ValidationReport / CURRENT / reuse tests.

Covers the frozen A3C contract (parent A-I4 sections 20-24, 28) plus the A3C
hardening requirements:

  * deterministic artifact / ValidationReport / CURRENT-pointer identity;
  * immutable persistence + a fail-closed typed loader;
  * exact A3 ValidationReport lineage + deterministic reuse verification;
  * **exact source ArtifactRef lineage** (forged refs / mismatched objects /
    incoherent chunk lineage all fail closed, no publish);
  * current-only semantic reuse (and every frozen identity-invalidation field);
  * **CURRENT is fully verified before the requested identity is compared**, so a
    corrupt / wrong-logical-target CURRENT fails closed even when the requested
    identity differs (never silently superseded);
  * supersession (new revision under the same logical artifact ID), historical
    retention, and no historical auto-resurrection;
  * a **pre-generation** reuse API usable before the provider call (zero provider
    calls) plus a **post-generation** publish API;
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
    PointerNotFoundError,
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
    ValidationResult,
    load_validation_report,
    persist_validation_report,
)
from short_drama.llm import (
    LLMInvocationProvenance,
    OutputSchema,
    ReasoningSettings,
    RenderedPrompt,
    SemanticLLMProfile,
    StructuredGenerationRequest,
    build_structured_request,
    compute_rendered_prompt_hash,
)
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
    extraction_semantic_identity,
    load_candidate_extraction,
    persist_candidate_extraction,
    persist_source_chunk,
    persist_source_document,
    request_semantic_fields,
)
from short_drama.story.persistence import (
    source_chunk_artifact_id,
    source_document_artifact_id,
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


def _default_output_schema() -> OutputSchema:
    return OutputSchema.create(
        schema_id="a3-candidate-payload", schema_version=1, schema={"type": "object"}
    )


def make_request(
    *,
    provider_family: str = "qwen",
    model: str = "qwen3-27b",
    semantic_profile_id: str = "story-llm-qwen-v1",
    temperature: float = 0.1,
    max_output_tokens: int = 20000,
    prompt_id: str = "a3.chunk-extraction",
    prompt_version: int = 1,
    prompt_content_hash: str = "b" * 64,
    system_text: str = "You extract story candidates.",
    user_text: str = "Chunk: 林晚走进教室。",
    output_schema: OutputSchema | None = None,
) -> StructuredGenerationRequest:
    """A provider-neutral A-I3 request whose semantic fields are controllable.

    ``semantic_profile_hash`` / ``rendered_prompt_hash`` / ``request_hash`` are
    derived from the underlying material, so changing the material changes the
    derived hash (as a real semantic change would).
    """
    semantic_profile = SemanticLLMProfile(
        schema_version=1,
        profile_id=semantic_profile_id,
        provider_family=provider_family,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(enabled=False),
    )
    variables_hash = "2" * 64
    rendered_prompt = RenderedPrompt(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        prompt_content_hash=prompt_content_hash,
        variables_hash=variables_hash,
        system_text=system_text,
        user_text=user_text,
        rendered_prompt_hash=compute_rendered_prompt_hash(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            variables_hash=variables_hash,
            system_text=system_text,
            user_text=user_text,
        ),
    )
    return build_structured_request(
        rendered_prompt=rendered_prompt,
        output_schema=output_schema if output_schema is not None else _default_output_schema(),
        semantic_profile=semantic_profile,
    )


def provenance_from_request(
    request: StructuredGenerationRequest, **overrides
) -> LLMInvocationProvenance:
    """A provenance whose ten A-I3 semantic fields exactly match the request."""
    values = dict(
        provider_family=request.semantic_profile.provider_family,
        model=request.model,
        semantic_profile_id=request.semantic_profile.profile_id,
        semantic_profile_hash=request.semantic_profile.semantic_profile_hash,
        prompt_id=request.rendered_prompt.prompt_id,
        prompt_version=request.rendered_prompt.prompt_version,
        prompt_content_hash=request.rendered_prompt.prompt_content_hash,
        rendered_prompt_hash=request.rendered_prompt.rendered_prompt_hash,
        output_schema_id=request.output_schema.schema_id,
        output_schema_version=request.output_schema.schema_version,
        output_schema_hash=request.output_schema.schema_hash,
        request_hash=request.request_hash,
        provider_response_id=None,
        finish_reason=None,
        usage=None,
    )
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


def build_source_document(paragraphs: dict[str, str] | None = None) -> SourceDocument:
    paragraphs = paragraphs or PARAGRAPHS
    chapter = SourceChapter(
        CHAPTER_ID,
        None,
        "synthetic",
        tuple(SourceParagraph(pid, paragraphs[pid], None) for pid in sorted(paragraphs)),
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
    request: StructuredGenerationRequest
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
    request = overrides.get("request", make_request())
    provenance = overrides.get(
        "provenance", provenance_from_request(request)
    )
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
        request=request,
        provenance=provenance,
        payload=payload,
    )


# -- two-phase flow helpers --------------------------------------------------


def publish(h: Harness, **overrides):
    """Post-generation publish (``publish_validated``)."""
    return h.service.publish_validated(
        source_document=overrides.get("source_document", h.source_document),
        source_document_ref=overrides.get("source_document_ref", h.source_document_ref),
        source_chunk=overrides.get("source_chunk", h.source_chunk),
        source_chunk_ref=overrides.get("source_chunk_ref", h.source_chunk_ref),
        chunk_profile_id=overrides.get("chunk_profile_id", CHUNK_PROFILE_ID),
        extraction_profile=overrides.get("extraction_profile", h.profile),
        generation_provenance=overrides.get("generation_provenance", h.provenance),
        payload=overrides.get("payload", h.payload),
    )


def reuse(h: Harness, *, request: StructuredGenerationRequest, **overrides):
    """Pre-generation reuse check (``try_reuse_current``)."""
    return h.service.try_reuse_current(
        project_id=overrides.get("project_id", PROJECT),
        document_id=overrides.get("document_id", DOCUMENT),
        chunk_profile_id=overrides.get("chunk_profile_id", CHUNK_PROFILE_ID),
        chunk_id=overrides.get("chunk_id", CHUNK_ID),
        source_document_ref=overrides.get("source_document_ref", h.source_document_ref),
        source_chunk_ref=overrides.get("source_chunk_ref", h.source_chunk_ref),
        extraction_profile=overrides.get("extraction_profile", h.profile),
        structured_request=request,
    )


def run(h: Harness, *, request: StructuredGenerationRequest, **overrides):
    """The A3D two-phase flow: pre-generation reuse, then post-generation
    publish on a miss. On publish, the provenance is derived from the request
    (as a real provider call would produce), unless overridden."""
    res = reuse(h, request=request, **overrides)
    if res is not None:
        return res
    overrides.setdefault("generation_provenance", provenance_from_request(request))
    return publish(h, **overrides)


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


# -- identity / path helpers -------------------------------------------------


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


def store_path(
    store: FileArtifactStore, artifact_type: str, artifact_id: str, revision: int
) -> Path:
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


def _pass_report(h: Harness, ref: ArtifactRef, **overrides) -> ValidationReport:
    return ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", h.source_chunk_ref),
            LineageRef("candidate_extraction", ref),
        ),
        findings=(),
    )


def _publish_current(h: Harness, extraction: CandidateExtraction, ref: ArtifactRef):
    """Persist a PASS report + point CURRENT at an already-persisted extraction."""
    persist_validation_report(
        h.store,
        _pass_report(h, ref),
        artifact_id=validation_artifact_id(),
        revision=ref.revision,
    )
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=ref,
    )


def _current_target_ref(h: Harness) -> ArtifactRef | None:
    """The CURRENT target ref, or None if no pointer exists yet."""
    try:
        return h.pointers.resolve_current(pointer_id()).target_ref
    except PointerNotFoundError:
        return None


def _alt_source_document(h: Harness) -> tuple[SourceDocument, ArtifactRef]:
    """Persist a *real* alternate SourceDocument revision (different content)."""
    alt_doc = build_source_document(
        paragraphs={
            "CH001_P0001": "不同的左上下文第一段。",
            "CH001_P0002": "不同的左上下文第二段。",
            "CH001_P0003": "不同的林晚走进教室。",
            "CH001_P0004": "不同的老师正在板书。",
            "CH001_P0005": "不同的右上下文第一段。",
            "CH001_P0006": "不同的右上下文第二段。",
        }
    )
    ref = persist_source_document(h.store, alt_doc, revision=2)
    return alt_doc, ref


def _alt_source_pair(
    h: Harness,
) -> tuple[SourceDocument, ArtifactRef, SourceChunk, ArtifactRef]:
    """A coherent alternate source pair: an alternate SourceDocument revision +
    an alternate SourceChunk revision that pins exactly that document ref.
    Coherent lineage is required for a legitimate (non-incoherent) source
    semantic change."""
    alt_doc, alt_doc_ref = _alt_source_document(h)
    alt_chunk = build_source_chunk(alt_doc_ref)
    alt_chunk_ref = persist_source_chunk(
        h.store, alt_chunk, profile_id=CHUNK_PROFILE_ID, revision=2
    )
    return alt_doc, alt_doc_ref, alt_chunk, alt_chunk_ref


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
# Persistence
# ---------------------------------------------------------------------------


def test_first_valid_publish(tmp_path):
    h = make_harness(tmp_path)
    result = publish(h)

    assert isinstance(result, CandidateExtractionPublication)
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 1
    assert result.validation_report_ref.revision == 1
    loaded = load_candidate_extraction(h.store, result.candidate_extraction_ref)
    assert loaded.candidates == h.payload
    report = load_validation_report(h.store, result.validation_report_ref)
    assert {(ref.role, ref.artifact_ref) for ref in report.validated_refs} == {
        ("source_document", h.source_document_ref),
        ("source_chunk", h.source_chunk_ref),
        ("candidate_extraction", result.candidate_extraction_ref),
    }
    assert report.summary.result is ValidationResult.PASS
    pointer = h.pointers.resolve_current(pointer_id())
    assert pointer.pointer_kind is PointerKind.CURRENT
    assert pointer.target_ref == result.candidate_extraction_ref


def test_typed_loader_round_trip(tmp_path):
    h = make_harness(tmp_path)
    result = publish(h)
    loaded = load_candidate_extraction(h.store, result.candidate_extraction_ref)
    assert loaded == make_extraction(h, payload=canonical_payload())
    assert loaded.candidates == h.payload


def test_bad_artifact_type_fails(tmp_path):
    h = make_harness(tmp_path)
    result = publish(h)
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
    result = publish(h)
    path = store_path(h.store, "candidate_extraction", extraction_artifact_id(), 1)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["candidates"]["characters"][0]["summary_zh"] = "被篡改的摘要。"
    path.write_bytes(canonical_json_bytes(data))
    with pytest.raises(StoryIntegrityError):
        load_candidate_extraction(h.store, result.candidate_extraction_ref)


# ---------------------------------------------------------------------------
# Exact current-only reuse (pre-generation API)
# ---------------------------------------------------------------------------


def test_identical_rerun_reuses_same_ref(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    second = reuse(h, request=h.request)
    assert second is not None
    assert second.reused is True
    assert second.candidate_extraction_ref == first.candidate_extraction_ref
    assert second.validation_report_ref == first.validation_report_ref
    assert second.current_pointer_ref == first.current_pointer_ref


def test_identical_rerun_does_not_allocate_new_revision(tmp_path):
    h = make_harness(tmp_path)
    publish(h)
    second = reuse(h, request=h.request)
    assert second is not None and second.reused is True
    assert second.candidate_extraction_ref.revision == 1
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_exact_matching_pass_report_required(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    second = reuse(h, request=h.request)
    assert second is not None and second.reused is True
    report = load_validation_report(h.store, second.validation_report_ref)
    assert report == load_validation_report(h.store, first.validation_report_ref)
    assert report.summary.result is ValidationResult.PASS


def test_missing_report_prevents_reuse_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h)
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=ref,
    )
    with pytest.raises(StoryIntegrityError, match="ValidationReport is missing"):
        reuse(h, request=h.request)


def test_mismatched_report_lineage_prevents_reuse(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h)
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
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
        h.store, bad_report, artifact_id=validation_artifact_id(), revision=1
    )
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=ref,
    )
    with pytest.raises(StoryIntegrityError, match="ValidationReport"):
        reuse(h, request=h.request)


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
        reuse(h, request=h.request)


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
        reuse(h, request=h.request)


def test_current_pointer_change_during_reuse_fails_closed(tmp_path, monkeypatch):
    h = make_harness(tmp_path)
    publish(h)

    original = h.pointers.resolve_current_pointer_ref
    calls = {"n": 0}

    def flaky_resolve(pointer_id):
        calls["n"] += 1
        if calls["n"] >= 3:  # the re-check call after initial resolution
            return ArtifactRef("current_pointer", pointer_id, 999, "f" * 64)
        return original(pointer_id)

    monkeypatch.setattr(h.pointers, "resolve_current_pointer_ref", flaky_resolve)
    with pytest.raises(
        StoryPersistenceError, match="changed during reuse verification"
    ):
        reuse(h, request=h.request)


def test_reuse_rejects_noncanonical_persisted_payload(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h, payload=noncanonical_payload())
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    _publish_current(h, extraction, ref)
    with pytest.raises(StoryIntegrityError, match="canonical form"):
        reuse(h, request=h.request)


# ---------------------------------------------------------------------------
# Identity invalidation (every frozen field)
# ---------------------------------------------------------------------------


def _publish_then_check_miss(h: Harness, *, request, **overrides):
    publish(h)
    result = reuse(h, request=request, **overrides)
    assert result is None
    # a normal miss must not allocate a new revision
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_source_document_ref_change_invalidates(tmp_path):
    """A coherent alternate source pair (real revisions, chunk pins the exact
    alternate document) is a legitimate source semantic change -> normal miss."""
    h = make_harness(tmp_path)
    _, alt_doc_ref, alt_chunk, alt_chunk_ref = _alt_source_pair(h)
    _publish_then_check_miss(
        h,
        request=h.request,
        source_document_ref=alt_doc_ref,
        source_chunk_ref=alt_chunk_ref,
    )


def test_source_chunk_ref_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    _, alt_doc_ref, alt_chunk, alt_chunk_ref = _alt_source_pair(h)
    _publish_then_check_miss(
        h,
        request=h.request,
        source_document_ref=alt_doc_ref,
        source_chunk_ref=alt_chunk_ref,
    )


def test_incoherent_source_pair_fails_closed_not_miss(tmp_path):
    """Changing only source_document_ref while the SourceChunk still pins the
    old document is structurally incoherent -> fail closed, not a normal miss."""
    h = make_harness(tmp_path)
    _, alt_doc_ref = _alt_source_document(h)
    # the (base) source chunk still pins the base document, so this pair is
    # incoherent (chunk.source_document_ref != requested source_document_ref)
    with pytest.raises(StoryIntegrityError, match="source_document_ref"):
        reuse(h, request=h.request, source_document_ref=alt_doc_ref)


def test_chunk_profile_id_is_part_of_identity():
    request = make_request()
    doc_ref = ArtifactRef("source_document", "classroom.src_001", 1, "f" * 64)
    chunk_ref = ArtifactRef(
        "source_chunk", "classroom.src_001.story-analysis-v1.ch001_c001", 1, "e" * 64
    )
    profile = make_profile()
    base = CandidateExtraction(
        schema_version=1,
        project_id=PROJECT,
        document_id=DOCUMENT,
        chunk_profile_id=CHUNK_PROFILE_ID,
        chunk_id=CHUNK_ID,
        source_document_ref=doc_ref,
        source_chunk_ref=chunk_ref,
        extraction_profile_id=profile.profile_id,
        extraction_profile_hash=profile.profile_hash,
        generation_provenance=provenance_from_request(request),
        candidates=canonical_payload(),
    )
    other = replace(base, chunk_profile_id="story-analysis-v2")
    assert extraction_semantic_identity(base) != extraction_semantic_identity(other)
    # and the artifact identity reflects the chunk_profile_id
    assert (
        candidate_extraction_artifact_id(
            PROJECT, DOCUMENT, "story-analysis-v2", CHUNK_ID, EXTRACTION_PROFILE_ID
        )
        != extraction_artifact_id()
    )


def test_extraction_profile_id_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    alt_profile = make_profile(profile_id="story-extraction-v2")
    _publish_then_check_miss(
        h,
        request=h.request,
        extraction_profile=alt_profile,
        chunk_profile_id=CHUNK_PROFILE_ID,
    )


def test_extraction_profile_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    alt_profile = make_profile(working_language="en-US")
    assert alt_profile.profile_hash != h.profile.profile_hash
    _publish_then_check_miss(
        h, request=h.request, extraction_profile=alt_profile
    )


def test_semantic_profile_id_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    _publish_then_check_miss(
        h, request=make_request(semantic_profile_id="story-llm-qwen-v2")
    )


def test_semantic_profile_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    # changing the semantic profile material changes semantic_profile_hash
    _publish_then_check_miss(h, request=make_request(temperature=0.5))


def test_prompt_id_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    _publish_then_check_miss(h, request=make_request(prompt_id="a3.other-prompt"))


def test_prompt_version_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    _publish_then_check_miss(h, request=make_request(prompt_version=2))


def test_prompt_content_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    _publish_then_check_miss(h, request=make_request(prompt_content_hash="f" * 64))


def test_rendered_prompt_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    # changing the rendered user text changes rendered_prompt_hash
    _publish_then_check_miss(h, request=make_request(user_text="Chunk: 完全不同的文本。"))


def test_output_schema_id_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    alt_schema = OutputSchema.create(
        schema_id="a3-candidate-payload-v2", schema_version=1, schema={"type": "object"}
    )
    _publish_then_check_miss(h, request=make_request(output_schema=alt_schema))


def test_output_schema_version_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    alt_schema = OutputSchema.create(
        schema_id="a3-candidate-payload", schema_version=2, schema={"type": "object"}
    )
    _publish_then_check_miss(h, request=make_request(output_schema=alt_schema))


def test_output_schema_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    alt_schema = OutputSchema.create(
        schema_id="a3-candidate-payload",
        schema_version=1,
        schema={"type": "object", "additionalProperties": False},
    )
    assert alt_schema.schema_hash != _default_output_schema().schema_hash
    _publish_then_check_miss(h, request=make_request(output_schema=alt_schema))


def test_request_hash_change_invalidates(tmp_path):
    h = make_harness(tmp_path)
    # changing the model changes request_hash (and semantic_profile_hash)
    _publish_then_check_miss(h, request=make_request(model="qwen3-9b"))


def test_non_semantic_provider_metadata_change_does_not_invalidate(tmp_path):
    h = make_harness(tmp_path)
    first = publish(
        h,
        generation_provenance=provenance_from_request(
            h.request,
            provider_response_id="resp-123",
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 7},
        ),
    )
    # the pre-generation request carries none of this metadata, and it is not
    # part of the reuse identity -> exact current is reused.
    result = reuse(h, request=h.request)
    assert result is not None
    assert result.reused is True
    assert result.candidate_extraction_ref == first.candidate_extraction_ref


def test_request_semantic_fields_exclude_non_semantic_metadata():
    base = make_request()
    p1 = provenance_from_request(
        base, provider_response_id="a", finish_reason="stop", usage={"x": 1}
    )
    p2 = provenance_from_request(
        base, provider_response_id="b", finish_reason="length", usage={"y": 2}
    )
    assert request_semantic_fields(p1) == request_semantic_fields(p2)
    # a genuine semantic change is captured by the semantic profile / request hash
    assert request_semantic_fields(p1) != request_semantic_fields(
        make_request(temperature=0.5)
    )


# ---------------------------------------------------------------------------
# Supersession / history
# ---------------------------------------------------------------------------


def test_stale_identity_creates_new_revision_same_artifact_id(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    second = run(h, request=make_request(model="qwen3-9b"))
    assert first.candidate_extraction_ref.revision == 1
    assert second.candidate_extraction_ref.revision == 2
    assert (
        first.candidate_extraction_ref.artifact_id
        == second.candidate_extraction_ref.artifact_id
        == extraction_artifact_id()
    )
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        second.candidate_extraction_ref
    )
    assert load_candidate_extraction(h.store, first.candidate_extraction_ref) == (
        make_extraction(h)
    )


def test_source_semantic_change_supersedes_with_real_alternates(tmp_path):
    """A legitimate source/chunk semantic change (real alternate immutable
    revisions with coherent lineage) prevents reuse and supersedes."""
    h = make_harness(tmp_path)
    first = publish(h)
    alt_doc, alt_doc_ref = _alt_source_document(h)
    alt_chunk = build_source_chunk(alt_doc_ref)
    alt_chunk_ref = persist_source_chunk(
        h.store, alt_chunk, profile_id=CHUNK_PROFILE_ID, revision=2
    )
    # pre-generation reuse with the alternate source refs is a normal miss
    assert (
        reuse(
            h,
            request=h.request,
            source_document_ref=alt_doc_ref,
            source_chunk_ref=alt_chunk_ref,
        )
        is None
    )
    # post-generation publish with the coherent alternate pair supersedes
    second = publish(
        h,
        source_document=alt_doc,
        source_document_ref=alt_doc_ref,
        source_chunk=alt_chunk,
        source_chunk_ref=alt_chunk_ref,
    )
    assert second.candidate_extraction_ref.revision == 2
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        second.candidate_extraction_ref
    )
    # first revision remains exactly resolvable (history preserved)
    assert load_candidate_extraction(h.store, first.candidate_extraction_ref) == (
        make_extraction(h)
    )


def test_no_historical_auto_resurrection(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    run(h, request=make_request(model="qwen3-9b"))  # -> revision 2
    # Request the OLD (historical) identity again: it publishes a NEW revision,
    # it does not resurrect the historical revision 1.
    result = run(h, request=h.request)
    assert result.reused is False
    assert result.candidate_extraction_ref.revision == 3
    assert result.candidate_extraction_ref != first.candidate_extraction_ref
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        result.candidate_extraction_ref
    )
    assert load_candidate_extraction(h.store, first.candidate_extraction_ref) == (
        make_extraction(h)
    )


# ---------------------------------------------------------------------------
# Failed candidate validation
# ---------------------------------------------------------------------------


def _invalid_payload() -> CandidatePayload:
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
    first = publish(h)
    # a DIFFERENT semantic identity + invalid payload: the publish fails closed
    # and never replaces the valid CURRENT.
    with pytest.raises(StoryIntegrityError, match="cannot become current"):
        publish(
            h,
            payload=_invalid_payload(),
            generation_provenance=provenance_from_request(
                make_request(model="qwen3-9b")
            ),
        )
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_invalid_candidate_same_identity_reuses_not_republishes(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    # same identity + valid current is reused before any (invalid) publish
    result = run(h, request=h.request, payload=_invalid_payload())
    assert result.reused is True
    assert result.candidate_extraction_ref == first.candidate_extraction_ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_invalid_payload_not_persisted_as_canonical(tmp_path):
    h = make_harness(tmp_path)
    with pytest.raises(StoryIntegrityError, match="cannot become current"):
        publish(h, payload=_invalid_payload())
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)


# ---------------------------------------------------------------------------
# Blocker 1 — exact source ArtifactRef lineage
# ---------------------------------------------------------------------------


def test_forged_source_document_ref_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    forged = ArtifactRef(
        "source_document",
        h.source_document_ref.artifact_id,
        h.source_document_ref.revision,
        "9" * 64,  # wrong content hash -> exact resolution must reject
    )
    with pytest.raises(StoryIntegrityError, match="failed to resolve SourceDocument"):
        publish(h, source_document_ref=forged)
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)
    # no CURRENT was created
    assert _current_target_ref(h) is None


def test_forged_source_chunk_ref_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    forged = ArtifactRef(
        "source_chunk",
        h.source_chunk_ref.artifact_id,
        h.source_chunk_ref.revision,
        "9" * 64,
    )
    with pytest.raises(StoryIntegrityError, match="failed to resolve SourceChunk"):
        publish(h, source_chunk_ref=forged)
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)
    assert _current_target_ref(h) is None


def test_source_chunk_lineage_mismatch_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    # a coherent alternate document the chunk does NOT point at
    alt_doc, alt_doc_ref = _alt_source_document(h)
    with pytest.raises(
        StoryIntegrityError, match="SourceChunk.source_document_ref"
    ):
        publish(
            h,
            source_document=alt_doc,
            source_document_ref=alt_doc_ref,
        )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)


def test_in_memory_source_document_mismatch_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    # ref resolves to the persisted document; the supplied object differs
    alt_doc = build_source_document(
        paragraphs={**PARAGRAPHS, "CH001_P0003": "不同的林晚走进教室。"}
    )
    with pytest.raises(
        StoryIntegrityError, match="does not exactly match.*source_document_ref"
    ):
        publish(h, source_document=alt_doc)
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)


def test_in_memory_source_chunk_mismatch_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    # ref resolves to the persisted chunk; the supplied object differs
    alt_chunk = replace(build_source_chunk(h.source_document_ref), ownership_token_count=99)
    with pytest.raises(
        StoryIntegrityError, match="does not exactly match.*source_chunk_ref"
    ):
        publish(h, source_chunk=alt_chunk)
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)


# ---------------------------------------------------------------------------
# Blocker 2 — a corrupt CURRENT fails closed even when identity is stale
# ---------------------------------------------------------------------------


def _changed_request() -> StructuredGenerationRequest:
    """A request whose semantic identity deliberately differs from the base."""
    return make_request(model="qwen3-9b")


def test_current_missing_report_changed_identity_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    store_path(
        h.store, "validation_report", validation_artifact_id(), 1
    ).unlink()
    with pytest.raises(StoryIntegrityError, match="ValidationReport is missing"):
        run(h, request=_changed_request())
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_current_mismatched_report_changed_identity_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    # overwrite the exact report with one carrying a wrong lineage ref
    wrong_chunk_ref = ArtifactRef(
        "source_chunk", h.source_chunk_ref.artifact_id, 1, "9" * 64
    )
    bad_report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", wrong_chunk_ref),
            LineageRef("candidate_extraction", first.candidate_extraction_ref),
        ),
        findings=(),
    )
    _overwrite_report(h, first.candidate_extraction_ref, bad_report)
    with pytest.raises(StoryIntegrityError, match="ValidationReport"):
        run(h, request=_changed_request())
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_current_fail_report_changed_identity_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    bad_report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", h.source_chunk_ref),
            LineageRef("candidate_extraction", first.candidate_extraction_ref),
        ),
        findings=(_finding(ValidationSeverity.BLOCKING, "A3_FAIL"),),
    )
    _overwrite_report(h, first.candidate_extraction_ref, bad_report)
    with pytest.raises(StoryIntegrityError):
        run(h, request=_changed_request())
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_current_noncanonical_changed_identity_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h, payload=noncanonical_payload())
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    _publish_current(h, extraction, ref)
    with pytest.raises(StoryIntegrityError, match="canonical form"):
        run(h, request=_changed_request())
    assert h.pointers.resolve_current(pointer_id()).target_ref == ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_current_semantic_invalid_changed_identity_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h, payload=_invalid_payload())
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    _publish_current(h, extraction, ref)
    with pytest.raises(StoryIntegrityError):
        run(h, request=_changed_request())
    assert h.pointers.resolve_current(pointer_id()).target_ref == ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_current_wrong_logical_target_changed_identity_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    # a valid CandidateExtraction for a *different* logical identity (chunk_id)
    other = replace(make_extraction(h), chunk_id="CH001_C002")
    other_ref = persist_candidate_extraction(h.store, other, revision=1)
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=other_ref,
    )
    with pytest.raises(
        StoryIntegrityError, match="different logical CandidateExtraction"
    ):
        run(h, request=_changed_request())
    # the (wrong) pointer is left untouched; nothing is published for ours
    assert h.pointers.resolve_current(pointer_id()).target_ref == other_ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)


# ---------------------------------------------------------------------------
# Blocker 1 — publish-time CURRENT re-verification (provider-call TOCTOU)
#
# These call ``publish_validated()`` DIRECTLY (without ``try_reuse_current``
# catching the problem first) to prove the CURRENT integrity invariant is
# upheld at the actual publication boundary.
# ---------------------------------------------------------------------------


def test_publish_current_missing_report_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    store_path(h.store, "validation_report", validation_artifact_id(), 1).unlink()
    with pytest.raises(StoryIntegrityError, match="ValidationReport is missing"):
        publish(h)
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_publish_current_mismatched_report_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    wrong_chunk_ref = ArtifactRef(
        "source_chunk", h.source_chunk_ref.artifact_id, 1, "9" * 64
    )
    bad_report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", wrong_chunk_ref),
            LineageRef("candidate_extraction", first.candidate_extraction_ref),
        ),
        findings=(),
    )
    _overwrite_report(h, first.candidate_extraction_ref, bad_report)
    with pytest.raises(StoryIntegrityError, match="ValidationReport"):
        publish(h)
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_publish_current_non_pass_report_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    bad_report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", h.source_document_ref),
            LineageRef("source_chunk", h.source_chunk_ref),
            LineageRef("candidate_extraction", first.candidate_extraction_ref),
        ),
        findings=(_finding(ValidationSeverity.BLOCKING, "A3_FAIL"),),
    )
    _overwrite_report(h, first.candidate_extraction_ref, bad_report)
    with pytest.raises(StoryIntegrityError):
        publish(h)
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_publish_current_semantic_invalid_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h, payload=_invalid_payload())
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    _publish_current(h, extraction, ref)
    with pytest.raises(StoryIntegrityError):
        publish(h)
    assert h.pointers.resolve_current(pointer_id()).target_ref == ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_publish_current_noncanonical_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    extraction = make_extraction(h, payload=noncanonical_payload())
    ref = persist_candidate_extraction(h.store, extraction, revision=1)
    _publish_current(h, extraction, ref)
    with pytest.raises(StoryIntegrityError, match="canonical form"):
        publish(h)
    assert h.pointers.resolve_current(pointer_id()).target_ref == ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_publish_current_wrong_logical_target_fails_closed(tmp_path):
    h = make_harness(tmp_path)
    other = replace(make_extraction(h), chunk_id="CH001_C002")
    other_ref = persist_candidate_extraction(h.store, other, revision=1)
    h.pointers.compare_and_set(
        pointer_id=pointer_id(),
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=other_ref,
    )
    with pytest.raises(
        StoryIntegrityError, match="different logical CandidateExtraction"
    ):
        publish(h)
    assert h.pointers.resolve_current(pointer_id()).target_ref == other_ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 1)


def test_publish_same_identity_race_reuses_not_republishes(tmp_path):
    """Provider-call same-identity race: worker B already published this exact
    post-generation semantic identity; worker A's direct ``publish_validated``
    must reuse the validated CURRENT (no new immutable revision)."""
    h = make_harness(tmp_path)
    first = publish(h)  # worker B: rev 1, base identity
    # worker A: direct publish of the SAME identity (no try_reuse_current)
    second = publish(h)
    assert second.reused is True
    assert second.candidate_extraction_ref == first.candidate_extraction_ref
    assert second.validation_report_ref == first.validation_report_ref
    assert second.current_pointer_ref == first.current_pointer_ref
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


# ---------------------------------------------------------------------------
# Blocker 2 — typed A1/A2 loader reuse (no second, weaker source loader)
# ---------------------------------------------------------------------------


def test_a3c_rejects_source_with_unsupported_envelope_schema(tmp_path):
    """A3C must not accept a source artifact the authoritative A1/A2 typed
    loader rejects: an immutable envelope with an unsupported
    ``envelope.schema_version`` but a valid v1 ``payload.schema_version``."""
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)
    doc = build_source_document()
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type="source_document",
        artifact_id=source_document_artifact_id(PROJECT, DOCUMENT),
        revision=1,
        schema_version=999,  # unsupported envelope schema_version
        payload=doc.to_dict(),  # valid v1 payload
    )
    bad_doc_ref = store.put(envelope)
    chunk = build_source_chunk(bad_doc_ref)
    chunk_ref = persist_source_chunk(
        store, chunk, profile_id=CHUNK_PROFILE_ID, revision=1
    )
    service = CandidateExtractionService(store, pointers)
    with pytest.raises(
        StoryIntegrityError, match="unsupported SourceDocument schema_version"
    ):
        service.try_reuse_current(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile_id=CHUNK_PROFILE_ID,
            chunk_id=CHUNK_ID,
            source_document_ref=bad_doc_ref,
            source_chunk_ref=chunk_ref,
            extraction_profile=make_profile(),
            structured_request=make_request(),
        )


def test_a3c_rejects_source_chunk_with_unsupported_envelope_schema(tmp_path):
    """Equivalent for SourceChunk: an immutable envelope with an unsupported
    ``envelope.schema_version`` is rejected by the A1/A2 typed loader."""
    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)
    doc = build_source_document()
    doc_ref = persist_source_document(store, doc, revision=1)
    chunk = build_source_chunk(doc_ref)
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type="source_chunk",
        artifact_id=source_chunk_artifact_id(
            PROJECT, DOCUMENT, CHUNK_PROFILE_ID, CHUNK_ID
        ),
        revision=1,
        schema_version=999,  # unsupported envelope schema_version
        payload=chunk.to_dict(),  # valid v1 payload
    )
    bad_chunk_ref = store.put(envelope)
    service = CandidateExtractionService(store, pointers)
    with pytest.raises(
        StoryIntegrityError, match="unsupported SourceChunk schema_version"
    ):
        service.try_reuse_current(
            project_id=PROJECT,
            document_id=DOCUMENT,
            chunk_profile_id=CHUNK_PROFILE_ID,
            chunk_id=CHUNK_ID,
            source_document_ref=doc_ref,
            source_chunk_ref=bad_chunk_ref,
            extraction_profile=make_profile(),
            structured_request=make_request(),
        )


# ---------------------------------------------------------------------------
# Blocker 3 — pre-generation reuse before the provider call
# ---------------------------------------------------------------------------


def test_pre_generation_reuse_before_llm(tmp_path):
    """Publish with real post-generation provenance, then reuse pre-generation
    using only the A-I3 request identity (no payload, no new provenance, no
    LLM call)."""
    h = make_harness(tmp_path)
    first = publish(h)  # uses h.provenance (matches h.request)
    assert first.reused is False

    # Build the exact same pre-generation A-I3 request identity and reuse it.
    reused = reuse(h, request=h.request)
    assert reused is not None
    assert reused.reused is True
    assert reused.candidate_extraction_ref == first.candidate_extraction_ref
    assert reused.validation_report_ref == first.validation_report_ref
    # no provider call happened: still exactly one immutable revision
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_pre_generation_reuse_miss_on_changed_request(tmp_path):
    h = make_harness(tmp_path)
    first = publish(h)
    # a changed request identity is a normal miss (no historical scan)
    result = reuse(h, request=make_request(model="qwen3-9b"))
    assert result is None
    assert h.pointers.resolve_current(pointer_id()).target_ref == (
        first.candidate_extraction_ref
    )
    with pytest.raises(Exception):
        h.store.get("candidate_extraction", extraction_artifact_id(), 2)


def test_pre_generation_reuse_no_current_is_miss(tmp_path):
    h = make_harness(tmp_path)
    # nothing published yet
    assert reuse(h, request=h.request) is None


# ---------------------------------------------------------------------------
# Helpers used by Blocker 2 tests
# ---------------------------------------------------------------------------


def _overwrite_report(h: Harness, extraction_ref: ArtifactRef, report: ValidationReport):
    """Overwrite the exact validation-report artifact bytes with a forged one.

    The store's content-hash verification still passes (we recompute the
    envelope's own content hash), but the *report* no longer matches the exact
    deterministic validation result for the CURRENT extraction.
    """
    from short_drama.foundation import validation_report_envelope

    envelope = validation_report_envelope(
        report,
        artifact_id=validation_artifact_id(),
        revision=extraction_ref.revision,
    )
    path = store_path(
        h.store, "validation_report", validation_artifact_id(), extraction_ref.revision
    )
    path.write_bytes(canonical_json_bytes(envelope.to_dict()))
