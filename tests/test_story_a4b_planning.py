"""A4B deterministic indexing + blocking tests.

Covers the frozen A4B contract:

  * snapshot/lineage coherence (exact order, fail closed, real artifact IDs)
  * candidate index (coverage, globalized refs, source_order_key)
  * name normalization v1 (NFKC, casefold, strip, whitespace collapse, idempotent)
  * strong/weak identity keys
  * blocking (exact key, token overlap, adjacent chunk, distant no-signal absent)
  * pair state (auto_same, must_not_merge, needs_semantic_decision)
  * must-not-merge v1 (production empty set, synthetic override, canonicalization)
  * no whole-document N² (exact pair count)
  * plan hash determinism
  * coverage audit
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.llm import LLMInvocationProvenance
from short_drama.story import (
    BLOCKING_POLICY_ID,
    CANONICALIZATION_POLICY_ID,
    NAME_NORMALIZATION_POLICY_ID,
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_MUST_NOT_MERGE,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    SIGNAL_ADJACENT_CHUNK,
    SIGNAL_EXACT_IDENTITY_KEY,
    SIGNAL_HARD_MUST_NOT_MERGE,
    SIGNAL_IDENTITY_TOKEN_OVERLAP,
    ChunkManifest,
    ChunkPlanningProfile,
    CandidateEntityIndex,
    CandidateExtraction,
    CandidatePayload,
    CharacterCandidate,
    ChunkCoverage,
    EvidenceRef,
    LocationCandidate,
    ParagraphSpan,
    ReconciliationInputSnapshot,
    ReconciliationPlanningError,
    SourceChunk,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    SourceChapter,
    UnresolvedMentionCandidate,
    derive_must_not_merge_constraints,
    effective_blocking_tokens,
    extract_blocking_tokens,
    extract_identity_keys,
    is_strong_identity_key,
    normalize_name,
    plan_reconciliation,
)
from short_drama.story.chunking import (
    CHUNK_PLANNER_VERSION,
    TOKEN_COUNTER_ID,
)
from short_drama.story.extraction_persistence import (
    candidate_extraction_artifact_id,
    candidate_extraction_validation_artifact_id,
)
from short_drama.story.persistence import (
    source_chunk_artifact_id,
    source_document_artifact_id,
)
from short_drama.story.source import NormalizationInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H = "a" * 64
H2 = "b" * 64
H3 = "c" * 64
H4 = "d" * 64
H5 = "e" * 64

PROJECT_ID = "proj1"
DOCUMENT_ID = "doc1"
PROFILE_ID = "a2-default"
EXTRACTION_PROFILE_ID = "a3-extraction-v1"


# ---------------------------------------------------------------------------
# Production-realistic ArtifactRef builders
# ---------------------------------------------------------------------------


def make_source_document_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_type="source_document",
        artifact_id=source_document_artifact_id(PROJECT_ID, DOCUMENT_ID),
        revision=1,
        content_hash=H,
    )


def make_source_chunk_ref(chunk_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_type="source_chunk",
        artifact_id=source_chunk_artifact_id(
            PROJECT_ID, DOCUMENT_ID, PROFILE_ID, chunk_id
        ),
        revision=1,
        content_hash=H2,
    )


def make_extraction_ref(chunk_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_type="candidate_extraction",
        artifact_id=candidate_extraction_artifact_id(
            PROJECT_ID, DOCUMENT_ID, PROFILE_ID, chunk_id, EXTRACTION_PROFILE_ID
        ),
        revision=1,
        content_hash=H3,
    )


def make_validation_report_ref(chunk_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_type="validation_report",
        artifact_id=candidate_extraction_validation_artifact_id(
            candidate_extraction_artifact_id(
                PROJECT_ID, DOCUMENT_ID, PROFILE_ID, chunk_id, EXTRACTION_PROFILE_ID
            )
        ),
        revision=1,
        content_hash=H4,
    )


def make_fake_artifact_ref(
    artifact_type: str = "source_chunk",
    artifact_id: str = "CH001_C001",
) -> ArtifactRef:
    """Create a ref with a non-production artifact_id (for negative tests)."""
    return ArtifactRef(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        revision=1,
        content_hash=H2,
    )


# ---------------------------------------------------------------------------
# Domain object builders
# ---------------------------------------------------------------------------


def make_provenance() -> LLMInvocationProvenance:
    return LLMInvocationProvenance(
        provider_family="llama-cpp",
        model="qwen2.5-7b",
        semantic_profile_id=EXTRACTION_PROFILE_ID,
        semantic_profile_hash=H,
        prompt_id="a3.extract-chunk",
        prompt_version=1,
        prompt_content_hash=H2,
        rendered_prompt_hash=H3,
        output_schema_id="a3.chunk-extraction",
        output_schema_version=1,
        output_schema_hash=H4,
        request_hash=H5,
        provider_response_id="resp-001",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 50},
    )


def make_source_document(num_paragraphs: int = 10) -> SourceDocument:
    paragraphs = tuple(
        SourceParagraph(
            paragraph_id=f"CH001_P{i:04d}",
            text_original=f"Paragraph {i}",
        )
        for i in range(1, num_paragraphs + 1)
    )
    chapter = SourceChapter(
        chapter_id="CH001",
        title_original=None,
        heading_kind="synthetic",
        paragraphs=paragraphs,
    )
    source = SourceInfo(
        type="txt",
        path="test.txt",
        raw_sha256=H,
        byte_size=100,
        declared_language="zh",
        detected_language="zh",
        language_detector="langid-1.1.6",
    )
    normalization = NormalizationInfo(
        input_encoding="utf-8",
        newline="LF",
        parser_id="short_drama_source_ingestion_v1",
        parser_version="1",
    )
    return SourceDocument(
        schema_version=1,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        source=source,
        normalization=normalization,
        chapters=(chapter,),
    )


def make_source_chunk(
    chunk_id: str = "CH001_C001",
    source_document_ref: ArtifactRef | None = None,
    paragraph_ids: tuple[str, ...] | None = None,
) -> SourceChunk:
    if source_document_ref is None:
        source_document_ref = make_source_document_ref()
    if paragraph_ids is None:
        paragraph_ids = ("CH001_P0001", "CH001_P0002")
    return SourceChunk(
        schema_version=1,
        chunk_id=chunk_id,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        chapter_id="CH001",
        source_document_ref=source_document_ref,
        context_span=ParagraphSpan(start=paragraph_ids[0], end=paragraph_ids[-1]),
        ownership_span=ParagraphSpan(start=paragraph_ids[0], end=paragraph_ids[-1]),
        paragraph_ids=paragraph_ids,
        token_count_method=TOKEN_COUNTER_ID,
        context_token_count=10,
        ownership_token_count=5,
    )


def make_chunk_manifest(
    chunk_refs: tuple[ArtifactRef, ...],
    source_document_ref: ArtifactRef | None = None,
) -> ChunkManifest:
    if source_document_ref is None:
        source_document_ref = make_source_document_ref()
    profile = ChunkPlanningProfile(
        schema_version=1,
        profile_id=PROFILE_ID,
        token_counter=TOKEN_COUNTER_ID,
        ownership_token_budget=100,
        context_overlap_token_budget=20,
        context_token_budget=200,
    )
    return ChunkManifest(
        schema_version=1,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        source_document_ref=source_document_ref,
        planner_version=CHUNK_PLANNER_VERSION,
        profile=profile,
        chunk_refs=chunk_refs,
        chunk_count=len(chunk_refs),
        coverage=ChunkCoverage(
            paragraphs_total=10, owned_once=10, unowned=0, multiply_owned=0
        ),
        state="CHUNKING_COMPLETE",
    )


def make_evidence(paragraph_id: str = "CH001_P0001") -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            paragraph_id=paragraph_id,
            role="primary",
            strength="explicit",
            excerpt="some text",
        ),
    )


def make_character(
    candidate_id: str = "cand_char_001",
    display_name_original: str = "John Smith",
    aliases_original: tuple[str, ...] = (),
    evidence: tuple[EvidenceRef, ...] | None = None,
) -> CharacterCandidate:
    if evidence is None:
        evidence = make_evidence()
    return CharacterCandidate(
        candidate_id=candidate_id,
        display_name_original=display_name_original,
        aliases_original=aliases_original,
        descriptors_zh=(),
        summary_zh="test character",
        evidence_strength="explicit",
        evidence=evidence,
    )


def make_location(
    candidate_id: str = "cand_loc_001",
    display_name_original: str = "北京",
    aliases_original: tuple[str, ...] = (),
    evidence: tuple[EvidenceRef, ...] | None = None,
) -> LocationCandidate:
    if evidence is None:
        evidence = make_evidence()
    return LocationCandidate(
        candidate_id=candidate_id,
        display_name_original=display_name_original,
        aliases_original=aliases_original,
        descriptors_zh=(),
        summary_zh="test location",
        evidence_strength="explicit",
        evidence=evidence,
    )


def make_unresolved(
    candidate_id: str = "cand_unres_001",
    mention_original: str = "那个人",
    mention_kind: str = "person",
    possible_candidate_refs: tuple[str, ...] = (),
    evidence: tuple[EvidenceRef, ...] | None = None,
) -> UnresolvedMentionCandidate:
    if evidence is None:
        evidence = make_evidence()
    return UnresolvedMentionCandidate(
        candidate_id=candidate_id,
        mention_original=mention_original,
        mention_kind=mention_kind,
        reason_zh="不确定",
        possible_candidate_refs=possible_candidate_refs,
        evidence_strength="uncertain",
        evidence=evidence,
    )


def make_extraction(
    chunk_id: str = "CH001_C001",
    source_document_ref: ArtifactRef | None = None,
    source_chunk_ref: ArtifactRef | None = None,
    extraction_profile_id: str = EXTRACTION_PROFILE_ID,
    extraction_profile_hash: str = H,
    characters: tuple[CharacterCandidate, ...] = (),
    locations: tuple[LocationCandidate, ...] = (),
    unresolved: tuple[UnresolvedMentionCandidate, ...] = (),
    facts: tuple = (),
    events: tuple = (),
    relationships: tuple = (),
) -> CandidateExtraction:
    if source_document_ref is None:
        source_document_ref = make_source_document_ref()
    if source_chunk_ref is None:
        source_chunk_ref = make_source_chunk_ref(chunk_id)
    candidates = CandidatePayload(
        characters=characters,
        locations=locations,
        facts=facts,
        events=events,
        relationships=relationships,
        unresolved_mentions=unresolved,
    )
    return CandidateExtraction(
        schema_version=1,
        project_id=PROJECT_ID,
        document_id=DOCUMENT_ID,
        chunk_profile_id=PROFILE_ID,
        chunk_id=chunk_id,
        source_document_ref=source_document_ref,
        source_chunk_ref=source_chunk_ref,
        extraction_profile_id=extraction_profile_id,
        extraction_profile_hash=extraction_profile_hash,
        generation_provenance=make_provenance(),
        candidates=candidates,
    )


def build_snapshot(
    chunk_ids: list[str] | None = None,
    *,
    characters_per_chunk: int = 0,
    locations_per_chunk: int = 0,
    unresolved_per_chunk: int = 0,
    display_names: list[str] | None = None,
) -> ReconciliationInputSnapshot:
    """Build a valid snapshot with production-realistic refs."""
    if chunk_ids is None:
        num_chunks = max(1, characters_per_chunk, locations_per_chunk, unresolved_per_chunk)
        if num_chunks <= 1:
            num_chunks = 1
        chunk_ids = [f"CH001_C{c+1:03d}" for c in range(num_chunks)]

    source_doc = make_source_document(num_paragraphs=len(chunk_ids) * 5 + 5)
    source_doc_ref = make_source_document_ref()

    chunk_refs = []
    source_chunks = []
    extractions = []
    extraction_refs = []
    report_refs = []

    for c, chunk_id in enumerate(chunk_ids):
        para_start = c * 5 + 1
        para_ids = tuple(
            f"CH001_P{i:04d}" for i in range(para_start, para_start + 5)
        )
        chunk_ref = make_source_chunk_ref(chunk_id)
        chunk = make_source_chunk(
            chunk_id=chunk_id,
            source_document_ref=source_doc_ref,
            paragraph_ids=para_ids,
        )
        chunk_refs.append(chunk_ref)
        source_chunks.append(chunk)

        chars = tuple(
            make_character(
                candidate_id=f"cand_char_{i+1:03d}",
                display_name_original=(
                    display_names[c * 2 + i]
                    if display_names and c * 2 + i < len(display_names)
                    else f"Character{c}_{i}"
                ),
                evidence=make_evidence(para_ids[0]),
            )
            for i in range(characters_per_chunk)
        )
        locs = tuple(
            make_location(
                candidate_id=f"cand_loc_{i+1:03d}",
                display_name_original=f"Location{c}_{i}",
                evidence=make_evidence(para_ids[0]),
            )
            for i in range(locations_per_chunk)
        )
        unres = tuple(
            make_unresolved(
                candidate_id=f"cand_unres_{i+1:03d}",
                evidence=make_evidence(para_ids[0]),
            )
            for i in range(unresolved_per_chunk)
        )
        ext = make_extraction(
            chunk_id=chunk_id,
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=chars,
            locations=locs,
            unresolved=unres,
        )
        extractions.append(ext)
        extraction_refs.append(make_extraction_ref(chunk_id))
        report_refs.append(make_validation_report_ref(chunk_id))

    manifest = make_chunk_manifest(
        chunk_refs=tuple(chunk_refs),
        source_document_ref=source_doc_ref,
    )

    return ReconciliationInputSnapshot(
        source_document=source_doc,
        source_document_ref=source_doc_ref,
        chunk_manifest=manifest,
        source_chunks=tuple(source_chunks),
        source_chunk_refs=tuple(chunk_refs),
        candidate_extractions=tuple(extractions),
        candidate_extraction_refs=tuple(extraction_refs),
        a3_validation_report_refs=tuple(report_refs),
    )


def build_single_chunk_snapshot(
    ext: CandidateExtraction,
    chunk_id: str = "CH001_C001",
) -> ReconciliationInputSnapshot:
    """Build a single-chunk snapshot for quick testing."""
    source_doc = make_source_document(num_paragraphs=10)
    source_doc_ref = make_source_document_ref()
    chunk_ref = make_source_chunk_ref(chunk_id)
    chunk = make_source_chunk(
        chunk_id=chunk_id,
        source_document_ref=source_doc_ref,
    )
    manifest = make_chunk_manifest(
        chunk_refs=(chunk_ref,),
        source_document_ref=source_doc_ref,
    )
    return ReconciliationInputSnapshot(
        source_document=source_doc,
        source_document_ref=source_doc_ref,
        chunk_manifest=manifest,
        source_chunks=(chunk,),
        source_chunk_refs=(chunk_ref,),
        candidate_extractions=(ext,),
        candidate_extraction_refs=(make_extraction_ref(chunk_id),),
        a3_validation_report_refs=(make_validation_report_ref(chunk_id),),
    )


# ===========================================================================
# Normalization tests
# ===========================================================================


class TestNormalization:
    def test_nfkc(self):
        assert normalize_name("\ufb01le") == "file"

    def test_casefold(self):
        assert normalize_name("John") == "john"
        assert normalize_name("JOHN") == "john"
        assert normalize_name("Straße") == "strasse"

    def test_strip(self):
        assert normalize_name("  John  ") == "john"
        assert normalize_name("\tJohn\n") == "john"

    def test_whitespace_collapse(self):
        assert normalize_name("John   Smith") == "john smith"
        assert normalize_name("John\tSmith") == "john smith"
        assert normalize_name("John \n Smith") == "john smith"

    def test_punctuation_retained(self):
        assert normalize_name("Mr. Smith") == "mr. smith"
        assert normalize_name("Dr. Smith") == "dr. smith"

    def test_idempotent(self):
        test_strings = [
            "John Smith",
            "  \tJohn   Smith  ",
            "Straße",
            "\ufb01le",
            "林晚",
            "Mr. Smith",
        ]
        for s in test_strings:
            once = normalize_name(s)
            twice = normalize_name(once)
            assert once == twice, f"not idempotent for {s!r}: {once!r} != {twice!r}"

    def test_chinese(self):
        assert normalize_name("林晚") == "林晚"
        assert normalize_name(" 林晚 ") == "林晚"


# ===========================================================================
# Identity key tests
# ===========================================================================


class TestIdentityKeys:
    def test_basic(self):
        keys = extract_identity_keys("John Smith", ())
        assert keys == ("john smith",)

    def test_with_aliases(self):
        keys = extract_identity_keys("John Smith", ("Johnny", "John S."))
        assert keys == ("john s.", "john smith", "johnny")

    def test_deduplication(self):
        keys = extract_identity_keys("John Smith", ("john smith",))
        assert keys == ("john smith",)

    def test_empty_after_normalization(self):
        keys = extract_identity_keys("   ", ("  ",))
        assert keys == ()

    def test_chinese(self):
        keys = extract_identity_keys("林晚", ("小林",))
        assert keys == ("小林", "林晚")


# ===========================================================================
# Blocking token tests
# ===========================================================================


class TestBlockingTokens:
    def test_english(self):
        tokens = extract_blocking_tokens(("john smith",))
        assert tokens == ("john", "smith")

    def test_mr_smith(self):
        tokens = extract_blocking_tokens(("mr. smith",))
        assert "mr" in tokens
        assert "smith" in tokens

    def test_chinese(self):
        tokens = extract_blocking_tokens(("林晚",))
        assert "林晚" in tokens

    def test_pure_numeric_excluded(self):
        tokens = extract_blocking_tokens(("123",))
        assert tokens == ()

    def test_short_token_excluded(self):
        tokens = extract_blocking_tokens(("a",))
        assert tokens == ()

    def test_unique_sorted(self):
        tokens = extract_blocking_tokens(("john smith", "smith john"))
        assert tokens == ("john", "smith")
        assert len(tokens) == len(set(tokens))

    def test_underscore_splits(self):
        """John_Smith → ("john", "smith"), not ("john_smith",)."""
        tokens = extract_blocking_tokens(("john_smith",))
        assert tokens == ("john", "smith")

    def test_mixed_separators(self):
        """Various non-alphanumeric separators all split."""
        tokens = extract_blocking_tokens(("john-smith",))
        assert tokens == ("john", "smith")
        tokens = extract_blocking_tokens(("john.smith",))
        assert tokens == ("john", "smith")


# ===========================================================================
# Strong/weak identity key tests
# ===========================================================================


class TestStrongWeakKeys:
    def test_john_weak(self):
        assert not is_strong_identity_key("john")

    def test_alice_weak(self):
        assert not is_strong_identity_key("alice")

    def test_john_smith_strong(self):
        assert is_strong_identity_key("john smith")

    def test_lin_wan_weak(self):
        assert not is_strong_identity_key("林晚")

    def test_lin_wan_wan_strong(self):
        assert is_strong_identity_key("林晚晚")

    def test_role_weak(self):
        assert not is_strong_identity_key("teacher")

    def test_the_teacher_weak(self):
        assert not is_strong_identity_key("the teacher")

    def test_zh_role_weak(self):
        assert not is_strong_identity_key("老师")
        assert not is_strong_identity_key("医生")
        assert not is_strong_identity_key("班主任")

    def test_empty_weak(self):
        assert not is_strong_identity_key("")

    def test_pure_numeric_weak(self):
        assert not is_strong_identity_key("12345")

    def test_three_cjk_non_generic_strong(self):
        assert is_strong_identity_key("张三丰")


# ===========================================================================
# Must-not-merge v1 tests
# ===========================================================================


def _make_index_entry(ref: str, kind: str, display: str = "Test") -> Any:
    from short_drama.story import CandidateEntityIndexEntry

    return CandidateEntityIndexEntry(
        candidate_ref=ref,
        candidate_kind=kind,
        candidate_extraction_ref=make_extraction_ref("CH001_C001"),
        source_order_key="000001:000000001:01:000000001:" + ref,
        display_name_original=display,
        aliases_original=(),
        descriptors_zh=(),
        evidence_refs=make_evidence(),
        possible_candidate_refs=(),
    )


class TestMustNotMerge:
    def test_production_empty(self):
        entry = _make_index_entry("CH001_C001:cand_char_001", "character")
        index = CandidateEntityIndex(schema_version=1, entries=(entry,))
        result = derive_must_not_merge_constraints(index)
        assert result == frozenset()

    def test_event_co_participation_not_hard(self):
        e1 = _make_index_entry("CH001_C001:cand_char_001", "character")
        e2 = _make_index_entry("CH001_C001:cand_char_002", "character")
        index = CandidateEntityIndex(schema_version=1, entries=(e1, e2))
        result = derive_must_not_merge_constraints(index)
        assert result == frozenset()

    def test_relationship_endpoints_not_hard(self):
        e1 = _make_index_entry("CH001_C001:cand_char_001", "character")
        e2 = _make_index_entry("CH001_C001:cand_char_002", "character")
        index = CandidateEntityIndex(schema_version=1, entries=(e1, e2))
        result = derive_must_not_merge_constraints(index)
        assert result == frozenset()


# ===========================================================================
# Snapshot coherence tests
# ===========================================================================


class TestSnapshotCoherence:
    def test_valid_snapshot_accepted(self):
        snapshot = build_snapshot(
            ["CH001_C001", "CH001_C002"], characters_per_chunk=1
        )
        result = plan_reconciliation(snapshot)
        assert len(result.candidate_index.entries) == 2

    def test_extraction_count_mismatch(self):
        """Fewer extractions than chunks → fail closed."""
        snapshot = build_snapshot(["CH001_C001"], characters_per_chunk=1)
        # Create a manifest with 2 chunks but only 1 extraction
        source_doc_ref = make_source_document_ref()
        chunk_ref2 = make_source_chunk_ref("CH001_C002")
        manifest = make_chunk_manifest(
            chunk_refs=(snapshot.source_chunk_refs[0], chunk_ref2),
            source_document_ref=source_doc_ref,
        )
        bad_snapshot = ReconciliationInputSnapshot(
            source_document=snapshot.source_document,
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=(snapshot.source_chunks[0],
                           make_source_chunk(chunk_id="CH001_C002", source_document_ref=source_doc_ref)),
            source_chunk_refs=(snapshot.source_chunk_refs[0], chunk_ref2),
            candidate_extractions=snapshot.candidate_extractions,  # only 1
            candidate_extraction_refs=snapshot.candidate_extraction_refs,
            a3_validation_report_refs=snapshot.a3_validation_report_refs,
        )
        with pytest.raises(ReconciliationPlanningError, match="len"):
            plan_reconciliation(bad_snapshot)

    def test_source_ref_mismatch(self):
        """Wrong source_document_ref in manifest vs snapshot → fail closed."""
        snapshot = build_snapshot(["CH001_C001"], characters_per_chunk=1)
        bad_ref = ArtifactRef(
            artifact_type="source_document",
            artifact_id="proj_OTHER.doc_OTHER",
            revision=1,
            content_hash=H,
        )
        bad_snapshot = ReconciliationInputSnapshot(
            source_document=snapshot.source_document,
            source_document_ref=bad_ref,
            chunk_manifest=snapshot.chunk_manifest,
            source_chunks=snapshot.source_chunks,
            source_chunk_refs=snapshot.source_chunk_refs,
            candidate_extractions=snapshot.candidate_extractions,
            candidate_extraction_refs=snapshot.candidate_extraction_refs,
            a3_validation_report_refs=snapshot.a3_validation_report_refs,
        )
        with pytest.raises(ReconciliationPlanningError):
            plan_reconciliation(bad_snapshot)

    def test_source_chunk_bare_id_rejected(self):
        """A source_chunk ref with bare CH001_C001 artifact_id (not production
        format) is rejected."""
        source_doc_ref = make_source_document_ref()
        bad_chunk_ref = make_fake_artifact_ref(
            artifact_type="source_chunk",
            artifact_id="CH001_C001",  # bare, not production format
        )
        chunk = make_source_chunk(
            chunk_id="CH001_C001", source_document_ref=source_doc_ref
        )
        manifest = ChunkManifest(
            schema_version=1,
            project_id=PROJECT_ID,
            document_id=DOCUMENT_ID,
            source_document_ref=source_doc_ref,
            planner_version=CHUNK_PLANNER_VERSION,
            profile=ChunkPlanningProfile(
                schema_version=1,
                profile_id=PROFILE_ID,
                token_counter=TOKEN_COUNTER_ID,
                ownership_token_budget=100,
                context_overlap_token_budget=20,
                context_token_budget=200,
            ),
            chunk_refs=(bad_chunk_ref,),
            chunk_count=1,
            coverage=ChunkCoverage(
                paragraphs_total=10, owned_once=10, unowned=0, multiply_owned=0
            ),
            state="CHUNKING_COMPLETE",
        )
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=bad_chunk_ref,
            characters=(make_character(),),
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=make_source_document(),
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=(chunk,),
            source_chunk_refs=(bad_chunk_ref,),
            candidate_extractions=(ext,),
            candidate_extraction_refs=(make_extraction_ref("CH001_C001"),),
            a3_validation_report_refs=(make_validation_report_ref("CH001_C001"),),
        )
        with pytest.raises(ReconciliationPlanningError, match="artifact_id"):
            plan_reconciliation(snapshot)

    def test_wrong_chunk_profile_artifact_id_rejected(self):
        """A source_chunk ref with wrong profile in artifact_id is rejected."""
        source_doc_ref = make_source_document_ref()
        # Use a different profile_id in the artifact_id
        bad_chunk_ref = ArtifactRef(
            artifact_type="source_chunk",
            artifact_id=f"{PROJECT_ID}.{DOCUMENT_ID}.wrong-profile.ch001_c001",
            revision=1,
            content_hash=H2,
        )
        chunk = make_source_chunk(
            chunk_id="CH001_C001", source_document_ref=source_doc_ref
        )
        manifest = ChunkManifest(
            schema_version=1,
            project_id=PROJECT_ID,
            document_id=DOCUMENT_ID,
            source_document_ref=source_doc_ref,
            planner_version=CHUNK_PLANNER_VERSION,
            profile=ChunkPlanningProfile(
                schema_version=1,
                profile_id=PROFILE_ID,
                token_counter=TOKEN_COUNTER_ID,
                ownership_token_budget=100,
                context_overlap_token_budget=20,
                context_token_budget=200,
            ),
            chunk_refs=(bad_chunk_ref,),
            chunk_count=1,
            coverage=ChunkCoverage(
                paragraphs_total=10, owned_once=10, unowned=0, multiply_owned=0
            ),
            state="CHUNKING_COMPLETE",
        )
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=bad_chunk_ref,
            characters=(make_character(),),
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=make_source_document(),
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=(chunk,),
            source_chunk_refs=(bad_chunk_ref,),
            candidate_extractions=(ext,),
            candidate_extraction_refs=(make_extraction_ref("CH001_C001"),),
            a3_validation_report_refs=(make_validation_report_ref("CH001_C001"),),
        )
        with pytest.raises(ReconciliationPlanningError, match="artifact_id"):
            plan_reconciliation(snapshot)

    def test_chunk_profile_mismatch_rejected(self):
        """ext.chunk_profile_id != manifest.profile.profile_id → rejected."""
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        chunk = make_source_chunk(
            chunk_id="CH001_C001", source_document_ref=source_doc_ref
        )
        manifest = make_chunk_manifest(
            chunk_refs=(chunk_ref,), source_document_ref=source_doc_ref
        )
        # Extraction with wrong chunk_profile_id
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(make_character(),),
        )
        # Override chunk_profile_id via a new extraction
        from dataclasses import replace
        ext_bad = CandidateExtraction(
            schema_version=1,
            project_id=PROJECT_ID,
            document_id=DOCUMENT_ID,
            chunk_profile_id="wrong-profile",  # mismatch!
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            extraction_profile_id=EXTRACTION_PROFILE_ID,
            extraction_profile_hash=H,
            generation_provenance=make_provenance(),
            candidates=CandidatePayload(characters=(make_character(),)),
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=make_source_document(),
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=(chunk,),
            source_chunk_refs=(chunk_ref,),
            candidate_extractions=(ext_bad,),
            candidate_extraction_refs=(make_extraction_ref("CH001_C001"),),
            a3_validation_report_refs=(make_validation_report_ref("CH001_C001"),),
        )
        with pytest.raises(ReconciliationPlanningError, match="chunk_profile_id"):
            plan_reconciliation(snapshot)

    def test_extraction_artifact_id_mismatch_rejected(self):
        """CandidateExtraction ref with wrong artifact_id → rejected."""
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        chunk = make_source_chunk(
            chunk_id="CH001_C001", source_document_ref=source_doc_ref
        )
        manifest = make_chunk_manifest(
            chunk_refs=(chunk_ref,), source_document_ref=source_doc_ref
        )
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(make_character(),),
        )
        # Use a wrong artifact_id for the extraction ref
        bad_ext_ref = ArtifactRef(
            artifact_type="candidate_extraction",
            artifact_id="wrong.artifact.id",
            revision=1,
            content_hash=H3,
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=make_source_document(),
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=(chunk,),
            source_chunk_refs=(chunk_ref,),
            candidate_extractions=(ext,),
            candidate_extraction_refs=(bad_ext_ref,),
            a3_validation_report_refs=(make_validation_report_ref("CH001_C001"),),
        )
        with pytest.raises(ReconciliationPlanningError, match="artifact_id"):
            plan_reconciliation(snapshot)

    def test_swapped_extraction_refs_rejected(self):
        """Swapped CandidateExtraction refs (position mismatch) → rejected."""
        snapshot = build_snapshot(
            ["CH001_C001", "CH001_C002"], characters_per_chunk=1
        )
        # Swap the extraction refs
        swapped_refs = (
            snapshot.candidate_extraction_refs[1],
            snapshot.candidate_extraction_refs[0],
        )
        bad_snapshot = ReconciliationInputSnapshot(
            source_document=snapshot.source_document,
            source_document_ref=snapshot.source_document_ref,
            chunk_manifest=snapshot.chunk_manifest,
            source_chunks=snapshot.source_chunks,
            source_chunk_refs=snapshot.source_chunk_refs,
            candidate_extractions=snapshot.candidate_extractions,
            candidate_extraction_refs=swapped_refs,
            a3_validation_report_refs=snapshot.a3_validation_report_refs,
        )
        with pytest.raises(ReconciliationPlanningError, match="artifact_id"):
            plan_reconciliation(bad_snapshot)

    def test_manifest_order_mismatch(self):
        """Source chunk refs not matching manifest order → rejected."""
        snapshot = build_snapshot(
            ["CH001_C001", "CH001_C002"], characters_per_chunk=1
        )
        swapped_refs = (
            snapshot.source_chunk_refs[1],
            snapshot.source_chunk_refs[0],
        )
        bad_snapshot = ReconciliationInputSnapshot(
            source_document=snapshot.source_document,
            source_document_ref=snapshot.source_document_ref,
            chunk_manifest=snapshot.chunk_manifest,
            source_chunks=snapshot.source_chunks,
            source_chunk_refs=swapped_refs,
            candidate_extractions=snapshot.candidate_extractions,
            candidate_extraction_refs=snapshot.candidate_extraction_refs,
            a3_validation_report_refs=snapshot.a3_validation_report_refs,
        )
        with pytest.raises(ReconciliationPlanningError, match="chunk_refs"):
            plan_reconciliation(bad_snapshot)

    def test_extraction_profile_mismatch(self):
        """Different extraction_profile_id across extractions → rejected."""
        source_doc_ref = make_source_document_ref()
        chunk_ref1 = make_source_chunk_ref("CH001_C001")
        chunk_ref2 = make_source_chunk_ref("CH001_C002")
        chunk1 = make_source_chunk(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            paragraph_ids=tuple(f"CH001_P{i:04d}" for i in range(1, 6)),
        )
        chunk2 = make_source_chunk(
            chunk_id="CH001_C002",
            source_document_ref=source_doc_ref,
            paragraph_ids=tuple(f"CH001_P{i:04d}" for i in range(6, 11)),
        )
        manifest = make_chunk_manifest(
            chunk_refs=(chunk_ref1, chunk_ref2), source_document_ref=source_doc_ref
        )
        ext1 = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref1,
            characters=(make_character(),),
        )
        ext2 = make_extraction(
            chunk_id="CH001_C002",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref2,
            characters=(make_character(),),
            extraction_profile_id="different-profile",
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=make_source_document(num_paragraphs=15),
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=(chunk1, chunk2),
            source_chunk_refs=(chunk_ref1, chunk_ref2),
            candidate_extractions=(ext1, ext2),
            candidate_extraction_refs=(
                make_extraction_ref("CH001_C001"),
                make_extraction_ref("CH001_C002"),
            ),
            a3_validation_report_refs=(
                make_validation_report_ref("CH001_C001"),
                make_validation_report_ref("CH001_C002"),
            ),
        )
        with pytest.raises(ReconciliationPlanningError, match="profile"):
            plan_reconciliation(snapshot)


# ===========================================================================
# Candidate index tests
# ===========================================================================


class TestCandidateIndex:
    def test_all_kinds_indexed(self):
        snapshot = build_snapshot(
            ["CH001_C001"],
            characters_per_chunk=2,
            locations_per_chunk=1,
            unresolved_per_chunk=1,
        )
        result = plan_reconciliation(snapshot)
        kinds = [e.candidate_kind for e in result.candidate_index.entries]
        assert kinds.count("character") == 2
        assert kinds.count("location") == 1
        assert sum(1 for k in kinds if k.startswith("unresolved")) == 1

    def test_unresolved_possible_refs_globalized(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(make_character(candidate_id="cand_char_001"),),
            unresolved=(
                make_unresolved(
                    possible_candidate_refs=("cand_char_001",),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        unres_entries = [
            e for e in result.candidate_index.entries
            if e.candidate_kind.startswith("unresolved_")
        ]
        assert len(unres_entries) == 1
        assert unres_entries[0].possible_candidate_refs == (
            "CH001_C001:cand_char_001",
        )

    def test_source_order_key_deterministic(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_002",
                    evidence=make_evidence("CH001_P0003"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        entry = result.candidate_index.entries[0]
        assert entry.source_order_key == (
            "000001:000000003:01:000000002:CH001_C001:cand_char_002"
        )

    def test_earliest_evidence_paragraph_controls_ordinal(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    evidence=(
                        EvidenceRef(
                            paragraph_id="CH001_P0005",
                            role="supporting",
                            strength="explicit",
                            excerpt="later",
                        ),
                        EvidenceRef(
                            paragraph_id="CH001_P0002",
                            role="primary",
                            strength="explicit",
                            excerpt="earlier",
                        ),
                    ),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        entry = result.candidate_index.entries[0]
        assert ":000000002:" in entry.source_order_key

    def test_entries_sorted_by_source_order_key(self):
        snapshot = build_snapshot(
            ["CH001_C001", "CH001_C002"], characters_per_chunk=1
        )
        result = plan_reconciliation(snapshot)
        keys = [e.source_order_key for e in result.candidate_index.entries]
        assert keys == sorted(keys)

    def test_evidence_paragraph_not_found_fails(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    evidence=make_evidence("CH001_P9999"),
                ),
            ),
        )
        # Use a source doc with only 3 paragraphs
        source_doc = make_source_document(num_paragraphs=3)
        chunk = make_source_chunk(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            paragraph_ids=("CH001_P0001", "CH001_P0002"),
        )
        manifest = make_chunk_manifest(
            chunk_refs=(chunk_ref,), source_document_ref=source_doc_ref
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=source_doc,
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=(chunk,),
            source_chunk_refs=(chunk_ref,),
            candidate_extractions=(ext,),
            candidate_extraction_refs=(make_extraction_ref("CH001_C001"),),
            a3_validation_report_refs=(make_validation_report_ref("CH001_C001"),),
        )
        with pytest.raises(ReconciliationPlanningError, match="paragraph"):
            plan_reconciliation(snapshot)


# ===========================================================================
# Blocking tests
# ===========================================================================


class TestBlocking:
    def test_exact_identity_key_produces_block(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert SIGNAL_EXACT_IDENTITY_KEY in plan.signals
        assert plan.shared_identity_keys == ("john smith",)

    def test_token_overlap_produces_block(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Williams",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert SIGNAL_IDENTITY_TOKEN_OVERLAP in plan.signals
        assert "john" in plan.shared_tokens

    def test_cross_chunk_adjacency_only_absent(self):
        """Characters in adjacent chunks with no exact key and no effective
        token overlap are NOT blocked (a4-blocking-v2: cross-chunk adjacency is
        supplemental, not generative → implicit not_compared)."""
        source_doc_ref = make_source_document_ref()
        chunk_ref1 = make_source_chunk_ref("CH001_C001")
        chunk_ref2 = make_source_chunk_ref("CH001_C002")
        ext1 = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref1,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="Alice",
                    evidence=make_evidence("CH001_P0001"),
                ),
            ),
        )
        ext2 = make_extraction(
            chunk_id="CH001_C002",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref2,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="Bob",
                    evidence=make_evidence("CH001_P0006"),
                ),
            ),
        )
        snapshot = build_snapshot(
            ["CH001_C001", "CH001_C002"], characters_per_chunk=0
        )
        # Override with our specific extractions
        source_doc = make_source_document(num_paragraphs=15)
        chunk1 = make_source_chunk(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            paragraph_ids=tuple(f"CH001_P{i:04d}" for i in range(1, 6)),
        )
        chunk2 = make_source_chunk(
            chunk_id="CH001_C002",
            source_document_ref=source_doc_ref,
            paragraph_ids=tuple(f"CH001_P{i:04d}" for i in range(6, 11)),
        )
        manifest = make_chunk_manifest(
            chunk_refs=(chunk_ref1, chunk_ref2), source_document_ref=source_doc_ref
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=source_doc,
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=(chunk1, chunk2),
            source_chunk_refs=(chunk_ref1, chunk_ref2),
            candidate_extractions=(ext1, ext2),
            candidate_extraction_refs=(
                make_extraction_ref("CH001_C001"),
                make_extraction_ref("CH001_C002"),
            ),
            a3_validation_report_refs=(
                make_validation_report_ref("CH001_C001"),
                make_validation_report_ref("CH001_C002"),
            ),
        )
        result = plan_reconciliation(snapshot)
        # a4-blocking-v2: cross-chunk adjacency alone is NOT generative, so
        # Alice (chunk 1) and Bob (chunk 2) with no exact key / no effective
        # token overlap produce NO explicit pair (implicit not_compared).
        assert len(result.pair_plans) == 0

    def test_distant_no_signal_absent(self):
        """Characters in chunks distance 2 apart with no shared signals → NOT blocked."""
        source_doc_ref = make_source_document_ref()
        chunk_ids = ["CH001_C001", "CH001_C002", "CH001_C003"]
        source_doc = make_source_document(num_paragraphs=20)
        chunk_refs = [make_source_chunk_ref(cid) for cid in chunk_ids]
        chunks = [
            make_source_chunk(
                chunk_id=cid,
                source_document_ref=source_doc_ref,
                paragraph_ids=tuple(
                    f"CH001_P{i:04d}" for i in range(c * 5 + 1, c * 5 + 6)
                ),
            )
            for c, cid in enumerate(chunk_ids)
        ]
        exts = []
        for c, (cid, cref) in enumerate(zip(chunk_ids, chunk_refs)):
            if c == 0:
                exts.append(make_extraction(
                    chunk_id=cid,
                    source_document_ref=source_doc_ref,
                    source_chunk_ref=cref,
                    characters=(make_character(
                        candidate_id="cand_char_001",
                        display_name_original="Alice",
                        evidence=make_evidence(f"CH001_P{c*5+1:04d}"),
                    ),),
                ))
            elif c == 2:
                exts.append(make_extraction(
                    chunk_id=cid,
                    source_document_ref=source_doc_ref,
                    source_chunk_ref=cref,
                    characters=(make_character(
                        candidate_id="cand_char_001",
                        display_name_original="Bob",
                        evidence=make_evidence(f"CH001_P{c*5+1:04d}"),
                    ),),
                ))
            else:
                exts.append(make_extraction(
                    chunk_id=cid,
                    source_document_ref=source_doc_ref,
                    source_chunk_ref=cref,
                ))
        manifest = make_chunk_manifest(
            chunk_refs=tuple(chunk_refs), source_document_ref=source_doc_ref
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=source_doc,
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=tuple(chunks),
            source_chunk_refs=tuple(chunk_refs),
            candidate_extractions=tuple(exts),
            candidate_extraction_refs=tuple(
                make_extraction_ref(cid) for cid in chunk_ids
            ),
            a3_validation_report_refs=tuple(
                make_validation_report_ref(cid) for cid in chunk_ids
            ),
        )
        result = plan_reconciliation(snapshot)
        # Alice (chunk 1) and Bob (chunk 3) are distance 2 apart → NOT blocked
        assert len(result.pair_plans) == 0

    def test_char_loc_cross_type_absent(self):
        """Character and location are never paired."""
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="Test",
                    evidence=make_evidence("CH001_P0001"),
                ),
            ),
            locations=(
                make_location(
                    candidate_id="cand_loc_001",
                    display_name_original="Test",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 0

    def test_unresolved_never_paired(self):
        """Unresolved candidates never enter pair plans."""
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
            unresolved=(
                make_unresolved(
                    candidate_id="cand_unres_001",
                    mention_original="John Smith",
                    evidence=make_evidence("CH001_P0003"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        for plan in result.pair_plans:
            assert "cand_unres" not in plan.left_candidate_ref
            assert "cand_unres" not in plan.right_candidate_ref

    def test_signals_deduplicated_sorted(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert plan.signals == tuple(sorted(plan.signals))
        assert len(plan.signals) == len(set(plan.signals))


# ===========================================================================
# a4-blocking-v2 tests (token exclusion + adjacency semantics)
# ===========================================================================


def _build_char_snapshot_by_chunk(
    chunk_order: list[str],
    chunk_to_names: dict[str, list[str]],
) -> ReconciliationInputSnapshot:
    """Build a multi-chunk snapshot with specific character names per chunk.

    Chunks are ordered by ``chunk_order``. Candidate ``i`` in a chunk uses the
    chunk's i-th paragraph as evidence. A chunk not present in ``chunk_to_names``
    (or with an empty list) yields an extraction with no candidates.
    """
    source_doc_ref = make_source_document_ref()
    source_doc = make_source_document(num_paragraphs=len(chunk_order) * 5 + 5)
    chunk_refs = [make_source_chunk_ref(cid) for cid in chunk_order]
    chunks = [
        make_source_chunk(
            chunk_id=cid,
            source_document_ref=source_doc_ref,
            paragraph_ids=tuple(
                f"CH001_P{i:04d}" for i in range(c * 5 + 1, c * 5 + 6)
            ),
        )
        for c, cid in enumerate(chunk_order)
    ]
    exts = []
    for c, (cid, cref) in enumerate(zip(chunk_order, chunk_refs)):
        para_ids = tuple(f"CH001_P{i:04d}" for i in range(c * 5 + 1, c * 5 + 6))
        names = chunk_to_names.get(cid, [])
        chars = tuple(
            make_character(
                candidate_id=f"cand_char_{i + 1:03d}",
                display_name_original=name,
                evidence=make_evidence(para_ids[i]),
            )
            for i, name in enumerate(names)
        )
        exts.append(
            make_extraction(
                chunk_id=cid,
                source_document_ref=source_doc_ref,
                source_chunk_ref=cref,
                characters=chars,
            )
        )
    manifest = make_chunk_manifest(
        chunk_refs=tuple(chunk_refs), source_document_ref=source_doc_ref
    )
    return ReconciliationInputSnapshot(
        source_document=source_doc,
        source_document_ref=source_doc_ref,
        chunk_manifest=manifest,
        source_chunks=tuple(chunks),
        source_chunk_refs=tuple(chunk_refs),
        candidate_extractions=tuple(exts),
        candidate_extraction_refs=tuple(
            make_extraction_ref(cid) for cid in chunk_order
        ),
        a3_validation_report_refs=tuple(
            make_validation_report_ref(cid) for cid in chunk_order
        ),
    )


class TestBlockingV2TokenExclusion:
    """Direct tests of the a4-blocking-v2 blocking-only minimal token filter."""

    def test_the_token_only_excluded(self):
        assert effective_blocking_tokens(("the",)) == ()

    def test_of_token_only_excluded(self):
        assert effective_blocking_tokens(("of",)) == ()

    def test_the_and_of_combined_excluded(self):
        assert effective_blocking_tokens(("the", "of")) == ()

    def test_meaningful_token_survives(self):
        assert effective_blocking_tokens(("the", "white", "rabbit")) == (
            "rabbit",
            "white",
        )

    def test_filter_deterministic_and_idempotent(self):
        raw = ("of", "the", "white", "rabbit", "white")
        once = effective_blocking_tokens(raw)
        twice = effective_blocking_tokens(once)
        assert once == ("rabbit", "white")
        assert once == twice
        # sorted + unique
        assert once == tuple(sorted(set(once)))

    def test_raw_extraction_unchanged(self):
        """extract_blocking_tokens() (raw) still emits 'the'/'of'; only the
        blocking seam filters them. This is what keeps strong-key classification
        (which reads raw tokens) stable."""
        assert extract_blocking_tokens(("the white rabbit",)) == (
            "rabbit",
            "the",
            "white",
        )

    def test_identity_keys_unchanged_by_filter(self):
        """The filter must not touch normalized identity keys."""
        assert extract_identity_keys("the", ()) == ("the",)
        assert extract_identity_keys("the White Rabbit", ()) == ("the white rabbit",)
        # ...yet it yields no effective blocking token for pure function-word key
        assert effective_blocking_tokens(extract_blocking_tokens(("the",))) == ()

    def test_strong_key_classification_unchanged_by_filter(self):
        """A 2-token key containing 'the' remains strong even though 'the' is in
        the exclusion set — strong/weak classification must not drift."""
        assert is_strong_identity_key("the smith") is True
        # raw tokens drive the classifier (2 tokens → strong)
        assert extract_blocking_tokens(("the smith",)) == ("smith", "the")
        # effective blocking tokens drop 'the' (1 token)
        assert effective_blocking_tokens(("smith", "the")) == ("smith",)


class TestBlockingV2PairGeneration:
    """End-to-end pair generation under a4-blocking-v2 semantics."""

    def test_the_token_only_does_not_create_lexical_block(self):
        # Shared raw token == {the} only, candidates in chunks distance 2 apart
        # (so no adjacency), no exact key → NO pair.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002", "CH001_C003"],
            {"CH001_C001": ["the Alpha"], "CH001_C003": ["the Beta"]},
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 0

    def test_of_token_only_does_not_create_lexical_block(self):
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002", "CH001_C003"],
            {"CH001_C001": ["of Alpha"], "CH001_C003": ["of Beta"]},
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 0

    def test_the_plus_of_token_only_does_not_create_lexical_block(self):
        # Shared raw tokens == {the, of} only → NO effective overlap.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002", "CH001_C003"],
            {
                "CH001_C001": ["the Alpha of Gamma"],
                "CH001_C003": ["the Beta of Delta"],
            },
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 0

    def test_meaningful_token_overlap_survives(self):
        # White Rabbit <-> the White Rabbit: effective shared tokens rabbit+white.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002", "CH001_C003"],
            {
                "CH001_C001": ["the White Rabbit"],
                "CH001_C003": ["White Rabbit"],
            },
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert SIGNAL_IDENTITY_TOKEN_OVERLAP in plan.signals
        # recorded shared_tokens are EFFECTIVE (no 'the'), deterministic + sorted
        assert plan.shared_tokens == ("rabbit", "white")
        assert "the" not in plan.shared_tokens
        assert "of" not in plan.shared_tokens
        # distance 2 → no supplemental adjacent_chunk
        assert SIGNAL_ADJACENT_CHUNK not in plan.signals

    def test_same_chunk_adjacency_only_survives(self):
        # No exact key, no effective token overlap, but same chunk → adjacency.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001"],
            {"CH001_C001": ["Zebra", "Yak"]},
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
        assert plan.signals == (SIGNAL_ADJACENT_CHUNK,)

    def test_cross_chunk_adjacency_only_absent(self):
        # Adjacent chunks, no exact key, no effective token overlap → NO pair.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["Zebra"], "CH001_C002": ["Yak"]},
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 0

    def test_cross_chunk_exact_key_pair_survives(self):
        # Adjacent chunks, exact strong key → pair exists with exact signal.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["John Smith"], "CH001_C002": ["John Smith"]},
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert SIGNAL_EXACT_IDENTITY_KEY in plan.signals
        assert plan.shared_identity_keys == ("john smith",)

    def test_cross_chunk_meaningful_token_pair_survives(self):
        # Adjacent chunks, effective token overlap (no exact key) → pair exists.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["White Rabbit"], "CH001_C002": ["the White Rabbit"]},
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert SIGNAL_IDENTITY_TOKEN_OVERLAP in plan.signals
        assert plan.shared_tokens == ("rabbit", "white")
        assert SIGNAL_EXACT_IDENTITY_KEY not in plan.signals

    def test_supplemental_adjacent_chunk_on_existing_exact_key_pair(self):
        # An already-existing cross-chunk exact-key pair in adjacent chunks gets
        # the supplemental adjacent_chunk deterministic signal.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["John Smith"], "CH001_C002": ["John Smith"]},
        )
        result = plan_reconciliation(snapshot)
        plan = result.pair_plans[0]
        assert SIGNAL_EXACT_IDENTITY_KEY in plan.signals
        assert SIGNAL_ADJACENT_CHUNK in plan.signals
        assert plan.signals == tuple(sorted(plan.signals))

    def test_supplemental_adjacent_chunk_on_existing_token_pair(self):
        # An already-existing cross-chunk token pair in adjacent chunks gets the
        # supplemental adjacent_chunk deterministic signal.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["White Rabbit"], "CH001_C002": ["the White Rabbit"]},
        )
        result = plan_reconciliation(snapshot)
        plan = result.pair_plans[0]
        assert SIGNAL_IDENTITY_TOKEN_OVERLAP in plan.signals
        assert SIGNAL_ADJACENT_CHUNK in plan.signals

    def test_same_chunk_adjacency_supplemental_not_duplicated(self):
        # A same-chunk pair already has adjacent_chunk from the generative window;
        # the supplemental step (distance 0) must not corrupt/duplicate signals.
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["Zebra", "Yak"], "CH001_C002": ["Fox"]},
        )
        result = plan_reconciliation(snapshot)
        # only the same-chunk (Zebra,Yak) pair exists; (Fox,*) is cross-chunk-only
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert plan.signals == (SIGNAL_ADJACENT_CHUNK,)


class TestBlockingV2UnchangedSemantics:
    """Confirm v2 does not disturb exact-key / strong-weak / auto_same / scope."""

    def test_exact_key_semantics_unchanged(self):
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001"],
            {"CH001_C001": ["John Smith", "John Smith"]},
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert SIGNAL_EXACT_IDENTITY_KEY in plan.signals
        assert plan.shared_identity_keys == ("john smith",)

    def test_auto_same_semantics_unchanged(self):
        # shared strong exact key → deterministic auto_same (unchanged by v2).
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001"],
            {"CH001_C001": ["John Smith", "John Smith"]},
        )
        result = plan_reconciliation(snapshot)
        plan = result.pair_plans[0]
        assert plan.state == PAIR_STATE_AUTO_SAME
        assert len(result.decisions) == 1
        assert result.decisions[0].reason_code == "same_strong_exact_identity_key"

    def test_weak_key_stays_semantic(self):
        # single-token exact key (weak) → needs_semantic_decision (unchanged).
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001"],
            {"CH001_C001": ["John", "John"]},
        )
        result = plan_reconciliation(snapshot)
        plan = result.pair_plans[0]
        assert plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION

    def test_char_location_still_absent(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="White Rabbit",
                    evidence=make_evidence("CH001_P0001"),
                ),
            ),
            locations=(
                make_location(
                    candidate_id="cand_loc_001",
                    display_name_original="White Rabbit",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 0

    def test_unresolved_still_absent(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="White Rabbit",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="White Rabbit",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
            unresolved=(
                make_unresolved(
                    candidate_id="cand_unres_001",
                    mention_original="White Rabbit",
                    evidence=make_evidence("CH001_P0003"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        for plan in result.pair_plans:
            assert "cand_unres" not in plan.left_candidate_ref
            assert "cand_unres" not in plan.right_candidate_ref


class TestBlockingV2DeterministicIdentity:
    """Policy identity + deterministic ordering/sorting/hash under v2."""

    def test_blocking_policy_id_is_v2(self):
        assert BLOCKING_POLICY_ID == "a4-blocking-v2"
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001"], {"CH001_C001": ["Zebra", "Yak"]}
        )
        result = plan_reconciliation(snapshot)
        assert result.blocking_policy_id == "a4-blocking-v2"

    def test_pair_ordering_deterministic(self):
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001"], {"CH001_C001": ["Zebra", "Yak", "Fox"]}
        )
        result1 = plan_reconciliation(snapshot)
        result2 = plan_reconciliation(snapshot)
        keys1 = [(p.left_candidate_ref, p.right_candidate_ref) for p in result1.pair_plans]
        keys2 = [(p.left_candidate_ref, p.right_candidate_ref) for p in result2.pair_plans]
        assert keys1 == keys2
        assert keys1 == sorted(keys1)

    def test_signals_deterministic_and_sorted(self):
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["White Rabbit"], "CH001_C002": ["the White Rabbit"]},
        )
        result = plan_reconciliation(snapshot)
        for plan in result.pair_plans:
            assert plan.signals == tuple(sorted(plan.signals))
            assert len(plan.signals) == len(set(plan.signals))

    def test_shared_tokens_deterministic_and_sorted(self):
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002", "CH001_C003"],
            {
                "CH001_C001": ["the White Rabbit"],
                "CH001_C003": ["White Rabbit"],
            },
        )
        result = plan_reconciliation(snapshot)
        plan = result.pair_plans[0]
        assert plan.shared_tokens == tuple(sorted(plan.shared_tokens))
        assert plan.shared_tokens == ("rabbit", "white")

    def test_plan_hash_deterministic(self):
        snapshot1 = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["White Rabbit"], "CH001_C002": ["the White Rabbit"]},
        )
        snapshot2 = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": ["White Rabbit"], "CH001_C002": ["the White Rabbit"]},
        )
        assert plan_reconciliation(snapshot1).plan_hash == plan_reconciliation(snapshot2).plan_hash

    def test_v2_identity_not_silently_equivalent_to_v1(self):
        """plan_hash material embeds the v2 blocking policy id; reconstructing
        the same material under a4-blocking-v1 yields a different hash."""
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001"], {"CH001_C001": ["Zebra", "Yak"]}
        )
        result = plan_reconciliation(snapshot)
        sorted_plans = sorted(
            result.pair_plans,
            key=lambda p: (p.left_candidate_ref, p.right_candidate_ref),
        )
        base = {
            "normalization_policy_id": result.normalization_policy_id,
            "canonicalization_policy_id": result.canonicalization_policy_id,
            "candidate_index": result.candidate_index.to_dict(),
            "pair_plans": [p.to_dict() for p in sorted_plans],
        }
        material_v2 = dict(base, blocking_policy_id="a4-blocking-v2")
        material_v1 = dict(base, blocking_policy_id="a4-blocking-v1")
        assert content_hash(material_v2) == result.plan_hash
        assert content_hash(material_v1) != result.plan_hash


class TestBlockingV2Complexity:
    """No whole-document N² / no cross-chunk Cartesian product under v2."""

    def test_no_cross_chunk_cartesian_product(self):
        """Two adjacent chunks, 3 distinct-name candidates each → exactly 6
        same-chunk pairs (2 × C(3,2)). A cross-chunk Cartesian product would add
        3×3 = 9 (total 15); a whole-document N² would give C(6,2) = 15 too."""
        names_a = ["Zebra", "Yak", "Wombat"]
        names_b = ["Koala", "Quokka", "Emu"]
        snapshot = _build_char_snapshot_by_chunk(
            ["CH001_C001", "CH001_C002"],
            {"CH001_C001": names_a, "CH001_C002": names_b},
        )
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 6
        # every pair is same-chunk
        for plan in result.pair_plans:
            left_chunk = plan.left_candidate_ref.split(":", 1)[0]
            right_chunk = plan.right_candidate_ref.split(":", 1)[0]
            assert left_chunk == right_chunk
            assert plan.signals == (SIGNAL_ADJACENT_CHUNK,)


# ===========================================================================
# Pair state tests
# ===========================================================================


class TestPairState:
    def test_strong_exact_overlap_auto_same(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        assert result.pair_plans[0].state == PAIR_STATE_AUTO_SAME
        assert len(result.decisions) == 1
        dec = result.decisions[0]
        assert dec.decision == "same_entity"
        assert dec.method == "deterministic"
        assert dec.reason_code == "same_strong_exact_identity_key"
        assert dec.prompt_id is None
        assert dec.prompt_version is None
        assert dec.generation_provenance is None

    def test_weak_exact_overlap_semantic(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        assert result.pair_plans[0].state == PAIR_STATE_NEEDS_SEMANTIC_DECISION

    def test_token_only_semantic(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Williams",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        assert result.pair_plans[0].state == PAIR_STATE_NEEDS_SEMANTIC_DECISION

    def test_adjacency_only_semantic(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="Alice",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="Bob",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.pair_plans) == 1
        assert result.pair_plans[0].state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
        assert result.pair_plans[0].signals == (SIGNAL_ADJACENT_CHUNK,)


# ===========================================================================
# Must-not-merge override tests
# ===========================================================================


class TestMustNotMergeOverride:
    def test_synthetic_hard_constraint_overrides_auto_same(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        hard = frozenset({
            ("CH001_C001:cand_char_001", "CH001_C001:cand_char_002"),
        })
        result = plan_reconciliation(snapshot, must_not_merge=hard)
        assert len(result.pair_plans) == 1
        plan = result.pair_plans[0]
        assert plan.state == PAIR_STATE_MUST_NOT_MERGE
        assert SIGNAL_HARD_MUST_NOT_MERGE in plan.signals
        assert len(result.decisions) == 1
        assert result.decisions[0].decision == "different_entity"
        assert result.decisions[0].reason_code == "hard_must_not_merge"

    def test_reversed_hard_constraint_canonicalized(self):
        """Reversed synthetic hard pair → still must_not_merge."""
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        # Reversed: right comes before left
        hard = frozenset({
            ("CH001_C001:cand_char_002", "CH001_C001:cand_char_001"),
        })
        result = plan_reconciliation(snapshot, must_not_merge=hard)
        assert len(result.pair_plans) == 1
        assert result.pair_plans[0].state == PAIR_STATE_MUST_NOT_MERGE

    def test_same_ref_hard_constraint_rejected(self):
        """left == right in hard constraint → rejected."""
        snapshot = build_snapshot(["CH001_C001"], characters_per_chunk=1)
        hard = frozenset({
            ("CH001_C001:cand_char_001", "CH001_C001:cand_char_001"),
        })
        with pytest.raises(ReconciliationPlanningError, match="distinct"):
            plan_reconciliation(snapshot, must_not_merge=hard)


# ===========================================================================
# No N² / exact adjacency count test
# ===========================================================================


class TestNoN2:
    def test_same_chunk_adjacency_only_pair_count(self):
        """5 chunks × 2 chars (all distinct names) → exactly 5 same-chunk
        adjacency pairs (a4-blocking-v2).

        Same chunk: 5 × C(2,2) = 5
        Cross-chunk adjacency: NOT generative in v2, so 0 cross-chunk pairs.
        Total: 5

        This proves the planner uses indexed/same-chunk-window generation (not a
        whole-document N² scan, and not a cross-chunk bucket Cartesian product):
        N² would give C(10,2) = 45 and the v1 cross-chunk window would give 21.
        """
        source_doc_ref = make_source_document_ref()
        num_chunks = 5
        chars_per_chunk = 2
        chunk_ids = [f"CH001_C{c+1:03d}" for c in range(num_chunks)]

        source_doc = make_source_document(num_paragraphs=num_chunks * 5 + 5)
        chunk_refs = [make_source_chunk_ref(cid) for cid in chunk_ids]
        chunks = [
            make_source_chunk(
                chunk_id=cid,
                source_document_ref=source_doc_ref,
                paragraph_ids=tuple(
                    f"CH001_P{i:04d}" for i in range(c * 5 + 1, c * 5 + 6)
                ),
            )
            for c, cid in enumerate(chunk_ids)
        ]
        exts = []
        for c, (cid, cref) in enumerate(zip(chunk_ids, chunk_refs)):
            para_id = f"CH001_P{c*5+1:04d}"
            exts.append(make_extraction(
                chunk_id=cid,
                source_document_ref=source_doc_ref,
                source_chunk_ref=cref,
                characters=tuple(
                    make_character(
                        candidate_id=f"cand_char_{i+1:03d}",
                        display_name_original=f"测试甲{c}乙{i}",
                        evidence=make_evidence(para_id),
                    )
                    for i in range(chars_per_chunk)
                ),
            ))
        manifest = make_chunk_manifest(
            chunk_refs=tuple(chunk_refs), source_document_ref=source_doc_ref
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=source_doc,
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=tuple(chunks),
            source_chunk_refs=tuple(chunk_refs),
            candidate_extractions=tuple(exts),
            candidate_extraction_refs=tuple(
                make_extraction_ref(cid) for cid in chunk_ids
            ),
            a3_validation_report_refs=tuple(
                make_validation_report_ref(cid) for cid in chunk_ids
            ),
        )
        result = plan_reconciliation(snapshot)

        # Exact count: 5 same-chunk only (v2: cross-chunk adjacency not generative)
        assert len(result.pair_plans) == 5

        # Verify every pair is SAME chunk (distance 0): no cross-chunk pair is
        # generated on adjacency alone, and no whole-document N² scan occurs.
        def get_chunk_ordinal(ref: str) -> int:
            chunk_id = ref.split(":", 1)[0]
            return chunk_ids.index(chunk_id) + 1

        for plan in result.pair_plans:
            left_ord = get_chunk_ordinal(plan.left_candidate_ref)
            right_ord = get_chunk_ordinal(plan.right_candidate_ref)
            assert left_ord == right_ord, (
                f"pair {plan.left_candidate_ref} (ord {left_ord}) vs "
                f"{plan.right_candidate_ref} (ord {right_ord}) is cross-chunk; "
                f"v2 cross-chunk adjacency must not be generative"
            )
            assert plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
            assert SIGNAL_ADJACENT_CHUNK in plan.signals
            assert SIGNAL_EXACT_IDENTITY_KEY not in plan.signals
            assert SIGNAL_IDENTITY_TOKEN_OVERLAP not in plan.signals

    def test_distance_2_pair_absent(self):
        """Chunk 1 candidate + chunk 3 candidate, no shared name → no pair."""
        source_doc_ref = make_source_document_ref()
        chunk_ids = ["CH001_C001", "CH001_C002", "CH001_C003"]
        source_doc = make_source_document(num_paragraphs=20)
        chunk_refs = [make_source_chunk_ref(cid) for cid in chunk_ids]
        chunks = [
            make_source_chunk(
                chunk_id=cid,
                source_document_ref=source_doc_ref,
                paragraph_ids=tuple(
                    f"CH001_P{i:04d}" for i in range(c * 5 + 1, c * 5 + 6)
                ),
            )
            for c, cid in enumerate(chunk_ids)
        ]
        # Only chunk 1 and chunk 3 have characters
        ext1 = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_refs[0],
            characters=(make_character(
                candidate_id="cand_char_001",
                display_name_original="Alice",
                evidence=make_evidence("CH001_P0001"),
            ),),
        )
        ext2 = make_extraction(
            chunk_id="CH001_C002",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_refs[1],
        )
        ext3 = make_extraction(
            chunk_id="CH001_C003",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_refs[2],
            characters=(make_character(
                candidate_id="cand_char_001",
                display_name_original="Bob",
                evidence=make_evidence("CH001_P0016"),
            ),),
        )
        manifest = make_chunk_manifest(
            chunk_refs=tuple(chunk_refs), source_document_ref=source_doc_ref
        )
        snapshot = ReconciliationInputSnapshot(
            source_document=source_doc,
            source_document_ref=source_doc_ref,
            chunk_manifest=manifest,
            source_chunks=tuple(chunks),
            source_chunk_refs=tuple(chunk_refs),
            candidate_extractions=(ext1, ext2, ext3),
            candidate_extraction_refs=tuple(
                make_extraction_ref(cid) for cid in chunk_ids
            ),
            a3_validation_report_refs=tuple(
                make_validation_report_ref(cid) for cid in chunk_ids
            ),
        )
        result = plan_reconciliation(snapshot)
        # Alice (chunk 1) and Bob (chunk 3) are distance 2 → NOT blocked
        assert len(result.pair_plans) == 0


# ===========================================================================
# Plan hash determinism tests
# ===========================================================================


class TestPlanHash:
    def test_same_input_same_hash(self):
        snapshot1 = build_snapshot(["CH001_C001", "CH001_C002"], characters_per_chunk=2)
        snapshot2 = build_snapshot(["CH001_C001", "CH001_C002"], characters_per_chunk=2)
        result1 = plan_reconciliation(snapshot1)
        result2 = plan_reconciliation(snapshot2)
        assert result1.plan_hash == result2.plan_hash

    def test_different_input_different_hash(self):
        snapshot1 = build_snapshot(["CH001_C001"], characters_per_chunk=1)
        snapshot2 = build_snapshot(["CH001_C001", "CH001_C002"], characters_per_chunk=1)
        result1 = plan_reconciliation(snapshot1)
        result2 = plan_reconciliation(snapshot2)
        assert result1.plan_hash != result2.plan_hash

    def test_hash_is_sha256(self):
        snapshot = build_snapshot(["CH001_C001"], characters_per_chunk=1)
        result = plan_reconciliation(snapshot)
        assert len(result.plan_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.plan_hash)

    def test_policy_ids_in_result(self):
        snapshot = build_snapshot(["CH001_C001"], characters_per_chunk=1)
        result = plan_reconciliation(snapshot)
        assert result.normalization_policy_id == NAME_NORMALIZATION_POLICY_ID
        assert result.blocking_policy_id == BLOCKING_POLICY_ID
        assert result.canonicalization_policy_id == CANONICALIZATION_POLICY_ID


# ===========================================================================
# Coverage audit tests
# ===========================================================================


class TestCoverageAudit:
    def test_coverage_complete(self):
        snapshot = build_snapshot(
            ["CH001_C001", "CH001_C002"],
            characters_per_chunk=1,
            locations_per_chunk=1,
            unresolved_per_chunk=1,
        )
        result = plan_reconciliation(snapshot)
        assert len(result.candidate_index.entries) == 6


# ===========================================================================
# Decision determinism tests
# ===========================================================================


class TestDecisionDeterminism:
    def test_decision_id_deterministic(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result1 = plan_reconciliation(snapshot)
        result2 = plan_reconciliation(snapshot)
        assert result1.decisions[0].decision_id == result2.decisions[0].decision_id

    def test_decision_id_format(self):
        source_doc_ref = make_source_document_ref()
        chunk_ref = make_source_chunk_ref("CH001_C001")
        ext = make_extraction(
            chunk_id="CH001_C001",
            source_document_ref=source_doc_ref,
            source_chunk_ref=chunk_ref,
            characters=(
                make_character(
                    candidate_id="cand_char_001",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0001"),
                ),
                make_character(
                    candidate_id="cand_char_002",
                    display_name_original="John Smith",
                    evidence=make_evidence("CH001_P0002"),
                ),
            ),
        )
        snapshot = build_single_chunk_snapshot(ext)
        result = plan_reconciliation(snapshot)
        assert len(result.decisions) == 1
        dec_id = result.decisions[0].decision_id
        assert dec_id.startswith("dec_")
        assert len(dec_id) == 4 + 20
        assert all(c in "0123456789abcdef" for c in dec_id[4:])
