"""A3B chunk-extraction candidate semantic validation tests.

Covers the frozen A3B contract for ``validate_candidate_payload`` /
``canonicalize_candidate_payload``:

  * valid ownership-primary / context-supporting / exact-excerpt paths;
  * evidence failures (missing paragraph, outside context, primary outside
    ownership in LEFT/RIGHT, missing ownership-primary, excerpt mismatch);
  * upstream source-lineage failures fail closed (not as findings);
  * local-ID integrity (duplicate, non-001 start, gap, wrong namespace);
  * cross-reference graph (dangling, wrong-type, relationship self-reference,
    unresolved chain / forbidden category);
  * deterministic, idempotent canonicalization (candidate order, string
    collections, evidence order, byte-equivalent serialization);
  * deterministic finding codes / severity / owner stage / repair route.

Deliberately does NOT require: an artifact store, a CURRENT pointer, a
persisted ValidationReport, an LLM client, or a real-novel fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.story import (
    A3_EVIDENCE_EXCERPT_MISMATCH,
    A3_EVIDENCE_OUTSIDE_CONTEXT,
    A3_INVALID_UNRESOLVED_REFERENCE,
    A3_LOCAL_ID_DUPLICATE,
    A3_LOCAL_ID_GAP,
    A3_LOCAL_ID_NAMESPACE,
    A3_LOCAL_REF_NOT_FOUND,
    A3_LOCAL_REF_WRONG_TYPE,
    A3_MISSING_PRIMARY_EVIDENCE,
    A3_OWNER_STAGE,
    A3_PRIMARY_EVIDENCE_OUTSIDE_OWNERSHIP,
    A3_RELATIONSHIP_SELF_REFERENCE,
    A3_REPAIR_REGENERATE_CANDIDATE_PAYLOAD,
    A3_SOURCE_REF_NOT_FOUND,
    CandidatePayload,
    CharacterCandidate,
    EventCandidate,
    EvidenceRef,
    FactCandidate,
    LocationCandidate,
    RelationshipCandidate,
    SourceChapter,
    SourceDocument,
    SourceInfo,
    SourceParagraph,
    StoryIntegrityError,
    UnresolvedMentionCandidate,
    canonicalize_candidate_payload,
    validate_candidate_payload,
)
from short_drama.foundation import ValidationSeverity, ValidationResult
from short_drama.story.chunking import ParagraphSpan, SourceChunk, estimate_paragraphs
from short_drama.story.source import NormalizationInfo

HASH = "a" * 64

# Ownership paragraphs (used as the valid primary-evidence default).
OWN_A = "CH001_P0003"
OWN_B = "CH001_P0004"
# Context (non-ownership) paragraphs in the same chapter.
LEFT_A = "CH001_P0001"
LEFT_B = "CH001_P0002"
RIGHT_A = "CH001_P0005"
RIGHT_B = "CH001_P0006"
# A paragraph that exists in the SourceDocument but outside the chunk context.
OUTSIDE_CH002 = "CH002_P0001"
# A paragraph that does not exist in the SourceDocument at all.
MISSING_PID = "CH001_P9999"

CHAPTER_1_TEXTS = {
    "CH001_P0001": "左上下文第一段文字。",
    "CH001_P0002": "左上下文第二段文字。",
    "CH001_P0003": "林晚走进教室。",
    "CH001_P0004": "老师正在板书。",
    "CH001_P0005": "右上下文第一段文字。",
    "CH001_P0006": "右上下文第二段文字。",
}
CHAPTER_2_TEXTS = {
    "CH002_P0001": "另一章节第一段。",
    "CH002_P0002": "另一章节第二段。",
}


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def make_evidence(
    paragraph_id: str = OWN_A,
    role: str = "primary",
    strength: str = "explicit",
    excerpt: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role=role, strength=strength, excerpt=excerpt
    )


def _paragraphs(chapter_id: str, texts: dict[str, str]) -> tuple[SourceParagraph, ...]:
    return tuple(
        SourceParagraph(
            paragraph_id=paragraph_id,
            text_original=text,
            source_pages=None,
        )
        for paragraph_id, text in texts.items()
    )


def source_document() -> SourceDocument:
    chapters = (
        SourceChapter("CH001", None, "synthetic", _paragraphs("CH001", CHAPTER_1_TEXTS)),
        SourceChapter("CH002", None, "synthetic", _paragraphs("CH002", CHAPTER_2_TEXTS)),
    )
    return SourceDocument(
        schema_version=1,
        project_id="demo",
        document_id="src_001",
        source=SourceInfo("txt", "source/novel.txt", HASH, 123, "zh-CN", "zh", "langid-1.1.6"),
        normalization=NormalizationInfo(
            "utf-8", "LF", "short_drama_source_ingestion_v1", "1"
        ),
        chapters=chapters,
    )


def source_ref() -> ArtifactRef:
    return ArtifactRef("source_document", "demo.src_001", 1, HASH)


def source_chunk(document: SourceDocument | None = None, **overrides) -> SourceChunk:
    document = document if document is not None else source_document()
    index = document.paragraph_index()
    context = [f"CH001_P{i:04d}" for i in range(1, 7)]
    ownership = [OWN_A, OWN_B]
    values = {
        "schema_version": 1,
        "chunk_id": "CH001_C001",
        "project_id": document.project_id,
        "document_id": document.document_id,
        "chapter_id": "CH001",
        "source_document_ref": source_ref(),
        "context_span": ParagraphSpan(context[0], context[-1]),
        "ownership_span": ParagraphSpan(ownership[0], ownership[-1]),
        "paragraph_ids": tuple(context),
        "token_count_method": "utf8-bytes-div3-v1",
        "context_token_count": estimate_paragraphs(
            tuple(index[pid] for pid in context)
        ),
        "ownership_token_count": estimate_paragraphs(
            tuple(index[pid] for pid in ownership)
        ),
    }
    values.update(overrides)
    return SourceChunk(**values)


def make_character(**overrides) -> CharacterCandidate:
    values = {
        "candidate_id": "cand_char_001",
        "display_name_original": "林晚",
        "aliases_original": ("阿晚",),
        "descriptors_zh": ("学生",),
        "summary_zh": "本章中的女主角。",
        "evidence_strength": "explicit",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return CharacterCandidate(**values)


def make_location(**overrides) -> LocationCandidate:
    values = {
        "candidate_id": "cand_loc_001",
        "display_name_original": "教室",
        "aliases_original": (),
        "descriptors_zh": ("教室",),
        "summary_zh": "故事发生的教室。",
        "evidence_strength": "explicit",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return LocationCandidate(**values)


def make_fact(**overrides) -> FactCandidate:
    values = {
        "candidate_id": "cand_fact_001",
        "fact_type": "identity",
        "statement_zh": "林晚是一名学生。",
        "subject_refs": ("cand_char_001",),
        "object_refs": (),
        "evidence_strength": "explicit",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return FactCandidate(**values)


def make_event(**overrides) -> EventCandidate:
    values = {
        "candidate_id": "cand_evt_001",
        "summary_zh": "林晚走进教室。",
        "participant_refs": ("cand_char_001",),
        "location_refs": ("cand_loc_001",),
        "temporal_mode": "normal",
        "evidence_strength": "explicit",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return EventCandidate(**values)


def make_relationship(**overrides) -> RelationshipCandidate:
    values = {
        "candidate_id": "cand_rel_001",
        "source_ref": "cand_char_001",
        "target_ref": "cand_char_002",
        "relationship_type_zh": "师生",
        "state_zh": None,
        "direction": "directed",
        "evidence_strength": "implied",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    return RelationshipCandidate(**values)


def make_unresolved(**overrides) -> UnresolvedMentionCandidate:
    values = {
        "candidate_id": "cand_unres_001",
        "mention_original": "他",
        "mention_kind": "person",
        "reason_zh": "无法确定该代词指代的具体人物。",
        "possible_candidate_refs": ("cand_char_001",),
        "evidence_strength": "uncertain",
        "evidence": (make_evidence(strength="uncertain"),),
    }
    values.update(overrides)
    return UnresolvedMentionCandidate(**values)


def full_payload() -> CandidatePayload:
    return CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_001"),
            make_character(candidate_id="cand_char_002", display_name_original="老师"),
        ),
        locations=(make_location(),),
        facts=(
            make_fact(
                subject_refs=("cand_char_001",),
                object_refs=("cand_loc_001",),
            ),
        ),
        events=(
            make_event(
                participant_refs=("cand_char_001",),
                location_refs=("cand_loc_001",),
            ),
        ),
        relationships=(
            make_relationship(source_ref="cand_char_001", target_ref="cand_char_002"),
        ),
        unresolved_mentions=(
            make_unresolved(possible_candidate_refs=("cand_char_001",)),
        ),
    )


def raw_character(candidate_id: str, **overrides) -> CharacterCandidate:
    """Bypass A3A ``__post_init__`` to build a malformed candidate.

    Used to prove A3B independently re-checks the ID namespace for a
    "malformed object" that should not exist per A3A.
    """
    values = {
        "candidate_id": candidate_id,
        "display_name_original": "林晚",
        "aliases_original": ("阿晚",),
        "descriptors_zh": ("学生",),
        "summary_zh": "本章中的女主角。",
        "evidence_strength": "explicit",
        "evidence": (make_evidence(),),
    }
    values.update(overrides)
    obj = object.__new__(CharacterCandidate)
    for key, value in values.items():
        object.__setattr__(obj, key, value)
    return obj


# ---------------------------------------------------------------------------
# Small assertion helpers
# ---------------------------------------------------------------------------


def validate(payload: CandidatePayload) -> "Any":
    return validate_candidate_payload(payload, source_document(), source_chunk())


def finding_codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


def assert_invalid(result, code: str) -> None:
    assert result.is_valid is False
    assert result.canonical_payload is None
    assert code in finding_codes(result)


def assert_valid(result) -> None:
    assert result.is_valid is True
    assert result.findings == ()
    assert result.summary.result is ValidationResult.PASS
    assert result.canonical_payload is not None


# ---------------------------------------------------------------------------
# Valid paths
# ---------------------------------------------------------------------------


def test_full_valid_cross_reference_graph_passes():
    # Valid ownership-primary evidence + a fully closed local cross-reference
    # graph across all six categories.
    assert_valid(validate(full_payload()))


def test_ownership_only_evidence_passes():
    payload = CandidatePayload(
        characters=(
            make_character(
                evidence=(make_evidence(OWN_B, role="primary"),),
            ),
        )
    )
    assert_valid(validate(payload))


def test_ownership_primary_plus_context_supporting_passes():
    payload = CandidatePayload(
        characters=(
            make_character(
                evidence=(
                    make_evidence(OWN_A, role="primary"),
                    make_evidence(LEFT_A, role="supporting"),
                    make_evidence(RIGHT_B, role="supporting"),
                ),
            ),
        )
    )
    assert_valid(validate(payload))


def test_excerpt_exact_substring_passes():
    # OWN_A text is "林晚走进教室。"; a verbatim substring passes.
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt="走进教室"),)),
        )
    )
    assert_valid(validate(payload))


def test_legitimate_unresolved_output_passes():
    # Uncertainty is a valid output: an unresolved mention with a person ref is
    # NOT a finding.
    payload = CandidatePayload(
        characters=(make_character(),),
        unresolved_mentions=(
            make_unresolved(possible_candidate_refs=("cand_char_001",)),
        ),
    )
    assert_valid(validate(payload))


def test_empty_possible_candidate_refs_passes():
    payload = CandidatePayload(
        characters=(make_character(),),
        unresolved_mentions=(
            make_unresolved(possible_candidate_refs=()),
        ),
    )
    assert_valid(validate(payload))


def test_temporal_mode_unknown_passes():
    payload = CandidatePayload(
        characters=(make_character(),),
        locations=(make_location(),),
        events=(
            make_event(temporal_mode="unknown"),
        ),
    )
    assert_valid(validate(payload))


def test_uncertain_evidence_strength_passes():
    payload = CandidatePayload(
        characters=(
            make_character(
                evidence_strength="uncertain",
                evidence=(make_evidence(strength="uncertain"),),
            ),
        )
    )
    assert_valid(validate(payload))


def test_all_categories_empty_passes():
    assert_valid(validate(CandidatePayload()))


# ---------------------------------------------------------------------------
# Evidence failures
# ---------------------------------------------------------------------------


def test_missing_paragraph_id_fails():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(MISSING_PID, role="primary"),)),
        )
    )
    assert_invalid(validate(payload), A3_SOURCE_REF_NOT_FOUND)


def test_paragraph_exists_globally_but_outside_chunk_context_fails():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OUTSIDE_CH002, role="primary"),)),
        )
    )
    assert_invalid(validate(payload), A3_EVIDENCE_OUTSIDE_CONTEXT)


def test_primary_evidence_in_left_context_fails():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(LEFT_A, role="primary"),)),
        )
    )
    assert_invalid(validate(payload), A3_PRIMARY_EVIDENCE_OUTSIDE_OWNERSHIP)


def test_primary_evidence_in_right_context_fails():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(RIGHT_B, role="primary"),)),
        )
    )
    assert_invalid(validate(payload), A3_PRIMARY_EVIDENCE_OUTSIDE_OWNERSHIP)


def test_supporting_only_evidence_fails_missing_primary():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, role="supporting"),)),
        )
    )
    assert_invalid(validate(payload), A3_MISSING_PRIMARY_EVIDENCE)


def test_bad_excerpt_substring_fails():
    payload = CandidatePayload(
        characters=(
            make_character(
                evidence=(make_evidence(OWN_A, excerpt="不存在的文字"),),
            ),
        )
    )
    assert_invalid(validate(payload), A3_EVIDENCE_EXCERPT_MISMATCH)


def test_excerpt_mismatch_is_not_normalized():
    # Same words, different punctuation -> NOT an exact substring.
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt="林晚 走进 教室"),),
            ),
        )
    )
    assert_invalid(validate(payload), A3_EVIDENCE_EXCERPT_MISMATCH)


def test_invalid_source_lineage_chunk_paragraph_not_in_document_fails_closed():
    bad_chunk = source_chunk(
        paragraph_ids=(LEFT_A, LEFT_B, MISSING_PID),
        context_span=ParagraphSpan(LEFT_A, MISSING_PID),
        ownership_span=ParagraphSpan(LEFT_A, LEFT_B),
    )
    payload = CandidatePayload(characters=(make_character(),))
    with pytest.raises(StoryIntegrityError):
        validate_candidate_payload(payload, source_document(), bad_chunk)


def test_invalid_source_lineage_project_mismatch_fails_closed():
    bad_chunk = source_chunk(project_id="other-project")
    payload = CandidatePayload(characters=(make_character(),))
    with pytest.raises(StoryIntegrityError):
        validate_candidate_payload(payload, source_document(), bad_chunk)


def test_invalid_source_lineage_document_mismatch_fails_closed():
    bad_chunk = source_chunk(document_id="other-doc")
    payload = CandidatePayload(characters=(make_character(),))
    with pytest.raises(StoryIntegrityError):
        validate_candidate_payload(payload, source_document(), bad_chunk)


def test_invalid_source_lineage_wrong_type_fails_closed():
    payload = CandidatePayload(characters=(make_character(),))
    with pytest.raises(StoryIntegrityError):
        validate_candidate_payload(payload, object(), source_chunk())


# ---------------------------------------------------------------------------
# Local ID failures
# ---------------------------------------------------------------------------


def test_duplicate_candidate_id_fails():
    payload = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_001"),
            make_character(candidate_id="cand_char_001", display_name_original="老师"),
        )
    )
    assert_invalid(validate(payload), A3_LOCAL_ID_DUPLICATE)


def test_numbering_starts_at_002_fails():
    payload = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_002"),
        )
    )
    assert_invalid(validate(payload), A3_LOCAL_ID_GAP)


def test_numbering_gap_fails():
    payload = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_001"),
            make_character(candidate_id="cand_char_003", display_name_original="老师"),
        )
    )
    assert_invalid(validate(payload), A3_LOCAL_ID_GAP)


def test_wrong_namespace_fails_when_malformed_object_reaches_validator():
    # A "malformed" character whose id is a location namespace: A3B must catch
    # it even though A3A would normally reject it at construction.
    payload = CandidatePayload(
        characters=(raw_character("cand_loc_001"),),
    )
    assert_invalid(validate(payload), A3_LOCAL_ID_NAMESPACE)


# ---------------------------------------------------------------------------
# Cross-reference failures
# ---------------------------------------------------------------------------


def test_dangling_subject_ref_fails():
    payload = CandidatePayload(
        characters=(make_character(),),
        facts=(
            make_fact(subject_refs=("cand_char_999",)),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_NOT_FOUND)


def test_dangling_object_ref_fails():
    payload = CandidatePayload(
        characters=(make_character(),),
        locations=(make_location(),),
        facts=(
            make_fact(subject_refs=("cand_char_001",), object_refs=("cand_loc_999",)),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_NOT_FOUND)


def test_dangling_participant_ref_fails():
    payload = CandidatePayload(
        characters=(make_character(),),
        locations=(make_location(),),
        events=(
            make_event(participant_refs=("cand_char_999",), location_refs=("cand_loc_001",)),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_NOT_FOUND)


def test_dangling_location_ref_fails():
    payload = CandidatePayload(
        characters=(make_character(),),
        locations=(make_location(),),
        events=(
            make_event(participant_refs=("cand_char_001",), location_refs=("cand_loc_999",)),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_NOT_FOUND)


def test_relationship_dangling_source_fails():
    payload = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_001"),
            make_character(candidate_id="cand_char_002", display_name_original="老师"),
        ),
        relationships=(
            make_relationship(source_ref="cand_char_999", target_ref="cand_char_002"),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_NOT_FOUND)


def test_relationship_dangling_target_fails():
    payload = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_001"),
            make_character(candidate_id="cand_char_002", display_name_original="老师"),
        ),
        relationships=(
            make_relationship(source_ref="cand_char_001", target_ref="cand_char_999"),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_NOT_FOUND)


def test_wrong_type_participant_ref_fails():
    # A location referenced as a participant is a wrong-type reference.
    payload = CandidatePayload(
        locations=(make_location(),),
        events=(
            make_event(participant_refs=("cand_loc_001",), location_refs=()),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_WRONG_TYPE)


def test_wrong_type_location_ref_fails():
    # A character referenced as a location is a wrong-type reference.
    payload = CandidatePayload(
        characters=(make_character(),),
        events=(
            make_event(participant_refs=(), location_refs=("cand_char_001",)),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_WRONG_TYPE)


def test_relationship_wrong_type_ref_fails():
    # A location referenced as a relationship endpoint is a wrong-type reference.
    payload = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_001"),
            make_character(candidate_id="cand_char_002", display_name_original="老师"),
        ),
        locations=(make_location(),),
        relationships=(
            make_relationship(source_ref="cand_loc_001", target_ref="cand_char_002"),
        ),
    )
    assert_invalid(validate(payload), A3_LOCAL_REF_WRONG_TYPE)


def test_relationship_self_reference_fails():
    payload = CandidatePayload(
        characters=(make_character(),),
        relationships=(
            make_relationship(source_ref="cand_char_001", target_ref="cand_char_001"),
        ),
    )
    assert_invalid(validate(payload), A3_RELATIONSHIP_SELF_REFERENCE)


def test_unresolved_to_unresolved_ref_fails():
    # An unresolved mention may not reference another (or itself) unresolved.
    payload = CandidatePayload(
        characters=(make_character(),),
        unresolved_mentions=(
            make_unresolved(candidate_id="cand_unres_001", possible_candidate_refs=("cand_unres_002",)),
            make_unresolved(candidate_id="cand_unres_002", possible_candidate_refs=()),
        ),
    )
    assert_invalid(validate(payload), A3_INVALID_UNRESOLVED_REFERENCE)


def test_unresolved_possible_ref_to_forbidden_category_fails():
    # An unresolved mention may not reference a fact (forbidden category).
    payload = CandidatePayload(
        characters=(make_character(),),
        facts=(make_fact(subject_refs=("cand_char_001",)),),
        unresolved_mentions=(
            make_unresolved(possible_candidate_refs=("cand_fact_001",)),
        ),
    )
    assert_invalid(validate(payload), A3_INVALID_UNRESOLVED_REFERENCE)


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def test_reversed_candidate_arrays_canonicalize_to_numeric_order():
    payload = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_003", display_name_original="三人"),
            make_character(candidate_id="cand_char_001"),
            make_character(candidate_id="cand_char_002", display_name_original="老师"),
        ),
    )
    canonical = canonicalize_candidate_payload(payload, source_document())
    assert [c.candidate_id for c in canonical.characters] == [
        "cand_char_001",
        "cand_char_002",
        "cand_char_003",
    ]


def test_string_collections_canonicalize_deterministically():
    payload = CandidatePayload(
        characters=(
            make_character(
                aliases_original=("z", "a", "m"),
                descriptors_zh=("乙", "甲", "丙"),
            ),
        ),
        facts=(
            make_fact(
                subject_refs=("cand_char_002", "cand_char_001"),
                object_refs=("cand_loc_002", "cand_loc_001"),
            ),
        ),
    )
    canonical = canonicalize_candidate_payload(payload, source_document())
    assert canonical.characters[0].aliases_original == ("a", "m", "z")
    # Plain lexicographic (Unicode code-point) order: 丙(U+4E19) 乙(U+4E59) 甲(U+7532).
    assert canonical.characters[0].descriptors_zh == ("丙", "乙", "甲")
    assert canonical.facts[0].subject_refs == ("cand_char_001", "cand_char_002")
    assert canonical.facts[0].object_refs == ("cand_loc_001", "cand_loc_002")


def test_evidence_canonicalizes_by_source_order_and_role_precedence():
    payload = CandidatePayload(
        characters=(
            make_character(
                evidence=(
                    make_evidence(OWN_B, role="supporting"),
                    make_evidence(OWN_B, role="primary"),
                    make_evidence(OWN_A, role="primary"),
                ),
            ),
        )
    )
    canonical = canonicalize_candidate_payload(payload, source_document())
    got = [(e.paragraph_id, e.role) for e in canonical.characters[0].evidence]
    # Source paragraph order first (OWN_A before OWN_B); within OWN_B primary
    # before supporting.
    assert got == [
        (OWN_A, "primary"),
        (OWN_B, "primary"),
        (OWN_B, "supporting"),
    ]


def test_canonicalization_is_deterministic():
    payload = CandidatePayload(
        characters=(
            make_character(
                candidate_id="cand_char_002",
                aliases_original=("z", "a"),
                evidence=(
                    make_evidence(OWN_B, role="supporting"),
                    make_evidence(OWN_A, role="primary"),
                ),
            ),
            make_character(candidate_id="cand_char_001"),
        )
    )
    document = source_document()
    first = canonicalize_candidate_payload(payload, document)
    second = canonicalize_candidate_payload(payload, document)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_canonicalization_is_idempotent():
    payload = CandidatePayload(
        characters=(
            make_character(
                candidate_id="cand_char_002",
                aliases_original=("z", "a"),
                evidence=(
                    make_evidence(OWN_B, role="supporting"),
                    make_evidence(OWN_A, role="primary"),
                ),
            ),
            make_character(candidate_id="cand_char_001"),
        ),
        unresolved_mentions=(
            make_unresolved(possible_candidate_refs=("cand_char_002", "cand_char_001")),
        ),
    )
    document = source_document()
    once = canonicalize_candidate_payload(payload, document)
    twice = canonicalize_candidate_payload(once, document)
    assert once == twice
    # typed round-trip remains exact
    assert CandidatePayload.from_dict(once.to_dict()) == once


def test_same_semantic_input_produces_byte_equivalent_to_dict():
    document = source_document()

    payload_a = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_002", display_name_original="老师"),
            make_character(candidate_id="cand_char_001"),
        ),
        facts=(
            make_fact(
                subject_refs=("cand_char_002", "cand_char_001"),
                evidence=(
                    make_evidence(OWN_B, role="supporting"),
                    make_evidence(OWN_A, role="primary"),
                ),
            ),
        ),
    )
    # Semantically identical payload with a different input ordering.
    payload_b = CandidatePayload(
        characters=(
            make_character(candidate_id="cand_char_001"),
            make_character(candidate_id="cand_char_002", display_name_original="老师"),
        ),
        facts=(
            make_fact(
                subject_refs=("cand_char_001", "cand_char_002"),
                evidence=(
                    make_evidence(OWN_A, role="primary"),
                    make_evidence(OWN_B, role="supporting"),
                ),
            ),
        ),
    )
    canonical_a = canonicalize_candidate_payload(payload_a, document)
    canonical_b = canonicalize_candidate_payload(payload_b, document)
    assert canonical_a.to_dict() == canonical_b.to_dict()
    assert content_hash(canonical_a.to_dict()) == content_hash(canonical_b.to_dict())


def test_validate_returns_canonical_form_for_valid_scrambled_payload():
    payload = CandidatePayload(
        characters=(
            make_character(
                candidate_id="cand_char_002",
                aliases_original=("z", "a"),
                evidence=(
                    make_evidence(OWN_B, role="supporting"),
                    make_evidence(OWN_A, role="primary"),
                ),
            ),
            make_character(candidate_id="cand_char_001"),
        ),
    )
    document = source_document()
    result = validate_candidate_payload(payload, document, source_chunk())
    assert_valid(result)
    assert result.canonical_payload == canonicalize_candidate_payload(
        payload, document
    )


# ---------------------------------------------------------------------------
# Finding metadata / repair routing
# ---------------------------------------------------------------------------


def test_findings_are_blocking_with_a3_owner_and_regeneration_route():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(MISSING_PID, role="primary"),)),
        )
    )
    result = validate(payload)
    assert result.is_valid is False
    assert result.summary.blocking_count >= 1
    for finding in result.findings:
        assert finding.severity is ValidationSeverity.BLOCKING
        assert finding.owner_stage == A3_OWNER_STAGE == "A3"
        assert finding.repair_route == A3_REPAIR_REGENERATE_CANDIDATE_PAYLOAD
        assert finding.finding_id
        assert finding.code.startswith("A3_")
