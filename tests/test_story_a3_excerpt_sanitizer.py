"""Issue #37 — deterministic excerpt sanitization before semantic validation.

Focused tests for the A3 pre-validation excerpt sanitizer
``sanitize_candidate_payload_excerpts`` (a service-boundary transformation that
runs between typed ``CandidatePayload`` loading and the existing
``validate_candidate_payload``):

  * exact non-null excerpt preserved byte-for-byte;
  * mismatched non-null excerpt -> None (only the excerpt value changes);
  * already-null excerpt stays None;
  * unresolvable ``paragraph_id`` left unchanged (validator still BLOCKS it);
  * evidence outside context still BLOCKS (sanitizer does not hide it);
  * primary evidence outside ownership still BLOCKS;
  * relationship self-reference still BLOCKS;
  * direct (unsanitized) validator behavior unchanged: a raw mismatched
    excerpt still produces ``A3_EVIDENCE_EXCERPT_MISMATCH``;
  * the transformation is deterministic, total, and non-mutating.

The service-boundary behavior (round-1 pass with no second generation call,
published ``excerpt: null``, exact-rerun reuse, genuine non-excerpt failure
still using bounded regeneration) is covered in ``test_story_a3_service.py``.
"""

from __future__ import annotations

from short_drama.story import (
    A3_EVIDENCE_EXCERPT_MISMATCH,
    A3_EVIDENCE_OUTSIDE_CONTEXT,
    A3_PRIMARY_EVIDENCE_OUTSIDE_OWNERSHIP,
    A3_RELATIONSHIP_SELF_REFERENCE,
    A3_SOURCE_REF_NOT_FOUND,
    CandidatePayload,
    EvidenceRef,
    FactCandidate,
    LocationCandidate,
    sanitize_candidate_payload_excerpts,
    validate_candidate_payload,
)

# Reuse the A3B validation-test source/payload authorities (a coherent exact
# SourceDocument/SourceChunk pair plus candidate builders + assertion helpers).
from test_story_a3_validation import (
    LEFT_A,
    MISSING_PID,
    OUTSIDE_CH002,
    OWN_A,
    assert_invalid,
    assert_valid,
    make_character,
    make_evidence,
    make_relationship,
    source_chunk,
    source_document,
)


# ---------------------------------------------------------------------------
# Sanitizer unit behavior
# ---------------------------------------------------------------------------


def test_exact_excerpt_preserved_byte_for_byte():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt="走进教室"),)),
        )
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, source_document())
    assert sanitized.characters[0].evidence[0].excerpt == "走进教室"
    assert sanitized.characters[0].evidence[0] == payload.characters[0].evidence[0]
    # original is never mutated
    assert payload.characters[0].evidence[0].excerpt == "走进教室"


def test_mismatched_excerpt_becomes_none():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt="不存在的文字"),)),
        )
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, source_document())
    assert sanitized.characters[0].evidence[0].excerpt is None
    # only the excerpt value changes; paragraph_id/role/strength are untouched
    assert sanitized.characters[0].evidence[0].paragraph_id == OWN_A
    assert sanitized.characters[0].evidence[0].role == "primary"
    assert sanitized.characters[0].evidence[0].strength == "explicit"


def test_already_null_excerpt_stays_none():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt=None),)),
        )
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, source_document())
    assert sanitized.characters[0].evidence[0].excerpt is None


def test_sanitizer_is_deterministic_and_non_mutating():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt="不存在的文字"),)),
        )
    )
    a = sanitize_candidate_payload_excerpts(payload, source_document())
    b = sanitize_candidate_payload_excerpts(payload, source_document())
    assert a == b
    # returns a new object (does not mutate the input)
    assert a is not payload
    assert a.characters is not payload.characters
    assert payload.characters[0].evidence[0].excerpt == "不存在的文字"


def test_sanitizer_applies_across_all_candidate_categories():
    doc = source_document()
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt="不存在的文字"),)),
        ),
        locations=(
            LocationCandidate(
                candidate_id="cand_loc_001",
                display_name_original="教室",
                aliases_original=(),
                descriptors_zh=("教室",),
                summary_zh="故事发生的教室。",
                evidence_strength="explicit",
                evidence=(make_evidence(OWN_A, excerpt="不存在的文字"),),
            ),
        ),
        facts=(
            FactCandidate(
                candidate_id="cand_fact_001",
                fact_type="identity",
                statement_zh="林晚是一名学生。",
                subject_refs=("cand_char_001",),
                object_refs=(),
                evidence_strength="explicit",
                evidence=(make_evidence(OWN_A, excerpt="不存在的文字"),),
            ),
        ),
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, doc)
    assert sanitized.characters[0].evidence[0].excerpt is None
    assert sanitized.locations[0].evidence[0].excerpt is None
    assert sanitized.facts[0].evidence[0].excerpt is None


# ---------------------------------------------------------------------------
# Fail-closed boundary: sanitizer must NOT hide any non-excerpt violation
# ---------------------------------------------------------------------------


def test_missing_paragraph_left_unchanged_and_validator_blocks():
    payload = CandidatePayload(
        characters=(
            make_character(
                evidence=(
                    make_evidence(
                        MISSING_PID, role="primary", excerpt="不存在的文字"
                    ),
                )
            ),
        )
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, source_document())
    # unresolvable id: the excerpt is preserved (NOT nulled)
    assert sanitized.characters[0].evidence[0].excerpt == "不存在的文字"
    # and the existing validator still BLOCKS it
    result = validate_candidate_payload(
        sanitized, source_document(), source_chunk()
    )
    assert_invalid(result, A3_SOURCE_REF_NOT_FOUND)


def test_evidence_outside_context_still_blocks():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OUTSIDE_CH002, role="primary"),)),
        )
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, source_document())
    result = validate_candidate_payload(
        sanitized, source_document(), source_chunk()
    )
    assert_invalid(result, A3_EVIDENCE_OUTSIDE_CONTEXT)


def test_primary_evidence_outside_ownership_still_blocks():
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(LEFT_A, role="primary"),)),
        )
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, source_document())
    result = validate_candidate_payload(
        sanitized, source_document(), source_chunk()
    )
    assert_invalid(result, A3_PRIMARY_EVIDENCE_OUTSIDE_OWNERSHIP)


def test_relationship_self_reference_still_blocks():
    payload = CandidatePayload(
        characters=(make_character(),),
        relationships=(
            make_relationship(source_ref="cand_char_001", target_ref="cand_char_001"),
        ),
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, source_document())
    result = validate_candidate_payload(
        sanitized, source_document(), source_chunk()
    )
    assert_invalid(result, A3_RELATIONSHIP_SELF_REFERENCE)


# ---------------------------------------------------------------------------
# Direct validator behavior unchanged (raw, unsanitized) + round-1 pass
# ---------------------------------------------------------------------------


def test_direct_validator_raw_mismatch_still_blocking():
    # A raw (unsanitized) mismatched excerpt passed DIRECTLY to the validator
    # must still produce A3_EVIDENCE_EXCERPT_MISMATCH (the validator is strict).
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt="不存在的文字"),)),
        )
    )
    result = validate_candidate_payload(payload, source_document(), source_chunk())
    assert_invalid(result, A3_EVIDENCE_EXCERPT_MISMATCH)


def test_only_excerpt_mismatch_sanitizes_then_validates_pass():
    # A payload whose ONLY defect is a mismatched excerpt sanitizes to a
    # payload that PASSes the existing validator (the round-1-pass precondition).
    payload = CandidatePayload(
        characters=(
            make_character(evidence=(make_evidence(OWN_A, excerpt="不存在的文字"),)),
        )
    )
    sanitized = sanitize_candidate_payload_excerpts(payload, source_document())
    assert_valid(validate_candidate_payload(sanitized, source_document(), source_chunk()))


def test_evidence_ref_is_not_mutated():
    # The frozen EvidenceRef in the input payload is not modified in place.
    ev = make_evidence(OWN_A, excerpt="不存在的文字")
    payload = CandidatePayload(characters=(make_character(evidence=(ev,)),))
    sanitize_candidate_payload_excerpts(payload, source_document())
    assert ev.excerpt == "不存在的文字"
    assert isinstance(ev, EvidenceRef)
