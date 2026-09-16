"""v1.2 A3B — chunk-extraction candidate semantic validation + canonicalization.

This module is the deterministic *semantic* validation layer for A3 (chunk
extraction). It sits strictly on top of A3A: it receives a structurally valid
:class:`~short_drama.story.extraction.CandidatePayload` plus the exact
A1/A2 authorities (``SourceDocument`` + ``SourceChunk``) and answers, without
any LLM and without touching the filesystem:

  1. does every evidence reference point at a real paragraph inside this
     chunk, with primary evidence owned by the chunk?
  2. does every (non-null) excerpt match the referenced paragraph verbatim?
  3. are the local candidate IDs unique, continuous, and namespaced?
  4. does the local cross-reference graph close inside this payload with the
     frozen per-field type restrictions?
  5. what is the single deterministic canonical form of the payload?

It reuses the Foundation finding layer (``ValidationFinding`` /
``ValidationSummary`` / ``derive_validation_summary``) rather than inventing a
second validation abstraction. It does NOT persist a ``ValidationReport``:
A3C owns persisted reports, CURRENT pointers, and reuse.

Deliberately out of scope (later slices): artifact persistence, CURRENT/reuse,
LLM orchestration/retry, the ``extract-chunks`` CLI, real-Qwen smoke, and any
A4/A5/A6 logic.

Authority discipline: the exact ``SourceDocument`` (``paragraph_id`` +
``text_original``) is the only source-location authority and the exact
``SourceChunk`` (``paragraph_ids`` + ``ownership_span``) is the only
context/ownership authority. A3B derives ownership from the chunk; it never
trusts an LLM-declared ownership flag. A structurally inconsistent
SourceDocument/SourceChunk pair is an A1/A2 lineage/integrity failure and fails
closed with :class:`StoryIntegrityError`, never as an LLM regeneration finding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from short_drama.artifacts import content_hash
from short_drama.foundation import (
    ValidationFinding,
    ValidationResult,
    ValidationSeverity,
    ValidationSummary,
    derive_validation_summary,
)

from .chunking import SourceChunk
from .errors import StoryIntegrityError
from .extraction import (
    CharacterCandidate,
    CandidatePayload,
    EventCandidate,
    EvidenceRef,
    FactCandidate,
    LocationCandidate,
    RelationshipCandidate,
    UnresolvedMentionCandidate,
    _CANDIDATE_ID_PATTERNS,
)
from .source import SourceDocument

# ---------------------------------------------------------------------------
# Frozen A3 finding codes (parent A-I4 contract, section 17.1)
# ---------------------------------------------------------------------------

A3_SOURCE_REF_NOT_FOUND = "A3_SOURCE_REF_NOT_FOUND"
A3_EVIDENCE_OUTSIDE_CONTEXT = "A3_EVIDENCE_OUTSIDE_CONTEXT"
A3_PRIMARY_EVIDENCE_OUTSIDE_OWNERSHIP = "A3_PRIMARY_EVIDENCE_OUTSIDE_OWNERSHIP"
A3_EVIDENCE_EXCERPT_MISMATCH = "A3_EVIDENCE_EXCERPT_MISMATCH"
A3_MISSING_PRIMARY_EVIDENCE = "A3_MISSING_PRIMARY_EVIDENCE"
A3_LOCAL_ID_NAMESPACE = "A3_LOCAL_ID_NAMESPACE"
A3_LOCAL_ID_DUPLICATE = "A3_LOCAL_ID_DUPLICATE"
A3_LOCAL_ID_GAP = "A3_LOCAL_ID_GAP"
A3_LOCAL_REF_NOT_FOUND = "A3_LOCAL_REF_NOT_FOUND"
A3_LOCAL_REF_WRONG_TYPE = "A3_LOCAL_REF_WRONG_TYPE"
A3_RELATIONSHIP_SELF_REFERENCE = "A3_RELATIONSHIP_SELF_REFERENCE"
A3_INVALID_UNRESOLVED_REFERENCE = "A3_INVALID_UNRESOLVED_REFERENCE"

# Repair route for LLM-produced semantic-invalid payloads (parent contract
# section 18/17.1). A3B only *names* this route; it never performs the
# regeneration itself (that is A3D).
A3_REPAIR_REGENERATE_CANDIDATE_PAYLOAD = "A3_REGENERATE_CANDIDATE_PAYLOAD"

# Owner stage for every A3 semantic finding (parent contract: owner_stage=A3).
A3_OWNER_STAGE = "A3"

# Deterministic top-level category order (matches A3A serialization order):
# (field name on CandidatePayload, candidate type, category name).
_CATEGORIES: tuple[tuple[str, type, str], ...] = (
    ("characters", CharacterCandidate, "character"),
    ("locations", LocationCandidate, "location"),
    ("facts", FactCandidate, "fact"),
    ("events", EventCandidate, "event"),
    ("relationships", RelationshipCandidate, "relationship"),
    ("unresolved_mentions", UnresolvedMentionCandidate, "unresolved"),
)

# Maps each A3A namespace-pattern key to this module's internal category name.
# A3A's ``_CANDIDATE_ID_PATTERNS`` remains the single namespace authority; this
# is only a key->label normalization (``unresolved_mention`` -> ``unresolved``).
_PATTERN_KEY_TO_CATEGORY = {
    "character": "character",
    "location": "location",
    "fact": "fact",
    "event": "event",
    "relationship": "relationship",
    "unresolved_mention": "unresolved",
}

# Reference-bearing fields and their frozen type restrictions. A reference is
# resolved to a category (and, for unresolved mentions, a mention_kind) and the
# predicate decides whether that target type is legal for the field.
_PERSON_REF_FIELDS = ("participant_refs", "source_ref", "target_ref")
_LOCATION_REF_FIELDS = ("location_refs",)
_ANY_ENTITY_REF_FIELDS = ("subject_refs", "object_refs")
_UNRESOLVED_POSSIBLE_REF_FIELD = "possible_candidate_refs"


def _id_category(candidate_id: str) -> str | None:
    """Return the A3A namespace category for a candidate id, or ``None``.

    Reuses A3A's single namespace-pattern authority (no second scheme).
    """
    if not isinstance(candidate_id, str):
        return None
    for pattern_key, pattern in _CANDIDATE_ID_PATTERNS.items():
        if pattern.fullmatch(candidate_id) is not None:
            return _PATTERN_KEY_TO_CATEGORY[pattern_key]
    return None


def _candidate_id_suffix(candidate_id: str) -> int:
    # A namespaced id is ``cand_<ns>_<digits>``; the digits are the suffix.
    return int(candidate_id.rsplit("_", 1)[1])


def _finding(
    code: str,
    message: str,
    path: tuple[str | int, ...],
) -> ValidationFinding:
    material = {"code": code, "path": list(path), "message": message}
    return ValidationFinding(
        finding_id=f"A3-{content_hash(material)[:20]}",
        code=code,
        severity=ValidationSeverity.BLOCKING,
        owner_stage=A3_OWNER_STAGE,
        repair_route=A3_REPAIR_REGENERATE_CANDIDATE_PAYLOAD,
        message=message,
        path=tuple(path),
    )


# ---------------------------------------------------------------------------
# Source authority helpers
# ---------------------------------------------------------------------------


def _source_paragraph_ids(source_document: SourceDocument) -> list[str]:
    return [paragraph.paragraph_id for paragraph in source_document.paragraphs]


def _ownership_ids(source_chunk: SourceChunk) -> frozenset[str]:
    """Ownership paragraph ids derived from the canonical SourceChunk.

    ``ownership_span`` is inclusive; its endpoints are guaranteed (by A3A) to be
    present in ``paragraph_ids`` with start <= end.
    """
    paragraph_ids = source_chunk.paragraph_ids
    start = paragraph_ids.index(source_chunk.ownership_span.start)
    end = paragraph_ids.index(source_chunk.ownership_span.end)
    return frozenset(paragraph_ids[start : end + 1])


def _check_source_lineage(
    source_document: SourceDocument, source_chunk: SourceChunk
) -> None:
    """Fail closed on a structurally inconsistent exact SourceDocument/Chunk pair.

    These are A1/A2 lineage/integrity failures: the upstream artifacts disagree
    with each other. They must NOT be disguised as an LLM regeneration finding,
    so they raise :class:`StoryIntegrityError` rather than produce a finding.
    """
    if not isinstance(source_document, SourceDocument):
        raise StoryIntegrityError(
            "source_document must be an exact A1 SourceDocument"
        )
    if not isinstance(source_chunk, SourceChunk):
        raise StoryIntegrityError(
            "source_chunk must be an exact A2 SourceChunk"
        )
    if (
        source_chunk.project_id != source_document.project_id
        or source_chunk.document_id != source_document.document_id
    ):
        raise StoryIntegrityError(
            "SourceChunk does not target the exact SourceDocument "
            "project/document identity"
        )
    paragraph_index = source_document.paragraph_index()
    for paragraph_id in source_chunk.paragraph_ids:
        if paragraph_id not in paragraph_index:
            raise StoryIntegrityError(
                f"SourceChunk paragraph {paragraph_id!r} is not present in the "
                "exact SourceDocument"
            )
    if (
        source_chunk.ownership_span.start not in paragraph_index
        or source_chunk.ownership_span.end not in paragraph_index
    ):
        raise StoryIntegrityError(
            "SourceChunk ownership span does not resolve in the exact "
            "SourceDocument"
        )
    return None


# ---------------------------------------------------------------------------
# Local ID integrity (payload-wide invariants on top of A3A shape checks)
# ---------------------------------------------------------------------------


def _validate_category_ids(
    candidates: tuple[Any, ...],
    field_name: str,
    category: str,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    ids = [candidate.candidate_id for candidate in candidates]

    # 8.x namespace (defensive re-check: A3A enforces it, but a malformed object
    # that reaches A3B must still be caught).
    for index, candidate_id in enumerate(ids):
        if _id_category(candidate_id) != category:
            findings.append(
                _finding(
                    A3_LOCAL_ID_NAMESPACE,
                    f"{candidate_id!r} is not a valid {category} candidate id "
                    f"in {field_name}",
                    (field_name, index, "candidate_id"),
                )
            )

    # 8.1 uniqueness.
    if len(ids) != len(set(ids)):
        duplicates = sorted(
            {candidate_id for candidate_id in ids if ids.count(candidate_id) > 1}
        )
        findings.append(
            _finding(
                A3_LOCAL_ID_DUPLICATE,
                f"duplicate {category} candidate ids: {', '.join(duplicates)}",
                (field_name, "candidate_id"),
            )
        )

    # 8.2 continuous numbering starting at 001 with no gaps (only meaningful
    # over namespace-valid ids; an empty category is valid).
    suffixes = sorted(
        _candidate_id_suffix(candidate_id)
        for candidate_id in ids
        if _id_category(candidate_id) == category
    )
    if suffixes != list(range(1, len(suffixes) + 1)):
        findings.append(
            _finding(
                A3_LOCAL_ID_GAP,
                f"{category} candidate ids are not continuous starting at 001: "
                f"{suffixes}",
                (field_name, "candidate_id"),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Cross-reference graph validation
# ---------------------------------------------------------------------------


def _candidate_ref_fields(candidate: Any) -> list[tuple[str, tuple[str, ...]]]:
    """Enumerate the reference-bearing fields of a candidate."""
    if isinstance(candidate, FactCandidate):
        return [
            ("subject_refs", candidate.subject_refs),
            ("object_refs", candidate.object_refs),
        ]
    if isinstance(candidate, EventCandidate):
        return [
            ("participant_refs", candidate.participant_refs),
            ("location_refs", candidate.location_refs),
        ]
    if isinstance(candidate, RelationshipCandidate):
        return [
            ("source_ref", (candidate.source_ref,)),
            ("target_ref", (candidate.target_ref,)),
        ]
    if isinstance(candidate, UnresolvedMentionCandidate):
        return [("possible_candidate_refs", candidate.possible_candidate_refs)]
    return []


def _ref_is_allowed(field_name: str, category: str | None, mention_kind: str | None) -> bool:
    if category is None:
        return False
    if field_name in _ANY_ENTITY_REF_FIELDS:
        return category in {"character", "location", "unresolved"}
    if field_name in _PERSON_REF_FIELDS:
        return category == "character" or (
            category == "unresolved" and mention_kind in {"person", "unknown"}
        )
    if field_name in _LOCATION_REF_FIELDS:
        return category == "location" or (
            category == "unresolved" and mention_kind in {"location", "unknown"}
        )
    if field_name == _UNRESOLVED_POSSIBLE_REF_FIELD:
        # unresolved mentions may only point at concrete character/location
        # candidates; unresolved->unresolved chains are forbidden.
        return category in {"character", "location"}
    return False


def _wrong_type_code(field_name: str) -> str:
    return (
        A3_INVALID_UNRESOLVED_REFERENCE
        if field_name == _UNRESOLVED_POSSIBLE_REF_FIELD
        else A3_LOCAL_REF_WRONG_TYPE
    )


def _validate_candidate_refs(
    candidate: Any,
    field_name: str,
    index: int,
    id_category: dict[str, str],
    id_mention_kind: dict[str, str],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    base = (field_name, index)
    for ref_field, ref_values in _candidate_ref_fields(candidate):
        for ref_position, ref in enumerate(ref_values):
            if ref not in id_category:
                findings.append(
                    _finding(
                        A3_LOCAL_REF_NOT_FOUND,
                        f"{candidate.candidate_id}.{ref_field} references "
                        f"unknown local candidate {ref!r}",
                        (*base, ref_field, ref_position),
                    )
                )
                continue
            category = id_category[ref]
            mention_kind = id_mention_kind.get(ref)
            if not _ref_is_allowed(ref_field, category, mention_kind):
                findings.append(
                    _finding(
                        _wrong_type_code(ref_field),
                        f"{candidate.candidate_id}.{ref_field} reference "
                        f"{ref!r} has disallowed candidate type {category!r}",
                        (*base, ref_field, ref_position),
                    )
                )
    return findings


def _validate_relationship_self_reference(
    relationships: tuple[RelationshipCandidate, ...],
    id_category: dict[str, str],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for index, relationship in enumerate(relationships):
        if (
            relationship.source_ref == relationship.target_ref
            and relationship.source_ref in id_category
        ):
            findings.append(
                _finding(
                    A3_RELATIONSHIP_SELF_REFERENCE,
                    f"relationship {relationship.candidate_id} source and "
                    f"target both resolve to {relationship.source_ref!r}",
                    ("relationships", index, "source_ref"),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Evidence / source validation
# ---------------------------------------------------------------------------


def _validate_candidate_evidence(
    candidate: Any,
    field_name: str,
    index: int,
    paragraph_index: dict[str, Any],
    context_ids: frozenset[str],
    ownership_ids: frozenset[str],
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    base = (field_name, index)

    for evidence_index, evidence in enumerate(candidate.evidence):
        path = (*base, "evidence", evidence_index)
        paragraph = paragraph_index.get(evidence.paragraph_id)
        if paragraph is None:
            findings.append(
                _finding(
                    A3_SOURCE_REF_NOT_FOUND,
                    f"{candidate.candidate_id} evidence references unknown "
                    f"paragraph {evidence.paragraph_id!r}",
                    (*path, "paragraph_id"),
                )
            )
            continue
        if evidence.paragraph_id not in context_ids:
            findings.append(
                _finding(
                    A3_EVIDENCE_OUTSIDE_CONTEXT,
                    f"{candidate.candidate_id} evidence paragraph "
                    f"{evidence.paragraph_id!r} is outside the chunk context",
                    (*path, "paragraph_id"),
                )
            )
        if evidence.role == "primary" and evidence.paragraph_id not in ownership_ids:
            findings.append(
                _finding(
                    A3_PRIMARY_EVIDENCE_OUTSIDE_OWNERSHIP,
                    f"{candidate.candidate_id} primary evidence paragraph "
                    f"{evidence.paragraph_id!r} is outside the ownership span",
                    (*path, "paragraph_id"),
                )
            )
        # Exact substring against the exact SourceDocument text_original; no
        # whitespace/punctuation/case/Unicode normalization is applied.
        if evidence.excerpt is not None and evidence.excerpt not in paragraph.text_original:
            findings.append(
                _finding(
                    A3_EVIDENCE_EXCERPT_MISMATCH,
                    f"{candidate.candidate_id} excerpt is not an exact "
                    f"substring of {evidence.paragraph_id!r} text_original",
                    (*path, "excerpt"),
                )
            )

    # Every candidate must own at least one primary evidence paragraph inside
    # the ownership span. Supporting context evidence does not satisfy this.
    has_ownership_primary = any(
        evidence.role == "primary" and evidence.paragraph_id in ownership_ids
        for evidence in candidate.evidence
    )
    if not has_ownership_primary:
        findings.append(
            _finding(
                A3_MISSING_PRIMARY_EVIDENCE,
                f"{candidate.candidate_id} has no primary evidence inside the "
                "ownership span",
                (base[0], base[1], "evidence"),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def _canonical_evidence(
    evidence: tuple[EvidenceRef, ...],
    paragraph_rank: dict[str, int],
) -> tuple[EvidenceRef, ...]:
    """Canonical evidence order: source paragraph order, then primary before
    supporting, then stable semantic tie-breakers (strength, then excerpt)."""

    fallback_rank = len(paragraph_rank)

    def sort_key(ref: EvidenceRef) -> tuple:
        excerpt_key = (0, "") if ref.excerpt is None else (1, ref.excerpt)
        return (
            paragraph_rank.get(ref.paragraph_id, fallback_rank),
            0 if ref.role == "primary" else 1,
            ref.strength,
            excerpt_key,
        )

    # ``sorted`` is stable, so equal keys preserve input order (idempotent).
    return tuple(sorted(evidence, key=sort_key))


def _canonical_candidate(candidate: Any, paragraph_rank: dict[str, int]) -> Any:
    if isinstance(candidate, (CharacterCandidate, LocationCandidate)):
        return replace(
            candidate,
            aliases_original=tuple(sorted(candidate.aliases_original)),
            descriptors_zh=tuple(sorted(candidate.descriptors_zh)),
            evidence=_canonical_evidence(candidate.evidence, paragraph_rank),
        )
    if isinstance(candidate, FactCandidate):
        return replace(
            candidate,
            subject_refs=tuple(sorted(candidate.subject_refs)),
            object_refs=tuple(sorted(candidate.object_refs)),
            evidence=_canonical_evidence(candidate.evidence, paragraph_rank),
        )
    if isinstance(candidate, EventCandidate):
        return replace(
            candidate,
            participant_refs=tuple(sorted(candidate.participant_refs)),
            location_refs=tuple(sorted(candidate.location_refs)),
            evidence=_canonical_evidence(candidate.evidence, paragraph_rank),
        )
    if isinstance(candidate, RelationshipCandidate):
        return replace(
            candidate,
            evidence=_canonical_evidence(candidate.evidence, paragraph_rank),
        )
    if isinstance(candidate, UnresolvedMentionCandidate):
        return replace(
            candidate,
            possible_candidate_refs=tuple(sorted(candidate.possible_candidate_refs)),
            evidence=_canonical_evidence(candidate.evidence, paragraph_rank),
        )
    raise StoryIntegrityError(
        f"cannot canonicalize unknown candidate type {type(candidate).__name__!r}"
    )


def canonicalize_candidate_payload(
    payload: CandidatePayload,
    source_document: SourceDocument,
) -> CandidatePayload:
    """Produce the deterministic canonical form of a candidate payload.

    Contract (parent A-I4 section 19): fixed top-level category order; each
    category sorted by numeric candidate-id suffix; the identified string
    collections sorted with one deterministic rule (plain lexicographic); and
    evidence ordered by source paragraph order, then ``primary`` before
    ``supporting``, then stable semantic tie-breakers.

    This is total (never raises on ordering) and idempotent:
    ``canonicalize(canonicalize(p)) == canonicalize(p)``. It is intended to be
    called on a payload whose evidence references resolve in ``source_document``
    (i.e. after semantic validation passes); an unresolvable paragraph id is
    ordered deterministically last rather than raising.
    """
    if not isinstance(payload, CandidatePayload):
        raise StoryIntegrityError("payload must be a CandidatePayload")
    if not isinstance(source_document, SourceDocument):
        raise StoryIntegrityError(
            "source_document must be an exact A1 SourceDocument"
        )
    paragraph_rank = {
        paragraph_id: index
        for index, paragraph_id in enumerate(_source_paragraph_ids(source_document))
    }

    def canonical_category(candidates: tuple[Any, ...]) -> tuple[Any, ...]:
        ordered = tuple(
            sorted(candidates, key=lambda candidate: _candidate_id_suffix(candidate.candidate_id))
        )
        return tuple(_canonical_candidate(candidate, paragraph_rank) for candidate in ordered)

    return CandidatePayload(
        characters=canonical_category(payload.characters),
        locations=canonical_category(payload.locations),
        facts=canonical_category(payload.facts),
        events=canonical_category(payload.events),
        relationships=canonical_category(payload.relationships),
        unresolved_mentions=canonical_category(payload.unresolved_mentions),
    )


# ---------------------------------------------------------------------------
# Validation result + entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateValidationResult:
    """In-memory semantic validation outcome for a single chunk payload.

    ``canonical_payload`` is non-``None`` only when the payload is valid (no
    blocking findings, i.e. the derived summary is ``PASS``). This is NOT a
    persisted ``ValidationReport``; A3C owns persistence and reuse.
    """

    findings: tuple[ValidationFinding, ...]
    canonical_payload: CandidatePayload | None
    summary: ValidationSummary

    @property
    def is_valid(self) -> bool:
        return self.summary.result is ValidationResult.PASS


def validate_candidate_payload(
    payload: CandidatePayload,
    source_document: SourceDocument,
    source_chunk: SourceChunk,
) -> CandidateValidationResult:
    """Deterministically validate + canonicalize a chunk candidate payload.

    Fails closed (``StoryIntegrityError``) on a structurally inconsistent exact
    SourceDocument/SourceChunk lineage. Otherwise collects deterministic A3
    semantic ``ValidationFinding``s (all ``BLOCKING``, ``owner_stage=A3``) and,
    when the payload is valid, returns its deterministic canonical form.
    """
    if not isinstance(payload, CandidatePayload):
        raise StoryIntegrityError("payload must be a CandidatePayload")
    _check_source_lineage(source_document, source_chunk)

    paragraph_index = source_document.paragraph_index()
    context_ids = frozenset(source_chunk.paragraph_ids)
    ownership_ids = _ownership_ids(source_chunk)

    # Resolve the local candidate graph once. Namespaces are disjoint per
    # category, so an id maps to at most one category.
    id_category: dict[str, str] = {}
    for field_name, _candidate_type, category in _CATEGORIES:
        for candidate in getattr(payload, field_name):
            id_category[candidate.candidate_id] = category
    id_mention_kind: dict[str, str] = {
        candidate.candidate_id: candidate.mention_kind
        for candidate in payload.unresolved_mentions
    }

    findings: list[ValidationFinding] = []

    # 1. local-ID integrity (payload-wide invariants).
    for field_name, _candidate_type, category in _CATEGORIES:
        findings.extend(
            _validate_category_ids(getattr(payload, field_name), field_name, category)
        )

    # 2. per-candidate cross-reference + evidence validation, in a fixed order.
    for field_name, _candidate_type, _category in _CATEGORIES:
        for index, candidate in enumerate(getattr(payload, field_name)):
            findings.extend(
                _validate_candidate_refs(
                    candidate, field_name, index, id_category, id_mention_kind
                )
            )
            findings.extend(
                _validate_candidate_evidence(
                    candidate,
                    field_name,
                    index,
                    paragraph_index,
                    context_ids,
                    ownership_ids,
                )
            )

    # 3. relationship self-reference.
    findings.extend(_validate_relationship_self_reference(payload.relationships, id_category))

    findings = tuple(sorted(findings, key=ValidationFinding.sort_key))
    summary = derive_validation_summary(findings)

    canonical_payload = (
        canonicalize_candidate_payload(payload, source_document)
        if summary.result is ValidationResult.PASS
        else None
    )
    return CandidateValidationResult(
        findings=findings,
        canonical_payload=canonical_payload,
        summary=summary,
    )
