"""A5B deterministic consolidation input planning (Phase A).

This module implements the *zero-provider* front half of A5B (issue #51):

  * a read-only downstream resolver for the exact current-eligible A4 CURRENT
    (``EntityMap``) via :meth:`ReconciliationPersistenceService.
    require_current_validated`;
  * exact loading of the pinned A3 input (``SourceDocument`` / ``ChunkManifest``
    / every ordered ``CandidateExtraction`` / per-chunk ``SourceChunk``) and the
    exact A4 dependent outputs (canonical character / location registries and the
    unresolved entity set);
  * local entity-ref **binding**: every A3 local ``char_* / loc_* / unres_*``
    reference is mapped to its exact A4-bound A5 entity id (``char_*`` /
    ``loc_*`` / ``unres_*``) through the ``EntityMap`` coverage universe, with
    fail-closed field-kind compatibility and registry membership;
  * construction of the in-memory ``ConsolidationCandidateIndex`` (facts /
    events / relationships) where every candidate's ``source_order_key`` is
    computed by the A5B source-order authority from the exact source location
    (never copied from the LLM payload order);
  * coverage validation (every A3 fact / event / relationship candidate is
    accounted for) and a ``ConsolidationCoverageSummary``.

This slice does NOT implement: candidate collection blocking, any LLM/provider
call, semantic decision generation, canonical-id allocation, A5 persistence or
CURRENT publication, or a CLI. After the zero-provider audit it STOPs and waits
for the architecture review to generate the A5B blocking refinement.

The module is read-only: it never writes an artifact, never sets or advances a
pointer, and never invokes a provider.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.artifacts.canonical import content_hash
from short_drama.foundation import FilePointerStore

from .chunking import ChunkManifest, SourceChunk
from .consolidation import (
    ConsolidationCandidateIndex,
    ConsolidationCandidateRef,
    ConsolidationCoverageSummary,
    ConsolidationDecisionSet,
    ConsolidationProfile,
    EventSemanticDecision,
    FactSemanticDecision,
    IndexedEventCandidate,
    IndexedFactCandidate,
    IndexedRelationshipCandidate,
    RelationshipSemanticDecision,
)
from .reconciliation_planning import (
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
)
from .errors import ConsolidationCurrentMissingError, StoryIntegrityError
from .extraction import CandidateExtraction
from .extraction_persistence import load_candidate_extraction
from .persistence import (
    load_chunk_manifest,
    load_source_chunk,
    load_source_document,
)
from .reconciliation import (
    A3InputIdentity,
    CanonicalCharacterRegistry,
    CanonicalLocationRegistry,
    EntityMap,
    EntityMapEntry,
    UnresolvedEntitySet,
)
from .reconciliation_persistence import (
    ReconciliationPersistenceService,
    ValidatedEntityMapCurrent,
    a4_base_artifact_id,
    canonical_character_registry_artifact_id,
    canonical_location_registry_artifact_id,
    load_canonical_character_registry,
    load_canonical_location_registry,
    load_unresolved_entity_set,
    unresolved_entity_set_artifact_id,
)
from .source import SourceDocument


# ---------------------------------------------------------------------------
# Field-kind authority (mirrors the A3 extraction validation authority)
# ---------------------------------------------------------------------------

# The reference-bearing fields and their frozen type restrictions, matching the
# A3 ``extraction_validation`` authority (which is the "A3 validation authority"
# the A5B binding must reproduce on the *A4-bound* ids):
#   * subject_refs / object_refs            -> any of character / location /
#                                              unresolved (any mention_kind);
#   * participant_refs / source_ref /
#     target_ref                            -> character, or unresolved
#                                              person / unknown;
#   * location_refs                         -> location, or unresolved
#                                              location / unknown.
_FIELD_ANY = "any"
_FIELD_PERSON = "person"
_FIELD_LOCATION = "location"

# For an ``unres_*`` bound id the effective mention kind is the exact
# ``UnresolvedEntity.entity_kind`` (looked up in the pinned set -- never
# inferred from the id shape). "Unknown" is dual: it is person-like AND
# location-like, matching the A3 authority. "Character" (an A4 uncertain
# character ambiguity group) is person-like; "location" is location-like.
_PERSON_UNRES_KINDS = frozenset({"character", "person", "unknown"})
_LOCATION_UNRES_KINDS = frozenset({"location", "unknown"})

# A5B source-order category ordinals (facts / events / relationships).
_CATEGORY_ORDINAL = {"fact": 1, "event": 2, "relationship": 3}


# ---------------------------------------------------------------------------
# Result objects (in-memory only -- Phase A persists nothing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsolidationInputSnapshot:
    """The exact, fully-verified A5 input set resolved read-only from A4/A3.

    Carries the validated A4 CURRENT (EntityMap + refs) and the exact A4
    dependent outputs (canonical registries + unresolved set), plus the exact
    A3 input (SourceDocument / ChunkManifest / every ordered
    ``CandidateExtraction``) and the per-chunk ``SourceChunk`` objects (in
    ``ChunkManifest.chunk_refs`` order) required for source-order derivation.

    This is an in-memory value: Phase A never serializes or persists it.
    """

    entity_map: EntityMap
    entity_map_ref: ArtifactRef
    a4_validation_report_ref: ArtifactRef
    a4_current_pointer_ref: ArtifactRef
    a3_input: A3InputIdentity
    canonical_character_registry: CanonicalCharacterRegistry
    canonical_location_registry: CanonicalLocationRegistry
    unresolved_entity_set: UnresolvedEntitySet
    source_document: SourceDocument
    chunk_manifest: ChunkManifest
    source_chunks: tuple[SourceChunk, ...]
    candidate_extractions: tuple[CandidateExtraction, ...]
    candidate_extraction_refs: tuple[ArtifactRef, ...]


# ---------------------------------------------------------------------------
# Phase B -- deterministic blocking-v1 pair planning (zero provider)
#
# The frozen post-audit A5B blocking-v1 contract is implemented here as pure,
# deterministic, bucket/index-based pair generation over the Phase A
# ``ConsolidationCandidateIndex``. It never iterates the naive N-choose-2
# cross-product per domain, never calls a provider, and never writes an
# artifact or pointer. The ``needs_semantic_decision`` pairs form the semantic
# input stream for #52/#53 (no prompt rendering, provider call, or result
# parsing is performed here).
# ---------------------------------------------------------------------------

#: The frozen A5B blocking policy id (from profiles/consolidation_v1.yaml).
A5B_BLOCKING_POLICY_ID = "consolidation-blocking-v1"
#: The frozen deterministic text-normalization policy id.
TEXT_NORMALIZATION_POLICY_ID = "a5-text-normalization-v1"
#: The frozen exact-safe auto-same policy id.
EXACT_SAFE_POLICY_ID = "a5-exact-safe-v1"
#: The frozen deterministic pair-planning policy id.
PLANNING_POLICY_ID = "a5-pair-planning-v1"
#: The frozen deterministic method for exact-safe auto-same decisions.
DETERMINISTIC_METHOD = "deterministic"

# -- Frozen atomic signal vocabularies (section 8) ---------------------------------------
# Each candidate domain's pair plans carry the COMPLETE set of frozen atomic signals that
# hold for that pair (a lexical-sorted, unique tuple) -- NOT a single "generator label".
# Production atomic signals are frozen and exhaustive per domain:
FACT_SIGNALS = frozenset(
    {
        "same_fact_type",
        "exact_normalized_statement",
        "subject_overlap",
        "object_overlap",
        "bound_entity_overlap",
        "evidence_paragraph_overlap",
        "same_chunk",
        "adjacent_chunk",
        "exact_safe_key",
    }
)
EVENT_SIGNALS = frozenset(
    {
        "exact_normalized_summary",
        "participant_overlap",
        "location_overlap",
        "bound_entity_overlap",
        "evidence_paragraph_overlap",
        "same_chunk",
        "adjacent_chunk",
        "exact_safe_key",
    }
)
RELATIONSHIP_SIGNALS = frozenset(
    {
        "same_endpoint_group",
        "exact_normalized_relationship_type",
        "state_equal",
        "evidence_paragraph_overlap",
        "same_chunk",
        "adjacent_chunk",
        "exact_safe_key",
    }
)

#: All Unicode whitespace (collapsed to a single ASCII space by normalization).
_WS_RE = re.compile(r"\s+")


def normalize_consolidation_text(value: str) -> str:
    """Deterministic A5 text normalization (``a5-text-normalization-v1``).

    Unicode NFKC -> casefold -> strip -> collapse internal whitespace to a
    single ASCII space. No punctuation removal, no stemming, no synonym
    mapping, no fuzzy threshold, no edit distance, no embedding, and no LLM.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.casefold()
    text = text.strip()
    return _WS_RE.sub(" ", text)


def _require_exact_safe_text(value: str, name: str) -> str:
    """Fail closed if an exact-safe required text field is empty/missing."""
    if value is None or value.strip() == "":
        raise StoryIntegrityError(
            f"exact-safe: {name!r} must be a non-empty text value"
        )
    return value


def _endpoint_identity_key(
    source_ref: str, target_ref: str, direction: str
) -> tuple:
    """The A5 relationship endpoint identity key (direction-aware, canonical).

    The frozen A5B relationship endpoint identity (section 9.1):

    * directed  -> ``("directed", source, target)``  (order is authoritative,
      so A->B != B->A);
    * symmetric -> ``("symmetric", min(source, target), max(source, target))``
      (canonical endpoint order, so A/B == B/A);
    * unknown   -> ``("unknown", source, target)``  (order is authoritative,
      so A->B != B->A; an unknown direction is NOT evidence of symmetry).

    This is the single production authority for both the relationship blocking
    group and the relationship exact-safe key -- there is no separate unordered
    endpoint grouping.
    """
    if direction == "directed":
        return ("directed", source_ref, target_ref)
    if direction == "symmetric":
        lo, hi = (source_ref, target_ref) if source_ref <= target_ref else (
            target_ref, source_ref
        )
        return ("symmetric", lo, hi)
    if direction == "unknown":
        return ("unknown", source_ref, target_ref)
    raise StoryIntegrityError(
        f"exact-safe: unknown relationship direction {direction!r}"
    )


def fact_exact_safe_key(candidate: IndexedFactCandidate) -> tuple:
    """Fact exact-safe auto-same key (exact-safe-v1).

    ``(fact_type, normalized statement, exact subject_refs tuple, exact
    object_refs tuple)``.
    """
    _require_exact_safe_text(candidate.fact_type, "fact_type")
    _require_exact_safe_text(candidate.statement_zh, "statement_zh")
    return (
        candidate.fact_type,
        normalize_consolidation_text(candidate.statement_zh),
        tuple(candidate.subject_refs),
        tuple(candidate.object_refs),
    )


def event_exact_safe_key(candidate: IndexedEventCandidate) -> tuple:
    """Event exact-safe auto-same key (exact-safe-v1).

    ``(normalized summary, exact participants tuple, exact locations tuple,
    temporal_mode)``.
    """
    _require_exact_safe_text(candidate.summary_zh, "summary_zh")
    return (
        normalize_consolidation_text(candidate.summary_zh),
        tuple(candidate.participants),
        tuple(candidate.locations),
        candidate.temporal_mode,
    )


def relationship_exact_safe_key(candidate: IndexedRelationshipCandidate) -> tuple:
    """Relationship exact-safe auto-same key (exact-safe-v1).

    ``(endpoint_identity_key, normalized type, normalized optional state)``.
    ``None`` and the empty string are distinct for the optional state.
    """
    _require_exact_safe_text(
        candidate.relationship_type_zh, "relationship_type_zh"
    )
    endpoint_key = _endpoint_identity_key(
        candidate.source_entity_ref,
        candidate.target_entity_ref,
        candidate.direction,
    )
    normalized_state = (
        None
        if candidate.state_zh is None
        else normalize_consolidation_text(candidate.state_zh)
    )
    return (
        endpoint_key,
        normalize_consolidation_text(candidate.relationship_type_zh),
        normalized_state,
    )


def _chunk_ordinal_from_source_order_key(source_order_key: str) -> int:
    """Recover the authoritative 1-based chunk ordinal from the source key.

    The first field of the A5B source-order key is the zero-padded chunk
    ordinal that the Phase A index construction used.
    """
    first_field = source_order_key.split(":", 1)[0]
    if not first_field.isdigit():
        raise StoryIntegrityError(
            f"source_order_key {source_order_key!r} does not start with a chunk ordinal"
        )
    ordinal = int(first_field)
    if ordinal < 1:
        raise StoryIntegrityError(f"chunk ordinal must be >= 1, got {ordinal}")
    return ordinal


def _validate_pair_plan(
    *,
    left_ref: object,
    right_ref: object,
    state: object,
    signals: object,
    namespace: str,
    vocabulary: frozenset,
    name: str,
) -> None:
    """Fail-closed pair-plan invariants (BLOCK 3).

    A pair plan must have ``left_ref < right_ref`` (canonical order), both refs
    in the correct candidate namespace, a valid state, and a non-empty,
    lexical-sorted, unique tuple of atomic signals drawn from the domain's
    frozen vocabulary. Any deviation raises :class:`StoryIntegrityError`.
    """
    if not isinstance(left_ref, str) or not isinstance(right_ref, str):
        raise StoryIntegrityError(f"{name}: refs must be strings")
    if not left_ref < right_ref:
        raise StoryIntegrityError(
            f"{name}: left_ref must be canonically < right_ref "
            f"({left_ref!r} !< {right_ref!r})"
        )
    for side, ref in (("left", left_ref), ("right", right_ref)):
        if ConsolidationCandidateRef.parse(ref).namespace != namespace:
            raise StoryIntegrityError(
                f"{name}: {side} ref {ref!r} is not a {namespace!r} candidate"
            )
    if state not in (PAIR_STATE_AUTO_SAME, PAIR_STATE_NEEDS_SEMANTIC_DECISION):
        raise StoryIntegrityError(f"{name}: invalid pair state {state!r}")
    if not isinstance(signals, tuple) or not signals:
        raise StoryIntegrityError(f"{name}: signals must be a non-empty tuple")
    if tuple(signals) != tuple(sorted(signals)):
        raise StoryIntegrityError(f"{name}: signals must be lexical-sorted")
    if len(set(signals)) != len(signals):
        raise StoryIntegrityError(f"{name}: signals must be unique")
    unknown = set(signals) - vocabulary
    if unknown:
        raise StoryIntegrityError(f"{name}: unknown signal(s) {sorted(unknown)!r}")


@dataclass(frozen=True, slots=True)
class FactPairPlan:
    """One A5 fact pair plan (``a5-pair-planning-v1``).

    ``signals`` is the COMPLETE set of frozen atomic fact signals that hold for
    this pair (lexical-sorted, unique) -- not a single generator label.
    ``left_ref < right_ref`` (canonical order) is validated fail-closed.
    """

    left_ref: str
    right_ref: str
    state: str  # PAIR_STATE_AUTO_SAME | PAIR_STATE_NEEDS_SEMANTIC_DECISION
    signals: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_pair_plan(
            left_ref=self.left_ref,
            right_ref=self.right_ref,
            state=self.state,
            signals=self.signals,
            namespace="fact",
            vocabulary=FACT_SIGNALS,
            name="FactPairPlan",
        )

    def to_dict(self) -> dict:
        return {
            "left_ref": self.left_ref,
            "right_ref": self.right_ref,
            "state": self.state,
            "signals": list(self.signals),
        }


@dataclass(frozen=True, slots=True)
class EventPairPlan:
    """One A5 event pair plan (``a5-pair-planning-v1``).

    ``signals`` is the COMPLETE set of frozen atomic event signals that hold for
    this pair (lexical-sorted, unique). ``left_ref < right_ref`` is validated
    fail-closed.
    """

    left_ref: str
    right_ref: str
    state: str
    signals: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_pair_plan(
            left_ref=self.left_ref,
            right_ref=self.right_ref,
            state=self.state,
            signals=self.signals,
            namespace="event",
            vocabulary=EVENT_SIGNALS,
            name="EventPairPlan",
        )

    def to_dict(self) -> dict:
        return {
            "left_ref": self.left_ref,
            "right_ref": self.right_ref,
            "state": self.state,
            "signals": list(self.signals),
        }


@dataclass(frozen=True, slots=True)
class RelationshipPairPlan:
    """One A5 relationship pair plan (``a5-pair-planning-v1``).

    ``signals`` is the COMPLETE set of frozen atomic relationship signals that
    hold for this pair (lexical-sorted, unique). ``left_ref < right_ref`` is
    validated fail-closed.
    """

    left_ref: str
    right_ref: str
    state: str
    signals: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_pair_plan(
            left_ref=self.left_ref,
            right_ref=self.right_ref,
            state=self.state,
            signals=self.signals,
            namespace="relationship",
            vocabulary=RELATIONSHIP_SIGNALS,
            name="RelationshipPairPlan",
        )

    def to_dict(self) -> dict:
        return {
            "left_ref": self.left_ref,
            "right_ref": self.right_ref,
            "state": self.state,
            "signals": list(self.signals),
        }


@dataclass(frozen=True, slots=True)
class ConsolidationPlanningResult:
    """The zero-provider A5B planning outcome (Phase A + Phase B).

    Carries the exact input snapshot, the A5B-built ``ConsolidationCandidateIndex``
    (source-ordered), and the ``ConsolidationCoverageSummary`` (Phase A), plus
    the deterministic blocking-v1 pair plans, the deterministic auto-same
    decision set, the frozen policy ids, and the deterministic plan hash
    (Phase B). This is in-memory only; it performs no A5 persistence, no
    CURRENT publication, and no provider call.
    """

    # Phase A
    snapshot: ConsolidationInputSnapshot
    index: ConsolidationCandidateIndex
    coverage: ConsolidationCoverageSummary
    # Phase B
    fact_pair_plans: tuple
    event_pair_plans: tuple
    relationship_pair_plans: tuple
    # The deterministic auto-same decisions are A5A-domain decisions (BLOCK 4):
    # a ``ConsolidationDecisionSet`` whose fact / event / relationship decisions
    # use ``method=deterministic`` and the frozen exact-safe reason strings.
    deterministic_decision_set: ConsolidationDecisionSet
    blocking_policy_id: str
    text_normalization_policy_id: str
    exact_safe_policy_id: str
    planning_policy_id: str
    plan_hash: str


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _bound_namespace(bound_id: str) -> str:
    """Return ``char`` / ``loc`` / ``unres`` for a bound entity id (fail closed)."""
    for prefix, ns in (("char_", "char"), ("loc_", "loc"), ("unres_", "unres")):
        if bound_id.startswith(prefix):
            return ns
    raise StoryIntegrityError(
        f"bound entity id {bound_id!r} is not in the char_/loc_/unres_ namespace"
    )


def _local_candidate_suffix(local_id: str) -> int:
    """Return the numeric suffix of a local candidate id (fail closed).

    ``cand_fact_007`` -> ``7``. The id has already been validated by the A3A
    domain contract (``cand_<category>_<digits>``), so the final ``_``-segment
    is a decimal integer.
    """
    suffix = local_id.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise StoryIntegrityError(
            f"local candidate id {local_id!r} has a non-numeric suffix"
        )
    return int(suffix)


def _source_order_key(
    *, chunk_ordinal: int, para_ordinal: int, category_ordinal: int,
    suffix: int, global_ref: str,
) -> str:
    """Compute the A5B source-order key for one consolidation candidate.

    Format (the frozen A5B source-order authority)::

        {chunk_ordinal:06d}:{paragraph_ordinal:09d}:{category_ordinal:02d}:
        {candidate_suffix:09d}:{candidate_ref}

    where ``candidate_ref`` is the exact A5 *global* candidate ref
    (``<chunk_id>:<local_candidate_id>``). All numeric fields are zero-padded
    to a fixed width, so the lexicographic order of the keys is exactly the
    total (chunk, paragraph, category, suffix) source order; the embedded
    global candidate ref is a deterministic final identity marker / tie-break
    (never copied from the LLM payload order).
    """
    if chunk_ordinal < 1 or para_ordinal < 1 or category_ordinal < 1 or suffix < 1:
        raise StoryIntegrityError(
            "source-order key ordinals/suffix must all be >= 1"
        )
    return (
        f"{chunk_ordinal:06d}:{para_ordinal:09d}:{category_ordinal:02d}"
        f":{suffix:09d}:{global_ref}"
    )


def _anchor_paragraph_ordinal(
    evidence: tuple,
    *,
    paragraph_rank: dict[str, int],
    candidate_ref: str,
) -> int:
    """Return the exact SourceDocument paragraph ordinal of a candidate anchor.

    The A5B source anchor is the earliest (by exact SourceDocument rank) among:

      * the PRIMARY evidence refs, if there is at least one; otherwise
      * all evidence refs (the frozen no-primary fallback).

    Evidence-array order is NOT the source authority: the minimum SourceDocument
    rank wins (frozen rule: earliest PRIMARY by source rank; else earliest
    evidence). Fail closed when the candidate has no evidence, or an evidence
    paragraph is absent from the exact SourceDocument (never inferred).
    """
    if not evidence:
        raise StoryIntegrityError(
            f"candidate {candidate_ref!r} has no evidence; cannot derive "
            "source-order key"
        )
    primary = [ref for ref in evidence if ref.role == "primary"]
    pool = primary if primary else list(evidence)
    best: int | None = None
    for ref in pool:
        ordinal = paragraph_rank.get(ref.paragraph_id)
        if ordinal is None:
            raise StoryIntegrityError(
                f"candidate {candidate_ref!r} evidence paragraph "
                f"{ref.paragraph_id!r} is not in the exact SourceDocument; "
                "source anchor is unresolvable"
            )
        if best is None or ordinal < best:
            best = ordinal
    assert best is not None
    return best


def _stable_dedupe(seq: tuple[str, ...]) -> tuple[str, ...]:
    """Stable exact dedupe: preserve first occurrence, drop later duplicates.

    The identity is the exact bound-ID string. This is NOT a lexical reorder
    (upstream source order may carry semantic value) and it does NOT merge
    distinct ``unres_*`` ids -- it only collapses exact duplicates produced by
    A4 aliasing (multiple local refs binding to one A4 canonical id).
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _verify_ref_closure(
    extraction: CandidateExtraction,
    local_ids: frozenset[str],
    *,
    chunk_id: str,
) -> None:
    """Verify every entity ref in an extraction closes within its own chunk.

    Re-checks the A3B cross-reference closure locally (fail closed) rather than
    trusting prior validation: every fact subject/object, event
    participant/location, and relationship source/target ref must name a local
    ``char_* / loc_* / unres_*`` candidate in the *same* extraction.
    """
    payload = extraction.candidates
    refs: list[str] = []
    for fact in payload.facts:
        refs.extend(fact.subject_refs)
        refs.extend(fact.object_refs)
    for event in payload.events:
        refs.extend(event.participant_refs)
        refs.extend(event.location_refs)
    for rel in payload.relationships:
        refs.append(rel.source_ref)
        refs.append(rel.target_ref)
    for ref in refs:
        if ref not in local_ids:
            raise StoryIntegrityError(
                f"chunk {chunk_id}: entity ref {ref!r} does not name a local "
                f"char_/loc_/unres_ candidate in the same extraction"
            )


# ---------------------------------------------------------------------------
# A3 exact-load + binding validation
# ---------------------------------------------------------------------------


def _validate_a3_input(
    *,
    project_id: str,
    document_id: str,
    a3: A3InputIdentity,
    source_document: SourceDocument,
    chunk_manifest: ChunkManifest,
    source_chunks: tuple[SourceChunk, ...],
    extractions: tuple[CandidateExtraction, ...],
) -> None:
    """Validate the exact A3 input pinned by the EntityMap (fail closed).

    Checks project/document identity, the exact SourceDocument / ChunkManifest
    refs, the ordered CandidateExtraction set (refs + identity + chunk
    position consistency), and per-extraction A3 ref closure.
    """
    if source_document.project_id != project_id or source_document.document_id != document_id:
        raise StoryIntegrityError(
            "exact SourceDocument project/document does not match the requested A5 input"
        )
    if chunk_manifest.project_id != project_id or chunk_manifest.document_id != document_id:
        raise StoryIntegrityError(
            "exact ChunkManifest project/document does not match the requested A5 input"
        )
    if chunk_manifest.source_document_ref != a3.source_document_ref:
        raise StoryIntegrityError(
            "ChunkManifest.source_document_ref does not match the A3 input SourceDocument ref"
        )

    if len(extractions) != len(a3.candidate_extraction_refs):
        raise StoryIntegrityError(
            "A3 candidate_extraction count does not match the pinned A3 input identity"
        )
    if len(source_chunks) != len(chunk_manifest.chunk_refs):
        raise StoryIntegrityError(
            "loaded SourceChunk count does not match ChunkManifest.chunk_refs"
        )

    for position, (ref, extraction) in enumerate(
        zip(a3.candidate_extraction_refs, extractions), start=1
    ):
        label = f"candidate_extraction[{position}]"
        if (
            extraction.project_id != project_id
            or extraction.document_id != document_id
        ):
            raise StoryIntegrityError(f"{label} project/document does not match the A5 input")
        if extraction.source_document_ref != a3.source_document_ref:
            raise StoryIntegrityError(f"{label} source_document_ref does not match the A3 input")
        if (
            extraction.extraction_profile_id != a3.extraction_profile_id
            or extraction.extraction_profile_hash != a3.extraction_profile_hash
        ):
            raise StoryIntegrityError(f"{label} extraction-profile identity does not match the A3 input")
        if extraction.chunk_profile_id != chunk_manifest.profile.profile_id:
            raise StoryIntegrityError(f"{label} chunk_profile_id does not match the ChunkManifest profile")

        # Position-consistent chunk identity: the extraction at this position
        # must reference the exact chunk artifact at this position, and the
        # authoritative per-chunk SourceChunk must agree on the chunk id.
        chunk_ref = chunk_manifest.chunk_refs[position - 1]
        if extraction.source_chunk_ref != chunk_ref:
            raise StoryIntegrityError(
                f"{label} source_chunk_ref does not match ChunkManifest.chunk_refs[{position}]"
            )
        source_chunk = source_chunks[position - 1]
        if source_chunk.chunk_id != extraction.chunk_id:
            raise StoryIntegrityError(
                f"{label} chunk_id does not match the authoritative SourceChunk chunk_id"
            )
        if source_chunk.source_document_ref != a3.source_document_ref:
            raise StoryIntegrityError(
                f"{label} SourceChunk.source_document_ref does not match the A3 input"
            )

        # Per-extraction A3 ref closure (fail closed; re-verified, not trusted).
        payload = extraction.candidates
        local_ids = frozenset(
            [c.candidate_id for c in payload.characters]
            + [c.candidate_id for c in payload.locations]
            + [c.candidate_id for c in payload.unresolved_mentions]
        )
        _verify_ref_closure(extraction, local_ids, chunk_id=extraction.chunk_id)


def _validate_a4_entity_binding(
    *,
    entity_map: EntityMap,
    extractions: tuple[CandidateExtraction, ...],
    canonical_character_registry: CanonicalCharacterRegistry,
    canonical_location_registry: CanonicalLocationRegistry,
    unresolved_entity_set: UnresolvedEntitySet,
) -> dict[str, EntityMapEntry]:
    """Validate the A4 EntityMap coverage universe against the exact A3 input.

    The ``EntityMap`` must form a partition of *exactly* the A3
    ``char_* + loc_* + unres_*`` candidate universe (one entry per global
    candidate ref, no gaps, no extras), and every entry must resolve to a
    canonical registry id or an unresolved-set id of a consistent namespace.

    Returns the ``global_candidate_ref -> EntityMapEntry`` binding map used for
    local-ref binding.
    """
    char_ids = {e.canonical_id for e in canonical_character_registry.entities}
    loc_ids = {e.canonical_id for e in canonical_location_registry.entities}
    unres_ids = {e.unresolved_id for e in unresolved_entity_set.entities}

    binding: dict[str, EntityMapEntry] = {}
    for entry in entity_map.entries:
        if entry.candidate_ref in binding:
            raise StoryIntegrityError(
                f"duplicate EntityMap entry for candidate ref {entry.candidate_ref!r}"
            )
        binding[entry.candidate_ref] = entry
        if entry.status == "resolved":
            if entry.canonical_id is None or entry.unresolved_id is not None:
                raise StoryIntegrityError(
                    f"resolved EntityMap entry {entry.candidate_ref!r} has no "
                    f"canonical_id (or also carries an unresolved_id)"
                )
            if entry.canonical_id.startswith("char_"):
                if entry.canonical_id not in char_ids:
                    raise StoryIntegrityError(
                        f"EntityMap entry {entry.candidate_ref!r} resolves to "
                        f"unknown canonical char id {entry.canonical_id!r}"
                    )
            elif entry.canonical_id.startswith("loc_"):
                if entry.canonical_id not in loc_ids:
                    raise StoryIntegrityError(
                        f"EntityMap entry {entry.candidate_ref!r} resolves to "
                        f"unknown canonical loc id {entry.canonical_id!r}"
                    )
            else:
                raise StoryIntegrityError(
                    f"EntityMap entry {entry.candidate_ref!r} resolves to a "
                    f"non-canonical id {entry.canonical_id!r}"
                )
        elif entry.status == "unresolved":
            if entry.unresolved_id is None or entry.canonical_id is not None:
                raise StoryIntegrityError(
                    f"unresolved EntityMap entry {entry.candidate_ref!r} has no "
                    f"unresolved_id (or also carries a canonical_id)"
                )
            if entry.unresolved_id not in unres_ids:
                raise StoryIntegrityError(
                    f"EntityMap entry {entry.candidate_ref!r} references "
                    f"unknown unresolved id {entry.unresolved_id!r}"
                )
        else:
            raise StoryIntegrityError(
                f"EntityMap entry {entry.candidate_ref!r} has invalid status {entry.status!r}"
            )

    # The coverage universe must equal the exact A3 char/loc/unres candidates.
    expected: dict[str, None] = {}
    for extraction in extractions:
        payload = extraction.candidates
        for candidate in (
            list(payload.characters)
            + list(payload.locations)
            + list(payload.unresolved_mentions)
        ):
            global_ref = f"{extraction.chunk_id}:{candidate.candidate_id}"
            if global_ref in expected:
                raise StoryIntegrityError(
                    f"duplicate A3 candidate global ref {global_ref!r}"
                )
            expected[global_ref] = None

    if set(expected) != set(binding):
        missing = sorted(set(expected) - set(binding))
        extra = sorted(set(binding) - set(expected))
        raise StoryIntegrityError(
            "EntityMap coverage universe does not exactly partition the A3 "
            f"char/loc/unres candidates; missing={missing!r} extra={extra!r}"
        )

    return binding


# ---------------------------------------------------------------------------
# Local entity-ref binding
# ---------------------------------------------------------------------------


def _bind_local_ref(
    *,
    chunk_id: str,
    local_ref: str,
    field: str,
    binding: dict[str, EntityMapEntry],
    unres_kinds: dict[str, str],
) -> str:
    """Bind one A3 local entity ref to its exact A4-bound A5 entity id.

    Fails closed when the ref has no EntityMap entry, the resolved/unresolved
    target is missing, the id is not a member of the matching registry/set, or
    the bound id is not a legal target for the given field (field-kind
    compatibility via the exact UnresolvedEntitySet authority).
    """
    global_ref = f"{chunk_id}:{local_ref}"
    entry = binding.get(global_ref)
    if entry is None:
        raise StoryIntegrityError(
            f"chunk {chunk_id}: local ref {local_ref!r} has no EntityMap binding "
            f"(no entry for {global_ref!r})"
        )

    if entry.status == "resolved":
        bound_id = entry.canonical_id
        if bound_id is None:
            raise StoryIntegrityError(
                f"chunk {chunk_id}: ref {local_ref!r} resolves to no canonical id"
            )
    else:
        bound_id = entry.unresolved_id
        if bound_id is None:
            raise StoryIntegrityError(
                f"chunk {chunk_id}: ref {local_ref!r} resolves to no unresolved id"
            )

    namespace = _bound_namespace(bound_id)
    kind: str | None = None
    if namespace == "unres":
        kind = unres_kinds.get(bound_id)
        if kind is None:
            raise StoryIntegrityError(
                f"chunk {chunk_id}: ref {local_ref!r} binds to unknown "
                f"unresolved id {bound_id!r}"
            )

    if field == _FIELD_ANY:
        return bound_id
    if field == _FIELD_PERSON:
        legal = namespace == "char" or (
            namespace == "unres" and kind in _PERSON_UNRES_KINDS
        )
    elif field == _FIELD_LOCATION:
        legal = namespace == "loc" or (
            namespace == "unres" and kind in _LOCATION_UNRES_KINDS
        )
    else:  # pragma: no cover - defensive (field is validated by callers)
        raise StoryIntegrityError(f"unknown reference field {field!r}")

    if not legal:
        raise StoryIntegrityError(
            f"chunk {chunk_id}: ref {local_ref!r} binds to {bound_id!r} which is "
            f"not a legal target for field {field!r} "
            f"(bound kind={kind!r} if unresolved)"
        )
    return bound_id


# ---------------------------------------------------------------------------
# A5B index construction
# ---------------------------------------------------------------------------


def build_consolidation_candidate_index(
    snapshot: ConsolidationInputSnapshot,
) -> ConsolidationCandidateIndex:
    """Build the exact in-memory A5B ``ConsolidationCandidateIndex``.

    Every fact / event / relationship candidate in every exact A3
    ``CandidateExtraction`` is indexed with (a) the exact A5 global candidate
    ref (``{chunk_id}:{local_id}``), (b) the exact A4-bound entity ids, and (c)
    the A5B-derived ``source_order_key``. Each category list is then sorted by
    ``source_order_key`` (the source-order authority) and validated to be a
    unique, total, deterministic order.
    """
    binding = _validate_a4_entity_binding(
        entity_map=snapshot.entity_map,
        extractions=snapshot.candidate_extractions,
        canonical_character_registry=snapshot.canonical_character_registry,
        canonical_location_registry=snapshot.canonical_location_registry,
        unresolved_entity_set=snapshot.unresolved_entity_set,
    )
    unres_kinds = {
        e.unresolved_id: e.entity_kind for e in snapshot.unresolved_entity_set.entities
    }
    # Exact SourceDocument paragraph-rank authority: 1-based position in the
    # exact SourceDocument.paragraphs (NOT the position within a SourceChunk).
    # This is the single A5B source-order paragraph authority for every
    # candidate anchor, derived from the exact A4-pinned SourceDocument.
    paragraph_rank = {
        paragraph.paragraph_id: index + 1
        for index, paragraph in enumerate(snapshot.source_document.paragraphs)
    }

    facts: list[IndexedFactCandidate] = []
    events: list[IndexedEventCandidate] = []
    relationships: list[IndexedRelationshipCandidate] = []

    for chunk_ordinal, extraction in enumerate(
        snapshot.candidate_extractions, start=1
    ):
        chunk_id = extraction.chunk_id
        ext_ref = snapshot.candidate_extraction_refs[chunk_ordinal - 1]

        def _key(local_id: str, category_ordinal: int, evidence: tuple) -> str:
            global_ref = f"{chunk_id}:{local_id}"
            para_ordinal = _anchor_paragraph_ordinal(
                evidence,
                paragraph_rank=paragraph_rank,
                candidate_ref=global_ref,
            )
            return _source_order_key(
                chunk_ordinal=chunk_ordinal,
                para_ordinal=para_ordinal,
                category_ordinal=category_ordinal,
                suffix=_local_candidate_suffix(local_id),
                global_ref=global_ref,
            )

        for fact in extraction.candidates.facts:
            subject_refs = _stable_dedupe(
                tuple(
                    _bind_local_ref(
                        chunk_id=chunk_id, local_ref=ref, field=_FIELD_ANY,
                        binding=binding, unres_kinds=unres_kinds,
                    )
                    for ref in fact.subject_refs
                )
            )
            object_refs = _stable_dedupe(
                tuple(
                    _bind_local_ref(
                        chunk_id=chunk_id, local_ref=ref, field=_FIELD_ANY,
                        binding=binding, unres_kinds=unres_kinds,
                    )
                    for ref in fact.object_refs
                )
            )
            facts.append(
                IndexedFactCandidate(
                    global_candidate_ref=f"{chunk_id}:{fact.candidate_id}",
                    chunk_id=chunk_id,
                    local_candidate_id=fact.candidate_id,
                    source_order_key=_key(
                        fact.candidate_id, _CATEGORY_ORDINAL["fact"], fact.evidence
                    ),
                    fact_type=fact.fact_type,
                    statement_zh=fact.statement_zh,
                    subject_refs=subject_refs,
                    object_refs=object_refs,
                    evidence_strength=fact.evidence_strength,
                    evidence_refs=fact.evidence,
                    candidate_extraction_ref=ext_ref,
                )
            )

        for event in extraction.candidates.events:
            participants = _stable_dedupe(
                tuple(
                    _bind_local_ref(
                        chunk_id=chunk_id, local_ref=ref, field=_FIELD_PERSON,
                        binding=binding, unres_kinds=unres_kinds,
                    )
                    for ref in event.participant_refs
                )
            )
            locations = _stable_dedupe(
                tuple(
                    _bind_local_ref(
                        chunk_id=chunk_id, local_ref=ref, field=_FIELD_LOCATION,
                        binding=binding, unres_kinds=unres_kinds,
                    )
                    for ref in event.location_refs
                )
            )
            events.append(
                IndexedEventCandidate(
                    global_candidate_ref=f"{chunk_id}:{event.candidate_id}",
                    chunk_id=chunk_id,
                    local_candidate_id=event.candidate_id,
                    source_order_key=_key(
                        event.candidate_id, _CATEGORY_ORDINAL["event"], event.evidence
                    ),
                    summary_zh=event.summary_zh,
                    participants=participants,
                    locations=locations,
                    temporal_mode=event.temporal_mode,
                    evidence_strength=event.evidence_strength,
                    evidence_refs=event.evidence,
                    candidate_extraction_ref=ext_ref,
                )
            )

        for rel in extraction.candidates.relationships:
            source_entity_ref = _bind_local_ref(
                chunk_id=chunk_id, local_ref=rel.source_ref, field=_FIELD_PERSON,
                binding=binding, unres_kinds=unres_kinds,
            )
            target_entity_ref = _bind_local_ref(
                chunk_id=chunk_id, local_ref=rel.target_ref, field=_FIELD_PERSON,
                binding=binding, unres_kinds=unres_kinds,
            )
            relationships.append(
                IndexedRelationshipCandidate(
                    global_candidate_ref=f"{chunk_id}:{rel.candidate_id}",
                    chunk_id=chunk_id,
                    local_candidate_id=rel.candidate_id,
                    source_order_key=_key(
                        rel.candidate_id, _CATEGORY_ORDINAL["relationship"], rel.evidence
                    ),
                    source_entity_ref=source_entity_ref,
                    target_entity_ref=target_entity_ref,
                    relationship_type_zh=rel.relationship_type_zh,
                    state_zh=rel.state_zh,
                    direction=rel.direction,
                    evidence_strength=rel.evidence_strength,
                    evidence_refs=rel.evidence,
                    candidate_extraction_ref=ext_ref,
                )
            )

    # Source-order authority: sort each category by its source_order_key and
    # require a unique, total order (no duplicate keys).
    facts = _require_unique_source_order(facts)
    events = _require_unique_source_order(events)
    relationships = _require_unique_source_order(relationships)

    return ConsolidationCandidateIndex(
        schema_version=1,
        facts=tuple(facts),
        events=tuple(events),
        relationships=tuple(relationships),
    )


def _require_unique_source_order(candidates: list) -> list:
    """Sort a category's candidates by source_order_key and require uniqueness."""
    ordered = sorted(candidates, key=lambda c: c.source_order_key)
    seen: set[str] = set()
    for candidate in ordered:
        if candidate.source_order_key in seen:
            raise StoryIntegrityError(
                f"duplicate source_order_key {candidate.source_order_key!r} "
                f"(candidate {candidate.global_candidate_ref!r})"
            )
        seen.add(candidate.source_order_key)
    return ordered


def _coverage_summary(index: ConsolidationCandidateIndex) -> ConsolidationCoverageSummary:
    """Build the A5B Phase A coverage summary (candidate counts only).

    The canonical fact/event/relationship counts, uncertain-decision count, and
    story-conflict count are all ``0`` in Phase A: semantic decisions,
    canonical-id allocation, and conflict detection are later A5B-A5H slices.
    """
    return ConsolidationCoverageSummary(
        fact_candidate_count=len(index.facts),
        event_candidate_count=len(index.events),
        relationship_candidate_count=len(index.relationships),
        canonical_fact_count=0,
        canonical_event_count=0,
        canonical_relationship_count=0,
        uncertain_decision_count=0,
        story_conflict_count=0,
    )


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def build_consolidation_input_snapshot(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    reconciliation_profile_id: str,
) -> ConsolidationInputSnapshot:
    """Resolve the exact A5 input snapshot read-only from the A4 CURRENT.

    A5B resolves the exact current-eligible A4 CURRENT EntityMap (via the
    read-only :meth:`require_current_validated` seam), loads the exact pinned
    A3 input and the exact A4 dependent outputs, and validates the A3/A4
    binding. Returns an in-memory :class:`ConsolidationInputSnapshot`.

    Raises :class:`ConsolidationCurrentMissingError` (via the seam) when no
    current-eligible A4 CURRENT exists, and :class:`StoryIntegrityError` when
    the exact A3 input or the A4 binding is not coherent.
    """
    persistence = ReconciliationPersistenceService(store, pointers)
    validated: ValidatedEntityMapCurrent = persistence.require_current_validated(
        project_id=project_id,
        document_id=document_id,
        reconciliation_profile_id=reconciliation_profile_id,
    )
    entity_map = validated.entity_map
    a3 = entity_map.a3_input

    # Exact A3 input load.
    source_document = load_source_document(store, a3.source_document_ref)
    chunk_manifest = load_chunk_manifest(store, a3.chunk_manifest_ref)
    source_chunks = tuple(
        load_source_chunk(store, ref) for ref in chunk_manifest.chunk_refs
    )
    extractions = tuple(
        load_candidate_extraction(store, ref) for ref in a3.candidate_extraction_refs
    )

    # Exact A4 dependent outputs (independent of the seam's earlier verification:
    # re-load against the base-derived artifact ids, fail closed on any drift).
    base = a4_base_artifact_id(project_id, document_id, reconciliation_profile_id)
    canonical_character_registry = load_canonical_character_registry(
        store,
        entity_map.canonical_character_registry_ref,
        expected_artifact_id=canonical_character_registry_artifact_id(base),
    )
    canonical_location_registry = load_canonical_location_registry(
        store,
        entity_map.canonical_location_registry_ref,
        expected_artifact_id=canonical_location_registry_artifact_id(base),
    )
    unresolved_entity_set = load_unresolved_entity_set(
        store,
        entity_map.unresolved_entity_set_ref,
        expected_artifact_id=unresolved_entity_set_artifact_id(base),
    )

    # Exact A3 + A4 binding validation (fail closed).
    _validate_a3_input(
        project_id=project_id,
        document_id=document_id,
        a3=a3,
        source_document=source_document,
        chunk_manifest=chunk_manifest,
        source_chunks=source_chunks,
        extractions=extractions,
    )
    # The A4 EntityMap must exactly partition the A3 char/loc/unres universe.
    _validate_a4_entity_binding(
        entity_map=entity_map,
        extractions=extractions,
        canonical_character_registry=canonical_character_registry,
        canonical_location_registry=canonical_location_registry,
        unresolved_entity_set=unresolved_entity_set,
    )

    return ConsolidationInputSnapshot(
        entity_map=entity_map,
        entity_map_ref=validated.entity_map_ref,
        a4_validation_report_ref=validated.validation_report_ref,
        a4_current_pointer_ref=validated.current_pointer_ref,
        a3_input=a3,
        canonical_character_registry=canonical_character_registry,
        canonical_location_registry=canonical_location_registry,
        unresolved_entity_set=unresolved_entity_set,
        source_document=source_document,
        chunk_manifest=chunk_manifest,
        source_chunks=source_chunks,
        candidate_extractions=extractions,
        candidate_extraction_refs=tuple(a3.candidate_extraction_refs),
    )


# Frozen Chinese reasons for the deterministic exact-safe auto-same decisions.
# These exact strings are the Phase-B reason authority and are embedded verbatim in
# the A5A-domain decision records (BLOCK 4 / section 10).
_FACT_REASON_ZH = "精确事实键完全一致"
_EVENT_REASON_ZH = "精确事件键完全一致"
_RELATIONSHIP_REASON_ZH = "精确关系键完全一致"


# ---------------------------------------------------------------------------
# Phase B -- pure bucket/index-based pair generation (no N-choose-2 cross product)
# ---------------------------------------------------------------------------
#
# Architecture (BLOCK 3):
#   1. Block/index generation: each domain's *blocking* signals are produced
#      from buckets/indices only (never the naive N-choose-2 cross product);
#      the explicit pair universe is the UNION of those buckets. This step is
#      instrumented (``_PairGenerationProbe``) so tests can prove the generator
#      never examines a whole-domain cross product.
#   2. Signal enrichment: for each explicit pair (O(P), never O(N^2)) the
#      COMPLETE set of frozen atomic signals is computed and stored on the pair
#      plan (lexical-sorted, unique). This is not a single "generator label".


def _ref_pair(left_ref: str, right_ref: str) -> tuple:
    """An ordered ``(left, right)`` pair with ``left < right`` (fail closed)."""
    if left_ref == right_ref:
        raise StoryIntegrityError(
            f"cannot form a pair from a single ref {left_ref!r}"
        )
    return (left_ref, right_ref) if left_ref < right_ref else (right_ref, left_ref)


class _PairGenerationProbe:
    """Instrumentation proving block generation is bucket/index-based (no N^2).

    Counts the within-bucket / adjacent-cross pair-combination examinations
    performed by the block generators. A whole-domain N-choose-2 iterator would
    examine exactly ``n*(n-1)//2`` combinations in a single bucket; bucket-based
    generation examines far fewer (summed over the small buckets it actually
    uses). Used only by structural no-N^2 tests -- the production path passes
    ``probe=None``.
    """

    def __init__(self) -> None:
        self.total_compositions = 0
        self.max_bucket_size = 0

    def record_within_bucket(self, size: int) -> None:
        self.total_compositions += size * (size - 1) // 2
        self.max_bucket_size = max(self.max_bucket_size, size)

    def record_cross(self, size_a: int, size_b: int) -> None:
        self.total_compositions += size_a * size_b
        self.max_bucket_size = max(self.max_bucket_size, size_a, size_b)


def _pairs_within_buckets(buckets, refs, *, probe=None) -> set:
    """Every unordered pair inside each bucket that has at least two members."""
    pairs = set()
    for group in buckets.values():
        count = len(group)
        if probe is not None:
            probe.record_within_bucket(count)
        for i in range(count):
            for j in range(i + 1, count):
                pairs.add(_ref_pair(refs[group[i]], refs[group[j]]))
    return pairs


def _adjacent_chunk_pairs(by_chunk, refs, *, probe=None) -> set:
    """Unordered pairs across chunk ordinals exactly one apart (distance == 1)."""
    pairs = set()
    for ord_a, group_a in by_chunk.items():
        for ord_b in (ord_a - 1, ord_a + 1):
            group_b = by_chunk.get(ord_b)
            if not group_b:
                continue
            if probe is not None:
                probe.record_cross(len(group_a), len(group_b))
            for i in group_a:
                for j in group_b:
                    pairs.add(_ref_pair(refs[i], refs[j]))
    return pairs


def _chunk_distance_signals(left_ord: int, right_ord: int) -> set:
    """The chunk-distance atomic signals (``same_chunk`` / ``adjacent_chunk``)."""
    if left_ord == right_ord:
        return {"same_chunk"}
    if abs(left_ord - right_ord) == 1:
        return {"adjacent_chunk"}
    return set()


def _fact_block_pairs(facts, ordinal, *, probe=None) -> set:
    """The explicit fact block-pair universe (F1-F4), via buckets/indices only.

    A pair is a candidate iff any of the four blocking conditions fires. Every
    condition is generated from buckets/indices (never the naive N-choose-2
    cross product) and the result is their union:

    * F1: ``exact_normalized_statement``  -- same non-empty normalized statement
    * F2: ``evidence_paragraph_overlap``  -- share a global evidence paragraph
    * F3: ``same_fact_type_bound_entity_overlap`` -- same normalized fact_type
      AND a shared bound entity
    * F4: ``same_fact_type_same_chunk``   -- same normalized fact_type AND
      same/adjacent chunk (distance <= 1)
    """
    n = len(facts)
    refs = [f.global_candidate_ref for f in facts]
    norm_stmt = [normalize_consolidation_text(f.statement_zh) for f in facts]
    norm_type = [normalize_consolidation_text(f.fact_type) for f in facts]
    bound = [
        frozenset(f.subject_refs) | frozenset(f.object_refs) for f in facts
    ]
    ev_para = [
        frozenset(ref.paragraph_id for ref in f.evidence_refs) for f in facts
    ]
    chunk_ord = [ordinal[f.global_candidate_ref] for f in facts]

    pairs = set()

    # F1: exact normalized statement.
    buckets: dict = defaultdict(list)
    for i in range(n):
        if norm_stmt[i]:
            buckets[norm_stmt[i]].append(i)
    pairs |= _pairs_within_buckets(buckets, refs, probe=probe)

    # F2: evidence paragraph overlap.
    buckets = defaultdict(list)
    for i in range(n):
        for paragraph in ev_para[i]:
            buckets[paragraph].append(i)
    pairs |= _pairs_within_buckets(buckets, refs, probe=probe)

    # F3: same normalized fact_type AND a shared bound entity.
    buckets = defaultdict(list)
    for i in range(n):
        if norm_type[i]:
            for entity in bound[i]:
                buckets[(norm_type[i], entity)].append(i)
    pairs |= _pairs_within_buckets(buckets, refs, probe=probe)

    # F4: same normalized fact_type AND same/adjacent chunk (distance <= 1).
    by_type_chunk = defaultdict(lambda: defaultdict(list))
    for i in range(n):
        if norm_type[i]:
            by_type_chunk[norm_type[i]][chunk_ord[i]].append(i)
    for by_chunk in by_type_chunk.values():
        pairs |= _pairs_within_buckets(by_chunk, refs, probe=probe)  # same chunk
        pairs |= _adjacent_chunk_pairs(by_chunk, refs, probe=probe)  # adjacent

    return pairs


def _fact_atomic_signals(
    left, right, *, exact_match: bool, left_ord: int, right_ord: int
) -> set:
    """The COMPLETE set of frozen atomic fact signals that hold for a pair."""
    signals: set = set()
    ntype_l = normalize_consolidation_text(left.fact_type)
    ntype_r = normalize_consolidation_text(right.fact_type)
    if ntype_l and ntype_l == ntype_r:
        signals.add("same_fact_type")
    stmt_l = normalize_consolidation_text(left.statement_zh)
    stmt_r = normalize_consolidation_text(right.statement_zh)
    if stmt_l and stmt_l == stmt_r:
        signals.add("exact_normalized_statement")
    subj_l, subj_r = set(left.subject_refs), set(right.subject_refs)
    obj_l, obj_r = set(left.object_refs), set(right.object_refs)
    if subj_l & subj_r:
        signals.add("subject_overlap")
    if obj_l & obj_r:
        signals.add("object_overlap")
    if (subj_l | obj_l) & (subj_r | obj_r):
        signals.add("bound_entity_overlap")
    if (
        {ref.paragraph_id for ref in left.evidence_refs}
        & {ref.paragraph_id for ref in right.evidence_refs}
    ):
        signals.add("evidence_paragraph_overlap")
    signals |= _chunk_distance_signals(left_ord, right_ord)
    if exact_match:
        signals.add("exact_safe_key")
    return signals


def _event_block_pairs(events, ordinal, *, probe=None) -> set:
    """The explicit event block-pair universe (E1-E5), via buckets/indices only.

    A pair is a candidate iff any of the five blocking conditions fires. Every
    condition is generated from buckets/indices (never the naive N-choose-2
    cross product) and the result is their union:

    * E1: ``exact_normalized_summary``  -- same non-empty normalized summary
    * E2: ``evidence_paragraph_overlap`` -- share a global evidence paragraph
    * E3: ``same_chunk``                -- same chunk
    * E4: ``adjacent_chunk_shared_entity`` -- adjacent chunk AND a shared bound
      entity (participant or location)
    * E5: ``participant_location_overlap`` -- a shared participant AND a shared
      location
    """
    n = len(events)
    refs = [e.global_candidate_ref for e in events]
    norm_summary = [normalize_consolidation_text(e.summary_zh) for e in events]
    participants = [frozenset(e.participants) for e in events]
    locations = [frozenset(e.locations) for e in events]
    bound = [participants[i] | locations[i] for i in range(n)]
    ev_para = [
        frozenset(ref.paragraph_id for ref in e.evidence_refs) for e in events
    ]
    chunk_ord = [ordinal[e.global_candidate_ref] for e in events]

    pairs = set()

    # E1: exact normalized summary.
    buckets: dict = defaultdict(list)
    for i in range(n):
        if norm_summary[i]:
            buckets[norm_summary[i]].append(i)
    pairs |= _pairs_within_buckets(buckets, refs, probe=probe)

    # E2: evidence paragraph overlap.
    buckets = defaultdict(list)
    for i in range(n):
        for paragraph in ev_para[i]:
            buckets[paragraph].append(i)
    pairs |= _pairs_within_buckets(buckets, refs, probe=probe)

    # E3: same chunk.
    buckets = defaultdict(list)
    for i in range(n):
        buckets[chunk_ord[i]].append(i)
    pairs |= _pairs_within_buckets(buckets, refs, probe=probe)

    # E4: adjacent chunk AND a shared bound entity (participant or location).
    by_entity_chunk = defaultdict(lambda: defaultdict(list))
    for i in range(n):
        for entity in bound[i]:
            by_entity_chunk[entity][chunk_ord[i]].append(i)
    for by_chunk in by_entity_chunk.values():
        pairs |= _adjacent_chunk_pairs(by_chunk, refs, probe=probe)

    # E5: a shared participant AND a shared location.
    buckets = defaultdict(list)
    for i in range(n):
        for participant in participants[i]:
            for location in locations[i]:
                buckets[(participant, location)].append(i)
    pairs |= _pairs_within_buckets(buckets, refs, probe=probe)

    return pairs


def _event_atomic_signals(
    left, right, *, exact_match: bool, left_ord: int, right_ord: int
) -> set:
    """The COMPLETE set of frozen atomic event signals that hold for a pair."""
    signals: set = set()
    summary_l = normalize_consolidation_text(left.summary_zh)
    summary_r = normalize_consolidation_text(right.summary_zh)
    if summary_l and summary_l == summary_r:
        signals.add("exact_normalized_summary")
    part_l, part_r = set(left.participants), set(right.participants)
    loc_l, loc_r = set(left.locations), set(right.locations)
    if part_l & part_r:
        signals.add("participant_overlap")
    if loc_l & loc_r:
        signals.add("location_overlap")
    if (part_l | loc_l) & (part_r | loc_r):
        signals.add("bound_entity_overlap")
    if (
        {ref.paragraph_id for ref in left.evidence_refs}
        & {ref.paragraph_id for ref in right.evidence_refs}
    ):
        signals.add("evidence_paragraph_overlap")
    signals |= _chunk_distance_signals(left_ord, right_ord)
    if exact_match:
        signals.add("exact_safe_key")
    return signals


def _relationship_block_pairs(rels, *, probe=None) -> set:
    """The explicit relationship block-pair universe, via buckets only.

    Relationship blocking-v1 is ``same_endpoint_group`` only (R1): two
    relationships are a blocking pair iff they share the same DIRECTION-AWARE
    endpoint identity key (BLOCK 1 / section 9.1). The frozen
    ``_endpoint_identity_key`` is the single production authority -- there is
    no separate unordered endpoint grouping. (No evidence overlap, chunk
    distance, type, or state bucket is used for blocking.)
    """
    refs = [r.global_candidate_ref for r in rels]
    buckets: dict = defaultdict(list)
    for i, rel in enumerate(rels):
        key = _endpoint_identity_key(
            rel.source_entity_ref, rel.target_entity_ref, rel.direction
        )
        buckets[key].append(i)
    return _pairs_within_buckets(buckets, refs, probe=probe)


def _relationship_state_component(state_zh) -> object:
    """The normalized optional-state component of a relationship exact-safe key.

    ``None`` and the empty/whitespace state are distinct (the exact-safe key
    maps ``None`` -> ``None`` and a non-``None`` value -> normalized text).
    """
    return None if state_zh is None else normalize_consolidation_text(state_zh)


def _relationship_atomic_signals(
    left, right, *, exact_match: bool, left_ord: int, right_ord: int
) -> set:
    """The COMPLETE set of frozen atomic relationship signals for a pair."""
    signals: set = set()
    left_key = _endpoint_identity_key(
        left.source_entity_ref, left.target_entity_ref, left.direction
    )
    right_key = _endpoint_identity_key(
        right.source_entity_ref, right.target_entity_ref, right.direction
    )
    # ``same_endpoint_group`` is guaranteed for every relationship pair plan
    # (it is the sole blocking signal) but is verified explicitly here.
    if left_key == right_key:
        signals.add("same_endpoint_group")
    ntype_l = normalize_consolidation_text(left.relationship_type_zh)
    ntype_r = normalize_consolidation_text(right.relationship_type_zh)
    if ntype_l and ntype_l == ntype_r:
        signals.add("exact_normalized_relationship_type")
    if _relationship_state_component(left.state_zh) == _relationship_state_component(
        right.state_zh
    ):
        signals.add("state_equal")
    if (
        {ref.paragraph_id for ref in left.evidence_refs}
        & {ref.paragraph_id for ref in right.evidence_refs}
    ):
        signals.add("evidence_paragraph_overlap")
    signals |= _chunk_distance_signals(left_ord, right_ord)
    if exact_match:
        signals.add("exact_safe_key")
    return signals


def _union_evidence(left_evidence, right_evidence) -> tuple:
    """Union of both sides' evidence: stable exact dedupe, first-appearance order."""
    seen = set()
    result = []
    for evidence in (*left_evidence, *right_evidence):
        key = (
            evidence.paragraph_id,
            evidence.role,
            evidence.strength,
            evidence.excerpt,
        )
        if key not in seen:
            seen.add(key)
            result.append(evidence)
    return tuple(result)


def _build_deterministic_decision(
    *,
    domain: str,
    left,
    right,
    decision: str,
    reason_zh: str,
    decision_cls,
):
    """Build one A5A-domain deterministic exact-safe auto-same decision (BLOCK 4).

    The decision reuses the A5A-domain semantic-decision record -- A5B does NOT
    re-declare a second decision contract. The decision id is ``dec_`` + the
    first 20 hex chars of the canonical content hash of the exact canonical
    identity material:

        {domain, left_candidate_ref, right_candidate_ref, decision, method,
         reason_zh, evidence_refs}

    No timestamp, hostname, PID, worker id, random salt, provider data, prompt
    text, or model name. ``method=deterministic``; ``prompt_id`` /
    ``prompt_version`` / ``generation_provenance`` are ``None``.
    """
    left_ref = left.global_candidate_ref
    right_ref = right.global_candidate_ref
    evidence_refs = _union_evidence(left.evidence_refs, right.evidence_refs)
    identity_material = {
        "domain": domain,
        "left_candidate_ref": left_ref,
        "right_candidate_ref": right_ref,
        "decision": decision,
        "method": DETERMINISTIC_METHOD,
        "reason_zh": reason_zh,
        "evidence_refs": [e.to_dict() for e in evidence_refs],
    }
    decision_id = "dec_" + content_hash(identity_material)[:20]
    return decision_cls(
        decision_id=decision_id,
        left_candidate_ref=left_ref,
        right_candidate_ref=right_ref,
        decision=decision,
        method=DETERMINISTIC_METHOD,
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
        prompt_id=None,
        prompt_version=None,
        generation_provenance=None,
    )


def _fact_auto_same_decision(left, right) -> FactSemanticDecision:
    """A5A-domain deterministic fact auto-same decision (``same_fact``)."""
    return _build_deterministic_decision(
        domain="fact",
        left=left,
        right=right,
        decision="same_fact",
        reason_zh=_FACT_REASON_ZH,
        decision_cls=FactSemanticDecision,
    )


def _event_auto_same_decision(left, right) -> EventSemanticDecision:
    """A5A-domain deterministic event auto-same decision (``same_event``)."""
    return _build_deterministic_decision(
        domain="event",
        left=left,
        right=right,
        decision="same_event",
        reason_zh=_EVENT_REASON_ZH,
        decision_cls=EventSemanticDecision,
    )


def _relationship_auto_same_decision(left, right) -> RelationshipSemanticDecision:
    """A5A-domain deterministic relationship auto-same decision."""
    return _build_deterministic_decision(
        domain="relationship",
        left=left,
        right=right,
        decision="same_relationship",
        reason_zh=_RELATIONSHIP_REASON_ZH,
        decision_cls=RelationshipSemanticDecision,
    )


def _plan_domain_pairs(
    candidates,
    block_pairs: set,
    *,
    ordinal,
    signal_fn,
    exact_key,
    plan_cls,
    decision_fn,
) -> tuple:
    """Assemble ordered pair plans + deterministic auto-same decisions.

    ``block_pairs`` is the set of explicit ``(left, right)`` pairs (left <
    right) produced by the bucket/index block generators. For each explicit
    pair (O(P), never O(N^2)) the COMPLETE atomic signal set is computed and
    stored on the pair plan (lexical-sorted, unique); the exact-safe key
    decides the pair state, and auto_same pairs yield A5A-domain deterministic
    decisions. Pair plans are ordered by ``(left_ref, right_ref)``. No
    ``not_compared`` pair is materialized.
    """
    by_ref = {c.global_candidate_ref: c for c in candidates}
    plans = []
    decisions = []
    for left_ref, right_ref in sorted(block_pairs):
        left = by_ref[left_ref]
        right = by_ref[right_ref]
        exact_match = exact_key(left) == exact_key(right)
        state = (
            PAIR_STATE_AUTO_SAME
            if exact_match
            else PAIR_STATE_NEEDS_SEMANTIC_DECISION
        )
        signals = tuple(
            sorted(
                signal_fn(
                    left,
                    right,
                    exact_match=exact_match,
                    left_ord=ordinal[left_ref],
                    right_ord=ordinal[right_ref],
                )
            )
        )
        plans.append(
            plan_cls(
                left_ref=left_ref,
                right_ref=right_ref,
                state=state,
                signals=signals,
            )
        )
        if exact_match:
            decisions.append(decision_fn(left, right))
    return plans, decisions


def plan_consolidation_pairs(
    index: ConsolidationCandidateIndex,
) -> tuple:
    """Deterministic blocking-v1 pair planning over the Phase A index.

    Pure and zero-provider: it reads only the ``ConsolidationCandidateIndex``
    and returns ``(fact_pair_plans, event_pair_plans, relationship_pair_plans,
    deterministic_decision_set)``. The pair plans are the semantic input stream
    (``needs_semantic_decision``) plus the deterministic ``auto_same`` pairs;
    no prompt is rendered, no provider is called, and no result is parsed.
    """
    ordinal = {
        cand.global_candidate_ref: _chunk_ordinal_from_source_order_key(
            cand.source_order_key
        )
        for cand in (*index.facts, *index.events, *index.relationships)
    }
    fact_plans, fact_decisions = _plan_domain_pairs(
        index.facts,
        _fact_block_pairs(index.facts, ordinal),
        ordinal=ordinal,
        signal_fn=_fact_atomic_signals,
        exact_key=fact_exact_safe_key,
        plan_cls=FactPairPlan,
        decision_fn=_fact_auto_same_decision,
    )
    event_plans, event_decisions = _plan_domain_pairs(
        index.events,
        _event_block_pairs(index.events, ordinal),
        ordinal=ordinal,
        signal_fn=_event_atomic_signals,
        exact_key=event_exact_safe_key,
        plan_cls=EventPairPlan,
        decision_fn=_event_auto_same_decision,
    )
    rel_plans, rel_decisions = _plan_domain_pairs(
        index.relationships,
        _relationship_block_pairs(index.relationships),
        ordinal=ordinal,
        signal_fn=_relationship_atomic_signals,
        exact_key=relationship_exact_safe_key,
        plan_cls=RelationshipPairPlan,
        decision_fn=_relationship_auto_same_decision,
    )
    # BLOCK 4: the deterministic auto-same decisions are A5A-domain decisions
    # (Fact / Event / Relationship semantic decisions) grouped in the A5A
    # ConsolidationDecisionSet -- no second decision contract is re-declared.
    decision_set = ConsolidationDecisionSet(
        schema_version=1,
        fact_decisions=tuple(fact_decisions),
        event_decisions=tuple(event_decisions),
        relationship_decisions=tuple(rel_decisions),
    )
    return (
        tuple(fact_plans),
        tuple(event_plans),
        tuple(rel_plans),
        decision_set,
    )


def build_consolidation_planning(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    reconciliation_profile_id: str,
    consolidation_profile: ConsolidationProfile,
) -> ConsolidationPlanningResult:
    """Run the full zero-provider A5B planning (Phase A + Phase B, read-only).

    Phase A: resolves the exact input snapshot, builds the source-ordered
    ``ConsolidationCandidateIndex``, and computes the coverage summary.
    Phase B: verifies the frozen blocking policy id, then runs deterministic
    blocking-v1 pair planning (bucket/index based, never the naive N-choose-2
    cross product), the exact-safe auto-same decisions, and the deterministic
    plan hash.

    No provider is invoked and nothing is persisted. The used
    ``ConsolidationProfile.blocking_policy_id`` must equal
    ``consolidation-blocking-v1`` or the planner fails closed (no silent
    fallback).
    """
    if consolidation_profile.blocking_policy_id != A5B_BLOCKING_POLICY_ID:
        raise StoryIntegrityError(
            "consolidation blocking_policy_id "
            f"{consolidation_profile.blocking_policy_id!r} does not match the "
            f"frozen A5B policy {A5B_BLOCKING_POLICY_ID!r}; refusing to plan"
        )
    snapshot = build_consolidation_input_snapshot(
        store,
        pointers,
        project_id=project_id,
        document_id=document_id,
        reconciliation_profile_id=reconciliation_profile_id,
    )
    index = build_consolidation_candidate_index(snapshot)
    coverage = _coverage_summary(index)

    (
        fact_pair_plans,
        event_pair_plans,
        relationship_pair_plans,
        deterministic_decision_set,
    ) = plan_consolidation_pairs(index)

    plan_material = {
        "blocking_policy_id": A5B_BLOCKING_POLICY_ID,
        "text_normalization_policy_id": TEXT_NORMALIZATION_POLICY_ID,
        "exact_safe_policy_id": EXACT_SAFE_POLICY_ID,
        "planning_policy_id": PLANNING_POLICY_ID,
        "candidate_index": index.to_dict(),
        "fact_pair_plans": [p.to_dict() for p in fact_pair_plans],
        "event_pair_plans": [p.to_dict() for p in event_pair_plans],
        "relationship_pair_plans": [p.to_dict() for p in relationship_pair_plans],
        "deterministic_decision_set": deterministic_decision_set.to_dict(),
    }
    plan_hash = content_hash(plan_material)

    return ConsolidationPlanningResult(
        snapshot=snapshot,
        index=index,
        coverage=coverage,
        fact_pair_plans=fact_pair_plans,
        event_pair_plans=event_pair_plans,
        relationship_pair_plans=relationship_pair_plans,
        deterministic_decision_set=deterministic_decision_set,
        blocking_policy_id=A5B_BLOCKING_POLICY_ID,
        text_normalization_policy_id=TEXT_NORMALIZATION_POLICY_ID,
        exact_safe_policy_id=EXACT_SAFE_POLICY_ID,
        planning_policy_id=PLANNING_POLICY_ID,
        plan_hash=plan_hash,
    )


__all__ = [
    "ConsolidationCurrentMissingError",
    "ConsolidationInputSnapshot",
    "ConsolidationPlanningResult",
    "ConsolidationProfile",
    "DETERMINISTIC_METHOD",
    "EVENT_SIGNALS",
    "EventPairPlan",
    "EXACT_SAFE_POLICY_ID",
    "FACT_SIGNALS",
    "FactPairPlan",
    "PAIR_STATE_AUTO_SAME",
    "PAIR_STATE_NEEDS_SEMANTIC_DECISION",
    "PLANNING_POLICY_ID",
    "RELATIONSHIP_SIGNALS",
    "RelationshipPairPlan",
    "TEXT_NORMALIZATION_POLICY_ID",
    "A5B_BLOCKING_POLICY_ID",
    "build_consolidation_candidate_index",
    "build_consolidation_input_snapshot",
    "build_consolidation_planning",
    "event_exact_safe_key",
    "fact_exact_safe_key",
    "normalize_consolidation_text",
    "plan_consolidation_pairs",
    "relationship_exact_safe_key",
]
