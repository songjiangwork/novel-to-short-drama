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

from dataclasses import dataclass

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore

from .chunking import ChunkManifest, SourceChunk
from .consolidation import (
    ConsolidationCandidateIndex,
    ConsolidationCoverageSummary,
    IndexedEventCandidate,
    IndexedFactCandidate,
    IndexedRelationshipCandidate,
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


@dataclass(frozen=True, slots=True)
class ConsolidationPlanningResult:
    """The zero-provider A5B Phase A planning outcome.

    Carries the exact input snapshot, the A5B-built
    ``ConsolidationCandidateIndex`` (source-ordered), and the
    ``ConsolidationCoverageSummary``. This is in-memory only; Phase A performs
    no A5 persistence / CURRENT / blocking.
    """

    snapshot: ConsolidationInputSnapshot
    index: ConsolidationCandidateIndex
    coverage: ConsolidationCoverageSummary


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


def build_consolidation_planning(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    reconciliation_profile_id: str,
) -> ConsolidationPlanningResult:
    """Run the full zero-provider A5B Phase A planning (read-only).

    Resolves the exact input snapshot, builds the source-ordered
    ``ConsolidationCandidateIndex``, and computes the coverage summary. No
    provider is invoked and nothing is persisted.
    """
    snapshot = build_consolidation_input_snapshot(
        store,
        pointers,
        project_id=project_id,
        document_id=document_id,
        reconciliation_profile_id=reconciliation_profile_id,
    )
    index = build_consolidation_candidate_index(snapshot)
    coverage = _coverage_summary(index)
    return ConsolidationPlanningResult(
        snapshot=snapshot, index=index, coverage=coverage
    )


__all__ = [
    "ConsolidationCurrentMissingError",
    "ConsolidationInputSnapshot",
    "ConsolidationPlanningResult",
    "build_consolidation_candidate_index",
    "build_consolidation_input_snapshot",
    "build_consolidation_planning",
]
