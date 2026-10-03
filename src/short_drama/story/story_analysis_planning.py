"""v1.2 A6B — exact A5 input binding, deterministic A6 planning, and measurement.

This module is the A6B *planning* layer. It sits on top of the A6A domain
contracts (``story_analysis.py``) and reuses the A5F3 exact A5 CURRENT seam
(``ConsolidationPersistenceService.require_current_validated``) as the SOLE A5
input authority. It owns:

  * the read-only :class:`StoryAnalysisInputSnapshot`: the exact current-eligible
    A5 ``ConsolidationManifest`` plus the exact pinned A4/A5 leaves it binds
    (EntityMap via ``manifest.entity_map_ref``, then the exact A4 canonical
    registries / unresolved set via the EntityMap's own refs, and the six A5
    leaves via the manifest refs). The A3 upstream identity is verified equal
    across the exact A4 EntityMap and the A5 manifest. A4 CURRENT is never
    resolved independently.
  * deterministic character evidence-package planning: 100% canonical-character
    coverage via exact ref joins (no fuzzy text matching, semantic search,
    sampling, or first-N truncation);
  * deterministic event-stream / plot-window planning: the complete canonical
    event stream ordered by A5 ``narrative_order`` partitioned into windows with
    ``owned_event_ids`` / ``context_event_ids`` and the exact-one owned-event
    invariant (context overlap never confers ownership);
  * a deterministic planning identity / plan hash that binds the exact A5
    manifest ref, the versioned A6 planning policy + numeric values, the ordered
    character package hashes, the ordered event stream, the ordered window plan,
    and the compact global-index hash (backend-neutral: no provider / model /
    base_url / timeout / GPU / NP / slots / concurrency);
  * the measurement primitives (canonical JSON bytes + the versioned
    ``utf8-bytes-div3-v1`` token estimate) used by the zero-provider Alice audit.

It performs NO provider calls, NO A6 persistence, NO CURRENT mutation, and NO
creative / adaptation semantics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore, content_hash
from short_drama.artifacts.canonical import canonical_json_bytes
from short_drama.foundation import FilePointerStore

from .chunking import estimate_tokens
from .consolidation import (
    CanonicalEvent,
    CanonicalFact,
    CanonicalFactSet,
    CanonicalRelationship,
    StoryConflict,
    StoryConflictSet,
)
from .consolidation_persistence import (
    ConsolidationPersistenceService,
    a5_base_artifact_id,
    canonical_event_set_artifact_id,
    canonical_fact_set_artifact_id,
    canonical_relationship_set_artifact_id,
    load_canonical_event_set,
    load_canonical_fact_set,
    load_canonical_relationship_set,
    load_story_conflict_set,
    story_conflict_set_artifact_id,
)
from .errors import StoryAnalysisPlanningError, StoryIntegrityError
from .reconciliation import (
    CanonicalCharacterRegistry,
    CanonicalEntity,
    EntityMap,
    UnresolvedEntity,
)
from .reconciliation_persistence import (
    a4_base_artifact_id,
    canonical_character_registry_artifact_id,
    canonical_location_registry_artifact_id,
    entity_map_artifact_id,
    load_canonical_character_registry,
    load_canonical_location_registry,
    load_entity_map,
    load_unresolved_entity_set,
    unresolved_entity_set_artifact_id,
)
from .story_analysis import StoryAnalysisPlanningPolicy, StoryAnalysisProfile

# A4 EntityMap logical-id suffix (``<a4_base>.entity-map``).
_ENTITY_MAP_ID_SUFFIX = ".entity-map"


def _estimate_bytes_tokens(data: bytes) -> int:
    """Reuse the versioned ``utf8-bytes-div3-v1`` estimator over canonical bytes.

    ``estimate_tokens`` computes ``max(1, ceil(utf8_bytes / 3))`` on the text's
    UTF-8 encoding; decoding the canonical bytes (valid UTF-8) and re-encoding
    reproduces the byte length exactly, so this reuses the estimator algorithm
    (and its stable identity) without inventing a second estimator.
    """
    return estimate_tokens(data.decode("utf-8"))


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryAnalysisInputSnapshot:
    """A read-only, exact A5 + pinned A4/A5-leaf snapshot for A6 planning.

    The snapshot binds exactly one current-eligible A5 ``ConsolidationManifest``
    and the exact leaves it pins (A5 six leaves + A4 EntityMap + A4 canonical
    character / location registries + A4 unresolved set). It carries no
    provider / model / runtime material and performs no writes.
    """

    consolidation_manifest: Any  # ConsolidationManifest
    consolidation_manifest_ref: ArtifactRef
    a5_validation_report_ref: ArtifactRef
    a5_current_pointer_ref: ArtifactRef
    entity_map: EntityMap
    canonical_character_registry: CanonicalCharacterRegistry
    canonical_location_registry: Any  # CanonicalLocationRegistry
    unresolved_entity_set: Any  # UnresolvedEntitySet
    canonical_fact_set: CanonicalFactSet
    canonical_event_set: Any  # CanonicalEventSet
    canonical_relationship_set: Any  # CanonicalRelationshipSet
    story_conflict_set: StoryConflictSet

    @property
    def canonical_characters(self) -> tuple[CanonicalEntity, ...]:
        return self.canonical_character_registry.entities

    @property
    def canonical_locations(self) -> tuple[CanonicalEntity, ...]:
        return self.canonical_location_registry.entities

    @property
    def unresolved_entities(self) -> tuple[UnresolvedEntity, ...]:
        return self.unresolved_entity_set.entities

    @property
    def facts(self) -> tuple[CanonicalFact, ...]:
        return self.canonical_fact_set.facts

    @property
    def state_transitions(self) -> tuple[Any, ...]:
        return self.canonical_fact_set.state_transitions

    @property
    def events(self) -> tuple[CanonicalEvent, ...]:
        return self.canonical_event_set.events

    @property
    def relationships(self) -> tuple[CanonicalRelationship, ...]:
        return self.canonical_relationship_set.relationships

    @property
    def conflicts(self) -> tuple[StoryConflict, ...]:
        return self.story_conflict_set.conflicts


def build_story_analysis_snapshot(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    consolidation_profile_id: str,
) -> StoryAnalysisInputSnapshot:
    """Resolve the exact current-eligible A5 CURRENT and load the exact pinned
    A4/A5 leaves into a read-only snapshot. Fails closed on any missing /
    corrupt / incoherent state.

    The A5 CURRENT is consumed strictly through
    :meth:`ConsolidationPersistenceService.require_current_validated` (the sole
    A5 input seam). The A4 EntityMap is loaded through the manifest's exact
    ``entity_map_ref`` (A4 CURRENT is NEVER resolved independently), and the
    A4 canonical registries / unresolved set are loaded through the EntityMap's
    own exact refs. The A3 upstream identity is required to be exactly equal
    across the exact A4 EntityMap and the A5 manifest.
    """
    current = ConsolidationPersistenceService(store, pointers).require_current_validated(
        project_id=project_id,
        document_id=document_id,
        consolidation_profile_id=consolidation_profile_id,
    )
    manifest = current.manifest
    manifest_ref = current.manifest_ref

    # Exact A5 leaves (pinned by the manifest).
    base = a5_base_artifact_id(project_id, document_id, consolidation_profile_id)
    fact_set = load_canonical_fact_set(
        store, manifest.canonical_fact_set_ref,
        expected_artifact_id=canonical_fact_set_artifact_id(base),
    )
    event_set = load_canonical_event_set(
        store, manifest.canonical_event_set_ref,
        expected_artifact_id=canonical_event_set_artifact_id(base),
    )
    rel_set = load_canonical_relationship_set(
        store, manifest.canonical_relationship_set_ref,
        expected_artifact_id=canonical_relationship_set_artifact_id(base),
    )
    conflict_set = load_story_conflict_set(
        store, manifest.story_conflict_set_ref,
        expected_artifact_id=story_conflict_set_artifact_id(base),
    )

    # Exact A4 EntityMap (pinned by the manifest) — derived A4 base identity.
    em_id = manifest.entity_map_ref.artifact_id
    if not em_id.endswith(_ENTITY_MAP_ID_SUFFIX):
        raise StoryIntegrityError(
            "A5 manifest entity_map_ref has an unexpected logical id"
        )
    a4_base = em_id[: -len(_ENTITY_MAP_ID_SUFFIX)]
    entity_map = load_entity_map(
        store, manifest.entity_map_ref,
        expected_artifact_id=entity_map_artifact_id(a4_base),
    )
    # Bind the A4 base identity to this project/document (fail closed on a
    # cross-project / cross-document EntityMap).
    expected_a4_base = a4_base_artifact_id(
        project_id, document_id, entity_map.semantic_identity.reconciliation_profile_id
    )
    if expected_a4_base != a4_base:
        raise StoryIntegrityError(
            "A5 manifest entity_map_ref does not match the exact A4 base identity "
            "for this project/document"
        )
    # The exact upstream A3 identity must agree across the A4 EntityMap and the
    # A5 manifest (the A5 -> A4 -> A3 chain is exact, never independently latest).
    if entity_map.a3_input != manifest.upstream_identity.a3_input:
        raise StoryIntegrityError(
            "A5 manifest upstream A3 identity does not match the exact A4 "
            "EntityMap A3 input"
        )

    # Exact A4 leaves (pinned by the EntityMap), verified against the A4 base
    # identity and the EntityMap run revision.
    run_revision = manifest.entity_map_ref.revision
    char_registry = load_canonical_character_registry(
        store, entity_map.canonical_character_registry_ref,
        expected_artifact_id=canonical_character_registry_artifact_id(a4_base),
    )
    loc_registry = load_canonical_location_registry(
        store, entity_map.canonical_location_registry_ref,
        expected_artifact_id=canonical_location_registry_artifact_id(a4_base),
    )
    unresolved_set = load_unresolved_entity_set(
        store, entity_map.unresolved_entity_set_ref,
        expected_artifact_id=unresolved_entity_set_artifact_id(a4_base),
    )
    for ref in (
        entity_map.canonical_character_registry_ref,
        entity_map.canonical_location_registry_ref,
        entity_map.unresolved_entity_set_ref,
    ):
        if ref.revision != run_revision:
            raise StoryIntegrityError(
                "A4 canonical registry / unresolved set does not share the exact "
                "EntityMap run revision"
            )

    return StoryAnalysisInputSnapshot(
        consolidation_manifest=manifest,
        consolidation_manifest_ref=manifest_ref,
        a5_validation_report_ref=current.validation_report_ref,
        a5_current_pointer_ref=current.current_pointer_ref,
        entity_map=entity_map,
        canonical_character_registry=char_registry,
        canonical_location_registry=loc_registry,
        unresolved_entity_set=unresolved_set,
        canonical_fact_set=fact_set,
        canonical_event_set=event_set,
        canonical_relationship_set=rel_set,
        story_conflict_set=conflict_set,
    )


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _compact_descriptor(entity: CanonicalEntity) -> dict[str, Any]:
    return {
        "entity_id": entity.canonical_id,
        "display_name_original": entity.display_name_original,
        "aliases_original": list(entity.aliases_original),
    }


def _sorted_events(events: tuple[CanonicalEvent, ...]) -> tuple[CanonicalEvent, ...]:
    return tuple(sorted(events, key=lambda e: (e.narrative_order, e.event_id)))


# ---------------------------------------------------------------------------
# Character evidence planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CharacterEvidencePackage:
    """The complete deterministic A4/A5 evidence package for one canonical
    character, built by exact ref joins (never by fuzzy / semantic retrieval).

    Membership is structural: facts/events/relationships/transitions that
    reference the character, conflicts that touch the character's own
    fact/relationship refs, and unresolved entities that share an exact
    candidate-ref structural link with the character.
    """

    character_ref: str
    character: CanonicalEntity
    related_facts: tuple[CanonicalFact, ...]
    participating_events: tuple[CanonicalEvent, ...]
    relationships: tuple[CanonicalRelationship, ...]
    state_transitions: tuple[Any, ...]
    story_conflicts: tuple[StoryConflict, ...]
    linked_unresolved: tuple[UnresolvedEntity, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_ref": self.character_ref,
            "character": self.character.to_dict(),
            "related_facts": [f.to_dict() for f in self.related_facts],
            "participating_events": [e.to_dict() for e in self.participating_events],
            "relationships": [r.to_dict() for r in self.relationships],
            "state_transitions": [t.to_dict() for t in self.state_transitions],
            "story_conflicts": [c.to_dict() for c in self.story_conflicts],
            "linked_unresolved": [u.to_dict() for u in self.linked_unresolved],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def estimated_tokens(self) -> int:
        return _estimate_bytes_tokens(self.canonical_bytes())

    def content_hash(self) -> str:
        return content_hash(self.to_dict())


def build_character_evidence_packages(
    snapshot: StoryAnalysisInputSnapshot,
) -> tuple[CharacterEvidencePackage, ...]:
    """Build the complete deterministic evidence package for EVERY canonical
    character (100% coverage), ordered by canonical id.

    All membership is an exact ref join against the snapshot's canonical sets:
      * related_facts           : char id in subject_refs / object_refs
      * participating_events    : char id in participants
      * relationships           : char id in source / target endpoint
      * state_transitions       : char id in subject_refs
      * story_conflicts         : conflict fact_ids / relationship_ids touch the
                                  character's own fact / relationship refs
      * linked_unresolved       : shared exact candidate-ref structural link

    There is no fuzzy text matching, semantic search, sampling, or first-N
    truncation, and the model never selects the evidence universe.
    """
    facts = snapshot.facts
    events = snapshot.events
    relationships = snapshot.relationships
    transitions = snapshot.state_transitions
    conflicts = snapshot.conflicts
    unresolved = snapshot.unresolved_entities

    char_to_facts: dict[str, list] = {}
    for f in facts:
        for ref in set(f.subject_refs) | set(f.object_refs):
            char_to_facts.setdefault(ref, []).append(f)
    char_to_events: dict[str, list] = {}
    for e in events:
        for ref in set(e.participants):
            char_to_events.setdefault(ref, []).append(e)
    char_to_rels: dict[str, list] = {}
    for r in relationships:
        for ref in {r.source_entity_ref, r.target_entity_ref}:
            char_to_rels.setdefault(ref, []).append(r)
    char_to_transitions: dict[str, list] = {}
    for t in transitions:
        for ref in set(t.subject_refs):
            char_to_transitions.setdefault(ref, []).append(t)

    # candidate-ref -> unresolved entities (exact structural link index).
    unres_by_cand: dict[str, set[str]] = {}
    for u in unresolved:
        for ref in set(u.candidate_refs) | set(u.possible_candidate_refs):
            unres_by_cand.setdefault(ref, set()).add(u.unresolved_id)

    packages: list[CharacterEvidencePackage] = []
    for char in sorted(snapshot.canonical_characters, key=lambda c: c.canonical_id):
        cid = char.canonical_id
        rel_facts = sorted(
            char_to_facts.get(cid, ()), key=lambda f: f.fact_id
        )
        rel_facts_ids = frozenset(f.fact_id for f in rel_facts)
        rel_events = _sorted_events(tuple(char_to_events.get(cid, ())))
        rel_rels = sorted(
            char_to_rels.get(cid, ()), key=lambda r: r.relationship_id
        )
        rel_rels_ids = frozenset(r.relationship_id for r in rel_rels)
        rel_transitions = sorted(
            char_to_transitions.get(cid, ()),
            key=lambda t: (t.narrative_order, t.transition_id),
        )
        # Conflicts touching the character's own included fact / relationship refs.
        linked_conflicts = tuple(
            sorted(
                (
                    c
                    for c in conflicts
                    if (set(c.fact_ids) & rel_facts_ids)
                    or (set(c.relationship_ids) & rel_rels_ids)
                ),
                key=lambda c: c.conflict_id,
            )
        )
        # Unresolved entities with an exact candidate-ref structural link.
        linked_unres_ids = set()
        for cand in char.candidate_refs:
            linked_unres_ids |= unres_by_cand.get(cand, set())
        linked_unresolved = tuple(
            sorted(
                (u for u in unresolved if u.unresolved_id in linked_unres_ids),
                key=lambda u: u.unresolved_id,
            )
        )
        packages.append(
            CharacterEvidencePackage(
                character_ref=cid,
                character=char,
                related_facts=tuple(rel_facts),
                participating_events=rel_events,
                relationships=tuple(rel_rels),
                state_transitions=tuple(rel_transitions),
                story_conflicts=linked_conflicts,
                linked_unresolved=linked_unresolved,
            )
        )
    return tuple(packages)


def assert_full_character_coverage(
    snapshot: StoryAnalysisInputSnapshot,
    packages: tuple[CharacterEvidencePackage, ...],
) -> None:
    """Fail closed unless every canonical character is accounted for exactly once
    (coverage = 100%)."""
    expected = frozenset(c.canonical_id for c in snapshot.canonical_characters)
    got = [p.character_ref for p in packages]
    if len(set(got)) != len(got):
        raise StoryAnalysisPlanningError("duplicate character in evidence plan")
    if frozenset(got) != expected:
        raise StoryAnalysisPlanningError(
            "character evidence plan does not cover every canonical character "
            f"exactly once (missing={sorted(expected - set(got))!r}, "
            f"extra={sorted(set(got) - expected)!r})"
        )


# ---------------------------------------------------------------------------
# Event stream / plot-window planning
# ---------------------------------------------------------------------------


def ordered_event_stream(
    snapshot: StoryAnalysisInputSnapshot,
) -> tuple[CanonicalEvent, ...]:
    """The complete canonical event stream ordered by A5 ``narrative_order``.

    A5's narrative order is the authority; A6 never redefines it. Ties on
    ``narrative_order`` are broken deterministically by ``event_id``.
    """
    return _sorted_events(snapshot.events)


@dataclass(frozen=True, slots=True)
class PlotWindowPlan:
    """Deterministic plot-window ownership / context assignment.

    ``owned_event_ids`` are the events this window exclusively owns;
    ``context_event_ids`` are neighbouring events provided as context (they may
    overlap between windows and NEVER confer ownership).
    """

    window_id: str
    window_ordinal: int
    owned_event_ids: tuple[str, ...]
    context_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_ordinal": self.window_ordinal,
            "owned_event_ids": list(self.owned_event_ids),
            "context_event_ids": list(self.context_event_ids),
        }


def plan_plot_windows(
    ordered_events: tuple[CanonicalEvent, ...],
    *,
    owned_event_target: int,
    context_event_count: int,
) -> tuple[PlotWindowPlan, ...]:
    """Deterministically partition the narrative-ordered event stream into plot
    windows.

    Owned events form non-overlapping contiguous runs of ``owned_event_target``
    (the final window takes the remainder). Context events are the
    ``context_event_count`` events immediately preceding / following the window's
    owned run (clamped to the stream bounds), excluding the owned events. This
    guarantees the exact-one owned-event invariant while allowing context
    overlap across boundaries.
    """
    if owned_event_target < 1:
        raise StoryAnalysisPlanningError("owned_event_target must be >= 1")
    if context_event_count < 0:
        raise StoryAnalysisPlanningError("context_event_count must be >= 0")
    events = tuple(ordered_events)
    n = len(events)
    windows: list[PlotWindowPlan] = []
    for start in range(0, n, owned_event_target):
        end = min(start + owned_event_target, n)
        owned = events[start:end]
        owned_ids = frozenset(e.event_id for e in owned)
        ctx_start = max(0, start - context_event_count)
        ctx_end = min(n, end + context_event_count)
        context = [e for e in events[ctx_start:ctx_end] if e.event_id not in owned_ids]
        ordinal = len(windows) + 1
        windows.append(
            PlotWindowPlan(
                window_id=f"window_{ordinal:04d}",
                window_ordinal=ordinal,
                owned_event_ids=tuple(e.event_id for e in owned),
                context_event_ids=tuple(e.event_id for e in context),
            )
        )
    return tuple(windows)


def validate_window_ownership(
    windows: tuple[PlotWindowPlan, ...],
    all_event_ids: frozenset[str],
) -> None:
    """Fail closed unless the exact-one owned-event invariant holds.

    Requires: union(owned) == all canonical events (missing = 0); every event is
    owned exactly once (duplicate = 0); no unknown owned / context refs; window
    ids are unique and strictly ordered by ordinal.
    """
    owned_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    prev_ordinal = 0
    for w in windows:
        if w.window_id in seen_ids:
            raise StoryAnalysisPlanningError(f"duplicate window id {w.window_id!r}")
        seen_ids.add(w.window_id)
        if w.window_ordinal != prev_ordinal + 1:
            raise StoryAnalysisPlanningError(
                "window ordinals are not strictly sequential"
            )
        prev_ordinal = w.window_ordinal
        for eid in w.owned_event_ids:
            if eid not in all_event_ids:
                raise StoryAnalysisPlanningError(
                    f"window {w.window_id!r} owns unknown event {eid!r}"
                )
            owned_counts[eid] += 1
        for eid in w.context_event_ids:
            if eid not in all_event_ids:
                raise StoryAnalysisPlanningError(
                    f"window {w.window_id!r} references unknown context event "
                    f"{eid!r}"
                )
    missing = sorted(all_event_ids - set(owned_counts))
    if missing:
        raise StoryAnalysisPlanningError(
            f"{len(missing)} canonical event(s) have no owned window: {missing!r}"
        )
    duplicates = sorted(eid for eid, c in owned_counts.items() if c != 1)
    if duplicates:
        raise StoryAnalysisPlanningError(
            f"events owned by more than one window: {duplicates!r}"
        )


# ---------------------------------------------------------------------------
# Window semantic packet (deterministic join)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowPacket:
    """The deterministic evidence packet for one plot window (owned + context
    events plus the structurally relevant facts / relationships / transitions /
    conflicts and compact character / location descriptors)."""

    window_id: str
    window_ordinal: int
    owned_event_ids: tuple[str, ...]
    context_event_ids: tuple[str, ...]
    owned_events: tuple[CanonicalEvent, ...]
    context_events: tuple[CanonicalEvent, ...]
    relevant_facts: tuple[CanonicalFact, ...]
    relevant_relationships: tuple[CanonicalRelationship, ...]
    relevant_transitions: tuple[Any, ...]
    relevant_conflicts: tuple[StoryConflict, ...]
    character_descriptors: tuple[dict[str, Any], ...]
    location_descriptors: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_ordinal": self.window_ordinal,
            "owned_event_ids": list(self.owned_event_ids),
            "context_event_ids": list(self.context_event_ids),
            "owned_events": [e.to_dict() for e in self.owned_events],
            "context_events": [e.to_dict() for e in self.context_events],
            "relevant_facts": [f.to_dict() for f in self.relevant_facts],
            "relevant_relationships": [r.to_dict() for r in self.relevant_relationships],
            "relevant_transitions": [t.to_dict() for t in self.relevant_transitions],
            "relevant_conflicts": [c.to_dict() for c in self.relevant_conflicts],
            "character_descriptors": list(self.character_descriptors),
            "location_descriptors": list(self.location_descriptors),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def estimated_tokens(self) -> int:
        return _estimate_bytes_tokens(self.canonical_bytes())

    def owned_event_bytes(self) -> int:
        return sum(len(_canonical_bytes(e.to_dict())) for e in self.owned_events)

    def context_event_bytes(self) -> int:
        return sum(len(_canonical_bytes(e.to_dict())) for e in self.context_events)


def build_window_packet(
    window: PlotWindowPlan,
    snapshot: StoryAnalysisInputSnapshot,
) -> WindowPacket:
    """Deterministically join the window's owned/context events with the
    structurally relevant facts / relationships / transitions / conflicts and
    compact character / location descriptors.

    "Relevant" is a pure ref join: the window's bound entities are the union of
    participants + locations across its owned + context events; a fact is
    relevant when it references a bound entity, a relationship when an endpoint
    is bound, a transition when a subject is bound, and a conflict when it
    touches a relevant fact / relationship. The model never selects this
    evidence.
    """
    events_by_id = {e.event_id: e for e in snapshot.events}
    owned = tuple(events_by_id[eid] for eid in window.owned_event_ids)
    context = tuple(events_by_id[eid] for eid in window.context_event_ids)

    bound: set[str] = set()
    for e in (*owned, *context):
        bound.update(e.participants)
        bound.update(e.locations)
    bound = frozenset(bound)

    relevant_facts = tuple(
        sorted(
            (
                f
                for f in snapshot.facts
                if (set(f.subject_refs) | set(f.object_refs)) & bound
            ),
            key=lambda f: f.fact_id,
        )
    )
    relevant_rels = tuple(
        sorted(
            (
                r
                for r in snapshot.relationships
                if r.source_entity_ref in bound or r.target_entity_ref in bound
            ),
            key=lambda r: r.relationship_id,
        )
    )
    relevant_transitions = tuple(
        sorted(
            (t for t in snapshot.state_transitions if set(t.subject_refs) & bound),
            key=lambda t: (t.narrative_order, t.transition_id),
        )
    )
    rel_fact_ids = frozenset(f.fact_id for f in relevant_facts)
    rel_rel_ids = frozenset(r.relationship_id for r in relevant_rels)
    relevant_conflicts = tuple(
        sorted(
            (
                c
                for c in snapshot.conflicts
                if (set(c.fact_ids) & rel_fact_ids)
                or (set(c.relationship_ids) & rel_rel_ids)
            ),
            key=lambda c: c.conflict_id,
        )
    )

    char_descriptors = tuple(
        sorted(
            (_compact_descriptor(c) for c in snapshot.canonical_characters
             if c.canonical_id in bound),
            key=lambda d: d["entity_id"],
        )
    )
    loc_descriptors = tuple(
        sorted(
            (_compact_descriptor(c) for c in snapshot.canonical_locations
             if c.canonical_id in bound),
            key=lambda d: d["entity_id"],
        )
    )

    return WindowPacket(
        window_id=window.window_id,
        window_ordinal=window.window_ordinal,
        owned_event_ids=window.owned_event_ids,
        context_event_ids=window.context_event_ids,
        owned_events=owned,
        context_events=context,
        relevant_facts=relevant_facts,
        relevant_relationships=relevant_rels,
        relevant_transitions=relevant_transitions,
        relevant_conflicts=relevant_conflicts,
        character_descriptors=char_descriptors,
        location_descriptors=loc_descriptors,
    )


def build_window_packets(
    windows: tuple[PlotWindowPlan, ...],
    snapshot: StoryAnalysisInputSnapshot,
) -> tuple[WindowPacket, ...]:
    return tuple(build_window_packet(w, snapshot) for w in windows)


# ---------------------------------------------------------------------------
# Compact global index (A5-derived base)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GlobalIndexBase:
    """The compact whole-story / global-index base derived purely from the
    exact A5/A4 snapshot.

    This is the deterministic A5-derived portion of the A6E global-skeleton
    input. It does NOT include any A6C/A6D/A6E semantic output (which does not
    yet exist); those will be layered on top by the later semantic passes.
    """

    character_descriptors: tuple[dict[str, Any], ...]
    location_descriptors: tuple[dict[str, Any], ...]
    unresolved_entities: tuple[UnresolvedEntity, ...]
    event_index: tuple[dict[str, Any], ...]
    relationship_summaries: tuple[dict[str, Any], ...]
    state_transition_summaries: tuple[dict[str, Any], ...]
    conflict_summaries: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_descriptors": list(self.character_descriptors),
            "location_descriptors": list(self.location_descriptors),
            "unresolved_entities": [u.to_dict() for u in self.unresolved_entities],
            "event_index": list(self.event_index),
            "relationship_summaries": list(self.relationship_summaries),
            "state_transition_summaries": list(self.state_transition_summaries),
            "conflict_summaries": list(self.conflict_summaries),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def estimated_tokens(self) -> int:
        return _estimate_bytes_tokens(self.canonical_bytes())

    def content_hash(self) -> str:
        return content_hash(self.to_dict())


def _event_index_entry(event: CanonicalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "narrative_order": event.narrative_order,
        "summary_zh": event.summary_zh,
        "participants": list(event.participants),
        "locations": list(event.locations),
        "temporal_mode": event.temporal_mode,
    }


def _relationship_summary(rel: CanonicalRelationship) -> dict[str, Any]:
    return {
        "relationship_id": rel.relationship_id,
        "source_entity_ref": rel.source_entity_ref,
        "target_entity_ref": rel.target_entity_ref,
        "direction": rel.direction,
        "relationship_type_zh": rel.relationship_type_zh,
        "state_count": len(rel.state_history),
    }


def _state_transition_summary(t: Any) -> dict[str, Any]:
    return {
        "transition_id": t.transition_id,
        "from_fact_id": t.from_fact_id,
        "to_fact_id": t.to_fact_id,
        "subject_refs": list(t.subject_refs),
        "transition_kind": t.transition_kind,
        "narrative_order": t.narrative_order,
    }


def _conflict_summary(c: StoryConflict) -> dict[str, Any]:
    return {
        "conflict_id": c.conflict_id,
        "conflict_kind": c.conflict_kind,
        "fact_ids": list(c.fact_ids),
        "relationship_ids": list(c.relationship_ids),
        "status": c.status,
    }


def build_global_index_base(
    snapshot: StoryAnalysisInputSnapshot,
) -> GlobalIndexBase:
    """Build the compact whole-story / global-index base from the exact A5/A4
    snapshot (deterministic, no semantic output)."""
    return GlobalIndexBase(
        character_descriptors=tuple(
            sorted(
                (_compact_descriptor(c) for c in snapshot.canonical_characters),
                key=lambda d: d["entity_id"],
            )
        ),
        location_descriptors=tuple(
            sorted(
                (_compact_descriptor(c) for c in snapshot.canonical_locations),
                key=lambda d: d["entity_id"],
            )
        ),
        unresolved_entities=tuple(
            sorted(snapshot.unresolved_entities, key=lambda u: u.unresolved_id)
        ),
        event_index=tuple(_event_index_entry(e) for e in ordered_event_stream(snapshot)),
        relationship_summaries=tuple(
            _relationship_summary(r)
            for r in sorted(snapshot.relationships, key=lambda r: r.relationship_id)
        ),
        state_transition_summaries=tuple(
            _state_transition_summary(t)
            for t in sorted(
                snapshot.state_transitions,
                key=lambda t: (t.narrative_order, t.transition_id),
            )
        ),
        conflict_summaries=tuple(
            _conflict_summary(c)
            for c in sorted(snapshot.conflicts, key=lambda c: c.conflict_id)
        ),
    )


# ---------------------------------------------------------------------------
# Planning identity / plan hash
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryAnalysisPlan:
    """A deterministic A6 planning result over one exact A5 snapshot + one A6
    planning policy. The :attr:`plan_hash` is a stable canonical hash that binds
    the exact A5 manifest ref, the policy identity + numeric values, the ordered
    character package hashes, the ordered event stream, the ordered window
    plan, and the compact global-index hash. It contains no provider / model /
    runtime topology material."""

    consolidation_manifest_ref: ArtifactRef
    planning_policy: StoryAnalysisPlanningPolicy
    character_packages: tuple[CharacterEvidencePackage, ...]
    event_stream: tuple[str, ...]
    windows: tuple[PlotWindowPlan, ...]
    global_index: GlobalIndexBase
    plan_hash: str

    def plan_identity_payload(self) -> dict[str, Any]:
        return {
            "consolidation_manifest_ref": self.consolidation_manifest_ref.to_dict(),
            "planning_policy": self.planning_policy.to_dict(),
            "character_package_hashes": [p.content_hash() for p in self.character_packages],
            "event_stream": list(self.event_stream),
            "windows": [w.to_dict() for w in self.windows],
            "global_index_hash": self.global_index.content_hash(),
        }


def compute_plan_hash(
    consolidation_manifest_ref: ArtifactRef,
    planning_policy: StoryAnalysisPlanningPolicy,
    character_packages: tuple[CharacterEvidencePackage, ...],
    event_stream: tuple[str, ...],
    windows: tuple[PlotWindowPlan, ...],
    global_index: GlobalIndexBase,
) -> str:
    """Compute the stable A6 plan hash from the deterministic canonical
    planning identity (backend-neutral)."""
    payload = {
        "consolidation_manifest_ref": consolidation_manifest_ref.to_dict(),
        "planning_policy": planning_policy.to_dict(),
        "character_package_hashes": [p.content_hash() for p in character_packages],
        "event_stream": list(event_stream),
        "windows": [w.to_dict() for w in windows],
        "global_index_hash": global_index.content_hash(),
    }
    return content_hash(payload)


def build_story_analysis_plan(
    snapshot: StoryAnalysisInputSnapshot,
    planning_policy: StoryAnalysisPlanningPolicy,
) -> StoryAnalysisPlan:
    """Build the deterministic A6 plan for an exact snapshot + planning policy.

    Fails closed (never truncates) when a measured packet exceeds the frozen
    policy budget: an oversized character package, an oversized window packet,
    or an oversized global-index base raises
    :class:`StoryAnalysisPlanningError`.
    """
    packages = build_character_evidence_packages(snapshot)
    assert_full_character_coverage(snapshot, packages)

    stream = ordered_event_stream(snapshot)
    windows = plan_plot_windows(
        stream,
        owned_event_target=planning_policy.plot_window_owned_event_target,
        context_event_count=planning_policy.plot_window_context_event_count,
    )
    validate_window_ownership(windows, frozenset(e.event_id for e in stream))
    packets = build_window_packets(windows, snapshot)

    global_index = build_global_index_base(snapshot)

    # Budget enforcement (FAIL CLOSED, never truncate / sample / drop).
    for p in packages:
        tokens = p.estimated_tokens()
        if tokens > planning_policy.character_packet_max_estimated_tokens:
            raise StoryAnalysisPlanningError(
                f"character package {p.character_ref!r} estimate {tokens} tokens "
                "exceeds the frozen "
                f"character_packet_max_estimated_tokens="
                f"{planning_policy.character_packet_max_estimated_tokens}; "
                "refusing to truncate"
            )
    for pk in packets:
        tokens = pk.estimated_tokens()
        if tokens > planning_policy.plot_window_packet_max_estimated_tokens:
            raise StoryAnalysisPlanningError(
                f"plot-window packet {pk.window_id!r} estimate {tokens} tokens "
                "exceeds the frozen "
                f"plot_window_packet_max_estimated_tokens="
                f"{planning_policy.plot_window_packet_max_estimated_tokens}; "
                "refusing to truncate"
            )
    gi_tokens = global_index.estimated_tokens()
    if gi_tokens > planning_policy.global_skeleton_packet_max_estimated_tokens:
        raise StoryAnalysisPlanningError(
            f"global-index base estimate {gi_tokens} tokens exceeds the frozen "
            "global_skeleton_packet_max_estimated_tokens="
            f"{planning_policy.global_skeleton_packet_max_estimated_tokens}; "
            "refusing to truncate"
        )

    event_ids = tuple(e.event_id for e in stream)
    plan_hash = compute_plan_hash(
        snapshot.consolidation_manifest_ref,
        planning_policy,
        packages,
        event_ids,
        windows,
        global_index,
    )
    return StoryAnalysisPlan(
        consolidation_manifest_ref=snapshot.consolidation_manifest_ref,
        planning_policy=planning_policy,
        character_packages=packages,
        event_stream=event_ids,
        windows=windows,
        global_index=global_index,
        plan_hash=plan_hash,
    )


def build_story_analysis_plan_from_profile(
    snapshot: StoryAnalysisInputSnapshot,
    profile: StoryAnalysisProfile,
) -> StoryAnalysisPlan:
    """Convenience: build the plan from a full A6 :class:`StoryAnalysisProfile`
    (using its pinned :attr:`planning_policy`)."""
    return build_story_analysis_plan(snapshot, profile.planning_policy)
