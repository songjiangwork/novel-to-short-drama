"""v1.2 A4D — reconciliation identity-graph finalization.

This module is the A4D graph-finalization slice:

    ReconciliationSemanticResult (A4C)
        ↓
    identity graph (same-entity connected components via union-find,
    different_entity conflict detection, uncertainty connected groups)
        ↓
    deterministic canonical ID allocation (char_NNNN / loc_NNNN)
        ↓
    deterministic unresolved ID allocation (unres_NNNN)
        ↓
    CanonicalCharacterRegistry / CanonicalLocationRegistry /
    UnresolvedEntitySet / EntityMap entries
        ↓
    ReconciliationFinalizationResult (in-memory, carries findings)

A4D does NOT persist and does NOT write CURRENT: that is the A4D
persistence service's job (``reconciliation_persistence``). The finalization
result is an in-memory aggregate that the persistence service validates
(zero blocking findings) and then persists.

The identity graph is deterministic: it is independent of decision input
order, set/dict traversal order, and provider provenance. Backend provenance
never influences any A4D structural output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from short_drama.foundation.validation import ValidationFinding, ValidationSeverity
from short_drama.story.reconciliation import (
    CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
    RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
    UNRESOLVED_ENTITY_SET_SCHEMA_VERSION,
    A4SemanticIdentity,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
    CanonicalCharacterRegistry,
    CanonicalEntity,
    CanonicalLocationRegistry,
    EntityMapEntry,
    EntityReconciliationProfile,
    ReconciliationDecision,
    ReconciliationDecisionSet,
    ReconciliationModelError,
    UnresolvedEntity,
    UnresolvedEntitySet,
)
from short_drama.story.reconciliation_semantic import (
    ReconciliationSemanticPreparation,
    ReconciliationSemanticResult,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReconciliationFinalizationError(ReconciliationModelError):
    """A4D finalization failed (structural corruption or blocking findings).

    May carry the deterministic :class:`ValidationFinding` values that caused
    the failure (``findings``).
    """

    def __init__(self, message: str, *, findings: tuple[ValidationFinding, ...] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.findings: tuple[ValidationFinding, ...] = tuple(findings)


# ---------------------------------------------------------------------------
# Identity graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityGraph:
    """Deterministic identity graph over character/location candidates.

    * ``same_components``: same-entity connected components (union-find over
      ``same_entity`` edges), each a frozenset of candidate refs.
    * ``ref_to_component``: candidate ref -> index into ``same_components``.
    * ``conflict_decisions``: ``different_entity`` decisions whose endpoints
      fall in the same same-component (a hard reconciliation conflict).
    * ``uncertainty_groups``: connected components of the uncertainty graph,
      each as ``(member candidate refs, uncertain decision_ids)``.
    * ``resolved_components``: same-components with NO incident uncertainty
      edge (these receive canonical IDs).
    """

    same_components: tuple[frozenset[str], ...]
    ref_to_component: dict[str, int]
    conflict_decisions: tuple[ReconciliationDecision, ...]
    uncertainty_groups: tuple[tuple[frozenset[str], frozenset[str]], ...]
    resolved_components: tuple[frozenset[str], ...]


def _entry_kind(entry: CandidateEntityIndexEntry) -> str | None:
    return entry.candidate_kind if entry.candidate_kind in ("character", "location") else None


def build_identity_graph(
    index_entries: tuple[CandidateEntityIndexEntry, ...],
    decisions: tuple[ReconciliationDecision, ...],
) -> IdentityGraph:
    """Compute the deterministic identity graph (see :class:`IdentityGraph`).

    Only character/location candidates participate. Edges are added only
    between same-kind endpoints, so components and uncertainty groups are
    never mixed-type (cross-type decisions are reported as validation
    findings and never structurally merged).
    """
    ref_to_entry: dict[str, CandidateEntityIndexEntry] = {}
    ordered_refs: list[str] = []
    for entry in index_entries:
        if entry.candidate_ref in ref_to_entry:
            continue  # duplicate candidate refs are reported as findings
        if entry.candidate_kind not in ("character", "location"):
            continue
        ref_to_entry[entry.candidate_ref] = entry
        ordered_refs.append(entry.candidate_ref)

    def source_key(ref: str) -> int:
        return ref_to_entry[ref].source_order_key

    # Union-find (union by lexicographic representative for determinism).
    parent = {ref: ref for ref in ordered_refs}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    def same_kind_pair(left: str, right: str) -> bool:
        return (
            left in ref_to_entry
            and right in ref_to_entry
            and ref_to_entry[left].candidate_kind == ref_to_entry[right].candidate_kind
        )

    for decision in decisions:
        if decision.decision == "same_entity" and same_kind_pair(
            decision.left_candidate_ref, decision.right_candidate_ref
        ):
            union(decision.left_candidate_ref, decision.right_candidate_ref)

    groups: dict[str, list[str]] = {}
    for ref in ordered_refs:
        groups.setdefault(find(ref), []).append(ref)

    component_members: list[frozenset[str]] = []
    for refs in groups.values():
        ordered = sorted(refs, key=lambda r: (source_key(r), r))
        component_members.append(frozenset(ordered))
    component_members.sort(key=lambda m: (min(source_key(r) for r in m), min(m)))

    ref_to_component = {
        ref: i for i, members in enumerate(component_members) for ref in members
    }

    # Conflict: different_entity decision whose endpoints share a component.
    conflict_decisions: list[ReconciliationDecision] = []
    for decision in decisions:
        if decision.decision != "different_entity":
            continue
        if (
            decision.left_candidate_ref in ref_to_component
            and decision.right_candidate_ref in ref_to_component
            and ref_to_component[decision.left_candidate_ref]
            == ref_to_component[decision.right_candidate_ref]
        ):
            conflict_decisions.append(decision)

    # Uncertainty graph over components.
    num_components = len(component_members)
    adjacency: dict[int, set[int]] = {i: set() for i in range(num_components)}
    uncertain_by_component: dict[int, set[str]] = {i: set() for i in range(num_components)}
    for decision in decisions:
        if decision.decision != "uncertain":
            continue
        left = decision.left_candidate_ref
        right = decision.right_candidate_ref
        if left not in ref_to_component or right not in ref_to_component:
            continue
        if not (
            ref_to_entry[left].candidate_kind == ref_to_entry[right].candidate_kind
        ):
            continue  # cross-type uncertainty is reported as a finding
        a = ref_to_component[left]
        b = ref_to_component[right]
        uncertain_by_component[a].add(decision.decision_id)
        uncertain_by_component[b].add(decision.decision_id)
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)

    incident = [i for i in range(num_components) if uncertain_by_component[i]]
    visited: set[int] = set()
    uncertainty_groups: list[tuple[frozenset[str], frozenset[str]]] = []
    for start in incident:
        if start in visited:
            continue
        comp: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            stack.extend(n for n in adjacency[node] if n not in visited)
        member_refs: set[str] = set()
        decision_ids: set[str] = set()
        for member in comp:
            member_refs |= set(component_members[member])
            decision_ids |= uncertain_by_component[member]
        uncertainty_groups.append((frozenset(member_refs), frozenset(decision_ids)))
    uncertainty_groups.sort(
        key=lambda g: (min(source_key(r) for r in g[0]), min(g[0]))
    )

    resolved_components = tuple(
        component_members[i]
        for i in range(num_components)
        if not uncertain_by_component[i]
    )

    return IdentityGraph(
        same_components=tuple(component_members),
        ref_to_component=ref_to_component,
        conflict_decisions=tuple(conflict_decisions),
        uncertainty_groups=tuple(uncertainty_groups),
        resolved_components=resolved_components,
    )


# ---------------------------------------------------------------------------
# Canonical entity + unresolved construction
# ---------------------------------------------------------------------------

_A3_KIND_TO_ENTITY_KIND = {
    "unresolved_person": "person",
    "unresolved_location": "location",
    "unresolved_unknown": "unknown",
    "unresolved_other": "other",
}


def _ordered_members(
    component: frozenset[str],
    ref_to_entry: dict[str, CandidateEntityIndexEntry],
) -> tuple[str, ...]:
    return tuple(sorted(component, key=lambda r: (ref_to_entry[r].source_order_key, r)))


def _build_canonical_entity(
    component: frozenset[str],
    entity_type: str,
    entity_id: str,
    ref_to_entry: dict[str, CandidateEntityIndexEntry],
) -> CanonicalEntity:
    members = _ordered_members(component, ref_to_entry)
    first = members[0]
    seen: set[str] = set()
    ordered_names: list[str] = []
    for ref in members:
        entry = ref_to_entry[ref]
        for name in (entry.display_name_original, *entry.aliases_original):
            if name and name not in seen:
                seen.add(name)
                ordered_names.append(name)
    return CanonicalEntity(
        canonical_id=entity_id,
        entity_type=entity_type,
        candidate_refs=members,
        display_name_original=ref_to_entry[first].display_name_original,
        aliases_original=tuple(ordered_names),
        first_appearance_candidate_ref=first,
    )


def _assign_canonical_entities(
    graph: IdentityGraph,
    ref_to_entry: dict[str, CandidateEntityIndexEntry],
) -> tuple[tuple[CanonicalEntity, ...], tuple[CanonicalEntity, ...]]:
    characters: list[frozenset[str]] = []
    locations: list[frozenset[str]] = []
    for component in graph.resolved_components:
        first_ref = min(component)  # deterministic representative
        if _entry_kind(ref_to_entry[first_ref]) == "character":
            characters.append(component)
        else:
            locations.append(component)

    def sort_key(component: frozenset[str]) -> tuple[int, str]:
        members = _ordered_members(component, ref_to_entry)
        return (ref_to_entry[members[0]].source_order_key, members[0])

    characters.sort(key=sort_key)
    locations.sort(key=sort_key)

    char_entities = tuple(
        _build_canonical_entity(comp, "character", f"char_{i:04d}", ref_to_entry)
        for i, comp in enumerate(characters, start=1)
    )
    loc_entities = tuple(
        _build_canonical_entity(comp, "location", f"loc_{i:04d}", ref_to_entry)
        for i, comp in enumerate(locations, start=1)
    )
    return char_entities, loc_entities


def _build_unresolved_entities(
    graph: IdentityGraph,
    index_entries: tuple[CandidateEntityIndexEntry, ...],
) -> tuple[UnresolvedEntity, ...]:
    ref_to_entry = {e.candidate_ref: e for e in index_entries}

    items: list[dict[str, Any]] = []

    # A3 unresolved candidates -> passthrough (each its own entity).
    for entry in index_entries:
        if entry.candidate_kind not in _A3_KIND_TO_ENTITY_KIND:
            continue
        items.append(
            {
                "sort_key": (entry.source_order_key, entry.candidate_ref, "a3_passthrough"),
                "entity_kind": _A3_KIND_TO_ENTITY_KIND[entry.candidate_kind],
                "candidate_refs": (entry.candidate_ref,),
                "decision_refs": (),
                "possible_candidate_refs": entry.possible_candidate_refs,
                "first_appearance": entry.candidate_ref,
            }
        )

    # A4 uncertainty connected groups -> one entity each.
    for member_refs, decision_ids in graph.uncertainty_groups:
        members = tuple(
            sorted(member_refs, key=lambda r: (ref_to_entry[r].source_order_key, r))
        )
        kind = ref_to_entry[members[0]].candidate_kind
        items.append(
            {
                "sort_key": (
                    ref_to_entry[members[0]].source_order_key,
                    members[0],
                    "a4_uncertain",
                ),
                "entity_kind": "character" if kind == "character" else "location",
                "candidate_refs": members,
                "decision_refs": tuple(sorted(decision_ids)),
                "possible_candidate_refs": (),
                "first_appearance": members[0],
            }
        )

    items.sort(key=lambda it: it["sort_key"])
    return tuple(
        UnresolvedEntity(
            unresolved_id=f"unres_{i:04d}",
            entity_kind=it["entity_kind"],
            candidate_refs=it["candidate_refs"],  # type: ignore[arg-type]
            decision_refs=it["decision_refs"],  # type: ignore[arg-type]
            possible_candidate_refs=it["possible_candidate_refs"],  # type: ignore[arg-type]
            first_appearance_candidate_ref=it["first_appearance"],
        )
        for i, it in enumerate(items, start=1)
    )


def _build_entity_map_entries(
    index_entries: tuple[CandidateEntityIndexEntry, ...],
    char_entities: tuple[CanonicalEntity, ...],
    loc_entities: tuple[CanonicalEntity, ...],
    unresolved_entities: tuple[UnresolvedEntity, ...],
) -> tuple[EntityMapEntry, ...]:
    ref_to_canonical: dict[str, str] = {}
    for entity in (*char_entities, *loc_entities):
        for ref in entity.candidate_refs:
            ref_to_canonical[ref] = entity.canonical_id
    ref_to_unresolved: dict[str, str] = {}
    for entity in unresolved_entities:
        for ref in entity.candidate_refs:
            ref_to_unresolved[ref] = entity.unresolved_id

    entries: list[EntityMapEntry] = []
    for entry in index_entries:
        ref = entry.candidate_ref
        if ref in ref_to_canonical:
            entries.append(
                EntityMapEntry(
                    candidate_ref=ref,
                    status="resolved",
                    canonical_id=ref_to_canonical[ref],
                    unresolved_id=None,
                )
            )
        elif ref in ref_to_unresolved:
            entries.append(
                EntityMapEntry(
                    candidate_ref=ref,
                    status="unresolved",
                    canonical_id=None,
                    unresolved_id=ref_to_unresolved[ref],
                )
            )
        else:
            raise ReconciliationFinalizationError(
                f"candidate {ref!r} has no resolved/unresolved target; "
                "identity graph did not account for it"
            )
    return tuple(entries)


# ---------------------------------------------------------------------------
# Finalization result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationFinalizationResult:
    """In-memory A4D finalization aggregate.

    Carries the exact A4 output domain objects plus the deterministic
    :class:`ValidationFinding` values. Any blocking finding makes the result
    non-publishable (the persistence service refuses to write).
    """

    candidate_index: CandidateEntityIndex
    decision_set: ReconciliationDecisionSet
    canonical_character_registry: CanonicalCharacterRegistry
    canonical_location_registry: CanonicalLocationRegistry
    unresolved_entity_set: UnresolvedEntitySet
    entity_map_entries: tuple[EntityMapEntry, ...]
    findings: tuple[ValidationFinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "entity_map_entries", tuple(self.entity_map_entries))

    @property
    def has_blocking_findings(self) -> bool:
        return any(
            f.severity is ValidationSeverity.BLOCKING for f in self.findings
        )


def finalize_reconciliation(
    semantic_result: ReconciliationSemanticResult,
    *,
    validate: bool = True,
) -> ReconciliationFinalizationResult:
    """Finalize the A4C result into the A4D identity graph + registries.

    Deterministic, order-independent. Produces the canonical registries,
    unresolved set, and EntityMap entries, then (by default) runs the
    deterministic validation to attach findings.
    """
    planning = semantic_result.planning_result
    index = planning.candidate_index
    index_entries = index.entries
    decisions = semantic_result.all_decisions

    graph = build_identity_graph(index_entries, decisions)
    ref_to_entry = {e.candidate_ref: e for e in index_entries}

    char_entities, loc_entities = _assign_canonical_entities(graph, ref_to_entry)
    unresolved_entities = _build_unresolved_entities(graph, index_entries)
    entity_map_entries = _build_entity_map_entries(
        index_entries, char_entities, loc_entities, unresolved_entities
    )

    char_registry = CanonicalCharacterRegistry(
        schema_version=CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION, entities=char_entities
    )
    loc_registry = CanonicalLocationRegistry(
        schema_version=CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION, entities=loc_entities
    )
    unresolved_set = UnresolvedEntitySet(
        schema_version=UNRESOLVED_ENTITY_SET_SCHEMA_VERSION, entities=unresolved_entities
    )
    decision_set = ReconciliationDecisionSet(
        schema_version=RECONCILIATION_DECISION_SET_SCHEMA_VERSION, decisions=decisions
    )

    findings: tuple[ValidationFinding, ...] = ()
    if validate:
        from short_drama.story.reconciliation_validation import validate_finalization

        findings = validate_finalization(
            candidate_index=index,
            decision_set=decision_set,
            pair_plans=planning.pair_plans,
            graph=graph,
            canonical_character_registry=char_registry,
            canonical_location_registry=loc_registry,
            unresolved_entity_set=unresolved_set,
            entity_map_entries=entity_map_entries,
        )

    return ReconciliationFinalizationResult(
        candidate_index=index,
        decision_set=decision_set,
        canonical_character_registry=char_registry,
        canonical_location_registry=loc_registry,
        unresolved_entity_set=unresolved_set,
        entity_map_entries=entity_map_entries,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# A4 semantic identity builder
# ---------------------------------------------------------------------------


def build_a4_semantic_identity(
    profile: EntityReconciliationProfile,
    preparation: ReconciliationSemanticPreparation,
    planning_result: Any,
) -> A4SemanticIdentity:
    """Build the exact backend-neutral A4 semantic identity from the zero-provider
    A4C preparation + the authoritative profile/plan.

    The reuse-identity material is copied from the authoritative sources, so the
    invariants ``reconciliation_profile_hash == profile.profile_hash`` and
    ``plan_hash == planning_result.plan_hash`` hold by construction.
    ``semantic_request_hashes`` come verbatim from the A4C preparation, so they
    equal the exact resolve-path request hashes.
    """
    plan_hash = planning_result.plan_hash
    return A4SemanticIdentity(
        reconciliation_profile_id=profile.profile_id,
        reconciliation_profile_hash=profile.profile_hash,
        semantic_profile_id=preparation.semantic_profile_id,
        semantic_profile_hash=preparation.semantic_profile_hash,
        prompt_id=preparation.prompt_id,
        prompt_version=preparation.prompt_version,
        prompt_content_hash=preparation.prompt_content_hash,
        output_schema_id=preparation.output_schema_id,
        output_schema_version=preparation.output_schema_version,
        output_schema_hash=preparation.output_schema_hash,
        plan_hash=plan_hash,
        semantic_request_hashes=preparation.semantic_request_hashes,
    )


__all__ = [
    "IdentityGraph",
    "ReconciliationFinalizationError",
    "ReconciliationFinalizationResult",
    "build_a4_semantic_identity",
    "build_identity_graph",
    "finalize_reconciliation",
]
