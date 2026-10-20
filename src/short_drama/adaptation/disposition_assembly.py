"""B1.3-B3B: assemble a reviewed local-candidate graph in memory.

This module deliberately has no provider, artifact-store, or CURRENT-pointer
dependency.  B3A remains the sole reconciliation/review authority; this code
only revalidates that authority, assigns deterministic Python-owned identities,
and lets the frozen B1.3-A plan validate the resulting graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from short_drama.artifacts import content_hash

from .disposition import (
    AdaptationMapping, AdaptedCharacter, AdaptedContinuityOverlay,
    AdaptedEndpoint, AdaptedFactDivergence, AdaptedStoryBeat, AnonymousRole,
    B13DispositionPlan, BeatParticipant, CharacterFunctionProvenance,
    ContinuityStoryState, EventDisposition, RelationshipArcRequirement,
)
from .disposition_reconciliation import (
    B13CandidateKind, B13CandidateRef, B13ReconciliationCandidate,
    B13ReconciliationDecisionKind, B13ReconciliationDecisionSet,
    B13ReconciliationError, B13ReconciliationInventory,
    B13ReconciliationReview, review_b13_reconciliation,
)
from .target import CanonicalTargetReference


class B13DispositionAssemblyError(ValueError):
    """A resolved B3A review cannot safely become a B1.3-A graph."""


_IDENTITY_KINDS = frozenset({
    B13CandidateKind.ADAPTED_CHARACTER,
    B13CandidateKind.ANONYMOUS_ROLE,
    B13CandidateKind.ADAPTED_STORY_BEAT,
})


@dataclass(frozen=True, slots=True)
class B13GlobalIdentityBinding:
    """Auditable scoped-local candidate to final Python-derived identity link."""

    candidate_ref: B13CandidateRef
    identity_key: str
    adapted_id: str


@dataclass(frozen=True, slots=True)
class B13AssemblyWindowAudit:
    """The exact accepted B2B proposal whose candidates fed assembly."""

    window_id: str
    proposal_hash: str
    request_identity: str
    candidate_refs: tuple[B13CandidateRef, ...]


@dataclass(frozen=True, slots=True)
class B13DispositionAssemblyResult:
    """Complete in-memory plan and its exact reconciliation provenance."""

    plan: B13DispositionPlan
    inventory_identity: str
    decision_identity: str
    strategy_ref: dict[str, Any]
    strategy_hash: str
    preparation_identity: str
    identity_bindings: tuple[B13GlobalIdentityBinding, ...]
    window_audit: tuple[B13AssemblyWindowAudit, ...]

    def __post_init__(self) -> None:
        if any(len(value) != 64 for value in (
            self.inventory_identity, self.decision_identity, self.strategy_hash, self.preparation_identity,
        )):
            raise B13DispositionAssemblyError("assembly identities must be SHA-256 hashes")
        if not isinstance(self.strategy_ref, dict):
            raise B13DispositionAssemblyError("assembly must retain the exact B1.2 strategy reference")
        if len({item.candidate_ref for item in self.identity_bindings}) != len(self.identity_bindings):
            raise B13DispositionAssemblyError("assembly identity bindings must be unique")
        if tuple(item.window_id for item in self.window_audit) != tuple(sorted(item.window_id for item in self.window_audit)):
            raise B13DispositionAssemblyError("assembly window audit must be deterministic")


def _decision_identity(decisions: B13ReconciliationDecisionSet) -> str:
    return content_hash({
        "inventory_identity": decisions.inventory_identity,
        "decisions": [item.to_dict() for item in decisions.decisions],
    })


def _verified_review(review: B13ReconciliationReview) -> tuple[B13ReconciliationInventory, B13ReconciliationDecisionSet]:
    if not isinstance(review, B13ReconciliationReview):
        raise B13DispositionAssemblyError("assembly requires a B13ReconciliationReview")
    try:
        # Reconstructing the contracts catches post-construction mutation as
        # well as stale inventory/decision identity before finalization.
        inventory = B13ReconciliationInventory(
            review.inventory.source_snapshot, review.inventory.strategy_ref,
            review.inventory.strategy_hash, review.inventory.preparation_identity,
            review.inventory.windows, review.inventory.candidates,
            review.inventory.findings, review.inventory.inventory_identity,
        )
        decisions = B13ReconciliationDecisionSet(
            review.decisions.inventory_identity, review.decisions.decisions,
        )
        derived = review_b13_reconciliation(inventory, decisions)
        if review.inventory != inventory or review.decisions != decisions or review != derived:
            raise B13DispositionAssemblyError("review is stale or was not derived from its exact reconciliation inputs")
        derived.require_resolved()
    except B13ReconciliationError as exc:
        raise B13DispositionAssemblyError(f"reconciliation review is invalid: {exc}") from exc
    return inventory, decisions


def _rejected_refs(decisions: B13ReconciliationDecisionSet) -> frozenset[B13CandidateRef]:
    return frozenset(
        decision.candidate_refs[0]
        for decision in decisions.decisions
        if decision.kind is B13ReconciliationDecisionKind.REJECT_CONTRIBUTION
    )


def _equivalence_components(
    candidates: Iterable[B13ReconciliationCandidate], decisions: B13ReconciliationDecisionSet,
) -> tuple[tuple[B13CandidateRef, ...], ...]:
    refs = {item.ref for item in candidates if item.ref.candidate_kind in _IDENTITY_KINDS}
    parent = {ref: ref for ref in refs}

    def find(ref: B13CandidateRef) -> B13CandidateRef:
        if parent[ref] != ref:
            parent[ref] = find(parent[ref])
        return parent[ref]

    for decision in decisions.decisions:
        if decision.kind is not B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT:
            continue
        left, right = decision.candidate_refs
        if left not in parent or right not in parent:
            raise B13DispositionAssemblyError("identity decision references a rejected or non-identity candidate")
        # B3A has already rejected contradictory KEEP_DISTINCT edges.  Union
        # only declared edges; names/source refs/order are never evidence.
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    grouped: dict[B13CandidateRef, list[B13CandidateRef]] = {}
    for ref in refs:
        grouped.setdefault(find(ref), []).append(ref)
    return tuple(sorted((tuple(sorted(group)) for group in grouped.values()), key=lambda group: group[0]))


def _same_material_payload(
    component: tuple[B13CandidateRef, ...], candidates: dict[B13CandidateRef, B13ReconciliationCandidate],
    *, ignored_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for ref in component:
        payload = dict(candidates[ref].payload())
        payload.pop("local_key", None)
        for field in ignored_fields:
            payload.pop(field, None)
        payloads.append(payload)
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise B13DispositionAssemblyError(
            "explicitly equivalent candidates disagree on material output fields; "
            "an explicit provenance-bearing content-selection contract is required"
        )
    return payloads[0]


def _identity_key(kind: B13CandidateKind, component: tuple[B13CandidateRef, ...]) -> str:
    return f"b13-{kind.value}-{content_hash({'kind': kind.value, 'members': [item.to_dict() for item in component]})[:24]}"


def _ref(value: dict[str, Any]) -> CanonicalTargetReference:
    return CanonicalTargetReference.from_dict(value)


def _require_live(ref: B13CandidateRef, candidates: dict[B13CandidateRef, B13ReconciliationCandidate]) -> B13ReconciliationCandidate:
    try:
        return candidates[ref]
    except KeyError as exc:
        raise B13DispositionAssemblyError(
            f"accepted graph contribution depends on rejected or absent candidate {ref.window_id}/{ref.candidate_kind.value}/{ref.local_key}"
        ) from exc


def _candidate_label(ref: B13CandidateRef) -> str:
    return f"{ref.window_id}/{ref.candidate_kind.value}/{ref.local_key}"


def _resolved_beat_participants(
    component: tuple[B13CandidateRef, ...], candidates: dict[B13CandidateRef, B13ReconciliationCandidate],
    identity_by_ref: dict[B13CandidateRef, tuple[str, str]],
) -> tuple[BeatParticipant, ...]:
    """Resolve every equivalent beat's scoped participant graph before merging.

    A local key has authority only within the proposal window that declared the
    beat.  Comparing final ``(kind, adapted_id)`` sets permits different local
    keys that were explicitly reconciled, while refusing a representative-only
    participant selection or an unauditable union.
    """
    resolved_by_beat: dict[B13CandidateRef, tuple[BeatParticipant, ...]] = {}
    for beat_ref in component:
        resolved: list[BeatParticipant] = []
        for item in candidates[beat_ref].payload()["participants"]:
            kind = (
                B13CandidateKind.ADAPTED_CHARACTER
                if item["kind"] == "adapted_character"
                else B13CandidateKind.ANONYMOUS_ROLE
            )
            participant_ref = B13CandidateRef(beat_ref.window_id, kind, item["local_key"])
            target = _require_live(participant_ref, candidates)
            target_identity = identity_by_ref.get(participant_ref)
            if target.ref.candidate_kind is not kind or target_identity is None or not target_identity[1]:
                raise B13DispositionAssemblyError(
                    f"beat participant {_candidate_label(participant_ref)} does not resolve to a final adapted identity"
                )
            resolved.append(BeatParticipant(item["kind"], target_identity[1]))
        if len(set(resolved)) != len(resolved):
            raise B13DispositionAssemblyError(
                f"equivalent beat {_candidate_label(beat_ref)} resolves duplicate material participants"
            )
        resolved_by_beat[beat_ref] = tuple(sorted(resolved, key=lambda item: (item.kind.value, item.participant_id)))

    baseline_ref = component[0]
    baseline = frozenset(resolved_by_beat[baseline_ref])
    for beat_ref in component[1:]:
        resolved = frozenset(resolved_by_beat[beat_ref])
        if resolved != baseline:
            raise B13DispositionAssemblyError(
                "equivalent beats resolve different material participants: "
                f"{_candidate_label(baseline_ref)}={sorted((item.kind.value, item.participant_id) for item in baseline)!r}; "
                f"{_candidate_label(beat_ref)}={sorted((item.kind.value, item.participant_id) for item in resolved)!r}"
            )
    return resolved_by_beat[baseline_ref]


def assemble_b13_disposition_plan(review: B13ReconciliationReview) -> B13DispositionAssemblyResult:
    """Materialize one complete validated B1.3-A plan from an exact B3A review.

    Unmentioned contributions remain accepted.  This is intentional: B3A's
    contribution decisions are conflict resolutions, not a default-drop list.
    Any explicit rejection with surviving graph dependents fails closed rather
    than causing an unauditable cascade.
    """
    inventory, decisions = _verified_review(review)
    rejected = _rejected_refs(decisions)
    live_items = tuple(item for item in inventory.candidates if item.ref not in rejected)
    candidates = {item.ref: item for item in live_items}

    # Check all local edges first, including inactive mappings: a rejected
    # endpoint cannot be silently forgotten merely because another validator
    # would not currently traverse it.
    for candidate in live_items:
        for link in candidate.local_links:
            _require_live(link, candidates)

    identity_by_ref: dict[B13CandidateRef, tuple[str, str]] = {}
    characters: list[AdaptedCharacter] = []
    roles: list[AnonymousRole] = []
    beats: list[AdaptedStoryBeat] = []
    for component in _equivalence_components(live_items, decisions):
        representative = candidates[component[0]]
        kind = representative.ref.candidate_kind
        if any(candidates[ref].ref.candidate_kind is not kind for ref in component):
            raise B13DispositionAssemblyError("equivalence component spans candidate kinds")
        identity_key = _identity_key(kind, component)
        if kind is B13CandidateKind.ADAPTED_CHARACTER:
            payload = _same_material_payload(component, candidates)
            value = AdaptedCharacter(
                identity_key, payload["display_name_zh"], payload["origin"],
                tuple(CharacterFunctionProvenance(
                    _ref(item["source_character_ref"]), item["function_zh"], item["provenance_zh"],
                ) for item in payload["function_provenance"]),
                payload["invention_rationale_zh"],
            )
            characters.append(value)
            adapted_id = value.adapted_character_id
        elif kind is B13CandidateKind.ANONYMOUS_ROLE:
            payload = _same_material_payload(component, candidates)
            value = AnonymousRole(identity_key, payload["role_kind"], payload["function_zh"])
            roles.append(value)
            adapted_id = value.anonymous_role_id
        elif kind is B13CandidateKind.ADAPTED_STORY_BEAT:
            # Participants need identity bindings, which may belong to a later
            # component.  Create beats in a second pass below.
            for ref in component:
                identity_by_ref[ref] = (identity_key, "")
            continue
        else:  # pragma: no cover - _IDENTITY_KINDS controls this loop
            raise B13DispositionAssemblyError("unsupported identity-bearing candidate")
        for ref in component:
            identity_by_ref[ref] = (identity_key, adapted_id)

    # All character/role bindings exist now; story-beat payloads can resolve
    # their local participant links deterministically.
    beat_components = [item for item in _equivalence_components(live_items, decisions)
                       if candidates[item[0]].ref.candidate_kind is B13CandidateKind.ADAPTED_STORY_BEAT]
    for component in beat_components:
        payload = _same_material_payload(component, candidates, ignored_fields=frozenset({"participants"}))
        identity_key = identity_by_ref[component[0]][0]
        participants = _resolved_beat_participants(component, candidates, identity_by_ref)
        value = AdaptedStoryBeat(
            identity_key, payload["summary_zh"], payload["origin"], payload["canon_compatibility"],
            tuple(_ref(item) for item in payload["source_event_refs"]), participants,
            tuple(_ref(item) for item in payload["canonical_anchor_refs"]),
            payload["invention_purpose"], payload["invention_rationale_zh"], payload["invention_provenance_zh"],
        )
        beats.append(value)
        for ref in component:
            identity_by_ref[ref] = (identity_key, value.adapted_story_beat_id)

    # There must be a final identity for every live local identity candidate.
    if any(not adapted_id for _key, adapted_id in identity_by_ref.values()):
        raise B13DispositionAssemblyError("global identity allocation is incomplete")

    dispositions: list[EventDisposition] = []
    for candidate in live_items:
        if candidate.ref.candidate_kind is not B13CandidateKind.EVENT_DISPOSITION:
            continue
        payload = candidate.payload()
        event_ref = _ref({"kind": "event", "canonical_id": payload["source_event_id"]})
        window = next(item for item in inventory.windows if item.window_id == candidate.ref.window_id)
        if payload["source_event_id"] not in window.owned_event_ids:
            raise B13DispositionAssemblyError("only an authoritative owner window may materialize an event disposition")
        dispositions.append(EventDisposition(event_ref, payload["disposition"], payload["rationale_zh"]))
    expected_events = {item for item in inventory.source_snapshot.canonical_references if item.kind == "event"}
    if len(dispositions) != len(expected_events) or {item.source_event_ref for item in dispositions} != expected_events:
        raise B13DispositionAssemblyError("owner-only event dispositions do not cover every canonical event exactly once")
    disposition_by_ref = {item.source_event_ref: item for item in dispositions}

    # No contextual contribution may revive a source event the owner omitted.
    for beat in beats:
        if any(disposition_by_ref[ref].disposition.value == "omit" for ref in beat.source_event_refs):
            raise B13DispositionAssemblyError("owner-omitted event cannot materialize through a contextual beat")

    mappings: list[AdaptationMapping] = []
    for candidate in live_items:
        if candidate.ref.candidate_kind is not B13CandidateKind.ADAPTATION_MAPPING:
            continue
        payload = candidate.payload()
        endpoint = payload["adapted_endpoint"]
        endpoint_kind = B13CandidateKind.ADAPTED_STORY_BEAT if endpoint["kind"] == "story_beat" else B13CandidateKind.ADAPTED_CHARACTER
        target_ref = B13CandidateRef(candidate.ref.window_id, endpoint_kind, endpoint["local_key"])
        _require_live(target_ref, candidates)
        target = identity_by_ref.get(target_ref)
        if target is None or not target[1]:
            raise B13DispositionAssemblyError("mapping endpoint does not resolve to a final adapted identity")
        mappings.append(AdaptationMapping(
            _ref(payload["source_ref"]), AdaptedEndpoint(endpoint["kind"], target[1]),
            payload["contribution_role"], payload["retained_semantics_zh"],
            payload["transformed_semantics_zh"], payload["provenance_zh"], payload["active"],
        ))

    facts: list[AdaptedFactDivergence] = []
    relationships: list[RelationshipArcRequirement] = []
    states: list[ContinuityStoryState] = []
    for candidate in live_items:
        payload = candidate.payload()
        if candidate.ref.candidate_kind is B13CandidateKind.FACTUAL_DIVERGENCE:
            facts.append(AdaptedFactDivergence(_ref(payload["source_fact_ref"]), payload["adapted_assertion_zh"], payload["provenance_zh"]))
        elif candidate.ref.candidate_kind is B13CandidateKind.RELATIONSHIP_REQUIREMENT:
            relationships.append(RelationshipArcRequirement(_ref(payload["source_relationship_ref"]), payload["requirement_zh"], payload["provenance_zh"]))
        elif candidate.ref.candidate_kind is B13CandidateKind.STORY_STATE_REQUIREMENT:
            beat_ref = B13CandidateRef(candidate.ref.window_id, B13CandidateKind.ADAPTED_STORY_BEAT, payload["adapted_story_beat_local_key"])
            _require_live(beat_ref, candidates)
            target = identity_by_ref.get(beat_ref)
            if target is None or not target[1]:
                raise B13DispositionAssemblyError("story-state requirement refers to a rejected or absent beat")
            states.append(ContinuityStoryState(target[1], payload["assertion_zh"], payload["provenance_zh"]))

    try:
        plan = B13DispositionPlan(
            inventory.source_snapshot, tuple(dispositions), tuple(characters), tuple(roles), tuple(beats),
            tuple(mappings), AdaptedContinuityOverlay(tuple(facts), tuple(relationships), tuple(states)),
        )
        plan.validate()
    except Exception as exc:
        raise B13DispositionAssemblyError(f"assembled B1.3 graph is invalid: {exc}") from exc

    bindings = tuple(sorted(
        (B13GlobalIdentityBinding(ref, identity_key, adapted_id)
         for ref, (identity_key, adapted_id) in identity_by_ref.items()),
        key=lambda item: item.candidate_ref,
    ))
    audit = tuple(B13AssemblyWindowAudit(
        window.window_id, window.proposal_hash, window.request_identity,
        tuple(item.ref for item in live_items if item.ref.window_id == window.window_id),
    ) for window in inventory.windows)
    return B13DispositionAssemblyResult(
        plan, inventory.inventory_identity, _decision_identity(decisions),
        dict(inventory.strategy_ref), inventory.strategy_hash, inventory.preparation_identity,
        bindings, audit,
    )


__all__ = [
    "B13DispositionAssemblyError", "B13GlobalIdentityBinding", "B13AssemblyWindowAudit",
    "B13DispositionAssemblyResult", "assemble_b13_disposition_plan",
]
