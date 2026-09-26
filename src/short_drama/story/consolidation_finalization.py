"""v1.2 A5E1 — deterministic identity components and canonical IDs.

This module is the A5E1 slice of Issue #54 (A5E deterministic finalization):
the deterministic *identity* half of consolidation finalization, driven by the
authoritative A5B planning identity and the three whole-domain semantic
resolution results.

    ConsolidationPlanningResult (A5B)
    FactSemanticResolutionResult        (A5C)
    EventSemanticResolutionResult       (A5D)
    RelationshipSemanticResolutionResult(A5D)
        ↓
    per-domain same_* identity graphs (union-find over the FULL candidate
    universe, not just the pairs that appeared in a decision)
        ↓
    hard-negative contradiction validation (same-component hard-negatives
    FAIL CLOSED; relationship direction-aware endpoint signature invariant)
        ↓
    source-authoritative member/component ordering
        ↓
    stable fact_/evt_/rel_ canonical ID allocation (1-based, sorted order)
        ↓
    ConsolidationIdentityPlan (in-memory, exact candidate coverage)

Frozen rules implemented here (authoritative: ``docs/
v1.2-A5E-deterministic-finalization-refinement-plan.md``):

* Only ``same_fact`` / ``same_event`` / ``same_relationship`` create an
  identity merge edge. ``uncertain`` is a non-edge and a non-veto.
* The graph node universe is EXACTLY the planning index for the domain
  (``planning_result.index.{facts,events,relationships}``): zero-edge,
  singleton, and unblocked candidates each form a component.
* An explicit hard-negative whose endpoints fall in the same component is a
  *structural* contradiction and FAILS CLOSED (it is never converted to a
  StoryConflict; A5E1 creates no StoryConflict).
    * Fact hard-negatives: ``compatible_fact`` / ``state_change`` / ``conflict``
      / ``unrelated``.
    * Event hard-negative: ``different_event``.
    * Relationship hard-negative: ``different_relationship``.
* The unique member order is ``(source_order_key, global_candidate_ref)``
  ascending; a component's key is its earliest member under the same tuple;
  components are sorted by that key. Union-find roots / parent pointers /
  decision arrival order / provider order are NOT the ordering authority.
* Canonical IDs are allocated 1-based in sorted component order
  (``fact_000001...`` / ``evt_000001...`` / ``rel_000001...``, >= 6 digits) and
  never depend on provider/model/decision_id/graph-root/timestamp.
* A relationship component's members must share one direction-aware endpoint
  signature: directed ``("directed", s, t)`` (ordered), symmetric
  ``("symmetric", min(s, t), max(s, t))``, unknown ``("unknown", s, t)``
  (ordered -- NOT treated as symmetric). A mismatch FAILS CLOSED.

A5E1 is deterministic Python-only: **provider calls = 0**, **persistence
writes = 0**, **CURRENT writes = 0**. It does NOT build CanonicalFact /
CanonicalEvent / CanonicalRelationship, StateTransition, or StoryConflict
(those are A5E2 / A5E3), and it does NOT persist anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .consolidation import (
    ConsolidationCandidateRef,
    EVENT_ID_PATTERN,
    FACT_ID_PATTERN,
    RELATIONSHIP_ID_PATTERN,
)
from .consolidation_planning import (
    ConsolidationPlanningResult,
    EventPairPlan,
    FactPairPlan,
    IndexedEventCandidate,
    IndexedFactCandidate,
    IndexedRelationshipCandidate,
    RelationshipPairPlan,
)
from .consolidation_semantic import (
    EventSemanticDecision,
    EventSemanticResolutionResult,
    FactSemanticDecision,
    FactSemanticResolutionResult,
    RelationshipSemanticDecision,
    RelationshipSemanticResolutionResult,
)
from .errors import ConsolidationModelError, ConsolidationSemanticError

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConsolidationFinalizationError(ConsolidationSemanticError):
    """A5E1 deterministic identity finalization failed a structural invariant.

    This is a narrow A5 finalization error: an identity-graph / contradiction /
    signature / coverage / planning-identity failure. It is deliberately
    distinct from A5A static model errors and from A5C/A5D semantic-generation
    findings. Every message carries the domain and the offending candidate
    refs / decision pair / reason so a review can locate the fault.
    """


# ---------------------------------------------------------------------------
# Frozen identity decision vocabulary (per domain)
# ---------------------------------------------------------------------------

# The single decision that creates an identity merge edge.
_FACT_MERGE_DECISION = "same_fact"
_EVENT_MERGE_DECISION = "same_event"
_RELATIONSHIP_MERGE_DECISION = "same_relationship"

# Explicit identity hard-negatives: an explicit decision of one of these kinds
# whose endpoints fall in the SAME component is a structural contradiction.
# ``uncertain`` is deliberately absent (non-edge / non-veto).
_FACT_HARD_NEGATIVES: frozenset[str] = frozenset(
    {"compatible_fact", "state_change", "conflict", "unrelated"}
)
_EVENT_HARD_NEGATIVES: frozenset[str] = frozenset({"different_event"})
_RELATIONSHIP_HARD_NEGATIVES: frozenset[str] = frozenset({"different_relationship"})

# Minimum zero-padded numeric width for a stable canonical id (>= 6 digits).
_ID_NUMERIC_WIDTH = 6

_DOMAINS: frozenset[str] = frozenset({"fact", "event", "relationship"})
_DOMAIN_ID_PATTERN: dict[str, re.Pattern[str]] = {
    "fact": re.compile(FACT_ID_PATTERN),
    "event": re.compile(EVENT_ID_PATTERN),
    "relationship": re.compile(RELATIONSHIP_ID_PATTERN),
}


def _canonical_id(prefix: str, ordinal: int) -> str:
    """Allocate a stable canonical id (1-based, >= 6 zero-padded digits)."""
    if ordinal < 1:
        raise ConsolidationFinalizationError(
            f"canonical id ordinal must be >= 1, got {ordinal}"
        )
    return f"{prefix}_{ordinal:0{_ID_NUMERIC_WIDTH}d}"


# ---------------------------------------------------------------------------
# Component + plan result objects (in-memory only, no persistence schema)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsolidationIdentityComponent:
    """One deterministic same_* identity component (A5E1).

    ``member_candidate_refs`` are the component members in the authoritative
    source order ``(source_order_key, global_candidate_ref)`` ascending; the
    earliest member (``member_candidate_refs[0]``) is the representative.
    ``first_source_order`` is that representative's ``source_order_key``.
    ``canonical_id`` is the stable 1-based ``<prefix>_NNNNNN`` id allocated by
    sorted-component order within the domain.

    This is an in-memory value: it carries identity membership + canonical id +
    source order only (no canonical semantic payload, which is A5E2/A5E3).
    """

    domain: str
    canonical_id: str
    member_candidate_refs: tuple[str, ...]
    first_source_order: str

    def __post_init__(self) -> None:
        if self.domain not in _DOMAINS:
            raise ConsolidationFinalizationError(
                f"component domain must be fact/event/relationship, got {self.domain!r}"
            )
        if not isinstance(self.canonical_id, str) or not _DOMAIN_ID_PATTERN[
            self.domain
        ].fullmatch(self.canonical_id):
            raise ConsolidationFinalizationError(
                f"component.canonical_id {self.canonical_id!r} does not match the "
                f"{self.domain!r} canonical id pattern"
            )
        if (
            not isinstance(self.member_candidate_refs, tuple)
            or not self.member_candidate_refs
        ):
            raise ConsolidationFinalizationError(
                "component.member_candidate_refs must be a non-empty tuple"
            )
        seen: set[str] = set()
        for ref in self.member_candidate_refs:
            if not isinstance(ref, str) or not ref:
                raise ConsolidationFinalizationError(
                    "component.member_candidate_refs must contain non-empty strings"
                )
            if ref in seen:
                raise ConsolidationFinalizationError(
                    f"component.member_candidate_refs has a duplicate {ref!r}"
                )
            seen.add(ref)
        if not isinstance(self.first_source_order, str) or not self.first_source_order:
            raise ConsolidationFinalizationError(
                "component.first_source_order must be a non-empty string"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "canonical_id": self.canonical_id,
            "member_candidate_refs": list(self.member_candidate_refs),
            "first_source_order": self.first_source_order,
        }


@dataclass(frozen=True, slots=True)
class ConsolidationIdentityPlan:
    """The deterministic A5E1 identity component plan (in-memory only).

    Carries the authoritative planning identity (``plan_hash``) and the three
    ordered same_* component collections. ``to_dict()`` is a deterministic
    normalized representation for comparison; it is NOT a persistence schema.
    """

    plan_hash: str
    fact_components: tuple[ConsolidationIdentityComponent, ...]
    event_components: tuple[ConsolidationIdentityComponent, ...]
    relationship_components: tuple[ConsolidationIdentityComponent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan_hash, str) or not self.plan_hash:
            raise ConsolidationFinalizationError(
                "plan.plan_hash must be a non-empty string"
            )
        for name, domain in (
            ("fact_components", "fact"),
            ("event_components", "event"),
            ("relationship_components", "relationship"),
        ):
            components = getattr(self, name)
            if not isinstance(components, tuple):
                raise ConsolidationFinalizationError(
                    f"plan.{name} must be a tuple"
                )
            for component in components:
                if not isinstance(component, ConsolidationIdentityComponent):
                    raise ConsolidationFinalizationError(
                        f"plan.{name} must contain ConsolidationIdentityComponent"
                    )
                if component.domain != domain:
                    raise ConsolidationFinalizationError(
                        f"plan.{name} contains a component with domain "
                        f"{component.domain!r} (expected {domain!r})"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "fact_components": [c.to_dict() for c in self.fact_components],
            "event_components": [c.to_dict() for c in self.event_components],
            "relationship_components": [
                c.to_dict() for c in self.relationship_components
            ],
        }


# ---------------------------------------------------------------------------
# Generic deterministic identity-graph core
# ---------------------------------------------------------------------------


def _validate_endpoint(ref: Any, domain: str, candidate_by_ref: dict[str, Any]) -> str:
    """Validate one decision/plan endpoint ref for ``domain`` (fail closed).

    The ref must be a valid consolidation candidate ref in the ``domain``
    namespace (domain namespace mismatch) AND present in the domain candidate
    index (candidate ref missing from index).
    """
    if not isinstance(ref, str):
        raise ConsolidationFinalizationError(
            f"{domain}: candidate ref must be a string, got {ref!r}"
        )
    try:
        parsed = ConsolidationCandidateRef.parse(ref)
    except ConsolidationModelError as exc:
        raise ConsolidationFinalizationError(
            f"{domain}: invalid candidate ref {ref!r}: {exc}"
        ) from exc
    if parsed.namespace != domain:
        raise ConsolidationFinalizationError(
            f"{domain}: candidate ref {ref!r} is in the {parsed.namespace!r} "
            "namespace (domain namespace mismatch)"
        )
    if ref not in candidate_by_ref:
        raise ConsolidationFinalizationError(
            f"{domain}: candidate ref {ref!r} is not in the candidate index"
        )
    return ref


def _source_order_key(candidate_by_ref: dict[str, Any], ref: str) -> str:
    return candidate_by_ref[ref].source_order_key


def _build_identity_components(
    *,
    domain: str,
    id_prefix: str,
    candidates: Sequence[Any],
    decisions: Sequence[Any],
    pair_plans: Sequence[Any],
    merge_decision: str,
    hard_negative_decisions: frozenset[str],
    signature_check: Callable[[tuple[str, ...], dict[str, Any]], None] | None = None,
) -> tuple[ConsolidationIdentityComponent, ...]:
    """Deterministically build the ``domain`` same_* identity components.

    The node universe is the FULL ``candidates`` index (not the decision
    endpoints): every candidate forms exactly one component. ``merge_decision``
    creates undirected merge edges; any explicit decision in
    ``hard_negative_decisions`` whose endpoints end in the same component FAILS
    CLOSED. Members and components are ordered by the frozen source authority
    ``(source_order_key, global_candidate_ref)``, and canonical ids are
    allocated 1-based in that order. The result is byte-identical for any
    logically-identical input regardless of candidate / decision / edge order.
    """
    # 1. Node universe (authoritative source) -- the FULL candidate index.
    candidate_by_ref: dict[str, Any] = {}
    for candidate in candidates:
        ref = candidate.global_candidate_ref
        if ref in candidate_by_ref:
            raise ConsolidationFinalizationError(
                f"{domain}: duplicate candidate {ref!r} in the candidate index"
            )
        candidate_by_ref[ref] = candidate

    # 2. Planned pair universe: endpoints in-index + namespace, canonical
    #    order, unique.
    expected_pairs: set[tuple[str, str]] = set()
    for plan in pair_plans:
        left = _validate_endpoint(plan.left_ref, domain, candidate_by_ref)
        right = _validate_endpoint(plan.right_ref, domain, candidate_by_ref)
        if not left < right:
            raise ConsolidationFinalizationError(
                f"{domain}: planned pair is not in canonical order: "
                f"{left!r} !< {right!r}"
            )
        pair = (left, right)
        if pair in expected_pairs:
            raise ConsolidationFinalizationError(
                f"{domain}: duplicate planned pair {pair!r}"
            )
        expected_pairs.add(pair)

    # 3. Decision universe: endpoints in-index + namespace, canonical order,
    #    no duplicate pair, exact coverage of the planned pairs.
    decision_by_pair: dict[tuple[str, str], Any] = {}
    for decision in decisions:
        left = _validate_endpoint(decision.left_candidate_ref, domain, candidate_by_ref)
        right = _validate_endpoint(decision.right_candidate_ref, domain, candidate_by_ref)
        if not left < right:
            raise ConsolidationFinalizationError(
                f"{domain}: decision pair is not in canonical order: "
                f"{left!r} !< {right!r}"
            )
        pair = (left, right)
        if pair in decision_by_pair:
            raise ConsolidationFinalizationError(
                f"{domain}: duplicate decision for pair {pair!r}"
            )
        decision_by_pair[pair] = decision

    for pair in decision_by_pair:
        if pair not in expected_pairs:
            raise ConsolidationFinalizationError(
                f"{domain}: decision found for pair {pair!r} that is not a planned "
                "pair (unknown decision pair)"
            )
    for pair in expected_pairs:
        if pair not in decision_by_pair:
            raise ConsolidationFinalizationError(
                f"{domain}: no decision found for planned pair {pair!r} "
                "(missing explicit pair decision)"
            )

    # 4. Union-find over the full node universe; merge edges only.
    refs = list(candidate_by_ref)
    parent = {ref: ref for ref in refs}

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

    for pair, decision in decision_by_pair.items():
        if decision.decision == merge_decision:
            union(pair[0], pair[1])

    # 5. Group members by root; order members by the frozen source authority.
    groups: dict[str, list[str]] = {}
    for ref in refs:
        groups.setdefault(find(ref), []).append(ref)
    member_lists: list[list[str]] = []
    for members in groups.values():
        members.sort(key=lambda r: (_source_order_key(candidate_by_ref, r), r))
        member_lists.append(members)

    # 6. Hard-negative contradiction validation (same component => FAIL).
    for (left, right), decision in decision_by_pair.items():
        if decision.decision in hard_negative_decisions and find(left) == find(right):
            raise ConsolidationFinalizationError(
                f"{domain}: hard-negative decision {decision.decision!r} for pair "
                f"{left!r} <-> {right!r} resolves to the same identity component "
                "(structural contradiction)"
            )

    # 7. Domain-specific component invariant (relationship endpoint signature).
    if signature_check is not None:
        for members in member_lists:
            signature_check(tuple(members), candidate_by_ref)

    # 8. Order components by earliest member; allocate stable canonical ids.
    member_lists.sort(
        key=lambda m: (_source_order_key(candidate_by_ref, m[0]), m[0])
    )
    components = tuple(
        ConsolidationIdentityComponent(
            domain=domain,
            canonical_id=_canonical_id(id_prefix, ordinal),
            member_candidate_refs=tuple(members),
            first_source_order=_source_order_key(candidate_by_ref, members[0]),
        )
        for ordinal, members in enumerate(member_lists, start=1)
    )

    # 9. Exact candidate coverage: every indexed ref in exactly one component.
    covered: set[str] = set()
    for component in components:
        for ref in component.member_candidate_refs:
            if ref in covered:
                raise ConsolidationFinalizationError(
                    f"{domain}: candidate {ref!r} appears in multiple components"
                )
            covered.add(ref)
    if len(covered) != len(candidate_by_ref):
        raise ConsolidationFinalizationError(
            f"{domain}: component members ({len(covered)}) do not exactly cover the "
            f"candidate universe ({len(candidate_by_ref)})"
        )

    return components


# ---------------------------------------------------------------------------
# Relationship direction-aware endpoint signature (frozen rule)
# ---------------------------------------------------------------------------


def _relationship_signature(candidate: Any) -> tuple:
    """The direction-aware endpoint signature of one relationship candidate.

    * directed  -> ``("directed", source, target)``     (order is authoritative)
    * symmetric -> ``("symmetric", min(source, target), max(source, target))``
    * unknown   -> ``("unknown", source, target)``      (ordered; NOT symmetric)
    """
    source = candidate.source_entity_ref
    target = candidate.target_entity_ref
    direction = candidate.direction
    if direction == "directed":
        return ("directed", source, target)
    if direction == "symmetric":
        lo, hi = (source, target) if source <= target else (target, source)
        return ("symmetric", lo, hi)
    if direction == "unknown":
        return ("unknown", source, target)
    raise ConsolidationFinalizationError(
        f"relationship: candidate {candidate.global_candidate_ref!r} has an "
        f"unknown direction {direction!r}"
    )


def _check_relationship_signature(
    members: tuple[str, ...], candidate_by_ref: dict[str, Any]
) -> None:
    """FAIL CLOSED unless all members of a relationship component share one
    direction-aware endpoint signature."""
    signatures: dict[str, tuple] = {}
    for ref in members:
        signatures[ref] = _relationship_signature(candidate_by_ref[ref])
    unique = set(signatures.values())
    if len(unique) > 1:
        raise ConsolidationFinalizationError(
            f"relationship: component with members {list(members)!r} has "
            f"mismatched direction-aware endpoint signatures {sorted(unique)!r}"
        )


# ---------------------------------------------------------------------------
# Domain entry points (also usable directly by A5E2 / A5E3)
# ---------------------------------------------------------------------------


def build_fact_identity_components(
    candidates: Sequence[IndexedFactCandidate],
    decisions: Sequence[FactSemanticDecision],
    pair_plans: Sequence[FactPairPlan],
) -> tuple[ConsolidationIdentityComponent, ...]:
    """Build the deterministic fact same_* identity components (A5E1)."""
    return _build_identity_components(
        domain="fact",
        id_prefix="fact",
        candidates=candidates,
        decisions=decisions,
        pair_plans=pair_plans,
        merge_decision=_FACT_MERGE_DECISION,
        hard_negative_decisions=_FACT_HARD_NEGATIVES,
    )


def build_event_identity_components(
    candidates: Sequence[IndexedEventCandidate],
    decisions: Sequence[EventSemanticDecision],
    pair_plans: Sequence[EventPairPlan],
) -> tuple[ConsolidationIdentityComponent, ...]:
    """Build the deterministic event same_* identity components (A5E1)."""
    return _build_identity_components(
        domain="event",
        id_prefix="evt",
        candidates=candidates,
        decisions=decisions,
        pair_plans=pair_plans,
        merge_decision=_EVENT_MERGE_DECISION,
        hard_negative_decisions=_EVENT_HARD_NEGATIVES,
    )


def build_relationship_identity_components(
    candidates: Sequence[IndexedRelationshipCandidate],
    decisions: Sequence[RelationshipSemanticDecision],
    pair_plans: Sequence[RelationshipPairPlan],
) -> tuple[ConsolidationIdentityComponent, ...]:
    """Build the deterministic relationship same_* identity components (A5E1)."""
    return _build_identity_components(
        domain="relationship",
        id_prefix="rel",
        candidates=candidates,
        decisions=decisions,
        pair_plans=pair_plans,
        merge_decision=_RELATIONSHIP_MERGE_DECISION,
        hard_negative_decisions=_RELATIONSHIP_HARD_NEGATIVES,
        signature_check=_check_relationship_signature,
    )


# ---------------------------------------------------------------------------
# Production entry point (whole-domain, planning-identity bound)
# ---------------------------------------------------------------------------


def finalize_consolidation_identity(
    planning_result: ConsolidationPlanningResult,
    fact_resolution: FactSemanticResolutionResult,
    event_resolution: EventSemanticResolutionResult,
    relationship_resolution: RelationshipSemanticResolutionResult,
) -> ConsolidationIdentityPlan:
    """Deterministically finalize the A5E1 identity component plan.

    Production A5E1 entry point. Verifies that all three semantic resolution
    results belong to the SAME authoritative planning identity as
    ``planning_result`` (mismatch FAILS CLOSED), then builds the fact / event /
    relationship identity components from the full candidate index + the
    whole-domain ``all_*_decisions``. No provider call, no persistence.
    """
    if not isinstance(planning_result, ConsolidationPlanningResult):
        raise ConsolidationFinalizationError(
            "planning_result must be a ConsolidationPlanningResult"
        )
    if not isinstance(fact_resolution, FactSemanticResolutionResult):
        raise ConsolidationFinalizationError(
            "fact_resolution must be a FactSemanticResolutionResult"
        )
    if not isinstance(event_resolution, EventSemanticResolutionResult):
        raise ConsolidationFinalizationError(
            "event_resolution must be an EventSemanticResolutionResult"
        )
    if not isinstance(relationship_resolution, RelationshipSemanticResolutionResult):
        raise ConsolidationFinalizationError(
            "relationship_resolution must be a RelationshipSemanticResolutionResult"
        )

    plan_hash = planning_result.plan_hash
    for domain, resolution in (
        ("fact", fact_resolution),
        ("event", event_resolution),
        ("relationship", relationship_resolution),
    ):
        if resolution.planning_result.plan_hash != plan_hash:
            raise ConsolidationFinalizationError(
                f"{domain} resolution planning identity "
                f"{resolution.planning_result.plan_hash!r} does not match "
                f"planning_result.plan_hash {plan_hash!r}"
            )

    index = planning_result.index
    fact_components = build_fact_identity_components(
        index.facts,
        fact_resolution.all_fact_decisions,
        planning_result.fact_pair_plans,
    )
    event_components = build_event_identity_components(
        index.events,
        event_resolution.all_event_decisions,
        planning_result.event_pair_plans,
    )
    relationship_components = build_relationship_identity_components(
        index.relationships,
        relationship_resolution.all_relationship_decisions,
        planning_result.relationship_pair_plans,
    )

    return ConsolidationIdentityPlan(
        plan_hash=plan_hash,
        fact_components=fact_components,
        event_components=event_components,
        relationship_components=relationship_components,
    )


__all__ = [
    "ConsolidationFinalizationError",
    "ConsolidationIdentityComponent",
    "ConsolidationIdentityPlan",
    "build_event_identity_components",
    "build_fact_identity_components",
    "build_relationship_identity_components",
    "finalize_consolidation_identity",
]
