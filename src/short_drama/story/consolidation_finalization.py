"""v1.2 A5E deterministic finalization helpers.

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
writes = 0**, **CURRENT writes = 0**.  The A5E2 helpers consume Fact/Event
components for canonical assembly and fact-addressed side objects; A5E3 adds
canonical relationship/state finalization and the final in-memory composition.
This module never persists anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .consolidation import (
    CanonicalEvent,
    CanonicalEventSet,
    CanonicalFact,
    CanonicalFactSet,
    CanonicalRelationship,
    CanonicalRelationshipSet,
    ConsolidationCandidateRef,
    EVENT_ID_PATTERN,
    FACT_ID_PATTERN,
    RELATIONSHIP_ID_PATTERN,
    RelationshipState,
    StateTransition,
    StoryConflict,
    StoryConflictSet,
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


def _evidence_identity(evidence: Any) -> tuple[str, str, str, str | None]:
    """Return the frozen exact EvidenceRef identity, failing closed on shape."""
    try:
        identity = (
            evidence.paragraph_id,
            evidence.role,
            evidence.strength,
            evidence.excerpt,
        )
    except AttributeError as exc:
        raise ConsolidationFinalizationError(
            "evidence ref does not expose paragraph_id/role/strength/excerpt"
        ) from exc
    if (
        not isinstance(identity[0], str)
        or not isinstance(identity[1], str)
        or not isinstance(identity[2], str)
        or (identity[3] is not None and not isinstance(identity[3], str))
    ):
        raise ConsolidationFinalizationError("evidence ref has an invalid exact identity")
    return identity


def _require_unique_evidence(evidence_refs: Sequence[Any], *, label: str) -> None:
    """Reject duplicate exact evidence rather than changing a semantic payload."""
    seen: set[tuple[str, str, str, str | None]] = set()
    for evidence in evidence_refs:
        identity = _evidence_identity(evidence)
        if identity in seen:
            raise ConsolidationFinalizationError(
                f"{label} contains duplicate exact EvidenceRef identity {identity!r}"
            )
        seen.add(identity)


def _stable_evidence_union(members: Sequence[Any]) -> tuple[Any, ...]:
    """Stable exact EvidenceRef union in already-authoritative member order."""
    seen: set[tuple[str, str, str, str | None]] = set()
    evidence_refs: list[Any] = []
    for member in members:
        for evidence in member.evidence_refs:
            identity = _evidence_identity(evidence)
            if identity not in seen:
                seen.add(identity)
                evidence_refs.append(evidence)
    return tuple(evidence_refs)


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
    if not isinstance(fact_resolution.planning_result, ConsolidationPlanningResult):
        raise ConsolidationFinalizationError(
            "fact_resolution.planning_result must be a ConsolidationPlanningResult"
        )
    if not isinstance(event_resolution, EventSemanticResolutionResult):
        raise ConsolidationFinalizationError(
            "event_resolution must be an EventSemanticResolutionResult"
        )
    if not isinstance(relationship_resolution, RelationshipSemanticResolutionResult):
        raise ConsolidationFinalizationError(
            "relationship_resolution must be a RelationshipSemanticResolutionResult"
        )
    for domain, resolution in (
        ("event", event_resolution),
        ("relationship", relationship_resolution),
    ):
        if not isinstance(resolution.planning_result, ConsolidationPlanningResult):
            raise ConsolidationFinalizationError(
                f"{domain}_resolution.planning_result must be a "
                "ConsolidationPlanningResult"
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


# ---------------------------------------------------------------------------
# A5E2: Canonical fact/event assembly and fact side objects
# ---------------------------------------------------------------------------


def _candidate_index_by_ref(
    candidates: Sequence[Any], *, domain: str
) -> dict[str, Any]:
    """Build a checked candidate lookup; input tuple order is never authority."""
    result: dict[str, Any] = {}
    for candidate in candidates:
        # ``_validate_endpoint`` cannot be used until the lookup is complete.
        ref = getattr(candidate, "global_candidate_ref", None)
        if not isinstance(ref, str):
            raise ConsolidationFinalizationError(
                f"{domain}: indexed candidate ref must be a string"
            )
        try:
            parsed = ConsolidationCandidateRef.parse(ref)
        except ConsolidationModelError as exc:
            raise ConsolidationFinalizationError(
                f"{domain}: invalid indexed candidate ref {ref!r}: {exc}"
            ) from exc
        if parsed.namespace != domain:
            raise ConsolidationFinalizationError(
                f"{domain}: indexed candidate ref {ref!r} has wrong namespace "
                f"{parsed.namespace!r}"
            )
        if ref in result:
            raise ConsolidationFinalizationError(
                f"{domain}: duplicate candidate {ref!r} in planning index"
            )
        result[ref] = candidate
    return result


def _validate_identity_components(
    *,
    planning_result: ConsolidationPlanningResult,
    identity_plan: ConsolidationIdentityPlan,
    domain: str,
) -> tuple[tuple[ConsolidationIdentityComponent, ...], dict[str, Any]]:
    """Validate A5E1 membership/order/coverage before consuming its IDs.

    This deliberately verifies the supplied plan rather than recomputing
    identity.  A5E1 remains the sole authority that decides components.
    """
    if not isinstance(planning_result, ConsolidationPlanningResult):
        raise ConsolidationFinalizationError(
            "planning_result must be a ConsolidationPlanningResult"
        )
    if not isinstance(identity_plan, ConsolidationIdentityPlan):
        raise ConsolidationFinalizationError(
            "identity_plan must be a ConsolidationIdentityPlan"
        )
    if identity_plan.plan_hash != planning_result.plan_hash:
        raise ConsolidationFinalizationError(
            "identity plan planning identity does not match planning_result.plan_hash"
        )

    candidates = getattr(planning_result.index, f"{domain}s")
    candidate_by_ref = _candidate_index_by_ref(candidates, domain=domain)
    components = getattr(identity_plan, f"{domain}_components")
    expected_pattern = _DOMAIN_ID_PATTERN[domain]
    canonical_id_prefix = {
        "fact": "fact",
        "event": "evt",
        "relationship": "rel",
    }[domain]
    covered: set[str] = set()
    canonical_ids: set[str] = set()
    previous_component_key: tuple[str, str] | None = None

    for ordinal, component in enumerate(components, start=1):
        if not isinstance(component, ConsolidationIdentityComponent):
            raise ConsolidationFinalizationError(
                f"{domain}: identity components must be ConsolidationIdentityComponent"
            )
        if component.domain != domain:
            raise ConsolidationFinalizationError(
                f"{domain}: component has wrong domain {component.domain!r}"
            )
        if not expected_pattern.fullmatch(component.canonical_id):
            raise ConsolidationFinalizationError(
                f"{domain}: component canonical id {component.canonical_id!r} has "
                "wrong namespace"
            )
        expected_canonical_id = _canonical_id(canonical_id_prefix, ordinal)
        if component.canonical_id != expected_canonical_id:
            raise ConsolidationFinalizationError(
                f"{domain}: component canonical id {component.canonical_id!r} does "
                f"not match authoritative source-order ordinal {ordinal} "
                f"({expected_canonical_id!r})"
            )
        if component.canonical_id in canonical_ids:
            raise ConsolidationFinalizationError(
                f"{domain}: duplicate canonical id {component.canonical_id!r}"
            )
        canonical_ids.add(component.canonical_id)

        member_key_previous: tuple[str, str] | None = None
        for ref in component.member_candidate_refs:
            _validate_endpoint(ref, domain, candidate_by_ref)
            if ref in covered:
                raise ConsolidationFinalizationError(
                    f"{domain}: candidate {ref!r} has duplicate component membership"
                )
            member_key = (candidate_by_ref[ref].source_order_key, ref)
            if member_key_previous is not None and member_key <= member_key_previous:
                raise ConsolidationFinalizationError(
                    f"{domain}: component members are not in authoritative source order"
                )
            member_key_previous = member_key
            covered.add(ref)

        first_ref = component.member_candidate_refs[0]
        first_key = (candidate_by_ref[first_ref].source_order_key, first_ref)
        if component.first_source_order != first_key[0]:
            raise ConsolidationFinalizationError(
                f"{domain}: component.first_source_order does not match earliest member"
            )
        if previous_component_key is not None and first_key <= previous_component_key:
            raise ConsolidationFinalizationError(
                f"{domain}: components are not in authoritative source order"
            )
        previous_component_key = first_key

    if covered != set(candidate_by_ref):
        missing = sorted(set(candidate_by_ref) - covered)
        foreign = sorted(covered - set(candidate_by_ref))
        raise ConsolidationFinalizationError(
            f"{domain}: identity component coverage is not exact "
            f"(missing={missing!r}, foreign={foreign!r})"
        )
    return components, candidate_by_ref


def _fact_assembly_context(
    planning_result: ConsolidationPlanningResult,
    identity_plan: ConsolidationIdentityPlan,
    fact_resolution: FactSemanticResolutionResult,
) -> tuple[
    tuple[ConsolidationIdentityComponent, ...],
    dict[str, Any],
    dict[str, tuple[str, int]],
    dict[str, FactSemanticDecision],
]:
    """Return checked fact A5E1 mappings and an unambiguous decision lookup."""
    if not isinstance(fact_resolution, FactSemanticResolutionResult):
        raise ConsolidationFinalizationError(
            "fact_resolution must be a FactSemanticResolutionResult"
        )
    if fact_resolution.planning_result.plan_hash != planning_result.plan_hash:
        raise ConsolidationFinalizationError(
            "fact resolution planning identity does not match planning_result.plan_hash"
        )
    components, candidate_by_ref = _validate_identity_components(
        planning_result=planning_result, identity_plan=identity_plan, domain="fact"
    )
    ref_to_fact: dict[str, tuple[str, int]] = {}
    for ordinal, component in enumerate(components, start=1):
        for ref in component.member_candidate_refs:
            if ref in ref_to_fact:
                raise ConsolidationFinalizationError(
                    f"fact: candidate {ref!r} maps to multiple canonical facts"
                )
            ref_to_fact[ref] = (component.canonical_id, ordinal)

    decisions_by_id: dict[str, FactSemanticDecision] = {}
    for decision in fact_resolution.all_fact_decisions:
        if not isinstance(decision, FactSemanticDecision):
            raise ConsolidationFinalizationError(
                "fact_resolution.all_fact_decisions must contain FactSemanticDecision"
            )
        if decision.decision_id in decisions_by_id:
            raise ConsolidationFinalizationError(
                f"fact: duplicate decision_id {decision.decision_id!r} is ambiguous"
            )
        _validate_endpoint(decision.left_candidate_ref, "fact", candidate_by_ref)
        _validate_endpoint(decision.right_candidate_ref, "fact", candidate_by_ref)
        decisions_by_id[decision.decision_id] = decision
    return components, candidate_by_ref, ref_to_fact, decisions_by_id


def _canonical_fact_records(
    components: Sequence[ConsolidationIdentityComponent], candidate_by_ref: dict[str, Any]
) -> tuple[CanonicalFact, ...]:
    facts: list[CanonicalFact] = []
    for component in components:
        members = tuple(candidate_by_ref[ref] for ref in component.member_candidate_refs)
        representative = members[0]
        evidence_refs = _stable_evidence_union(members)
        _require_unique_evidence(evidence_refs, label=f"fact {component.canonical_id}")
        facts.append(
            CanonicalFact(
                fact_id=component.canonical_id,
                fact_type=representative.fact_type,
                statement_zh=representative.statement_zh,
                subject_refs=representative.subject_refs,
                object_refs=representative.object_refs,
                candidate_fact_refs=component.member_candidate_refs,
                evidence_refs=evidence_refs,
                first_source_order=representative.source_order_key,
                continuity_relevant=any(
                    member.fact_type == "continuity_relevant" for member in members
                ),
            )
        )
    return tuple(facts)


def _ordered_fact_endpoints(
    decision: FactSemanticDecision, ref_to_fact: dict[str, tuple[str, int]]
) -> tuple[str, int, str, int]:
    left_id, left_ordinal = ref_to_fact[decision.left_candidate_ref]
    right_id, right_ordinal = ref_to_fact[decision.right_candidate_ref]
    if left_id == right_id:
        raise ConsolidationFinalizationError(
            f"fact: {decision.decision!r} decision {decision.decision_id!r} endpoints "
            "resolve to the same canonical fact"
        )
    if (left_ordinal, left_id) < (right_ordinal, right_id):
        return left_id, left_ordinal, right_id, right_ordinal
    return right_id, right_ordinal, left_id, left_ordinal


def _fact_state_transitions(
    *,
    facts: Sequence[CanonicalFact],
    ref_to_fact: dict[str, tuple[str, int]],
    decisions_by_id: dict[str, FactSemanticDecision],
) -> tuple[StateTransition, ...]:
    facts_by_id = {fact.fact_id: fact for fact in facts}
    pending: list[tuple[int, int, str, str, str, FactSemanticDecision]] = []
    for decision in decisions_by_id.values():
        if decision.decision != "state_change":
            continue
        from_id, from_ordinal, to_id, to_ordinal = _ordered_fact_endpoints(
            decision, ref_to_fact
        )
        _require_unique_evidence(
            decision.evidence_refs, label=f"state_change decision {decision.decision_id}"
        )
        pending.append(
            (to_ordinal, from_ordinal, decision.decision_id, from_id, to_id, decision)
        )
    pending.sort(key=lambda item: item[:3])

    transitions: list[StateTransition] = []
    for ordinal, (to_ordinal, _, _, from_id, to_id, decision) in enumerate(
        pending, start=1
    ):
        if from_id not in facts_by_id or to_id not in facts_by_id or from_id == to_id:
            raise ConsolidationFinalizationError(
                f"state_change decision {decision.decision_id!r} has invalid fact endpoints"
            )
        to_subjects = set(facts_by_id[to_id].subject_refs)
        subjects = tuple(
            subject for subject in facts_by_id[from_id].subject_refs if subject in to_subjects
        )
        transition = StateTransition(
            transition_id=_canonical_id("trans", ordinal),
            from_fact_id=from_id,
            to_fact_id=to_id,
            subject_refs=subjects,
            transition_kind="state_change",
            source_decision_ref=decision.decision_id,
            evidence_refs=decision.evidence_refs,
            narrative_order=to_ordinal,
        )
        if (
            decisions_by_id.get(transition.source_decision_ref) is not decision
            or transition.transition_kind != "state_change"
        ):
            raise ConsolidationFinalizationError("state transition decision cross-reference invalid")
        transitions.append(transition)
    return tuple(transitions)


def build_canonical_fact_set(
    planning_result: ConsolidationPlanningResult,
    identity_plan: ConsolidationIdentityPlan,
    fact_resolution: FactSemanticResolutionResult,
) -> CanonicalFactSet:
    """Build A5E2 CanonicalFactSet and fact-addressed state transitions.

    A5E1 supplies component membership and canonical IDs.  This function does
    not re-decide identity, call a provider, or persist anything.
    """
    components, candidate_by_ref, ref_to_fact, decisions_by_id = _fact_assembly_context(
        planning_result, identity_plan, fact_resolution
    )
    facts = _canonical_fact_records(components, candidate_by_ref)
    transitions = _fact_state_transitions(
        facts=facts, ref_to_fact=ref_to_fact, decisions_by_id=decisions_by_id
    )
    if {ref for fact in facts for ref in fact.candidate_fact_refs} != set(candidate_by_ref):
        raise ConsolidationFinalizationError("fact: canonical fact candidate coverage is not exact")
    if len({fact.fact_id for fact in facts}) != len(facts):
        raise ConsolidationFinalizationError("fact: canonical fact ids are not unique")
    return CanonicalFactSet(schema_version=1, facts=facts, state_transitions=transitions)


def build_fact_story_conflict_set(
    planning_result: ConsolidationPlanningResult,
    identity_plan: ConsolidationIdentityPlan,
    fact_resolution: FactSemanticResolutionResult,
) -> StoryConflictSet:
    """Build the A5E2 fact-only unresolved StoryConflictSet."""
    components, candidate_by_ref, ref_to_fact, decisions_by_id = _fact_assembly_context(
        planning_result, identity_plan, fact_resolution
    )
    # Constructing facts here also validates the canonical candidate assembly
    # that conflict endpoints reference, without creating a new intermediate API.
    facts = _canonical_fact_records(components, candidate_by_ref)
    facts_by_id = {fact.fact_id: fact for fact in facts}
    pending: list[tuple[int, int, str, str, str, FactSemanticDecision]] = []
    for decision in decisions_by_id.values():
        if decision.decision != "conflict":
            continue
        earlier_id, earlier_ordinal, later_id, later_ordinal = _ordered_fact_endpoints(
            decision, ref_to_fact
        )
        _require_unique_evidence(
            decision.evidence_refs, label=f"conflict decision {decision.decision_id}"
        )
        pending.append(
            (later_ordinal, earlier_ordinal, decision.decision_id, earlier_id, later_id, decision)
        )
    pending.sort(key=lambda item: item[:3])

    conflicts: list[StoryConflict] = []
    for ordinal, (_, _, _, earlier_id, later_id, decision) in enumerate(pending, start=1):
        if earlier_id not in facts_by_id or later_id not in facts_by_id:
            raise ConsolidationFinalizationError("fact conflict has an unknown canonical fact")
        conflict = StoryConflict(
            conflict_id=_canonical_id("conf", ordinal),
            conflict_kind="fact_conflict",
            fact_ids=(earlier_id, later_id),
            relationship_ids=(),
            candidate_refs=(decision.left_candidate_ref, decision.right_candidate_ref),
            decision_refs=(decision.decision_id,),
            evidence_refs=decision.evidence_refs,
            status="unresolved",
        )
        resolved = decisions_by_id.get(conflict.decision_refs[0])
        if (
            resolved is not decision
            or resolved.decision != "conflict"
            or conflict.relationship_ids != ()
            or conflict.status != "unresolved"
        ):
            raise ConsolidationFinalizationError("fact conflict decision cross-reference invalid")
        for ref in conflict.candidate_refs:
            _validate_endpoint(ref, "fact", candidate_by_ref)
        conflicts.append(conflict)
    return StoryConflictSet(schema_version=1, conflicts=tuple(conflicts))


def build_canonical_event_set(
    planning_result: ConsolidationPlanningResult,
    identity_plan: ConsolidationIdentityPlan,
) -> CanonicalEventSet:
    """Build A5E2 CanonicalEventSet from A5E1 components, with no side objects."""
    components, candidate_by_ref = _validate_identity_components(
        planning_result=planning_result, identity_plan=identity_plan, domain="event"
    )
    events: list[CanonicalEvent] = []
    for ordinal, component in enumerate(components, start=1):
        members = tuple(candidate_by_ref[ref] for ref in component.member_candidate_refs)
        representative = members[0]
        known_modes = {
            member.temporal_mode
            for member in members
            if member.temporal_mode != "unknown"
        }
        temporal_mode = next(iter(known_modes)) if len(known_modes) == 1 else "unknown"
        evidence_refs = _stable_evidence_union(members)
        _require_unique_evidence(evidence_refs, label=f"event {component.canonical_id}")
        events.append(
            CanonicalEvent(
                event_id=component.canonical_id,
                narrative_order=ordinal,
                summary_zh=representative.summary_zh,
                participants=representative.participants,
                locations=representative.locations,
                temporal_mode=temporal_mode,
                candidate_event_refs=component.member_candidate_refs,
                evidence_refs=evidence_refs,
                first_source_order=representative.source_order_key,
            )
        )
    if len({event.event_id for event in events}) != len(events):
        raise ConsolidationFinalizationError("event: canonical event ids are not unique")
    if tuple(event.narrative_order for event in events) != tuple(range(1, len(events) + 1)):
        raise ConsolidationFinalizationError("event: narrative_order is not contiguous")
    if {ref for event in events for ref in event.candidate_event_refs} != set(candidate_by_ref):
        raise ConsolidationFinalizationError("event: canonical event candidate coverage is not exact")
    return CanonicalEventSet(schema_version=1, events=tuple(events))


# ---------------------------------------------------------------------------
# A5E3: Canonical relationships and final in-memory composition
# ---------------------------------------------------------------------------


def _relationship_source_ranks(
    candidates: Sequence[IndexedRelationshipCandidate],
) -> dict[str, int]:
    """Return the frozen global relationship candidate source ranks (1-based)."""
    candidate_by_ref = _candidate_index_by_ref(candidates, domain="relationship")
    ordered_refs = sorted(
        candidate_by_ref,
        key=lambda ref: (candidate_by_ref[ref].source_order_key, ref),
    )
    return {ref: ordinal for ordinal, ref in enumerate(ordered_refs, start=1)}


def _relationship_states(
    members: Sequence[IndexedRelationshipCandidate],
    source_ranks: dict[str, int],
) -> tuple[RelationshipState, ...]:
    """Build exact consecutive state groups from source-ordered members."""
    groups: list[list[IndexedRelationshipCandidate]] = []
    for member in members:
        if member.state_zh is None:
            continue
        if groups and groups[-1][0].state_zh == member.state_zh:
            groups[-1].append(member)
        else:
            groups.append([member])

    states: list[RelationshipState] = []
    for group in groups:
        first = group[0]
        evidence_refs = _stable_evidence_union(group)
        _require_unique_evidence(
            evidence_refs,
            label=f"relationship state {first.state_zh!r}",
        )
        states.append(
            RelationshipState(
                state_zh=first.state_zh,
                candidate_relationship_refs=tuple(
                    member.global_candidate_ref for member in group
                ),
                evidence_refs=evidence_refs,
                narrative_order=source_ranks[first.global_candidate_ref],
            )
        )
    return tuple(states)


def _validate_relationship_state_history(
    relationship: CanonicalRelationship,
    candidate_by_ref: dict[str, IndexedRelationshipCandidate],
    source_ranks: dict[str, int],
) -> None:
    """Validate exact state provenance against its canonical parent (fail closed)."""
    parent_refs = relationship.candidate_relationship_refs
    parent_ref_set = set(parent_refs)
    member_positions = {ref: position for position, ref in enumerate(parent_refs)}
    previous_position = -1
    for state in relationship.state_history:
        if not isinstance(state, RelationshipState):
            raise ConsolidationFinalizationError("relationship state_history has invalid item")
        _require_unique_evidence(
            state.evidence_refs,
            label=f"relationship state {state.state_zh!r}",
        )
        if not state.candidate_relationship_refs:
            raise ConsolidationFinalizationError("relationship state has no candidate refs")
        positions: list[int] = []
        contributors: list[IndexedRelationshipCandidate] = []
        for ref in state.candidate_relationship_refs:
            if ref not in parent_ref_set:
                raise ConsolidationFinalizationError(
                    f"relationship state candidate {ref!r} is outside its parent"
                )
            candidate = candidate_by_ref.get(ref)
            if candidate is None:
                raise ConsolidationFinalizationError(
                    f"relationship state candidate {ref!r} is not indexed"
                )
            if candidate.state_zh != state.state_zh:
                raise ConsolidationFinalizationError(
                    f"relationship state candidate {ref!r} does not exactly match "
                    "the state_zh"
                )
            positions.append(member_positions[ref])
            contributors.append(candidate)
        if positions != sorted(positions) or positions[0] <= previous_position:
            raise ConsolidationFinalizationError(
                "relationship state candidate refs are not in parent source order"
            )
        previous_position = positions[-1]
        expected_evidence = _stable_evidence_union(contributors)
        if state.evidence_refs != expected_evidence:
            raise ConsolidationFinalizationError(
                "relationship state evidence is not the exact stable contributor union"
            )
        expected_rank = source_ranks[state.candidate_relationship_refs[0]]
        if state.narrative_order != expected_rank:
            raise ConsolidationFinalizationError(
                "relationship state narrative_order is not its global source rank"
            )


def build_canonical_relationship_set(
    planning_result: ConsolidationPlanningResult,
    identity_plan: ConsolidationIdentityPlan,
) -> CanonicalRelationshipSet:
    """Build A5E3 CanonicalRelationshipSet from checked A5E1 components.

    Identity remains A5E1 authority.  This only assembles the representative
    relationship payload and exact candidate-derived state histories.
    """
    components, candidate_by_ref = _validate_identity_components(
        planning_result=planning_result,
        identity_plan=identity_plan,
        domain="relationship",
    )
    source_ranks = _relationship_source_ranks(planning_result.index.relationships)
    relationships: list[CanonicalRelationship] = []
    for component in components:
        members = tuple(candidate_by_ref[ref] for ref in component.member_candidate_refs)
        _check_relationship_signature(component.member_candidate_refs, candidate_by_ref)
        direction, source, target = _relationship_signature(members[0])
        relationship = CanonicalRelationship(
            relationship_id=component.canonical_id,
            source_entity_ref=source,
            target_entity_ref=target,
            direction=direction,
            relationship_type_zh=members[0].relationship_type_zh,
            candidate_relationship_refs=component.member_candidate_refs,
            state_history=_relationship_states(members, source_ranks),
            first_source_order=members[0].source_order_key,
        )
        _validate_relationship_state_history(relationship, candidate_by_ref, source_ranks)
        relationships.append(relationship)

    refs = {
        ref for relationship in relationships for ref in relationship.candidate_relationship_refs
    }
    if refs != set(candidate_by_ref):
        raise ConsolidationFinalizationError(
            "relationship: canonical relationship candidate coverage is not exact"
        )
    if len(refs) != sum(len(r.candidate_relationship_refs) for r in relationships):
        raise ConsolidationFinalizationError(
            "relationship: canonical relationship candidate refs are not globally unique"
        )
    if len({r.relationship_id for r in relationships}) != len(relationships):
        raise ConsolidationFinalizationError("relationship: canonical ids are not unique")
    return CanonicalRelationshipSet(schema_version=1, relationships=tuple(relationships))


@dataclass(frozen=True, slots=True)
class A5FinalizationResult:
    """Final A5E in-memory boundary; ``to_dict`` is comparison-only, not persistence."""

    planning_result: ConsolidationPlanningResult
    canonical_fact_set: CanonicalFactSet
    canonical_event_set: CanonicalEventSet
    canonical_relationship_set: CanonicalRelationshipSet
    story_conflict_set: StoryConflictSet

    def __post_init__(self) -> None:
        if not isinstance(self.planning_result, ConsolidationPlanningResult):
            raise ConsolidationFinalizationError(
                "planning_result must be a ConsolidationPlanningResult"
            )
        for name, expected_type in (
            ("canonical_fact_set", CanonicalFactSet),
            ("canonical_event_set", CanonicalEventSet),
            ("canonical_relationship_set", CanonicalRelationshipSet),
            ("story_conflict_set", StoryConflictSet),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise ConsolidationFinalizationError(f"{name} has an invalid type")

    def to_dict(self) -> dict[str, Any]:
        """Normalized in-memory comparison material; NOT a persistence schema."""
        return {
            "plan_hash": self.planning_result.plan_hash,
            "canonical_fact_set": self.canonical_fact_set.to_dict(),
            "canonical_event_set": self.canonical_event_set.to_dict(),
            "canonical_relationship_set": self.canonical_relationship_set.to_dict(),
            "story_conflict_set": self.story_conflict_set.to_dict(),
        }


def _validate_final_composition(
    result: A5FinalizationResult,
    identity_plan: ConsolidationIdentityPlan,
    fact_resolution: FactSemanticResolutionResult,
) -> None:
    """Validate the frozen final A5E cross-artifact invariants (no mutation)."""
    planning_result = result.planning_result
    fact_components, fact_candidates = _validate_identity_components(
        planning_result=planning_result, identity_plan=identity_plan, domain="fact"
    )
    event_components, event_candidates = _validate_identity_components(
        planning_result=planning_result, identity_plan=identity_plan, domain="event"
    )
    relationship_components, relationship_candidates = _validate_identity_components(
        planning_result=planning_result, identity_plan=identity_plan, domain="relationship"
    )

    facts_by_id = {fact.fact_id: fact for fact in result.canonical_fact_set.facts}
    if (
        len(facts_by_id) != len(result.canonical_fact_set.facts)
        or len(result.canonical_fact_set.facts) != len(fact_components)
    ):
        raise ConsolidationFinalizationError("final facts have duplicate ids")
    fact_ref_to_id: dict[str, tuple[str, int]] = {}
    for ordinal, (component, fact) in enumerate(
        zip(fact_components, result.canonical_fact_set.facts), start=1
    ):
        if fact.fact_id != component.canonical_id or fact.candidate_fact_refs != component.member_candidate_refs:
            raise ConsolidationFinalizationError("final facts do not match identity components")
        _require_unique_evidence(fact.evidence_refs, label=f"fact {fact.fact_id}")
        for ref in fact.candidate_fact_refs:
            if ref in fact_ref_to_id:
                raise ConsolidationFinalizationError("final fact refs are not globally unique")
            fact_ref_to_id[ref] = (fact.fact_id, ordinal)
    if set(fact_ref_to_id) != set(fact_candidates):
        raise ConsolidationFinalizationError("final fact candidate coverage is not exact")

    decisions_by_id: dict[str, FactSemanticDecision] = {}
    for decision in fact_resolution.all_fact_decisions:
        if not isinstance(decision, FactSemanticDecision) or decision.decision_id in decisions_by_id:
            raise ConsolidationFinalizationError("final fact decision lookup is ambiguous")
        decisions_by_id[decision.decision_id] = decision
    transition_decision_ids: set[str] = set()
    for ordinal, transition in enumerate(
        result.canonical_fact_set.state_transitions, start=1
    ):
        if transition.transition_id != _canonical_id("trans", ordinal):
            raise ConsolidationFinalizationError(
                "final StateTransition canonical id is invalid"
            )
        if transition.source_decision_ref in transition_decision_ids:
            raise ConsolidationFinalizationError(
                "final StateTransition source decision is duplicated"
            )
        _require_unique_evidence(transition.evidence_refs, label=transition.transition_id)
        decision = decisions_by_id.get(transition.source_decision_ref)
        if (
            transition.transition_kind != "state_change"
            or decision is None
            or decision.decision != "state_change"
            or transition.evidence_refs != decision.evidence_refs
            or transition.from_fact_id not in facts_by_id
            or transition.to_fact_id not in facts_by_id
            or transition.from_fact_id == transition.to_fact_id
        ):
            raise ConsolidationFinalizationError("final StateTransition is invalid")
        to_ordinal = next(
            ordinal for fact_id, ordinal in fact_ref_to_id.values() if fact_id == transition.to_fact_id
        )
        if transition.narrative_order != to_ordinal:
            raise ConsolidationFinalizationError("final StateTransition narrative_order is invalid")
        left_id, left_ordinal = fact_ref_to_id[decision.left_candidate_ref]
        right_id, right_ordinal = fact_ref_to_id[decision.right_candidate_ref]
        expected_from, expected_to = (
            (left_id, right_id)
            if left_ordinal < right_ordinal
            else (right_id, left_id)
        )
        if (transition.from_fact_id, transition.to_fact_id) != (expected_from, expected_to):
            raise ConsolidationFinalizationError("final StateTransition direction is invalid")
        transition_decision_ids.add(transition.source_decision_ref)
    if transition_decision_ids != {
        decision_id
        for decision_id, decision in decisions_by_id.items()
        if decision.decision == "state_change"
    }:
        raise ConsolidationFinalizationError("final StateTransition decision coverage is invalid")

    events = result.canonical_event_set.events
    if len(events) != len(event_components) or tuple(e.narrative_order for e in events) != tuple(range(1, len(events) + 1)):
        raise ConsolidationFinalizationError("final event sequence is invalid")
    event_refs: set[str] = set()
    for ordinal, (component, event) in enumerate(zip(event_components, events), start=1):
        if event.event_id != component.canonical_id or event.candidate_event_refs != component.member_candidate_refs:
            raise ConsolidationFinalizationError("final events do not match identity components")
        if event.event_id != _canonical_id("evt", ordinal):
            raise ConsolidationFinalizationError("final event canonical id is invalid")
        _require_unique_evidence(event.evidence_refs, label=f"event {event.event_id}")
        if event_refs.intersection(event.candidate_event_refs):
            raise ConsolidationFinalizationError("final event refs are not globally unique")
        event_refs.update(event.candidate_event_refs)
    if event_refs != set(event_candidates):
        raise ConsolidationFinalizationError("final event candidate coverage is not exact")

    relationships = result.canonical_relationship_set.relationships
    if len(relationships) != len(relationship_components):
        raise ConsolidationFinalizationError("final relationship count is invalid")
    source_ranks = _relationship_source_ranks(planning_result.index.relationships)
    relationship_refs: set[str] = set()
    for ordinal, (component, relationship) in enumerate(
        zip(relationship_components, relationships), start=1
    ):
        if (
            relationship.relationship_id != component.canonical_id
            or relationship.candidate_relationship_refs != component.member_candidate_refs
        ):
            raise ConsolidationFinalizationError(
                "final relationships do not match identity components"
            )
        _check_relationship_signature(component.member_candidate_refs, relationship_candidates)
        signature = _relationship_signature(relationship_candidates[component.member_candidate_refs[0]])
        if (relationship.direction, relationship.source_entity_ref, relationship.target_entity_ref) != signature:
            raise ConsolidationFinalizationError("final relationship endpoint signature is invalid")
        representative = relationship_candidates[component.member_candidate_refs[0]]
        if (
            relationship.relationship_id != _canonical_id("rel", ordinal)
            or relationship.relationship_type_zh != representative.relationship_type_zh
            or relationship.first_source_order != representative.source_order_key
        ):
            raise ConsolidationFinalizationError("final relationship representative fields are invalid")
        _validate_relationship_state_history(relationship, relationship_candidates, source_ranks)
        expected_states = _relationship_states(
            tuple(relationship_candidates[ref] for ref in component.member_candidate_refs),
            source_ranks,
        )
        if relationship.state_history != expected_states:
            raise ConsolidationFinalizationError("final relationship state history is invalid")
        if relationship_refs.intersection(relationship.candidate_relationship_refs):
            raise ConsolidationFinalizationError(
                "final relationship refs are not globally unique"
            )
        relationship_refs.update(relationship.candidate_relationship_refs)
    if relationship_refs != set(relationship_candidates):
        raise ConsolidationFinalizationError("final relationship candidate coverage is not exact")

    conflict_decision_ids: set[str] = set()
    for ordinal, conflict in enumerate(result.story_conflict_set.conflicts, start=1):
        if conflict.conflict_id != _canonical_id("conf", ordinal):
            raise ConsolidationFinalizationError(
                "final StoryConflict canonical id is invalid"
            )
        if len(conflict.decision_refs) == 1 and conflict.decision_refs[0] in conflict_decision_ids:
            raise ConsolidationFinalizationError(
                "final StoryConflict source decision is duplicated"
            )
        _require_unique_evidence(conflict.evidence_refs, label=conflict.conflict_id)
        if (
            conflict.conflict_kind != "fact_conflict"
            or conflict.relationship_ids != ()
            or conflict.status != "unresolved"
            or len(conflict.decision_refs) != 1
            or any(fact_id not in facts_by_id for fact_id in conflict.fact_ids)
            or any(ref not in fact_candidates for ref in conflict.candidate_refs)
        ):
            raise ConsolidationFinalizationError("final StoryConflict is invalid")
        decision = decisions_by_id.get(conflict.decision_refs[0])
        if (
            decision is None
            or decision.decision != "conflict"
            or conflict.evidence_refs != decision.evidence_refs
            or conflict.candidate_refs
            != (decision.left_candidate_ref, decision.right_candidate_ref)
        ):
            raise ConsolidationFinalizationError("final StoryConflict decision reference is invalid")
        expected_fact_ids = tuple(
            fact_ref_to_id[ref][0]
            for ref in sorted(
                conflict.candidate_refs, key=lambda ref: fact_ref_to_id[ref][1]
            )
        )
        if conflict.fact_ids != expected_fact_ids:
            raise ConsolidationFinalizationError("final StoryConflict fact ordering is invalid")
        conflict_decision_ids.add(conflict.decision_refs[0])
    if conflict_decision_ids != {
        decision_id
        for decision_id, decision in decisions_by_id.items()
        if decision.decision == "conflict"
    }:
        raise ConsolidationFinalizationError("final StoryConflict decision coverage is invalid")


def finalize_consolidation(
    planning_result: ConsolidationPlanningResult,
    fact_resolution: FactSemanticResolutionResult,
    event_resolution: EventSemanticResolutionResult,
    relationship_resolution: RelationshipSemanticResolutionResult,
) -> A5FinalizationResult:
    """Finalize all A5E domains in memory, with no provider or persistence work."""
    if not isinstance(planning_result, ConsolidationPlanningResult):
        raise ConsolidationFinalizationError("planning_result must be a ConsolidationPlanningResult")
    expected = (
        ("fact", fact_resolution, FactSemanticResolutionResult),
        ("event", event_resolution, EventSemanticResolutionResult),
        ("relationship", relationship_resolution, RelationshipSemanticResolutionResult),
    )
    for domain, resolution, resolution_type in expected:
        if not isinstance(resolution, resolution_type):
            raise ConsolidationFinalizationError(f"{domain}_resolution has an invalid type")
        if not isinstance(resolution.planning_result, ConsolidationPlanningResult):
            raise ConsolidationFinalizationError(
                f"{domain}_resolution.planning_result must be a ConsolidationPlanningResult"
            )
        if resolution.planning_result.plan_hash != planning_result.plan_hash:
            raise ConsolidationFinalizationError(
                f"{domain} resolution planning identity does not match planning_result.plan_hash"
            )
    identity_plan = finalize_consolidation_identity(
        planning_result, fact_resolution, event_resolution, relationship_resolution
    )
    result = A5FinalizationResult(
        planning_result=planning_result,
        canonical_fact_set=build_canonical_fact_set(
            planning_result, identity_plan, fact_resolution
        ),
        canonical_event_set=build_canonical_event_set(planning_result, identity_plan),
        canonical_relationship_set=build_canonical_relationship_set(
            planning_result, identity_plan
        ),
        story_conflict_set=build_fact_story_conflict_set(
            planning_result, identity_plan, fact_resolution
        ),
    )
    _validate_final_composition(result, identity_plan, fact_resolution)
    return result


__all__ = [
    "ConsolidationFinalizationError",
    "A5FinalizationResult",
    "ConsolidationIdentityComponent",
    "ConsolidationIdentityPlan",
    "build_event_identity_components",
    "build_fact_identity_components",
    "build_relationship_identity_components",
    "build_canonical_event_set",
    "build_canonical_fact_set",
    "build_fact_story_conflict_set",
    "build_canonical_relationship_set",
    "finalize_consolidation",
    "finalize_consolidation_identity",
]
