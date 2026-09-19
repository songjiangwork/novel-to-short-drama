"""v1.2 A4D — deterministic reconciliation validation findings.

This module produces the deterministic A4D :class:`ValidationFinding` values
for the reconciliation graph finalization. Every finding is BLOCKING, owned by
stage ``A4`` with repair route ``rerun_a4``. Finding IDs are deterministic
(via shared ``content_hash``) so identical corruption yields identical
findings.

The findings are computed over in-memory domain objects (no artifact refs are
required). ``pair_plans`` is optional: when provided, the pair-plan-dependent
findings (duplicate / missing / unexpected pair + deterministic-constraint
override) are checked; when omitted (CURRENT verification over persisted
artifacts) those are skipped because pair plans are not persisted.
"""

from __future__ import annotations

from typing import Any

from short_drama.artifacts import content_hash
from short_drama.foundation.validation import (
    ValidationFinding,
    ValidationSeverity,
)
from short_drama.story.reconciliation import (
    CandidateEntityIndex,
    CanonicalCharacterRegistry,
    CanonicalLocationRegistry,
    EntityMapEntry,
    ReconciliationDecisionSet,
    UnresolvedEntitySet,
)
from short_drama.story.reconciliation_planning import (
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_MUST_NOT_MERGE,
)

# Finding codes (all BLOCKING, owner A4, repair route rerun_a4).
A4_CANDIDATE_REF_DUPLICATE = "A4_CANDIDATE_REF_DUPLICATE"
A4_CANDIDATE_REF_NOT_FOUND = "A4_CANDIDATE_REF_NOT_FOUND"
A4_DECISION_PAIR_DUPLICATE = "A4_DECISION_PAIR_DUPLICATE"
A4_DECISION_PAIR_MISSING = "A4_DECISION_PAIR_MISSING"
A4_DECISION_PAIR_UNEXPECTED = "A4_DECISION_PAIR_UNEXPECTED"
A4_DECISION_REF_NOT_FOUND = "A4_DECISION_REF_NOT_FOUND"
A4_DECISION_TYPE_MISMATCH = "A4_DECISION_TYPE_MISMATCH"
A4_DETERMINISTIC_CONSTRAINT_OVERRIDE = "A4_DETERMINISTIC_CONSTRAINT_OVERRIDE"
A4_RECONCILIATION_CONFLICT = "A4_RECONCILIATION_CONFLICT"
A4_CANONICAL_ID_GAP = "A4_CANONICAL_ID_GAP"
A4_ENTITY_MAP_DUPLICATE = "A4_ENTITY_MAP_DUPLICATE"
A4_ENTITY_MAP_UNACCOUNTED = "A4_ENTITY_MAP_UNACCOUNTED"
A4_ENTITY_MAP_TARGET_NOT_FOUND = "A4_ENTITY_MAP_TARGET_NOT_FOUND"


def _normalize(value: Any) -> Any:
    """Reduce to a canonical-JSON value (tuples/sets become lists)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize(v) for v in value)
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _finding(code: str, message: str, **identity: Any) -> ValidationFinding:
    material = {"code": code, "identity": _normalize(identity), "message": message}
    finding_id = f"a4-{code.lower()}-{content_hash(material)[:20]}"
    return ValidationFinding(
        finding_id=finding_id,
        code=code,
        severity=ValidationSeverity.BLOCKING,
        owner_stage="A4",
        repair_route="rerun_a4",
        message=message,
        artifact_refs=(),
    )


def _check_canonical_id_gaps(
    entity_type: str, registry: Any
) -> tuple[ValidationFinding, ...]:
    """Verify canonical IDs are gap-free starting at 1 (char_NNNN / loc_NNNN)."""
    findings = []
    ids = sorted(entity.canonical_id for entity in registry.entities)
    prefix = "char_" if entity_type == "character" else "loc_"
    for expected_index, entity_id in enumerate(ids, start=1):
        expected = f"{prefix}{expected_index:04d}"
        if entity_id != expected:
            findings.append(
                _finding(
                    A4_CANONICAL_ID_GAP,
                    f"{entity_type} registry canonical id gap: expected {expected!r}, "
                    f"got {entity_id!r}",
                    entity_type=entity_type,
                    canonical_id=entity_id,
                    expected_canonical_id=expected,
                )
            )
    return tuple(findings)


def validate_finalization(
    *,
    candidate_index: CandidateEntityIndex,
    decision_set: ReconciliationDecisionSet,
    graph: Any,
    canonical_character_registry: CanonicalCharacterRegistry,
    canonical_location_registry: CanonicalLocationRegistry,
    unresolved_entity_set: UnresolvedEntitySet,
    entity_map_entries: tuple[EntityMapEntry, ...],
    pair_plans: tuple | None = None,
) -> tuple[ValidationFinding, ...]:
    """Compute the deterministic A4D validation findings.

    See the module docstring for which findings depend on ``pair_plans``.
    """
    findings: list[ValidationFinding] = []

    # Index candidate-ref uniqueness + lookup.
    ref_to_entry: dict[str, Any] = {}
    seen_refs: set[str] = set()
    for entry in candidate_index.entries:
        if entry.candidate_ref in seen_refs:
            findings.append(
                _finding(
                    A4_CANDIDATE_REF_DUPLICATE,
                    f"duplicate candidate_ref {entry.candidate_ref!r} in candidate index",
                    candidate_ref=entry.candidate_ref,
                )
            )
        else:
            seen_refs.add(entry.candidate_ref)
            ref_to_entry[entry.candidate_ref] = entry

    decisions = decision_set.decisions
    decision_ids: set[str] = set()
    decision_by_pair: dict[tuple[str, str], Any] = {}
    for decision in decisions:
        decision_ids.add(decision.decision_id)
        key = (decision.left_candidate_ref, decision.right_candidate_ref)
        if key in decision_by_pair:
            findings.append(
                _finding(
                    A4_DECISION_PAIR_DUPLICATE,
                    f"duplicate decision for pair {key!r}",
                    pair=key,
                )
            )
        else:
            decision_by_pair[key] = decision

    # Decision endpoints must exist in the index.
    for decision in decisions:
        for ref in (decision.left_candidate_ref, decision.right_candidate_ref):
            if ref not in ref_to_entry:
                findings.append(
                    _finding(
                        A4_CANDIDATE_REF_NOT_FOUND,
                        f"decision {decision.decision_id!r} references unknown "
                        f"candidate {ref!r}",
                        decision_id=decision.decision_id,
                        candidate_ref=ref,
                    )
                )

    # Cross-type decision pair endpoints.
    for decision in decisions:
        left = ref_to_entry.get(decision.left_candidate_ref)
        right = ref_to_entry.get(decision.right_candidate_ref)
        if (
            left is not None
            and right is not None
            and left.candidate_kind != right.candidate_kind
        ):
            findings.append(
                _finding(
                    A4_DECISION_TYPE_MISMATCH,
                    f"decision {decision.decision_id!r} mixes candidate kinds "
                    f"{left.candidate_kind!r} and {right.candidate_kind!r}",
                    decision_id=decision.decision_id,
                    left_candidate_ref=decision.left_candidate_ref,
                    right_candidate_ref=decision.right_candidate_ref,
                )
            )

    # Pair-plan-dependent findings (only when pair_plans provided).
    if pair_plans is not None:
        plan_by_pair: dict[tuple[str, str], Any] = {}
        for plan in pair_plans:
            plan_by_pair[(plan.left_candidate_ref, plan.right_candidate_ref)] = plan

        for key in decision_by_pair:
            if key not in plan_by_pair:
                findings.append(
                    _finding(
                        A4_DECISION_PAIR_UNEXPECTED,
                        f"decision found for pair {key!r} not in planning pair plans",
                        pair=key,
                    )
                )
        for key in plan_by_pair:
            if key not in decision_by_pair:
                findings.append(
                    _finding(
                        A4_DECISION_PAIR_MISSING,
                        f"no decision for pair {key!r} present in planning pair plans",
                        pair=key,
                    )
                )
        for key, plan in plan_by_pair.items():
            decision = decision_by_pair.get(key)
            if decision is None:
                continue
            if plan.state == PAIR_STATE_AUTO_SAME and decision.decision != "same_entity":
                findings.append(
                    _finding(
                        A4_DETERMINISTIC_CONSTRAINT_OVERRIDE,
                        f"pair {key!r} (auto_same) resolved as "
                        f"{decision.decision!r}; expected same_entity",
                        pair=key,
                        expected="same_entity",
                    )
                )
            elif (
                plan.state == PAIR_STATE_MUST_NOT_MERGE
                and decision.decision != "different_entity"
            ):
                findings.append(
                    _finding(
                        A4_DETERMINISTIC_CONSTRAINT_OVERRIDE,
                        f"pair {key!r} (must_not_merge) resolved as "
                        f"{decision.decision!r}; expected different_entity",
                        pair=key,
                        expected="different_entity",
                    )
                )

    # Graph conflict.
    for decision in graph.conflict_decisions:
        findings.append(
            _finding(
                A4_RECONCILIATION_CONFLICT,
                f"different_entity decision {decision.decision_id!r} contradicts a "
                f"same_entity connected component",
                decision_id=decision.decision_id,
                left_candidate_ref=decision.left_candidate_ref,
                right_candidate_ref=decision.right_candidate_ref,
            )
        )

    # Canonical ID gaps.
    findings.extend(
        _check_canonical_id_gaps("character", canonical_character_registry)
    )
    findings.extend(_check_canonical_id_gaps("location", canonical_location_registry))

    # EntityMap coverage.
    entries_by_ref: dict[str, list[EntityMapEntry]] = {}
    for entry in entity_map_entries:
        entries_by_ref.setdefault(entry.candidate_ref, []).append(entry)

    for ref, entries in entries_by_ref.items():
        if len(entries) > 1:
            findings.append(
                _finding(
                    A4_ENTITY_MAP_DUPLICATE,
                    f"candidate {ref!r} mapped to {len(entries)} EntityMap entries",
                    candidate_ref=ref,
                )
            )

    for entry in entity_map_entries:
        if entry.candidate_ref not in ref_to_entry:
            findings.append(
                _finding(
                    A4_CANDIDATE_REF_NOT_FOUND,
                    f"EntityMap entry references unknown candidate "
                    f"{entry.candidate_ref!r}",
                    candidate_ref=entry.candidate_ref,
                )
            )

    for ref in ref_to_entry:
        if ref not in entries_by_ref:
            findings.append(
                _finding(
                    A4_ENTITY_MAP_UNACCOUNTED,
                    f"candidate {ref!r} has no EntityMap entry",
                    candidate_ref=ref,
                )
            )

    character_ids = {e.canonical_id: e for e in canonical_character_registry.entities}
    location_ids = {e.canonical_id: e for e in canonical_location_registry.entities}
    unresolved_ids = {e.unresolved_id: e for e in unresolved_entity_set.entities}
    for entry in entity_map_entries:
        index_entry = ref_to_entry.get(entry.candidate_ref)
        if index_entry is None:
            continue
        if entry.status == "resolved":
            registry = (
                character_ids
                if index_entry.candidate_kind == "character"
                else location_ids
            )
            target = registry.get(entry.canonical_id)
            if target is None or entry.candidate_ref not in target.candidate_refs:
                findings.append(
                    _finding(
                        A4_ENTITY_MAP_TARGET_NOT_FOUND,
                        f"EntityMap entry for {entry.candidate_ref!r} does not "
                        f"resolve to a valid canonical target {entry.canonical_id!r}",
                        candidate_ref=entry.candidate_ref,
                        canonical_id=entry.canonical_id,
                    )
                )
        else:
            target = unresolved_ids.get(entry.unresolved_id)
            if target is None or entry.candidate_ref not in target.candidate_refs:
                findings.append(
                    _finding(
                        A4_ENTITY_MAP_TARGET_NOT_FOUND,
                        f"EntityMap entry for {entry.candidate_ref!r} does not "
                        f"resolve to a valid unresolved target {entry.unresolved_id!r}",
                        candidate_ref=entry.candidate_ref,
                        unresolved_id=entry.unresolved_id,
                    )
                )

    # A target entity must not claim a candidate mapped elsewhere.
    claimant: dict[str, set[str]] = {}
    for entity in (*canonical_character_registry.entities, *canonical_location_registry.entities):
        for ref in entity.candidate_refs:
            claimant.setdefault(ref, set()).add(entity.canonical_id)
    for entity in unresolved_entity_set.entities:
        for ref in entity.candidate_refs:
            claimant.setdefault(ref, set()).add(entity.unresolved_id)
    for ref, owners in claimant.items():
        if len(owners) > 1:
            findings.append(
                _finding(
                    A4_ENTITY_MAP_TARGET_NOT_FOUND,
                    f"candidate {ref!r} claimed by multiple target entities "
                    f"{sorted(owners)!r}",
                    candidate_ref=ref,
                    target_entities=sorted(owners),
                )
            )

    # Unresolved decision_refs must reference existing decisions.
    for entity in unresolved_entity_set.entities:
        for decision_ref in entity.decision_refs:
            if decision_ref not in decision_ids:
                findings.append(
                    _finding(
                        A4_DECISION_REF_NOT_FOUND,
                        f"unresolved entity {entity.unresolved_id!r} references unknown "
                        f"decision {decision_ref!r}",
                        unresolved_id=entity.unresolved_id,
                        decision_ref=decision_ref,
                    )
                )

    # Deduplicate by finding_id (some checks can produce identical findings).
    deduped: dict[str, ValidationFinding] = {}
    for finding in findings:
        deduped.setdefault(finding.finding_id, finding)
    return tuple(deduped.values())


__all__ = [
    "A4_CANDIDATE_REF_DUPLICATE",
    "A4_CANDIDATE_REF_NOT_FOUND",
    "A4_DECISION_PAIR_DUPLICATE",
    "A4_DECISION_PAIR_MISSING",
    "A4_DECISION_PAIR_UNEXPECTED",
    "A4_DECISION_REF_NOT_FOUND",
    "A4_DECISION_TYPE_MISMATCH",
    "A4_DETERMINISTIC_CONSTRAINT_OVERRIDE",
    "A4_RECONCILIATION_CONFLICT",
    "A4_CANONICAL_ID_GAP",
    "A4_ENTITY_MAP_DUPLICATE",
    "A4_ENTITY_MAP_UNACCOUNTED",
    "A4_ENTITY_MAP_TARGET_NOT_FOUND",
    "validate_finalization",
]
