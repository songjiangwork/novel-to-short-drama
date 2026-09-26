"""Focused A5E3 tests: canonical relationships and final in-memory composition."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from short_drama.artifacts import ArtifactRef
from short_drama.story import (
    A5FinalizationResult,
    ConsolidationCandidateIndex,
    ConsolidationCoverageSummary,
    ConsolidationFinalizationError,
    ConsolidationIdentityComponent,
    ConsolidationIdentityPlan,
    ConsolidationPlanningResult,
    EventSemanticResolutionResult,
    FactSemanticDecision,
    FactSemanticResolutionResult,
    IndexedEventCandidate,
    IndexedFactCandidate,
    IndexedRelationshipCandidate,
    RelationshipSemanticResolutionResult,
    build_canonical_relationship_set,
    finalize_consolidation,
)
from short_drama.story.extraction import EvidenceRef


def _artifact() -> ArtifactRef:
    return ArtifactRef("candidate_extraction", "ce000001", 1, "0" * 64)


def _evidence(n: int, excerpt: str | None = "证据") -> EvidenceRef:
    return EvidenceRef(f"CH001_P{n:04d}", "primary", "explicit", excerpt)


def _rel_ref(n: int) -> str:
    return f"CH001_C001:cand_rel_{n:03d}"


def _relationship(
    n: int, *, source_order: int | None = None, source: str = "char_0001",
    target: str = "char_0002", direction: str = "directed", kind: str = "朋友",
    state: str | None = None, evidence: tuple[EvidenceRef, ...] | None = None,
) -> IndexedRelationshipCandidate:
    return IndexedRelationshipCandidate(
        global_candidate_ref=_rel_ref(n), chunk_id="CH001_C001", local_candidate_id=f"cand_rel_{n:03d}",
        source_order_key=f"{source_order or n:09d}", source_entity_ref=source,
        target_entity_ref=target, relationship_type_zh=kind, state_zh=state,
        direction=direction, evidence_strength="explicit", evidence_refs=evidence or (_evidence(n),),
        candidate_extraction_ref=_artifact(),
    )


def _component(ordinal: int, *refs: str, first: int | None = None) -> ConsolidationIdentityComponent:
    return ConsolidationIdentityComponent(
        domain="relationship", canonical_id=f"rel_{ordinal:06d}", member_candidate_refs=tuple(refs),
        first_source_order=f"{first if first is not None else int(refs[0][-3:]):09d}",
    )


def _planning(*, relationships=(), facts=(), events=(), plan_hash="a" * 64):
    return ConsolidationPlanningResult(
        snapshot=object(),
        index=ConsolidationCandidateIndex(1, tuple(facts), tuple(events), tuple(relationships)),
        coverage=ConsolidationCoverageSummary(len(facts), len(events), len(relationships), 0, 0, 0, 0, 0),
        fact_pair_plans=(), event_pair_plans=(), relationship_pair_plans=(), deterministic_decision_set=None,
        blocking_policy_id="bp", text_normalization_policy_id="tn", exact_safe_policy_id="es",
        planning_policy_id="pp", plan_hash=plan_hash,
    )


def _identity(planning, relationships=()):
    return ConsolidationIdentityPlan(planning.plan_hash, (), (), tuple(relationships))


class TestCanonicalRelationship:
    def test_directed_representative_and_singleton_are_preserved(self):
        first = _relationship(1, kind="同事", state="戒备")
        second = _relationship(2, kind="陌生", state=None)
        planning = _planning(relationships=(second, first))
        identity = _identity(planning, (_component(1, _rel_ref(1), _rel_ref(2)),))
        relationship = build_canonical_relationship_set(planning, identity).relationships[0]
        assert relationship.relationship_id == "rel_000001"
        assert (relationship.source_entity_ref, relationship.target_entity_ref, relationship.direction) == (
            "char_0001", "char_0002", "directed"
        )
        assert relationship.relationship_type_zh == "同事"
        assert relationship.candidate_relationship_refs == (_rel_ref(1), _rel_ref(2))
        assert relationship.first_source_order == "000000001"
        assert relationship.state_history[0].candidate_relationship_refs == (_rel_ref(1),)

    def test_symmetric_canonicalizes_and_unknown_does_not(self):
        sym_a = _relationship(1, source="char_0002", target="char_0001", direction="symmetric")
        sym_b = _relationship(2, source="char_0001", target="char_0002", direction="symmetric")
        planning = _planning(relationships=(sym_a, sym_b))
        symmetric = build_canonical_relationship_set(planning, _identity(planning, (_component(1, _rel_ref(1), _rel_ref(2)),))).relationships[0]
        assert (symmetric.direction, symmetric.source_entity_ref, symmetric.target_entity_ref) == (
            "symmetric", "char_0001", "char_0002"
        )
        unknown = _relationship(1, source="char_0002", target="char_0001", direction="unknown")
        planning = _planning(relationships=(unknown,))
        output = build_canonical_relationship_set(planning, _identity(planning, (_component(1, _rel_ref(1)),))).relationships[0]
        assert (output.direction, output.source_entity_ref, output.target_entity_ref) == (
            "unknown", "char_0002", "char_0001"
        )

    @pytest.mark.parametrize(
        "left,right",
        [
            (_relationship(1, source="char_0001", target="char_0002"), _relationship(2, source="char_0002", target="char_0001")),
            (_relationship(1, direction="directed"), _relationship(2, direction="symmetric")),
            (_relationship(1, direction="unknown"), _relationship(2, source="char_0002", target="char_0001", direction="unknown")),
        ],
    )
    def test_mismatched_signature_fails_closed(self, left, right):
        planning = _planning(relationships=(left, right))
        identity = _identity(planning, (_component(1, _rel_ref(1), _rel_ref(2)),))
        with pytest.raises(ConsolidationFinalizationError, match="mismatched"):
            build_canonical_relationship_set(planning, identity)


class TestStateHistory:
    def _one_component(self, states):
        relationships = tuple(_relationship(n, state=state) for n, state in enumerate(states, 1))
        planning = _planning(relationships=relationships)
        identity = _identity(planning, (_component(1, *[_rel_ref(n) for n in range(1, len(states) + 1)]),))
        return build_canonical_relationship_set(planning, identity).relationships[0]

    @pytest.mark.parametrize(
        "states, expected",
        [
            ((None, None), ()),
            (("A", "A"), (("A", (_rel_ref(1), _rel_ref(2))),)),
            (("A", "A", "B", "B", "A"), (("A", (_rel_ref(1), _rel_ref(2))), ("B", (_rel_ref(3), _rel_ref(4))), ("A", (_rel_ref(5),)))),
            (("A", None, "A"), (("A", (_rel_ref(1), _rel_ref(3))),)),
            ((None, "A", None, "B"), (("A", (_rel_ref(2),)), ("B", (_rel_ref(4),)))),
        ],
    )
    def test_observed_sequence_and_none_semantics(self, states, expected):
        relationship = self._one_component(states)
        assert tuple((state.state_zh, state.candidate_relationship_refs) for state in relationship.state_history) == expected
        assert relationship.candidate_relationship_refs == tuple(_rel_ref(n) for n in range(1, len(states) + 1))

    def test_state_evidence_and_global_source_rank(self):
        # Relationship A has members at global ranks 1 and 3; B interleaves at ranks 2 and 4.
        a1 = _relationship(1, source_order=1, state="A", evidence=(_evidence(1, None), _evidence(2)))
        b1 = _relationship(2, source_order=2, source="char_0003", target="char_0004", state="X")
        a2 = _relationship(3, source_order=3, state="B", evidence=(_evidence(2), _evidence(1, "文字")))
        b2 = _relationship(4, source_order=4, source="char_0003", target="char_0004", state="Y")
        planning = _planning(relationships=(b2, a2, b1, a1))
        identity = _identity(planning, (_component(1, _rel_ref(1), _rel_ref(3), first=1), _component(2, _rel_ref(2), _rel_ref(4), first=2)))
        output = build_canonical_relationship_set(planning, identity).relationships
        assert [state.narrative_order for state in output[0].state_history] == [1, 3]
        assert output[0].state_history[1].evidence_refs == (_evidence(2), _evidence(1, "文字"))


class TestRelationshipIdentityValidation:
    @pytest.mark.parametrize("ids", [("rel_000002",), ("rel_000002", "rel_000001"), ("rel_000001", "rel_000003")])
    def test_canonical_ids_follow_component_ordinals(self, ids):
        relationships = tuple(_relationship(n) for n in range(1, len(ids) + 1))
        planning = _planning(relationships=relationships)
        components = tuple(replace(_component(ordinal, _rel_ref(ordinal)), canonical_id=cid) for ordinal, cid in enumerate(ids, 1))
        with pytest.raises(ConsolidationFinalizationError, match="source-order ordinal"):
            build_canonical_relationship_set(planning, _identity(planning, components))

    def test_coverage_and_order_fail_closed(self):
        planning = _planning(relationships=(_relationship(1), _relationship(2)))
        missing = _identity(planning, (_component(1, _rel_ref(1)),))
        with pytest.raises(ConsolidationFinalizationError, match="coverage"):
            build_canonical_relationship_set(planning, missing)
        reversed_members = _identity(planning, (_component(1, _rel_ref(2), _rel_ref(1)),))
        with pytest.raises(ConsolidationFinalizationError, match="members"):
            build_canonical_relationship_set(planning, reversed_members)

    def test_malformed_membership_namespace_component_order_and_first_order_fail_closed(self):
        planning = _planning(relationships=(_relationship(1), _relationship(2)))
        duplicate = _identity(planning, (_component(1, _rel_ref(1)), _component(2, _rel_ref(1))))
        with pytest.raises(ConsolidationFinalizationError, match="duplicate component membership"):
            build_canonical_relationship_set(planning, duplicate)
        foreign = _identity(planning, (_component(1, _rel_ref(1), "CH001_C001:cand_rel_999"), _component(2, _rel_ref(2))))
        with pytest.raises(ConsolidationFinalizationError, match="not in the candidate index"):
            build_canonical_relationship_set(planning, foreign)
        wrong_namespace = _identity(planning, (_component(1, "CH001_C001:cand_fact_001"), _component(2, _rel_ref(2))))
        with pytest.raises(ConsolidationFinalizationError, match="namespace"):
            build_canonical_relationship_set(planning, wrong_namespace)
        bad_first = _identity(planning, (_component(1, _rel_ref(1), first=99), _component(2, _rel_ref(2))))
        with pytest.raises(ConsolidationFinalizationError, match="first_source_order"):
            build_canonical_relationship_set(planning, bad_first)
        swapped_components = _identity(planning, (_component(1, _rel_ref(2)), _component(2, _rel_ref(1))))
        with pytest.raises(ConsolidationFinalizationError, match="components"):
            build_canonical_relationship_set(planning, swapped_components)


@dataclass(frozen=True)
class _Pair:
    left_ref: str
    right_ref: str


def _fact(n: int) -> IndexedFactCandidate:
    ref = f"CH001_C001:cand_fact_{n:03d}"
    return IndexedFactCandidate(ref, "CH001_C001", f"cand_fact_{n:03d}", f"{n:09d}", "world_fact", f"事实{n}", ("char_0001",), (), "explicit", (_evidence(50 + n),), _artifact())


def _event(n: int) -> IndexedEventCandidate:
    ref = f"CH001_C001:cand_evt_{n:03d}"
    return IndexedEventCandidate(ref, "CH001_C001", f"cand_evt_{n:03d}", f"{n:09d}", f"事件{n}", ("char_0001",), (), "normal", "explicit", (_evidence(60 + n),), _artifact())


def _fact_decision(n: int, left: str, right: str, decision: str) -> FactSemanticDecision:
    left, right = sorted((left, right))
    return FactSemanticDecision(f"dec_{n}", left, right, decision, "manual", "测试", (_evidence(70 + n),), None, None, None)


class TestFinalComposition:
    def _resolutions(self, planning, fact_decisions=()):
        return (
            FactSemanticResolutionResult(planning, None, (), tuple(fact_decisions), ()),
            EventSemanticResolutionResult(planning, None, (), (), ()),
            RelationshipSemanticResolutionResult(planning, None, (), (), ()),
        )

    def test_empty_domains_and_mixed_result(self):
        empty = _planning()
        fr, er, rr = self._resolutions(empty)
        result = finalize_consolidation(empty, fr, er, rr)
        assert isinstance(result, A5FinalizationResult)
        assert result.to_dict()["canonical_relationship_set"]["relationships"] == []

        facts = (_fact(1), _fact(2), _fact(3))
        event = _event(1)
        relationship = _relationship(1, state="敌对")
        planning = _planning(facts=facts, events=(event,), relationships=(relationship,))
        f1, f2, f3 = (fact.global_candidate_ref for fact in facts)
        planning = replace(planning, fact_pair_plans=(_Pair(*sorted((f1, f2))), _Pair(*sorted((f1, f3)))))
        decisions = (_fact_decision(1, f1, f2, "state_change"), _fact_decision(2, f1, f3, "conflict"))
        fr, er, rr = self._resolutions(planning, decisions)
        result = finalize_consolidation(planning, fr, er, rr)
        assert result.planning_result is planning
        assert len(result.canonical_fact_set.state_transitions) == 1
        assert len(result.story_conflict_set.conflicts) == 1
        assert len(result.canonical_relationship_set.relationships[0].state_history) == 1
        assert all(t.transition_kind == "state_change" for t in result.canonical_fact_set.state_transitions)
        assert all(c.conflict_kind == "fact_conflict" and c.relationship_ids == () for c in result.story_conflict_set.conflicts)

    def test_deterministic_normalized_result_and_bad_nested_plan(self):
        relationship = _relationship(1, state="友好")
        planning = _planning(relationships=(relationship,))
        fr, er, rr = self._resolutions(planning)
        assert finalize_consolidation(planning, fr, er, rr).to_dict() == finalize_consolidation(planning, fr, er, rr).to_dict()
        bad = replace(fr, planning_result=object())
        with pytest.raises(ConsolidationFinalizationError, match="planning_result"):
            finalize_consolidation(planning, bad, er, rr)

    @pytest.mark.parametrize("domain", ["fact", "event", "relationship"])
    def test_plan_hash_mismatch_fails_closed_for_each_resolution(self, domain):
        planning = _planning()
        other = replace(planning, plan_hash="b" * 64)
        fr, er, rr = self._resolutions(planning)
        if domain == "fact":
            fr = replace(fr, planning_result=other)
        elif domain == "event":
            er = replace(er, planning_result=other)
        else:
            rr = replace(rr, planning_result=other)
        with pytest.raises(ConsolidationFinalizationError, match="planning identity"):
            finalize_consolidation(planning, fr, er, rr)

    def test_permuted_relationship_candidate_arrival_is_deterministic(self):
        relationships = (_relationship(1, state="A"), _relationship(2, state="B"))
        outputs = []
        for candidate_order in (relationships, tuple(reversed(relationships))):
            planning = _planning(relationships=candidate_order)
            fr, er, rr = self._resolutions(planning)
            outputs.append(finalize_consolidation(planning, fr, er, rr).to_dict())
        assert outputs[0] == outputs[1]
