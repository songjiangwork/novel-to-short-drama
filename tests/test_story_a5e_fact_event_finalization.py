"""Focused A5E2 finalization tests: facts/events and fact side objects only."""

from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.artifacts import ArtifactRef
from short_drama.story import (
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
    build_canonical_event_set,
    build_canonical_fact_set,
    build_fact_story_conflict_set,
)
from short_drama.story.extraction import EvidenceRef


def _ref(domain: str, n: int) -> str:
    local = "evt" if domain == "event" else "fact"
    return f"CH001_C001:cand_{local}_{n:03d}"


def _evidence(n: int, excerpt: str | None = "x") -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=f"CH001_P{n:04d}", role="primary", strength="explicit", excerpt=excerpt
    )


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_type="candidate_extraction", artifact_id="ce000001", revision=1,
        content_hash="0" * 64,
    )


def _fact(
    n: int, *, source: int | None = None, fact_type: str = "world_fact",
    subjects: tuple[str, ...] = ("char_0001",), objects: tuple[str, ...] = (),
    evidence: tuple[EvidenceRef, ...] | None = None,
) -> IndexedFactCandidate:
    return IndexedFactCandidate(
        global_candidate_ref=_ref("fact", n), chunk_id="CH001_C001",
        local_candidate_id=f"cand_fact_{n:03d}", source_order_key=f"{source or n:09d}",
        fact_type=fact_type, statement_zh=f"事实 {n}", subject_refs=subjects,
        object_refs=objects, evidence_strength="explicit", evidence_refs=evidence or (_evidence(n),),
        candidate_extraction_ref=_artifact(),
    )


def _event(
    n: int, *, source: int | None = None, temporal_mode: str = "normal",
    participants: tuple[str, ...] = ("char_0001",), locations: tuple[str, ...] = (),
    evidence: tuple[EvidenceRef, ...] | None = None,
) -> IndexedEventCandidate:
    return IndexedEventCandidate(
        global_candidate_ref=_ref("event", n), chunk_id="CH001_C001",
        local_candidate_id=f"cand_evt_{n:03d}", source_order_key=f"{source or n:09d}",
        summary_zh=f"事件 {n}", participants=participants, locations=locations,
        temporal_mode=temporal_mode, evidence_strength="explicit", evidence_refs=evidence or (_evidence(n),),
        candidate_extraction_ref=_artifact(),
    )


def _planning(facts=(), events=(), *, plan_hash="a" * 64):
    return ConsolidationPlanningResult(
        snapshot=object(),
        index=ConsolidationCandidateIndex(schema_version=1, facts=tuple(facts), events=tuple(events), relationships=()),
        coverage=ConsolidationCoverageSummary(
            fact_candidate_count=len(facts), event_candidate_count=len(events), relationship_candidate_count=0,
            canonical_fact_count=0, canonical_event_count=0, canonical_relationship_count=0,
            uncertain_decision_count=0, story_conflict_count=0,
        ),
        fact_pair_plans=(), event_pair_plans=(), relationship_pair_plans=(), deterministic_decision_set=None,
        blocking_policy_id="bp", text_normalization_policy_id="tn", exact_safe_policy_id="es",
        planning_policy_id="pp", plan_hash=plan_hash,
    )


def _plan(planning, fact_components=(), event_components=()):
    return ConsolidationIdentityPlan(
        plan_hash=planning.plan_hash, fact_components=tuple(fact_components),
        event_components=tuple(event_components), relationship_components=(),
    )


def _component(domain: str, ordinal: int, *refs: str, first: int | None = None):
    prefix = "evt" if domain == "event" else "fact"
    return ConsolidationIdentityComponent(
        domain=domain, canonical_id=f"{prefix}_{ordinal:06d}", member_candidate_refs=tuple(refs),
        first_source_order=f"{first if first is not None else int(refs[0][-3:]):09d}",
    )


def _decision(n: int, a: str, b: str, kind: str, evidence=None) -> FactSemanticDecision:
    left, right = sorted((a, b))
    return FactSemanticDecision(
        decision_id=f"dec_{n}", left_candidate_ref=left, right_candidate_ref=right,
        decision=kind, method="manual", reason_zh="测试", evidence_refs=tuple(evidence or (_evidence(90 + n),)),
        prompt_id=None, prompt_version=None, generation_provenance=None,
    )


def _resolution(planning, decisions=()):
    return FactSemanticResolutionResult(
        planning_result=planning, preparation=None, semantic_decisions=(),
        all_fact_decisions=tuple(decisions), block_results=(),
    )


class TestCanonicalFact:
    def test_representative_evidence_and_continuity_rules(self):
        ev_none, ev_text, ev_shared = _evidence(1, None), _evidence(1, "不同摘录"), _evidence(2)
        later = _fact(2, source=20, fact_type="continuity_relevant", subjects=("char_0002",),
                      objects=("loc_0001",), evidence=(ev_shared, ev_none))
        early = _fact(1, source=10, subjects=("char_0001",), objects=(), evidence=(ev_none, ev_text, ev_shared))
        planning = _planning((later, early))
        identity = _plan(planning, (_component("fact", 1, _ref("fact", 1), _ref("fact", 2), first=10),))
        result = build_canonical_fact_set(planning, identity, _resolution(planning))
        fact = result.facts[0]
        assert (fact.fact_type, fact.statement_zh, fact.subject_refs, fact.object_refs) == (
            "world_fact", "事实 1", ("char_0001",), ()
        )
        assert fact.candidate_fact_refs == (_ref("fact", 1), _ref("fact", 2))
        assert fact.evidence_refs == (ev_none, ev_text, ev_shared)
        assert fact.continuity_relevant is True

    def test_singleton_and_no_implicit_continuity(self):
        planning = _planning((_fact(1, fact_type="location_state"),))
        identity = _plan(planning, (_component("fact", 1, _ref("fact", 1)),))
        fact = build_canonical_fact_set(planning, identity, _resolution(planning)).facts[0]
        assert fact.continuity_relevant is False


class TestFactSideObjects:
    def test_state_change_uses_component_direction_and_subject_intersection(self):
        # Lexical endpoints are fact_001 then fact_002, but source order reverses them.
        first = _fact(2, source=10, subjects=("char_0002", "char_0001", "char_0002"))
        later = _fact(1, source=20, subjects=("char_0001", "char_0002"))
        planning = _planning((later, first))
        identity = _plan(planning, (
            _component("fact", 1, _ref("fact", 2), first=10),
            _component("fact", 2, _ref("fact", 1), first=20),
        ))
        decision = _decision(1, _ref("fact", 1), _ref("fact", 2), "state_change", (_evidence(7),))
        transition = build_canonical_fact_set(planning, identity, _resolution(planning, (decision,))).state_transitions[0]
        assert (transition.from_fact_id, transition.to_fact_id, transition.narrative_order) == (
            "fact_000001", "fact_000002", 2
        )
        assert transition.subject_refs == ("char_0002", "char_0001", "char_0002")
        assert transition.evidence_refs == decision.evidence_refs
        assert transition.source_decision_ref == "dec_1"

    def test_transitions_and_conflicts_are_sorted_and_not_aggregated(self):
        facts = (_fact(1), _fact(2), _fact(3))
        planning = _planning(facts)
        identity = _plan(planning, tuple(_component("fact", n, _ref("fact", n)) for n in (1, 2, 3)))
        decisions = (
            _decision(9, _ref("fact", 1), _ref("fact", 3), "state_change"),
            _decision(3, _ref("fact", 1), _ref("fact", 2), "state_change"),
            _decision(8, _ref("fact", 1), _ref("fact", 3), "conflict"),
            _decision(2, _ref("fact", 1), _ref("fact", 2), "conflict"),
        )
        resolution = _resolution(planning, tuple(reversed(decisions)))
        fact_set = build_canonical_fact_set(planning, identity, resolution)
        conflicts = build_fact_story_conflict_set(planning, identity, resolution).conflicts
        assert [t.transition_id for t in fact_set.state_transitions] == ["trans_000001", "trans_000002"]
        assert [t.source_decision_ref for t in fact_set.state_transitions] == ["dec_3", "dec_9"]
        assert [c.conflict_id for c in conflicts] == ["conf_000001", "conf_000002"]
        assert [c.decision_refs for c in conflicts] == [("dec_2",), ("dec_8",)]
        assert all(c.conflict_kind == "fact_conflict" and c.relationship_ids == () and c.status == "unresolved" for c in conflicts)
        assert conflicts[0].candidate_refs == (_ref("fact", 1), _ref("fact", 2))

    @pytest.mark.parametrize("kind", ["compatible_fact", "unrelated", "uncertain", "state_change"])
    def test_non_conflict_decisions_emit_no_conflict(self, kind):
        planning = _planning((_fact(1), _fact(2)))
        identity = _plan(planning, (_component("fact", 1, _ref("fact", 1)), _component("fact", 2, _ref("fact", 2))))
        conflicts = build_fact_story_conflict_set(planning, identity, _resolution(planning, (_decision(1, _ref("fact", 1), _ref("fact", 2), kind),)))
        assert conflicts.conflicts == ()

    @pytest.mark.parametrize("kind, builder", [("state_change", build_canonical_fact_set), ("conflict", build_fact_story_conflict_set)])
    def test_same_canonical_endpoint_fails_closed(self, kind, builder):
        planning = _planning((_fact(1), _fact(2)))
        identity = _plan(planning, (_component("fact", 1, _ref("fact", 1), _ref("fact", 2)),))
        with pytest.raises(ConsolidationFinalizationError, match="same canonical fact"):
            builder(planning, identity, _resolution(planning, (_decision(1, _ref("fact", 1), _ref("fact", 2), kind),)))

    def test_ambiguous_decision_id_and_duplicate_side_evidence_fail_closed(self):
        planning = _planning((_fact(1), _fact(2)))
        identity = _plan(planning, (_component("fact", 1, _ref("fact", 1)), _component("fact", 2, _ref("fact", 2))))
        one = _decision(1, _ref("fact", 1), _ref("fact", 2), "state_change")
        duplicate_id = replace(one, decision="conflict")
        with pytest.raises(ConsolidationFinalizationError, match="duplicate decision_id"):
            build_canonical_fact_set(planning, identity, _resolution(planning, (one, duplicate_id)))
        duplicate_evidence = _decision(2, _ref("fact", 1), _ref("fact", 2), "state_change", (_evidence(5), _evidence(5)))
        with pytest.raises(ConsolidationFinalizationError, match="duplicate exact"):
            build_canonical_fact_set(planning, identity, _resolution(planning, (duplicate_evidence,)))


class TestCanonicalEvent:
    @pytest.mark.parametrize(
        "modes, expected",
        [(("normal", "normal"), "normal"), (("unknown", "unknown"), "unknown"),
         (("unknown", "memory", "unknown"), "memory"), (("normal", "unknown"), "normal"),
         (("normal", "memory"), "unknown"), (("flashback", "dream"), "unknown")],
    )
    def test_temporal_modes(self, modes, expected):
        events = tuple(_event(n, temporal_mode=mode) for n, mode in enumerate(modes, 1))
        planning = _planning(events=events)
        identity = _plan(planning, event_components=(_component("event", 1, *[_ref("event", n) for n in range(1, len(events) + 1)]),))
        assert build_canonical_event_set(planning, identity).events[0].temporal_mode == expected

    def test_representative_evidence_singleton_and_contiguous_orders(self):
        shared = _evidence(3)
        early = _event(1, participants=("char_0001",), locations=("loc_0001",), evidence=(shared,))
        later = _event(2, participants=("char_0002",), locations=(), evidence=(shared, _evidence(4)))
        singleton = _event(3)
        planning = _planning(events=(singleton, later, early))
        identity = _plan(planning, event_components=(
            _component("event", 1, _ref("event", 1), _ref("event", 2)),
            _component("event", 2, _ref("event", 3)),
        ))
        result = build_canonical_event_set(planning, identity)
        assert result.events[0].summary_zh == "事件 1"
        assert result.events[0].participants == ("char_0001",)
        assert result.events[0].locations == ("loc_0001",)
        assert result.events[0].evidence_refs == (shared, _evidence(4))
        assert [event.narrative_order for event in result.events] == [1, 2]


class TestMalformedAndDeterminism:
    def test_plan_hash_and_identity_plan_integrity_fail_closed(self):
        planning = _planning((_fact(1), _fact(2)))
        base = _plan(planning, (_component("fact", 1, _ref("fact", 1)), _component("fact", 2, _ref("fact", 2))))
        bad_hash = replace(base, plan_hash="b" * 64)
        with pytest.raises(ConsolidationFinalizationError, match="planning identity"):
            build_canonical_fact_set(planning, bad_hash, _resolution(planning))
        duplicate_id = replace(base, fact_components=(base.fact_components[0], replace(base.fact_components[1], canonical_id="fact_000001")))
        with pytest.raises(ConsolidationFinalizationError, match="duplicate canonical id"):
            build_canonical_fact_set(planning, duplicate_id, _resolution(planning))
        missing = replace(base, fact_components=(base.fact_components[0],))
        with pytest.raises(ConsolidationFinalizationError, match="coverage"):
            build_canonical_fact_set(planning, missing, _resolution(planning))

    def test_bad_member_order_first_source_and_foreign_ref_fail_closed(self):
        planning = _planning((_fact(1), _fact(2)))
        wrong_order = _plan(planning, (_component("fact", 1, _ref("fact", 2), _ref("fact", 1)),))
        with pytest.raises(ConsolidationFinalizationError, match="members"):
            build_canonical_fact_set(planning, wrong_order, _resolution(planning))
        bad_first = _plan(planning, (_component("fact", 1, _ref("fact", 1), _ref("fact", 2), first=99),))
        with pytest.raises(ConsolidationFinalizationError, match="first_source_order"):
            build_canonical_fact_set(planning, bad_first, _resolution(planning))
        foreign = _plan(planning, (_component("fact", 1, _ref("fact", 1), "CH001_C001:cand_fact_999"), _component("fact", 2, _ref("fact", 2))))
        with pytest.raises(ConsolidationFinalizationError, match="not in the candidate index"):
            build_canonical_fact_set(planning, foreign, _resolution(planning))

    def test_permuted_candidate_and_decision_arrival_is_identical(self):
        facts = (_fact(1), _fact(2), _fact(3))
        components = tuple(_component("fact", n, _ref("fact", n)) for n in (1, 2, 3))
        decisions = (_decision(2, _ref("fact", 1), _ref("fact", 3), "conflict"), _decision(1, _ref("fact", 1), _ref("fact", 2), "state_change"))
        normalized = []
        for candidate_order, decision_order in ((facts, decisions), (tuple(reversed(facts)), tuple(reversed(decisions)))):
            planning = _planning(candidate_order)
            identity = _plan(planning, components)
            normalized.append((
                build_canonical_fact_set(planning, identity, _resolution(planning, decision_order)).to_dict(),
                build_fact_story_conflict_set(planning, identity, _resolution(planning, decision_order)).to_dict(),
            ))
        assert normalized[0] == normalized[1]
