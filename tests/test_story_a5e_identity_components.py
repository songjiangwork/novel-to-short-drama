"""v1.2 A5E1 -- deterministic identity components and canonical IDs.

Covers A5E1 implemented in ``short_drama.story.consolidation_finalization``:

  * deterministic same_* connected components over the FULL candidate index
    (zero-edge / singleton / unblocked candidates each form a component);
  * source-authoritative member order ``(source_order_key, global_candidate_ref)``
    and component order (earliest member under the same tuple);
  * stable 1-based ``fact_/evt_/rel_`` canonical ID allocation (>= 6 digits);
  * hard-negative contradiction validation (same-component hard-negatives FAIL
    CLOSED; ``uncertain`` is a non-edge and a non-veto);
  * relationship direction-aware endpoint signature invariant (directed ordered,
    symmetric min/max, unknown ordered-not-symmetric);
  * decision coverage defense (duplicate / unknown / missing pair, candidate
    ref not in index, wrong namespace);
  * production entry-point planning-identity binding (plan_hash match);
  * determinism: shuffled decision / candidate / edge order gives byte-identical
    ``to_dict()``.

A5E1 is deterministic Python-only: no provider call, no persistence. All
inputs are synthetic in-memory objects (lightweight stand-ins for the generic
domain builders; real ``Indexed*Candidate`` / resolution dataclasses for the
production entry point).
"""

from __future__ import annotations

from dataclasses import dataclass

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
    FactSemanticResolutionResult,
    IndexedFactCandidate,
    RelationshipSemanticResolutionResult,
    build_event_identity_components,
    build_fact_identity_components,
    build_relationship_identity_components,
    finalize_consolidation_identity,
)
from short_drama.story.extraction import EvidenceRef


# ---------------------------------------------------------------------------
# Lightweight duck-typed stand-ins for the generic domain builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cand:
    global_candidate_ref: str
    source_order_key: str
    # relationship-specific (unused for fact / event)
    source_entity_ref: str = ""
    target_entity_ref: str = ""
    direction: str = ""


@dataclass(frozen=True)
class _Dec:
    left_candidate_ref: str
    right_candidate_ref: str
    decision: str
    decision_id: str = "dec_x"


@dataclass(frozen=True)
class _Plan:
    left_ref: str
    right_ref: str


def _fact_ref(n: int) -> str:
    return f"CH001_C001:cand_fact_{n:03d}"


def _evt_ref(n: int) -> str:
    return f"CH001_C001:cand_evt_{n:03d}"


def _rel_ref(n: int) -> str:
    return f"CH001_C001:cand_rel_{n:03d}"


def _so(n: int) -> str:
    """A synthetic source_order_key whose lexicographic order matches ``n``."""
    return f"{n:09d}"


def _fact_cands(*ns: int) -> tuple[_Cand, ...]:
    return tuple(_Cand(_fact_ref(n), _so(n)) for n in ns)


def _evt_cands(*ns: int) -> tuple[_Cand, ...]:
    return tuple(_Cand(_evt_ref(n), _so(n)) for n in ns)


def _rel_cands(*specs: tuple) -> tuple[_Cand, ...]:
    return tuple(
        _Cand(
            _rel_ref(n),
            _so(n),
            source_entity_ref=source,
            target_entity_ref=target,
            direction=direction,
        )
        for (n, source, target, direction) in specs
    )


def _dec(ref_a: str, ref_b: str, decision: str) -> _Dec:
    left, right = (ref_a, ref_b) if ref_a < ref_b else (ref_b, ref_a)
    return _Dec(left, right, decision)


def _plans(*pairs: tuple[str, str]) -> tuple[_Plan, ...]:
    out = []
    for (a, b) in pairs:
        left, right = (a, b) if a < b else (b, a)
        out.append(_Plan(left, right))
    return tuple(out)


def _plan_from(fact_components, plan_hash: str = "hash_1") -> ConsolidationIdentityPlan:
    """Wrap fact components in a plan for deterministic ``to_dict()`` comparison."""
    return ConsolidationIdentityPlan(
        plan_hash=plan_hash,
        fact_components=tuple(fact_components),
        event_components=(),
        relationship_components=(),
    )


def _real_fact_cand(n: int) -> IndexedFactCandidate:
    ref = f"CH001_C001:cand_fact_{n:03d}"
    return IndexedFactCandidate(
        global_candidate_ref=ref,
        chunk_id="CH001_C001",
        local_candidate_id=f"cand_fact_{n:03d}",
        source_order_key=f"{n:09d}",
        fact_type="world_fact",
        statement_zh=f"fact {n}",
        subject_refs=("char_0001",),
        object_refs=(),
        evidence_strength="explicit",
        evidence_refs=(
            EvidenceRef(
                paragraph_id="CH001_P0001",
                role="primary",
                strength="explicit",
                excerpt="txt",
            ),
        ),
        candidate_extraction_ref=ArtifactRef(
            artifact_type="candidate_extraction",
            artifact_id="ce000001",
            revision=1,
            content_hash="0" * 64,
        ),
    )


def _zero_coverage(f: int, e: int, r: int) -> ConsolidationCoverageSummary:
    return ConsolidationCoverageSummary(
        fact_candidate_count=f,
        event_candidate_count=e,
        relationship_candidate_count=r,
        canonical_fact_count=0,
        canonical_event_count=0,
        canonical_relationship_count=0,
        uncertain_decision_count=0,
        story_conflict_count=0,
    )


# ---------------------------------------------------------------------------
# Generic graph
# ---------------------------------------------------------------------------


class TestGenericGraph:
    def test_empty_universe(self):
        assert build_fact_identity_components((), (), ()) == ()

    def test_single_singleton(self):
        comps = build_fact_identity_components(_fact_cands(1), (), ())
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (_fact_ref(1),)
        assert comps[0].canonical_id == "fact_000001"
        assert comps[0].first_source_order == _so(1)

    def test_multiple_singletons_source_ordered(self):
        # input order (3, 1, 2); components must come out in source order (1,2,3)
        comps = build_fact_identity_components(_fact_cands(3, 1, 2), (), ())
        assert [c.member_candidate_refs for c in comps] == [
            (_fact_ref(1),),
            (_fact_ref(2),),
            (_fact_ref(3),),
        ]
        assert [c.canonical_id for c in comps] == [
            "fact_000001",
            "fact_000002",
            "fact_000003",
        ]

    def test_simple_merge(self):
        a, b = _fact_ref(1), _fact_ref(2)
        decs = (_dec(a, b, "same_fact"),)
        comps = build_fact_identity_components(_fact_cands(1, 2), decs, _plans((a, b)))
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (a, b)

    def test_transitive_component(self):
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        decs = (_dec(a, b, "same_fact"), _dec(b, c, "same_fact"))
        comps = build_fact_identity_components(
            _fact_cands(1, 2, 3), decs, _plans((a, b), (b, c))
        )
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (a, b, c)

    def test_multiple_disconnected_components(self):
        a, b, c, d = (_fact_ref(i) for i in (1, 2, 3, 4))
        decs = (_dec(a, b, "same_fact"), _dec(c, d, "same_fact"))
        comps = build_fact_identity_components(
            _fact_cands(1, 2, 3, 4), decs, _plans((a, b), (c, d))
        )
        assert len(comps) == 2
        assert comps[0].member_candidate_refs == (a, b)
        assert comps[1].member_candidate_refs == (c, d)

    def test_members_source_ordered(self):
        # source keys: cand 3 -> so(1), cand 2 -> so(2), cand 1 -> so(5)
        cands = (
            _Cand(_fact_ref(1), _so(5)),
            _Cand(_fact_ref(2), _so(2)),
            _Cand(_fact_ref(3), _so(1)),
        )
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        decs = (_dec(a, b, "same_fact"), _dec(b, c, "same_fact"))
        comps = build_fact_identity_components(cands, decs, _plans((a, b), (b, c)))
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (c, b, a)
        assert comps[0].first_source_order == _so(1)

    def test_components_source_ordered(self):
        # component {a,b} earliest so=1 ; component {c,d} earliest so=3.
        # decisions are supplied with the {c,d} edge first -- order must not
        # change the component ordering.
        a, b, c, d = (_fact_ref(i) for i in (1, 2, 3, 4))
        decs = (_dec(c, d, "same_fact"), _dec(a, b, "same_fact"))
        comps = build_fact_identity_components(
            _fact_cands(1, 2, 3, 4), decs, _plans((c, d), (a, b))
        )
        assert [c.member_candidate_refs for c in comps] == [(a, b), (c, d)]
        assert [c.canonical_id for c in comps] == ["fact_000001", "fact_000002"]

    def test_stable_ids(self):
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        decs = (_dec(a, b, "same_fact"),)
        comps = build_fact_identity_components(
            _fact_cands(1, 2, 3), decs, _plans((a, b))
        )
        # {a,b} (earliest so=1) then singleton {c} (so=3)
        assert [c.canonical_id for c in comps] == ["fact_000001", "fact_000002"]

    def test_full_node_coverage(self):
        a, b, c, d, e = (_fact_ref(i) for i in range(1, 6))
        decs = (_dec(a, b, "same_fact"), _dec(d, e, "same_fact"))
        comps = build_fact_identity_components(
            _fact_cands(1, 2, 3, 4, 5), decs, _plans((a, b), (d, e))
        )
        members = [r for comp in comps for r in comp.member_candidate_refs]
        assert sorted(members) == sorted(_fact_ref(i) for i in range(1, 6))
        assert len(members) == len(set(members)) == 5


# ---------------------------------------------------------------------------
# Fact identity + hard-negatives
# ---------------------------------------------------------------------------


class TestFactIdentity:
    def test_same_fact_one_component(self):
        a, b = _fact_ref(1), _fact_ref(2)
        decs = (_dec(a, b, "same_fact"),)
        comps = build_fact_identity_components(_fact_cands(1, 2), decs, _plans((a, b)))
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (a, b)

    def test_same_fact_transitive(self):
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        decs = (_dec(a, b, "same_fact"), _dec(b, c, "same_fact"))
        comps = build_fact_identity_components(
            _fact_cands(1, 2, 3), decs, _plans((a, b), (b, c))
        )
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (a, b, c)

    @pytest.mark.parametrize(
        "decision",
        ["compatible_fact", "state_change", "conflict", "unrelated", "uncertain"],
    )
    def test_hard_negative_or_uncertain_alone_no_edge(self, decision):
        a, b = _fact_ref(1), _fact_ref(2)
        decs = (_dec(a, b, decision),)
        comps = build_fact_identity_components(_fact_cands(1, 2), decs, _plans((a, b)))
        assert len(comps) == 2  # two singletons -- no merge edge
        assert sorted(c.member_candidate_refs for c in comps) == [
            (a,),
            (b,),
        ]

    @pytest.mark.parametrize(
        "decision", ["compatible_fact", "state_change", "conflict", "unrelated"]
    )
    def test_same_path_hard_negative_fails(self, decision):
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        decs = (_dec(a, b, "same_fact"), _dec(b, c, "same_fact"), _dec(a, c, decision))
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_fact_identity_components(
                _fact_cands(1, 2, 3), decs, _plans((a, b), (b, c), (a, c))
            )
        assert "fact" in str(exc.value)
        assert decision in str(exc.value)

    def test_same_path_uncertain_legal(self):
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        decs = (_dec(a, b, "same_fact"), _dec(b, c, "same_fact"), _dec(a, c, "uncertain"))
        comps = build_fact_identity_components(
            _fact_cands(1, 2, 3), decs, _plans((a, b), (b, c), (a, c))
        )
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (a, b, c)

    def test_singleton_retained(self):
        comps = build_fact_identity_components(_fact_cands(1, 2), (), ())
        assert len(comps) == 2

    def test_stable_fact_ordering(self):
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        decs = (_dec(b, c, "same_fact"),)
        comps = build_fact_identity_components(
            _fact_cands(1, 2, 3), decs, _plans((b, c))
        )
        # {a} (so=1) then {b,c} (earliest so=2)
        assert [c.canonical_id for c in comps] == ["fact_000001", "fact_000002"]
        assert comps[0].member_candidate_refs == (a,)
        assert comps[1].member_candidate_refs == (b, c)


# ---------------------------------------------------------------------------
# Event identity + hard-negatives
# ---------------------------------------------------------------------------


class TestEventIdentity:
    def test_same_event_merge(self):
        a, b = _evt_ref(1), _evt_ref(2)
        decs = (_dec(a, b, "same_event"),)
        comps = build_event_identity_components(_evt_cands(1, 2), decs, _plans((a, b)))
        assert len(comps) == 1
        assert comps[0].canonical_id == "evt_000001"
        assert comps[0].member_candidate_refs == (a, b)

    def test_same_event_transitive(self):
        a, b, c = _evt_ref(1), _evt_ref(2), _evt_ref(3)
        decs = (_dec(a, b, "same_event"), _dec(b, c, "same_event"))
        comps = build_event_identity_components(
            _evt_cands(1, 2, 3), decs, _plans((a, b), (b, c))
        )
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (a, b, c)

    @pytest.mark.parametrize("decision", ["different_event", "uncertain"])
    def test_no_edge_decisions(self, decision):
        a, b = _evt_ref(1), _evt_ref(2)
        decs = (_dec(a, b, decision),)
        comps = build_event_identity_components(_evt_cands(1, 2), decs, _plans((a, b)))
        assert len(comps) == 2

    def test_same_path_different_event_fails(self):
        a, b, c = _evt_ref(1), _evt_ref(2), _evt_ref(3)
        decs = (_dec(a, b, "same_event"), _dec(b, c, "same_event"), _dec(a, c, "different_event"))
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_event_identity_components(
                _evt_cands(1, 2, 3), decs, _plans((a, b), (b, c), (a, c))
            )
        assert "event" in str(exc.value)
        assert "different_event" in str(exc.value)

    def test_same_path_uncertain_legal(self):
        a, b, c = _evt_ref(1), _evt_ref(2), _evt_ref(3)
        decs = (_dec(a, b, "same_event"), _dec(b, c, "same_event"), _dec(a, c, "uncertain"))
        comps = build_event_identity_components(
            _evt_cands(1, 2, 3), decs, _plans((a, b), (b, c), (a, c))
        )
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == (a, b, c)

    def test_singleton_retained(self):
        assert len(build_event_identity_components(_evt_cands(1, 2), (), ())) == 2

    def test_stable_evt_ordering(self):
        a, b, c = _evt_ref(1), _evt_ref(2), _evt_ref(3)
        decs = (_dec(a, b, "same_event"),)
        comps = build_event_identity_components(_evt_cands(1, 2, 3), decs, _plans((a, b)))
        assert [c.canonical_id for c in comps] == ["evt_000001", "evt_000002"]


# ---------------------------------------------------------------------------
# Relationship identity + signature invariant
# ---------------------------------------------------------------------------


class TestRelationshipIdentity:
    def test_same_relationship_merge(self):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        b = _rel_cands((2, "char_0001", "char_0002", "directed"))[0]
        decs = (_dec(a.global_candidate_ref, b.global_candidate_ref, "same_relationship"),)
        comps = build_relationship_identity_components(
            (a, b), decs, _plans((a.global_candidate_ref, b.global_candidate_ref))
        )
        assert len(comps) == 1
        assert comps[0].canonical_id == "rel_000001"

    def test_same_relationship_transitive(self):
        specs = (
            (1, "char_0001", "char_0002", "directed"),
            (2, "char_0001", "char_0002", "directed"),
            (3, "char_0001", "char_0002", "directed"),
        )
        cands = _rel_cands(*specs)
        refs = [c.global_candidate_ref for c in cands]
        decs = (
            _dec(refs[0], refs[1], "same_relationship"),
            _dec(refs[1], refs[2], "same_relationship"),
        )
        comps = build_relationship_identity_components(
            cands, decs, _plans((refs[0], refs[1]), (refs[1], refs[2]))
        )
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == tuple(refs)

    @pytest.mark.parametrize(
        "decision", ["different_relationship", "uncertain"]
    )
    def test_no_edge_decisions(self, decision):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        b = _rel_cands((2, "char_0001", "char_0002", "directed"))[0]
        decs = (_dec(a.global_candidate_ref, b.global_candidate_ref, decision),)
        comps = build_relationship_identity_components(
            (a, b), decs, _plans((a.global_candidate_ref, b.global_candidate_ref))
        )
        assert len(comps) == 2

    def test_same_path_different_relationship_fails(self):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        b = _rel_cands((2, "char_0001", "char_0002", "directed"))[0]
        c = _rel_cands((3, "char_0001", "char_0002", "directed"))[0]
        refs = [x.global_candidate_ref for x in (a, b, c)]
        decs = (
            _dec(refs[0], refs[1], "same_relationship"),
            _dec(refs[1], refs[2], "same_relationship"),
            _dec(refs[0], refs[2], "different_relationship"),
        )
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_relationship_identity_components(
                (a, b, c), decs, _plans((refs[0], refs[1]), (refs[1], refs[2]), (refs[0], refs[2]))
            )
        assert "relationship" in str(exc.value)
        assert "different_relationship" in str(exc.value)

    def test_same_path_uncertain_legal(self):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        b = _rel_cands((2, "char_0001", "char_0002", "directed"))[0]
        c = _rel_cands((3, "char_0001", "char_0002", "directed"))[0]
        refs = [x.global_candidate_ref for x in (a, b, c)]
        decs = (
            _dec(refs[0], refs[1], "same_relationship"),
            _dec(refs[1], refs[2], "same_relationship"),
            _dec(refs[0], refs[2], "uncertain"),
        )
        comps = build_relationship_identity_components(
            (a, b, c), decs, _plans((refs[0], refs[1]), (refs[1], refs[2]), (refs[0], refs[2]))
        )
        assert len(comps) == 1
        assert comps[0].member_candidate_refs == tuple(refs)

    def test_directed_same_direction_legal(self):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        b = _rel_cands((2, "char_0001", "char_0002", "directed"))[0]
        decs = (_dec(a.global_candidate_ref, b.global_candidate_ref, "same_relationship"),)
        comps = build_relationship_identity_components(
            (a, b), decs, _plans((a.global_candidate_ref, b.global_candidate_ref))
        )
        assert len(comps) == 1

    def test_directed_reversed_mismatch_fails(self):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        b = _rel_cands((2, "char_0002", "char_0001", "directed"))[0]
        decs = (_dec(a.global_candidate_ref, b.global_candidate_ref, "same_relationship"),)
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_relationship_identity_components(
                (a, b), decs, _plans((a.global_candidate_ref, b.global_candidate_ref))
            )
        assert "signature" in str(exc.value).lower()

    def test_symmetric_reversed_same_signature_legal(self):
        a = _rel_cands((1, "char_0001", "char_0002", "symmetric"))[0]
        b = _rel_cands((2, "char_0002", "char_0001", "symmetric"))[0]
        decs = (_dec(a.global_candidate_ref, b.global_candidate_ref, "same_relationship"),)
        comps = build_relationship_identity_components(
            (a, b), decs, _plans((a.global_candidate_ref, b.global_candidate_ref))
        )
        assert len(comps) == 1

    def test_unknown_reversed_distinct_fails(self):
        a = _rel_cands((1, "char_0001", "char_0002", "unknown"))[0]
        b = _rel_cands((2, "char_0002", "char_0001", "unknown"))[0]
        decs = (_dec(a.global_candidate_ref, b.global_candidate_ref, "same_relationship"),)
        with pytest.raises(ConsolidationFinalizationError):
            build_relationship_identity_components(
                (a, b), decs, _plans((a.global_candidate_ref, b.global_candidate_ref))
            )

    def test_same_endpoint_different_direction_enum_fails(self):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        b = _rel_cands((2, "char_0001", "char_0002", "symmetric"))[0]
        decs = (_dec(a.global_candidate_ref, b.global_candidate_ref, "same_relationship"),)
        with pytest.raises(ConsolidationFinalizationError):
            build_relationship_identity_components(
                (a, b), decs, _plans((a.global_candidate_ref, b.global_candidate_ref))
            )

    def test_singleton_retained(self):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        assert len(build_relationship_identity_components((a,), (), ())) == 1

    def test_stable_rel_ordering(self):
        a = _rel_cands((1, "char_0001", "char_0002", "directed"))[0]
        b = _rel_cands((2, "char_0001", "char_0002", "directed"))[0]
        decs = (_dec(a.global_candidate_ref, b.global_candidate_ref, "same_relationship"),)
        c = _rel_cands((3, "char_0003", "char_0004", "directed"))[0]
        comps = build_relationship_identity_components(
            (a, b, c), decs, _plans((a.global_candidate_ref, b.global_candidate_ref))
        )
        assert [x.canonical_id for x in comps] == ["rel_000001", "rel_000002"]


# ---------------------------------------------------------------------------
# Malformed inputs (fail closed, no repair)
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    def test_candidate_ref_not_in_index(self):
        a, b = _fact_ref(1), _fact_ref(2)
        missing = _fact_ref(3)  # not in the candidate index
        decs = (_dec(a, missing, "same_fact"),)
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_fact_identity_components(
                _fact_cands(1, 2), decs, _plans((a, missing))
            )
        assert "not in the candidate index" in str(exc.value)

    def test_wrong_candidate_namespace(self):
        a = _fact_ref(1)
        b = _evt_ref(1)  # event-namespace ref in a fact decision
        decs = (_dec(a, b, "same_fact"),)
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_fact_identity_components(_fact_cands(1), decs, _plans((a, b)))
        assert "namespace" in str(exc.value)

    def test_duplicate_decision_pair(self):
        a, b = _fact_ref(1), _fact_ref(2)
        decs = (_dec(a, b, "same_fact"), _dec(a, b, "compatible_fact"))
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_fact_identity_components(_fact_cands(1, 2), decs, _plans((a, b)))
        assert "duplicate decision" in str(exc.value)

    def test_missing_explicit_decision_coverage(self):
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        decs = (_dec(a, b, "same_fact"),)
        plans = _plans((a, b), (b, c))  # planned pair (b,c) has no decision
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_fact_identity_components(_fact_cands(1, 2, 3), decs, plans)
        assert "no decision found" in str(exc.value)

    def test_unknown_decision_pair(self):
        a, b, c, d = _fact_ref(1), _fact_ref(2), _fact_ref(3), _fact_ref(4)
        decs = (_dec(a, b, "same_fact"), _dec(c, d, "same_fact"))
        plans = _plans((a, b))  # decision (c,d) has no plan
        with pytest.raises(ConsolidationFinalizationError) as exc:
            build_fact_identity_components(_fact_cands(1, 2, 3, 4), decs, plans)
        assert "not a planned" in str(exc.value)

    def test_duplicate_component_membership_detected(self):
        with pytest.raises(ConsolidationFinalizationError):
            ConsolidationIdentityComponent(
                domain="fact",
                canonical_id="fact_000001",
                member_candidate_refs=(_fact_ref(1), _fact_ref(1)),
                first_source_order=_so(1),
            )


# ---------------------------------------------------------------------------
# Determinism (byte-identical for logically-identical inputs)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_decision_order_permutation_identical(self):
        a, b, c, d, e = (_fact_ref(i) for i in range(1, 6))
        cands = _fact_cands(1, 2, 3, 4, 5)
        decs = [
            _dec(a, b, "same_fact"),
            _dec(b, c, "same_fact"),
            _dec(d, e, "compatible_fact"),
            _dec(a, c, "uncertain"),
        ]
        plans = _plans((a, b), (b, c), (d, e), (a, c))
        results = []
        for order in (
            decs,
            list(reversed(decs)),
            [decs[2], decs[0], decs[3], decs[1]],
        ):
            comps = build_fact_identity_components(cands, tuple(order), plans)
            results.append(_plan_from(comps).to_dict())
        assert results[0] == results[1] == results[2]

    def test_candidate_order_permutation_identical(self):
        a, b, c = _fact_ref(1), _fact_ref(2), _fact_ref(3)
        cands = _fact_cands(1, 2, 3)
        decs = (_dec(a, b, "same_fact"), _dec(b, c, "same_fact"))
        plans = _plans((a, b), (b, c))
        base = _plan_from(build_fact_identity_components(cands, decs, plans)).to_dict()
        shuffled = _plan_from(
            build_fact_identity_components(tuple(reversed(cands)), decs, plans)
        ).to_dict()
        assert base == shuffled

    def test_different_edge_order_identical(self):
        # a 4-node chain; shuffling the same-edge (merge-edge) order is identical
        a, b, c, d = (_fact_ref(i) for i in range(1, 5))
        cands = _fact_cands(1, 2, 3, 4)
        edges = [
            _dec(a, b, "same_fact"),
            _dec(b, c, "same_fact"),
            _dec(c, d, "same_fact"),
        ]
        plans = _plans((a, b), (b, c), (c, d))
        results = []
        for order in (
            edges,
            list(reversed(edges)),
            [edges[2], edges[0], edges[1]],
        ):
            comps = build_fact_identity_components(cands, tuple(order), plans)
            results.append(_plan_from(comps).to_dict())
        assert results[0] == results[1] == results[2]
        assert results[0]["fact_components"][0]["member_candidate_refs"] == [a, b, c, d]


# ---------------------------------------------------------------------------
# Production entry point (planning-identity binding + whole-domain decisions)
# ---------------------------------------------------------------------------


class TestProductionEntryPoint:
    def _planning(self, fact_cands, fact_plans, plan_hash):
        index = ConsolidationCandidateIndex(
            schema_version=1,
            facts=tuple(fact_cands),
            events=(),
            relationships=(),
        )
        return ConsolidationPlanningResult(
            snapshot=object(),
            index=index,
            coverage=_zero_coverage(len(fact_cands), 0, 0),
            fact_pair_plans=fact_plans,
            event_pair_plans=(),
            relationship_pair_plans=(),
            deterministic_decision_set=None,
            blocking_policy_id="bp",
            text_normalization_policy_id="tn",
            exact_safe_policy_id="es",
            planning_policy_id="pp",
            plan_hash=plan_hash,
        )

    def _resolutions(self, planning, fact_decisions):
        return (
            FactSemanticResolutionResult(
                planning_result=planning,
                preparation=None,
                semantic_decisions=(),
                all_fact_decisions=tuple(fact_decisions),
                block_results=(),
            ),
            EventSemanticResolutionResult(
                planning_result=planning,
                preparation=None,
                semantic_decisions=(),
                all_event_decisions=(),
                block_results=(),
            ),
            RelationshipSemanticResolutionResult(
                planning_result=planning,
                preparation=None,
                semantic_decisions=(),
                all_relationship_decisions=(),
                block_results=(),
            ),
        )

    def test_plan_hash_match_ok(self):
        cands = [_real_fact_cand(1), _real_fact_cand(2)]
        a, b = cands[0].global_candidate_ref, cands[1].global_candidate_ref
        decs = (_dec(a, b, "same_fact"),)
        planning = self._planning(cands, _plans((a, b)), "hashX")
        fr, er, rr = self._resolutions(planning, decs)
        plan = finalize_consolidation_identity(planning, fr, er, rr)
        assert plan.plan_hash == "hashX"
        assert plan.fact_components[0].canonical_id == "fact_000001"
        assert plan.fact_components[0].member_candidate_refs == (a, b)
        assert plan.event_components == ()
        assert plan.relationship_components == ()

    def test_plan_hash_mismatch_fails(self):
        cands = [_real_fact_cand(1)]
        planning = self._planning(cands, (), "hashA")
        other = self._planning(cands, (), "hashB")
        fr = FactSemanticResolutionResult(
            planning_result=other,
            preparation=None,
            semantic_decisions=(),
            all_fact_decisions=(),
            block_results=(),
        )
        er = EventSemanticResolutionResult(
            planning_result=planning,
            preparation=None,
            semantic_decisions=(),
            all_event_decisions=(),
            block_results=(),
        )
        rr = RelationshipSemanticResolutionResult(
            planning_result=planning,
            preparation=None,
            semantic_decisions=(),
            all_relationship_decisions=(),
            block_results=(),
        )
        with pytest.raises(ConsolidationFinalizationError) as exc:
            finalize_consolidation_identity(planning, fr, er, rr)
        assert "planning identity" in str(exc.value)

    def test_production_determinism(self):
        cands = [_real_fact_cand(i) for i in range(1, 5)]
        refs = [c.global_candidate_ref for c in cands]
        a, b, c, d = refs
        dec_list = [
            _dec(a, b, "same_fact"),
            _dec(b, c, "same_fact"),
            _dec(c, d, "uncertain"),
        ]
        plans = _plans((a, b), (b, c), (c, d))
        results = []
        for order in (
            dec_list,
            list(reversed(dec_list)),
            [dec_list[2], dec_list[1], dec_list[0]],
        ):
            planning = self._planning(cands, plans, "hashZ")
            fr, er, rr = self._resolutions(planning, tuple(order))
            results.append(finalize_consolidation_identity(planning, fr, er, rr).to_dict())
        assert results[0] == results[1] == results[2]
        # a-b same, b-c same, c-d uncertain -> component {a,b,c} + singleton {d}
        assert results[0]["fact_components"][0]["member_candidate_refs"] == [a, b, c]
        assert results[0]["fact_components"][1]["member_candidate_refs"] == [d]
