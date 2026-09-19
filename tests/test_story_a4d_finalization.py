"""A4D reconciliation finalization + validation tests.

Covers the frozen A4D contract:

  * the deterministic identity graph (same-entity union-find components,
    different-entity conflicts, uncertainty groups, self-ambiguity, singletons,
    cross-type decisions never structurally merge);
  * deterministic canonical / unresolved id allocation (char_NNNN / loc_NNNN
    gap-free in source order; unres_NNNN for A3 passthrough + uncertain groups);
  * EntityMap coverage (every index candidate exactly once; resolved ->
    canonical id, unresolved -> unresolved id);
  * the 13 deterministic BLOCKING findings (owner A4 / repair rerun_a4), each
    with a specific corruption trigger, deterministic finding_id;
  * the backend-neutral A4 semantic identity builder and its parity with the A4C
    zero-provider preparation.

Deliberately does NOT require: a running LLM, a stage CLI, or a real-novel
fixture. It is a pure in-memory graph/finalization/validation test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from short_drama.artifacts import ArtifactRef, content_hash
from short_drama.llm.config import load_semantic_profile
from short_drama.story import (
    A4SemanticIdentity,
    CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
    CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION,
    RECONCILIATION_DECISION_SET_SCHEMA_VERSION,
    UNRESOLVED_ENTITY_SET_SCHEMA_VERSION,
    CandidateEntityIndex,
    CandidateEntityIndexEntry,
    CanonicalCharacterRegistry,
    CanonicalEntity,
    CanonicalLocationRegistry,
    EntityMapEntry,
    ReconciliationDecision,
    ReconciliationDecisionSet,
    ReconciliationFinalizationResult,
    ReconciliationPairPlan,
    ReconciliationPlanningResult,
    ReconciliationSemanticResult,
    UnresolvedEntity,
    UnresolvedEntitySet,
    build_a4_semantic_identity,
    build_identity_graph,
    finalize_reconciliation,
    load_entity_reconciliation_profile,
    prepare_semantic_resolution,
    validate_finalization,
)
from short_drama.story.reconciliation_validation import (
    A4_CANDIDATE_REF_DUPLICATE,
    A4_CANDIDATE_REF_NOT_FOUND,
    A4_CANONICAL_ID_GAP,
    A4_DECISION_PAIR_DUPLICATE,
    A4_DECISION_PAIR_MISSING,
    A4_DECISION_PAIR_UNEXPECTED,
    A4_DECISION_REF_NOT_FOUND,
    A4_DECISION_TYPE_MISMATCH,
    A4_DETERMINISTIC_CONSTRAINT_OVERRIDE,
    A4_ENTITY_MAP_DUPLICATE,
    A4_ENTITY_MAP_TARGET_NOT_FOUND,
    A4_ENTITY_MAP_UNACCOUNTED,
    A4_RECONCILIATION_CONFLICT,
)
from short_drama.foundation.validation import ValidationSeverity

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
RECON_PROFILE_PATH = PROFILES / "entity_reconciliation_v1.yaml"
A4_LLM_PROFILE_PATH = PROFILES / "entity_reconciliation_llm_v1.yaml"

H = "a" * 64
H2 = "b" * 64
A4_PROMPT_ID = "a4.entity-reconciliation"
A4_PROMPT_VERSION = 1


def make_ref(artifact_type: str, artifact_id: str, revision: int = 1) -> ArtifactRef:
    return ArtifactRef(
        artifact_type=artifact_type, artifact_id=artifact_id, revision=revision, content_hash=H
    )


def did(left: str, right: str, decision: str) -> str:
    return f"dec_{content_hash({'l': left, 'r': right, 'd': decision})[:20]}"


def entry(
    candidate_ref: str,
    kind: str = "character",
    source_key: str = "CH001_C001:P0001",
    name: str = "林晚",
    **overrides,
) -> CandidateEntityIndexEntry:
    values = dict(
        candidate_ref=candidate_ref,
        candidate_kind=kind,
        candidate_extraction_ref=make_ref("candidate_extraction", "ce-1"),
        source_order_key=source_key,
        display_name_original=name,
        aliases_original=(),
        descriptors_zh=(),
        evidence_refs=(),
        possible_candidate_refs=(),
    )
    values.update(overrides)
    return CandidateEntityIndexEntry(**values)


def decision(
    left: str,
    right: str,
    decision: str = "same_entity",
    method: str = "deterministic",
    decision_id: str | None = None,
    **overrides,
) -> ReconciliationDecision:
    values = dict(
        decision_id=decision_id or did(left, right, decision),
        left_candidate_ref=left,
        right_candidate_ref=right,
        decision=decision,
        method=method,
        reason_code="test",
        reason_zh="test",
        evidence_refs=(),
        prompt_id=None if method == "deterministic" else A4_PROMPT_ID,
        prompt_version=None if method == "deterministic" else A4_PROMPT_VERSION,
        generation_provenance=None,
    )
    values.update(overrides)
    return ReconciliationDecision(**values)


def plan(left: str, right: str, state: str = "auto_same") -> ReconciliationPairPlan:
    return ReconciliationPairPlan(
        left_candidate_ref=left,
        right_candidate_ref=right,
        state=state,
        signals=(),
        shared_identity_keys=(),
        shared_tokens=(),
    )


def char_entity(canonical_id: str, candidate_refs, name: str = "林晚") -> CanonicalEntity:
    return CanonicalEntity(
        canonical_id=canonical_id,
        entity_type="character",
        display_name_original=name,
        aliases_original=(),
        candidate_refs=tuple(candidate_refs),
        first_appearance_candidate_ref=candidate_refs[0],
    )


def loc_entity(canonical_id: str, candidate_refs, name: str = "教室") -> CanonicalEntity:
    return CanonicalEntity(
        canonical_id=canonical_id,
        entity_type="location",
        display_name_original=name,
        aliases_original=(),
        candidate_refs=tuple(candidate_refs),
        first_appearance_candidate_ref=candidate_refs[0],
    )


def unresolved_entity(unresolved_id: str, kind: str, candidate_refs, decision_refs=()) -> UnresolvedEntity:
    return UnresolvedEntity(
        unresolved_id=unresolved_id,
        entity_kind=kind,
        candidate_refs=tuple(candidate_refs),
        decision_refs=tuple(decision_refs),
        possible_candidate_refs=(),
        first_appearance_candidate_ref=candidate_refs[0],
    )


def char_registry(entities) -> CanonicalCharacterRegistry:
    return CanonicalCharacterRegistry(
        schema_version=CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION, entities=tuple(entities)
    )


def loc_registry(entities) -> CanonicalLocationRegistry:
    return CanonicalLocationRegistry(
        schema_version=CANONICAL_ENTITY_REGISTRY_SCHEMA_VERSION, entities=tuple(entities)
    )


def unresolved_set(entities) -> UnresolvedEntitySet:
    return UnresolvedEntitySet(
        schema_version=UNRESOLVED_ENTITY_SET_SCHEMA_VERSION, entities=tuple(entities)
    )


# Canonical pair: left (CH001_C001) < right (CH001_C002).
L = "CH001_C001:cand_char_001"
R = "CH001_C002:cand_char_002"
L3 = "CH001_C003:cand_char_003"


def make_planning(index, pair_plans, decisions, plan_hash: str = H2) -> ReconciliationPlanningResult:
    return ReconciliationPlanningResult(
        candidate_index=index,
        pair_plans=tuple(pair_plans),
        decisions=tuple(decisions),
        normalization_policy_id="norm",
        blocking_policy_id="block",
        canonicalization_policy_id="canon",
        plan_hash=plan_hash,
    )


def make_semantic(planning, decisions) -> ReconciliationSemanticResult:
    return ReconciliationSemanticResult(
        planning_result=planning,
        blocks=(),
        semantic_decisions=(),
        all_decisions=tuple(decisions),
        semantic_request_hashes=(),
        block_results=(),
    )


def finalize(index, pair_plans, decisions) -> ReconciliationFinalizationResult:
    planning = make_planning(index, pair_plans, decisions)
    return finalize_reconciliation(make_semantic(planning, decisions))


def make_bundle(
    index_entries=None,
    decisions=None,
    pair_plans=None,
    char_entities=None,
    loc_entities=None,
    unresolved_entities=None,
    map_entries=None,
) -> dict:
    """A fully consistent clean bundle; overrides build the graph from the
    (possibly overridden) index entries + decisions."""
    ie = index_entries if index_entries is not None else [entry(L, source_key="CH001_C001:P0001"), entry(R, source_key="CH001_C002:P0002")]
    dcs = decisions if decisions is not None else [decision(L, R, "same_entity")]
    index = CandidateEntityIndex(
        schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION, entries=tuple(ie)
    )
    decision_set = ReconciliationDecisionSet(
        schema_version=RECONCILIATION_DECISION_SET_SCHEMA_VERSION, decisions=tuple(dcs)
    )
    graph = build_identity_graph(tuple(ie), tuple(dcs))
    if char_entities is None:
        char_entities = [char_entity("char_0001", (L, R))]
    if loc_entities is None:
        loc_entities = []
    if unresolved_entities is None:
        unresolved_entities = []
    if map_entries is None:
        map_entries = [
            EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
            EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0001", unresolved_id=None),
        ]
    if pair_plans is None:
        pair_plans = [plan(L, R, "auto_same")]
    return {
        "index_entries": tuple(ie),
        "decisions": tuple(dcs),
        "candidate_index": index,
        "decision_set": decision_set,
        "graph": graph,
        "canonical_character_registry": char_registry(char_entities),
        "canonical_location_registry": loc_registry(loc_entities),
        "unresolved_entity_set": unresolved_set(unresolved_entities),
        "entity_map_entries": tuple(map_entries),
        "pair_plans": tuple(pair_plans),
    }


def check(bundle: dict, pair_plans: tuple | None = ...) -> tuple:
    pp = bundle["pair_plans"] if pair_plans is ... else pair_plans
    return validate_finalization(
        candidate_index=bundle["candidate_index"],
        decision_set=bundle["decision_set"],
        graph=bundle["graph"],
        canonical_character_registry=bundle["canonical_character_registry"],
        canonical_location_registry=bundle["canonical_location_registry"],
        unresolved_entity_set=bundle["unresolved_entity_set"],
        entity_map_entries=bundle["entity_map_entries"],
        pair_plans=pp,
    )


def codes(findings) -> set:
    return {f.code for f in findings}


def assert_blocking(findings, owner="A4", repair="rerun_a4"):
    assert findings
    for f in findings:
        assert f.severity is ValidationSeverity.BLOCKING
        assert f.owner_stage == owner
        assert f.repair_route == repair


# ---------------------------------------------------------------------------
# Identity graph
# ---------------------------------------------------------------------------


class TestIdentityGraph:
    def test_same_entity_unions_into_one_component(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R), entry(L3, source_key="CH001_C003:P0003")],
            decisions=[decision(L, R, "same_entity"), decision(R, L3, "same_entity")],
            pair_plans=[],
        )
        assert b["graph"].same_components == (frozenset({L, R, L3}),)
        assert b["graph"].resolved_components == (frozenset({L, R, L3}),)

    def test_different_entity_does_not_union(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "different_entity")],
            pair_plans=[],
        )
        assert b["graph"].same_components == (frozenset({L}), frozenset({R}))

    def test_conflict_different_entity_within_same_component(self):
        # L-R same, R-M same (transitive), but L-M different -> conflict.
        M = "CH001_C004:cand_char_004"
        d_conflict = decision(L, M, "different_entity")
        b = make_bundle(
            index_entries=[entry(L), entry(R), entry(M, source_key="CH001_C004:P0004")],
            decisions=[
                decision(L, R, "same_entity"),
                decision(R, M, "same_entity"),
                d_conflict,
            ],
            pair_plans=[],
        )
        assert frozenset({L, R, M}) in b["graph"].same_components
        assert b["graph"].conflict_decisions == (d_conflict,)

    def test_self_ambiguity_single_component_ignores_intra_component_edge(self):
        # L-R same (one component); uncertain L-R is self-ambiguity -> one group.
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "same_entity"), decision(L, R, "uncertain")],
            pair_plans=[],
        )
        assert b["graph"].uncertainty_groups
        members, decision_ids = b["graph"].uncertainty_groups[0]
        assert members == frozenset({L, R})
        assert decision_ids == frozenset({did(L, R, "uncertain")})
        # A self-ambiguity component is NOT resolved (it is uncertain).
        assert b["graph"].resolved_components == ()

    def test_cross_component_uncertainty_forms_group(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "uncertain")],
            pair_plans=[],
        )
        assert len(b["graph"].uncertainty_groups) == 1
        members, decision_ids = b["graph"].uncertainty_groups[0]
        assert members == frozenset({L, R})
        assert b["graph"].resolved_components == ()

    def test_transitive_uncertainty_connects_group(self):
        # L-R uncertain, R-M uncertain (no same edges) -> one group of {L,R,M}.
        M = "CH001_C004:cand_char_004"
        b = make_bundle(
            index_entries=[entry(L), entry(R), entry(M, source_key="CH001_C004:P0004")],
            decisions=[decision(L, R, "uncertain"), decision(R, M, "uncertain")],
            pair_plans=[],
        )
        members, _ = b["graph"].uncertainty_groups[0]
        assert members == frozenset({L, R, M})
        assert len(b["graph"].uncertainty_groups) == 1

    def test_singleton_component_has_no_group_and_is_resolved(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "different_entity")],
            pair_plans=[],
        )
        assert b["graph"].uncertainty_groups == ()
        assert b["graph"].resolved_components == (frozenset({L}), frozenset({R}))

    def test_cross_type_decision_never_merges(self):
        loc = "CH001_C001:cand_loc_001"
        b = make_bundle(
            index_entries=[entry(L, kind="character"), entry(loc, kind="location")],
            decisions=[decision(L, loc, "same_entity")],
            pair_plans=[],
        )
        # Character and location never union; both stay singletons.
        assert b["graph"].same_components == (frozenset({L}), frozenset({loc}))


# ---------------------------------------------------------------------------
# Deterministic id allocation + EntityMap coverage
# ---------------------------------------------------------------------------


class TestDeterministicAllocation:
    def test_character_ids_gap_free_in_source_order(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
            entries=(
                entry(L, source_key="CH001_C001:P0001"),
                entry(R, source_key="CH001_C002:P0002"),
            ),
        )
        result = finalize(index, [], [])
        ids = [e.canonical_id for e in result.canonical_character_registry.entities]
        assert ids == ["char_0001", "char_0002"]
        # char_0001 is the earlier source-order member (L in CH001_C001).
        assert result.canonical_character_registry.entities[0].candidate_refs == (L,)
        assert result.canonical_character_registry.entities[1].candidate_refs == (R,)

    def test_location_ids_gap_free(self):
        loc1 = "CH001_C001:cand_loc_001"
        loc2 = "CH001_C002:cand_loc_002"
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
            entries=(entry(loc1, kind="location"), entry(loc2, kind="location")),
        )
        result = finalize(index, [], [])
        ids = [e.canonical_id for e in result.canonical_location_registry.entities]
        assert ids == ["loc_0001", "loc_0002"]

    def test_a3_unresolved_passthrough_kind_mapping(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
            entries=(
                entry("CH001_C001:cand_unres_001", kind="unresolved_person", name="某人"),
                entry("CH001_C002:cand_unres_002", kind="unresolved_location", name="某地"),
            ),
        )
        result = finalize(index, [], [])
        kinds = [e.entity_kind for e in result.unresolved_entity_set.entities]
        assert kinds == ["person", "location"]
        ids = [e.unresolved_id for e in result.unresolved_entity_set.entities]
        assert ids == ["unres_0001", "unres_0002"]

    def test_uncertain_group_becomes_unresolved(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
            entries=(entry(L), entry(R)),
        )
        result = finalize(index, [], [decision(L, R, "uncertain")])
        assert len(result.unresolved_entity_set.entities) == 1
        u = result.unresolved_entity_set.entities[0]
        assert u.entity_kind == "character"
        assert u.unresolved_id == "unres_0001"
        assert set(u.candidate_refs) == {L, R}
        # The uncertain decision is recorded.
        assert tuple(u.decision_refs) == (did(L, R, "uncertain"),)

    def test_every_index_candidate_has_exactly_one_map_entry(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
            entries=(entry(L), entry(R), entry("CH001_C001:cand_unres_001", kind="unresolved_person")),
        )
        result = finalize(index, [], [decision(L, R, "same_entity")])
        refs = [e.candidate_ref for e in result.entity_map_entries]
        assert sorted(refs) == sorted(
            [L, R, "CH001_C001:cand_unres_001"]
        )
        # Resolved -> canonical_id; unresolved -> unresolved_id.
        by_ref = {e.candidate_ref: e for e in result.entity_map_entries}
        assert by_ref[L].status == "resolved" and by_ref[L].canonical_id == "char_0001"
        assert by_ref["CH001_C001:cand_unres_001"].status == "unresolved"
        assert by_ref["CH001_C001:cand_unres_001"].unresolved_id == "unres_0001"

    def test_clean_finalization_has_no_blocking_findings(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION,
            entries=(entry(L), entry(R)),
        )
        result = finalize(index, [plan(L, R, "auto_same")], [decision(L, R, "same_entity")])
        assert result.findings == ()
        assert result.has_blocking_findings is False


# ---------------------------------------------------------------------------
# The 13 deterministic findings
# ---------------------------------------------------------------------------


class TestFindings:
    def test_candidate_ref_duplicate(self):
        b = make_bundle(
            index_entries=[entry(L), entry(L)],
            decisions=[],
            pair_plans=[],
            char_entities=[char_entity("char_0001", (L,))],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None)
            ],
        )
        findings = check(b)
        assert A4_CANDIDATE_REF_DUPLICATE in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_CANDIDATE_REF_DUPLICATE])

    def test_candidate_ref_not_found(self):
        b = make_bundle(
            index_entries=[entry(L)],
            decisions=[decision(L, R, "same_entity")],
            pair_plans=[],
            char_entities=[char_entity("char_0001", (L,))],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None)
            ],
        )
        findings = check(b)
        assert A4_CANDIDATE_REF_NOT_FOUND in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_CANDIDATE_REF_NOT_FOUND])

    def test_decision_pair_duplicate(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "same_entity"), decision(L, R, "same_entity")],
            pair_plans=[plan(L, R, "auto_same")],
        )
        findings = check(b)
        assert A4_DECISION_PAIR_DUPLICATE in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_DECISION_PAIR_DUPLICATE])

    def test_decision_pair_missing(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[],
            pair_plans=[plan(L, R, "auto_same")],
            char_entities=[
                char_entity("char_0001", (L,)),
                char_entity("char_0002", (R,)),
            ],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0002", unresolved_id=None),
            ],
        )
        findings = check(b)
        assert A4_DECISION_PAIR_MISSING in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_DECISION_PAIR_MISSING])

    def test_decision_pair_unexpected(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "same_entity")],
            pair_plans=[],  # a decision for a pair not in the plan
        )
        findings = check(b)
        assert A4_DECISION_PAIR_UNEXPECTED in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_DECISION_PAIR_UNEXPECTED])

    def test_decision_ref_not_found(self):
        b = make_bundle(
            index_entries=[entry("CH001_C001:cand_unres_001", kind="unresolved_person")],
            decisions=[],
            pair_plans=[],
            char_entities=[],
            unresolved_entities=[
                unresolved_entity("unres_0001", "person", ("CH001_C001:cand_unres_001",), ("dec_missing",))
            ],
            map_entries=[
                EntityMapEntry(
                    candidate_ref="CH001_C001:cand_unres_001",
                    status="unresolved",
                    canonical_id=None,
                    unresolved_id="unres_0001",
                )
            ],
        )
        findings = check(b)
        assert A4_DECISION_REF_NOT_FOUND in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_DECISION_REF_NOT_FOUND])

    def test_decision_type_mismatch(self):
        loc = "CH001_C001:cand_loc_001"
        b = make_bundle(
            index_entries=[entry(L, kind="character"), entry(loc, kind="location")],
            decisions=[decision(L, loc, "same_entity")],
            pair_plans=[],
            char_entities=[char_entity("char_0001", (L,))],
            loc_entities=[loc_entity("loc_0001", (loc,))],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=loc, status="resolved", canonical_id="loc_0001", unresolved_id=None),
            ],
        )
        findings = check(b)
        assert A4_DECISION_TYPE_MISMATCH in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_DECISION_TYPE_MISMATCH])

    def test_deterministic_constraint_override_auto_same(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "different_entity")],
            pair_plans=[plan(L, R, "auto_same")],
            char_entities=[
                char_entity("char_0001", (L,)),
                char_entity("char_0002", (R,)),
            ],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0002", unresolved_id=None),
            ],
        )
        findings = check(b)
        assert A4_DETERMINISTIC_CONSTRAINT_OVERRIDE in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_DETERMINISTIC_CONSTRAINT_OVERRIDE])

    def test_deterministic_constraint_override_must_not_merge(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "same_entity")],
            pair_plans=[plan(L, R, "must_not_merge")],
        )
        findings = check(b)
        assert A4_DETERMINISTIC_CONSTRAINT_OVERRIDE in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_DETERMINISTIC_CONSTRAINT_OVERRIDE])

    def test_reconciliation_conflict(self):
        M = "CH001_C004:cand_char_004"
        b = make_bundle(
            index_entries=[entry(L), entry(R), entry(M, source_key="CH001_C004:P0004")],
            decisions=[
                decision(L, R, "same_entity"),
                decision(R, M, "same_entity"),
                decision(L, M, "different_entity"),
            ],
            pair_plans=[],
            char_entities=[char_entity("char_0001", (L, R, M))],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=M, status="resolved", canonical_id="char_0001", unresolved_id=None),
            ],
        )
        findings = check(b)
        assert A4_RECONCILIATION_CONFLICT in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_RECONCILIATION_CONFLICT])

    def test_canonical_id_gap(self):
        # Two character entities whose ids skip char_0002.
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[],
            pair_plans=[],
            char_entities=[
                char_entity("char_0001", (L,)),
                char_entity("char_0003", (R,)),
            ],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0003", unresolved_id=None),
            ],
        )
        findings = check(b)
        assert A4_CANONICAL_ID_GAP in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_CANONICAL_ID_GAP])

    def test_entity_map_duplicate(self):
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[],
            pair_plans=[],
            char_entities=[
                char_entity("char_0001", (L,)),
                char_entity("char_0002", (R,)),
            ],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0002", unresolved_id=None),
            ],
        )
        findings = check(b)
        assert A4_ENTITY_MAP_DUPLICATE in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_ENTITY_MAP_DUPLICATE])

    def test_entity_map_unaccounted(self):
        # R has no EntityMap entry.
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[],
            pair_plans=[],
            char_entities=[
                char_entity("char_0001", (L,)),
                char_entity("char_0002", (R,)),
            ],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
            ],
        )
        findings = check(b)
        assert A4_ENTITY_MAP_UNACCOUNTED in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_ENTITY_MAP_UNACCOUNTED])

    def test_entity_map_target_not_found(self):
        # EntityMap entry claims char_0002 for L, but L is in char_0001.
        b = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[],
            pair_plans=[],
            char_entities=[
                char_entity("char_0001", (L,)),
                char_entity("char_0002", (R,)),
            ],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0002", unresolved_id=None),
                EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0002", unresolved_id=None),
            ],
        )
        findings = check(b)
        assert A4_ENTITY_MAP_TARGET_NOT_FOUND in codes(findings)
        assert_blocking([f for f in findings if f.code == A4_ENTITY_MAP_TARGET_NOT_FOUND])

    def test_finding_id_is_deterministic(self):
        b1 = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "different_entity")],
            pair_plans=[plan(L, R, "auto_same")],
            char_entities=[
                char_entity("char_0001", (L,)),
                char_entity("char_0002", (R,)),
            ],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0002", unresolved_id=None),
            ],
        )
        f1 = [f for f in check(b1) if f.code == A4_DETERMINISTIC_CONSTRAINT_OVERRIDE]
        b2 = make_bundle(
            index_entries=[entry(L), entry(R)],
            decisions=[decision(L, R, "different_entity")],
            pair_plans=[plan(L, R, "auto_same")],
            char_entities=[
                char_entity("char_0001", (L,)),
                char_entity("char_0002", (R,)),
            ],
            map_entries=[
                EntityMapEntry(candidate_ref=L, status="resolved", canonical_id="char_0001", unresolved_id=None),
                EntityMapEntry(candidate_ref=R, status="resolved", canonical_id="char_0002", unresolved_id=None),
            ],
        )
        f2 = [f for f in check(b2) if f.code == A4_DETERMINISTIC_CONSTRAINT_OVERRIDE]
        assert [f.finding_id for f in f1] == [f.finding_id for f in f2]
        assert f1[0].finding_id.startswith("a4-a4_deterministic_constraint_override-")

    def test_all_thirteen_codes_are_distinct(self):
        expected = {
            A4_CANDIDATE_REF_DUPLICATE,
            A4_CANDIDATE_REF_NOT_FOUND,
            A4_DECISION_PAIR_DUPLICATE,
            A4_DECISION_PAIR_MISSING,
            A4_DECISION_PAIR_UNEXPECTED,
            A4_DECISION_REF_NOT_FOUND,
            A4_DECISION_TYPE_MISMATCH,
            A4_DETERMINISTIC_CONSTRAINT_OVERRIDE,
            A4_RECONCILIATION_CONFLICT,
            A4_CANONICAL_ID_GAP,
            A4_ENTITY_MAP_DUPLICATE,
            A4_ENTITY_MAP_UNACCOUNTED,
            A4_ENTITY_MAP_TARGET_NOT_FOUND,
        }
        assert len(expected) == 13
        assert expected == {
            A4_CANDIDATE_REF_DUPLICATE,
            A4_CANDIDATE_REF_NOT_FOUND,
            A4_DECISION_PAIR_DUPLICATE,
            A4_DECISION_PAIR_MISSING,
            A4_DECISION_PAIR_UNEXPECTED,
            A4_DECISION_REF_NOT_FOUND,
            A4_DECISION_TYPE_MISMATCH,
            A4_DETERMINISTIC_CONSTRAINT_OVERRIDE,
            A4_RECONCILIATION_CONFLICT,
            A4_CANONICAL_ID_GAP,
            A4_ENTITY_MAP_DUPLICATE,
            A4_ENTITY_MAP_UNACCOUNTED,
            A4_ENTITY_MAP_TARGET_NOT_FOUND,
        }


# ---------------------------------------------------------------------------
# A4 semantic identity + parity with A4C preparation
# ---------------------------------------------------------------------------


class TestSemanticIdentity:
    def _profile(self):
        return load_entity_reconciliation_profile(RECON_PROFILE_PATH)

    def _sem_profile(self):
        return load_semantic_profile(A4_LLM_PROFILE_PATH)

    def test_identity_fields_from_authoritative_sources(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION, entries=(entry(L), entry(R))
        )
        planning = make_planning(index, [plan(L, R, "needs_semantic_decision")], (), plan_hash=H2)
        profile = self._profile()
        prep = prepare_semantic_resolution(planning, profile, self._sem_profile())
        identity = build_a4_semantic_identity(profile, prep, planning)
        assert isinstance(identity, A4SemanticIdentity)
        assert identity.reconciliation_profile_id == profile.profile_id
        assert identity.reconciliation_profile_hash == profile.profile_hash
        assert identity.plan_hash == H2
        assert identity.prompt_id == prep.prompt_id
        assert identity.prompt_version == prep.prompt_version
        assert identity.prompt_content_hash == prep.prompt_content_hash
        assert identity.output_schema_id == prep.output_schema_id
        assert identity.output_schema_version == prep.output_schema_version
        assert identity.output_schema_hash == prep.output_schema_hash
        assert identity.semantic_profile_id == prep.semantic_profile_id
        assert identity.semantic_profile_hash == prep.semantic_profile_hash

    def test_identity_request_hashes_equal_prepare_hashes(self):
        # A needs_llm pair produces real request hashes; the identity must carry
        # them verbatim (backend-neutral request hashes, not provider metadata).
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION, entries=(entry(L), entry(R))
        )
        planning = make_planning(index, [plan(L, R, "needs_semantic_decision")], ())
        profile = self._profile()
        prep = prepare_semantic_resolution(planning, profile, self._sem_profile())
        identity = build_a4_semantic_identity(profile, prep, planning)
        assert len(prep.semantic_request_hashes) == 1
        assert identity.semantic_request_hashes == prep.semantic_request_hashes
        assert identity.semantic_request_hashes == tuple(
            request.request_hash for request in prep.structured_requests
        )

    def test_zero_pair_preparation_gives_empty_request_hashes(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION, entries=(entry(L), entry(R))
        )
        planning = make_planning(index, [plan(L, R, "auto_same")], ())
        profile = self._profile()
        prep = prepare_semantic_resolution(planning, profile, self._sem_profile())
        identity = build_a4_semantic_identity(profile, prep, planning)
        assert identity.semantic_request_hashes == ()

    def test_backend_switch_does_not_change_identity(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION, entries=(entry(L), entry(R))
        )
        planning = make_planning(index, [plan(L, R, "needs_semantic_decision")], ())
        profile = self._profile()
        prep = prepare_semantic_resolution(planning, profile, self._sem_profile())
        base = build_a4_semantic_identity(profile, prep, planning)
        # Same identity must hold regardless of backend (no provider metadata in it).
        assert all(
            provider not in identity_field_to_str(base)
            for provider in ("qwen", "openai", "anthropic")
        )

    def test_identity_round_trip(self):
        index = CandidateEntityIndex(
            schema_version=CANDIDATE_ENTITY_INDEX_SCHEMA_VERSION, entries=(entry(L), entry(R))
        )
        planning = make_planning(index, [plan(L, R, "needs_semantic_decision")], ())
        profile = self._profile()
        prep = prepare_semantic_resolution(planning, profile, self._sem_profile())
        identity = build_a4_semantic_identity(profile, prep, planning)
        assert A4SemanticIdentity.from_dict(identity.to_dict()) == identity


def identity_field_to_str(identity: A4SemanticIdentity) -> str:
    return " ".join(
        str(value)
        for value in identity.to_dict().values()
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
