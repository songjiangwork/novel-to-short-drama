"""v1.2 A5B Phase B -- deterministic blocking-v1 pair planning (zero provider).

Covers the frozen post-audit A5B blocking-v1 contract implemented on top of
the Phase A input binding / index:

  * deterministic text normalization (``a5-text-normalization-v1``);
  * fact / event / relationship blocking-v1 pair generation (bucket/index
    based, never the naive N-choose-2 cross product) with the exact signal set;
  * a structural proof that the planner does NOT materialize the naive N^2
    cross product;
  * exact-safe auto-same keys (fact / event / relationship) and the
    ``None`` vs ``""`` distinction for the optional relationship state;
  * the deterministic decision set (auto_same only, deterministic id,
    method/reason, null prompt/provenance, stable evidence union + dedupe);
  * the deterministic plan hash (over the documented material incl. policy
    ids; deterministic; changes with the index);
  * frozen blocking-policy-id verification (fail closed on mismatch);
  * the pair-plan structure (left < right, sorted, integer signal counts,
    only ``auto_same`` / ``needs_semantic_decision`` states, no
    ``not_compared`` materialization);
  * the Alice zero-provider smoke test (exact acceptance gates, read-only,
    deterministic);
  * the semantic-stream boundary (``needs_semantic_decision`` pairs carry no
    prompt / provenance -- those belong to #52/#53).

All fixtures are self-contained synthetic run trees (no dependency on the
local Alice run tree; the Alice smoke test is skipped when the tree is absent).
No provider is called and nothing is persisted anywhere in this file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from short_drama.artifacts.canonical import content_hash
from short_drama.story import (
    A5B_BLOCKING_POLICY_ID,
    EXACT_SAFE_POLICY_ID,
    PLANNING_POLICY_ID,
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    TEXT_NORMALIZATION_POLICY_ID,
    DeterministicConsolidationDecision,
    DeterministicConsolidationDecisionSet,
    EventCandidate,
    FactCandidate,
    RelationshipCandidate,
    StoryIntegrityError,
    build_consolidation_planning,
    event_exact_safe_key,
    fact_exact_safe_key,
    normalize_consolidation_text,
    relationship_exact_safe_key,
)
from short_drama.story.extraction import EvidenceRef
from test_story_a5b_audit import (
    ChunkSpec,
    _build_multi_chunk_tree,
    _fingerprint_dir,
    _plan,
    _rich_corpus,
)
from test_story_a5b_planning import (
    DOCUMENT,
    PROJECT,
    RECON_PROFILE_ID,
    _char,
    _consolidation_profile,
    _loc,
)


# ---------------------------------------------------------------------------
# Candidate builders with fully controlled fields
# ---------------------------------------------------------------------------


def _ev(paragraph_id: str) -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role="primary", strength="explicit", excerpt="txt"
    )


def _mk_fact(
    cid: str,
    *,
    fact_type: str = "world_fact",
    statement_zh: str = "stmt",
    subject_refs=(),
    object_refs=(),
    para: str,
) -> FactCandidate:
    return FactCandidate(
        candidate_id=cid,
        fact_type=fact_type,
        statement_zh=statement_zh,
        subject_refs=tuple(subject_refs),
        object_refs=tuple(object_refs),
        evidence_strength="explicit",
        evidence=(_ev(para),),
    )


def _mk_event(
    cid: str,
    *,
    summary_zh: str = "evt",
    participant_refs=(),
    location_refs=(),
    temporal_mode: str = "normal",
    para: str,
) -> EventCandidate:
    return EventCandidate(
        candidate_id=cid,
        summary_zh=summary_zh,
        participant_refs=tuple(participant_refs),
        location_refs=tuple(location_refs),
        temporal_mode=temporal_mode,
        evidence_strength="explicit",
        evidence=(_ev(para),),
    )


def _mk_rel(
    cid: str,
    *,
    source_ref: str,
    target_ref: str,
    relationship_type_zh: str = "meets",
    state_zh=None,
    direction: str = "directed",
    para: str,
) -> RelationshipCandidate:
    return RelationshipCandidate(
        candidate_id=cid,
        source_ref=source_ref,
        target_ref=target_ref,
        relationship_type_zh=relationship_type_zh,
        state_zh=state_zh,
        direction=direction,
        evidence_strength="explicit",
        evidence=(_ev(para),),
    )


def _pair_by_refs(plans, left, right):
    for plan in plans:
        if {plan.left_ref, plan.right_ref} == {left, right}:
            return plan
    return None


def _global_refs_for(result, category: str, ids) -> dict:
    """Map a local candidate id to its A5 global ref (``<chunk_id>:<local_id>``)."""
    out = {}
    for cand in getattr(result.index, category):
        if cand.local_candidate_id in ids:
            out[cand.local_candidate_id] = cand.global_candidate_ref
    return out


# ---------------------------------------------------------------------------
# 1. Deterministic normalization
# ---------------------------------------------------------------------------


def test_normalize_deterministic_exact():
    # casefold + strip + collapse internal whitespace to a single ASCII space
    assert normalize_consolidation_text("  Alice   met \nBob\tCarol ") == (
        "alice met bob carol"
    )
    # CJK is preserved verbatim (NFKC is a no-op for ideographs), only spaces collapsed
    assert normalize_consolidation_text("  姐妹  关系 ") == "姐妹 关系"
    # NFKC folds full-width latin to ASCII
    assert normalize_consolidation_text("Ａｌｉｃｅ") == "alice"
    # empty / whitespace-only -> empty
    assert normalize_consolidation_text("") == ""
    assert normalize_consolidation_text("   ") == ""
    # idempotent / deterministic
    once = normalize_consolidation_text("  X  Y  ")
    assert normalize_consolidation_text(once) == once == "x y"
    # no fuzzy: distinct texts stay distinct
    assert normalize_consolidation_text("alice") != normalize_consolidation_text("alicia")
    assert normalize_consolidation_text("run") != normalize_consolidation_text("ran")


# ---------------------------------------------------------------------------
# 2. Fact blocking-v1
# ---------------------------------------------------------------------------


def test_fact_signal_exact_normalized_statement(tmp_path):
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(_mk_fact("cand_fact_001", statement_zh="Same Statement",
                            subject_refs=("cand_char_001",), para="CH001_P0001"),),
        ),
        ChunkSpec(
            "CH002", ("CH002_P0001",),
            chars=(_char("cand_char_002", "Bob"),),
            facts=(_mk_fact("cand_fact_002", fact_type="identity",
                            statement_zh="  same   statement ",
                            subject_refs=("cand_char_002",), para="CH002_P0001"),),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "facts", {"cand_fact_001", "cand_fact_002"})
    plan = _pair_by_refs(result.fact_pair_plans, refs["cand_fact_001"], refs["cand_fact_002"])
    assert plan is not None, "same normalized statement must form a candidate pair"
    assert plan.blocking_signal_counts.get("exact_normalized_statement") == 1


def test_fact_signal_evidence_paragraph_overlap(tmp_path):
    # Two facts in the SAME chunk sharing a paragraph (different statement/type/entity).
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001", "CH001_P0002"),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob")),
            facts=(
                _mk_fact("cand_fact_001", statement_zh="stmt A", fact_type="world_fact",
                         subject_refs=("cand_char_001",), para="CH001_P0001"),
                _mk_fact("cand_fact_002", statement_zh="stmt B", fact_type="identity",
                         subject_refs=("cand_char_002",), para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "facts", {"cand_fact_001", "cand_fact_002"})
    plan = _pair_by_refs(result.fact_pair_plans, refs["cand_fact_001"], refs["cand_fact_002"])
    assert plan is not None, "shared evidence paragraph must form a candidate pair"
    assert plan.blocking_signal_counts.get("evidence_paragraph_overlap") == 1


def test_fact_signal_same_type_bound_entity(tmp_path):
    # Same normalized type + a shared bound entity. The two same-name chars are
    # reconciled to ONE canonical entity (same_entity), so the two facts share a
    # bound entity -> the bound-entity signal fires.
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(_mk_fact("cand_fact_001", statement_zh="stmt A",
                            subject_refs=("cand_char_001",), para="CH001_P0001"),),
        ),
        ChunkSpec(
            "CH002", ("CH002_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(_mk_fact("cand_fact_002", statement_zh="stmt B",
                            subject_refs=("cand_char_001",), para="CH002_P0001"),),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs, decision="same_entity")
    result = _plan(tree)
    refs = _global_refs_for(result, "facts", {"cand_fact_001", "cand_fact_002"})
    plan = _pair_by_refs(result.fact_pair_plans, refs["cand_fact_001"], refs["cand_fact_002"])
    assert plan is not None, "same type + shared bound entity must form a candidate pair"
    assert plan.blocking_signal_counts.get("same_fact_type_bound_entity_overlap") == 1


def test_fact_signal_same_chunk(tmp_path):
    # Same normalized type + same chunk (distinct statement / paragraph / entity).
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001", "CH001_P0002"),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob")),
            facts=(
                _mk_fact("cand_fact_001", statement_zh="stmt A",
                         subject_refs=("cand_char_001",), para="CH001_P0001"),
                _mk_fact("cand_fact_002", statement_zh="stmt B",
                         subject_refs=("cand_char_002",), para="CH001_P0002"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "facts", {"cand_fact_001", "cand_fact_002"})
    plan = _pair_by_refs(result.fact_pair_plans, refs["cand_fact_001"], refs["cand_fact_002"])
    assert plan is not None
    assert plan.blocking_signal_counts.get("same_fact_type_same_chunk") == 1


def test_fact_signal_adjacent_chunk(tmp_path):
    # Same normalized type, adjacent chunks, no shared entity / paragraph / statement.
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(_mk_fact("cand_fact_001", statement_zh="stmt A",
                            subject_refs=("cand_char_001",), para="CH001_P0001"),),
        ),
        ChunkSpec(
            "CH002", ("CH002_P0001",),
            chars=(_char("cand_char_002", "Bob"),),
            facts=(_mk_fact("cand_fact_002", statement_zh="stmt B",
                            subject_refs=("cand_char_002",), para="CH002_P0001"),),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "facts", {"cand_fact_001", "cand_fact_002"})
    plan = _pair_by_refs(result.fact_pair_plans, refs["cand_fact_001"], refs["cand_fact_002"])
    assert plan is not None
    assert plan.blocking_signal_counts.get("same_fact_type_adjacent_chunk") == 1


def test_fact_far_chunk_no_signals_not_materialized(tmp_path):
    # Different type, statement, paragraph, entity, and far chunk (distance 2):
    # no fact signal fires, so the pair must NOT be materialized.
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(_mk_fact("cand_fact_001", fact_type="world_fact", statement_zh="stmt A",
                            subject_refs=("cand_char_001",), para="CH001_P0001"),),
        ),
        ChunkSpec("CH002", ("CH002_P0001",), unres=()),  # filler
        ChunkSpec(
            "CH003", ("CH003_P0001",),
            chars=(_char("cand_char_002", "Bob"),),
            facts=(_mk_fact("cand_fact_002", fact_type="identity", statement_zh="stmt B",
                            subject_refs=("cand_char_002",), para="CH003_P0001"),),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "facts", {"cand_fact_001", "cand_fact_002"})
    plan = _pair_by_refs(result.fact_pair_plans, refs["cand_fact_001"], refs["cand_fact_002"])
    assert plan is None, "no fact signal fired -> pair must not be materialized"


# ---------------------------------------------------------------------------
# 3. Event blocking-v1
# ---------------------------------------------------------------------------


def test_event_signal_exact_normalized_summary(tmp_path):
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            events=(_mk_event("cand_evt_001", summary_zh="Same Summary",
                              participant_refs=("cand_char_001",), para="CH001_P0001"),),
        ),
        ChunkSpec(
            "CH002", ("CH002_P0001",),
            chars=(_char("cand_char_002", "Bob"),),
            events=(_mk_event("cand_evt_002", summary_zh="  same   summary ",
                              participant_refs=("cand_char_002",), para="CH002_P0001"),),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "events", {"cand_evt_001", "cand_evt_002"})
    plan = _pair_by_refs(result.event_pair_plans, refs["cand_evt_001"], refs["cand_evt_002"])
    assert plan is not None
    assert plan.blocking_signal_counts.get("exact_normalized_summary") == 1


def test_event_signal_same_chunk_and_adjacent(tmp_path):
    # Same chunk -> same_chunk signal; adjacent chunk + shared bound entity ->
    # adjacent_chunk_shared_entity. The two same-name chars reconcile to one
    # canonical entity (same_entity).
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001", "CH001_P0002"),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob")),
            events=(
                _mk_event("cand_evt_001", summary_zh="sum A",
                          participant_refs=("cand_char_001",), para="CH001_P0001"),
                _mk_event("cand_evt_002", summary_zh="sum B",
                          participant_refs=("cand_char_002",), para="CH001_P0002"),
            ),
        ),
        ChunkSpec(
            "CH002", ("CH002_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            events=(_mk_event("cand_evt_003", summary_zh="sum C",
                              participant_refs=("cand_char_001",), para="CH002_P0001"),),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs, decision="same_entity")
    result = _plan(tree)
    refs = _global_refs_for(result, "events", {"cand_evt_001", "cand_evt_002", "cand_evt_003"})
    plan_same = _pair_by_refs(result.event_pair_plans, refs["cand_evt_001"], refs["cand_evt_002"])
    assert plan_same is not None
    assert plan_same.blocking_signal_counts.get("same_chunk") == 1
    plan_adj = _pair_by_refs(result.event_pair_plans, refs["cand_evt_001"], refs["cand_evt_003"])
    assert plan_adj is not None
    assert plan_adj.blocking_signal_counts.get("adjacent_chunk_shared_entity") == 1


def test_event_signal_participant_location_overlap(tmp_path):
    # Two events sharing a participant AND a location -> participant_location_overlap.
    # Different summaries keep the exact-summary signal out.
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            locs=(_loc("cand_loc_001", "Wonderland"),),
            events=(
                _mk_event("cand_evt_001", summary_zh="sum A",
                          participant_refs=("cand_char_001",),
                          location_refs=("cand_loc_001",), para="CH001_P0001"),
                _mk_event("cand_evt_002", summary_zh="sum B",
                          participant_refs=("cand_char_001",),
                          location_refs=("cand_loc_001",), para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "events", {"cand_evt_001", "cand_evt_002"})
    plan = _pair_by_refs(result.event_pair_plans, refs["cand_evt_001"], refs["cand_evt_002"])
    assert plan is not None
    assert plan.blocking_signal_counts.get("participant_location_overlap") == 1


def test_event_no_signals_not_materialized(tmp_path):
    # Different summary, far chunk (distance 2), no shared participant/location/entity.
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            events=(_mk_event("cand_evt_001", summary_zh="sum A",
                              participant_refs=("cand_char_001",), para="CH001_P0001"),),
        ),
        ChunkSpec("CH002", ("CH002_P0001",), unres=()),  # filler
        ChunkSpec(
            "CH003", ("CH003_P0001",),
            chars=(_char("cand_char_002", "Bob"),),
            events=(_mk_event("cand_evt_002", summary_zh="sum B",
                              participant_refs=("cand_char_002",), para="CH003_P0001"),),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "events", {"cand_evt_001", "cand_evt_002"})
    plan = _pair_by_refs(result.event_pair_plans, refs["cand_evt_001"], refs["cand_evt_002"])
    assert plan is None


# ---------------------------------------------------------------------------
# 4. Relationship blocking-v1
# ---------------------------------------------------------------------------


def test_relationship_same_endpoint_group_pairs(tmp_path):
    # Same endpoint group (same two endpoints), different direction order and
    # different type/state -> all in the same endpoint group -> candidate pairs.
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob")),
            rels=(
                _mk_rel("cand_rel_001", source_ref="cand_char_001", target_ref="cand_char_002",
                        relationship_type_zh="helps", state_zh="x", para="CH001_P0001"),
                _mk_rel("cand_rel_002", source_ref="cand_char_002", target_ref="cand_char_001",
                        relationship_type_zh="fears", state_zh=None, para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "relationships", {"cand_rel_001", "cand_rel_002"})
    plan = _pair_by_refs(result.relationship_pair_plans, refs["cand_rel_001"], refs["cand_rel_002"])
    assert plan is not None, "same endpoint group must form a candidate pair"
    assert plan.blocking_signal_counts.get("same_endpoint_group") == 1


def test_relationship_different_endpoint_group_not_materialized(tmp_path):
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob"),
                   _char("cand_char_003", "Carol")),
            rels=(
                _mk_rel("cand_rel_001", source_ref="cand_char_001", target_ref="cand_char_002",
                        para="CH001_P0001"),
                _mk_rel("cand_rel_002", source_ref="cand_char_001", target_ref="cand_char_003",
                        para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "relationships", {"cand_rel_001", "cand_rel_002"})
    plan = _pair_by_refs(result.relationship_pair_plans, refs["cand_rel_001"], refs["cand_rel_002"])
    assert plan is None, "different endpoint group must NOT form a candidate pair"


# ---------------------------------------------------------------------------
# 5. No N^2 structural proof
# ---------------------------------------------------------------------------


def test_planner_does_not_materialize_n2_cross_product(tmp_path):
    # N facts in N consecutive chunks, all the SAME type, but each with a distinct
    # statement / paragraph and NO shared entity. The ONLY signal that can fire is
    # ``same_fact_type_adjacent_chunk`` (N-1 pairs). The naive N-choose-2 count is
    # N*(N-1)/2, which must be far larger than the planner's output.
    n = 40
    specs = []
    for i in range(n):
        para = f"CH{i + 1:03d}_P0001"
        specs.append(
            ChunkSpec(
                f"CH{i + 1:03d}",
                (para,),
                chars=(_char(f"cand_char_{i + 1:03d}", f"Char{i}"),),
                facts=(_mk_fact(
                    f"cand_fact_{i + 1:03d}",
                    statement_zh=f"distinct statement {i}",
                    subject_refs=(f"cand_char_{i + 1:03d}",),
                    para=para,
                ),),
            )
        )
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    assert len(result.index.facts) == n
    planned = len(result.fact_pair_plans)
    naive = n * (n - 1) // 2
    # Exactly the N-1 adjacent-chunk pairs; NOT the naive cross product.
    assert planned == n - 1
    assert naive > planned * 10, f"planned {planned} should be far below naive {naive}"
    assert planned < naive


# ---------------------------------------------------------------------------
# 6. Exact-safe auto-same
# ---------------------------------------------------------------------------


def test_fact_exact_safe_auto_same(tmp_path):
    # Two facts with identical exact fact_type + normalized statement + refs ->
    # auto_same (same chunk so a blocking pair exists).
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            locs=(_loc("cand_loc_001", "Wonderland"),),
            facts=(
                _mk_fact("cand_fact_001", statement_zh="Alice lives in Wonderland",
                         subject_refs=("cand_char_001",), object_refs=("cand_loc_001",),
                         para="CH001_P0001"),
                _mk_fact("cand_fact_002", statement_zh="  alice   lives in wonderland ",
                         subject_refs=("cand_char_001",), object_refs=("cand_loc_001",),
                         para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "facts", {"cand_fact_001", "cand_fact_002"})
    plan = _pair_by_refs(result.fact_pair_plans, refs["cand_fact_001"], refs["cand_fact_002"])
    assert plan is not None
    assert plan.state == PAIR_STATE_AUTO_SAME


def test_fact_not_auto_same_when_refs_differ(tmp_path):
    # Same normalized statement + type but different object refs -> NOT auto_same.
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            locs=(_loc("cand_loc_001", "Wonderland"), _loc("cand_loc_002", "Looking Glass")),
            facts=(
                _mk_fact("cand_fact_001", statement_zh="Alice lives here",
                         subject_refs=("cand_char_001",), object_refs=("cand_loc_001",),
                         para="CH001_P0001"),
                _mk_fact("cand_fact_002", statement_zh="Alice lives here",
                         subject_refs=("cand_char_001",), object_refs=("cand_loc_002",),
                         para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "facts", {"cand_fact_001", "cand_fact_002"})
    plan = _pair_by_refs(result.fact_pair_plans, refs["cand_fact_001"], refs["cand_fact_002"])
    assert plan is not None
    assert plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION


def test_event_exact_safe_auto_same(tmp_path):
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            locs=(_loc("cand_loc_001", "Wonderland"),),
            events=(
                _mk_event("cand_evt_001", summary_zh="Alice falls down the hole",
                          participant_refs=("cand_char_001",),
                          location_refs=("cand_loc_001",), temporal_mode="normal",
                          para="CH001_P0001"),
                _mk_event("cand_evt_002", summary_zh=" alice  falls down the hole ",
                          participant_refs=("cand_char_001",),
                          location_refs=("cand_loc_001",), temporal_mode="normal",
                          para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "events", {"cand_evt_001", "cand_evt_002"})
    plan = _pair_by_refs(result.event_pair_plans, refs["cand_evt_001"], refs["cand_evt_002"])
    assert plan is not None
    assert plan.state == PAIR_STATE_AUTO_SAME


def test_relationship_exact_safe_auto_same_and_none_vs_state(tmp_path):
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob")),
            rels=(
                _mk_rel("cand_rel_001", source_ref="cand_char_001", target_ref="cand_char_002",
                        relationship_type_zh="helps", state_zh=None, para="CH001_P0001"),
                _mk_rel("cand_rel_002", source_ref="cand_char_001", target_ref="cand_char_002",
                        relationship_type_zh=" helps ", state_zh=None, para="CH001_P0001"),
                # same endpoints + type but a PRESENT (non-empty) state -> not auto_same
                _mk_rel("cand_rel_003", source_ref="cand_char_001", target_ref="cand_char_002",
                        relationship_type_zh="helps", state_zh="active", para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "relationships", {"cand_rel_001", "cand_rel_002", "cand_rel_003"})
    plan_same = _pair_by_refs(result.relationship_pair_plans, refs["cand_rel_001"], refs["cand_rel_002"])
    assert plan_same is not None
    assert plan_same.state == PAIR_STATE_AUTO_SAME
    plan_none_vs_state = _pair_by_refs(
        result.relationship_pair_plans, refs["cand_rel_001"], refs["cand_rel_003"]
    )
    assert plan_none_vs_state is not None
    assert plan_none_vs_state.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION


def test_relationship_opposite_direction_not_auto_same(tmp_path):
    # Same endpoint group + type + state but OPPOSITE directed order -> the
    # endpoint identity key differs -> NOT auto_same (needs a semantic decision).
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob")),
            rels=(
                _mk_rel("cand_rel_001", source_ref="cand_char_001", target_ref="cand_char_002",
                        relationship_type_zh="helps", state_zh="x", para="CH001_P0001"),
                _mk_rel("cand_rel_002", source_ref="cand_char_002", target_ref="cand_char_001",
                        relationship_type_zh="helps", state_zh="x", para="CH001_P0001"),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    refs = _global_refs_for(result, "relationships", {"cand_rel_001", "cand_rel_002"})
    plan = _pair_by_refs(result.relationship_pair_plans, refs["cand_rel_001"], refs["cand_rel_002"])
    assert plan is not None
    assert plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION


def test_exact_safe_keys_pure():
    """The exact-safe key functions are pure and expose the documented material."""
    from short_drama.artifacts import ArtifactRef
    from short_drama.story.consolidation import (
        IndexedEventCandidate,
        IndexedFactCandidate,
        IndexedRelationshipCandidate,
    )

    def _ref():
        return ArtifactRef(
            artifact_type="candidate_extraction", artifact_id="x", revision=1,
            content_hash="f" * 64,
        )

    def _key(n: int) -> str:
        return f"000001:000000001:{n:02d}:{n:09d}:CH001_C001:cand_{n:03d}"

    f1 = IndexedFactCandidate(
        global_candidate_ref="CH001_C001:cand_fact_001", chunk_id="CH001_C001",
        local_candidate_id="cand_fact_001", source_order_key=_key(1),
        fact_type="world_fact", statement_zh="Alice lives here",
        subject_refs=("char_0001",), object_refs=("loc_0001",),
        evidence_strength="explicit", evidence_refs=(_ev("CH001_P0001"),),
        candidate_extraction_ref=_ref(),
    )
    f2 = IndexedFactCandidate(
        global_candidate_ref="CH001_C001:cand_fact_002", chunk_id="CH001_C001",
        local_candidate_id="cand_fact_002", source_order_key=_key(2),
        fact_type="world_fact", statement_zh="  alice  lives here ",
        subject_refs=("char_0001",), object_refs=("loc_0001",),
        evidence_strength="explicit", evidence_refs=(_ev("CH001_P0001"),),
        candidate_extraction_ref=_ref(),
    )
    assert fact_exact_safe_key(f1) == fact_exact_safe_key(f2)
    f3 = IndexedFactCandidate(
        global_candidate_ref="CH001_C001:cand_fact_003", chunk_id="CH001_C001",
        local_candidate_id="cand_fact_003", source_order_key=_key(3),
        fact_type="world_fact", statement_zh="Alice lives here",
        subject_refs=("char_0001",), object_refs=("loc_0002",),
        evidence_strength="explicit", evidence_refs=(_ev("CH001_P0001"),),
        candidate_extraction_ref=_ref(),
    )
    assert fact_exact_safe_key(f1) != fact_exact_safe_key(f3)

    e1 = IndexedEventCandidate(
        global_candidate_ref="CH001_C001:cand_evt_001", chunk_id="CH001_C001",
        local_candidate_id="cand_evt_001", source_order_key=_key(4),
        summary_zh="Alice falls down the hole", participants=("char_0001",),
        locations=("loc_0001",), temporal_mode="normal",
        evidence_strength="explicit", evidence_refs=(_ev("CH001_P0001"),),
        candidate_extraction_ref=_ref(),
    )
    e2 = IndexedEventCandidate(
        global_candidate_ref="CH001_C001:cand_evt_002", chunk_id="CH001_C001",
        local_candidate_id="cand_evt_002", source_order_key=_key(5),
        summary_zh=" alice  falls down the hole ", participants=("char_0001",),
        locations=("loc_0001",), temporal_mode="normal",
        evidence_strength="explicit", evidence_refs=(_ev("CH001_P0001"),),
        candidate_extraction_ref=_ref(),
    )
    assert event_exact_safe_key(e1) == event_exact_safe_key(e2)

    # The relationship key distinguishes an ABSENT state (None) from a present
    # one. The A5A/A3 models only permit None or non-empty state, so the
    # defensive ``None`` vs ``""`` distinction is exercised on the key function
    # itself (via a duck-typed namespace that bypasses model validation).
    import types

    def _rel_ns(state):
        return types.SimpleNamespace(
            relationship_type_zh="helps",
            source_entity_ref="char_0001", target_entity_ref="char_0002",
            direction="directed", state_zh=state,
        )

    # None vs empty string are distinct for the optional state.
    assert relationship_exact_safe_key(_rel_ns(None)) != relationship_exact_safe_key(_rel_ns(""))
    # None vs a present state are distinct.
    assert relationship_exact_safe_key(_rel_ns(None)) != relationship_exact_safe_key(_rel_ns("active"))
    # A valid indexed candidate with a None state is also accepted by the key.
    r_none = IndexedRelationshipCandidate(
        global_candidate_ref="CH001_C001:cand_rel_001", chunk_id="CH001_C001",
        local_candidate_id="cand_rel_001", source_order_key=_key(6),
        source_entity_ref="char_0001", target_entity_ref="char_0002",
        relationship_type_zh="helps", state_zh=None, direction="directed",
        evidence_strength="explicit", evidence_refs=(_ev("CH001_P0001"),),
        candidate_extraction_ref=_ref(),
    )
    assert relationship_exact_safe_key(r_none) == relationship_exact_safe_key(_rel_ns(None))


# ---------------------------------------------------------------------------
# 7. Deterministic decision set
# ---------------------------------------------------------------------------


def test_decision_set_auto_same_only_and_fields(tmp_path):
    # A rich corpus (from the audit helper). Every decision must be
    # deterministic, complete, and exactly the auto_same pairs (across domains).
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    result = _plan(tree)
    decision_set = result.deterministic_decision_set
    assert isinstance(decision_set, DeterministicConsolidationDecisionSet)
    auto_pairs = {
        frozenset((p.left_ref, p.right_ref))
        for plans in (result.fact_pair_plans, result.event_pair_plans,
                      result.relationship_pair_plans)
        for p in plans if p.state == PAIR_STATE_AUTO_SAME
    }
    decision_pairs = {frozenset((d.left_ref, d.right_ref)) for d in decision_set.decisions}
    assert decision_pairs == auto_pairs
    for d in decision_set.decisions:
        assert d.method == "deterministic"
        assert d.reason_code == "exact_safe"
        assert d.prompt_id is None
        assert d.prompt_version is None
        assert d.generation_provenance is None
        assert d.pair_kind in ("fact", "event", "relationship")
        assert d.decision_id.startswith("dec_")
        assert len(d.decision_id) == 4 + 20
        assert d.reason_zh
        assert all(e.paragraph_id for e in d.evidence_refs)


def test_decision_id_deterministic_and_evidence_union(tmp_path):
    # Two identical facts (same chunk, different paragraphs) auto_same; the
    # decision id is deterministic and the evidence is the union of both sides.
    para_a = "CH001_P0001"
    para_b = "CH001_P0002"
    specs = [
        ChunkSpec(
            "CH001", (para_a, para_b),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(
                _mk_fact("cand_fact_001", statement_zh="Alice lives here",
                         subject_refs=("cand_char_001",), para=para_a),
                _mk_fact("cand_fact_002", statement_zh="Alice lives here",
                         subject_refs=("cand_char_001",), para=para_b),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    assert len(result.deterministic_decision_set.decisions) == 1
    decision = result.deterministic_decision_set.decisions[0]
    assert tuple(e.paragraph_id for e in decision.evidence_refs) == (para_a, para_b)
    result2 = _plan(tree)
    assert result2.deterministic_decision_set.decisions[0].decision_id == decision.decision_id


def test_decision_evidence_stable_exact_dedupe(tmp_path):
    # Two facts citing the SAME evidence paragraph -> the union dedupes to one.
    para = "CH001_P0001"
    specs = [
        ChunkSpec(
            "CH001", (para,),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(
                _mk_fact("cand_fact_001", statement_zh="Alice lives here",
                         subject_refs=("cand_char_001",), para=para),
                _mk_fact("cand_fact_002", statement_zh="Alice lives here",
                         subject_refs=("cand_char_001",), para=para),
            ),
        ),
    ]
    tree = _build_multi_chunk_tree(tmp_path, *specs)
    result = _plan(tree)
    assert len(result.deterministic_decision_set.decisions) == 1
    decision = result.deterministic_decision_set.decisions[0]
    assert len(decision.evidence_refs) == 1
    assert decision.evidence_refs[0].paragraph_id == para


# ---------------------------------------------------------------------------
# 8. Deterministic plan hash
# ---------------------------------------------------------------------------


def test_plan_hash_deterministic(tmp_path):
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    result1 = _plan(tree)
    result2 = _plan(tree)
    assert result1.plan_hash == result2.plan_hash
    assert isinstance(result1.plan_hash, str) and len(result1.plan_hash) == 64


def test_plan_hash_over_documented_material(tmp_path):
    # The plan hash is the canonical content hash of the documented material
    # (policy ids + index.to_dict() + ordered pair plans + decision set).
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    result = _plan(tree)
    material = {
        "blocking_policy_id": result.blocking_policy_id,
        "text_normalization_policy_id": result.text_normalization_policy_id,
        "exact_safe_policy_id": result.exact_safe_policy_id,
        "planning_policy_id": result.planning_policy_id,
        "candidate_index": result.index.to_dict(),
        "fact_pair_plans": [p.to_dict() for p in result.fact_pair_plans],
        "event_pair_plans": [p.to_dict() for p in result.event_pair_plans],
        "relationship_pair_plans": [p.to_dict() for p in result.relationship_pair_plans],
        "deterministic_decision_set": result.deterministic_decision_set.to_dict(),
    }
    assert result.plan_hash == content_hash(material)


def test_plan_hash_changes_with_index(tmp_path):
    # Different candidates -> different index -> different plan hash.
    def _build(root, statement):
        return _build_multi_chunk_tree(
            root,
            ChunkSpec(
                "CH001", ("CH001_P0001",),
                chars=(_char("cand_char_001", "Alice"),),
                facts=(_mk_fact("cand_fact_001", statement_zh=statement,
                                subject_refs=("cand_char_001",), para="CH001_P0001"),),
            ),
        )

    tree_a = _build(tmp_path / "a", "one")
    tree_b = _build(tmp_path / "b", "two")
    assert _plan(tree_a).plan_hash != _plan(tree_b).plan_hash


# ---------------------------------------------------------------------------
# 9. Frozen blocking-policy-id verification (fail closed)
# ---------------------------------------------------------------------------


def test_wrong_blocking_policy_id_fails_closed(tmp_path):
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    bad_profile = dataclasses.replace(
        _consolidation_profile(), blocking_policy_id="some-other-blocking-v9"
    )
    with pytest.raises(StoryIntegrityError, match="blocking_policy_id"):
        build_consolidation_planning(
            tree.store,
            tree.pointers,
            project_id=PROJECT,
            document_id=DOCUMENT,
            reconciliation_profile_id=RECON_PROFILE_ID,
            consolidation_profile=bad_profile,
        )


def test_profile_policy_ids_are_frozen():
    profile = _consolidation_profile()
    assert profile.blocking_policy_id == A5B_BLOCKING_POLICY_ID == "consolidation-blocking-v1"
    assert TEXT_NORMALIZATION_POLICY_ID == "a5-text-normalization-v1"
    assert EXACT_SAFE_POLICY_ID == "a5-exact-safe-v1"
    assert PLANNING_POLICY_ID == "a5-pair-planning-v1"


# ---------------------------------------------------------------------------
# 10. Pair-plan structure
# ---------------------------------------------------------------------------


def test_pair_plan_structure(tmp_path):
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    result = _plan(tree)
    for plans in (
        result.fact_pair_plans, result.event_pair_plans, result.relationship_pair_plans
    ):
        keys = [(p.left_ref, p.right_ref) for p in plans]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)
        for plan in plans:
            assert plan.left_ref < plan.right_ref
            assert plan.state in (PAIR_STATE_AUTO_SAME, PAIR_STATE_NEEDS_SEMANTIC_DECISION)
            assert plan.blocking_signal_counts
            for signal, count in plan.blocking_signal_counts.items():
                assert isinstance(signal, str)
                assert count == 1
                assert isinstance(count, int) and not isinstance(count, bool)


def test_no_not_compared_state_anywhere(tmp_path):
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    result = _plan(tree)
    for plans in (
        result.fact_pair_plans, result.event_pair_plans, result.relationship_pair_plans
    ):
        for plan in plans:
            assert plan.state in (PAIR_STATE_AUTO_SAME, PAIR_STATE_NEEDS_SEMANTIC_DECISION)


# ---------------------------------------------------------------------------
# 11. Alice zero-provider smoke test (read-only, deterministic, exact gates)
# ---------------------------------------------------------------------------

_ALICE_RUNS = Path("runs") / "a4e_real_novel" / "a3e-real-novel"
_ALICE_PROJECT = "a3e-real-novel"
_ALICE_DOCUMENT = "src_001"
_ALICE_PROFILE = "entity-reconciliation-v2"


def _alice_tree_present() -> bool:
    root = Path(__file__).resolve().parents[1] / _ALICE_RUNS / "story"
    return (root / "artifacts").is_dir() and (root / "pointers").is_dir()


@pytest.mark.skipif(not _alice_tree_present(), reason="Alice run tree not present (local, gitignored)")
def test_alice_zero_provider_smoke():
    from short_drama.artifacts import FileArtifactStore
    from short_drama.foundation import FilePointerStore

    root = Path(__file__).resolve().parents[1] / _ALICE_RUNS / "story"
    store = FileArtifactStore(root / "artifacts")
    pointers = FilePointerStore(root / "pointers", store)

    before = _fingerprint_dir(Path(__file__).resolve().parents[1] / _ALICE_RUNS)
    result = build_consolidation_planning(
        store, pointers, project_id=_ALICE_PROJECT, document_id=_ALICE_DOCUMENT,
        reconciliation_profile_id=_ALICE_PROFILE, consolidation_profile=_consolidation_profile(),
    )
    after = _fingerprint_dir(Path(__file__).resolve().parents[1] / _ALICE_RUNS)
    assert before == after

    assert len(result.index.facts) == 158
    assert len(result.index.events) == 167
    assert len(result.index.relationships) == 116

    fact_total = len(result.fact_pair_plans)
    event_total = len(result.event_pair_plans)
    rel_total = len(result.relationship_pair_plans)
    auto_total = sum(
        1 for plans in (result.fact_pair_plans, result.event_pair_plans,
                        result.relationship_pair_plans)
        for p in plans if p.state == PAIR_STATE_AUTO_SAME
    )
    total_plans = fact_total + event_total + rel_total

    assert auto_total == 0
    assert fact_total <= 5300
    assert event_total <= 4300
    assert rel_total == 845
    assert total_plans <= 10450
    assert len(result.deterministic_decision_set.decisions) == 0

    result2 = build_consolidation_planning(
        store, pointers, project_id=_ALICE_PROJECT, document_id=_ALICE_DOCUMENT,
        reconciliation_profile_id=_ALICE_PROFILE, consolidation_profile=_consolidation_profile(),
    )
    assert result2.plan_hash == result.plan_hash


# ---------------------------------------------------------------------------
# 12. Semantic-stream boundary
# ---------------------------------------------------------------------------


def test_semantic_stream_boundary(tmp_path):
    # The A5B output is ONLY the pair plans (needs_semantic_decision stream) +
    # deterministic decisions. No prompt is rendered, no provider is called,
    # no LLM result is parsed, and no canonical A5 set is produced here.
    tree = _build_multi_chunk_tree(tmp_path, *_rich_corpus())
    result = _plan(tree)
    needs = [
        p for plans in (result.fact_pair_plans, result.event_pair_plans,
                        result.relationship_pair_plans)
        for p in plans if p.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
    ]
    assert needs, "a rich corpus must produce needs_semantic_decision pairs"
    for plan in needs:
        assert not hasattr(plan, "prompt_id")
        assert not hasattr(plan, "generation_provenance")
        assert not hasattr(plan, "provider")
    for decision in result.deterministic_decision_set.decisions:
        assert isinstance(decision, DeterministicConsolidationDecision)
        assert decision.prompt_id is None
        assert decision.generation_provenance is None
    assert not hasattr(result, "canonical_facts")
    assert not hasattr(result, "canonical_events")
