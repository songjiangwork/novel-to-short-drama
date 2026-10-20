"""B1.3-B3B graph assembly: synthetic, zero-provider only."""
from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.adaptation import (
    B13CandidateKind, B13CandidateRef, B13DispositionAssemblyError,
    B13ReconciliationDecision, B13ReconciliationDecisionKind,
    B13ReconciliationDecisionSet, assemble_b13_disposition_plan,
    build_b13_reconciliation_inventory, review_b13_reconciliation,
)

from test_b13_disposition_reconciliation import (
    _character, _result, _source_character, _three_window_result,
    _three_window_preparation, _two_window_preparation,
)
from test_b13_window_semantic import _mapping, _proposal


def _review(result, decisions=()):
    inventory = build_b13_reconciliation_inventory(result)
    return review_b13_reconciliation(
        inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, decisions),
    )


def _character_ref(window_id, local_key):
    return B13CandidateRef(window_id, B13CandidateKind.ADAPTED_CHARACTER, local_key)


def test_same_local_key_is_scoped_and_different_keys_can_be_explicitly_equivalent(tmp_path):
    source = _source_character(_result(tmp_path))
    preparation, _runtime = _two_window_preparation(tmp_path)
    first, second = (_proposal(packet) for packet in preparation.packets)
    first["adapted_characters"] = [_character("lead", source)]
    second["adapted_characters"] = [_character("lead", source)]
    distinct = assemble_b13_disposition_plan(_review(_result(tmp_path, first, second)))
    lead_ids = [item.adapted_id for item in distinct.identity_bindings if item.candidate_ref.candidate_kind is B13CandidateKind.ADAPTED_CHARACTER]
    assert len(set(lead_ids)) == 2

    first["adapted_characters"] = [_character("lead-a", source)]
    second["adapted_characters"] = [_character("lead-b", source)]
    # The semantic content is deliberately equal: this decision, not names or
    # source provenance, is the only reason these local candidates coalesce.
    second["adapted_characters"][0]["display_name_zh"] = first["adapted_characters"][0]["display_name_zh"]
    left, right = _character_ref("window_0001", "lead-a"), _character_ref("window_0002", "lead-b")
    decision = B13ReconciliationDecision("identify_equivalent", (left, right), (), "人工确认。", "审阅记录。")
    equivalent = assemble_b13_disposition_plan(_review(_result(tmp_path, first, second), (decision,)))
    bindings = [item for item in equivalent.identity_bindings if item.candidate_ref in {left, right}]
    assert len({item.adapted_id for item in bindings}) == 1


def test_transitive_equivalence_and_decision_order_have_stable_global_ids(tmp_path):
    preparation, _runtime = _three_window_preparation(tmp_path)
    source = _source_character(_result(tmp_path))
    values = [_proposal(packet) for packet in preparation.packets]
    for value, key in zip(values, ("lead-a", "lead-b", "lead-c")):
        value["adapted_characters"] = [_character(key, source)]
        value["adapted_characters"][0]["display_name_zh"] = "同一主角"
    a, b, c = (_character_ref(f"window_{ordinal:04d}", key) for ordinal, key in enumerate(("lead-a", "lead-b", "lead-c"), 1))
    ab = B13ReconciliationDecision("identify_equivalent", (a, b), (), "确认。", "审阅。")
    bc = B13ReconciliationDecision("identify_equivalent", (b, c), (), "确认。", "审阅。")
    inventory = build_b13_reconciliation_inventory(_three_window_result(tmp_path, values))
    first = assemble_b13_disposition_plan(review_b13_reconciliation(
        inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (ab, bc)),
    ))
    second = assemble_b13_disposition_plan(review_b13_reconciliation(
        inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (bc, ab)),
    ))
    assert first.plan.content_hash() == second.plan.content_hash()
    assert first.identity_bindings == second.identity_bindings
    assert len({item.adapted_id for item in first.identity_bindings if item.candidate_ref in {a, b, c}}) == 1


def test_equivalent_candidates_with_material_content_disagreement_fail_closed(tmp_path):
    source = _source_character(_result(tmp_path))
    preparation, _runtime = _two_window_preparation(tmp_path)
    first, second = (_proposal(packet) for packet in preparation.packets)
    first["adapted_characters"] = [_character("lead-a", source)]
    second["adapted_characters"] = [_character("lead-b", source)]
    decision = B13ReconciliationDecision(
        "identify_equivalent", (_character_ref("window_0001", "lead-a"), _character_ref("window_0002", "lead-b")),
        (), "确认身份。", "审阅。",
    )
    with pytest.raises(B13DispositionAssemblyError, match="content-selection"):
        assemble_b13_disposition_plan(_review(_result(tmp_path, first, second), (decision,)))


def test_complete_event_coverage_one_to_many_mapping_and_participants_materialize(tmp_path):
    source = _source_character(_result(tmp_path))
    preparation, _runtime = _two_window_preparation(tmp_path)
    first, second = (_proposal(packet) for packet in preparation.packets)
    first["adapted_characters"] = [_character("lead", source)]
    first["anonymous_roles"] = [{"local_key": "guard", "role_kind": "guard", "function_zh": "阻拦。"}]
    first["adapted_story_beats"][0]["participants"] = [
        {"kind": "adapted_character", "local_key": "lead"},
        {"kind": "anonymous_role", "local_key": "guard"},
    ]
    first["adapted_story_beats"].append({
        "local_key": "echo", "summary_zh": "事件后果再次显现。", "origin": "source_direct",
        "canon_compatibility": "compatible", "source_event_refs": [{"kind": "event", "canonical_id": "evt_000001"}],
        "participants": [], "canonical_anchor_refs": [], "invention_purpose": None,
        "invention_rationale_zh": None, "invention_provenance_zh": None,
    })
    first["adaptation_mappings"].append(_mapping("evt_000001", "echo"))
    plan = assemble_b13_disposition_plan(_review(_result(tmp_path, first, second))).plan
    assert len(plan.event_dispositions) == 2
    assert len([item for item in plan.adaptation_mappings if item.source_ref.canonical_id == "evt_000001" and item.active]) == 2
    assert len(plan.adapted_characters) == len(plan.anonymous_roles) == 1
    assert plan.anonymous_roles[0].anonymous_role_id in {
        participant.participant_id for beat in plan.adapted_story_beats for participant in beat.participants
    }


def test_overlay_candidates_translate_story_state_and_preserve_uniqueness(tmp_path):
    preparation, _runtime = _two_window_preparation(tmp_path)
    first, second = (_proposal(packet) for packet in preparation.packets)
    snapshot = preparation.source_snapshot
    fact = next(item for item in snapshot.canonical_references if item.kind == "fact")
    relationship = next(item for item in snapshot.canonical_references if item.kind == "relationship")
    first["continuity_overlay_candidates"] = {
        "factual_divergences": [{"source_fact_ref": fact.to_dict(), "adapted_assertion_zh": "事实改写。", "provenance_zh": "审阅。"}],
        "relationship_requirements": [],
        "story_state_requirements": [{"adapted_story_beat_local_key": "beat1", "assertion_zh": "状态成立。", "provenance_zh": "审阅。"}],
    }
    second["continuity_overlay_candidates"] = {
        "factual_divergences": [],
        "relationship_requirements": [{"source_relationship_ref": relationship.to_dict(), "requirement_zh": "关系升级。", "provenance_zh": "审阅。"}],
        "story_state_requirements": [],
    }
    overlay = assemble_b13_disposition_plan(_review(_result(tmp_path, first, second))).plan.continuity_overlay
    assert len(overlay.factual_divergences) == len(overlay.relationship_requirements) == len(overlay.story_state_requirements) == 1


def test_owner_omit_and_rejected_beat_dependencies_fail_closed_without_cascade(tmp_path):
    preparation, _runtime = _two_window_preparation(tmp_path, second_context_first=True)
    first, second = (_proposal(packet) for packet in preparation.packets)
    first["event_dispositions"][0]["disposition"] = "omit"
    first["adapted_story_beats"], first["adaptation_mappings"] = [], []
    second["anonymous_roles"] = [{"local_key": "guard", "role_kind": "guard", "function_zh": "旁观。"}]
    second["adapted_story_beats"].append({
        "local_key": "context", "summary_zh": "不该 materialize 的上下文。", "origin": "source_direct",
        "canon_compatibility": "uncertain", "source_event_refs": [{"kind": "event", "canonical_id": "evt_000001"}],
        "participants": [{"kind": "anonymous_role", "local_key": "guard"}], "canonical_anchor_refs": [], "invention_purpose": None,
        "invention_rationale_zh": None, "invention_provenance_zh": None,
    })
    second["adaptation_mappings"].append(_mapping("evt_000001", "context"))
    second["continuity_overlay_candidates"]["story_state_requirements"] = [
        {"adapted_story_beat_local_key": "context", "assertion_zh": "不应留下。", "provenance_zh": "审阅。"},
    ]
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first, second, second_context_first=True))
    finding = next(item for item in inventory.findings if item.kind.value == "owner_omit_context_membership")
    rejected_beat = next(item.ref for item in inventory.candidates if item.ref.window_id == "window_0002" and item.ref.local_key == "context")
    reject = B13ReconciliationDecision("reject_contribution", (rejected_beat,), (finding.finding_id,), "删除冲突节拍。", "审阅。")
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (reject,)))
    with pytest.raises(B13DispositionAssemblyError, match="depends on rejected"):
        assemble_b13_disposition_plan(review)


def test_forged_or_unresolved_review_never_assembles(tmp_path):
    review = _review(_result(tmp_path))
    object.__setattr__(review, "unresolved_finding_ids", ("f" * 64,))
    with pytest.raises(B13DispositionAssemblyError, match="review is stale"):
        assemble_b13_disposition_plan(review)
