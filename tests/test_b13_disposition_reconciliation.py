"""B1.3-B3A reconciliation inventory: synthetic, zero-provider only."""
from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.adaptation import (
    B12StrategySelection, B13CandidateKind, B13CandidateRef, B13DispositionExecutionResult,
    B13DispositionPacketMeasurement, B13DispositionPreparation,
    B13DispositionWindowPacket,
    B13ReconciliationDecision, B13ReconciliationDecisionKind,
    B13ReconciliationDecisionSet, B13ReconciliationError,
    B13ReconciliationFindingKind, B13ReconciliationReview,
    B13WindowAcceptedProposal,
    build_b13_reconciliation_inventory, build_b13_window_semantic_preparation,
    review_b13_reconciliation,
)
from short_drama.adaptation.target import CanonicalTargetReference
from short_drama.artifacts import canonical_json_bytes, content_hash
from short_drama.llm import LLMInvocationProvenance

from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_execution import _two_window_inputs
from test_b13_window_semantic import _mapping, _proposal


def _preparation_identity(preparation):
    return content_hash({
        "stage": "B1.3-B1", "source_snapshot": preparation.source_snapshot.to_dict(),
        "strategy_ref": preparation.strategy_selection.strategy_ref.to_dict(),
        "strategy_hash": preparation.strategy_selection.strategy_hash,
        "packet_hashes": [packet.content_hash() for packet in preparation.packets],
    })


def _two_window_preparation(tmp_path, *, second_context_first=False):
    _tree, original, _requests, _budgets, runtime, first, second = _two_window_inputs(tmp_path)
    if second_context_first:
        payload = second.to_dict()
        payload["window"]["context_event_refs"] = ["evt_000001"]
        payload["context_source_events"] = [{"event_id": "evt_000001", "narrative_order": 1}]
        second = B13DispositionWindowPacket(
            second.source_snapshot, second.strategy_selection, second.window_id,
            second.window_ordinal, canonical_json_bytes(payload),
        )
    packets = (first, second)
    preparation = B13DispositionPreparation(
        original.source_snapshot, original.strategy_selection, packets,
        tuple(B13DispositionPacketMeasurement(packet.window_id, len(packet.canonical_bytes()), 1, packet.content_hash()) for packet in packets),
        ("evt_000001", "evt_000002"), "0" * 64,
    )
    return replace(preparation, preparation_identity=_preparation_identity(preparation)), runtime


def _accepted(request, value):
    proposal = request.decode_and_validate_proposal(value)
    provenance = LLMInvocationProvenance(
        provider_family=request.runtime_config.provider_family,
        model=request.runtime_config.request_model,
        semantic_profile_id=request.semantic_profile.profile_id,
        semantic_profile_hash=request.semantic_profile.semantic_profile_hash,
        prompt_id=request.request.rendered_prompt.prompt_id,
        prompt_version=request.request.rendered_prompt.prompt_version,
        prompt_content_hash=request.request.rendered_prompt.prompt_content_hash,
        rendered_prompt_hash=request.request.rendered_prompt.rendered_prompt_hash,
        output_schema_id=request.output_schema.schema_id,
        output_schema_version=request.output_schema.schema_version,
        output_schema_hash=request.output_schema.schema_hash,
        request_hash=request.request.request_hash,
        provider_response_id=None, finish_reason="stop", usage=None,
    )
    return B13WindowAcceptedProposal(
        window_id=request.packet.window_id, window_ordinal=request.packet.window_ordinal,
        request_identity=request.request_identity, semantic_request_hash=request.request.request_hash,
        semantic_profile_identity={"profile_id": request.semantic_profile.profile_id, "profile_hash": request.semantic_profile.semantic_profile_hash},
        prompt_identity={"prompt_id": request.request.rendered_prompt.prompt_id, "prompt_version": request.request.rendered_prompt.prompt_version, "prompt_content_hash": request.request.rendered_prompt.prompt_content_hash},
        output_schema_identity={"schema_id": request.output_schema.schema_id, "schema_version": request.output_schema.schema_version, "schema_hash": request.output_schema.schema_hash},
        rendered_prompt_hash=request.request.rendered_prompt.rendered_prompt_hash,
        invocation_provenance=provenance, semantic_rounds=1, provider_attempts=1,
        accepted_round_provider_attempts=1, semantic_failures=(), rejected_semantic_rounds=(),
        proposal=proposal,
    )


def _result(tmp_path, first_value=None, second_value=None, *, second_context_first=False):
    preparation, runtime = _two_window_preparation(tmp_path, second_context_first=second_context_first)
    requests = tuple(build_b13_window_semantic_preparation(packet, runtime_config=runtime) for packet in preparation.packets)
    first_value = _proposal(requests[0].packet) if first_value is None else first_value
    second_value = _proposal(requests[1].packet) if second_value is None else second_value
    return B13DispositionExecutionResult(
        preparation.source_snapshot, preparation.strategy_selection, preparation.preparation_identity,
        (_accepted(requests[0], first_value), _accepted(requests[1], second_value)),
    )


def _three_window_preparation(tmp_path):
    _tree, original, _requests, _budgets, runtime, first, _second = _two_window_inputs(tmp_path)
    source = replace(
        first.source_snapshot,
        canonical_references=(*first.source_snapshot.canonical_references, CanonicalTargetReference("event", "evt_000003")),
    )
    strategy = replace(
        first.strategy_selection.artifact.strategy,
        resolved_target=replace(first.strategy_selection.artifact.strategy.resolved_target, source_snapshot=source),
    )
    selection = B12StrategySelection(
        first.strategy_selection.strategy_ref,
        replace(first.strategy_selection.artifact, source_snapshot=source, strategy=strategy),
        first.strategy_selection.current_pointer_ref,
    )

    def packet_for(event_id, ordinal):
        payload = first.to_dict()
        payload["source_snapshot"] = source.to_dict()
        payload["global_adaptation_strategy"]["strategy"] = strategy.to_dict()
        payload["global_adaptation_strategy"]["strategy_hash"] = strategy.content_hash()
        payload["window"] = {"window_id": f"window_{ordinal:04d}", "window_ordinal": ordinal, "owned_event_refs": [event_id], "context_event_refs": []}
        payload["owned_source_events"] = [{"event_id": event_id, "narrative_order": ordinal}]
        payload["context_source_events"] = []
        return B13DispositionWindowPacket(source, selection, f"window_{ordinal:04d}", ordinal, canonical_json_bytes(payload))

    packets = tuple(packet_for(f"evt_{ordinal:06d}", ordinal) for ordinal in range(1, 4))
    preparation = B13DispositionPreparation(
        source, selection, packets,
        tuple(B13DispositionPacketMeasurement(packet.window_id, len(packet.canonical_bytes()), 1, packet.content_hash()) for packet in packets),
        ("evt_000001", "evt_000002", "evt_000003"), "0" * 64,
    )
    return replace(preparation, preparation_identity=_preparation_identity(preparation)), runtime


def _three_window_result(tmp_path, values):
    preparation, runtime = _three_window_preparation(tmp_path)
    requests = tuple(build_b13_window_semantic_preparation(packet, runtime_config=runtime) for packet in preparation.packets)
    return B13DispositionExecutionResult(
        preparation.source_snapshot, preparation.strategy_selection, preparation.preparation_identity,
        tuple(_accepted(request, value) for request, value in zip(requests, values)),
    )


def _source_character(result):
    return next(item for item in result.source_snapshot.canonical_references if item.kind == "character")


def _character(local_key, source_ref):
    return {
        "local_key": local_key, "display_name_zh": f"人物-{local_key}", "origin": "source_single",
        "function_provenance": [{"source_character_ref": source_ref.to_dict(), "function_zh": "承担行动功能。", "provenance_zh": "窗口候选的功能证据。"}],
        "invention_rationale_zh": None,
    }


def test_scoped_refs_preserve_identical_local_keys_in_distinct_windows(tmp_path):
    inventory = build_b13_reconciliation_inventory(_result(tmp_path))
    beats = [item.ref for item in inventory.candidates if item.ref.candidate_kind is B13CandidateKind.ADAPTED_STORY_BEAT]
    assert [(item.window_id, item.local_key) for item in beats] == [("window_0001", "beat1"), ("window_0002", "beat1")]
    assert len(set(beats)) == 2


def test_explicitly_equivalent_differently_named_local_candidates_need_no_text_heuristic(tmp_path):
    provisional = _result(tmp_path)
    character = _source_character(provisional)
    preparation, runtime = _two_window_preparation(tmp_path)
    first, second = (_proposal(packet) for packet in preparation.packets)
    first["adapted_characters"] = [_character("lead-a", character)]
    second["adapted_characters"] = [_character("lead-b", character)]
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first, second))
    a = B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_CHARACTER, "lead-a")
    b = B13CandidateRef("window_0002", B13CandidateKind.ADAPTED_CHARACTER, "lead-b")
    decision = B13ReconciliationDecision(B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT, (a, b), (), "人工确认同一改编角色。", "跨窗口审阅记录。")
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (decision,)))
    assert review.is_resolved


def test_one_source_to_many_beat_mappings_remains_lawful_and_distinct(tmp_path):
    preparation, _runtime = _two_window_preparation(tmp_path)
    first = _proposal(preparation.packets[0])
    event_id = "evt_000001"
    first["adapted_story_beats"].append({
        "local_key": "echo", "summary_zh": "同一事件的另一条改编后果。", "origin": "source_direct",
        "canon_compatibility": "compatible", "source_event_refs": [{"kind": "event", "canonical_id": event_id}],
        "participants": [], "canonical_anchor_refs": [], "invention_purpose": None,
        "invention_rationale_zh": None, "invention_provenance_zh": None,
    })
    first["adaptation_mappings"].append(_mapping(event_id, "echo"))
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first))
    beats = [item for item in inventory.candidates if item.ref.window_id == "window_0001" and item.ref.candidate_kind is B13CandidateKind.ADAPTED_STORY_BEAT]
    assert {item.ref.local_key for item in beats} == {"beat1", "echo"}
    assert not inventory.findings


def test_owner_omit_conflicts_with_contextual_beat_membership_and_stays_unresolved(tmp_path):
    preparation, _runtime = _two_window_preparation(tmp_path, second_context_first=True)
    first = _proposal(preparation.packets[0])
    first["event_dispositions"][0]["disposition"] = "omit"
    first["adapted_story_beats"] = []
    first["adaptation_mappings"] = []
    second = _proposal(preparation.packets[1])
    second["adapted_story_beats"].append({
        "local_key": "context-beat", "summary_zh": "跨窗口上下文候选。", "origin": "source_direct",
        "canon_compatibility": "uncertain", "source_event_refs": [{"kind": "event", "canonical_id": "evt_000001"}],
        "participants": [], "canonical_anchor_refs": [], "invention_purpose": None,
        "invention_rationale_zh": None, "invention_provenance_zh": None,
    })
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first, second, second_context_first=True))
    finding = next(item for item in inventory.findings if item.kind is B13ReconciliationFindingKind.OWNER_OMIT_CONTEXT_MEMBERSHIP)
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    assert finding.finding_id in review.unresolved_finding_ids
    with pytest.raises(B13ReconciliationError, match="unresolved"):
        review.require_resolved()
    marked = B13ReconciliationDecision(
        B13ReconciliationDecisionKind.MARK_UNRESOLVED, (), (finding.finding_id,),
        "等待语义复审。", "跨窗口审阅记录。",
    )
    assert finding.finding_id in review_b13_reconciliation(
        inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (marked,))
    ).unresolved_finding_ids


def test_contradictory_equivalence_decisions_fail_closed(tmp_path):
    result = _result(tmp_path)
    source = _source_character(result)
    preparation, _runtime = _two_window_preparation(tmp_path)
    first, second = (_proposal(packet) for packet in preparation.packets)
    first["adapted_characters"] = [_character("lead-a", source)]
    second["adapted_characters"] = [_character("lead-b", source)]
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first, second))
    refs = (B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_CHARACTER, "lead-a"), B13CandidateRef("window_0002", B13CandidateKind.ADAPTED_CHARACTER, "lead-b"))
    same = B13ReconciliationDecision(B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT, refs, (), "确认相同。", "审阅。")
    distinct = B13ReconciliationDecision(B13ReconciliationDecisionKind.KEEP_DISTINCT, refs, (), "确认不同。", "审阅。")
    with pytest.raises(B13ReconciliationError, match="contradictory"):
        review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (same, distinct)))


def test_unknown_duplicate_and_cross_inventory_decisions_fail_closed(tmp_path):
    inventory = build_b13_reconciliation_inventory(_result(tmp_path))
    finding = next(iter(inventory.findings), None)
    unknown = B13CandidateRef("window_9999", B13CandidateKind.ADAPTED_STORY_BEAT, "unknown")
    decision = B13ReconciliationDecision(B13ReconciliationDecisionKind.KEEP_DISTINCT, (unknown, B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_STORY_BEAT, "beat1")), (), "不能自动合并。", "审阅。")
    with pytest.raises(B13ReconciliationError, match="unknown candidate"):
        review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (decision,)))
    known = B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_STORY_BEAT, "beat1")
    with pytest.raises(B13ReconciliationError, match="exactly two"):
        B13ReconciliationDecision(B13ReconciliationDecisionKind.KEEP_DISTINCT, (known,), (), "候选不足。", "审阅。")
    with pytest.raises(B13ReconciliationError, match="unique"):
        B13ReconciliationDecision(B13ReconciliationDecisionKind.KEEP_DISTINCT, (known, known), (), "重复引用。", "审阅。")
    with pytest.raises(B13ReconciliationError, match="duplicate"):
        B13ReconciliationDecisionSet(inventory.inventory_identity, (decision, decision))
    with pytest.raises(B13ReconciliationError, match="stale"):
        review_b13_reconciliation(inventory, B13ReconciliationDecisionSet("f" * 64, ()))
    assert finding is None  # Baseline has no forced conflict; unknown refs still reject.


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "reordered"))
def test_missing_duplicated_or_reordered_accepted_windows_fail_closed(tmp_path, mutation):
    result = _result(tmp_path)
    if mutation == "missing":
        object.__setattr__(result, "accepted_windows", result.accepted_windows[:1])
    elif mutation == "duplicate":
        object.__setattr__(result, "accepted_windows", (result.accepted_windows[0], result.accepted_windows[0]))
    else:
        object.__setattr__(result, "accepted_windows", tuple(reversed(result.accepted_windows)))
    with pytest.raises(B13ReconciliationError):
        build_b13_reconciliation_inventory(result)


def test_proposal_or_semantic_request_change_changes_or_invalidates_inventory_identity(tmp_path):
    baseline = build_b13_reconciliation_inventory(_result(tmp_path))
    preparation, _runtime = _two_window_preparation(tmp_path)
    changed = _proposal(preparation.packets[0])
    changed["adapted_story_beats"][0]["summary_zh"] = "改动后的候选节拍。"
    proposal_changed = build_b13_reconciliation_inventory(_result(tmp_path, changed))
    assert proposal_changed.inventory_identity != baseline.inventory_identity
    result = _result(tmp_path)
    changed_request = replace(result.accepted_windows[0], request_identity="b" * 64)
    request_changed = replace(result, accepted_windows=(changed_request, result.accepted_windows[1]))
    with pytest.raises(B13ReconciliationError, match="request identity"):
        build_b13_reconciliation_inventory(request_changed)


def test_inventory_ordering_and_hashes_are_stable_across_repeated_runs(tmp_path):
    first = build_b13_reconciliation_inventory(_result(tmp_path / "one"))
    second = build_b13_reconciliation_inventory(_result(tmp_path / "two"))
    assert first.inventory_identity == second.inventory_identity
    assert first.to_dict() == second.to_dict()
    assert [item.ref for item in first.candidates] == sorted(item.ref for item in first.candidates)


def test_inventory_identity_rejects_erased_or_modified_derived_evidence(tmp_path):
    preparation, _runtime = _two_window_preparation(tmp_path, second_context_first=True)
    first = _proposal(preparation.packets[0])
    first["event_dispositions"][0]["disposition"] = "omit"
    first["adapted_story_beats"], first["adaptation_mappings"] = [], []
    second = _proposal(preparation.packets[1])
    second["adapted_story_beats"].append({
        "local_key": "context-beat", "summary_zh": "上下文候选。", "origin": "source_direct",
        "canon_compatibility": "uncertain", "source_event_refs": [{"kind": "event", "canonical_id": "evt_000001"}],
        "participants": [], "canonical_anchor_refs": [], "invention_purpose": None,
        "invention_rationale_zh": None, "invention_provenance_zh": None,
    })
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first, second, second_context_first=True))
    with pytest.raises(B13ReconciliationError, match="identity|findings"):
        replace(inventory, findings=())
    changed = next(item for item in inventory.candidates if item.ref.local_key == "context-beat")
    altered_payload = changed.payload()
    altered_payload["summary_zh"] = "被篡改的候选。"
    altered = replace(changed, _canonical_payload=canonical_json_bytes(altered_payload))
    with pytest.raises(B13ReconciliationError, match="candidates|identity"):
        replace(inventory, candidates=tuple(altered if item == changed else item for item in inventory.candidates))
    with pytest.raises(B13ReconciliationError, match="findings|identity"):
        replace(inventory, findings=(replace(inventory.findings[0], detail="被篡改的 finding。"),))


def test_three_way_overlay_conflicts_require_all_but_one_contribution_rejected(tmp_path):
    preparation, _runtime = _three_window_preparation(tmp_path)
    fact = next(item for item in preparation.source_snapshot.canonical_references if item.kind == "fact")
    relationship = next(item for item in preparation.source_snapshot.canonical_references if item.kind == "relationship")
    values = []
    for ordinal, packet in enumerate(preparation.packets, 1):
        value = _proposal(packet)
        value["continuity_overlay_candidates"] = {
            "factual_divergences": [{"source_fact_ref": fact.to_dict(), "adapted_assertion_zh": f"事实偏离候选 {ordinal}。", "provenance_zh": "跨窗口候选。"}],
            "relationship_requirements": [{"source_relationship_ref": relationship.to_dict(), "requirement_zh": f"关系要求候选 {ordinal}。", "provenance_zh": "跨窗口候选。"}],
            "story_state_requirements": [],
        }
        values.append(value)
    inventory = build_b13_reconciliation_inventory(_three_window_result(tmp_path, values))
    findings = {item.kind: item for item in inventory.findings}
    fact_finding = findings[B13ReconciliationFindingKind.COMPETING_FACTUAL_DIVERGENCES]
    relationship_finding = findings[B13ReconciliationFindingKind.COMPETING_RELATIONSHIP_REQUIREMENTS]

    def reject(ref, finding):
        return B13ReconciliationDecision(B13ReconciliationDecisionKind.REJECT_CONTRIBUTION, (ref,), (finding.finding_id,), "拒绝该冲突贡献。", "跨窗口审阅。")

    one_each = (reject(fact_finding.candidate_refs[0], fact_finding), reject(relationship_finding.candidate_refs[0], relationship_finding))
    partial = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, one_each))
    assert {fact_finding.finding_id, relationship_finding.finding_id} <= set(partial.unresolved_finding_ids)

    two_each = (*one_each, reject(fact_finding.candidate_refs[1], fact_finding), reject(relationship_finding.candidate_refs[1], relationship_finding))
    assert review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, two_each)).is_resolved


def test_transitive_equivalence_cannot_contradict_keep_distinct(tmp_path):
    result = _result(tmp_path)
    source = _source_character(result)
    preparation, _runtime = _two_window_preparation(tmp_path)
    first, second = (_proposal(packet) for packet in preparation.packets)
    first["adapted_characters"] = [_character("lead-a", source), _character("lead-c", source)]
    second["adapted_characters"] = [_character("lead-b", source)]
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first, second))
    a = B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_CHARACTER, "lead-a")
    b = B13CandidateRef("window_0002", B13CandidateKind.ADAPTED_CHARACTER, "lead-b")
    c = B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_CHARACTER, "lead-c")
    same_ab = B13ReconciliationDecision(B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT, (a, b), (), "确认相同。", "审阅。")
    same_bc = B13ReconciliationDecision(B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT, (b, c), (), "确认相同。", "审阅。")
    distinct_ac = B13ReconciliationDecision(B13ReconciliationDecisionKind.KEEP_DISTINCT, (a, c), (), "确认不同。", "审阅。")
    with pytest.raises(B13ReconciliationError, match="transitively"):
        review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (same_ab, same_bc, distinct_ac)))


def test_incompatible_explicit_equivalence_fails_immediately_without_dynamic_finding(tmp_path):
    result = _result(tmp_path)
    source = _source_character(result)
    preparation, _runtime = _two_window_preparation(tmp_path)
    first, second = (_proposal(packet) for packet in preparation.packets)
    first["adapted_characters"] = [_character("lead-a", source)]
    second["adapted_characters"] = [_character("lead-b", source)]
    second["adapted_characters"][0]["function_provenance"][0]["function_zh"] = "不兼容的另一项功能。"
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first, second))
    decision = B13ReconciliationDecision(
        B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT,
        (B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_CHARACTER, "lead-a"), B13CandidateRef("window_0002", B13CandidateKind.ADAPTED_CHARACTER, "lead-b")),
        (), "错误地要求等同。", "审阅。",
    )
    with pytest.raises(B13ReconciliationError, match="incompatible"):
        review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (decision,)))


def test_review_rederives_integrity_before_resolved_handoff(tmp_path):
    preparation, _runtime = _two_window_preparation(tmp_path, second_context_first=True)
    first = _proposal(preparation.packets[0])
    first["event_dispositions"][0]["disposition"] = "omit"
    first["adapted_story_beats"], first["adaptation_mappings"] = [], []
    second = _proposal(preparation.packets[1])
    second["adapted_story_beats"].append({
        "local_key": "context-beat", "summary_zh": "跨窗口上下文候选。", "origin": "source_direct",
        "canon_compatibility": "uncertain", "source_event_refs": [{"kind": "event", "canonical_id": "evt_000001"}],
        "participants": [], "canonical_anchor_refs": [], "invention_purpose": None,
        "invention_rationale_zh": None, "invention_provenance_zh": None,
    })
    inventory = build_b13_reconciliation_inventory(_result(tmp_path, first, second, second_context_first=True))
    decisions = B13ReconciliationDecisionSet(inventory.inventory_identity, ())
    review = review_b13_reconciliation(inventory, decisions)
    assert not review.is_resolved
    with pytest.raises(B13ReconciliationError, match="unresolved finding IDs"):
        replace(review, unresolved_finding_ids=())
    with pytest.raises(B13ReconciliationError, match="findings"):
        replace(review, findings=())
    with pytest.raises(B13ReconciliationError, match="findings"):
        B13ReconciliationReview(inventory, decisions, (), ())
    object.__setattr__(review, "unresolved_finding_ids", ())
    with pytest.raises(B13ReconciliationError, match="unresolved finding IDs"):
        review.require_resolved()
    with pytest.raises(B13ReconciliationError, match="stale"):
        B13ReconciliationReview(
            inventory, B13ReconciliationDecisionSet("f" * 64, ()),
            inventory.findings, review_b13_reconciliation(inventory, decisions).unresolved_finding_ids,
        )


def test_valid_resolved_review_remains_usable(tmp_path):
    inventory = build_b13_reconciliation_inventory(_result(tmp_path))
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    assert review.is_resolved
    review.require_resolved()
