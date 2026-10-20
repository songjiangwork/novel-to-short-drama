"""B1.3-B2B is entirely exercised with the injected Fake LLMClient."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from short_drama.adaptation import (
    B12StrategySelection, B13DispositionPacketMeasurement,
    B13DispositionPreparation, B13DispositionWindowPacket,
    B13WindowExecutionError,
    B13WindowGenerationExhaustedError, B13WindowProvenanceError,
    B13WindowSemanticBudget, B13WindowSemanticBudgetError,
    build_b13_window_semantic_preparation, execute_b13_disposition_windows,
    prepare_b13_disposition_windows,
)
from short_drama.adaptation.target import CanonicalTargetReference
from short_drama.artifacts import ImmutableArtifactEnvelope, canonical_json_bytes
from short_drama.foundation import PointerKind
from short_drama.llm import LLMTransportError
from short_drama.llm import GenerationExecutionOptions
from short_drama.story.chunking import estimate_tokens
from short_drama.story.story_analysis import StoryAnalysisManifest
from short_drama.story.story_analysis_persistence import a6_pointer_id

from test_adaptation_strategy import _b12_test_runtime_config
from test_adaptation_strategy_execution import _FakeClient, _move_current_to_payload
from test_b13_disposition_preparation import _published_strategy
from test_b13_window_semantic import _proposal


def _execution_inputs(tmp_path):
    tree, semantic, publication = _published_strategy(tmp_path)
    preparation = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
    )
    runtime = _b12_test_runtime_config()
    requests = tuple(
        build_b13_window_semantic_preparation(packet, runtime_config=runtime)
        for packet in preparation.packets
    )
    budgets = tuple(
        B13WindowSemanticBudget(runtime.request_model, 1_000_000, 1_000_000)
        for _ in requests
    )
    return tree, semantic, publication, preparation, requests, budgets, runtime


def _execute(tree, preparation, requests, budgets, runtime, client):
    return execute_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers, preparation=preparation,
        semantic_preparations=requests, budgets=budgets, runtime_config=runtime,
        llm_client=client,
    )


def _tree_bytes(root: Path):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


class _AttemptCountingFakeClient(_FakeClient):
    """Fake adapter results with explicit per-invocation retry counts."""

    def __init__(self, results, runtime, attempts):
        super().__init__(results, runtime)
        self._attempts = iter(attempts)

    def generate_structured(self, *args, **kwargs):
        return replace(super().generate_structured(*args, **kwargs), attempts=next(self._attempts))


def test_preflight_checks_both_mapped_attempt_budgets_before_any_call(tmp_path):
    tree, _semantic, _publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)
    insufficient = (replace(budgets[0], input_ceiling_tokens=1),)
    fake = _FakeClient([], runtime)
    with pytest.raises(B13WindowSemanticBudgetError, match="input ceiling"):
        _execute(tree, preparation, requests, insufficient, runtime, fake)
    assert fake.calls == []


def test_valid_execution_retains_exact_window_proposal_and_provenance_in_memory(tmp_path):
    tree, _semantic, _publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)
    fake = _FakeClient([(_proposal(requests[0].packet), False)], runtime)
    store_before = _tree_bytes(tree.store.root)
    pointers_before = _tree_bytes(tree.pointers.root)
    result = _execute(tree, preparation, requests, budgets, runtime, fake)
    accepted = result.accepted_windows[0]
    assert len(fake.calls) == result.semantic_call_count == 1
    assert accepted.request_identity == requests[0].request_identity
    assert accepted.semantic_request_hash == requests[0].request.request_hash
    assert accepted.proposal.to_dict() == _proposal(requests[0].packet)
    assert accepted.semantic_rounds == 1
    assert accepted.provider_attempts == 1
    assert accepted.accepted_round_provider_attempts == 1
    assert accepted.semantic_failures == ()
    assert accepted.rejected_semantic_rounds == ()
    assert _tree_bytes(tree.store.root) == store_before
    assert _tree_bytes(tree.pointers.root) == pointers_before


def test_schema_valid_semantic_invalidity_gets_one_cache_disabled_regeneration(tmp_path):
    tree, _semantic, _publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)
    invalid = _proposal(requests[0].packet)
    invalid["event_dispositions"][0]["disposition"] = "omit"
    fake = _FakeClient([(invalid, False), (_proposal(requests[0].packet), False)], runtime)
    result = _execute(tree, preparation, requests, budgets, runtime, fake)
    accepted = result.accepted_windows[0]
    assert len(fake.calls) == 2
    assert fake.calls[0] is None
    assert fake.calls[1].prompt_context_reuse == "disabled"
    assert accepted.semantic_rounds == 2
    assert accepted.provider_attempts == 2
    assert accepted.accepted_round_provider_attempts == 1
    assert len(accepted.semantic_failures) == 1
    assert accepted.rejected_semantic_rounds[0].provider_attempts == 1
    assert accepted.rejected_semantic_rounds[0].invocation_provenance.request_hash == requests[0].request.request_hash


def test_provider_attempts_accumulate_non_unit_adapter_attempts_across_regeneration(tmp_path):
    tree, _semantic, _publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)
    invalid = _proposal(requests[0].packet)
    invalid["event_dispositions"][0]["disposition"] = "omit"
    fake = _AttemptCountingFakeClient(
        [(invalid, False), (_proposal(requests[0].packet), False)], runtime, attempts=(3, 2),
    )
    accepted = _execute(tree, preparation, requests, budgets, runtime, fake).accepted_windows[0]
    assert accepted.semantic_rounds == 2
    assert accepted.provider_attempts == 5
    assert accepted.accepted_round_provider_attempts == 2
    assert accepted.rejected_semantic_rounds[0].provider_attempts == 3


def test_semantic_regeneration_exhaustion_does_not_return_partial_result(tmp_path):
    tree, _semantic, _publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)
    invalid = _proposal(requests[0].packet)
    invalid["event_dispositions"][0]["disposition"] = "omit"
    fake = _FakeClient([(invalid, False), (invalid, False)], runtime)
    with pytest.raises(B13WindowGenerationExhaustedError) as raised:
        _execute(tree, preparation, requests, budgets, runtime, fake)
    assert raised.value.rounds_attempted == 2
    assert len(fake.calls) == 2


def test_provenance_mismatch_fails_closed_without_regeneration(tmp_path):
    tree, _semantic, _publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)
    fake = _FakeClient([(_proposal(requests[0].packet), True)], runtime)
    with pytest.raises(B13WindowProvenanceError):
        _execute(tree, preparation, requests, budgets, runtime, fake)
    assert len(fake.calls) == 1


def test_transport_failure_is_adapter_owned_and_does_not_consume_semantic_round(tmp_path):
    tree, _semantic, _publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)
    fake = _FakeClient([LLMTransportError("offline")], runtime)
    with pytest.raises(LLMTransportError):
        _execute(tree, preparation, requests, budgets, runtime, fake)
    assert len(fake.calls) == 1


def test_b12_current_change_after_response_fails_before_proposal_is_retained(tmp_path):
    tree, semantic, publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)

    class MovingClient(_FakeClient):
        def generate_structured(self, *args, **kwargs):
            response = super().generate_structured(*args, **kwargs)
            _move_current_to_payload(tree, semantic, publication, publication.artifact.to_dict())
            return response

    fake = MovingClient([(_proposal(requests[0].packet), False)], runtime)
    with pytest.raises(B13WindowExecutionError, match="B1.2 CURRENT"):
        _execute(tree, preparation, requests, budgets, runtime, fake)
    assert len(fake.calls) == 1


def test_source_current_change_after_response_fails_before_proposal_is_retained(tmp_path):
    tree, semantic, _publication, preparation, requests, budgets, runtime = _execution_inputs(tmp_path)
    source = semantic.preparation.packet.source_snapshot
    manifest = StoryAnalysisManifest.from_dict(tree.store.get_ref(source.story_analysis_manifest_ref).payload)

    class MovingClient(_FakeClient):
        def generate_structured(self, *args, **kwargs):
            response = super().generate_structured(*args, **kwargs)
            replacement = tree.store.put(ImmutableArtifactEnvelope.create(
                artifact_type=source.story_analysis_manifest_ref.artifact_type,
                artifact_id=source.story_analysis_manifest_ref.artifact_id,
                revision=source.story_analysis_manifest_ref.revision + 1,
                schema_version=1, payload=manifest.to_dict(),
            ))
            pointer_id = a6_pointer_id(
                manifest.project_id, manifest.document_id,
                manifest.semantic_identity.story_analysis_profile_id,
            )
            old = tree.pointers.resolve_current_pointer_ref(pointer_id)
            tree.pointers.compare_and_set(
                pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=old, target_ref=replacement,
            )
            return response

    fake = MovingClient([(_proposal(requests[0].packet), False)], runtime)
    with pytest.raises(B13WindowExecutionError, match="A5/A6"):
        _execute(tree, preparation, requests, budgets, runtime, fake)
    assert len(fake.calls) == 1


def test_explicit_historical_b12_pin_does_not_require_it_to_remain_current(tmp_path):
    tree, semantic, publication, _preparation, _requests, _budgets, runtime = _execution_inputs(tmp_path)
    historical = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
        strategy_ref=publication.strategy_ref,
    )
    requests = tuple(build_b13_window_semantic_preparation(x, runtime_config=runtime) for x in historical.packets)
    budgets = tuple(B13WindowSemanticBudget(runtime.request_model, 1_000_000, 1_000_000) for _ in requests)
    # Advance B1.2 CURRENT externally.  The explicit historical selection has
    # no pointer pin and must preserve its deliberately chosen revision.
    _move_current_to_payload(tree, semantic, publication, publication.artifact.to_dict())
    fake = _FakeClient([(_proposal(requests[0].packet), False)], runtime)
    result = _execute(tree, historical, requests, budgets, runtime, fake)
    assert result.strategy_selection.strategy_ref == publication.strategy_ref
    assert result.strategy_selection.current_pointer_ref is None


def _two_window_inputs(tmp_path):
    """Synthetic B2B-only two-window fixture; B1/B2A coverage is tested elsewhere."""
    tree, _semantic, _publication, preparation, _requests, _budgets, runtime = _execution_inputs(tmp_path)
    base_packet = preparation.packets[0]
    source = replace(
        base_packet.source_snapshot,
        canonical_references=(*base_packet.source_snapshot.canonical_references,
                              CanonicalTargetReference("event", "evt_000002")),
    )
    strategy = replace(
        base_packet.strategy_selection.artifact.strategy,
        resolved_target=replace(
            base_packet.strategy_selection.artifact.strategy.resolved_target,
            source_snapshot=source,
        ),
    )
    selection = B12StrategySelection(
        base_packet.strategy_selection.strategy_ref,
        replace(base_packet.strategy_selection.artifact, source_snapshot=source, strategy=strategy),
        base_packet.strategy_selection.current_pointer_ref,
    )

    def packet_for(event_id, ordinal):
        value = base_packet.to_dict()
        value["source_snapshot"] = source.to_dict()
        value["global_adaptation_strategy"]["strategy"] = strategy.to_dict()
        value["global_adaptation_strategy"]["strategy_hash"] = strategy.content_hash()
        value["window"] = {
            "window_id": f"window_{ordinal:04d}", "window_ordinal": ordinal,
            "owned_event_refs": [event_id], "context_event_refs": [],
        }
        value["owned_source_events"] = [{"event_id": event_id, "narrative_order": ordinal}]
        value["context_source_events"] = []
        return B13DispositionWindowPacket(
            source, selection, f"window_{ordinal:04d}", ordinal,
            canonical_json_bytes(value),
        )

    first, second = packet_for("evt_000001", 1), packet_for("evt_000002", 2)
    packets = (first, second)
    two = B13DispositionPreparation(
        source, selection, packets,
        tuple(B13DispositionPacketMeasurement(
            packet.window_id, len(packet.canonical_bytes()), 1, packet.content_hash(),
        ) for packet in packets),
        ("evt_000001", "evt_000002"), preparation.preparation_identity,
    )
    requests = tuple(build_b13_window_semantic_preparation(packet, runtime_config=runtime) for packet in packets)
    budgets = tuple(B13WindowSemanticBudget(runtime.request_model, 1_000_000, 1_000_000) for _ in packets)
    return tree, two, requests, budgets, runtime, first, second


def _allow_synthetic_two_window_preflight(monkeypatch, two):
    # Source/strategy races and full A6 ownership have dedicated real-store
    # tests above.  This fixture isolates only the two-window execution order.
    import short_drama.adaptation.disposition_execution as execution_module
    monkeypatch.setattr(execution_module, "prepare_b13_disposition_windows", lambda **_kwargs: two)
    monkeypatch.setattr(execution_module, "_revalidate_authority", lambda *_args: None)


def test_multiple_in_memory_windows_execute_in_deterministic_ordinal_order(tmp_path, monkeypatch):
    """The executor serializes all preflight-approved windows by ordinal."""
    tree, two, requests, budgets, runtime, first, second = _two_window_inputs(tmp_path)
    _allow_synthetic_two_window_preflight(monkeypatch, two)
    fake = _FakeClient([(_proposal(first), False), (_proposal(second), False)], runtime)
    result = _execute(tree, two, requests, budgets, runtime, fake)
    assert [item.window_id for item in result.accepted_windows] == ["window_0001", "window_0002"]
    assert len(fake.calls) == 2


def test_second_window_retry_budget_failure_stops_all_provider_calls(tmp_path, monkeypatch):
    tree, two, requests, budgets, runtime, _first, _second = _two_window_inputs(tmp_path)
    _allow_synthetic_two_window_preflight(monkeypatch, two)
    from short_drama.adaptation.disposition_semantic import _mapped_provider_request_bytes

    retry_options = GenerationExecutionOptions(prompt_context_reuse="disabled")
    retry_tokens = estimate_tokens(_mapped_provider_request_bytes(
        requests[1].request, runtime, retry_options,
    ).decode("utf-8"))
    assert retry_tokens > requests[1].measurement.provider_request_estimated_tokens
    insufficient_second = replace(budgets[1], input_ceiling_tokens=retry_tokens - 1)
    fake = _FakeClient([], runtime)
    with pytest.raises(B13WindowSemanticBudgetError, match="input ceiling"):
        _execute(tree, two, requests, (budgets[0], insufficient_second), runtime, fake)
    assert fake.calls == []
