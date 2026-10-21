"""B1.3-B4A publication evidence is synthetic and strictly zero-provider."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from short_drama.adaptation import (
    B12StrategySelection, B13DispositionExecutionResult, B13DispositionPublication,
    B13DispositionPublicationError, B13DispositionWindowPacket,
    B13ReconciliationDecisionSet, B13WindowRejectedSemanticRound,
    assemble_b13_disposition_plan, build_b13_reconciliation_inventory,
    build_b13_window_semantic_preparation, prepare_b13_disposition_publication,
    review_b13_reconciliation, verify_b13_disposition_publication,
)
from short_drama.artifacts import ArtifactRef, artifact_content_hash, canonical_json_bytes, content_hash
from short_drama.llm import LLMClient
from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_assembly import _result
from test_b13_disposition_reconciliation import _accepted
from short_drama.adaptation.disposition_publication import _publication_identity


def _persistable_result(tmp_path):
    """Give the B2B synthetic fixture a real immutable B1.2 content ref."""
    original = _result(tmp_path)
    artifact = original.strategy_selection.artifact
    ref = ArtifactRef(
        original.strategy_selection.strategy_ref.artifact_type,
        original.strategy_selection.strategy_ref.artifact_id,
        original.strategy_selection.strategy_ref.revision,
        artifact_content_hash(schema_version=1, payload=artifact.to_dict()),
    )
    selection = B12StrategySelection(ref, artifact, original.strategy_selection.current_pointer_ref)
    runtime = _b12_test_runtime_config()
    accepted = []
    for old in original.accepted_windows:
        packet_dict = old.proposal.packet.to_dict()
        packet_dict["global_adaptation_strategy"]["strategy_ref"] = ref.to_dict()
        packet = B13DispositionWindowPacket(
            original.source_snapshot, selection, old.window_id, old.window_ordinal,
            canonical_json_bytes(packet_dict),
        )
        semantic = build_b13_window_semantic_preparation(packet, runtime_config=runtime)
        accepted.append(_accepted(semantic, old.proposal.to_dict()))
    preparation_identity = content_hash({
        "stage": "B1.3-B1", "source_snapshot": original.source_snapshot.to_dict(),
        "strategy_ref": ref.to_dict(), "strategy_hash": selection.strategy_hash,
        "packet_hashes": [item.proposal.packet.content_hash() for item in accepted],
    })
    return B13DispositionExecutionResult(
        original.source_snapshot, selection, preparation_identity, tuple(accepted),
    )


def _publication(tmp_path):
    execution = _persistable_result(tmp_path)
    inventory = build_b13_reconciliation_inventory(execution)
    # A decision set with no findings is the exact resolved B3A review here.
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    assembly = assemble_b13_disposition_plan(review)
    return prepare_b13_disposition_publication(review, assembly, execution), review, assembly, execution


def _review_and_assembly(execution):
    inventory = build_b13_reconciliation_inventory(execution)
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    return review, assemble_b13_disposition_plan(review)


def _two_round_result(tmp_path):
    """One bounded semantic rejection followed by a valid accepted proposal."""
    execution = _persistable_result(tmp_path)
    accepted = execution.accepted_windows[0]
    rejected = B13WindowRejectedSemanticRound(
        semantic_round=1, provider_attempts=1,
        invocation_provenance=accepted.invocation_provenance,
        diagnostic="schema 有效但语义不满足约束。",
    )
    retried = replace(
        accepted, semantic_rounds=2, provider_attempts=2,
        accepted_round_provider_attempts=1,
        semantic_failures=(rejected.diagnostic,), rejected_semantic_rounds=(rejected,),
    )
    return replace(execution, accepted_windows=(retried, *execution.accepted_windows[1:]))


def _two_round_publication(tmp_path):
    execution = _two_round_result(tmp_path)
    review, assembly = _review_and_assembly(execution)
    return prepare_b13_disposition_publication(review, assembly, execution), review, assembly, execution


def _reidentified(payload):
    payload["publication_identity"] = _publication_identity(payload)
    return payload


def test_prepare_and_verify_two_window_publication_is_deterministic_and_zero_write(tmp_path, monkeypatch):
    execution = _persistable_result(tmp_path)
    inventory = build_b13_reconciliation_inventory(execution)
    review = review_b13_reconciliation(inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()))
    assembly = assemble_b13_disposition_plan(review)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    monkeypatch.setattr(LLMClient, "generate_structured", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM call")))
    publication = prepare_b13_disposition_publication(review, assembly, execution)
    again = prepare_b13_disposition_publication(review, assembly, execution)
    assert publication.canonical_bytes() == again.canonical_bytes()
    assert publication.publication_identity == again.publication_identity
    assert verify_b13_disposition_publication(publication) == assembly
    assert B13DispositionPublication.from_dict(publication.to_dict()).canonical_bytes() == publication.canonical_bytes()
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_two_round_publication_reconstructs_with_consistent_provenance(tmp_path):
    publication, _review, assembly, _execution = _two_round_publication(tmp_path)
    assert publication.to_dict()["evidence"]["accepted_windows"][0]["semantic_rounds"] == 2
    assert verify_b13_disposition_publication(publication) == assembly


@pytest.mark.parametrize(("field", "value", "match"), [
    ("request_hash", "f" * 64, "rejected semantic round.*exact reconstructed request"),
    ("output_schema_hash", "e" * 64, "rejected semantic round.*exact reconstructed request"),
    ("prompt_content_hash", "d" * 64, "rejected semantic round.*exact reconstructed request"),
    ("semantic_profile_hash", "c" * 64, "rejected semantic round.*exact reconstructed request"),
    ("provider_family", "other-provider", "declared routing"),
    ("model", "other-model", "declared routing"),
])
def test_rejected_round_provenance_mismatch_fails_prepare_and_reidentified_verification(tmp_path, field, value, match):
    publication, review, assembly, execution = _two_round_publication(tmp_path)
    rejected = execution.accepted_windows[0].rejected_semantic_rounds[0]
    forged_round = replace(rejected, invocation_provenance=replace(rejected.invocation_provenance, **{field: value}))
    forged_window = replace(execution.accepted_windows[0], rejected_semantic_rounds=(forged_round,))
    forged_execution = replace(execution, accepted_windows=(forged_window, *execution.accepted_windows[1:]))
    forged_review, forged_assembly = _review_and_assembly(forged_execution)
    with pytest.raises(B13DispositionPublicationError, match=match):
        prepare_b13_disposition_publication(forged_review, forged_assembly, forged_execution)

    payload = deepcopy(publication.to_dict())
    payload["evidence"]["accepted_windows"][0]["rejected_semantic_rounds"][0]["invocation_provenance"][field] = value
    with pytest.raises(B13DispositionPublicationError, match=match):
        verify_b13_disposition_publication(_reidentified(payload))


def test_accepted_round_provenance_mismatch_fails_with_reidentified_outer_payload(tmp_path):
    publication, _review, _assembly, _execution = _two_round_publication(tmp_path)
    payload = deepcopy(publication.to_dict())
    payload["evidence"]["accepted_windows"][0]["invocation_provenance"]["request_hash"] = "b" * 64
    with pytest.raises(B13DispositionPublicationError, match="accepted invocation provenance.*exact reconstructed request"):
        verify_b13_disposition_publication(_reidentified(payload))


@pytest.mark.parametrize("mutate", [
    lambda value: value["evidence"]["assembly"]["plan"]["event_dispositions"][0].update({"rationale_zh": "伪造。"}),
    lambda value: value["evidence"]["assembly"]["identity_bindings"][0].update({"adapted_id": "beat_forged"}),
    lambda value: value["evidence"]["reconciliation"]["decisions"].update({"decisions": [{
        "kind": "mark_unresolved", "candidate_refs": [], "finding_ids": ["f" * 64],
        "rationale_zh": "伪造。", "provenance_zh": "伪造。",
    }]}),
    lambda value: value["evidence"]["accepted_windows"][0]["proposal"]["event_dispositions"][0].update({"rationale_zh": "篡改。"}),
    lambda value: value["evidence"]["accepted_windows"].reverse(),
    lambda value: value["evidence"]["accepted_windows"].append(deepcopy(value["evidence"]["accepted_windows"][0])),
    lambda value: value["evidence"]["accepted_windows"][0]["invocation_provenance"].update({"model": "forged-model"}),
    lambda value: value.update({"semantic_identity": "0" * 64}),
    lambda value: value["evidence"]["source_snapshot"]["canonical_event_set_ref"].update({"content_hash": "0" * 64}),
    lambda value: value["evidence"]["strategy_selection"]["strategy_ref"].update({"revision": 99}),
])
def test_tampered_publication_evidence_fails_closed(tmp_path, mutate):
    publication, *_ = _publication(tmp_path)
    forged = deepcopy(publication.to_dict())
    mutate(forged)
    with pytest.raises(B13DispositionPublicationError):
        verify_b13_disposition_publication(forged)


def test_invalid_schema_and_forged_b12_ref_fail_closed(tmp_path):
    publication, *_ = _publication(tmp_path)
    invalid = deepcopy(publication.to_dict())
    invalid["schema_version"] = 999
    with pytest.raises(B13DispositionPublicationError, match="schema"):
        B13DispositionPublication.from_dict(invalid)
    forged = deepcopy(publication.to_dict())
    forged["evidence"]["strategy_selection"]["strategy_ref"]["content_hash"] = "f" * 64
    with pytest.raises(B13DispositionPublicationError, match="identity|strategy artifact"):
        verify_b13_disposition_publication(forged)


def test_prepare_rejects_modified_supplied_plan_and_binding(tmp_path):
    publication, review, assembly, execution = _publication(tmp_path)
    assert publication
    forged_plan = replace(assembly, plan=replace(assembly.plan, event_dispositions=assembly.plan.event_dispositions))
    object.__setattr__(forged_plan.plan, "event_dispositions", ())
    with pytest.raises(B13DispositionPublicationError, match="assembly"):
        prepare_b13_disposition_publication(review, forged_plan, execution)
    forged_binding = replace(assembly, identity_bindings=())
    with pytest.raises(B13DispositionPublicationError, match="assembly"):
        prepare_b13_disposition_publication(review, forged_binding, execution)
