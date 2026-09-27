"""Offline tests for A5G3 acceptance-only gates."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from short_drama.llm import LLMRetryExhaustedError
from short_drama.story import ConsolidationSemanticGenerationError


SPEC = importlib.util.spec_from_file_location(
    "a5g3_acceptance", Path(__file__).parents[1] / "scripts" / "a5g3_real_novel_acceptance.py"
)
acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(acceptance)


def _evidence(paragraph="p1"):
    return {"paragraph_id": paragraph, "role": "primary", "strength": "strong", "excerpt": "text"}


def _bundle():
    index = {
        "facts": [{"global_candidate_ref": "fact:one", "subject_refs": ["char_a"], "object_refs": []}],
        "events": [{"global_candidate_ref": "event:one", "participants": ["unres_x"], "locations": []}],
        "relationships": [{"global_candidate_ref": "relationship:one", "source_entity_ref": "char_a", "target_entity_ref": "loc_a"}],
    }
    facts = {"facts": [{"fact_id": "fact_001", "subject_refs": ["char_a"], "object_refs": [], "candidate_fact_refs": ["fact:one"], "evidence_refs": [_evidence()]}], "state_transitions": [{"transition_id": "transition_001", "from_fact_id": "fact_001", "to_fact_id": "fact_001", "subject_refs": ["char_a"], "evidence_refs": [_evidence("p2")]}]}
    events = {"events": [{"event_id": "event_001", "participants": ["unres_x"], "locations": [], "candidate_event_refs": ["event:one"], "evidence_refs": [_evidence("p3")]}]}
    relationships = {"relationships": [{"relationship_id": "relationship_001", "source_entity_ref": "char_a", "target_entity_ref": "loc_a", "candidate_relationship_refs": ["relationship:one"], "state_history": [{"candidate_relationship_refs": ["relationship:one"], "evidence_refs": [_evidence("p4")]}]}]}
    conflicts = {"conflicts": [{"conflict_id": "conflict_001", "fact_ids": ["fact_001"], "relationship_ids": ["relationship_001"], "evidence_refs": [_evidence("p5")]}]}
    return index, facts, events, relationships, conflicts


def _audit(**changes):
    index, facts, events, relationships, conflicts = _bundle()
    index, facts, events, relationships, conflicts = changes.get("bundle", (index, facts, events, relationships, conflicts))
    return acceptance.audit_persisted_bundle(index=index, facts=facts, events=events, relationships=relationships, conflicts=conflicts, allowed_entity_ids={"char_a", "loc_a", "unres_x"}, unresolved_ids={"unres_x"})


def test_candidate_accounting_and_unresolved_preservation_pass():
    result = _audit()
    assert result["candidate_accounting"]["fact"] == {"accounted": 1, "total": 1, "missing": 0, "extra": 0}
    assert result["unresolved"]["dangling_unresolved_ids"] == 0


@pytest.mark.parametrize("domain, field, bad_ref", [("facts", "candidate_fact_refs", "fact:extra"), ("events", "candidate_event_refs", "event:extra")])
def test_extra_candidate_accounting_fails(domain, field, bad_ref):
    index, facts, events, relationships, conflicts = _bundle()
    {"facts": facts["facts"][0], "events": events["events"][0]}[domain][field].append(bad_ref)
    with pytest.raises(acceptance.AcceptanceError, match="candidate accounting"):
        _audit(bundle=(index, facts, events, relationships, conflicts))


def test_missing_candidate_accounting_fails():
    index, facts, events, relationships, conflicts = _bundle()
    facts["facts"][0]["candidate_fact_refs"] = []
    with pytest.raises(acceptance.AcceptanceError, match="fact candidate accounting"):
        _audit(bundle=(index, facts, events, relationships, conflicts))


def test_dangling_bound_ref_transition_and_conflict_refs_fail():
    index, facts, events, relationships, conflicts = _bundle()
    facts["facts"][0]["subject_refs"] = ["char_missing"]
    with pytest.raises(acceptance.AcceptanceError, match="dangling bound"):
        _audit(bundle=(index, facts, events, relationships, conflicts))
    index, facts, events, relationships, conflicts = _bundle()
    facts["state_transitions"][0]["to_fact_id"] = "fact_999"
    with pytest.raises(acceptance.AcceptanceError, match="graph consistency"):
        _audit(bundle=(index, facts, events, relationships, conflicts))
    index, facts, events, relationships, conflicts = _bundle()
    conflicts["conflicts"][0]["relationship_ids"] = ["relationship_999"]
    with pytest.raises(acceptance.AcceptanceError, match="graph consistency"):
        _audit(bundle=(index, facts, events, relationships, conflicts))


def test_duplicate_exact_evidence_within_tuple_fails_but_cross_object_is_allowed():
    index, facts, events, relationships, conflicts = _bundle()
    facts["facts"][0]["evidence_refs"].append(_evidence())
    with pytest.raises(acceptance.AcceptanceError, match="duplicate exact EvidenceRefs"):
        _audit(bundle=(index, facts, events, relationships, conflicts))
    index, facts, events, relationships, conflicts = _bundle()
    events["events"][0]["evidence_refs"] = [_evidence()]
    assert _audit(bundle=(index, facts, events, relationships, conflicts))["evidence"]["duplicate_exact_within_tuples"] == 0


@pytest.mark.parametrize(
    "collection, duplicate",
    [
        ("events", {"event_id": "event_001", "participants": [], "locations": [], "candidate_event_refs": [], "evidence_refs": []}),
        ("transitions", {"transition_id": "transition_001", "from_fact_id": "fact_001", "to_fact_id": "fact_001", "subject_refs": [], "evidence_refs": []}),
        ("conflicts", {"conflict_id": "conflict_001", "fact_ids": [], "relationship_ids": [], "evidence_refs": []}),
    ],
)
def test_every_canonical_id_namespace_is_checked(collection, duplicate):
    index, facts, events, relationships, conflicts = _bundle()
    target = {
        "events": events["events"],
        "transitions": facts["state_transitions"],
        "conflicts": conflicts["conflicts"],
    }[collection]
    target.append(duplicate)
    with pytest.raises(acceptance.AcceptanceError, match="canonical graph consistency"):
        _audit(bundle=(index, facts, events, relationships, conflicts))


def test_upstream_identity_requires_exact_twelve_a3_refs():
    ref = SimpleNamespace(to_dict=lambda: {"artifact_id": "x"})
    a3 = SimpleNamespace(source_document_ref=ref, chunk_manifest_ref=ref, candidate_extraction_refs=(ref,) * 12)
    current = SimpleNamespace(entity_map=SimpleNamespace(a3_input=a3), entity_map_ref=ref, validation_report_ref=ref, current_pointer_ref=ref)
    assert len(acceptance.require_upstream_identity(current, project_id="a3e-real-novel", reconciliation_profile_id="entity-reconciliation-v2")["candidate_extraction_refs"]) == 12
    current.entity_map.a3_input = SimpleNamespace(source_document_ref=ref, chunk_manifest_ref=ref, candidate_extraction_refs=(ref,) * 11)
    with pytest.raises(acceptance.AcceptanceError, match="expected 12"):
        acceptance.require_upstream_identity(current, project_id="a3e-real-novel", reconciliation_profile_id="entity-reconciliation-v2")


def _result(manifest="m", report="r", pointer="p", *, reused=False, calls=1):
    ref = lambda value: SimpleNamespace(to_dict=lambda: {"artifact_id": value})
    return SimpleNamespace(reused=reused, semantic_generation_call_count=calls, consolidation_candidate_index_ref=ref("i"), consolidation_decision_set_ref=ref("d"), canonical_fact_set_ref=ref("f"), canonical_event_set_ref=ref("e"), canonical_relationship_set_ref=ref("rel"), story_conflict_set_ref=ref("c"), consolidation_manifest_ref=ref(manifest), validation_report_ref=ref(report), current_pointer_ref=ref(pointer))


def test_exact_rerun_and_transport_gates_require_same_refs_zero_calls_and_no_writes():
    first, rerun = _result(), _result(reused=True, calls=0)
    acceptance.check_reuse_gate(first, rerun, semantic_delta=0, attempt_delta=0, before={"a": "1"}, after={"a": "1"})
    with pytest.raises(acceptance.AcceptanceError):
        acceptance.check_reuse_gate(first, _result("new", reused=True, calls=0), semantic_delta=0, attempt_delta=0, before={}, after={})
    with pytest.raises(acceptance.AcceptanceError):
        acceptance.check_reuse_gate(first, rerun, semantic_delta=1, attempt_delta=0, before={}, after={})
    with pytest.raises(acceptance.AcceptanceError):
        acceptance.check_reuse_gate(first, rerun, semantic_delta=0, attempt_delta=0, before={"a": "1"}, after={"a": "2"})


def test_semantic_invalidation_gate_requires_new_hash_miss_calls_and_temp_current():
    result = _result("new")
    current = SimpleNamespace(manifest_ref=result.consolidation_manifest_ref)
    acceptance.check_invalidation_gate(old_hash="old", new_hash="new", result=result, semantic_calls=1, provider_attempts=1, old_manifest={"artifact_id": "old"}, temp_current=current)
    with pytest.raises(acceptance.AcceptanceError):
        acceptance.check_invalidation_gate(old_hash="same", new_hash="same", result=result, semantic_calls=1, provider_attempts=1, old_manifest={"artifact_id": "old"}, temp_current=current)
    with pytest.raises(acceptance.AcceptanceError):
        acceptance.check_invalidation_gate(old_hash="old", new_hash="new", result=_result("new", reused=True), semantic_calls=1, provider_attempts=1, old_manifest={"artifact_id": "old"}, temp_current=current)


def test_counting_client_counts_successful_result_attempts():
    inner = SimpleNamespace(generate_structured=lambda *args: SimpleNamespace(attempts=2))
    client = acceptance.CountingLLMClient(inner)
    result = client.generate_structured(None, None, None)
    assert result.attempts == 2
    assert client.semantic_generation_calls == 1
    assert client.provider_attempts == 2


def test_counting_client_counts_exhausted_attempts_and_reraises_same_exception():
    error = LLMRetryExhaustedError(attempts=3, message="connection refused")

    def raise_error(*args):
        raise error

    client = acceptance.CountingLLMClient(SimpleNamespace(generate_structured=raise_error))
    with pytest.raises(LLMRetryExhaustedError) as caught:
        client.generate_structured(None, None, None)
    assert caught.value is error
    assert client.semantic_generation_calls == 1
    assert client.provider_attempts == 3
    assert acceptance.live_failure_payload(error, client) == {
        "valid": False,
        "error_type": "LLMRetryExhaustedError",
        "error": "connection refused",
        "semantic_generation_calls_observed": 1,
        "provider_attempts_observed": 3,
        "attempts": 3,
    }


def test_semantic_failure_payload_uses_frozen_block_prefix_diagnostics():
    error = ConsolidationSemanticGenerationError(
        block_id="a5eblk_001",
        request_hash="a" * 64,
        rounds_attempted=2,
        last_failure_details="selector invalid",
        expected_pairs=(("event:one", "event:two"),),
    )
    payload = acceptance.live_failure_payload(error, None)
    assert {key: payload[key] for key in ("domain", "block_id", "rounds_attempted", "pair_count", "last_failure_details")} == {
        "domain": "event",
        "block_id": "a5eblk_001",
        "rounds_attempted": 2,
        "pair_count": 1,
        "last_failure_details": "selector invalid",
    }
