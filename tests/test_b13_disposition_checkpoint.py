"""Zero-provider tests for the B1.3 single-window checkpoint evidence codec.

Every case uses temporary stores and synthetic fixtures; no provider is called,
no artifact is written, and no CURRENT pointer is mutated.
"""
import json
import sys
from dataclasses import replace as _dc_replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import b13_alice_runner_preparation as runner  # noqa: E402

from short_drama.adaptation import (
    B13WindowRejectedSemanticRound, B13WindowSemanticBudget, build_b13_window_semantic_preparation,
    load_b13_window_semantic_assets, prepare_b13_disposition_windows,
)
import short_drama.adaptation.disposition_checkpoint as disposition_checkpoint
from short_drama.adaptation.disposition_checkpoint import (
    B13_CHECKPOINT_MAX_BYTES, B13CheckpointEvidenceError, encode_checkpoint,
    verify_and_decode_checkpoint,
)
from short_drama.adaptation.disposition_execution import B13WindowAcceptedProposal
from short_drama.adaptation.disposition_semantic import B13WindowProposal
from short_drama.artifacts import canonical_json_bytes, content_hash, strict_json_loads
from short_drama.llm import LLMInvocationProvenance

from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_execution import _two_window_inputs
from test_b13_disposition_preparation import _published_strategy
from test_b13_window_semantic import _proposal


def _provenance(semantic, runtime, **overrides):
    base = dict(
        provider_family=runtime.provider_family, model=runtime.request_model,
        semantic_profile_id=semantic.semantic_profile.profile_id,
        semantic_profile_hash=semantic.semantic_profile.semantic_profile_hash,
        prompt_id=semantic.request.rendered_prompt.prompt_id,
        prompt_version=semantic.request.rendered_prompt.prompt_version,
        prompt_content_hash=semantic.request.rendered_prompt.prompt_content_hash,
        rendered_prompt_hash=semantic.request.rendered_prompt.rendered_prompt_hash,
        output_schema_id=semantic.output_schema.schema_id,
        output_schema_version=semantic.output_schema.schema_version,
        output_schema_hash=semantic.output_schema.schema_hash,
        request_hash=semantic.request.request_hash,
        provider_response_id="fake-response", finish_reason="stop", usage={"prompt_tokens": 1},
    )
    base.update(overrides)
    return LLMInvocationProvenance(**base)


def _accepted(packet, semantic, runtime, *, rejected=(), accepted_attempts=1):
    proposal = B13WindowProposal.from_dict(packet, _proposal(packet), load_b13_window_semantic_assets().output_schema)
    request = semantic.request
    return B13WindowAcceptedProposal(
        packet.window_id, packet.window_ordinal, semantic.request_identity, request.request_hash,
        {"profile_id": semantic.semantic_profile.profile_id, "profile_hash": semantic.semantic_profile.semantic_profile_hash},
        {"prompt_id": request.rendered_prompt.prompt_id, "prompt_version": request.rendered_prompt.prompt_version,
         "prompt_content_hash": request.rendered_prompt.prompt_content_hash},
        {"schema_id": semantic.output_schema.schema_id, "schema_version": semantic.output_schema.schema_version,
         "schema_hash": semantic.output_schema.schema_hash},
        request.rendered_prompt.rendered_prompt_hash, _provenance(semantic, runtime), len(rejected) + 1,
        accepted_attempts + sum(item.provider_attempts for item in rejected), accepted_attempts,
        tuple(item.diagnostic for item in rejected), tuple(rejected), proposal)


def _session(preparation, requests, runtime):
    budget = B13WindowSemanticBudget(runtime.request_model, 1_000_000, 1_000_000)
    return runner.B13AliceExecutionSession(
        preparation=preparation, semantic_preparations=tuple(requests),
        budgets=tuple(budget for _ in requests), runtime_config=runtime, preflight_verdict="PASS",
        request_model=runtime.request_model, effective_n_ctx=1_000_000, input_ceiling_tokens=1_000_000,
        strategy_ref=preparation.strategy_selection.strategy_ref,
        window_evidence=tuple(runner.B13AliceWindowEvidence(
            p.window_id, p.window_ordinal, s.request_identity, "PASS", "PASS")
            for p, s in zip(preparation.packets, requests)))


def _single(tmp_path):
    tree, semantic, _publication = _published_strategy(tmp_path)
    preparation = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id)
    runtime = _b12_test_runtime_config()
    requests = (build_b13_window_semantic_preparation(preparation.packets[0], runtime_config=runtime),)
    return tree, preparation, runtime, requests


def _fixture(tmp_path):
    tree, preparation, runtime, requests = _single(tmp_path)
    return tree, _session(preparation, requests, runtime), preparation.packets[0], requests[0]


def _reencode(payload):
    payload = dict(payload)
    payload["checkpoint_identity"] = content_hash({k: v for k, v in payload.items() if k != "checkpoint_identity"})
    return canonical_json_bytes(payload)


def _payload(accepted, session):
    return strict_json_loads(encode_checkpoint(accepted, session))


def _tree_bytes(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_valid_round_trip_is_deterministic_and_rebuilt_from_session(tmp_path):
    _tree, session, packet, semantic = _fixture(tmp_path)
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    first = encode_checkpoint(accepted, session)
    assert first == encode_checkpoint(accepted, session) and len(first) <= B13_CHECKPOINT_MAX_BYTES
    decoded = verify_and_decode_checkpoint(first, session)
    assert decoded == accepted and isinstance(decoded, B13WindowAcceptedProposal)
    assert decoded.proposal.to_dict() == accepted.proposal.to_dict()
    assert decoded.invocation_provenance == accepted.invocation_provenance


def test_rejects_oversize_and_non_bytes(tmp_path):
    _tree, session, _packet, _semantic = _fixture(tmp_path)
    with pytest.raises(B13CheckpointEvidenceError, match="bounded"):
        verify_and_decode_checkpoint(b"0" * (B13_CHECKPOINT_MAX_BYTES + 1), session)
    with pytest.raises(B13CheckpointEvidenceError):
        verify_and_decode_checkpoint("not-bytes", session)


def test_rejects_duplicate_keys_unexpected_fields_and_corruption(tmp_path):
    _tree, session, packet, semantic = _fixture(tmp_path)
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    good = encode_checkpoint(accepted, session)
    for bad in (b'{"a":1,"a":2}', good[:-1], b"\xff\xfe\x00", b"[1,2,3]", b"null"):
        with pytest.raises(B13CheckpointEvidenceError):
            verify_and_decode_checkpoint(bad, session)
    payload = _payload(accepted, session)
    payload["unexpected_field"] = "x"
    with pytest.raises(B13CheckpointEvidenceError):
        verify_and_decode_checkpoint(_reencode(payload), session)


_MISMATCHES = [
    ("preparation", lambda p: p.__setitem__("preparation_identity", "0" * 64)),
    ("source", lambda p: p.__setitem__("source_snapshot_hash", "0" * 64)),
    ("strategy_ref", lambda p: p["strategy_ref"].__setitem__("revision", p["strategy_ref"]["revision"] + 1)),
    ("strategy_hash", lambda p: p.__setitem__("strategy_hash", "0" * 64)),
    ("window_id", lambda p: p["window"].__setitem__("window_id", "window_missing")),
    ("ordinal", lambda p: p["window"].__setitem__("window_ordinal", p["window"]["window_ordinal"] + 1)),
    ("packet_hash", lambda p: p["window"].__setitem__("packet_hash", "0" * 64)),
    ("request_identity", lambda p: p.__setitem__("request_identity", "0" * 64)),
    ("request_hash", lambda p: p.__setitem__("semantic_request_hash", "0" * 64)),
    ("profile", lambda p: p["semantic_profile_identity"].__setitem__("profile_hash", "0" * 64)),
    ("prompt", lambda p: p["prompt_identity"].__setitem__("prompt_content_hash", "0" * 64)),
    ("schema", lambda p: p["output_schema_identity"].__setitem__("schema_hash", "0" * 64)),
    ("rendered", lambda p: p.__setitem__("rendered_prompt_hash", "0" * 64)),
    ("proposal", lambda p: p["proposal"]["event_dispositions"][0].__setitem__("disposition", "omit")),
    ("provenance", lambda p: p["invocation_provenance"].__setitem__("request_hash", "0" * 64)),
    ("accounting", lambda p: p.__setitem__("provider_attempts", p["provider_attempts"] + 1)),
    ("failures", lambda p: p.__setitem__("semantic_failures", ["mismatched"])),
]


@pytest.mark.parametrize("mutate", [item[1] for item in _MISMATCHES], ids=[item[0] for item in _MISMATCHES])
def test_field_mismatches_are_rejected(tmp_path, mutate):
    _tree, session, packet, semantic = _fixture(tmp_path)
    payload = _payload(_accepted(packet, semantic, _b12_test_runtime_config()), session)
    mutate(payload)
    with pytest.raises(B13CheckpointEvidenceError):
        verify_and_decode_checkpoint(_reencode(payload), session)


def test_two_distinct_windows_cannot_exchange_checkpoints(tmp_path):
    _tree, two, requests, _budgets, runtime, first, second = _two_window_inputs(tmp_path)
    session = _session(two, requests, runtime)
    accepted_a = _accepted(first, requests[0], runtime)
    accepted_b = _accepted(second, requests[1], runtime)
    payload = _payload(accepted_b, session)
    payload["proposal"] = _payload(accepted_a, session)["proposal"]
    with pytest.raises(B13CheckpointEvidenceError):
        verify_and_decode_checkpoint(_reencode(payload), session)
    assert verify_and_decode_checkpoint(encode_checkpoint(accepted_a, session), session) == accepted_a
    assert verify_and_decode_checkpoint(encode_checkpoint(accepted_b, session), session) == accepted_b


def test_two_round_preserves_rejected_provenance_and_accounting(tmp_path):
    _tree, session, packet, semantic = _fixture(tmp_path)
    runtime = _b12_test_runtime_config()
    rejected = (B13WindowRejectedSemanticRound(
        1, 2, _provenance(semantic, runtime, provider_response_id="r1"),
        "schema-valid but semantically invalid"),)
    accepted = _accepted(packet, semantic, runtime, rejected=rejected, accepted_attempts=1)
    assert accepted.semantic_rounds == 2 and accepted.provider_attempts == 3
    decoded = verify_and_decode_checkpoint(encode_checkpoint(accepted, session), session)
    assert decoded == accepted
    assert decoded.rejected_semantic_rounds[0].provider_attempts == 2
    assert decoded.rejected_semantic_rounds[0].invocation_provenance == rejected[0].invocation_provenance
    assert decoded.provider_attempts == 3 and decoded.accepted_round_provider_attempts == 1


def test_codec_performs_no_writes_and_no_generation(tmp_path):
    tree, session, packet, semantic = _fixture(tmp_path)
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    store_before, pointers_before = _tree_bytes(tree.store.root), _tree_bytes(tree.pointers.root)
    raw = encode_checkpoint(accepted, session)
    verify_and_decode_checkpoint(raw, session)
    assert _tree_bytes(tree.store.root) == store_before
    assert _tree_bytes(tree.pointers.root) == pointers_before


def _broken_session(session, name):
    row, budget = session.window_evidence[0], B13WindowSemanticBudget("other", 100, 100)
    return _dc_replace(session, **{
        "non_pass_preflight": dict(preflight_verdict="FAIL"), "wrong_model": dict(request_model="other-model"),
        "wrong_strategy_ref": dict(strategy_ref=_dc_replace(session.strategy_ref, revision=session.strategy_ref.revision + 1)),
        "missing_budgets": dict(budgets=()), "wrong_budget_model": dict(budgets=(budget,) * len(session.budgets)),
        "missing_evidence": dict(window_evidence=()),
        "evidence_request_mismatch": dict(window_evidence=(_dc_replace(row, request_identity="0" * 64),) + session.window_evidence[1:]),
        "evidence_ordinal_mismatch": dict(window_evidence=(_dc_replace(row, window_ordinal=row.window_ordinal + 1),) + session.window_evidence[1:]),
        "missing_semantic": dict(semantic_preparations=()),
    }[name])


@pytest.mark.parametrize("name", [
    "non_pass_preflight", "wrong_model", "wrong_strategy_ref", "missing_budgets", "wrong_budget_model",
    "missing_evidence", "evidence_request_mismatch", "evidence_ordinal_mismatch", "missing_semantic",
])
def test_session_evidence_gate_rejects(tmp_path, name):
    _tree, session, packet, semantic = _fixture(tmp_path)
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    broken = _broken_session(session, name)
    with pytest.raises(B13CheckpointEvidenceError):
        encode_checkpoint(accepted, broken)
    with pytest.raises(B13CheckpointEvidenceError):
        verify_and_decode_checkpoint(encode_checkpoint(accepted, session), broken)


def test_rejects_valid_but_non_canonical_bytes(tmp_path):
    _tree, session, packet, semantic = _fixture(tmp_path)
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    payload = _payload(accepted, session)
    canonical = encode_checkpoint(accepted, session)
    reversed_keys = json.dumps({key: payload[key] for key in reversed(list(payload))},
                               separators=(",", ":"), sort_keys=False).encode()
    spaced = json.dumps(payload, separators=(", ", ": "), sort_keys=True).encode()
    escaped = canonical.replace(b"_", b"\\u005f", 1)
    for bad in (reversed_keys, spaced, escaped):
        with pytest.raises(B13CheckpointEvidenceError):
            verify_and_decode_checkpoint(bad, session)


def test_encode_rejects_oversized_evidence(tmp_path, monkeypatch):
    _tree, session, packet, semantic = _fixture(tmp_path)
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    monkeypatch.setattr(disposition_checkpoint, "B13_CHECKPOINT_MAX_BYTES", 16)
    with pytest.raises(B13CheckpointEvidenceError, match="bounded"):
        encode_checkpoint(accepted, session)
