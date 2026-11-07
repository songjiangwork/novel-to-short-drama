"""Focused production-entry wiring tests; isolated fake Provider only."""
from dataclasses import replace

import pytest

from short_drama.adaptation import disposition_checkpoint_storage as storage
from short_drama.adaptation.disposition_execution_claim import (
    ClaimState, claim_path, create_claim, execution_guard, inspect_claim,
)
from short_drama.adaptation.disposition_resume import inspect_b13_resume_versions
from short_drama.adaptation.disposition_runner import (
    B13RunnerError, B13RunnerStatus, run_b13_resumable,
)
from test_b13_disposition_checkpoint import _fixture, _session
from test_b13_disposition_execution import (
    _allow_synthetic_two_window_preflight, _two_window_inputs,
)
from test_b13_feedback_synthetic_loop import RecordingFakeClient, SyntheticTokenizer
from test_b13_window_semantic import _proposal


def _run(tree, session, tmp_path, client, selected, *, refresh=None, tokenizer=None, limit=1):
    # Existing RecordingFakeClient intentionally omits runtime_config unless
    # provided explicitly.  The real OpenAICompatibleLLMClient always exposes
    # its bound RuntimeConfig; match that contract rather than loosening the
    # production runner's pre-dispatch identity check.
    client.runtime_config = session.runtime_config
    return run_b13_resumable(
        store=tree.store, pointers=tree.pointers, session=session,
        llm_client=client, execution_root=tmp_path / "execution",
        checkpoint_root=tmp_path / "checkpoints", max_new_windows=limit,
        enable_feedback=True, feedback_tokenizer=tokenizer or SyntheticTokenizer(),
        refresh_session=refresh or (lambda: session),
        selected_window_ids=selected,
    )


def test_feedback_checkpoint_v2_durable_and_exact_reuse(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    first = _proposal(packet)
    first["adaptation_mappings"] = []
    client = RecordingFakeClient(
        [(first, False), (_proposal(packet), False)], session.runtime_config)
    result = _run(tree, session, tmp_path, client, (packet.window_id,))
    assert result.status is B13RunnerStatus.COMPLETE
    assert (result.generated_windows, result.new_semantic_calls, result.new_provider_attempts) == (1, 2, 2)
    accepted = result.execution_result.accepted_windows[0]
    assert accepted.feedback_evidence is not None
    assert len(client.calls) == 2
    assert client.responses[0].provenance.request_hash != client.responses[1].provenance.request_hash
    files = list((tmp_path / "checkpoints").glob("b13-checkpoint-v2-*.json"))
    assert len(files) == 1 and files[0].stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "checkpoints").stat().st_mode & 0o777 == 0o700
    assert files[0].read_bytes() == accepted.feedback_evidence.canonical_bytes()
    assert inspect_claim(tmp_path / "execution", session, packet.window_id).state is ClaimState.CLAIMED_UNRESOLVED
    before = files[0].read_bytes()
    again = _run(tree, session, tmp_path, client, (packet.window_id,))
    assert again.status is B13RunnerStatus.COMPLETE
    assert again.generated_windows == 0 and again.reused_windows == 1
    assert again.execution_result.accepted_windows == (accepted,)
    assert files[0].read_bytes() == before and len(client.calls) == 2


def test_feedback_enabled_first_round_still_uses_v1(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([(_proposal(packet), False)], session.runtime_config)
    result = _run(tree, session, tmp_path, client, (packet.window_id,))
    assert result.status is B13RunnerStatus.COMPLETE
    assert result.execution_result.accepted_windows[0].feedback_evidence is None
    assert len(list((tmp_path / "checkpoints").glob("b13-checkpoint-v1-*.json"))) == 1
    assert not list((tmp_path / "checkpoints").glob("b13-checkpoint-v2-*.json"))


def test_explicit_selection_skips_existing_unresolved_claim_without_completion(tmp_path, monkeypatch):
    tree, preparation, requests, _, runtime, _, _ = _two_window_inputs(tmp_path)
    _allow_synthetic_two_window_preflight(monkeypatch, preparation)
    session = _session(preparation, requests, runtime)
    first, second = preparation.packets
    with execution_guard(tmp_path / "execution", store=tree.store, pointers=tree.pointers) as guard:
        create_claim(guard=guard, store=tree.store, pointers=tree.pointers, session=session,
                     checkpoint_root=tmp_path / "checkpoints", window_id=first.window_id)
    original_claim = claim_path(tmp_path / "execution", session, first.window_id).read_bytes()
    client = RecordingFakeClient([(_proposal(second), False)], session.runtime_config)
    result = _run(tree, session, tmp_path, client, (second.window_id,))
    assert result.status is B13RunnerStatus.BOUNDED_PROGRESS
    assert result.execution_result is None and result.remaining_windows == 1
    assert (result.generated_windows, result.reused_windows) == (1, 0)
    assert claim_path(tmp_path / "execution", session, first.window_id).read_bytes() == original_claim
    assert not (tmp_path / "checkpoints").joinpath(
        storage._checkpoint_name(session, first.window_id)).exists()
    plan = inspect_b13_resume_versions(store=tree.store, pointers=tree.pointers,
                                        session=session, checkpoint_root=tmp_path / "checkpoints")
    assert plan.needs_generation_count == 1 and plan.verified_reuse_count == 1
    again = _run(tree, session, tmp_path, client, (second.window_id,))
    assert again.status is B13RunnerStatus.BOUNDED_PROGRESS
    assert again.generated_windows == 0 and len(client.calls) == 1
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (first.window_id,))
    assert len(client.calls) == 1
    assert claim_path(tmp_path / "execution", session, first.window_id).read_bytes() == original_claim


@pytest.mark.parametrize("selection", [(), ("missing",), ("duplicate", "duplicate")])
def test_invalid_window_selection_has_no_side_effects(tmp_path, selection):
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([], session.runtime_config)
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, selection)
    assert not client.calls and not (tmp_path / "execution").exists()


def test_mismatched_client_runtime_refuses_before_claim(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([], session.runtime_config)
    client.runtime_config = replace(session.runtime_config, request_model="wrong-model")
    with pytest.raises(B13RunnerError):
        run_b13_resumable(
            store=tree.store, pointers=tree.pointers, session=session,
            llm_client=client, execution_root=tmp_path / "execution",
            checkpoint_root=tmp_path / "checkpoints", max_new_windows=1,
            enable_feedback=True, feedback_tokenizer=SyntheticTokenizer(),
            refresh_session=lambda: session, selected_window_ids=(packet.window_id,))
    assert not client.calls and not (tmp_path / "execution").exists()


def test_fresh_runtime_required_before_claim(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([], session.runtime_config)
    stale = replace(session, effective_n_ctx=session.effective_n_ctx - 1)
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,), refresh=lambda: stale)
    assert not client.calls
    assert inspect_claim(tmp_path / "execution", session, packet.window_id).state is ClaimState.ABSENT


def test_runtime_change_before_round_two_prevents_second_provider_call(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected["adaptation_mappings"] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    calls = [0]
    def refresh():
        calls[0] += 1
        return session if calls[0] < 4 else replace(session, effective_n_ctx=session.effective_n_ctx - 1)

    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,), refresh=refresh)
    assert len(client.calls) == 1
    assert inspect_claim(tmp_path / "execution", session, packet.window_id).state is ClaimState.CLAIMED_UNRESOLVED
    assert not list((tmp_path / "checkpoints").glob("*.json"))


def test_v2_writer_failure_preserves_claim_and_prevents_replay(tmp_path, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected["adaptation_mappings"] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    def fail(*_args):
        raise OSError("test-only injected v2 fsync failure")
    monkeypatch.setattr(storage, "_fsync_directory", fail)
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    assert len(client.calls) == 2
    # An uncertain write cannot be interpreted as authorization to call again.
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    assert len(client.calls) == 2
