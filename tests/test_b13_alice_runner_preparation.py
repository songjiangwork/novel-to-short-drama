"""Offline, zero-provider tests for the B1.3 Alice execution-session preparation.

No network, no generation, no production-run tree.  Discovery/tokenizer use an
injected fake transport and the deterministic B1.3 preparation fixture from the
preflight tests; the session is built purely from existing production APIs.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import b13_alice_runner_preparation as runner  # noqa: E402
import b13_alice_token_budget_preflight as pf  # noqa: E402
import test_b13_alice_token_budget_preflight as pft  # noqa: E402

# A realistic ceiling: well above the fixture prompt but below its verified
# per-slot n_ctx (163840), mirroring the real Alice 65,536 declaration.
_CEILING = 65_536


def _call(tmp_path, *, responses=None, client=None, input_ceiling_tokens=_CEILING,
          model_context_tokens=None):
    tree, source, publication, target_id = pft._preparation(tmp_path)
    client = client or pft._client(responses if responses is not None else pft._responses())
    session = runner.prepare_b13_alice_execution_session(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        runtime_config=pft._runtime(), client=client,
        input_ceiling_tokens=input_ceiling_tokens, model_context_tokens=model_context_tokens,
        strategy_ref=publication.strategy_ref, adaptation_target_id=target_id)
    return tree, client, session


def test_complete_successful_session(tmp_path):
    tree, client, session = _call(tmp_path)
    assert session.preflight_verdict == "PASS"
    assert session.preparation is not None
    assert len(session.semantic_preparations) == len(session.preparation.packets)
    assert len(session.budgets) == len(session.preparation.packets)
    assert session.effective_n_ctx == 163840 and session.request_model == pft.REQUEST_MODEL


def test_exact_tuple_lengths_and_ordering(tmp_path):
    _, _, session = _call(tmp_path)
    packets = session.preparation.packets
    assert len(session.semantic_preparations) == len(packets) == len(session.budgets)
    assert [s.packet.window_id for s in session.semantic_preparations] == [p.window_id for p in packets]
    assert [e.window_ordinal for e in session.window_evidence] == [p.window_ordinal for p in packets]


def test_budget_values_match_verified_capacity(tmp_path):
    _, _, session = _call(tmp_path)
    for budget in session.budgets:
        assert budget.runtime_model_id == pft.REQUEST_MODEL
        assert budget.model_context_tokens == session.effective_n_ctx
        assert budget.input_ceiling_tokens == session.input_ceiling_tokens


def test_runtime_config_identity_consistency(tmp_path):
    _, _, session = _call(tmp_path)
    runtime = pft._runtime()
    assert session.runtime_config == runtime
    assert all(s.runtime_config == runtime for s in session.semantic_preparations)


def test_stable_request_identities(tmp_path):
    _, _, session = _call(tmp_path)
    runtime = session.runtime_config
    for packet, semantic, evidence in zip(
            session.preparation.packets, session.semantic_preparations, session.window_evidence):
        rebuilt = pf.build_b13_window_semantic_preparation(packet, runtime_config=runtime)
        assert rebuilt.request_identity == semantic.request_identity == evidence.request_identity


def test_unknown_preflight_refuses_session(tmp_path):
    responses = pft._responses()
    responses[("GET", "/slots")] = 500  # capacity unverified -> UNKNOWN
    with pytest.raises(runner.B13AliceSessionError):
        _call(tmp_path, responses=responses)


def test_fail_preflight_refuses_session(tmp_path):
    with pytest.raises(runner.B13AliceSessionError):
        _call(tmp_path, input_ceiling_tokens=1)  # every window's estimator FAILs


def test_model_mismatch_refuses_session(tmp_path):
    with pytest.raises(runner.B13AliceSessionError):
        _call(tmp_path, responses=pft._responses(model="other-model"))


def test_changed_request_evidence_refuses_session(tmp_path, monkeypatch):
    real_build = runner.build_b13_window_semantic_preparation

    def forge(packet, **kwargs):
        return replace(real_build(packet, **kwargs), request_identity="f" * 64)

    monkeypatch.setattr(runner, "build_b13_window_semantic_preparation", forge)
    with pytest.raises(runner.B13AliceSessionError):
        _call(tmp_path)


def test_zero_generation_calls(tmp_path):
    _, client, _ = _call(tmp_path)
    assert client.generation_calls == []


def test_zero_writes_and_current_mutations(tmp_path):
    tree, source, publication, target_id = pft._preparation(tmp_path)
    files_before = pft._tree_files(tree.store.root, tree.pointers.root)
    runner.prepare_b13_alice_execution_session(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        runtime_config=pft._runtime(), client=pft._client(pft._responses()),
        input_ceiling_tokens=_CEILING, strategy_ref=publication.strategy_ref,
        adaptation_target_id=target_id)
    assert pft._tree_files(tree.store.root, tree.pointers.root) == files_before
    assert tree.pointers.resolve_current_pointer_ref(
        publication.current_pointer_ref.artifact_id) == publication.current_pointer_ref


def test_deterministic_repeated_preparation(tmp_path):
    tree, source, publication, target_id = pft._preparation(tmp_path)

    def build():
        return runner.prepare_b13_alice_execution_session(
            store=tree.store, pointers=tree.pointers, source_snapshot=source,
            runtime_config=pft._runtime(), client=pft._client(pft._responses()),
            input_ceiling_tokens=_CEILING, strategy_ref=publication.strategy_ref,
            adaptation_target_id=target_id)

    first, second = build(), build()
    assert first.preparation.preparation_identity == second.preparation.preparation_identity
    assert [e.request_identity for e in first.window_evidence] == \
        [e.request_identity for e in second.window_evidence]
    assert (first.effective_n_ctx, first.input_ceiling_tokens) == \
        (second.effective_n_ctx, second.input_ceiling_tokens)
