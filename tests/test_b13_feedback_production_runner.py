"""Focused production-entry wiring tests; isolated fake Provider only."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from short_drama.adaptation import disposition_checkpoint_storage as storage
from short_drama.adaptation.disposition_execution_claim import (
    ClaimState, claim_path, create_claim, execution_guard, inspect_claim,
)
from short_drama.adaptation.disposition_resume import inspect_b13_resume_versions
from short_drama.adaptation.disposition_runner import (
    B13RunnerError, B13RunnerStatus, _run_b13_feedback_resumable, run_b13_resumable,
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
    with patch('short_drama.adaptation.disposition_runner._build_feedback_refresh',
               return_value=refresh or (lambda: session)), patch(
                   'short_drama.adaptation.disposition_runner._verify_feedback_client_binding'):
        return _run_b13_feedback_resumable(
            store=tree.store, pointers=tree.pointers, session=session,
            llm_client=client, execution_root=tmp_path / "execution",
            checkpoint_root=tmp_path / "checkpoints", max_new_windows=limit,
            feedback_tokenizer=tokenizer or SyntheticTokenizer(), selected_window_ids=selected,
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
        _run_b13_feedback_resumable(
            store=tree.store, pointers=tree.pointers, session=session,
            llm_client=client, execution_root=tmp_path / "execution",
            checkpoint_root=tmp_path / "checkpoints", max_new_windows=1,
            feedback_tokenizer=SyntheticTokenizer(),
            selected_window_ids=(packet.window_id,))
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


def test_public_runner_has_no_feedback_opt_in_surface(tmp_path):
    tree, session, _packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([], session.runtime_config)
    with pytest.raises(TypeError):
        run_b13_resumable(
            store=tree.store, pointers=tree.pointers, session=session,
            llm_client=client, execution_root=tmp_path / "execution",
            checkpoint_root=tmp_path / "checkpoints", max_new_windows=1,
            enable_feedback=True, feedback_tokenizer=SyntheticTokenizer())
    assert not client.calls and not (tmp_path / "execution").exists()


def test_feedback_error_has_no_provider_content_in_exception_graph(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([], session.runtime_config)
    secret = "SECRET_NOVEL_CONTENT_SHOULD_NOT_ESCAPE"
    def schema_leak(*_args, **_kwargs):
        raise ValueError(secret)
    client.generate_structured = schema_leak
    with pytest.raises(B13RunnerError) as caught:
        _run(tree, session, tmp_path, client, (packet.window_id,))
    error = caught.value
    assert secret not in str(error)
    assert error.__cause__ is None and error.__context__ is None
    assert error.__suppress_context__ is False


def test_production_entry_rejects_synthetic_client_without_preflight_or_dispatch(tmp_path, monkeypatch):
    import b13_alice_bounded_runner as alice
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([], session.runtime_config)
    client.runtime_config = session.runtime_config
    def should_not_prepare(**_kw):
        raise AssertionError("must fail before production preflight")
    monkeypatch.setattr(alice, "prepare_b13_alice_execution_session", should_not_prepare)
    with pytest.raises(B13RunnerError, match="matched live generation/tokenizer"):
        alice.run_b13_alice_bounded(
            store=tree.store, pointers=tree.pointers, source_snapshot=session.preparation.source_snapshot,
            runtime_config=session.runtime_config, preflight_client=SyntheticTokenizer(),
            llm_client=client, input_ceiling_tokens=10_000,
            execution_root=tmp_path / "execution", checkpoint_root=tmp_path / "checkpoints",
            max_new_windows=1, enable_feedback=True, selected_window_ids=(packet.window_id,))
    assert not client.calls and not (tmp_path / "execution").exists()


def test_production_entry_rejects_injected_preflight_transport(tmp_path, monkeypatch):
    import b13_alice_bounded_runner as alice
    from b13_alice_token_budget_preflight import LlamaClient, _server_root
    from short_drama.llm import OpenAICompatibleLLMClient
    tree, session, packet, _ = _fixture(tmp_path)
    preflight = LlamaClient(base_url=_server_root(session.runtime_config.base_url),
                            transport=lambda *_args: (200, b"{}"))
    client = OpenAICompatibleLLMClient(session.runtime_config)
    monkeypatch.setattr(alice, "prepare_b13_alice_execution_session",
                        lambda **_kw: (_ for _ in ()).throw(AssertionError("preflight must not start")))
    with pytest.raises(B13RunnerError, match="matched live generation/tokenizer"):
        alice.run_b13_alice_bounded(
            store=tree.store, pointers=tree.pointers, source_snapshot=session.preparation.source_snapshot,
            runtime_config=session.runtime_config, preflight_client=preflight,
            llm_client=client, input_ceiling_tokens=10_000,
            execution_root=tmp_path / "execution", checkpoint_root=tmp_path / "checkpoints",
            max_new_windows=1, enable_feedback=True, selected_window_ids=(packet.window_id,))
    assert not preflight.requested_paths


def test_postlink_fsync_failure_cannot_yield_complete_until_guarded_confirmation(tmp_path, monkeypatch):
    import os
    import stat
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected["adaptation_mappings"] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    target_root = tmp_path / "checkpoints"
    original_fsync = storage.os.fsync
    refusal = [True]

    def fail_v2_root_sync(fd):
        if (refusal[0] and stat.S_ISDIR(os.fstat(fd).st_mode)
                and target_root.exists()
                and any(target_root.glob("b13-checkpoint-v2-*.json"))):
            raise OSError("test-only post-link directory fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", fail_v2_root_sync)
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    assert len(client.calls) == 2
    visible = list(target_root.glob("b13-checkpoint-v2-*.json"))
    assert len(visible) == 1 and visible[0].stat().st_nlink == 1
    # The file is readable, but its directory durability has not been
    # confirmed. A new invocation must fail without declaring COMPLETE.
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    assert len(client.calls) == 2
    refusal[0] = False
    recovered = _run(tree, session, tmp_path, client, (packet.window_id,))
    assert recovered.status is B13RunnerStatus.COMPLETE
    assert recovered.generated_windows == 0 and recovered.reused_windows == 1
    assert len(client.calls) == 2


def test_failed_staging_unlink_never_returns_success_or_replays(tmp_path, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected["adaptation_mappings"] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    original_unlink = storage.os.unlink

    def fail_v2_staging_cleanup(name, *args, **kwargs):
        if isinstance(name, str) and name.startswith(".b13-checkpoint-v2-") and name.endswith(".tmp"):
            raise OSError("test-only temporary link cleanup failure")
        return original_unlink(name, *args, **kwargs)

    monkeypatch.setattr(storage.os, "unlink", fail_v2_staging_cleanup)
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    assert len(client.calls) == 2
    files = list((tmp_path / "checkpoints").glob("b13-checkpoint-v2-*.json"))
    assert len(files) == 1 and files[0].stat().st_nlink == 2
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    assert len(client.calls) == 2


def test_private_directory_swap_never_redirects_v2_bytes(tmp_path, monkeypatch):
    import os
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected["adaptation_mappings"] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    target_root = tmp_path / "checkpoints"
    alternate = tmp_path / "alternate"
    relocated = tmp_path / "relocated-original"
    alternate.mkdir(mode=0o700)
    original_link = storage.os.link
    switched = [False]

    def swap_before_link(source, target, *args, **kwargs):
        if (not switched[0] and isinstance(target, str)
                and target.startswith("b13-checkpoint-v2-")
                and kwargs.get("dst_dir_fd") is not None):
            switched[0] = True
            os.rename(target_root, relocated)
            os.symlink(alternate, target_root)
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(storage.os, "link", swap_before_link)
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    assert switched[0] and len(client.calls) == 2
    assert list(alternate.iterdir()) == []
    assert claim_path(tmp_path / "execution", session, packet.window_id).exists()


def test_production_entry_rejects_generation_transport_substitution(tmp_path):
    import b13_alice_bounded_runner as alice
    from b13_alice_token_budget_preflight import LlamaClient, _server_root
    from short_drama.llm import OpenAICompatibleLLMClient
    tree, session, packet, _ = _fixture(tmp_path)
    preflight = LlamaClient(base_url=_server_root(session.runtime_config.base_url))
    client = OpenAICompatibleLLMClient(session.runtime_config, transport=object())
    with pytest.raises(B13RunnerError, match="matched live generation/tokenizer"):
        alice.run_b13_alice_bounded(
            store=tree.store, pointers=tree.pointers, source_snapshot=session.preparation.source_snapshot,
            runtime_config=session.runtime_config, preflight_client=preflight,
            llm_client=client, input_ceiling_tokens=10_000,
            execution_root=tmp_path / "execution", checkpoint_root=tmp_path / "checkpoints",
            max_new_windows=1, enable_feedback=True, selected_window_ids=(packet.window_id,))
    assert not preflight.requested_paths


def test_production_feedback_preflight_failure_does_not_leak_exception_graph(tmp_path, monkeypatch):
    import b13_alice_bounded_runner as alice
    from b13_alice_token_budget_preflight import LlamaClient, _server_root
    from short_drama.llm import OpenAICompatibleLLMClient
    tree, session, packet, _ = _fixture(tmp_path)
    preflight = LlamaClient(base_url=_server_root(session.runtime_config.base_url))
    client = OpenAICompatibleLLMClient(session.runtime_config)
    secret = "SOURCE_NOVEL_TEXT_NOT_FOR_EXCEPTION_CAUSE"

    def fail_preflight(**_kw):
        raise ValueError(secret)

    monkeypatch.setattr(alice, "prepare_b13_alice_execution_session", fail_preflight)
    with pytest.raises(B13RunnerError) as caught:
        alice.run_b13_alice_bounded(
            store=tree.store, pointers=tree.pointers, source_snapshot=session.preparation.source_snapshot,
            runtime_config=session.runtime_config, preflight_client=preflight,
            llm_client=client, input_ceiling_tokens=10_000,
            execution_root=tmp_path / "execution", checkpoint_root=tmp_path / "checkpoints",
            max_new_windows=1, enable_feedback=True, selected_window_ids=(packet.window_id,))
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert not preflight.requested_paths


def test_internal_feedback_rejects_fake_tokenizer_for_real_client(tmp_path):
    from short_drama.llm import OpenAICompatibleLLMClient
    tree, session, packet, _ = _fixture(tmp_path)
    client = OpenAICompatibleLLMClient(session.runtime_config)
    with pytest.raises(B13RunnerError):
        _run_b13_feedback_resumable(store=tree.store, pointers=tree.pointers, session=session,
            llm_client=client, execution_root=tmp_path / 'execution', checkpoint_root=tmp_path / 'checkpoints',
            max_new_windows=1, feedback_tokenizer=SyntheticTokenizer(), selected_window_ids=(packet.window_id,))
    assert not (tmp_path / 'execution').exists()


def test_internal_impl_rejects_caller_supplied_refresh(tmp_path):
    from short_drama.adaptation.disposition_runner import _run_b13_resumable_impl
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([], session.runtime_config)
    client.runtime_config = session.runtime_config
    with pytest.raises(B13RunnerError, match='feedback execution failed closed'):
        _run_b13_resumable_impl(store=tree.store, pointers=tree.pointers, session=session,
            llm_client=client, execution_root=tmp_path / 'execution', checkpoint_root=tmp_path / 'checkpoints',
            max_new_windows=1, enable_feedback=True, feedback_tokenizer=SyntheticTokenizer(),
            refresh_session=lambda: session, selected_window_ids=(packet.window_id,))
    assert not client.calls


def test_feedback_tokens_rechecked_after_runtime_refresh(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected['adaptation_mappings'] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    tokenizer = SyntheticTokenizer()
    def refresh():
        if len(client.calls) == 1:
            tokenizer.count = session.input_ceiling_tokens + 1
        return session
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,), refresh=refresh, tokenizer=tokenizer)
    assert len(client.calls) == 1


@pytest.mark.parametrize('replacement', ['symlink', 'directory'])
@pytest.mark.parametrize('existing', [True, False])
def test_v2_root_swap_during_legacy_resolution_does_not_redirect(tmp_path, monkeypatch, replacement, existing):
    import os
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected['adaptation_mappings'] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    root = tmp_path / 'checkpoints'
    if existing:
        root.mkdir(mode=0o700)
    alternate = tmp_path / 'alternate'
    alternate.mkdir(mode=0o700)
    old_root = tmp_path / 'old-root'
    original = storage._root
    switched = False
    def switch_after_check(path, *, create, **kwargs):
        nonlocal switched
        result = original(path, create=create, **kwargs)
        if create and not switched:
            switched = True
            os.rename(root, old_root)
            if replacement == 'symlink':
                os.symlink(alternate, root)
                return alternate
            os.rename(alternate, root)
            return root
        return result
    monkeypatch.setattr(storage, '_root', switch_after_check)
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    destination = alternate if replacement == 'symlink' else root
    assert switched and len(client.calls) == 2 and list(destination.iterdir()) == []


def test_v2_writer_detects_conflict_inserted_during_readback(tmp_path, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected['adaptation_mappings'] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    original = storage._read_private_v2_at
    def add_conflict(fd, name):
        raw = original(fd, name)
        other = name.replace('b13-checkpoint-v2-', 'b13-checkpoint-v1-', 1)
        import os
        other_fd = os.open(other, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd)
        os.close(other_fd)
        return raw
    monkeypatch.setattr(storage, '_read_private_v2_at', add_conflict)
    with pytest.raises(B13RunnerError):
        _run(tree, session, tmp_path, client, (packet.window_id,))
    assert len(client.calls) == 2
    assert inspect_claim(tmp_path / 'execution', session, packet.window_id).state is ClaimState.CLAIMED_UNRESOLVED


def test_complete_result_repr_excludes_window_text(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([(_proposal(packet), False)], session.runtime_config)
    result = _run(tree, session, tmp_path, client, (packet.window_id,))
    assert result.execution_result is not None
    assert packet.window_id not in repr(result)
    assert repr(packet.canonical_bytes()) not in repr(result)
    assert repr(packet.canonical_bytes()) not in repr(packet)
    assert repr(result.execution_result.accepted_windows[0].proposal.canonical_bytes()) not in repr(result.execution_result.accepted_windows[0].proposal)


@pytest.mark.parametrize('method', ['transport_post', 'generation'])
def test_internal_binding_rejects_instance_method_substitution(tmp_path, method):
    from b13_alice_token_budget_preflight import LlamaClient, _server_root
    from short_drama.llm import OpenAICompatibleLLMClient
    tree, session, packet, _ = _fixture(tmp_path)
    tokenizer = LlamaClient(base_url=_server_root(session.runtime_config.base_url))
    client = OpenAICompatibleLLMClient(session.runtime_config)
    if method == 'transport_post':
        client._transport.post = lambda **_kwargs: None
    else:
        client.generate_structured = lambda *_a, **_kw: None
    with pytest.raises(B13RunnerError):
        _run_b13_feedback_resumable(store=tree.store, pointers=tree.pointers, session=session,
            llm_client=client, execution_root=tmp_path / 'execution', checkpoint_root=tmp_path / 'checkpoints',
            max_new_windows=1, feedback_tokenizer=tokenizer, selected_window_ids=(packet.window_id,))
    assert not tokenizer.requested_paths and not (tmp_path / 'execution').exists()


def test_internal_binding_accepts_matched_unmodified_clients_without_io(tmp_path, monkeypatch):
    from b13_alice_token_budget_preflight import LlamaClient, _server_root
    import b13_alice_runner_preparation as preparation_module
    from short_drama.llm import OpenAICompatibleLLMClient
    from short_drama.adaptation.disposition_runner import _build_feedback_refresh
    tree, session, _packet, _ = _fixture(tmp_path)
    tokenizer = LlamaClient(base_url=_server_root(session.runtime_config.base_url))
    client = OpenAICompatibleLLMClient(session.runtime_config)
    observed = []
    def prepare(**kwargs):
        observed.append(kwargs)
        return session
    monkeypatch.setattr(preparation_module, 'prepare_b13_alice_execution_session', prepare)
    refresh = _build_feedback_refresh(store=tree.store, pointers=tree.pointers,
        session=session, llm_client=client, tokenizer=tokenizer)
    assert refresh() == session and not tokenizer.requested_paths
    assert observed[0]['client'] is tokenizer
    assert observed[0]['model_context_tokens'] == session.effective_n_ctx
    assert observed[0]['strategy_ref'] == session.strategy_ref


def test_refresh_rechecks_runtime_binding_before_preflight(tmp_path):
    from b13_alice_token_budget_preflight import LlamaClient, _server_root
    from short_drama.llm import OpenAICompatibleLLMClient
    from short_drama.adaptation.disposition_runner import _build_feedback_refresh
    tree, session, _packet, _ = _fixture(tmp_path)
    tokenizer = LlamaClient(base_url=_server_root(session.runtime_config.base_url))
    client = OpenAICompatibleLLMClient(session.runtime_config)
    refresh = _build_feedback_refresh(store=tree.store, pointers=tree.pointers,
        session=session, llm_client=client, tokenizer=tokenizer)
    client._transport.post = lambda **_kwargs: None
    with pytest.raises(B13RunnerError):
        refresh()
    assert not tokenizer.requested_paths


def test_binding_change_blocks_feedback_before_tokenizer(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    rejected = _proposal(packet)
    rejected['adaptation_mappings'] = []
    client = RecordingFakeClient([(rejected, False), (_proposal(packet), False)], session.runtime_config)
    client.runtime_config = session.runtime_config
    tokenizer = SyntheticTokenizer()
    def binding(**_kwargs):
        if client.calls:
            raise B13RunnerError('runtime binding changed')
    with patch('short_drama.adaptation.disposition_runner._build_feedback_refresh', return_value=lambda: session), patch(
            'short_drama.adaptation.disposition_runner._verify_feedback_client_binding', side_effect=binding):
        with pytest.raises(B13RunnerError):
            _run_b13_feedback_resumable(store=tree.store, pointers=tree.pointers, session=session,
                llm_client=client, execution_root=tmp_path / 'execution', checkpoint_root=tmp_path / 'checkpoints',
                max_new_windows=1, feedback_tokenizer=tokenizer, selected_window_ids=(packet.window_id,))
    assert len(client.calls) == 1 and tokenizer.messages is None


def test_internal_impl_feedback_failure_has_no_exception_graph(tmp_path):
    from short_drama.adaptation.disposition_runner import _run_b13_resumable_impl
    tree, session, packet, _ = _fixture(tmp_path)
    client = RecordingFakeClient([], session.runtime_config)
    client.runtime_config = session.runtime_config
    secret = 'SYNTHETIC_NOVEL_SECRET'
    def fail(*_a, **_kw):
        raise ValueError(secret)
    client.generate_structured = fail
    with patch('short_drama.adaptation.disposition_runner._build_feedback_refresh', return_value=lambda: session), patch(
            'short_drama.adaptation.disposition_runner._verify_feedback_client_binding'):
        with pytest.raises(B13RunnerError) as caught:
            _run_b13_resumable_impl(store=tree.store, pointers=tree.pointers, session=session,
                llm_client=client, execution_root=tmp_path / 'execution', checkpoint_root=tmp_path / 'checkpoints',
                max_new_windows=1, enable_feedback=True, feedback_tokenizer=SyntheticTokenizer(),
                selected_window_ids=(packet.window_id,))
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
