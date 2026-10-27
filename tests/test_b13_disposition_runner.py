"""Offline orchestration tests using synthetic stores and fake clients."""
from dataclasses import replace

import pytest

import short_drama.adaptation.disposition_runner as runner
import short_drama.adaptation.disposition_execution as execution
from short_drama.adaptation.disposition_checkpoint_storage import load_checkpoint, save_checkpoint
from short_drama.adaptation.disposition_execution_claim import claim_path, create_claim, execution_guard
from test_adaptation_strategy_execution import _FakeClient
from test_b13_disposition_checkpoint import _accepted, _fixture, _session
from test_b13_disposition_execution import (_two_window_inputs, _allow_synthetic_two_window_preflight,
                                             _AttemptCountingFakeClient, _tree_bytes)
from test_b13_disposition_resume import _move_a6
from test_b13_window_semantic import _proposal


def _run(tmp_path, tree, session, client, limit=1):
    return runner.run_b13_resumable(store=tree.store, pointers=tree.pointers, session=session,
        llm_client=client, execution_root=tmp_path / 'execution', checkpoint_root=tmp_path / 'checkpoints',
        max_new_windows=limit)


def _two(tmp_path, monkeypatch):
    tree, prep, requests, _, runtime, _, _ = _two_window_inputs(tmp_path)
    _allow_synthetic_two_window_preflight(monkeypatch, prep)
    return tree, _session(prep, requests, runtime)


def test_bounded_non_prefix_reuse_and_rerun(tmp_path, monkeypatch):
    tree, session = _two(tmp_path, monkeypatch)
    p = session.preparation.packets
    s = session.semantic_preparations
    accepted = _accepted(p[1], s[1], session.runtime_config)
    save_checkpoint(tmp_path / 'checkpoints', accepted, session)
    before = (_tree_bytes(tree.store.root), _tree_bytes(tree.pointers.root))
    client = _FakeClient([(_proposal(p[0]), False)], session.runtime_config)
    result = _run(tmp_path, tree, session, client)
    assert result.status is runner.B13RunnerStatus.COMPLETE
    assert (result.reused_windows, result.generated_windows, result.new_semantic_calls,
            result.new_provider_attempts, result.remaining_windows) == (1, 1, 1, 1, 0)
    assert result.execution_result.accepted_windows[1] == accepted
    again = _run(tmp_path, tree, session, client)
    assert again.reused_windows == 2 and again.generated_windows == 0 and len(client.calls) == 1
    assert before == (_tree_bytes(tree.store.root), _tree_bytes(tree.pointers.root))


def test_partial_checkpoint_saved_before_next_run(tmp_path, monkeypatch):
    tree, session = _two(tmp_path, monkeypatch)
    p = session.preparation.packets
    client = _FakeClient([(_proposal(x), False) for x in p], session.runtime_config)
    first = _run(tmp_path, tree, session, client)
    assert first.status is runner.B13RunnerStatus.BOUNDED_PROGRESS
    assert first.execution_result is None and first.remaining_windows == 1
    assert load_checkpoint(tmp_path / 'checkpoints', session, p[0].window_id) is not None
    second = _run(tmp_path, tree, session, client)
    assert second.status is runner.B13RunnerStatus.COMPLETE
    assert second.reused_windows == 1 and second.generated_windows == 1


def test_all_reuse_no_writes(tmp_path):
    tree, session, packet, semantic = _fixture(tmp_path)
    accepted = _accepted(packet, semantic, session.runtime_config)
    save_checkpoint(tmp_path / 'checkpoints', accepted, session)
    before = _tree_bytes(tmp_path)
    client = _FakeClient([], session.runtime_config)
    result = _run(tmp_path, tree, session, client)
    assert result.execution_result.accepted_windows == (accepted,)
    assert not client.calls and not (tmp_path / 'execution').exists()
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize('invalid', [False, True])
def test_any_claim_blocks_entire_batch(tmp_path, monkeypatch, invalid):
    tree, session = _two(tmp_path, monkeypatch)
    window = session.preparation.packets[1].window_id
    with execution_guard(tmp_path / 'execution', store=tree.store, pointers=tree.pointers) as guard:
        create_claim(guard=guard, store=tree.store, pointers=tree.pointers, session=session,
                     checkpoint_root=tmp_path / 'checkpoints', window_id=window)
    if invalid:
        claim_path(tmp_path / 'execution', session, window).write_bytes(b'bad')
    client = _FakeClient([], session.runtime_config)
    with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)
    assert not client.calls


def test_lock_blocks_generation(tmp_path):
    tree, session, _, _ = _fixture(tmp_path)
    client = _FakeClient([], session.runtime_config)
    with execution_guard(tmp_path / 'execution', store=tree.store, pointers=tree.pointers):
        with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)
    assert not client.calls


@pytest.mark.parametrize('failure', ['provider', 'exhaustion', 'save', 'after_claim', 'after_response'])
def test_failure_retains_claim_and_blocks_replay(tmp_path, monkeypatch, failure):
    tree, session, packet, _ = _fixture(tmp_path)
    invalid = _proposal(packet)
    invalid['event_dispositions'][0]['disposition'] = 'omit'
    client = _FakeClient([(invalid, False), (invalid, False)] if failure == 'exhaustion'
                         else [(_proposal(packet), False)], session.runtime_config)
    if failure == 'provider':
        def fail(*a, **kw): raise RuntimeError('transport')
        monkeypatch.setattr(client, 'generate_structured', fail)
    elif failure == 'save':
        monkeypatch.setattr(runner, 'save_checkpoint', lambda *a: (_ for _ in ()).throw(OSError('fsync')))
    elif failure == 'after_claim':
        monkeypatch.setattr(runner, '_execute_one_b13_window', lambda **kw: (_ for _ in ()).throw(RuntimeError('crash')))
    elif failure == 'after_response':
        monkeypatch.setattr(runner, 'save_checkpoint', lambda *a: (_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)
    assert claim_path(tmp_path / 'execution', session, packet.window_id).exists()
    with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)


def test_retry_accounting(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    invalid = _proposal(packet)
    invalid['event_dispositions'][0]['disposition'] = 'omit'
    client = _AttemptCountingFakeClient([(invalid, False), (_proposal(packet), False)], session.runtime_config, (3, 2))
    result = _run(tmp_path, tree, session, client)
    assert (result.new_semantic_calls, result.new_provider_attempts, result.historical_semantic_rounds) == (2, 5, 2)
    assert result.execution_result.accepted_windows[0].rejected_semantic_rounds[0].provider_attempts == 3


def test_authority_changes_during_call(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    class Moving(_FakeClient):
        def generate_structured(self, *args, **kwargs):
            result = super().generate_structured(*args, **kwargs)
            _move_a6(tree, session)
            return result
    client = Moving([(_proposal(packet), False)], session.runtime_config)
    with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)
    assert claim_path(tmp_path / 'execution', session, packet.window_id).exists()
    assert not (tmp_path / 'checkpoints').exists()


@pytest.mark.parametrize('case', ['unknown', 'budget', 'mismatch', 'limit', 'overlap'])
def test_preflight_failure_before_generation(tmp_path, case):
    tree, session, _, _ = _fixture(tmp_path)
    client = _FakeClient([], session.runtime_config)
    if case == 'unknown': session = replace(session, preflight_verdict='UNKNOWN')
    if case == 'budget': session = replace(session, input_ceiling_tokens=1, budgets=(replace(session.budgets[0], input_ceiling_tokens=1),))
    if case == 'mismatch': client.runtime_config = replace(session.runtime_config, request_model='wrong')
    with pytest.raises(runner.B13RunnerError):
        if case == 'overlap':
            runner.run_b13_resumable(store=tree.store, pointers=tree.pointers, session=session, llm_client=client,
                execution_root=tree.store.root / 'bad', checkpoint_root=tmp_path / 'checkpoints', max_new_windows=1)
        else: _run(tmp_path, tree, session, client, 2 if case == 'limit' else 1)
    assert not client.calls


def test_checkpoint_is_durable_before_next_dispatch(tmp_path, monkeypatch):
    tree, session = _two(tmp_path, monkeypatch)
    packets = session.preparation.packets
    class Inspecting(_FakeClient):
        def generate_structured(self, *args, **kwargs):
            if self.calls:
                assert load_checkpoint(tmp_path / 'checkpoints', session, packets[0].window_id)
            return super().generate_structured(*args, **kwargs)
    client = Inspecting([(_proposal(p), False) for p in packets], session.runtime_config)
    assert _run(tmp_path, tree, session, client, 2).generated_windows == 2


@pytest.mark.parametrize('point', ['claim', 'response', 'checkpoint'])
def test_process_death_boundaries(tmp_path, point):
    import os
    tree, session, packet, _ = _fixture(tmp_path)
    pid = os.fork()
    if pid == 0:
        if point == 'claim':
            runner._execute_one_b13_window = lambda **kw: os._exit(71)
        elif point == 'response':
            runner.save_checkpoint = lambda *a: os._exit(72)
        else:
            original = runner.save_checkpoint
            def save_then_die(*args):
                original(*args)
                os._exit(73)
            runner.save_checkpoint = save_then_die
        client = _FakeClient([(_proposal(packet), False)], session.runtime_config)
        _run(tmp_path, tree, session, client)
        os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == {'claim': 71, 'response': 72, 'checkpoint': 73}[point]
    client = _FakeClient([], session.runtime_config)
    if point == 'checkpoint':
        assert _run(tmp_path, tree, session, client).status is runner.B13RunnerStatus.COMPLETE
    else:
        with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)
    assert not client.calls


@pytest.mark.parametrize('point', ['file', 'directory'])
def test_checkpoint_fsync_failure_prevents_replay(tmp_path, monkeypatch, point):
    import short_drama.adaptation.disposition_checkpoint_storage as storage
    tree, session, packet, _ = _fixture(tmp_path)
    (tmp_path / 'checkpoints').mkdir(mode=0o700)
    original = storage.os.fsync
    def fsync(fd):
        import os, stat
        mode = os.fstat(fd).st_mode
        # Only fail checkpoint sync, after the execution claim is published.
        if claim_path(tmp_path / 'execution', session, packet.window_id).exists():
            if (point == 'file' and stat.S_ISREG(mode)) or (point == 'directory' and stat.S_ISDIR(mode)):
                raise OSError('injected fsync failure')
        return original(fd)
    save = runner.save_checkpoint
    def failing_save(*args):
        monkeypatch.setattr(storage.os, 'fsync', fsync)
        return save(*args)
    monkeypatch.setattr(runner, 'save_checkpoint', failing_save)
    client = _FakeClient([(_proposal(packet), False)], session.runtime_config)
    with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)
    monkeypatch.setattr(storage.os, 'fsync', original)
    assert claim_path(tmp_path / 'execution', session, packet.window_id).exists()
    again = _FakeClient([], session.runtime_config)
    try:
        result = _run(tmp_path, tree, session, again)
        assert point == 'directory' and result.status is runner.B13RunnerStatus.COMPLETE
    except runner.B13RunnerError:
        assert point == 'file'
    assert not again.calls


def test_authority_change_before_claim(tmp_path, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    original = runner.create_claim
    def moved(**kwargs):
        _move_a6(tree, session)
        return original(**kwargs)
    monkeypatch.setattr(runner, 'create_claim', moved)
    client = _FakeClient([], session.runtime_config)
    with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)
    assert not client.calls and not claim_path(tmp_path / 'execution', session, packet.window_id).exists()


def test_authority_change_before_checkpoint(tmp_path, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    original = runner._execute_one_b13_window
    def moved(**kwargs):
        accepted = original(**kwargs)
        _move_a6(tree, session)
        return accepted
    monkeypatch.setattr(runner, '_execute_one_b13_window', moved)
    client = _FakeClient([(_proposal(packet), False)], session.runtime_config)
    with pytest.raises(runner.B13RunnerError): _run(tmp_path, tree, session, client)
    assert claim_path(tmp_path / 'execution', session, packet.window_id).exists()
    assert not (tmp_path / 'checkpoints').exists()


def test_fourteen_checkpointed_windows_are_read_only(tmp_path, monkeypatch):
    from short_drama.adaptation import (B12StrategySelection, B13DispositionPreparation,
        B13DispositionPacketMeasurement, B13DispositionWindowPacket, build_b13_window_semantic_preparation)
    from short_drama.adaptation.target import CanonicalTargetReference
    from short_drama.artifacts import canonical_json_bytes
    tree, old, _, _, runtime, _, _ = _two_window_inputs(tmp_path)
    base = old.packets[0]
    ids = tuple(f'evt_{n:06d}' for n in range(1, 15))
    source = replace(base.source_snapshot, canonical_references=tuple(CanonicalTargetReference('event', x) for x in ids))
    strategy = replace(base.strategy_selection.artifact.strategy,
        resolved_target=replace(base.strategy_selection.artifact.strategy.resolved_target, source_snapshot=source))
    selection = B12StrategySelection(base.strategy_selection.strategy_ref,
        replace(base.strategy_selection.artifact, source_snapshot=source, strategy=strategy),
        base.strategy_selection.current_pointer_ref)
    packets = []
    for i, event in enumerate(ids, 1):
        value = base.to_dict()
        value['source_snapshot'] = source.to_dict()
        value['global_adaptation_strategy']['strategy'] = strategy.to_dict()
        value['global_adaptation_strategy']['strategy_hash'] = strategy.content_hash()
        value['window'] = {'window_id': f'window_{i:04d}', 'window_ordinal': i,
                           'owned_event_refs': [event], 'context_event_refs': []}
        value['owned_source_events'] = [{'event_id': event, 'narrative_order': i}]
        value['context_source_events'] = []
        packets.append(B13DispositionWindowPacket(source, selection, f'window_{i:04d}', i, canonical_json_bytes(value)))
    packets = tuple(packets)
    prep = B13DispositionPreparation(source, selection, packets,
        tuple(B13DispositionPacketMeasurement(p.window_id, len(p.canonical_bytes()), 1, p.content_hash()) for p in packets),
        ids, old.preparation_identity)
    _allow_synthetic_two_window_preflight(monkeypatch, prep)
    requests = tuple(build_b13_window_semantic_preparation(p, runtime_config=runtime) for p in packets)
    session = _session(prep, requests, runtime)
    for p, s in zip(packets, requests):
        save_checkpoint(tmp_path / 'checkpoints', _accepted(p, s, runtime), session)
    before = _tree_bytes(tmp_path)
    client = _FakeClient([], runtime)
    result = _run(tmp_path, tree, session, client, 14)
    assert result.status is runner.B13RunnerStatus.COMPLETE and result.reused_windows == 14
    assert len(result.execution_result.accepted_windows) == 14 and not client.calls
    assert not (tmp_path / 'execution').exists() and before == _tree_bytes(tmp_path)


def test_alice_entry_prepares_session_in_same_call(tmp_path, monkeypatch):
    import b13_alice_bounded_runner as alice
    tree, session, packet, semantic = _fixture(tmp_path)
    accepted = _accepted(packet, semantic, session.runtime_config)
    save_checkpoint(tmp_path / 'checkpoints', accepted, session)
    seen = []
    def prepare(**kwargs):
        seen.append(kwargs)
        return session
    monkeypatch.setattr(alice, 'prepare_b13_alice_execution_session', prepare)
    fake = _FakeClient([], session.runtime_config)
    result = alice.run_b13_alice_bounded(store=tree.store, pointers=tree.pointers,
        source_snapshot=session.preparation.source_snapshot, runtime_config=session.runtime_config,
        preflight_client=object(), llm_client=fake, input_ceiling_tokens=1000000,
        execution_root=tmp_path / 'execution', checkpoint_root=tmp_path / 'checkpoints', max_new_windows=1)
    assert len(seen) == 1 and result.status is runner.B13RunnerStatus.COMPLETE
    assert not fake.calls
