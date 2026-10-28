"""Offline tests use only fake sessions and isolated temporary roots."""
import os

import pytest

from short_drama.adaptation import disposition_execution_claim as claim
from short_drama.adaptation import disposition_recovery_audit as audit
from short_drama.adaptation import disposition_recovery_storage as storage
from test_b13_recovery_audit import setup_case, decision


def case(tmp_path):
    args, binding, roots, _ = setup_case(tmp_path)
    decision(args, binding, roots)
    path = args['execution_root'] / audit.record_name(binding['claim_identity'], 'decision')
    raw = path.read_bytes()
    path.unlink()
    return args, path, raw


def persist(args, guard, raw):
    return storage._persist_decision(guard=guard, store=args['store'], pointers=args['pointers'],
        session=args['session'], checkpoint_root=args['checkpoint_root'], window_id=args['window_id'], raw=raw)


def guard(args):
    return claim.execution_guard(args['execution_root'], store=args['store'], pointers=args['pointers'])


def test_roundtrip_and_existing(tmp_path, monkeypatch):
    monkeypatch.setattr('short_drama.llm.LLMClient.generate_structured', lambda *_a, **_k: pytest.fail('provider called'))
    args, path, raw = case(tmp_path)
    original = claim.claim_path(args['execution_root'], args['session'], args['window_id'])
    before = (original.read_bytes(), original.stat().st_ino)
    checkpoint = {p.name: p.read_bytes() for p in args['checkpoint_root'].iterdir()}
    from test_b13_disposition_execution import _tree_bytes
    protected = (_tree_bytes(args['store'].root), _tree_bytes(args['pointers'].root))
    with guard(args) as g:
        assert persist(args, g, raw) == storage._Result.PERSISTED
        assert persist(args, g, raw) == storage._Result.EXISTS_VERIFIED
        with pytest.raises(claim.B13ExecutionClaimError):
            with guard(args): pass
    assert path.read_bytes() == raw
    assert (original.read_bytes(), original.stat().st_ino) == before
    assert {p.name: p.read_bytes() for p in args['checkpoint_root'].iterdir()} == checkpoint
    assert (_tree_bytes(args['store'].root), _tree_bytes(args['pointers'].root)) == protected
    assert audit.inspect_recovery(**args) == audit.Audit.ORDINARY_AUDIT


@pytest.mark.parametrize('bad', ['missing_guard', 'inactive', 'claim', 'binding', 'roots', 'noncanonical', 'missing_checkpoint'])
def test_preconditions_do_not_write(tmp_path, bad):
    args, path, raw = case(tmp_path)
    g = guard(args)
    if bad == 'missing_guard': g = None
    elif bad == 'inactive': pass
    else:
        g.__enter__()
        if bad == 'claim': claim.claim_path(args['execution_root'], args['session'], args['window_id']).write_bytes(b'bad')
        if bad == 'binding':
            obj = audit.decode_record(raw, 'decision'); fields = {k: v for k, v in obj.items() if k not in ('record_type', 'schema_version', 'content_hash')}
            fields['binding']['authority_hash'] = 'a' * 64; raw = audit.encode_record('decision', fields)
        if bad == 'roots': args['checkpoint_root'] = args['execution_root']
        if bad == 'noncanonical': raw += b' '
        if bad == 'missing_checkpoint': args['checkpoint_root'] = tmp_path / 'absent'
    before = set(args['execution_root'].iterdir())
    try:
        with pytest.raises(storage._StorageError) as err: persist(args, g, raw)
        assert not err.value.link_attempted
        assert set(args['execution_root'].iterdir()) == before
    finally:
        if g is not None and g._locked: g.__exit__(None, None, None)


@pytest.mark.parametrize('kind', ['conflict', 'symlink', 'hardlink', 'mode', 'remnant'])
def test_unsafe_existing_slots_block(tmp_path, kind):
    args, path, raw = case(tmp_path)
    if kind == 'remnant': (path.parent / ('.' + path.name + '.tmp')).touch(mode=0o600)
    elif kind == 'symlink': path.symlink_to(tmp_path / 'missing')
    else:
        path.write_bytes(raw if kind != 'conflict' else b'bad'); path.chmod(0o600)
        if kind == 'mode': path.chmod(0o644)
        if kind == 'hardlink': os.link(path, tmp_path / 'alias')
    with guard(args) as g:
        with pytest.raises(storage._StorageError): persist(args, g, raw)
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE


@pytest.mark.parametrize('point', ['chmod', 'write', 'file_sync', 'link', 'dir_sync', 'unlink', 'final_read'])
def test_failures_retain_or_surface_ambiguous_evidence(tmp_path, monkeypatch, point):
    args, path, raw = case(tmp_path)
    with guard(args) as g:
        if point in ('chmod', 'write', 'link', 'unlink'):
            monkeypatch.setattr(storage.os, {'chmod':'fchmod', 'write':'write', 'link':'link', 'unlink':'unlink'}[point],
                                lambda *_a, **_k: (_ for _ in ()).throw(OSError('injected')))
        elif point in ('file_sync', 'dir_sync'):
            real = os.fsync
            def fsync(fd):
                if (fd == g._dirfd) == (point == 'dir_sync'): raise OSError('injected')
                return real(fd)
            monkeypatch.setattr(storage.os, 'fsync', fsync)
        else:
            real = audit._read_slot
            def read(directory, fd, name):
                if name == path.name and path.exists(): raise OSError('injected')
                return real(directory, fd, name)
            monkeypatch.setattr(audit, '_read_slot', read)
        with pytest.raises(storage._StorageError) as err: persist(args, g, raw)
        assert err.value.link_attempted == (point in ('link', 'dir_sync', 'unlink', 'final_read'))
    monkeypatch.undo()
    # Before cleanup the tracked remnant blocks audit and every later writer.
    if point != 'final_read':
        assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE
        with guard(args) as g:
            with pytest.raises(storage._StorageError): persist(args, g, raw)
    else:
        assert path.read_bytes() == raw
        with guard(args) as g: assert persist(args, g, raw) == storage._Result.EXISTS_VERIFIED


def test_link_success_then_exception_blocks_restart(tmp_path, monkeypatch):
    args, path, raw = case(tmp_path)
    real = os.link
    with guard(args) as g:
        def link_then_crash(*a, **kw):
            real(*a, **kw)
            raise OSError('simulated crash after link')
        monkeypatch.setattr(storage.os, 'link', link_then_crash)
        with pytest.raises(storage._StorageError) as err: persist(args, g, raw)
        assert err.value.link_attempted
    monkeypatch.undo()
    assert path.read_bytes() == raw
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE
    with guard(args) as g:
        with pytest.raises(storage._StorageError): persist(args, g, raw)


def test_short_writes_are_completed(tmp_path, monkeypatch):
    args, path, raw = case(tmp_path)
    real = os.write
    with guard(args) as g:
        monkeypatch.setattr(storage.os, 'write', lambda fd, data: real(fd, data[:max(1, len(data) // 2)]))
        assert persist(args, g, raw) == storage._Result.PERSISTED
    assert path.read_bytes() == raw


def test_guard_store_mismatch_and_fork_identity(tmp_path):
    args, path, raw = case(tmp_path)
    with guard(args) as g:
        with pytest.raises(storage._StorageError):
            storage._persist_decision(guard=g, store=object(), pointers=args['pointers'], session=args['session'],
                checkpoint_root=args['checkpoint_root'], window_id=args['window_id'], raw=raw)
        original_pid = g._pid
        g._pid = -1
        try:
            with pytest.raises(storage._StorageError): persist(args, g, raw)
        finally:
            g._pid = original_pid
    assert not path.exists()


def _try_guard_in_child(args, queue):
    try:
        with guard(args):
            queue.put('acquired')
    except claim.B13ExecutionClaimError:
        queue.put('blocked')


def test_other_process_cannot_unlock_guard(tmp_path):
    import multiprocessing
    args, path, raw = case(tmp_path)
    ctx = multiprocessing.get_context('fork')
    with guard(args) as g:
        queue = ctx.Queue()
        process = ctx.Process(target=_try_guard_in_child, args=(args, queue))
        process.start(); process.join(30)
        assert process.exitcode == 0
        assert queue.get(timeout=2) == 'blocked'
        assert persist(args, g, raw) == storage._Result.PERSISTED
        g.verify()
    with guard(args) as g:
        assert persist(args, g, raw) == storage._Result.EXISTS_VERIFIED
    assert path.read_bytes() == raw


def test_replacement_during_link_blocks(tmp_path, monkeypatch):
    args, path, raw = case(tmp_path)
    real = os.link
    with guard(args) as g:
        def replace_after_link(*a, **kw):
            real(*a, **kw)
            path.unlink()
            path.write_bytes(raw)
            path.chmod(0o600)
        monkeypatch.setattr(storage.os, 'link', replace_after_link)
        with pytest.raises(storage._StorageError) as err: persist(args, g, raw)
        assert err.value.link_attempted
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE


def test_wrong_execution_root_blocks(tmp_path):
    args, path, raw = case(tmp_path)
    other = tmp_path / 'other-execution'
    with claim.execution_guard(other, store=args['store'], pointers=args['pointers']) as g:
        before = set(other.iterdir())
        with pytest.raises(storage._StorageError): persist(args, g, raw)
        assert set(other.iterdir()) == before
    assert not path.exists()


def test_runner_reuses_checkpoint_without_generation_under_guard(tmp_path):
    from short_drama.adaptation.disposition_runner import run_b13_resumable
    from test_adaptation_strategy_execution import _FakeClient
    args, _, raw = case(tmp_path)
    client = _FakeClient([], args['session'].runtime_config)
    with guard(args) as g:
        assert persist(args, g, raw) == storage._Result.PERSISTED
        result = run_b13_resumable(store=args['store'], pointers=args['pointers'], session=args['session'],
            llm_client=client, execution_root=args['execution_root'], checkpoint_root=args['checkpoint_root'],
            max_new_windows=1)
        assert result.reused_windows == 1 and result.generated_windows == 0
        assert not client.calls
        g.verify()


def test_guard_parent_replaced_before_storage_blocks(tmp_path):
    args, path, raw = case(tmp_path)
    parent = tmp_path / 'parent'
    parent.mkdir()
    args['execution_root'].rename(parent / 'execution')
    args['execution_root'] = parent / 'execution'
    obj = audit.decode_record(raw, 'decision')
    fields = {k: v for k, v in obj.items() if k not in ('record_type', 'schema_version', 'content_hash')}
    fields['roots']['execution'] = audit._root_identity(args['execution_root'])
    raw = audit.encode_record('decision', fields)
    path = args['execution_root'] / path.name
    with guard(args) as g:
        moved = tmp_path / 'moved'
        parent.rename(moved)
        parent.mkdir()
        (moved / 'execution').rename(parent / 'execution')
        with pytest.raises(storage._StorageError): persist(args, g, raw)
        assert not path.exists()


def test_selection_mode_mismatch_writes_nothing(tmp_path):
    from short_drama.adaptation import prepare_b13_disposition_windows, build_b13_window_semantic_preparation
    from test_b13_disposition_checkpoint import _session
    args, path, raw = case(tmp_path)
    current = args['session']
    selection = current.preparation.strategy_selection
    prep = prepare_b13_disposition_windows(store=args['store'], pointers=args['pointers'],
        source_snapshot=current.preparation.source_snapshot,
        adaptation_target_id=selection.artifact.strategy.resolved_target.adaptation_target_id,
        strategy_ref=selection.strategy_ref)
    requests = tuple(build_b13_window_semantic_preparation(p, runtime_config=current.runtime_config) for p in prep.packets)
    args['session'] = _session(prep, requests, current.runtime_config)
    with guard(args) as g:
        before = set(args['execution_root'].iterdir())
        with pytest.raises(storage._StorageError): persist(args, g, raw)
        assert set(args['execution_root'].iterdir()) == before
    assert not path.exists()


def test_partial_write_then_failure_retains_remnant(tmp_path, monkeypatch):
    args, path, raw = case(tmp_path)
    real = os.write
    count = 0
    def partial(fd, data):
        nonlocal count
        count += 1
        if count == 1: return real(fd, data[:7])
        raise OSError('injected second write failure')
    with guard(args) as g:
        monkeypatch.setattr(storage.os, 'write', partial)
        with pytest.raises(storage._StorageError) as err: persist(args, g, raw)
        assert not err.value.link_attempted
    monkeypatch.undo()
    assert not path.exists()
    assert audit.inspect_recovery(**args) == audit.Audit.AUDIT_INCOMPLETE
