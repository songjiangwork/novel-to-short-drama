"""Isolated zero-provider root initialization and restart checks."""
import os
import stat

import pytest

from short_drama.adaptation import disposition_execution_claim as claim
from short_drama.adaptation import disposition_recovery_audit as audit
from short_drama.adaptation import disposition_recovery_root as root
from test_b13_disposition_checkpoint import _fixture
from test_b13_disposition_execution import _tree_bytes


def case(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    execution = tmp_path / 'execution'
    parent = tmp_path / 'checkpoint-parent'
    parent.mkdir(mode=0o700)
    checkpoint = parent / 'checkpoint'
    with claim.execution_guard(execution, store=tree.store, pointers=tree.pointers) as guard:
        claim.create_claim(guard=guard, store=tree.store, pointers=tree.pointers, session=session,
                           checkpoint_root=checkpoint, window_id=packet.window_id)
    return dict(store=tree.store, pointers=tree.pointers, session=session, execution=execution,
                checkpoint_root=checkpoint, window_id=packet.window_id)


def run(args):
    with claim.execution_guard(args['execution'], store=args['store'], pointers=args['pointers']) as guard:
        return root._initialize_root(guard=guard, store=args['store'], pointers=args['pointers'],
            session=args['session'], checkpoint_root=args['checkpoint_root'], window_id=args['window_id'],
            operator_id='operator', recorded_at='2026-01-01T00:00:00Z', reason='offline test', historically_absent=True)


def test_initialization_and_read_only_restart(tmp_path, monkeypatch):
    monkeypatch.setattr('short_drama.llm.LLMClient.generate_structured', lambda *_a, **_k: pytest.fail('provider called'))
    args = case(tmp_path)
    original = claim.claim_path(args['execution'], args['session'], args['window_id'])
    before = (original.read_bytes(), original.stat().st_ino, _tree_bytes(args['store'].root), _tree_bytes(args['pointers'].root))
    assert run(args) is root._RootState.INITIALIZED
    st = args['checkpoint_root'].stat()
    assert stat.S_IMODE(st.st_mode) == 0o700
    ident = claim._identity(args['session'], args['window_id'])
    from short_drama.artifacts import content_hash
    cid = content_hash(ident)
    i = audit.decode_record((args['execution'] / audit.record_name(cid, 'root_intent')).read_bytes(), 'root_intent')
    r = audit.decode_record((args['execution'] / audit.record_name(cid, 'root_result')).read_bytes(), 'root_result')
    assert r['intent_identity'] == i['content_hash'] and r['root_uid'] == st.st_uid and r['root_mode'] == 0o700
    assert audit.inspect_recovery(store=args['store'], pointers=args['pointers'], session=args['session'],
        execution_root=args['execution'], checkpoint_root=args['checkpoint_root'], window_id=args['window_id']) is audit.Audit.BLOCKED
    monkeypatch.setattr(root.os, 'mkdir', lambda *_a, **_kw: pytest.fail('restart attempted mkdir'))
    monkeypatch.setattr(root.os, 'link', lambda *_a, **_kw: pytest.fail('restart attempted publication'))
    assert run(args) is root._RootState.EVIDENCE_VERIFIED
    assert (original.read_bytes(), original.stat().st_ino, _tree_bytes(args['store'].root), _tree_bytes(args['pointers'].root)) == before


@pytest.mark.parametrize('point', ['intent_fsync', 'intent_link_after', 'intent_dir_fsync', 'mkdir_after', 'root_open', 'root_fstat', 'parent_fsync', 'result_fsync', 'result_link_after', 'result_dir_fsync', 'cleanup'])
def test_failure_preserves_evidence_and_restart_blocks(tmp_path, monkeypatch, point):
    args = case(tmp_path)
    real_fsync, real_link, real_mkdir, real_open, real_unlink, real_fstat = os.fsync, os.link, os.mkdir, os.open, os.unlink, os.fstat
    def fsync(fd):
        if point == 'intent_fsync' and stat.S_ISREG(os.fstat(fd).st_mode): raise OSError('injected')
        if point == 'result_fsync' and stat.S_ISREG(os.fstat(fd).st_mode) and 'root_result' in os.readlink(f'/proc/self/fd/{fd}'): raise OSError('injected')
        if point in ('intent_dir_fsync', 'result_dir_fsync') and stat.S_ISDIR(os.fstat(fd).st_mode) and os.fstat(fd).st_ino == args['execution'].stat().st_ino:
            kind = 'root_intent' if point == 'intent_dir_fsync' else 'root_result'
            if any(kind in name and name.endswith('.json') for name in os.listdir(args['execution'])): raise OSError('injected')
        if point == 'parent_fsync' and os.fstat(fd).st_ino == args['checkpoint_root'].parent.stat().st_ino: raise OSError('injected')
        return real_fsync(fd)
    def link(src, dst, **kw):
        real_link(src, dst, **kw)
        if (point == 'intent_link_after' and 'root_intent' in dst) or (point == 'result_link_after' and 'root_result' in dst):
            raise OSError('injected after link')
    def mkdir(name, *a, **kw):
        real_mkdir(name, *a, **kw)
        if point == 'mkdir_after' and name == args['checkpoint_root'].name: raise OSError('injected after mkdir')
    def fstat(fd):
        if point == 'root_fstat' and os.readlink(f'/proc/self/fd/{fd}') == str(args['checkpoint_root']):
            raise OSError('injected')
        return real_fstat(fd)
    def open_file(name, flags, *a, **kw):
        if point == 'root_open' and name == args['checkpoint_root'].name: raise OSError('injected')
        return real_open(name, flags, *a, **kw)
    def unlink(name, *a, **kw):
        if point == 'cleanup' and 'root_intent' in str(name): raise OSError('injected')
        return real_unlink(name, *a, **kw)
    with monkeypatch.context() as m:
        m.setattr(root.os, 'fsync', fsync); m.setattr(root.os, 'link', link); m.setattr(root.os, 'mkdir', mkdir)
        m.setattr(root.os, 'open', open_file); m.setattr(root.os, 'unlink', unlink); m.setattr(root.os, 'fstat', fstat)
        with pytest.raises(root._RootError): run(args)
    before = _tree_bytes(args['execution'])
    with pytest.raises(root._RootError): run(args)
    assert _tree_bytes(args['execution']) == before


def test_existing_root_and_unsafe_parent(tmp_path):
    args = case(tmp_path)
    args['checkpoint_root'].mkdir(mode=0o700)
    assert run(args) is root._RootState.EXISTING
    args['checkpoint_root'].parent.chmod(0o777)
    try:
        with pytest.raises(root._RootError): run(args)
    finally:
        args['checkpoint_root'].parent.chmod(0o700)


def test_result_schema_is_closed():
    binding = dict(claim_identity='a'*64, claim_bytes_hash='b'*64, preparation_identity='c'*64,
        window_id='window_0001', window_ordinal=1, request_identity='d'*64,
        semantic_request_hash='e'*64, authority_hash='f'*64)
    directory = dict(path='/tmp/checkpoint', dev=1, ino=2, type='directory')
    fields = dict(binding=binding, intent_identity='a'*64, parent=directory, root=directory,
                  root_uid=1, root_mode=0o700, absence_observed=True, parent_synced=True)
    assert audit.decode_record(audit.encode_record('root_result', fields), 'root_result')['root_uid'] == 1
    for bad in (dict(fields, root_mode=0o755), {k: v for k, v in fields.items() if k != 'root_uid'}):
        with pytest.raises(audit.RecoveryEvidenceError): audit.encode_record('root_result', bad)


def test_unsafe_paths_claim_and_slots_block_without_creation(tmp_path):
    args = case(tmp_path)
    original = args['checkpoint_root']
    claim_file = claim.claim_path(args['execution'], args['session'], args['window_id'])
    raw = claim_file.read_bytes()
    claim_file.write_bytes(b'invalid')
    with pytest.raises(root._RootError): run(args)
    claim_file.write_bytes(raw)
    original.symlink_to(tmp_path / 'missing')
    with pytest.raises(root._RootError): run(args)
    original.unlink()
    args['checkpoint_root'] = 'relative/checkpoint'
    with pytest.raises(root._RootError): run(args)
    args['checkpoint_root'] = str(original.parent) + '/./checkpoint'
    with pytest.raises(root._RootError): run(args)
    args['checkpoint_root'] = original / 'nested'
    with pytest.raises(root._RootError): run(args)
    unsafe = tmp_path / 'unsafe-ancestor'
    unsafe.mkdir(mode=0o700)
    (unsafe / 'parent').mkdir(mode=0o700)
    unsafe.chmod(0o777)
    args['checkpoint_root'] = unsafe / 'parent' / 'checkpoint'
    try:
        with pytest.raises(root._RootError): run(args)
    finally:
        unsafe.chmod(0o700)
    args['checkpoint_root'] = args['store'].root / 'nested'
    with pytest.raises(root._RootError): run(args)
    args['checkpoint_root'] = original
    from short_drama.artifacts import content_hash
    cid = content_hash(claim._identity(args['session'], args['window_id']))
    remnant = args['execution'] / ('.' + audit.record_name(cid, 'root_intent') + '.tmp')
    remnant.touch(mode=0o600)
    with pytest.raises(root._RootError): run(args)
    assert not original.exists()


def test_unexpected_eexist_never_adopted(tmp_path, monkeypatch):
    args = case(tmp_path)
    real = os.mkdir
    def competing(name, *a, **kw):
        if name == args['checkpoint_root'].name:
            real(name, *a, **kw)
        return real(name, *a, **kw)
    with monkeypatch.context() as m:
        m.setattr(root.os, 'mkdir', competing)
        with pytest.raises(root._RootError): run(args)
    assert args['checkpoint_root'].is_dir()
    with pytest.raises(root._RootError): run(args)


def test_complete_evidence_rejects_replaced_root(tmp_path):
    args = case(tmp_path)
    assert run(args) is root._RootState.INITIALIZED
    args['checkpoint_root'].rename(args['checkpoint_root'].with_name('original'))
    args['checkpoint_root'].mkdir(mode=0o700)
    with pytest.raises(root._RootError): run(args)


def test_valid_checkpoint_takes_precedence_and_audit_reads_linkage(tmp_path):
    from test_b13_disposition_checkpoint import _accepted
    from short_drama.adaptation.disposition_checkpoint_storage import save_checkpoint
    args = case(tmp_path)
    assert run(args) is root._RootState.INITIALIZED
    session = args['session']
    packet = next(p for p in session.preparation.packets if p.window_id == args['window_id'])
    semantic = next(s for s in session.semantic_preparations if s.packet.window_id == args['window_id'])
    checkpoint = save_checkpoint(args['checkpoint_root'], _accepted(packet, semantic, session.runtime_config), session)
    assert run(args) is root._RootState.VERIFIED_REUSE
    assert audit.inspect_recovery(store=args['store'], pointers=args['pointers'], session=session,
        execution_root=args['execution'], checkpoint_root=args['checkpoint_root'], window_id=args['window_id']) is audit.Audit.ORDINARY_AUDIT
    checkpoint.write_bytes(b'invalid')
    with pytest.raises(root._RootError): run(args)


def test_final_read_error_does_not_claim_prior_success(tmp_path, monkeypatch):
    args = case(tmp_path)
    real = audit._read_slot
    def read(directory, fd, name):
        if name.endswith('root_result.json') and (args['execution'] / name).exists():
            raise OSError('injected final read error')
        return real(directory, fd, name)
    with monkeypatch.context() as m:
        m.setattr(audit, '_read_slot', read)
        with pytest.raises(root._RootError): run(args)
    assert run(args) is root._RootState.EVIDENCE_VERIFIED


def test_guard_excludes_other_operator(tmp_path):
    args = case(tmp_path)
    with claim.execution_guard(args['execution'], store=args['store'], pointers=args['pointers']) as guard:
        with pytest.raises(claim.B13ExecutionClaimError):
            with claim.execution_guard(args['execution'], store=args['store'], pointers=args['pointers']):
                pass
        guard.verify()
    assert not args['checkpoint_root'].exists()


def test_replacement_after_open_blocks(tmp_path, monkeypatch):
    args = case(tmp_path)
    real_open = os.open
    def replace_after_open(name, flags, *a, **kw):
        fd = real_open(name, flags, *a, **kw)
        if name == args['checkpoint_root'].name and kw.get('dir_fd') is not None:
            args['checkpoint_root'].rename(args['checkpoint_root'].with_name('displaced'))
            args['checkpoint_root'].mkdir(mode=0o700)
        return fd
    with monkeypatch.context() as m:
        m.setattr(root.os, 'open', replace_after_open)
        with pytest.raises(root._RootError): run(args)
    with pytest.raises(root._RootError): run(args)


def test_existing_evidence_rejects_other_selection_mode(tmp_path):
    from short_drama.adaptation import prepare_b13_disposition_windows, build_b13_window_semantic_preparation
    from test_b13_disposition_checkpoint import _session
    args = case(tmp_path)
    assert run(args) is root._RootState.INITIALIZED
    current = args['session']
    selection = current.preparation.strategy_selection
    prep = prepare_b13_disposition_windows(store=args['store'], pointers=args['pointers'],
        source_snapshot=current.preparation.source_snapshot,
        adaptation_target_id=selection.artifact.strategy.resolved_target.adaptation_target_id,
        strategy_ref=selection.strategy_ref)
    requests = tuple(build_b13_window_semantic_preparation(p, runtime_config=current.runtime_config) for p in prep.packets)
    args['session'] = _session(prep, requests, current.runtime_config)
    with pytest.raises(root._RootError): run(args)
