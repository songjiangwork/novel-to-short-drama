"""Offline crash and filesystem boundaries for B1.3 execution claims."""
import multiprocessing
import os
from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore

import pytest

from short_drama.adaptation import disposition_execution_claim as claim
from short_drama.adaptation.disposition_checkpoint_storage import save_checkpoint
from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_checkpoint import _accepted, _fixture, _broken_session



@pytest.fixture
def stores(tmp_path):
    store = FileArtifactStore(tmp_path / 'artifacts')
    return store, FilePointerStore(tmp_path / 'pointers', store)


def _create(tree, session, packet, guard, root):
    return claim.create_claim(guard=guard, store=tree.store, pointers=tree.pointers,
                              session=session, checkpoint_root=root, window_id=packet.window_id)


def _compete(root, store, pointers, queue):
    try:
        with claim.execution_guard(root, store=store, pointers=pointers):
            queue.put('acquired')
    except claim.B13ExecutionClaimError:
        queue.put('blocked')


def test_missing_canonical_claim_and_no_replay(tmp_path, monkeypatch):
    tree, session, packet, _ = _fixture(tmp_path)
    root = tmp_path / 'execution'
    monkeypatch.setattr('short_drama.llm.LLMClient.generate_structured', lambda *_a, **_k: pytest.fail('provider'))
    path = claim.claim_path(root, session, packet.window_id)
    assert claim.inspect_claim(root, session, packet.window_id).state == claim.ClaimState.ABSENT
    assert not root.exists()
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        result = _create(tree, session, packet, guard, tmp_path / 'checkpoints')
        assert result.path == path and result.state == claim.ClaimState.CLAIMED_UNRESOLVED
        assert path.read_bytes() == claim._encoded(claim._identity(session, packet.window_id))
        with pytest.raises(claim.B13ExecutionClaimError, match='existing claim'):
            _create(tree, session, packet, guard, tmp_path / 'checkpoints')
    # A crash before dispatch or after response has exactly the same state.
    assert claim.inspect_claim(root, session, packet.window_id).state == claim.ClaimState.CLAIMED_UNRESOLVED
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        with pytest.raises(claim.B13ExecutionClaimError, match='existing claim'):
            _create(tree, session, packet, guard, tmp_path / 'checkpoints')
    with pytest.raises(claim.B13ExecutionClaimError, match='not active'):
        _create(tree, session, packet, guard, tmp_path / 'checkpoints')


def test_checkpoint_prevents_claim(tmp_path):
    tree, session, packet, semantic = _fixture(tmp_path)
    checkpoints = tmp_path / 'checkpoints'
    accepted = _accepted(packet, semantic, _b12_test_runtime_config())
    save_checkpoint(checkpoints, accepted, session)
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        with pytest.raises(claim.B13ExecutionClaimError, match='checkpoint already exists'):
            _create(tree, session, packet, guard, checkpoints)
    assert claim.inspect_claim(root, session, packet.window_id).state == claim.ClaimState.ABSENT


@pytest.mark.parametrize('bad', ['corrupt', 'truncated', 'oversize', 'symlink', 'fifo', 'conflict'])
def test_invalid_claim_blocks(tmp_path, bad):
    tree, session, packet, _ = _fixture(tmp_path)
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        path = claim.claim_path(root, session, packet.window_id)
        if bad == 'symlink': path.symlink_to(tmp_path / 'missing')
        elif bad == 'fifo': os.mkfifo(path)
        else:
            raw = claim._encoded(claim._identity(session, packet.window_id))
            path.write_bytes({'corrupt': b'{}', 'truncated': raw[:-1], 'oversize': b'x' * (claim.MAX_CLAIM_BYTES + 1),
                              'conflict': raw.replace(b'CLAIMED_UNRESOLVED', b'CLAIMED_RESOLVED__')}[bad])
            path.chmod(0o600)
        assert claim.inspect_claim(root, session, packet.window_id).state == claim.ClaimState.INVALID
        with pytest.raises(claim.B13ExecutionClaimError, match='existing claim'):
            _create(tree, session, packet, guard, tmp_path / 'checkpoints')


def test_lock_contention_and_unsafe_targets(tmp_path, stores):
    store, pointers = stores
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=store, pointers=pointers):
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        process = ctx.Process(target=_compete, args=(root, store, pointers, queue))
        process.start(); process.join(10)
        assert process.exitcode == 0 and queue.get(timeout=2) == 'blocked'
    lock = root / '.b13-execution.lock'
    lock.unlink(); lock.symlink_to(tmp_path / 'elsewhere')
    with pytest.raises(claim.B13ExecutionClaimError):
        with claim.execution_guard(root, store=store, pointers=pointers): pass


@pytest.mark.parametrize('point', ['write', 'file_sync', 'link', 'directory_sync'])
def test_publication_failures_fail_closed(tmp_path, monkeypatch, point):
    tree, session, packet, _ = _fixture(tmp_path)
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        if point == 'write': monkeypatch.setattr(claim.storage, '_write_all', lambda *_: (_ for _ in ()).throw(OSError('write')))
        elif point == 'link': monkeypatch.setattr(claim.os, 'link', lambda *_a, **_k: (_ for _ in ()).throw(OSError('link')))
        else:
            real = os.fsync
            def fail(fd):
                is_dir = os.fstat(fd).st_ino == os.fstat(guard._dirfd).st_ino
                if is_dir == (point == 'directory_sync'): raise OSError('sync')
                return real(fd)
            monkeypatch.setattr(claim.os, 'fsync', fail)
        with pytest.raises(claim.B13ExecutionClaimError):
            _create(tree, session, packet, guard, tmp_path / 'checkpoints')
    state = claim.inspect_claim(root, session, packet.window_id).state
    assert state == (claim.ClaimState.CLAIMED_UNRESOLVED if point == 'directory_sync' else claim.ClaimState.ABSENT)
    assert not list(root.glob('*.tmp'))


def test_root_replacement_and_invalid_session(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        root.rename(tmp_path / 'old')
        root.mkdir(mode=0o700)
        with pytest.raises(claim.B13ExecutionClaimError, match='root changed'):
            _create(tree, session, packet, guard, tmp_path / 'checkpoints')
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        for name in ('wrong_strategy_ref', 'evidence_request_mismatch', 'tokenizer_unknown'):
            with pytest.raises(claim.B13ExecutionClaimError):
                _create(tree, _broken_session(session, name), packet, guard, tmp_path / 'checkpoints')


def test_root_parent_sync_failure_and_root_symlink(tmp_path, monkeypatch, stores):
    store, pointers = stores
    parent = tmp_path / 'parent'
    parent.mkdir()
    root = parent / 'execution'
    real = os.fsync
    def fail(fd):
        if os.fstat(fd).st_ino == parent.stat().st_ino:
            raise OSError('parent sync')
        return real(fd)
    monkeypatch.setattr(claim.os, 'fsync', fail)
    with pytest.raises(claim.B13ExecutionClaimError):
        with claim.execution_guard(root, store=store, pointers=pointers): pass
    assert root.is_dir() and not list(root.iterdir())
    monkeypatch.undo()
    link = tmp_path / 'link'
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(claim.B13ExecutionClaimError):
        with claim.execution_guard(link, store=store, pointers=pointers): pass


def test_lock_replacement_refuses_claim(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        (root / '.b13-execution.lock').unlink()
        (root / '.b13-execution.lock').touch(mode=0o600)
        with pytest.raises(claim.B13ExecutionClaimError, match='lock target changed'):
            _create(tree, session, packet, guard, tmp_path / 'checkpoints')


def test_authority_change_and_claim_binding(tmp_path):
    from test_b13_disposition_resume import _move_a6
    tree, session, packet, _ = _fixture(tmp_path)
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=tree.store, pointers=tree.pointers) as guard:
        path = claim.claim_path(root, session, packet.window_id)
        identity = claim._identity(session, packet.window_id)
        for field in ('source_snapshot_hash', 'preparation_identity', 'request_identity'):
            altered = dict(identity, **{field: '0' * 64})
            path.write_bytes(claim._encoded(altered)); path.chmod(0o600)
            assert claim.inspect_claim(root, session, packet.window_id).state == claim.ClaimState.INVALID
        path.unlink()
        _move_a6(tree, session)
        with pytest.raises(claim.B13ExecutionClaimError):
            _create(tree, session, packet, guard, tmp_path / 'checkpoints')
        assert not path.exists()


def _snapshot(root):
    """Include directories, file bytes and symlink targets, without following links."""
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = os.path.join(directory, name)
            key = os.path.relpath(path, root)
            if os.path.islink(path):
                result[key] = ('link', os.readlink(path))
            elif os.path.isdir(path):
                result[key] = ('directory',)
            else:
                with open(path, 'rb') as stream:
                    result[key] = ('file', stream.read())
    return result


@pytest.mark.parametrize('protected_name', ['artifact', 'pointer'])
@pytest.mark.parametrize('alias', ['direct', 'dotdot', 'intermediate_symlink', 'existing_alias'])
def test_protected_root_aliases_rejected_before_any_mutation(tmp_path, stores, protected_name, alias):
    store, pointers = stores
    protected = store.root if protected_name == 'artifact' else pointers.root
    outside = tmp_path / 'outside'
    outside.mkdir()
    link = tmp_path / 'alias'
    if alias == 'direct':
        root = protected / 'execution'
    elif alias == 'dotdot':
        root = outside / '..' / protected.name / 'execution'
    elif alias == 'intermediate_symlink':
        link.symlink_to(protected, target_is_directory=True)
        root = link / 'execution'
    else:
        link.symlink_to(protected, target_is_directory=True)
        root = link
    before = _snapshot(tmp_path)
    with pytest.raises(claim.B13ExecutionClaimError):
        with claim.execution_guard(root, store=store, pointers=pointers):
            pytest.fail('protected root acquired')
    assert _snapshot(tmp_path) == before
    assert not (protected / 'execution').exists()
    assert not (protected / '.b13-execution.lock').exists()


def test_valid_root_and_missing_trusted_parent(tmp_path, stores):
    store, pointers = stores
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=store, pointers=pointers) as guard:
        guard.verify()
        assert guard.path == root.resolve()
    before = _snapshot(tmp_path)
    with pytest.raises(claim.B13ExecutionClaimError):
        with claim.execution_guard(tmp_path / 'missing' / 'root', store=store, pointers=pointers):
            pytest.fail('missing parent acquired')
    assert _snapshot(tmp_path) == before


def test_root_replaced_during_guard_initialization(tmp_path, stores, monkeypatch):
    store, pointers = stores
    root = tmp_path / 'execution'
    root.mkdir(mode=0o700)
    before_artifacts, before_pointers = _snapshot(store.root), _snapshot(pointers.root)
    real_open = os.open
    replaced = False
    def replace_after_open(path, flags, *args, **kwargs):
        nonlocal replaced
        fd = real_open(path, flags, *args, **kwargs)
        if path == root.name and kwargs.get('dir_fd') is not None and not replaced:
            replaced = True
            root.rename(tmp_path / 'original-root')
            root.mkdir(mode=0o700)
        return fd
    monkeypatch.setattr(claim.os, 'open', replace_after_open)
    with pytest.raises(claim.B13ExecutionClaimError, match='root changed'):
        with claim.execution_guard(root, store=store, pointers=pointers):
            pytest.fail('replaced root acquired')
    assert replaced
    assert _snapshot(store.root) == before_artifacts and _snapshot(pointers.root) == before_pointers
    assert not list(root.iterdir()) and not list((tmp_path / 'original-root').iterdir())


def test_guard_requires_matching_stores(tmp_path, stores):
    store, pointers = stores
    other = FileArtifactStore(tmp_path / 'other-store')
    root = tmp_path / 'execution'
    before = _snapshot(tmp_path)
    with pytest.raises(claim.B13ExecutionClaimError, match='matching'):
        with claim.execution_guard(root, store=other, pointers=pointers):
            pytest.fail('mismatched stores acquired')
    assert _snapshot(tmp_path) == before


def test_claim_rejects_guard_for_different_stores(tmp_path, stores):
    store, pointers = stores
    other = FileArtifactStore(tmp_path / 'other-store')
    other_pointers = FilePointerStore(tmp_path / 'other-pointers', other)
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=store, pointers=pointers) as guard:
        with pytest.raises(claim.B13ExecutionClaimError, match='different stores'):
            claim.create_claim(guard=guard, store=other, pointers=other_pointers,
                               session=None, checkpoint_root=tmp_path / 'checkpoints', window_id='window')
    assert not list(root.glob('*.json'))


def test_guard_rejects_changed_protected_root(tmp_path, stores):
    store, pointers = stores
    root = tmp_path / 'execution'
    with claim.execution_guard(root, store=store, pointers=pointers) as guard:
        store.root = root
        with pytest.raises(claim.B13ExecutionClaimError, match='protected roots changed'):
            guard.verify()
