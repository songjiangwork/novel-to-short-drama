"""Test-only synthetic v2 durability adapter; not a production save interface.

The fixture closes over the fixed synthetic case and pytest's tmp_path. It has
no candidate, bytes, session, root, CLI, or environment override input.
"""
import os
import stat
from types import SimpleNamespace

import pytest

import short_drama.adaptation.disposition_checkpoint_storage as storage
from short_drama.adaptation.disposition_checkpoint_inspection_v2 import inspect_checkpoint_versions_offline
from short_drama.adaptation.disposition_checkpoint_v2 import (
    encode_verified_feedback_checkpoint_v2, verify_and_decode_feedback_checkpoint_v2,
)
from short_drama.adaptation.disposition_checkpoint import encode_checkpoint
from test_b13_disposition_checkpoint import _accepted
from test_b13_feedback_checkpoint_v2 import case  # fixed repository synthetic fixture


class SyntheticDurabilityError(ValueError):
    pass


@pytest.fixture
def lab(tmp_path, case, monkeypatch):
    """Build one adapter with no caller-supplied evidence or destination."""
    session, packet, semantic, _kwargs, fixture_raw = case
    accepted = verify_and_decode_feedback_checkpoint_v2(fixture_raw, session)
    # Both the parent and root are children of pytest's own tmp_path.
    parent = tmp_path / 'synthetic-parent'
    parent.mkdir(mode=0o700)
    root = parent / 'checkpoint-root'
    prep, ordinal, base = storage._session_window(session, packet.window_id)
    name = f'b13-checkpoint-v2-p{prep}-w{ordinal:08d}-q{base}.json'
    target = root / name
    monkeypatch.setattr('short_drama.llm.LLMClient.generate_structured',
                        lambda *_a, **_kw: pytest.fail('Provider must not be called'))

    def private_directory(path):
        st = path.lstat()
        if (not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) != 0o700):
            raise ValueError

    def check_root(create):
        if parent.is_symlink() or parent.parent != tmp_path:
            raise ValueError
        private_directory(parent)
        path = storage._root(root, create=create)
        if path is None or path != root:
            raise ValueError
        private_directory(root)

    def check_conflicts():
        if any(n != name for n in storage._related_checkpoint_names(root, prep, ordinal)):
            raise ValueError

    def verify_existing(raw):
        check_root(False)
        check_conflicts()
        st = storage._target_stat(target)
        if st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) != 0o600:
            raise ValueError
        decoded = inspect_checkpoint_versions_offline(root, session, packet.window_id)
        if decoded != accepted or storage._read_bounded(target) != raw:
            raise ValueError
        check_conflicts()
        check_root(False)

    def save():
        # Fixed fixture only; verify carrier through the real v2 codec every time.
        try:
            raw = encode_verified_feedback_checkpoint_v2(accepted, session)
            if raw != fixture_raw:
                raise ValueError
            check_root(True)
            check_conflicts()
            if storage._target_stat(target, missing_ok=True) is not None:
                verify_existing(raw)
                storage._fsync_directory(root)
                verify_existing(raw)
                return target
            with storage._staged_checkpoint(target, raw) as temporary:
                check_root(False)
                check_conflicts()
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    verify_existing(raw)
                storage._fsync_directory(root)
                verify_existing(raw)
                return target
        except Exception:
            raise SyntheticDurabilityError('synthetic checkpoint operation failed') from None

    return SimpleNamespace(root=root, parent=parent, target=target, save=save,
                           raw=fixture_raw, accepted=accepted, session=session,
                           packet=packet, semantic=semantic, prep=prep, ordinal=ordinal)


def test_real_os_save_fsync_readback_and_idempotence(lab, monkeypatch):
    real_fsync = storage.os.fsync
    synced = []
    def record_fsync(fd):
        mode = os.fstat(fd).st_mode
        synced.append('file' if stat.S_ISREG(mode) else 'directory' if stat.S_ISDIR(mode) else 'other')
        return real_fsync(fd)
    monkeypatch.setattr(storage.os, 'fsync', record_fsync)
    assert lab.save() == lab.target
    assert 'file' in synced and 'directory' in synced
    assert stat.S_IMODE(lab.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(lab.target.stat().st_mode) == 0o600
    assert lab.target.read_bytes() == lab.raw
    assert verify_and_decode_feedback_checkpoint_v2(lab.target.read_bytes(), lab.session) == lab.accepted
    before = lab.target.stat().st_ino
    assert lab.save() == lab.target
    assert lab.target.stat().st_ino == before and lab.target.read_bytes() == lab.raw


@pytest.mark.parametrize('kind', ['v1', 'other_request', 'unknown', 'different', 'corrupt', 'file_mode', 'symlink', 'directory', 'fifo'])
def test_existing_conflicts_never_overwrite(lab, kind):
    lab.root.mkdir(mode=0o700)
    suffix = f'-p{lab.prep}-w{lab.ordinal:08d}-q{lab.semantic.request_identity}.json'
    if kind == 'v1':
        v1 = _accepted(lab.packet, lab.semantic, lab.session.runtime_config)
        (lab.root / ('b13-checkpoint-v1' + suffix)).write_bytes(encode_checkpoint(v1, lab.session))
    elif kind == 'other_request':
        (lab.root / (f'b13-checkpoint-v2-p{lab.prep}-w{lab.ordinal:08d}-q' + ('0' * 64) + '.json')).write_bytes(b'conflict')
    elif kind == 'unknown': (lab.root / ('b13-checkpoint-v99' + suffix)).write_bytes(b'conflict')
    elif kind == 'different': lab.target.write_bytes(lab.raw + b' ')
    elif kind == 'corrupt': lab.target.write_bytes(b'SECRET SYNTHETIC')
    elif kind == 'file_mode':
        lab.target.write_bytes(lab.raw)
        lab.target.chmod(0o644)
    elif kind == 'symlink': lab.target.symlink_to(lab.parent / 'outside')
    elif kind == 'directory': lab.target.mkdir()
    else:
        if not hasattr(os, 'mkfifo'): pytest.skip('FIFO unsupported')
        os.mkfifo(lab.target)
    before = sorted(p.name for p in lab.root.iterdir())
    with pytest.raises(SyntheticDurabilityError) as error:
        lab.save()
    assert 'SECRET' not in str(error.value) and 'evt_' not in str(error.value)
    assert sorted(p.name for p in lab.root.iterdir()) == before


@pytest.mark.parametrize('kind', ['root_symlink', 'root_file', 'root_mode', 'parent_symlink', 'parent_mode', 'parent_missing'])
def test_unsafe_roots_fail_closed(lab, kind, tmp_path):
    if kind == 'root_symlink': lab.root.symlink_to(tmp_path, target_is_directory=True)
    elif kind == 'root_file': lab.root.write_bytes(b'x')
    elif kind == 'root_mode':
        lab.root.mkdir(mode=0o700)
        lab.root.chmod(0o755)
    elif kind == 'parent_mode': lab.parent.chmod(0o755)
    elif kind == 'parent_missing': lab.parent.rmdir()
    else:
        lab.parent.rename(tmp_path / 'old-parent')
        lab.parent.symlink_to(tmp_path / 'old-parent', target_is_directory=True)
    with pytest.raises(SyntheticDurabilityError): lab.save()


@pytest.mark.parametrize('fault', ['temp', 'zero_write', 'file_fsync', 'link', 'directory_fsync', 'readback'])
def test_injected_faults_never_report_success(lab, monkeypatch, fault):
    # Existing safe root means directory fsync fault occurs after publication.
    lab.root.mkdir(mode=0o700)
    if fault == 'temp':
        monkeypatch.setattr(storage.tempfile, 'mkstemp', lambda *_a, **_k: (_ for _ in ()).throw(OSError('secret')))
    elif fault == 'zero_write': monkeypatch.setattr(storage.os, 'write', lambda *_a: 0)
    elif fault == 'file_fsync':
        original = storage.os.fsync
        def fail_file(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode): raise OSError('secret')
            return original(fd)
        monkeypatch.setattr(storage.os, 'fsync', fail_file)
    elif fault == 'link': monkeypatch.setattr(storage.os, 'link', lambda *_a: (_ for _ in ()).throw(FileExistsError('secret')))
    elif fault == 'directory_fsync': monkeypatch.setattr(storage, '_fsync_directory', lambda *_a: (_ for _ in ()).throw(OSError('secret')))
    else: monkeypatch.setattr(storage, '_read_bounded', lambda *_a: (_ for _ in ()).throw(OSError('secret')))
    with pytest.raises(SyntheticDurabilityError) as error: lab.save()
    assert 'secret' not in str(error.value)
    if fault in ('directory_fsync', 'readback'):
        assert lab.target.exists() and lab.target.read_bytes() == lab.raw
    else:
        assert not lab.target.exists()


def test_short_positive_writes_are_completed(lab, monkeypatch):
    original = storage.os.write
    monkeypatch.setattr(storage.os, 'write', lambda fd, data: original(fd, data[:128]))
    assert lab.save().read_bytes() == lab.raw


def test_link_conflict_does_not_replace_concurrent_file(lab, monkeypatch):
    lab.root.mkdir(mode=0o700)
    def concurrent_file(_source, target):
        lab.target.write_bytes(b'concurrent')
        lab.target.chmod(0o600)
        raise FileExistsError('synthetic conflict')
    monkeypatch.setattr(storage.os, 'link', concurrent_file)
    with pytest.raises(SyntheticDurabilityError): lab.save()
    assert lab.target.read_bytes() == b'concurrent'


def test_post_link_conflict_keeps_published_evidence(lab, monkeypatch):
    lab.root.mkdir(mode=0o700)
    original = storage.os.link
    other = lab.root / f'b13-checkpoint-v99-p{lab.prep}-w{lab.ordinal:08d}-q'  # marker is sufficient
    def publish_then_conflict(source, target):
        original(source, target)
        other.write_bytes(b'conflict')
    monkeypatch.setattr(storage.os, 'link', publish_then_conflict)
    with pytest.raises(SyntheticDurabilityError): lab.save()
    assert lab.target.read_bytes() == lab.raw and other.read_bytes() == b'conflict'


def test_version_conflict_after_initial_save_is_not_reused(lab):
    lab.save()
    suffix = f'-p{lab.prep}-w{lab.ordinal:08d}-q{lab.semantic.request_identity}.json'
    v1 = _accepted(lab.packet, lab.semantic, lab.session.runtime_config)
    other = lab.root / ('b13-checkpoint-v1' + suffix)
    original_v1 = encode_checkpoint(v1, lab.session)
    other.write_bytes(original_v1)
    with pytest.raises(SyntheticDurabilityError): lab.save()
    assert lab.target.read_bytes() == lab.raw and other.read_bytes() == original_v1


def test_post_link_replacement_is_not_overwritten(lab, monkeypatch):
    lab.root.mkdir(mode=0o700)
    replacement = lab.parent / 'replacement'
    replacement.write_bytes(b'changed')
    original = storage.os.link
    def publish_then_replace(source, target):
        original(source, target)
        os.replace(replacement, target)
    monkeypatch.setattr(storage.os, 'link', publish_then_replace)
    with pytest.raises(SyntheticDurabilityError): lab.save()
    assert lab.target.read_bytes() == b'changed'
