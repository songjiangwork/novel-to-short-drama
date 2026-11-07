"""Synthetic, read-only v1/v2 discovery; no production runner or Provider."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import short_drama.adaptation.disposition_checkpoint_storage as storage
import short_drama.adaptation.disposition_resume as resume
from short_drama.adaptation.disposition_checkpoint_inspection_v2 import (
    B13OfflineCheckpointConflict, inspect_checkpoint_versions_offline,
)
from short_drama.adaptation.disposition_checkpoint_v2 import encode_feedback_checkpoint_v2
from short_drama.adaptation.disposition_checkpoint import encode_checkpoint
from short_drama.adaptation.disposition_feedback import build_feedback_request
from test_b13_disposition_checkpoint import _accepted, _fixture, _provenance, _session
from test_b13_disposition_execution import _two_window_inputs, _allow_synthetic_two_window_preflight
from test_b13_window_semantic import _proposal


def _v2(session, packet, semantic):
    candidate = _proposal(packet)
    candidate['adaptation_mappings'] = []
    feedback = build_feedback_request(semantic, candidate)
    second = SimpleNamespace(request=feedback.request, semantic_profile=semantic.semantic_profile,
                             output_schema=semantic.output_schema)
    return encode_feedback_checkpoint_v2(
        session=session, window_id=packet.window_id, rejected_candidate=candidate,
        first_provenance=_provenance(semantic, session.runtime_config), first_attempts=1,
        accepted_proposal=_proposal(packet), second_provenance=_provenance(second, session.runtime_config),
        second_attempts=1,
    )


def _name(session, packet, version='v2', request=None):
    semantic = session.semantic_preparations[packet.window_ordinal - 1]
    return (f'b13-checkpoint-{version}-p{session.preparation.preparation_identity}'
            f'-w{packet.window_ordinal:08d}-q{request or semantic.request_identity}.json')


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    tree, session, packet, semantic = _fixture(tmp_path_factory.mktemp('version-read'))
    return tree, session, packet, semantic, _v2(session, packet, semantic)


def test_v2_only_blocks_v1_loader_and_save_preserves_carrier(case, tmp_path):
    tree, session, packet, semantic, raw = case
    path = tmp_path / _name(session, packet)
    path.write_bytes(raw)
    path.chmod(0o600)
    before = path.read_bytes()
    with pytest.raises(storage.B13CheckpointStorageError):
        storage.load_checkpoint(tmp_path, session, packet.window_id)
    with pytest.raises(storage.B13CheckpointStorageError):
        storage.save_checkpoint(tmp_path, _accepted(packet, semantic, session.runtime_config), session)
    assert path.read_bytes() == before and len(list(tmp_path.iterdir())) == 1
    accepted = inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id)
    assert accepted.feedback_evidence.canonical_bytes() == raw
    assert accepted.request_identity == semantic.request_identity
    assert accepted.invocation_provenance.request_hash != semantic.request.request_hash
    plan = resume.inspect_b13_resume_versions(store=tree.store, pointers=tree.pointers, session=session,
                                             checkpoint_root=tmp_path)
    assert plan.verified_reuse_count == 1 and plan.windows[0].accepted_proposal == accepted
    with pytest.raises(resume.B13ResumeError):
        resume.plan_b13_resume(store=tree.store, pointers=tree.pointers, session=session, checkpoint_root=tmp_path)
    assert path.read_bytes() == before


def test_empty_and_v1_reuse_are_read_only(case, tmp_path):
    tree, session, packet, semantic, _raw = case
    missing = tmp_path / 'missing'
    assert inspect_checkpoint_versions_offline(missing, session, packet.window_id) is None
    plan = resume.inspect_b13_resume_versions(store=tree.store, pointers=tree.pointers, session=session,
                                             checkpoint_root=missing)
    assert plan.needs_generation_count == 1 and not missing.exists()
    accepted = _accepted(packet, semantic, session.runtime_config)
    path = storage.save_checkpoint(tmp_path / 'v1', accepted, session)
    data, name = path.read_bytes(), path.name
    assert name == _name(session, packet, version='v1')
    assert data == encode_checkpoint(accepted, session)
    assert storage.load_checkpoint(path.parent, session, packet.window_id) == accepted
    assert inspect_checkpoint_versions_offline(path.parent, session, packet.window_id) == accepted
    assert path.read_bytes() == data and path.name == name


def test_mixed_windows_read_only_resume(tmp_path, monkeypatch):
    tree, preparation, requests, _budgets, runtime, first, second = _two_window_inputs(tmp_path)
    _allow_synthetic_two_window_preflight(monkeypatch, preparation)
    session = _session(preparation, requests, runtime)
    root = tmp_path / 'evidence'
    one = storage.save_checkpoint(root, _accepted(first, requests[0], runtime), session)
    two = root / _name(session, second)
    raw = _v2(session, second, requests[1])
    two.write_bytes(raw)
    two.chmod(0o600)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    plan = resume.inspect_b13_resume_versions(store=tree.store, pointers=tree.pointers, session=session,
                                             checkpoint_root=root)
    assert plan.verified_reuse_count == 2
    assert plan.windows[0].accepted_proposal.feedback_evidence is None
    assert plan.windows[1].accepted_proposal.feedback_evidence.canonical_bytes() == raw
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert one.name in before and two.name in before
    # A valid checkpoint for another window cannot be accepted under this name.
    two.write_bytes(_v2(session, first, requests[0]))
    with pytest.raises(B13OfflineCheckpointConflict):
        inspect_checkpoint_versions_offline(root, session, second.window_id)


@pytest.mark.parametrize('kind', ['both', 'unknown', 'wrong_request', 'corrupt', 'truncated', 'oversize', 'symlink', 'directory', 'fifo'])
def test_conflicts_and_unsafe_files_fail_closed(case, tmp_path, kind):
    _tree, session, packet, semantic, raw = case
    target = tmp_path / _name(session, packet)
    if kind == 'both':
        target.write_bytes(raw)
        storage_path = tmp_path / _name(session, packet, version='v1')
        storage_path.write_bytes(encode_checkpoint(_accepted(packet, semantic, session.runtime_config), session))
    elif kind == 'unknown':
        (tmp_path / _name(session, packet, version='v99')).write_bytes(raw)
    elif kind == 'wrong_request':
        (tmp_path / _name(session, packet, request='0' * 64)).write_bytes(raw)
    elif kind == 'corrupt': target.write_bytes(b'NOT A CHECKPOINT SECRET')
    elif kind == 'truncated': target.write_bytes(raw[:-1])
    elif kind == 'oversize': target.write_bytes(b'x' * 1_000_001)
    elif kind == 'symlink': target.symlink_to(tmp_path / 'SECRET')
    elif kind == 'directory': target.mkdir()
    else:
        if not hasattr(os, 'mkfifo'):
            pytest.skip('FIFO unavailable')
        os.mkfifo(target)
    with pytest.raises(B13OfflineCheckpointConflict) as error:
        inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id)
    assert 'SECRET' not in str(error.value) and 'evt_' not in str(error.value)
    with pytest.raises(storage.B13CheckpointStorageError):
        storage.load_checkpoint(tmp_path, session, packet.window_id)
    with pytest.raises(storage.B13CheckpointStorageError):
        storage.save_checkpoint(tmp_path, _accepted(packet, semantic, session.runtime_config), session)


@pytest.mark.parametrize('unsafe', ['file_mode', 'root_mode', 'hardlink'])
def test_feedback_v2_requires_private_unlinked_file_and_root(case, tmp_path, unsafe):
    _tree, session, packet, _semantic, raw = case
    target = tmp_path / _name(session, packet)
    target.write_bytes(raw)
    target.chmod(0o600)
    if unsafe == 'file_mode':
        target.chmod(0o644)
    elif unsafe == 'root_mode':
        tmp_path.chmod(0o750)
    else:
        os.link(target, tmp_path / 'unexpected-second-link')

    try:
        with pytest.raises(B13OfflineCheckpointConflict):
            inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id)
        assert target.read_bytes() == raw
    finally:
        if unsafe == 'root_mode':
            tmp_path.chmod(0o700)


def test_file_replaced_during_read_is_rejected(case, tmp_path, monkeypatch):
    _tree, session, packet, _semantic, raw = case
    target = tmp_path / _name(session, packet)
    target.write_bytes(raw)
    target.chmod(0o600)
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(raw)
    replacement.chmod(0o600)
    real_read = storage.os.read
    switched = False
    def read_and_replace(fd, n):
        nonlocal switched
        result = real_read(fd, n)
        if result and not switched:
            switched = True
            os.replace(replacement, target)
        return result
    monkeypatch.setattr(storage.os, 'read', read_and_replace)
    with pytest.raises(B13OfflineCheckpointConflict):
        inspect_checkpoint_versions_offline(tmp_path, session, packet.window_id)
    assert switched


def test_version_appearing_during_v1_read_fails_closed(case, tmp_path, monkeypatch):
    _tree, session, packet, semantic, raw = case
    storage.save_checkpoint(tmp_path, _accepted(packet, semantic, session.runtime_config), session)
    other = tmp_path / _name(session, packet)
    original = storage._read_bounded
    def old_writer_race(path):
        result = original(path)
        other.write_bytes(raw)
        return result
    monkeypatch.setattr(storage, '_read_bounded', old_writer_race)
    with pytest.raises(storage.B13CheckpointStorageError):
        storage.load_checkpoint(tmp_path, session, packet.window_id)
    assert other.exists()


def test_version_appearing_during_v1_save_fails_closed(case, tmp_path, monkeypatch):
    _tree, session, packet, semantic, raw = case
    original = storage.os.link
    other = tmp_path / _name(session, packet)
    def concurrent_publish(source, target):
        original(source, target)
        other.write_bytes(raw)
    monkeypatch.setattr(storage.os, 'link', concurrent_publish)
    with pytest.raises(storage.B13CheckpointStorageError):
        storage.save_checkpoint(tmp_path, _accepted(packet, semantic, session.runtime_config), session)
    assert other.exists()
