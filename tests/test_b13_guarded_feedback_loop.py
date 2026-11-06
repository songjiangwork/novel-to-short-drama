"""Guarded feedback durability in pytest temporary roots; not a v2 writer API."""
import os
import stat
from types import SimpleNamespace

import pytest

from short_drama.adaptation import disposition_checkpoint_storage as storage
from short_drama.adaptation.disposition_checkpoint_v2 import (
    encode_verified_feedback_checkpoint_v2, verify_and_decode_feedback_checkpoint_v2,
)
from short_drama.adaptation.disposition_checkpoint_inspection_v2 import inspect_checkpoint_versions_offline
from short_drama.adaptation.disposition_execution_claim import (
    B13ExecutionClaimError, ClaimState, create_claim, execution_guard, inspect_claim,
)
from short_drama.adaptation.disposition_resume import B13ResumeDecision, B13ResumeError, inspect_b13_resume_versions
from short_drama.adaptation.disposition_runner import _roots
from test_b13_disposition_checkpoint import _accepted, _fixture
from test_b13_feedback_synthetic_loop import RecordingFakeClient, SyntheticTokenizer, run_fake
from test_b13_window_semantic import _proposal


class GuardedFakeClient(RecordingFakeClient):
    guard = None

    def generate_structured(self, *args, **kwargs):
        self.guard.verify()
        result = super().generate_structured(*args, **kwargs)
        self.guard.verify()
        return result


@pytest.fixture
def guarded_lab(tmp_path):
    tree, session, packet, semantic = _fixture(tmp_path)
    execution, checkpoint = _roots(tree.store, tree.pointers, tmp_path / 'execution', tmp_path / 'checkpoints')
    candidate = _proposal(packet)
    candidate['adaptation_mappings'] = []
    client = GuardedFakeClient([(candidate, False), (_proposal(packet), False)], session.runtime_config)
    accepted = None

    def claim(guard):
        return create_claim(guard=guard, store=tree.store, pointers=tree.pointers,
                            session=session, checkpoint_root=checkpoint, window_id=packet.window_id)

    def generate(guard):
        nonlocal accepted
        guard.verify()
        client.guard = guard
        accepted = run_fake(tree, session, client, SyntheticTokenizer()).accepted_windows[0]
        guard.verify()
        return accepted

    def dispatch(guard):
        assert claim(guard).state is ClaimState.CLAIMED_UNRESOLVED
        return generate(guard)

    def save(guard):
        # No caller-supplied bytes, candidate, session or root. Only the actual
        # fake execution above can populate this closure; no production API.
        try:
            guard.verify()
            if accepted is None:
                raise ValueError
            raw = encode_verified_feedback_checkpoint_v2(accepted, session)
            if verify_and_decode_feedback_checkpoint_v2(raw, session) != accepted:
                raise ValueError
            _roots(tree.store, tree.pointers, execution, checkpoint)
            root = storage._root(checkpoint, create=True)
            if root != checkpoint or stat.S_IMODE(root.stat().st_mode) != 0o700:
                raise ValueError
            prep, ordinal, request = storage._session_window(session, packet.window_id)
            name = f'b13-checkpoint-v2-p{prep}-w{ordinal:08d}-q{request}.json'
            target = root / name
            if storage._related_checkpoint_names(root, prep, ordinal):
                raise ValueError
            with storage._staged_checkpoint(target, raw) as temporary:
                guard.verify()
                if storage._related_checkpoint_names(root, prep, ordinal):
                    raise ValueError
                os.link(temporary, target)
                storage._fsync_directory(root)
                guard.verify()
                _roots(tree.store, tree.pointers, execution, checkpoint)
                st = storage._target_stat(target)
                if stat.S_IMODE(st.st_mode) != 0o600:
                    raise ValueError
                if (storage._read_bounded(target) != raw
                        or inspect_checkpoint_versions_offline(root, session, packet.window_id) != accepted):
                    raise ValueError
                guard.verify()
                return target
        except Exception:
            raise ValueError('synthetic v2 checkpoint failed') from None

    def resume():
        return inspect_b13_resume_versions(store=tree.store, pointers=tree.pointers,
                                           session=session, checkpoint_root=checkpoint)

    return SimpleNamespace(tree=tree, session=session, packet=packet, semantic=semantic,
                           execution=execution, checkpoint=checkpoint, client=client,
                           claim=claim, dispatch=dispatch, save=save, resume=resume)


def test_guard_claim_feedback_v2_durable_resume(guarded_lab, monkeypatch):
    lab = guarded_lab
    assert lab.resume().needs_generation_count == 1
    with execution_guard(lab.execution, store=lab.tree.store, pointers=lab.tree.pointers) as guard:
        accepted = lab.dispatch(guard)
        assert len(lab.client.calls) == accepted.provider_attempts == accepted.semantic_rounds == 2
        assert lab.client.responses[0].provenance.request_hash != lab.client.responses[1].provenance.request_hash
        assert accepted.feedback_evidence is not None
        original_fsync = storage.os.fsync
        synced = []
        def record_fsync(fd):
            guard.verify()
            mode = os.fstat(fd).st_mode
            synced.append('file' if stat.S_ISREG(mode) else 'directory' if stat.S_ISDIR(mode) else 'other')
            return original_fsync(fd)
        monkeypatch.setattr(storage.os, 'fsync', record_fsync)
        target = lab.save(guard)
        assert 'file' in synced and 'directory' in synced
        assert accepted.request_identity in target.name
        assert accepted.invocation_provenance.request_hash not in target.name
        assert target.read_bytes() == accepted.feedback_evidence.canonical_bytes()
        assert lab.resume().windows[0].accepted_proposal == accepted
        guard.verify()
    before = len(lab.client.calls)
    plan = lab.resume()
    assert plan.windows[0].decision is B13ResumeDecision.VERIFIED_REUSE
    assert plan.windows[0].accepted_proposal == accepted
    assert plan.windows[0].accepted_proposal.feedback_evidence == accepted.feedback_evidence
    assert len(lab.client.calls) == before
    assert inspect_claim(lab.execution, lab.session, lab.packet.window_id).state is ClaimState.CLAIMED_UNRESOLVED


def test_claim_blocks_second_dispatch(guarded_lab):
    lab = guarded_lab
    with execution_guard(lab.execution, store=lab.tree.store, pointers=lab.tree.pointers) as guard:
        lab.claim(guard)
    with execution_guard(lab.execution, store=lab.tree.store, pointers=lab.tree.pointers) as guard:
        with pytest.raises(B13ExecutionClaimError):
            lab.dispatch(guard)
    assert lab.client.calls == []


def test_first_round_success_v1_checkpoint(tmp_path):
    tree, session, packet, _ = _fixture(tmp_path)
    execution, checkpoint = _roots(tree.store, tree.pointers, tmp_path / 'execution', tmp_path / 'checkpoints')
    client = RecordingFakeClient([(_proposal(packet), False)], session.runtime_config)
    with execution_guard(execution, store=tree.store, pointers=tree.pointers) as guard:
        create_claim(guard=guard, store=tree.store, pointers=tree.pointers, session=session,
                     checkpoint_root=checkpoint, window_id=packet.window_id)
        accepted = run_fake(tree, session, client, SyntheticTokenizer()).accepted_windows[0]
        assert accepted.feedback_evidence is None and len(client.calls) == 1
        guard.verify()
        saved = storage.save_checkpoint(checkpoint, accepted, session)
        assert saved.name.startswith('b13-checkpoint-v1-')
        assert storage.load_checkpoint(checkpoint, session, packet.window_id) == accepted
        guard.verify()
    plan = inspect_b13_resume_versions(store=tree.store, pointers=tree.pointers, session=session, checkpoint_root=checkpoint)
    assert plan.windows[0].accepted_proposal == accepted


def test_cross_version_conflict_fails_closed(guarded_lab):
    from short_drama.adaptation.disposition_checkpoint import encode_checkpoint
    lab = guarded_lab
    with execution_guard(lab.execution, store=lab.tree.store, pointers=lab.tree.pointers) as guard:
        lab.dispatch(guard)
        target = lab.save(guard)
        v1 = _accepted(lab.packet, lab.semantic, lab.session.runtime_config)
        v1_path = target.with_name(target.name.replace('b13-checkpoint-v2-', 'b13-checkpoint-v1-', 1))
        v1_path.write_bytes(encode_checkpoint(v1, lab.session))
        v1_path.chmod(0o600)
        with pytest.raises(B13ResumeError):
            lab.resume()
        with pytest.raises(ValueError, match='synthetic v2 checkpoint failed'):
            lab.save(guard)
        assert target.exists() and v1_path.exists()


@pytest.mark.parametrize('fault', ['write', 'file_fsync', 'link', 'directory_fsync', 'readback'])
def test_save_fault_preserves_claim_and_fails(guarded_lab, monkeypatch, fault):
    lab = guarded_lab
    with execution_guard(lab.execution, store=lab.tree.store, pointers=lab.tree.pointers) as guard:
        lab.dispatch(guard)
        if fault == 'write':
            monkeypatch.setattr(storage, '_write_all', lambda *_: (_ for _ in ()).throw(OSError('synthetic')))
        elif fault == 'file_fsync':
            original = storage.os.fsync
            def fail_file(fd):
                if stat.S_ISREG(os.fstat(fd).st_mode):
                    raise OSError('synthetic')
                return original(fd)
            monkeypatch.setattr(storage.os, 'fsync', fail_file)
        elif fault == 'link':
            monkeypatch.setattr(storage.os, 'link', lambda *_a, **_kw: (_ for _ in ()).throw(FileExistsError('synthetic')))
        elif fault == 'directory_fsync':
            # Root creation precedes the injected post-link fault.
            lab.checkpoint.mkdir(mode=0o700)
            monkeypatch.setattr(storage, '_fsync_directory', lambda *_: (_ for _ in ()).throw(OSError('synthetic')))
        else:
            monkeypatch.setattr(storage, '_read_bounded', lambda *_: (_ for _ in ()).throw(OSError('synthetic')))
        with pytest.raises(ValueError, match='synthetic v2 checkpoint failed'):
            lab.save(guard)
        guard.verify()
        if fault in ('directory_fsync', 'readback'):
            assert len(list(lab.checkpoint.glob('b13-checkpoint-v2-*'))) == 1
        assert len(lab.client.calls) == 2
    assert inspect_claim(lab.execution, lab.session, lab.packet.window_id).state is ClaimState.CLAIMED_UNRESOLVED
