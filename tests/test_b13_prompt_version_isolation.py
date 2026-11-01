"""Synthetic v1/v2 records must never be silently reused or ignored."""
import pytest

import short_drama.adaptation.disposition_semantic as semantic
from short_drama.adaptation.disposition_checkpoint import (
    B13CheckpointEvidenceError, encode_checkpoint, verify_and_decode_checkpoint,
)
from short_drama.adaptation.disposition_checkpoint_storage import (
    B13CheckpointStorageError, _checkpoint_name, load_checkpoint,
)
from short_drama.adaptation.disposition_execution_claim import (
    B13ExecutionClaimError, ClaimState, _encoded, _identity, claim_path,
    create_claim, execution_guard, inspect_claim,
)
from test_b13_disposition_checkpoint import _accepted, _fixture, _session


def test_v1_records_block_v2_without_mixing(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(semantic, 'B13_WINDOW_PROMPT_VERSION', 1)
        m.setattr(semantic, 'B13_WINDOW_PROMPT_CONTENT_HASH_V2', semantic.B13_WINDOW_PROMPT_CONTENT_HASH_V1)
        tree, old_session, packet, old_request = _fixture(tmp_path / 'fixture')
        accepted = _accepted(packet, old_request, old_session.runtime_config)
        old_checkpoint = encode_checkpoint(accepted, old_session)
        assert verify_and_decode_checkpoint(old_checkpoint, old_session) == accepted
        old_checkpoint_name = _checkpoint_name(old_session, packet.window_id)
        old_claim = _encoded(_identity(old_session, packet.window_id))
        old_claim_name = claim_path(tmp_path / 'claims', old_session, packet.window_id).name
    new_request = semantic.build_b13_window_semantic_preparation(packet, runtime_config=old_session.runtime_config)
    new_session = _session(old_session.preparation, (new_request,), old_session.runtime_config)
    assert old_request.request_identity != new_request.request_identity
    with pytest.raises(B13CheckpointEvidenceError):
        verify_and_decode_checkpoint(old_checkpoint, new_session)
    checkpoint_root = tmp_path / 'checkpoints'
    checkpoint_root.mkdir(mode=0o700)
    (checkpoint_root / old_checkpoint_name).write_bytes(old_checkpoint)
    with pytest.raises(B13CheckpointStorageError, match='another request'):
        load_checkpoint(checkpoint_root, new_session, packet.window_id)
    (checkpoint_root / _checkpoint_name(new_session, packet.window_id)).write_bytes(b'synthetic')
    with pytest.raises(B13CheckpointStorageError, match='another request'):
        load_checkpoint(checkpoint_root, new_session, packet.window_id)
    claim_root = tmp_path / 'claims'
    claim_root.mkdir(mode=0o700)
    claim_file = claim_root / old_claim_name
    claim_file.write_bytes(old_claim)
    claim_file.chmod(0o600)
    assert inspect_claim(claim_root, new_session, packet.window_id).state is ClaimState.INVALID
    with execution_guard(claim_root, store=tree.store, pointers=tree.pointers) as guard:
        with pytest.raises(B13ExecutionClaimError):
            create_claim(guard=guard, store=tree.store, pointers=tree.pointers, session=new_session,
                         checkpoint_root=tmp_path / 'empty-checkpoints', window_id=packet.window_id)
    new_claim = claim_path(claim_root, new_session, packet.window_id)
    new_claim.write_bytes(_encoded(_identity(new_session, packet.window_id)))
    new_claim.chmod(0o600)
    assert inspect_claim(claim_root, new_session, packet.window_id).state is ClaimState.INVALID
    assert claim_file.read_bytes() == old_claim
