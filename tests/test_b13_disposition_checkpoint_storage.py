"""Offline filesystem-boundary tests for one B1.3 checkpoint."""
from dataclasses import replace
import os
from pathlib import Path

import pytest

import short_drama.adaptation.disposition_checkpoint_storage as storage
from short_drama.adaptation.disposition_checkpoint import (
    B13_CHECKPOINT_MAX_BYTES, B13CheckpointEvidenceError, encode_checkpoint,
)

from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_checkpoint import _accepted, _fixture, _provenance, _session
from test_b13_disposition_execution import _two_window_inputs


def _case(tmp_path):
    tree, session, packet, semantic = _fixture(tmp_path)
    return tree, session, packet, semantic, _accepted(packet, semantic, _b12_test_runtime_config())


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_save_load_is_atomic_idempotent_and_zero_provider_or_artifact_writes(tmp_path, monkeypatch):
    tree, session, packet, _semantic, accepted = _case(tmp_path)
    root = tmp_path / "private-checkpoints"
    before = _tree_bytes(tree.store.root), _tree_bytes(tree.pointers.root)
    monkeypatch.setattr("short_drama.llm.LLMClient.generate_structured",
                        lambda *_a, **_k: pytest.fail("provider called"))
    target = storage.save_checkpoint(root, accepted, session)
    assert target == storage.checkpoint_path(root, session, packet.window_id)
    assert target.stat().st_mode & 0o777 == 0o600
    assert storage.load_checkpoint(root, session, packet.window_id) == accepted
    assert storage.save_checkpoint(root, accepted, session) == target
    assert target.read_bytes() == encode_checkpoint(accepted, session)
    assert (_tree_bytes(tree.store.root), _tree_bytes(tree.pointers.root)) == before


def test_filename_uses_full_preparation_request_and_ordinal_without_aliasing(tmp_path):
    _tree, session, packet, _semantic, _accepted_one = _case(tmp_path)
    _tree2, two, requests, _budgets, _runtime, first, second = _two_window_inputs(tmp_path / "two")
    second_session = _session(two, requests, _b12_test_runtime_config())
    one = storage.checkpoint_path(tmp_path / "one", session, packet.window_id)
    first_path = storage.checkpoint_path(tmp_path / "two-root", second_session, first.window_id)
    second_path = storage.checkpoint_path(tmp_path / "two-root", second_session, second.window_id)
    assert one.name.startswith("b13-checkpoint-v1-p")
    assert session.preparation.preparation_identity in one.name
    assert session.semantic_preparations[0].request_identity in one.name
    assert first_path != second_path and first_path.name != second_path.name


def test_conflict_cannot_replace_valid_checkpoint_and_missing_is_only_miss(tmp_path):
    _tree, session, packet, semantic, accepted = _case(tmp_path)
    root = tmp_path / "private"
    assert storage.load_checkpoint(root, session, packet.window_id) is None
    target = storage.save_checkpoint(root, accepted, session)
    conflicting = replace(accepted, invocation_provenance=_provenance(
        semantic, _b12_test_runtime_config(), provider_response_id="other-response"))
    original = target.read_bytes()
    with pytest.raises(storage.B13CheckpointStorageError, match="conflicting"):
        storage.save_checkpoint(root, conflicting, session)
    assert target.read_bytes() == original
    with pytest.raises(storage.B13CheckpointStorageError, match="exactly one"):
        storage.load_checkpoint(root, session, "not-a-session-window")


@pytest.mark.parametrize("bad", [b'{"x":1}', b'{"a":1,"a":2}', b"{"])
def test_corrupt_truncated_and_noncanonical_final_files_reject(tmp_path, bad):
    _tree, session, packet, _semantic, accepted = _case(tmp_path)
    root = tmp_path / "private"
    target = storage.save_checkpoint(root, accepted, session)
    target.write_bytes(bad)
    with pytest.raises(B13CheckpointEvidenceError):
        storage.load_checkpoint(root, session, packet.window_id)


def test_oversize_final_file_rejects_before_decode(tmp_path):
    _tree, session, packet, _semantic, accepted = _case(tmp_path)
    root = tmp_path / "private"
    target = storage.save_checkpoint(root, accepted, session)
    target.write_bytes(b"x" * (B13_CHECKPOINT_MAX_BYTES + 1))
    with pytest.raises(storage.B13CheckpointStorageError, match="maximum"):
        storage.load_checkpoint(root, session, packet.window_id)


def test_symlink_root_and_target_are_rejected(tmp_path):
    _tree, session, packet, _semantic, accepted = _case(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(storage.B13CheckpointStorageError, match="root.*symlink"):
        storage.save_checkpoint(root_link, accepted, session)
    root = tmp_path / "private"
    target = storage.checkpoint_path(root, session, packet.window_id)
    outside = tmp_path / "outside"
    outside.write_bytes(encode_checkpoint(accepted, session))
    target.symlink_to(outside)
    with pytest.raises(storage.B13CheckpointStorageError, match="target.*symlink"):
        storage.load_checkpoint(root, session, packet.window_id)


def test_fifo_final_target_rejects_without_opening_or_blocking(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unavailable")
    _tree, session, packet, _semantic, _accepted_one = _case(tmp_path)
    root = tmp_path / "private"
    target = storage.checkpoint_path(root, session, packet.window_id)
    os.mkfifo(target)
    with pytest.raises(storage.B13CheckpointStorageError, match="regular"):
        storage.load_checkpoint(root, session, packet.window_id)


def test_publish_write_and_fsync_failures_never_report_success_and_clean_temp(tmp_path, monkeypatch):
    _tree, session, packet, _semantic, accepted = _case(tmp_path)
    root = tmp_path / "private"
    target = storage.checkpoint_path(root, session, packet.window_id)
    monkeypatch.setattr(storage.tempfile, "mkstemp",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("temp failed")))
    with pytest.raises(storage.B13CheckpointStorageError, match="atomically"):
        storage.save_checkpoint(root, accepted, session)
    assert not target.exists() and not list(root.glob("*.tmp"))

    monkeypatch.undo()
    monkeypatch.setattr(storage.os, "write", lambda *_a: 0)
    with pytest.raises(storage.B13CheckpointStorageError, match="short"):
        storage.save_checkpoint(root, accepted, session)
    assert not target.exists() and not list(root.glob("*.tmp"))

    monkeypatch.undo()
    monkeypatch.setattr(storage.os, "link", lambda *_a: (_ for _ in ()).throw(OSError("link failed")))
    with pytest.raises(storage.B13CheckpointStorageError, match="atomically"):
        storage.save_checkpoint(root, accepted, session)
    assert not target.exists() and not list(root.glob("*.tmp"))

    monkeypatch.undo()
    original_fsync = storage.os.fsync
    calls = []

    def fail_directory_fsync(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError("directory sync failed")
        original_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", fail_directory_fsync)
    with pytest.raises(storage.B13CheckpointStorageError, match="directory"):
        storage.save_checkpoint(root, accepted, session)
    assert target.exists() and not list(root.glob("*.tmp"))


def test_preexisting_verified_checkpoint_survives_failed_retry_and_leftover_temp_is_ignored(tmp_path, monkeypatch):
    _tree, session, packet, _semantic, accepted = _case(tmp_path)
    root = tmp_path / "private"
    target = storage.save_checkpoint(root, accepted, session)
    original = target.read_bytes()
    (root / f".{target.name}.orphan.tmp").write_bytes(b"incomplete")
    monkeypatch.setattr(storage.os, "link", lambda *_a: pytest.fail("idempotent save must not publish"))
    assert storage.save_checkpoint(root, accepted, session) == target
    assert target.read_bytes() == original
    assert storage.load_checkpoint(root, session, packet.window_id) == accepted


def test_wrong_session_is_an_error_not_a_cache_hit(tmp_path):
    _tree, session, packet, _semantic, accepted = _case(tmp_path)
    root = tmp_path / "private"
    storage.save_checkpoint(root, accepted, session)
    broken = replace(session, preflight_verdict="FAIL")
    with pytest.raises(B13CheckpointEvidenceError):
        storage.load_checkpoint(root, broken, packet.window_id)
