from __future__ import annotations

import json
from pathlib import Path

import pytest

from short_drama.artifacts import ArtifactRef, ImmutableArtifactEnvelope, canonical_json_bytes
from short_drama.foundation import (
    CurrentPointer,
    FilePointerStore,
    LineageRef,
    LineageResolutionError,
    PointerConflictError,
    PointerIntegrityError,
    PointerKind,
    PointerLockedError,
    SupersessionError,
    resolve_lineage_ref,
)

from conftest import put_artifact


def test_lineage_round_trip_and_exact_resolution(artifact_store):
    ref = put_artifact(artifact_store)
    lineage = LineageRef("source", ref)
    assert LineageRef.from_dict(lineage.to_dict()) == lineage
    assert resolve_lineage_ref(artifact_store, lineage).ref == ref


def test_lineage_hash_mismatch_fails(artifact_store):
    ref = put_artifact(artifact_store)
    bad = ArtifactRef(ref.artifact_type, ref.artifact_id, ref.revision, "0" * 64)
    with pytest.raises(LineageResolutionError):
        resolve_lineage_ref(artifact_store, LineageRef("source", bad))


def test_current_pointer_kind_contract(artifact_store):
    target = put_artifact(artifact_store)
    with pytest.raises(Exception):
        CurrentPointer("demo", PointerKind.CURRENT, target, authority_ref=target)
    with pytest.raises(Exception):
        CurrentPointer("demo", PointerKind.CURRENT_APPROVED, target)


def test_pointer_create_move_history_and_supersession(tmp_path, artifact_store):
    v1 = put_artifact(artifact_store, revision=1, payload={"v": 1})
    v2 = put_artifact(artifact_store, revision=2, payload={"v": 2})
    pointers = FilePointerStore(tmp_path / "pointer-state", artifact_store)

    p1 = pointers.compare_and_set(
        pointer_id="story-current",
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=v1,
    )
    assert pointers.resolve_current_pointer_ref("story-current") == p1
    assert pointers.resolve_current("story-current").target_ref == v1

    p2 = pointers.compare_and_set(
        pointer_id="story-current",
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=p1,
        target_ref=v2,
    )
    assert pointers.resolve_current("story-current").target_ref == v2
    assert pointers.active_history("story-current") == (p2, p1)
    assert artifact_store.get_ref(v1).payload == {"v": 1}
    assert artifact_store.get_ref(p1).payload["target_ref"] == v1.to_dict()

    supersession = pointers.resolve_supersession("story-current", p2)
    assert supersession.superseded_pointer_ref == p1
    assert supersession.superseding_pointer_ref == p2
    assert supersession.superseded_target_ref == v1
    assert supersession.superseding_target_ref == v2


def test_pointer_stale_cas_fails(tmp_path, artifact_store):
    v1 = put_artifact(artifact_store, revision=1)
    v2 = put_artifact(artifact_store, revision=2)
    v3 = put_artifact(artifact_store, revision=3)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    p1 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=v1)
    p2 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=p1, target_ref=v2)
    with pytest.raises(PointerConflictError):
        pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=p1, target_ref=v3)
    assert pointers.resolve_current_pointer_ref("p") == p2


def test_pointer_exact_retry_is_idempotent(tmp_path, artifact_store):
    v1 = put_artifact(artifact_store, revision=1)
    v2 = put_artifact(artifact_store, revision=2)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    p1 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=v1)
    p2 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=p1, target_ref=v2)
    assert pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=p1, target_ref=v2) == p2


def test_pointer_first_create_retry_is_idempotent(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    p1 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=target)
    assert pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=target) == p1


def test_orphan_candidate_never_becomes_current_and_revision_is_skipped(tmp_path, artifact_store):
    v1 = put_artifact(artifact_store, revision=1)
    v2 = put_artifact(artifact_store, revision=2)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    p1 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=v1)

    orphan = CurrentPointer("p", PointerKind.CURRENT, v2, previous_pointer_ref=p1)
    orphan_env = ImmutableArtifactEnvelope.create(
        artifact_type="current_pointer",
        artifact_id="p",
        revision=2,
        schema_version=1,
        payload=orphan.to_dict(),
    )
    artifact_store.put(orphan_env)
    assert pointers.resolve_current_pointer_ref("p") == p1

    p3 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=p1, target_ref=v2)
    assert p3.revision == 3
    assert pointers.resolve_current_pointer_ref("p") == p3


def test_pointer_head_reload_preserves_authority(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    root = tmp_path / "state"
    p1 = FilePointerStore(root, artifact_store).compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=target)
    reloaded = FilePointerStore(root, artifact_store)
    assert reloaded.resolve_current_pointer_ref("p") == p1
    assert reloaded.require_current("p", target).target_ref == target


def test_pointer_head_noncanonical_tamper_fails(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    root = tmp_path / "state"
    pointers = FilePointerStore(root, artifact_store)
    p1 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=target)
    head = root / "heads" / "p.json"
    head.write_text(json.dumps({"current_pointer_ref": p1.to_dict()}, indent=2), encoding="utf-8")
    with pytest.raises(PointerIntegrityError):
        pointers.resolve_current("p")


def test_pointer_target_hash_mismatch_fails(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    bad = ArtifactRef(target.artifact_type, target.artifact_id, target.revision, "0" * 64)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    with pytest.raises(Exception):
        pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=bad)


def test_pointer_lock_fails_explicitly(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    root = tmp_path / "state"
    pointers = FilePointerStore(root, artifact_store)
    (root / "locks" / "p.lock").write_text("stale", encoding="utf-8")
    with pytest.raises(PointerLockedError):
        pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=target)


def test_initial_pointer_has_no_supersession(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    p1 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=target)
    with pytest.raises(SupersessionError):
        pointers.resolve_supersession("p", p1)


def test_pointer_noop_when_expected_current_already_has_desired_state(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    pointers = FilePointerStore(tmp_path / "state", artifact_store)
    p1 = pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=None, target_ref=target)
    assert pointers.compare_and_set(pointer_id="p", pointer_kind="CURRENT", expected_pointer_ref=p1, target_ref=target) == p1
    assert pointers.active_history("p") == (p1,)


def test_supersession_rejects_non_forward_pointer_revision(artifact_store):
    target = put_artifact(artifact_store)
    p2 = put_artifact(
        artifact_store, artifact_type="current_pointer", artifact_id="p", revision=2
    )
    p1 = put_artifact(
        artifact_store, artifact_type="current_pointer", artifact_id="p", revision=1
    )
    from short_drama.foundation import SupersessionRecord

    with pytest.raises(SupersessionError):
        SupersessionRecord("p", PointerKind.CURRENT, p2, p1, target, target)


def test_current_pointer_reload_revalidates_exact_target(tmp_path, artifact_store):
    target = put_artifact(artifact_store)
    root = tmp_path / "state"
    pointers = FilePointerStore(root, artifact_store)
    pointer_ref = pointers.compare_and_set(
        pointer_id="p",
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=None,
        target_ref=target,
    )

    # Replace the target with canonical bytes for a different envelope while keeping the path.
    target_path = artifact_store.root / target.artifact_type / target.artifact_id / "r00000001.json"
    tampered = ImmutableArtifactEnvelope.create(
        artifact_type=target.artifact_type,
        artifact_id=target.artifact_id,
        revision=target.revision,
        schema_version=1,
        payload={"tampered": True},
    )
    target_path.write_bytes(tampered.canonical_bytes())

    with pytest.raises(PointerIntegrityError):
        pointers.resolve_current_pointer_ref("p")
