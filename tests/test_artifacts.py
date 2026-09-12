from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from short_drama.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPathError,
    ArtifactRef,
    ArtifactStoreError,
    ArtifactValidationError,
    CanonicalSerializationError,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
    canonical_json_bytes,
    content_hash,
)


def make_envelope(payload=None, *, revision=1):
    return ImmutableArtifactEnvelope.create(
        artifact_type="story_bible",
        artifact_id="demo_001",
        revision=revision,
        schema_version=1,
        payload=payload if payload is not None else {"title": "测试", "beats": [1, 2]},
    )


def only_artifact_file(root: Path) -> Path:
    files = list(root.rglob("r*.json"))
    assert len(files) == 1
    return files[0]


def test_canonical_mapping_order_is_stable():
    left = {"a": 1, "b": {"x": 1, "y": 2}}
    right = {"b": {"y": 2, "x": 1}, "a": 1}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert content_hash(left) == content_hash(right)


def test_canonical_array_order_is_significant():
    assert canonical_json_bytes([1, 2]) != canonical_json_bytes([2, 1])
    assert content_hash([1, 2]) != content_hash([2, 1])


def test_canonical_unicode_round_trip_and_repeatability():
    value = {"中文": "人物与场景", "emoji": "🎬"}
    first = canonical_json_bytes(value)
    assert first == canonical_json_bytes(value)
    assert json.loads(first) == value
    assert b"\\u" not in first


def test_semantic_content_change_changes_hash():
    assert content_hash({"value": 1}) != content_hash({"value": 2})


@pytest.mark.parametrize("value", [{1: "bad"}, {"bad": {1, 2}}, (1, 2)])
def test_non_json_values_are_rejected(value):
    with pytest.raises(CanonicalSerializationError):
        canonical_json_bytes(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(value):
    with pytest.raises(CanonicalSerializationError):
        canonical_json_bytes({"value": value})


def test_cyclic_containers_are_rejected():
    value = []
    value.append(value)
    with pytest.raises(CanonicalSerializationError):
        canonical_json_bytes(value)


def test_artifact_ref_validates_and_is_frozen():
    digest = "a" * 64
    ref = ArtifactRef("story_bible", "demo", 1, digest)
    assert ArtifactRef.from_dict(ref.to_dict()) == ref
    with pytest.raises(FrozenInstanceError):
        ref.revision = 2
    with pytest.raises(ArtifactValidationError):
        ArtifactRef("story_bible", "demo", 0, digest)
    with pytest.raises(ArtifactValidationError):
        ArtifactRef("story_bible", "demo", 1, "ABC")


def test_envelope_snapshots_caller_payload_and_returns_copies():
    payload = {"nested": {"value": 1}, "items": [1, 2]}
    envelope = make_envelope(payload)
    original_hash = envelope.content_hash

    payload["nested"]["value"] = 99
    payload["items"].append(3)
    assert envelope.payload == {"nested": {"value": 1}, "items": [1, 2]}
    assert envelope.content_hash == original_hash

    returned = envelope.payload
    returned["nested"]["value"] = 100
    assert envelope.payload["nested"]["value"] == 1


def test_envelope_round_trip_and_ref_consistency():
    envelope = make_envelope()
    restored = ImmutableArtifactEnvelope.from_dict(envelope.to_dict())
    assert restored == envelope
    assert restored.ref.content_hash == content_hash(
        {"schema_version": restored.schema_version, "payload": restored.payload}
    )
    assert restored.canonical_bytes() == envelope.canonical_bytes()


def test_schema_version_is_part_of_content_hash_material():
    payload = {"value": 1}
    v1 = ImmutableArtifactEnvelope.create(
        artifact_type="story_bible",
        artifact_id="demo",
        revision=1,
        schema_version=1,
        payload=payload,
    )
    v2 = ImmutableArtifactEnvelope.create(
        artifact_type="story_bible",
        artifact_id="demo",
        revision=1,
        schema_version=2,
        payload=payload,
    )
    assert v1.content_hash != v2.content_hash


def test_invalid_utf8_surrogate_is_rejected():
    with pytest.raises(CanonicalSerializationError):
        canonical_json_bytes({"text": "\ud800"})


def test_envelope_rejects_content_hash_mismatch():
    value = make_envelope().to_dict()
    value["content_hash"] = "0" * 64
    with pytest.raises(ArtifactValidationError):
        ImmutableArtifactEnvelope.from_dict(value)


def test_external_schemas_are_valid_and_accept_models():
    schema_root = Path(__file__).resolve().parents[1] / "schemas"
    ref_schema = json.loads((schema_root / "artifact-ref.schema.json").read_text(encoding="utf-8"))
    envelope_schema = json.loads(
        (schema_root / "artifact-envelope.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(ref_schema)
    Draft202012Validator.check_schema(envelope_schema)
    envelope = make_envelope()
    Draft202012Validator(ref_schema).validate(envelope.ref.to_dict())
    Draft202012Validator(envelope_schema).validate(envelope.to_dict())


def test_store_put_get_and_idempotent_replay(tmp_path):
    store = FileArtifactStore(tmp_path)
    envelope = make_envelope()
    ref = store.put(envelope)
    assert ref == envelope.ref
    assert store.put(envelope) == ref
    assert store.get("story_bible", "demo_001", 1) == envelope
    assert store.get_ref(ref) == envelope


def test_store_missing_artifact_fails(tmp_path):
    store = FileArtifactStore(tmp_path)
    with pytest.raises(ArtifactNotFoundError):
        store.get("story_bible", "missing", 1)


def test_store_detects_payload_tampering(tmp_path):
    store = FileArtifactStore(tmp_path)
    store.put(make_envelope())
    path = only_artifact_file(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["payload"]["title"] = "tampered"
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ArtifactIntegrityError):
        store.get("story_bible", "demo_001", 1)


def test_store_detects_declared_hash_tampering(tmp_path):
    store = FileArtifactStore(tmp_path)
    store.put(make_envelope())
    path = only_artifact_file(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["content_hash"] = "0" * 64
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ArtifactIntegrityError):
        store.get("story_bible", "demo_001", 1)


def test_store_detects_noncanonical_persisted_bytes(tmp_path):
    store = FileArtifactStore(tmp_path)
    store.put(make_envelope())
    path = only_artifact_file(tmp_path)
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        store.get("story_bible", "demo_001", 1)


def test_store_ref_hash_mismatch_fails(tmp_path):
    store = FileArtifactStore(tmp_path)
    envelope = make_envelope()
    store.put(envelope)
    wrong = ArtifactRef(
        artifact_type=envelope.artifact_type,
        artifact_id=envelope.artifact_id,
        revision=envelope.revision,
        content_hash="0" * 64,
    )
    with pytest.raises(ArtifactIntegrityError):
        store.get_ref(wrong)


def test_store_prevents_conflicting_overwrite(tmp_path):
    store = FileArtifactStore(tmp_path)
    store.put(make_envelope({"value": 1}))
    with pytest.raises(ArtifactConflictError):
        store.put(make_envelope({"value": 2}))
    assert store.get("story_bible", "demo_001", 1).payload == {"value": 1}


def test_failed_publication_leaves_no_partial_target(tmp_path, monkeypatch):
    store = FileArtifactStore(tmp_path)
    envelope = make_envelope()

    def fail_link(src, dst):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(ArtifactStoreError):
        store.put(envelope)
    assert not list(tmp_path.rglob("r*.json"))
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    "artifact_type,artifact_id",
    [
        ("../escape", "safe"),
        ("safe", "../escape"),
        ("safe/type", "id"),
        ("type", "id\\escape"),
        ("type:windows", "id"),
    ],
)
def test_store_rejects_unsafe_storage_identity(tmp_path, artifact_type, artifact_id):
    store = FileArtifactStore(tmp_path)
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        revision=1,
        schema_version=1,
        payload={"value": 1},
    )
    with pytest.raises(ArtifactPathError):
        store.put(envelope)


def test_multiple_revisions_coexist(tmp_path):
    store = FileArtifactStore(tmp_path)
    first = make_envelope({"value": 1}, revision=1)
    second = make_envelope({"value": 2}, revision=2)
    store.put(first)
    store.put(second)
    assert store.get("story_bible", "demo_001", 1) == first
    assert store.get("story_bible", "demo_001", 2) == second
    assert len(list(tmp_path.rglob("r*.json"))) == 2
