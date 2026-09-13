from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import rfc8785

from short_drama.artifacts import (
    ArtifactPathError,
    CanonicalSerializationError,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
    artifact_content_hash,
    artifact_hash_material,
    canonical_json_bytes,
)


def test_rfc8785_number_formatting_contract():
    assert canonical_json_bytes(1.0) == b"1"
    assert canonical_json_bytes(-0.0) == b"0"
    assert canonical_json_bytes(1e-7) == b"1e-7"
    assert canonical_json_bytes(1e21) == b"1e+21"


def test_rfc8785_rejects_integer_outside_safe_domain():
    with pytest.raises(CanonicalSerializationError):
        canonical_json_bytes(9007199254740992)


def test_artifact_hash_preimage_is_externalizable():
    payload = {"duration": 1.0, "title": "demo"}
    material = artifact_hash_material(schema_version=3, payload=payload)
    expected = hashlib.sha256(rfc8785.dumps(material)).hexdigest()
    assert artifact_content_hash(schema_version=3, payload=payload) == expected


def test_persisted_hash_can_be_recomputed_from_raw_json(tmp_path):
    store = FileArtifactStore(tmp_path)
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type="story_bible",
        artifact_id="demo",
        revision=1,
        schema_version=2,
        payload={"duration": -0.0, "beats": [1, 2]},
    )
    store.put(envelope)
    path = next(tmp_path.rglob("r*.json"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    preimage = {"schema_version": raw["schema_version"], "payload": raw["payload"]}
    assert hashlib.sha256(rfc8785.dumps(preimage)).hexdigest() == raw["content_hash"]


def test_envelope_schema_documents_hash_preimage():
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "artifact-envelope.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    description = schema["description"]
    assert "RFC 8785" in description
    assert "schema_version" in description
    assert "payload" in description


@pytest.mark.parametrize("reserved", ["CON", "aux.txt", "Lpt9", "safe."])
def test_store_rejects_cross_platform_reserved_components(tmp_path, reserved):
    store = FileArtifactStore(tmp_path)
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type="story_bible",
        artifact_id=reserved,
        revision=1,
        schema_version=1,
        payload={"value": 1},
    )
    with pytest.raises(ArtifactPathError):
        store.put(envelope)
