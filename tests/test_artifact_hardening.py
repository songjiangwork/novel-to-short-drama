from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import short_drama.artifacts.canonical as canonical_module
from short_drama.artifacts import (
    ArtifactRef,
    ArtifactValidationError,
    CanonicalSerializationError,
    ImmutableArtifactEnvelope,
    strict_json_loads,
)


def _invalid_identities():
    return ["bad" + chr(0) + "id", "bad" + chr(0xD800) + "id"]


@pytest.mark.parametrize("bad_identity", _invalid_identities())
def test_artifact_ref_rejects_noncanonical_identity_text(bad_identity):
    with pytest.raises(ArtifactValidationError):
        ArtifactRef(
            artifact_type="story_bible",
            artifact_id=bad_identity,
            revision=1,
            content_hash="a" * 64,
        )


def test_artifact_schemas_match_runtime_identity_text_rules():
    schema_root = Path(__file__).resolve().parents[1] / "schemas"
    ref_schema = json.loads((schema_root / "artifact-ref.schema.json").read_text(encoding="utf-8"))
    envelope_schema = json.loads(
        (schema_root / "artifact-envelope.schema.json").read_text(encoding="utf-8")
    )
    ref_validator = Draft202012Validator(ref_schema)
    envelope_validator = Draft202012Validator(envelope_schema)

    ref = ArtifactRef("story_bible", "demo", 1, "a" * 64).to_dict()
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type="story_bible",
        artifact_id="demo",
        revision=1,
        schema_version=1,
        payload={"value": 1},
    ).to_dict()

    for bad_identity in _invalid_identities():
        bad_ref = dict(ref)
        bad_ref["artifact_id"] = bad_identity
        assert list(ref_validator.iter_errors(bad_ref))

        bad_envelope = dict(envelope)
        bad_envelope["artifact_id"] = bad_identity
        assert list(envelope_validator.iter_errors(bad_envelope))


def test_strict_json_loads_wraps_parser_value_error(monkeypatch):
    def fail_parser(*args, **kwargs):
        raise ValueError("simulated parser numeric limit")

    monkeypatch.setattr(canonical_module.json, "loads", fail_parser)
    with pytest.raises(CanonicalSerializationError, match="invalid JSON"):
        strict_json_loads("0")
