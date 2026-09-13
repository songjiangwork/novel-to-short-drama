from __future__ import annotations

import pytest

from short_drama.artifacts import FileArtifactStore, ImmutableArtifactEnvelope


@pytest.fixture
def artifact_store(tmp_path):
    return FileArtifactStore(tmp_path / "artifacts")


def put_artifact(store, *, artifact_type="story", artifact_id="demo", revision=1, payload=None):
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        revision=revision,
        schema_version=1,
        payload={} if payload is None else payload,
    )
    store.put(envelope)
    return envelope.ref
