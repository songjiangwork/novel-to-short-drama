from __future__ import annotations

from pathlib import Path

import pytest

from short_drama.artifacts import (
    ArtifactPathError,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
)


def _make_envelope(*, artifact_type: str = "story_bible", artifact_id: str = "demo"):
    return ImmutableArtifactEnvelope.create(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        revision=1,
        schema_version=1,
        payload={"value": 1},
    )


def _symlink_or_skip(link: Path, target: Path, *, is_directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=is_directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")


@pytest.mark.parametrize(
    "artifact_type,artifact_id",
    [
        ("Story_Bible", "demo"),
        ("story_bible", "Demo"),
    ],
)
def test_store_rejects_case_ambiguous_storage_components(tmp_path, artifact_type, artifact_id):
    store = FileArtifactStore(tmp_path)
    with pytest.raises(ArtifactPathError):
        store.put(_make_envelope(artifact_type=artifact_type, artifact_id=artifact_id))


def test_store_rejects_symlink_root(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "store-link"
    _symlink_or_skip(link, actual, is_directory=True)

    with pytest.raises(ArtifactPathError):
        FileArtifactStore(link)


def test_store_rejects_symlinked_artifact_directory(tmp_path):
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    store = FileArtifactStore(root)
    outside.mkdir()
    _symlink_or_skip(root / "story_bible", outside, is_directory=True)

    with pytest.raises(ArtifactPathError):
        store.put(_make_envelope())

    assert not list(outside.iterdir())


def test_store_rejects_symlinked_revision_target(tmp_path):
    root = tmp_path / "store"
    store = FileArtifactStore(root)
    target_dir = root / "story_bible" / "demo"
    target_dir.mkdir(parents=True)
    outside_file = tmp_path / "outside.json"
    outside_file.write_text("{}", encoding="utf-8")
    revision_path = target_dir / "r00000001.json"
    _symlink_or_skip(revision_path, outside_file, is_directory=False)

    with pytest.raises(ArtifactPathError):
        store.get("story_bible", "demo", 1)
