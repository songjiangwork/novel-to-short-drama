from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from short_drama.artifacts import (
    ArtifactRef,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
)
from short_drama.foundation import FilePointerStore, PointerKind
from short_drama.story import (
    StoryIntegrityError,
    build_source_document,
    ingest_source_project,
    plan_chunks_project,
)
from short_drama.story.chunking import ChunkManifest
from short_drama.story.persistence import (
    chunk_pointer_id,
    load_chunk_manifest,
    load_source_chunk,
    load_source_document,
    persist_chunk_manifest,
    persist_source_chunk,
    source_document_artifact_id,
)


def _patch_language(monkeypatch, language: str) -> None:
    import short_drama.story.source as source_module

    monkeypatch.setattr(
        source_module,
        "detect_language",
        lambda _text: (language, "langid-1.1.6"),
    )


def _project(tmp_path: Path, text: str) -> Path:
    root = tmp_path / "project"
    (root / "source").mkdir(parents=True)
    (root / "source" / "novel.txt").write_text(
        text,
        encoding="utf-8",
    )
    data = {
        "schema_version": 1,
        "project_id": "story-hardening",
        "title": "Story Hardening",
        "source": {
            "type": "txt",
            "path": "source/novel.txt",
            "language": "en-CA",
        },
        "production": {
            "output_language": "en-CA",
            "profile": "h3_v1",
        },
        "approval_policy": {
            "adaptation_requires_approval": True,
            "generation_preflight_requires_approval": True,
            "shot_qc_requires_approval": True,
        },
    }
    project_path = root / "project.yaml"
    project_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return project_path


def _profile(
    path: Path,
    *,
    ownership: int,
    overlap: int = 0,
) -> Path:
    data = {
        "schema_version": 1,
        "profile_id": "story-hardening-v1",
        "token_counter": "utf8-bytes-div3-v1",
        "ownership_token_budget": ownership,
        "context_overlap_token_budget": overlap,
        "context_token_budget": ownership + 2 * overlap,
    }
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_heading_detection_does_not_promote_prose(monkeypatch):
    _patch_language(monkeypatch, "en")
    document = build_source_document(
        project_id="story-hardening",
        document_id="src_001",
        source_type="txt",
        source_path="source/novel.txt",
        raw=(
            "Chapter one begins with a storm.\n"
            "Prologue to the story was already over.\n\n"
            "Chapter One\n"
            "The real first chapter begins here.\n\n"
            "Chapter Two: Return\n"
            "Mara came home.\n"
        ).encode("utf-8"),
        declared_language="en-CA",
    )

    assert [chapter.heading_kind for chapter in document.chapters] == [
        "synthetic",
        "chapter",
        "chapter",
    ]
    assert document.chapters[0].title_original is None
    assert [
        paragraph.text_original
        for paragraph in document.chapters[0].paragraphs
    ] == [
        "Chapter one begins with a storm.",
        "Prologue to the story was already over.",
    ]
    assert document.chapters[1].title_original == "Chapter One"
    assert document.chapters[2].title_original == "Chapter Two: Return"


def test_language_mismatch_is_warning_not_blocking(monkeypatch, tmp_path):
    _patch_language(monkeypatch, "fr")
    project_path = _project(
        tmp_path,
        "Chapter 1\nThis source is intentionally English.\n",
    )
    result = ingest_source_project(
        project_path,
        runs_root=tmp_path / "runs",
    )
    assert result["validation"] == {
        "blocking_count": 0,
        "review_required_count": 0,
        "warning_count": 1,
        "result": "PASS",
    }
    assert result["reused"] is False


def test_profile_semantic_change_creates_new_plan_revision(
    monkeypatch,
    tmp_path,
):
    _patch_language(monkeypatch, "en")
    project_path = _project(
        tmp_path,
        "Chapter 1\naaaaaa\n\nbbbbbb\n\ncccccc\n",
    )
    profile_path = _profile(
        tmp_path / "profile.yaml",
        ownership=8,
    )
    runs_root = tmp_path / "runs"
    ingest_source_project(project_path, runs_root=runs_root)
    first = plan_chunks_project(
        project_path,
        runs_root=runs_root,
        profile_path=profile_path,
    )

    _profile(profile_path, ownership=5)
    second = plan_chunks_project(
        project_path,
        runs_root=runs_root,
        profile_path=profile_path,
    )
    assert second["reused"] is False
    assert (
        second["chunk_manifest_ref"]["revision"]
        > first["chunk_manifest_ref"]["revision"]
    )

    store = FileArtifactStore(
        runs_root / "story-hardening" / "story" / "artifacts"
    )
    assert store.get_ref(
        ArtifactRef.from_dict(first["chunk_manifest_ref"])
    )
    assert store.get_ref(
        ArtifactRef.from_dict(second["chunk_manifest_ref"])
    )


def test_non_normal_current_manifest_chunk_order_fails_closed(
    monkeypatch,
    tmp_path,
):
    _patch_language(monkeypatch, "en")
    project_path = _project(
        tmp_path,
        "Chapter 1\naaaaaa\n\nbbbbbb\n\ncccccc\n",
    )
    profile_path = _profile(
        tmp_path / "profile.yaml",
        ownership=5,
    )
    runs_root = tmp_path / "runs"
    ingest_source_project(project_path, runs_root=runs_root)
    first = plan_chunks_project(
        project_path,
        runs_root=runs_root,
        profile_path=profile_path,
    )

    store = FileArtifactStore(
        runs_root / "story-hardening" / "story" / "artifacts"
    )
    pointers = FilePointerStore(
        runs_root / "story-hardening" / "story" / "pointers",
        store,
    )
    first_ref = ArtifactRef.from_dict(first["chunk_manifest_ref"])
    manifest = load_chunk_manifest(store, first_ref)
    assert manifest.chunk_count >= 2

    copied_refs = tuple(
        persist_source_chunk(
            store,
            load_source_chunk(store, ref),
            profile_id=manifest.profile.profile_id,
            revision=99,
        )
        for ref in manifest.chunk_refs
    )
    non_normal = ChunkManifest(
        schema_version=manifest.schema_version,
        project_id=manifest.project_id,
        document_id=manifest.document_id,
        source_document_ref=manifest.source_document_ref,
        planner_version=manifest.planner_version,
        profile=manifest.profile,
        chunk_refs=tuple(reversed(copied_refs)),
        chunk_count=len(copied_refs),
        coverage=manifest.coverage,
        state=manifest.state,
    )
    non_normal_ref = persist_chunk_manifest(
        store,
        non_normal,
        revision=99,
    )

    pointer_id = chunk_pointer_id(
        manifest.project_id,
        manifest.document_id,
        manifest.profile.profile_id,
    )
    old_pointer_ref = pointers.resolve_current_pointer_ref(pointer_id)
    pointers.compare_and_set(
        pointer_id=pointer_id,
        pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=old_pointer_ref,
        target_ref=non_normal_ref,
    )

    with pytest.raises(
        StoryIntegrityError,
        match="deterministic planner output",
    ):
        plan_chunks_project(
            project_path,
            runs_root=runs_root,
            profile_path=profile_path,
        )


def test_source_loader_rejects_wrong_artifact_identity(monkeypatch, tmp_path):
    _patch_language(monkeypatch, "en")
    document = build_source_document(
        project_id="story-hardening",
        document_id="src_001",
        source_type="txt",
        source_path="source/novel.txt",
        raw=b"Chapter One\nAlpha paragraph.\n",
        declared_language="en-CA",
    )
    store = FileArtifactStore(tmp_path / "artifacts")
    wrong = ImmutableArtifactEnvelope.create(
        artifact_type="source_document",
        artifact_id="other-project.src_001",
        revision=1,
        schema_version=1,
        payload=document.to_dict(),
    )
    ref = store.put(wrong)

    with pytest.raises(
        StoryIntegrityError,
        match="artifact_id does not match payload identity",
    ):
        load_source_document(store, ref)


def test_source_loader_rejects_non_contract_parser_metadata(
    monkeypatch,
    tmp_path,
):
    _patch_language(monkeypatch, "en")
    document = build_source_document(
        project_id="story-hardening",
        document_id="src_001",
        source_type="txt",
        source_path="source/novel.txt",
        raw=b"Chapter One\nAlpha paragraph.\n",
        declared_language="en-CA",
    )
    payload = document.to_dict()
    payload["normalization"]["parser_id"] = "unexpected-parser"

    store = FileArtifactStore(tmp_path / "artifacts")
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type="source_document",
        artifact_id=source_document_artifact_id(
            "story-hardening",
            "src_001",
        ),
        revision=1,
        schema_version=1,
        payload=payload,
    )
    ref = store.put(envelope)

    with pytest.raises(
        StoryIntegrityError,
        match="invalid persisted SourceDocument",
    ):
        load_source_document(store, ref)
