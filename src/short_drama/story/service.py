from __future__ import annotations

from pathlib import Path
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import (
    FilePointerStore,
    LineageRef,
    PointerKind,
    PointerNotFoundError,
    ValidationFinding,
    ValidationReport,
    ValidationResult,
    load_validation_report,
    persist_validation_report,
)
from short_drama.io import load_yaml
from short_drama.validation import validate_file

from .chunking import (
    CHUNK_PLANNER_VERSION,
    ChunkManifest,
    ChunkPlanningProfile,
    SourceChunk,
    plan_chunks,
    validate_chunks_against_source,
)
from .errors import (
    ChunkCoverageError,
    StoryConfigError,
    StoryError,
    StoryIntegrityError,
    StoryPersistenceError,
    SourceReadError,
)
from .persistence import (
    chunk_pointer_id,
    chunk_validation_artifact_id,
    load_chunk_manifest,
    load_source_chunk,
    load_source_document,
    next_plan_revision,
    next_source_revision,
    persist_chunk_manifest,
    persist_source_chunk,
    persist_source_document,
    source_chunk_artifact_id,
    source_pointer_id,
    source_validation_artifact_id,
)
from .source import SourceDocument, base_language, build_source_document

DOCUMENT_ID = "src_001"


def _load_project(project_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(project_path).expanduser()
    if not path.is_file():
        raise StoryConfigError(f"project file not found: {path}")
    try:
        errors = validate_file(path, "project")
    except Exception as exc:
        raise StoryConfigError(f"failed to validate project file: {exc}") from exc
    if errors:
        raise StoryConfigError("invalid project file: " + "; ".join(errors))
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise StoryConfigError("project file must contain an object")
    return path.resolve(), data


def _load_profile(profile_path: str | Path) -> ChunkPlanningProfile:
    path = Path(profile_path).expanduser()
    if not path.is_file():
        raise StoryConfigError(f"chunk profile not found: {path}")
    try:
        data = load_yaml(path)
    except Exception as exc:
        raise StoryConfigError(f"failed to load chunk profile: {exc}") from exc
    if not isinstance(data, dict):
        raise StoryConfigError("chunk profile must contain an object")
    try:
        return ChunkPlanningProfile.from_dict(data)
    except StoryError:
        raise
    except Exception as exc:
        raise StoryConfigError(f"invalid chunk profile: {exc}") from exc


def _stores(runs_root: str | Path, project_id: str) -> tuple[FileArtifactStore, FilePointerStore]:
    root = Path(runs_root).expanduser() / project_id / "story"
    try:
        artifact_store = FileArtifactStore(root / "artifacts")
        pointer_store = FilePointerStore(root / "pointers", artifact_store)
    except Exception as exc:
        raise StoryPersistenceError(f"failed to initialize Story artifact stores: {exc}") from exc
    return artifact_store, pointer_store


def _current_pointer(
    pointer_store: FilePointerStore,
    pointer_id: str,
) -> tuple[ArtifactRef | None, ArtifactRef | None]:
    try:
        pointer_ref = pointer_store.resolve_current_pointer_ref(pointer_id)
        pointer = pointer_store.resolve_current(pointer_id)
        return pointer_ref, pointer.target_ref
    except PointerNotFoundError:
        return None, None


def _validation_ref_for_revision(
    store: FileArtifactStore,
    *,
    artifact_id: str,
    revision: int,
    expected_refs: tuple[LineageRef, ...],
    expected_findings: tuple[ValidationFinding, ...] = (),
) -> ArtifactRef:
    try:
        ref = store.get("validation_report", artifact_id, revision).ref
        report = load_validation_report(store, ref)
        expected_report = ValidationReport(
            validated_refs=expected_refs,
            findings=expected_findings,
        )
        if report != expected_report:
            raise StoryIntegrityError(
                "ValidationReport does not match the exact expected stage validation result"
            )
        if report.summary.result is ValidationResult.FAIL:
            raise StoryIntegrityError("current artifact is backed by a FAIL ValidationReport")
        return ref
    except Exception as exc:
        if isinstance(exc, StoryIntegrityError):
            raise
        raise StoryIntegrityError(
            f"matching ValidationReport is missing or invalid for {artifact_id} revision {revision}"
        ) from exc


def _source_findings(document: SourceDocument, source_ref: ArtifactRef) -> tuple[ValidationFinding, ...]:
    if base_language(document.source.declared_language) == base_language(
        document.source.detected_language
    ):
        return ()
    return (
        ValidationFinding(
            finding_id="a1-source-language-mismatch",
            code="a1.source.language-mismatch",
            severity="WARNING",
            owner_stage="A1",
            repair_route="A1_SOURCE_CONFIGURATION",
            message=(
                "declared source language "
                f"{document.source.declared_language!r} does not match detected base language "
                f"{document.source.detected_language!r}"
            ),
            artifact_refs=(source_ref,),
            path=("source", "declared_language"),
        ),
    )


def ingest_source_project(
    project_path: str | Path,
    *,
    runs_root: str | Path = "runs",
    encoding: str | None = None,
) -> dict[str, object]:
    project_file, project = _load_project(project_path)
    project_id = project["project_id"]
    source_cfg = project["source"]
    source_path_string = source_cfg["path"]
    source_path = project_file.parent / source_path_string
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise SourceReadError(f"failed to read source file {source_path}: {exc}") from exc

    candidate = build_source_document(
        project_id=project_id,
        document_id=DOCUMENT_ID,
        source_type=source_cfg["type"],
        source_path=source_path_string,
        raw=raw,
        declared_language=source_cfg["language"],
        explicit_encoding=encoding,
    )

    store, pointers = _stores(runs_root, project_id)
    pointer_id = source_pointer_id(project_id, DOCUMENT_ID)
    current_pointer_ref, current_source_ref = _current_pointer(pointers, pointer_id)

    if current_source_ref is not None:
        current = load_source_document(store, current_source_ref)
        if current.to_dict() == candidate.to_dict():
            report_ref = _validation_ref_for_revision(
                store,
                artifact_id=source_validation_artifact_id(project_id, DOCUMENT_ID),
                revision=current_source_ref.revision,
                expected_refs=(LineageRef("source_document", current_source_ref),),
                expected_findings=_source_findings(current, current_source_ref),
            )
            if pointers.resolve_current_pointer_ref(pointer_id) != current_pointer_ref:
                raise StoryPersistenceError(
                    "SourceDocument CURRENT pointer changed during reuse verification"
                )
            return {
                "source_document_ref": current_source_ref.to_dict(),
                "source_validation_report_ref": report_ref.to_dict(),
                "source_current_pointer_ref": current_pointer_ref.to_dict(),
                "reused": True,
                "validation": load_validation_report(store, report_ref).summary.to_dict(),
            }

    revision = next_source_revision(
        store,
        project_id,
        DOCUMENT_ID,
        current_source_ref,
    )
    source_ref = persist_source_document(store, candidate, revision=revision)
    report = ValidationReport(
        validated_refs=(LineageRef("source_document", source_ref),),
        findings=_source_findings(candidate, source_ref),
    )
    if report.summary.result is ValidationResult.FAIL:
        raise StoryIntegrityError("A1 validation failed; SourceDocument cannot become current")
    report_ref = persist_validation_report(
        store,
        report,
        artifact_id=source_validation_artifact_id(project_id, DOCUMENT_ID),
        revision=revision,
    )
    try:
        pointer_ref = pointers.compare_and_set(
            pointer_id=pointer_id,
            pointer_kind=PointerKind.CURRENT,
            expected_pointer_ref=current_pointer_ref,
            target_ref=source_ref,
        )
    except Exception as exc:
        raise StoryPersistenceError(
            "failed to publish SourceDocument CURRENT pointer; immutable artifacts remain historical"
        ) from exc

    return {
        "source_document_ref": source_ref.to_dict(),
        "source_validation_report_ref": report_ref.to_dict(),
        "source_current_pointer_ref": pointer_ref.to_dict(),
        "reused": False,
        "validation": report.summary.to_dict(),
    }


def _validate_persisted_manifest(
    store: FileArtifactStore,
    *,
    source: SourceDocument,
    source_ref: ArtifactRef,
    manifest_ref: ArtifactRef,
    expected_profile: ChunkPlanningProfile | None = None,
) -> tuple[ChunkManifest, tuple[SourceChunk, ...]]:
    manifest = load_chunk_manifest(store, manifest_ref)
    if manifest.source_document_ref != source_ref:
        raise StoryIntegrityError("ChunkManifest does not pin the expected SourceDocument ref")
    if manifest.project_id != source.project_id or manifest.document_id != source.document_id:
        raise StoryIntegrityError("ChunkManifest source identity mismatch")
    if expected_profile is not None and manifest.profile != expected_profile:
        raise StoryIntegrityError("ChunkManifest profile does not match expected profile")
    if manifest.chunk_count != len(manifest.chunk_refs):
        raise StoryIntegrityError("ChunkManifest chunk_count/ref count mismatch")

    chunks: list[SourceChunk] = []
    seen_chunk_ids: set[str] = set()
    for ref in manifest.chunk_refs:
        if ref.revision != manifest_ref.revision:
            raise StoryIntegrityError("SourceChunk revision must match ChunkManifest revision")
        chunk = load_source_chunk(store, ref)
        expected_id = source_chunk_artifact_id(
            manifest.project_id,
            manifest.document_id,
            manifest.profile.profile_id,
            chunk.chunk_id,
        )
        if ref.artifact_id != expected_id:
            raise StoryIntegrityError(
                f"SourceChunk artifact identity does not match chunk_id {chunk.chunk_id}"
            )
        if chunk.chunk_id in seen_chunk_ids:
            raise StoryIntegrityError(f"duplicate persisted chunk_id: {chunk.chunk_id}")
        seen_chunk_ids.add(chunk.chunk_id)
        chunks.append(chunk)

    coverage = validate_chunks_against_source(
        source,
        source_ref,
        manifest.profile,
        chunks,
    )
    if coverage != manifest.coverage:
        raise StoryIntegrityError("ChunkManifest coverage does not match resolved SourceChunks")
    return manifest, tuple(chunks)


def plan_chunks_project(
    project_path: str | Path,
    *,
    runs_root: str | Path = "runs",
    profile_path: str | Path,
) -> dict[str, object]:
    _project_file, project = _load_project(project_path)
    project_id = project["project_id"]
    profile = _load_profile(profile_path)
    store, pointers = _stores(runs_root, project_id)

    source_pointer = source_pointer_id(project_id, DOCUMENT_ID)
    source_pointer_ref, source_ref = _current_pointer(pointers, source_pointer)
    if source_ref is None or source_pointer_ref is None:
        raise StoryIntegrityError(
            "A1 SourceDocument is not current; run short-drama ingest-source first"
        )
    source = load_source_document(store, source_ref)

    manifest_pointer = chunk_pointer_id(project_id, DOCUMENT_ID, profile.profile_id)
    current_manifest_pointer_ref, current_manifest_ref = _current_pointer(
        pointers, manifest_pointer
    )

    if current_manifest_ref is not None:
        current_manifest, _chunks = _validate_persisted_manifest(
            store,
            source=source,
            source_ref=source_ref,
            manifest_ref=current_manifest_ref,
        )
        if (
            current_manifest.source_document_ref == source_ref
            and current_manifest.profile == profile
            and current_manifest.planner_version == CHUNK_PLANNER_VERSION
        ):
            report_ref = _validation_ref_for_revision(
                store,
                artifact_id=chunk_validation_artifact_id(
                    project_id, DOCUMENT_ID, profile.profile_id
                ),
                revision=current_manifest_ref.revision,
                expected_refs=(
                    LineageRef("source_document", source_ref),
                    LineageRef("chunk_manifest", current_manifest_ref),
                ),
            )
            if pointers.resolve_current(source_pointer).target_ref != source_ref:
                raise StoryPersistenceError(
                    "SourceDocument CURRENT pointer changed during chunk-plan reuse verification"
                )
            if (
                pointers.resolve_current_pointer_ref(manifest_pointer)
                != current_manifest_pointer_ref
            ):
                raise StoryPersistenceError(
                    "ChunkManifest CURRENT pointer changed during reuse verification"
                )
            return {
                "source_document_ref": source_ref.to_dict(),
                "chunk_manifest_ref": current_manifest_ref.to_dict(),
                "chunk_validation_report_ref": report_ref.to_dict(),
                "chunk_current_pointer_ref": current_manifest_pointer_ref.to_dict(),
                "chunk_count": current_manifest.chunk_count,
                "coverage": current_manifest.coverage.to_dict(),
                "reused": True,
                "validation": load_validation_report(store, report_ref).summary.to_dict(),
            }

    chunks, coverage = plan_chunks(source, source_ref, profile)
    revision = next_plan_revision(
        store,
        project_id=project_id,
        document_id=DOCUMENT_ID,
        profile_id=profile.profile_id,
        chunks=chunks,
        current_manifest_ref=current_manifest_ref,
    )

    chunk_refs = tuple(
        persist_source_chunk(
            store,
            chunk,
            profile_id=profile.profile_id,
            revision=revision,
        )
        for chunk in chunks
    )
    manifest = ChunkManifest(
        schema_version=1,
        project_id=project_id,
        document_id=DOCUMENT_ID,
        source_document_ref=source_ref,
        planner_version=CHUNK_PLANNER_VERSION,
        profile=profile,
        chunk_refs=chunk_refs,
        chunk_count=len(chunk_refs),
        coverage=coverage,
        state="CHUNKING_COMPLETE",
    )
    manifest_ref = persist_chunk_manifest(store, manifest, revision=revision)

    _validate_persisted_manifest(
        store,
        source=source,
        source_ref=source_ref,
        manifest_ref=manifest_ref,
        expected_profile=profile,
    )

    report = ValidationReport(
        validated_refs=(
            LineageRef("source_document", source_ref),
            LineageRef("chunk_manifest", manifest_ref),
        ),
        findings=(),
    )
    if report.summary.result is ValidationResult.FAIL:
        raise ChunkCoverageError("A2 validation failed; ChunkManifest cannot become current")
    report_ref = persist_validation_report(
        store,
        report,
        artifact_id=chunk_validation_artifact_id(
            project_id, DOCUMENT_ID, profile.profile_id
        ),
        revision=revision,
    )

    latest_source = pointers.resolve_current(source_pointer)
    if latest_source.target_ref != source_ref:
        raise StoryPersistenceError(
            "SourceDocument CURRENT pointer changed during A2 planning; "
            "new chunk artifacts remain historical/orphaned and are not published"
        )

    try:
        manifest_pointer_ref = pointers.compare_and_set(
            pointer_id=manifest_pointer,
            pointer_kind=PointerKind.CURRENT,
            expected_pointer_ref=current_manifest_pointer_ref,
            target_ref=manifest_ref,
        )
    except Exception as exc:
        raise StoryPersistenceError(
            "failed to publish ChunkManifest CURRENT pointer; immutable artifacts remain historical"
        ) from exc

    return {
        "source_document_ref": source_ref.to_dict(),
        "chunk_manifest_ref": manifest_ref.to_dict(),
        "chunk_validation_report_ref": report_ref.to_dict(),
        "chunk_current_pointer_ref": manifest_pointer_ref.to_dict(),
        "chunk_count": manifest.chunk_count,
        "coverage": manifest.coverage.to_dict(),
        "reused": False,
        "validation": report.summary.to_dict(),
    }
