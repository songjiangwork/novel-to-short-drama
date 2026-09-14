from __future__ import annotations

from collections.abc import Iterable

from short_drama.artifacts import (
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactRef,
    FileArtifactStore,
    ImmutableArtifactEnvelope,
)

from .chunking import (
    CHUNK_MANIFEST_ARTIFACT_TYPE,
    CHUNK_MANIFEST_SCHEMA_VERSION,
    SOURCE_CHUNK_ARTIFACT_TYPE,
    SOURCE_CHUNK_SCHEMA_VERSION,
    ChunkManifest,
    SourceChunk,
    plan_chunks,
)
from .errors import StoryIntegrityError, StoryPersistenceError
from .source import (
    SOURCE_DOCUMENT_ARTIFACT_TYPE,
    SOURCE_DOCUMENT_SCHEMA_VERSION,
    SourceDocument,
)


def source_document_artifact_id(project_id: str, document_id: str) -> str:
    return f"{project_id}.{document_id}"


def source_pointer_id(project_id: str, document_id: str) -> str:
    return f"{project_id}.source.{document_id}"


def source_validation_artifact_id(project_id: str, document_id: str) -> str:
    return f"{project_id}.{document_id}.a1-validation"


def chunk_manifest_artifact_id(
    project_id: str,
    document_id: str,
    profile_id: str,
) -> str:
    return f"{project_id}.{document_id}.{profile_id}"


def chunk_pointer_id(
    project_id: str,
    document_id: str,
    profile_id: str,
) -> str:
    return f"{project_id}.chunks.{document_id}.{profile_id}"


def chunk_validation_artifact_id(
    project_id: str,
    document_id: str,
    profile_id: str,
) -> str:
    return f"{project_id}.{document_id}.{profile_id}.a2-validation"


def source_chunk_artifact_id(
    project_id: str,
    document_id: str,
    profile_id: str,
    chunk_id: str,
) -> str:
    return (
        f"{project_id}.{document_id}.{profile_id}.{chunk_id.lower()}"
    )


def _next_free_revision(
    store: FileArtifactStore,
    *,
    artifact_type: str,
    artifact_id: str,
    start: int = 1,
) -> int:
    revision = max(1, start)
    while True:
        try:
            store.get(artifact_type, artifact_id, revision)
        except ArtifactNotFoundError:
            return revision
        except ArtifactError as exc:
            raise StoryPersistenceError(
                f"cannot inspect {artifact_type}/{artifact_id} "
                f"revision {revision}"
            ) from exc
        revision += 1


def next_source_revision(
    store: FileArtifactStore,
    project_id: str,
    document_id: str,
    current_target_ref: ArtifactRef | None,
) -> int:
    start = (
        1
        if current_target_ref is None
        else current_target_ref.revision + 1
    )
    return _next_free_revision(
        store,
        artifact_type=SOURCE_DOCUMENT_ARTIFACT_TYPE,
        artifact_id=source_document_artifact_id(
            project_id,
            document_id,
        ),
        start=start,
    )


def next_plan_revision(
    store: FileArtifactStore,
    *,
    project_id: str,
    document_id: str,
    profile_id: str,
    chunks: Iterable[SourceChunk],
    current_manifest_ref: ArtifactRef | None,
) -> int:
    chunks = tuple(chunks)
    revision = (
        1
        if current_manifest_ref is None
        else current_manifest_ref.revision + 1
    )
    manifest_id = chunk_manifest_artifact_id(
        project_id,
        document_id,
        profile_id,
    )
    chunk_ids = tuple(
        source_chunk_artifact_id(
            project_id,
            document_id,
            profile_id,
            chunk.chunk_id,
        )
        for chunk in chunks
    )

    while True:
        targets = [
            (CHUNK_MANIFEST_ARTIFACT_TYPE, manifest_id),
            *[
                (SOURCE_CHUNK_ARTIFACT_TYPE, chunk_id)
                for chunk_id in chunk_ids
            ],
        ]
        occupied = False
        for artifact_type, artifact_id in targets:
            try:
                store.get(artifact_type, artifact_id, revision)
            except ArtifactNotFoundError:
                continue
            except ArtifactError as exc:
                raise StoryPersistenceError(
                    f"cannot inspect {artifact_type}/{artifact_id} "
                    f"revision {revision}"
                ) from exc
            occupied = True
            break
        if not occupied:
            return revision
        revision += 1


def persist_source_document(
    store: FileArtifactStore,
    document: SourceDocument,
    *,
    revision: int,
) -> ArtifactRef:
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=SOURCE_DOCUMENT_ARTIFACT_TYPE,
        artifact_id=source_document_artifact_id(
            document.project_id,
            document.document_id,
        ),
        revision=revision,
        schema_version=SOURCE_DOCUMENT_SCHEMA_VERSION,
        payload=document.to_dict(),
    )
    try:
        return store.put(envelope)
    except ArtifactError as exc:
        raise StoryPersistenceError(
            f"failed to persist SourceDocument: {exc}"
        ) from exc


def load_source_document(
    store: FileArtifactStore,
    ref: ArtifactRef,
) -> SourceDocument:
    if (
        not isinstance(ref, ArtifactRef)
        or ref.artifact_type != SOURCE_DOCUMENT_ARTIFACT_TYPE
    ):
        raise StoryIntegrityError(
            "SourceDocument ref has the wrong artifact_type"
        )
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise StoryIntegrityError(
            f"failed to resolve SourceDocument: {ref!r}"
        ) from exc
    if envelope.schema_version != SOURCE_DOCUMENT_SCHEMA_VERSION:
        raise StoryIntegrityError(
            "unsupported SourceDocument schema_version: "
            f"{envelope.schema_version}; "
            f"supported={SOURCE_DOCUMENT_SCHEMA_VERSION}"
        )
    payload = envelope.payload
    if not isinstance(payload, dict):
        raise StoryIntegrityError(
            "persisted SourceDocument payload must be an object"
        )
    try:
        document = SourceDocument.from_dict(payload)
    except Exception as exc:
        raise StoryIntegrityError(
            f"invalid persisted SourceDocument: {exc}"
        ) from exc

    expected_id = source_document_artifact_id(
        document.project_id,
        document.document_id,
    )
    if ref.artifact_id != expected_id:
        raise StoryIntegrityError(
            "SourceDocument artifact_id does not match payload identity"
        )
    if document.to_dict() != payload:
        raise StoryIntegrityError(
            "persisted SourceDocument payload is not in canonical "
            "semantic form"
        )
    return document


def persist_source_chunk(
    store: FileArtifactStore,
    chunk: SourceChunk,
    *,
    profile_id: str,
    revision: int,
) -> ArtifactRef:
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=SOURCE_CHUNK_ARTIFACT_TYPE,
        artifact_id=source_chunk_artifact_id(
            chunk.project_id,
            chunk.document_id,
            profile_id,
            chunk.chunk_id,
        ),
        revision=revision,
        schema_version=SOURCE_CHUNK_SCHEMA_VERSION,
        payload=chunk.to_dict(),
    )
    try:
        return store.put(envelope)
    except ArtifactError as exc:
        raise StoryPersistenceError(
            f"failed to persist SourceChunk {chunk.chunk_id}: {exc}"
        ) from exc


def load_source_chunk(
    store: FileArtifactStore,
    ref: ArtifactRef,
) -> SourceChunk:
    if (
        not isinstance(ref, ArtifactRef)
        or ref.artifact_type != SOURCE_CHUNK_ARTIFACT_TYPE
    ):
        raise StoryIntegrityError(
            "SourceChunk ref has the wrong artifact_type"
        )
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise StoryIntegrityError(
            f"failed to resolve SourceChunk: {ref!r}"
        ) from exc
    if envelope.schema_version != SOURCE_CHUNK_SCHEMA_VERSION:
        raise StoryIntegrityError(
            "unsupported SourceChunk schema_version: "
            f"{envelope.schema_version}; "
            f"supported={SOURCE_CHUNK_SCHEMA_VERSION}"
        )
    payload = envelope.payload
    if not isinstance(payload, dict):
        raise StoryIntegrityError(
            "persisted SourceChunk payload must be an object"
        )
    try:
        chunk = SourceChunk.from_dict(payload)
    except Exception as exc:
        raise StoryIntegrityError(
            f"invalid persisted SourceChunk: {exc}"
        ) from exc

    expected_prefix = f"{chunk.project_id}.{chunk.document_id}."
    expected_suffix = f".{chunk.chunk_id.lower()}"
    if (
        not ref.artifact_id.startswith(expected_prefix)
        or not ref.artifact_id.endswith(expected_suffix)
        or len(ref.artifact_id)
        <= len(expected_prefix) + len(expected_suffix)
    ):
        raise StoryIntegrityError(
            "SourceChunk artifact_id does not match payload identity"
        )
    if chunk.to_dict() != payload:
        raise StoryIntegrityError(
            "persisted SourceChunk payload is not in canonical "
            "semantic form"
        )
    return chunk


def persist_chunk_manifest(
    store: FileArtifactStore,
    manifest: ChunkManifest,
    *,
    revision: int,
) -> ArtifactRef:
    envelope = ImmutableArtifactEnvelope.create(
        artifact_type=CHUNK_MANIFEST_ARTIFACT_TYPE,
        artifact_id=chunk_manifest_artifact_id(
            manifest.project_id,
            manifest.document_id,
            manifest.profile.profile_id,
        ),
        revision=revision,
        schema_version=CHUNK_MANIFEST_SCHEMA_VERSION,
        payload=manifest.to_dict(),
    )
    try:
        return store.put(envelope)
    except ArtifactError as exc:
        raise StoryPersistenceError(
            f"failed to persist ChunkManifest: {exc}"
        ) from exc


def load_chunk_manifest(
    store: FileArtifactStore,
    ref: ArtifactRef,
) -> ChunkManifest:
    if (
        not isinstance(ref, ArtifactRef)
        or ref.artifact_type != CHUNK_MANIFEST_ARTIFACT_TYPE
    ):
        raise StoryIntegrityError(
            "ChunkManifest ref has the wrong artifact_type"
        )
    try:
        envelope = store.get_ref(ref)
    except ArtifactError as exc:
        raise StoryIntegrityError(
            f"failed to resolve ChunkManifest: {ref!r}"
        ) from exc
    if envelope.schema_version != CHUNK_MANIFEST_SCHEMA_VERSION:
        raise StoryIntegrityError(
            "unsupported ChunkManifest schema_version: "
            f"{envelope.schema_version}; "
            f"supported={CHUNK_MANIFEST_SCHEMA_VERSION}"
        )
    payload = envelope.payload
    if not isinstance(payload, dict):
        raise StoryIntegrityError(
            "persisted ChunkManifest payload must be an object"
        )
    try:
        manifest = ChunkManifest.from_dict(payload)
    except Exception as exc:
        raise StoryIntegrityError(
            f"invalid persisted ChunkManifest: {exc}"
        ) from exc

    expected_id = chunk_manifest_artifact_id(
        manifest.project_id,
        manifest.document_id,
        manifest.profile.profile_id,
    )
    if ref.artifact_id != expected_id:
        raise StoryIntegrityError(
            "ChunkManifest artifact_id does not match payload identity"
        )
    if manifest.to_dict() != payload:
        raise StoryIntegrityError(
            "persisted ChunkManifest payload is not in canonical "
            "semantic form"
        )

    try:
        source = load_source_document(
            store,
            manifest.source_document_ref,
        )
        persisted_chunks = tuple(
            load_source_chunk(store, chunk_ref)
            for chunk_ref in manifest.chunk_refs
        )
        canonical_chunks, canonical_coverage = plan_chunks(
            source,
            manifest.source_document_ref,
            manifest.profile,
        )
    except StoryIntegrityError:
        raise
    except Exception as exc:
        raise StoryIntegrityError(
            "failed to verify ChunkManifest deterministic planner output"
        ) from exc

    if (
        persisted_chunks != canonical_chunks
        or manifest.coverage != canonical_coverage
    ):
        raise StoryIntegrityError(
            "persisted ChunkManifest does not match deterministic "
            "planner output"
        )

    for chunk_ref, chunk in zip(
        manifest.chunk_refs,
        persisted_chunks,
        strict=True,
    ):
        expected_chunk_id = source_chunk_artifact_id(
            manifest.project_id,
            manifest.document_id,
            manifest.profile.profile_id,
            chunk.chunk_id,
        )
        if (
            chunk_ref.artifact_id != expected_chunk_id
            or chunk_ref.revision != ref.revision
        ):
            raise StoryIntegrityError(
                "ChunkManifest child ref identity/revision does not match "
                "deterministic plan"
            )

    return manifest
