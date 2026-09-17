"""v1.2 A3E-A — deterministic PROJECT/BATCH chunk-extraction orchestration core.

This is the thin, provider-neutral *orchestration* layer above the already
merged A3D single-chunk authority. For one exact project/document/chunk-profile
it answers the batch question without adding a second reuse / validation /
persistence layer:

  * resolve the authoritative **current** A1 ``SourceDocument`` and verify its
    A1 ``ValidationReport`` (reusing the A1 project/persistence primitives);
  * resolve the authoritative **current** A2 ``ChunkManifest`` for the requested
    ``ChunkPlanningProfile`` and verify its A2 ``ValidationReport`` + deterministic
    plan (reusing the A2 project/persistence primitives);
  * iterate the manifest chunks in the exact deterministic **manifest order**
    (sequential — no concurrency), resolving each exact ``SourceChunk`` through
    the manifest child ref;
  * invoke the existing A3D :meth:`ChunkExtractionService.extract_chunk`
    *exactly once* per chunk;
  * let A3D/A3C decide, per chunk, whether the chunk is an exact validated reuse
    (``publication.reused is True``) or a fresh generation (``reused is False``);
  * return a **non-persisted** batch summary (execution/reporting data only).

Authoritative architecture boundary (frozen A-I4 / A3E contract):

  * A3D ``ChunkExtractionService.extract_chunk(...)`` remains the *sole*
    single-chunk semantic + persistence authority: prompt rendering, LEFT /
    OWNERSHIP / RIGHT partitioning, ``CandidatePayload`` parsing, A3 semantic
    validation, bounded semantic regeneration, ``CandidateExtraction`` /
    ``ValidationReport`` persistence, CURRENT reuse verification, and
    request/provenance identity are all reused, never duplicated here.
  * A3D/A3C remain the sole authority for CURRENT eligibility, exact semantic
    identity, PASS ``ValidationReport`` matching, stale/missing detection, and
    publication race handling. The batch only *counts* the returned
    ``publication.reused`` value; it does NOT implement a second reuse layer,
    does NOT scan historical revisions, and does NOT resurrect anything.

Deliberately minimal (fail closed, no workflow engine):

  * If the exact A1/A2 project state or any chunk lineage is invalid, the
    resolution helpers (and the per-chunk A3D source/lineage gate) fail closed
    *before* any provider call — an invalid chunk never consumes an LLM attempt.
  * If extraction of a chunk fails (A-I3 ``LLMError`` or
    :class:`ExtractionSemanticGenerationError`), the error propagates unchanged
    and the batch does NOT manufacture a successful summary. Earlier chunks that
    already resolved to a validated CURRENT remain persisted; a future rerun
    resumes through the existing per-chunk A3 CURRENT reuse (the v1 resume
    mechanism). No retry manager, queue, checkpoint database, or journal.

The :class:`ChunkExtractionBatchSummary` is in-memory execution/reporting data
only: it is NOT an artifact type, a CURRENT pointer, a canonical authority, or a
database record.

Deliberately out of scope (later slices): the ``extract-chunks`` CLI (A3E-B),
real-Qwen execution and real-novel / semantic-profile invalidation smoke
(A3E-C), concurrency / ``np=2`` slot usage, historical replay, A4/A5/A6,
H3/ComfyUI/B-stage, provider infrastructure, DB/ORM, raw-LLM persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import LLMClient, SemanticLLMProfile

from .chunking import ChunkManifest, ChunkPlanningProfile
from .errors import StoryIntegrityError
from .extraction import StoryExtractionProfile
from .extraction_service import (
    DEFAULT_OUTPUT_SCHEMA_PATH,
    ChunkExtractionService,
)
from .persistence import chunk_pointer_id, load_source_document, source_pointer_id
from .service import (
    _current_pointer,
    _require_current_source_validation,
    _require_source_identity,
    _validate_current_manifest_snapshot,
)


# ---------------------------------------------------------------------------
# Non-persisted batch summary (execution / reporting only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkExtractionBatchSummary:
    """Outcome of one sequential A3E-A batch run, in deterministic manifest order.

    This value is execution/reporting data ONLY. It is deliberately NOT:
    an artifact type, a CURRENT pointer, a canonical authority, or a database
    record. Resume after a failure is achieved by a future rerun, which reuses
    the per-chunk A3 CURRENT that A3D/A3C already persisted.

    ``candidate_extraction_refs`` and ``validation_report_refs`` are ordered to
    match the manifest order. In a successful run ``chunks_failed`` is ``0``; a
    failing chunk makes the batch raise (fail closed) rather than return a
    successful summary.
    """

    chunks_total: int
    chunks_reused: int
    chunks_generated: int
    chunks_failed: int
    candidate_extraction_refs: tuple[ArtifactRef, ...]
    validation_report_refs: tuple[ArtifactRef, ...]


# ---------------------------------------------------------------------------
# Current A1 / A2 resolution (reuses the existing project/persistence primitives)
# ---------------------------------------------------------------------------


def _resolve_current_source_ref(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    project_id: str,
    document_id: str,
) -> ArtifactRef:
    """Resolve + verify the authoritative current A1 ``SourceDocument``.

    Fails closed (``StoryIntegrityError``) before any provider call if the A1
    source is not current, targets the wrong project/document, or is not backed
    by its exact current A1 ``ValidationReport``.
    """
    source_pointer = source_pointer_id(project_id, document_id)
    _source_pointer_ref, source_ref = _current_pointer(pointers, source_pointer)
    if source_ref is None:
        raise StoryIntegrityError(
            "A1 SourceDocument is not current; run short-drama ingest-source first"
        )
    source = load_source_document(store, source_ref)
    _require_source_identity(source, project_id=project_id, document_id=document_id)
    _require_current_source_validation(store, source, source_ref)
    return source_ref


def _resolve_current_chunk_refs(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    source_ref: ArtifactRef,
    project_id: str,
    document_id: str,
    chunk_profile: ChunkPlanningProfile,
) -> tuple[ChunkManifest, tuple[ArtifactRef, ...]]:
    """Resolve + verify the authoritative current A2 ``ChunkManifest`` for the
    requested ``ChunkPlanningProfile`` and return its child chunk refs in the
    exact deterministic manifest order.

    Fails closed (``StoryIntegrityError``) before any provider call if the A2
    manifest is not current, targets the wrong project/document/profile, is not
    backed by its exact current A2 ``ValidationReport``, does not pin the
    current A1 ``SourceDocument``, or does not match the requested profile.
    """
    profile_id = chunk_profile.profile_id
    manifest_pointer = chunk_pointer_id(project_id, document_id, profile_id)
    _manifest_pointer_ref, manifest_ref = _current_pointer(pointers, manifest_pointer)
    if manifest_ref is None:
        raise StoryIntegrityError(
            "A2 ChunkManifest is not current for chunk profile "
            f"{profile_id!r}; run short-drama plan-chunks first"
        )
    # A2 authority: identity + deterministic plan + exact A2 ValidationReport.
    manifest, _pinned_source, _report_ref = _validate_current_manifest_snapshot(
        store,
        manifest_ref,
        project_id=project_id,
        document_id=document_id,
        profile_id=profile_id,
    )
    # The current manifest must pin the CURRENT A1 source and the exact requested
    # profile (a stale / superseded manifest fails closed rather than being run).
    if manifest.source_document_ref != source_ref:
        raise StoryIntegrityError(
            "A2 ChunkManifest does not pin the current A1 SourceDocument; "
            "the current manifest is stale relative to the current source"
        )
    if manifest.profile != chunk_profile:
        raise StoryIntegrityError(
            "requested chunk profile does not match the current ChunkManifest "
            "profile"
        )
    return manifest, tuple(manifest.chunk_refs)


# ---------------------------------------------------------------------------
# Batch service
# ---------------------------------------------------------------------------


class ChunkExtractionBatchService:
    """A3E-A deterministic PROJECT/BATCH chunk-extraction orchestration service.

    Holds the same exact source/persistence authorities as the A3D
    :class:`ChunkExtractionService` (single ``FileArtifactStore`` +
    ``FilePointerStore``) and owns one A3D instance as its per-chunk authority.
    It knows nothing about base_url, HTTP, credentials, or a concrete model: it
    depends only on the provider-neutral ``LLMClient``.

    The batch is sequential and deterministic: chunks are processed in manifest
    order, and each chunk is handed to A3D ``extract_chunk(...)`` exactly once.
    There is no concurrency, retry, or reordering in this layer.
    """

    def __init__(
        self,
        store: FileArtifactStore,
        pointers: FilePointerStore,
        prompt_registry=None,
        output_schema_path: str | Path = DEFAULT_OUTPUT_SCHEMA_PATH,
    ) -> None:
        self.store = store
        self.pointers = pointers
        # A3D remains the single-chunk semantic + persistence authority.
        self._extraction = ChunkExtractionService(
            store, pointers, prompt_registry, output_schema_path
        )

    def extract_chunks(
        self,
        *,
        project_id: str,
        document_id: str,
        chunk_profile: ChunkPlanningProfile,
        extraction_profile: StoryExtractionProfile,
        semantic_profile: SemanticLLMProfile,
        llm_client: LLMClient,
    ) -> ChunkExtractionBatchSummary:
        """Extract every chunk in the current A2 manifest, in manifest order.

        Resolves + verifies the current A1 ``SourceDocument`` and A2
        ``ChunkManifest`` (fail closed before any provider call), then invokes
        the A3D single-chunk service exactly once per chunk. A chunk is counted
        as *reused* or *generated* from the A3D ``publication.reused`` value;
        A3D/A3C are the sole authority for that decision.

        Returns a non-persisted :class:`ChunkExtractionBatchSummary`. Fails
        closed (propagating the A-I3 / A3 error unchanged) if resolution is
        invalid or any chunk cannot produce a validated extraction; in that case
        no successful summary is returned.
        """
        source_ref = _resolve_current_source_ref(
            self.store, self.pointers, project_id, document_id
        )
        _manifest, chunk_refs = _resolve_current_chunk_refs(
            self.store,
            self.pointers,
            source_ref,
            project_id,
            document_id,
            chunk_profile,
        )

        extraction_refs: list[ArtifactRef] = []
        validation_report_refs: list[ArtifactRef] = []
        reused = 0
        generated = 0
        for chunk_ref in chunk_refs:
            publication = self._extraction.extract_chunk(
                source_document_ref=source_ref,
                source_chunk_ref=chunk_ref,
                chunk_profile_id=chunk_profile.profile_id,
                extraction_profile=extraction_profile,
                semantic_profile=semantic_profile,
                llm_client=llm_client,
            )
            if publication.reused:
                reused += 1
            else:
                generated += 1
            extraction_refs.append(publication.candidate_extraction_ref)
            validation_report_refs.append(publication.validation_report_ref)

        return ChunkExtractionBatchSummary(
            chunks_total=len(chunk_refs),
            chunks_reused=reused,
            chunks_generated=generated,
            chunks_failed=0,
            candidate_extraction_refs=tuple(extraction_refs),
            validation_report_refs=tuple(validation_report_refs),
        )


__all__ = [
    "ChunkExtractionBatchService",
    "ChunkExtractionBatchSummary",
]
