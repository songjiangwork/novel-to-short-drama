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
from short_drama.llm import LLMClient, SemanticLLMProfile, load_semantic_profile

from .chunking import ChunkPlanningProfile
from .extraction import StoryExtractionProfile, load_story_extraction_profile
from .extraction_service import (
    DEFAULT_OUTPUT_SCHEMA_PATH,
    ChunkExtractionService,
)
from .service import (
    DOCUMENT_ID,
    _load_profile,
    _load_project,
    _stores,
    resolve_current_story_snapshot,
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

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible reporting view of this (non-persisted) summary.

        Artifact refs are emitted in their existing canonical ``to_dict()``
        representation, in deterministic manifest order. Reporting data only.
        """
        return {
            "chunks_total": self.chunks_total,
            "chunks_reused": self.chunks_reused,
            "chunks_generated": self.chunks_generated,
            "chunks_failed": self.chunks_failed,
            "candidate_extraction_refs": [
                ref.to_dict() for ref in self.candidate_extraction_refs
            ],
            "validation_report_refs": [
                ref.to_dict() for ref in self.validation_report_refs
            ],
        }





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
        # A1/A2 resolution: single shared public authority (also used by A4E).
        snapshot = resolve_current_story_snapshot(
            self.store,
            self.pointers,
            project_id=project_id,
            document_id=document_id,
            chunk_profile=chunk_profile,
        )
        source_ref = snapshot.source_document_ref
        chunk_refs = snapshot.source_chunk_refs

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


# ---------------------------------------------------------------------------
# A3E-B project-level composition (provider-neutral)
# ---------------------------------------------------------------------------


def extract_chunks_project(
    project_path: str | Path,
    *,
    runs_root: str | Path = "runs",
    chunk_profile_path: str | Path,
    extraction_profile_path: str | Path,
    semantic_profile_path: str | Path,
    llm_client: LLMClient,
) -> ChunkExtractionBatchSummary:
    """A3E-B provider-neutral project-level composition for ``extract-chunks``.

    This small application-level wrapper is what the CLI calls. It reuses the
    existing authorities and does NOT duplicate any A3E-A orchestration / reuse /
    persistence logic:

      * loads + validates the project (reusing the A1 project loader) and reads
        its ``project_id``;
      * loads the requested ``ChunkPlanningProfile``, ``StoryExtractionProfile``,
        and ``SemanticLLMProfile`` (reusing the existing loaders);
      * initializes the project's story artifact/pointer stores (reusing the
        existing store-creation helper);
      * uses the existing fixed A1 ``SourceDocument`` identity (``DOCUMENT_ID``);
      * instantiates the merged A3E-A :class:`ChunkExtractionBatchService` and
        invokes :meth:`ChunkExtractionBatchService.extract_chunks`.

    Provider/runtime composition is deliberately NOT done here: the caller passes
    an already-created, provider-neutral ``LLMClient``. This helper never touches
    ``base_url``, HTTP, credentials, or a concrete provider client, so it remains
    safe to reuse offline and in tests.
    """
    _project_file, project = _load_project(project_path)
    project_id = project["project_id"]
    chunk_profile = _load_profile(chunk_profile_path)
    extraction_profile = load_story_extraction_profile(extraction_profile_path)
    semantic_profile = load_semantic_profile(semantic_profile_path)
    store, pointers = _stores(runs_root, project_id)
    service = ChunkExtractionBatchService(store, pointers)
    return service.extract_chunks(
        project_id=project_id,
        document_id=DOCUMENT_ID,
        chunk_profile=chunk_profile,
        extraction_profile=extraction_profile,
        semantic_profile=semantic_profile,
        llm_client=llm_client,
    )


__all__ = [
    "ChunkExtractionBatchService",
    "ChunkExtractionBatchSummary",
    "extract_chunks_project",
]
