"""v1.2 A4E-A — production orchestration core for entity reconciliation.

This module implements the provider-neutral A4 production orchestration
(A4E-A). It composes the already-merged A4B / A4C / A4D authorities into the
frozen production flow:

    resolve + verify current A1 SourceDocument
    resolve + verify current A2 ChunkManifest
    resolve + verify every current A3 CandidateExtraction (manifest order)
    build ReconciliationInputSnapshot + A3InputIdentity
    plan_reconciliation()                    # A4B
    prepare_semantic_resolution()            # A4C, ZERO provider
    build_a4_semantic_identity()             # A4D
    try_reuse_current()                      # A4D
      ├─ exact hit → upstream stability → return reused result
      └─ miss → resolve_semantic_ambiguity() # A4C
                finalize_reconciliation()    # A4D
                upstream stability check
                publish_validated()          # A4D
                return fresh result

A4E is *composition only*: it does NOT duplicate A4B planning, A4C block
packing / retries, A4D graph finalization / persistence / CURRENT verification.

Deliberately out of scope (A4E-B / A4E-C): CLI, runtime-config loading,
OpenAICompatibleLLMClient construction, live Qwen, real-novel smoke,
controlled live smoke, semantic invalidation smoke.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import LLMClient, PromptRegistry, SemanticLLMProfile

from .chunking import ChunkManifest, ChunkPlanningProfile, SourceChunk
from .errors import StoryIntegrityError, StoryPersistenceError
from .extraction import StoryExtractionProfile
from .extraction_persistence import (
    CandidateExtractionService,
    ValidatedCandidateExtractionCurrent,
    candidate_extraction_pointer_id,
)
from .persistence import (
    load_source_chunk,
    load_source_document,
    source_pointer_id,
    chunk_pointer_id,
)
from .reconciliation import (
    A3InputIdentity,
    A4SemanticIdentity,
    EntityReconciliationProfile,
)
from .reconciliation_finalization import (
    build_a4_semantic_identity,
    finalize_reconciliation,
)
from .reconciliation_persistence import (
    ReconciliationPersistenceService,
    ReconciliationPublication,
)
from .reconciliation_planning import (
    ReconciliationInputSnapshot,
    plan_reconciliation,
)
from .reconciliation_semantic import (
    DEFAULT_OUTPUT_SCHEMA_PATH,
    prepare_semantic_resolution,
    resolve_semantic_ambiguity,
)
from .service import DOCUMENT_ID
from .source import SourceDocument

# ---------------------------------------------------------------------------
# A1/A2 public resolver seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CurrentStorySnapshot:
    """The fully validated current A1 + A2 story snapshot.

    Carries the authoritative A1 ``SourceDocument``, A2 ``ChunkManifest``, and
    every A2 ``SourceChunk`` in exact manifest order, along with their artifact
    refs and the captured CURRENT pointer head refs for upstream stability.

    This is an in-memory, non-persisted, reporting-internal value. It is NOT
    an artifact, NOT a CURRENT pointer, and NOT a second canonical authority.
    """

    source_document: SourceDocument
    source_document_ref: ArtifactRef
    source_current_pointer_ref: ArtifactRef

    chunk_manifest: ChunkManifest
    chunk_manifest_ref: ArtifactRef
    chunk_current_pointer_ref: ArtifactRef

    source_chunks: tuple[SourceChunk, ...]
    source_chunk_refs: tuple[ArtifactRef, ...]


def resolve_current_story_snapshot(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    chunk_profile: ChunkPlanningProfile,
) -> CurrentStorySnapshot:
    """Resolve + fully verify the current A1 SourceDocument and A2 ChunkManifest.

    Reuses the existing A1/A2 validation authorities (exact project/document
    identity, exact ValidationReport, deterministic manifest order, coverage,
    SourceChunk refs, pointer heads) without duplicating them.

    Fails closed (``StoryIntegrityError`` / ``StoryPersistenceError``) before
    any provider call if the A1 or A2 state is invalid, stale, or missing.
    """
    from .service import (
        _current_pointer,
        _require_current_source_validation,
        _require_source_identity,
        _validate_current_manifest_snapshot,
        _validate_persisted_manifest,
    )

    # --- A1: resolve + verify current SourceDocument ---
    source_pointer = source_pointer_id(project_id, document_id)
    source_pointer_ref, source_ref = _current_pointer(pointers, source_pointer)
    if source_ref is None or source_pointer_ref is None:
        raise StoryIntegrityError(
            "A1 SourceDocument is not current; "
            "run short-drama ingest-source first"
        )
    source_document = load_source_document(store, source_ref)
    _require_source_identity(
        source_document, project_id=project_id, document_id=document_id
    )
    _require_current_source_validation(store, source_document, source_ref)

    # --- A2: resolve + verify current ChunkManifest ---
    profile_id = chunk_profile.profile_id
    manifest_pointer = chunk_pointer_id(project_id, document_id, profile_id)
    manifest_pointer_ref, manifest_ref = _current_pointer(pointers, manifest_pointer)
    if manifest_ref is None or manifest_pointer_ref is None:
        raise StoryIntegrityError(
            f"A2 ChunkManifest is not current for chunk profile {profile_id!r}; "
            "run short-drama plan-chunks first"
        )

    # A2 authority: identity + deterministic plan + exact A2 ValidationReport.
    manifest, _pinned_source, _report_ref = _validate_current_manifest_snapshot(
        store,
        manifest_ref,
        project_id=project_id,
        document_id=document_id,
        profile_id=profile_id,
    )

    # The current manifest must pin the CURRENT A1 source and the exact
    # requested profile (a stale / superseded manifest fails closed).
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

    # Load every SourceChunk in exact manifest order (reuses A2 authority).
    _validated_manifest, source_chunks = _validate_persisted_manifest(
        store,
        source=source_document,
        source_ref=source_ref,
        manifest_ref=manifest_ref,
        expected_profile=chunk_profile,
    )
    source_chunk_refs = tuple(manifest.chunk_refs)

    return CurrentStorySnapshot(
        source_document=source_document,
        source_document_ref=source_ref,
        source_current_pointer_ref=source_pointer_ref,
        chunk_manifest=manifest,
        chunk_manifest_ref=manifest_ref,
        chunk_current_pointer_ref=manifest_pointer_ref,
        source_chunks=source_chunks,
        source_chunk_refs=source_chunk_refs,
    )


# ---------------------------------------------------------------------------
# Ordered A3 reconciliation inputs (in-memory only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CurrentA3ReconciliationInputs:
    """The complete ordered A3 CURRENT set for A4 reconciliation.

    All candidate arrays are in exact ``ChunkManifest.chunk_refs`` order.
    This is an in-memory, non-persisted, reporting-internal value. It is NOT
    an artifact, NOT a CURRENT pointer, and NOT a second canonical authority.
    """

    story_snapshot: CurrentStorySnapshot
    candidate_extractions: tuple  # tuple[CandidateExtraction, ...]
    candidate_extraction_refs: tuple[ArtifactRef, ...]
    candidate_validation_report_refs: tuple[ArtifactRef, ...]
    candidate_current_pointer_refs: tuple[ArtifactRef, ...]
    extraction_profile: StoryExtractionProfile


def resolve_current_a3_reconciliation_inputs(
    store: FileArtifactStore,
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    chunk_profile: ChunkPlanningProfile,
    extraction_profile: StoryExtractionProfile,
) -> CurrentA3ReconciliationInputs:
    """Resolve + fully verify the complete A3 CURRENT set in manifest order.

    For every chunk in the current ChunkManifest, resolves the exact A3
    CandidateExtraction CURRENT via the public A3C authority
    :meth:`CandidateExtractionService.require_current_validated`.

    Fails closed (zero provider calls) if any manifest chunk lacks a valid
    A3 CURRENT or if the A3 state is invalid / stale / wrong-profile.
    """
    story_snapshot = resolve_current_story_snapshot(
        store,
        pointers,
        project_id=project_id,
        document_id=document_id,
        chunk_profile=chunk_profile,
    )

    service = CandidateExtractionService(store, pointers)
    extractions = []
    extraction_refs = []
    report_refs = []
    current_pointer_refs = []

    for chunk, chunk_ref in zip(
        story_snapshot.source_chunks, story_snapshot.source_chunk_refs
    ):
        validated = service.require_current_validated(
            project_id=project_id,
            document_id=document_id,
            chunk_profile_id=chunk_profile.profile_id,
            chunk_id=chunk.chunk_id,
            source_document_ref=story_snapshot.source_document_ref,
            source_chunk_ref=chunk_ref,
            extraction_profile=extraction_profile,
        )
        extractions.append(validated.candidate_extraction)
        extraction_refs.append(validated.candidate_extraction_ref)
        report_refs.append(validated.validation_report_ref)
        current_pointer_refs.append(validated.current_pointer_ref)

    # Complete-set proof: len must equal manifest chunk_count.
    if len(extractions) != story_snapshot.chunk_manifest.chunk_count:
        raise StoryIntegrityError(
            "A3 reconciliation input count "
            f"{len(extractions)} does not equal "
            f"ChunkManifest.chunk_count "
            f"{story_snapshot.chunk_manifest.chunk_count}"
        )

    return CurrentA3ReconciliationInputs(
        story_snapshot=story_snapshot,
        candidate_extractions=tuple(extractions),
        candidate_extraction_refs=tuple(extraction_refs),
        candidate_validation_report_refs=tuple(report_refs),
        candidate_current_pointer_refs=tuple(current_pointer_refs),
        extraction_profile=extraction_profile,
    )


# ---------------------------------------------------------------------------
# Upstream CURRENT stability guard
# ---------------------------------------------------------------------------


def _assert_upstream_heads_unchanged(
    pointers: FilePointerStore,
    *,
    project_id: str,
    document_id: str,
    chunk_profile_id: str,
    extraction_profile_id: str,
    story_snapshot: CurrentStorySnapshot,
    a3_current_pointer_refs: tuple[ArtifactRef, ...],
) -> None:
    """FAIL CLOSED if any upstream A1/A2/A3 CURRENT pointer head moved.

    Re-resolves the A1 source pointer, A2 manifest pointer, and every A3
    candidate-extraction pointer, and requires each to exactly equal the
    captured pointer ref. Any movement raises ``StoryPersistenceError``.
    """
    from .service import _current_pointer

    # A1 stability
    a1_pointer_id = source_pointer_id(project_id, document_id)
    a1_pointer_ref_now, _a1_target_now = _current_pointer(pointers, a1_pointer_id)
    if a1_pointer_ref_now != story_snapshot.source_current_pointer_ref:
        raise StoryPersistenceError(
            "A1 SourceDocument CURRENT pointer moved during A4 reconciliation; "
            "upstream is no longer stable"
        )

    # A2 stability
    a2_pointer_id = chunk_pointer_id(project_id, document_id, chunk_profile_id)
    a2_pointer_ref_now, _a2_target_now = _current_pointer(pointers, a2_pointer_id)
    if a2_pointer_ref_now != story_snapshot.chunk_current_pointer_ref:
        raise StoryPersistenceError(
            "A2 ChunkManifest CURRENT pointer moved during A4 reconciliation; "
            "upstream is no longer stable"
        )

    # A3 stability (every chunk, in manifest order)
    for i, expected_pointer_ref in enumerate(a3_current_pointer_refs):
        chunk = story_snapshot.source_chunks[i]
        a3_pointer_id = candidate_extraction_pointer_id(
            project_id,
            document_id,
            chunk_profile_id,
            chunk.chunk_id,
            extraction_profile_id,
        )
        a3_pointer_ref_now, _a3_target_now = _current_pointer(pointers, a3_pointer_id)
        if a3_pointer_ref_now != expected_pointer_ref:
            raise StoryPersistenceError(
                f"A3 CandidateExtraction CURRENT pointer for chunk "
                f"{chunk.chunk_id!r} moved during A4 reconciliation; "
                "upstream is no longer stable"
            )


# ---------------------------------------------------------------------------
# Non-persisted stage result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntityReconciliationStageResult:
    """Non-persisted A4E stage result (reporting only).

    This value is execution/reporting data ONLY. It is deliberately NOT:
    an artifact type, a CURRENT pointer, a canonical authority, or a database
    record. All canonical data still comes from persisted A4 artifacts.
    """

    # Upstream refs
    source_document_ref: ArtifactRef
    chunk_manifest_ref: ArtifactRef
    candidate_extraction_refs: tuple[ArtifactRef, ...]

    # Candidate counts
    candidate_count_total: int
    character_candidate_count: int
    location_candidate_count: int
    a3_unresolved_candidate_count: int

    # Pair counts
    pair_count_total: int
    auto_same_pair_count: int
    must_not_merge_pair_count: int
    semantic_pair_count: int

    # Decision counts
    deterministic_decision_count: int
    llm_same_count: int
    llm_different_count: int
    llm_uncertain_count: int

    # Canonical / unresolved counts
    canonical_character_count: int
    canonical_location_count: int
    unresolved_entity_count: int

    # A4 output refs
    entity_map_ref: ArtifactRef
    validation_report_ref: ArtifactRef
    current_pointer_ref: ArtifactRef

    # Semantic generation
    semantic_block_count: int
    semantic_generation_call_count: int
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible reporting view. Reporting data only."""
        return {
            "source_document_ref": self.source_document_ref.to_dict(),
            "chunk_manifest_ref": self.chunk_manifest_ref.to_dict(),
            "candidate_extraction_refs": [
                ref.to_dict() for ref in self.candidate_extraction_refs
            ],
            "candidate_count_total": self.candidate_count_total,
            "character_candidate_count": self.character_candidate_count,
            "location_candidate_count": self.location_candidate_count,
            "a3_unresolved_candidate_count": self.a3_unresolved_candidate_count,
            "pair_count_total": self.pair_count_total,
            "auto_same_pair_count": self.auto_same_pair_count,
            "must_not_merge_pair_count": self.must_not_merge_pair_count,
            "semantic_pair_count": self.semantic_pair_count,
            "deterministic_decision_count": self.deterministic_decision_count,
            "llm_same_count": self.llm_same_count,
            "llm_different_count": self.llm_different_count,
            "llm_uncertain_count": self.llm_uncertain_count,
            "canonical_character_count": self.canonical_character_count,
            "canonical_location_count": self.canonical_location_count,
            "unresolved_entity_count": self.unresolved_entity_count,
            "entity_map_ref": self.entity_map_ref.to_dict(),
            "validation_report_ref": self.validation_report_ref.to_dict(),
            "current_pointer_ref": self.current_pointer_ref.to_dict(),
            "semantic_block_count": self.semantic_block_count,
            "semantic_generation_call_count": self.semantic_generation_call_count,
            "reused": self.reused,
        }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _count_candidates_from_snapshot(
    snapshot: ReconciliationInputSnapshot,
) -> tuple[int, int, int, int]:
    """Count (total, character, location, unresolved) candidates."""
    char_count = 0
    loc_count = 0
    unres_count = 0
    for ext in snapshot.candidate_extractions:
        char_count += len(ext.candidates.characters)
        loc_count += len(ext.candidates.locations)
        unres_count += len(ext.candidates.unresolved_mentions)
    total = char_count + loc_count + unres_count
    return total, char_count, loc_count, unres_count


def _count_pairs_from_planning(
    planning_result,
) -> tuple[int, int, int, int]:
    """Count (total, auto_same, must_not_merge, needs_semantic_decision) pairs."""
    total = len(planning_result.pair_plans)
    auto_same = sum(
        1 for p in planning_result.pair_plans if p.state == "auto_same"
    )
    must_not_merge = sum(
        1 for p in planning_result.pair_plans if p.state == "must_not_merge"
    )
    semantic = sum(
        1 for p in planning_result.pair_plans if p.state == "needs_semantic_decision"
    )
    return total, auto_same, must_not_merge, semantic


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class EntityReconciliationService:
    """A4E-A provider-neutral production orchestration service for A4 entity
    reconciliation.

    Holds the same exact source/persistence authorities as the A4B/A4C/A4D
    services (single ``FileArtifactStore`` + ``FilePointerStore``) and composes
    them into the frozen A4 production flow. It knows nothing about base_url,
    HTTP, credentials, or a concrete model: it depends only on the
    provider-neutral ``LLMClient``.
    """

    def __init__(
        self,
        store: FileArtifactStore,
        pointers: FilePointerStore,
        prompt_registry: PromptRegistry | None = None,
        output_schema_path: str | Path | None = None,
    ) -> None:
        self.store = store
        self.pointers = pointers
        self._prompt_registry = prompt_registry
        self._output_schema_path = output_schema_path

    def reconcile_entities(
        self,
        *,
        project_id: str,
        document_id: str,
        chunk_profile: ChunkPlanningProfile,
        extraction_profile: StoryExtractionProfile,
        reconciliation_profile: EntityReconciliationProfile,
        semantic_profile: SemanticLLMProfile,
        llm_client: LLMClient,
    ) -> EntityReconciliationStageResult:
        """Execute the full A4 production orchestration flow.

        Resolves + verifies the complete A1/A2/A3 CURRENT set, builds the
        A4B/A4C/A4D identity material, attempts pre-provider reuse, and (on
        miss) drives the A4C semantic generation + A4D finalization +
        publication. Returns a non-persisted stage result.

        Fails closed (zero provider calls) on any upstream A1/A2/A3 invalidity.
        Propagates ``LLMError`` unchanged (no outer provider retry).
        """
        # 1. Resolve the complete A3 CURRENT set (A1/A2/A3 validation).
        current_a3 = resolve_current_a3_reconciliation_inputs(
            self.store,
            self.pointers,
            project_id=project_id,
            document_id=document_id,
            chunk_profile=chunk_profile,
            extraction_profile=extraction_profile,
        )
        story_snapshot = current_a3.story_snapshot

        # 2. Build the A4B ReconciliationInputSnapshot.
        snapshot = ReconciliationInputSnapshot(
            source_document=story_snapshot.source_document,
            source_document_ref=story_snapshot.source_document_ref,
            chunk_manifest=story_snapshot.chunk_manifest,
            source_chunks=story_snapshot.source_chunks,
            source_chunk_refs=story_snapshot.source_chunk_refs,
            candidate_extractions=current_a3.candidate_extractions,
            candidate_extraction_refs=current_a3.candidate_extraction_refs,
            a3_validation_report_refs=current_a3.candidate_validation_report_refs,
        )

        # 3. A4B deterministic planning.
        planning_result = plan_reconciliation(snapshot)

        # 4. A4C deterministic preparation (ZERO provider calls).
        output_schema_path = (
            self._output_schema_path
            if self._output_schema_path is not None
            else DEFAULT_OUTPUT_SCHEMA_PATH
        )
        preparation = prepare_semantic_resolution(
            planning_result,
            reconciliation_profile,
            semantic_profile,
            prompt_registry=self._prompt_registry,
            output_schema_path=output_schema_path,
        )

        # 5. A4 semantic identity (backend-neutral, pre-provider).
        semantic_identity = build_a4_semantic_identity(
            reconciliation_profile,
            preparation,
            planning_result,
        )

        # 6. A3InputIdentity (EntityMap lineage authority).
        a3_input = A3InputIdentity(
            source_document_ref=story_snapshot.source_document_ref,
            chunk_manifest_ref=story_snapshot.chunk_manifest_ref,
            candidate_extraction_refs=current_a3.candidate_extraction_refs,
            extraction_profile_id=extraction_profile.profile_id,
            extraction_profile_hash=extraction_profile.profile_hash,
        )

        # 7. Pre-provider A4 reuse.
        persistence = ReconciliationPersistenceService(self.store, self.pointers)
        reuse_result = persistence.try_reuse_current(
            project_id=project_id,
            document_id=document_id,
            reconciliation_profile_id=reconciliation_profile.profile_id,
            a3_input=a3_input,
            semantic_identity=semantic_identity,
        )

        if reuse_result is not None:
            # 8a. Reuse hit: verify upstream stability before returning.
            _assert_upstream_heads_unchanged(
                self.pointers,
                project_id=project_id,
                document_id=document_id,
                chunk_profile_id=chunk_profile.profile_id,
                extraction_profile_id=extraction_profile.profile_id,
                story_snapshot=story_snapshot,
                a3_current_pointer_refs=current_a3.candidate_current_pointer_refs,
            )
            # Build the stage result from the reused publication.
            return self._build_reuse_stage_result(
                snapshot=snapshot,
                chunk_manifest_ref=story_snapshot.chunk_manifest_ref,
                planning_result=planning_result,
                publication=reuse_result,
                semantic_block_count=len(preparation.blocks),
            )

        # 8b. Reuse miss: drive A4C semantic generation.
        semantic_result = resolve_semantic_ambiguity(
            planning_result,
            reconciliation_profile,
            semantic_profile,
            llm_client,
            prompt_registry=self._prompt_registry,
            output_schema_path=output_schema_path,
        )

        # 9. A4D finalization.
        finalization = finalize_reconciliation(semantic_result)

        # 10. Upstream stability check before publication.
        _assert_upstream_heads_unchanged(
            self.pointers,
            project_id=project_id,
            document_id=document_id,
            chunk_profile_id=chunk_profile.profile_id,
            extraction_profile_id=extraction_profile.profile_id,
            story_snapshot=story_snapshot,
            a3_current_pointer_refs=current_a3.candidate_current_pointer_refs,
        )

        # 11. A4D publication.
        publication = persistence.publish_validated(
            project_id=project_id,
            document_id=document_id,
            reconciliation_profile=reconciliation_profile,
            semantic_identity=semantic_identity,
            a3_input=a3_input,
            finalization_result=finalization,
        )

        # 12. Build the fresh stage result.
        total, char_count, loc_count, unres_count = (
            _count_candidates_from_snapshot(snapshot)
        )
        pair_total, auto_same, must_not_merge, semantic_pairs = (
            _count_pairs_from_planning(planning_result)
        )
        det_count = len(planning_result.decisions)
        llm_same = sum(
            1
            for d in semantic_result.all_decisions
            if d.method == "llm" and d.decision == "same_entity"
        )
        llm_diff = sum(
            1
            for d in semantic_result.all_decisions
            if d.method == "llm" and d.decision == "different_entity"
        )
        llm_unc = sum(
            1
            for d in semantic_result.all_decisions
            if d.method == "llm" and d.decision == "uncertain"
        )
        canonical_char_count = len(finalization.canonical_character_registry.entities)
        canonical_loc_count = len(finalization.canonical_location_registry.entities)
        unresolved_count = len(finalization.unresolved_entity_set.entities)
        semantic_generation_call_count = sum(
            br.semantic_rounds for br in semantic_result.block_results
        )

        return EntityReconciliationStageResult(
            source_document_ref=story_snapshot.source_document_ref,
            chunk_manifest_ref=story_snapshot.chunk_manifest_ref,
            candidate_extraction_refs=current_a3.candidate_extraction_refs,
            candidate_count_total=total,
            character_candidate_count=char_count,
            location_candidate_count=loc_count,
            a3_unresolved_candidate_count=unres_count,
            pair_count_total=pair_total,
            auto_same_pair_count=auto_same,
            must_not_merge_pair_count=must_not_merge,
            semantic_pair_count=semantic_pairs,
            deterministic_decision_count=det_count,
            llm_same_count=llm_same,
            llm_different_count=llm_diff,
            llm_uncertain_count=llm_unc,
            canonical_character_count=canonical_char_count,
            canonical_location_count=canonical_loc_count,
            unresolved_entity_count=unresolved_count,
            entity_map_ref=publication.entity_map_ref,
            validation_report_ref=publication.validation_report_ref,
            current_pointer_ref=publication.current_pointer_ref,
            semantic_block_count=len(preparation.blocks),
            semantic_generation_call_count=semantic_generation_call_count,
            reused=False,
        )

    # -- reuse stage result -------------------------------------------------

    def _build_reuse_stage_result(
        self,
        *,
        snapshot: ReconciliationInputSnapshot,
        chunk_manifest_ref: ArtifactRef,
        planning_result,
        publication: ReconciliationPublication,
        semantic_block_count: int,
    ) -> EntityReconciliationStageResult:
        """Build the non-persisted stage result for a reuse hit.

        Counts are derived from the in-memory planning result (identical for
        the same inputs) and the verified persisted A4 artifacts (loaded via
        existing typed A4D loaders for the decision/registry counts).
        """
        from .reconciliation_persistence import (
            a4_base_artifact_id,
            canonical_character_registry_artifact_id,
            canonical_location_registry_artifact_id,
            load_canonical_character_registry,
            load_canonical_location_registry,
            load_reconciliation_decision_set,
            load_unresolved_entity_set,
            reconciliation_decision_set_artifact_id,
            unresolved_entity_set_artifact_id,
        )

        total, char_count, loc_count, unres_count = (
            _count_candidates_from_snapshot(snapshot)
        )
        pair_total, auto_same, must_not_merge, semantic_pairs = (
            _count_pairs_from_planning(planning_result)
        )
        det_count = len(planning_result.decisions)

        # Load the persisted A4 artifacts for accurate decision/registry counts.
        project_id = snapshot.source_document.project_id
        document_id = snapshot.source_document.document_id
        # Derive the A4 base from the entity map ref's artifact_id.
        base = publication.entity_map_ref.artifact_id.rsplit(
            ".entity-map", 1
        )[0]
        decision_set = load_reconciliation_decision_set(
            self.store,
            publication.reconciliation_decision_set_ref,
            expected_artifact_id=reconciliation_decision_set_artifact_id(base),
        )
        llm_same = sum(
            1
            for d in decision_set.decisions
            if d.method == "llm" and d.decision == "same_entity"
        )
        llm_diff = sum(
            1
            for d in decision_set.decisions
            if d.method == "llm" and d.decision == "different_entity"
        )
        llm_unc = sum(
            1
            for d in decision_set.decisions
            if d.method == "llm" and d.decision == "uncertain"
        )
        char_registry = load_canonical_character_registry(
            self.store,
            publication.canonical_character_registry_ref,
            expected_artifact_id=canonical_character_registry_artifact_id(base),
        )
        loc_registry = load_canonical_location_registry(
            self.store,
            publication.canonical_location_registry_ref,
            expected_artifact_id=canonical_location_registry_artifact_id(base),
        )
        unres_set = load_unresolved_entity_set(
            self.store,
            publication.unresolved_entity_set_ref,
            expected_artifact_id=unresolved_entity_set_artifact_id(base),
        )

        return EntityReconciliationStageResult(
            source_document_ref=snapshot.source_document_ref,
            chunk_manifest_ref=chunk_manifest_ref,
            candidate_extraction_refs=snapshot.candidate_extraction_refs,
            candidate_count_total=total,
            character_candidate_count=char_count,
            location_candidate_count=loc_count,
            a3_unresolved_candidate_count=unres_count,
            pair_count_total=pair_total,
            auto_same_pair_count=auto_same,
            must_not_merge_pair_count=must_not_merge,
            semantic_pair_count=semantic_pairs,
            deterministic_decision_count=det_count,
            llm_same_count=llm_same,
            llm_different_count=llm_diff,
            llm_uncertain_count=llm_unc,
            canonical_character_count=len(char_registry.entities),
            canonical_location_count=len(loc_registry.entities),
            unresolved_entity_count=len(unres_set.entities),
            entity_map_ref=publication.entity_map_ref,
            validation_report_ref=publication.validation_report_ref,
            current_pointer_ref=publication.current_pointer_ref,
            semantic_block_count=semantic_block_count,
            semantic_generation_call_count=0,
            reused=True,
        )


# ---------------------------------------------------------------------------
# A4E-A project-level composition (provider-neutral)
# ---------------------------------------------------------------------------


def reconcile_entities_project(
    project_path: str | Path,
    *,
    runs_root: str | Path,
    chunk_profile_path: str | Path,
    extraction_profile_path: str | Path,
    reconciliation_profile_path: str | Path,
    semantic_profile_path: str | Path,
    llm_client: LLMClient,
) -> EntityReconciliationStageResult:
    """A4E-A provider-neutral project-level composition for reconcile-entities.

    This small application-level wrapper loads the existing tracked profile
    types using their existing loaders, initializes the story artifact/pointer
    stores, and calls the narrow orchestration service.

    Provider/runtime composition is deliberately NOT done here: the caller
    passes an already-created, provider-neutral ``LLMClient``. This helper
    never touches ``base_url``, HTTP, credentials, or a concrete provider
    client, so it remains safe to reuse offline and in tests.
    """
    from .extraction import load_story_extraction_profile
    from .reconciliation import load_entity_reconciliation_profile
    from .service import _load_project, _load_profile, _stores
    from short_drama.llm import load_semantic_profile

    _project_file, project = _load_project(project_path)
    project_id = project["project_id"]
    chunk_profile = _load_profile(chunk_profile_path)
    extraction_profile = load_story_extraction_profile(extraction_profile_path)
    reconciliation_profile = load_entity_reconciliation_profile(
        reconciliation_profile_path
    )
    semantic_profile = load_semantic_profile(semantic_profile_path)
    store, pointers = _stores(runs_root, project_id)
    service = EntityReconciliationService(store, pointers)
    return service.reconcile_entities(
        project_id=project_id,
        document_id=DOCUMENT_ID,
        chunk_profile=chunk_profile,
        extraction_profile=extraction_profile,
        reconciliation_profile=reconciliation_profile,
        semantic_profile=semantic_profile,
        llm_client=llm_client,
    )


__all__ = [
    "CurrentA3ReconciliationInputs",
    "CurrentStorySnapshot",
    "EntityReconciliationService",
    "EntityReconciliationStageResult",
    "reconcile_entities_project",
    "resolve_current_a3_reconciliation_inputs",
    "resolve_current_story_snapshot",
]
