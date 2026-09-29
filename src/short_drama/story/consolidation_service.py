"""Provider-neutral A5G1 production orchestration for evidence consolidation.

This composition layer deliberately delegates A4/A3 resolution and A5B planning
to :func:`build_consolidation_planning`, reuse to
``ConsolidationPersistenceService.try_reuse_current``, and publication to
``ConsolidationPersistenceService.publish_validated``.  It contains no runtime
transport or provider-routing configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import LLMClient, PromptRegistry, SemanticLLMProfile
from short_drama.paths import REPO_ROOT

from .consolidation import ConsolidationProfile, load_consolidation_profile
from .consolidation_finalization import finalize_consolidation
from .consolidation_persistence import (
    ConsolidationPersistenceService,
    ConsolidationPublication,
    canonical_event_set_artifact_id,
    canonical_fact_set_artifact_id,
    canonical_relationship_set_artifact_id,
    consolidation_decision_set_artifact_id,
    load_canonical_event_set,
    load_canonical_fact_set,
    load_canonical_relationship_set,
    load_consolidation_decision_set,
    load_story_conflict_set,
    story_conflict_set_artifact_id,
)
from .consolidation_planning import ConsolidationPlanningResult, build_consolidation_planning
from .consolidation_semantic import (
    EVENT_SEMANTIC_PACKING_V1,
    FACT_SEMANTIC_PACKING_V2,
    RELATIONSHIP_SEMANTIC_PACKING_V1,
    build_event_semantic_preparation,
    build_fact_semantic_preparation,
    build_relationship_semantic_preparation,
    resolve_event_semantic_ambiguity,
    resolve_fact_semantic_ambiguity,
    resolve_relationship_semantic_ambiguity,
    validate_a5_max_concurrency,
)


DEFAULT_CONSOLIDATION_PROMPT_BASE_DIR = REPO_ROOT / "prompts" / "story"


@dataclass(frozen=True, slots=True)
class EvidenceConsolidationStageResult:
    """Non-persisted A5G1 reporting result; persisted A5 refs stay canonical."""

    entity_map_ref: ArtifactRef
    candidate_extraction_refs: tuple[ArtifactRef, ...]
    fact_candidate_count: int
    event_candidate_count: int
    relationship_candidate_count: int
    resolved_bound_reference_count: int
    unresolved_bound_reference_count: int
    fact_planned_pair_count: int
    event_planned_pair_count: int
    relationship_planned_pair_count: int
    fact_deterministic_decision_count: int
    event_deterministic_decision_count: int
    relationship_deterministic_decision_count: int
    fact_semantic_decision_count: int
    event_semantic_decision_count: int
    relationship_semantic_decision_count: int
    fact_semantic_block_count: int
    event_semantic_block_count: int
    relationship_semantic_block_count: int
    semantic_generation_call_count: int
    canonical_fact_count: int
    canonical_event_count: int
    canonical_relationship_count: int
    state_transition_count: int
    story_conflict_count: int
    consolidation_candidate_index_ref: ArtifactRef
    consolidation_decision_set_ref: ArtifactRef
    canonical_fact_set_ref: ArtifactRef
    canonical_event_set_ref: ArtifactRef
    canonical_relationship_set_ref: ArtifactRef
    story_conflict_set_ref: ArtifactRef
    consolidation_manifest_ref: ArtifactRef
    validation_report_ref: ArtifactRef
    current_pointer_ref: ArtifactRef
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible reporting data only."""
        return {
            "entity_map_ref": self.entity_map_ref.to_dict(),
            "candidate_extraction_refs": [
                ref.to_dict() for ref in self.candidate_extraction_refs
            ],
            "fact_candidate_count": self.fact_candidate_count,
            "event_candidate_count": self.event_candidate_count,
            "relationship_candidate_count": self.relationship_candidate_count,
            "resolved_bound_reference_count": self.resolved_bound_reference_count,
            "unresolved_bound_reference_count": self.unresolved_bound_reference_count,
            "fact_planned_pair_count": self.fact_planned_pair_count,
            "event_planned_pair_count": self.event_planned_pair_count,
            "relationship_planned_pair_count": self.relationship_planned_pair_count,
            "fact_deterministic_decision_count": self.fact_deterministic_decision_count,
            "event_deterministic_decision_count": self.event_deterministic_decision_count,
            "relationship_deterministic_decision_count": self.relationship_deterministic_decision_count,
            "fact_semantic_decision_count": self.fact_semantic_decision_count,
            "event_semantic_decision_count": self.event_semantic_decision_count,
            "relationship_semantic_decision_count": self.relationship_semantic_decision_count,
            "fact_semantic_block_count": self.fact_semantic_block_count,
            "event_semantic_block_count": self.event_semantic_block_count,
            "relationship_semantic_block_count": self.relationship_semantic_block_count,
            "semantic_generation_call_count": self.semantic_generation_call_count,
            "canonical_fact_count": self.canonical_fact_count,
            "canonical_event_count": self.canonical_event_count,
            "canonical_relationship_count": self.canonical_relationship_count,
            "state_transition_count": self.state_transition_count,
            "story_conflict_count": self.story_conflict_count,
            "consolidation_candidate_index_ref": self.consolidation_candidate_index_ref.to_dict(),
            "consolidation_decision_set_ref": self.consolidation_decision_set_ref.to_dict(),
            "canonical_fact_set_ref": self.canonical_fact_set_ref.to_dict(),
            "canonical_event_set_ref": self.canonical_event_set_ref.to_dict(),
            "canonical_relationship_set_ref": self.canonical_relationship_set_ref.to_dict(),
            "story_conflict_set_ref": self.story_conflict_set_ref.to_dict(),
            "consolidation_manifest_ref": self.consolidation_manifest_ref.to_dict(),
            "validation_report_ref": self.validation_report_ref.to_dict(),
            "current_pointer_ref": self.current_pointer_ref.to_dict(),
            "reused": self.reused,
        }


def _bound_reference_counts(planning: ConsolidationPlanningResult) -> tuple[int, int]:
    """Count indexed A5 entity-reference occurrences by A4 bound namespace."""
    refs: list[str] = []
    for candidate in planning.index.facts:
        refs.extend(candidate.subject_refs)
        refs.extend(candidate.object_refs)
    for candidate in planning.index.events:
        refs.extend(candidate.participants)
        refs.extend(candidate.locations)
    for candidate in planning.index.relationships:
        refs.extend((candidate.source_entity_ref, candidate.target_entity_ref))
    return (
        sum(not ref.startswith("unres_") for ref in refs),
        sum(ref.startswith("unres_") for ref in refs),
    )


class EvidenceConsolidationService:
    """Provider-neutral A5G1 composition over the frozen A5B--A5F seams."""

    def __init__(
        self,
        store: FileArtifactStore,
        pointers: FilePointerStore,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._store = store
        self._pointers = pointers
        self._prompts = prompt_registry or PromptRegistry(
            DEFAULT_CONSOLIDATION_PROMPT_BASE_DIR
        )

    def consolidate_evidence(
        self,
        *,
        project_id: str,
        document_id: str,
        reconciliation_profile_id: str,
        consolidation_profile: ConsolidationProfile,
        semantic_profile: SemanticLLMProfile,
        llm_client: LLMClient,
        max_concurrency: int = 1,
    ) -> EvidenceConsolidationStageResult:
        """Execute A5G1 planning, pre-provider reuse, or fresh consolidation."""
        max_concurrency = validate_a5_max_concurrency(max_concurrency)
        planning = build_consolidation_planning(
            self._store,
            self._pointers,
            project_id=project_id,
            document_id=document_id,
            reconciliation_profile_id=reconciliation_profile_id,
            consolidation_profile=consolidation_profile,
        )
        # These three preparations are the frozen identity material and must
        # precede the only reuse authority and every provider call.
        fact_preparation = build_fact_semantic_preparation(
            planning, consolidation_profile, semantic_profile,
            prompts=self._prompts, packing_policy=FACT_SEMANTIC_PACKING_V2,
        )
        event_preparation = build_event_semantic_preparation(
            planning, consolidation_profile, semantic_profile,
            prompts=self._prompts, packing_policy=EVENT_SEMANTIC_PACKING_V1,
        )
        relationship_preparation = build_relationship_semantic_preparation(
            planning, consolidation_profile, semantic_profile,
            prompts=self._prompts, packing_policy=RELATIONSHIP_SEMANTIC_PACKING_V1,
        )
        persistence = ConsolidationPersistenceService(self._store, self._pointers)
        publication = persistence.try_reuse_current(
            project_id=project_id,
            document_id=document_id,
            consolidation_profile=consolidation_profile,
            planning_result=planning,
            fact_preparation=fact_preparation,
            event_preparation=event_preparation,
            relationship_preparation=relationship_preparation,
        )
        block_counts = (
            len(fact_preparation.blocks),
            len(event_preparation.blocks),
            len(relationship_preparation.blocks),
        )
        if publication is not None:
            return self._stage_result_from_persisted(
                planning, publication, block_counts=block_counts
            )

        # The existing domain authorities execute sequentially in frozen order.
        fact_resolution = resolve_fact_semantic_ambiguity(
            planning, consolidation_profile, semantic_profile, llm_client,
            prompts=self._prompts, max_concurrency=max_concurrency,
        )
        event_resolution = resolve_event_semantic_ambiguity(
            planning, consolidation_profile, semantic_profile, llm_client,
            prompts=self._prompts, max_concurrency=max_concurrency,
        )
        relationship_resolution = resolve_relationship_semantic_ambiguity(
            planning, consolidation_profile, semantic_profile, llm_client,
            prompts=self._prompts, max_concurrency=max_concurrency,
        )
        finalization = finalize_consolidation(
            planning, fact_resolution, event_resolution, relationship_resolution
        )
        publication = persistence.publish_validated(
            project_id=project_id,
            document_id=document_id,
            consolidation_profile=consolidation_profile,
            planning_result=planning,
            fact_resolution=fact_resolution,
            event_resolution=event_resolution,
            relationship_resolution=relationship_resolution,
            finalization_result=finalization,
        )
        return self._stage_result(
            planning=planning,
            publication=publication,
            block_counts=block_counts,
            semantic_decision_counts=(
                len(fact_resolution.semantic_decisions),
                len(event_resolution.semantic_decisions),
                len(relationship_resolution.semantic_decisions),
            ),
            semantic_generation_call_count=sum(
                result.semantic_rounds
                for result in (
                    *fact_resolution.block_results,
                    *event_resolution.block_results,
                    *relationship_resolution.block_results,
                )
            ),
            canonical_counts=(
                len(finalization.canonical_fact_set.facts),
                len(finalization.canonical_event_set.events),
                len(finalization.canonical_relationship_set.relationships),
                len(finalization.canonical_fact_set.state_transitions),
                len(finalization.story_conflict_set.conflicts),
            ),
            reused=False,
        )

    def _stage_result_from_persisted(
        self,
        planning: ConsolidationPlanningResult,
        publication: ConsolidationPublication,
        *,
        block_counts: tuple[int, int, int],
    ) -> EvidenceConsolidationStageResult:
        """Load only typed persisted A5 leaves for a pre-provider reuse report."""
        base = publication.consolidation_manifest_ref.artifact_id.rsplit(".manifest", 1)[0]
        decisions = load_consolidation_decision_set(
            self._store, publication.consolidation_decision_set_ref,
            expected_artifact_id=consolidation_decision_set_artifact_id(base),
        )
        facts = load_canonical_fact_set(
            self._store, publication.canonical_fact_set_ref,
            expected_artifact_id=canonical_fact_set_artifact_id(base),
        )
        events = load_canonical_event_set(
            self._store, publication.canonical_event_set_ref,
            expected_artifact_id=canonical_event_set_artifact_id(base),
        )
        relationships = load_canonical_relationship_set(
            self._store, publication.canonical_relationship_set_ref,
            expected_artifact_id=canonical_relationship_set_artifact_id(base),
        )
        conflicts = load_story_conflict_set(
            self._store, publication.story_conflict_set_ref,
            expected_artifact_id=story_conflict_set_artifact_id(base),
        )
        return self._stage_result(
            planning=planning,
            publication=publication,
            block_counts=block_counts,
            semantic_decision_counts=(
                sum(d.method == "llm" for d in decisions.fact_decisions),
                sum(d.method == "llm" for d in decisions.event_decisions),
                sum(d.method == "llm" for d in decisions.relationship_decisions),
            ),
            semantic_generation_call_count=0,
            canonical_counts=(
                len(facts.facts), len(events.events), len(relationships.relationships),
                len(facts.state_transitions), len(conflicts.conflicts),
            ),
            reused=True,
        )

    @staticmethod
    def _stage_result(
        *,
        planning: ConsolidationPlanningResult,
        publication: ConsolidationPublication,
        block_counts: tuple[int, int, int],
        semantic_decision_counts: tuple[int, int, int],
        semantic_generation_call_count: int,
        canonical_counts: tuple[int, int, int, int, int],
        reused: bool,
    ) -> EvidenceConsolidationStageResult:
        resolved, unresolved = _bound_reference_counts(planning)
        deterministic = planning.deterministic_decision_set
        return EvidenceConsolidationStageResult(
            entity_map_ref=planning.snapshot.entity_map_ref,
            candidate_extraction_refs=planning.snapshot.candidate_extraction_refs,
            fact_candidate_count=len(planning.index.facts),
            event_candidate_count=len(planning.index.events),
            relationship_candidate_count=len(planning.index.relationships),
            resolved_bound_reference_count=resolved,
            unresolved_bound_reference_count=unresolved,
            fact_planned_pair_count=len(planning.fact_pair_plans),
            event_planned_pair_count=len(planning.event_pair_plans),
            relationship_planned_pair_count=len(planning.relationship_pair_plans),
            fact_deterministic_decision_count=len(deterministic.fact_decisions),
            event_deterministic_decision_count=len(deterministic.event_decisions),
            relationship_deterministic_decision_count=len(deterministic.relationship_decisions),
            fact_semantic_decision_count=semantic_decision_counts[0],
            event_semantic_decision_count=semantic_decision_counts[1],
            relationship_semantic_decision_count=semantic_decision_counts[2],
            fact_semantic_block_count=block_counts[0],
            event_semantic_block_count=block_counts[1],
            relationship_semantic_block_count=block_counts[2],
            semantic_generation_call_count=semantic_generation_call_count,
            canonical_fact_count=canonical_counts[0],
            canonical_event_count=canonical_counts[1],
            canonical_relationship_count=canonical_counts[2],
            state_transition_count=canonical_counts[3],
            story_conflict_count=canonical_counts[4],
            consolidation_candidate_index_ref=publication.consolidation_candidate_index_ref,
            consolidation_decision_set_ref=publication.consolidation_decision_set_ref,
            canonical_fact_set_ref=publication.canonical_fact_set_ref,
            canonical_event_set_ref=publication.canonical_event_set_ref,
            canonical_relationship_set_ref=publication.canonical_relationship_set_ref,
            story_conflict_set_ref=publication.story_conflict_set_ref,
            consolidation_manifest_ref=publication.consolidation_manifest_ref,
            validation_report_ref=publication.validation_report_ref,
            current_pointer_ref=publication.current_pointer_ref,
            reused=reused,
        )


def consolidate_evidence_project(
    project_path: str | Path,
    *,
    runs_root: str | Path,
    reconciliation_profile_id: str,
    consolidation_profile_path: str | Path,
    semantic_profile_path: str | Path,
    llm_client: LLMClient,
    max_concurrency: int = 1,
) -> EvidenceConsolidationStageResult:
    """Provider-neutral project composition; runtime/client wiring is A5G2."""
    from short_drama.llm import load_semantic_profile

    from .service import DOCUMENT_ID, _load_project, _stores

    _project_file, project = _load_project(project_path)
    project_id = project["project_id"]
    consolidation_profile = load_consolidation_profile(Path(consolidation_profile_path))
    semantic_profile = load_semantic_profile(semantic_profile_path)
    store, pointers = _stores(runs_root, project_id)
    return EvidenceConsolidationService(store, pointers).consolidate_evidence(
        project_id=project_id,
        document_id=DOCUMENT_ID,
        reconciliation_profile_id=reconciliation_profile_id,
        consolidation_profile=consolidation_profile,
        semantic_profile=semantic_profile,
        llm_client=llm_client,
        max_concurrency=max_concurrency,
    )


__all__ = [
    "DEFAULT_CONSOLIDATION_PROMPT_BASE_DIR",
    "EvidenceConsolidationService",
    "EvidenceConsolidationStageResult",
    "consolidate_evidence_project",
]
