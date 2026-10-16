"""Provider-neutral A6 production composition.

This module deliberately only joins the already-frozen A6B--A6F authorities.
It does not create an A6 cache, persist semantic checkpoints, or contain any
provider/runtime policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import LLMClient, PromptRegistry, SemanticLLMProfile
from short_drama.paths import REPO_ROOT

from .story_analysis import (
    StoryAnalysisManifest,
    StoryAnalysisProfile,
    load_story_analysis_profile,
)
from .story_analysis_global_skeleton_semantic import resolve_global_skeleton
from .story_analysis_persistence import (
    StoryAnalysisPersistenceService,
    StoryAnalysisPublication,
    validate_whole_a6,
)
from .story_analysis_planning import (
    StoryAnalysisInputSnapshot,
    build_story_analysis_plan_from_profile,
    build_story_analysis_snapshot,
)
from .story_analysis_semantic import resolve_character_analysis
from .story_analysis_story_bible_semantic import resolve_story_bible
from .story_analysis_window_semantic import resolve_plot_window_analysis


DEFAULT_STORY_ANALYSIS_PROMPT_BASE_DIR = REPO_ROOT / "prompts" / "story"


@dataclass(frozen=True, slots=True)
class StoryAnalysisStageResult:
    """Non-persisted A6 reporting result; refs remain the sole authority."""

    project_id: str
    document_id: str
    a5_manifest_ref: ArtifactRef
    a6_manifest_ref: ArtifactRef
    validation_report_ref: ArtifactRef
    current_pointer_ref: ArtifactRef
    character_count: int
    character_analysis_count: int
    event_count: int
    owned_event_count: int
    plot_window_count: int
    global_event_importance_count: int
    arc_count: int
    turning_point_count: int
    reveal_count: int
    foreshadow_payoff_count: int
    semantic_generation_call_count: int
    semantic_regeneration_retry_count: int
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "document_id": self.document_id,
            "a5_manifest_ref": self.a5_manifest_ref.to_dict(),
            "a6_manifest_ref": self.a6_manifest_ref.to_dict(),
            "validation_report_ref": self.validation_report_ref.to_dict(),
            "current_pointer_ref": self.current_pointer_ref.to_dict(),
            "a6_validation_report_ref": self.validation_report_ref.to_dict(),
            "a6_current_pointer_ref": self.current_pointer_ref.to_dict(),
            "character_count": self.character_count,
            "character_analysis_count": self.character_analysis_count,
            "event_count": self.event_count,
            "owned_event_count": self.owned_event_count,
            "plot_window_count": self.plot_window_count,
            "global_event_importance_count": self.global_event_importance_count,
            "arc_count": self.arc_count,
            "turning_point_count": self.turning_point_count,
            "reveal_count": self.reveal_count,
            "foreshadow_payoff_count": self.foreshadow_payoff_count,
            "semantic_generation_call_count": self.semantic_generation_call_count,
            "semantic_regeneration_retry_count": self.semantic_regeneration_retry_count,
            "reused": self.reused,
        }


class StoryAnalysisService:
    """Minimal A6B--A6F orchestration over existing typed authorities."""

    def __init__(
        self,
        store: FileArtifactStore,
        pointers: FilePointerStore,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._store = store
        self._pointers = pointers
        self._prompts = prompt_registry or PromptRegistry(
            DEFAULT_STORY_ANALYSIS_PROMPT_BASE_DIR
        )

    def build_story_bible(
        self,
        *,
        project_id: str,
        document_id: str,
        consolidation_profile_id: str,
        story_analysis_profile: StoryAnalysisProfile,
        semantic_profile: SemanticLLMProfile,
        llm_client: LLMClient,
    ) -> StoryAnalysisStageResult:
        """Resolve exact A5, reuse exact current, or run complete A6C--A6F.

        Every semantic value remains in memory until all four passes have
        completed and whole-A6 validation has succeeded.  Consequently a
        semantic failure cannot publish a partial A6 revision.
        """
        snapshot = build_story_analysis_snapshot(
            self._store,
            self._pointers,
            project_id=project_id,
            document_id=document_id,
            consolidation_profile_id=consolidation_profile_id,
        )
        plan = build_story_analysis_plan_from_profile(
            snapshot, story_analysis_profile
        )
        persistence = StoryAnalysisPersistenceService(self._store, self._pointers)
        reused = persistence.try_reuse_current(
            project_id=project_id,
            document_id=document_id,
            profile=story_analysis_profile,
            semantic_profile=semantic_profile,
            prompts=self._prompts,
            snapshot=snapshot,
            plan=plan,
        )
        if reused is not None:
            return self._result_from_publication(
                snapshot, reused, semantic_calls=0, semantic_retries=0
            )

        characters = resolve_character_analysis(
            plan, story_analysis_profile, semantic_profile, llm_client,
            prompts=self._prompts,
        )
        windows = resolve_plot_window_analysis(
            snapshot, plan, story_analysis_profile, semantic_profile, llm_client,
            prompts=self._prompts,
        )
        skeleton = resolve_global_skeleton(
            snapshot,
            plan,
            story_analysis_profile,
            semantic_profile,
            characters.analyses,
            windows.analyses,
            llm_client,
            prompts=self._prompts,
            character_request_identity_hashes=(
                characters.preparation.character_request_identity_hashes
            ),
            window_request_identity_hashes=(
                windows.preparation.window_request_identity_hashes
            ),
        )
        bible = resolve_story_bible(
            snapshot,
            plan,
            story_analysis_profile,
            semantic_profile,
            characters.analyses,
            skeleton.global_event_analysis,
            skeleton.arc_analysis,
            skeleton.global_structure,
            llm_client,
            prompts=self._prompts,
            character_request_identity_hashes=(
                characters.preparation.character_request_identity_hashes
            ),
            window_request_identity_hashes=(
                windows.preparation.window_request_identity_hashes
            ),
            global_skeleton_request_identity_hash=(
                skeleton.preparation.request_identity_hash
            ),
        )
        validate_whole_a6(
            snapshot,
            plan,
            characters.character_analysis_set,
            skeleton.global_event_analysis,
            skeleton.arc_analysis,
            skeleton.global_structure,
            bible.story_bible,
        )
        publication = persistence.publish_validated(
            project_id=project_id,
            document_id=document_id,
            profile=story_analysis_profile,
            semantic_profile=semantic_profile,
            prompts=self._prompts,
            snapshot=snapshot,
            plan=plan,
            characters=characters.character_analysis_set,
            events=skeleton.global_event_analysis,
            arcs=skeleton.arc_analysis,
            structure=skeleton.global_structure,
            bible=bible.story_bible,
        )
        rounds = (
            sum(round_count for _, round_count in characters.character_rounds)
            + sum(round_count for _, round_count in windows.window_rounds)
            + skeleton.rounds_consumed
            + bible.rounds_consumed
        )
        request_count = len(characters.analyses) + len(windows.analyses) + 2
        return self._result_from_publication(
            snapshot,
            publication,
            semantic_calls=rounds,
            semantic_retries=rounds - request_count,
        )

    def _result_from_publication(
        self,
        snapshot: StoryAnalysisInputSnapshot,
        publication: StoryAnalysisPublication,
        *,
        semantic_calls: int,
        semantic_retries: int,
    ) -> StoryAnalysisStageResult:
        envelope = self._store.get_ref(publication.story_analysis_manifest_ref)
        manifest = StoryAnalysisManifest.from_dict(envelope.payload)
        coverage = manifest.coverage_summary
        return StoryAnalysisStageResult(
            project_id=manifest.project_id,
            document_id=manifest.document_id,
            a5_manifest_ref=snapshot.consolidation_manifest_ref,
            a6_manifest_ref=publication.story_analysis_manifest_ref,
            validation_report_ref=publication.validation_report_ref,
            current_pointer_ref=publication.current_pointer_ref,
            character_count=coverage.canonical_character_count,
            character_analysis_count=coverage.character_analysis_count,
            event_count=coverage.canonical_event_count,
            owned_event_count=coverage.owned_event_count,
            plot_window_count=coverage.plot_window_count,
            global_event_importance_count=coverage.global_event_importance_count,
            arc_count=coverage.arc_count,
            turning_point_count=coverage.turning_point_count,
            reveal_count=coverage.reveal_count,
            foreshadow_payoff_count=coverage.foreshadow_payoff_count,
            semantic_generation_call_count=semantic_calls,
            semantic_regeneration_retry_count=semantic_retries,
            reused=publication.reused,
        )


def build_story_bible_project(
    project_path: str | Path,
    *,
    runs_root: str | Path,
    consolidation_profile_id: str,
    story_analysis_profile_path: str | Path,
    semantic_profile_path: str | Path,
    llm_client: LLMClient,
) -> StoryAnalysisStageResult:
    """Project-level A6 composition; runtime/client construction stays outside."""
    from short_drama.llm import load_semantic_profile

    from .service import DOCUMENT_ID, _load_project, _stores

    _project_file, project = _load_project(project_path)
    profile = load_story_analysis_profile(Path(story_analysis_profile_path))
    semantic_profile = load_semantic_profile(semantic_profile_path)
    store, pointers = _stores(runs_root, project["project_id"])
    return StoryAnalysisService(store, pointers).build_story_bible(
        project_id=project["project_id"],
        document_id=DOCUMENT_ID,
        consolidation_profile_id=consolidation_profile_id,
        story_analysis_profile=profile,
        semantic_profile=semantic_profile,
        llm_client=llm_client,
    )


__all__ = [
    "DEFAULT_STORY_ANALYSIS_PROMPT_BASE_DIR",
    "StoryAnalysisService",
    "StoryAnalysisStageResult",
    "build_story_bible_project",
]
