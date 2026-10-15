"""Focused offline contract checks for A6F assets and identity boundaries."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from short_drama.paths import REPO_ROOT
from short_drama.story import (
    A6F_STORY_BIBLE_OUTPUT_SCHEMA_ID,
    A6F_STORY_BIBLE_PROMPT_ID,
    enforce_story_bible_ceiling,
    load_story_analysis_profile,
    load_story_bible_output_schema,
    StoryAnalysisPersistenceService,
)
from short_drama.foundation import PointerKind
from short_drama.llm import PromptRegistry, load_semantic_profile
from test_story_a6b_planning import _snapshot
from short_drama.story import (
    ArcAnalysis, CharacterAnalysis, CharacterAnalysisSet, EvidenceBackedInterpretation,
    GlobalEventAnalysis, GlobalStoryBible, GlobalStructure, PlotWindowAnalysis,
    StoryAnalysisPersistenceService, build_story_analysis_plan_from_profile,
)
from short_drama.story.errors import StoryAnalysisSemanticError, StoryIntegrityError


def test_tracked_a6f_assets_are_loadable_and_pinned() -> None:
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    schema = load_story_bible_output_schema()
    assert profile.story_bible.prompt_id == A6F_STORY_BIBLE_PROMPT_ID
    assert schema.schema_id == A6F_STORY_BIBLE_OUTPUT_SCHEMA_ID
    assert profile.planning_policy.story_bible_packet_max_estimated_tokens == 105_000


def test_deferred_a6f_ceiling_fails_closed_before_provider() -> None:
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    deferred = replace(profile, planning_policy=replace(
        profile.planning_policy, story_bible_packet_max_estimated_tokens=None
    ))
    class Context:
        def estimated_tokens(self): return 1
    with pytest.raises(StoryAnalysisSemanticError, match="DEFERRED"):
        enforce_story_bible_ceiling(deferred, Context())


def test_valid_current_with_changed_plan_is_normal_miss_after_integrity_check() -> None:
    """A new deterministic plan must not validate old windows and call them corrupt."""
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    semantic = __import__("short_drama.llm", fromlist=["load_semantic_profile"]).load_semantic_profile(
        REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml"
    )
    pointers = MagicMock()
    pointers.resolve_current_pointer_ref.return_value = MagicMock()
    pointers.resolve_current.return_value = SimpleNamespace(target_ref=MagicMock(), pointer_kind=PointerKind.CURRENT)
    service = StoryAnalysisPersistenceService(MagicMock(), pointers)
    upstream = MagicMock()
    manifest = SimpleNamespace(
        project_id="project", document_id="document", consolidation_manifest_ref=upstream,
        semantic_identity=SimpleNamespace(plan_hash="old-plan"),
    )
    service._load_current_integrity = MagicMock(return_value=(
        manifest, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
    ))
    assert service.try_reuse_current(
        project_id="project", document_id="document", profile=profile,
        semantic_profile=semantic, prompts=MagicMock(),
        snapshot=SimpleNamespace(consolidation_manifest_ref=upstream),
        plan=SimpleNamespace(plan_hash="new-plan"),
    ) is None
    service._load_current_integrity.assert_called_once()


def test_corrupt_current_fails_closed_before_identity_miss() -> None:
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    pointers = MagicMock()
    pointers.resolve_current_pointer_ref.return_value = MagicMock()
    pointers.resolve_current.return_value = SimpleNamespace(target_ref=MagicMock(), pointer_kind=PointerKind.CURRENT)
    service = StoryAnalysisPersistenceService(MagicMock(), pointers)
    service._load_current_integrity = MagicMock(side_effect=StoryIntegrityError("corrupt"))
    with pytest.raises(StoryIntegrityError, match="corrupt"):
        service.try_reuse_current(
            project_id="project", document_id="document", profile=profile,
            semantic_profile=MagicMock(), prompts=MagicMock(),
            snapshot=SimpleNamespace(consolidation_manifest_ref=MagicMock()),
            plan=SimpleNamespace(plan_hash="new-plan"),
        )


def _interp() -> EvidenceBackedInterpretation:
    return EvidenceBackedInterpretation(text_zh="测试。", evidence_mode="explicit")


def test_real_store_publication_and_exact_reuse(tmp_path) -> None:
    """Exercise immutable A6 leaves/CURRENT, not mocked persistence methods."""
    tree, _a5_planning, _a5_pub, snapshot = _snapshot(tmp_path)
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    semantic = load_semantic_profile(REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml")
    plan = build_story_analysis_plan_from_profile(snapshot, profile)
    analyses = tuple(CharacterAnalysis(
        character_ref=p.character_ref, role=_interp(), goals=(), motivations=(), traits=(),
        key_event_refs=(), key_fact_refs=(), important_relationship_refs=(),
        arc_summary=_interp(), unresolved_or_conflicting_points=(),
    ) for p in plan.character_packages)
    windows = tuple(PlotWindowAnalysis(
        window_id=w.window_id, window_ordinal=w.window_ordinal,
        owned_event_refs=w.owned_event_ids, context_event_refs=w.context_event_ids,
        interpretation=_interp(),
    ) for w in plan.windows)
    chars = CharacterAnalysisSet(1, analyses)
    events = GlobalEventAnalysis(1, plan.plan_hash, windows, (), (
        ("canonical_event_count", len(plan.event_stream)),
        ("event_importance_count", 0), ("plot_window_count", len(windows)),
    ))
    arcs = ArcAnalysis(1, (), (), (), ())
    structure = GlobalStructure(1, _interp(), (), _interp(), (), _interp(), (), (), (), (), ())
    bible = GlobalStoryBible(1, _interp(), _interp(), "测试", "测试", _interp(), (), (), (), (), (), _interp(), _interp(), (
        ("canonical_character_count", len(analyses)), ("canonical_event_count", len(plan.event_stream)),
    ))
    service = StoryAnalysisPersistenceService(tree.store, tree.pointers)
    prompts = PromptRegistry(REPO_ROOT / "prompts" / "story")
    publication = service.publish_validated(
        project_id=snapshot.consolidation_manifest.project_id,
        document_id=snapshot.consolidation_manifest.document_id, profile=profile,
        semantic_profile=semantic, prompts=prompts, snapshot=snapshot, plan=plan,
        characters=chars, events=events, arcs=arcs, structure=structure, bible=bible,
    )
    refs = (publication.character_analysis_ref, publication.global_event_analysis_ref,
            publication.arc_analysis_ref, publication.global_structure_ref,
            publication.global_story_bible_ref, publication.story_analysis_manifest_ref,
            publication.validation_report_ref)
    assert len({ref.revision for ref in refs}) == 1
    assert tree.pointers.resolve_current(
        f"{snapshot.consolidation_manifest.project_id}.a6.{snapshot.consolidation_manifest.document_id}.{profile.profile_id}"
    ).target_ref == publication.story_analysis_manifest_ref
    assert service.try_reuse_current(
        project_id=snapshot.consolidation_manifest.project_id,
        document_id=snapshot.consolidation_manifest.document_id, profile=profile,
        semantic_profile=semantic, prompts=prompts, snapshot=snapshot, plan=plan,
    ).story_analysis_manifest_ref == publication.story_analysis_manifest_ref
