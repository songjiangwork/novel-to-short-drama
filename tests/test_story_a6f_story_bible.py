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
    pointers.resolve_current.return_value = SimpleNamespace(target_ref=MagicMock())
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
    pointers.resolve_current.return_value = SimpleNamespace(target_ref=MagicMock())
    service = StoryAnalysisPersistenceService(MagicMock(), pointers)
    service._load_current_integrity = MagicMock(side_effect=StoryIntegrityError("corrupt"))
    with pytest.raises(StoryIntegrityError, match="corrupt"):
        service.try_reuse_current(
            project_id="project", document_id="document", profile=profile,
            semantic_profile=MagicMock(), prompts=MagicMock(),
            snapshot=SimpleNamespace(consolidation_manifest_ref=MagicMock()),
            plan=SimpleNamespace(plan_hash="new-plan"),
        )
