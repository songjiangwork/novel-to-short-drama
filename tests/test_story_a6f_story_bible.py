"""Focused offline contract checks for A6F assets and identity boundaries."""
from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.paths import REPO_ROOT
from short_drama.story import (
    A6F_STORY_BIBLE_OUTPUT_SCHEMA_ID,
    A6F_STORY_BIBLE_PROMPT_ID,
    enforce_story_bible_ceiling,
    load_story_analysis_profile,
    load_story_bible_output_schema,
)
from short_drama.story.errors import StoryAnalysisSemanticError


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
