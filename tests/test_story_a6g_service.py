"""A6G production composition tests with real temporary artifact stores."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from short_drama.story import (
    ArcAnalysis,
    CharacterAnalysis,
    CharacterAnalysisSet,
    EvidenceBackedInterpretation,
    GlobalEventAnalysis,
    GlobalStoryBible,
    GlobalStructure,
    PlotWindowAnalysis,
    StoryAnalysisService,
    a5_pointer_id,
    a6_pointer_id,
    build_story_analysis_plan_from_profile,
    load_story_analysis_profile,
)
from short_drama.story.errors import ConsolidationCurrentMissingError, StoryIntegrityError, StoryPersistenceError
from short_drama.paths import REPO_ROOT
from short_drama.foundation import PointerKind, PointerNotFoundError
from short_drama.llm import LLMError

from test_story_a6c_character_analysis import (
    CONSOLIDATION_PROFILE_ID,
    DOCUMENT,
    PROJECT,
    _semantic_profile,
    _snapshot,
)


def _interp() -> EvidenceBackedInterpretation:
    return EvidenceBackedInterpretation(text_zh="测试解读。", evidence_mode="explicit")


def _profile():
    return load_story_analysis_profile(
        REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
    )


def _fake_semantic_results(plan):
    analyses = tuple(
        CharacterAnalysis(
            character_ref=package.character_ref,
            role=_interp(),
            goals=(), motivations=(), traits=(), key_event_refs=(), key_fact_refs=(),
            important_relationship_refs=(), arc_summary=_interp(),
            unresolved_or_conflicting_points=(),
        )
        for package in plan.character_packages
    )
    windows = tuple(
        PlotWindowAnalysis(
            window_id=window.window_id,
            window_ordinal=window.window_ordinal,
            owned_event_refs=window.owned_event_ids,
            context_event_refs=window.context_event_ids,
            interpretation=_interp(),
        )
        for window in plan.windows
    )
    events = GlobalEventAnalysis(
        1, plan.plan_hash, windows, (), (
            ("canonical_event_count", len(plan.event_stream)),
            ("event_importance_count", 0),
            ("plot_window_count", len(windows)),
        )
    )
    arcs = ArcAnalysis(1, (), (), (), ())
    structure = GlobalStructure(
        1, _interp(), (), _interp(), (), _interp(), (), (), (), (), ()
    )
    bible = GlobalStoryBible(
        1, _interp(), _interp(), "测试", "测试", _interp(), (), (), (), (), (),
        _interp(), _interp(), (
            ("canonical_character_count", len(analyses)),
            ("canonical_event_count", len(plan.event_stream)),
        )
    )
    return analyses, windows, events, arcs, structure, bible


def _patch_fresh_passes(monkeypatch, plan):
    import short_drama.story.story_analysis_service as module

    analyses, windows, events, arcs, structure, bible = _fake_semantic_results(plan)
    char_hashes = tuple("a" * 64 for _ in analyses)
    window_hashes = tuple("b" * 64 for _ in windows)
    monkeypatch.setattr(module, "resolve_character_analysis", lambda *args, **kwargs: SimpleNamespace(
        analyses=analyses,
        character_analysis_set=CharacterAnalysisSet(1, analyses),
        character_rounds=tuple((x.character_ref, 1) for x in analyses),
        preparation=SimpleNamespace(character_request_identity_hashes=char_hashes),
    ))
    monkeypatch.setattr(module, "resolve_plot_window_analysis", lambda *args, **kwargs: SimpleNamespace(
        analyses=windows,
        window_rounds=tuple((x.window_id, 1) for x in windows),
        preparation=SimpleNamespace(window_request_identity_hashes=window_hashes),
    ))
    monkeypatch.setattr(module, "resolve_global_skeleton", lambda *args, **kwargs: SimpleNamespace(
        global_event_analysis=events,
        arc_analysis=arcs,
        global_structure=structure,
        rounds_consumed=1,
        preparation=SimpleNamespace(request_identity_hash="c" * 64),
    ))
    monkeypatch.setattr(module, "resolve_story_bible", lambda *args, **kwargs: SimpleNamespace(
        story_bible=bible, rounds_consumed=1,
    ))


def test_missing_a5_current_fails_before_provider(tmp_path):
    from short_drama.artifacts import FileArtifactStore
    from short_drama.foundation import FilePointerStore

    store = FileArtifactStore(tmp_path / "artifacts")
    pointers = FilePointerStore(tmp_path / "pointers", store)
    service = StoryAnalysisService(store, pointers)
    with pytest.raises(ConsolidationCurrentMissingError, match="A5"):
        service.build_story_bible(
            project_id=PROJECT,
            document_id=DOCUMENT,
            consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
            story_analysis_profile=_profile(),
            semantic_profile=_semantic_profile(),
            llm_client=SimpleNamespace(),
        )


def test_fresh_publication_then_exact_current_reuse_is_zero_provider(tmp_path, monkeypatch):
    tree, _planning, _a5_pub, snapshot = _snapshot(tmp_path)
    profile = _profile()
    plan = build_story_analysis_plan_from_profile(snapshot, profile)
    _patch_fresh_passes(monkeypatch, plan)
    service = StoryAnalysisService(tree.store, tree.pointers)

    fresh = service.build_story_bible(
        project_id=PROJECT,
        document_id=DOCUMENT,
        consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
        story_analysis_profile=profile,
        semantic_profile=_semantic_profile(),
        llm_client=SimpleNamespace(),
    )
    assert fresh.reused is False
    assert fresh.semantic_generation_call_count == len(plan.character_packages) + len(plan.windows) + 2
    assert fresh.semantic_regeneration_retry_count == 0
    assert fresh.character_analysis_count == len(plan.character_packages)
    assert fresh.owned_event_count == len(plan.event_stream)

    def provider_must_not_run(*args, **kwargs):
        raise AssertionError("exact A6 reuse must precede every provider call")

    import short_drama.story.story_analysis_service as module
    monkeypatch.setattr(module, "resolve_character_analysis", provider_must_not_run)
    reused = service.build_story_bible(
        project_id=PROJECT,
        document_id=DOCUMENT,
        consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
        story_analysis_profile=profile,
        semantic_profile=_semantic_profile(),
        llm_client=SimpleNamespace(),
    )
    assert reused.reused is True
    assert reused.semantic_generation_call_count == 0
    assert reused.a6_manifest_ref == fresh.a6_manifest_ref
    assert reused.current_pointer_ref == fresh.current_pointer_ref


def test_profile_identity_change_is_a_normal_fresh_miss(tmp_path, monkeypatch):
    tree, _planning, _a5_pub, snapshot = _snapshot(tmp_path)
    profile = _profile()
    plan = build_story_analysis_plan_from_profile(snapshot, profile)
    _patch_fresh_passes(monkeypatch, plan)
    service = StoryAnalysisService(tree.store, tree.pointers)
    first = service.build_story_bible(
        project_id=PROJECT, document_id=DOCUMENT,
        consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
        story_analysis_profile=profile, semantic_profile=_semantic_profile(),
        llm_client=SimpleNamespace(),
    )
    changed = replace(profile, character_analysis_policy_id="a6-character-v2")
    changed_plan = build_story_analysis_plan_from_profile(snapshot, changed)
    _patch_fresh_passes(monkeypatch, changed_plan)
    second = service.build_story_bible(
        project_id=PROJECT, document_id=DOCUMENT,
        consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
        story_analysis_profile=changed, semantic_profile=_semantic_profile(),
        llm_client=SimpleNamespace(),
    )
    assert second.reused is False
    assert second.a6_manifest_ref.revision == first.a6_manifest_ref.revision + 1


def test_corrupt_a6_current_fails_closed_before_provider(tmp_path, monkeypatch):
    tree, _planning, _a5_pub, snapshot = _snapshot(tmp_path)
    profile = _profile()
    plan = build_story_analysis_plan_from_profile(snapshot, profile)
    _patch_fresh_passes(monkeypatch, plan)
    service = StoryAnalysisService(tree.store, tree.pointers)
    fresh = service.build_story_bible(
        project_id=PROJECT, document_id=DOCUMENT,
        consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
        story_analysis_profile=profile, semantic_profile=_semantic_profile(),
        llm_client=SimpleNamespace(),
    )
    pointer_id = a6_pointer_id(PROJECT, DOCUMENT, profile.profile_id)
    tree.pointers.compare_and_set(
        pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=fresh.current_pointer_ref,
        target_ref=snapshot.consolidation_manifest_ref,
    )
    import short_drama.story.story_analysis_service as module
    monkeypatch.setattr(module, "resolve_character_analysis", lambda *args, **kwargs: pytest.fail("provider called"))
    with pytest.raises(StoryIntegrityError):
        service.build_story_bible(
            project_id=PROJECT, document_id=DOCUMENT,
            consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
            story_analysis_profile=profile, semantic_profile=_semantic_profile(),
            llm_client=SimpleNamespace(),
        )


def test_a5_head_change_during_semantic_execution_never_publishes_a6_current(tmp_path, monkeypatch):
    tree, _planning, _a5_pub, snapshot = _snapshot(tmp_path)
    profile = _profile()
    plan = build_story_analysis_plan_from_profile(snapshot, profile)
    _patch_fresh_passes(monkeypatch, plan)
    import short_drama.story.story_analysis_service as module
    original = module.resolve_character_analysis

    def move_a5_head(*args, **kwargs):
        tree.pointers.compare_and_set(
            pointer_id=a5_pointer_id(PROJECT, DOCUMENT, CONSOLIDATION_PROFILE_ID),
            pointer_kind=PointerKind.CURRENT,
            expected_pointer_ref=snapshot.a5_current_pointer_ref,
            target_ref=snapshot.consolidation_manifest.entity_map_ref,
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "resolve_character_analysis", move_a5_head)
    with pytest.raises(StoryPersistenceError, match="A5 CURRENT changed"):
        StoryAnalysisService(tree.store, tree.pointers).build_story_bible(
            project_id=PROJECT, document_id=DOCUMENT,
            consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
            story_analysis_profile=profile, semantic_profile=_semantic_profile(),
            llm_client=SimpleNamespace(),
        )
    with pytest.raises(PointerNotFoundError):
        tree.pointers.resolve_current(a6_pointer_id(PROJECT, DOCUMENT, profile.profile_id))


def test_provider_failure_does_not_publish_any_a6_current(tmp_path, monkeypatch):
    tree, _planning, _a5_pub, _snapshot_value = _snapshot(tmp_path)
    profile = _profile()
    import short_drama.story.story_analysis_service as module
    monkeypatch.setattr(
        module,
        "resolve_character_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(LLMError("provider failed")),
    )
    with pytest.raises(LLMError, match="provider failed"):
        StoryAnalysisService(tree.store, tree.pointers).build_story_bible(
            project_id=PROJECT, document_id=DOCUMENT,
            consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
            story_analysis_profile=profile, semantic_profile=_semantic_profile(),
            llm_client=SimpleNamespace(),
        )
    with pytest.raises(PointerNotFoundError):
        tree.pointers.resolve_current(a6_pointer_id(PROJECT, DOCUMENT, profile.profile_id))
