"""A6F semantic-output assembly regressions; all provider calls are fake."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from shutil import copytree
from typing import Any

import pytest

from short_drama.llm import (
    LLMClient, LLMInvocationProvenance, PromptRegistry,
    StructuredGenerationResult, build_structured_request, load_semantic_profile,
    compute_prompt_content_hash, validate_against_output_schema,
)
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    ArcAnalysis, CharacterAnalysis, CharacterAnalysisSet, EvidenceBackedInterpretation,
    GlobalEventAnalysis, GlobalStructure, PlotWindowAnalysis, Reveal, StoryArc,
    TurningPoint, build_character_semantic_preparation,
    build_a6_semantic_identity,
    build_global_skeleton_semantic_preparation,
    build_plot_window_semantic_preparation, build_story_analysis_plan_from_profile,
    load_story_analysis_profile, load_story_bible_output_schema, resolve_story_bible,
    validate_story_bible_output, validate_whole_a6,
)
from short_drama.llm.errors import LLMStructuredOutputError
from short_drama.story.consolidation import StoryConflict, StoryConflictSet

from test_story_a6b_planning import _snapshot


def _interpretation() -> EvidenceBackedInterpretation:
    return EvidenceBackedInterpretation(text_zh="测试解读。", evidence_mode="explicit")


def _provider_payload(event_ref: str) -> dict[str, Any]:
    """Exactly the eight fields owned by the current provider JSON Schema."""
    interpretation = {
        "text_zh": "测试文本。",
        "evidence_mode": "explicit",
        "supporting_fact_refs": [],
        "supporting_event_refs": [event_ref],
        "supporting_relationship_refs": [],
        "supporting_conflict_refs": [],
    }
    return {
        "premise_zh": deepcopy(interpretation),
        "synopsis_zh": deepcopy(interpretation),
        "genre": "悬疑",
        "tone": "克制",
        "setting": deepcopy(interpretation),
        "themes": [deepcopy(interpretation)],
        "unresolved_summary": deepcopy(interpretation),
        "source_conflict_summary": deepcopy(interpretation),
    }


def _matching_provenance(request) -> LLMInvocationProvenance:
    rendered = request.rendered_prompt
    schema = request.output_schema
    return LLMInvocationProvenance(
        provider_family="fake", model="fake-model",
        semantic_profile_id=request.semantic_profile.profile_id,
        semantic_profile_hash=request.semantic_profile.semantic_profile_hash,
        prompt_id=rendered.prompt_id, prompt_version=rendered.prompt_version,
        prompt_content_hash=rendered.prompt_content_hash,
        rendered_prompt_hash=rendered.rendered_prompt_hash,
        output_schema_id=schema.schema_id,
        output_schema_version=schema.schema_version,
        output_schema_hash=schema.schema_hash, request_hash=request.request_hash,
        provider_response_id="fake-response", finish_reason="stop", usage=None,
    )


class _FakeStoryBibleLLM(LLMClient):
    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.call_count = 0

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile, *, execution_options=None):
        self.call_count += 1
        # Preserve A-I3's local authoritative Provider-schema boundary.
        validate_against_output_schema(self.payload, output_schema)
        request = build_structured_request(
            rendered_prompt=rendered_prompt, output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        return StructuredGenerationResult(
            parsed_json=self.payload, provenance=_matching_provenance(request), attempts=1,
        )


def _inputs(tmp_path, *, with_source_conflict: bool = False):
    _tree, _a5_planning, _a5_publication, snapshot = _snapshot(tmp_path)
    if with_source_conflict:
        fact = snapshot.facts[0]
        snapshot = replace(snapshot, story_conflict_set=StoryConflictSet(1, (
            StoryConflict(
                conflict_id="conf_000001", conflict_kind="fact_conflict",
                fact_ids=(fact.fact_id,), relationship_ids=(),
                candidate_refs=tuple(fact.candidate_fact_refs), decision_refs=(),
                evidence_refs=(), status="unresolved",
            ),
        )))
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    semantic_profile = load_semantic_profile(REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml")
    prompts = PromptRegistry(REPO_ROOT / "prompts" / "story")
    plan = build_story_analysis_plan_from_profile(snapshot, profile)
    character_analyses = tuple(CharacterAnalysis(
        character_ref=package.character_ref, role=_interpretation(), goals=(), motivations=(),
        traits=(), key_event_refs=(), key_fact_refs=(), important_relationship_refs=(),
        arc_summary=_interpretation(), unresolved_or_conflicting_points=(),
    ) for package in plan.character_packages)
    windows = tuple(PlotWindowAnalysis(
        window_id=window.window_id, window_ordinal=window.window_ordinal,
        owned_event_refs=window.owned_event_ids, context_event_refs=window.context_event_ids,
        interpretation=_interpretation(),
    ) for window in plan.windows)
    global_event_analysis = GlobalEventAnalysis(
        schema_version=1, plot_window_plan_hash=plan.plan_hash, windows=windows,
        event_importance_overlay=(), coverage_metadata=(
            ("canonical_event_count", len(plan.event_stream)),
            ("event_importance_count", 0), ("plot_window_count", len(windows)),
        ),
    )

    character_ref = snapshot.canonical_characters[0].canonical_id
    event_ref = snapshot.events[0].event_id
    fact_ref = snapshot.facts[0].fact_id
    arc_analysis = ArcAnalysis(
        schema_version=1,
        arcs=(StoryArc("arc_000001", "main", (character_ref,), (), (event_ref,),
                       (fact_ref,), event_ref, event_ref, _interpretation()),),
        turning_points=(TurningPoint("turn_000001", event_ref, (fact_ref,), (), _interpretation()),),
        reveals=(Reveal("reveal_000001", (event_ref,), (fact_ref,), (character_ref,),
                        (), _interpretation()),),
        foreshadow_payoffs=(),
    )
    global_structure = GlobalStructure(
        schema_version=1, main_conflict=_interpretation(), secondary_conflicts=(),
        main_plot=_interpretation(), subplots=(), ending_state=_interpretation(),
        global_sections=(), main_character_refs=(character_ref,),
        major_arc_refs=("arc_000001",), major_turning_point_refs=("turn_000001",),
        major_reveal_refs=("reveal_000001",),
    )
    a6c = build_character_semantic_preparation(plan, profile, semantic_profile, prompts=prompts)
    a6d = build_plot_window_semantic_preparation(snapshot, plan, profile, semantic_profile, prompts=prompts)
    a6e = build_global_skeleton_semantic_preparation(
        snapshot, plan, profile, semantic_profile, character_analyses, windows,
        prompts=prompts, character_request_identity_hashes=a6c.character_request_identity_hashes,
        window_request_identity_hashes=a6d.window_request_identity_hashes,
    )
    return (snapshot, plan, profile, semantic_profile, prompts, character_analyses,
            global_event_analysis, arc_analysis, global_structure, a6c, a6d, a6e)


def test_schema_only_payload_validates_and_resolves_with_python_owned_a6_refs(tmp_path) -> None:
    (snapshot, plan, profile, semantic_profile, prompts, character_analyses,
     global_event_analysis, arc_analysis, global_structure, a6c, a6d, a6e) = _inputs(tmp_path)
    payload = _provider_payload(snapshot.events[0].event_id)

    assert set(payload) == set(load_story_bible_output_schema().schema["required"])
    validate_against_output_schema(payload, load_story_bible_output_schema())
    assert validate_story_bible_output(
        payload, snapshot, global_event_analysis, arc_analysis, global_structure,
    ) is None

    llm = _FakeStoryBibleLLM(payload)
    result = resolve_story_bible(
        snapshot, plan, profile, semantic_profile, character_analyses,
        global_event_analysis, arc_analysis, global_structure, llm, prompts=prompts,
        character_request_identity_hashes=a6c.character_request_identity_hashes,
        window_request_identity_hashes=a6d.window_request_identity_hashes,
        global_skeleton_request_identity_hash=a6e.request_identity_hash,
    )

    assert llm.call_count == 1
    assert result.story_bible.main_character_refs == global_structure.main_character_refs
    assert result.story_bible.major_arc_refs == global_structure.major_arc_refs
    assert result.story_bible.major_turning_point_refs == global_structure.major_turning_point_refs
    assert result.story_bible.major_reveal_refs == global_structure.major_reveal_refs


def test_story_bible_output_keeps_strict_ref_and_provider_key_rejection(tmp_path) -> None:
    (snapshot, _plan, _profile, _semantic_profile, _prompts, _character_analyses,
     global_event_analysis, arc_analysis, global_structure, _a6c, _a6d, _a6e) = _inputs(tmp_path)
    invalid_ref = _provider_payload(snapshot.events[0].event_id)
    invalid_ref["premise_zh"]["supporting_event_refs"] = ["evt_999999"]
    assert validate_story_bible_output(
        invalid_ref, snapshot, global_event_analysis, arc_analysis, global_structure,
    ) == "StoryBible has ref outside A5 evidence universe"

    extra = _provider_payload(snapshot.events[0].event_id)
    extra["not_in_provider_schema"] = "reject"
    with pytest.raises(LLMStructuredOutputError):
        validate_against_output_schema(extra, load_story_bible_output_schema())
    detail = validate_story_bible_output(
        extra, snapshot, global_event_analysis, arc_analysis, global_structure,
    )
    assert detail is not None
    assert "missing_keys=[]" in detail
    assert "unexpected_keys=['not_in_provider_schema']" in detail


def test_zero_source_conflicts_normalize_narrative_hallucination_without_losing_unresolved(tmp_path) -> None:
    (snapshot, plan, profile, semantic_profile, prompts, character_analyses,
     global_event_analysis, arc_analysis, global_structure, a6c, a6d, a6e) = _inputs(tmp_path)
    assert not snapshot.conflicts
    assert snapshot.unresolved_entities
    payload = _provider_payload(snapshot.events[0].event_id)
    payload["unresolved_summary"]["text_zh"] = "仍有待确认的实体身份。"
    payload["source_conflict_summary"]["text_zh"] = "这是小说内部的叙事冲突。"

    result = resolve_story_bible(
        snapshot, plan, profile, semantic_profile, character_analyses,
        global_event_analysis, arc_analysis, global_structure,
        _FakeStoryBibleLLM(payload), prompts=prompts,
        character_request_identity_hashes=a6c.character_request_identity_hashes,
        window_request_identity_hashes=a6d.window_request_identity_hashes,
        global_skeleton_request_identity_hash=a6e.request_identity_hash,
    )

    assert result.story_bible.source_conflict_summary.text_zh == "未发现 canonical 来源证据之间的冲突。"
    assert result.story_bible.source_conflict_summary.evidence_mode == "explicit"
    assert result.story_bible.source_conflict_summary.supporting_fact_refs == ()
    assert result.story_bible.source_conflict_summary.supporting_event_refs == ()
    assert result.story_bible.source_conflict_summary.supporting_relationship_refs == ()
    assert result.story_bible.source_conflict_summary.supporting_conflict_refs == ()
    assert result.story_bible.unresolved_summary.text_zh == "仍有待确认的实体身份。"
    assert global_structure.main_conflict.text_zh == "测试解读。"
    validate_whole_a6(
        snapshot, plan, CharacterAnalysisSet(1, character_analyses),
        global_event_analysis, arc_analysis, global_structure, result.story_bible,
    )


def test_nonzero_source_conflicts_require_and_preserve_canonical_conflict_evidence(tmp_path) -> None:
    (snapshot, plan, profile, semantic_profile, prompts, character_analyses,
     global_event_analysis, arc_analysis, global_structure, a6c, a6d, a6e) = _inputs(
        tmp_path, with_source_conflict=True,
    )
    payload = _provider_payload(snapshot.events[0].event_id)
    source_summary = payload["source_conflict_summary"]
    source_summary["text_zh"] = "两条 canonical 来源事实尚未消解的矛盾。"
    source_summary["supporting_fact_refs"] = [snapshot.facts[0].fact_id]
    source_summary["supporting_event_refs"] = []
    source_summary["supporting_conflict_refs"] = ["conf_000001"]

    assert validate_story_bible_output(
        payload, snapshot, global_event_analysis, arc_analysis, global_structure,
    ) is None
    result = resolve_story_bible(
        snapshot, plan, profile, semantic_profile, character_analyses,
        global_event_analysis, arc_analysis, global_structure,
        _FakeStoryBibleLLM(payload), prompts=prompts,
        character_request_identity_hashes=a6c.character_request_identity_hashes,
        window_request_identity_hashes=a6d.window_request_identity_hashes,
        global_skeleton_request_identity_hash=a6e.request_identity_hash,
    )
    assert result.story_bible.source_conflict_summary.text_zh == source_summary["text_zh"]
    assert result.story_bible.source_conflict_summary.supporting_conflict_refs == ("conf_000001",)

    missing_conflict_ref = deepcopy(payload)
    missing_conflict_ref["source_conflict_summary"]["supporting_conflict_refs"] = []
    assert validate_story_bible_output(
        missing_conflict_ref, snapshot, global_event_analysis, arc_analysis, global_structure,
    ) == "StoryBible source_conflict_summary must reference every A5 source conflict"


def test_changed_a6f_prompt_identity_cannot_reuse_legacy_current_semantics(tmp_path) -> None:
    (snapshot, plan, profile, semantic_profile, prompts, character_analyses,
     global_event_analysis, arc_analysis, global_structure, _a6c, _a6d, _a6e) = _inputs(tmp_path)
    current_identity = build_a6_semantic_identity(
        snapshot=snapshot, plan=plan, profile=profile, semantic_profile=semantic_profile,
        characters=CharacterAnalysisSet(1, character_analyses), events=global_event_analysis,
        arcs=arc_analysis, structure=global_structure, prompts=prompts,
    )
    legacy_root = tmp_path / "legacy-prompts"
    copytree(REPO_ROOT / "prompts" / "story", legacy_root)
    legacy_dir = legacy_root / "a6.story-bible-synthesis" / "v1"
    legacy_system = legacy_dir / "system.txt"
    legacy_system.write_text(
        "你是小说全局分析助手。只使用提供的 canonical context；不得发明引用。区分 explicit 与 inferred，保留 unresolved/conflict，不做改编决策。使用 zh-CN，仅输出结构化 JSON。\n",
        encoding="utf-8",
    )
    legacy_hash = compute_prompt_content_hash(
        prompt_id="a6.story-bible-synthesis", version=1,
        system_template=legacy_system.read_text(encoding="utf-8"),
        user_template=(legacy_dir / "user.txt").read_text(encoding="utf-8"),
        required_variables=["story_context_json"],
    )
    assert legacy_hash == "bdeb964ac8408b95de5946677cedb6a97f607685c2afe002a3ab1a8d34972bc9"
    metadata_path = legacy_dir / "prompt.yaml"
    metadata = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text(
        "\n".join(
            f"content_hash: {legacy_hash}" if line.startswith("content_hash:") else line
            for line in metadata.splitlines()
        ) + "\n",
        encoding="utf-8",
    )
    legacy_identity = build_a6_semantic_identity(
        snapshot=snapshot, plan=plan, profile=profile, semantic_profile=semantic_profile,
        characters=CharacterAnalysisSet(1, character_analyses), events=global_event_analysis,
        arcs=arc_analysis, structure=global_structure, prompts=PromptRegistry(legacy_root),
    )

    assert legacy_identity != current_identity
    assert legacy_identity.story_bible_request_hashes != current_identity.story_bible_request_hashes
