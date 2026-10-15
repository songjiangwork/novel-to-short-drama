"""A6F semantic-output assembly regressions; all provider calls are fake."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from short_drama.llm import (
    LLMClient, LLMInvocationProvenance, PromptRegistry,
    StructuredGenerationResult, build_structured_request, load_semantic_profile,
    validate_against_output_schema,
)
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    ArcAnalysis, CharacterAnalysis, EvidenceBackedInterpretation,
    GlobalEventAnalysis, GlobalStructure, PlotWindowAnalysis, Reveal, StoryArc,
    TurningPoint, build_character_semantic_preparation,
    build_global_skeleton_semantic_preparation,
    build_plot_window_semantic_preparation, build_story_analysis_plan_from_profile,
    load_story_analysis_profile, load_story_bible_output_schema, resolve_story_bible,
    validate_story_bible_output,
)
from short_drama.llm.errors import LLMStructuredOutputError

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


def _inputs(tmp_path):
    _tree, _a5_planning, _a5_publication, snapshot = _snapshot(tmp_path)
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
