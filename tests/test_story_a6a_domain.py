"""Focused A6A static-contract and tracked-asset tests (zero provider)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import yaml
from jsonschema import Draft202012Validator

from short_drama.artifacts import ArtifactRef
from short_drama.llm import PromptRegistry, load_semantic_profile
from short_drama.paths import REPO_ROOT, SCHEMAS_DIR
from short_drama.story import (
    A6SemanticIdentity, A6UpstreamIdentity, ArcAnalysis, CharacterAnalysis,
    CharacterAnalysisSet, EvidenceBackedInterpretation, ForeshadowPayoff,
    GlobalEventAnalysis, GlobalEventImportance, GlobalSection, GlobalStoryBible,
    GlobalStructure, PlotWindowAnalysis, PromptAssetIdentity, OutputSchemaAssetIdentity,
    Reveal, StoryAnalysisCoverageSummary, StoryAnalysisManifest, StoryAnalysisModelError,
    StoryAnalysisPlanningPolicy, StoryAnalysisProfile, StoryAnalysisSemanticPass,
    STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1, StoryArc, TurningPoint,
)
from short_drama.story.chunking import ChunkPlanningProfile


def _profile():
    semantic = StoryAnalysisSemanticPass("story-analysis-llm-v1", "a6.character-analysis", 1, "a6-character-analysis-output", 1)
    return StoryAnalysisProfile(1, "story-analysis-v1", "zh-CN", "a6-character-v1", "a6-window-v1", "a6-skeleton-v1", "a6-bible-v1", 2, StoryAnalysisPlanningPolicy(1, 1, 1, 0, None, None), semantic, semantic, semantic, semantic)


def _schema(name): return json.loads((SCHEMAS_DIR / name).read_text())


H = "a" * 64


def _interpretation(text="解释"):
    return EvidenceBackedInterpretation(text, "inferred", ("fact_000001",), ("evt_000001",), ("rel_000001",), ("conf_000001",))


def _artifact(kind="a6-leaf"):
    return ArtifactRef(kind, f"{kind}-0001", 1, H)


def _validate(name, value):
    assert list(Draft202012Validator(_schema(name)).iter_errors(value.to_dict())) == []


def test_profile_round_trip_hash_schema_and_a2_a6_profile_boundaries():
    profile = _profile()
    assert StoryAnalysisProfile.from_dict(profile.to_dict()) == profile
    assert profile.content_hash() == profile.content_hash()
    assert list(Draft202012Validator(_schema("story-analysis-profile.schema.json")).iter_errors(profile.to_dict())) == []
    a2_profile_path = REPO_ROOT / "profiles/story_analysis_v1.yaml"
    a2_bytes = a2_profile_path.read_bytes()
    a2_profile = ChunkPlanningProfile.from_dict(yaml.safe_load(a2_bytes))
    assert a2_profile.profile_id == "story-analysis-v1"
    assert a2_profile_path.read_bytes() == a2_bytes
    assert not (REPO_ROOT / "profiles/global_story_analysis_v1.yaml").exists()
    with pytest.raises(StoryAnalysisModelError): StoryAnalysisProfile.from_dict({**profile.to_dict(), "base_url": "x"})
    assert STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1 == 2
    for rounds in (1, 3):
        invalid = {**profile.to_dict(), "max_generation_rounds": rounds}
        with pytest.raises(StoryAnalysisModelError):
            StoryAnalysisProfile.from_dict(invalid)
        assert list(Draft202012Validator(_schema("story-analysis-profile.schema.json")).iter_errors(invalid))


def test_planning_policy_stages_whole_story_ceilings():
    deferred = StoryAnalysisPlanningPolicy(10, 20, 3, 1, None, None)
    assert StoryAnalysisPlanningPolicy.from_dict(deferred.to_dict()) == deferred
    assert list(Draft202012Validator(_schema("story-analysis-profile.schema.json")).iter_errors(_profile().to_dict())) == []

    concrete = StoryAnalysisPlanningPolicy(10, 20, 3, 1, 30, 40)
    assert StoryAnalysisPlanningPolicy.from_dict(concrete.to_dict()) == concrete

    for field in (
        "global_skeleton_packet_max_estimated_tokens",
        "story_bible_packet_max_estimated_tokens",
    ):
        raw = deferred.to_dict()
        raw[field] = 0
        with pytest.raises(StoryAnalysisModelError):
            StoryAnalysisPlanningPolicy.from_dict(raw)

        profile_raw = _profile().to_dict()
        profile_raw["planning_policy"][field] = 0
        assert list(
            Draft202012Validator(_schema("story-analysis-profile.schema.json")).iter_errors(profile_raw)
        )


def test_interpretation_closed_evidence_mode_and_namespace_round_trip():
    item = EvidenceBackedInterpretation("爱丽丝作出选择", "inferred", ("fact_000001",), ("evt_000001",), ("rel_000001",), ("conf_000001",))
    assert EvidenceBackedInterpretation.from_dict(item.to_dict()) == item
    with pytest.raises(StoryAnalysisModelError): EvidenceBackedInterpretation("x", "guess")
    with pytest.raises(StoryAnalysisModelError): EvidenceBackedInterpretation("x", "explicit", ("evt_000001",))


def test_semantic_profile_and_four_prompts_are_tracked_and_backend_neutral():
    profile = load_semantic_profile(REPO_ROOT / "profiles/story_analysis_llm_v1.yaml")
    assert (profile.profile_id, profile.temperature, profile.max_output_tokens, profile.structured_output_mode) == ("story-analysis-llm-v1", 0.0, 16384, "json_schema")
    registry = PromptRegistry(REPO_ROOT / "prompts/story")
    expected = {
        "a6.character-analysis": ("character_context_json", "3e8b1ed4a1b5981c31a4e4caadcb482dde4af8ed32f045ad918c5af4878c0e0d"),
        "a6.plot-window-analysis": ("window_context_json", "41a4b489a3745ae3f8bc96f478bbc4a3d97a9a05665b3ea930c31ae0d7c1c22a"),
        "a6.global-skeleton": ("global_context_json", "256f94a61c08bfca6ddb11aff13c123eb7d6ecffef03b858433fa777b240cc7b"),
        "a6.story-bible-synthesis": ("story_context_json", "bdeb964ac8408b95de5946677cedb6a97f607685c2afe002a3ab1a8d34972bc9"),
    }
    for prompt_id, (variable, expected_hash) in expected.items():
        asset = registry.load(prompt_id, version=1)
        assert asset.required_variables == (variable,)
        assert asset.content_hash == expected_hash


def test_tampered_same_version_prompt_fails_hash_validation(tmp_path):
    copied = tmp_path / "story"
    shutil.copytree(REPO_ROOT / "prompts" / "story", copied)
    system = copied / "a6.character-analysis" / "v1" / "system.txt"
    system.write_text(system.read_text(encoding="utf-8") + "篡改", encoding="utf-8")
    with pytest.raises(Exception):
        PromptRegistry(copied).load("a6.character-analysis", version=1)


def test_a6_schemas_are_strict_draft_2020_12():
    names = ("story-analysis-profile", "character-analysis-set", "global-event-analysis", "arc-analysis", "global-structure", "global-story-bible", "story-analysis-manifest", "a6-character-analysis-output", "a6-plot-window-analysis-output", "a6-global-skeleton-output", "a6-story-bible-output")
    for name in names:
        schema = _schema(f"{name}.schema.json")
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
    for name in ("a6-character-analysis-output", "a6-plot-window-analysis-output", "a6-global-skeleton-output", "a6-story-bible-output"):
        serialized = json.dumps(_schema(f"{name}.schema.json"), ensure_ascii=False)
        assert not any(field in serialized for field in ("arc_id", "turning_point_id", "reveal_id", "payoff_id"))
        assert "(?=" not in serialized


def test_global_skeleton_provider_schema_has_complete_request_local_semantics():
    schema = _schema("a6-global-skeleton-output.schema.json")
    props = schema["properties"]
    assert set(props) == {
        "event_importance_overlay", "arc_proposals", "turning_point_proposals",
        "reveal_proposals", "foreshadow_payoff_proposals", "global_structure",
    }
    arc = props["arc_proposals"]["items"]["properties"]
    assert {"proposal_ordinal", "arc_kind", "involved_character_refs",
            "involved_relationship_refs", "supporting_event_refs", "supporting_fact_refs",
            "start_event_ref", "end_event_ref", "interpretation"} <= set(arc)
    structure = props["global_structure"]["properties"]
    assert {"main_conflict", "secondary_conflicts", "main_plot", "subplots",
            "ending_state", "global_sections", "main_character_refs",
            "major_arc_proposal_ordinals", "major_turning_point_proposal_ordinals",
            "major_reveal_proposal_ordinals"} <= set(structure)
    serialized = json.dumps(schema, ensure_ascii=False)
    assert not any(field in serialized for field in ("arc_id", "turning_point_id", "reveal_id", "payoff_id"))


def test_character_event_arc_structure_and_bible_round_trip_schema_parity():
    item = _interpretation()
    characters = CharacterAnalysisSet(1, (CharacterAnalysis(
        "char_0001", item, (item,), (item,), (item,), ("evt_000001",),
        ("fact_000001",), ("rel_000001",), item, (item,),
    ),))
    windows = GlobalEventAnalysis(1, H, (PlotWindowAnalysis(
        "window_0001", 1, ("evt_000001",), (), item, (item,), (item,), (item,),
    ),), (GlobalEventImportance("evt_000001", item),), (("owned_event_count", 1),))
    arcs = ArcAnalysis(1, (StoryArc(
        "arc_000001", "plot", ("char_0001",), ("rel_000001",), ("evt_000001",),
        ("fact_000001",), "evt_000001", "evt_000001", item,
    ),), (TurningPoint("turn_000001", "evt_000001", (), (), item),), (Reveal(
        "reveal_000001", ("evt_000001",), (), ("char_0001",), (), item,
    ),), (ForeshadowPayoff("payoff_000001", ("evt_000001",), ("evt_000001",), item),))
    structure = GlobalStructure(1, item, (), item, (), item, (GlobalSection(1, "开端", ("evt_000001",), item),), ("char_0001",), ("arc_000001",), ("turn_000001",), ("reveal_000001",))
    bible = GlobalStoryBible(1, item, item, "悬疑", "克制", item, (item,), ("char_0001",), ("arc_000001",), ("turn_000001",), ("reveal_000001",), item, item, (("canonical_event_count", 1),))
    for value, name in ((characters, "character-analysis-set.schema.json"), (windows, "global-event-analysis.schema.json"), (arcs, "arc-analysis.schema.json"), (structure, "global-structure.schema.json"), (bible, "global-story-bible.schema.json")):
        assert type(value).from_dict(value.to_dict()) == value
        _validate(name, value)
    with pytest.raises(StoryAnalysisModelError): StoryArc("arc_1", "plot", (), (), (), (), "evt_000001", "evt_000001", item)
    with pytest.raises(StoryAnalysisModelError): CharacterAnalysisSet(1, (CharacterAnalysis("loc_0001", item, (), (), (), (), (), (), item, ()),))


def test_persisted_schema_rejects_same_direct_invalid_text_and_duplicate_refs_as_python():
    item = _interpretation()
    characters = CharacterAnalysisSet(1, (CharacterAnalysis(
        "char_0001", item, (), (), (), ("evt_000001",), (), (), item, (),
    ),))
    for mutate in (
        lambda raw: raw["analyses"][0]["role"].__setitem__("text_zh", ""),
        lambda raw: raw["analyses"][0].__setitem__("key_event_refs", ["evt_000001", "evt_000001"]),
    ):
        raw = characters.to_dict()
        mutate(raw)
        with pytest.raises(StoryAnalysisModelError):
            CharacterAnalysisSet.from_dict(raw)
        assert list(Draft202012Validator(_schema("character-analysis-set.schema.json")).iter_errors(raw))


def test_persisted_schema_matches_python_text_whitespace_and_nul_rejection():
    item = _interpretation()
    characters = CharacterAnalysisSet(1, (CharacterAnalysis(
        "char_0001", item, (), (), (), (), (), (), item, (),
    ),))
    for invalid_text in ("   ", "解释\x00内容"):
        raw = characters.to_dict()
        raw["analyses"][0]["role"]["text_zh"] = invalid_text
        with pytest.raises(StoryAnalysisModelError):
            CharacterAnalysisSet.from_dict(raw)
        assert list(Draft202012Validator(_schema("character-analysis-set.schema.json")).iter_errors(raw))

    bible = GlobalStoryBible(
        1, item, item, "悬疑", "克制", item, (), (), (), (), (), item, item, (),
    )
    raw_bible = bible.to_dict()
    raw_bible["genre"] = "\t\n"
    with pytest.raises(StoryAnalysisModelError):
        GlobalStoryBible.from_dict(raw_bible)
    assert list(Draft202012Validator(_schema("global-story-bible.schema.json")).iter_errors(raw_bible))


def test_manifest_identity_is_backend_neutral_and_schema_parity():
    ref = _artifact("consolidation-manifest")
    semantic = A6SemanticIdentity(
        "global-story-analysis-v1", H, "story-analysis-llm-v1", H,
        (PromptAssetIdentity("a6.character-analysis", 1, H),),
        (OutputSchemaAssetIdentity("a6-character-analysis-output", 1, H),), H,
        (H,), (H,), (H,), (H,),
    )
    coverage = StoryAnalysisCoverageSummary(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    manifest = StoryAnalysisManifest(1, "project-1", "document-1", ref, _artifact("character-analysis"), _artifact("global-event-analysis"), _artifact("arc-analysis"), _artifact("global-structure"), _artifact("global-story-bible"), semantic, A6UpstreamIdentity(ref), coverage)
    assert StoryAnalysisManifest.from_dict(manifest.to_dict()) == manifest
    _validate("story-analysis-manifest.schema.json", manifest)
    assert not ({"base_url", "provider_family", "model", "timeout", "gpu", "max_concurrency"} & set(semantic.to_dict()))
    with pytest.raises(StoryAnalysisModelError): A6SemanticIdentity.from_dict({**semantic.to_dict(), "base_url": "x"})
