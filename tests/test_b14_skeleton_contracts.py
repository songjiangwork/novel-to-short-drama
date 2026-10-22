"""B1.4-A1 typed skeleton contracts: synthetic and zero-provider only."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from short_drama.adaptation import (
    AdaptedStoryBeat, AdaptedStorySkeleton, ArcCheckpoint, AudienceKnowledgeState, B14SkeletonError,
    BeatAccounting, CharacterKnowledgeState, CharacterKnowledgeTransition, ChronologyPrecedence,
    GapAnchor,
    NarrativeDependency, NarrativeOutcome, PresentationEntry, RevealTransition,
    SetupPayoffLink, SkeletonGap, WithholdingBoundary, WorldTruthState,
)
from short_drama.artifacts import ArtifactRef
from short_drama.adaptation.target import CanonicalTargetReference


def _ref(artifact_type: str, suffix: str) -> ArtifactRef:
    return ArtifactRef(artifact_type, f"synthetic.{suffix}", 1, suffix * 64)


def _skeleton() -> AdaptedStorySkeleton:
    return AdaptedStorySkeleton(
        _ref("adaptation_strategy", "a"), _ref("b13_disposition_publication", "b"),
        chronology=(ChronologyPrecedence("beat_discovery", "beat_choice"),),
        primary_presentation=(
            PresentationEntry("pe_teaser", "beat_choice", "teaser", "先展示选择的后果。"),
            PresentationEntry("pe_flashback_discovery", "beat_discovery", "flashback", "之后才以闪回展示发现过程。"),
            PresentationEntry("pe_reveal", "beat_choice", "delayed_reveal", "随后解释选择原因。"),
        ),
        outcomes=(NarrativeOutcome("outcome_key", "beat_discovery", "获得开启密室的钥匙。"),),
        dependencies=(NarrativeDependency("dep_entry", "outcome_key", "beat_choice", "选择需要先取得钥匙。"),),
        world_truth_states=(WorldTruthState("truth_identity", "identity_status", "antagonist", "对手真实身份已成立。", "beat_discovery", "B1.3 retained fact。"),),
        character_knowledge_states=(CharacterKnowledgeState("ck_lead", "adapted_char_lead", "identity_secret", "unknown", None, "故事开始时主角尚不知情。"),),
        character_knowledge_transitions=(CharacterKnowledgeTransition("ck_transition", "adapted_char_lead", "identity_secret", "unknown", "confirmed", "beat_discovery"),),
        audience_knowledge_states=(AudienceKnowledgeState("ak_audience", "identity_secret", "unknown", "观众初始不知情。"),),
        reveal_transitions=(
            RevealTransition("reveal_suspect", "identity_secret", "unknown", "suspected", "pe_teaser"),
            RevealTransition("reveal_confirm", "identity_secret", "suspected", "confirmed", "pe_flashback_discovery"),
        ),
        withholding_boundaries=(WithholdingBoundary("hold_identity", "identity_secret", "audience", None, "pe_flashback_discovery", "在正式闪回揭示前仅允许暗示。"),),
        setup_payoffs=(SetupPayoffLink("sp_key", "beat_discovery", "beat_choice", (CanonicalTargetReference("fact", "fact_000001"),), "改编层面的钥匙 setup/payoff。"),),
        arc_checkpoints=(ArcCheckpoint("arc_lead_choice", "character", "adapted_char_lead", "commitment", "从回避转为承诺行动。", "beat_choice", "主角弧线检查点。"),),
        gaps=(SkeletonGap("gap_mystery", "knowledge", "intentional_ambiguity", "身份细节暂不说明。", "保留悬念，不是缺陷。", (GapAnchor("beat", "beat_discovery"), GapAnchor("knowledge", "identity_secret"))),),
        beat_accounting=(
            BeatAccounting("beat_discovery", "directly_presented", "以紧凑一对一 presentation 呈现。"),
            BeatAccounting("beat_choice", "directly_presented", "先 teaser 后 delayed reveal，仍是同一 occurrence。"),
        ),
    )


def test_minimal_legal_skeleton_and_partial_chronology_round_trip():
    skeleton = _skeleton()
    assert AdaptedStorySkeleton.from_dict(skeleton.to_dict()) == skeleton
    assert skeleton.chronology == (ChronologyPrecedence("beat_discovery", "beat_choice"),)
    assert skeleton.primary_presentation[1].presentation_entry_id == "pe_flashback_discovery"


def test_chronology_and_presentation_are_independent_and_beat_can_recur():
    skeleton = _skeleton()
    # Story world: discovery precedes choice. Audience gets choice first, then
    # receives discovery as a flashback, without altering story-world chronology.
    assert skeleton.chronology[0].to_dict() == {"before_beat_id": "beat_discovery", "after_beat_id": "beat_choice"}
    entries = tuple(skeleton.primary_presentation)
    assert [item.adapted_story_beat_id for item in entries] == ["beat_choice", "beat_discovery", "beat_choice"]
    assert [item.semantics.value for item in entries] == ["teaser", "flashback", "delayed_reveal"]


def test_world_character_and_audience_knowledge_are_distinct():
    skeleton = _skeleton()
    assert skeleton.world_truth_states[0].assertion_zh.startswith("对手")
    assert skeleton.character_knowledge_states[0].status.value == "unknown"
    assert skeleton.character_knowledge_states[0].established_by_beat_id is None
    assert skeleton.audience_knowledge_states[0].status.value == "unknown"


def test_reveal_withholding_setup_arc_and_intentional_ambiguity_round_trip():
    skeleton = _skeleton()
    restored = AdaptedStorySkeleton.from_dict(skeleton.to_dict())
    assert [item.to_status.value for item in restored.reveal_transitions] == ["confirmed", "suspected"]
    assert restored.withholding_boundaries[0].must_not_be_confirmed_before_presentation_id == "pe_flashback_discovery"
    assert restored.setup_payoffs[0].source_provenance_refs[0].canonical_id == "fact_000001"
    assert restored.arc_checkpoints[0].trigger_beat_id == "beat_choice"
    assert restored.gaps[0].disposition.value == "intentional_ambiguity"
    assert [item.kind.value for item in restored.gaps[0].anchors] == ["beat", "knowledge"]


def test_character_learning_is_beat_anchored_before_later_audience_flashback():
    skeleton = _skeleton()
    acquired_state = CharacterKnowledgeState("ck_lead_after", "adapted_char_lead", "identity_secret", "confirmed", "beat_discovery", "主角在发现时获得确定证据。")
    skeleton = replace(skeleton, character_knowledge_states=(*skeleton.character_knowledge_states, acquired_state))
    character_transition = skeleton.character_knowledge_transitions[0]
    audience_reveal = next(item for item in skeleton.reveal_transitions if item.reveal_id == "reveal_confirm")
    assert character_transition.triggered_by_beat_id == "beat_discovery"
    assert next(item for item in skeleton.character_knowledge_states if item.state_id == "ck_lead_after").established_by_beat_id == "beat_discovery"
    assert audience_reveal.from_status.value == "suspected"
    assert audience_reveal.triggered_by_presentation_id == "pe_flashback_discovery"
    assert skeleton.primary_presentation[1].semantics.value == "flashback"
    assert AdaptedStorySkeleton.from_dict(skeleton.to_dict()) == skeleton


def test_direct_confirmation_is_valid_but_reverse_or_repeated_transitions_fail_closed():
    assert RevealTransition("direct_audience", "secret", "unknown", "confirmed", "pe_reveal").to_status.value == "confirmed"
    assert CharacterKnowledgeTransition("direct_character", "adapted_char_lead", "secret", "unknown", "confirmed", "beat_discovery").to_status.value == "confirmed"
    with pytest.raises(B14SkeletonError, match="transition"):
        RevealTransition("reverse", "secret", "confirmed", "suspected", "pe_reveal")
    with pytest.raises(B14SkeletonError, match="transition"):
        CharacterKnowledgeTransition("repeat", "adapted_char_lead", "secret", "confirmed", "confirmed", "beat_discovery")


def test_initial_world_truth_and_global_gap_are_representable_without_artificial_beat_anchor():
    initial_truth = WorldTruthState("truth_preexisting", "life_status", "mentor", "导师在故事开始时已失踪。", None, "B1.3 continuity overlay。")
    global_gap = SkeletonGap("gap_global", "motivation", "accidental", "整体动机缺口。", "B1.4 audit。")
    skeleton = replace(_skeleton(), world_truth_states=(_skeleton().world_truth_states[0], initial_truth), gaps=(global_gap,))
    restored = AdaptedStorySkeleton.from_dict(skeleton.to_dict())
    assert next(item for item in restored.world_truth_states if item.state_id == "truth_preexisting").established_by_beat_id is None
    assert restored.gaps[0].anchors == ()


def test_gap_anchors_are_typed_round_trip_and_duplicate_anchors_fail_closed():
    skeleton = _skeleton()
    assert AdaptedStorySkeleton.from_dict(skeleton.to_dict()).gaps[0] == skeleton.gaps[0]
    with pytest.raises(B14SkeletonError, match="unique GapAnchor"):
        replace(skeleton.gaps[0], anchors=(GapAnchor("beat", "beat_discovery"), GapAnchor("beat", "beat_discovery")))


def test_unordered_collections_normalize_without_reordering_primary_presentation():
    skeleton = _skeleton()
    reordered = replace(
        skeleton,
        reveal_transitions=tuple(reversed(skeleton.reveal_transitions)),
        beat_accounting=tuple(reversed(skeleton.beat_accounting)),
    )
    assert reordered.to_dict() == skeleton.to_dict()
    assert [item.presentation_entry_id for item in reordered.primary_presentation] == ["pe_teaser", "pe_flashback_discovery", "pe_reveal"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.__setitem__("schema_version", 99), "schema_version"),
        (lambda value: value.__setitem__("schema_version", True), "schema_version"),
        (lambda value: value["world_truth_states"][0].__setitem__("kind", "costume"), "permitted"),
        (lambda value: value["primary_presentation"].append(copy.deepcopy(value["primary_presentation"][0])), "duplicate presentation"),
        (lambda value: value["chronology"][0].__setitem__("before_beat_id", "bad id"), "safe identifier"),
        (lambda value: value["reveal_transitions"][0].__setitem__("from_status", "confirmed"), "transition"),
    ],
)
def test_invalid_schema_enum_malformed_field_and_duplicate_identity_fail_closed(mutate, match):
    value = copy.deepcopy(_skeleton().to_dict())
    mutate(value)
    with pytest.raises(B14SkeletonError, match=match):
        AdaptedStorySkeleton.from_dict(value)


def test_canonical_round_trip_stable_hash_and_tampering_fail_closed():
    skeleton = _skeleton()
    assert AdaptedStorySkeleton.from_dict(skeleton.to_dict()).content_hash() == skeleton.content_hash()
    tampered = copy.deepcopy(skeleton.to_dict())
    tampered["strategy_ref"]["artifact_type"] = "current_strategy"
    with pytest.raises(B14SkeletonError, match="adaptation_strategy"):
        AdaptedStorySkeleton.from_dict(tampered)
    malformed = copy.deepcopy(skeleton.to_dict())
    malformed["gaps"][0]["unexpected"] = True
    with pytest.raises(B14SkeletonError, match="exactly"):
        AdaptedStorySkeleton.from_dict(malformed)


def test_local_invented_beat_registry_is_explicit_and_only_accepts_b13_invented_beats():
    skeleton = _skeleton()
    anchor = CanonicalTargetReference("fact", "fact_000001")
    bridge = AdaptedStoryBeat(
        "b14-bridge", "以一段行动衔接两次发现。", "adaptation_invented", "compatible",
        (), (), (anchor,), "bridge", "补足连续性。", "B1.4-local bridge registry。",
    )
    with_bridge = replace(skeleton, local_invented_beats=(bridge,))
    assert AdaptedStorySkeleton.from_dict(with_bridge.to_dict()).local_invented_beats == (bridge,)
    source_beat = AdaptedStoryBeat("source", "原始事件。", "source_direct", "compatible", (CanonicalTargetReference("event", "evt_000001"),))
    with pytest.raises(B14SkeletonError, match="adaptation_invented"):
        replace(skeleton, local_invented_beats=(source_beat,))


def test_duplicate_state_identity_across_truth_and_knowledge_domains_fails_closed():
    skeleton = _skeleton()
    duplicate = replace(skeleton.audience_knowledge_states[0], state_id="truth_identity")
    with pytest.raises(B14SkeletonError, match="must not overlap"):
        replace(skeleton, audience_knowledge_states=(duplicate,))
