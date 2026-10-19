import copy
from dataclasses import replace

import pytest

from short_drama.adaptation import (
    AdaptationMapping, AdaptationSourceSnapshot, AdaptedCharacter,
    AdaptedContinuityOverlay, AdaptedEndpoint, AdaptedFactDivergence,
    AdaptedStoryBeat, AnonymousRole, B13DispositionError, B13DispositionPlan,
    BeatParticipant, CanonicalTargetReference, CharacterFunctionProvenance,
    ContinuityStoryState, EventDisposition, RelationshipArcRequirement,
)
from short_drama.artifacts import ArtifactRef


H = "b" * 64


def _ref(kind, artifact_id):
    return ArtifactRef(kind, artifact_id, 1, H)


def _source_snapshot():
    refs = (
        CanonicalTargetReference("character", "char_0001"),
        CanonicalTargetReference("character", "char_0002"),
        CanonicalTargetReference("event", "evt_000001"),
        CanonicalTargetReference("event", "evt_000002"),
        CanonicalTargetReference("event", "evt_000003"),
        CanonicalTargetReference("event", "evt_000004"),
        CanonicalTargetReference("fact", "fact_000001"),
        CanonicalTargetReference("relationship", "rel_000001"),
    )
    return AdaptationSourceSnapshot(
        _ref("story_analysis_manifest", "p.a6.manifest"), _ref("consolidation_manifest", "p.a5.manifest"),
        _ref("entity_map", "p.a4.map"), _ref("canonical_character_registry", "p.a4.characters"),
        _ref("canonical_location_registry", "p.a4.locations"), _ref("canonical_fact_set", "p.a5.facts"),
        _ref("canonical_event_set", "p.a5.events"), _ref("canonical_relationship_set", "p.a5.relationships"),
        _ref("arc_analysis", "p.a6.arcs"), refs,
    )


def _plan():
    snapshot = _source_snapshot()
    refs = {(item.kind, item.canonical_id): item for item in snapshot.canonical_references}
    char = AdaptedCharacter(
        "lead", "林舟", "source_composite",
        (CharacterFunctionProvenance(refs["character", "char_0001"], "承担行动主视角。", "改编策略。"),
         CharacterFunctionProvenance(refs["character", "char_0002"], "承担道德压力。", "改编策略。")),
    )
    role = AnonymousRole("guard-at-gate", "guard", "阻拦主角进入。")
    direct = AdaptedStoryBeat(
        "inciting", "林舟得知危机。", "source_direct", "compatible",
        (refs["event", "evt_000001"],),
        (BeatParticipant("adapted_character", char.adapted_character_id),),
    )
    composite = AdaptedStoryBeat(
        "confrontation", "两次冲突压缩为一次正面交锋。", "source_composite", "deliberate_divergence",
        (refs["event", "evt_000002"], refs["event", "evt_000003"]),
        (BeatParticipant("adapted_character", char.adapted_character_id), BeatParticipant("anonymous_role", role.anonymous_role_id)),
    )
    invented = AdaptedStoryBeat(
        "bridge", "以电话场景外化选择压力。", "adaptation_invented", "compatible", (),
        (BeatParticipant("adapted_character", char.adapted_character_id),),
        (refs["event", "evt_000001"], refs["fact", "fact_000001"]), "externalization", "把内心冲突变成可拍动作。", "B1.3 改编理由。",
    )
    def event_mapping(event_id, beat, transformed):
        return AdaptationMapping(refs["event", event_id], AdaptedEndpoint("story_beat", beat.adapted_story_beat_id), "primary", "保留因果功能。", transformed, "B1.3 语义决策。")
    mappings = (
        event_mapping("evt_000001", direct, "直接呈现。"),
        event_mapping("evt_000002", composite, "合并冲突过程。"),
        event_mapping("evt_000003", composite, "合并冲突过程。"),
        AdaptationMapping(refs["character", "char_0001"], AdaptedEndpoint("character", char.adapted_character_id), "primary", "保留行动功能。", "转移至复合角色。", "B1.3 角色映射。"),
    )
    overlay = AdaptedContinuityOverlay(
        (AdaptedFactDivergence(refs["fact", "fact_000001"], "危机提前一天发生。", "明确的有意偏离。"),),
        (RelationshipArcRequirement(refs["relationship", "rel_000001"], "对立必须升级后才和解。", "连续性约束。"),),
        (ContinuityStoryState(composite.adapted_story_beat_id, "林舟已公开承诺。", "衔接要求。"),),
    )
    return B13DispositionPlan(
        snapshot,
        (EventDisposition(refs["event", "evt_000001"], "keep", "保留开端。"),
         EventDisposition(refs["event", "evt_000002"], "merge_member", "并入正面交锋。"),
         EventDisposition(refs["event", "evt_000003"], "merge_member", "并入正面交锋。"),
         EventDisposition(refs["event", "evt_000004"], "omit", "不再发生。")),
        (char,), (role,), (direct, composite, invented), mappings, overlay,
    )


def test_b13_round_trip_stable_identity_and_ordering():
    plan = _plan()
    assert B13DispositionPlan.from_dict(plan.to_dict()) == plan
    assert B13DispositionPlan.from_dict(plan.to_dict()).content_hash() == plan.content_hash()
    reversed_plan = replace(
        plan,
        event_dispositions=tuple(reversed(plan.event_dispositions)),
        adapted_story_beats=tuple(reversed(plan.adapted_story_beats)),
        adaptation_mappings=tuple(reversed(plan.adaptation_mappings)),
    )
    assert reversed_plan.to_dict() == plan.to_dict()
    assert reversed_plan.content_hash() == plan.content_hash()
    composite = plan.adapted_characters[0]
    assert composite.origin.value == "source_composite"
    assert composite.adapted_character_id.startswith("adapted_char_")
    assert composite.adapted_character_id not in {"char_0001", "char_0002"}


def test_b13_rejects_tampered_derived_ids_and_unknown_sources():
    plan = _plan()
    tampered = copy.deepcopy(plan.to_dict())
    tampered["adapted_story_beats"][0]["adapted_story_beat_id"] = "adapted_beat_tampered"
    with pytest.raises(B13DispositionError, match="deterministic identity"):
        B13DispositionPlan.from_dict(tampered)
    unknown = CanonicalTargetReference("event", "evt_999999")
    bad_mapping = AdaptationMapping(
        unknown, AdaptedEndpoint("story_beat", plan.adapted_story_beats[0].adapted_story_beat_id),
        "primary", "保留。", "变形。", "来源。",
    )
    with pytest.raises(B13DispositionError, match="absent from exact source snapshot"):
        replace(plan, adaptation_mappings=(*plan.adaptation_mappings, bad_mapping))


def test_b13_requires_exhaustive_disposition_and_mapping_integrity():
    plan = _plan()
    with pytest.raises(B13DispositionError, match="cover exactly"):
        replace(plan, event_dispositions=plan.event_dispositions[:-1])
    mappings_without_keep = tuple(item for item in plan.adaptation_mappings if item.source_ref.canonical_id != "evt_000001")
    with pytest.raises(B13DispositionError, match="requires an active mapping"):
        replace(plan, adaptation_mappings=mappings_without_keep)
    omitted = next(item for item in plan.event_dispositions if item.source_event_ref.canonical_id == "evt_000004")
    omitted_beat = AdaptedStoryBeat("omitted-event", "不应被采用的事件。", "source_direct", "compatible", (omitted.source_event_ref,))
    extra_active = AdaptationMapping(
        omitted.source_event_ref, AdaptedEndpoint("story_beat", omitted_beat.adapted_story_beat_id),
        "supporting", "保留。", "借用。", "错误输入。",
    )
    with pytest.raises(B13DispositionError, match="omit event forbids"):
        replace(plan, adapted_story_beats=(*plan.adapted_story_beats, omitted_beat), adaptation_mappings=(*plan.adaptation_mappings, extra_active))


def test_b13_merge_requires_multi_source_composite_and_participants_resolve():
    plan = _plan()
    merge_event = next(item for item in plan.event_dispositions if item.source_event_ref.canonical_id == "evt_000002")
    wrong_beat = AdaptedStoryBeat("wrong-merge", "未合并的错误版本。", "source_direct", "compatible", (merge_event.source_event_ref,))
    wrong_target = AdaptationMapping(
        merge_event.source_event_ref, AdaptedEndpoint("story_beat", wrong_beat.adapted_story_beat_id),
        "primary", "保留。", "错误地未合并。", "错误输入。",
    )
    with pytest.raises(B13DispositionError, match="multi-source source_composite"):
        replace(plan, adapted_story_beats=(*plan.adapted_story_beats, wrong_beat), adaptation_mappings=tuple(item for item in plan.adaptation_mappings if item.source_ref != merge_event.source_event_ref) + (wrong_target,))
    broken = copy.deepcopy(plan)
    object.__setattr__(broken.adapted_story_beats[0], "participants", (BeatParticipant("adapted_character", "adapted_char_missing"),))
    with pytest.raises(B13DispositionError, match="does not resolve"):
        broken.validate()


def test_b13_rejects_invalid_anonymous_role_lifetime_and_overlay_refs():
    plan = _plan()
    role = plan.anonymous_roles[0]
    direct = next(item for item in plan.adapted_story_beats if item.origin.value == "source_direct")
    duplicated_direct = replace(direct, participants=(*direct.participants, BeatParticipant("anonymous_role", role.anonymous_role_id)))
    with pytest.raises(B13DispositionError, match="exactly one material beat"):
        replace(plan, adapted_story_beats=tuple(duplicated_direct if item.adapted_story_beat_id == direct.adapted_story_beat_id else item for item in plan.adapted_story_beats))
    bad_overlay = replace(plan.continuity_overlay, story_state_requirements=(ContinuityStoryState("adapted_beat_missing", "状态。", "来源。"),))
    with pytest.raises(B13DispositionError, match="unknown adapted beat"):
        replace(plan, continuity_overlay=bad_overlay)
