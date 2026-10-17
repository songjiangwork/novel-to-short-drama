import json

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from short_drama.adaptation import (
    AdaptationSourceSnapshot, AdaptationStrategy, AdaptationStrategyError,
    CanonicalTargetReference, CharacterStrategy, DeliberateDivergence,
    PlotlineStrategy, PriorityEntry, RelationshipStrategy, RevealStrategy,
    ResolvedAdaptationTarget, SnapshotLocalPlotline,
    B12Packet, prepare_b12_packet,
)
from short_drama.artifacts import ArtifactRef
from short_drama.io import load_json
from short_drama.paths import SCHEMAS_DIR


H = "a" * 64


def _ref(kind, artifact_id):
    return ArtifactRef(kind, artifact_id, 1, H)


def _target():
    refs = (
        CanonicalTargetReference("character", "char_0001"),
        CanonicalTargetReference("event", "evt_000001"),
        CanonicalTargetReference("relationship", "rel_000001"),
        CanonicalTargetReference("reveal", "reveal_000001"),
    )
    snapshot = AdaptationSourceSnapshot(
        _ref("story_analysis_manifest", "p.a6.manifest"), _ref("consolidation_manifest", "p.a5.manifest"),
        _ref("entity_map", "p.a4.map"), _ref("canonical_character_registry", "p.a4.characters"),
        _ref("canonical_location_registry", "p.a4.locations"), _ref("canonical_fact_set", "p.a5.facts"),
        _ref("canonical_event_set", "p.a5.events"), _ref("canonical_relationship_set", "p.a5.relationships"),
        _ref("arc_analysis", "p.a6.arcs"), refs,
    )
    return ResolvedAdaptationTarget("b12-test", snapshot)


def _strategy(plotlines=()):
    target = _target()
    refs = {x.kind: x for x in target.source_snapshot.canonical_references}
    return AdaptationStrategy(
        target, "聚焦主线。", "持续的危机承诺。", "主角能否逃离？",
        tuple(PlotlineStrategy(x.handle, "preserve", "保留核心。") for x in plotlines),
        (CharacterStrategy(refs["character"], "retain", "保留。"),),
        (RelationshipStrategy(refs["relationship"], "compress", "压缩。"),),
        (RevealStrategy(refs["reveal"], "delay", "延后。"),),
        "最终面对核心选择。", (refs["event"],), "source_ending_reframed", "重构结尾视角。",
        (PriorityEntry(refs["event"], "critical"),),
        (DeliberateDivergence("结尾重构。", (refs["event"],), "user direction"),), (),
    )


def test_strategy_round_trip_schema_hash_and_distinct_global_fields():
    plotlines = (
        SnapshotLocalPlotline("a6pl_001_abcdef0123456789", H, 1, "subplot"),
        SnapshotLocalPlotline("a6pl_000_abcdef0123456789", H, 0, "main"),
    )
    strategy = _strategy(plotlines)
    assert strategy.adaptation_thesis_zh != strategy.dramatic_promise_zh
    assert strategy.primary_dramatic_question_zh != strategy.dramatic_promise_zh
    assert AdaptationStrategy.from_dict(strategy.to_dict()) == strategy
    reversed_strategy = _strategy(tuple(reversed(plotlines)))
    assert reversed_strategy.content_hash() == strategy.content_hash()
    schema = load_json(SCHEMAS_DIR / "adaptation-strategy.schema.json")
    resolved = load_json(SCHEMAS_DIR / "resolved-adaptation-target.schema.json")
    raw = load_json(SCHEMAS_DIR / "adaptation-target.schema.json")
    registry = (Registry().with_resource(raw["$id"], Resource.from_contents(raw))
                .with_resource(resolved["$id"], Resource.from_contents(resolved)))
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema, registry=registry).iter_errors(strategy.to_dict())) == []
    malformed = json.loads(json.dumps(strategy.to_dict()))
    malformed["ending_disposition"] = "not-an-ending"
    assert list(Draft202012Validator(schema, registry=registry).iter_errors(malformed))


def test_snapshot_local_plotline_merge_validation_rejects_unknown_or_self():
    plotlines = (
        SnapshotLocalPlotline("a6pl_000_abcdef0123456789", H, 0, "main"),
        SnapshotLocalPlotline("a6pl_001_abcdef0123456789", H, 1, "subplot"),
    )
    strategy = _strategy(plotlines)
    strategy.validate_source_membership(plotlines)
    bad = _strategy(plotlines)
    object.__setattr__(bad, "plotline_strategies", (PlotlineStrategy(plotlines[0].handle, "merge_into", "合并。", "a6pl_missing"), PlotlineStrategy(plotlines[1].handle, "preserve", "保留。")))
    with pytest.raises(AdaptationStrategyError, match="merge_into"):
        bad.validate_source_membership(plotlines)
    self_merge = _strategy(plotlines)
    object.__setattr__(self_merge, "plotline_strategies", (PlotlineStrategy(plotlines[0].handle, "merge_into", "合并。", plotlines[0].handle), PlotlineStrategy(plotlines[1].handle, "preserve", "保留。")))
    with pytest.raises(AdaptationStrategyError, match="merge_into"):
        self_merge.validate_source_membership(plotlines)


def test_unknown_source_reference_is_rejected():
    strategy = _strategy()
    object.__setattr__(strategy, "series_climax_source_refs", (CanonicalTargetReference("event", "evt_999999"),))
    with pytest.raises(AdaptationStrategyError, match="absent"):
        strategy.validate_source_membership()


def _published_a6(tmp_path):
    """Reuse the established A5/A6 test fixture; provider-free by construction."""
    from test_story_a6b_planning import _snapshot
    from short_drama.llm import PromptRegistry, load_semantic_profile
    from short_drama.paths import REPO_ROOT
    from short_drama.story import (
        ArcAnalysis, CharacterAnalysis, CharacterAnalysisSet, EvidenceBackedInterpretation,
        GlobalEventAnalysis, GlobalStoryBible, GlobalStructure, PlotWindowAnalysis,
        StoryAnalysisPersistenceService, build_story_analysis_plan_from_profile,
        load_story_analysis_profile,
    )
    def interp(): return EvidenceBackedInterpretation(text_zh="测试。", evidence_mode="explicit")
    tree, _planning, _publication, snap = _snapshot(tmp_path)
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    plan = build_story_analysis_plan_from_profile(snap, profile)
    chars = CharacterAnalysisSet(1, tuple(CharacterAnalysis(p.character_ref, interp(), (), (), (), (), (), (), interp(), ()) for p in plan.character_packages))
    events = GlobalEventAnalysis(1, plan.plan_hash, tuple(PlotWindowAnalysis(w.window_id, w.window_ordinal, w.owned_event_ids, w.context_event_ids, interp()) for w in plan.windows), (), (("canonical_event_count", len(plan.event_stream)), ("event_importance_count", 0), ("plot_window_count", len(plan.windows))))
    publication = StoryAnalysisPersistenceService(tree.store, tree.pointers).publish_validated(
        project_id=snap.consolidation_manifest.project_id, document_id=snap.consolidation_manifest.document_id,
        profile=profile, semantic_profile=load_semantic_profile(REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml"),
        prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"), snapshot=snap, plan=plan, characters=chars,
        events=events, arcs=ArcAnalysis(1, (), (), (), ()),
        structure=GlobalStructure(1, interp(), (), interp(), (), interp(), (), (), (), (), ()),
        bible=GlobalStoryBible(1, interp(), interp(), "测试", "测试", interp(), (), (), (), (), (), interp(), interp(), (("canonical_character_count", len(chars.analyses)), ("canonical_event_count", len(plan.event_stream)))),
    )
    return tree, publication


def test_zero_provider_packet_is_complete_deterministic_and_read_only(tmp_path):
    from short_drama.adaptation import AdaptationTarget, AdaptationSourceSnapshot, resolve_adaptation_target
    tree, publication = _published_a6(tmp_path)
    before = sorted(path.relative_to(tree.store.root) for path in tree.store.root.rglob("*"))
    source = AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, publication.story_analysis_manifest_ref)
    target = resolve_adaptation_target(AdaptationTarget("packet-test"), source_snapshot=source)
    first = prepare_b12_packet(tree.store, tree.pointers, target)
    second = prepare_b12_packet(tree.store, tree.pointers, target)
    after = sorted(path.relative_to(tree.store.root) for path in tree.store.root.rglob("*"))
    assert before == after
    assert first.measurement.canonical_event_count == len(first.packet.canonical_event_index)
    assert {item["event_id"] for item in first.packet.canonical_event_index} == {
        ref.canonical_id for ref in source.canonical_references if ref.kind == "event"
    }
    assert {item["relationship_id"] for item in first.packet.canonical_relationship_index} == {
        ref.canonical_id for ref in source.canonical_references if ref.kind == "relationship"
    }
    assert [(item["narrative_order"], item["event_id"]) for item in first.packet.canonical_event_index] == sorted(
        (item["narrative_order"], item["event_id"]) for item in first.packet.canonical_event_index
    )
    assert first.measurement.packet_hash == second.measurement.packet_hash
    assert first.semantic_input_identity == second.semantic_input_identity
    assert first.measurement.effective_b1_direction_hash == second.measurement.effective_b1_direction_hash


def test_b4_direction_does_not_change_b12_identity_but_b1_direction_does(tmp_path):
    from short_drama.adaptation import AdaptationTarget, AdaptationSourceSnapshot, resolve_adaptation_target
    from short_drama.guidance import (
        DirectionCategory, DirectionOrigin, DirectionStage, DirectionStrength, TargetScope,
        UserDirection, UserDirectionSet, persist_user_direction,
    )
    tree, publication = _published_a6(tmp_path)
    source = AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, publication.story_analysis_manifest_ref)
    target = resolve_adaptation_target(AdaptationTarget("direction-packet"), source_snapshot=source)
    project_id = tree.store.get_ref(publication.story_analysis_manifest_ref).payload["project_id"]
    empty = UserDirectionSet(project_id, "directions", 1, ())
    before = prepare_b12_packet(tree.store, tree.pointers, target, empty)
    camera = UserDirection("camera", project_id, 1, DirectionOrigin.USER, DirectionCategory.CAMERA,
        DirectionStrength.PREFER, DirectionStage.B4_SHOT, TargetScope.PROJECT, (), "多用近景。")
    camera_ref = persist_user_direction(tree.store, camera)
    b4_only = prepare_b12_packet(tree.store, tree.pointers, target, UserDirectionSet(project_id, "directions", 2, (camera_ref,)))
    narrative = UserDirection("narrative", project_id, 1, DirectionOrigin.USER, DirectionCategory.CHARACTER_NARRATIVE,
        DirectionStrength.PREFER, DirectionStage.B1_ADAPTATION, TargetScope.PROJECT, (), "强化主角能动性。")
    narrative_ref = persist_user_direction(tree.store, narrative)
    b1 = prepare_b12_packet(tree.store, tree.pointers, target, UserDirectionSet(project_id, "directions", 3, (camera_ref, narrative_ref)))
    assert before.measurement.effective_b1_direction_hash == b4_only.measurement.effective_b1_direction_hash
    assert before.semantic_input_identity == b4_only.semantic_input_identity
    assert b1.measurement.effective_b1_direction_hash != before.measurement.effective_b1_direction_hash
    assert b1.semantic_input_identity != before.semantic_input_identity


def _context_strategy(target, plotlines, *, merge_into=None):
    event = next(ref for ref in target.source_snapshot.canonical_references if ref.kind == "event")
    strategies = tuple(
        PlotlineStrategy(item.handle, "merge_into" if merge_into and item.handle == merge_into[0] else "preserve", "策略。", merge_into[1] if merge_into and item.handle == merge_into[0] else None)
        for item in plotlines
    )
    return AdaptationStrategy(target, "论文。", "承诺。", "问题？", strategies, (), (), (), "高潮。", (event,), "source_ending_retained", "保留结尾。", ())


def test_preparation_is_mandatory_authoritative_plotline_acceptance_seam(tmp_path):
    from short_drama.adaptation import AdaptationTarget, AdaptationSourceSnapshot, resolve_adaptation_target
    tree, publication = _published_a6(tmp_path)
    source = AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, publication.story_analysis_manifest_ref)
    target = resolve_adaptation_target(AdaptationTarget("strategy-context"), source_snapshot=source)
    preparation = prepare_b12_packet(tree.store, tree.pointers, target)
    valid = _context_strategy(target, preparation.packet.plotlines)
    preparation.validate_strategy(AdaptationStrategy.from_dict(valid.to_dict()))
    with pytest.raises(AdaptationStrategyError, match="cover exactly"):
        preparation.validate_strategy(_context_strategy(target, ()))
    forged = SnapshotLocalPlotline("a6pl_999_forged000000000", source.story_analysis_manifest_ref.content_hash, 999, "subplot")
    with pytest.raises(AdaptationStrategyError, match="cover exactly"):
        preparation.validate_strategy(_context_strategy(target, (forged,)))
    handle = preparation.packet.plotlines[0].handle
    with pytest.raises(AdaptationStrategyError, match="merge_into"):
        preparation.validate_strategy(_context_strategy(target, preparation.packet.plotlines, merge_into=(handle, "a6pl_missing")))
    with pytest.raises(AdaptationStrategyError, match="exact target A6 snapshot"):
        valid.validate_source_membership((SnapshotLocalPlotline(handle, "b" * 64, 0, "main"),))
    with pytest.raises(AdaptationStrategyError, match="exact A6 snapshot"):
        B12Packet.build(source, target, {}, (SnapshotLocalPlotline(handle, "b" * 64, 0, "main"),), {}, {}, (), {}, {},
            ({"event_id": ref.canonical_id, "narrative_order": 1} for ref in source.canonical_references if ref.kind == "event"), (), (), ())


def test_packet_bytes_hash_and_identity_survive_public_component_mutation(tmp_path):
    from short_drama.adaptation import AdaptationTarget, AdaptationSourceSnapshot, resolve_adaptation_target
    tree, publication = _published_a6(tmp_path)
    source = AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, publication.story_analysis_manifest_ref)
    preparation = prepare_b12_packet(tree.store, tree.pointers, resolve_adaptation_target(AdaptationTarget("immutable-packet"), source_snapshot=source))
    before = preparation.packet.canonical_bytes()
    inspected = preparation.packet.story_bible
    inspected["genre"] = "被篡改"
    rendered_copy = preparation.packet.to_dict()
    rendered_copy["global_structure"]["schema_version"] = 999
    assert preparation.packet.canonical_bytes() == before
    assert preparation.packet.content_hash() == preparation.measurement.packet_hash
    assert preparation.packet.story_bible["genre"] != "被篡改"


def test_packet_requires_complete_sorted_compact_event_and_relationship_indexes():
    base = _target().source_snapshot
    snapshot = AdaptationSourceSnapshot(
        base.story_analysis_manifest_ref, base.consolidation_manifest_ref, base.entity_map_ref,
        base.canonical_character_registry_ref, base.canonical_location_registry_ref,
        base.canonical_fact_set_ref, base.canonical_event_set_ref, base.canonical_relationship_set_ref,
        base.arc_analysis_ref, (CanonicalTargetReference("event", "evt_000001"), CanonicalTargetReference("event", "evt_000002"), CanonicalTargetReference("relationship", "rel_000001")),
    )
    target = ResolvedAdaptationTarget("ordered-packet", snapshot)
    with pytest.raises(AdaptationStrategyError, match="narrative order"):
        B12Packet.build(snapshot, target, {}, (), {}, {}, (), {}, {},
            ({"event_id": "evt_000002", "narrative_order": 2}, {"event_id": "evt_000001", "narrative_order": 1}),
            ({"relationship_id": "rel_000001"},), (), ())
    packet = B12Packet.build(snapshot, target, {}, (), {}, {}, (), {}, {},
        ({"event_id": "evt_000001", "narrative_order": 1}, {"event_id": "evt_000002", "narrative_order": 2}),
        ({"relationship_id": "rel_000001", "relationship_type_zh": "关系"},), (), ())
    assert [entry["event_id"] for entry in packet.canonical_event_index] == ["evt_000001", "evt_000002"]
    assert packet.canonical_relationship_index[0]["relationship_id"] == "rel_000001"
