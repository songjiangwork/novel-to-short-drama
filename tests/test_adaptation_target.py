import json
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from short_drama.artifacts import ArtifactRef, FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.io import load_json
from short_drama.paths import SCHEMAS_DIR
from short_drama.adaptation.target import _canonical_references_from_a4

from short_drama.adaptation import (
    AdaptationTarget,
    AdaptationTargetDefaults,
    AdaptationTargetError,
    AdaptationTargetOverrides,
    AdaptationSourceSnapshot,
    CanonicalTargetReference,
    ConstraintKind,
    ConstraintStrength,
    DurationCountFeasibility,
    FidelityProfile,
    NumericConstraint,
    Provenance,
    StableIdIndex,
    TargetElement,
    TargetValue,
    UnresolvedTargetReference,
    resolve_adaptation_target,
)

H = "a" * 64


def artifact_ref(artifact_type, artifact_id, revision=1):
    return ArtifactRef(artifact_type, artifact_id, revision, H)


def source_snapshot(*references, revision=1):
    if not references:
        references = (
            CanonicalTargetReference("character", "char_0001"),
            CanonicalTargetReference("event", "evt_000001"),
            CanonicalTargetReference("fact", "fact_000001"),
            CanonicalTargetReference("relationship", "rel_000001"),
        )
    return AdaptationSourceSnapshot(
        story_analysis_manifest_ref=artifact_ref("story_analysis_manifest", "project.a6.manifest", revision),
        consolidation_manifest_ref=artifact_ref("consolidation_manifest", "project.a5.manifest", revision),
        entity_map_ref=artifact_ref("entity_map", "project.a4.entity-map", revision),
        canonical_character_registry_ref=artifact_ref("canonical_character_registry", "project.a4.characters", revision),
        canonical_location_registry_ref=artifact_ref("canonical_location_registry", "project.a4.locations", revision),
        canonical_fact_set_ref=artifact_ref("canonical_fact_set", "project.a5.facts", revision),
        canonical_event_set_ref=artifact_ref("canonical_event_set", "project.a5.events", revision),
        canonical_relationship_set_ref=artifact_ref("canonical_relationship_set", "project.a5.relationships", revision),
        arc_analysis_ref=artifact_ref("arc_analysis", "project.a6.arcs", revision),
        canonical_references=references,
    )


def value(value, provenance=Provenance.USER):
    return TargetValue(value, provenance)


def constraint(kind, strength, **kwargs):
    return NumericConstraint(kind, strength, **kwargs)


def test_resolution_is_field_deterministic_and_preserves_provenance():
    defaults = AdaptationTargetDefaults(
        "vertical-defaults",
        narrative_format=value("short drama", Provenance.PROFILE_DEFAULT),
        fidelity=value(FidelityProfile.HIGH, Provenance.PROFILE_DEFAULT),
        pacing=value("measured", Provenance.PROFILE_DEFAULT),
        compression=value("moderate", Provenance.PROFILE_DEFAULT),
        working_language=value("zh-CN", Provenance.PROFILE_DEFAULT),
        episode_count_constraints=value((constraint("target", "soft", value=12),), Provenance.PROFILE_DEFAULT),
    )
    raw = AdaptationTarget(
        "alice-adaptation",
        fidelity=value("balanced"),
        pacing=value("fast"),
        required_elements=value((TargetElement(CanonicalTargetReference("event", "evt_000001")),)),
    )
    overrides = AdaptationTargetOverrides(
        "alice-override-1",
        pacing=value("deliberate"),
        compression=value("high"),
    )

    resolved = resolve_adaptation_target(raw, defaults, overrides, source_snapshot=source_snapshot())

    assert resolved.narrative_format.value == "short drama"
    assert resolved.narrative_format.provenance is Provenance.PROFILE_DEFAULT
    assert resolved.fidelity.value is FidelityProfile.BALANCED
    assert resolved.fidelity.provenance is Provenance.USER
    assert resolved.pacing.value == "deliberate"
    assert resolved.pacing.provenance is Provenance.USER
    assert resolved.compression.value == "high"
    assert resolved.compression.provenance is Provenance.USER
    assert resolved.episode_count_constraints.provenance is Provenance.PROFILE_DEFAULT
    # The three policies are separate fields; no implication changes one from another.
    assert (resolved.fidelity.value, resolved.pacing.value, resolved.compression.value) == (
        FidelityProfile.BALANCED, "deliberate", "high"
    )


def test_strict_round_trip_serialization_and_schema_validation():
    raw = AdaptationTarget(
        "round-trip",
        working_language=value("zh-CN"),
        primary_focus=value((
            CanonicalTargetReference("character", "char_0001"),
            UnresolvedTargetReference("the recurring clock motif"),
        )),
        total_duration_constraints=value((constraint("range", "hard", minimum=150, maximum=210),)),
        per_episode_duration_constraints=value((constraint("target", "soft", value=90),)),
        episode_count_constraints=value((constraint("derived", "soft", derivation="B1.5 episode partitioning"),)),
    )
    resolved = resolve_adaptation_target(raw, source_snapshot=source_snapshot())
    data = resolved.to_dict()

    assert type(resolved).from_dict(data) == resolved
    assert type(resolved).from_dict(data).content_hash() == resolved.content_hash()
    raw_schema = load_json(SCHEMAS_DIR / "adaptation-target.schema.json")
    resolved_schema = load_json(SCHEMAS_DIR / "resolved-adaptation-target.schema.json")
    registry = Registry().with_resource(raw_schema["$id"], Resource.from_contents(raw_schema))
    Draft202012Validator.check_schema(raw_schema)
    Draft202012Validator.check_schema(resolved_schema)
    assert list(Draft202012Validator(raw_schema).iter_errors(raw.to_dict())) == []
    assert list(Draft202012Validator(raw_schema).iter_errors(data)) == []
    assert list(Draft202012Validator(resolved_schema, registry=registry).iter_errors(data)) == []
    malformed_duration = json.loads(json.dumps(data))
    malformed_duration["total_duration_constraints"]["value"][0]["value"] = 90.5
    assert list(Draft202012Validator(resolved_schema, registry=registry).iter_errors(malformed_duration))
    malformed_count = json.loads(json.dumps(data))
    malformed_count["episode_count_constraints"] = {
        "value": [{"kind": "fixed", "strength": "hard", "value": True, "minimum": None, "maximum": None, "derivation": None}],
        "provenance": "user",
    }
    assert list(Draft202012Validator(resolved_schema, registry=registry).iter_errors(malformed_count))
    bad = dict(data)
    bad["unknown"] = True
    with pytest.raises(AdaptationTargetError, match="exactly"):
        type(resolved).from_dict(bad)


def test_all_constraint_kinds_and_strengths_are_orthogonal():
    target = AdaptationTarget(
        "constraint-kinds",
        total_duration_constraints=value((
            constraint(ConstraintKind.FIXED, ConstraintStrength.HARD, value=180),
            constraint(ConstraintKind.TARGET, ConstraintStrength.SOFT, value=175),
        )),
        per_episode_duration_constraints=value((
            constraint(ConstraintKind.RANGE, ConstraintStrength.HARD, maximum=100),
            constraint(ConstraintKind.TARGET, ConstraintStrength.SOFT, value=90),
        )),
        episode_count_constraints=value((
            constraint(ConstraintKind.DERIVED, ConstraintStrength.HARD, derivation="resolve after B1.5 partitioning"),
        )),
    )
    resolved = resolve_adaptation_target(target, source_snapshot=source_snapshot())
    assert [item.kind for item in resolved.total_duration_constraints.value] == [
        ConstraintKind.FIXED, ConstraintKind.TARGET
    ]
    assert resolved.per_episode_duration_constraints.value[0].maximum == 100
    assert resolved.episode_count_constraints.value[0].strength is ConstraintStrength.HARD


@pytest.mark.parametrize(
    "total,per_episode,count",
    [
        (1200, 90, 20),  # Explicit frozen-document example: 1800 != 1200.
        (100, 60, 2),
    ],
)
def test_hard_duration_count_contradictions_fail_closed(total, per_episode, count):
    raw = AdaptationTarget(
        "bad-budget",
        total_duration_constraints=value((constraint("fixed", "hard", value=total),)),
        per_episode_duration_constraints=value((constraint("fixed", "hard", value=per_episode),)),
        episode_count_constraints=value((constraint("fixed", "hard", value=count),)),
    )
    with pytest.raises(AdaptationTargetError, match="contradictory"):
        resolve_adaptation_target(raw, source_snapshot=source_snapshot())


def test_hard_constraint_self_conflict_and_element_conflict_fail_closed():
    with pytest.raises(AdaptationTargetError, match="no feasible"):
        resolve_adaptation_target(AdaptationTarget(
            "no-count",
            episode_count_constraints=value((
                constraint("fixed", "hard", value=2),
                constraint("range", "hard", minimum=3, maximum=4),
            )),
        ), source_snapshot=source_snapshot())
    ref = CanonicalTargetReference("relationship", "rel_000001")
    with pytest.raises(AdaptationTargetError, match="contradict"):
        resolve_adaptation_target(AdaptationTarget(
            "elements-conflict",
            required_elements=value((TargetElement(ref),)),
            forbidden_elements=value((TargetElement(ref),)),
        ), source_snapshot=source_snapshot())


def test_only_exact_stable_ids_are_resolved_and_unknowns_remain_explicit():
    character = CanonicalTargetReference("character", "char_0001")
    event = CanonicalTargetReference("event", "evt_000001")
    index = StableIdIndex((character, event), (("Alice", character),))

    assert index.resolve("char_0001") == character
    assert index.resolve(" alice ") == character
    unresolved = index.resolve("The Duchess")
    assert unresolved == UnresolvedTargetReference("The Duchess")
    with pytest.raises(AdaptationTargetError, match="stable ID"):
        CanonicalTargetReference("event", "event_1")
    with pytest.raises(AdaptationTargetError, match="ambiguously"):
        StableIdIndex((character, event), (("lead", character), ("LEAD", event)))


def test_count_constraints_reject_fractional_values_and_invalid_shapes():
    with pytest.raises(AdaptationTargetError, match="positive integer"):
        AdaptationTarget(
            "fractional-count",
            episode_count_constraints=value((constraint("fixed", "hard", value=1.5),)),
        )
    with pytest.raises(AdaptationTargetError, match="may only have derivation"):
        NumericConstraint("derived", "soft", value=2, derivation="not allowed")


def test_bound_snapshot_rejects_dangling_and_cross_revision_references():
    current = source_snapshot(CanonicalTargetReference("event", "evt_000001"), revision=2)
    old = source_snapshot(CanonicalTargetReference("event", "evt_000002"), revision=1)
    dangling = AdaptationTarget(
        "dangling", primary_focus=value((CanonicalTargetReference("event", "evt_999999"),))
    )
    cross_revision = AdaptationTarget(
        "cross-revision", primary_focus=value((CanonicalTargetReference("event", "evt_000002"),))
    )
    valid = AdaptationTarget(
        "valid", primary_focus=value((CanonicalTargetReference("event", "evt_000001"),))
    )
    with pytest.raises(AdaptationTargetError, match="outside the exact bound"):
        resolve_adaptation_target(dangling, source_snapshot=current)
    with pytest.raises(AdaptationTargetError, match="outside the exact bound"):
        resolve_adaptation_target(cross_revision, source_snapshot=current)
    assert resolve_adaptation_target(valid, source_snapshot=current).source_snapshot == current
    assert old.story_analysis_manifest_ref.revision != current.story_analysis_manifest_ref.revision


def test_fixed_seconds_reject_non_integral_count_feasibility_and_float_precision():
    raw = AdaptationTarget(
        "integral-count",
        total_duration_constraints=value((constraint("fixed", "hard", value=150),)),
        per_episode_duration_constraints=value((constraint("fixed", "hard", value=60),)),
        episode_count_constraints=value((constraint("range", "hard", minimum=2, maximum=3),)),
    )
    with pytest.raises(AdaptationTargetError, match="no feasible episode count"):
        resolve_adaptation_target(raw, source_snapshot=source_snapshot())
    with pytest.raises(AdaptationTargetError, match="positive integer"):
        AdaptationTarget(
            "float-seconds",
            total_duration_constraints=value((constraint("fixed", "hard", value=90.5),)),
        )


@pytest.mark.parametrize(
    "total,per_episode,count",
    [
        (("fixed", 121), ("range", 60, 61), ("fixed", 2)),
        (("range", 125, 130), ("fixed", 60), ("range", 2, 3)),
    ],
)
def test_discrete_hard_ranges_reject_continuous_hull_false_passes(total, per_episode, count):
    def build(spec):
        if spec[0] == "fixed":
            return constraint("fixed", "hard", value=spec[1])
        return constraint("range", "hard", minimum=spec[1], maximum=spec[2])

    raw = AdaptationTarget(
        "discrete-range",
        total_duration_constraints=value((build(total),)),
        per_episode_duration_constraints=value((build(per_episode),)),
        episode_count_constraints=value((build(count),)),
    )
    with pytest.raises(AdaptationTargetError, match="no feasible episode count"):
        resolve_adaptation_target(raw, source_snapshot=source_snapshot())


def test_large_range_discrete_feasibility_is_exact_when_one_dimension_is_narrow():
    """Do not skip a contradiction solely because the count range is large."""
    raw = AdaptationTarget(
        "large-narrow-duration-range",
        total_duration_constraints=value((constraint("range", "hard", minimum=5_000_006_000, maximum=5_000_007_000),)),
        per_episode_duration_constraints=value((constraint("range", "hard", minimum=1_000_000, maximum=1_000_001),)),
        episode_count_constraints=value((constraint("range", "hard", minimum=1, maximum=10_000),)),
    )
    with pytest.raises(AdaptationTargetError, match="no feasible episode count"):
        resolve_adaptation_target(raw, source_snapshot=source_snapshot())


def test_unbounded_integer_search_is_explicitly_pending_for_b1_5():
    raw = AdaptationTarget(
        "large-integer-rectangle",
        total_duration_constraints=value((constraint("range", "hard", minimum=5_000_000_000, maximum=5_000_001_000),)),
        per_episode_duration_constraints=value((constraint("range", "hard", minimum=1_000_000, maximum=1_010_000),)),
        episode_count_constraints=value((constraint("range", "hard", minimum=1, maximum=10_000),)),
    )
    resolved = resolve_adaptation_target(raw, source_snapshot=source_snapshot())
    assert resolved.duration_count_feasibility is DurationCountFeasibility.PENDING_B1_5
    tampered = resolved.to_dict()
    tampered["duration_count_feasibility"] = "feasible"
    with pytest.raises(AdaptationTargetError, match="must match deterministic"):
        type(resolved).from_dict(tampered)


def test_raw_and_overrides_cannot_elevate_profile_or_model_values_over_user():
    with pytest.raises(AdaptationTargetError, match="raw intent"):
        AdaptationTarget("bad-raw", pacing=value("fast", Provenance.PROFILE_DEFAULT))
    with pytest.raises(AdaptationTargetError, match="explicit user acceptance"):
        AdaptationTargetOverrides("bad-override", pacing=value("fast", Provenance.MODEL_PROPOSED))
    raw = AdaptationTarget("raw-user", pacing=value("fast"))
    override = AdaptationTargetOverrides("accepted", pacing=value("measured"))
    assert resolve_adaptation_target(raw, explicit_overrides=override, source_snapshot=source_snapshot()).pacing.value == "measured"


def test_missing_store_snapshot_input_fails_closed_without_writes(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    with pytest.raises(AdaptationTargetError, match="cannot resolve exact"):
        AdaptationSourceSnapshot.from_store(
            store, FilePointerStore(tmp_path / "pointers", store), artifact_ref("story_analysis_manifest", "project.a6.manifest")
        )
    assert not any((tmp_path / "artifacts").iterdir())


def test_store_snapshot_binds_exact_a5_a6_members_without_writes(tmp_path):
    """Use the existing validated A5/A6 test fixture; this path makes no writes."""
    from test_story_a6b_planning import _snapshot
    from short_drama.llm import PromptRegistry, load_semantic_profile
    from short_drama.paths import REPO_ROOT
    from short_drama.story import (
        ArcAnalysis, CharacterAnalysis, CharacterAnalysisSet, EvidenceBackedInterpretation,
        GlobalEventAnalysis, GlobalStoryBible, GlobalStructure, PlotWindowAnalysis,
        StoryAnalysisPersistenceService, build_story_analysis_plan_from_profile,
        load_story_analysis_profile,
    )

    def interp():
        return EvidenceBackedInterpretation(text_zh="测试。", evidence_mode="explicit")

    tree, _planning, _publication, a5_snapshot = _snapshot(tmp_path)
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    semantic = load_semantic_profile(REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml")
    plan = build_story_analysis_plan_from_profile(a5_snapshot, profile)
    chars = CharacterAnalysisSet(1, tuple(CharacterAnalysis(
        character_ref=packet.character_ref, role=interp(), goals=(), motivations=(), traits=(),
        key_event_refs=(), key_fact_refs=(), important_relationship_refs=(),
        arc_summary=interp(), unresolved_or_conflicting_points=(),
    ) for packet in plan.character_packages))
    events = GlobalEventAnalysis(1, plan.plan_hash, tuple(PlotWindowAnalysis(
        window_id=window.window_id, window_ordinal=window.window_ordinal,
        owned_event_refs=window.owned_event_ids, context_event_refs=window.context_event_ids,
        interpretation=interp(),
    ) for window in plan.windows), (), (
        ("canonical_event_count", len(plan.event_stream)), ("event_importance_count", 0),
        ("plot_window_count", len(plan.windows)),
    ))
    bible = GlobalStoryBible(1, interp(), interp(), "测试", "测试", interp(), (), (), (), (), (), interp(), interp(), (
        ("canonical_character_count", len(chars.analyses)), ("canonical_event_count", len(plan.event_stream)),
    ))
    publication = StoryAnalysisPersistenceService(tree.store, tree.pointers).publish_validated(
        project_id=a5_snapshot.consolidation_manifest.project_id,
        document_id=a5_snapshot.consolidation_manifest.document_id,
        profile=profile, semantic_profile=semantic, prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"),
        snapshot=a5_snapshot, plan=plan, characters=chars, events=events,
        arcs=ArcAnalysis(1, (), (), (), ()),
        structure=GlobalStructure(1, interp(), (), interp(), (), interp(), (), (), (), (), ()), bible=bible,
    )
    before = sorted(path.relative_to(tree.store.root) for path in tree.store.root.rglob("*"))
    bound = AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, publication.story_analysis_manifest_ref)
    after = sorted(path.relative_to(tree.store.root) for path in tree.store.root.rglob("*"))

    assert before == after
    assert bound.story_analysis_manifest_ref == publication.story_analysis_manifest_ref
    assert CanonicalTargetReference("event", "evt_000001") in bound.canonical_references
    report_path = tree.store._path(
        publication.validation_report_ref.artifact_type,
        publication.validation_report_ref.artifact_id,
        publication.validation_report_ref.revision,
    )
    report_path.unlink()
    with pytest.raises(AdaptationTargetError, match="exact validated A6"):
        AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, publication.story_analysis_manifest_ref)


def test_a4_registry_universe_allows_many_candidates_per_canonical_entity(tmp_path):
    from test_story_a6b_planning import _snapshot

    _tree, _planning, _publication, a5_snapshot = _snapshot(tmp_path)
    duplicated_candidate_map = replace(
        a5_snapshot.entity_map,
        entries=(*a5_snapshot.entity_map.entries, a5_snapshot.entity_map.entries[0]),
    )
    refs = _canonical_references_from_a4(
        duplicated_candidate_map,
        a5_snapshot.canonical_characters,
        a5_snapshot.canonical_locations,
    )
    assert [ref.canonical_id for ref in refs if ref.kind == "character"] == [
        entity.canonical_id for entity in a5_snapshot.canonical_characters
    ]
    invalid_map = replace(
        a5_snapshot.entity_map,
        entries=(replace(a5_snapshot.entity_map.entries[0], canonical_id="char_9999"), *a5_snapshot.entity_map.entries[1:]),
    )
    with pytest.raises(AdaptationTargetError, match="disagree"):
        _canonical_references_from_a4(
            invalid_map, a5_snapshot.canonical_characters, a5_snapshot.canonical_locations
        )
