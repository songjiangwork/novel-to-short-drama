import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from short_drama.adaptation import (
    AdaptationTarget,
    AdaptationTargetDefaults,
    AdaptationTargetError,
    AdaptationTargetOverrides,
    CanonicalTargetReference,
    ConstraintKind,
    ConstraintStrength,
    FidelityProfile,
    NumericConstraint,
    Provenance,
    StableIdIndex,
    TargetElement,
    TargetValue,
    UnresolvedTargetReference,
    resolve_adaptation_target,
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
        compression=value("high", Provenance.MODEL_PROPOSED),
    )

    resolved = resolve_adaptation_target(raw, defaults, overrides)

    assert resolved.narrative_format.value == "short drama"
    assert resolved.narrative_format.provenance is Provenance.PROFILE_DEFAULT
    assert resolved.fidelity.value is FidelityProfile.BALANCED
    assert resolved.fidelity.provenance is Provenance.USER
    assert resolved.pacing.value == "deliberate"
    assert resolved.pacing.provenance is Provenance.USER
    assert resolved.compression.value == "high"
    assert resolved.compression.provenance is Provenance.MODEL_PROPOSED
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
    resolved = resolve_adaptation_target(raw)
    data = resolved.to_dict()

    assert type(resolved).from_dict(data) == resolved
    assert type(resolved).from_dict(data).content_hash() == resolved.content_hash()
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "adaptation-target.schema.json").read_text())
    assert list(Draft202012Validator(schema).iter_errors(data)) == []
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
    resolved = resolve_adaptation_target(target)
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
        resolve_adaptation_target(raw)


def test_hard_constraint_self_conflict_and_element_conflict_fail_closed():
    with pytest.raises(AdaptationTargetError, match="no feasible"):
        resolve_adaptation_target(AdaptationTarget(
            "no-count",
            episode_count_constraints=value((
                constraint("fixed", "hard", value=2),
                constraint("range", "hard", minimum=3, maximum=4),
            )),
        ))
    ref = CanonicalTargetReference("relationship", "rel_000001")
    with pytest.raises(AdaptationTargetError, match="contradict"):
        resolve_adaptation_target(AdaptationTarget(
            "elements-conflict",
            required_elements=value((TargetElement(ref),)),
            forbidden_elements=value((TargetElement(ref),)),
        ))


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
