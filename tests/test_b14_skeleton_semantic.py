"""B1.4-B1 stays zero-provider and leaves no B1.3/A6 mutation behind."""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib

import pytest

from short_drama.adaptation import (
    B14SkeletonPacket, B14SkeletonSemanticBudget, B14SkeletonSemanticBudgetError,
    B14SkeletonSemanticError, build_b14_skeleton_semantic_preparation,
    load_b14_skeleton_semantic_assets, prepare_b14_skeleton_packet,
)
from short_drama.llm import LLMStructuredOutputError, OutputSchema, validate_against_output_schema
from short_drama.adaptation.disposition_publication import (
    B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
    b13_disposition_publication_artifact_id,
)
from short_drama.artifacts import ImmutableArtifactEnvelope, canonical_json_bytes
from short_drama.guidance import (
    DirectionCategory, DirectionOrigin, DirectionStage, DirectionStrength,
    TargetScope, UserDirection,
)
from short_drama.guidance.models import GUIDANCE_SCHEMA_VERSION
from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_persistence import _prepared, _publication
from test_b14_skeleton_validation import _skeleton as _typed_skeleton


def _packet(tmp_path):
    tree, _semantic, strategy, preparation, requests = _prepared(tmp_path)
    publication = _publication(preparation, requests)
    publication_ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        artifact_id=b13_disposition_publication_artifact_id(preparation.source_snapshot, strategy.artifact),
        revision=1, schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=publication.to_dict(),
    ))
    return tree, strategy.strategy_ref, publication_ref, prepare_b14_skeleton_packet(
        store=tree.store, pointers=tree.pointers, strategy_ref=strategy.strategy_ref,
        disposition_publication_ref=publication_ref,
    )


def test_whole_story_packet_assets_identity_and_runtime_independence(tmp_path):
    tree, strategy_ref, publication_ref, packet = _packet(tmp_path)
    assert packet.to_dict()["authority"]["strategy_ref"] == strategy_ref.to_dict()
    assert packet.to_dict()["authority"]["disposition_publication_ref"] == publication_ref.to_dict()
    assert "accepted_windows" not in packet.to_dict()
    assert packet.to_dict()["verified_b13_disposition"]["adapted_story_beats"]
    assets = load_b14_skeleton_semantic_assets()
    assert assets.semantic_profile.max_output_tokens == 16384
    one = build_b14_skeleton_semantic_preparation(packet, runtime_config=_b12_test_runtime_config())
    two = build_b14_skeleton_semantic_preparation(
        packet, runtime_config=replace(_b12_test_runtime_config(), base_url="http://other.invalid/v1", timeout_seconds=99),
    )
    assert one.request_identity == two.request_identity
    assert one.request.request_hash == two.request.request_hash
    assert one.measurement.packet_bytes == len(packet.canonical_bytes())
    assert one.measurement.rendered_prompt_bytes > one.measurement.packet_bytes
    assert one.measurement.total_requested_context_tokens == (
        one.measurement.provider_request_estimated_tokens + one.measurement.reserved_output_tokens
    )
    # Preparation has no client parameter and only reads these immutable refs.
    assert tree.store.get_ref(strategy_ref).ref == strategy_ref


def test_explicit_budget_fails_closed_and_wrong_ref_cannot_resolve(tmp_path):
    tree, strategy_ref, publication_ref, packet = _packet(tmp_path)
    semantic = build_b14_skeleton_semantic_preparation(packet, runtime_config=_b12_test_runtime_config())
    semantic.enforce_budget(None)  # Measurement remains usable without pretending a default n_ctx.
    with pytest.raises(B14SkeletonSemanticBudgetError, match="input ceiling"):
        semantic.enforce_budget(B14SkeletonSemanticBudget(semantic.runtime_config.request_model, 10**9, 1))
    with pytest.raises(B14SkeletonSemanticBudgetError, match="output reserve"):
        semantic.enforce_budget(B14SkeletonSemanticBudget(semantic.runtime_config.request_model, 1, 10**9))
    with pytest.raises(B14SkeletonSemanticError):
        prepare_b14_skeleton_packet(store=tree.store, pointers=tree.pointers,
            strategy_ref=publication_ref, disposition_publication_ref=publication_ref)


def test_schema_tamper_and_unknown_adapted_id_do_not_become_accepted(tmp_path):
    _tree, _strategy_ref, _publication_ref, packet = _packet(tmp_path)
    semantic = build_b14_skeleton_semantic_preparation(packet, runtime_config=_b12_test_runtime_config())
    fields = semantic.output_schema.schema["required"]
    bad = {name: [] for name in fields}
    bad["extra"] = []
    with pytest.raises(B14SkeletonSemanticError, match="schema/typed"):
        semantic.decode_and_validate_proposal(bad)
    proposal = {name: [] for name in fields}
    proposal["primary_presentation"] = ["adapted_missing"]
    _skeleton, result = semantic.decode_and_validate_proposal(proposal)
    assert not result.valid
    assert {finding.code for finding in result.findings} & {"presentation.beat", "beat.accounting"}


def test_asset_pair_rejects_tampered_schema_or_profile(tmp_path):
    _tree, _strategy_ref, _publication_ref, packet = _packet(tmp_path)
    assets = load_b14_skeleton_semantic_assets()
    altered_schema = deepcopy(assets.output_schema.schema)
    altered_schema["title"] = "tampered"
    with pytest.raises(B14SkeletonSemanticError, match="pinned B1.4 pair"):
        build_b14_skeleton_semantic_preparation(packet, runtime_config=_b12_test_runtime_config(),
            semantic_profile=assets.semantic_profile,
            output_schema=OutputSchema.create(schema_id=assets.output_schema.schema_id,
                schema_version=assets.output_schema.schema_version, schema=altered_schema))
    with pytest.raises(B14SkeletonSemanticError, match="pinned B1.4 pair"):
        build_b14_skeleton_semantic_preparation(packet, runtime_config=_b12_test_runtime_config(),
            semantic_profile=replace(assets.semantic_profile, temperature=0.1), output_schema=assets.output_schema)


def test_schema_accepts_a1_shape_and_rejects_malformed_nested_objects(tmp_path):
    _tree, _strategy_ref, _publication_ref, packet = _packet(tmp_path)
    schema = load_b14_skeleton_semantic_assets().output_schema
    proposal = _typed_skeleton().to_dict()
    proposal.pop("schema_version")
    proposal.pop("strategy_ref")
    proposal.pop("disposition_publication_ref")
    validate_against_output_schema(proposal, schema)
    malformed = []
    missing_outcome_field = deepcopy(proposal)
    missing_outcome_field["outcomes"][0].pop("description_zh")
    malformed.append(missing_outcome_field)
    extra_presentation_field = deepcopy(proposal)
    extra_presentation_field["primary_presentation"] = [{
        "presentation_entry_id": "pe_one", "adapted_story_beat_id": "adapted_one",
        "semantics": "full", "description_zh": "完整呈现。", "extra": "forbidden",
    }]
    malformed.append(extra_presentation_field)
    invalid_local_beat = deepcopy(proposal)
    invalid_local_beat["local_invented_beats"] = [{"origin": "source_direct"}]
    malformed.append(invalid_local_beat)
    forbidden_authority = deepcopy(proposal)
    forbidden_authority["strategy_ref"] = packet.authority.strategy_ref.to_dict()
    malformed.append(forbidden_authority)
    for value in malformed:
        with pytest.raises(LLMStructuredOutputError):
            validate_against_output_schema(value, schema)


@pytest.mark.parametrize("field", ["resolved_target", "effective_b1_directions", "global_and_canonical_context"])
def test_packet_payload_content_cannot_be_detached_from_verified_construction(tmp_path, field):
    _tree, _strategy_ref, _publication_ref, packet = _packet(tmp_path)
    altered = packet.to_dict()
    if field == "resolved_target":
        altered[field]["adaptation_target_id"] = "tampered-target"
    elif field == "effective_b1_directions":
        altered[field]["effective_direction_hash"] = "0" * 64
    else:
        altered[field]["a5_canonical_events"] = []
    with pytest.raises(B14SkeletonSemanticError, match="packet"):
        replace(packet, _canonical_payload=canonical_json_bytes(altered))


def test_b14_identity_consumes_only_the_exact_b12_b1_guidance_snapshot(tmp_path):
    _tree, _strategy_ref, _publication_ref, packet = _packet(tmp_path)
    baseline = build_b14_skeleton_semantic_preparation(packet, runtime_config=_b12_test_runtime_config())
    direction = UserDirection(
        "b14-story-focus", "test-project", 1, DirectionOrigin.USER,
        DirectionCategory.ADAPTATION_STORY, DirectionStrength.PREFER,
        DirectionStage.B1_ADAPTATION, TargetScope.PROJECT, (), "强化主线悬念。",
    )
    direction_hash = canonical_json_bytes({
        "schema_version": GUIDANCE_SCHEMA_VERSION,
        "direction_refs": [direction.artifact_ref.to_dict()],
    })
    changed_strategy = replace(
        packet.strategy, effective_direction_refs=(direction.artifact_ref,),
        effective_b1_direction_hash=hashlib.sha256(direction_hash).hexdigest(),
    )
    changed = packet.to_dict()
    changed["effective_b1_directions"] = {
        "effective_direction_hash": changed_strategy.effective_b1_direction_hash,
        "direction_refs": [direction.artifact_ref.to_dict()], "directions": [direction.to_dict()],
    }
    guided = B14SkeletonPacket(
        packet.authority, changed_strategy, (direction,), packet._verified_global_context,
        canonical_json_bytes(changed),
    )
    guided_request = build_b14_skeleton_semantic_preparation(guided, runtime_config=_b12_test_runtime_config())
    assert guided_request.request_identity != baseline.request_identity
    # A B4-only direction cannot enter B1.2's B1-effective snapshot and so
    # cannot invalidate this B1.4 request without an upstream B1 revision.
    b4_direction = UserDirection(
        "b14-camera", "test-project", 1, DirectionOrigin.USER,
        DirectionCategory.CAMERA, DirectionStrength.PREFER, DirectionStage.B4_SHOT,
        TargetScope.PROJECT, (), "更常用近景。",
    )
    assert b4_direction.artifact_ref not in packet.strategy.effective_direction_refs
    assert build_b14_skeleton_semantic_preparation(packet, runtime_config=_b12_test_runtime_config()).request_identity == baseline.request_identity
