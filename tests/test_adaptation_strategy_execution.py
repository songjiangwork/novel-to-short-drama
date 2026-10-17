"""B1.2-B2 tests use an injected fake client; they never open a provider route."""
from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.adaptation import (
    B12SemanticBudget, B12SemanticBudgetError, B12SemanticGenerationExhaustedError,
    B12SemanticExecutionError, B12SemanticProvenanceError,
    B12_STRATEGY_ARTIFACT_TYPE, b12_strategy_pointer_id,
    build_b12_semantic_preparation, execute_b12_strategy, prepare_b12_packet,
)
from short_drama.artifacts import ImmutableArtifactEnvelope
from short_drama.foundation import PointerKind
from short_drama.guidance import (
    DirectionCategory, DirectionOrigin, DirectionStage, DirectionStrength,
    TargetScope, UserDirection, UserDirectionSet, persist_user_direction,
)
from short_drama.llm import (
    LLMClient, LLMInvocationProvenance, StructuredGenerationResult,
    build_structured_request,
)

from test_adaptation_strategy import _context_strategy, _proposal, _semantic_preparation


class _FakeClient(LLMClient):
    supported_structured_output_modes = frozenset({"json_schema"})

    def __init__(self, results, runtime, *, client_runtime=None, provenance_runtime=None):
        self._results = list(results)
        self.runtime = runtime
        if client_runtime is not None:
            self.runtime_config = client_runtime
        self.provenance_runtime = runtime if provenance_runtime is None else provenance_runtime
        self.calls = []

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile, *, execution_options=None):
        self.calls.append(execution_options)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        parsed, forged, forged_routing = (*result, False)[:3]
        routing = self.provenance_runtime
        if forged_routing:
            routing = replace(routing, request_model="forged-model")
        provenance = LLMInvocationProvenance(
            provider_family=routing.provider_family,
            model=routing.request_model,
            semantic_profile_id=semantic_profile.profile_id,
            semantic_profile_hash=semantic_profile.semantic_profile_hash,
            prompt_id=rendered_prompt.prompt_id,
            prompt_version=rendered_prompt.prompt_version,
            prompt_content_hash=rendered_prompt.prompt_content_hash,
            rendered_prompt_hash=rendered_prompt.rendered_prompt_hash,
            output_schema_id=output_schema.schema_id,
            output_schema_version=output_schema.schema_version,
            output_schema_hash=output_schema.schema_hash,
            request_hash=("f" * 64) if forged else build_structured_request(
                rendered_prompt=rendered_prompt, output_schema=output_schema, semantic_profile=semantic_profile
            ).request_hash,
            provider_response_id="fake-response", finish_reason="stop", usage={"prompt_tokens": 1},
        )
        return StructuredGenerationResult(parsed_json=parsed, provenance=provenance, attempts=1)


def _budget(semantic):
    return B12SemanticBudget(
        semantic.runtime_config.request_model,
        1_000_000,
        1_000_000,
    )


def _valid_proposal(semantic):
    return _proposal(_context_strategy(
        semantic.preparation.packet.resolved_target, semantic.preparation.packet.plotlines
    ))


def _project_id(tree, semantic):
    return tree.store.get_ref(
        semantic.preparation.packet.source_snapshot.story_analysis_manifest_ref
    ).payload["project_id"]


def _semantic_with_directions(tree, semantic, directions):
    prepared = prepare_b12_packet(
        tree.store, tree.pointers,
        semantic.preparation.packet.resolved_target, directions,
    )
    return build_b12_semantic_preparation(
        prepared, runtime_config=semantic.runtime_config,
    )


def _move_current_to_payload(tree, semantic, publication, payload):
    project_id = _project_id(tree, semantic)
    pointer_id = b12_strategy_pointer_id(
        project_id, semantic.preparation.packet.resolved_target.adaptation_target_id
    )
    forged = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B12_STRATEGY_ARTIFACT_TYPE,
        artifact_id=publication.strategy_ref.artifact_id,
        revision=publication.strategy_ref.revision + 1,
        schema_version=1,
        payload=payload,
    ))
    tree.pointers.compare_and_set(
        pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=publication.current_pointer_ref, target_ref=forged,
    )


def test_valid_execution_persists_then_exactly_reuses_without_provider_call(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    fake = _FakeClient([(_valid_proposal(semantic), False)], semantic.runtime_config)

    publication = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=fake,
    )

    assert len(fake.calls) == 1
    assert publication.reused is False
    assert publication.artifact.strategy.resolved_target == semantic.preparation.packet.resolved_target
    assert publication.artifact.requires_human_review is False
    assert publication.current_pointer_ref.artifact_type == "current_pointer"
    reused_client = _FakeClient([], semantic.runtime_config)
    reused = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=reused_client,
    )
    assert reused.reused is True
    assert reused.strategy_ref == publication.strategy_ref
    assert reused_client.calls == []


def test_changed_b1_direction_is_a_healthy_cache_miss_and_publishes_new_revision(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    first = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic),
        llm_client=_FakeClient([(_valid_proposal(semantic), False)], semantic.runtime_config),
    )
    project_id = _project_id(tree, semantic)
    direction_ref = persist_user_direction(tree.store, UserDirection(
        "b12-focus", project_id, 1, DirectionOrigin.USER,
        DirectionCategory.ADAPTATION_STORY, DirectionStrength.PREFER,
        DirectionStage.B1_ADAPTATION, TargetScope.PROJECT, (), "强化主线冲突。",
    ))
    changed = _semantic_with_directions(
        tree, semantic, UserDirectionSet(project_id, "b12-directions", 1, (direction_ref,))
    )
    assert changed.request_identity != semantic.request_identity
    fake = _FakeClient([(_valid_proposal(changed), False)], changed.runtime_config)
    second = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=changed,
        runtime_config=changed.runtime_config, budget=_budget(changed), llm_client=fake,
    )
    assert second.reused is False
    assert second.strategy_ref.revision == first.strategy_ref.revision + 1
    assert len(fake.calls) == 1


def test_tampered_old_request_identity_blocks_changed_b1_cache_miss_before_provider(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    publication = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic),
        llm_client=_FakeClient([(_valid_proposal(semantic), False)], semantic.runtime_config),
    )
    payload = publication.artifact.to_dict()
    payload["request_identity"] = "f" * 64
    _move_current_to_payload(tree, semantic, publication, payload)
    project_id = _project_id(tree, semantic)
    direction_ref = persist_user_direction(tree.store, UserDirection(
        "b12-changed-focus", project_id, 1, DirectionOrigin.USER,
        DirectionCategory.ADAPTATION_STORY, DirectionStrength.PREFER,
        DirectionStage.B1_ADAPTATION, TargetScope.PROJECT, (), "强化悬念。",
    ))
    changed = _semantic_with_directions(
        tree, semantic, UserDirectionSet(project_id, "b12-directions", 1, (direction_ref,))
    )
    fake = _FakeClient([(_valid_proposal(changed), False)], changed.runtime_config)
    with pytest.raises(B12SemanticExecutionError, match="request_identity"):
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=changed,
            runtime_config=changed.runtime_config, budget=_budget(changed), llm_client=fake,
        )
    assert fake.calls == []


def test_historical_plotline_handles_are_checked_without_requiring_a6_current_reload(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    publication = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic),
        llm_client=_FakeClient([(_valid_proposal(semantic), False)], semantic.runtime_config),
    )
    payload = publication.artifact.to_dict()
    payload["strategy"]["plotline_strategies"][0]["plotline_handle"] = "a6pl_999_forged000000000"
    _move_current_to_payload(tree, semantic, publication, payload)
    fake = _FakeClient([], semantic.runtime_config)
    with pytest.raises(B12SemanticExecutionError, match="pinned historical A6 structure"):
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=semantic,
            runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=fake,
        )
    assert fake.calls == []


def test_b4_only_direction_leaves_exact_reuse_eligible(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic),
        llm_client=_FakeClient([(_valid_proposal(semantic), False)], semantic.runtime_config),
    )
    project_id = _project_id(tree, semantic)
    camera_ref = persist_user_direction(tree.store, UserDirection(
        "b4-camera", project_id, 1, DirectionOrigin.USER,
        DirectionCategory.CAMERA, DirectionStrength.PREFER,
        DirectionStage.B4_SHOT, TargetScope.PROJECT, (), "多用近景。",
    ))
    b4_only = _semantic_with_directions(
        tree, semantic, UserDirectionSet(project_id, "b12-directions", 1, (camera_ref,))
    )
    assert b4_only.request_identity == semantic.request_identity
    no_call = _FakeClient([], b4_only.runtime_config)
    reused = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=b4_only,
        runtime_config=b4_only.runtime_config, budget=_budget(b4_only), llm_client=no_call,
    )
    assert reused.reused is True
    assert no_call.calls == []


def test_budget_failures_happen_before_fake_provider_call(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    fake = _FakeClient([], semantic.runtime_config)
    with pytest.raises(B12SemanticBudgetError, match="requires an explicit"):
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=semantic,
            runtime_config=semantic.runtime_config, budget=None, llm_client=fake,
        )
    with pytest.raises(B12SemanticBudgetError, match="input ceiling"):
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=semantic,
            runtime_config=semantic.runtime_config,
            budget=B12SemanticBudget(semantic.runtime_config.request_model, 1_000_000, 1), llm_client=fake,
        )
    assert fake.calls == []


def test_only_schema_valid_but_rejected_strategy_is_regenerated_once_with_rechecked_options(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    bad = _valid_proposal(semantic)
    bad["plotline_strategies"] = []  # schema-shaped but fails exact plotline coverage.
    fake = _FakeClient([(bad, False), (_valid_proposal(semantic), False)], semantic.runtime_config)
    publication = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=fake,
    )
    assert publication.artifact.semantic_rounds == 2
    assert fake.calls[0] is None
    assert fake.calls[1].prompt_context_reuse == "disabled"


def test_semantic_exhaustion_does_not_publish_strategy(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    bad = _valid_proposal(semantic)
    bad["plotline_strategies"] = []
    fake = _FakeClient([(bad, False), (bad, False)], semantic.runtime_config)
    with pytest.raises(B12SemanticGenerationExhaustedError) as error:
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=semantic,
            runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=fake,
        )
    assert error.value.rounds_attempted == 2
    assert len(fake.calls) == 2
    assert not list(tree.store.root.glob("adaptation_strategy/*"))


def test_forged_provider_provenance_fails_closed_without_second_call(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    fake = _FakeClient([(_valid_proposal(semantic), True)], semantic.runtime_config)
    with pytest.raises(B12SemanticProvenanceError):
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=semantic,
            runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=fake,
        )
    assert len(fake.calls) == 1


def test_mismatched_concrete_client_runtime_fails_before_provider_call(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    wrong_runtime = replace(semantic.runtime_config, request_model="other-model")
    fake = _FakeClient(
        [], semantic.runtime_config, client_runtime=wrong_runtime,
    )
    with pytest.raises(B12SemanticExecutionError, match="LLMClient runtime_config"):
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=semantic,
            runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=fake,
        )
    assert fake.calls == []


def test_forged_provider_routing_provenance_fails_closed(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    fake = _FakeClient([(_valid_proposal(semantic), False, True)], semantic.runtime_config)
    with pytest.raises(B12SemanticProvenanceError, match="routing provenance"):
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=semantic,
            runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=fake,
        )
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "field",
    [
        "semantic_identity", "prompt_identity", "output_schema_identity",
        "semantic_profile_identity", "rendered_prompt_hash", "review_metadata",
    ],
)
def test_forged_persisted_identity_or_review_metadata_fails_before_provider(tmp_path, field):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    publication = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic),
        llm_client=_FakeClient([(_valid_proposal(semantic), False)], semantic.runtime_config),
    )
    payload = publication.artifact.to_dict()
    if field == "semantic_identity":
        # Self-consistent bogus provenance must still be rejected against the
        # authoritative current preparation, not treated as a cache hit.
        payload["semantic_request_hash"] = "f" * 64
        payload["invocation_provenance"]["request_hash"] = "f" * 64
    elif field == "prompt_identity":
        payload["prompt_identity"]["prompt_id"] = "forged-prompt"
        payload["invocation_provenance"]["prompt_id"] = "forged-prompt"
    elif field == "output_schema_identity":
        payload["output_schema_identity"]["schema_id"] = "forged-schema"
        payload["invocation_provenance"]["output_schema_id"] = "forged-schema"
    elif field == "semantic_profile_identity":
        payload["semantic_profile_identity"]["profile_id"] = "forged-profile"
        payload["invocation_provenance"]["semantic_profile_id"] = "forged-profile"
    elif field == "rendered_prompt_hash":
        payload["rendered_request_hash"] = "f" * 64
        payload["invocation_provenance"]["rendered_prompt_hash"] = "f" * 64
    else:
        payload["requires_human_review"] = True
        payload["review_reasons"] = ["explicit_conflict"]
    _move_current_to_payload(tree, semantic, publication, payload)
    fake = _FakeClient([], semantic.runtime_config)
    with pytest.raises(B12SemanticExecutionError):
        execute_b12_strategy(
            store=tree.store, pointers=tree.pointers, semantic=semantic,
            runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=fake,
        )
    assert fake.calls == []


def test_runtime_endpoint_change_keeps_exact_reuse_identity(tmp_path):
    tree, packet, semantic = _semantic_preparation(tmp_path)
    first = _FakeClient([(_valid_proposal(semantic), False)], semantic.runtime_config)
    execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic), llm_client=first,
    )
    changed_runtime = replace(semantic.runtime_config, base_url="http://127.0.0.1:8081/v1")
    remapped = build_b12_semantic_preparation(packet, runtime_config=changed_runtime)
    assert remapped.request_identity == semantic.request_identity
    no_call = _FakeClient([], changed_runtime)
    reused = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=remapped,
        runtime_config=changed_runtime, budget=_budget(remapped), llm_client=no_call,
    )
    assert reused.reused is True
    assert no_call.calls == []
