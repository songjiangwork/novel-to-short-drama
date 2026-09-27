"""A5F3 current-only, pre-provider consolidation reuse acceptance tests.

The synthetic fixture is deliberately shared with A5F2: it publishes a real,
fully verified A4 CURRENT and a real A5F2 bundle, but has zero provider calls.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.llm import OutputSchema, PromptSpec, build_structured_request
from short_drama.llm.config import RUNTIME_CONFIG_SCHEMA_VERSION, RuntimeConfig
from short_drama.foundation import PointerIntegrityError
from short_drama.story import (
    ConsolidationPersistenceService,
    ConsolidationUpstreamUnstableError,
    StoryIntegrityError,
    StoryPersistenceError,
    build_a5_semantic_identity,
)
from short_drama.story.consolidation_persistence import (
    CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
)
from short_drama.story.consolidation_semantic import FactSemanticBlock

# The A5F2 fixture builds all upstream state through the production persistence
# authorities.  Keeping this focused test module dependent on it prevents a
# second, weaker A5 CURRENT fixture from accidentally masking corruption.
from test_story_a5f2_current_publication import (  # noqa: E402
    DOCUMENT,
    PROJECT,
    _HASH2,
    _build_tree_with_a4,
    _current_a5_target,
    _highest_revision,
    _make_preparations,
    _planning,
    _publish_a4,
    _publish_a5,
    _store_path,
)


def _published_current(tmp_path):
    tree, _ext_ref, _a3_input, _index = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    publication = _publish_a5(tree, planning)
    preparations = _make_preparations(planning, tree.consolidation_profile)
    return (
        tree,
        planning,
        publication,
        preparations,
        ConsolidationPersistenceService(tree.store, tree.pointers),
    )


def _reuse(service, tree, planning, preparations, profile=None):
    fact, event, relationship = preparations
    return service.try_reuse_current(
        project_id=PROJECT,
        document_id=DOCUMENT,
        consolidation_profile=profile or tree.consolidation_profile,
        planning_result=planning,
        fact_preparation=fact,
        event_preparation=event,
        relationship_preparation=relationship,
    )


def test_missing_current_is_read_only_normal_miss(tmp_path):
    tree, _ext_ref, _a3_input, _index = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    preparations = _make_preparations(planning, tree.consolidation_profile)
    service = ConsolidationPersistenceService(tree.store, tree.pointers)

    assert _reuse(service, tree, planning, preparations) is None
    assert _current_a5_target(tree) is None


def test_exact_hit_returns_identical_refs_without_writes(tmp_path):
    tree, planning, published, preparations, service = _published_current(tmp_path)

    reused = _reuse(service, tree, planning, preparations)

    assert reused is not None
    assert reused.reused is True
    assert reused == replace(published, reused=True)
    assert _current_a5_target(tree) == published.consolidation_manifest_ref


def test_transport_only_runtime_change_remains_reusable(tmp_path):
    tree, planning, published, preparations, service = _published_current(tmp_path)
    runtime_a = RuntimeConfig(
        schema_version=RUNTIME_CONFIG_SCHEMA_VERSION,
        transport_id="local-qwen",
        base_url="http://127.0.0.1:8080/v1",
        request_model="qwen3-27b",
        provider_family="qwen",
        credential_environment_name="QWEN_TOKEN",
        timeout_seconds=30,
    )
    runtime_b = RuntimeConfig(
        schema_version=RUNTIME_CONFIG_SCHEMA_VERSION,
        transport_id="remote-gemma",
        base_url="https://example.invalid/v1",
        request_model="gemma-3",
        provider_family="gemma",
        credential_environment_name="GEMMA_TOKEN",
        timeout_seconds=120,
    )
    assert runtime_a != runtime_b
    identity_before = build_a5_semantic_identity(*preparations)
    assert identity_before == build_a5_semantic_identity(*preparations)
    assert tuple(
        request_hash
        for preparation in preparations
        for request_hash in preparation.semantic_request_hashes
    ) == identity_before.semantic_request_hashes

    # RuntimeConfig is intentionally test context only: it is absent from the
    # production reuse API and cannot affect the preparation/request hashes.
    assert _reuse(service, tree, planning, preparations) == replace(published, reused=True)


def test_ordered_semantic_request_hashes_are_not_sorted_or_deduplicated(tmp_path):
    tree, planning, _published, preparations, service = _published_current(tmp_path)
    fact, event, relationship = preparations
    schema = OutputSchema.create(
        schema_id="a5f3-test-schema",
        schema_version=1,
        schema={"type": "object", "additionalProperties": False},
    )

    def request(user_text):
        prompt = PromptSpec.create(
            prompt_id="a5f3-test-prompt",
            version=1,
            system_template="",
            user_template=user_text,
            required_variables=(),
        )
        return build_structured_request(
            rendered_prompt=prompt.render({}),
            output_schema=schema,
            semantic_profile=fact.semantic_profile,
        )

    first, second = request("first"), request("second")
    blocks = (
        FactSemanticBlock(0, "a5fblk_" + "a" * 20, (), (), (), "[]"),
        FactSemanticBlock(1, "a5fblk_" + "b" * 20, (), (), (), "[]"),
    )
    ordered = replace(
        fact,
        blocks=blocks,
        structured_requests=(first, second),
        semantic_request_hashes=(first.request_hash, second.request_hash),
    )
    reversed_order = replace(
        fact,
        blocks=blocks,
        structured_requests=(second, first),
        semantic_request_hashes=(second.request_hash, first.request_hash),
    )
    ordered_identity = build_a5_semantic_identity(ordered, event, relationship)
    reversed_identity = build_a5_semantic_identity(reversed_order, event, relationship)

    assert ordered_identity.semantic_request_hashes == (first.request_hash, second.request_hash)
    assert reversed_identity.semantic_request_hashes == (second.request_hash, first.request_hash)
    assert ordered_identity != reversed_identity
    assert _reuse(service, tree, planning, (ordered, event, relationship)) is None


@pytest.mark.parametrize("change", ["profile", "semantic", "prompt", "schema", "plan"])
def test_result_affecting_semantic_contract_changes_are_misses(tmp_path, change):
    tree, planning, _published, preparations, service = _published_current(tmp_path)
    fact, event, relationship = preparations
    profile = tree.consolidation_profile

    if change == "profile":
        profile = replace(profile, max_generation_rounds=profile.max_generation_rounds + 1)
        preparations = _make_preparations(planning, profile)
    elif change == "semantic":
        temperature = 0.9 if fact.semantic_profile.temperature != 0.9 else 0.8
        preparations = (replace(fact, semantic_profile=replace(fact.semantic_profile, temperature=temperature)), event, relationship)
    elif change == "prompt":
        preparations = (replace(fact, prompt_content_hash=_HASH2), event, relationship)
    elif change == "schema":
        preparations = (fact, replace(event, output_schema_hash=_HASH2), relationship)
    else:
        changed_plan = replace(planning, plan_hash=_HASH2)
        planning = changed_plan
        preparations = tuple(
            replace(preparation, planning_result=changed_plan)
            for preparation in preparations
        )

    assert _reuse(service, tree, planning, preparations, profile) is None


def test_upstream_entity_map_mismatch_is_a_normal_miss(tmp_path):
    tree, planning, _published, preparations, service = _published_current(tmp_path)
    _publish_a4(tree, suffix="002", revision=2)
    new_planning = _planning(tree)
    new_preparations = _make_preparations(new_planning, tree.consolidation_profile)

    assert _reuse(service, tree, new_planning, new_preparations) is None


def test_hit_rechecks_a4_current_stability(tmp_path):
    tree, planning, published, preparations, service = _published_current(tmp_path)
    _publish_a4(tree, suffix="002", revision=2)

    with pytest.raises(ConsolidationUpstreamUnstableError):
        _reuse(service, tree, planning, preparations)
    assert _current_a5_target(tree) == published.consolidation_manifest_ref


def test_a5_current_advance_during_reuse_eligibility_fails_closed(
    tmp_path, monkeypatch
):
    tree, planning, published, preparations, service = _published_current(tmp_path)
    original_upstream_check = service._require_a4_upstream_stable
    winning_publications = []

    def advance_a5_current_after_upstream_check(**kwargs):
        original_upstream_check(**kwargs)
        winning_publications.append(_publish_a5(tree, planning))

    monkeypatch.setattr(
        service, "_require_a4_upstream_stable", advance_a5_current_after_upstream_check
    )

    with pytest.raises(
        StoryPersistenceError, match="A5 CURRENT pointer changed during reuse eligibility check"
    ):
        _reuse(service, tree, planning, preparations)

    winner = winning_publications[0]
    assert winner.consolidation_manifest_ref.revision == 2
    assert _current_a5_target(tree) == winner.consolidation_manifest_ref
    assert (
        tree.pointers.resolve_current_pointer_ref(winner.current_pointer_ref.artifact_id)
        == winner.current_pointer_ref
    )
    assert _highest_revision(
        tree.store,
        CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
        winner.consolidation_manifest_ref.artifact_id,
    ) == winner.consolidation_manifest_ref.revision
    assert published.consolidation_manifest_ref != winner.consolidation_manifest_ref


def test_corrupt_current_fails_closed_not_as_a_miss(tmp_path):
    tree, planning, published, preparations, service = _published_current(tmp_path)
    _store_path(
        tree.store,
        CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
        published.consolidation_manifest_ref.artifact_id,
        published.consolidation_manifest_ref.revision,
    ).write_text("{not-json", encoding="utf-8")

    with pytest.raises((StoryIntegrityError, PointerIntegrityError)):
        _reuse(service, tree, planning, preparations)


def test_historical_matching_revision_is_never_reused(tmp_path):
    tree, planning_a, published_a, preparations_a, service = _published_current(tmp_path)
    _publish_a4(tree, suffix="002", revision=2)
    planning_b = _planning(tree)
    published_b = _publish_a5(tree, planning_b)

    assert _reuse(service, tree, planning_a, preparations_a) is None
    assert _current_a5_target(tree) == published_b.consolidation_manifest_ref
    assert published_a.consolidation_manifest_ref != published_b.consolidation_manifest_ref
