"""A5G1 provider-neutral consolidation composition tests (no live provider)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.foundation import PointerIntegrityError
from short_drama.llm import LLMError
from short_drama.story import (
    ConsolidationCurrentMissingError,
    EvidenceConsolidationService,
    FACT_SEMANTIC_PACKING_V2,
)
from short_drama.story import consolidation_service as service_module
from short_drama.story.consolidation_persistence import (
    CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
)
from test_story_a5f2_current_publication import (
    DOCUMENT,
    PROJECT,
    RECON_PROFILE_ID,
    _build_base_tree,
    _build_tree_with_a4,
    _current_a5_target,
    _highest_revision,
    _store_path,
)
from test_story_a5b_planning import (
    DOCUMENT as A5B_DOCUMENT,
    PROJECT as A5B_PROJECT,
    _build_run_tree,
    _consolidation_profile,
    _default_candidates,
)
from test_story_a5c_fact_resolution import FakeLLMClient, _valid_responses


class ExplodingLLM:
    """Fails if an offline reuse/zero-semantic path tries a provider call."""

    def __init__(self):
        self.calls = 0

    def generate_structured(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("provider must not be called")


def _inputs(tmp_path):
    tree, _ext_ref, _a3_input, _index = _build_tree_with_a4(tmp_path)
    from short_drama.story import load_fact_semantic_profile

    return tree, load_fact_semantic_profile(), ExplodingLLM()


def _run(tree, semantic_profile, llm, *, max_concurrency=1):
    return EvidenceConsolidationService(tree.store, tree.pointers).consolidate_evidence(
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=tree.consolidation_profile,
        semantic_profile=semantic_profile,
        llm_client=llm,
        max_concurrency=max_concurrency,
    )


def _semantic_pair_inputs(tmp_path):
    """Build a valid A4 CURRENT whose fact stream needs one LLM decision."""
    candidates = _default_candidates()
    first, second = candidates["facts"][0], candidates["facts"][0]
    candidates["facts"] = (
        replace(first, statement_zh="Alice knows the key"),
        replace(
            second,
            candidate_id="cand_fact_002",
            statement_zh="Alice lost the key",
        ),
    )
    tree = _build_run_tree(tmp_path, **candidates)
    from short_drama.story import build_consolidation_planning, load_fact_semantic_profile

    profile = _consolidation_profile()
    semantic_profile = load_fact_semantic_profile()
    planning = build_consolidation_planning(
        tree.store,
        tree.pointers,
        project_id=A5B_PROJECT,
        document_id=A5B_DOCUMENT,
        reconciliation_profile_id=tree.recon_profile.profile_id,
        consolidation_profile=profile,
    )
    assert len(planning.fact_pair_plans) == 1
    return tree, profile, semantic_profile, planning


def _run_semantic_pair(tree, profile, semantic_profile, llm):
    return EvidenceConsolidationService(tree.store, tree.pointers).consolidate_evidence(
        project_id=A5B_PROJECT,
        document_id=A5B_DOCUMENT,
        reconciliation_profile_id=tree.recon_profile.profile_id,
        consolidation_profile=profile,
        semantic_profile=semantic_profile,
        llm_client=llm,
    )


def test_fresh_miss_publishes_through_existing_authorities(tmp_path):
    tree, semantic_profile, llm = _inputs(tmp_path)

    result = _run(tree, semantic_profile, llm)

    assert result.reused is False
    assert result.semantic_generation_call_count == 0
    assert llm.calls == 0
    assert _current_a5_target(tree) == result.consolidation_manifest_ref
    assert result.to_dict()["entity_map_ref"] == result.entity_map_ref.to_dict()


def test_exact_reuse_is_pre_provider_and_returns_same_refs(tmp_path):
    tree, semantic_profile, llm = _inputs(tmp_path)
    fresh = _run(tree, semantic_profile, llm)
    reuse_llm = ExplodingLLM()

    reused = _run(tree, semantic_profile, reuse_llm, max_concurrency=8)

    assert reused.reused is True
    assert reused.semantic_generation_call_count == 0
    assert reuse_llm.calls == 0
    assert reused.consolidation_manifest_ref == fresh.consolidation_manifest_ref
    assert reused.validation_report_ref == fresh.validation_report_ref
    assert reused.current_pointer_ref == fresh.current_pointer_ref
    assert _highest_revision(
        tree.store,
        CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
        fresh.consolidation_manifest_ref.artifact_id,
    ) == fresh.consolidation_manifest_ref.revision


def test_all_three_preparations_exist_before_reuse_lookup(tmp_path, monkeypatch):
    tree, semantic_profile, llm = _inputs(tmp_path)
    original = service_module.ConsolidationPersistenceService.try_reuse_current
    observed = []

    def check_preparations(self, **kwargs):
        observed.append(
            (
                tuple(len(kwargs[name].structured_requests) for name in (
                    "fact_preparation", "event_preparation", "relationship_preparation"
                )),
                kwargs["fact_preparation"].packing_policy,
            )
        )
        return original(self, **kwargs)

    monkeypatch.setattr(
        service_module.ConsolidationPersistenceService,
        "try_reuse_current",
        check_preparations,
    )

    _run(tree, semantic_profile, llm)

    assert observed == [((0, 0, 0), FACT_SEMANTIC_PACKING_V2)]


def test_semantic_invalidation_of_existing_current_executes_provider(tmp_path):
    tree, profile, semantic_profile, planning = _semantic_pair_inputs(tmp_path)
    first_client = FakeLLMClient(_valid_responses(planning))
    first = _run_semantic_pair(tree, profile, semantic_profile, first_client)
    changed = replace(
        semantic_profile,
        temperature=0.9 if semantic_profile.temperature != 0.9 else 0.8,
    )
    second_client = FakeLLMClient(_valid_responses(planning))

    second = _run_semantic_pair(tree, profile, changed, second_client)

    assert second.reused is False
    assert second.semantic_generation_call_count > 0
    assert second_client.call_count == second.semantic_generation_call_count
    assert second.consolidation_manifest_ref.revision > first.consolidation_manifest_ref.revision
    assert (
        tree.pointers.resolve_current(second.current_pointer_ref.artifact_id).target_ref
        == second.consolidation_manifest_ref
    )


def test_fact_prompt_v1_current_is_not_reused_by_v2_production_execution(
    tmp_path, monkeypatch
):
    """Issue #80: an old Fact prompt identity must force a fresh provider path."""
    from short_drama.story import consolidation_semantic as semantic_module

    tree, profile_v2, semantic_profile, _planning = _semantic_pair_inputs(tmp_path)
    profile_v1 = replace(profile_v2, fact=replace(profile_v2.fact, prompt_version=1))
    responses = _valid_responses(_planning)
    with monkeypatch.context() as prior_contract:
        prior_contract.setattr(semantic_module, "A5C_FACT_PROMPT_VERSION", 1)
        first = _run_semantic_pair(
            tree, profile_v1, semantic_profile, FakeLLMClient(responses)
        )

    fresh_client = FakeLLMClient(_valid_responses(_planning))
    second = _run_semantic_pair(tree, profile_v2, semantic_profile, fresh_client)

    assert first.reused is False
    assert second.reused is False
    assert fresh_client.call_count == second.semantic_generation_call_count == 1
    assert second.consolidation_manifest_ref.revision > first.consolidation_manifest_ref.revision


def test_semantic_miss_executes_existing_provider_resolution(tmp_path):
    tree, profile, semantic_profile, planning = _semantic_pair_inputs(tmp_path)
    client = FakeLLMClient(_valid_responses(planning))
    result = _run_semantic_pair(tree, profile, semantic_profile, client)

    assert result.reused is False
    assert client.call_count == result.semantic_generation_call_count == 1
    assert result.fact_semantic_decision_count == 1


def test_corrupt_current_propagates_before_provider(tmp_path):
    tree, profile, semantic_profile, planning = _semantic_pair_inputs(tmp_path)
    first = _run_semantic_pair(
        tree, profile, semantic_profile, FakeLLMClient(_valid_responses(planning))
    )
    revision_before = _highest_revision(
        tree.store,
        CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
        first.consolidation_manifest_ref.artifact_id,
    )
    head_path = tree.pointers._head_path(first.current_pointer_ref.artifact_id)
    head_before = head_path.read_bytes()
    _store_path(
        tree.store,
        CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
        first.consolidation_manifest_ref.artifact_id,
        first.consolidation_manifest_ref.revision,
    ).write_text("{broken", encoding="utf-8")
    provider = ExplodingLLM()

    with pytest.raises(PointerIntegrityError):
        _run_semantic_pair(tree, profile, semantic_profile, provider)
    assert provider.calls == 0
    assert not _store_path(
        tree.store,
        CONSOLIDATION_MANIFEST_ARTIFACT_TYPE,
        first.consolidation_manifest_ref.artifact_id,
        revision_before + 1,
    ).exists()
    assert head_path.read_bytes() == head_before


def test_missing_a4_current_fails_before_provider(tmp_path):
    tree = _build_base_tree(tmp_path)
    from short_drama.story import load_fact_semantic_profile

    llm = ExplodingLLM()
    with pytest.raises(ConsolidationCurrentMissingError):
        _run(tree, load_fact_semantic_profile(), llm)
    assert llm.calls == 0


def test_llm_error_propagates_without_publication(tmp_path, monkeypatch):
    tree, semantic_profile, llm = _inputs(tmp_path)

    def fail_fact(*args, **kwargs):
        raise LLMError("provider failure")

    monkeypatch.setattr(service_module, "resolve_fact_semantic_ambiguity", fail_fact)
    with pytest.raises(LLMError, match="provider failure"):
        _run(tree, semantic_profile, llm)
    assert _current_a5_target(tree) is None


def test_domain_execution_order_and_reporting_projection(tmp_path, monkeypatch):
    tree, semantic_profile, llm = _inputs(tmp_path)
    order = []
    originals = (
        service_module.resolve_fact_semantic_ambiguity,
        service_module.resolve_event_semantic_ambiguity,
        service_module.resolve_relationship_semantic_ambiguity,
    )

    def wrap(name, original):
        def call(*args, **kwargs):
            order.append(name)
            return original(*args, **kwargs)
        return call

    monkeypatch.setattr(service_module, "resolve_fact_semantic_ambiguity", wrap("fact", originals[0]))
    monkeypatch.setattr(service_module, "resolve_event_semantic_ambiguity", wrap("event", originals[1]))
    monkeypatch.setattr(service_module, "resolve_relationship_semantic_ambiguity", wrap("relationship", originals[2]))

    result = _run(tree, semantic_profile, llm)

    assert order == ["fact", "event", "relationship"]
    projection = result.to_dict()
    assert projection == result.to_dict()
    assert projection["reused"] is False
    assert "runtime_config" not in projection
    assert "raw_response" not in projection
    assert "credential" not in projection
