"""B1.2-B2 tests use an injected fake client; they never open a provider route."""
from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.adaptation import (
    B12SemanticBudget, B12SemanticBudgetError, B12SemanticGenerationExhaustedError,
    B12SemanticProvenanceError, build_b12_semantic_preparation, execute_b12_strategy,
)
from short_drama.llm import (
    LLMClient, LLMInvocationProvenance, StructuredGenerationResult,
    build_structured_request,
)

from test_adaptation_strategy import _context_strategy, _proposal, _semantic_preparation


class _FakeClient(LLMClient):
    supported_structured_output_modes = frozenset({"json_schema"})

    def __init__(self, results, runtime):
        self._results = list(results)
        self.runtime = runtime
        self.calls = []

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile, *, execution_options=None):
        self.calls.append(execution_options)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        parsed, forged = result
        provenance = LLMInvocationProvenance(
            provider_family=self.runtime.provider_family,
            model=self.runtime.request_model,
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
