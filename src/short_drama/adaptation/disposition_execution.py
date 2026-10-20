"""B1.3-B2B bounded, read-only disposition-window semantic execution.

This module intentionally stops before B1.3-B3 reconciliation.  It never
creates adapted identities, persists a proposal, or advances a pointer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import FileArtifactStore
from short_drama.foundation import FilePointerStore
from short_drama.llm import (
    GenerationExecutionOptions, LLMClient, LLMInvocationProvenance,
    RuntimeConfig,
)
from short_drama.story.chunking import estimate_tokens

from .disposition_preparation import (
    B12StrategySelection, B13DispositionPreparation, B13PreparationError,
    prepare_b13_disposition_windows, resolve_b12_strategy_selection,
)
from .disposition_semantic import (
    B13WindowProposal, B13WindowProposalError, B13WindowSchemaError,
    B13WindowSemanticBudget, B13WindowSemanticBudgetError,
    B13WindowSemanticPreparation, _mapped_provider_request_bytes,
    build_b13_window_semantic_preparation,
)
from .target import AdaptationSourceSnapshot


B13_MAX_SEMANTIC_GENERATION_ROUNDS = 2


class B13WindowExecutionError(B13PreparationError):
    """A B1.3-B2B execution precondition or authority check failed."""


class B13WindowProvenanceError(B13WindowExecutionError):
    """A provider result is not for the exact prepared B1.3 request."""


class B13WindowGenerationExhaustedError(B13WindowExecutionError):
    """The bounded semantic regeneration policy found no valid proposal."""

    def __init__(
        self, *, window_id: str, request_identity: str, rounds_attempted: int,
        failures: tuple[str, ...],
    ) -> None:
        self.window_id = window_id
        self.request_identity = request_identity
        self.rounds_attempted = rounds_attempted
        self.failures = failures
        super().__init__(
            f"B1.3 semantic generation exhausted after {rounds_attempted} rounds "
            f"for window {window_id!r} ({request_identity}): {'; '.join(failures)}"
        )


@dataclass(frozen=True, slots=True)
class B13WindowSemanticGenerationPolicy:
    """Pinned policy: one proposal plus at most one semantic regeneration."""

    max_rounds: int = B13_MAX_SEMANTIC_GENERATION_ROUNDS

    def __post_init__(self) -> None:
        if self.max_rounds != B13_MAX_SEMANTIC_GENERATION_ROUNDS:
            raise B13WindowExecutionError(
                "B1.3 semantic generation policy is frozen at exactly two rounds"
            )


@dataclass(frozen=True, slots=True)
class B13WindowAcceptedProposal:
    """One accepted local proposal and the exact invocation that produced it."""

    window_id: str
    window_ordinal: int
    request_identity: str
    semantic_request_hash: str
    semantic_profile_identity: dict[str, Any]
    prompt_identity: dict[str, Any]
    output_schema_identity: dict[str, Any]
    rendered_prompt_hash: str
    invocation_provenance: LLMInvocationProvenance
    semantic_rounds: int
    provider_attempts: int
    semantic_failures: tuple[str, ...]
    proposal: B13WindowProposal

    def __post_init__(self) -> None:
        if (not isinstance(self.window_id, str) or not self.window_id
                or not isinstance(self.window_ordinal, int) or self.window_ordinal < 1):
            raise B13WindowExecutionError("accepted window identity is invalid")
        if any(not isinstance(value, str) or len(value) != 64 for value in (
            self.request_identity, self.semantic_request_hash, self.rendered_prompt_hash,
        )):
            raise B13WindowExecutionError("accepted request identities must be SHA-256 hashes")
        if self.semantic_rounds not in (1, 2) or self.provider_attempts < 1:
            raise B13WindowExecutionError("accepted window attempt accounting is invalid")
        if not isinstance(self.proposal, B13WindowProposal):
            raise B13WindowExecutionError("accepted window requires B13WindowProposal")
        if (self.proposal.packet.window_id != self.window_id
                or self.proposal.packet.window_ordinal != self.window_ordinal):
            raise B13WindowExecutionError("accepted proposal/window identity mismatch")


@dataclass(frozen=True, slots=True)
class B13DispositionExecutionResult:
    """Deterministic in-memory B2B output for the later B1.3-B3 reconciler."""

    source_snapshot: AdaptationSourceSnapshot
    strategy_selection: B12StrategySelection
    preparation_identity: str
    accepted_windows: tuple[B13WindowAcceptedProposal, ...]

    def __post_init__(self) -> None:
        accepted = tuple(self.accepted_windows)
        if not isinstance(self.source_snapshot, AdaptationSourceSnapshot):
            raise B13WindowExecutionError("execution result requires exact source snapshot")
        if not isinstance(self.strategy_selection, B12StrategySelection):
            raise B13WindowExecutionError("execution result requires exact B1.2 selection")
        if self.source_snapshot != self.strategy_selection.artifact.source_snapshot:
            raise B13WindowExecutionError("execution result source/strategy mismatch")
        if not isinstance(self.preparation_identity, str) or len(self.preparation_identity) != 64:
            raise B13WindowExecutionError("execution result preparation_identity must be a SHA-256 hash")
        if (not accepted
                or tuple(item.window_ordinal for item in accepted) != tuple(range(1, len(accepted) + 1))
                or len({item.window_id for item in accepted}) != len(accepted)):
            raise B13WindowExecutionError("accepted windows must retain deterministic complete window order")
        if any(item.proposal.packet.source_snapshot != self.source_snapshot
               or item.proposal.packet.strategy_selection != self.strategy_selection
               for item in accepted):
            raise B13WindowExecutionError("accepted proposal has cross-window authority")
        object.__setattr__(self, "accepted_windows", accepted)

    @property
    def semantic_call_count(self) -> int:
        return sum(item.semantic_rounds for item in self.accepted_windows)


def _verify_client_runtime(llm_client: LLMClient, runtime: RuntimeConfig) -> None:
    configured = getattr(llm_client, "runtime_config", None)
    if configured is not None and configured != runtime:
        raise B13WindowExecutionError(
            "injected LLMClient runtime_config differs from the preparation-bound RuntimeConfig"
        )


def _verify_budget(
    semantic: B13WindowSemanticPreparation, runtime: RuntimeConfig,
    options: GenerationExecutionOptions | None, budget: B13WindowSemanticBudget | None,
) -> None:
    if not isinstance(runtime, RuntimeConfig) or runtime != semantic.runtime_config:
        raise B13WindowExecutionError(
            "execution RuntimeConfig differs from the preparation-bound RuntimeConfig"
        )
    if semantic.semantic_profile.structured_output_mode != "json_schema":
        raise B13WindowExecutionError("B1.3 requires structured_output_mode=json_schema")
    if budget is None:
        raise B13WindowSemanticBudgetError("B1.3 provider execution requires an explicit semantic budget")
    if budget.runtime_model_id != runtime.request_model:
        raise B13WindowSemanticBudgetError(
            "semantic budget runtime_model_id does not match the exact runtime request_model"
        )
    body = _mapped_provider_request_bytes(semantic.request, runtime, options)
    measured = estimate_tokens(body.decode("utf-8"))
    if options == semantic.execution_options and (
        body != semantic.provider_request_bytes
        or measured != semantic.measurement.provider_request_estimated_tokens
    ):
        raise B13WindowExecutionError(
            "prepared provider request measurement does not match execution request"
        )
    if measured > budget.input_ceiling_tokens:
        raise B13WindowSemanticBudgetError(
            "B1.3 mapped provider request exceeds input ceiling; refusing to truncate"
        )
    if measured + semantic.semantic_profile.max_output_tokens > budget.model_context_tokens:
        raise B13WindowSemanticBudgetError(
            "B1.3 mapped provider request plus reserved output exceeds runtime model context"
        )


def _verify_provenance(
    provenance: LLMInvocationProvenance, semantic: B13WindowSemanticPreparation,
    runtime: RuntimeConfig,
) -> None:
    request = semantic.request
    checks = (
        (provenance.semantic_profile_id, semantic.semantic_profile.profile_id),
        (provenance.semantic_profile_hash, semantic.semantic_profile.semantic_profile_hash),
        (provenance.prompt_id, request.rendered_prompt.prompt_id),
        (provenance.prompt_version, request.rendered_prompt.prompt_version),
        (provenance.prompt_content_hash, request.rendered_prompt.prompt_content_hash),
        (provenance.rendered_prompt_hash, request.rendered_prompt.rendered_prompt_hash),
        (provenance.output_schema_id, semantic.output_schema.schema_id),
        (provenance.output_schema_version, semantic.output_schema.schema_version),
        (provenance.output_schema_hash, semantic.output_schema.schema_hash),
        (provenance.request_hash, request.request_hash),
        (provenance.provider_family, runtime.provider_family),
        (provenance.model, runtime.request_model),
    )
    if any(got != expected for got, expected in checks):
        raise B13WindowProvenanceError(
            "B1.3 provider provenance does not match the exact prepared request"
        )


def _revalidate_authority(
    store: FileArtifactStore, pointers: FilePointerStore,
    preparation: B13DispositionPreparation,
) -> None:
    """Recheck A5/A6, then the chosen immutable B1.2 lineage.

    An explicit historical B1.2 ref is deliberately reloaded as that ref; a
    selection originating at B1.2 CURRENT additionally requires the exact
    pointer revision observed at preparation time to remain CURRENT.
    """
    source = preparation.source_snapshot
    try:
        current = AdaptationSourceSnapshot.from_store(
            store, pointers, source.story_analysis_manifest_ref
        )
    except Exception as exc:
        raise B13WindowExecutionError("cannot revalidate exact A5/A6 CURRENT authority") from exc
    if current != source:
        raise B13WindowExecutionError("exact A5/A6 source authority changed during B1.3 execution")
    target_id = preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id
    try:
        pinned = resolve_b12_strategy_selection(
            store=store, pointers=pointers, source_snapshot=source,
            adaptation_target_id=target_id,
            strategy_ref=preparation.strategy_selection.strategy_ref,
        )
    except Exception as exc:
        raise B13WindowExecutionError("selected immutable B1.2 strategy lineage is no longer valid") from exc
    selection = preparation.strategy_selection
    if pinned.strategy_ref != selection.strategy_ref or pinned.artifact != selection.artifact:
        raise B13WindowExecutionError("selected immutable B1.2 strategy lineage changed during execution")
    if selection.current_pointer_ref is not None:
        try:
            current_selection = resolve_b12_strategy_selection(
                store=store, pointers=pointers, source_snapshot=source,
                adaptation_target_id=target_id,
            )
        except Exception as exc:
            raise B13WindowExecutionError("B1.2 CURRENT strategy authority changed during execution") from exc
        if (current_selection.current_pointer_ref != selection.current_pointer_ref
                or current_selection.strategy_ref != selection.strategy_ref):
            raise B13WindowExecutionError("B1.2 CURRENT strategy authority changed during execution")


def _verify_preflight(
    *, store: FileArtifactStore, pointers: FilePointerStore,
    preparation: B13DispositionPreparation,
    semantic_preparations: tuple[B13WindowSemanticPreparation, ...],
    budgets: tuple[B13WindowSemanticBudget | None, ...], runtime: RuntimeConfig,
) -> None:
    if not isinstance(preparation, B13DispositionPreparation):
        raise B13WindowExecutionError("B1.3 execution requires complete B1.3-B1 preparation")
    if (len(semantic_preparations) != len(preparation.packets)
            or len(budgets) != len(preparation.packets)):
        raise B13WindowExecutionError("B1.3 execution requires one request and budget per prepared window")
    # Rebuild the exact source/strategy/window packets before *any* client call.
    # It verifies complete A6 ownership and does not touch the provider or disk.
    try:
        rebuilt = prepare_b13_disposition_windows(
            store=store, pointers=pointers, source_snapshot=preparation.source_snapshot,
            adaptation_target_id=preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id,
            strategy_ref=(None if preparation.strategy_selection.current_pointer_ref is not None
                          else preparation.strategy_selection.strategy_ref),
        )
    except Exception as exc:
        raise B13WindowExecutionError("cannot revalidate complete B1.3 window preparation") from exc
    if rebuilt != preparation:
        raise B13WindowExecutionError("prepared B1.3 windows no longer match exact authority")
    _revalidate_authority(store, pointers, preparation)
    retry_options = GenerationExecutionOptions(prompt_context_reuse="disabled")
    for packet, semantic, budget in zip(preparation.packets, semantic_preparations, budgets):
        if not isinstance(semantic, B13WindowSemanticPreparation) or semantic.packet != packet:
            raise B13WindowExecutionError("window semantic request does not match prepared window identity")
        # Rebuilding catches forged/tampered profile, prompt/schema, request
        # hashes and initial adapter-mapped measurement before the first call.
        expected = build_b13_window_semantic_preparation(
            packet, runtime_config=runtime, execution_options=semantic.execution_options,
        )
        if semantic != expected:
            raise B13WindowExecutionError("window semantic request or pinned assets do not match preparation")
        _verify_budget(semantic, runtime, semantic.execution_options, budget)
        # The retry body can differ (notably cache_prompt=false), so measure it
        # and reserve output against the same declared runtime n_ctx now.
        _verify_budget(semantic, runtime, retry_options, budget)


def execute_b13_disposition_windows(
    *, store: FileArtifactStore, pointers: FilePointerStore,
    preparation: B13DispositionPreparation,
    semantic_preparations: tuple[B13WindowSemanticPreparation, ...],
    budgets: tuple[B13WindowSemanticBudget | None, ...],
    runtime_config: RuntimeConfig, llm_client: LLMClient,
    policy: B13WindowSemanticGenerationPolicy = B13WindowSemanticGenerationPolicy(),
) -> B13DispositionExecutionResult:
    """Execute complete B1.3 windows in order and retain accepted local proposals.

    Technical adapter failures are intentionally allowed to propagate: adapter
    retry is transport-owned and never consumes a semantic regeneration round.
    Schema output failures are also not regenerated here; only explicit
    schema-valid B1.3 proposal invalidity gets the one bounded regeneration.
    """
    if not isinstance(llm_client, LLMClient) or not isinstance(policy, B13WindowSemanticGenerationPolicy):
        raise B13WindowExecutionError("B1.3 execution requires injected LLMClient and bounded policy")
    semantic_preparations = tuple(semantic_preparations)
    budgets = tuple(budgets)
    _verify_preflight(
        store=store, pointers=pointers, preparation=preparation,
        semantic_preparations=semantic_preparations, budgets=budgets,
        runtime=runtime_config,
    )
    _verify_client_runtime(llm_client, runtime_config)
    accepted: list[B13WindowAcceptedProposal] = []
    retry_options = GenerationExecutionOptions(prompt_context_reuse="disabled")
    for semantic, budget in zip(semantic_preparations, budgets):
        failures: list[str] = []
        for round_number in range(1, policy.max_rounds + 1):
            options = semantic.execution_options if round_number == 1 else retry_options
            _verify_budget(semantic, runtime_config, options, budget)
            _revalidate_authority(store, pointers, preparation)
            result = llm_client.generate_structured(
                semantic.request.rendered_prompt, semantic.output_schema,
                semantic.semantic_profile, execution_options=options,
            )
            # Authority races win over proposal acceptance or retry decisions.
            _revalidate_authority(store, pointers, preparation)
            _verify_provenance(result.provenance, semantic, runtime_config)
            try:
                proposal = semantic.decode_and_validate_proposal(result.parsed_json)
            except B13WindowProposalError as exc:
                failures.append(str(exc)[:500])
                continue
            except B13WindowSchemaError:
                raise
            accepted.append(B13WindowAcceptedProposal(
                window_id=semantic.packet.window_id,
                window_ordinal=semantic.packet.window_ordinal,
                request_identity=semantic.request_identity,
                semantic_request_hash=semantic.request.request_hash,
                semantic_profile_identity={
                    "profile_id": semantic.semantic_profile.profile_id,
                    "profile_hash": semantic.semantic_profile.semantic_profile_hash,
                },
                prompt_identity={
                    "prompt_id": semantic.request.rendered_prompt.prompt_id,
                    "prompt_version": semantic.request.rendered_prompt.prompt_version,
                    "prompt_content_hash": semantic.request.rendered_prompt.prompt_content_hash,
                },
                output_schema_identity={
                    "schema_id": semantic.output_schema.schema_id,
                    "schema_version": semantic.output_schema.schema_version,
                    "schema_hash": semantic.output_schema.schema_hash,
                },
                rendered_prompt_hash=semantic.request.rendered_prompt.rendered_prompt_hash,
                invocation_provenance=result.provenance,
                semantic_rounds=round_number,
                provider_attempts=result.attempts,
                semantic_failures=tuple(failures), proposal=proposal,
            ))
            break
        else:
            raise B13WindowGenerationExhaustedError(
                window_id=semantic.packet.window_id,
                request_identity=semantic.request_identity,
                rounds_attempted=policy.max_rounds, failures=tuple(failures),
            )
    return B13DispositionExecutionResult(
        preparation.source_snapshot, preparation.strategy_selection,
        preparation.preparation_identity, tuple(accepted),
    )


__all__ = [
    "B13_MAX_SEMANTIC_GENERATION_ROUNDS", "B13WindowExecutionError",
    "B13WindowProvenanceError", "B13WindowGenerationExhaustedError",
    "B13WindowSemanticGenerationPolicy", "B13WindowAcceptedProposal",
    "B13DispositionExecutionResult", "execute_b13_disposition_windows",
]
