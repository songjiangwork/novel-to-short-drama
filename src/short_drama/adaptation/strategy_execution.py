"""B1.2-B2 bounded semantic execution and immutable strategy persistence.

The only semantic acceptance seam is
``B12SemanticPreparation.decode_and_validate_proposal``.  This module keeps
adapter-owned technical retry separate from the two explicitly pinned semantic
rounds, and never publishes an approval record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from short_drama.artifacts import ArtifactError, ArtifactNotFoundError, ArtifactRef, FileArtifactStore, ImmutableArtifactEnvelope
from short_drama.foundation import FilePointerStore, LineageRef, PointerKind, PointerNotFoundError
from short_drama.guidance import DirectionStrength
from short_drama.guidance.projection import load_user_direction
from short_drama.llm import GenerationExecutionOptions, LLMClient, LLMInvocationProvenance, RuntimeConfig
from short_drama.story.chunking import estimate_tokens

from .strategy import AdaptationStrategy, AdaptationStrategyError
from .strategy_packet import B12Preparation
from .strategy_semantic import B12SemanticBudget, B12SemanticBudgetError, B12SemanticPreparation, _mapped_provider_request_bytes
from .target import AdaptationSourceSnapshot


B12_STRATEGY_ARTIFACT_TYPE = "adaptation_strategy"
B12_STRATEGY_ARTIFACT_SCHEMA_VERSION = 1
B12_MAX_SEMANTIC_GENERATION_ROUNDS = 2


class B12SemanticExecutionError(AdaptationStrategyError):
    """A deterministic B1.2 execution precondition failed."""


class B12SemanticProvenanceError(B12SemanticExecutionError):
    """The injected client result is not for the exact prepared request."""


class B12SemanticGenerationExhaustedError(B12SemanticExecutionError):
    """The explicit bounded semantic rounds produced no acceptable strategy."""

    def __init__(self, *, request_identity: str, rounds_attempted: int, failures: tuple[str, ...]) -> None:
        self.request_identity = request_identity
        self.rounds_attempted = rounds_attempted
        self.failures = failures
        super().__init__(
            f"B1.2 semantic generation exhausted after {rounds_attempted} rounds "
            f"for {request_identity}: {'; '.join(failures)}"
        )


@dataclass(frozen=True, slots=True)
class B12SemanticGenerationPolicy:
    """Pinned B1.2 policy: first call plus at most one semantic regeneration."""

    max_rounds: int = B12_MAX_SEMANTIC_GENERATION_ROUNDS

    def __post_init__(self) -> None:
        if self.max_rounds != B12_MAX_SEMANTIC_GENERATION_ROUNDS:
            raise B12SemanticExecutionError(
                "B1.2 semantic generation policy is frozen at exactly two rounds"
            )


@dataclass(frozen=True, slots=True)
class B12StrategyArtifact:
    """Persisted generated strategy plus all exact B1.2 lineage/identity pins."""

    request_identity: str
    semantic_request_hash: str
    source_snapshot: AdaptationSourceSnapshot
    effective_b1_direction_hash: str
    semantic_view_hash: str
    prompt_identity: dict[str, Any]
    output_schema_identity: dict[str, Any]
    semantic_profile_identity: dict[str, Any]
    rendered_request_hash: str
    invocation_provenance: dict[str, Any]
    semantic_rounds: int
    lineage: tuple[LineageRef, ...]
    effective_direction_refs: tuple[ArtifactRef, ...]
    requires_human_review: bool
    review_reasons: tuple[str, ...]
    review_required_direction_refs: tuple[ArtifactRef, ...]
    strategy: AdaptationStrategy

    def __post_init__(self) -> None:
        if not isinstance(self.request_identity, str) or len(self.request_identity) != 64:
            raise B12SemanticExecutionError("strategy artifact request_identity must be a SHA-256 hash")
        if not isinstance(self.semantic_request_hash, str) or len(self.semantic_request_hash) != 64:
            raise B12SemanticExecutionError("strategy artifact semantic_request_hash must be a SHA-256 hash")
        if not isinstance(self.source_snapshot, AdaptationSourceSnapshot):
            raise B12SemanticExecutionError("strategy artifact requires an exact AdaptationSourceSnapshot")
        if not isinstance(self.strategy, AdaptationStrategy):
            raise B12SemanticExecutionError("strategy artifact requires AdaptationStrategy")
        if self.strategy.resolved_target.source_snapshot != self.source_snapshot:
            raise B12SemanticExecutionError("strategy artifact target/source snapshot mismatch")
        if self.semantic_rounds not in (1, 2):
            raise B12SemanticExecutionError("strategy artifact semantic_rounds must be 1 or 2")
        lineage = tuple(self.lineage)
        if any(not isinstance(item, LineageRef) for item in lineage) or len({item.role for item in lineage}) != len(lineage):
            raise B12SemanticExecutionError("strategy artifact lineage must have unique typed roles")
        object.__setattr__(self, "lineage", tuple(sorted(lineage, key=LineageRef.sort_key)))
        effective_refs = tuple(self.effective_direction_refs)
        if any(not isinstance(ref, ArtifactRef) or ref.artifact_type != "user_direction" for ref in effective_refs) or len(set(effective_refs)) != len(effective_refs):
            raise B12SemanticExecutionError("effective_direction_refs must be unique user_direction refs")
        object.__setattr__(self, "effective_direction_refs", tuple(sorted(effective_refs, key=lambda x: (x.artifact_id, x.revision, x.content_hash))))
        if not isinstance(self.requires_human_review, bool):
            raise B12SemanticExecutionError("strategy artifact requires_human_review must be boolean")
        reasons = tuple(self.review_reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons) or len(set(reasons)) != len(reasons):
            raise B12SemanticExecutionError("strategy artifact review_reasons must be unique non-empty text")
        if bool(reasons) != self.requires_human_review:
            raise B12SemanticExecutionError("strategy artifact review flag/reasons disagree")
        object.__setattr__(self, "review_reasons", tuple(sorted(reasons)))
        refs = tuple(self.review_required_direction_refs)
        if any(not isinstance(ref, ArtifactRef) or ref.artifact_type != "user_direction" for ref in refs):
            raise B12SemanticExecutionError("review_required_direction_refs must be user_direction refs")
        if len(set(refs)) != len(refs):
            raise B12SemanticExecutionError("review_required_direction_refs must be unique")
        object.__setattr__(self, "review_required_direction_refs", tuple(sorted(refs, key=lambda x: (x.artifact_id, x.revision, x.content_hash))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": B12_STRATEGY_ARTIFACT_SCHEMA_VERSION,
            "request_identity": self.request_identity,
            "semantic_request_hash": self.semantic_request_hash,
            "source_snapshot": self.source_snapshot.to_dict(),
            "effective_b1_direction_hash": self.effective_b1_direction_hash,
            "semantic_view_hash": self.semantic_view_hash,
            "prompt_identity": self.prompt_identity,
            "output_schema_identity": self.output_schema_identity,
            "semantic_profile_identity": self.semantic_profile_identity,
            "rendered_request_hash": self.rendered_request_hash,
            "invocation_provenance": self.invocation_provenance,
            "semantic_rounds": self.semantic_rounds,
            "lineage": [item.to_dict() for item in self.lineage],
            "effective_direction_refs": [ref.to_dict() for ref in self.effective_direction_refs],
            "requires_human_review": self.requires_human_review,
            "review_reasons": list(self.review_reasons),
            "review_required_direction_refs": [ref.to_dict() for ref in self.review_required_direction_refs],
            "strategy": self.strategy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "B12StrategyArtifact":
        fields = {
            "schema_version", "request_identity", "semantic_request_hash", "source_snapshot",
            "effective_b1_direction_hash", "semantic_view_hash", "prompt_identity",
            "output_schema_identity", "semantic_profile_identity", "rendered_request_hash",
            "invocation_provenance", "semantic_rounds", "lineage", "effective_direction_refs", "requires_human_review", "review_reasons",
            "review_required_direction_refs", "strategy",
        }
        if not isinstance(value, dict) or set(value) != fields or value["schema_version"] != B12_STRATEGY_ARTIFACT_SCHEMA_VERSION:
            raise B12SemanticExecutionError("invalid persisted B1.2 strategy artifact shape")
        for name in ("prompt_identity", "output_schema_identity", "semantic_profile_identity", "invocation_provenance"):
            if not isinstance(value[name], dict):
                raise B12SemanticExecutionError(f"strategy artifact {name} must be an object")
        if not isinstance(value["review_required_direction_refs"], list) or not isinstance(value["review_reasons"], list) or not isinstance(value["lineage"], list) or not isinstance(value["effective_direction_refs"], list):
            raise B12SemanticExecutionError("strategy artifact review fields must be arrays")
        return cls(
            request_identity=value["request_identity"], semantic_request_hash=value["semantic_request_hash"],
            source_snapshot=AdaptationSourceSnapshot.from_dict(value["source_snapshot"]),
            effective_b1_direction_hash=value["effective_b1_direction_hash"], semantic_view_hash=value["semantic_view_hash"],
            prompt_identity=value["prompt_identity"], output_schema_identity=value["output_schema_identity"],
            semantic_profile_identity=value["semantic_profile_identity"], rendered_request_hash=value["rendered_request_hash"],
            invocation_provenance=value["invocation_provenance"], semantic_rounds=value["semantic_rounds"],
            lineage=tuple(LineageRef.from_dict(x) for x in value["lineage"]),
            effective_direction_refs=tuple(ArtifactRef.from_dict(x) for x in value["effective_direction_refs"]),
            requires_human_review=value["requires_human_review"], review_reasons=tuple(value["review_reasons"]),
            review_required_direction_refs=tuple(ArtifactRef.from_dict(x) for x in value["review_required_direction_refs"]),
            strategy=AdaptationStrategy.from_dict(value["strategy"]),
        )


@dataclass(frozen=True, slots=True)
class B12StrategyPublication:
    strategy_ref: ArtifactRef
    current_pointer_ref: ArtifactRef
    artifact: B12StrategyArtifact
    reused: bool = False


def b12_strategy_artifact_id(project_id: str, adaptation_target_id: str) -> str:
    return f"{project_id}.{adaptation_target_id}.b12.strategy"


def b12_strategy_pointer_id(project_id: str, adaptation_target_id: str) -> str:
    return f"{project_id}.b12.{adaptation_target_id}.strategy"


def _provenance_checks(provenance: LLMInvocationProvenance, semantic: B12SemanticPreparation, runtime: RuntimeConfig) -> None:
    request = semantic.request
    # Runtime backend/model are recorded audit data, not semantic identity (the
    # A6 convention).  ``runtime`` is checked against preparation separately
    # before each call; it must not make a valid stored strategy unreusable.
    del runtime
    checks = (
        (provenance.semantic_profile_id, request.semantic_profile.profile_id),
        (provenance.semantic_profile_hash, request.semantic_profile.semantic_profile_hash),
        (provenance.prompt_id, request.rendered_prompt.prompt_id),
        (provenance.prompt_version, request.rendered_prompt.prompt_version),
        (provenance.prompt_content_hash, request.rendered_prompt.prompt_content_hash),
        (provenance.rendered_prompt_hash, request.rendered_prompt.rendered_prompt_hash),
        (provenance.output_schema_id, request.output_schema.schema_id),
        (provenance.output_schema_version, request.output_schema.schema_version),
        (provenance.output_schema_hash, request.output_schema.schema_hash),
        (provenance.request_hash, request.request_hash),
    )
    if any(got != expected for got, expected in checks):
        raise B12SemanticProvenanceError("B1.2 provider provenance does not match the exact prepared request")


def _verify_execution_request(semantic: B12SemanticPreparation, runtime: RuntimeConfig, options: GenerationExecutionOptions | None, budget: B12SemanticBudget) -> None:
    if not isinstance(runtime, RuntimeConfig):
        raise B12SemanticExecutionError("B1.2 execution requires explicit RuntimeConfig")
    if runtime != semantic.runtime_config:
        raise B12SemanticExecutionError("execution RuntimeConfig differs from the preparation-bound RuntimeConfig")
    if semantic.semantic_profile.structured_output_mode != "json_schema":
        raise B12SemanticExecutionError("B1.2 requires structured_output_mode=json_schema")
    body = _mapped_provider_request_bytes(semantic.request, runtime, options)
    measured = estimate_tokens(body.decode("utf-8"))
    if options == semantic.execution_options:
        if body != semantic.provider_request_bytes or measured != semantic.measurement.provider_request_estimated_tokens:
            raise B12SemanticExecutionError("prepared provider request measurement does not match execution request")
    # The budget is deliberately checked against this exact round's request;
    # cache-disabled regeneration may have different mapped bytes.
    if budget is None:
        raise B12SemanticBudgetError("B1.2 provider execution requires an explicit semantic budget")
    if budget.runtime_model_id != runtime.request_model:
        raise B12SemanticBudgetError("semantic budget runtime_model_id does not match execution RuntimeConfig")
    if measured > budget.input_ceiling_tokens:
        raise B12SemanticBudgetError("B1.2 mapped provider request exceeds input ceiling; refusing to truncate")
    if measured + semantic.semantic_profile.max_output_tokens > budget.model_context_tokens:
        raise B12SemanticBudgetError("B1.2 mapped provider request plus reserved output exceeds runtime model context")


def _ensure_source_current(store: FileArtifactStore, pointers: FilePointerStore, semantic: B12SemanticPreparation) -> None:
    current = AdaptationSourceSnapshot.from_store(
        store, pointers, semantic.preparation.packet.source_snapshot.story_analysis_manifest_ref
    )
    if current != semantic.preparation.packet.source_snapshot:
        raise B12SemanticExecutionError("exact A5/A6 source authority changed during B1.2 execution")


def _verify_effective_direction_authority(store: FileArtifactStore, semantic: B12SemanticPreparation) -> tuple[ArtifactRef, ...]:
    """Resolve every exact direction pin; no stored guidance is trusted by hash alone."""
    effective = semantic.preparation.packet.effective_directions.get("effective_set", {})
    raw_refs = effective.get("direction_refs", []) if isinstance(effective, dict) else []
    try:
        refs = tuple(ArtifactRef.from_dict(item) for item in raw_refs)
        for ref in refs:
            load_user_direction(store, ref)
    except Exception as exc:
        raise B12SemanticExecutionError("effective B1 user-direction authority is missing or invalid") from exc
    return refs


def _review_required_direction_refs(store: FileArtifactStore, semantic: B12SemanticPreparation, strategy: AdaptationStrategy) -> tuple[ArtifactRef, ...]:
    refs = _verify_effective_direction_authority(store, semantic)
    # Python does not infer whether natural language was fulfilled.  Every hard
    # B1 direction remains a review item, as do explicit conflicts/divergences.
    hard = tuple(ref for ref in refs if load_user_direction(store, ref).strength is DirectionStrength.MUST or load_user_direction(store, ref).strength is DirectionStrength.MUST_NOT)
    return hard if (hard or strategy.explicit_conflicts or strategy.deliberate_divergences) else ()


def _strategy_lineage(semantic: B12SemanticPreparation, direction_refs: tuple[ArtifactRef, ...]) -> tuple[LineageRef, ...]:
    snapshot = semantic.preparation.packet.source_snapshot
    source_refs = (
        ("a6_manifest", snapshot.story_analysis_manifest_ref),
        ("a5_consolidation_manifest", snapshot.consolidation_manifest_ref),
        ("a4_entity_map", snapshot.entity_map_ref),
        ("a4_character_registry", snapshot.canonical_character_registry_ref),
        ("a4_location_registry", snapshot.canonical_location_registry_ref),
        ("a5_canonical_fact_set", snapshot.canonical_fact_set_ref),
        ("a5_canonical_event_set", snapshot.canonical_event_set_ref),
        ("a5_canonical_relationship_set", snapshot.canonical_relationship_set_ref),
        ("a6_arc_analysis", snapshot.arc_analysis_ref),
    )
    return tuple(sorted((
        LineageRef(role, ref)
        for role, ref in (*source_refs, *((f"b1_effective_direction_{index:04d}", ref) for index, ref in enumerate(direction_refs)))
    ), key=LineageRef.sort_key))


class B12StrategyPersistenceService:
    """Small CURRENT-only B1.2 persistence adapter; never emits approval."""

    def __init__(self, store: FileArtifactStore, pointers: FilePointerStore) -> None:
        self._store, self._pointers = store, pointers

    def _project_and_ids(self, semantic: B12SemanticPreparation) -> tuple[str, str, str]:
        from short_drama.story.story_analysis import StoryAnalysisManifest
        ref = semantic.preparation.packet.source_snapshot.story_analysis_manifest_ref
        try:
            payload = self._store.get_ref(ref).payload
            manifest = StoryAnalysisManifest.from_dict(payload)
        except Exception as exc:  # exact ref/hash verification belongs to store
            raise B12SemanticExecutionError("cannot derive B1.2 project identity from exact A6 manifest") from exc
        target_id = semantic.preparation.packet.resolved_target.adaptation_target_id
        return manifest.project_id, b12_strategy_artifact_id(manifest.project_id, target_id), b12_strategy_pointer_id(manifest.project_id, target_id)

    def try_reuse_current(self, semantic: B12SemanticPreparation) -> B12StrategyPublication | None:
        project_id, artifact_id, pointer_id = self._project_and_ids(semantic)
        del project_id
        try:
            pointer_ref = self._pointers.resolve_current_pointer_ref(pointer_id)
            pointer = self._pointers.resolve_current(pointer_id)
        except PointerNotFoundError:
            return None
        if pointer.pointer_kind is not PointerKind.CURRENT or pointer.target_ref.artifact_type != B12_STRATEGY_ARTIFACT_TYPE or pointer.target_ref.artifact_id != artifact_id:
            raise B12SemanticExecutionError("B1.2 CURRENT pointer has invalid identity")
        try:
            envelope = self._store.get_ref(pointer.target_ref)
        except ArtifactError as exc:
            raise B12SemanticExecutionError("B1.2 CURRENT strategy artifact fails integrity verification") from exc
        if envelope.schema_version != B12_STRATEGY_ARTIFACT_SCHEMA_VERSION or not isinstance(envelope.payload, dict):
            raise B12SemanticExecutionError("B1.2 CURRENT strategy artifact has unsupported schema")
        artifact = B12StrategyArtifact.from_dict(envelope.payload)
        self._validate_reusable_artifact(artifact, semantic)
        return B12StrategyPublication(pointer.target_ref, pointer_ref, artifact, reused=True)

    def _validate_reusable_artifact(self, artifact: B12StrategyArtifact, semantic: B12SemanticPreparation) -> None:
        if artifact.request_identity != semantic.request_identity or artifact.semantic_request_hash != semantic.request.request_hash:
            raise B12SemanticExecutionError("B1.2 CURRENT strategy is not an exact semantic request match")
        if artifact.source_snapshot != semantic.preparation.packet.source_snapshot:
            raise B12SemanticExecutionError("B1.2 CURRENT strategy has cross-project or stale source lineage")
        if artifact.strategy.resolved_target != semantic.preparation.packet.resolved_target:
            raise B12SemanticExecutionError("B1.2 CURRENT strategy has wrong resolved target")
        if artifact.effective_b1_direction_hash != semantic.preparation.measurement.effective_b1_direction_hash:
            raise B12SemanticExecutionError("B1.2 CURRENT strategy has wrong effective direction identity")
        if artifact.semantic_view_hash != semantic.preparation.semantic_view.content_hash():
            raise B12SemanticExecutionError("B1.2 CURRENT strategy has wrong semantic view identity")
        direction_refs = _verify_effective_direction_authority(self._store, semantic)
        if artifact.effective_direction_refs != direction_refs or artifact.lineage != _strategy_lineage(semantic, direction_refs):
            raise B12SemanticExecutionError("B1.2 CURRENT strategy has wrong exact lineage")
        _provenance_checks(_provenance_from_dict(artifact.invocation_provenance), semantic, semantic.runtime_config)
        semantic.decode_and_validate_proposal(_without_target(artifact.strategy))

    def publish(self, semantic: B12SemanticPreparation, artifact: B12StrategyArtifact,
                *, source_guard: Callable[[], None] | None = None) -> B12StrategyPublication:
        _project, artifact_id, pointer_id = self._project_and_ids(semantic)
        try:
            old_pointer_ref = self._pointers.resolve_current_pointer_ref(pointer_id)
            old = self._pointers.resolve_current(pointer_id)
            if old.pointer_kind is not PointerKind.CURRENT:
                raise B12SemanticExecutionError("B1.2 pointer must be a stage-local CURRENT pointer")
            revision = old.target_ref.revision + 1
        except PointerNotFoundError:
            old_pointer_ref, revision = None, 1
        while True:
            try:
                self._store.get(B12_STRATEGY_ARTIFACT_TYPE, artifact_id, revision)
                revision += 1
            except ArtifactNotFoundError:
                break
            except ArtifactError as exc:
                raise B12SemanticExecutionError("cannot allocate immutable B1.2 strategy revision") from exc
        envelope = ImmutableArtifactEnvelope.create(
            artifact_type=B12_STRATEGY_ARTIFACT_TYPE, artifact_id=artifact_id, revision=revision,
            schema_version=B12_STRATEGY_ARTIFACT_SCHEMA_VERSION, payload=artifact.to_dict(),
        )
        try:
            strategy_ref = self._store.put(envelope)
        except ArtifactError as exc:
            raise B12SemanticExecutionError("failed to persist immutable B1.2 strategy") from exc
        if source_guard is not None:
            source_guard()
        try:
            pointer_ref = self._pointers.compare_and_set(pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT, expected_pointer_ref=old_pointer_ref, target_ref=strategy_ref)
        except Exception as exc:
            raise B12SemanticExecutionError("B1.2 CURRENT publication CAS failed") from exc
        return B12StrategyPublication(strategy_ref, pointer_ref, artifact)


def _without_target(strategy: AdaptationStrategy) -> dict[str, Any]:
    value = strategy.to_dict()
    value.pop("resolved_target")
    return value


def _provenance_from_dict(value: dict[str, Any]) -> LLMInvocationProvenance:
    try:
        return LLMInvocationProvenance(**value)
    except Exception as exc:
        raise B12SemanticExecutionError("persisted B1.2 invocation provenance is invalid") from exc


def execute_b12_strategy(
    *,
    store: FileArtifactStore,
    pointers: FilePointerStore,
    semantic: B12SemanticPreparation,
    runtime_config: RuntimeConfig,
    budget: B12SemanticBudget | None,
    llm_client: LLMClient,
    policy: B12SemanticGenerationPolicy = B12SemanticGenerationPolicy(),
) -> B12StrategyPublication:
    """Reuse or execute one B1.2 request, then publish only accepted strategy."""
    if not isinstance(semantic, B12SemanticPreparation) or not isinstance(llm_client, LLMClient):
        raise B12SemanticExecutionError("B1.2 execution requires B12SemanticPreparation and injected LLMClient")
    if not isinstance(policy, B12SemanticGenerationPolicy):
        raise B12SemanticExecutionError("B1.2 execution requires an explicit bounded semantic generation policy")
    # All deterministic failures happen before either reuse/provider work.
    _verify_execution_request(semantic, runtime_config, semantic.execution_options, budget)  # type: ignore[arg-type]
    _ensure_source_current(store, pointers, semantic)
    _verify_effective_direction_authority(store, semantic)
    persistence = B12StrategyPersistenceService(store, pointers)
    reused = persistence.try_reuse_current(semantic)
    if reused is not None:
        _ensure_source_current(store, pointers, semantic)
        _verify_effective_direction_authority(store, semantic)
        return reused

    failures: list[str] = []
    for round_number in range(1, policy.max_rounds + 1):
        options = semantic.execution_options if round_number == 1 else GenerationExecutionOptions(prompt_context_reuse="disabled")
        _verify_execution_request(semantic, runtime_config, options, budget)  # type: ignore[arg-type]
        _ensure_source_current(store, pointers, semantic)
        _verify_effective_direction_authority(store, semantic)
        result = llm_client.generate_structured(
            semantic.request.rendered_prompt, semantic.output_schema, semantic.semantic_profile,
            execution_options=options,
        )
        _provenance_checks(result.provenance, semantic, runtime_config)
        try:
            strategy = semantic.decode_and_validate_proposal(result.parsed_json)
        except AdaptationStrategyError as exc:
            failures.append(str(exc)[:500])
            continue
        # Upstream A5/A6 authority is re-read after the provider call and before
        # writing any B1 owned bytes, closing the CURRENT race.
        _ensure_source_current(store, pointers, semantic)
        review_refs = _review_required_direction_refs(store, semantic, strategy)
        direction_refs = _verify_effective_direction_authority(store, semantic)
        reasons = tuple(reason for reason, present in (
            ("deliberate_divergence", bool(strategy.deliberate_divergences)),
            ("explicit_conflict", bool(strategy.explicit_conflicts)),
            ("hard_user_direction_semantic_review", bool(review_refs)),
        ) if present)
        artifact = B12StrategyArtifact(
            request_identity=semantic.request_identity,
            semantic_request_hash=semantic.request.request_hash,
            source_snapshot=semantic.preparation.packet.source_snapshot,
            effective_b1_direction_hash=semantic.preparation.measurement.effective_b1_direction_hash,
            semantic_view_hash=semantic.preparation.semantic_view.content_hash(),
            prompt_identity={"prompt_id": semantic.request.rendered_prompt.prompt_id, "prompt_version": semantic.request.rendered_prompt.prompt_version, "prompt_content_hash": semantic.request.rendered_prompt.prompt_content_hash},
            output_schema_identity={"schema_id": semantic.output_schema.schema_id, "schema_version": semantic.output_schema.schema_version, "schema_content_hash": semantic.output_schema.schema_hash},
            semantic_profile_identity={"profile_id": semantic.semantic_profile.profile_id, "profile_hash": semantic.semantic_profile.semantic_profile_hash},
            rendered_request_hash=semantic.request.request_hash,
            invocation_provenance=result.provenance.to_dict(), semantic_rounds=round_number,
            lineage=_strategy_lineage(semantic, direction_refs), effective_direction_refs=direction_refs,
            requires_human_review=bool(reasons), review_reasons=reasons,
            review_required_direction_refs=review_refs, strategy=strategy,
        )
        return persistence.publish(semantic, artifact, source_guard=lambda: _ensure_source_current(store, pointers, semantic))
    raise B12SemanticGenerationExhaustedError(
        request_identity=semantic.request_identity, rounds_attempted=policy.max_rounds, failures=tuple(failures),
    )
