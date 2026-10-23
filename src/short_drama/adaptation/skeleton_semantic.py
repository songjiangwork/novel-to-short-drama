"""B1.4-B1 deterministic whole-story skeleton semantic preparation.

This module is intentionally a read-only, zero-provider boundary.  It pins
the B1.2 strategy and B1.3 publication supplied by the caller, constructs one
whole-story packet, renders a versioned structured request, and measures that
actual request.  It neither persists a skeleton nor invokes a client.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from short_drama.artifacts import ArtifactRef, FileArtifactStore, canonical_json_bytes, content_hash, strict_json_loads
from short_drama.foundation import FilePointerStore
from short_drama.guidance.projection import load_user_direction
from short_drama.io import load_json
from short_drama.llm import (GenerationExecutionOptions, OpenAICompatibleLLMClient, OutputSchema,
    PromptRegistry, RuntimeConfig, SemanticLLMProfile, StructuredGenerationRequest,
    build_structured_request, load_semantic_profile, validate_against_output_schema)
from short_drama.paths import REPO_ROOT
from short_drama.story.chunking import estimate_tokens
from short_drama.story.consolidation import CanonicalEventSet, CanonicalFactSet, CanonicalRelationshipSet
from short_drama.story.story_analysis import ArcAnalysis, GlobalEventAnalysis, GlobalStoryBible, GlobalStructure, StoryAnalysisManifest

from .skeleton import AdaptedStorySkeleton, B14SkeletonError
from .skeleton_validation import (B14SkeletonAuthority, B14SkeletonValidationResult,
    resolve_b14_skeleton_semantic_authority, validate_adapted_story_skeleton)
from .strategy_execution import B12StrategyArtifact


B14_SKELETON_PROMPT_ID = "b14.adapted-story-skeleton"
B14_SKELETON_PROMPT_VERSION = 1
B14_SKELETON_OUTPUT_SCHEMA_ID = "b14-adapted-story-skeleton-proposal"
B14_SKELETON_PROMPTS_DIR = REPO_ROOT / "prompts" / "adaptation"
B14_SKELETON_PACKET_SCHEMA_VERSION = 1
B14_SKELETON_PROMPT_CONTENT_HASH_V1 = "f31bb574c259d7e39f61f440cd47f5f8ad8f4068c81ed4c7a46d4a4b286bf66e"
B14_SKELETON_PROFILE_HASH_V1 = "bfc31ec94b1bc2285429639811a5704b09459618e8a0390eb8f1ef548b86ab39"
B14_SKELETON_SCHEMA_HASH_V1 = "f08580b34e55909fedd25d2dbaffe2720697ddbd22218dd59af6f6ead6faacd6"


class B14SkeletonSemanticError(ValueError):
    """A pinned B1.4 semantic input, asset, or proposed output is invalid."""


class B14SkeletonSemanticBudgetError(B14SkeletonSemanticError):
    """An explicit operator supplied context ceiling is exceeded."""


@dataclass(frozen=True, slots=True)
class B14SkeletonSemanticAssetSelection:
    version: int
    semantic_profile: SemanticLLMProfile
    output_schema: OutputSchema


@dataclass(frozen=True, slots=True)
class B14SkeletonSemanticBudget:
    """Runtime-only capacity declaration; the repository has no default n_ctx."""
    runtime_model_id: str
    model_context_tokens: int
    input_ceiling_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_model_id, str) or not self.runtime_model_id.strip():
            raise B14SkeletonSemanticBudgetError("runtime_model_id is required")
        for name in ("model_context_tokens", "input_ceiling_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise B14SkeletonSemanticBudgetError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class B14SkeletonSemanticMeasurement:
    packet_bytes: int
    packet_estimated_tokens: int
    rendered_prompt_bytes: int
    rendered_prompt_estimated_tokens: int
    provider_request_bytes: int
    provider_request_estimated_tokens: int
    reserved_output_tokens: int

    @property
    def total_requested_context_tokens(self) -> int:
        return self.provider_request_estimated_tokens + self.reserved_output_tokens


@dataclass(frozen=True, slots=True)
class B14SkeletonPacket:
    authority: B14SkeletonAuthority
    strategy: B12StrategyArtifact
    _canonical_payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.authority, B14SkeletonAuthority) or not isinstance(self.strategy, B12StrategyArtifact):
            raise B14SkeletonSemanticError("packet requires verified B1.4 authority and B1.2 strategy")
        try:
            value = strict_json_loads(self._canonical_payload)
        except Exception as exc:
            raise B14SkeletonSemanticError("packet must be canonical JSON") from exc
        if not isinstance(value, dict) or canonical_json_bytes(value) != self._canonical_payload:
            raise B14SkeletonSemanticError("packet must be canonical JSON")
        if value.get("schema_version") != B14_SKELETON_PACKET_SCHEMA_VERSION:
            raise B14SkeletonSemanticError("unsupported B1.4 packet schema version")
        pins = value.get("authority")
        if pins != {"strategy_ref": self.authority.strategy_ref.to_dict(), "disposition_publication_ref": self.authority.disposition_publication_ref.to_dict()}:
            raise B14SkeletonSemanticError("packet authority pins do not match verified authority")
        plan = value.get("verified_b13_disposition")
        if plan != self.authority.disposition_plan.to_dict():
            raise B14SkeletonSemanticError("packet B1.3 content does not match verified authority")
        if value.get("verified_b12_strategy") != self.strategy.strategy.to_dict():
            raise B14SkeletonSemanticError("packet B1.2 content does not match verified authority")

    def to_dict(self) -> dict[str, Any]:
        value = strict_json_loads(self._canonical_payload)
        assert isinstance(value, dict)
        return value

    def canonical_bytes(self) -> bytes:
        return self._canonical_payload

    def content_hash(self) -> str:
        return content_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class B14SkeletonSemanticPreparation:
    packet: B14SkeletonPacket
    semantic_profile: SemanticLLMProfile
    output_schema: OutputSchema
    request: StructuredGenerationRequest
    runtime_config: RuntimeConfig
    execution_options: GenerationExecutionOptions | None
    provider_request_bytes: bytes
    measurement: B14SkeletonSemanticMeasurement
    request_identity: str

    def enforce_budget(self, budget: B14SkeletonSemanticBudget | None) -> None:
        """Check only an explicit capacity declaration, with no truncation."""
        if budget is None:
            return
        if budget.runtime_model_id != self.runtime_config.request_model:
            raise B14SkeletonSemanticBudgetError("budget runtime_model_id does not match RuntimeConfig")
        if self.measurement.provider_request_estimated_tokens > budget.input_ceiling_tokens:
            raise B14SkeletonSemanticBudgetError("B1.4 rendered provider request exceeds input ceiling; refusing to truncate")
        if self.measurement.total_requested_context_tokens > budget.model_context_tokens:
            raise B14SkeletonSemanticBudgetError("B1.4 rendered provider request plus output reserve exceeds model context")

    def decode_and_validate_proposal(self, proposal: dict[str, Any]) -> tuple[AdaptedStorySkeleton, B14SkeletonValidationResult]:
        """Optional B2 seam: schema-valid proposal is not graph-valid output."""
        if not isinstance(proposal, dict):
            raise B14SkeletonSemanticError("B1.4 proposal must be an object")
        if {"strategy_ref", "disposition_publication_ref"} & set(proposal):
            raise B14SkeletonSemanticError("provider proposal must not select upstream authority")
        try:
            validate_against_output_schema(proposal, self.output_schema)
            value = {"schema_version": 1, "strategy_ref": self.packet.authority.strategy_ref.to_dict(),
                     "disposition_publication_ref": self.packet.authority.disposition_publication_ref.to_dict(), **proposal}
            skeleton = AdaptedStorySkeleton.from_dict(value)
        except Exception as exc:
            raise B14SkeletonSemanticError("B1.4 proposal is not schema/typed valid") from exc
        return skeleton, validate_adapted_story_skeleton(skeleton, authority=self.packet.authority)


def _load(store: FileArtifactStore, ref: ArtifactRef, model: Any) -> Any:
    try:
        payload = store.get_ref(ref).payload
        value = model.from_dict(payload)
    except Exception as exc:
        raise B14SkeletonSemanticError(f"cannot resolve exact {getattr(ref, 'artifact_type', 'unknown')} authority") from exc
    if value.to_dict() != payload:
        raise B14SkeletonSemanticError(f"persisted {ref.artifact_type} authority is not canonical")
    return value


def _relevant_source_context(store: FileArtifactStore, source: Any) -> dict[str, Any]:
    """Use pinned historical A5/A6 leaves, not a mutable current projection."""
    manifest = _load(store, source.story_analysis_manifest_ref, StoryAnalysisManifest)
    if (manifest.consolidation_manifest_ref != source.consolidation_manifest_ref
            or manifest.arc_analysis_ref != source.arc_analysis_ref):
        raise B14SkeletonSemanticError("exact A6 manifest does not match B1 authority")
    structure = _load(store, manifest.global_structure_ref, GlobalStructure)
    bible = _load(store, manifest.global_story_bible_ref, GlobalStoryBible)
    events = _load(store, source.canonical_event_set_ref, CanonicalEventSet)
    facts = _load(store, source.canonical_fact_set_ref, CanonicalFactSet)
    relationships = _load(store, source.canonical_relationship_set_ref, CanonicalRelationshipSet)
    arcs = _load(store, source.arc_analysis_ref, ArcAnalysis)
    event_analysis = _load(store, manifest.global_event_analysis_ref, GlobalEventAnalysis)
    return {
        "a6_global_story_bible": bible.to_dict(), "a6_global_structure": structure.to_dict(),
        "a6_global_event_analysis": event_analysis.to_dict(), "a6_arc_analysis": arcs.to_dict(),
        # Full canonical sets are intentional: B1.3 may retain a sparse overlay
        # while an A5 fact/relationship supplies critical causal or knowledge context.
        "a5_canonical_events": [item.to_dict() for item in sorted(events.events, key=lambda x: (x.narrative_order, x.event_id))],
        "a5_canonical_facts": [item.to_dict() for item in sorted(facts.facts, key=lambda x: x.fact_id)],
        "a5_canonical_relationships": [item.to_dict() for item in sorted(relationships.relationships, key=lambda x: x.relationship_id)],
    }


def prepare_b14_skeleton_packet(*, store: FileArtifactStore, pointers: FilePointerStore,
                                strategy_ref: ArtifactRef, disposition_publication_ref: ArtifactRef) -> B14SkeletonPacket:
    """Resolve explicit historical authority and build one whole-story packet."""
    try:
        authority = resolve_b14_skeleton_semantic_authority(
            strategy_ref=strategy_ref, disposition_publication_ref=disposition_publication_ref,
            store=store, pointers=pointers,
        )
    except Exception as exc:
        raise B14SkeletonSemanticError("pinned B1.2/B1.3 authority cannot be prepared") from exc
    strategy = _load(store, strategy_ref, B12StrategyArtifact)
    if strategy.source_snapshot != authority.disposition_plan.source_snapshot:
        raise B14SkeletonSemanticError("B1.2 strategy and B1.3 plan source snapshots differ")
    directions = []
    try:
        for ref in strategy.effective_direction_refs:
            directions.append(load_user_direction(store, ref).to_dict())
    except Exception as exc:
        raise B14SkeletonSemanticError("effective B1 user-direction authority is missing or invalid") from exc
    payload = {
        "schema_version": B14_SKELETON_PACKET_SCHEMA_VERSION,
        "authority": {"strategy_ref": strategy_ref.to_dict(), "disposition_publication_ref": disposition_publication_ref.to_dict()},
        "resolved_target": strategy.strategy.resolved_target.to_dict(),
        "verified_b12_strategy": strategy.strategy.to_dict(),
        "effective_b1_directions": {"effective_direction_hash": strategy.effective_b1_direction_hash,
                                    "direction_refs": [x.to_dict() for x in strategy.effective_direction_refs], "directions": directions},
        "verified_b13_disposition": authority.disposition_plan.to_dict(),
        "global_and_canonical_context": _relevant_source_context(store, authority.disposition_plan.source_snapshot),
    }
    return B14SkeletonPacket(authority, strategy, canonical_json_bytes(payload))


_ASSETS = ("b14_skeleton_llm_v1.yaml", "b14-adapted-story-skeleton-proposal-v1.schema.json",
           "b14-adapted-story-skeleton-llm-v1")


def load_b14_skeleton_semantic_assets(*, version: int = 1) -> B14SkeletonSemanticAssetSelection:
    if isinstance(version, bool) or version != 1:
        raise B14SkeletonSemanticError("unsupported B1.4 skeleton semantic asset version")
    profile_name, schema_name, profile_id = _ASSETS
    try:
        profile = load_semantic_profile(REPO_ROOT / "profiles" / profile_name)
        schema = OutputSchema.create(schema_id=B14_SKELETON_OUTPUT_SCHEMA_ID, schema_version=version,
                                     schema=load_json(REPO_ROOT / "schemas" / schema_name))
    except Exception as exc:
        raise B14SkeletonSemanticError("cannot load B1.4 skeleton semantic assets") from exc
    if (profile.profile_id != profile_id or profile.semantic_profile_hash != B14_SKELETON_PROFILE_HASH_V1
            or schema.schema_hash != B14_SKELETON_SCHEMA_HASH_V1
            or profile.structured_output_mode != "json_schema"):
        raise B14SkeletonSemanticError("B1.4 skeleton semantic profile is not the pinned structured asset")
    return B14SkeletonSemanticAssetSelection(version, profile, schema)


def _provider_bytes(request: StructuredGenerationRequest, runtime: RuntimeConfig,
                    options: GenerationExecutionOptions | None) -> bytes:
    body = OpenAICompatibleLLMClient(runtime).build_request_body(
        request.rendered_prompt, request.output_schema, request.semantic_profile, execution_options=options)
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def b14_skeleton_request_identity(*, packet: B14SkeletonPacket, profile: SemanticLLMProfile,
                                  schema: OutputSchema, request: StructuredGenerationRequest) -> str:
    return content_hash({"stage": "B1.4-B1", "strategy_ref": packet.authority.strategy_ref.to_dict(),
        "disposition_publication_ref": packet.authority.disposition_publication_ref.to_dict(),
        "packet_hash": packet.content_hash(),
        "effective_b1_direction_hash": packet.strategy.effective_b1_direction_hash,
        "prompt": {"prompt_id": request.rendered_prompt.prompt_id, "prompt_version": request.rendered_prompt.prompt_version,
                   "prompt_content_hash": request.rendered_prompt.prompt_content_hash},
        "output_schema": {"schema_id": schema.schema_id, "schema_version": schema.schema_version, "schema_content_hash": schema.schema_hash},
        "semantic_profile": {"profile_id": profile.profile_id, "profile_hash": profile.semantic_profile_hash},
        "rendered_request_hash": request.request_hash})


def build_b14_skeleton_semantic_preparation(packet: B14SkeletonPacket, *, prompts: PromptRegistry | None = None,
    semantic_asset_version: int = 1, semantic_profile: SemanticLLMProfile | None = None,
    output_schema: OutputSchema | None = None, runtime_config: RuntimeConfig | None = None,
    execution_options: GenerationExecutionOptions | None = None) -> B14SkeletonSemanticPreparation:
    if not isinstance(packet, B14SkeletonPacket) or not isinstance(runtime_config, RuntimeConfig):
        raise B14SkeletonSemanticError("B1.4 semantic preparation requires packet and explicit RuntimeConfig")
    if execution_options is not None and not isinstance(execution_options, GenerationExecutionOptions):
        raise B14SkeletonSemanticError("execution_options must be GenerationExecutionOptions or None")
    assets = load_b14_skeleton_semantic_assets(version=semantic_asset_version)
    if (semantic_profile is None) != (output_schema is None):
        raise B14SkeletonSemanticError("semantic profile and output schema must be supplied together")
    if semantic_profile is not None and (semantic_profile != assets.semantic_profile or output_schema != assets.output_schema):
        raise B14SkeletonSemanticError("semantic assets do not match the pinned B1.4 pair")
    prompts = PromptRegistry(B14_SKELETON_PROMPTS_DIR) if prompts is None else prompts
    prompt = prompts.load(B14_SKELETON_PROMPT_ID, version=B14_SKELETON_PROMPT_VERSION)
    if prompt.content_hash != B14_SKELETON_PROMPT_CONTENT_HASH_V1:
        raise B14SkeletonSemanticError("B1.4 skeleton prompt content hash does not match pinned v1 asset")
    rendered = prompt.render({"skeleton_packet_json": packet.canonical_bytes().decode("utf-8")})
    request = build_structured_request(rendered_prompt=rendered, output_schema=assets.output_schema, semantic_profile=assets.semantic_profile)
    provider = _provider_bytes(request, runtime_config, execution_options)
    measurement = B14SkeletonSemanticMeasurement(len(packet.canonical_bytes()), estimate_tokens(packet.canonical_bytes().decode("utf-8")),
        len((rendered.system_text + rendered.user_text).encode("utf-8")), estimate_tokens(rendered.system_text + rendered.user_text),
        len(provider), estimate_tokens(provider.decode("utf-8")), assets.semantic_profile.max_output_tokens)
    return B14SkeletonSemanticPreparation(packet, assets.semantic_profile, assets.output_schema, request, runtime_config,
        execution_options, provider, measurement, b14_skeleton_request_identity(packet=packet, profile=assets.semantic_profile,
        schema=assets.output_schema, request=request))


__all__ = ["B14_SKELETON_PROMPT_ID", "B14_SKELETON_PROMPT_VERSION", "B14_SKELETON_OUTPUT_SCHEMA_ID",
    "B14_SKELETON_PROMPT_CONTENT_HASH_V1", "B14_SKELETON_PROFILE_HASH_V1", "B14_SKELETON_SCHEMA_HASH_V1",
    "B14SkeletonSemanticError", "B14SkeletonSemanticBudgetError", "B14SkeletonSemanticAssetSelection",
    "B14SkeletonSemanticBudget", "B14SkeletonSemanticMeasurement", "B14SkeletonPacket", "B14SkeletonSemanticPreparation",
    "prepare_b14_skeleton_packet", "load_b14_skeleton_semantic_assets", "b14_skeleton_request_identity",
    "build_b14_skeleton_semantic_preparation"]
