"""B1.2-B1 measured semantic preparation; deliberately no provider execution.

This module owns the bounded hand-off to the shared LLM primitives.  It does
not construct an LLM client, retry, persist an artifact, or call a provider.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from short_drama.artifacts import canonical_json_bytes, content_hash
from short_drama.io import load_json
from short_drama.llm import (
    OutputSchema, PromptRegistry, SemanticLLMProfile, StructuredGenerationRequest,
    build_structured_request, load_semantic_profile,
)
from short_drama.paths import REPO_ROOT
from short_drama.story.chunking import estimate_tokens

from .strategy import AdaptationStrategy, AdaptationStrategyError
from .strategy_packet import B12Preparation


B12_STRATEGY_PROMPT_ID = "b12.adaptation-strategy"
B12_STRATEGY_PROMPT_VERSION = 1
B12_STRATEGY_OUTPUT_SCHEMA_ID = "b12-adaptation-strategy-proposal"
B12_STRATEGY_OUTPUT_SCHEMA_VERSION = 1
B12_PROMPTS_DIR = REPO_ROOT / "prompts" / "adaptation"
B12_SEMANTIC_PROFILE_PATH = REPO_ROOT / "profiles" / "adaptation_strategy_llm_v1.yaml"
B12_OUTPUT_SCHEMA_PATH = REPO_ROOT / "schemas" / "b12-adaptation-strategy-proposal.schema.json"


class B12SemanticBudgetError(AdaptationStrategyError):
    """The explicitly configured provider context cannot safely hold B1.2."""


@dataclass(frozen=True, slots=True)
class B12SemanticBudget:
    """Runtime-only context declaration, intentionally absent from identity.

    ``model_context_tokens`` must come from the operator's actual target model
    configuration.  The repository deliberately carries no guessed production
    context size.  ``input_ceiling_tokens`` is an independently tracked input
    policy ceiling; output capacity is always reserved from the semantic
    profile before a future provider call can happen.
    """

    runtime_model_id: str
    model_context_tokens: int
    input_ceiling_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_model_id, str) or not self.runtime_model_id.strip():
            raise B12SemanticBudgetError("runtime_model_id is required for a provider budget")
        for field in ("model_context_tokens", "input_ceiling_tokens"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise B12SemanticBudgetError(f"{field} must be a positive integer")


@dataclass(frozen=True, slots=True)
class B12SemanticMeasurement:
    semantic_view_bytes: int
    semantic_view_estimated_tokens: int
    rendered_request_bytes: int
    rendered_request_estimated_tokens: int
    reserved_output_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "semantic_view_bytes": self.semantic_view_bytes,
            "semantic_view_estimated_tokens": self.semantic_view_estimated_tokens,
            "rendered_request_bytes": self.rendered_request_bytes,
            "rendered_request_estimated_tokens": self.rendered_request_estimated_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
        }


@dataclass(frozen=True, slots=True)
class B12SemanticPreparation:
    """Complete deterministic request material for the later B1.2-B2 call."""

    preparation: B12Preparation
    semantic_profile: SemanticLLMProfile
    output_schema: OutputSchema
    request: StructuredGenerationRequest
    measurement: B12SemanticMeasurement
    request_identity: str

    def enforce_budget(self, budget: B12SemanticBudget | None) -> None:
        """Fail before provider execution on missing or insufficient budget.

        The caller for B1.2-B2 must invoke this immediately before its shared
        client.  There is intentionally no truncation, alternate packet, or
        default context window in this method.
        """
        if budget is None:
            raise B12SemanticBudgetError("B1.2 provider execution requires an explicit semantic budget")
        measured = self.measurement.rendered_request_estimated_tokens
        if measured > budget.input_ceiling_tokens:
            raise B12SemanticBudgetError(
                f"B1.2 rendered request ({measured} repository-estimated tokens) exceeds "
                f"input ceiling ({budget.input_ceiling_tokens}); refusing to truncate"
            )
        required = measured + self.measurement.reserved_output_tokens
        if required > budget.model_context_tokens:
            raise B12SemanticBudgetError(
                f"B1.2 rendered request plus reserved output ({required} repository-estimated tokens) "
                f"exceeds runtime model context ({budget.model_context_tokens})"
            )

    def decode_and_validate_proposal(self, proposal: dict[str, Any]) -> AdaptationStrategy:
        """Attach Python-owned target authority then cross the acceptance seam.

        JSON Schema is only a provider-shape constraint.  This method is the
        mandatory B1.2-B2 acceptance entry point: it attaches the exact target
        from this preparation and calls ``B12Preparation.validate_strategy``.
        Semantic hard-direction compliance and divergence review remain an
        explicit human/semantic review surface, not Python NLP heuristics.
        """
        if not isinstance(proposal, dict):
            raise AdaptationStrategyError("provider strategy proposal must be an object")
        if "resolved_target" in proposal:
            raise AdaptationStrategyError("provider proposal must not select Python-owned resolved_target")
        material = dict(proposal)
        material["resolved_target"] = self.preparation.packet.resolved_target.to_dict()
        strategy = AdaptationStrategy.from_dict(material)
        self.preparation.validate_strategy(strategy)
        return strategy


def load_b12_semantic_profile() -> SemanticLLMProfile:
    profile = load_semantic_profile(B12_SEMANTIC_PROFILE_PATH)
    if profile.profile_id != "b12-adaptation-strategy-llm-v1":
        raise AdaptationStrategyError("B1.2 semantic profile ID is not the pinned asset")
    return profile


def load_b12_output_schema() -> OutputSchema:
    try:
        data = load_json(B12_OUTPUT_SCHEMA_PATH)
    except Exception as exc:  # noqa: BLE001
        raise AdaptationStrategyError("cannot load B1.2 provider proposal schema") from exc
    if not isinstance(data, dict):
        raise AdaptationStrategyError("B1.2 provider proposal schema must be an object")
    return OutputSchema.create(
        schema_id=B12_STRATEGY_OUTPUT_SCHEMA_ID,
        schema_version=B12_STRATEGY_OUTPUT_SCHEMA_VERSION,
        schema=data,
    )


def b12_request_identity(*, preparation: B12Preparation, profile: SemanticLLMProfile,
                         output_schema: OutputSchema, request: StructuredGenerationRequest) -> str:
    """Backend-neutral identity for exactly the semantic material B1.2 uses."""
    packet = preparation.packet
    return content_hash({
        "stage": "B1.2",
        "a6_manifest_ref": packet.source_snapshot.story_analysis_manifest_ref.to_dict(),
        "a5_manifest_ref": packet.source_snapshot.consolidation_manifest_ref.to_dict(),
        "resolved_target_hash": packet.resolved_target.content_hash(),
        "effective_b1_direction_hash": preparation.measurement.effective_b1_direction_hash,
        "semantic_view_hash": preparation.semantic_view.content_hash(),
        "prompt": {
            "prompt_id": request.rendered_prompt.prompt_id,
            "prompt_version": request.rendered_prompt.prompt_version,
            "prompt_content_hash": request.rendered_prompt.prompt_content_hash,
        },
        "output_schema": {
            "schema_id": output_schema.schema_id,
            "schema_version": output_schema.schema_version,
            "schema_content_hash": output_schema.schema_hash,
        },
        "semantic_profile": {"profile_id": profile.profile_id, "profile_hash": profile.semantic_profile_hash},
        "rendered_request_hash": request.request_hash,
    })


def _provider_request_material(request: StructuredGenerationRequest) -> dict[str, Any]:
    """Shared OpenAI-compatible body minus runtime routing metadata.

    This is the complete prompt/schema/profile material transmitted to the
    provider.  ``model`` is intentionally omitted because it is runtime-only;
    it changes neither semantic identity nor the bounded context payload.
    """
    profile = request.semantic_profile
    return {
        "messages": list(request.messages),
        "temperature": profile.temperature,
        "max_tokens": profile.max_output_tokens,
        "reasoning_effort": profile.reasoning.request_effort,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.output_schema.schema_id,
                "schema": request.output_schema.schema,
                "strict": True,
            },
        },
    }


def build_b12_semantic_preparation(
    preparation: B12Preparation,
    *,
    prompts: PromptRegistry | None = None,
    semantic_profile: SemanticLLMProfile | None = None,
    output_schema: OutputSchema | None = None,
) -> B12SemanticPreparation:
    """Render the entire B1.2 proposal request with no provider side effects."""
    if not isinstance(preparation, B12Preparation):
        raise AdaptationStrategyError("B1.2 semantic preparation requires B12Preparation")
    prompts = PromptRegistry(B12_PROMPTS_DIR) if prompts is None else prompts
    semantic_profile = load_b12_semantic_profile() if semantic_profile is None else semantic_profile
    output_schema = load_b12_output_schema() if output_schema is None else output_schema
    prompt = prompts.load(B12_STRATEGY_PROMPT_ID, version=B12_STRATEGY_PROMPT_VERSION)
    rendered = prompt.render({"semantic_view_json": preparation.semantic_view.canonical_bytes().decode("utf-8")})
    request = build_structured_request(
        rendered_prompt=rendered, output_schema=output_schema, semantic_profile=semantic_profile,
    )
    # Do not measure the in-memory request's diagnostic duplication of rendered
    # prompt and messages.  Measure the full transport-neutral wire body the
    # existing OpenAI-compatible adapter maps (minus runtime-only ``model``).
    rendered_bytes = canonical_json_bytes(_provider_request_material(request))
    measurement = B12SemanticMeasurement(
        semantic_view_bytes=len(preparation.semantic_view.canonical_bytes()),
        semantic_view_estimated_tokens=estimate_tokens(preparation.semantic_view.canonical_bytes().decode("utf-8")),
        rendered_request_bytes=len(rendered_bytes),
        rendered_request_estimated_tokens=estimate_tokens(rendered_bytes.decode("utf-8")),
        reserved_output_tokens=semantic_profile.max_output_tokens,
    )
    return B12SemanticPreparation(
        preparation=preparation,
        semantic_profile=semantic_profile,
        output_schema=output_schema,
        request=request,
        measurement=measurement,
        request_identity=b12_request_identity(
            preparation=preparation, profile=semantic_profile, output_schema=output_schema, request=request,
        ),
    )
