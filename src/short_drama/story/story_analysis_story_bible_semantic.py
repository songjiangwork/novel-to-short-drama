"""A6F StoryBible structured semantic pass; all persisted IDs remain Python-owned."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from short_drama.artifacts import ArtifactRef
from short_drama.artifacts.canonical import content_hash
from short_drama.io import load_json
from short_drama.llm import (GenerationExecutionOptions, LLMClient, LLMInvocationProvenance,
    OutputSchema, PromptRegistry, SemanticLLMProfile, StructuredGenerationRequest,
    build_structured_request, load_semantic_profile)
from short_drama.paths import PROFILES_DIR, SCHEMAS_DIR
from .consolidation import OutputSchemaAssetIdentity, PromptAssetIdentity
from .errors import (StoryAnalysisProvenanceError, StoryAnalysisSemanticError,
    StoryAnalysisStoryBibleSemanticGenerationError)
from .story_analysis import (ArcAnalysis, CharacterAnalysis, EvidenceBackedInterpretation,
    GlobalEventAnalysis, GlobalStoryBible, GlobalStructure, StoryAnalysisProfile)
from .story_analysis_global_skeleton_semantic import _global_evidence_universe
from .story_analysis_planning import StoryAnalysisInputSnapshot, StoryAnalysisPlan
from .story_analysis_story_bible import (A6F_STORY_BIBLE_PROMPT_ID, A6F_STORY_BIBLE_PROMPT_VERSION,
    StoryBibleContext, build_story_bible_context, build_story_bible_rendered_request,
    estimate_story_bible_request_tokens, validate_story_bible_context)

A6F_STORY_BIBLE_OUTPUT_SCHEMA_ID = "a6-story-bible-output"
A6F_STORY_BIBLE_OUTPUT_SCHEMA_VERSION = 1
A6F_SEMANTIC_PROFILE_ID = "story-analysis-llm-v1"

# These are exactly the top-level fields in the tracked provider output schema.
# The four reference collections on GlobalStoryBible are deliberately absent:
# their authoritative source is the already-validated A6E GlobalStructure.
_PROVIDER_STORY_BIBLE_KEYS = frozenset((
    "premise_zh", "synopsis_zh", "genre", "tone", "setting", "themes",
    "unresolved_summary", "source_conflict_summary",
))


@dataclass(frozen=True, slots=True)
class StoryBibleSemanticPreparation:
    context: StoryBibleContext; request: StructuredGenerationRequest; request_identity_hash: str
    request_hash: str; prompt_identity: PromptAssetIdentity; output_schema_identity: OutputSchemaAssetIdentity
    packet_token_estimate: int; rendered_request_token_estimate: int
    character_request_identity_hashes: tuple[str, ...]; window_request_identity_hashes: tuple[str, ...]; global_skeleton_request_identity_hash: str


@dataclass(frozen=True, slots=True)
class StoryBibleSemanticResult:
    preparation: StoryBibleSemanticPreparation; story_bible: GlobalStoryBible; rounds_consumed: int


def load_story_bible_semantic_profile() -> SemanticLLMProfile:
    return load_semantic_profile(PROFILES_DIR / "story_analysis_llm_v1.yaml")


def load_story_bible_output_schema() -> OutputSchema:
    data = load_json(SCHEMAS_DIR / "a6-story-bible-output.schema.json")
    if not isinstance(data, dict): raise StoryAnalysisSemanticError("story-bible output schema must be object")
    return OutputSchema.create(
        schema_id=A6F_STORY_BIBLE_OUTPUT_SCHEMA_ID,
        schema_version=A6F_STORY_BIBLE_OUTPUT_SCHEMA_VERSION,
        schema=data,
    )


def enforce_story_bible_ceiling(profile: StoryAnalysisProfile, context: StoryBibleContext) -> None:
    ceiling = profile.planning_policy.story_bible_packet_max_estimated_tokens
    if ceiling is None: raise StoryAnalysisSemanticError("story_bible_packet_max_estimated_tokens is DEFERRED (None); refusing provider call")
    if context.estimated_tokens() > ceiling:
        raise StoryAnalysisSemanticError(f"story-bible packet ({context.estimated_tokens()} estimated tokens) exceeds frozen ceiling ({ceiling}); refusing to truncate")


def story_bible_request_identity_hash(*, consolidation_manifest_ref: ArtifactRef, plan_hash: str,
    profile: StoryAnalysisProfile, semantic_profile: SemanticLLMProfile,
    prompt_identity: PromptAssetIdentity, output_schema_identity: OutputSchemaAssetIdentity,
    packet_hash: str, request_hash: str, character_request_identity_hashes: Sequence[str],
    window_request_identity_hashes: Sequence[str], global_skeleton_request_identity_hash: str) -> str:
    return content_hash({"upstream_a5": consolidation_manifest_ref.to_dict(), "a6_plan_hash": plan_hash,
        "story_analysis_profile_id": profile.profile_id, "story_analysis_profile_hash": profile.content_hash(),
        "semantic_profile_id": semantic_profile.profile_id, "semantic_profile_hash": semantic_profile.semantic_profile_hash,
        "prompt": prompt_identity.to_dict(), "output_schema": output_schema_identity.to_dict(),
        "story_packet_hash": packet_hash, "rendered_request_hash": request_hash,
        "character_request_identity_hashes": list(character_request_identity_hashes),
        "window_request_identity_hashes": list(window_request_identity_hashes),
        "global_skeleton_request_identity_hash": global_skeleton_request_identity_hash})


def build_story_bible_semantic_preparation(snapshot: StoryAnalysisInputSnapshot, plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile, semantic_profile: SemanticLLMProfile, character_analyses: Sequence[CharacterAnalysis],
    global_event_analysis: GlobalEventAnalysis, arc_analysis: ArcAnalysis, global_structure: GlobalStructure, *,
    prompts: PromptRegistry, character_request_identity_hashes: Sequence[str], window_request_identity_hashes: Sequence[str],
    global_skeleton_request_identity_hash: str) -> StoryBibleSemanticPreparation:
    if snapshot.consolidation_manifest_ref != plan.consolidation_manifest_ref: raise StoryAnalysisSemanticError("snapshot and plan disagree on exact A5 manifest")
    if semantic_profile.profile_id != A6F_SEMANTIC_PROFILE_ID or profile.max_generation_rounds != 2: raise StoryAnalysisSemanticError("invalid A6F semantic profile")
    sb = profile.story_bible
    if (sb.prompt_id, sb.prompt_version, sb.output_schema_id, sb.output_schema_version) != (A6F_STORY_BIBLE_PROMPT_ID, 1, A6F_STORY_BIBLE_OUTPUT_SCHEMA_ID, 1): raise StoryAnalysisSemanticError("profile does not pin frozen A6F assets")
    # Upstream identities are not caller assertions. Rebuild A6C/A6D and A6E
    # identities from the exact current snapshot/profile before a final request.
    from .story_analysis_semantic import build_character_semantic_preparation
    from .story_analysis_window_semantic import build_plot_window_semantic_preparation
    from .story_analysis_global_skeleton_semantic import build_global_skeleton_semantic_preparation
    expected_chars = build_character_semantic_preparation(plan, profile, semantic_profile, prompts=prompts).character_request_identity_hashes
    expected_windows = build_plot_window_semantic_preparation(snapshot, plan, profile, semantic_profile, prompts=prompts).window_request_identity_hashes
    if tuple(character_request_identity_hashes) != expected_chars or tuple(window_request_identity_hashes) != expected_windows:
        raise StoryAnalysisSemanticError("A6F upstream A6C/A6D request identities are stale, incomplete, or reordered")
    expected_global = build_global_skeleton_semantic_preparation(snapshot, plan, profile, semantic_profile, character_analyses, global_event_analysis.windows, prompts=prompts, character_request_identity_hashes=expected_chars, window_request_identity_hashes=expected_windows).request_identity_hash
    if global_skeleton_request_identity_hash != expected_global:
        raise StoryAnalysisSemanticError("A6F upstream A6E request identity is stale or fabricated")
    context = build_story_bible_context(character_analyses, global_event_analysis, arc_analysis, global_structure, plan.global_index)
    enforce_story_bible_ceiling(profile, context); validate_story_bible_context(context, plan)
    spec = prompts.load(A6F_STORY_BIBLE_PROMPT_ID, version=1); schema = load_story_bible_output_schema()
    pi = PromptAssetIdentity(spec.prompt_id, spec.version, spec.content_hash)
    oi = OutputSchemaAssetIdentity(schema.schema_id, schema.schema_version, schema.schema_hash)
    request = build_structured_request(rendered_prompt=build_story_bible_rendered_request(context, spec), output_schema=schema, semantic_profile=semantic_profile)
    identity = story_bible_request_identity_hash(consolidation_manifest_ref=plan.consolidation_manifest_ref, plan_hash=plan.plan_hash, profile=profile, semantic_profile=semantic_profile, prompt_identity=pi, output_schema_identity=oi, packet_hash=context.content_hash(), request_hash=request.request_hash, character_request_identity_hashes=character_request_identity_hashes, window_request_identity_hashes=window_request_identity_hashes, global_skeleton_request_identity_hash=global_skeleton_request_identity_hash)
    return StoryBibleSemanticPreparation(context, request, identity, request.request_hash, pi, oi, context.estimated_tokens(), estimate_story_bible_request_tokens(request.rendered_prompt), tuple(character_request_identity_hashes), tuple(window_request_identity_hashes), global_skeleton_request_identity_hash)


def _refs_ok(interp: EvidenceBackedInterpretation, universe: dict[str, frozenset[str]]) -> bool:
    return all(ref in universe[k] for k, refs in (("fact", interp.supporting_fact_refs), ("evt", interp.supporting_event_refs), ("rel", interp.supporting_relationship_refs), ("conf", interp.supporting_conflict_refs)) for ref in refs)


def build_story_bible_from_provider_output(payload: dict[str, Any], snapshot: StoryAnalysisInputSnapshot,
    global_event_analysis: GlobalEventAnalysis, arc_analysis: ArcAnalysis,
    global_structure: GlobalStructure) -> GlobalStoryBible:
    """Build the typed StoryBible from model fields plus Python-owned A6E refs.

    Keeping this as the sole construction path prevents semantic validation from
    accepting a payload that the final typed construction would later reject.
    """
    keys = set(payload)
    if keys != _PROVIDER_STORY_BIBLE_KEYS:
        missing_keys = sorted(_PROVIDER_STORY_BIBLE_KEYS - keys)
        unexpected_keys = sorted(keys - _PROVIDER_STORY_BIBLE_KEYS)
        raise StoryAnalysisSemanticError(
            "provider StoryBible keys mismatch: "
            f"missing_keys={missing_keys!r}, unexpected_keys={unexpected_keys!r}"
        )
    return GlobalStoryBible.from_dict({
        "schema_version": 1,
        **payload,
        "main_character_refs": list(global_structure.main_character_refs),
        "major_arc_refs": list(global_structure.major_arc_refs),
        "major_turning_point_refs": list(global_structure.major_turning_point_refs),
        "major_reveal_refs": list(global_structure.major_reveal_refs),
        "coverage_metadata": {
            "canonical_character_count": len(snapshot.canonical_characters),
            "canonical_event_count": len(snapshot.events),
            "plot_window_count": len(global_event_analysis.windows),
            "arc_count": len(arc_analysis.arcs),
        },
    })


def validate_story_bible_output(payload: dict[str, Any], snapshot: StoryAnalysisInputSnapshot,
    global_event_analysis: GlobalEventAnalysis, arc_analysis: ArcAnalysis,
    global_structure: GlobalStructure) -> str | None:
    try:
        bible = build_story_bible_from_provider_output(
            payload, snapshot, global_event_analysis, arc_analysis, global_structure,
        )
    except Exception as exc: return f"typed StoryBible load failed: {exc}"
    universe = _global_evidence_universe(snapshot)
    interpretations = (bible.premise_zh, bible.synopsis_zh, bible.setting, bible.unresolved_summary, bible.source_conflict_summary, *bible.themes)
    if not all(_refs_ok(x, universe) for x in interpretations): return "StoryBible has ref outside A5 evidence universe"
    if not set(bible.main_character_refs) <= universe["char"]: return "StoryBible has unknown character refs"
    if not set(bible.major_arc_refs) <= {x.arc_id for x in arc_analysis.arcs}: return "StoryBible has unknown arc refs"
    if not set(bible.major_turning_point_refs) <= {x.turning_point_id for x in arc_analysis.turning_points}: return "StoryBible has unknown turning-point refs"
    if not set(bible.major_reveal_refs) <= {x.reveal_id for x in arc_analysis.reveals}: return "StoryBible has unknown reveal refs"
    if not set(bible.main_character_refs) <= set(global_structure.main_character_refs): return "StoryBible main characters must be in GlobalStructure"
    return None


def _verify_provenance(p: LLMInvocationProvenance, r: StructuredGenerationRequest) -> None:
    checks = ((p.semantic_profile_id, r.semantic_profile.profile_id), (p.semantic_profile_hash, r.semantic_profile.semantic_profile_hash), (p.prompt_id, r.rendered_prompt.prompt_id), (p.prompt_version, r.rendered_prompt.prompt_version), (p.prompt_content_hash, r.rendered_prompt.prompt_content_hash), (p.rendered_prompt_hash, r.rendered_prompt.rendered_prompt_hash), (p.output_schema_id, r.output_schema.schema_id), (p.output_schema_version, r.output_schema.schema_version), (p.output_schema_hash, r.output_schema.schema_hash), (p.request_hash, r.request_hash))
    if any(a != b for a, b in checks): raise StoryAnalysisProvenanceError("story-bible provenance mismatch")


def resolve_story_bible(snapshot: StoryAnalysisInputSnapshot, plan: StoryAnalysisPlan, profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile, character_analyses: Sequence[CharacterAnalysis], global_event_analysis: GlobalEventAnalysis,
    arc_analysis: ArcAnalysis, global_structure: GlobalStructure, llm_client: LLMClient, *, prompts: PromptRegistry,
    character_request_identity_hashes: Sequence[str], window_request_identity_hashes: Sequence[str], global_skeleton_request_identity_hash: str) -> StoryBibleSemanticResult:
    prep = build_story_bible_semantic_preparation(snapshot, plan, profile, semantic_profile, character_analyses, global_event_analysis, arc_analysis, global_structure, prompts=prompts, character_request_identity_hashes=character_request_identity_hashes, window_request_identity_hashes=window_request_identity_hashes, global_skeleton_request_identity_hash=global_skeleton_request_identity_hash)
    failures: list[str] = []
    for round_number in (1, 2):
        result = llm_client.generate_structured(prep.request.rendered_prompt, prep.request.output_schema, semantic_profile, execution_options=GenerationExecutionOptions(prompt_context_reuse="disabled") if round_number == 2 else None)
        _verify_provenance(result.provenance, prep.request)
        if not isinstance(result.parsed_json, dict): detail = "provider output must be a JSON object"
        else: detail = validate_story_bible_output(
            result.parsed_json, snapshot, global_event_analysis, arc_analysis,
            global_structure,
        )
        if detail is None:
            bible = build_story_bible_from_provider_output(
                result.parsed_json, snapshot, global_event_analysis, arc_analysis,
                global_structure,
            )
            return StoryBibleSemanticResult(prep, bible, round_number)
        failures.append(detail)
    raise StoryAnalysisStoryBibleSemanticGenerationError(
        request_hash=prep.request_hash, rounds_attempted=2,
        last_failure_details=tuple(failures),
    )
