"""v1.2 A6E-2 -- global-skeleton semantic execution (provider execution).

This slice implements the **global-skeleton** semantic pass of A6 (Issue #102)
as specified by the frozen A6 architecture doc (``06-A6-global-story-bible.md``)
and the frozen A6 implementation plan. It is the whole-story semantic synthesis
pass that consumes the complete A6C character analyses + A6D plot-window
analyses + A5-derived GlobalIndexBase (assembled by A6E-1) and produces the
three in-memory A6 analysis artifacts:

  * :class:`~short_drama.story.GlobalEventAnalysis` (event-importance overlay)
  * :class:`~short_drama.story.ArcAnalysis` (arcs / turning points / reveals /
    foreshadow-payoffs)
  * :class:`~short_drama.story.GlobalStructure` (narrative structure)

A6E-2 owns:

* **frozen ceiling enforcement** -- the ``global_skeleton_packet_max_estimated_tokens``
  ceiling is loaded from the tracked ``StoryAnalysisProfile``; if it is ``None``
  (DEFERRED) or the actual packet exceeds it, the pass fails closed BEFORE any
  provider call. No truncation or sampling is permitted.
* **deterministic complete request identity** -- one stable request identity
  that binds: the exact A5 ConsolidationManifest ref, the A6B plan hash, the
  ``StoryAnalysisProfile`` id/hash, the ``SemanticLLMProfile`` id/hash, the
  frozen prompt and output-schema asset identities, the complete global packet
  hash, the rendered request hash, AND the ordered upstream A6C/A6D request
  identity hashes. Excludes runtime/provider/hardware topology.
* **bounded semantic regeneration** -- up to the frozen ``max_generation_rounds``
  (2) semantic rounds for the single global-skeleton request. Only *semantic*
  invalidity (a schema-valid provider result rejected by typed load / canonical
  ref validation / structural validation) triggers a regeneration round. A
  schema-invalid output and every other technical LLM failure remain owned by
  A-I3 (``short_drama.llm``) and are propagated, never translated into a
  semantic retry. Provenance / request-hash mismatches fail closed without
  retry.
* **Python-assigned deterministic IDs** -- the model proposes semantics (the
  content); Python assigns the final persisted A6 analysis IDs deterministically
  (``arc_000001``, ``turn_000001``, ``reveal_000001``, ``payoff_000001``, ...)
  based on the provider's ``proposal_ordinal``. The model NEVER assigns final
  IDs.
* **canonical ref validation** -- every referenced canonical ref (evt / char /
  fact / rel / conf) must belong to the complete A5 evidence universe (the
  entire snapshot). A ref that does not exist in the A5 snapshot is rejected.
* **structural validation** -- arc start/end events must exist in narrative
  order (start before end), global section ordinals must be unique and
  strictly increasing, internal cross-references (major_arc_proposal_ordinals,
  etc.) must reference valid proposal ordinals, and Python-assigned IDs must
  be unique.

A6E-2 is **in-memory only**: it consumes the exact A6B
:class:`~short_drama.story.StoryAnalysisPlan`, the A6C
:class:`~short_drama.story.CharacterAnalysis` set, the A6D
:class:`~short_drama.story.PlotWindowAnalysis` set, and the A5 snapshot, and
produces the complete in-memory typed result set. It does NOT persist any A6
artifact, does NOT mutate any A5/A4 CURRENT, does NOT implement A6F StoryBible,
CLI, B1, or UserDirection. A failed global execution must NOT return a
partially authoritative result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from short_drama.artifacts import ArtifactRef
from short_drama.artifacts.canonical import content_hash
from short_drama.io import load_json
from short_drama.llm import (
    GenerationExecutionOptions,
    LLMClient,
    LLMInvocationProvenance,
    OutputSchema,
    PromptRegistry,
    RenderedPrompt,
    SemanticLLMProfile,
    StructuredGenerationRequest,
    build_structured_request,
    load_semantic_profile,
    render_prompt,
)
from short_drama.paths import PROFILES_DIR, SCHEMAS_DIR

from .chunking import estimate_tokens
from .consolidation import OutputSchemaAssetIdentity, PromptAssetIdentity
from .errors import (
    StoryAnalysisGlobalSkeletonSemanticGenerationError,
    StoryAnalysisProvenanceError,
    StoryAnalysisSemanticError,
    StoryIntegrityError,
)
from .story_analysis import (
    ArcAnalysis,
    CharacterAnalysis,
    EvidenceBackedInterpretation,
    ForeshadowPayoff,
    GlobalEventAnalysis,
    GlobalEventImportance,
    GlobalSection,
    GlobalStructure,
    PlotWindowAnalysis,
    Reveal,
    STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1,
    STORY_ANALYSIS_SCHEMA_VERSION,
    StoryAnalysisProfile,
    StoryArc,
    TurningPoint,
)
from .story_analysis_global_skeleton import (
    A6E_GLOBAL_SKELETON_PROMPT_ID,
    A6E_GLOBAL_SKELETON_PROMPT_VERSION,
    GlobalSkeletonContext,
    build_global_skeleton_context,
    build_global_skeleton_rendered_request,
    validate_global_skeleton_coverage,
)
from .story_analysis_planning import (
    StoryAnalysisInputSnapshot,
    StoryAnalysisPlan,
)


# ---------------------------------------------------------------------------
# Frozen A6E-2 identity (prompt / output schema / semantic profile).
# ---------------------------------------------------------------------------

#: The exact A6E global-skeleton output schema (frozen, reviewed).
A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_ID = "a6-global-skeleton-output"
A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_VERSION = 1
A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_PATH = (
    SCHEMAS_DIR / "a6-global-skeleton-output.schema.json"
)
#: The exact A6E semantic profile (frozen, reviewed; shared with A6C/A6D).
A6E_SEMANTIC_PROFILE_ID = "story-analysis-llm-v1"
A6E_SEMANTIC_PROFILE_PATH = PROFILES_DIR / "story_analysis_llm_v1.yaml"
#: The frozen A6E semantic regeneration budget (initial + at most 1
#: regeneration), re-exposed from the A6A contract so A6E and A6A can never
#: drift.
A6E_MAX_GENERATION_ROUNDS = STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1


# ---------------------------------------------------------------------------
# Result objects (in-memory, no persistence).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GlobalSkeletonSemanticPreparation:
    """Deterministic, zero-provider A6E-2 global-skeleton request preparation.

    Carries the complete global packet, the rendered request, the stable
    request identity hash, and the token estimates.
    """

    plan: StoryAnalysisPlan
    profile: StoryAnalysisProfile
    semantic_profile: SemanticLLMProfile
    prompt_identity: PromptAssetIdentity
    output_schema_identity: OutputSchemaAssetIdentity
    context: GlobalSkeletonContext
    request: StructuredGenerationRequest
    request_identity_hash: str
    request_hash: str
    packet_token_estimate: int
    rendered_request_token_estimate: int
    #: Ordered upstream A6C request identity hashes (canonical character order).
    character_request_identity_hashes: tuple[str, ...]
    #: Ordered upstream A6D request identity hashes (canonical window order).
    window_request_identity_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GlobalSkeletonSemanticResult:
    """Complete in-memory A6E-2 global-skeleton semantic result (no persistence).

    ``global_event_analysis``, ``arc_analysis``, and ``global_structure`` are
    the three typed A6 analysis artifacts produced by the global-skeleton
    pass. They are in-memory only: no A6 artifact is persisted and no
    A5/A4 CURRENT is mutated.
    """

    preparation: GlobalSkeletonSemanticPreparation
    global_event_analysis: GlobalEventAnalysis
    arc_analysis: ArcAnalysis
    global_structure: GlobalStructure
    #: Number of semantic rounds consumed (1 or 2).
    rounds_consumed: int


@dataclass(frozen=True, slots=True)
class _GlobalSkeletonRetryableInvalid:
    """A schema-valid provider result rejected by semantic validation.

    Carries a bounded failure detail (never the raw provider body) for the
    bounded-regeneration diagnostic. This is a *semantic* invalidity, not a
    technical LLM failure.
    """

    detail: str


# ---------------------------------------------------------------------------
# Frozen asset loading.
# ---------------------------------------------------------------------------


def load_global_skeleton_semantic_profile() -> SemanticLLMProfile:
    """Load the A6E global-skeleton semantic profile."""
    return load_semantic_profile(A6E_SEMANTIC_PROFILE_PATH)


def load_global_skeleton_output_schema() -> OutputSchema:
    """Load the frozen A6E global-skeleton output schema."""
    try:
        schema_data = load_json(A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_PATH)
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(
            f"failed to load global-skeleton output schema: {exc}"
        ) from exc
    if not isinstance(schema_data, dict):
        raise StoryIntegrityError(
            "global-skeleton output schema must be a JSON object"
        )
    return OutputSchema.create(
        schema_id=A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_ID,
        schema_version=A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_VERSION,
        schema=schema_data,
    )


def load_global_skeleton_semantic_assets() -> tuple[SemanticLLMProfile, OutputSchema]:
    """Load the exact (semantic profile, output schema) A6E asset pair."""
    return load_global_skeleton_semantic_profile(), load_global_skeleton_output_schema()


# ---------------------------------------------------------------------------
# Profile identity verification (fail closed).
# ---------------------------------------------------------------------------


def _verify_global_skeleton_profile(
    profile: StoryAnalysisProfile, semantic_profile: SemanticLLMProfile
) -> None:
    """Verify the profile pins the exact frozen A6E global-skeleton pass identity."""
    if semantic_profile.profile_id != A6E_SEMANTIC_PROFILE_ID:
        raise StoryAnalysisSemanticError(
            f"semantic profile must be {A6E_SEMANTIC_PROFILE_ID!r}, "
            f"got {semantic_profile.profile_id!r}"
        )
    if profile.max_generation_rounds != A6E_MAX_GENERATION_ROUNDS:
        raise StoryAnalysisSemanticError(
            f"max_generation_rounds must be {A6E_MAX_GENERATION_ROUNDS}, "
            f"got {profile.max_generation_rounds}"
        )
    gs = profile.global_skeleton
    if gs.semantic_profile_id != A6E_SEMANTIC_PROFILE_ID:
        raise StoryAnalysisSemanticError(
            f"global_skeleton.semantic_profile_id must be "
            f"{A6E_SEMANTIC_PROFILE_ID!r}, got {gs.semantic_profile_id!r}"
        )
    if (
        gs.prompt_id != A6E_GLOBAL_SKELETON_PROMPT_ID
        or gs.prompt_version != A6E_GLOBAL_SKELETON_PROMPT_VERSION
    ):
        raise StoryAnalysisSemanticError(
            f"global_skeleton prompt must be "
            f"{A6E_GLOBAL_SKELETON_PROMPT_ID!r} v{A6E_GLOBAL_SKELETON_PROMPT_VERSION}, "
            f"got {gs.prompt_id!r} v{gs.prompt_version}"
        )
    if (
        gs.output_schema_id != A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_ID
        or gs.output_schema_version != A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_VERSION
    ):
        raise StoryAnalysisSemanticError(
            f"global_skeleton output schema must be "
            f"{A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_ID!r} "
            f"v{A6E_GLOBAL_SKELETON_OUTPUT_SCHEMA_VERSION}, "
            f"got {gs.output_schema_id!r} v{gs.output_schema_version}"
        )


# ---------------------------------------------------------------------------
# Ceiling enforcement (fail closed).
# ---------------------------------------------------------------------------


def enforce_global_skeleton_ceiling(
    profile: StoryAnalysisProfile,
    context: GlobalSkeletonContext,
) -> None:
    """Fail closed if the global-skeleton ceiling is DEFERRED (None) or the
    actual packet exceeds it. No truncation or sampling is permitted."""
    ceiling = profile.planning_policy.global_skeleton_packet_max_estimated_tokens
    if ceiling is None:
        raise StoryAnalysisSemanticError(
            "global_skeleton_packet_max_estimated_tokens is DEFERRED (None); "
            "A6E global-skeleton semantic execution refuses to run without a "
            "frozen ceiling"
        )
    packet_tokens = context.estimated_tokens()
    if packet_tokens > ceiling:
        raise StoryAnalysisSemanticError(
            f"global-skeleton packet ({packet_tokens} estimated tokens) "
            f"exceeds the frozen global_skeleton_packet_max_estimated_tokens "
            f"ceiling ({ceiling}); refusing to truncate"
        )


# ---------------------------------------------------------------------------
# Upstream binding enforcement (fail closed, before any provider call).
# ---------------------------------------------------------------------------

import re as _re

_SHA256_HEX = _re.compile(r"^[0-9a-f]{64}$")


def _verify_snapshot_plan_manifest_identity(
    snapshot: StoryAnalysisInputSnapshot, plan: StoryAnalysisPlan
) -> None:
    """Fail closed if the A5 snapshot and the A6B plan do not refer to the same
    exact A5 ConsolidationManifest.

    Mirrors the A6D seam: if the two inputs name different exact A5 upstreams,
    the semantic requests would claim one A5 identity while consuming evidence
    from another, so A6E-2 must fail closed before any packet construction or
    provider execution.
    """
    if snapshot.consolidation_manifest_ref != plan.consolidation_manifest_ref:
        raise StoryAnalysisSemanticError(
            "story-analysis input snapshot and plan disagree on the exact A5 "
            "consolidation manifest identity (snapshot "
            f"{snapshot.consolidation_manifest_ref!r} != plan "
            f"{plan.consolidation_manifest_ref!r}); A6E-2 global-skeleton "
            "semantic analysis must consume one consistent exact A5 upstream "
            "and refuses to join evidence from a mismatched pair"
        )


def _verify_upstream_request_identities(
    snapshot: StoryAnalysisInputSnapshot,
    plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    character_request_identity_hashes: Sequence[str],
    window_request_identity_hashes: Sequence[str],
    *,
    prompts: PromptRegistry,
) -> None:
    """Enforce the frozen exact-upstream identity contract.

    Recomputes the deterministic A6C/A6D request identity hashes using the
    existing preparation functions under the current exact profile and A6B
    plan, then compares them against the provided sequences. This is NOT a
    format check -- it is an actual identity verification that catches:
    * stale-profile identities (e.g. from a null-ceiling profile);
    * swapped or reordered sequences;
    * fabricated or placeholder hashes;
    * any mismatch between the claimed upstream and the actual current
      profile/plan.

    Rejects empty, missing, mismatched, or stale identities before any
    provider execution.
    """
    from .story_analysis_semantic import build_character_semantic_preparation
    from .story_analysis_window_semantic import (
        build_plot_window_semantic_preparation,
    )

    expected_char_count = len(plan.character_packages)
    expected_window_count = len(plan.windows)

    if len(character_request_identity_hashes) != expected_char_count:
        raise StoryAnalysisSemanticError(
            f"character_request_identity_hashes count "
            f"{len(character_request_identity_hashes)} does not match the "
            f"planned character count {expected_char_count}; A6E-2 requires "
            f"complete ordered upstream A6C request identities"
        )
    if len(window_request_identity_hashes) != expected_window_count:
        raise StoryAnalysisSemanticError(
            f"window_request_identity_hashes count "
            f"{len(window_request_identity_hashes)} does not match the "
            f"planned window count {expected_window_count}; A6E-2 requires "
            f"complete ordered upstream A6D request identities"
        )

    # Recompute the expected A6C identity hashes under the current exact
    # profile and plan (zero provider calls).
    a6c_prep = build_character_semantic_preparation(
        plan, profile, semantic_profile, prompts=prompts
    )
    expected_char_hashes = a6c_prep.character_request_identity_hashes

    # Recompute the expected A6D identity hashes under the current exact
    # profile and plan (zero provider calls).
    a6d_prep = build_plot_window_semantic_preparation(
        snapshot, plan, profile, semantic_profile, prompts=prompts
    )
    expected_window_hashes = a6d_prep.window_request_identity_hashes

    # Compare each provided hash against the recomputed expected hash.
    for i, (got, want) in enumerate(
        zip(character_request_identity_hashes, expected_char_hashes)
    ):
        if got != want:
            raise StoryAnalysisSemanticError(
                f"character_request_identity_hashes[{i}] mismatch: provided "
                f"{got!r} does not match the deterministically recomputed "
                f"A6C identity {want!r} under the current profile/plan; "
                f"the upstream A6C result is stale, fabricated, or from a "
                f"different profile"
            )
    for i, (got, want) in enumerate(
        zip(window_request_identity_hashes, expected_window_hashes)
    ):
        if got != want:
            raise StoryAnalysisSemanticError(
                f"window_request_identity_hashes[{i}] mismatch: provided "
                f"{got!r} does not match the deterministically recomputed "
                f"A6D identity {want!r} under the current profile/plan; "
                f"the upstream A6D result is stale, fabricated, or from a "
                f"different profile"
            )


# ---------------------------------------------------------------------------
# Stable request identity (backend-neutral).
# ---------------------------------------------------------------------------


def global_skeleton_request_identity_hash(
    *,
    consolidation_manifest_ref: ArtifactRef,
    plan_hash: str,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    prompt_identity: PromptAssetIdentity,
    output_schema_identity: OutputSchemaAssetIdentity,
    packet_hash: str,
    request_hash: str,
    character_request_identity_hashes: Sequence[str],
    window_request_identity_hashes: Sequence[str],
) -> str:
    """Compute the A6E-2 *stable request identity* for the global-skeleton
    request.

    This binds (via the existing :func:`short_drama.artifacts.canonical.
    content_hash` authority) every element required by the frozen A6E-2 spec:
    the exact upstream A5 identity, the A6 plan identity, the
    ``StoryAnalysisProfile`` id/hash, the ``SemanticLLMProfile`` id/hash, the
    prompt and output-schema asset identities, the complete global packet hash,
    the rendered request hash, and the ordered upstream A6C/A6D request
    identity hashes. It is intentionally backend-neutral: it never includes
    provider family / model / base_url / timeout / GPU / NP / slots /
    concurrency.
    """
    return content_hash(
        {
            "upstream_a5": consolidation_manifest_ref.to_dict(),
            "a6_plan_hash": plan_hash,
            "story_analysis_profile_id": profile.profile_id,
            "story_analysis_profile_hash": profile.content_hash(),
            "semantic_profile_id": semantic_profile.profile_id,
            "semantic_profile_hash": semantic_profile.semantic_profile_hash,
            "prompt": {
                "prompt_id": prompt_identity.prompt_id,
                "prompt_version": prompt_identity.prompt_version,
                "prompt_content_hash": prompt_identity.prompt_content_hash,
            },
            "output_schema": {
                "schema_id": output_schema_identity.schema_id,
                "schema_version": output_schema_identity.schema_version,
                "schema_hash": output_schema_identity.schema_hash,
            },
            "global_packet_hash": packet_hash,
            "rendered_request_hash": request_hash,
            "character_request_identity_hashes": list(character_request_identity_hashes),
            "window_request_identity_hashes": list(window_request_identity_hashes),
        }
    )


# ---------------------------------------------------------------------------
# Deterministic, zero-provider preparation.
# ---------------------------------------------------------------------------


def build_global_skeleton_semantic_preparation(
    snapshot: StoryAnalysisInputSnapshot,
    plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    character_analyses: Sequence[CharacterAnalysis],
    window_analyses: Sequence[PlotWindowAnalysis],
    *,
    prompts: PromptRegistry,
    character_request_identity_hashes: Sequence[str],
    window_request_identity_hashes: Sequence[str],
) -> GlobalSkeletonSemanticPreparation:
    """Deterministically build the complete A6E-2 global-skeleton request.

    Zero provider invocations. Validates the frozen A6E-2 identity, enforces
    the frozen ceiling (fail closed), validates complete coverage, verifies
    the exact upstream binding, renders the request, and computes the stable
    request identity hash.
    """
    _verify_global_skeleton_profile(profile, semantic_profile)
    _verify_snapshot_plan_manifest_identity(snapshot, plan)
    _verify_upstream_request_identities(
        snapshot,
        plan,
        profile,
        semantic_profile,
        character_request_identity_hashes,
        window_request_identity_hashes,
        prompts=prompts,
    )

    # Assemble the complete global packet from the A6C/A6D outputs + A5 index.
    context = build_global_skeleton_context(
        character_analyses, window_analyses, plan.global_index
    )

    # Enforce the frozen ceiling (fail closed before any provider call).
    enforce_global_skeleton_ceiling(profile, context)

    # Validate complete coverage (no missing/duplicate/extra character or window).
    validate_global_skeleton_coverage(context, plan)

    # Render the exact request.
    prompt_spec = prompts.load(
        A6E_GLOBAL_SKELETON_PROMPT_ID, version=A6E_GLOBAL_SKELETON_PROMPT_VERSION
    )
    output_schema = load_global_skeleton_output_schema()
    prompt_identity = PromptAssetIdentity(
        prompt_spec.prompt_id, prompt_spec.version, prompt_spec.content_hash
    )
    output_schema_identity = OutputSchemaAssetIdentity(
        output_schema.schema_id,
        output_schema.schema_version,
        output_schema.schema_hash,
    )

    rendered = build_global_skeleton_rendered_request(context, prompt_spec)
    request = build_structured_request(
        rendered_prompt=rendered,
        output_schema=output_schema,
        semantic_profile=semantic_profile,
    )

    # Compute the stable request identity.
    identity_hash = global_skeleton_request_identity_hash(
        consolidation_manifest_ref=plan.consolidation_manifest_ref,
        plan_hash=plan.plan_hash,
        profile=profile,
        semantic_profile=semantic_profile,
        prompt_identity=prompt_identity,
        output_schema_identity=output_schema_identity,
        packet_hash=context.content_hash(),
        request_hash=request.request_hash,
        character_request_identity_hashes=character_request_identity_hashes,
        window_request_identity_hashes=window_request_identity_hashes,
    )

    packet_tokens = context.estimated_tokens()
    rendered_tokens = estimate_tokens(rendered.system_text) + estimate_tokens(
        rendered.user_text
    )

    return GlobalSkeletonSemanticPreparation(
        plan=plan,
        profile=profile,
        semantic_profile=semantic_profile,
        prompt_identity=prompt_identity,
        output_schema_identity=output_schema_identity,
        context=context,
        request=request,
        request_identity_hash=identity_hash,
        request_hash=request.request_hash,
        packet_token_estimate=packet_tokens,
        rendered_request_token_estimate=rendered_tokens,
        character_request_identity_hashes=tuple(character_request_identity_hashes),
        window_request_identity_hashes=tuple(window_request_identity_hashes),
    )


# ---------------------------------------------------------------------------
# Evidence universe (complete A5 snapshot for the global pass).
# ---------------------------------------------------------------------------


def _global_evidence_universe(
    snapshot: StoryAnalysisInputSnapshot,
) -> dict[str, frozenset[str]]:
    """The complete A5 evidence universe for the global-skeleton pass.

    The global skeleton sees the WHOLE story, so every canonical ref in the
    A5 snapshot is valid. This is distinct from A6C/A6D which use per-character
    / per-window local evidence universes.
    """
    return {
        "fact": frozenset(f.fact_id for f in snapshot.facts),
        "evt": frozenset(e.event_id for e in snapshot.events),
        "rel": frozenset(r.relationship_id for r in snapshot.relationships),
        "conf": frozenset(c.conflict_id for c in snapshot.conflicts),
        "char": frozenset(c.canonical_id for c in snapshot.canonical_characters),
    }


# ---------------------------------------------------------------------------
# Semantic validation.
# ---------------------------------------------------------------------------


def _check_refs_in_universe(
    refs: Sequence[str], universe: frozenset[str], label: str
) -> str | None:
    """Check that every ref belongs to the evidence universe."""
    for ref in refs:
        if ref not in universe:
            return f"{label}: ref {ref!r} is not in the A5 evidence universe"
    return None


def _validate_interpretation_refs(
    interp: EvidenceBackedInterpretation,
    universe: dict[str, frozenset[str]],
    label: str,
) -> str | None:
    """Validate all supporting refs of an interpretation against the universe."""
    for refs, ns in (
        (interp.supporting_fact_refs, "fact"),
        (interp.supporting_event_refs, "evt"),
        (interp.supporting_relationship_refs, "rel"),
        (interp.supporting_conflict_refs, "conf"),
    ):
        detail = _check_refs_in_universe(refs, universe[ns], f"{label} [{ns}]")
        if detail is not None:
            return detail
    return None


def _event_narrative_order(
    snapshot: StoryAnalysisInputSnapshot,
) -> dict[str, int]:
    """Map event_id -> narrative_order for narrative-order validation."""
    return {e.event_id: e.narrative_order for e in snapshot.events}


def validate_global_skeleton_output(
    payload: dict[str, Any],
    snapshot: StoryAnalysisInputSnapshot,
) -> str | None:
    """Validate the complete A6E-2 global-skeleton provider output.

    Returns ``None`` when the output is semantically valid, or a bounded
    failure-detail string otherwise. Validates:

    * every canonical ref (evt / char / fact / rel / conf) against the complete
      A5 evidence universe;
    * structural anchors: arc start_event_ref must precede end_event_ref in
      narrative order;
    * event narrative order: global sections must have strictly increasing
      ordinals;
    * internal cross-references: major_arc_proposal_ordinals,
      major_turning_point_proposal_ordinals, and major_reveal_proposal_ordinals
      must reference valid proposal ordinals;
    * proposal ordinals must be unique within each collection.
    """
    universe = _global_evidence_universe(snapshot)
    narrative_order = _event_narrative_order(snapshot)

    # --- event_importance_overlay ------------------------------------------
    event_importance = payload.get("event_importance_overlay", [])
    seen_events: set[str] = set()
    for i, entry in enumerate(event_importance):
        evt_ref = entry.get("event_ref", "")
        if evt_ref not in universe["evt"]:
            return (
                f"event_importance_overlay[{i}]: event_ref {evt_ref!r} "
                f"is not in the A5 evidence universe"
            )
        if evt_ref in seen_events:
            return (
                f"event_importance_overlay[{i}]: duplicate event_ref {evt_ref!r}"
            )
        seen_events.add(evt_ref)
        interp = entry.get("interpretation", {})
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"event_importance_overlay[{i}]"
        )
        if detail is not None:
            return detail

    # --- arc_proposals -----------------------------------------------------
    arc_proposals = payload.get("arc_proposals", [])
    arc_ordinals: set[int] = set()
    for i, arc in enumerate(arc_proposals):
        ordinal = arc.get("proposal_ordinal")
        if not isinstance(ordinal, int) or ordinal < 0:
            return f"arc_proposals[{i}]: invalid proposal_ordinal {ordinal!r}"
        if ordinal in arc_ordinals:
            return f"arc_proposals[{i}]: duplicate proposal_ordinal {ordinal!r}"
        arc_ordinals.add(ordinal)

        for ref in arc.get("involved_character_refs", []):
            if ref not in universe["char"]:
                return (
                    f"arc_proposals[{i}]: involved_character_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        for ref in arc.get("involved_relationship_refs", []):
            if ref not in universe["rel"]:
                return (
                    f"arc_proposals[{i}]: involved_relationship_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        for ref in arc.get("supporting_event_refs", []):
            if ref not in universe["evt"]:
                return (
                    f"arc_proposals[{i}]: supporting_event_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        for ref in arc.get("supporting_fact_refs", []):
            if ref not in universe["fact"]:
                return (
                    f"arc_proposals[{i}]: supporting_fact_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        start_evt = arc.get("start_event_ref", "")
        end_evt = arc.get("end_event_ref", "")
        if start_evt not in universe["evt"]:
            return (
                f"arc_proposals[{i}]: start_event_ref {start_evt!r} "
                f"is not in the A5 evidence universe"
            )
        if end_evt not in universe["evt"]:
            return (
                f"arc_proposals[{i}]: end_event_ref {end_evt!r} "
                f"is not in the A5 evidence universe"
            )
        # Narrative order: start must precede end.
        if narrative_order[start_evt] >= narrative_order[end_evt]:
            return (
                f"arc_proposals[{i}]: start_event_ref {start_evt!r} "
                f"(narrative_order={narrative_order[start_evt]}) does not "
                f"precede end_event_ref {end_evt!r} "
                f"(narrative_order={narrative_order[end_evt]})"
            )
        interp = arc.get("interpretation", {})
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"arc_proposals[{i}]"
        )
        if detail is not None:
            return detail

    # --- turning_point_proposals -------------------------------------------
    tp_proposals = payload.get("turning_point_proposals", [])
    tp_ordinals: set[int] = set()
    for i, tp in enumerate(tp_proposals):
        ordinal = tp.get("proposal_ordinal")
        if not isinstance(ordinal, int) or ordinal < 0:
            return f"turning_point_proposals[{i}]: invalid proposal_ordinal {ordinal!r}"
        if ordinal in tp_ordinals:
            return f"turning_point_proposals[{i}]: duplicate proposal_ordinal {ordinal!r}"
        tp_ordinals.add(ordinal)

        evt_ref = tp.get("event_ref", "")
        if evt_ref not in universe["evt"]:
            return (
                f"turning_point_proposals[{i}]: event_ref {evt_ref!r} "
                f"is not in the A5 evidence universe"
            )
        for ref in tp.get("supporting_fact_refs", []):
            if ref not in universe["fact"]:
                return (
                    f"turning_point_proposals[{i}]: supporting_fact_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        for ref in tp.get("supporting_relationship_refs", []):
            if ref not in universe["rel"]:
                return (
                    f"turning_point_proposals[{i}]: supporting_relationship_ref "
                    f"{ref!r} is not in the A5 evidence universe"
                )
        interp = tp.get("interpretation", {})
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"turning_point_proposals[{i}]"
        )
        if detail is not None:
            return detail

    # --- reveal_proposals --------------------------------------------------
    reveal_proposals = payload.get("reveal_proposals", [])
    reveal_ordinals: set[int] = set()
    for i, rv in enumerate(reveal_proposals):
        ordinal = rv.get("proposal_ordinal")
        if not isinstance(ordinal, int) or ordinal < 0:
            return f"reveal_proposals[{i}]: invalid proposal_ordinal {ordinal!r}"
        if ordinal in reveal_ordinals:
            return f"reveal_proposals[{i}]: duplicate proposal_ordinal {ordinal!r}"
        reveal_ordinals.add(ordinal)

        for ref in rv.get("reveal_event_refs", []):
            if ref not in universe["evt"]:
                return (
                    f"reveal_proposals[{i}]: reveal_event_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        for ref in rv.get("supporting_fact_refs", []):
            if ref not in universe["fact"]:
                return (
                    f"reveal_proposals[{i}]: supporting_fact_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        for ref in rv.get("affected_character_refs", []):
            if ref not in universe["char"]:
                return (
                    f"reveal_proposals[{i}]: affected_character_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        for ref in rv.get("setup_event_refs", []):
            if ref not in universe["evt"]:
                return (
                    f"reveal_proposals[{i}]: setup_event_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        interp = rv.get("interpretation", {})
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"reveal_proposals[{i}]"
        )
        if detail is not None:
            return detail

    # --- foreshadow_payoff_proposals ---------------------------------------
    payoff_proposals = payload.get("foreshadow_payoff_proposals", [])
    payoff_ordinals: set[int] = set()
    for i, pf in enumerate(payoff_proposals):
        ordinal = pf.get("proposal_ordinal")
        if not isinstance(ordinal, int) or ordinal < 0:
            return (
                f"foreshadow_payoff_proposals[{i}]: invalid "
                f"proposal_ordinal {ordinal!r}"
            )
        if ordinal in payoff_ordinals:
            return (
                f"foreshadow_payoff_proposals[{i}]: duplicate "
                f"proposal_ordinal {ordinal!r}"
            )
        payoff_ordinals.add(ordinal)

        for ref in pf.get("setup_event_refs", []):
            if ref not in universe["evt"]:
                return (
                    f"foreshadow_payoff_proposals[{i}]: setup_event_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        for ref in pf.get("payoff_event_refs", []):
            if ref not in universe["evt"]:
                return (
                    f"foreshadow_payoff_proposals[{i}]: payoff_event_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        interp = pf.get("interpretation", {})
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"foreshadow_payoff_proposals[{i}]"
        )
        if detail is not None:
            return detail

    # --- global_structure --------------------------------------------------
    structure = payload.get("global_structure", {})
    for field in ("main_conflict", "main_plot", "ending_state"):
        interp = structure.get(field, {})
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"global_structure.{field}"
        )
        if detail is not None:
            return detail
    for i, interp in enumerate(structure.get("secondary_conflicts", [])):
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"global_structure.secondary_conflicts[{i}]"
        )
        if detail is not None:
            return detail
    for i, interp in enumerate(structure.get("subplots", [])):
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"global_structure.subplots[{i}]"
        )
        if detail is not None:
            return detail

    # Global sections: unique strictly-increasing ordinals, valid event refs.
    sections = structure.get("global_sections", [])
    seen_section_ordinals: set[int] = set()
    prev_ordinal = 0
    for i, section in enumerate(sections):
        ordinal = section.get("section_ordinal")
        if not isinstance(ordinal, int) or ordinal < 1:
            return (
                f"global_structure.global_sections[{i}]: invalid "
                f"section_ordinal {ordinal!r}"
            )
        if ordinal in seen_section_ordinals:
            return (
                f"global_structure.global_sections[{i}]: duplicate "
                f"section_ordinal {ordinal!r}"
            )
        if ordinal <= prev_ordinal:
            return (
                f"global_structure.global_sections[{i}]: section_ordinal "
                f"{ordinal!r} is not strictly increasing (prev={prev_ordinal})"
            )
        seen_section_ordinals.add(ordinal)
        prev_ordinal = ordinal
        for ref in section.get("event_refs", []):
            if ref not in universe["evt"]:
                return (
                    f"global_structure.global_sections[{i}]: event_ref {ref!r} "
                    f"is not in the A5 evidence universe"
                )
        interp = section.get("interpretation", {})
        detail = _validate_interpretation_payload_refs(
            interp, universe, f"global_structure.global_sections[{i}]"
        )
        if detail is not None:
            return detail

    # main_character_refs
    for ref in structure.get("main_character_refs", []):
        if ref not in universe["char"]:
            return (
                f"global_structure.main_character_refs: ref {ref!r} "
                f"is not in the A5 evidence universe"
            )

    # Internal cross-references: major_*_proposal_ordinals must be valid.
    for field, valid_ordinals in (
        ("major_arc_proposal_ordinals", arc_ordinals),
        ("major_turning_point_proposal_ordinals", tp_ordinals),
        ("major_reveal_proposal_ordinals", reveal_ordinals),
    ):
        for ordinal in structure.get(field, []):
            if ordinal not in valid_ordinals:
                return (
                    f"global_structure.{field}: ordinal {ordinal!r} does not "
                    f"reference a valid proposal"
                )

    return None


def _validate_interpretation_payload_refs(
    interp: dict[str, Any],
    universe: dict[str, frozenset[str]],
    label: str,
) -> str | None:
    """Validate the supporting refs of a raw interpretation payload dict."""
    for field, ns in (
        ("supporting_fact_refs", "fact"),
        ("supporting_event_refs", "evt"),
        ("supporting_relationship_refs", "rel"),
        ("supporting_conflict_refs", "conf"),
    ):
        for ref in interp.get(field, []):
            if ref not in universe[ns]:
                return f"{label}: {ns} ref {ref!r} is not in the A5 evidence universe"
    return None


# ---------------------------------------------------------------------------
# Python-assigned deterministic ID allocation.
# ---------------------------------------------------------------------------


def _assign_arc_ids(arc_proposals: list[dict[str, Any]]) -> list[str]:
    """Deterministically assign arc IDs based on proposal order."""
    return [f"arc_{i + 1:06d}" for i in range(len(arc_proposals))]


def _assign_turning_point_ids(tp_proposals: list[dict[str, Any]]) -> list[str]:
    """Deterministically assign turning-point IDs based on proposal order."""
    return [f"turn_{i + 1:06d}" for i in range(len(tp_proposals))]


def _assign_reveal_ids(reveal_proposals: list[dict[str, Any]]) -> list[str]:
    """Deterministically assign reveal IDs based on proposal order."""
    return [f"reveal_{i + 1:06d}" for i in range(len(reveal_proposals))]


def _assign_payoff_ids(payoff_proposals: list[dict[str, Any]]) -> list[str]:
    """Deterministically assign foreshadow-payoff IDs based on proposal order."""
    return [f"payoff_{i + 1:06d}" for i in range(len(payoff_proposals))]


# ---------------------------------------------------------------------------
# Typed artifact construction (Python-assigned IDs, model-proposed semantics).
# ---------------------------------------------------------------------------


def _build_global_event_analysis(
    payload: dict[str, Any],
    plan: StoryAnalysisPlan,
    window_analyses: tuple[PlotWindowAnalysis, ...],
) -> GlobalEventAnalysis:
    """Construct the typed GlobalEventAnalysis from the validated payload."""
    event_importance_raw = payload.get("event_importance_overlay", [])
    event_importance = tuple(
        GlobalEventImportance(
            event_ref=entry["event_ref"],
            interpretation=EvidenceBackedInterpretation.from_dict(entry["interpretation"]),
        )
        for entry in event_importance_raw
    )
    return GlobalEventAnalysis(
        schema_version=STORY_ANALYSIS_SCHEMA_VERSION,
        plot_window_plan_hash=plan.plan_hash,
        windows=window_analyses,
        event_importance_overlay=event_importance,
        coverage_metadata=(
            ("canonical_event_count", len(plan.event_stream)),
            ("event_importance_count", len(event_importance)),
            ("plot_window_count", len(window_analyses)),
        ),
    )


def _build_arc_analysis(payload: dict[str, Any]) -> ArcAnalysis:
    """Construct the typed ArcAnalysis from the validated payload with
    Python-assigned deterministic IDs."""
    # Arcs
    arc_proposals = payload.get("arc_proposals", [])
    arc_ids = _assign_arc_ids(arc_proposals)
    arcs = tuple(
        StoryArc(
            arc_id=arc_ids[i],
            arc_kind=arc["arc_kind"],
            involved_character_refs=tuple(arc.get("involved_character_refs", [])),
            involved_relationship_refs=tuple(arc.get("involved_relationship_refs", [])),
            supporting_event_refs=tuple(arc.get("supporting_event_refs", [])),
            supporting_fact_refs=tuple(arc.get("supporting_fact_refs", [])),
            start_event_ref=arc["start_event_ref"],
            end_event_ref=arc["end_event_ref"],
            interpretation=EvidenceBackedInterpretation.from_dict(arc["interpretation"]),
        )
        for i, arc in enumerate(arc_proposals)
    )

    # Turning points
    tp_proposals = payload.get("turning_point_proposals", [])
    tp_ids = _assign_turning_point_ids(tp_proposals)
    turning_points = tuple(
        TurningPoint(
            turning_point_id=tp_ids[i],
            event_ref=tp["event_ref"],
            supporting_fact_refs=tuple(tp.get("supporting_fact_refs", [])),
            supporting_relationship_refs=tuple(
                tp.get("supporting_relationship_refs", [])
            ),
            interpretation=EvidenceBackedInterpretation.from_dict(
                tp["interpretation"]
            ),
        )
        for i, tp in enumerate(tp_proposals)
    )

    # Reveals
    reveal_proposals = payload.get("reveal_proposals", [])
    reveal_ids = _assign_reveal_ids(reveal_proposals)
    reveals = tuple(
        Reveal(
            reveal_id=reveal_ids[i],
            reveal_event_refs=tuple(rv.get("reveal_event_refs", [])),
            supporting_fact_refs=tuple(rv.get("supporting_fact_refs", [])),
            affected_character_refs=tuple(rv.get("affected_character_refs", [])),
            setup_event_refs=tuple(rv.get("setup_event_refs", [])),
            interpretation=EvidenceBackedInterpretation.from_dict(
                rv["interpretation"]
            ),
        )
        for i, rv in enumerate(reveal_proposals)
    )

    # Foreshadow payoffs
    payoff_proposals = payload.get("foreshadow_payoff_proposals", [])
    payoff_ids = _assign_payoff_ids(payoff_proposals)
    payoffs = tuple(
        ForeshadowPayoff(
            payoff_id=payoff_ids[i],
            setup_event_refs=tuple(pf.get("setup_event_refs", [])),
            payoff_event_refs=tuple(pf.get("payoff_event_refs", [])),
            interpretation=EvidenceBackedInterpretation.from_dict(
                pf["interpretation"]
            ),
        )
        for i, pf in enumerate(payoff_proposals)
    )

    return ArcAnalysis(
        schema_version=STORY_ANALYSIS_SCHEMA_VERSION,
        arcs=arcs,
        turning_points=turning_points,
        reveals=reveals,
        foreshadow_payoffs=payoffs,
    )


def _build_global_structure(payload: dict[str, Any]) -> GlobalStructure:
    """Construct the typed GlobalStructure from the validated payload.

    The ``major_arc_refs``, ``major_turning_point_refs``, and
    ``major_reveal_refs`` fields are resolved from the provider's proposal
    ordinals to the Python-assigned IDs using explicit maps (the provider
    schema allows non-negative proposal ordinals that are not necessarily
    equal to array indices).
    """
    structure_raw = payload.get("global_structure", {})

    # Build explicit maps: proposal_ordinal -> Python-assigned final ID.
    # Python assigns IDs deterministically by array position (0-indexed);
    # the cross-reference resolution uses the provider's proposal_ordinal.
    arc_proposals = payload.get("arc_proposals", [])
    arc_ids = _assign_arc_ids(arc_proposals)
    arc_ordinal_to_id: dict[int, str] = {
        arc["proposal_ordinal"]: arc_ids[i] for i, arc in enumerate(arc_proposals)
    }
    major_arc_refs = tuple(
        arc_ordinal_to_id[ordinal]
        for ordinal in structure_raw.get("major_arc_proposal_ordinals", [])
    )

    tp_proposals = payload.get("turning_point_proposals", [])
    tp_ids = _assign_turning_point_ids(tp_proposals)
    tp_ordinal_to_id: dict[int, str] = {
        tp["proposal_ordinal"]: tp_ids[i] for i, tp in enumerate(tp_proposals)
    }
    major_tp_refs = tuple(
        tp_ordinal_to_id[ordinal]
        for ordinal in structure_raw.get("major_turning_point_proposal_ordinals", [])
    )

    reveal_proposals = payload.get("reveal_proposals", [])
    reveal_ids = _assign_reveal_ids(reveal_proposals)
    reveal_ordinal_to_id: dict[int, str] = {
        rv["proposal_ordinal"]: reveal_ids[i]
        for i, rv in enumerate(reveal_proposals)
    }
    major_reveal_refs = tuple(
        reveal_ordinal_to_id[ordinal]
        for ordinal in structure_raw.get("major_reveal_proposal_ordinals", [])
    )

    # Global sections
    sections_raw = structure_raw.get("global_sections", [])
    sections = tuple(
        GlobalSection(
            section_ordinal=section["section_ordinal"],
            label_zh=section["label_zh"],
            event_refs=tuple(section.get("event_refs", [])),
            interpretation=EvidenceBackedInterpretation.from_dict(
                section["interpretation"]
            ),
        )
        for section in sections_raw
    )

    return GlobalStructure(
        schema_version=STORY_ANALYSIS_SCHEMA_VERSION,
        main_conflict=EvidenceBackedInterpretation.from_dict(
            structure_raw["main_conflict"]
        ),
        secondary_conflicts=tuple(
            EvidenceBackedInterpretation.from_dict(x)
            for x in structure_raw.get("secondary_conflicts", [])
        ),
        main_plot=EvidenceBackedInterpretation.from_dict(
            structure_raw["main_plot"]
        ),
        subplots=tuple(
            EvidenceBackedInterpretation.from_dict(x)
            for x in structure_raw.get("subplots", [])
        ),
        ending_state=EvidenceBackedInterpretation.from_dict(
            structure_raw["ending_state"]
        ),
        global_sections=sections,
        main_character_refs=tuple(
            structure_raw.get("main_character_refs", [])
        ),
        major_arc_refs=major_arc_refs,
        major_turning_point_refs=major_tp_refs,
        major_reveal_refs=major_reveal_refs,
    )


# ---------------------------------------------------------------------------
# Provenance verification (backend-neutral, fail closed, no retry).
# ---------------------------------------------------------------------------


def _verify_global_skeleton_provenance(
    provenance: LLMInvocationProvenance,
    request: StructuredGenerationRequest,
) -> None:
    """Verify the backend-neutral semantic/request identity of a provider
    result against the exact global-skeleton request.

    A mismatch fails closed with :class:`StoryAnalysisProvenanceError`
    (no semantic retry).
    """
    checks = (
        (provenance.semantic_profile_id, request.semantic_profile.profile_id),
        (
            provenance.semantic_profile_hash,
            request.semantic_profile.semantic_profile_hash,
        ),
        (provenance.prompt_id, request.rendered_prompt.prompt_id),
        (provenance.prompt_version, request.rendered_prompt.prompt_version),
        (
            provenance.prompt_content_hash,
            request.rendered_prompt.prompt_content_hash,
        ),
        (
            provenance.rendered_prompt_hash,
            request.rendered_prompt.rendered_prompt_hash,
        ),
        (provenance.output_schema_id, request.output_schema.schema_id),
        (
            provenance.output_schema_version,
            request.output_schema.schema_version,
        ),
        (provenance.output_schema_hash, request.output_schema.schema_hash),
        (provenance.request_hash, request.request_hash),
    )
    for got, want in checks:
        if got != want:
            raise StoryAnalysisProvenanceError(
                f"global-skeleton provenance field mismatch: got {got!r}, "
                f"expected {want!r}"
            )


# ---------------------------------------------------------------------------
# Bounded semantic execution.
# ---------------------------------------------------------------------------


def _attempt_global_skeleton_round(
    request: StructuredGenerationRequest,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    snapshot: StoryAnalysisInputSnapshot,
    *,
    round_number: int,
) -> dict[str, Any] | _GlobalSkeletonRetryableInvalid:
    """One semantic round for the global-skeleton request.

    A technical :class:`short_drama.llm.LLMError` (transport, malformed JSON,
    or schema-invalid output) propagates unchanged (A-I3-owned); a provenance /
    request-hash mismatch fails closed (no retry); only a schema-valid result
    rejected by semantic validation is returned as a bounded semantic
    invalidity.
    """
    execution_options = (
        GenerationExecutionOptions(prompt_context_reuse="disabled")
        if round_number > 1
        else None
    )
    result = llm_client.generate_structured(
        request.rendered_prompt,
        request.output_schema,
        semantic_profile,
        execution_options=execution_options,
    )
    _verify_global_skeleton_provenance(result.provenance, request)

    payload = result.parsed_json
    if not isinstance(payload, dict):
        return _GlobalSkeletonRetryableInvalid(
            detail="provider output must be a JSON object"
        )

    # Semantic validation against the A5 evidence universe.
    detail = validate_global_skeleton_output(payload, snapshot)
    if detail is not None:
        return _GlobalSkeletonRetryableInvalid(detail=detail)

    return payload


def _execute_global_skeleton_request(
    preparation: GlobalSkeletonSemanticPreparation,
    snapshot: StoryAnalysisInputSnapshot,
    llm_client: LLMClient,
) -> tuple[dict[str, Any], int]:
    """Execute the bounded semantic rounds for the global-skeleton request.

    Returns the accepted payload and the number of semantic rounds consumed.
    Raises :class:`StoryAnalysisGlobalSkeletonSemanticGenerationError` once the
    two frozen semantic rounds are both exhausted by schema-valid-but-
    semantically-invalid results.
    """
    outcome = _attempt_global_skeleton_round(
        preparation.request,
        preparation.semantic_profile,
        llm_client,
        snapshot,
        round_number=1,
    )
    rounds = 1
    if isinstance(outcome, _GlobalSkeletonRetryableInvalid):
        first_detail = outcome.detail
        outcome = _attempt_global_skeleton_round(
            preparation.request,
            preparation.semantic_profile,
            llm_client,
            snapshot,
            round_number=2,
        )
        rounds = 2
        if isinstance(outcome, _GlobalSkeletonRetryableInvalid):
            raise StoryAnalysisGlobalSkeletonSemanticGenerationError(
                request_hash=preparation.request_hash,
                rounds_attempted=A6E_MAX_GENERATION_ROUNDS,
                last_failure_details=(first_detail, outcome.detail),
            )
    return outcome, rounds


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def resolve_global_skeleton(
    snapshot: StoryAnalysisInputSnapshot,
    plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    character_analyses: Sequence[CharacterAnalysis],
    window_analyses: Sequence[PlotWindowAnalysis],
    llm_client: LLMClient,
    *,
    prompts: PromptRegistry,
    character_request_identity_hashes: Sequence[str],
    window_request_identity_hashes: Sequence[str],
) -> GlobalSkeletonSemanticResult:
    """Run the complete A6E-2 global-skeleton semantic pass.

    One provider request for the complete whole-story global skeleton, with up
    to the frozen ``max_generation_rounds`` (2) semantic rounds. Returns the
    complete in-memory typed result set (``GlobalEventAnalysis``,
    ``ArcAnalysis``, ``GlobalStructure``) plus the preparation metadata.

    A failed global execution does NOT return a partially authoritative result:
    any semantic exhaustion, provenance mismatch, or technical LLM failure
    propagates before the result is built.

    In-memory only: no A6 artifact is persisted and no A5/A4 CURRENT is
    mutated.
    """
    preparation = build_global_skeleton_semantic_preparation(
        snapshot,
        plan,
        profile,
        semantic_profile,
        character_analyses,
        window_analyses,
        prompts=prompts,
        character_request_identity_hashes=character_request_identity_hashes,
        window_request_identity_hashes=window_request_identity_hashes,
    )

    payload, rounds = _execute_global_skeleton_request(
        preparation, snapshot, llm_client
    )

    # Construct the three typed artifacts with Python-assigned IDs.
    window_analyses_tuple = tuple(window_analyses)
    global_event_analysis = _build_global_event_analysis(
        payload, plan, window_analyses_tuple
    )
    arc_analysis = _build_arc_analysis(payload)
    global_structure = _build_global_structure(payload)

    return GlobalSkeletonSemanticResult(
        preparation=preparation,
        global_event_analysis=global_event_analysis,
        arc_analysis=arc_analysis,
        global_structure=global_structure,
        rounds_consumed=rounds,
    )
