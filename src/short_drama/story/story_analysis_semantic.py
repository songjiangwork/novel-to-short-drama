"""v1.2 A6C -- character analysis semantic pass (provider execution).

This slice implements the **character analysis** semantic pass of A6 (Issue #88)
as specified by the frozen A6 implementation plan (slice ``A6C``) and the
frozen A6 architecture doc (``06-A6-global-story-bible.md``). It is the first
semantic pass of the hierarchical A6 synthesis: A6A produced the domain
contracts, A6B produced the exact deterministic planning (character evidence
packages over the complete canonical character universe, joined by exact refs),
and A6C turns those **exact** evidence packages into a complete in-memory
:class:`~short_drama.story.CharacterAnalysisSet`.

A6C owns:

* **deterministic character request rendering** -- one provider request per
  canonical character, whose ``character_context_json`` is the canonical JSON of
  that character's exact A6B evidence package; the rendered request content and
  a *stable request identity* that binds the exact upstream A5 identity, the A6
  plan identity, the ``StoryAnalysisProfile`` id/hash, the
  ``SemanticLLMProfile`` id/hash, the prompt and output-schema asset
  identities, the character evidence packet hash, and the actual rendered
  request hash;
* **character evidence-ref validation** -- the typed
  :class:`~short_drama.story.CharacterAnalysis` (frozen A6A contract) plus a
  per-character evidence-universe membership check for every referenced fact /
  event / relationship / conflict ref (a ref that merely exists somewhere in
  the global A5 snapshot is **not** sufficient; it must belong to the evidence
  universe actually supplied to that character);
* **bounded semantic regeneration** -- up to the frozen ``max_generation_rounds``
  (2) semantic rounds per character, where only *semantic* invalidity (a
  schema-valid provider result rejected by typed load or exact evidence-ref
  validation) triggers a regeneration round. A schema-invalid output and every
  other technical LLM failure remain owned by A-I3 (``short_drama.llm``) and are
  propagated, never translated into a semantic retry.

A6C is **in-memory only**: it consumes the exact A6B
:class:`~short_drama.story.StoryAnalysisPlan` and produces a complete
:class:`~short_drama.story.CharacterAnalysisSet` plus the ordered stable request
identities that A6D/A6E/A6F will fold into the final
:class:`~short_drama.story.A6SemanticIdentity`. A6C does NOT persist any A6
artifact, does NOT mutate any A5/A4 CURRENT, does NOT adapt the character
universe, and does NOT emit any downstream (A6D/A6E/A6F / B1) semantics.

Execution is **serial** (canonical character order). Optional bounded
concurrency is intentionally deferred: the A5 two-stage provider-neutral
executor is reusable but wiring it here would expand the A6C surface without
changing the frozen semantics, so A6C keeps the minimal correct serial path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    StoryAnalysisModelError,
    StoryAnalysisProvenanceError,
    StoryAnalysisSemanticError,
    StoryAnalysisSemanticGenerationError,
    StoryIntegrityError,
)
from .story_analysis import (
    CharacterAnalysis,
    CharacterAnalysisSet,
    STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1,
    STORY_ANALYSIS_SCHEMA_VERSION,
    StoryAnalysisProfile,
)
from .story_analysis_planning import (
    CharacterEvidencePackage,
    StoryAnalysisPlan,
)


# ---------------------------------------------------------------------------
# Frozen A6C identity (prompt / output schema / semantic profile).
# ---------------------------------------------------------------------------

#: The exact A6C character semantic pass prompt (frozen, reviewed).
A6C_CHARACTER_PROMPT_ID = "a6.character-analysis"
A6C_CHARACTER_PROMPT_VERSION = 1
#: The exact A6C character analysis output schema (frozen, reviewed).
A6C_CHARACTER_OUTPUT_SCHEMA_ID = "a6-character-analysis-output"
A6C_CHARACTER_OUTPUT_SCHEMA_VERSION = 1
A6C_CHARACTER_OUTPUT_SCHEMA_PATH = (
    SCHEMAS_DIR / "a6-character-analysis-output.schema.json"
)
#: The exact A6C semantic profile (frozen, reviewed).
A6C_SEMANTIC_PROFILE_ID = "story-analysis-llm-v1"
A6C_SEMANTIC_PROFILE_PATH = PROFILES_DIR / "story_analysis_llm_v1.yaml"
#: The frozen A6C semantic regeneration budget (initial + at most 1
#: regeneration), re-exposed from the A6A contract so A6C and A6A can never
#: drift.
A6C_MAX_GENERATION_ROUNDS = STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1

#: The single prompt variable rendered for every character request.
_CHARACTER_PROMPT_VARIABLE = "character_context_json"


# ---------------------------------------------------------------------------
# Result objects (in-memory, no persistence).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CharacterSemanticPreparation:
    """Deterministic, zero-provider A6C character request preparation.

    ``character_refs`` and every per-character tuple are in the exact A6B
    canonical planned order and aligned with ``plan.character_packages``.
    ``character_request_identity_hashes`` are the backend-neutral *stable
    request identities* (see :func:`character_request_identity_hash`) that A6D/
    A6E/A6F fold into the final A6 semantic identity; ``character_request_hashes``
    are the raw :class:`~short_drama.llm.StructuredGenerationRequest.request_hash`
    values (the actual rendered-request content hashes).
    """

    plan: StoryAnalysisPlan
    profile: StoryAnalysisProfile
    semantic_profile: SemanticLLMProfile
    prompt_identity: PromptAssetIdentity
    output_schema_identity: OutputSchemaAssetIdentity
    character_refs: tuple[str, ...]
    character_requests: tuple[StructuredGenerationRequest, ...]
    character_request_identity_hashes: tuple[str, ...]
    character_request_hashes: tuple[str, ...]
    packet_token_estimates: tuple[int, ...]
    rendered_request_token_estimates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CharacterSemanticResult:
    """Complete in-memory A6C character semantic result (no persistence)."""

    preparation: CharacterSemanticPreparation
    analyses: tuple[CharacterAnalysis, ...]
    character_analysis_set: CharacterAnalysisSet
    #: ``(character_ref, semantic_rounds_consumed)`` in canonical planned order.
    character_rounds: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _CharacterRetryableInvalid:
    """A schema-valid provider result rejected by semantic validation.

    Carries a bounded failure detail (never the raw provider body) for the
    bounded-regeneration diagnostic. This is a *semantic* invalidity, not a
    technical LLM failure.
    """

    detail: str


# ---------------------------------------------------------------------------
# Frozen asset loading (A5C-B pattern: exact path + exact profile id).
# ---------------------------------------------------------------------------


def load_character_semantic_profile() -> SemanticLLMProfile:
    """Load the A6C character-analysis semantic profile.

    The exact identity (``story-analysis-llm-v1``) is verified during execution
    by :func:`_verify_character_profile`, matching the A5C reviewed pattern.
    """
    return load_semantic_profile(A6C_SEMANTIC_PROFILE_PATH)


def load_character_output_schema() -> OutputSchema:
    """Load the frozen A6C character analysis output schema."""
    try:
        schema_data = load_json(A6C_CHARACTER_OUTPUT_SCHEMA_PATH)
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(
            f"failed to load character analysis output schema: {exc}"
        ) from exc
    if not isinstance(schema_data, dict):
        raise StoryIntegrityError(
            "character analysis output schema must be a JSON object"
        )
    return OutputSchema.create(
        schema_id=A6C_CHARACTER_OUTPUT_SCHEMA_ID,
        schema_version=A6C_CHARACTER_OUTPUT_SCHEMA_VERSION,
        schema=schema_data,
    )


def load_character_semantic_assets() -> tuple[SemanticLLMProfile, OutputSchema]:
    """Load the exact (semantic profile, output schema) A6C asset pair."""
    return load_character_semantic_profile(), load_character_output_schema()


# ---------------------------------------------------------------------------
# Profile identity verification (fail closed).
# ---------------------------------------------------------------------------


def _verify_character_profile(
    profile: StoryAnalysisProfile, semantic_profile: SemanticLLMProfile
) -> None:
    """Verify the profile pins the exact frozen A6C character pass identity."""
    if semantic_profile.profile_id != A6C_SEMANTIC_PROFILE_ID:
        raise StoryAnalysisSemanticError(
            f"semantic profile must be {A6C_SEMANTIC_PROFILE_ID!r}, "
            f"got {semantic_profile.profile_id!r}"
        )
    if profile.max_generation_rounds != A6C_MAX_GENERATION_ROUNDS:
        raise StoryAnalysisSemanticError(
            f"max_generation_rounds must be {A6C_MAX_GENERATION_ROUNDS}, "
            f"got {profile.max_generation_rounds}"
        )
    ca = profile.character_analysis
    if ca.semantic_profile_id != A6C_SEMANTIC_PROFILE_ID:
        raise StoryAnalysisSemanticError(
            f"character_analysis.semantic_profile_id must be "
            f"{A6C_SEMANTIC_PROFILE_ID!r}, got {ca.semantic_profile_id!r}"
        )
    if (
        ca.prompt_id != A6C_CHARACTER_PROMPT_ID
        or ca.prompt_version != A6C_CHARACTER_PROMPT_VERSION
    ):
        raise StoryAnalysisSemanticError(
            f"character_analysis prompt must be "
            f"{A6C_CHARACTER_PROMPT_ID!r} v{A6C_CHARACTER_PROMPT_VERSION}, got "
            f"{ca.prompt_id!r} v{ca.prompt_version}"
        )
    if (
        ca.output_schema_id != A6C_CHARACTER_OUTPUT_SCHEMA_ID
        or ca.output_schema_version != A6C_CHARACTER_OUTPUT_SCHEMA_VERSION
    ):
        raise StoryAnalysisSemanticError(
            f"character_analysis output schema must be "
            f"{A6C_CHARACTER_OUTPUT_SCHEMA_ID!r} v{A6C_CHARACTER_OUTPUT_SCHEMA_VERSION}, "
            f"got {ca.output_schema_id!r} v{ca.output_schema_version}"
        )


# ---------------------------------------------------------------------------
# Stable request identity (backend-neutral).
# ---------------------------------------------------------------------------


def character_request_identity_hash(
    *,
    consolidation_manifest_ref,
    plan_hash: str,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    prompt_identity: PromptAssetIdentity,
    output_schema_identity: OutputSchemaAssetIdentity,
    character_ref: str,
    packet_hash: str,
    request_hash: str,
) -> str:
    """Compute the A6C *stable request identity* for one character request.

    This binds (via the existing :func:`short_drama.artifacts.canonical.
    content_hash` authority over the existing asset-identity / request-hash
    facilities) every element required by the frozen A6C spec: the exact
    upstream A5 identity, the A6 plan identity, the ``StoryAnalysisProfile``
    id/hash, the ``SemanticLLMProfile`` id/hash, the prompt and output-schema
    asset identities, the character evidence packet hash, and the actual
    rendered request hash. It is intentionally backend-neutral: it never
    includes provider family / model / base_url / timeout / GPU / NP / slots /
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
            "character_ref": character_ref,
            "character_evidence_packet_hash": packet_hash,
            "rendered_request_hash": request_hash,
        }
    )


# ---------------------------------------------------------------------------
# Deterministic, zero-provider preparation.
# ---------------------------------------------------------------------------


def _complete_rendered_request_tokens(rendered: RenderedPrompt) -> int:
    """utf8-bytes-div3-v1 estimate of the complete rendered provider request
    (system + user framing), which is strictly larger than the evidence packet
    alone and is measured separately from the frozen packet budget."""
    return estimate_tokens(rendered.system_text) + estimate_tokens(rendered.user_text)


def build_character_semantic_preparation(
    plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
) -> CharacterSemanticPreparation:
    """Deterministically render one request per canonical planned character.

    Zero provider invocations. Validates the frozen A6C identity and the
    frozen character packet budget (fail closed -- never truncate) before any
    provider execution would occur.
    """
    _verify_character_profile(profile, semantic_profile)
    prompt_spec = prompts.load(
        A6C_CHARACTER_PROMPT_ID, version=A6C_CHARACTER_PROMPT_VERSION
    )
    output_schema = load_character_output_schema()
    prompt_identity = PromptAssetIdentity(
        prompt_spec.prompt_id, prompt_spec.version, prompt_spec.content_hash
    )
    output_schema_identity = OutputSchemaAssetIdentity(
        output_schema.schema_id,
        output_schema.schema_version,
        output_schema.schema_hash,
    )

    budget = profile.planning_policy.character_packet_max_estimated_tokens
    character_refs: list[str] = []
    character_requests: list[StructuredGenerationRequest] = []
    character_request_identity_hashes: list[str] = []
    character_request_hashes: list[str] = []
    packet_token_estimates: list[int] = []
    rendered_request_token_estimates: list[int] = []

    for package in plan.character_packages:
        packet_tokens = package.estimated_tokens()
        if packet_tokens > budget:
            raise StoryAnalysisSemanticError(
                f"character evidence packet for {package.character_ref} "
                f"({packet_tokens} estimated tokens) exceeds the frozen "
                f"character_packet_max_estimated_tokens budget ({budget}); "
                f"refusing to truncate"
            )
        character_context_json = package.canonical_bytes().decode("utf-8")
        rendered = render_prompt(
            prompt_spec, {_CHARACTER_PROMPT_VARIABLE: character_context_json}
        )
        request = build_structured_request(
            rendered_prompt=rendered,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        identity_hash = character_request_identity_hash(
            consolidation_manifest_ref=plan.consolidation_manifest_ref,
            plan_hash=plan.plan_hash,
            profile=profile,
            semantic_profile=semantic_profile,
            prompt_identity=prompt_identity,
            output_schema_identity=output_schema_identity,
            character_ref=package.character_ref,
            packet_hash=package.content_hash(),
            request_hash=request.request_hash,
        )
        character_refs.append(package.character_ref)
        character_requests.append(request)
        character_request_identity_hashes.append(identity_hash)
        character_request_hashes.append(request.request_hash)
        packet_token_estimates.append(packet_tokens)
        rendered_request_token_estimates.append(_complete_rendered_request_tokens(rendered))

    return CharacterSemanticPreparation(
        plan=plan,
        profile=profile,
        semantic_profile=semantic_profile,
        prompt_identity=prompt_identity,
        output_schema_identity=output_schema_identity,
        character_refs=tuple(character_refs),
        character_requests=tuple(character_requests),
        character_request_identity_hashes=tuple(character_request_identity_hashes),
        character_request_hashes=tuple(character_request_hashes),
        packet_token_estimates=tuple(packet_token_estimates),
        rendered_request_token_estimates=tuple(rendered_request_token_estimates),
    )


# ---------------------------------------------------------------------------
# Semantic validation (typed load + exact evidence-ref membership).
# ---------------------------------------------------------------------------


def _character_evidence_universe(package: CharacterEvidencePackage) -> dict[str, frozenset[str]]:
    """The exact per-character supporting-reference universe.

    A referenced ref is only valid if it belongs to this character's own
    evidence package (facts / events / relationships / conflicts). Unresolved
    entities are supplied as context but are not referenceable through the A6A
    contract, so they are not part of the ref universe.
    """
    return {
        "fact": frozenset(fact.fact_id for fact in package.related_facts),
        "evt": frozenset(event.event_id for event in package.participating_events),
        "rel": frozenset(rel.relationship_id for rel in package.relationships),
        "conf": frozenset(conflict.conflict_id for conflict in package.story_conflicts),
    }


def _all_interpretations(analysis: CharacterAnalysis) -> list[tuple[str, "EvidenceBackedInterpretation"]]:
    """Every evidence-backed interpretation field of an analysis, with a
    stable label, in schema order."""
    out: list[tuple[str, object]] = [("role", analysis.role)]
    for index, interp in enumerate(analysis.goals):
        out.append((f"goals[{index}]", interp))
    for index, interp in enumerate(analysis.motivations):
        out.append((f"motivations[{index}]", interp))
    for index, interp in enumerate(analysis.traits):
        out.append((f"traits[{index}]", interp))
    out.append(("arc_summary", analysis.arc_summary))
    for index, interp in enumerate(analysis.unresolved_or_conflicting_points):
        out.append((f"unresolved_or_conflicting_points[{index}]", interp))
    return out


def validate_character_evidence(
    analysis: CharacterAnalysis, package: CharacterEvidencePackage
) -> str | None:
    """Validate one analysis against its exact evidence universe.

    Returns ``None`` when the analysis is semantically valid, or a bounded
    failure-detail string otherwise. This layers on top of the frozen A6A
    domain contract (which already enforces the exact keys, ref namespaces,
    ``explicit``/``inferred`` evidence modes, and duplicate-free refs). The
    A6C-specific invariant is:

    * the analysis ``character_ref`` must equal the package character; and
    * every referenced fact / event / relationship / conflict ref (both the
      top-level ``key_*_refs`` and the ``supporting_*_refs`` inside every
      evidence-backed interpretation) must belong to the evidence universe
      actually supplied to that character.

    An empty supporting-ref set is NOT rejected here: the frozen A6A contract
    allows it (explicit vs inferred is carried by ``evidence_mode``), and
    forcing non-empty support would manufacture unsupported claims.
    """
    if analysis.character_ref != package.character_ref:
        return (
            f"wrong character_ref {analysis.character_ref!r} "
            f"(expected {package.character_ref!r})"
        )
    universe = _character_evidence_universe(package)

    def _check(refs: Sequence[str], ns: str, label: str) -> str | None:
        allowed = universe[ns]
        for ref in refs:
            if ref not in allowed:
                return f"{label}: {ns} ref {ref!r} is not in the supplied evidence universe"
        return None

    for ref in analysis.key_fact_refs:
        if ref not in universe["fact"]:
            return f"key_fact_refs: fact ref {ref!r} is not in the supplied evidence universe"
    for ref in analysis.key_event_refs:
        if ref not in universe["evt"]:
            return f"key_event_refs: event ref {ref!r} is not in the supplied evidence universe"
    for ref in analysis.important_relationship_refs:
        if ref not in universe["rel"]:
            return (
                f"important_relationship_refs: relationship ref {ref!r} "
                f"is not in the supplied evidence universe"
            )
    for label, interp in _all_interpretations(analysis):
        for refs, ns in (
            (interp.supporting_fact_refs, "fact"),
            (interp.supporting_event_refs, "evt"),
            (interp.supporting_relationship_refs, "rel"),
            (interp.supporting_conflict_refs, "conf"),
        ):
            detail = _check(refs, ns, label)
            if detail is not None:
                return detail
    return None


def validate_character_coverage(
    analyses: Sequence[CharacterAnalysis], expected_character_refs: Sequence[str]
) -> None:
    """Validate the complete-character-coverage invariant (frozen A6C section 9).

    Required final invariant: every canonical character -> exactly one
    :class:`~short_drama.story.CharacterAnalysis`. Rejects missing characters,
    duplicate characters, and extra/unknown characters. A structurally invalid
    (non-``char_*``) ``character_ref`` is already rejected by the frozen A6A
    domain contract during typed load, so "unknown character_ref" here means a
    well-formed char ref that is not part of the planned universe.
    """
    expected = frozenset(expected_character_refs)
    counts: dict[str, int] = {}
    for analysis in analyses:
        counts[analysis.character_ref] = counts.get(analysis.character_ref, 0) + 1
    duplicates = sorted(ref for ref, count in counts.items() if count > 1)
    if duplicates:
        raise StoryAnalysisSemanticError(
            f"duplicate character analysis: {duplicates}"
        )
    missing = sorted(expected - set(counts))
    if missing:
        raise StoryAnalysisSemanticError(f"missing character analysis: {missing}")
    extra = sorted(set(counts) - expected)
    if extra:
        raise StoryAnalysisSemanticError(
            f"extra/unknown character analysis: {extra}"
        )


# ---------------------------------------------------------------------------
# Provenance verification (backend-neutral, fail closed, no retry).
# ---------------------------------------------------------------------------


def _verify_character_provenance(
    provenance: LLMInvocationProvenance,
    request: StructuredGenerationRequest,
) -> None:
    """Verify the backend-neutral semantic/request identity of a provider
    result against the exact character request.

    Mirrors the A5C provenance verification: only the semantic profile,
    prompt, rendered prompt, output schema, and request-hash identities are
    checked. Provider family / model / response id / usage are audit
    provenance only and are intentionally NOT part of the identity (see the
    frozen A-I3 boundary). A mismatch fails closed with
    :class:`StoryAnalysisProvenanceError` (no semantic retry).
    """
    checks = (
        (provenance.semantic_profile_id, request.semantic_profile.profile_id),
        (
            provenance.semantic_profile_hash,
            request.semantic_profile.semantic_profile_hash,
        ),
        (provenance.prompt_id, request.rendered_prompt.prompt_id),
        (provenance.prompt_version, request.rendered_prompt.prompt_version),
        (provenance.prompt_content_hash, request.rendered_prompt.prompt_content_hash),
        (provenance.rendered_prompt_hash, request.rendered_prompt.rendered_prompt_hash),
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
                f"character analysis provenance field mismatch: got {got!r}, "
                f"expected {want!r}"
            )


# ---------------------------------------------------------------------------
# Bounded semantic execution (serial, canonical order).
# ---------------------------------------------------------------------------


def _attempt_character_round(
    package: CharacterEvidencePackage,
    request: StructuredGenerationRequest,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    *,
    round_number: int,
) -> CharacterAnalysis | _CharacterRetryableInvalid:
    """One semantic round for one character request.

    A technical :class:`short_drama.llm.LLMError` (transport, malformed JSON,
    or schema-invalid output) propagates unchanged (A-I3-owned); only a
    schema-valid result rejected by typed load or exact evidence-ref
    validation is returned as a bounded semantic invalidity.
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
    _verify_character_provenance(result.provenance, request)
    try:
        analysis = CharacterAnalysis.from_dict(result.parsed_json)
    except StoryAnalysisModelError as exc:
        return _CharacterRetryableInvalid(detail=f"typed load failed: {exc}")
    detail = validate_character_evidence(analysis, package)
    if detail is not None:
        return _CharacterRetryableInvalid(detail=detail)
    return analysis


def _execute_character_request(
    package: CharacterEvidencePackage,
    request: StructuredGenerationRequest,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
) -> tuple[CharacterAnalysis, int]:
    """Execute the bounded semantic rounds for one character request (serial).

    Returns the accepted :class:`CharacterAnalysis` and the number of semantic
    rounds consumed. Raises
    :class:`StoryAnalysisSemanticGenerationError` once the two frozen semantic
    rounds are both exhausted by schema-valid-but-semantically-invalid results.
    """
    outcome = _attempt_character_round(
        package, request, semantic_profile, llm_client, round_number=1
    )
    rounds = 1
    if isinstance(outcome, _CharacterRetryableInvalid):
        first_detail = outcome.detail
        outcome = _attempt_character_round(
            package, request, semantic_profile, llm_client, round_number=2
        )
        rounds = 2
        if isinstance(outcome, _CharacterRetryableInvalid):
            raise StoryAnalysisSemanticGenerationError(
                character_ref=package.character_ref,
                request_hash=request.request_hash,
                rounds_attempted=A6C_MAX_GENERATION_ROUNDS,
                last_failure_details=(first_detail, outcome.detail),
            )
    return outcome, rounds


def resolve_character_analysis(
    plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    *,
    prompts: PromptRegistry,
) -> CharacterSemanticResult:
    """Run the complete A6C character analysis semantic pass.

    One provider request per canonical planned character (serial, canonical
    order), each with up to the frozen ``max_generation_rounds`` (2) semantic
    rounds. Returns the complete in-memory
    :class:`~short_drama.story.CharacterAnalysisSet` (coverage validated) plus
    the ordered stable request identities. In-memory only: no A6 artifact is
    persisted and no A5/A4 CURRENT is mutated.
    """
    preparation = build_character_semantic_preparation(
        plan, profile, semantic_profile, prompts=prompts
    )
    # Pair packages with their prepared requests (both in canonical planned
    # order, guaranteed aligned by build_character_semantic_preparation).
    packages = plan.character_packages
    analyses: list[CharacterAnalysis] = []
    character_rounds: list[tuple[str, int]] = []
    for package, request in zip(packages, preparation.character_requests):
        analysis, rounds = _execute_character_request(
            package, request, preparation.semantic_profile, llm_client
        )
        analyses.append(analysis)
        character_rounds.append((package.character_ref, rounds))

    validate_character_coverage(analyses, preparation.character_refs)
    character_analysis_set = CharacterAnalysisSet(
        STORY_ANALYSIS_SCHEMA_VERSION, tuple(analyses)
    )
    return CharacterSemanticResult(
        preparation=preparation,
        analyses=tuple(analyses),
        character_analysis_set=character_analysis_set,
        character_rounds=tuple(character_rounds),
    )
