"""v1.2 A6D -- plot-window analysis semantic pass (provider execution).

This slice implements the **plot-window analysis** semantic pass of A6 (Issue
#89) as specified by the frozen A6 implementation plan (slice ``A6D``) and the
frozen A6 architecture doc (``06-A6-global-story-bible.md``). It is the second
semantic pass of the hierarchical A6 synthesis: A6A produced the domain
contracts, A6B produced the exact deterministic planning (plot windows over the
complete canonical event stream, joined by exact refs into evidence packets
with ``owned_event_ids`` / ``context_event_ids``), and A6D turns those **exact**
window packets into a complete in-memory ordered set of typed
:class:`~short_drama.story.PlotWindowAnalysis`.

A6D owns:

* **deterministic window request rendering** -- one provider request per planned
  window, whose ``window_context_json`` is the canonical JSON of that window's
  exact A6B evidence packet; the rendered request content and a *stable request
  identity* that binds the exact upstream A5 identity, the A6 plan identity, the
  ``StoryAnalysisProfile`` id/hash, the ``SemanticLLMProfile`` id/hash, the
  prompt and output-schema asset identities, the window id/ordinal, the exact
  owned/context event membership, the deterministic packet content hash, and the
  actual rendered request hash;
* **local evidence-ref validation** -- every supporting fact / event /
  relationship / conflict ref in the main interpretation and in every candidate
  turning point / reveal / arc-continuation marker must belong to the current
  window's own evidence universe (canonical events = the union of the window's
  owned and context events; facts / relationships / conflicts = the packet's
  relevant collections). A ref that merely exists somewhere in the global A5
  snapshot is **not** sufficient. The provider never allocates, reorders, or
  changes ownership: ``window_ordinal`` / ``owned_event_refs`` /
  ``context_event_refs`` are Python-owned and copied from the authoritative A6B
  plan, not taken from the provider;
* **bounded semantic regeneration** -- up to the frozen ``max_generation_rounds``
  (2) semantic rounds per window, where only *semantic* invalidity (a
  schema-valid provider result rejected by typed load, a wrong returned
  ``window_id``, or exact local-evidence-ref validation) triggers a
  regeneration round. A schema-invalid output and every other technical LLM
  failure remain owned by A-I3 (``short_drama.llm``) and are propagated, never
  translated into a semantic retry. Provenance / request-hash mismatches fail
  closed without retry.

A6D is **in-memory only**: it consumes the exact A6B
:class:`~short_drama.story.StoryAnalysisPlan` (plus the exact A5 snapshot from
which the window packets are joined) and produces a complete ordered
:class:`~short_drama.story.PlotWindowAnalysis` set plus the ordered stable
request identities that A6E/A6F will fold into the final
:class:`~short_drama.story.A6SemanticIdentity`. A6D does NOT persist any A6
artifact, does NOT mutate any A5/A4 CURRENT, does NOT synthesize a final
:class:`~short_drama.story.GlobalEventAnalysis` importance overlay (whole-story
importance belongs to A6E), and does NOT modify A5 canonical events or narrative
order.

Execution is **serial** (canonical window order). Optional bounded concurrency
is intentionally deferred: the frozen windows are independent, but wiring the
A5 provider-neutral executor here would expand the A6D surface without changing
the frozen semantics, so A6D keeps the minimal correct serial path. Concurrency
would remain runtime-only and must never enter the semantic identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

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
    StoryAnalysisWindowSemanticGenerationError,
    StoryIntegrityError,
)
from .story_analysis import (
    EvidenceBackedInterpretation,
    PlotWindowAnalysis,
    STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1,
    StoryAnalysisProfile,
)
from .story_analysis_planning import (
    StoryAnalysisInputSnapshot,
    StoryAnalysisPlan,
    WindowPacket,
    build_window_packets,
    validate_window_ownership,
)


# ---------------------------------------------------------------------------
# Frozen A6D identity (prompt / output schema / semantic profile).
# ---------------------------------------------------------------------------

#: The exact A6D plot-window semantic pass prompt (frozen, reviewed).
A6D_WINDOW_PROMPT_ID = "a6.plot-window-analysis"
A6D_WINDOW_PROMPT_VERSION = 1
#: The exact A6D plot-window analysis output schema (frozen, reviewed).
A6D_WINDOW_OUTPUT_SCHEMA_ID = "a6-plot-window-analysis-output"
A6D_WINDOW_OUTPUT_SCHEMA_VERSION = 1
A6D_WINDOW_OUTPUT_SCHEMA_PATH = (
    SCHEMAS_DIR / "a6-plot-window-analysis-output.schema.json"
)
#: The exact A6D semantic profile (frozen, reviewed; shared with A6C).
A6D_SEMANTIC_PROFILE_ID = "story-analysis-llm-v1"
A6D_SEMANTIC_PROFILE_PATH = PROFILES_DIR / "story_analysis_llm_v1.yaml"
#: The frozen A6D semantic regeneration budget (initial + at most 1
#: regeneration), re-exposed from the A6A contract so A6D and A6A can never
#: drift.
A6D_MAX_GENERATION_ROUNDS = STORY_ANALYSIS_MAX_GENERATION_ROUNDS_V1

#: The single prompt variable rendered for every window request.
_WINDOW_PROMPT_VARIABLE = "window_context_json"


# ---------------------------------------------------------------------------
# Result objects (in-memory, no persistence).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlotWindowSemanticPreparation:
    """Deterministic, zero-provider A6D window request preparation.

    ``windows`` and every per-window tuple are in the exact A6B canonical
    planned order and aligned with ``plan.windows``. ``window_ids`` and
    ``window_ordinals`` are Python-owned (copied from the plan).
    ``window_request_identity_hashes`` are the backend-neutral *stable request
    identities* (see :func:`window_request_identity_hash`) that A6E/A6F fold
    into the final A6 semantic identity; ``window_request_hashes`` are the raw
    :class:`~short_drama.llm.StructuredGenerationRequest.request_hash` values
    (the actual rendered-request content hashes).
    """

    plan: StoryAnalysisPlan
    profile: StoryAnalysisProfile
    semantic_profile: SemanticLLMProfile
    prompt_identity: PromptAssetIdentity
    output_schema_identity: OutputSchemaAssetIdentity
    windows: tuple[WindowPacket, ...]
    window_ids: tuple[str, ...]
    window_ordinals: tuple[int, ...]
    window_requests: tuple[StructuredGenerationRequest, ...]
    window_request_identity_hashes: tuple[str, ...]
    window_request_hashes: tuple[str, ...]
    packet_token_estimates: tuple[int, ...]
    rendered_request_token_estimates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PlotWindowSemanticResult:
    """Complete in-memory A6D plot-window semantic result (no persistence).

    ``analyses`` is the complete ordered set of typed
    :class:`~short_drama.story.PlotWindowAnalysis` (canonical planned window
    order); ``window_rounds`` is ``(window_id, semantic_rounds_consumed)`` in
    the same order. No final :class:`~short_drama.story.GlobalEventAnalysis`
    importance overlay is synthesized here (that belongs to A6E).
    """

    preparation: PlotWindowSemanticPreparation
    analyses: tuple[PlotWindowAnalysis, ...]
    #: ``(window_id, semantic_rounds_consumed)`` in canonical planned order.
    window_rounds: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _WindowRetryableInvalid:
    """A schema-valid provider result rejected by semantic validation.

    Carries a bounded failure detail (never the raw provider body) for the
    bounded-regeneration diagnostic. This is a *semantic* invalidity, not a
    technical LLM failure.
    """

    detail: str


# ---------------------------------------------------------------------------
# Frozen asset loading (A5C/A6C pattern: exact path + exact profile id).
# ---------------------------------------------------------------------------


def load_window_semantic_profile() -> SemanticLLMProfile:
    """Load the A6D plot-window-analysis semantic profile.

    The exact identity (``story-analysis-llm-v1``) is verified during execution
    by :func:`_verify_window_profile`, matching the A5C/A6C reviewed pattern.
    """
    return load_semantic_profile(A6D_SEMANTIC_PROFILE_PATH)


def load_window_output_schema() -> OutputSchema:
    """Load the frozen A6D plot-window analysis output schema."""
    try:
        schema_data = load_json(A6D_WINDOW_OUTPUT_SCHEMA_PATH)
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(
            f"failed to load plot-window analysis output schema: {exc}"
        ) from exc
    if not isinstance(schema_data, dict):
        raise StoryIntegrityError(
            "plot-window analysis output schema must be a JSON object"
        )
    return OutputSchema.create(
        schema_id=A6D_WINDOW_OUTPUT_SCHEMA_ID,
        schema_version=A6D_WINDOW_OUTPUT_SCHEMA_VERSION,
        schema=schema_data,
    )


def load_window_semantic_assets() -> tuple[SemanticLLMProfile, OutputSchema]:
    """Load the exact (semantic profile, output schema) A6D asset pair."""
    return load_window_semantic_profile(), load_window_output_schema()


# ---------------------------------------------------------------------------
# Profile identity verification (fail closed).
# ---------------------------------------------------------------------------


def _verify_window_profile(
    profile: StoryAnalysisProfile, semantic_profile: SemanticLLMProfile
) -> None:
    """Verify the profile pins the exact frozen A6D plot-window pass identity."""
    if semantic_profile.profile_id != A6D_SEMANTIC_PROFILE_ID:
        raise StoryAnalysisSemanticError(
            f"semantic profile must be {A6D_SEMANTIC_PROFILE_ID!r}, "
            f"got {semantic_profile.profile_id!r}"
        )
    if profile.max_generation_rounds != A6D_MAX_GENERATION_ROUNDS:
        raise StoryAnalysisSemanticError(
            f"max_generation_rounds must be {A6D_MAX_GENERATION_ROUNDS}, "
            f"got {profile.max_generation_rounds}"
        )
    pw = profile.plot_window_analysis
    if pw.semantic_profile_id != A6D_SEMANTIC_PROFILE_ID:
        raise StoryAnalysisSemanticError(
            f"plot_window_analysis.semantic_profile_id must be "
            f"{A6D_SEMANTIC_PROFILE_ID!r}, got {pw.semantic_profile_id!r}"
        )
    if (
        pw.prompt_id != A6D_WINDOW_PROMPT_ID
        or pw.prompt_version != A6D_WINDOW_PROMPT_VERSION
    ):
        raise StoryAnalysisSemanticError(
            f"plot_window_analysis prompt must be "
            f"{A6D_WINDOW_PROMPT_ID!r} v{A6D_WINDOW_PROMPT_VERSION}, got "
            f"{pw.prompt_id!r} v{pw.prompt_version}"
        )
    if (
        pw.output_schema_id != A6D_WINDOW_OUTPUT_SCHEMA_ID
        or pw.output_schema_version != A6D_WINDOW_OUTPUT_SCHEMA_VERSION
    ):
        raise StoryAnalysisSemanticError(
            f"plot_window_analysis output schema must be "
            f"{A6D_WINDOW_OUTPUT_SCHEMA_ID!r} v{A6D_WINDOW_OUTPUT_SCHEMA_VERSION}, "
            f"got {pw.output_schema_id!r} v{pw.output_schema_version}"
        )


def _verify_snapshot_plan_manifest_identity(
    snapshot: StoryAnalysisInputSnapshot, plan: StoryAnalysisPlan
) -> None:
    """Fail closed if the A5 snapshot and the A6B plan do not refer to the same
    exact A5 ConsolidationManifest.

    A6D joins evidence from the snapshot's canonical event stream while binding
    the plan's consolidation manifest ref into the stable request identity. If
    the two inputs name different exact A5 upstreams, the semantic requests
    would claim one A5 identity while consuming evidence from another, so A6D
    must fail closed before any packet construction or provider execution.
    """
    if snapshot.consolidation_manifest_ref != plan.consolidation_manifest_ref:
        raise StoryAnalysisSemanticError(
            "story-analysis input snapshot and plan disagree on the exact A5 "
            "consolidation manifest identity (snapshot "
            f"{snapshot.consolidation_manifest_ref!r} != plan "
            f"{plan.consolidation_manifest_ref!r}); A6D plot-window semantic "
            "analysis must consume one consistent exact A5 upstream and "
            "refuses to join evidence from a mismatched pair"
        )


# ---------------------------------------------------------------------------
# Stable request identity (backend-neutral).
# ---------------------------------------------------------------------------


def window_request_identity_hash(
    *,
    consolidation_manifest_ref: Any,
    plan_hash: str,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    prompt_identity: PromptAssetIdentity,
    output_schema_identity: OutputSchemaAssetIdentity,
    window_id: str,
    window_ordinal: int,
    owned_event_ids: Sequence[str],
    context_event_ids: Sequence[str],
    packet_hash: str,
    request_hash: str,
) -> str:
    """Compute the A6D *stable request identity* for one window request.

    This binds (via the existing :func:`short_drama.artifacts.canonical.
    content_hash` authority over the existing asset-identity / request-hash
    facilities) every element required by the frozen A6D spec: the exact
    upstream A5 identity, the A6 plan identity, the ``StoryAnalysisProfile``
    id/hash, the ``SemanticLLMProfile`` id/hash, the prompt and output-schema
    asset identities, the window identity/ordinal, the exact owned/context event
    membership, the deterministic packet content hash, and the actual rendered
    request hash. It is intentionally backend-neutral: it never includes
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
            "window_id": window_id,
            "window_ordinal": window_ordinal,
            "owned_event_ids": list(owned_event_ids),
            "context_event_ids": list(context_event_ids),
            "window_packet_hash": packet_hash,
            "rendered_request_hash": request_hash,
        }
    )


# ---------------------------------------------------------------------------
# Deterministic, zero-provider preparation.
# ---------------------------------------------------------------------------


def _complete_rendered_request_tokens(rendered: RenderedPrompt) -> int:
    """utf8-bytes-div3-v1 estimate of the complete rendered provider request
    (system + user framing), which is strictly larger than the window packet
    alone and is measured separately from the frozen packet budget."""
    return estimate_tokens(rendered.system_text) + estimate_tokens(rendered.user_text)


def build_plot_window_semantic_preparation(
    snapshot: StoryAnalysisInputSnapshot,
    plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
) -> PlotWindowSemanticPreparation:
    """Deterministically render one request per canonical planned window.

    Zero provider invocations. Validates the frozen A6D identity, re-validates
    the complete window plan (exact-one owned-event ownership over the complete
    canonical event universe) before any provider call, and enforces the frozen
    window packet budget (fail closed -- never truncate). Also measures the
    complete rendered request (system + user framing) separately from the
    packet budget.
    """
    _verify_window_profile(profile, semantic_profile)
    _verify_snapshot_plan_manifest_identity(snapshot, plan)

    # Validate the complete window plan BEFORE any provider call (fail closed):
    # exact-one owned-event ownership over the complete canonical event
    # universe, unique/strictly-ordered window ids, no unknown refs.
    all_event_ids = frozenset(event.event_id for event in snapshot.events)
    validate_window_ownership(plan.windows, all_event_ids)

    prompt_spec = prompts.load(
        A6D_WINDOW_PROMPT_ID, version=A6D_WINDOW_PROMPT_VERSION
    )
    output_schema = load_window_output_schema()
    prompt_identity = PromptAssetIdentity(
        prompt_spec.prompt_id, prompt_spec.version, prompt_spec.content_hash
    )
    output_schema_identity = OutputSchemaAssetIdentity(
        output_schema.schema_id,
        output_schema.schema_version,
        output_schema.schema_hash,
    )

    budget = profile.planning_policy.plot_window_packet_max_estimated_tokens
    packets = build_window_packets(plan.windows, snapshot)

    window_ids: list[str] = []
    window_ordinals: list[int] = []
    window_requests: list[StructuredGenerationRequest] = []
    window_request_identity_hashes: list[str] = []
    window_request_hashes: list[str] = []
    packet_token_estimates: list[int] = []
    rendered_request_token_estimates: list[int] = []

    for packet in packets:
        packet_tokens = packet.estimated_tokens()
        if packet_tokens > budget:
            raise StoryAnalysisSemanticError(
                f"plot-window packet {packet.window_id!r} "
                f"({packet_tokens} estimated tokens) exceeds the frozen "
                f"plot_window_packet_max_estimated_tokens budget ({budget}); "
                f"refusing to truncate"
            )
        window_context_json = packet.canonical_bytes().decode("utf-8")
        rendered = render_prompt(
            prompt_spec, {_WINDOW_PROMPT_VARIABLE: window_context_json}
        )
        request = build_structured_request(
            rendered_prompt=rendered,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        identity_hash = window_request_identity_hash(
            consolidation_manifest_ref=plan.consolidation_manifest_ref,
            plan_hash=plan.plan_hash,
            profile=profile,
            semantic_profile=semantic_profile,
            prompt_identity=prompt_identity,
            output_schema_identity=output_schema_identity,
            window_id=packet.window_id,
            window_ordinal=packet.window_ordinal,
            owned_event_ids=packet.owned_event_ids,
            context_event_ids=packet.context_event_ids,
            packet_hash=content_hash(packet.to_dict()),
            request_hash=request.request_hash,
        )
        window_ids.append(packet.window_id)
        window_ordinals.append(packet.window_ordinal)
        window_requests.append(request)
        window_request_identity_hashes.append(identity_hash)
        window_request_hashes.append(request.request_hash)
        packet_token_estimates.append(packet_tokens)
        rendered_request_token_estimates.append(
            _complete_rendered_request_tokens(rendered)
        )

    return PlotWindowSemanticPreparation(
        plan=plan,
        profile=profile,
        semantic_profile=semantic_profile,
        prompt_identity=prompt_identity,
        output_schema_identity=output_schema_identity,
        windows=packets,
        window_ids=tuple(window_ids),
        window_ordinals=tuple(window_ordinals),
        window_requests=tuple(window_requests),
        window_request_identity_hashes=tuple(window_request_identity_hashes),
        window_request_hashes=tuple(window_request_hashes),
        packet_token_estimates=tuple(packet_token_estimates),
        rendered_request_token_estimates=tuple(rendered_request_token_estimates),
    )


# ---------------------------------------------------------------------------
# Semantic validation (typed load + exact local-evidence-ref membership).
# ---------------------------------------------------------------------------


def _window_evidence_universe(packet: WindowPacket) -> dict[str, frozenset[str]]:
    """The exact per-window supporting-reference universe.

    A referenced ref is only valid if it belongs to the current window's own
    evidence universe (canonical events = the union of the window's owned and
    context events; facts / relationships / conflicts = the packet's relevant
    collections). Character / location descriptors are supplied as context but
    are not referenceable through the A6A contract, so they are not part of the
    ref universe.
    """
    event_ids = frozenset(packet.owned_event_ids) | frozenset(
        packet.context_event_ids
    )
    return {
        "fact": frozenset(fact.fact_id for fact in packet.relevant_facts),
        "evt": event_ids,
        "rel": frozenset(
            rel.relationship_id for rel in packet.relevant_relationships
        ),
        "conf": frozenset(
            conflict.conflict_id for conflict in packet.relevant_conflicts
        ),
    }


def _all_window_interpretations(
    analysis: PlotWindowAnalysis,
) -> list[tuple[str, EvidenceBackedInterpretation]]:
    """Every evidence-backed interpretation field of a window analysis, with a
    stable label, in schema order (main interpretation, then the three local
    candidate-marker arrays)."""
    out: list[tuple[str, EvidenceBackedInterpretation]] = [
        ("interpretation", analysis.interpretation)
    ]
    for index, interp in enumerate(analysis.candidate_turning_points):
        out.append((f"candidate_turning_points[{index}]", interp))
    for index, interp in enumerate(analysis.candidate_reveals):
        out.append((f"candidate_reveals[{index}]", interp))
    for index, interp in enumerate(analysis.arc_continuation_markers):
        out.append((f"arc_continuation_markers[{index}]", interp))
    return out


def validate_plot_window_evidence(
    analysis: PlotWindowAnalysis, packet: WindowPacket
) -> str | None:
    """Validate one window analysis against its exact local evidence universe.

    Returns ``None`` when the analysis is semantically valid, or a bounded
    failure-detail string otherwise. This layers on top of the frozen A6A
    domain contract (which already enforces the exact keys, ref namespaces,
    ``explicit``/``inferred`` evidence modes, and duplicate-free refs). The
    A6D-specific invariant is:

    * every supporting fact / event / relationship / conflict ref in the main
      interpretation and in every candidate turning point / reveal /
      arc-continuation marker must belong to the current window's own evidence
      universe (canonical events = owned + context; facts / relationships /
      conflicts = the packet's relevant collections). A ref that exists
      elsewhere in the global A5 snapshot is NOT valid.

    An empty supporting-ref set is NOT rejected here: the frozen A6A contract
    allows it (explicit vs inferred is carried by ``evidence_mode``), and
    forcing non-empty support would manufacture unsupported claims. The
    Python-owned ``window_ordinal`` / ``owned_event_refs`` /
    ``context_event_refs`` are not validated against the provider (they are
    copied from the authoritative plan), and context supporting refs do not
    alter ownership.
    """
    universe = _window_evidence_universe(packet)

    def _check(refs: Sequence[str], ns: str, label: str) -> str | None:
        allowed = universe[ns]
        for ref in refs:
            if ref not in allowed:
                return (
                    f"{label}: {ns} ref {ref!r} is not in the supplied window "
                    f"evidence universe"
                )
        return None

    for label, interp in _all_window_interpretations(analysis):
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


def validate_plot_window_coverage(
    analyses: Sequence[PlotWindowAnalysis],
    windows: Sequence[Any],
) -> None:
    """Validate the exact-window-coverage invariant (frozen A6D section 9).

    Required final invariant: every planned window -> exactly one
    :class:`~short_drama.story.PlotWindowAnalysis`, in canonical planned order,
    with the correct window id, ordinal, owned-event membership, and
    context-event membership. Rejects missing windows, duplicate windows, and
    extra/unknown windows, and any mismatch between the completed result set and
    the authoritative A6B plan. The provider-created canonical refs are already
    rejected per-window by :func:`validate_plot_window_evidence`; the exact
    owned-event coverage of the complete canonical event universe is guaranteed
    by the plan (validated in
    :func:`build_plot_window_semantic_preparation`), so completing every planned
    window establishes complete canonical-event ownership.
    """
    if len(analyses) != len(windows):
        raise StoryAnalysisSemanticError(
            f"window result count {len(analyses)} does not equal planned window "
            f"count {len(windows)}"
        )
    for analysis, window in zip(analyses, windows):
        if analysis.window_id != window.window_id:
            raise StoryAnalysisSemanticError(
                f"window id mismatch: {analysis.window_id!r} != "
                f"{window.window_id!r}"
            )
        if analysis.window_ordinal != window.window_ordinal:
            raise StoryAnalysisSemanticError(
                f"window ordinal mismatch for {analysis.window_id!r}: "
                f"{analysis.window_ordinal} != {window.window_ordinal}"
            )
        if tuple(analysis.owned_event_refs) != tuple(window.owned_event_ids):
            raise StoryAnalysisSemanticError(
                f"owned event membership mismatch for {analysis.window_id!r}"
            )
        if tuple(analysis.context_event_refs) != tuple(
            window.context_event_ids
        ):
            raise StoryAnalysisSemanticError(
                f"context event membership mismatch for {analysis.window_id!r}"
            )


# ---------------------------------------------------------------------------
# Provenance verification (backend-neutral, fail closed, no retry).
# ---------------------------------------------------------------------------


def _verify_window_provenance(
    provenance: LLMInvocationProvenance,
    request: StructuredGenerationRequest,
) -> None:
    """Verify the backend-neutral semantic/request identity of a provider
    result against the exact window request.

    Mirrors the A5C/A6C provenance verification: only the semantic profile,
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
                f"plot-window analysis provenance field mismatch: got {got!r}, "
                f"expected {want!r}"
            )


# ---------------------------------------------------------------------------
# Bounded semantic execution (serial, canonical order).
# ---------------------------------------------------------------------------


def _attempt_window_round(
    packet: WindowPacket,
    request: StructuredGenerationRequest,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    *,
    round_number: int,
) -> PlotWindowAnalysis | _WindowRetryableInvalid:
    """One semantic round for one window request.

    A technical :class:`short_drama.llm.LLMError` (transport, malformed JSON,
    or schema-invalid output) propagates unchanged (A-I3-owned); a provenance /
    request-hash mismatch fails closed (no retry); only a schema-valid result
    rejected by typed load, a wrong returned ``window_id``, or exact
    local-evidence-ref validation is returned as a bounded semantic invalidity.
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
    _verify_window_provenance(result.provenance, request)

    payload = result.parsed_json
    if not isinstance(payload, dict):
        return _WindowRetryableInvalid(
            detail="provider output must be a JSON object"
        )
    got_window_id = payload.get("window_id")
    if got_window_id != packet.window_id:
        return _WindowRetryableInvalid(
            detail=(
                f"wrong window_id {got_window_id!r} "
                f"(expected {packet.window_id!r})"
            )
        )
    try:
        interpretation = EvidenceBackedInterpretation.from_dict(
            payload["interpretation"]
        )
        candidate_turning_points = tuple(
            EvidenceBackedInterpretation.from_dict(x)
            for x in payload["candidate_turning_points"]
        )
        candidate_reveals = tuple(
            EvidenceBackedInterpretation.from_dict(x)
            for x in payload["candidate_reveals"]
        )
        arc_continuation_markers = tuple(
            EvidenceBackedInterpretation.from_dict(x)
            for x in payload["arc_continuation_markers"]
        )
        # Python-owned fields (ordinal + owned/context membership) are copied
        # from the authoritative A6B plan; the provider never allocates,
        # reorders, or changes ownership.
        analysis = PlotWindowAnalysis(
            window_id=packet.window_id,
            window_ordinal=packet.window_ordinal,
            owned_event_refs=packet.owned_event_ids,
            context_event_refs=packet.context_event_ids,
            interpretation=interpretation,
            candidate_turning_points=candidate_turning_points,
            candidate_reveals=candidate_reveals,
            arc_continuation_markers=arc_continuation_markers,
        )
    except (StoryAnalysisModelError, KeyError, TypeError) as exc:
        return _WindowRetryableInvalid(detail=f"typed load failed: {exc}")
    detail = validate_plot_window_evidence(analysis, packet)
    if detail is not None:
        return _WindowRetryableInvalid(detail=detail)
    return analysis


def _execute_window_request(
    packet: WindowPacket,
    request: StructuredGenerationRequest,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
) -> tuple[PlotWindowAnalysis, int]:
    """Execute the bounded semantic rounds for one window request (serial).

    Returns the accepted :class:`PlotWindowAnalysis` and the number of semantic
    rounds consumed. Raises
    :class:`StoryAnalysisWindowSemanticGenerationError` once the two frozen
    semantic rounds are both exhausted by schema-valid-but-semantically-invalid
    results.
    """
    outcome = _attempt_window_round(
        packet, request, semantic_profile, llm_client, round_number=1
    )
    rounds = 1
    if isinstance(outcome, _WindowRetryableInvalid):
        first_detail = outcome.detail
        outcome = _attempt_window_round(
            packet, request, semantic_profile, llm_client, round_number=2
        )
        rounds = 2
        if isinstance(outcome, _WindowRetryableInvalid):
            raise StoryAnalysisWindowSemanticGenerationError(
                window_id=packet.window_id,
                request_hash=request.request_hash,
                rounds_attempted=A6D_MAX_GENERATION_ROUNDS,
                last_failure_details=(first_detail, outcome.detail),
            )
    return outcome, rounds


def resolve_plot_window_analysis(
    snapshot: StoryAnalysisInputSnapshot,
    plan: StoryAnalysisPlan,
    profile: StoryAnalysisProfile,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    *,
    prompts: PromptRegistry,
) -> PlotWindowSemanticResult:
    """Run the complete A6D plot-window analysis semantic pass.

    One provider request per canonical planned window (serial, canonical
    order), each with up to the frozen ``max_generation_rounds`` (2) semantic
    rounds. Returns the complete in-memory ordered
    :class:`~short_drama.story.PlotWindowAnalysis` set (coverage validated) plus
    the ordered stable request identities. A failed window prevents returning a
    partial authoritative result set: any semantic exhaustion, provenance
    mismatch, or technical LLM failure propagates before the result is built.
    In-memory only: no A6 artifact is persisted and no A5/A4 CURRENT is
    mutated; no final ``GlobalEventAnalysis`` importance overlay is synthesized
    (that belongs to A6E).
    """
    preparation = build_plot_window_semantic_preparation(
        snapshot, plan, profile, semantic_profile, prompts=prompts
    )
    # Pair window packets with their prepared requests (both in canonical
    # planned order, guaranteed aligned by
    # build_plot_window_semantic_preparation).
    analyses: list[PlotWindowAnalysis] = []
    window_rounds: list[tuple[str, int]] = []
    for packet, request in zip(preparation.windows, preparation.window_requests):
        analysis, rounds = _execute_window_request(
            packet, request, preparation.semantic_profile, llm_client
        )
        analyses.append(analysis)
        window_rounds.append((packet.window_id, rounds))

    validate_plot_window_coverage(analyses, plan.windows)
    return PlotWindowSemanticResult(
        preparation=preparation,
        analyses=tuple(analyses),
        window_rounds=tuple(window_rounds),
    )
