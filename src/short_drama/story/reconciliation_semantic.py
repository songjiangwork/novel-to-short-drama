"""v1.2 A4C — provider-neutral semantic ambiguity resolution.

This module implements the A4C semantic ambiguity-resolution slice:

    ReconciliationPlanningResult (A4B)
        ↓
    deterministic semantic block packing (MAX_PAIRS_PER_BLOCK=6,
    MAX_CANDIDATES_PER_BLOCK=12)
        ↓
    candidate packet rendering (endpoint-only, source-order)
        ↓
    requested pair rendering (needs_semantic_decision only)
        ↓
    PromptRegistry rendering (a4.entity-reconciliation v1)
        ↓
    OutputSchema build (reconciliation-decision-payload.schema.json)
        ↓
    LLMClient.generate_structured(...) — up to 2 semantic rounds per block
        ↓
    exact pair/evidence validation
        ↓
    ReconciliationDecision construction (method=llm)
        ↓
    combined deterministic + semantic decision result

A4C is in-memory: it does NOT persist, write CURRENT, or reuse. It does NOT
build the identity graph, detect conflicts, or allocate canonical IDs (A4D).

It reuses existing authorities:
  * Foundation ``content_hash`` / ``canonical_json_bytes`` for all hashing/JSON;
  * A-I3 ``LLMClient.generate_structured(...)`` for provider invocation;
  * A-I3 ``PromptRegistry`` / ``PromptSpec`` / ``RenderedPrompt`` for prompts;
  * A-I3 ``OutputSchema`` / ``LLMInvocationProvenance`` for structured output;
  * A-I3 ``SemanticLLMProfile`` for generation semantics;
  * A4A ``ReconciliationDecision`` / ``EvidenceRef`` / ``ReconciliationDecisionPayload``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from short_drama.artifacts import canonical_json_bytes, content_hash
from short_drama.io import load_json
from short_drama.llm import (
    LLMClient,
    LLMError,
    LLMInvocationProvenance,
    OutputSchema,
    PromptRegistry,
    PromptSpec,
    SemanticLLMProfile,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    build_structured_request,
    render_prompt,
)
from short_drama.paths import REPO_ROOT, SCHEMAS_DIR

from .errors import (
    ReconciliationModelError,
    ReconciliationProvenanceError,
    ReconciliationSemanticError,
    ReconciliationSemanticGenerationError,
)
from .extraction import EvidenceRef
from .reconciliation import (
    EntityReconciliationProfile,
    ReconciliationDecision,
    ReconciliationDecisionItem,
    ReconciliationDecisionPayload,
)
from .reconciliation_planning import (
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_MUST_NOT_MERGE,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    ReconciliationPairPlan,
    ReconciliationPlanningResult,
)

# ---------------------------------------------------------------------------
# Frozen block packing policy
# ---------------------------------------------------------------------------

MAX_PAIRS_PER_BLOCK = 6
MAX_CANDIDATES_PER_BLOCK = 12

# Default authorities (tracked prompt registry + tracked output schema).
DEFAULT_PROMPT_BASE_DIR = REPO_ROOT / "prompts" / "story"
DEFAULT_OUTPUT_SCHEMA_PATH = SCHEMAS_DIR / "reconciliation-decision-payload.schema.json"

# Reason code mapping: decision → reason_code
_REASON_CODE_MAP = {
    "same_entity": "llm_same_entity",
    "different_entity": "llm_different_entity",
    "uncertain": "llm_uncertain",
}


# ---------------------------------------------------------------------------
# In-memory semantic block
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationSemanticBlock:
    """One deterministic semantic block to be sent to the LLM.

    In-memory only; A4C does not persist blocks.
    """

    block_id: str
    pair_plans: tuple[ReconciliationPairPlan, ...]
    candidate_refs: tuple[str, ...]
    candidate_packets_json: str
    requested_pairs_json: str


# ---------------------------------------------------------------------------
# In-memory semantic block result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationSemanticBlockResult:
    """The result of processing one semantic block.

    In-memory only; A4C does not persist.
    """

    block_id: str
    request_hash: str
    semantic_rounds: int
    decisions: tuple[ReconciliationDecision, ...]
    generation_provenance: LLMInvocationProvenance


# ---------------------------------------------------------------------------
# Whole A4C result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationSemanticResult:
    """The complete A4C semantic resolution result.

    In-memory only; A4C does not persist.
    """

    planning_result: ReconciliationPlanningResult
    blocks: tuple[ReconciliationSemanticBlock, ...]
    semantic_decisions: tuple[ReconciliationDecision, ...]
    all_decisions: tuple[ReconciliationDecision, ...]
    semantic_request_hashes: tuple[str, ...]
    block_results: tuple[ReconciliationSemanticBlockResult, ...]


# ---------------------------------------------------------------------------
# Deterministic, zero-provider preparation (A4D reuse seam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationSemanticPreparation:
    """Deterministic, ZERO-provider A4C preparation.

    Built without any LLM call. Carries the exact blocks, the per-block
    ``StructuredGenerationRequest`` values (aligned with ``blocks``), the
    canonical-order semantic request hashes, and the backend-neutral
    prompt / output-schema / semantic-profile identity.

    This is the single deterministic request-construction path. Both
    :func:`resolve_semantic_ambiguity` (which then drives the provider per
    block) and A4D (which derives the :class:`A4SemanticIdentity` *before* any
    provider call for current-only reuse) consume it, so the prepared request
    hashes are guaranteed identical to the actual resolve-path request hashes.
    """

    blocks: tuple[ReconciliationSemanticBlock, ...]
    structured_requests: tuple[StructuredGenerationRequest, ...]
    semantic_request_hashes: tuple[str, ...]
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    output_schema_id: str
    output_schema_version: int
    output_schema_hash: str
    semantic_profile_id: str
    semantic_profile_hash: str


# ---------------------------------------------------------------------------
# Block packing
# ---------------------------------------------------------------------------


def _pack_semantic_blocks(
    planning_result: ReconciliationPlanningResult,
) -> list[list[ReconciliationPairPlan]]:
    """Deterministic greedy packing of needs_semantic_decision pairs into blocks.

    Returns a list of blocks, each a list of pair plans.
    """
    # Collect all needs_semantic_decision pairs, canonical sort
    semantic_pairs = [
        p
        for p in planning_result.pair_plans
        if p.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
    ]
    semantic_pairs.sort(key=lambda p: (p.left_candidate_ref, p.right_candidate_ref))

    if not semantic_pairs:
        return []

    blocks: list[list[ReconciliationPairPlan]] = []
    current_block: list[ReconciliationPairPlan] = []
    current_candidates: set[str] = set()

    for pair in semantic_pairs:
        pair_candidates = {pair.left_candidate_ref, pair.right_candidate_ref}

        # Check if adding this pair would violate the limits
        if current_block:
            would_exceed_pairs = len(current_block) + 1 > MAX_PAIRS_PER_BLOCK
            would_exceed_candidates = (
                len(current_candidates | pair_candidates) > MAX_CANDIDATES_PER_BLOCK
            )
            if would_exceed_pairs or would_exceed_candidates:
                blocks.append(current_block)
                current_block = []
                current_candidates = set()

        current_block.append(pair)
        current_candidates |= pair_candidates

    if current_block:
        blocks.append(current_block)

    return blocks


def _build_block_id(
    plan_hash: str,
    block_ordinal: int,
    pair_plans: list[ReconciliationPairPlan],
    candidate_refs: tuple[str, ...],
) -> str:
    """Compute the deterministic block_id from frozen hash material.

    Hash material:
      * planning_result.plan_hash
      * block ordinal
      * ordered requested pair canonical material
      * ordered candidate refs

    No runtime/backend metadata.
    """
    # Ordered requested pair canonical material
    pair_material = [
        {
            "left_candidate_ref": p.left_candidate_ref,
            "right_candidate_ref": p.right_candidate_ref,
        }
        for p in pair_plans
    ]
    # Ordered candidate refs (source_order_key order)
    candidate_refs_list = list(candidate_refs)

    material = {
        "plan_hash": plan_hash,
        "block_ordinal": block_ordinal,
        "requested_pairs": pair_material,
        "candidate_refs": candidate_refs_list,
    }
    return f"a4blk_{content_hash(material)[:20]}"


def _build_candidate_packets(
    planning_result: ReconciliationPlanningResult,
    pair_plans: list[ReconciliationPairPlan],
) -> tuple[str, tuple[str, ...]]:
    """Build candidate packets for a block.

    Returns (candidate_packets_json, candidate_refs_tuple).

    Only includes endpoint candidates of the requested pairs, ordered by
    source_order_key from CandidateEntityIndex.

    FAILS CLOSED if any endpoint ref is missing from the candidate index or
    has a non-merge-graph kind (not character/location).
    """
    # Collect unique endpoint refs
    endpoint_refs: set[str] = set()
    for p in pair_plans:
        endpoint_refs.add(p.left_candidate_ref)
        endpoint_refs.add(p.right_candidate_ref)

    # Get candidate index entries for these refs; FAIL CLOSED on missing or
    # non-merge-graph kind.
    ref_to_entry = {e.candidate_ref: e for e in planning_result.candidate_index.entries}
    endpoint_entries = []
    for ref in sorted(endpoint_refs):
        entry = ref_to_entry.get(ref)
        if entry is None:
            raise ReconciliationSemanticError(
                f"semantic pair endpoint {ref!r} not found in "
                f"CandidateEntityIndex; A4B planning input is invalid"
            )
        if entry.candidate_kind not in ("character", "location"):
            raise ReconciliationSemanticError(
                f"semantic pair endpoint {ref!r} has candidate_kind "
                f"{entry.candidate_kind!r}; expected 'character' or 'location'"
            )
        endpoint_entries.append(entry)

    # Sort by source_order_key (canonical A4B order)
    endpoint_entries.sort(key=lambda e: e.source_order_key)

    # Build packet dicts (exact fields only)
    packets = []
    candidate_refs_ordered = []
    for entry in endpoint_entries:
        candidate_refs_ordered.append(entry.candidate_ref)
        packet = {
            "candidate_ref": entry.candidate_ref,
            "candidate_kind": entry.candidate_kind,
            "display_name_original": entry.display_name_original,
            "aliases_original": list(entry.aliases_original),
            "descriptors_zh": list(entry.descriptors_zh),
            "source_order_key": entry.source_order_key,
            "evidence_refs": [e.to_dict() for e in entry.evidence_refs],
        }
        packets.append(packet)

    candidate_packets_json = canonical_json_bytes(packets).decode("utf-8")
    return candidate_packets_json, tuple(candidate_refs_ordered)


def _build_requested_pairs_json(
    pair_plans: list[ReconciliationPairPlan],
) -> str:
    """Build the requested_pairs_json for a block.

    Each pair exact fields from ReconciliationPairPlan:
      left_candidate_ref, right_candidate_ref, signals,
      shared_identity_keys, shared_tokens

    Pairs are in the given order (already canonical-sorted by pack).
    """
    pairs_data = []
    for p in pair_plans:
        pairs_data.append(
            {
                "left_candidate_ref": p.left_candidate_ref,
                "right_candidate_ref": p.right_candidate_ref,
                "signals": list(p.signals),
                "shared_identity_keys": list(p.shared_identity_keys),
                "shared_tokens": list(p.shared_tokens),
            }
        )
    return canonical_json_bytes(pairs_data).decode("utf-8")


def _build_blocks(
    planning_result: ReconciliationPlanningResult,
) -> tuple[ReconciliationSemanticBlock, ...]:
    """Build all semantic blocks for the planning result."""
    packed = _pack_semantic_blocks(planning_result)

    blocks: list[ReconciliationSemanticBlock] = []
    for ordinal, pair_plans in enumerate(packed):
        candidate_packets_json, candidate_refs = _build_candidate_packets(
            planning_result, pair_plans
        )
        requested_pairs_json = _build_requested_pairs_json(pair_plans)
        block_id = _build_block_id(
            planning_result.plan_hash,
            ordinal,
            pair_plans,
            candidate_refs,
        )
        blocks.append(
            ReconciliationSemanticBlock(
                block_id=block_id,
                pair_plans=tuple(pair_plans),
                candidate_refs=candidate_refs,
                candidate_packets_json=candidate_packets_json,
                requested_pairs_json=requested_pairs_json,
            )
        )

    return tuple(blocks)


# ---------------------------------------------------------------------------
# Profile consistency gate
# ---------------------------------------------------------------------------


def _check_profile_consistency(
    profile: EntityReconciliationProfile,
    prompt_spec: PromptSpec,
    output_schema: OutputSchema,
) -> None:
    """Fail closed unless the profile, prompt, and schema are consistent."""
    if profile.prompt_id != prompt_spec.prompt_id:
        raise ReconciliationSemanticError(
            f"EntityReconciliationProfile.prompt_id {profile.prompt_id!r} "
            f"does not match PromptSpec.prompt_id {prompt_spec.prompt_id!r}"
        )
    if profile.prompt_version != prompt_spec.version:
        raise ReconciliationSemanticError(
            f"EntityReconciliationProfile.prompt_version "
            f"{profile.prompt_version!r} does not match "
            f"PromptSpec.version {prompt_spec.version!r}"
        )
    if profile.output_schema_id != output_schema.schema_id:
        raise ReconciliationSemanticError(
            f"EntityReconciliationProfile.output_schema_id "
            f"{profile.output_schema_id!r} does not match "
            f"OutputSchema.schema_id {output_schema.schema_id!r}"
        )
    if profile.output_schema_version != output_schema.schema_version:
        raise ReconciliationSemanticError(
            f"EntityReconciliationProfile.output_schema_version "
            f"{profile.output_schema_version!r} does not match "
            f"OutputSchema.schema_version {output_schema.schema_version!r}"
        )
    if profile.max_generation_rounds != 2:
        raise ReconciliationSemanticError(
            f"EntityReconciliationProfile.max_generation_rounds must be 2, "
            f"got {profile.max_generation_rounds}"
        )


# ---------------------------------------------------------------------------
# Provenance verification
# ---------------------------------------------------------------------------


def _verify_provenance(
    provenance: LLMInvocationProvenance,
    request: Any,  # StructuredGenerationRequest
    rendered_prompt: Any,
    output_schema: OutputSchema,
    semantic_profile: SemanticLLMProfile,
) -> None:
    """Verify the successful generation's provenance matches the exact built request.

    Checks backend-neutral semantic/request identity only:
      semantic_profile_id/hash
      prompt_id/version/content_hash
      rendered_prompt_hash
      output_schema_id/version/hash
      request_hash

    Mismatch: ReconciliationProvenanceError (FAIL CLOSED, NO semantic retry).
    """
    # Expected values from the built request
    expected = {
        "semantic_profile_id": semantic_profile.profile_id,
        "semantic_profile_hash": semantic_profile.semantic_profile_hash,
        "prompt_id": rendered_prompt.prompt_id,
        "prompt_version": rendered_prompt.prompt_version,
        "prompt_content_hash": rendered_prompt.prompt_content_hash,
        "rendered_prompt_hash": rendered_prompt.rendered_prompt_hash,
        "output_schema_id": output_schema.schema_id,
        "output_schema_version": output_schema.schema_version,
        "output_schema_hash": output_schema.schema_hash,
        "request_hash": request.request_hash,
    }

    # Actual values from provenance
    actual = {
        "semantic_profile_id": provenance.semantic_profile_id,
        "semantic_profile_hash": provenance.semantic_profile_hash,
        "prompt_id": provenance.prompt_id,
        "prompt_version": provenance.prompt_version,
        "prompt_content_hash": provenance.prompt_content_hash,
        "rendered_prompt_hash": provenance.rendered_prompt_hash,
        "output_schema_id": provenance.output_schema_id,
        "output_schema_version": provenance.output_schema_version,
        "output_schema_hash": provenance.output_schema_hash,
        "request_hash": provenance.request_hash,
    }

    for key in expected:
        if expected[key] != actual[key]:
            raise ReconciliationProvenanceError(
                f"provenance field {key!r} mismatch: expected "
                f"{expected[key]!r}, got {actual[key]!r}"
            )


# ---------------------------------------------------------------------------
# Pair/evidence validation
# ---------------------------------------------------------------------------


def _validate_block_payload(
    payload: ReconciliationDecisionPayload,
    pair_plans: tuple[ReconciliationPairPlan, ...],
    candidate_packets: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Validate a block's provider payload against exact requested pairs.

    Returns (is_valid, failure_detail).

    Checks:
      * count exact
      * order exact
      * left/right refs exact
      * no duplicate/missing/extra
      * refs must be in block candidate packets
      * evidence refs must be exact endpoint evidence
      * no duplicate evidence within a decision
    """
    requested_pairs = [
        (p.left_candidate_ref, p.right_candidate_ref) for p in pair_plans
    ]
    requested_count = len(requested_pairs)
    decisions = payload.decisions

    # Count check
    if len(decisions) != requested_count:
        return False, (
            f"decision count mismatch: expected {requested_count}, "
            f"got {len(decisions)}"
        )

    # Build the set of valid evidence per candidate ref
    # For each candidate, collect its evidence_refs as tuples for comparison
    candidate_evidence: dict[str, tuple[tuple, ...]] = {}
    for packet in candidate_packets:
        ref = packet["candidate_ref"]
        ev_tuples = tuple(
            (ev["paragraph_id"], ev["role"], ev["strength"], ev["excerpt"])
            for ev in packet["evidence_refs"]
        )
        candidate_evidence[ref] = ev_tuples

    for i, item in enumerate(decisions):
        expected_left, expected_right = requested_pairs[i]

        # Order + ref exact
        if item.left_candidate_ref != expected_left:
            return False, (
                f"pair {i}: left_candidate_ref mismatch: expected "
                f"{expected_left!r}, got {item.left_candidate_ref!r}"
            )
        if item.right_candidate_ref != expected_right:
            return False, (
                f"pair {i}: right_candidate_ref mismatch: expected "
                f"{expected_right!r}, got {item.right_candidate_ref!r}"
            )

        # Refs must be in block candidate packets
        if item.left_candidate_ref not in candidate_evidence:
            return False, (
                f"pair {i}: left_candidate_ref {item.left_candidate_ref!r} "
                f"not in block candidate packets"
            )
        if item.right_candidate_ref not in candidate_evidence:
            return False, (
                f"pair {i}: right_candidate_ref {item.right_candidate_ref!r} "
                f"not in block candidate packets"
            )

        # Evidence validation
        valid_evidence = set(
            candidate_evidence[item.left_candidate_ref]
            + candidate_evidence[item.right_candidate_ref]
        )

        seen_evidence: set[tuple] = set()
        for ev in item.evidence_refs:
            ev_tuple = (ev.paragraph_id, ev.role, ev.strength, ev.excerpt)
            if ev_tuple not in valid_evidence:
                return False, (
                    f"pair {i}: evidence_ref not exact endpoint evidence: "
                    f"{ev_tuple!r}"
                )
            if ev_tuple in seen_evidence:
                return False, (
                    f"pair {i}: duplicate evidence_ref: {ev_tuple!r}"
                )
            seen_evidence.add(ev_tuple)

    return True, ""


# ---------------------------------------------------------------------------
# Decision ID computation
# ---------------------------------------------------------------------------


def _compute_llm_decision_id(
    left_ref: str,
    right_ref: str,
    decision: str,
    method: str,
    reason_code: str,
    reason_zh: str,
    evidence_refs: tuple[EvidenceRef, ...],
    prompt_id: str,
    prompt_version: int,
    request_hash: str,
) -> str:
    """Compute the deterministic LLM decision_id.

    Hash material (excludes provider_family, model, provider_response_id,
    endpoint, timestamp):
      left_candidate_ref, right_candidate_ref, decision, method, reason_code,
      reason_zh, evidence_refs, prompt_id, prompt_version, request_hash
    """
    material = {
        "left_candidate_ref": left_ref,
        "right_candidate_ref": right_ref,
        "decision": decision,
        "method": method,
        "reason_code": reason_code,
        "reason_zh": reason_zh,
        "evidence_refs": [e.to_dict() for e in evidence_refs],
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "request_hash": request_hash,
    }
    return f"dec_{content_hash(material)[:20]}"


# ---------------------------------------------------------------------------
# Convert provider item to ReconciliationDecision
# ---------------------------------------------------------------------------


def _convert_to_decision(
    item: ReconciliationDecisionItem,
    request_hash: str,
    prompt_id: str,
    prompt_version: int,
    provenance: LLMInvocationProvenance,
) -> ReconciliationDecision:
    """Convert a valid provider decision item to a ReconciliationDecision."""
    reason_code = _REASON_CODE_MAP[item.decision]
    decision_id = _compute_llm_decision_id(
        left_ref=item.left_candidate_ref,
        right_ref=item.right_candidate_ref,
        decision=item.decision,
        method="llm",
        reason_code=reason_code,
        reason_zh=item.reason_zh,
        evidence_refs=item.evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        request_hash=request_hash,
    )
    return ReconciliationDecision(
        decision_id=decision_id,
        left_candidate_ref=item.left_candidate_ref,
        right_candidate_ref=item.right_candidate_ref,
        decision=item.decision,
        method="llm",
        reason_code=reason_code,
        reason_zh=item.reason_zh,
        evidence_refs=item.evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        generation_provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Deterministic, zero-provider preparation (A4D reuse seam)
# ---------------------------------------------------------------------------


def _load_output_schema(
    profile: EntityReconciliationProfile, output_schema_path: str | Path
) -> OutputSchema:
    """Load the profile-pinned JSON Schema and build the OutputSchema."""
    try:
        schema_data = load_json(Path(output_schema_path))
    except Exception as exc:  # noqa: BLE001
        raise ReconciliationSemanticError(
            f"failed to load reconciliation-decision-payload output schema: {exc}"
        ) from exc
    if not isinstance(schema_data, dict):
        raise ReconciliationSemanticError(
            "reconciliation-decision-payload output schema must be a JSON object"
        )
    return OutputSchema.create(
        schema_id=profile.output_schema_id,
        schema_version=profile.output_schema_version,
        schema=schema_data,
    )


def prepare_semantic_resolution(
    planning_result: ReconciliationPlanningResult,
    profile: EntityReconciliationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompt_registry: PromptRegistry | None = None,
    output_schema_path: str | Path = DEFAULT_OUTPUT_SCHEMA_PATH,
) -> ReconciliationSemanticPreparation:
    """Deterministically prepare A4C semantic requests WITHOUT any provider call.

    This is the zero-provider preparation seam A4D consumes to derive the
    :class:`A4SemanticIdentity` (and decide current-only reuse) before any LLM
    call. It:

    1. Builds the semantic blocks (fail-closed endpoint validation first).
    2. Loads the profile-pinned prompt + output schema.
    3. Runs the profile consistency gate (fail closed, zero provider calls).
    4. Builds one ``StructuredGenerationRequest`` per block (no provider call).

    The returned ``structured_requests`` are aligned with ``blocks``; their
    request hashes equal the exact hashes the resolve path uses, so the
    prepared identity and the actual resolve path agree by construction.
    """
    blocks = _build_blocks(planning_result)

    registry = prompt_registry if prompt_registry is not None else PromptRegistry(
        DEFAULT_PROMPT_BASE_DIR
    )
    prompt_spec = registry.load(profile.prompt_id, version=profile.prompt_version)

    output_schema = _load_output_schema(profile, output_schema_path)

    # Profile consistency gate (FAIL CLOSED, zero provider calls)
    _check_profile_consistency(profile, prompt_spec, output_schema)

    structured_requests: list[StructuredGenerationRequest] = []
    for block in blocks:
        variables = {
            "block_id": block.block_id,
            "candidate_packets_json": block.candidate_packets_json,
            "requested_pairs_json": block.requested_pairs_json,
        }
        rendered_prompt = render_prompt(prompt_spec, variables)
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        structured_requests.append(request)

    return ReconciliationSemanticPreparation(
        blocks=blocks,
        structured_requests=tuple(structured_requests),
        semantic_request_hashes=tuple(r.request_hash for r in structured_requests),
        prompt_id=prompt_spec.prompt_id,
        prompt_version=prompt_spec.version,
        prompt_content_hash=prompt_spec.content_hash,
        output_schema_id=output_schema.schema_id,
        output_schema_version=output_schema.schema_version,
        output_schema_hash=output_schema.schema_hash,
        semantic_profile_id=semantic_profile.profile_id,
        semantic_profile_hash=semantic_profile.semantic_profile_hash,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def resolve_semantic_ambiguity(
    planning_result: ReconciliationPlanningResult,
    profile: EntityReconciliationProfile,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    *,
    prompt_registry: PromptRegistry | None = None,
    output_schema_path: str | Path = DEFAULT_OUTPUT_SCHEMA_PATH,
) -> ReconciliationSemanticResult:
    """Execute A4C semantic ambiguity resolution.

    Consumes the A4B ``ReconciliationPlanningResult`` and resolves all
    ``needs_semantic_decision`` pairs via bounded LLM semantic generation.

    The deterministic, zero-provider request construction is delegated to
    :func:`prepare_semantic_resolution` (the A4D reuse seam); this function
    only drives the provider per block and merges the results.

    Returns a :class:`ReconciliationSemanticResult` with:
      * all semantic blocks (in order);
      * all LLM decisions;
      * the combined deterministic + semantic decision set (all_decisions);
      * the semantic request hashes (for A4D reuse).

    Fails closed (raises) on:
      * profile/prompt/schema contradiction;
      * provenance mismatch;
      * A-I3 LLMError (propagated);
      * semantic exhaustion (both rounds invalid for a block).

    A4C does NOT persist, write CURRENT, or reuse.
    """
    preparation = prepare_semantic_resolution(
        planning_result,
        profile,
        semantic_profile,
        prompt_registry=prompt_registry,
        output_schema_path=output_schema_path,
    )
    blocks = preparation.blocks

    # 1. Zero semantic pairs → zero blocks → validate deterministic coverage
    if not blocks:
        all_decisions = _validate_decision_coverage(
            planning_result, planning_result.decisions, ()
        )
        return ReconciliationSemanticResult(
            planning_result=planning_result,
            blocks=blocks,
            semantic_decisions=(),
            all_decisions=all_decisions,
            semantic_request_hashes=preparation.semantic_request_hashes,
            block_results=(),
        )

    # 2. Process each block sequentially (provider invocation happens here)
    all_semantic_decisions: list[ReconciliationDecision] = []
    all_block_results: list[ReconciliationSemanticBlockResult] = []

    for block, request in zip(blocks, preparation.structured_requests):
        rendered_prompt = request.rendered_prompt
        output_schema = request.output_schema

        # Parse candidate packets for evidence validation
        candidate_packets = json.loads(block.candidate_packets_json)

        # Bounded semantic generation (max 2 rounds)
        max_rounds = profile.max_generation_rounds
        block_decisions: list[ReconciliationDecision] = []
        block_provenance: LLMInvocationProvenance | None = None
        rounds_attempted = 0
        last_failure = ""

        for round_number in range(1, max_rounds + 1):
            rounds_attempted = round_number

            # One provider-neutral generation (A-I3 owns technical retry)
            result = llm_client.generate_structured(
                rendered_prompt, output_schema, semantic_profile
            )

            # Provenance verification (FAIL CLOSED, NO semantic retry)
            _verify_provenance(
                result.provenance,
                request,
                rendered_prompt,
                output_schema,
                semantic_profile,
            )

            # Typed domain load (only typed-model rejection triggers retry)
            try:
                payload = ReconciliationDecisionPayload.from_dict(result.parsed_json)
            except ReconciliationModelError:
                last_failure = "typed payload load failed"
                continue

            # Exact pair/evidence validation
            is_valid, failure_detail = _validate_block_payload(
                payload, block.pair_plans, candidate_packets
            )
            if not is_valid:
                last_failure = failure_detail
                continue

            # Valid: convert to decisions
            for item in payload.decisions:
                block_decisions.append(
                    _convert_to_decision(
                        item,
                        request_hash=request.request_hash,
                        prompt_id=rendered_prompt.prompt_id,
                        prompt_version=rendered_prompt.prompt_version,
                        provenance=result.provenance,
                    )
                )
            block_provenance = result.provenance
            break  # success, stop retrying

        if block_decisions:
            all_semantic_decisions.extend(block_decisions)
            all_block_results.append(
                ReconciliationSemanticBlockResult(
                    block_id=block.block_id,
                    request_hash=request.request_hash,
                    semantic_rounds=rounds_attempted,
                    decisions=tuple(block_decisions),
                    generation_provenance=block_provenance,  # type: ignore[arg-type]
                )
            )
        else:
            # Semantic exhaustion
            raise ReconciliationSemanticGenerationError(
                block_id=block.block_id,
                rounds_attempted=rounds_attempted,
                last_failure_details=last_failure,
                expected_pairs=tuple(
                    (p.left_candidate_ref, p.right_candidate_ref)
                    for p in block.pair_plans
                ),
            )

    # 3. Combine + validate exact decision coverage
    all_decisions = _validate_decision_coverage(
        planning_result, planning_result.decisions, tuple(all_semantic_decisions)
    )

    return ReconciliationSemanticResult(
        planning_result=planning_result,
        blocks=blocks,
        semantic_decisions=tuple(all_semantic_decisions),
        all_decisions=all_decisions,
        semantic_request_hashes=preparation.semantic_request_hashes,
        block_results=tuple(all_block_results),
    )


def _validate_decision_coverage(
    planning_result: ReconciliationPlanningResult,
    deterministic_decisions: tuple[ReconciliationDecision, ...],
    semantic_decisions: tuple[ReconciliationDecision, ...],
) -> tuple[ReconciliationDecision, ...]:
    """Validate and combine all decisions against the authoritative pair plans.

    Enforces the frozen invariant:
      * every explicit A4B pair in planning_result.pair_plans has exactly one
        decision;
      * no extra decisions for pairs not in pair_plans;
      * state/method consistency:
          auto_same → method=deterministic, decision=same_entity
          must_not_merge → method=deterministic, decision=different_entity
          needs_semantic_decision → method=llm, decision ∈
              {same_entity, different_entity, uncertain}

    Returns canonical sort by (left_candidate_ref, right_candidate_ref).
    """
    all_decisions = list(deterministic_decisions) + list(semantic_decisions)

    # Build the decision lookup: pair key → decision
    decision_by_pair: dict[tuple[str, str], ReconciliationDecision] = {}
    for d in all_decisions:
        key = (d.left_candidate_ref, d.right_candidate_ref)
        if key in decision_by_pair:
            raise ReconciliationSemanticError(
                f"duplicate decision for pair {key!r}"
            )
        decision_by_pair[key] = d

    # Build the authoritative pair set from pair_plans
    expected_pairs: set[tuple[str, str]] = set()
    pair_state: dict[tuple[str, str], str] = {}
    for plan in planning_result.pair_plans:
        pair_key = (plan.left_candidate_ref, plan.right_candidate_ref)
        expected_pairs.add(pair_key)
        pair_state[pair_key] = plan.state

    # Check: no extra decisions
    for pair_key in decision_by_pair:
        if pair_key not in expected_pairs:
            raise ReconciliationSemanticError(
                f"decision found for pair {pair_key!r} not present in "
                f"planning_result.pair_plans; invalid decision coverage"
            )

    # Check: every expected pair has exactly one decision
    for pair_key in expected_pairs:
        if pair_key not in decision_by_pair:
            raise ReconciliationSemanticError(
                f"no decision found for pair {pair_key!r} in "
                f"planning_result.pair_plans; incomplete decision coverage"
            )

    # State/method consistency validation
    for pair_key, state in pair_state.items():
        d = decision_by_pair[pair_key]
        if state == PAIR_STATE_AUTO_SAME:
            if d.method != "deterministic":
                raise ReconciliationSemanticError(
                    f"pair {pair_key!r} (auto_same) has method {d.method!r}; "
                    f"expected 'deterministic'"
                )
            if d.decision != "same_entity":
                raise ReconciliationSemanticError(
                    f"pair {pair_key!r} (auto_same) has decision {d.decision!r}; "
                    f"expected 'same_entity'"
                )
        elif state == PAIR_STATE_MUST_NOT_MERGE:
            if d.method != "deterministic":
                raise ReconciliationSemanticError(
                    f"pair {pair_key!r} (must_not_merge) has method {d.method!r}; "
                    f"expected 'deterministic'"
                )
            if d.decision != "different_entity":
                raise ReconciliationSemanticError(
                    f"pair {pair_key!r} (must_not_merge) has decision "
                    f"{d.decision!r}; expected 'different_entity'"
                )
        elif state == PAIR_STATE_NEEDS_SEMANTIC_DECISION:
            if d.method != "llm":
                raise ReconciliationSemanticError(
                    f"pair {pair_key!r} (needs_semantic_decision) has method "
                    f"{d.method!r}; expected 'llm'"
                )
            if d.decision not in ("same_entity", "different_entity", "uncertain"):
                raise ReconciliationSemanticError(
                    f"pair {pair_key!r} (needs_semantic_decision) has decision "
                    f"{d.decision!r}; expected same_entity/different_entity/uncertain"
                )

    # Canonical sort
    all_decisions.sort(key=lambda d: (d.left_candidate_ref, d.right_candidate_ref))
    return tuple(all_decisions)
