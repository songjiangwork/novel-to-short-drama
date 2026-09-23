"""v1.2 A4C — provider-neutral semantic ambiguity resolution.

This module implements the A4C semantic ambiguity-resolution slice:

    ReconciliationPlanningResult (A4B)
        ↓
    deterministic semantic block packing (MAX_PAIRS_PER_BLOCK=6,
    MAX_CANDIDATES_PER_BLOCK=12)
        ↓
    pair-local context rendering (endpoint-only; each pair owns its own
    left/right endpoint packet; evidence cited by pair-local selectors)
        ↓
    PromptRegistry rendering (a4.entity-reconciliation v3)
        ↓
    OutputSchema build (reconciliation-decision-selector-payload.schema.json)
        ↓
    LLMClient.generate_structured(...) — up to 2 semantic rounds per block
        ↓
    exact pair + evidence-selector validation (fail closed, pair-scoped)
        ↓
    canonical pair-local selector order (left-by-index, then right-by-index)
        ↓
    selector -> exact endpoint EvidenceRef resolution (Python-owned identity)
        ↓
    stable exact-EvidenceRef alias dedup (first-occurrence in canonical order)
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
  * A4A ``ReconciliationDecision`` / ``EvidenceRef`` / the v3 pair-local
    ``ReconciliationSelectorDecisionPayload`` (provider cites evidence only by
    pair-local selectors; A4C resolves them to the exact endpoint EvidenceRefs).

The provider never reproduces paragraph_id / role / strength / excerpt: it
returns pair-local evidence selectors (L0/L1/... for the decision's own left
endpoint, R0/R1/... for its right endpoint). A4C validates each selector
against that exact pair (syntax, range, duplicates, no third candidate) and,
after every selector in a decision is valid, canonicalizes the selector set to
the frozen pair-local order (all left selectors by ascending index, then all
right selectors by ascending index) before resolving it to the exact endpoint
EvidenceRef objects. The resolved sequence is then projected to the canonical
exact-EvidenceRef form: two distinct valid selectors that resolve to the same
exact EvidenceRef identity (paragraph_id, role, strength, excerpt) are
collapsed to the first occurrence in canonical selector order. This is a
persisted-projection canonicalization (NOT selector deduplication or repair),
and it ensures the persisted ``evidence_refs`` tuple satisfies the A4D
invariant (no duplicate exact EvidenceRef). Canonicalizing means semantically
equivalent selector permutations (e.g. ["L0","R0"] vs ["R0","L0"]) resolve to
the SAME persisted EvidenceRef tuple and the SAME decision_id. An invalid
selector fails the A4C semantic output validation and consumes a bounded
semantic round exactly like the existing invalid-evidence behavior. The
persisted A4 decision contracts are unchanged and selector strings never enter
them.
"""

from __future__ import annotations

import re
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
    EVIDENCE_SELECTOR_PATTERN,
    EntityReconciliationProfile,
    ReconciliationDecision,
    ReconciliationSelectorDecisionItem,
    ReconciliationSelectorDecisionPayload,
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
# The production A4C provider contract is the pair-local evidence-selector
# payload (v3). The old raw-EvidenceRef payload schema is preserved for the
# historical v1/v2 profiles but is no longer the default.
DEFAULT_OUTPUT_SCHEMA_PATH = SCHEMAS_DIR / "reconciliation-decision-selector-payload.schema.json"

# Reason code mapping: decision → reason_code
_REASON_CODE_MAP = {
    "same_entity": "llm_same_entity",
    "different_entity": "llm_different_entity",
    "uncertain": "llm_uncertain",
}

# Pair-local evidence selector (v3 contract): "L<index>" selects the decision's
# own left endpoint evidence item at zero-based ``index`` and "R<index>" selects
# the right endpoint evidence item at ``index``. This is the single A4C authority
# for selector validity (form, range, duplicates, no third candidate); the
# tracked output schema is intentionally lenient (list of strings) so an invalid
# selector reaches this semantic validation and consumes a bounded round, exactly
# like the existing invalid-evidence behavior.
_EVIDENCE_SELECTOR_RE = re.compile(EVIDENCE_SELECTOR_PATTERN)


def llm_reason_code(decision: str) -> str:
    """The deterministic A4C ``reason_code`` for an LLM decision.

    Reused by A4D CURRENT verification so the recomputed decision_id derives
    its reason code from the decision itself (never from a persisted field).
    """
    try:
        return _REASON_CODE_MAP[decision]
    except KeyError:
        raise ReconciliationSemanticError(
            f"no A4C LLM reason code for decision {decision!r}"
        ) from None


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
    # Pair-local contexts: one entry per requested pair, each carrying the pair's
    # own left/right endpoint packet (the evidence-selection authority). The
    # left endpoint's evidence items are the pair's L0/L1/... selectors (in
    # order) and the right endpoint's evidence items are the R0/R1/... selectors
    # (in order). No block-wide evidence pool is presented.
    pair_contexts_json: str


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
    # Authoritative A4B plan identity this preparation was built from. A4D uses
    # this to require ``preparation.plan_hash == planning_result.plan_hash`` so
    # the derived A4SemanticIdentity is bound to the exact plan it describes.
    plan_hash: str


# ---------------------------------------------------------------------------
# Block packing
# ---------------------------------------------------------------------------


def pack_semantic_pairs_v1(
    planning_result: ReconciliationPlanningResult,
) -> tuple[tuple[ReconciliationPairPlan, ...], ...]:
    """Provider-free A4C semantic block packing authority.

    Deterministic greedy packing of the ``needs_semantic_decision`` pair plans
    into blocks using the frozen :data:`MAX_PAIRS_PER_BLOCK` / :data:`MAX_CANDIDATES_PER_BLOCK`
    limits and canonical pair ordering (``(left, right)``). This is the SAME
    packing :func:`prepare_semantic_resolution` uses to build the semantic
    blocks, exposed provider-free so A4D can bind each semantic pair to its
    exact block request hash (``semantic_request_hashes[block_ordinal]``) without
    prompt rendering or provider calls. Do NOT duplicate this algorithm.
    """
    semantic_pairs = [
        p
        for p in planning_result.pair_plans
        if p.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
    ]
    semantic_pairs.sort(key=lambda p: (p.left_candidate_ref, p.right_candidate_ref))

    if not semantic_pairs:
        return ()

    blocks: list[tuple[ReconciliationPairPlan, ...]] = []
    current_block: list[ReconciliationPairPlan] = []
    current_candidates: set[str] = set()

    for pair in semantic_pairs:
        pair_candidates = {pair.left_candidate_ref, pair.right_candidate_ref}
        if current_block:
            would_exceed_pairs = len(current_block) + 1 > MAX_PAIRS_PER_BLOCK
            would_exceed_candidates = (
                len(current_candidates | pair_candidates) > MAX_CANDIDATES_PER_BLOCK
            )
            if would_exceed_pairs or would_exceed_candidates:
                blocks.append(tuple(current_block))
                current_block = []
                current_candidates = set()
        current_block.append(pair)
        current_candidates |= pair_candidates

    if current_block:
        blocks.append(tuple(current_block))

    return tuple(blocks)


def _pack_semantic_blocks(
    planning_result: ReconciliationPlanningResult,
) -> list[list[ReconciliationPairPlan]]:
    """Backward-compatible list-of-lists view of the frozen packing authority."""
    return [list(block) for block in pack_semantic_pairs_v1(planning_result)]


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


def _endpoint_packet(entry: Any) -> dict[str, Any]:
    """Build the pair-local endpoint packet for one candidate index entry.

    The packet carries the endpoint identity material (name / aliases /
    descriptors) and the endpoint's evidence items. The evidence items are the
    pair's selector universe for this endpoint: index ``n`` is the selector
    ``L<n>`` (for the left endpoint) or ``R<n>`` (for the right endpoint), in
    the exact order of ``entry.evidence_refs``.

    FAILS CLOSED (structurally) if an evidence excerpt is not a string or null.
    """
    evidence = []
    for ev in entry.evidence_refs:
        if ev.excerpt is not None and not isinstance(ev.excerpt, str):
            raise ReconciliationSemanticError(
                f"endpoint {entry.candidate_ref!r} has a non-string, non-null "
                f"evidence excerpt; the tracked A3A corpus must be consistent "
                f"for the endpoint-only evidence contract"
            )
        evidence.append(ev.to_dict())
    return {
        "candidate_ref": entry.candidate_ref,
        "candidate_kind": entry.candidate_kind,
        "display_name_original": entry.display_name_original,
        "aliases_original": list(entry.aliases_original),
        "descriptors_zh": list(entry.descriptors_zh),
        "evidence": evidence,
    }


def _build_pair_contexts(
    planning_result: ReconciliationPlanningResult,
    pair_plans: list[ReconciliationPairPlan],
) -> tuple[str, tuple[str, ...]]:
    """Build pair-local contexts for a block.

    Returns (pair_contexts_json, candidate_refs_tuple).

    Each pair context carries the pair's own left/right endpoint packet (the
    evidence-selection authority) plus the pair signals / shared identity keys /
    shared tokens. The left endpoint's evidence items are the pair's ``L0`` /
    ``L1`` / ... selectors (in order) and the right endpoint's evidence items
    are the ``R0`` / ``R1`` / ... selectors (in order). No block-wide evidence
    pool is presented.

    ``candidate_refs`` is the unique set of endpoint refs in source_order_key
    order (the block_id material, unchanged from the prior candidate-packet
    path so block ids stay stable).

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

    # Sort by source_order_key (canonical A4B order) -> block_id candidate refs.
    endpoint_entries.sort(key=lambda e: e.source_order_key)
    candidate_refs = tuple(e.candidate_ref for e in endpoint_entries)

    contexts = []
    for p in pair_plans:
        left = ref_to_entry[p.left_candidate_ref]
        right = ref_to_entry[p.right_candidate_ref]
        contexts.append(
            {
                "left_candidate_ref": p.left_candidate_ref,
                "right_candidate_ref": p.right_candidate_ref,
                "signals": list(p.signals),
                "shared_identity_keys": list(p.shared_identity_keys),
                "shared_tokens": list(p.shared_tokens),
                "left_endpoint": _endpoint_packet(left),
                "right_endpoint": _endpoint_packet(right),
            }
        )

    pair_contexts_json = canonical_json_bytes(contexts).decode("utf-8")
    return pair_contexts_json, candidate_refs


def _build_blocks(
    planning_result: ReconciliationPlanningResult,
) -> tuple[ReconciliationSemanticBlock, ...]:
    """Build all semantic blocks for the planning result."""
    packed = _pack_semantic_blocks(planning_result)

    blocks: list[ReconciliationSemanticBlock] = []
    for ordinal, pair_plans in enumerate(packed):
        pair_contexts_json, candidate_refs = _build_pair_contexts(
            planning_result, pair_plans
        )
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
                pair_contexts_json=pair_contexts_json,
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
# Pair/evidence-selector validation + resolution
# ---------------------------------------------------------------------------


def _block_endpoint_evidence(
    planning_result: ReconciliationPlanningResult,
    pair_plans: list[ReconciliationPairPlan],
) -> list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]]:
    """Per-pair endpoint evidence for selector resolution (aligned with plans).

    Returns a list where element ``i`` is the pair ``i``'s
    ``(left_endpoint_evidence, right_endpoint_evidence)`` -- each a tuple of the
    exact endpoint EvidenceRefs in the SAME order shown in the pair context.
    ``L<n>`` therefore resolves to ``left_endpoint_evidence[n]`` and ``R<n>`` to
    ``right_endpoint_evidence[n]``. This is re-derived from the authoritative
    candidate index so the resolved EvidenceRef is byte-for-byte the endpoint's
    own EvidenceRef (exact identity, null excerpt preserved).
    """
    ref_to_entry = {e.candidate_ref: e for e in planning_result.candidate_index.entries}
    out: list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]] = []
    for p in pair_plans:
        left = ref_to_entry[p.left_candidate_ref]
        right = ref_to_entry[p.right_candidate_ref]
        out.append((left.evidence_refs, right.evidence_refs))
    return out


def _canonicalize_evidence_selectors(selectors: list[str]) -> list[str]:
    """Canonicalize validated, unique pair-local selectors to the frozen order.

    Frozen pair-local order (this order is authoritative for persistence):
      1. all left selectors first, numeric index ascending;
      2. then all right selectors, numeric index ascending.

    ``["R2", "L1", "R0", "L0"]`` -> ``["L0", "L1", "R0", "R2"]``.

    ``selectors`` must already be validated (``L<index>`` / ``R<index>`` form)
    and duplicate-free; nothing is re-validated or dropped here. Canonicalizing
    means semantically equivalent selector permutations resolve to the SAME
    persisted EvidenceRef tuple (and therefore the SAME ``decision_id``),
    independent of the order the provider returned them in. The endpoint
    evidence order itself is NOT changed.
    """
    left = sorted((s for s in selectors if s[0] == "L"), key=lambda s: int(s[1:]))
    right = sorted((s for s in selectors if s[0] == "R"), key=lambda s: int(s[1:]))
    return left + right


def _evidence_exact_identity(ev: EvidenceRef) -> tuple[str, str, str, "str | None"]:
    """The exact persisted EvidenceRef identity tuple.

    Two EvidenceRefs are the same evidence for dedupe purposes if and only if
    their complete identity tuple ``(paragraph_id, role, strength, excerpt)``
    is equal. ``excerpt=None`` is distinct from any string value. No
    normalization, case-folding, or semantic similarity is applied.
    """
    return (ev.paragraph_id, ev.role, ev.strength, ev.excerpt)


def _validate_selector_block_payload(
    payload: ReconciliationSelectorDecisionPayload,
    pair_plans: tuple[ReconciliationPairPlan, ...],
    endpoint_evidence: list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]],
) -> tuple[bool, str, list[tuple[EvidenceRef, ...]]]:
    """Validate a block's v3 provider payload against the exact requested pairs.

    This is the single A4C authority for selector validity. ``endpoint_evidence``
    is aligned with ``pair_plans``; ``L<n>`` resolves to the pair's own left
    endpoint evidence item ``n`` and ``R<n>`` to the right endpoint evidence item
    ``n`` (both zero-based, in the order shown in the pair context).

    Returns (is_valid, failure_detail, resolved_evidence). ``resolved_evidence``
    is aligned with ``payload.decisions`` when valid (each a tuple of the exact
    endpoint EvidenceRefs in the CANONICAL pair-local selector order with exact-
    EvidenceRef alias deduplication; see :func:`_canonicalize_evidence_selectors`
    and :func:`_evidence_exact_identity`), otherwise empty.

    Checks (the exact pair order / ref checks are UNCHANGED from the prior
    endpoint-only validator):
      * count exact, order exact, left/right refs exact
      * each selector matches ``L<index>`` / ``R<index>`` (only these forms are
        legal; invalid prefix / negative / non-integer index rejected)
      * the index is in range for the pair's own endpoint (out-of-range rejected)
      * no selector can reach a third candidate (L -> left endpoint, R -> right
        endpoint only)
      * no duplicate selector within a decision

    Only AFTER every selector in a decision has passed syntax + range +
    duplicate validation is the selector SET canonicalized (left selectors
    first by index, then right selectors by index) and resolved to the exact
    endpoint EvidenceRefs. The resolved sequence is then projected to the
    canonical exact-EvidenceRef form: duplicates by exact identity
    ``(paragraph_id, role, strength, excerpt)`` are collapsed to their first
    occurrence in canonical selector order. This is a persisted-projection
    canonicalization, NOT a selector repair: the provider's valid selectors
    are all preserved, and the resulting ``evidence_refs`` tuple satisfies the
    A4D invariant (no duplicate exact EvidenceRef). Invalid / out-of-range /
    duplicate selectors always fail BEFORE canonicalization and consume a
    bounded semantic round.
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
        ), []

    resolved: list[tuple[EvidenceRef, ...]] = []
    for i, item in enumerate(decisions):
        expected_left, expected_right = requested_pairs[i]

        # Order + ref exact (UNCHANGED)
        if item.left_candidate_ref != expected_left:
            return False, (
                f"pair {i}: left_candidate_ref mismatch: expected "
                f"{expected_left!r}, got {item.left_candidate_ref!r}"
            ), []
        if item.right_candidate_ref != expected_right:
            return False, (
                f"pair {i}: right_candidate_ref mismatch: expected "
                f"{expected_right!r}, got {item.right_candidate_ref!r}"
            ), []

        left_evidence, right_evidence = endpoint_evidence[i]
        seen_selectors: set[str] = set()
        validated_selectors: list[str] = []
        for selector in item.evidence_selectors:
            # Only L<index> / R<index> forms are legal (invalid prefix /
            # negative / non-integer index rejected here, at semantic time).
            if _EVIDENCE_SELECTOR_RE.fullmatch(selector) is None:
                return False, (
                    f"pair {i}: invalid evidence selector {selector!r}; only "
                    f"L<index>/R<index> forms are legal"
                ), []
            if selector in seen_selectors:
                return False, (
                    f"pair {i}: duplicate evidence selector: {selector!r}"
                ), []
            seen_selectors.add(selector)
            index = int(selector[1:])
            if selector[0] == "L":
                if index >= len(left_evidence):
                    return False, (
                        f"pair {i}: evidence selector {selector!r} out of range "
                        f"for the left endpoint "
                        f"({len(left_evidence)} evidence item(s))"
                    ), []
            else:  # "R"
                if index >= len(right_evidence):
                    return False, (
                        f"pair {i}: evidence selector {selector!r} out of range "
                        f"for the right endpoint "
                        f"({len(right_evidence)} evidence item(s))"
                    ), []
            validated_selectors.append(selector)

        # All selectors for this decision are valid, unique, and in range.
        # Canonicalize (left-by-index, then right-by-index) BEFORE resolving so
        # semantically equivalent permutations persist identically.
        canonical_selectors = _canonicalize_evidence_selectors(validated_selectors)

        # Resolve the canonical selector sequence to exact endpoint EvidenceRefs,
        # then apply stable exact-EvidenceRef alias deduplication: two distinct
        # valid selectors (e.g. L0 and R0) may resolve to the same exact
        # EvidenceRef identity; only the first occurrence (in canonical selector
        # order) is persisted. This is NOT selector deduplication — duplicate
        # selector strings are still rejected above.
        decision_evidence: list[EvidenceRef] = []
        seen_exact_evidence: set[tuple[str, str, str, "str | None"]] = set()
        for selector in canonical_selectors:
            index = int(selector[1:])
            if selector[0] == "L":
                ev = left_evidence[index]
            else:  # "R"
                ev = right_evidence[index]
            identity = _evidence_exact_identity(ev)
            if identity not in seen_exact_evidence:
                seen_exact_evidence.add(identity)
                decision_evidence.append(ev)
        resolved.append(tuple(decision_evidence))

    return True, "", resolved


# ---------------------------------------------------------------------------
# Decision ID computation
# ---------------------------------------------------------------------------


def compute_llm_decision_id(
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

    This is the single authority for A4C LLM decision ids, reused by A4D
    CURRENT verification to require ``persisted decision_id == deterministically
    recomputed decision_id``.

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
    left_ref: str,
    right_ref: str,
    decision: str,
    reason_zh: str,
    evidence_refs: tuple[EvidenceRef, ...],
    request_hash: str,
    prompt_id: str,
    prompt_version: int,
    provenance: LLMInvocationProvenance,
) -> ReconciliationDecision:
    """Convert a valid, selector-resolved decision to a ReconciliationDecision.

    ``evidence_refs`` are the exact endpoint EvidenceRefs resolved from the
    provider's pair-local selectors in the canonical pair-local selector order
    (left-by-index, then right-by-index; see
    :func:`_canonicalize_evidence_selectors`) with stable exact-EvidenceRef
    alias deduplication applied (first occurrence in canonical order; see
    :func:`_evidence_exact_identity`). The persisted
    :class:`ReconciliationDecision` carries those exact EvidenceRefs -- no
    selector strings ever appear in the persisted contract, the ordering is
    canonical (independent of the order the provider returned the selectors),
    and the tuple contains no duplicate exact EvidenceRefs.
    """
    reason_code = _REASON_CODE_MAP[decision]
    decision_id = compute_llm_decision_id(
        left_ref=left_ref,
        right_ref=right_ref,
        decision=decision,
        method="llm",
        reason_code=reason_code,
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        request_hash=request_hash,
    )
    return ReconciliationDecision(
        decision_id=decision_id,
        left_candidate_ref=left_ref,
        right_candidate_ref=right_ref,
        decision=decision,
        method="llm",
        reason_code=reason_code,
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
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
            "pair_contexts_json": block.pair_contexts_json,
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
        plan_hash=planning_result.plan_hash,
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

        # Per-pair endpoint evidence for selector resolution (re-derived from the
        # authoritative candidate index; aligned with block.pair_plans).
        endpoint_evidence = _block_endpoint_evidence(
            planning_result, list(block.pair_plans)
        )

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
                payload = ReconciliationSelectorDecisionPayload.from_dict(
                    result.parsed_json
                )
            except ReconciliationModelError:
                last_failure = "typed payload load failed"
                continue

            # Exact pair + evidence-selector validation (pair-scoped, fail
            # closed). An invalid selector reaches HERE (the tracked output
            # schema is lenient) and consumes a bounded semantic round, exactly
            # like the existing invalid-evidence behavior.
            is_valid, failure_detail, resolved_evidence = (
                _validate_selector_block_payload(
                    payload, block.pair_plans, endpoint_evidence
                )
            )
            if not is_valid:
                last_failure = failure_detail
                continue

            # Valid: resolve selectors to exact endpoint EvidenceRefs and convert
            for item, resolved in zip(payload.decisions, resolved_evidence):
                block_decisions.append(
                    _convert_to_decision(
                        left_ref=item.left_candidate_ref,
                        right_ref=item.right_candidate_ref,
                        decision=item.decision,
                        reason_zh=item.reason_zh,
                        evidence_refs=resolved,
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
