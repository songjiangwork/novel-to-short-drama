"""v1.2 A5C-A fact semantic preparation + packing audit (zero provider).

Implements A5C-A (Issue #52): build the A5C fact semantic preparation from an
A5B ``ConsolidationPlanningResult`` and audit the deterministic block-packing
candidates -- with NO provider call and NO persistence anywhere in this module
or the audit script.

* **Semantic stream** -- the fact semantic stream is EXACTLY the set of fact
  pair plans in state ``needs_semantic_decision`` (auto_same pairs are
  deterministic and excluded). Before packing the stream is validated fail
  closed (fact namespace, ``left_ref < right_ref``, no duplicate pair, canonical
  ``(left_ref, right_ref)`` order, exact ``needs_semantic_decision`` coverage,
  auto_same excluded).

* **Explicit packing policy** -- a block is bounded by TWO independent limits
  from a ``FactSemanticPackingPolicy``: at most ``max_pairs_per_block`` fact
  semantic pairs AND at most ``max_candidates_per_block`` unique candidate
  endpoints. ``fact-semantic-packing-v1`` is NOT YET FROZEN: A5C-A requires an
  explicit policy (no production default). The three audit-only candidates are
  P1 (6/12), P2 (12/24), P3 (24/48). Packing is deterministic in the canonical
  ``(left_ref, right_ref)`` order: no pair is split, lost, duplicated, reordered
  heuristically, or packed randomly.

* **Block identity** -- each block carries a deterministic ``candidate_refs``
  (the stable unique union of its pair endpoints, ordered by candidate
  ``source_order_key`` then ref) and a ``block_id`` = ``a5fblk_`` + first 20 hex
  chars of the ``content_hash`` of material binding the plan hash, the packing
  policy limits, the block ordinal, the ordered requested pairs, and the ordered
  candidate refs (no runtime/backend metadata).

* **Pair-local endpoint packets** -- each pair carries its own left/right
  endpoint fact packets, derived from the ``IndexedFactCandidate`` (no new
  decision/selector/profile model). Each packet carries the candidate identity,
  the normalized statement/type, the subject/object refs, the exact
  ``evidence_strength``, the ``source_order_key``, and the source-anchored
  evidence items labeled with pair-local selectors ``L0/L1/...`` (left) or
  ``R0/R1/...`` (right) in the EXACT indexed A5B evidence order (no re-sort).
  Provider context is strictly pair-local: there is NO block-wide semantic or
  evidence pool.

* **Real ``StructuredGenerationRequest`` objects** -- every block is rendered
  through the existing ``PromptRegistry`` / ``OutputSchema`` /
  ``SemanticLLMProfile`` infrastructure: the tracked ``a5.fact-consolidation``
  prompt v1 (``required_variables`` = exactly ``block_id`` +
  ``pair_contexts_json``), the ``consolidation-fact-selector-payload`` schema
  v1, and the ``consolidation-llm-v1`` semantic profile. Stable per-request
  hashes are computed before any provider call.

The consolidation / prompt / schema / semantic-profile identity is verified
fail closed (including ``max_generation_rounds == 2``). Nothing in this module
calls a provider or persists anything; the only provider boundary is the
rendered ``StructuredGenerationRequest`` set.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from short_drama.artifacts.canonical import canonical_json_bytes, content_hash
from short_drama.io import load_json
from short_drama.llm import (
    LLMClient,
    LLMInvocationProvenance,
    PromptRegistry,
    build_structured_request,
    load_semantic_profile,
    render_prompt,
)
from short_drama.llm.models import (
    OutputSchema,
    SemanticLLMProfile,
    StructuredGenerationRequest,
)
from short_drama.paths import PROFILES_DIR, SCHEMAS_DIR

from .consolidation import (
    A5_EVIDENCE_SELECTOR_PATTERN,
    ConsolidationCandidateRef,
    ConsolidationProfile,
    ConsolidationSemanticPass,
    EVENT_DECISIONS,
    FACT_DECISIONS,
    RELATIONSHIP_DECISIONS,
    EventSemanticDecision,
    FactSelectorDecisionPayload,
    FactSemanticDecision,
    RelationshipSemanticDecision,
)
from .consolidation_planning import (
    ConsolidationPlanningResult,
    EventPairPlan,
    FactPairPlan,
    IndexedEventCandidate,
    IndexedFactCandidate,
    IndexedRelationshipCandidate,
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    RelationshipPairPlan,
)
from .errors import (
    ConsolidationModelError,
    ConsolidationProvenanceError,
    ConsolidationSemanticError,
    ConsolidationSemanticGenerationError,
    StoryIntegrityError,
)
from .extraction import EvidenceRef

# ---------------------------------------------------------------------------
# Frozen A5C fact semantic identity (A5C-A BLOCK 2 / BLOCK 4)
# ---------------------------------------------------------------------------

#: Tracked A5 fact consolidation prompt identity.
A5C_FACT_PROMPT_ID = "a5.fact-consolidation"
A5C_FACT_PROMPT_VERSION = 1

#: Tracked A5 fact selector-payload output schema identity.
A5C_FACT_OUTPUT_SCHEMA_ID = "consolidation-fact-selector-payload"
A5C_FACT_OUTPUT_SCHEMA_VERSION = 1

#: Tracked A5 fact selector-payload output schema path.
A5C_FACT_OUTPUT_SCHEMA_PATH = SCHEMAS_DIR / "consolidation-fact-selector-payload.schema.json"

#: Tracked A5 semantic LLM profile identity (generation semantics only).
A5C_FACT_SEMANTIC_PROFILE_ID = "consolidation-llm-v1"

#: Tracked A5 semantic LLM profile path (no endpoint / model / credential).
A5C_FACT_SEMANTIC_PROFILE_PATH = PROFILES_DIR / "consolidation_llm_v1.yaml"

#: A5 fact block id prefix (A5C-A BLOCK 1: distinct from A4 ``a4rblk_``).
A5C_BLOCK_PREFIX = "a5fblk_"
#: First N hex chars of the block identity hash that identify a block.
A5C_BLOCK_ID_HEX_LENGTH = 20

#: Frozen requirement: the fact semantic pass generation budget (BLOCK 7).
A5C_FACT_MAX_GENERATION_ROUNDS = 2

#: The exact A5C fact semantic endpoint-packet fields (BLOCK 6).
A5C_FACT_ENDPOINT_PACKET_FIELDS = (
    "candidate_ref",
    "chunk_id",
    "local_candidate_id",
    "fact_type",
    "statement_zh",
    "subject_refs",
    "object_refs",
    "evidence_strength",
    "source_order_key",
    "evidence",
)


# ---------------------------------------------------------------------------
# Packing policy (A5C-A BLOCK 1 / BLOCK 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactSemanticPackingPolicy:
    """An explicit A5C fact packing policy with TWO independent limits.

    * ``max_pairs_per_block`` -- at most this many fact semantic pairs in a
      block.
    * ``max_candidates_per_block`` -- at most this many unique candidate
      endpoints across the pairs in a block.

    ``fact-semantic-packing-v1`` is NOT YET FROZEN. The audit candidates below
    are audit-only; A5C-A always requires an explicit policy (there is no
    production default), and A5C-B later consumes the policy selected after
    architecture review.
    """

    name: str
    max_pairs_per_block: int
    max_candidates_per_block: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise StoryIntegrityError(
                "FactSemanticPackingPolicy.name must be a non-empty string"
            )
        if not isinstance(self.max_pairs_per_block, int) or self.max_pairs_per_block < 1:
            raise StoryIntegrityError("max_pairs_per_block must be a positive integer")
        # A single pair has two distinct endpoints, so the candidate limit must
        # be able to hold at least one pair.
        if (
            not isinstance(self.max_candidates_per_block, int)
            or self.max_candidates_per_block < 2
        ):
            raise StoryIntegrityError(
                "max_candidates_per_block must be an integer >= 2 "
                "(one pair has two endpoints)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_pairs_per_block": self.max_pairs_per_block,
            "max_candidates_per_block": self.max_candidates_per_block,
        }


#: The three deterministic block-packing candidates audited by A5C-A (BLOCK 1):
#: P1 = 6 pairs / 12 candidates, P2 = 12 / 24, P3 = 24 / 48. Audit-only.
A5C_PACKING_CANDIDATES: tuple[FactSemanticPackingPolicy, ...] = (
    FactSemanticPackingPolicy("P1", 6, 12),
    FactSemanticPackingPolicy("P2", 12, 24),
    FactSemanticPackingPolicy("P3", 24, 48),
)

# FROZEN production fact semantic packing policy (A5C-B).
#
# ``fact-semantic-packing-v1`` is the FROZEN POST-AUDIT policy (docs
# ``v1.2-A5C-fact-semantic-packing-v1.md``): 12 pairs / 24 candidates (audit
# candidate P2). The canonical packing material is the frozen behavioral limits
# (12 / 24), NOT the policy label: ``FactSemanticPackingPolicy.to_dict()``
# carries only the two limits, so this policy produces byte-identical block ids
# and request hashes to the audited P2 preparation (identical block
# membership, request hashes). A5C-B consumes this exact policy and never
# re-selects P1 / P2 / P3 at runtime.
FACT_SEMANTIC_PACKING_V1: FactSemanticPackingPolicy = FactSemanticPackingPolicy(
    "fact-semantic-packing-v1",
    12,
    24,
)

# A5 pair-local evidence selector (L0 / R0 / L1 / ...). This is the single
# A5C authority for selector syntax; it reuses the frozen A5A pattern (no
# duplicate, subtly different regex). Pair-scoped range / duplicate /
# resolution validation is A5C-B (below).
_FACT_EVIDENCE_SELECTOR_RE = re.compile(A5_EVIDENCE_SELECTOR_PATTERN)


# ---------------------------------------------------------------------------
# Fact semantic block (A5C-A BLOCK 1 / BLOCK 3 / BLOCK 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactSemanticBlock:
    """One deterministic fact consolidation block (pair-local contexts).

    ``pair_contexts`` is the exact canonical block payload: one pair-context
    dict per pair, in the canonical ``(left_ref, right_ref)`` order.
    ``candidate_refs`` is the stable unique union of the block's pair endpoints
    ordered by candidate ``source_order_key`` then ref (bookkeeping / block
    identity only -- never a block-wide semantic or evidence pool).
    ``block_id`` binds the frozen material (plan hash, packing policy limits,
    block ordinal, ordered requested pairs, ordered candidate refs).
    """

    block_ordinal: int
    block_id: str
    pair_contexts: tuple[dict[str, Any], ...]
    candidate_refs: tuple[str, ...]
    pair_refs: tuple[tuple[str, str], ...]
    pair_contexts_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.block_ordinal, int) or self.block_ordinal < 0:
            raise StoryIntegrityError("block_ordinal must be a non-negative integer")
        if not self.block_id.startswith(A5C_BLOCK_PREFIX):
            raise StoryIntegrityError(
                f"fact block_id must start with {A5C_BLOCK_PREFIX!r}: {self.block_id!r}"
            )
        hex_part = self.block_id[len(A5C_BLOCK_PREFIX):]
        if len(hex_part) != A5C_BLOCK_ID_HEX_LENGTH or any(
            c not in "0123456789abcdef" for c in hex_part
        ):
            raise StoryIntegrityError(
                f"fact block_id must be {A5C_BLOCK_PREFIX!r} + "
                f"{A5C_BLOCK_ID_HEX_LENGTH} lowercase hex chars: {self.block_id!r}"
            )
        if tuple(self.pair_refs) != tuple(
            (pc["left_candidate_ref"], pc["right_candidate_ref"])
            for pc in self.pair_contexts
        ):
            raise StoryIntegrityError("fact block pair_refs does not match the pair contexts")
        # BLOCK 3: candidate_refs must equal the exact stable union of the
        # block's pair endpoints, ordered by candidate source_order_key then ref.
        keyed: list[tuple[str, str]] = []
        for pc in self.pair_contexts:
            for side in ("left", "right"):
                packet = pc[side]
                keyed.append((packet["source_order_key"], packet["candidate_ref"]))
        expected_refs = tuple(ref for _so, ref in sorted(set(keyed)))
        if tuple(self.candidate_refs) != expected_refs:
            raise StoryIntegrityError(
                "candidate_refs must equal the stable union of the block pair endpoints "
                "(ordered by candidate source_order_key then ref)"
            )

    @property
    def pair_count(self) -> int:
        return len(self.pair_contexts)

    @property
    def payload_bytes(self) -> int:
        return len(self.pair_contexts_json.encode("utf-8"))


# ---------------------------------------------------------------------------
# Fact semantic preparation result (A5C-A BLOCK 2 / BLOCK 11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactSemanticPreparation:
    """Immutable fact semantic preparation (A5B plan + blocks + requests + hashes).

    Carries the A5B planning result, the verified consolidation / semantic
    profile identities, the prompt / schema identities, the explicit packing
    policy, the deterministic fact blocks, the rendered
    ``StructuredGenerationRequest`` objects, and the stable per-request hashes.
    No provider is called and nothing is persisted.
    """

    planning_result: ConsolidationPlanningResult
    consolidation_profile: ConsolidationProfile
    semantic_profile: SemanticLLMProfile
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    output_schema_id: str
    output_schema_version: int
    output_schema_hash: str
    working_language: str
    packing_policy: FactSemanticPackingPolicy
    blocks: tuple[FactSemanticBlock, ...]
    structured_requests: tuple[StructuredGenerationRequest, ...]
    semantic_request_hashes: tuple[str, ...]
    semantic_pair_count: int
    auto_same_pair_count: int
    total_fact_pair_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.packing_policy, FactSemanticPackingPolicy):
            raise StoryIntegrityError("packing_policy must be a FactSemanticPackingPolicy")
        if len(self.blocks) != len(self.structured_requests):
            raise StoryIntegrityError(
                "each fact block must have exactly one structured request"
            )
        if len(self.blocks) != len(self.semantic_request_hashes):
            raise StoryIntegrityError(
                "each fact block must have exactly one semantic request hash"
            )
        expected_hash = tuple(request.request_hash for request in self.structured_requests)
        if self.semantic_request_hashes != expected_hash:
            raise StoryIntegrityError(
                "semantic_request_hashes must match the structured request hashes"
            )
        # Block ordinals must be the exact contiguous 0..n-1 sequence.
        if [block.block_ordinal for block in self.blocks] != list(range(len(self.blocks))):
            raise StoryIntegrityError("block_ordinal must be the contiguous 0..n-1 sequence")
        for block in self.blocks:
            if block.pair_count > self.packing_policy.max_pairs_per_block:
                raise StoryIntegrityError(
                    "block pair count exceeds packing_policy.max_pairs_per_block"
                )
            if len(block.candidate_refs) > self.packing_policy.max_candidates_per_block:
                raise StoryIntegrityError(
                    "block unique-candidate count exceeds packing_policy.max_candidates_per_block"
                )
        total_pairs = sum(block.pair_count for block in self.blocks)
        if total_pairs != self.semantic_pair_count:
            raise StoryIntegrityError(
                "block pair counts must sum to the semantic pair count"
            )


def load_fact_semantic_profile() -> SemanticLLMProfile:
    """Load the tracked ``consolidation-llm-v1`` semantic profile."""
    return load_semantic_profile(A5C_FACT_SEMANTIC_PROFILE_PATH)


def load_fact_output_schema() -> OutputSchema:
    """Load the tracked A5 fact selector-payload output schema."""
    try:
        schema_data = load_json(A5C_FACT_OUTPUT_SCHEMA_PATH)
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(f"failed to load fact output schema: {exc}") from exc
    if not isinstance(schema_data, dict):
        raise StoryIntegrityError("fact output schema must be a JSON object")
    return OutputSchema.create(
        schema_id=A5C_FACT_OUTPUT_SCHEMA_ID,
        schema_version=A5C_FACT_OUTPUT_SCHEMA_VERSION,
        schema=schema_data,
    )


def _verify_fact_profile(
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
) -> None:
    """Fail closed unless the fact pass pins the exact A5C-A identity (BLOCK 4/7)."""
    if not isinstance(consolidation_profile, ConsolidationProfile):
        raise StoryIntegrityError("consolidation_profile must be a ConsolidationProfile")
    if not isinstance(semantic_profile, SemanticLLMProfile):
        raise StoryIntegrityError("semantic_profile must be a SemanticLLMProfile")
    # BLOCK 7: the fact generation budget is a frozen A5C-A requirement.
    if consolidation_profile.max_generation_rounds != A5C_FACT_MAX_GENERATION_ROUNDS:
        raise StoryIntegrityError(
            "consolidation max_generation_rounds must be "
            f"{A5C_FACT_MAX_GENERATION_ROUNDS}, got {consolidation_profile.max_generation_rounds}"
        )
    fact: ConsolidationSemanticPass = consolidation_profile.fact
    if fact.semantic_profile_id != A5C_FACT_SEMANTIC_PROFILE_ID:
        raise StoryIntegrityError(
            "consolidation fact.semantic_profile_id must be "
            f"{A5C_FACT_SEMANTIC_PROFILE_ID!r}, got {fact.semantic_profile_id!r}"
        )
    if fact.prompt_id != A5C_FACT_PROMPT_ID or fact.prompt_version != A5C_FACT_PROMPT_VERSION:
        raise StoryIntegrityError(
            "consolidation fact prompt must be "
            f"{A5C_FACT_PROMPT_ID!r} v{A5C_FACT_PROMPT_VERSION}, "
            f"got {fact.prompt_id!r} v{fact.prompt_version}"
        )
    if (
        fact.output_schema_id != A5C_FACT_OUTPUT_SCHEMA_ID
        or fact.output_schema_version != A5C_FACT_OUTPUT_SCHEMA_VERSION
    ):
        raise StoryIntegrityError(
            "consolidation fact output schema must be "
            f"{A5C_FACT_OUTPUT_SCHEMA_ID!r} v{A5C_FACT_OUTPUT_SCHEMA_VERSION}, "
            f"got {fact.output_schema_id!r} v{fact.output_schema_version}"
        )
    if semantic_profile.profile_id != A5C_FACT_SEMANTIC_PROFILE_ID:
        raise StoryIntegrityError(
            "semantic profile id must be "
            f"{A5C_FACT_SEMANTIC_PROFILE_ID!r}, got {semantic_profile.profile_id!r}"
        )


# ---------------------------------------------------------------------------
# Pair-local endpoint packets (A5C-A BLOCK 5 / BLOCK 6)
# ---------------------------------------------------------------------------


def build_fact_endpoint_packet(cand: IndexedFactCandidate, side: str) -> dict[str, Any]:
    """Build the pair-local endpoint packet for one side of a fact pair.

    ``side`` is ``"left"`` or ``"right"``. The evidence items are labeled with
    pair-local selectors ``L0/L1/...`` (left) or ``R0/R1/...`` (right) in the
    EXACT indexed A5B evidence order (``cand.evidence_refs[0] -> L0/R0``,
    ``cand.evidence_refs[1] -> L1/R1``, ...) -- A5C does NOT re-sort evidence.
    The packet carries the exact ``evidence_strength`` and ``source_order_key``
    from the indexed candidate.
    """
    if side not in ("left", "right"):
        raise StoryIntegrityError("side must be 'left' or 'right'")
    prefix = "L" if side == "left" else "R"
    evidence = [
        {
            "selector": f"{prefix}{idx}",
            "paragraph_id": ev.paragraph_id,
            "role": ev.role,
            "strength": ev.strength,
            "excerpt": ev.excerpt,
        }
        for idx, ev in enumerate(cand.evidence_refs)
    ]
    return {
        "candidate_ref": cand.global_candidate_ref,
        "chunk_id": cand.chunk_id,
        "local_candidate_id": cand.local_candidate_id,
        "fact_type": cand.fact_type,
        "statement_zh": cand.statement_zh,
        "subject_refs": list(cand.subject_refs),
        "object_refs": list(cand.object_refs),
        "evidence_strength": cand.evidence_strength,
        "source_order_key": cand.source_order_key,
        "evidence": evidence,
    }


def build_fact_pair_context(
    plan: FactPairPlan, facts_by_ref: Mapping[str, IndexedFactCandidate]
) -> dict[str, Any]:
    """Build one pair context (left/right refs + signals + endpoint packets)."""
    left = facts_by_ref.get(plan.left_ref)
    right = facts_by_ref.get(plan.right_ref)
    if left is None or right is None:
        raise StoryIntegrityError(
            "fact pair plan references a candidate missing from the index"
        )
    return {
        "left_candidate_ref": plan.left_ref,
        "right_candidate_ref": plan.right_ref,
        "signals": list(plan.signals),
        "left": build_fact_endpoint_packet(left, "left"),
        "right": build_fact_endpoint_packet(right, "right"),
    }


# ---------------------------------------------------------------------------
# Semantic stream validation (A5C-A BLOCK 8)
# ---------------------------------------------------------------------------


def _fact_ref_namespace(ref: str) -> str:
    try:
        return ConsolidationCandidateRef.parse(ref).namespace
    except ConsolidationModelError as exc:
        raise StoryIntegrityError(f"invalid consolidation candidate ref {ref!r}: {exc}") from exc


def _validate_fact_semantic_stream(
    planning: ConsolidationPlanningResult,
) -> tuple[FactPairPlan, ...]:
    """Validate the fact semantic stream fail closed and return it in order.

    Enforces: every fact pair plan is in the fact namespace with
    ``left_ref < right_ref``; no duplicate pair; the plans are in canonical
    ``(left_ref, right_ref)`` order (verified, not silently re-sorted); the
    semantic stream is EXACTLY the ``needs_semantic_decision`` fact pairs with
    ``auto_same`` excluded. Any deviation raises :class:`StoryIntegrityError`.
    """
    plans = planning.fact_pair_plans
    seen_pairs: set[tuple[str, str]] = set()
    prev_key: tuple[str, str] | None = None
    semantic_count = 0
    for plan in plans:
        left, right = plan.left_ref, plan.right_ref
        if _fact_ref_namespace(left) != "fact" or _fact_ref_namespace(right) != "fact":
            raise StoryIntegrityError(
                f"fact pair plan is not in the fact namespace: {left!r} <-> {right!r}"
            )
        if not left < right:
            raise StoryIntegrityError(
                f"fact pair plan is malformed (left_ref < right_ref): {left!r} !< {right!r}"
            )
        key = (left, right)
        if key in seen_pairs:
            raise StoryIntegrityError(
                f"duplicate fact pair plan: {left!r} <-> {right!r}"
            )
        seen_pairs.add(key)
        if prev_key is not None and key < prev_key:
            raise StoryIntegrityError(
                f"fact pair plans are not in canonical (left_ref, right_ref) order at "
                f"{left!r} <-> {right!r}"
            )
        prev_key = key
        if plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION:
            semantic_count += 1
        elif plan.state != PAIR_STATE_AUTO_SAME:
            raise StoryIntegrityError(
                f"unexpected fact pair state {plan.state!r} "
                "(must be needs_semantic_decision or auto_same)"
            )
    # The semantic stream is EXACTLY the needs_semantic_decision plans
    # (auto_same excluded), in the canonical order verified above.
    stream = tuple(plan for plan in plans if plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION)
    if len(stream) != semantic_count:
        raise StoryIntegrityError("fact semantic stream coverage mismatch")
    return stream


# ---------------------------------------------------------------------------
# Block packing + identity (A5C-A BLOCK 1 / BLOCK 3 / BLOCK 4)
# ---------------------------------------------------------------------------


def _greedy_pack(
    stream,
    max_pairs_per_block: int,
    max_candidates_per_block: int,
) -> list[list]:
    """Deterministic greedy two-limit packing (domain-agnostic).

    ``stream`` is an iterable of pair plans in canonical
    ``(left_ref, right_ref)`` order (each with ``.left_ref`` / ``.right_ref``).
    A block is closed when appending the next pair would exceed either the pair
    limit or the unique-endpoint limit. No pair is split, lost, duplicated, or
    reordered. Returns the stream partitioned into a list of blocks (each a
    list of pair plans, in the original canonical order).
    """
    blocks: list[list] = []
    current: list = []
    current_endpoints: set[str] = set()
    for plan in stream:
        endpoints = {plan.left_ref, plan.right_ref}
        if current and (
            len(current) + 1 > max_pairs_per_block
            or len(current_endpoints | endpoints) > max_candidates_per_block
        ):
            blocks.append(current)
            current = []
            current_endpoints = set()
        current.append(plan)
        current_endpoints |= endpoints
    if current:
        blocks.append(current)
    return blocks


def _canonical_candidate_refs_ordered(refs: set[str], source_order_key_for_ref) -> tuple[str, ...]:
    """Stable unique union ordered by candidate source_order_key then ref."""
    keyed = [(source_order_key_for_ref(ref), ref) for ref in refs]
    keyed.sort()
    return tuple(ref for _so, ref in keyed)


def _compute_semantic_block_id(
    *,
    plan_hash: str,
    block_prefix: str,
    packing_policy_material: Mapping[str, Any],
    block_ordinal: int,
    pair_refs: tuple[tuple[str, str], ...],
    candidate_refs: tuple[str, ...],
    domain: str | None = None,
) -> str:
    """Bind a semantic block id to the frozen material (no runtime data).

    Shared by the A5C fact / A5D event / A5D relationship block identities:
    ``block_prefix`` + first ``A5C_BLOCK_ID_HEX_LENGTH`` hex chars of the
    ``content_hash`` of the canonical material (plan hash, packing-policy
    limits, block ordinal, ordered requested pairs, ordered candidate refs).

    The A5D event / relationship block identities also bind the explicit
    ``domain`` (per the A5D plan BLOCK 12); the A5C fact block identity passes
    no domain (preserving the A5C fact block id byte-for-byte).
    """
    material = {
        "plan_hash": plan_hash,
        "packing_policy": dict(packing_policy_material),
        "block_ordinal": block_ordinal,
        "ordered_requested_pairs": [list(pair) for pair in pair_refs],
        "ordered_candidate_refs": list(candidate_refs),
    }
    if domain is not None:
        material["domain"] = domain
    return block_prefix + content_hash(material)[:A5C_BLOCK_ID_HEX_LENGTH]


def _indexed_source_order_key(by_ref: Mapping, domain: str):
    """A ``source_order_key`` lookup that fails closed on a missing ref."""

    def _so(ref: str) -> str:
        cand = by_ref.get(ref)
        if cand is None:
            raise StoryIntegrityError(
                f"candidate ref {ref!r} is not in the {domain} candidate index"
            )
        return cand.source_order_key

    return _so


def _build_semantic_blocks(
    *,
    block_cls: type,
    block_prefix: str,
    packing_policy_material: Mapping[str, Any],
    plan_hash: str,
    max_pairs_per_block: int,
    max_candidates_per_block: int,
    semantic_stream,
    source_order_key_for_ref,
    pair_context_builder,
    domain: str | None = None,
) -> tuple:
    """Pack the semantic stream under BOTH limits into domain-specific blocks.

    Shared by the A5C fact / A5D event / A5D relationship preparation builders:
    greedy two-limit packing (``_greedy_pack``), stable candidate refs, the
    pair-local contexts, the deterministic block id, and the canonical
    ``pair_contexts_json``. ``block_cls`` is the domain block dataclass (the
    fact / event / relationship blocks share the identical field layout).
    """
    blocks: list = []
    for group in _greedy_pack(
        semantic_stream, max_pairs_per_block, max_candidates_per_block
    ):
        ordinal = len(blocks)
        pair_refs = tuple((p.left_ref, p.right_ref) for p in group)
        endpoints: set[str] = set()
        for p in group:
            endpoints.add(p.left_ref)
            endpoints.add(p.right_ref)
        candidate_refs = _canonical_candidate_refs_ordered(
            endpoints, source_order_key_for_ref
        )
        pair_contexts = tuple(pair_context_builder(p) for p in group)
        blocks.append(
            block_cls(
                block_ordinal=ordinal,
                block_id=_compute_semantic_block_id(
                    plan_hash=plan_hash,
                    block_prefix=block_prefix,
                    packing_policy_material=packing_policy_material,
                    block_ordinal=ordinal,
                    pair_refs=pair_refs,
                    candidate_refs=candidate_refs,
                    domain=domain,
                ),
                pair_contexts=pair_contexts,
                candidate_refs=candidate_refs,
                pair_refs=pair_refs,
                pair_contexts_json=canonical_json_bytes(
                    [dict(pc) for pc in pair_contexts]
                ).decode("utf-8"),
            )
        )
    return tuple(blocks)


def _compute_block_id(
    plan_hash: str,
    policy: FactSemanticPackingPolicy,
    block_ordinal: int,
    pair_refs: tuple[tuple[str, str], ...],
    candidate_refs: tuple[str, ...],
) -> str:
    """A5C-A BLOCK 4: bind the fact block id to the frozen material."""
    return _compute_semantic_block_id(
        plan_hash=plan_hash,
        block_prefix=A5C_BLOCK_PREFIX,
        packing_policy_material=policy.to_dict(),
        block_ordinal=block_ordinal,
        pair_refs=pair_refs,
        candidate_refs=candidate_refs,
    )


def _build_fact_blocks(
    policy: FactSemanticPackingPolicy,
    plan_hash: str,
    facts_by_ref: Mapping[str, IndexedFactCandidate],
    semantic_stream: tuple[FactPairPlan, ...],
) -> tuple[FactSemanticBlock, ...]:
    """Deterministically pack the fact semantic stream under BOTH policy limits.

    The stream is consumed in its canonical ``(left_ref, right_ref)`` order. A
    block is closed when appending the next pair would exceed either the pair
    limit or the unique-candidate limit. No pair is split, lost, duplicated, or
    reordered. (Shared two-limit packing: :func:`_build_semantic_blocks`.)
    """
    return _build_semantic_blocks(
        block_cls=FactSemanticBlock,
        block_prefix=A5C_BLOCK_PREFIX,
        packing_policy_material=policy.to_dict(),
        plan_hash=plan_hash,
        max_pairs_per_block=policy.max_pairs_per_block,
        max_candidates_per_block=policy.max_candidates_per_block,
        semantic_stream=semantic_stream,
        source_order_key_for_ref=_indexed_source_order_key(facts_by_ref, "fact"),
        pair_context_builder=lambda plan: build_fact_pair_context(plan, facts_by_ref),
    )


# ---------------------------------------------------------------------------
# Fact semantic preparation builder (A5C-A BLOCK 2)
# ---------------------------------------------------------------------------


def build_fact_semantic_preparation(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
    packing_policy: FactSemanticPackingPolicy,
) -> FactSemanticPreparation:
    """Build the fact semantic preparation from an A5B planning result.

    Selects the ``needs_semantic_decision`` fact pairs (the A5C fact semantic
    stream), validates the stream fail closed, builds the deterministic
    pair-local pair contexts, packs them into blocks under the explicit
    ``packing_policy`` (BOTH the pair and unique-candidate limits), and renders
    one real ``StructuredGenerationRequest`` per block. ``packing_policy`` is
    required (no production default: ``fact-semantic-packing-v1`` is not yet
    frozen). No provider is called and nothing is persisted.
    """
    if not isinstance(packing_policy, FactSemanticPackingPolicy):
        raise StoryIntegrityError(
            "packing_policy is required (fact-semantic-packing-v1 is not yet frozen)"
        )
    _verify_fact_profile(consolidation_profile, semantic_profile)
    prompt = prompts.load(A5C_FACT_PROMPT_ID, version=A5C_FACT_PROMPT_VERSION)
    output_schema = load_fact_output_schema()

    facts_by_ref = {
        cand.global_candidate_ref: cand for cand in planning_result.index.facts
    }
    fact_plans = planning_result.fact_pair_plans
    semantic_stream = _validate_fact_semantic_stream(planning_result)
    auto_same_count = sum(
        1 for plan in fact_plans if plan.state == PAIR_STATE_AUTO_SAME
    )
    plan_hash = planning_result.plan_hash

    blocks = _build_fact_blocks(packing_policy, plan_hash, facts_by_ref, semantic_stream)

    structured_requests: list[StructuredGenerationRequest] = []
    for block in blocks:
        variables = {
            "block_id": block.block_id,
            "pair_contexts_json": block.pair_contexts_json,
        }
        rendered = render_prompt(prompt, variables)
        structured_requests.append(
            build_structured_request(
                rendered_prompt=rendered,
                output_schema=output_schema,
                semantic_profile=semantic_profile,
            )
        )
    request_hashes = tuple(request.request_hash for request in structured_requests)

    return FactSemanticPreparation(
        planning_result=planning_result,
        consolidation_profile=consolidation_profile,
        semantic_profile=semantic_profile,
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.version,
        prompt_content_hash=prompt.content_hash,
        output_schema_id=output_schema.schema_id,
        output_schema_version=output_schema.schema_version,
        output_schema_hash=output_schema.schema_hash,
        working_language=consolidation_profile.working_language,
        packing_policy=packing_policy,
        blocks=blocks,
        structured_requests=tuple(structured_requests),
        semantic_request_hashes=request_hashes,
        semantic_pair_count=len(semantic_stream),
        auto_same_pair_count=auto_same_count,
        total_fact_pair_count=len(fact_plans),
    )


# ---------------------------------------------------------------------------
# Fact semantic identity (A5C-A BLOCK 4 / BLOCK 11)
# ---------------------------------------------------------------------------


def build_fact_semantic_identity(prep: FactSemanticPreparation) -> dict[str, Any]:
    """Build the canonical A5C fact semantic identity material.

    Pins the A5B planning identity (policy ids + plan hash), the verified
    consolidation / semantic profile identities, the prompt / schema identities,
    the explicit packing policy limits, and the exact request set (block ids +
    ordered per-request hashes). No backend routing metadata.
    """
    planning = prep.planning_result
    return {
        "schema_version": 1,
        "planning_policy_id": planning.planning_policy_id,
        "blocking_policy_id": planning.blocking_policy_id,
        "text_normalization_policy_id": planning.text_normalization_policy_id,
        "exact_safe_policy_id": planning.exact_safe_policy_id,
        "plan_hash": planning.plan_hash,
        "profile_id": prep.consolidation_profile.profile_id,
        "profile_hash": prep.consolidation_profile.content_hash(),
        "working_language": prep.working_language,
        "semantic_profile_id": prep.semantic_profile.profile_id,
        "semantic_profile_hash": prep.semantic_profile.semantic_profile_hash,
        "prompt_id": prep.prompt_id,
        "prompt_version": prep.prompt_version,
        "prompt_content_hash": prep.prompt_content_hash,
        "output_schema_id": prep.output_schema_id,
        "output_schema_version": prep.output_schema_version,
        "output_schema_hash": prep.output_schema_hash,
        "packing_policy": prep.packing_policy.to_dict(),
        "semantic_pair_count": prep.semantic_pair_count,
        "auto_same_pair_count": prep.auto_same_pair_count,
        "total_fact_pair_count": prep.total_fact_pair_count,
        "block_count": len(prep.blocks),
        "block_ids": tuple(block.block_id for block in prep.blocks),
        "semantic_request_hashes": tuple(prep.semantic_request_hashes),
    }


# ---------------------------------------------------------------------------
# Deterministic block-packing audit (A5C-A BLOCK 5 / BLOCK 9 / BLOCK 10)
# ---------------------------------------------------------------------------


def _percentile_nearest_rank(values: Sequence[float], q: float) -> float:
    """Deterministic nearest-rank percentile: index = ceil(q * n) - 1.

    q in [0, 1]: q=0 -> min, q=1 -> max, q=0.5 -> (upper) median, q=0.95 -> p95.
    """
    if not values:
        raise StoryIntegrityError("percentile of an empty sequence is undefined")
    if not 0.0 <= q <= 1.0:
        raise StoryIntegrityError(f"percentile q must be in [0, 1], got {q!r}")
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered))
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


def _distribution(values: Sequence[float]) -> dict[str, float]:
    """Deterministic min / median / p95 / max distribution (nearest-rank)."""
    return {
        "min": _percentile_nearest_rank(values, 0.0),
        "median": _percentile_nearest_rank(values, 0.5),
        "p95": _percentile_nearest_rank(values, 0.95),
        "max": _percentile_nearest_rank(values, 1.0),
    }


def _block_evidence_item_count(block: FactSemanticBlock) -> int:
    """Total source-anchored evidence items in the block (summed over pairs)."""
    total = 0
    for pc in block.pair_contexts:
        total += len(pc["left"]["evidence"]) + len(pc["right"]["evidence"])
    return total


def _request_prompt_bytes(request: StructuredGenerationRequest) -> int:
    """Actual rendered prompt bytes (system + user), per BLOCK 9."""
    return (
        len(request.rendered_prompt.system_text.encode("utf-8"))
        + len(request.rendered_prompt.user_text.encode("utf-8"))
    )


def _select_largest_block(prep: FactSemanticPreparation) -> dict[str, Any] | None:
    """Deterministic largest-block diagnostics (BLOCK 10).

    Largest by ``rendered_prompt_bytes``; ties broken by the smallest
    ``block_id`` (documented deterministic tie-break).
    """
    if not prep.blocks:
        return None
    prompt_bytes = {
        block.block_id: _request_prompt_bytes(request)
        for block, request in zip(prep.blocks, prep.structured_requests)
    }
    max_bytes = max(prompt_bytes.values())
    largest = min(
        (block for block in prep.blocks if prompt_bytes[block.block_id] == max_bytes),
        key=lambda block: block.block_id,
    )
    request = next(
        r for b, r in zip(prep.blocks, prep.structured_requests) if b.block_id == largest.block_id
    )
    return {
        "block_id": largest.block_id,
        "block_ordinal": largest.block_ordinal,
        "pair_count": largest.pair_count,
        "unique_candidate_count": len(largest.candidate_refs),
        "evidence_item_count": _block_evidence_item_count(largest),
        "pair_contexts_json_bytes": largest.payload_bytes,
        "rendered_prompt_bytes": _request_prompt_bytes(request),
        "first_pair": largest.pair_refs[0] if largest.pair_refs else None,
        "last_pair": largest.pair_refs[-1] if largest.pair_refs else None,
    }


def build_fact_packing_audit(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
    candidates: tuple[FactSemanticPackingPolicy, ...] = A5C_PACKING_CANDIDATES,
) -> tuple[dict[str, Any], ...]:
    """Audit deterministic fact block-packing candidates (zero provider).

    For each ``FactSemanticPackingPolicy`` candidate, build the fact semantic
    preparation and report the deterministic distribution statistics (BLOCK 9)
    and largest-block diagnostics (BLOCK 10). The candidates do NOT freeze the
    production default.
    """
    audit: list[dict[str, Any]] = []
    for policy in candidates:
        prep = build_fact_semantic_preparation(
            planning_result,
            consolidation_profile,
            semantic_profile,
            prompts=prompts,
            packing_policy=policy,
        )
        blocks = prep.blocks
        requests = prep.structured_requests
        hashes = prep.semantic_request_hashes
        audit.append(
            {
                "packing_name": policy.name,
                "max_pairs_per_block": policy.max_pairs_per_block,
                "max_candidates_per_block": policy.max_candidates_per_block,
                "block_count": len(blocks),
                "total_pairs_in_blocks": sum(b.pair_count for b in blocks),
                "pairs_per_block": _distribution([b.pair_count for b in blocks]),
                "unique_candidates_per_block": _distribution(
                    [len(b.candidate_refs) for b in blocks]
                ),
                "evidence_items_per_block": _distribution(
                    [_block_evidence_item_count(b) for b in blocks]
                ),
                "pair_contexts_json_bytes": _distribution(
                    [b.payload_bytes for b in blocks]
                ),
                "rendered_prompt_bytes": _distribution(
                    [_request_prompt_bytes(r) for r in requests]
                ),
                "request_hash_count": len(hashes),
                "unique_request_hash_count": len(set(hashes)),
                "largest_block": _select_largest_block(prep),
            }
        )
    return tuple(audit)


# ---------------------------------------------------------------------------
# A5D-A -- event + relationship semantic preparation (zero provider)
#
# The event / relationship equivalents of the A5C-A fact preparation. They
# reuse the EXACT A5C-A packing algorithm (``_build_semantic_blocks`` /
# ``_greedy_pack``) and the same pair-local endpoint-packet + request-rendering
# pattern, with domain-specific endpoint fields, pair-context signals, block-id
# prefixes, and the tracked event / relationship prompt + schema identity.
#
# As with A5C-A: NO provider call, NO persistence. Only the zero-provider
# preparation + packing audit + real ``StructuredGenerationRequest`` rendering
# is implemented. Response parsing, selector resolution, the ``method=llm``
# decision model, persistence, CURRENT, and A5D-B are all OUT OF SCOPE.
# ---------------------------------------------------------------------------


# --- Frozen event / relationship semantic identity (BLOCK 3 / BLOCK 4) ---

#: Tracked A5 event consolidation prompt identity.
A5D_EVENT_PROMPT_ID = "a5.event-consolidation"
A5D_EVENT_PROMPT_VERSION = 1

#: Tracked A5 event selector-payload output schema identity.
A5D_EVENT_OUTPUT_SCHEMA_ID = "consolidation-event-selector-payload"
A5D_EVENT_OUTPUT_SCHEMA_VERSION = 1
A5D_EVENT_OUTPUT_SCHEMA_PATH = SCHEMAS_DIR / "consolidation-event-selector-payload.schema.json"

#: Tracked A5 relationship consolidation prompt identity.
A5D_RELATIONSHIP_PROMPT_ID = "a5.relationship-consolidation"
A5D_RELATIONSHIP_PROMPT_VERSION = 1

#: Tracked A5 relationship selector-payload output schema identity.
A5D_RELATIONSHIP_OUTPUT_SCHEMA_ID = "consolidation-relationship-selector-payload"
A5D_RELATIONSHIP_OUTPUT_SCHEMA_VERSION = 1
A5D_RELATIONSHIP_OUTPUT_SCHEMA_PATH = SCHEMAS_DIR / "consolidation-relationship-selector-payload.schema.json"

#: Tracked A5 semantic LLM profile identity (generation semantics only, shared
#: by the fact / event / relationship passes).
A5D_SEMANTIC_PROFILE_ID = "consolidation-llm-v1"
A5D_SEMANTIC_PROFILE_PATH = PROFILES_DIR / "consolidation_llm_v1.yaml"

#: A5 event block id prefix (distinct from fact ``a5fblk_`` / relationship).
A5D_EVENT_BLOCK_PREFIX = "a5eblk_"
#: A5 relationship block id prefix.
A5D_RELATIONSHIP_BLOCK_PREFIX = "a5rblk_"

#: Frozen requirement: the event / relationship semantic pass generation budget
#: (the single ``consolidation_profile.max_generation_rounds`` pin).
A5D_MAX_GENERATION_ROUNDS = 2

#: The exact A5D event semantic endpoint-packet fields (BLOCK 7).
A5D_EVENT_ENDPOINT_PACKET_FIELDS = (
    "candidate_ref",
    "chunk_id",
    "local_candidate_id",
    "summary_zh",
    "participants",
    "locations",
    "temporal_mode",
    "evidence_strength",
    "source_order_key",
    "evidence",
)

#: The exact A5D relationship semantic endpoint-packet fields (BLOCK 8).
A5D_RELATIONSHIP_ENDPOINT_PACKET_FIELDS = (
    "candidate_ref",
    "chunk_id",
    "local_candidate_id",
    "source_entity_ref",
    "target_entity_ref",
    "relationship_type_zh",
    "state_zh",
    "direction",
    "evidence_strength",
    "source_order_key",
    "evidence",
)


# ---------------------------------------------------------------------------
# Packing policy (event / relationship -- audit-only, NOT frozen)
# ---------------------------------------------------------------------------


def _validate_semantic_packing_policy(name: str, max_pairs: int, max_candidates: int) -> None:
    """Shared validation for the event / relationship packing policies."""
    if not isinstance(name, str) or not name:
        raise StoryIntegrityError("packing policy name must be a non-empty string")
    if not isinstance(max_pairs, int) or max_pairs < 1:
        raise StoryIntegrityError("max_pairs_per_block must be a positive integer")
    # A single pair has two distinct endpoints, so the candidate limit must be
    # able to hold at least one pair.
    if not isinstance(max_candidates, int) or max_candidates < 2:
        raise StoryIntegrityError(
            "max_candidates_per_block must be an integer >= 2 (one pair has two endpoints)"
        )


@dataclass(frozen=True, slots=True)
class EventSemanticPackingPolicy:
    """An explicit A5D event packing policy with TWO independent limits.

    ``event-semantic-packing-v1`` is NOT YET FROZEN. The audit candidates below
    are audit-only; A5D-A always requires an explicit policy (there is no
    production default), and a later slice consumes the policy selected after
    architecture review.
    """

    name: str
    max_pairs_per_block: int
    max_candidates_per_block: int

    def __post_init__(self) -> None:
        _validate_semantic_packing_policy(
            self.name, self.max_pairs_per_block, self.max_candidates_per_block
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_pairs_per_block": self.max_pairs_per_block,
            "max_candidates_per_block": self.max_candidates_per_block,
        }


@dataclass(frozen=True, slots=True)
class RelationshipSemanticPackingPolicy:
    """An explicit A5D relationship packing policy with TWO independent limits.

    ``relationship-semantic-packing-v1`` is NOT YET FROZEN. The audit
    candidates below are audit-only; A5D-A always requires an explicit policy
    (there is no production default).
    """

    name: str
    max_pairs_per_block: int
    max_candidates_per_block: int

    def __post_init__(self) -> None:
        _validate_semantic_packing_policy(
            self.name, self.max_pairs_per_block, self.max_candidates_per_block
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_pairs_per_block": self.max_pairs_per_block,
            "max_candidates_per_block": self.max_candidates_per_block,
        }


#: The three deterministic event block-packing candidates audited by A5D-A
#: (BLOCK 10): P1 = 6 pairs / 12 candidates, P2 = 12 / 24, P3 = 24 / 48.
#: Audit-only (NOT a production default).
A5D_EVENT_PACKING_CANDIDATES: tuple[EventSemanticPackingPolicy, ...] = (
    EventSemanticPackingPolicy("P1", 6, 12),
    EventSemanticPackingPolicy("P2", 12, 24),
    EventSemanticPackingPolicy("P3", 24, 48),
)

#: The three deterministic relationship block-packing candidates audited by
#: A5D-A (BLOCK 11): P1 = 6 pairs / 12 candidates, P2 = 12 / 24, P3 = 24 / 48.
#: Audit-only (NOT a production default).
A5D_RELATIONSHIP_PACKING_CANDIDATES: tuple[RelationshipSemanticPackingPolicy, ...] = (
    RelationshipSemanticPackingPolicy("P1", 6, 12),
    RelationshipSemanticPackingPolicy("P2", 12, 24),
    RelationshipSemanticPackingPolicy("P3", 24, 48),
)


# ---------------------------------------------------------------------------
# Event / relationship semantic block (BLOCK 7 / BLOCK 8)
# ---------------------------------------------------------------------------


def _validate_semantic_block_shape(
    block_ordinal: int, block_id: str, prefix: str, domain: str
) -> None:
    """Shared structural validation for the event / relationship blocks."""
    if not isinstance(block_ordinal, int) or block_ordinal < 0:
        raise StoryIntegrityError(f"{domain} block_ordinal must be a non-negative integer")
    if not block_id.startswith(prefix):
        raise StoryIntegrityError(
            f"{domain} block_id must start with {prefix!r}: {block_id!r}"
        )
    hex_part = block_id[len(prefix):]
    if len(hex_part) != A5C_BLOCK_ID_HEX_LENGTH or any(
        c not in "0123456789abcdef" for c in hex_part
    ):
        raise StoryIntegrityError(
            f"{domain} block_id must be {prefix!r} + {A5C_BLOCK_ID_HEX_LENGTH} "
            f"lowercase hex chars: {block_id!r}"
        )


def _validate_semantic_block_payload(
    pair_contexts: tuple[dict[str, Any], ...],
    candidate_refs: tuple[str, ...],
    pair_refs: tuple[tuple[str, str], ...],
    domain: str,
) -> None:
    """Shared payload validation for the event / relationship blocks.

    ``pair_refs`` must equal the pair contexts' (left, right) refs and
    ``candidate_refs`` must equal the exact stable union of the block's pair
    endpoints, ordered by candidate source_order_key then ref (never a
    block-wide semantic or evidence pool).
    """
    if tuple(pair_refs) != tuple(
        (pc["left_candidate_ref"], pc["right_candidate_ref"]) for pc in pair_contexts
    ):
        raise StoryIntegrityError(f"{domain} block pair_refs does not match the pair contexts")
    keyed: list[tuple[str, str]] = []
    for pc in pair_contexts:
        for side in ("left", "right"):
            packet = pc[side]
            keyed.append((packet["source_order_key"], packet["candidate_ref"]))
    expected_refs = tuple(ref for _so, ref in sorted(set(keyed)))
    if tuple(candidate_refs) != expected_refs:
        raise StoryIntegrityError(
            f"{domain} candidate_refs must equal the stable union of the block pair "
            "endpoints (ordered by candidate source_order_key then ref)"
        )


@dataclass(frozen=True, slots=True)
class EventSemanticBlock:
    """One deterministic event consolidation block (pair-local contexts).

    Mirrors :class:`FactSemanticBlock` (identical field layout) with the event
    block id prefix ``a5eblk_`` and the event pair-local endpoint packets.
    """

    block_ordinal: int
    block_id: str
    pair_contexts: tuple[dict[str, Any], ...]
    candidate_refs: tuple[str, ...]
    pair_refs: tuple[tuple[str, str], ...]
    pair_contexts_json: str

    def __post_init__(self) -> None:
        _validate_semantic_block_shape(
            self.block_ordinal, self.block_id, A5D_EVENT_BLOCK_PREFIX, "event"
        )
        _validate_semantic_block_payload(
            self.pair_contexts, self.candidate_refs, self.pair_refs, "event"
        )

    @property
    def pair_count(self) -> int:
        return len(self.pair_contexts)

    @property
    def payload_bytes(self) -> int:
        return len(self.pair_contexts_json.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class RelationshipSemanticBlock:
    """One deterministic relationship consolidation block (pair-local contexts).

    Mirrors :class:`FactSemanticBlock` (identical field layout) with the
    relationship block id prefix ``a5rblk_`` and the relationship pair-local
    endpoint packets.
    """

    block_ordinal: int
    block_id: str
    pair_contexts: tuple[dict[str, Any], ...]
    candidate_refs: tuple[str, ...]
    pair_refs: tuple[tuple[str, str], ...]
    pair_contexts_json: str

    def __post_init__(self) -> None:
        _validate_semantic_block_shape(
            self.block_ordinal, self.block_id, A5D_RELATIONSHIP_BLOCK_PREFIX, "relationship"
        )
        _validate_semantic_block_payload(
            self.pair_contexts, self.candidate_refs, self.pair_refs, "relationship"
        )

    @property
    def pair_count(self) -> int:
        return len(self.pair_contexts)

    @property
    def payload_bytes(self) -> int:
        return len(self.pair_contexts_json.encode("utf-8"))


# ---------------------------------------------------------------------------
# Event / relationship semantic preparation result (BLOCK 2 / BLOCK 12 / 13)
# ---------------------------------------------------------------------------


def _validate_semantic_preparation_shape(
    packing_policy: Any,
    expected_policy: type,
    policy_name: str,
    blocks: tuple,
    structured_requests: tuple,
    semantic_request_hashes: tuple,
) -> None:
    """Shared structural validation for the event / relationship preparations."""
    if not isinstance(packing_policy, expected_policy):
        raise StoryIntegrityError(f"packing_policy must be a {policy_name}")
    if len(blocks) != len(structured_requests):
        raise StoryIntegrityError(
            "each block must have exactly one structured request"
        )
    if len(blocks) != len(semantic_request_hashes):
        raise StoryIntegrityError(
            "each block must have exactly one semantic request hash"
        )
    expected_hash = tuple(request.request_hash for request in structured_requests)
    if tuple(semantic_request_hashes) != expected_hash:
        raise StoryIntegrityError(
            "semantic_request_hashes must match the structured request hashes"
        )
    if [block.block_ordinal for block in blocks] != list(range(len(blocks))):
        raise StoryIntegrityError("block_ordinal must be the contiguous 0..n-1 sequence")
    for block in blocks:
        if block.pair_count > packing_policy.max_pairs_per_block:
            raise StoryIntegrityError(
                "block pair count exceeds packing_policy.max_pairs_per_block"
            )
        if len(block.candidate_refs) > packing_policy.max_candidates_per_block:
            raise StoryIntegrityError(
                "block unique-candidate count exceeds packing_policy.max_candidates_per_block"
            )


@dataclass(frozen=True)
class EventSemanticPreparation:
    """Immutable event semantic preparation (A5B plan + blocks + requests + hashes).

    Mirrors :class:`FactSemanticPreparation` for the event domain. No provider
    is called and nothing is persisted.
    """

    planning_result: ConsolidationPlanningResult
    consolidation_profile: ConsolidationProfile
    semantic_profile: SemanticLLMProfile
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    output_schema_id: str
    output_schema_version: int
    output_schema_hash: str
    working_language: str
    packing_policy: EventSemanticPackingPolicy
    blocks: tuple[EventSemanticBlock, ...]
    structured_requests: tuple[StructuredGenerationRequest, ...]
    semantic_request_hashes: tuple[str, ...]
    semantic_pair_count: int
    auto_same_pair_count: int
    total_event_pair_count: int

    def __post_init__(self) -> None:
        _validate_semantic_preparation_shape(
            self.packing_policy,
            EventSemanticPackingPolicy,
            "EventSemanticPackingPolicy",
            self.blocks,
            self.structured_requests,
            self.semantic_request_hashes,
        )
        total_pairs = sum(block.pair_count for block in self.blocks)
        if total_pairs != self.semantic_pair_count:
            raise StoryIntegrityError(
                "block pair counts must sum to the semantic pair count"
            )


@dataclass(frozen=True)
class RelationshipSemanticPreparation:
    """Immutable relationship semantic preparation (A5B plan + blocks + requests + hashes).

    Mirrors :class:`FactSemanticPreparation` for the relationship domain. No
    provider is called and nothing is persisted.
    """

    planning_result: ConsolidationPlanningResult
    consolidation_profile: ConsolidationProfile
    semantic_profile: SemanticLLMProfile
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    output_schema_id: str
    output_schema_version: int
    output_schema_hash: str
    working_language: str
    packing_policy: RelationshipSemanticPackingPolicy
    blocks: tuple[RelationshipSemanticBlock, ...]
    structured_requests: tuple[StructuredGenerationRequest, ...]
    semantic_request_hashes: tuple[str, ...]
    semantic_pair_count: int
    auto_same_pair_count: int
    total_relationship_pair_count: int

    def __post_init__(self) -> None:
        _validate_semantic_preparation_shape(
            self.packing_policy,
            RelationshipSemanticPackingPolicy,
            "RelationshipSemanticPackingPolicy",
            self.blocks,
            self.structured_requests,
            self.semantic_request_hashes,
        )
        total_pairs = sum(block.pair_count for block in self.blocks)
        if total_pairs != self.semantic_pair_count:
            raise StoryIntegrityError(
                "block pair counts must sum to the semantic pair count"
            )


# ---------------------------------------------------------------------------
# Profile / schema loaders + verification (event / relationship)
# ---------------------------------------------------------------------------


def load_event_semantic_profile() -> SemanticLLMProfile:
    """Load the tracked ``consolidation-llm-v1`` semantic profile (event pass)."""
    return load_semantic_profile(A5D_SEMANTIC_PROFILE_PATH)


def load_relationship_semantic_profile() -> SemanticLLMProfile:
    """Load the tracked ``consolidation-llm-v1`` semantic profile (relationship pass)."""
    return load_semantic_profile(A5D_SEMANTIC_PROFILE_PATH)


def _load_selector_output_schema(schema_id: str, schema_version: int, path, domain: str) -> OutputSchema:
    try:
        schema_data = load_json(path)
    except Exception as exc:  # noqa: BLE001
        raise StoryIntegrityError(f"failed to load {domain} output schema: {exc}") from exc
    if not isinstance(schema_data, dict):
        raise StoryIntegrityError(f"{domain} output schema must be a JSON object")
    return OutputSchema.create(
        schema_id=schema_id, schema_version=schema_version, schema=schema_data
    )


def load_event_output_schema() -> OutputSchema:
    """Load the tracked A5 event selector-payload output schema."""
    return _load_selector_output_schema(
        A5D_EVENT_OUTPUT_SCHEMA_ID,
        A5D_EVENT_OUTPUT_SCHEMA_VERSION,
        A5D_EVENT_OUTPUT_SCHEMA_PATH,
        "event",
    )


def load_relationship_output_schema() -> OutputSchema:
    """Load the tracked A5 relationship selector-payload output schema."""
    return _load_selector_output_schema(
        A5D_RELATIONSHIP_OUTPUT_SCHEMA_ID,
        A5D_RELATIONSHIP_OUTPUT_SCHEMA_VERSION,
        A5D_RELATIONSHIP_OUTPUT_SCHEMA_PATH,
        "relationship",
    )


def _verify_semantic_pass_profile(
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    pass_name: str,
    prompt_id: str,
    prompt_version: int,
    output_schema_id: str,
    output_schema_version: int,
) -> None:
    """Fail closed unless the pass pins the exact A5D identity."""
    if not isinstance(consolidation_profile, ConsolidationProfile):
        raise StoryIntegrityError("consolidation_profile must be a ConsolidationProfile")
    if not isinstance(semantic_profile, SemanticLLMProfile):
        raise StoryIntegrityError("semantic_profile must be a SemanticLLMProfile")
    if consolidation_profile.max_generation_rounds != A5D_MAX_GENERATION_ROUNDS:
        raise StoryIntegrityError(
            "consolidation max_generation_rounds must be "
            f"{A5D_MAX_GENERATION_ROUNDS}, got {consolidation_profile.max_generation_rounds}"
        )
    pass_pin: ConsolidationSemanticPass = getattr(consolidation_profile, pass_name)
    if pass_pin.semantic_profile_id != A5D_SEMANTIC_PROFILE_ID:
        raise StoryIntegrityError(
            f"consolidation {pass_name}.semantic_profile_id must be "
            f"{A5D_SEMANTIC_PROFILE_ID!r}, got {pass_pin.semantic_profile_id!r}"
        )
    if pass_pin.prompt_id != prompt_id or pass_pin.prompt_version != prompt_version:
        raise StoryIntegrityError(
            f"consolidation {pass_name} prompt must be {prompt_id!r} v{prompt_version}, "
            f"got {pass_pin.prompt_id!r} v{pass_pin.prompt_version}"
        )
    if (
        pass_pin.output_schema_id != output_schema_id
        or pass_pin.output_schema_version != output_schema_version
    ):
        raise StoryIntegrityError(
            f"consolidation {pass_name} output schema must be "
            f"{output_schema_id!r} v{output_schema_version}, "
            f"got {pass_pin.output_schema_id!r} v{pass_pin.output_schema_version}"
        )
    if semantic_profile.profile_id != A5D_SEMANTIC_PROFILE_ID:
        raise StoryIntegrityError(
            "semantic profile id must be "
            f"{A5D_SEMANTIC_PROFILE_ID!r}, got {semantic_profile.profile_id!r}"
        )


def _verify_event_profile(
    consolidation_profile: ConsolidationProfile, semantic_profile: SemanticLLMProfile
) -> None:
    """Fail closed unless the event pass pins the exact A5D identity."""
    _verify_semantic_pass_profile(
        consolidation_profile,
        semantic_profile,
        pass_name="event",
        prompt_id=A5D_EVENT_PROMPT_ID,
        prompt_version=A5D_EVENT_PROMPT_VERSION,
        output_schema_id=A5D_EVENT_OUTPUT_SCHEMA_ID,
        output_schema_version=A5D_EVENT_OUTPUT_SCHEMA_VERSION,
    )


def _verify_relationship_profile(
    consolidation_profile: ConsolidationProfile, semantic_profile: SemanticLLMProfile
) -> None:
    """Fail closed unless the relationship pass pins the exact A5D identity."""
    _verify_semantic_pass_profile(
        consolidation_profile,
        semantic_profile,
        pass_name="relationship",
        prompt_id=A5D_RELATIONSHIP_PROMPT_ID,
        prompt_version=A5D_RELATIONSHIP_PROMPT_VERSION,
        output_schema_id=A5D_RELATIONSHIP_OUTPUT_SCHEMA_ID,
        output_schema_version=A5D_RELATIONSHIP_OUTPUT_SCHEMA_VERSION,
    )


# ---------------------------------------------------------------------------
# Pair-local endpoint packets (event / relationship -- BLOCK 7 / BLOCK 8)
# ---------------------------------------------------------------------------


def _endpoint_evidence_items(cand, side: str) -> list[dict[str, Any]]:
    """Pair-local evidence items labeled ``L0..`` / ``R0..`` in indexed order."""
    prefix = "L" if side == "left" else "R"
    return [
        {
            "selector": f"{prefix}{idx}",
            "paragraph_id": ev.paragraph_id,
            "role": ev.role,
            "strength": ev.strength,
            "excerpt": ev.excerpt,
        }
        for idx, ev in enumerate(cand.evidence_refs)
    ]


def build_event_endpoint_packet(cand: IndexedEventCandidate, side: str) -> dict[str, Any]:
    """Build the pair-local endpoint packet for one side of an event pair.

    ``side`` is ``"left"`` or ``"right"``. The evidence items are labeled with
    pair-local selectors ``L0/L1/...`` (left) or ``R0/R1/...`` (right) in the
    EXACT indexed A5B evidence order (no re-sort). The packet carries the exact
    ``evidence_strength`` and ``source_order_key`` from the indexed candidate.
    """
    if side not in ("left", "right"):
        raise StoryIntegrityError("side must be 'left' or 'right'")
    return {
        "candidate_ref": cand.global_candidate_ref,
        "chunk_id": cand.chunk_id,
        "local_candidate_id": cand.local_candidate_id,
        "summary_zh": cand.summary_zh,
        "participants": list(cand.participants),
        "locations": list(cand.locations),
        "temporal_mode": cand.temporal_mode,
        "evidence_strength": cand.evidence_strength,
        "source_order_key": cand.source_order_key,
        "evidence": _endpoint_evidence_items(cand, side),
    }


def build_relationship_endpoint_packet(
    cand: IndexedRelationshipCandidate, side: str
) -> dict[str, Any]:
    """Build the pair-local endpoint packet for one side of a relationship pair.

    ``side`` is ``"left"`` or ``"right"``. The evidence items are labeled with
    pair-local selectors ``L0/L1/...`` (left) or ``R0/R1/...`` (right) in the
    EXACT indexed A5B evidence order (no re-sort). The packet carries the exact
    ``evidence_strength`` and ``source_order_key`` from the indexed candidate.
    ``state_zh`` may be ``None`` (carried through exactly as indexed).
    """
    if side not in ("left", "right"):
        raise StoryIntegrityError("side must be 'left' or 'right'")
    return {
        "candidate_ref": cand.global_candidate_ref,
        "chunk_id": cand.chunk_id,
        "local_candidate_id": cand.local_candidate_id,
        "source_entity_ref": cand.source_entity_ref,
        "target_entity_ref": cand.target_entity_ref,
        "relationship_type_zh": cand.relationship_type_zh,
        "state_zh": cand.state_zh,
        "direction": cand.direction,
        "evidence_strength": cand.evidence_strength,
        "source_order_key": cand.source_order_key,
        "evidence": _endpoint_evidence_items(cand, side),
    }


# ---------------------------------------------------------------------------
# Pair contexts (event / relationship)
# ---------------------------------------------------------------------------


def build_event_pair_context(
    plan: EventPairPlan, events_by_ref: Mapping[str, IndexedEventCandidate]
) -> dict[str, Any]:
    """Build one event pair context (left/right refs + signals + endpoint packets)."""
    left = events_by_ref.get(plan.left_ref)
    right = events_by_ref.get(plan.right_ref)
    if left is None or right is None:
        raise StoryIntegrityError(
            "event pair plan references a candidate missing from the index"
        )
    return {
        "left_candidate_ref": plan.left_ref,
        "right_candidate_ref": plan.right_ref,
        "signals": list(plan.signals),
        "left": build_event_endpoint_packet(left, "left"),
        "right": build_event_endpoint_packet(right, "right"),
    }


def build_relationship_pair_context(
    plan: RelationshipPairPlan, relationships_by_ref: Mapping[str, IndexedRelationshipCandidate]
) -> dict[str, Any]:
    """Build one relationship pair context (left/right refs + signals + endpoint packets)."""
    left = relationships_by_ref.get(plan.left_ref)
    right = relationships_by_ref.get(plan.right_ref)
    if left is None or right is None:
        raise StoryIntegrityError(
            "relationship pair plan references a candidate missing from the index"
        )
    return {
        "left_candidate_ref": plan.left_ref,
        "right_candidate_ref": plan.right_ref,
        "signals": list(plan.signals),
        "left": build_relationship_endpoint_packet(left, "left"),
        "right": build_relationship_endpoint_packet(right, "right"),
    }


# ---------------------------------------------------------------------------
# Semantic stream validation (event / relationship)
# ---------------------------------------------------------------------------


def _validate_semantic_pair_stream(plans, namespace: str, domain: str):
    """Validate a domain semantic stream fail closed and return it in order.

    Enforces: every pair plan is in ``namespace`` with ``left_ref < right_ref``;
    no duplicate pair; the plans are in canonical ``(left_ref, right_ref)``
    order (verified, not silently re-sorted); the semantic stream is EXACTLY the
    ``needs_semantic_decision`` pairs with ``auto_same`` excluded. Any deviation
    raises :class:`StoryIntegrityError`.
    """
    seen_pairs: set[tuple[str, str]] = set()
    prev_key: tuple[str, str] | None = None
    semantic_count = 0
    for plan in plans:
        left, right = plan.left_ref, plan.right_ref
        if _fact_ref_namespace(left) != namespace or _fact_ref_namespace(right) != namespace:
            raise StoryIntegrityError(
                f"{domain} pair plan is not in the {namespace} namespace: "
                f"{left!r} <-> {right!r}"
            )
        if not left < right:
            raise StoryIntegrityError(
                f"{domain} pair plan is malformed (left_ref < right_ref): "
                f"{left!r} !< {right!r}"
            )
        key = (left, right)
        if key in seen_pairs:
            raise StoryIntegrityError(f"duplicate {domain} pair plan: {left!r} <-> {right!r}")
        seen_pairs.add(key)
        if prev_key is not None and key < prev_key:
            raise StoryIntegrityError(
                f"{domain} pair plans are not in canonical (left_ref, right_ref) order at "
                f"{left!r} <-> {right!r}"
            )
        prev_key = key
        if plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION:
            semantic_count += 1
        elif plan.state != PAIR_STATE_AUTO_SAME:
            raise StoryIntegrityError(
                f"unexpected {domain} pair state {plan.state!r} "
                "(must be needs_semantic_decision or auto_same)"
            )
    stream = tuple(plan for plan in plans if plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION)
    if len(stream) != semantic_count:
        raise StoryIntegrityError(f"{domain} semantic stream coverage mismatch")
    return stream


def _validate_event_semantic_stream(planning: ConsolidationPlanningResult) -> tuple[EventPairPlan, ...]:
    """Validate the event semantic stream fail closed and return it in order."""
    return _validate_semantic_pair_stream(planning.event_pair_plans, "event", "event")


def _validate_relationship_semantic_stream(
    planning: ConsolidationPlanningResult,
) -> tuple[RelationshipPairPlan, ...]:
    """Validate the relationship semantic stream fail closed and return it in order."""
    return _validate_semantic_pair_stream(
        planning.relationship_pair_plans, "relationship", "relationship"
    )


# ---------------------------------------------------------------------------
# Event / relationship preparation builders (BLOCK 2)
# ---------------------------------------------------------------------------


def build_event_semantic_preparation(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
    packing_policy: EventSemanticPackingPolicy,
) -> EventSemanticPreparation:
    """Build the event semantic preparation from an A5B planning result.

    Selects the ``needs_semantic_decision`` event pairs (the A5D event semantic
    stream), validates the stream fail closed, builds the deterministic
    pair-local pair contexts, packs them into blocks under the explicit
    ``packing_policy`` (BOTH the pair and unique-candidate limits), and renders
    one real ``StructuredGenerationRequest`` per block. ``packing_policy`` is
    required (no production default). No provider is called and nothing is
    persisted.
    """
    if not isinstance(packing_policy, EventSemanticPackingPolicy):
        raise StoryIntegrityError(
            "packing_policy is required (event-semantic-packing is not yet frozen)"
        )
    _verify_event_profile(consolidation_profile, semantic_profile)
    prompt = prompts.load(A5D_EVENT_PROMPT_ID, version=A5D_EVENT_PROMPT_VERSION)
    output_schema = load_event_output_schema()

    events_by_ref = {
        cand.global_candidate_ref: cand for cand in planning_result.index.events
    }
    event_plans = planning_result.event_pair_plans
    semantic_stream = _validate_event_semantic_stream(planning_result)
    auto_same_count = sum(1 for plan in event_plans if plan.state == PAIR_STATE_AUTO_SAME)
    plan_hash = planning_result.plan_hash

    blocks = _build_semantic_blocks(
        block_cls=EventSemanticBlock,
        block_prefix=A5D_EVENT_BLOCK_PREFIX,
        packing_policy_material=packing_policy.to_dict(),
        plan_hash=plan_hash,
        max_pairs_per_block=packing_policy.max_pairs_per_block,
        max_candidates_per_block=packing_policy.max_candidates_per_block,
        semantic_stream=semantic_stream,
        source_order_key_for_ref=_indexed_source_order_key(events_by_ref, "event"),
        pair_context_builder=lambda plan: build_event_pair_context(plan, events_by_ref),
        domain="event",
    )

    structured_requests: list[StructuredGenerationRequest] = []
    for block in blocks:
        variables = {
            "block_id": block.block_id,
            "pair_contexts_json": block.pair_contexts_json,
        }
        rendered = render_prompt(prompt, variables)
        structured_requests.append(
            build_structured_request(
                rendered_prompt=rendered,
                output_schema=output_schema,
                semantic_profile=semantic_profile,
            )
        )
    request_hashes = tuple(request.request_hash for request in structured_requests)

    return EventSemanticPreparation(
        planning_result=planning_result,
        consolidation_profile=consolidation_profile,
        semantic_profile=semantic_profile,
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.version,
        prompt_content_hash=prompt.content_hash,
        output_schema_id=output_schema.schema_id,
        output_schema_version=output_schema.schema_version,
        output_schema_hash=output_schema.schema_hash,
        working_language=consolidation_profile.working_language,
        packing_policy=packing_policy,
        blocks=blocks,
        structured_requests=tuple(structured_requests),
        semantic_request_hashes=request_hashes,
        semantic_pair_count=len(semantic_stream),
        auto_same_pair_count=auto_same_count,
        total_event_pair_count=len(event_plans),
    )


def build_relationship_semantic_preparation(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
    packing_policy: RelationshipSemanticPackingPolicy,
) -> RelationshipSemanticPreparation:
    """Build the relationship semantic preparation from an A5B planning result.

    Selects the ``needs_semantic_decision`` relationship pairs (the A5D
    relationship semantic stream), validates the stream fail closed, builds the
    deterministic pair-local pair contexts, packs them into blocks under the
    explicit ``packing_policy`` (BOTH the pair and unique-candidate limits), and
    renders one real ``StructuredGenerationRequest`` per block. ``packing_policy``
    is required (no production default). No provider is called and nothing is
    persisted.
    """
    if not isinstance(packing_policy, RelationshipSemanticPackingPolicy):
        raise StoryIntegrityError(
            "packing_policy is required (relationship-semantic-packing is not yet frozen)"
        )
    _verify_relationship_profile(consolidation_profile, semantic_profile)
    prompt = prompts.load(A5D_RELATIONSHIP_PROMPT_ID, version=A5D_RELATIONSHIP_PROMPT_VERSION)
    output_schema = load_relationship_output_schema()

    relationships_by_ref = {
        cand.global_candidate_ref: cand for cand in planning_result.index.relationships
    }
    relationship_plans = planning_result.relationship_pair_plans
    semantic_stream = _validate_relationship_semantic_stream(planning_result)
    auto_same_count = sum(
        1 for plan in relationship_plans if plan.state == PAIR_STATE_AUTO_SAME
    )
    plan_hash = planning_result.plan_hash

    blocks = _build_semantic_blocks(
        block_cls=RelationshipSemanticBlock,
        block_prefix=A5D_RELATIONSHIP_BLOCK_PREFIX,
        packing_policy_material=packing_policy.to_dict(),
        plan_hash=plan_hash,
        max_pairs_per_block=packing_policy.max_pairs_per_block,
        max_candidates_per_block=packing_policy.max_candidates_per_block,
        semantic_stream=semantic_stream,
        source_order_key_for_ref=_indexed_source_order_key(
            relationships_by_ref, "relationship"
        ),
        pair_context_builder=lambda plan: build_relationship_pair_context(
            plan, relationships_by_ref
        ),
        domain="relationship",
    )

    structured_requests: list[StructuredGenerationRequest] = []
    for block in blocks:
        variables = {
            "block_id": block.block_id,
            "pair_contexts_json": block.pair_contexts_json,
        }
        rendered = render_prompt(prompt, variables)
        structured_requests.append(
            build_structured_request(
                rendered_prompt=rendered,
                output_schema=output_schema,
                semantic_profile=semantic_profile,
            )
        )
    request_hashes = tuple(request.request_hash for request in structured_requests)

    return RelationshipSemanticPreparation(
        planning_result=planning_result,
        consolidation_profile=consolidation_profile,
        semantic_profile=semantic_profile,
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.version,
        prompt_content_hash=prompt.content_hash,
        output_schema_id=output_schema.schema_id,
        output_schema_version=output_schema.schema_version,
        output_schema_hash=output_schema.schema_hash,
        working_language=consolidation_profile.working_language,
        packing_policy=packing_policy,
        blocks=blocks,
        structured_requests=tuple(structured_requests),
        semantic_request_hashes=request_hashes,
        semantic_pair_count=len(semantic_stream),
        auto_same_pair_count=auto_same_count,
        total_relationship_pair_count=len(relationship_plans),
    )


# ---------------------------------------------------------------------------
# Event / relationship semantic identity (BLOCK 12 / BLOCK 13)
# ---------------------------------------------------------------------------


def _build_semantic_identity(
    prep,
    *,
    total_pair_count_key: str,
) -> dict[str, Any]:
    """Shared canonical identity material for the event / relationship passes."""
    planning = prep.planning_result
    return {
        "schema_version": 1,
        "planning_policy_id": planning.planning_policy_id,
        "blocking_policy_id": planning.blocking_policy_id,
        "text_normalization_policy_id": planning.text_normalization_policy_id,
        "exact_safe_policy_id": planning.exact_safe_policy_id,
        "plan_hash": planning.plan_hash,
        "profile_id": prep.consolidation_profile.profile_id,
        "profile_hash": prep.consolidation_profile.content_hash(),
        "working_language": prep.working_language,
        "semantic_profile_id": prep.semantic_profile.profile_id,
        "semantic_profile_hash": prep.semantic_profile.semantic_profile_hash,
        "prompt_id": prep.prompt_id,
        "prompt_version": prep.prompt_version,
        "prompt_content_hash": prep.prompt_content_hash,
        "output_schema_id": prep.output_schema_id,
        "output_schema_version": prep.output_schema_version,
        "output_schema_hash": prep.output_schema_hash,
        "packing_policy": prep.packing_policy.to_dict(),
        "semantic_pair_count": prep.semantic_pair_count,
        "auto_same_pair_count": prep.auto_same_pair_count,
        total_pair_count_key: getattr(prep, total_pair_count_key),
        "block_count": len(prep.blocks),
        "block_ids": tuple(block.block_id for block in prep.blocks),
        "semantic_request_hashes": tuple(prep.semantic_request_hashes),
    }


def build_event_semantic_identity(prep: EventSemanticPreparation) -> dict[str, Any]:
    """Build the canonical A5D event semantic identity material (no backend data)."""
    return _build_semantic_identity(prep, total_pair_count_key="total_event_pair_count")


def build_relationship_semantic_identity(
    prep: RelationshipSemanticPreparation,
) -> dict[str, Any]:
    """Build the canonical A5D relationship semantic identity material."""
    return _build_semantic_identity(
        prep, total_pair_count_key="total_relationship_pair_count"
    )


# ---------------------------------------------------------------------------
# Event / relationship packing audit (BLOCK 10 / BLOCK 11 -- zero provider)
# ---------------------------------------------------------------------------


def build_event_packing_audit(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
    candidates: tuple[EventSemanticPackingPolicy, ...] = A5D_EVENT_PACKING_CANDIDATES,
) -> tuple[dict[str, Any], ...]:
    """Audit deterministic event block-packing candidates (zero provider).

    For each ``EventSemanticPackingPolicy`` candidate, build the event semantic
    preparation and report the deterministic distribution statistics and
    largest-block diagnostics. The candidates do NOT freeze the production
    default.
    """
    audit: list[dict[str, Any]] = []
    for policy in candidates:
        prep = build_event_semantic_preparation(
            planning_result,
            consolidation_profile,
            semantic_profile,
            prompts=prompts,
            packing_policy=policy,
        )
        blocks = prep.blocks
        requests = prep.structured_requests
        hashes = prep.semantic_request_hashes
        audit.append(
            {
                "packing_name": policy.name,
                "max_pairs_per_block": policy.max_pairs_per_block,
                "max_candidates_per_block": policy.max_candidates_per_block,
                "block_count": len(blocks),
                "total_pairs_in_blocks": sum(b.pair_count for b in blocks),
                "pairs_per_block": _distribution([b.pair_count for b in blocks]),
                "unique_candidates_per_block": _distribution(
                    [len(b.candidate_refs) for b in blocks]
                ),
                "evidence_items_per_block": _distribution(
                    [_block_evidence_item_count(b) for b in blocks]
                ),
                "pair_contexts_json_bytes": _distribution(
                    [b.payload_bytes for b in blocks]
                ),
                "rendered_prompt_bytes": _distribution(
                    [_request_prompt_bytes(r) for r in requests]
                ),
                "request_hash_count": len(hashes),
                "unique_request_hash_count": len(set(hashes)),
                "largest_block": _select_largest_block(prep),
            }
        )
    return tuple(audit)


def build_relationship_packing_audit(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
    candidates: tuple[
        RelationshipSemanticPackingPolicy, ...
    ] = A5D_RELATIONSHIP_PACKING_CANDIDATES,
) -> tuple[dict[str, Any], ...]:
    """Audit deterministic relationship block-packing candidates (zero provider).

    For each ``RelationshipSemanticPackingPolicy`` candidate, build the
    relationship semantic preparation and report the deterministic distribution
    statistics and largest-block diagnostics. The candidates do NOT freeze the
    production default.
    """
    audit: list[dict[str, Any]] = []
    for policy in candidates:
        prep = build_relationship_semantic_preparation(
            planning_result,
            consolidation_profile,
            semantic_profile,
            prompts=prompts,
            packing_policy=policy,
        )
        blocks = prep.blocks
        requests = prep.structured_requests
        hashes = prep.semantic_request_hashes
        audit.append(
            {
                "packing_name": policy.name,
                "max_pairs_per_block": policy.max_pairs_per_block,
                "max_candidates_per_block": policy.max_candidates_per_block,
                "block_count": len(blocks),
                "total_pairs_in_blocks": sum(b.pair_count for b in blocks),
                "pairs_per_block": _distribution([b.pair_count for b in blocks]),
                "unique_candidates_per_block": _distribution(
                    [len(b.candidate_refs) for b in blocks]
                ),
                "evidence_items_per_block": _distribution(
                    [_block_evidence_item_count(b) for b in blocks]
                ),
                "pair_contexts_json_bytes": _distribution(
                    [b.payload_bytes for b in blocks]
                ),
                "rendered_prompt_bytes": _distribution(
                    [_request_prompt_bytes(r) for r in requests]
                ),
                "request_hash_count": len(hashes),
                "unique_request_hash_count": len(set(hashes)),
                "largest_block": _select_largest_block(prep),
            }
        )
    return tuple(audit)


# ---------------------------------------------------------------------------
# A5C-B -- fact semantic provider execution + selector resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactSemanticBlockResult:
    """In-memory result of executing one fact semantic block (A5C-B).

    In-memory only; A5C-B does NOT persist. ``semantic_rounds`` is the number
    of semantic generation rounds consumed until this block succeeded (1 for a
    first-round success, 2 for a retry-then-success). ``decisions`` are the
    block's LLM fact decisions in the block's pair order; ``request_hash`` is
    the exact backend-neutral request identity for the block (identical across
    every retry round); ``generation_provenance`` is the exact provenance of
    the successful provider call.
    """

    block_id: str
    request_hash: str
    semantic_rounds: int
    decisions: tuple[FactSemanticDecision, ...]
    generation_provenance: LLMInvocationProvenance


@dataclass(frozen=True, slots=True)
class FactSemanticResolutionResult:
    """In-memory A5C-B fact semantic resolution result.

    In-memory only; A5C-B does NOT persist, does NOT allocate canonical fact
    ids, and does NOT build StateTransition / StoryConflict. Carries:
      * the exact A5B ``ConsolidationPlanningResult``;
      * the A5C-A ``FactSemanticPreparation`` the execution was driven from;
      * the LLM fact decisions only (``semantic_decisions``);
      * the combined deterministic auto_same + LLM fact decisions
        (``all_fact_decisions``), in canonical ``(left, right)`` order;
      * the per-block execution results (``block_results``).
    """

    planning_result: ConsolidationPlanningResult
    preparation: FactSemanticPreparation
    semantic_decisions: tuple[FactSemanticDecision, ...]
    all_fact_decisions: tuple[FactSemanticDecision, ...]
    block_results: tuple[FactSemanticBlockResult, ...]


# ---------------------------------------------------------------------------
# Decision id (backend-neutral semantic identity, NOT runtime provenance)
# ---------------------------------------------------------------------------


def compute_fact_llm_decision_id(
    left_ref: str,
    right_ref: str,
    decision: str,
    reason_zh: str,
    evidence_refs: tuple[EvidenceRef, ...],
    prompt_id: str,
    prompt_version: int,
    request_hash: str,
) -> str:
    """Compute the deterministic A5C fact LLM decision id.

    The single authority for A5C fact LLM decision ids. The canonical semantic
    material is backend-neutral: it binds the domain, the pair, the decision,
    the method (``llm``), the reason, the resolved exact (canonical,
    exact-deduped) evidence refs, the prompt identity, and the backend-neutral
    ``request_hash``. It does NOT include provider_family / model /
    provider_response_id / usage / finish_reason / endpoint / timeout /
    timestamp / PID: those are audit provenance only, so changing them alone
    does not alter the recomputed decision id.
    """
    material = {
        "domain": "fact",
        "left_candidate_ref": left_ref,
        "right_candidate_ref": right_ref,
        "decision": decision,
        "method": "llm",
        "reason_zh": reason_zh,
        "evidence_refs": [e.to_dict() for e in evidence_refs],
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "request_hash": request_hash,
    }
    return "dec_" + content_hash(material)[:20]


# ---------------------------------------------------------------------------
# Provenance verification (FAIL CLOSED, no semantic retry)
# ---------------------------------------------------------------------------


def _verify_fact_provenance(
    provenance: LLMInvocationProvenance,
    request: StructuredGenerationRequest,
    semantic_profile: SemanticLLMProfile,
) -> None:
    """Verify a successful generation's provenance matches the exact request.

    Checks backend-neutral semantic/request identity only:
      semantic_profile_id / hash
      prompt_id / version / content_hash
      rendered_prompt_hash
      output_schema_id / version / hash
      request_hash

    Any mismatch raises :class:`ConsolidationProvenanceError` (FAIL CLOSED, NO
    semantic retry). provider_family / model / provider_response_id / usage /
    finish_reason are audit provenance only and are deliberately NOT compared.
    """
    rendered = request.rendered_prompt
    schema = request.output_schema
    expected = {
        "semantic_profile_id": semantic_profile.profile_id,
        "semantic_profile_hash": semantic_profile.semantic_profile_hash,
        "prompt_id": rendered.prompt_id,
        "prompt_version": rendered.prompt_version,
        "prompt_content_hash": rendered.prompt_content_hash,
        "rendered_prompt_hash": rendered.rendered_prompt_hash,
        "output_schema_id": schema.schema_id,
        "output_schema_version": schema.schema_version,
        "output_schema_hash": schema.schema_hash,
        "request_hash": request.request_hash,
    }
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
            raise ConsolidationProvenanceError(
                f"provenance field {key!r} mismatch: expected "
                f"{expected[key]!r}, got {actual[key]!r}"
            )


# ---------------------------------------------------------------------------
# Pair-local selector canonicalization + exact endpoint evidence resolution
# ---------------------------------------------------------------------------


def _canonicalize_fact_selectors(selectors: list[str]) -> list[str]:
    """Canonicalize validated, unique pair-local selectors to the frozen order.

    Frozen pair-local order (authoritative for persistence): all left
    selectors first (numeric index ascending), then all right selectors
    (numeric index ascending). ``["R2", "L1", "R0", "L0"]`` ->
    ``["L0", "L1", "R0", "R2"]``. ``selectors`` must already be validated
    (``L<index>`` / ``R<index>`` form) and duplicate-free; nothing is
    re-validated or dropped here. This is NOT repair of invalid selectors.
    """
    left = sorted((s for s in selectors if s[0] == "L"), key=lambda s: int(s[1:]))
    right = sorted((s for s in selectors if s[0] == "R"), key=lambda s: int(s[1:]))
    return left + right


def _fact_evidence_exact_identity(
    ev: EvidenceRef,
) -> tuple[str, str, str, "str | None"]:
    """The exact persisted EvidenceRef identity tuple.

    Two EvidenceRefs are the same evidence for exact-alias dedupe purposes iff
    ``(paragraph_id, role, strength, excerpt)`` are equal. ``excerpt=None`` is
    distinct from any string value. No normalization or case-folding.
    """
    return (ev.paragraph_id, ev.role, ev.strength, ev.excerpt)


def _block_endpoint_evidence(
    planning: ConsolidationPlanningResult,
    pair_refs: tuple[tuple[str, str], ...],
) -> list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]]:
    """Per-pair endpoint evidence for selector resolution (aligned with pairs).

    Element ``i`` is pair ``i``'s ``(left_endpoint_evidence,
    right_endpoint_evidence)`` -- each the EXACT indexed A5B
    ``cand.evidence_refs`` tuple in order. ``L<n>`` therefore resolves to
    ``left_endpoint_evidence[n]`` and ``R<n>`` to ``right_endpoint_evidence[n]``
    (zero-based, in indexed order). This reuses the exact indexed candidates
    (no reload / re-sort), so the resolved EvidenceRef is byte-for-byte the
    endpoint's own EvidenceRef.
    """
    facts_by_ref = {c.global_candidate_ref: c for c in planning.index.facts}
    out: list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]] = []
    for left_ref, right_ref in pair_refs:
        left = facts_by_ref.get(left_ref)
        right = facts_by_ref.get(right_ref)
        if left is None or right is None:
            raise StoryIntegrityError(
                "fact pair endpoint missing from the fact candidate index"
            )
        out.append((left.evidence_refs, right.evidence_refs))
    return out


def _validate_fact_selector_block_payload(
    payload: FactSelectorDecisionPayload,
    pair_refs: tuple[tuple[str, str], ...],
    endpoint_evidence: list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]],
) -> tuple[bool, str, list[tuple[EvidenceRef, ...]]]:
    """Validate a fact block's provider payload against the exact requested pairs.

    The single A5C-B authority for fact selector validity. ``endpoint_evidence``
    is aligned with ``pair_refs``; ``L<n>`` resolves to the pair's own left
    endpoint evidence item ``n`` and ``R<n>`` to the right endpoint evidence
    item ``n`` (both zero-based, in indexed order).

    Returns ``(is_valid, failure_detail, resolved_evidence)``.
    ``resolved_evidence`` is aligned with ``payload.decisions`` when valid (each
    a tuple of the exact endpoint EvidenceRefs in canonical pair-local selector
    order with exact-EvidenceRef alias deduplication), otherwise empty.

    Checks: count exact, order exact, left/right refs exact; each selector
    matches ``L<index>`` / ``R<index>`` (invalid prefix / negative / non-integer
    index rejected); index in range for the pair's own endpoint (a selector can
    never reach a third candidate / another pair / block-level evidence); no
    duplicate selector string within a decision. Only after every selector in a
    decision passes syntax + range + duplicate validation is the selector set
    canonicalized and resolved to exact endpoint EvidenceRefs, then projected to
    the canonical exact-EvidenceRef form (first occurrence in canonical selector
    order wins). Invalid selectors always fail BEFORE canonicalization and
    consume a bounded semantic round.
    """
    requested_count = len(pair_refs)
    decisions = payload.decisions

    if len(decisions) != requested_count:
        return (
            False,
            f"decision count mismatch: expected {requested_count}, "
            f"got {len(decisions)}",
            [],
        )

    resolved: list[tuple[EvidenceRef, ...]] = []
    for i, item in enumerate(decisions):
        expected_left, expected_right = pair_refs[i]
        if item.left_candidate_ref != expected_left:
            return (
                False,
                f"pair {i}: left_candidate_ref mismatch: expected "
                f"{expected_left!r}, got {item.left_candidate_ref!r}",
                [],
            )
        if item.right_candidate_ref != expected_right:
            return (
                False,
                f"pair {i}: right_candidate_ref mismatch: expected "
                f"{expected_right!r}, got {item.right_candidate_ref!r}",
                [],
            )

        left_evidence, right_evidence = endpoint_evidence[i]
        seen_selectors: set[str] = set()
        validated_selectors: list[str] = []
        for selector in item.evidence_selectors:
            if _FACT_EVIDENCE_SELECTOR_RE.fullmatch(selector) is None:
                return (
                    False,
                    f"pair {i}: invalid evidence selector {selector!r}; only "
                    f"L<index>/R<index> forms are legal",
                    [],
                )
            if selector in seen_selectors:
                return (
                    False,
                    f"pair {i}: duplicate evidence selector: {selector!r}",
                    [],
                )
            seen_selectors.add(selector)
            index = int(selector[1:])
            if selector[0] == "L":
                if index >= len(left_evidence):
                    return (
                        False,
                        f"pair {i}: evidence selector {selector!r} out of range "
                        f"for the left endpoint ({len(left_evidence)} evidence "
                        f"item(s))",
                        [],
                    )
            else:  # "R"
                if index >= len(right_evidence):
                    return (
                        False,
                        f"pair {i}: evidence selector {selector!r} out of range "
                        f"for the right endpoint ({len(right_evidence)} evidence "
                        f"item(s))",
                        [],
                    )
            validated_selectors.append(selector)

        canonical_selectors = _canonicalize_fact_selectors(validated_selectors)

        decision_evidence: list[EvidenceRef] = []
        seen_exact: set[tuple[str, str, str, "str | None"]] = set()
        for selector in canonical_selectors:
            index = int(selector[1:])
            ev = (
                left_evidence[index]
                if selector[0] == "L"
                else right_evidence[index]
            )
            identity = _fact_evidence_exact_identity(ev)
            if identity not in seen_exact:
                seen_exact.add(identity)
                decision_evidence.append(ev)
        resolved.append(tuple(decision_evidence))

    return True, "", resolved


def _convert_to_fact_decision(
    left_ref: str,
    right_ref: str,
    decision: str,
    reason_zh: str,
    evidence_refs: tuple[EvidenceRef, ...],
    request_hash: str,
    prompt_id: str,
    prompt_version: int,
    provenance: LLMInvocationProvenance,
) -> FactSemanticDecision:
    """Convert a valid, selector-resolved decision to a FactSemanticDecision.

    ``evidence_refs`` are the exact endpoint EvidenceRefs resolved from the
    provider's pair-local selectors in canonical pair-local selector order with
    stable exact-EvidenceRef alias deduplication. The persisted decision carries
    those exact EvidenceRefs (no selector strings), ``method="llm"``, the exact
    prompt identity, the exact provider provenance, and the deterministic LLM
    decision id.
    """
    decision_id = compute_fact_llm_decision_id(
        left_ref=left_ref,
        right_ref=right_ref,
        decision=decision,
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        request_hash=request_hash,
    )
    return FactSemanticDecision(
        decision_id=decision_id,
        left_candidate_ref=left_ref,
        right_candidate_ref=right_ref,
        decision=decision,
        method="llm",
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        generation_provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Whole-fact decision coverage (deterministic auto_same + LLM semantic)
# ---------------------------------------------------------------------------


def _validate_fact_decision_coverage(
    planning: ConsolidationPlanningResult,
    deterministic_decisions: tuple[FactSemanticDecision, ...],
    semantic_decisions: tuple[FactSemanticDecision, ...],
) -> tuple[FactSemanticDecision, ...]:
    """Validate and combine all fact decisions against the authoritative plans.

    Enforces the frozen invariant:
      * every explicit A5B fact pair in ``planning.fact_pair_plans`` has exactly
        one decision;
      * no extra decision for a pair not in the plans;
      * no duplicate pair decision;
      * state/method consistency:
          auto_same -> method=deterministic, decision=same_fact
          needs_semantic_decision -> method=llm, decision in the six fact
            decisions
      * unblocked / not_compared pairs are not present (only explicit pairs).

    Returns canonical order by ``(left_candidate_ref, right_candidate_ref)``.
    """
    all_decisions = list(deterministic_decisions) + list(semantic_decisions)

    decision_by_pair: dict[tuple[str, str], FactSemanticDecision] = {}
    for d in all_decisions:
        key = (d.left_candidate_ref, d.right_candidate_ref)
        if key in decision_by_pair:
            raise ConsolidationSemanticError(
                f"duplicate decision for fact pair {key!r}"
            )
        decision_by_pair[key] = d

    expected_pairs: set[tuple[str, str]] = set()
    pair_state: dict[tuple[str, str], str] = {}
    for plan in planning.fact_pair_plans:
        pair_key = (plan.left_ref, plan.right_ref)
        expected_pairs.add(pair_key)
        pair_state[pair_key] = plan.state

    for pair_key in decision_by_pair:
        if pair_key not in expected_pairs:
            raise ConsolidationSemanticError(
                f"fact decision found for pair {pair_key!r} not present in "
                f"planning_result.fact_pair_plans; invalid decision coverage"
            )

    for pair_key in expected_pairs:
        if pair_key not in decision_by_pair:
            raise ConsolidationSemanticError(
                f"no decision found for fact pair {pair_key!r}; incomplete "
                f"fact decision coverage"
            )

    for pair_key, state in pair_state.items():
        d = decision_by_pair[pair_key]
        if state == PAIR_STATE_AUTO_SAME:
            if d.method != "deterministic":
                raise ConsolidationSemanticError(
                    f"fact pair {pair_key!r} (auto_same) has method "
                    f"{d.method!r}; expected 'deterministic'"
                )
            if d.decision != "same_fact":
                raise ConsolidationSemanticError(
                    f"fact pair {pair_key!r} (auto_same) has decision "
                    f"{d.decision!r}; expected 'same_fact'"
                )
        elif state == PAIR_STATE_NEEDS_SEMANTIC_DECISION:
            if d.method != "llm":
                raise ConsolidationSemanticError(
                    f"fact pair {pair_key!r} (needs_semantic_decision) has "
                    f"method {d.method!r}; expected 'llm'"
                )
            if d.decision not in FACT_DECISIONS:
                raise ConsolidationSemanticError(
                    f"fact pair {pair_key!r} (needs_semantic_decision) has "
                    f"decision {d.decision!r}; expected one of "
                    f"{sorted(FACT_DECISIONS)}"
                )

    all_decisions.sort(key=lambda d: (d.left_candidate_ref, d.right_candidate_ref))
    return tuple(all_decisions)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def resolve_fact_semantic_ambiguity(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    *,
    prompts: PromptRegistry,
) -> FactSemanticResolutionResult:
    """Execute A5C-B fact semantic ambiguity resolution.

    Consumes the A5B ``ConsolidationPlanningResult`` and resolves every
    ``needs_semantic_decision`` fact pair via bounded LLM semantic generation.

    The deterministic, zero-provider request construction is delegated to
    :func:`build_fact_semantic_preparation` with the FROZEN production policy
    :data:`FACT_SEMANTIC_PACKING_V1` (12 pairs / 24 candidates, identical block
    ids / request hashes to the audited P2). This function only drives the
    provider per block (sequentially, in exact preparation order) and merges the
    results. A5C-B does NOT persist, does NOT write CURRENT, and does NOT build
    CanonicalFactSet / StateTransition / StoryConflict.

    Per block (block atomicity):
      * ``llm_client.generate_structured(...)`` (A-I3 owns the provider call and
        technical retry);
      * provenance verification (FAIL CLOSED, no semantic retry);
      * ``FactSelectorDecisionPayload.from_dict(...)`` (typed load);
      * exact pair coverage/order + pair-local selector validation;
      * canonical selector order + exact endpoint EvidenceRef resolution +
        stable exact-EvidenceRef alias dedupe;
      * ``FactSemanticDecision(method="llm")`` construction.

    Semantic rounds are bounded to :data:`A5C_FACT_MAX_GENERATION_ROUNDS` (2).
    An ``LLMError`` raised by the provider after A-I3 technical retry is
    PROPAGATED (not turned into a semantic retry). Provenance mismatch fails
    closed with no retry. Two semantic-invalid rounds raise
    :class:`ConsolidationSemanticGenerationError` (the whole block is invalid; no
    partial decisions are retained). ``decision == uncertain`` is a valid
    successful semantic result and is NOT retried.

    After all blocks succeed, the combined deterministic auto_same + LLM fact
    decisions are validated for exact whole-fact-stream coverage.
    """
    preparation = build_fact_semantic_preparation(
        planning_result,
        consolidation_profile,
        semantic_profile,
        prompts=prompts,
        packing_policy=FACT_SEMANTIC_PACKING_V1,
    )
    blocks = preparation.blocks

    deterministic_fact_decisions = (
        planning_result.deterministic_decision_set.fact_decisions
    )

    # 1. Zero semantic pairs -> zero blocks -> validate deterministic coverage.
    if not blocks:
        all_fact_decisions = _validate_fact_decision_coverage(
            planning_result, deterministic_fact_decisions, ()
        )
        return FactSemanticResolutionResult(
            planning_result=planning_result,
            preparation=preparation,
            semantic_decisions=(),
            all_fact_decisions=all_fact_decisions,
            block_results=(),
        )

    all_semantic_decisions: list[FactSemanticDecision] = []
    all_block_results: list[FactSemanticBlockResult] = []

    # 2. Process each block sequentially in exact preparation order.
    for block, request in zip(blocks, preparation.structured_requests):
        rendered_prompt = request.rendered_prompt
        output_schema = request.output_schema

        endpoint_evidence = _block_endpoint_evidence(planning_result, block.pair_refs)

        max_rounds = consolidation_profile.max_generation_rounds
        block_decisions: list[FactSemanticDecision] = []
        block_provenance: LLMInvocationProvenance | None = None
        rounds_attempted = 0
        last_failure = ""

        for round_number in range(1, max_rounds + 1):
            rounds_attempted = round_number

            # One provider call (A-I3 owns technical retry). An LLMError after
            # A-I3 technical retry PROPAGATES (not a semantic retry).
            result = llm_client.generate_structured(
                rendered_prompt, output_schema, semantic_profile
            )

            # Provenance verification (FAIL CLOSED, NO semantic retry).
            _verify_fact_provenance(result.provenance, request, semantic_profile)

            # Typed domain load (typed-model rejection -> semantic invalid).
            try:
                payload = FactSelectorDecisionPayload.from_dict(result.parsed_json)
            except ConsolidationModelError:
                last_failure = "typed payload load failed"
                continue

            # Exact pair coverage/order + pair-local selector validation
            # (fail closed; consumes a bounded semantic round on any failure).
            is_valid, failure_detail, resolved_evidence = (
                _validate_fact_selector_block_payload(
                    payload, block.pair_refs, endpoint_evidence
                )
            )
            if not is_valid:
                last_failure = failure_detail
                continue

            # Valid: resolve selectors to exact endpoint EvidenceRefs (canonical
            # order, exact-alias deduped) and convert to LLM decisions.
            for item, resolved in zip(payload.decisions, resolved_evidence):
                block_decisions.append(
                    _convert_to_fact_decision(
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
                FactSemanticBlockResult(
                    block_id=block.block_id,
                    request_hash=request.request_hash,
                    semantic_rounds=rounds_attempted,
                    decisions=tuple(block_decisions),
                    generation_provenance=block_provenance,  # type: ignore[arg-type]
                )
            )
        else:
            # Semantic exhaustion: both rounds invalid for this block. Do not
            # return partial decisions; do not continue to later blocks.
            raise ConsolidationSemanticGenerationError(
                block_id=block.block_id,
                request_hash=request.request_hash,
                rounds_attempted=rounds_attempted,
                last_failure_details=last_failure,
                expected_pairs=tuple(block.pair_refs),
            )

    # 3. Combine deterministic + LLM fact decisions; validate exact coverage.
    all_fact_decisions = _validate_fact_decision_coverage(
        planning_result, deterministic_fact_decisions, tuple(all_semantic_decisions)
    )

    return FactSemanticResolutionResult(
        planning_result=planning_result,
        preparation=preparation,
        semantic_decisions=tuple(all_semantic_decisions),
        all_fact_decisions=all_fact_decisions,
        block_results=tuple(all_block_results),
    )


# ---------------------------------------------------------------------------
# A5D-B -- event + relationship semantic provider execution + selector resolution
#
# The event / relationship equivalents of the A5C-B fact provider-execution path.
# They reuse the EXACT A5C-B patterns: provenance verification (FAIL CLOSED),
# typed selector-payload load, exact pair coverage/order, pair-local selector
# validation, canonical selector order, exact endpoint EvidenceRef resolution,
# stable exact-EvidenceRef alias dedupe, domain SemanticDecision(method="llm")
# construction, and exact whole-domain decision coverage.
#
# A5D-B does NOT persist, does NOT write CURRENT, does NOT build canonical event/
# relationship IDs, StateTransition, StoryConflict, or graph components.
# ---------------------------------------------------------------------------

# Frozen production event / relationship semantic packing policies (A5D-B).
#
# ``event-semantic-packing-v1`` and ``relationship-semantic-packing-v1`` are the
# FROZEN POST-AUDIT policies (docs ``v1.2-A5D-event-relationship-semantic-packing-v1.md``):
# 12 pairs / 24 candidates each (audit candidate P2). The canonical packing
# material is the frozen behavioral limits (12 / 24), NOT the policy label:
# ``to_dict()`` carries only the two limits, so these policies produce
# byte-identical block ids and request hashes to the audited P2 preparation.
# A5D-B consumes these exact policies and never re-selects P1 / P2 / P3 at
# runtime.
EVENT_SEMANTIC_PACKING_V1: EventSemanticPackingPolicy = EventSemanticPackingPolicy(
    "event-semantic-packing-v1",
    12,
    24,
)
RELATIONSHIP_SEMANTIC_PACKING_V1: RelationshipSemanticPackingPolicy = (
    RelationshipSemanticPackingPolicy(
        "relationship-semantic-packing-v1",
        12,
        24,
    )
)

# Shared A5 selector regex (same as A5C fact: A5_EVIDENCE_SELECTOR_PATTERN).
_EVENT_EVIDENCE_SELECTOR_RE = re.compile(A5_EVIDENCE_SELECTOR_PATTERN)
_RELATIONSHIP_EVIDENCE_SELECTOR_RE = re.compile(A5_EVIDENCE_SELECTOR_PATTERN)


# ---------------------------------------------------------------------------
# Result models (A5D-B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventSemanticBlockResult:
    """In-memory result of executing one event semantic block (A5D-B).

    In-memory only; A5D-B does NOT persist. ``semantic_rounds`` is the number
    of semantic generation rounds consumed until this block succeeded. ``decisions``
    are the block's LLM event decisions in the block's pair order.
    """

    block_id: str
    request_hash: str
    semantic_rounds: int
    decisions: tuple[EventSemanticDecision, ...]
    generation_provenance: LLMInvocationProvenance


@dataclass(frozen=True, slots=True)
class EventSemanticResolutionResult:
    """In-memory A5D-B event semantic resolution result.

    In-memory only; A5D-B does NOT persist, does NOT allocate canonical event
    ids, and does NOT build StateTransition / StoryConflict / graph components.
    """

    planning_result: ConsolidationPlanningResult
    preparation: EventSemanticPreparation
    semantic_decisions: tuple[EventSemanticDecision, ...]
    all_event_decisions: tuple[EventSemanticDecision, ...]
    block_results: tuple[EventSemanticBlockResult, ...]


@dataclass(frozen=True, slots=True)
class RelationshipSemanticBlockResult:
    """In-memory result of executing one relationship semantic block (A5D-B)."""

    block_id: str
    request_hash: str
    semantic_rounds: int
    decisions: tuple[RelationshipSemanticDecision, ...]
    generation_provenance: LLMInvocationProvenance


@dataclass(frozen=True, slots=True)
class RelationshipSemanticResolutionResult:
    """In-memory A5D-B relationship semantic resolution result.

    In-memory only; A5D-B does NOT persist, does NOT allocate canonical
    relationship ids, and does NOT build state histories / graph components.
    """

    planning_result: ConsolidationPlanningResult
    preparation: RelationshipSemanticPreparation
    semantic_decisions: tuple[RelationshipSemanticDecision, ...]
    all_relationship_decisions: tuple[RelationshipSemanticDecision, ...]
    block_results: tuple[RelationshipSemanticBlockResult, ...]


# ---------------------------------------------------------------------------
# Decision id (event / relationship -- backend-neutral semantic identity)
# ---------------------------------------------------------------------------


def compute_event_llm_decision_id(
    left_ref: str,
    right_ref: str,
    decision: str,
    reason_zh: str,
    evidence_refs: tuple[EvidenceRef, ...],
    prompt_id: str,
    prompt_version: int,
    request_hash: str,
) -> str:
    """Compute the deterministic A5D event LLM decision id.

    Backend-neutral: binds domain, pair, decision, method, reason, evidence,
    prompt identity, and request_hash. Does NOT include runtime provider
    metadata.
    """
    material = {
        "domain": "event",
        "left_candidate_ref": left_ref,
        "right_candidate_ref": right_ref,
        "decision": decision,
        "method": "llm",
        "reason_zh": reason_zh,
        "evidence_refs": [e.to_dict() for e in evidence_refs],
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "request_hash": request_hash,
    }
    return "dec_" + content_hash(material)[:20]


def compute_relationship_llm_decision_id(
    left_ref: str,
    right_ref: str,
    decision: str,
    reason_zh: str,
    evidence_refs: tuple[EvidenceRef, ...],
    prompt_id: str,
    prompt_version: int,
    request_hash: str,
) -> str:
    """Compute the deterministic A5D relationship LLM decision id."""
    material = {
        "domain": "relationship",
        "left_candidate_ref": left_ref,
        "right_candidate_ref": right_ref,
        "decision": decision,
        "method": "llm",
        "reason_zh": reason_zh,
        "evidence_refs": [e.to_dict() for e in evidence_refs],
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "request_hash": request_hash,
    }
    return "dec_" + content_hash(material)[:20]


# ---------------------------------------------------------------------------
# Generic provenance verification (FAIL CLOSED, no semantic retry)
# ---------------------------------------------------------------------------


def _verify_semantic_provenance(
    provenance: LLMInvocationProvenance,
    request: StructuredGenerationRequest,
    semantic_profile: SemanticLLMProfile,
) -> None:
    """Verify a successful generation's provenance matches the exact request.

    Checks backend-neutral semantic/request identity only. Any mismatch raises
    :class:`ConsolidationProvenanceError` (FAIL CLOSED, NO semantic retry).
    provider_family / model / provider_response_id / usage / finish_reason are
    audit provenance only and are deliberately NOT compared.
    """
    rendered = request.rendered_prompt
    schema = request.output_schema
    expected = {
        "semantic_profile_id": semantic_profile.profile_id,
        "semantic_profile_hash": semantic_profile.semantic_profile_hash,
        "prompt_id": rendered.prompt_id,
        "prompt_version": rendered.prompt_version,
        "prompt_content_hash": rendered.prompt_content_hash,
        "rendered_prompt_hash": rendered.rendered_prompt_hash,
        "output_schema_id": schema.schema_id,
        "output_schema_version": schema.schema_version,
        "output_schema_hash": schema.schema_hash,
        "request_hash": request.request_hash,
    }
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
            raise ConsolidationProvenanceError(
                f"provenance field {key!r} mismatch: expected "
                f"{expected[key]!r}, got {actual[key]!r}"
            )


# ---------------------------------------------------------------------------
# Generic pair-local selector canonicalization + exact endpoint evidence resolution
# ---------------------------------------------------------------------------


def _canonicalize_selectors(selectors: list[str]) -> list[str]:
    """Canonicalize validated, unique pair-local selectors to the frozen order.

    All left selectors first (numeric ascending), then all right selectors
    (numeric ascending). Must already be validated and duplicate-free.
    """
    left = sorted((s for s in selectors if s[0] == "L"), key=lambda s: int(s[1:]))
    right = sorted((s for s in selectors if s[0] == "R"), key=lambda s: int(s[1:]))
    return left + right


def _evidence_exact_identity(ev: EvidenceRef) -> tuple[str, str, str, "str | None"]:
    """The exact persisted EvidenceRef identity tuple for alias dedup."""
    return (ev.paragraph_id, ev.role, ev.strength, ev.excerpt)


def _event_block_endpoint_evidence(
    planning: ConsolidationPlanningResult,
    pair_refs: tuple[tuple[str, str], ...],
) -> list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]]:
    """Per-pair endpoint evidence for event selector resolution."""
    events_by_ref = {c.global_candidate_ref: c for c in planning.index.events}
    out: list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]] = []
    for left_ref, right_ref in pair_refs:
        left = events_by_ref.get(left_ref)
        right = events_by_ref.get(right_ref)
        if left is None or right is None:
            raise StoryIntegrityError(
                "event pair endpoint missing from the event candidate index"
            )
        out.append((left.evidence_refs, right.evidence_refs))
    return out


def _relationship_block_endpoint_evidence(
    planning: ConsolidationPlanningResult,
    pair_refs: tuple[tuple[str, str], ...],
) -> list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]]:
    """Per-pair endpoint evidence for relationship selector resolution."""
    rels_by_ref = {
        c.global_candidate_ref: c for c in planning.index.relationships
    }
    out: list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]] = []
    for left_ref, right_ref in pair_refs:
        left = rels_by_ref.get(left_ref)
        right = rels_by_ref.get(right_ref)
        if left is None or right is None:
            raise StoryIntegrityError(
                "relationship pair endpoint missing from the relationship candidate index"
            )
        out.append((left.evidence_refs, right.evidence_refs))
    return out


# ---------------------------------------------------------------------------
# Generic selector block payload validation
# ---------------------------------------------------------------------------


def _validate_selector_block_payload(
    payload_decisions: list,
    pair_refs: tuple[tuple[str, str], ...],
    endpoint_evidence: list[tuple[tuple[EvidenceRef, ...], tuple[EvidenceRef, ...]]],
    selector_re: re.Pattern,
    domain: str,
) -> tuple[bool, str, list[tuple[EvidenceRef, ...]]]:
    """Validate a block's provider payload against the exact requested pairs.

    Generic for fact / event / relationship. Returns ``(is_valid, failure_detail,
    resolved_evidence)``. ``resolved_evidence`` is aligned with the payload
    decisions when valid, otherwise empty.
    """
    requested_count = len(pair_refs)
    decisions = payload_decisions

    if len(decisions) != requested_count:
        return (
            False,
            f"decision count mismatch: expected {requested_count}, "
            f"got {len(decisions)}",
            [],
        )

    resolved: list[tuple[EvidenceRef, ...]] = []
    for i, item in enumerate(decisions):
        expected_left, expected_right = pair_refs[i]
        if item.left_candidate_ref != expected_left:
            return (
                False,
                f"pair {i}: left_candidate_ref mismatch: expected "
                f"{expected_left!r}, got {item.left_candidate_ref!r}",
                [],
            )
        if item.right_candidate_ref != expected_right:
            return (
                False,
                f"pair {i}: right_candidate_ref mismatch: expected "
                f"{expected_right!r}, got {item.right_candidate_ref!r}",
                [],
            )

        left_evidence, right_evidence = endpoint_evidence[i]
        seen_selectors: set[str] = set()
        validated_selectors: list[str] = []
        for selector in item.evidence_selectors:
            if selector_re.fullmatch(selector) is None:
                return (
                    False,
                    f"pair {i}: invalid evidence selector {selector!r}; only "
                    f"L<index>/R<index> forms are legal",
                    [],
                )
            if selector in seen_selectors:
                return (
                    False,
                    f"pair {i}: duplicate evidence selector: {selector!r}",
                    [],
                )
            seen_selectors.add(selector)
            index = int(selector[1:])
            if selector[0] == "L":
                if index >= len(left_evidence):
                    return (
                        False,
                        f"pair {i}: evidence selector {selector!r} out of range "
                        f"for the left endpoint ({len(left_evidence)} evidence "
                        f"item(s))",
                        [],
                    )
            else:
                if index >= len(right_evidence):
                    return (
                        False,
                        f"pair {i}: evidence selector {selector!r} out of range "
                        f"for the right endpoint ({len(right_evidence)} evidence "
                        f"item(s))",
                        [],
                    )
            validated_selectors.append(selector)

        canonical_selectors = _canonicalize_selectors(validated_selectors)

        decision_evidence: list[EvidenceRef] = []
        seen_exact: set[tuple[str, str, str, "str | None"]] = set()
        for selector in canonical_selectors:
            index = int(selector[1:])
            ev = (
                left_evidence[index]
                if selector[0] == "L"
                else right_evidence[index]
            )
            identity = _evidence_exact_identity(ev)
            if identity not in seen_exact:
                seen_exact.add(identity)
                decision_evidence.append(ev)
        resolved.append(tuple(decision_evidence))

    return True, "", resolved


# ---------------------------------------------------------------------------
# Decision conversion (event / relationship)
# ---------------------------------------------------------------------------


def _convert_to_event_decision(
    left_ref: str,
    right_ref: str,
    decision: str,
    reason_zh: str,
    evidence_refs: tuple[EvidenceRef, ...],
    request_hash: str,
    prompt_id: str,
    prompt_version: int,
    provenance: LLMInvocationProvenance,
) -> EventSemanticDecision:
    """Convert a valid, selector-resolved event decision to EventSemanticDecision."""
    decision_id = compute_event_llm_decision_id(
        left_ref=left_ref,
        right_ref=right_ref,
        decision=decision,
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        request_hash=request_hash,
    )
    return EventSemanticDecision(
        decision_id=decision_id,
        left_candidate_ref=left_ref,
        right_candidate_ref=right_ref,
        decision=decision,
        method="llm",
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        generation_provenance=provenance,
    )


def _convert_to_relationship_decision(
    left_ref: str,
    right_ref: str,
    decision: str,
    reason_zh: str,
    evidence_refs: tuple[EvidenceRef, ...],
    request_hash: str,
    prompt_id: str,
    prompt_version: int,
    provenance: LLMInvocationProvenance,
) -> RelationshipSemanticDecision:
    """Convert a valid, selector-resolved relationship decision to RelationshipSemanticDecision."""
    decision_id = compute_relationship_llm_decision_id(
        left_ref=left_ref,
        right_ref=right_ref,
        decision=decision,
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        request_hash=request_hash,
    )
    return RelationshipSemanticDecision(
        decision_id=decision_id,
        left_candidate_ref=left_ref,
        right_candidate_ref=right_ref,
        decision=decision,
        method="llm",
        reason_zh=reason_zh,
        evidence_refs=evidence_refs,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        generation_provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Whole-domain decision coverage (event / relationship)
# ---------------------------------------------------------------------------


def _validate_event_decision_coverage(
    planning: ConsolidationPlanningResult,
    deterministic_decisions: tuple[EventSemanticDecision, ...],
    semantic_decisions: tuple[EventSemanticDecision, ...],
) -> tuple[EventSemanticDecision, ...]:
    """Validate and combine all event decisions against the authoritative plans.

    Enforces: every explicit event pair plan has exactly one decision; state/
    method consistency; canonical order by (left, right).
    """
    from .consolidation import EVENT_DECISIONS

    all_decisions = list(deterministic_decisions) + list(semantic_decisions)

    decision_by_pair: dict[tuple[str, str], EventSemanticDecision] = {}
    for d in all_decisions:
        key = (d.left_candidate_ref, d.right_candidate_ref)
        if key in decision_by_pair:
            raise ConsolidationSemanticError(
                f"duplicate decision for event pair {key!r}"
            )
        decision_by_pair[key] = d

    expected_pairs: set[tuple[str, str]] = set()
    pair_state: dict[tuple[str, str], str] = {}
    for plan in planning.event_pair_plans:
        pair_key = (plan.left_ref, plan.right_ref)
        expected_pairs.add(pair_key)
        pair_state[pair_key] = plan.state

    for pair_key in decision_by_pair:
        if pair_key not in expected_pairs:
            raise ConsolidationSemanticError(
                f"event decision found for pair {pair_key!r} not present in "
                f"planning_result.event_pair_plans; invalid decision coverage"
            )

    for pair_key in expected_pairs:
        if pair_key not in decision_by_pair:
            raise ConsolidationSemanticError(
                f"no decision found for event pair {pair_key!r}; incomplete "
                f"event decision coverage"
            )

    for pair_key, state in pair_state.items():
        d = decision_by_pair[pair_key]
        if state == PAIR_STATE_AUTO_SAME:
            if d.method != "deterministic":
                raise ConsolidationSemanticError(
                    f"event pair {pair_key!r} (auto_same) has method "
                    f"{d.method!r}; expected 'deterministic'"
                )
            if d.decision != "same_event":
                raise ConsolidationSemanticError(
                    f"event pair {pair_key!r} (auto_same) has decision "
                    f"{d.decision!r}; expected 'same_event'"
                )
        elif state == PAIR_STATE_NEEDS_SEMANTIC_DECISION:
            if d.method != "llm":
                raise ConsolidationSemanticError(
                    f"event pair {pair_key!r} (needs_semantic_decision) has "
                    f"method {d.method!r}; expected 'llm'"
                )
            if d.decision not in EVENT_DECISIONS:
                raise ConsolidationSemanticError(
                    f"event pair {pair_key!r} (needs_semantic_decision) has "
                    f"decision {d.decision!r}; expected one of "
                    f"{sorted(EVENT_DECISIONS)}"
                )

    all_decisions.sort(key=lambda d: (d.left_candidate_ref, d.right_candidate_ref))
    return tuple(all_decisions)


def _validate_relationship_decision_coverage(
    planning: ConsolidationPlanningResult,
    deterministic_decisions: tuple[RelationshipSemanticDecision, ...],
    semantic_decisions: tuple[RelationshipSemanticDecision, ...],
) -> tuple[RelationshipSemanticDecision, ...]:
    """Validate and combine all relationship decisions against the authoritative plans."""
    from .consolidation import RELATIONSHIP_DECISIONS

    all_decisions = list(deterministic_decisions) + list(semantic_decisions)

    decision_by_pair: dict[tuple[str, str], RelationshipSemanticDecision] = {}
    for d in all_decisions:
        key = (d.left_candidate_ref, d.right_candidate_ref)
        if key in decision_by_pair:
            raise ConsolidationSemanticError(
                f"duplicate decision for relationship pair {key!r}"
            )
        decision_by_pair[key] = d

    expected_pairs: set[tuple[str, str]] = set()
    pair_state: dict[tuple[str, str], str] = {}
    for plan in planning.relationship_pair_plans:
        pair_key = (plan.left_ref, plan.right_ref)
        expected_pairs.add(pair_key)
        pair_state[pair_key] = plan.state

    for pair_key in decision_by_pair:
        if pair_key not in expected_pairs:
            raise ConsolidationSemanticError(
                f"relationship decision found for pair {pair_key!r} not present in "
                f"planning_result.relationship_pair_plans; invalid decision coverage"
            )

    for pair_key in expected_pairs:
        if pair_key not in decision_by_pair:
            raise ConsolidationSemanticError(
                f"no decision found for relationship pair {pair_key!r}; incomplete "
                f"relationship decision coverage"
            )

    for pair_key, state in pair_state.items():
        d = decision_by_pair[pair_key]
        if state == PAIR_STATE_AUTO_SAME:
            if d.method != "deterministic":
                raise ConsolidationSemanticError(
                    f"relationship pair {pair_key!r} (auto_same) has method "
                    f"{d.method!r}; expected 'deterministic'"
                )
            if d.decision != "same_relationship":
                raise ConsolidationSemanticError(
                    f"relationship pair {pair_key!r} (auto_same) has decision "
                    f"{d.decision!r}; expected 'same_relationship'"
                )
        elif state == PAIR_STATE_NEEDS_SEMANTIC_DECISION:
            if d.method != "llm":
                raise ConsolidationSemanticError(
                    f"relationship pair {pair_key!r} (needs_semantic_decision) has "
                    f"method {d.method!r}; expected 'llm'"
                )
            if d.decision not in RELATIONSHIP_DECISIONS:
                raise ConsolidationSemanticError(
                    f"relationship pair {pair_key!r} (needs_semantic_decision) has "
                    f"decision {d.decision!r}; expected one of "
                    f"{sorted(RELATIONSHIP_DECISIONS)}"
                )

    all_decisions.sort(key=lambda d: (d.left_candidate_ref, d.right_candidate_ref))
    return tuple(all_decisions)


# ---------------------------------------------------------------------------
# Main entry points (A5D-B)
# ---------------------------------------------------------------------------


def resolve_event_semantic_ambiguity(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    *,
    prompts: PromptRegistry,
) -> EventSemanticResolutionResult:
    """Execute A5D-B event semantic ambiguity resolution.

    Consumes the A5B ``ConsolidationPlanningResult`` and resolves every
    ``needs_semantic_decision`` event pair via bounded LLM semantic generation.

    Per block (block atomicity):
      * ``llm_client.generate_structured(...)`` (A-I3 owns the provider call);
      * provenance verification (FAIL CLOSED, no semantic retry);
      * ``EventSelectorDecisionPayload.from_dict(...)`` (typed load);
      * exact pair coverage/order + pair-local selector validation;
      * canonical selector order + exact endpoint EvidenceRef resolution;
      * ``EventSemanticDecision(method="llm")`` construction.

    Semantic rounds are bounded to ``max_generation_rounds`` (2). An ``LLMError``
    raised by the provider is PROPAGATED (not a semantic retry). Provenance
    mismatch fails closed with no retry. Two semantic-invalid rounds raise
    ``ConsolidationSemanticGenerationError``. ``uncertain`` is a valid
    successful semantic result and is NOT retried.
    """
    from .consolidation import EventSelectorDecisionPayload

    preparation = build_event_semantic_preparation(
        planning_result,
        consolidation_profile,
        semantic_profile,
        prompts=prompts,
        packing_policy=EVENT_SEMANTIC_PACKING_V1,
    )
    blocks = preparation.blocks

    deterministic_event_decisions = (
        planning_result.deterministic_decision_set.event_decisions
    )

    if not blocks:
        all_event_decisions = _validate_event_decision_coverage(
            planning_result, deterministic_event_decisions, ()
        )
        return EventSemanticResolutionResult(
            planning_result=planning_result,
            preparation=preparation,
            semantic_decisions=(),
            all_event_decisions=all_event_decisions,
            block_results=(),
        )

    all_semantic_decisions: list[EventSemanticDecision] = []
    all_block_results: list[EventSemanticBlockResult] = []

    for block, request in zip(blocks, preparation.structured_requests):
        rendered_prompt = request.rendered_prompt
        output_schema = request.output_schema

        endpoint_evidence = _event_block_endpoint_evidence(
            planning_result, block.pair_refs
        )

        max_rounds = consolidation_profile.max_generation_rounds
        block_decisions: list[EventSemanticDecision] = []
        block_provenance: LLMInvocationProvenance | None = None
        rounds_attempted = 0
        last_failure = ""

        for round_number in range(1, max_rounds + 1):
            rounds_attempted = round_number

            result = llm_client.generate_structured(
                rendered_prompt, output_schema, semantic_profile
            )

            _verify_semantic_provenance(result.provenance, request, semantic_profile)

            try:
                payload = EventSelectorDecisionPayload.from_dict(result.parsed_json)
            except ConsolidationModelError:
                last_failure = "typed payload load failed"
                continue

            is_valid, failure_detail, resolved_evidence = (
                _validate_selector_block_payload(
                    list(payload.decisions),
                    block.pair_refs,
                    endpoint_evidence,
                    _EVENT_EVIDENCE_SELECTOR_RE,
                    "event",
                )
            )
            if not is_valid:
                last_failure = failure_detail
                continue

            for item, resolved in zip(payload.decisions, resolved_evidence):
                block_decisions.append(
                    _convert_to_event_decision(
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
            break

        if block_decisions:
            all_semantic_decisions.extend(block_decisions)
            all_block_results.append(
                EventSemanticBlockResult(
                    block_id=block.block_id,
                    request_hash=request.request_hash,
                    semantic_rounds=rounds_attempted,
                    decisions=tuple(block_decisions),
                    generation_provenance=block_provenance,  # type: ignore[arg-type]
                )
            )
        else:
            raise ConsolidationSemanticGenerationError(
                block_id=block.block_id,
                request_hash=request.request_hash,
                rounds_attempted=rounds_attempted,
                last_failure_details=last_failure,
                expected_pairs=tuple(block.pair_refs),
            )

    all_event_decisions = _validate_event_decision_coverage(
        planning_result, deterministic_event_decisions, tuple(all_semantic_decisions)
    )

    return EventSemanticResolutionResult(
        planning_result=planning_result,
        preparation=preparation,
        semantic_decisions=tuple(all_semantic_decisions),
        all_event_decisions=all_event_decisions,
        block_results=tuple(all_block_results),
    )


def resolve_relationship_semantic_ambiguity(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    llm_client: LLMClient,
    *,
    prompts: PromptRegistry,
) -> RelationshipSemanticResolutionResult:
    """Execute A5D-B relationship semantic ambiguity resolution.

    Consumes the A5B ``ConsolidationPlanningResult`` and resolves every
    ``needs_semantic_decision`` relationship pair via bounded LLM semantic
    generation. Same block-atomicity and provenance rules as the event path.
    """
    from .consolidation import RelationshipSelectorDecisionPayload

    preparation = build_relationship_semantic_preparation(
        planning_result,
        consolidation_profile,
        semantic_profile,
        prompts=prompts,
        packing_policy=RELATIONSHIP_SEMANTIC_PACKING_V1,
    )
    blocks = preparation.blocks

    deterministic_rel_decisions = (
        planning_result.deterministic_decision_set.relationship_decisions
    )

    if not blocks:
        all_rel_decisions = _validate_relationship_decision_coverage(
            planning_result, deterministic_rel_decisions, ()
        )
        return RelationshipSemanticResolutionResult(
            planning_result=planning_result,
            preparation=preparation,
            semantic_decisions=(),
            all_relationship_decisions=all_rel_decisions,
            block_results=(),
        )

    all_semantic_decisions: list[RelationshipSemanticDecision] = []
    all_block_results: list[RelationshipSemanticBlockResult] = []

    for block, request in zip(blocks, preparation.structured_requests):
        rendered_prompt = request.rendered_prompt
        output_schema = request.output_schema

        endpoint_evidence = _relationship_block_endpoint_evidence(
            planning_result, block.pair_refs
        )

        max_rounds = consolidation_profile.max_generation_rounds
        block_decisions: list[RelationshipSemanticDecision] = []
        block_provenance: LLMInvocationProvenance | None = None
        rounds_attempted = 0
        last_failure = ""

        for round_number in range(1, max_rounds + 1):
            rounds_attempted = round_number

            result = llm_client.generate_structured(
                rendered_prompt, output_schema, semantic_profile
            )

            _verify_semantic_provenance(result.provenance, request, semantic_profile)

            try:
                payload = RelationshipSelectorDecisionPayload.from_dict(
                    result.parsed_json
                )
            except ConsolidationModelError:
                last_failure = "typed payload load failed"
                continue

            is_valid, failure_detail, resolved_evidence = (
                _validate_selector_block_payload(
                    list(payload.decisions),
                    block.pair_refs,
                    endpoint_evidence,
                    _RELATIONSHIP_EVIDENCE_SELECTOR_RE,
                    "relationship",
                )
            )
            if not is_valid:
                last_failure = failure_detail
                continue

            for item, resolved in zip(payload.decisions, resolved_evidence):
                block_decisions.append(
                    _convert_to_relationship_decision(
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
            break

        if block_decisions:
            all_semantic_decisions.extend(block_decisions)
            all_block_results.append(
                RelationshipSemanticBlockResult(
                    block_id=block.block_id,
                    request_hash=request.request_hash,
                    semantic_rounds=rounds_attempted,
                    decisions=tuple(block_decisions),
                    generation_provenance=block_provenance,  # type: ignore[arg-type]
                )
            )
        else:
            raise ConsolidationSemanticGenerationError(
                block_id=block.block_id,
                request_hash=request.request_hash,
                rounds_attempted=rounds_attempted,
                last_failure_details=last_failure,
                expected_pairs=tuple(block.pair_refs),
            )

    all_rel_decisions = _validate_relationship_decision_coverage(
        planning_result,
        deterministic_rel_decisions,
        tuple(all_semantic_decisions),
    )

    return RelationshipSemanticResolutionResult(
        planning_result=planning_result,
        preparation=preparation,
        semantic_decisions=tuple(all_semantic_decisions),
        all_relationship_decisions=all_rel_decisions,
        block_results=tuple(all_block_results),
    )
