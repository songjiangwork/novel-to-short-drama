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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from short_drama.artifacts.canonical import canonical_json_bytes, content_hash
from short_drama.io import load_json
from short_drama.llm import (
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
    ConsolidationCandidateRef,
    ConsolidationProfile,
    ConsolidationSemanticPass,
)
from .consolidation_planning import (
    ConsolidationPlanningResult,
    FactPairPlan,
    IndexedFactCandidate,
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
)
from .errors import ConsolidationModelError, StoryIntegrityError

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


def _canonical_candidate_refs(
    refs: set[str], facts_by_ref: Mapping[str, IndexedFactCandidate]
) -> tuple[str, ...]:
    """Stable unique union ordered by candidate source_order_key then ref."""
    keyed: list[tuple[str, str]] = []
    for ref in refs:
        cand = facts_by_ref.get(ref)
        if cand is None:
            raise StoryIntegrityError(
                f"candidate ref {ref!r} is not in the fact candidate index"
            )
        keyed.append((cand.source_order_key, ref))
    keyed.sort()
    return tuple(ref for _so, ref in keyed)


def _compute_block_id(
    plan_hash: str,
    policy: FactSemanticPackingPolicy,
    block_ordinal: int,
    pair_refs: tuple[tuple[str, str], ...],
    candidate_refs: tuple[str, ...],
) -> str:
    """A5C-A BLOCK 4: bind the block id to the frozen material (no runtime data)."""
    material = {
        "plan_hash": plan_hash,
        "packing_policy": policy.to_dict(),
        "block_ordinal": block_ordinal,
        "ordered_requested_pairs": [list(pair) for pair in pair_refs],
        "ordered_candidate_refs": list(candidate_refs),
    }
    return A5C_BLOCK_PREFIX + content_hash(material)[:A5C_BLOCK_ID_HEX_LENGTH]


def _build_fact_blocks(
    policy: FactSemanticPackingPolicy,
    plan_hash: str,
    facts_by_ref: Mapping[str, IndexedFactCandidate],
    semantic_stream: tuple[FactPairPlan, ...],
) -> tuple[FactSemanticBlock, ...]:
    """Deterministically pack the semantic stream under BOTH policy limits.

    The stream is consumed in its canonical ``(left_ref, right_ref)`` order. A
    block is closed when appending the next pair would exceed either the pair
    limit or the unique-candidate limit. No pair is split, lost, duplicated, or
    reordered.
    """
    blocks: list[FactSemanticBlock] = []
    current_pairs: list[FactPairPlan] = []
    current_candidates: set[str] = set()

    def _close() -> None:
        nonlocal current_pairs, current_candidates
        if not current_pairs:
            return
        ordinal = len(blocks)
        pair_refs = tuple((p.left_ref, p.right_ref) for p in current_pairs)
        candidate_refs = _canonical_candidate_refs(current_candidates, facts_by_ref)
        pair_contexts = tuple(
            build_fact_pair_context(p, facts_by_ref) for p in current_pairs
        )
        blocks.append(
            FactSemanticBlock(
                block_ordinal=ordinal,
                block_id=_compute_block_id(plan_hash, policy, ordinal, pair_refs, candidate_refs),
                pair_contexts=pair_contexts,
                candidate_refs=candidate_refs,
                pair_refs=pair_refs,
                pair_contexts_json=canonical_json_bytes(
                    [dict(pc) for pc in pair_contexts]
                ).decode("utf-8"),
            )
        )
        current_pairs = []
        current_candidates = set()

    for plan in semantic_stream:
        pair_candidates = {plan.left_ref, plan.right_ref}
        if current_pairs and (
            len(current_pairs) + 1 > policy.max_pairs_per_block
            or len(current_candidates | pair_candidates) > policy.max_candidates_per_block
        ):
            _close()
        current_pairs.append(plan)
        current_candidates |= pair_candidates
    _close()
    return tuple(blocks)


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
