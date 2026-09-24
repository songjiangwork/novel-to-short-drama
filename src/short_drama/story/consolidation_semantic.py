"""v1.2 A5C-A fact semantic preparation + packing audit (zero provider).

Implements A5C-A (Issue #52, BLOCK 1-6): build the A5C fact semantic
preparation from an A5B ``ConsolidationPlanningResult`` and audit the
deterministic block-packing candidates -- with NO provider call and NO
persistence anywhere in this module or the audit script.

* **Semantic stream** -- the fact semantic stream is the set of fact pair plans
  in state ``needs_semantic_decision``. ``auto_same`` fact pairs are already
  resolved deterministically by A5B and are never sent to the provider.

* **Deterministic block packing** -- the fact stream is packed deterministically
  in the A5B fact-pair-plan order (``(left_ref, right_ref)``), each block at
  most the requested maximum size. Three deterministic packing candidates are
  audited (``P1`` 6/12, ``P2`` 12/24, ``P3`` 24/48); the chosen production
  default is NOT frozen here.

* **Pair-local endpoint packets** -- each pair carries its own left/right
  endpoint fact packets, derived from the ``FactCandidate`` (no new
  decision/selector/profile model). Each endpoint packet carries the candidate
  identity, the normalized statement/type, the subject/object refs, the
  ``source_order_key``, and the source-anchored evidence items labeled with
  pair-local selectors (``L0/L1/...`` for the left endpoint, ``R0/R1/...`` for
  the right endpoint). There is NO block-wide evidence pool.

* **Real ``StructuredGenerationRequest`` objects** -- every block is rendered
  through the existing ``PromptRegistry`` / ``OutputSchema`` /
  ``SemanticLLMProfile`` infrastructure: the tracked ``a5.fact-consolidation``
  prompt v1 (``required_variables`` = exactly ``block_id`` +
  ``pair_contexts_json``), the ``consolidation-fact-selector-payload`` schema
  v1, and the ``consolidation-llm-v1`` semantic profile. The ``block_id`` is
  ``a5fblk_`` + the first 20 hex chars of the sha256 of the canonical block
  payload. Stable per-request hashes are computed before any provider call.

The profile/prompt/schema/semantic-profile identity is verified fail closed.
Nothing in this module calls a provider or persists anything; the only provider
boundary is the rendered ``StructuredGenerationRequest`` set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

from .consolidation import ConsolidationProfile, ConsolidationSemanticPass
from .consolidation_planning import (
    ConsolidationPlanningResult,
    FactPairPlan,
    IndexedFactCandidate,
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
)
from .errors import StoryIntegrityError
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
#: First N hex chars of the block sha256 that identify a block (A5C-A BLOCK 1).
A5C_BLOCK_ID_HEX_LENGTH = 20

#: Default requested maximum fact block size (NOT a frozen production default).
A5C_DEFAULT_FACT_BLOCK_SIZE = 24

#: Deterministic block-packing candidates audited by A5C-A (name, min, max).
#: These are audit candidates only; A5C-A does not freeze the production default.
A5C_PACKING_CANDIDATES: tuple[tuple[str, int, int], ...] = (
    ("P1", 6, 12),
    ("P2", 12, 24),
    ("P3", 24, 48),
)


# ---------------------------------------------------------------------------
# Fact semantic block (A5C-A BLOCK 1 / BLOCK 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactSemanticBlock:
    """One deterministic fact consolidation block (pair-local contexts).

    ``pair_contexts`` is the exact canonical block payload: one pair-context
    dict per pair, in the A5B fact-pair-plan order. ``pair_contexts_json`` is
    the RFC 8785 canonical JSON string of that payload (the prompt variable),
    and ``block_id`` is ``a5fblk_`` + the first 20 hex chars of the sha256 of
    the same canonical payload.
    """

    block_id: str
    pair_contexts: tuple[dict[str, Any], ...]
    pair_contexts_json: str
    pair_refs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.block_id.startswith(A5C_BLOCK_PREFIX):
            raise StoryIntegrityError(
                f"fact block_id must start with {A5C_BLOCK_PREFIX!r}: {self.block_id!r}"
            )
        if self.block_id != A5C_BLOCK_PREFIX + content_hash(list(self.pair_contexts))[: A5C_BLOCK_ID_HEX_LENGTH]:
            raise StoryIntegrityError("fact block_id does not match the canonical block payload")
        if tuple(self.pair_refs) != tuple(
            (pc["left_candidate_ref"], pc["right_candidate_ref"]) for pc in self.pair_contexts
        ):
            raise StoryIntegrityError("fact block pair_refs does not match the pair contexts")

    @property
    def pair_count(self) -> int:
        return len(self.pair_contexts)

    @property
    def payload_bytes(self) -> int:
        return len(self.pair_contexts_json.encode("utf-8"))


def _pack_fact_blocks(pair_contexts: list[dict[str, Any]], max_block_size: int) -> tuple[FactSemanticBlock, ...]:
    """Deterministically pack pair contexts into blocks of at most ``max_block_size``."""
    if max_block_size < 1:
        raise StoryIntegrityError("max_block_size must be >= 1")
    blocks: list[FactSemanticBlock] = []
    for start in range(0, len(pair_contexts), max_block_size):
        payload = pair_contexts[start : start + max_block_size]
        pair_refs = tuple(
            (pc["left_candidate_ref"], pc["right_candidate_ref"]) for pc in payload
        )
        payload_json = canonical_json_bytes(payload).decode("utf-8")
        block_id = A5C_BLOCK_PREFIX + content_hash(payload)[:A5C_BLOCK_ID_HEX_LENGTH]
        blocks.append(
            FactSemanticBlock(
                block_id=block_id,
                pair_contexts=tuple(payload),
                pair_contexts_json=payload_json,
                pair_refs=pair_refs,
            )
        )
    return tuple(blocks)


# ---------------------------------------------------------------------------
# Fact semantic preparation result (A5C-A BLOCK 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactSemanticPreparation:
    """Immutable fact semantic preparation (A5B plan + blocks + requests + hashes).

    Carries the A5B planning result, the verified consolidation / semantic
    profile identities, the prompt / schema identities, the deterministic
    fact blocks, the rendered ``StructuredGenerationRequest`` objects, and the
    stable per-request hashes. No provider is called and nothing is persisted.
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
    max_block_size: int
    blocks: tuple[FactSemanticBlock, ...]
    structured_requests: tuple[StructuredGenerationRequest, ...]
    semantic_request_hashes: tuple[str, ...]
    semantic_pair_count: int
    auto_same_pair_count: int
    total_fact_pair_count: int

    def __post_init__(self) -> None:
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
    """Fail closed unless the fact pass pins the exact A5C-A identity."""
    if not isinstance(consolidation_profile, ConsolidationProfile):
        raise StoryIntegrityError("consolidation_profile must be a ConsolidationProfile")
    if not isinstance(semantic_profile, SemanticLLMProfile):
        raise StoryIntegrityError("semantic_profile must be a SemanticLLMProfile")
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


def _evidence_sort_key(ev: EvidenceRef) -> tuple[str, ...]:
    return (ev.paragraph_id, ev.role, ev.strength, ev.excerpt)


def build_fact_endpoint_packet(cand: IndexedFactCandidate, side: str) -> dict[str, Any]:
    """Build the pair-local endpoint packet for one side of a fact pair.

    ``side`` is ``"left"`` or ``"right"``; the evidence items are labeled with
    pair-local selectors ``L0/L1/...`` (left) or ``R0/R1/...`` (right) in the
    canonical evidence order. The packet is derived from the indexed candidate
    and carries the ``source_order_key``.
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
        for idx, ev in enumerate(sorted(cand.evidence_refs, key=_evidence_sort_key))
    ]
    return {
        "candidate_ref": cand.global_candidate_ref,
        "chunk_id": cand.chunk_id,
        "local_candidate_id": cand.local_candidate_id,
        "fact_type": cand.fact_type,
        "statement_zh": cand.statement_zh,
        "subject_refs": list(cand.subject_refs),
        "object_refs": list(cand.object_refs),
        "source_order_key": cand.source_order_key,
        "evidence": evidence,
    }


def build_fact_pair_context(
    plan: FactPairPlan, facts_by_ref: dict[str, IndexedFactCandidate]
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
# Fact semantic preparation builder (A5C-A BLOCK 3 / BLOCK 5 / BLOCK 6)
# ---------------------------------------------------------------------------


def build_fact_semantic_preparation(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
    max_block_size: int = A5C_DEFAULT_FACT_BLOCK_SIZE,
) -> FactSemanticPreparation:
    """Build the fact semantic preparation from an A5B planning result.

    Selects the ``needs_semantic_decision`` fact pairs (the A5C fact semantic
    stream), builds the deterministic pair-local pair contexts, packs them into
    blocks of at most ``max_block_size``, and renders one real
    ``StructuredGenerationRequest`` per block via the existing LLM
    infrastructure. No provider is called and nothing is persisted.
    """
    _verify_fact_profile(consolidation_profile, semantic_profile)
    prompt = prompts.load(A5C_FACT_PROMPT_ID, version=A5C_FACT_PROMPT_VERSION)
    output_schema = load_fact_output_schema()

    facts_by_ref = {
        cand.global_candidate_ref: cand for cand in planning_result.index.facts
    }
    fact_plans = planning_result.fact_pair_plans
    semantic_plans = [
        plan
        for plan in fact_plans
        if plan.state == PAIR_STATE_NEEDS_SEMANTIC_DECISION
    ]
    auto_same_count = sum(
        1 for plan in fact_plans if plan.state == PAIR_STATE_AUTO_SAME
    )

    pair_contexts = [
        build_fact_pair_context(plan, facts_by_ref) for plan in semantic_plans
    ]
    blocks = _pack_fact_blocks(pair_contexts, max_block_size)

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
        max_block_size=max_block_size,
        blocks=blocks,
        structured_requests=tuple(structured_requests),
        semantic_request_hashes=request_hashes,
        semantic_pair_count=len(semantic_plans),
        auto_same_pair_count=auto_same_count,
        total_fact_pair_count=len(fact_plans),
    )


# ---------------------------------------------------------------------------
# Fact semantic identity (A5C-A BLOCK 4)
# ---------------------------------------------------------------------------


def build_fact_semantic_identity(prep: FactSemanticPreparation) -> dict[str, Any]:
    """Build the canonical A5C fact semantic identity material.

    The identity binds the A5B planning identity (policy ids + plan hash), the
    verified consolidation / semantic profile identities, the prompt / schema
    identities, and the exact request set (max block size + per-request hashes).
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
        "max_block_size": prep.max_block_size,
        "semantic_pair_count": prep.semantic_pair_count,
        "auto_same_pair_count": prep.auto_same_pair_count,
        "total_fact_pair_count": prep.total_fact_pair_count,
        "block_count": len(prep.blocks),
        "block_ids": tuple(block.block_id for block in prep.blocks),
        "semantic_request_hashes": tuple(prep.semantic_request_hashes),
    }


# ---------------------------------------------------------------------------
# Deterministic block-packing audit (A5C-A BLOCK 5)
# ---------------------------------------------------------------------------


def _largest_block_id(blocks: tuple[FactSemanticBlock, ...]) -> str | None:
    """The block id of the largest block (by payload bytes; tie -> lexicographic)."""
    if not blocks:
        return None
    max_bytes = max(block.payload_bytes for block in blocks)
    largest = [block for block in blocks if block.payload_bytes == max_bytes]
    return max(block.block_id for block in largest)


def build_fact_packing_audit(
    planning_result: ConsolidationPlanningResult,
    consolidation_profile: ConsolidationProfile,
    semantic_profile: SemanticLLMProfile,
    *,
    prompts: PromptRegistry,
    candidates: tuple[tuple[str, int, int], ...] = A5C_PACKING_CANDIDATES,
) -> tuple[dict[str, Any], ...]:
    """Audit deterministic fact block-packing candidates (zero provider).

    For each ``(name, min_block_size, max_block_size)`` candidate, build the
    fact semantic preparation with the candidate's requested maximum block size
    and report the deterministic packing statistics. The audit candidates do
    NOT freeze the production default.
    """
    audit: list[dict[str, Any]] = []
    for name, min_block_size, max_block_size in candidates:
        prep = build_fact_semantic_preparation(
            planning_result,
            consolidation_profile,
            semantic_profile,
            prompts=prompts,
            max_block_size=max_block_size,
        )
        block_sizes = [block.pair_count for block in prep.blocks]
        block_bytes = [block.payload_bytes for block in prep.blocks]
        audit.append(
            {
                "candidate": name,
                "min_block_size": min_block_size,
                "requested_max_block_size": max_block_size,
                "total_fact_semantic_pairs": prep.semantic_pair_count,
                "block_count": len(prep.blocks),
                "min_block_size_actual": min(block_sizes) if block_sizes else 0,
                "avg_block_size": (sum(block_sizes) / len(block_sizes)) if block_sizes else 0.0,
                "max_block_size_actual": max(block_sizes) if block_sizes else 0,
                "max_block_bytes": max(block_bytes) if block_bytes else 0,
                "largest_block_id": _largest_block_id(prep.blocks),
                "total_request_count": len(prep.blocks),
                "request_hashes": tuple(prep.semantic_request_hashes),
            }
        )
    return tuple(audit)
