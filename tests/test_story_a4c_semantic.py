"""A4C semantic ambiguity resolution tests (Issue #41: pair-local evidence
selectors).

Covers the frozen A4C selector contract:

  * block packing (0/1/6/7 pairs, candidate cap, stable IDs, every pair once)
  * pair context (endpoint-only, source-order, exact evidence, null excerpt,
    canonical JSON, selector labels)
  * deterministic selector validation (L/R form, index range, duplicates,
    no-third-candidate) + exact EvidenceRef resolution (Python-owned)
  * decisions (same/different/uncertain accepted)
  * uncertain → exactly 1 semantic call, no retry
  * invalid output / invalid selector retry (missing/extra/duplicate/reordered/
    wrong ref, invalid selector form/range/duplicate)
  * first invalid + second valid → success, 2 semantic calls
  * 2 invalid → ReconciliationSemanticGenerationError
  * LLMError → propagate, semantic call count = 1
  * provenance mismatch → fail closed, no second call
  * backend neutrality (provider/model metadata does not change IDs)
  * profile consistency gate
  * complete decision coverage
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from short_drama.artifacts import ArtifactRef
from short_drama.llm import (
    LLMClient,
    LLMError,
    LLMInvocationProvenance,
    LLMRetryExhaustedError,
    OutputSchema,
    PromptRegistry,
    PromptSpec,
    ReasoningSettings,
    SemanticLLMProfile,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    build_structured_request,
    build_provenance,
    render_prompt,
    validate_against_output_schema,
)
from short_drama.llm.openai_compatible import ProviderMeta
from short_drama.story import (
    EntityReconciliationProfile,
    EvidenceRef,
    MAX_CANDIDATES_PER_BLOCK,
    MAX_PAIRS_PER_BLOCK,
    PAIR_STATE_AUTO_SAME,
    PAIR_STATE_MUST_NOT_MERGE,
    PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    ReconciliationDecision,
    ReconciliationInputSnapshot,
    ReconciliationModelError,
    ReconciliationPairPlan,
    ReconciliationPlanningError,
    ReconciliationPlanningResult,
    ReconciliationProvenanceError,
    ReconciliationSemanticError,
    ReconciliationSemanticGenerationError,
    ReconciliationSelectorDecisionItem,
    ReconciliationSelectorDecisionPayload,
    resolve_semantic_ambiguity,
)
from short_drama.story.reconciliation import CandidateEntityIndex, CandidateEntityIndexEntry
from short_drama.story.reconciliation_semantic import (
    MAX_CANDIDATES_PER_BLOCK as MAX_CANDS,
    MAX_PAIRS_PER_BLOCK as MAX_PAIRS,
    _block_endpoint_evidence,
    _build_blocks,
    _build_pair_contexts,
    _pack_semantic_blocks,
    _validate_decision_coverage,
    _validate_selector_block_payload,
    _verify_provenance,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H = "a" * 64
H2 = "b" * 64
H3 = "c" * 64

# A valid chunk id for global candidate refs
CHUNK_ID = "CH001_C001"


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def make_artifact_ref(artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_type="candidate_extraction",
        artifact_id=artifact_id,
        revision=1,
        content_hash="a" * 64,
    )


def make_candidate_entry(
    ref: str,
    kind: str = "character",
    display_name: str = "Test",
    source_order_key: str = "000001:000000001:01:000000001:ref",
    aliases: tuple[str, ...] = (),
    descriptors_zh: tuple[str, ...] = (),
    evidence: tuple[EvidenceRef, ...] = (),
) -> CandidateEntityIndexEntry:
    return CandidateEntityIndexEntry(
        candidate_ref=ref,
        candidate_kind=kind,
        candidate_extraction_ref=make_artifact_ref(f"ext_{ref[:20]}"),
        source_order_key=source_order_key,
        display_name_original=display_name,
        aliases_original=aliases,
        descriptors_zh=descriptors_zh,
        evidence_refs=evidence,
        possible_candidate_refs=(),
    )


def make_pair_plan(
    left: str,
    right: str,
    state: str = PAIR_STATE_NEEDS_SEMANTIC_DECISION,
    signals: tuple[str, ...] = ("identity_token_overlap",),
    shared_keys: tuple[str, ...] = (),
    shared_tokens: tuple[str, ...] = (),
) -> ReconciliationPairPlan:
    return ReconciliationPairPlan(
        left_candidate_ref=left,
        right_candidate_ref=right,
        state=state,
        signals=signals,
        shared_identity_keys=shared_keys,
        shared_tokens=shared_tokens,
    )


def make_planning_result(
    entries: tuple[CandidateEntityIndexEntry, ...],
    pair_plans: tuple[ReconciliationPairPlan, ...],
    decisions: tuple[ReconciliationDecision, ...] = (),
    plan_hash: str = H,
) -> ReconciliationPlanningResult:
    return ReconciliationPlanningResult(
        candidate_index=CandidateEntityIndex(schema_version=1, entries=entries),
        pair_plans=pair_plans,
        decisions=decisions,
        normalization_policy_id="a4-name-normalization-v1",
        blocking_policy_id="a4-blocking-v1",
        canonicalization_policy_id="a4-canonicalization-v1",
        plan_hash=plan_hash,
    )


def make_profile(
    prompt_id: str = "a4.entity-reconciliation",
    prompt_version: int = 3,
    output_schema_id: str = "a4-reconciliation-decision-selector-payload",
    output_schema_version: int = 1,
    max_generation_rounds: int = 2,
) -> EntityReconciliationProfile:
    return EntityReconciliationProfile(
        schema_version=1,
        profile_id="entity-reconciliation-v2",
        working_language="zh-CN",
        name_normalization_policy_id="a4-name-normalization-v1",
        blocking_policy_id="a4-blocking-v1",
        canonicalization_policy_id="a4-canonicalization-v1",
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        output_schema_id=output_schema_id,
        output_schema_version=output_schema_version,
        max_generation_rounds=max_generation_rounds,
    )


def make_semantic_profile() -> SemanticLLMProfile:
    return SemanticLLMProfile(
        schema_version=2,
        profile_id="entity-reconciliation-llm-v1",
        temperature=0.0,
        max_output_tokens=4096,
        structured_output_mode="json_schema",
        reasoning=ReasoningSettings(enabled=False),
    )


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """Deterministic fake provider for A4C offline tests.

    Scripted responses:
      * a ``dict`` → successful result with matching provenance
      * a ``(dict, LLMInvocationProvenance)`` tuple → successful result with exact provenance
      * an ``Exception`` → raised from generate_structured
    """

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.calls = []
        self.request_hashes = []
        self.provider_family = "qwen"
        self.request_model = "qwen3-27b"

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.call_count += 1
        self.calls.append((rendered_prompt, output_schema, semantic_profile))
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        self.request_hashes.append(request.request_hash)
        if not self.responses:
            raise AssertionError("unexpected extra generate_structured call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            parsed, provenance = response
        else:
            parsed = response
            provenance = build_provenance(
                request,
                ProviderMeta(),
                request_model=self.request_model,
                provider_family=self.provider_family,
            )
        # Trust boundary: validate against output schema (lenient: selectors are
        # plain strings; pair-local validity is enforced by the A4C validator).
        validate_against_output_schema(parsed, output_schema)
        return StructuredGenerationResult(
            parsed_json=parsed, provenance=provenance, attempts=1
        )


# ---------------------------------------------------------------------------
# Output schema fixture (shared)
# ---------------------------------------------------------------------------

SCHEMA_PATH = (
    __import__("pathlib").Path(__file__).parent.parent
    / "schemas"
    / "reconciliation-decision-selector-payload.schema.json"
)


def make_output_schema() -> OutputSchema:
    from short_drama.io import load_json

    schema_data = load_json(SCHEMA_PATH)
    return OutputSchema.create(
        schema_id="a4-reconciliation-decision-selector-payload",
        schema_version=1,
        schema=schema_data,
    )


# ---------------------------------------------------------------------------
# Prompt registry fixture (use tracked prompt)
# ---------------------------------------------------------------------------

PROMPT_REGISTRY = PromptRegistry(
    __import__("pathlib").Path(__file__).parent.parent
    / "prompts"
    / "story"
)


# ---------------------------------------------------------------------------
# Test: Block packing
# ---------------------------------------------------------------------------


class TestBlockPacking:
    def test_zero_pairs_zero_blocks(self):
        """0 semantic pairs → 0 blocks."""
        entries = (
            make_candidate_entry(f"{CHUNK_ID}:cand_char_001"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002"),
        )
        pair_plans = (
            make_pair_plan(
                f"{CHUNK_ID}:cand_char_001",
                f"{CHUNK_ID}:cand_char_002",
                state=PAIR_STATE_AUTO_SAME,
            ),
        )
        result = make_planning_result(entries, pair_plans)
        blocks = _build_blocks(result)
        assert len(blocks) == 0

    def test_one_pair_one_block(self):
        """1 pair → 1 block."""
        entries = (
            make_candidate_entry(f"{CHUNK_ID}:cand_char_001", source_order_key="000001:000000001:01:000000001:r1"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002", source_order_key="000001:000000002:01:000000001:r2"),
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans)
        blocks = _build_blocks(result)
        assert len(blocks) == 1
        assert len(blocks[0].pair_plans) == 1

    def test_six_pairs_twelve_candidates_one_block(self):
        """6 disjoint pairs / 12 candidates → 1 block."""
        entries = []
        pair_plans = []
        for i in range(6):
            left = f"{CHUNK_ID}:cand_char_{i * 2 + 1:03d}"
            right = f"{CHUNK_ID}:cand_char_{i * 2 + 2:03d}"
            entries.append(
                make_candidate_entry(left, source_order_key=f"000001:00000000{i + 1}:01:00000000{i + 1}:l{i}")
            )
            entries.append(
                make_candidate_entry(right, source_order_key=f"000001:00000001{i + 1}:01:00000001{i + 1}:r{i}")
            )
            pair_plans.append(make_pair_plan(left, right))
        entries_tuple = tuple(entries)
        pair_plans_tuple = tuple(pair_plans)
        result = make_planning_result(entries_tuple, pair_plans_tuple)
        blocks = _build_blocks(result)
        assert len(blocks) == 1
        assert len(blocks[0].pair_plans) == 6
        assert len(blocks[0].candidate_refs) == 12

    def test_seven_pairs_two_blocks(self):
        """7 pairs → 2 blocks (6 + 1)."""
        entries = []
        pair_plans = []
        for i in range(7):
            left = f"{CHUNK_ID}:cand_char_{i * 2 + 1:03d}"
            right = f"{CHUNK_ID}:cand_char_{i * 2 + 2:03d}"
            entries.append(
                make_candidate_entry(left, source_order_key=f"000001:00000000{i + 1}:01:00000000{i + 1}:l{i}")
            )
            entries.append(
                make_candidate_entry(right, source_order_key=f"000001:00000001{i + 1}:01:00000001{i + 1}:r{i}")
            )
            pair_plans.append(make_pair_plan(left, right))
        entries_tuple = tuple(entries)
        pair_plans_tuple = tuple(pair_plans)
        result = make_planning_result(entries_tuple, pair_plans_tuple)
        blocks = _build_blocks(result)
        assert len(blocks) == 2
        assert len(blocks[0].pair_plans) == 6
        assert len(blocks[1].pair_plans) == 1

    def test_block_ids_are_stable(self):
        """Block IDs are deterministic for the same input."""
        entries = (
            make_candidate_entry(f"{CHUNK_ID}:cand_char_001"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002"),
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans)
        blocks1 = _build_blocks(result)
        blocks2 = _build_blocks(result)
        assert blocks1[0].block_id == blocks2[0].block_id

    def test_every_pair_appears_exactly_once(self):
        """Every semantic pair appears in exactly one block."""
        entries = []
        pair_plans = []
        for i in range(10):
            left = f"{CHUNK_ID}:cand_char_{i * 2 + 1:03d}"
            right = f"{CHUNK_ID}:cand_char_{i * 2 + 2:03d}"
            entries.append(
                make_candidate_entry(left, source_order_key=f"000001:00000000{i + 1}:01:00000000{i + 1}:l{i}")
            )
            entries.append(
                make_candidate_entry(right, source_order_key=f"000001:00000001{i + 1}:01:00000001{i + 1}:r{i}")
            )
            pair_plans.append(make_pair_plan(left, right))
        entries_tuple = tuple(entries)
        pair_plans_tuple = tuple(pair_plans)
        result = make_planning_result(entries_tuple, pair_plans_tuple)
        blocks = _build_blocks(result)

        # 10 pairs → 2 blocks (6 + 4)
        assert len(blocks) == 2

        # Count pair occurrences across all blocks
        pair_count = 0
        for block in blocks:
            pair_count += len(block.pair_plans)
        assert pair_count == 10


# ---------------------------------------------------------------------------
# Test: Pair context
# ---------------------------------------------------------------------------


class TestPairContext:
    def test_endpoint_only(self):
        """Only pair endpoints are included in the pair contexts."""
        entries = (
            make_candidate_entry(f"{CHUNK_ID}:cand_char_001", source_order_key="000001:000000001:01:000000001:r1"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002", source_order_key="000001:000000002:01:000000001:r2"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_003", source_order_key="000001:000000003:01:000000001:r3"),  # not in pair
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans)
        blocks = _build_blocks(result)
        contexts = json.loads(blocks[0].pair_contexts_json)
        refs = set()
        for pc in contexts:
            refs.add(pc["left_endpoint"]["candidate_ref"])
            refs.add(pc["right_endpoint"]["candidate_ref"])
        assert refs == {f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"}
        assert f"{CHUNK_ID}:cand_char_003" not in refs

    def test_source_order(self):
        """Pair contexts are in source_order_key order (block_id stability)."""
        entries = (
            make_candidate_entry(f"{CHUNK_ID}:cand_char_001", source_order_key="000001:000000003:01:000000003:r1"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002", source_order_key="000001:000000001:01:000000001:r2"),
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans)
        contexts_json, candidate_refs = _build_pair_contexts(result, list(pair_plans))
        # Pair endpoints keep their pair roles (left/right), but the block_id
        # candidate refs are in source_order_key order (r2 < r1).
        assert candidate_refs == (
            f"{CHUNK_ID}:cand_char_002",
            f"{CHUNK_ID}:cand_char_001",
        )
        contexts = json.loads(contexts_json)
        assert contexts[0]["left_endpoint"]["candidate_ref"] == f"{CHUNK_ID}:cand_char_001"
        assert contexts[0]["right_endpoint"]["candidate_ref"] == f"{CHUNK_ID}:cand_char_002"

    def test_exact_fields(self):
        """Each endpoint has exactly the frozen fields."""
        evidence = (
            EvidenceRef(
                paragraph_id="CH001_P001",
                role="primary",
                strength="explicit",
                excerpt="Alice entered the room.",
            ),
        )
        entries = (
            make_candidate_entry(
                f"{CHUNK_ID}:cand_char_001",
                display_name="Alice",
                aliases=("Alicia",),
                descriptors_zh=("女主角",),
                evidence=evidence,
            ),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002", display_name="Bob"),
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans)
        contexts_json, _ = _build_pair_contexts(result, list(pair_plans))
        contexts = json.loads(contexts_json)
        for pc in contexts:
            for side in ("left_endpoint", "right_endpoint"):
                assert set(pc[side].keys()) == {
                    "candidate_ref",
                    "candidate_kind",
                    "display_name_original",
                    "aliases_original",
                    "descriptors_zh",
                    "evidence",
                }
        # Verify exact evidence preserved (null round-trips)
        alice_side = next(
            pc for pc in contexts
            if pc["left_endpoint"]["display_name_original"] == "Alice"
            or pc["right_endpoint"]["display_name_original"] == "Alice"
        )
        alice = alice_side["left_endpoint"] if alice_side["left_endpoint"]["display_name_original"] == "Alice" else alice_side["right_endpoint"]
        assert alice["aliases_original"] == ["Alicia"]
        assert alice["descriptors_zh"] == ["女主角"]
        assert alice["evidence"][0]["paragraph_id"] == "CH001_P001"
        assert alice["evidence"][0]["excerpt"] == "Alice entered the room."

    def test_canonical_json_stable(self):
        """Pair contexts JSON is stable across runs."""
        entries = (
            make_candidate_entry(f"{CHUNK_ID}:cand_char_001"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002"),
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans)
        blocks1 = _build_blocks(result)
        blocks2 = _build_blocks(result)
        assert blocks1[0].pair_contexts_json == blocks2[0].pair_contexts_json

    def test_null_excerpt_preserved(self):
        """A null endpoint excerpt is preserved in the pair context (round-trips)."""
        entries = (
            make_candidate_entry(
                f"{CHUNK_ID}:cand_char_001",
                evidence=(EvidenceRef("CH001_P001", "primary", "explicit", None),),
            ),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002", display_name="B"),
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans)
        contexts_json, _ = _build_pair_contexts(result, list(pair_plans))
        contexts = json.loads(contexts_json)
        ev = contexts[0]["left_endpoint"]["evidence"][0]
        assert ev["paragraph_id"] == "CH001_P001"
        assert ev["excerpt"] is None

    def test_no_unresolved(self):
        """Unresolved candidates never appear in semantic pair contexts."""
        entries = (
            make_candidate_entry(f"{CHUNK_ID}:cand_char_001", kind="character"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002", kind="character"),
            make_candidate_entry(f"{CHUNK_ID}:cand_unres_001", kind="unresolved_person"),
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans)
        blocks = _build_blocks(result)
        contexts = json.loads(blocks[0].pair_contexts_json)
        refs = set()
        for pc in contexts:
            refs.add(pc["left_endpoint"]["candidate_ref"])
            refs.add(pc["right_endpoint"]["candidate_ref"])
        assert f"{CHUNK_ID}:cand_unres_001" not in refs


# ---------------------------------------------------------------------------
# Test: Decisions (same / different / uncertain)
# ---------------------------------------------------------------------------


class TestDecisions:
    def _make_valid_payload(self, decisions: list[dict[str, Any]]) -> dict:
        return {"decisions": decisions}

    def test_same_entity_accepted(self):
        """A valid same_entity decision (with a selector) is accepted."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        evidence = EvidenceRef(
            paragraph_id="CH001_P001", role="primary", strength="explicit",
            excerpt="Alice, also known as Alicia, entered the room.",
        )
        entries = (
            make_candidate_entry(left, display_name="Alice", evidence=(evidence,)),
            make_candidate_entry(right, display_name="Alicia"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        payload = self._make_valid_payload([
            {
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": "same_entity",
                "reason_zh": "两人是同一角色。",
                "evidence_selectors": ["L0"],
            }
        ])

        client = FakeLLMClient([payload])
        profile = make_profile()
        sem_profile = make_semantic_profile()
        res = resolve_semantic_ambiguity(
            result, profile, sem_profile, client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 1
        assert len(res.semantic_decisions) == 1
        assert res.semantic_decisions[0].decision == "same_entity"
        assert res.semantic_decisions[0].reason_code == "llm_same_entity"
        assert res.semantic_decisions[0].method == "llm"
        # Selector resolved to the exact left endpoint EvidenceRef.
        assert res.semantic_decisions[0].evidence_refs[0].paragraph_id == "CH001_P001"

    def test_different_entity_accepted(self):
        """A valid different_entity decision is accepted."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="Bob"),
            make_candidate_entry(right, display_name="Carol"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        payload = self._make_valid_payload([
            {
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": "different_entity",
                "reason_zh": "两人是不同角色。",
                "evidence_selectors": [],
            }
        ])

        client = FakeLLMClient([payload])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 1
        assert res.semantic_decisions[0].decision == "different_entity"
        assert res.semantic_decisions[0].reason_code == "llm_different_entity"

    def test_uncertain_accepted(self):
        """A valid uncertain decision is accepted (SUCCESS, no retry)."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="The doctor"),
            make_candidate_entry(right, display_name="Dr. Chen"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        payload = self._make_valid_payload([
            {
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": "uncertain",
                "reason_zh": "证据不足。",
                "evidence_selectors": [],
            }
        ])

        client = FakeLLMClient([payload])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 1  # exactly 1 call, no retry
        assert res.semantic_decisions[0].decision == "uncertain"
        assert res.semantic_decisions[0].reason_code == "llm_uncertain"


# ---------------------------------------------------------------------------
# Test: Invalid output retry
# ---------------------------------------------------------------------------


class TestInvalidOutputRetry:
    def _entries(self):
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        return (
            make_candidate_entry(left, display_name="Alice"),
            make_candidate_entry(right, display_name="Alicia"),
        ), left, right

    def _valid_payload(self, left, right, decision="same_entity") -> dict:
        return {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": decision,
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                }
            ]
        }

    def test_missing_pair_retries(self):
        """Missing pair (count mismatch) → retry."""
        entries, left, right = self._entries()
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        invalid = {"decisions": []}
        valid = self._valid_payload(left, right)

        client = FakeLLMClient([invalid, valid])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 2
        assert len(res.semantic_decisions) == 1
        assert res.block_results[0].semantic_rounds == 2

    def test_extra_pair_retries(self):
        """Extra pair (count mismatch) → retry."""
        entries, left, right = self._entries()
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        invalid = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                },
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                },
            ]
        }
        valid = self._valid_payload(left, right)

        client = FakeLLMClient([invalid, valid])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 2
        assert len(res.semantic_decisions) == 1

    def test_invalid_selector_form_retries(self):
        """Invalid selector prefix (X0) → A4C semantic validation failure → retry."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A", evidence=(
                EvidenceRef("CH001_P001", "primary", "explicit", "A."),)),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        invalid = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": ["X0"],  # invalid prefix
                }
            ]
        }
        valid = self._valid_payload(left, right)

        client = FakeLLMClient([invalid, valid])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 2
        assert res.block_results[0].semantic_rounds == 2

    def test_out_of_range_selector_retries(self):
        """Selector index out of range for the pair's own endpoint → retry."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        invalid = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": ["L5"],  # left has 0 evidence
                }
            ]
        }
        valid = self._valid_payload(left, right)

        client = FakeLLMClient([invalid, valid])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 2

    def test_negative_selector_retries(self):
        """Negative index selector (L-1) → A4C validation failure → retry."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        invalid = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": ["L-1"],
                }
            ]
        }
        valid = self._valid_payload(left, right)

        client = FakeLLMClient([invalid, valid])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 2

    def test_duplicate_selector_retries(self):
        """Duplicate selector (L0 twice) → A4C validation failure → retry."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A", evidence=(
                EvidenceRef("CH001_P001", "primary", "explicit", "A."),)),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        invalid = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": ["L0", "L0"],
                }
            ]
        }
        valid = self._valid_payload(left, right)

        client = FakeLLMClient([invalid, valid])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 2

    def test_wrong_ref_retries(self):
        """Wrong candidate_ref (not in requested pairs) → retry."""
        entries, left, right = self._entries()
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        wrong_ref = f"{CHUNK_ID}:cand_char_999"
        invalid = {
            "decisions": [
                {
                    "left_candidate_ref": wrong_ref,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                }
            ]
        }
        valid = self._valid_payload(left, right)

        client = FakeLLMClient([invalid, valid])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 2

    def test_first_invalid_second_valid(self):
        """First invalid, second valid → success with 2 calls."""
        entries, left, right = self._entries()
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        invalid = {"decisions": []}  # count mismatch
        valid = self._valid_payload(left, right)

        client = FakeLLMClient([invalid, valid])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert client.call_count == 2
        assert len(res.semantic_decisions) == 1
        assert res.block_results[0].semantic_rounds == 2

    def test_two_invalid_exhausts_rounds(self):
        """Two invalid outputs → ReconciliationSemanticGenerationError."""
        entries, left, right = self._entries()
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        invalid1 = {"decisions": []}
        invalid2 = {"decisions": []}

        client = FakeLLMClient([invalid1, invalid2])
        with pytest.raises(ReconciliationSemanticGenerationError):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# Test: Selector validation (the A4C semantic authority for evidence identity)
# ---------------------------------------------------------------------------


class TestSelectorValidation:
    """Deterministic pair-local evidence-selector validation + resolution.

    ``_validate_selector_block_payload`` is the single authority for selector
    validity: L/R form, index in range for the pair's own endpoint, no
    duplicates, and no third-candidate (a selector can only reference the
    decision's own left or right endpoint). It resolves valid selectors to the
    exact endpoint EvidenceRef.
    """

    def _plans_and_evidence(self, entries, pair_plans):
        result = make_planning_result(entries, pair_plans)
        blocks = _build_blocks(result)
        assert len(blocks) == 1
        block = blocks[0]
        evidence = _block_endpoint_evidence(result, list(block.pair_plans))
        return block.pair_plans, evidence

    @staticmethod
    def _payload(decisions):
        return ReconciliationSelectorDecisionPayload.from_dict({"decisions": decisions})

    def _validate(self, entries, pair_plans, decisions):
        plans, evidence = self._plans_and_evidence(entries, pair_plans)
        payload = self._payload(decisions)
        return _validate_selector_block_payload(payload, plans, evidence)

    # -- Resolution cases ----------------------------------------------------

    def test_l0_resolves_to_left_evidence(self):
        """L0 resolves to the exact left endpoint EvidenceRef[0]."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        left_ev = EvidenceRef("CH001_P001", "primary", "explicit", "A evidence.")
        entries = (
            make_candidate_entry(a, display_name="A", evidence=(left_ev,)),
            make_candidate_entry(b, display_name="B"),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["L0"],
            },
        ]
        ok, detail, resolved = self._validate(entries, pair_plans, decisions)
        assert ok, detail
        assert resolved[0] == (left_ev,)

    def test_r0_resolves_to_right_evidence(self):
        """R0 resolves to the exact right endpoint EvidenceRef[0]."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        right_ev = EvidenceRef("CH001_P002", "primary", "explicit", "B evidence.")
        entries = (
            make_candidate_entry(a, display_name="A"),
            make_candidate_entry(b, display_name="B", evidence=(right_ev,)),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["R0"],
            },
        ]
        ok, detail, resolved = self._validate(entries, pair_plans, decisions)
        assert ok, detail
        assert resolved[0] == (right_ev,)

    def test_selector_order_preserved(self):
        """Resolved evidence follows the provider's selector order."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        left_ev = EvidenceRef("CH001_P001", "primary", "explicit", "A.")
        right_ev = EvidenceRef("CH001_P002", "primary", "explicit", "B.")
        entries = (
            make_candidate_entry(a, display_name="A", evidence=(left_ev,)),
            make_candidate_entry(b, display_name="B", evidence=(right_ev,)),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["R0", "L0"],
            },
        ]
        ok, detail, resolved = self._validate(entries, pair_plans, decisions)
        assert ok, detail
        assert resolved[0] == (right_ev, left_ev)

    def test_null_excerpt_round_trips(self):
        """A selector resolving to a null-excerpt EvidenceRef round-trips null."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(a, display_name="A", evidence=(
                EvidenceRef("CH001_P001", "primary", "explicit", None),)),
            make_candidate_entry(b, display_name="B"),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["L0"],
            },
        ]
        ok, detail, resolved = self._validate(entries, pair_plans, decisions)
        assert ok, detail
        assert resolved[0][0].excerpt is None

    def test_empty_selectors_accepted(self):
        """Empty evidence_selectors remains accepted (behavior unchanged)."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(a, display_name="A"),
            make_candidate_entry(b, display_name="B"),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "uncertain", "reason_zh": "t.",
                "evidence_selectors": [],
            },
        ]
        ok, detail, _ = self._validate(entries, pair_plans, decisions)
        assert ok, detail

    # -- Rejection cases -----------------------------------------------------

    def test_invalid_prefix_rejected(self):
        """Non-L/R prefix (X0) → rejected (consumes a semantic round)."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(a, display_name="A"),
            make_candidate_entry(b, display_name="B"),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["X0"],
            },
        ]
        ok, detail, _ = self._validate(entries, pair_plans, decisions)
        assert not ok
        assert "invalid evidence selector" in detail

    def test_negative_index_rejected(self):
        """Negative index (L-1) → rejected."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(a, display_name="A", evidence=(
                EvidenceRef("CH001_P001", "primary", "explicit", "A."),)),
            make_candidate_entry(b, display_name="B"),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["L-1"],
            },
        ]
        ok, detail, _ = self._validate(entries, pair_plans, decisions)
        assert not ok
        assert "invalid evidence selector" in detail

    def test_non_integer_index_rejected(self):
        """Non-integer index (Lx) → rejected."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(a, display_name="A"),
            make_candidate_entry(b, display_name="B"),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["Lx"],
            },
        ]
        ok, detail, _ = self._validate(entries, pair_plans, decisions)
        assert not ok
        assert "invalid evidence selector" in detail

    def test_out_of_range_rejected(self):
        """Index out of range for the pair's own endpoint (L5, 0 evidence) → rejected."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(a, display_name="A"),
            make_candidate_entry(b, display_name="B"),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["L5"],
            },
        ]
        ok, detail, _ = self._validate(entries, pair_plans, decisions)
        assert not ok
        assert "out of range" in detail

    def test_duplicate_selector_rejected(self):
        """Duplicate selector (L0, L0) → rejected."""
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(a, display_name="A", evidence=(
                EvidenceRef("CH001_P001", "primary", "explicit", "A."),)),
            make_candidate_entry(b, display_name="B"),
        )
        pair_plans = (make_pair_plan(a, b),)
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["L0", "L0"],
            },
        ]
        ok, detail, _ = self._validate(entries, pair_plans, decisions)
        assert not ok
        assert "duplicate" in detail

    def test_no_third_candidate(self):
        """A selector can only reference the decision's own endpoints.

        R0 when the right endpoint has no evidence (and there is a third
        candidate in the block) is out of range for this pair → rejected.
        """
        a, b = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        c, d = f"{CHUNK_ID}:cand_char_003", f"{CHUNK_ID}:cand_char_004"
        entries = (
            make_candidate_entry(a, display_name="A"),
            make_candidate_entry(b, display_name="B"),
            make_candidate_entry(c, display_name="C", evidence=(
                EvidenceRef("CH001_P003", "primary", "explicit", "C."),)),
            make_candidate_entry(d, display_name="D"),
        )
        pair_plans = (make_pair_plan(a, b), make_pair_plan(c, d))
        # Pair (a, b) tries to cite R0, but b has no evidence; C's evidence
        # belongs to the OTHER pair, not to (a, b)'s own endpoints.
        decisions = [
            {
                "left_candidate_ref": a, "right_candidate_ref": b,
                "decision": "same_entity", "reason_zh": "t.",
                "evidence_selectors": ["R0"],
            },
            {
                "left_candidate_ref": c, "right_candidate_ref": d,
                "decision": "different_entity", "reason_zh": "t.",
                "evidence_selectors": [],
            },
        ]
        ok, detail, _ = self._validate(entries, pair_plans, decisions)
        assert not ok
        assert "out of range" in detail


# ---------------------------------------------------------------------------
# Test: Retry separation
# ---------------------------------------------------------------------------


class TestRetrySeparation:
    def test_llm_error_propagates(self):
        """LLMError → propagate immediately, no second semantic round."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        client = FakeLLMClient([LLMError("transport failure")])
        with pytest.raises(LLMError):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 1  # only 1 semantic call

    def test_llm_retry_exhausted_propagates(self):
        """LLMRetryExhaustedError → propagate immediately."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        client = FakeLLMClient([LLMError("retries exhausted")])
        with pytest.raises(LLMError):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 1

    def test_provenance_mismatch_fail_closed(self):
        """Provenance mismatch → ReconciliationProvenanceError, no retry."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        valid_payload = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                }
            ]
        }

        def make_bad_provenance(request, rendered_prompt, output_schema, semantic_profile):
            good_prov = build_provenance(
                request, ProviderMeta(), request_model="qwen3-27b", provider_family="qwen"
            )
            bad_prov = LLMInvocationProvenance(
                provider_family="qwen",
                model="qwen3-27b",
                semantic_profile_id=good_prov.semantic_profile_id,
                semantic_profile_hash=good_prov.semantic_profile_hash,
                prompt_id=good_prov.prompt_id,
                prompt_version=good_prov.prompt_version,
                prompt_content_hash=good_prov.prompt_content_hash,
                rendered_prompt_hash=good_prov.rendered_prompt_hash,
                output_schema_id=good_prov.output_schema_id,
                output_schema_version=good_prov.output_schema_version,
                output_schema_hash=good_prov.output_schema_hash,
                request_hash="f" * 64,  # wrong hash
                provider_response_id=None,
                finish_reason=None,
                usage=None,
            )
            return bad_prov

        class BadProvenanceClient(FakeLLMClient):
            def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
                self.call_count += 1
                self.calls.append((rendered_prompt, output_schema, semantic_profile))
                request = build_structured_request(
                    rendered_prompt=rendered_prompt,
                    output_schema=output_schema,
                    semantic_profile=semantic_profile,
                )
                self.request_hashes.append(request.request_hash)
                response = self.responses.pop(0)
                if isinstance(response, BaseException):
                    raise response
                parsed = response
                bad_prov = make_bad_provenance(
                    request, rendered_prompt, output_schema, semantic_profile
                )
                validate_against_output_schema(parsed, output_schema)
                return StructuredGenerationResult(
                    parsed_json=parsed, provenance=bad_prov, attempts=1
                )

        client = BadProvenanceClient([valid_payload])
        with pytest.raises(ReconciliationProvenanceError):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 1  # no second semantic round


# ---------------------------------------------------------------------------
# Test: Backend neutrality
# ---------------------------------------------------------------------------


class TestBackendNeutrality:
    def test_block_id_independent_of_backend(self):
        """Block ID does not change with different backend metadata."""
        entries = (
            make_candidate_entry(f"{CHUNK_ID}:cand_char_001"),
            make_candidate_entry(f"{CHUNK_ID}:cand_char_002"),
        )
        pair_plans = (
            make_pair_plan(f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"),
        )
        result = make_planning_result(entries, pair_plans, plan_hash=H)
        blocks1 = _build_blocks(result)

        result2 = make_planning_result(entries, pair_plans, plan_hash=H)
        blocks2 = _build_blocks(result2)
        assert blocks1[0].block_id == blocks2[0].block_id

    def test_decision_id_excludes_backend(self):
        """Decision ID excludes provider_family/model metadata."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="Alice"),
            make_candidate_entry(right, display_name="Alicia"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        payload = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                }
            ]
        }

        client1 = FakeLLMClient([payload])
        client1.provider_family = "qwen"
        client1.request_model = "qwen3-27b"
        res1 = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client1,
            prompt_registry=PROMPT_REGISTRY,
        )

        result2 = make_planning_result(entries, pair_plans)  # same plan_hash
        client2 = FakeLLMClient([payload])
        client2.provider_family = "gemma"  # different backend
        client2.request_model = "gemma-7b"  # different model
        res2 = resolve_semantic_ambiguity(
            result2, make_profile(), make_semantic_profile(), client2,
            prompt_registry=PROMPT_REGISTRY,
        )

        assert res1.semantic_decisions[0].decision_id == res2.semantic_decisions[0].decision_id


# ---------------------------------------------------------------------------
# Test: Profile consistency gate
# ---------------------------------------------------------------------------


class TestProfileConsistencyGate:
    def test_unknown_prompt_id_fails_closed(self):
        """Unknown prompt_id → fails closed, zero provider calls."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        bad_profile = make_profile(prompt_id="wrong.prompt-id")
        client = FakeLLMClient([])
        with pytest.raises(Exception):
            resolve_semantic_ambiguity(
                result, bad_profile, make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 0  # zero provider calls

    def test_output_schema_id_mismatch_gate(self):
        """Profile output_schema_id mismatch → gate raises (direct test)."""
        from short_drama.story.reconciliation_semantic import _check_profile_consistency

        profile = make_profile(output_schema_id="wrong-schema-id")
        prompt_spec = PROMPT_REGISTRY.load("a4.entity-reconciliation", version=3)
        from short_drama.io import load_json
        schema_data = load_json(SCHEMA_PATH)
        output_schema = OutputSchema.create(
            schema_id="correct-schema-id",
            schema_version=1,
            schema=schema_data,
        )
        with pytest.raises(ReconciliationSemanticError, match="output_schema_id"):
            _check_profile_consistency(profile, prompt_spec, output_schema)

    def test_valid_profile_passes_gate(self):
        """A consistent v2 profile (prompt v3 + selector schema) passes the gate."""
        from short_drama.io import load_json
        from short_drama.story.reconciliation_semantic import _check_profile_consistency

        profile = make_profile(max_generation_rounds=2)
        prompt_spec = PROMPT_REGISTRY.load("a4.entity-reconciliation", version=3)
        schema_data = load_json(SCHEMA_PATH)
        output_schema = OutputSchema.create(
            schema_id="a4-reconciliation-decision-selector-payload",
            schema_version=1,
            schema=schema_data,
        )
        # This should pass (no exception)
        _check_profile_consistency(profile, prompt_spec, output_schema)


# ---------------------------------------------------------------------------
# Test: Complete decision coverage
# ---------------------------------------------------------------------------


class TestDecisionCoverage:
    def test_all_explicit_pairs_have_decision(self):
        """Every explicit A4B pair has exactly one decision in all_decisions."""
        left1, right1 = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        left2, right2 = f"{CHUNK_ID}:cand_char_003", f"{CHUNK_ID}:cand_char_004"
        left3, right3 = f"{CHUNK_ID}:cand_char_005", f"{CHUNK_ID}:cand_char_006"

        entries = (
            make_candidate_entry(left1, display_name="A1"),
            make_candidate_entry(right1, display_name="A2"),
            make_candidate_entry(left2, display_name="B1"),
            make_candidate_entry(right2, display_name="B2"),
            make_candidate_entry(left3, display_name="C1"),
            make_candidate_entry(right3, display_name="C2"),
        )

        auto_decision = ReconciliationDecision(
            decision_id="dec_" + "a" * 20,
            left_candidate_ref=left1,
            right_candidate_ref=right1,
            decision="same_entity",
            method="deterministic",
            reason_code="same_strong_exact_identity_key",
            reason_zh="确定性自动合并",
            evidence_refs=(),
            prompt_id=None,
            prompt_version=None,
            generation_provenance=None,
        )
        must_decision = ReconciliationDecision(
            decision_id="dec_" + "b" * 20,
            left_candidate_ref=left2,
            right_candidate_ref=right2,
            decision="different_entity",
            method="deterministic",
            reason_code="hard_must_not_merge",
            reason_zh="确定性硬约束",
            evidence_refs=(),
            prompt_id=None,
            prompt_version=None,
            generation_provenance=None,
        )

        pair_plans = (
            make_pair_plan(left1, right1, state=PAIR_STATE_AUTO_SAME),
            make_pair_plan(left2, right2, state=PAIR_STATE_MUST_NOT_MERGE),
            make_pair_plan(left3, right3, state=PAIR_STATE_NEEDS_SEMANTIC_DECISION),
        )
        result = make_planning_result(
            entries, pair_plans, decisions=(auto_decision, must_decision)
        )

        payload = {
            "decisions": [
                {
                    "left_candidate_ref": left3,
                    "right_candidate_ref": right3,
                    "decision": "same_entity",
                    "reason_zh": "LLM判定。",
                    "evidence_selectors": [],
                }
            ]
        }

        client = FakeLLMClient([payload])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )

        assert len(res.all_decisions) == 3

        pair_refs = set()
        for d in res.all_decisions:
            key = (d.left_candidate_ref, d.right_candidate_ref)
            assert key not in pair_refs, f"duplicate decision for {key}"
            pair_refs.add(key)

        assert (left1, right1) in pair_refs
        assert (left2, right2) in pair_refs
        assert (left3, right3) in pair_refs

        det_in_all = [d for d in res.all_decisions if d.method == "deterministic"]
        assert len(det_in_all) == 2
        llm_in_all = [d for d in res.all_decisions if d.method == "llm"]
        assert len(llm_in_all) == 1
        assert llm_in_all[0].decision == "same_entity"

    def test_canonical_sort_order(self):
        """all_decisions is canonically sorted by (left, right)."""
        left1, right1 = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        left2, right2 = f"{CHUNK_ID}:cand_char_003", f"{CHUNK_ID}:cand_char_004"

        entries = (
            make_candidate_entry(left1, display_name="A"),
            make_candidate_entry(right1, display_name="B"),
            make_candidate_entry(left2, display_name="C"),
            make_candidate_entry(right2, display_name="D"),
        )
        pair_plans = (
            make_pair_plan(left1, right1),
            make_pair_plan(left2, right2),
        )
        result = make_planning_result(entries, pair_plans)

        payload = {
            "decisions": [
                {
                    "left_candidate_ref": left1,
                    "right_candidate_ref": right1,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                },
                {
                    "left_candidate_ref": left2,
                    "right_candidate_ref": right2,
                    "decision": "different_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                },
            ]
        }

        client = FakeLLMClient([payload])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )

        for i in range(len(res.all_decisions) - 1):
            d1 = res.all_decisions[i]
            d2 = res.all_decisions[i + 1]
            assert (d1.left_candidate_ref, d1.right_candidate_ref) < (
                d2.left_candidate_ref, d2.right_candidate_ref
            )


# ---------------------------------------------------------------------------
# Test: Provenance / identity
# ---------------------------------------------------------------------------


class TestProvenanceIdentity:
    def test_exact_provenance_accepted(self):
        """Exact matching provenance is accepted."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        payload = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "测试。",
                    "evidence_selectors": [],
                }
            ]
        }

        client = FakeLLMClient([payload])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )
        assert res.semantic_decisions[0].generation_provenance is not None
        assert res.semantic_decisions[0].generation_provenance.request_hash == res.block_results[0].request_hash

    def test_decision_id_deterministic(self):
        """Same semantic material → same decision_id (deterministic)."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="Alice"),
            make_candidate_entry(right, display_name="Alicia"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        payload = {
            "decisions": [
                {
                    "left_candidate_ref": left,
                    "right_candidate_ref": right,
                    "decision": "same_entity",
                    "reason_zh": "同一人。",
                    "evidence_selectors": [],
                }
            ]
        }

        client1 = FakeLLMClient([payload])
        res1 = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client1,
            prompt_registry=PROMPT_REGISTRY,
        )

        result2 = make_planning_result(entries, pair_plans)
        client2 = FakeLLMClient([payload])
        res2 = resolve_semantic_ambiguity(
            result2, make_profile(), make_semantic_profile(), client2,
            prompt_registry=PROMPT_REGISTRY,
        )

        assert res1.semantic_decisions[0].decision_id == res2.semantic_decisions[0].decision_id
        assert res1.semantic_decisions[0].decision_id.startswith("dec_")
        assert len(res1.semantic_decisions[0].decision_id) == 4 + 20  # "dec_" + 20 hex


# ---------------------------------------------------------------------------
# Test: Zero pairs → zero calls
# ---------------------------------------------------------------------------


class TestZeroPairs:
    def test_zero_semantic_pairs_zero_calls(self):
        """No semantic pairs → zero blocks, zero LLM calls."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (
            make_pair_plan(left, right, state=PAIR_STATE_AUTO_SAME),
        )
        auto_decision = ReconciliationDecision(
            decision_id="dec_" + "c" * 20,
            left_candidate_ref=left,
            right_candidate_ref=right,
            decision="same_entity",
            method="deterministic",
            reason_code="same_strong_exact_identity_key",
            reason_zh="确定性自动合并",
            evidence_refs=(),
            prompt_id=None,
            prompt_version=None,
            generation_provenance=None,
        )
        result = make_planning_result(entries, pair_plans, decisions=(auto_decision,))

        client = FakeLLMClient([])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )

        assert client.call_count == 0
        assert res.blocks == ()
        assert res.semantic_decisions == ()
        assert res.all_decisions == (auto_decision,)
        assert res.semantic_request_hashes == ()


# ---------------------------------------------------------------------------
# Finding 1: Missing semantic-pair endpoint must fail before provider call
# ---------------------------------------------------------------------------


class TestMissingEndpointFailClosed:
    """Invalid A4B planning input (missing endpoint / wrong kind) → fail closed,
    zero provider calls, NOT semantic retry."""

    def test_left_endpoint_missing_from_index(self):
        left = f"{CHUNK_ID}:cand_char_001"  # not in index
        right = f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        client = FakeLLMClient([make_valid_payload(left, right, "same_entity")])
        with pytest.raises(ReconciliationSemanticError, match="not found in"):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 0

    def test_right_endpoint_missing_from_index(self):
        left = f"{CHUNK_ID}:cand_char_001"
        right = f"{CHUNK_ID}:cand_char_002"  # not in index
        entries = (
            make_candidate_entry(left, display_name="A"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        client = FakeLLMClient([make_valid_payload(left, right, "same_entity")])
        with pytest.raises(ReconciliationSemanticError, match="not found in"):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 0

    def test_non_merge_graph_candidate_kind(self):
        left = f"{CHUNK_ID}:cand_char_001"
        right = f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, kind="unresolved_person", display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        client = FakeLLMClient([make_valid_payload(left, right, "same_entity")])
        with pytest.raises(ReconciliationSemanticError, match="candidate_kind"):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 0


# ---------------------------------------------------------------------------
# Finding 2: Enforce exact A4B explicit-pair decision coverage
# ---------------------------------------------------------------------------


def _make_det_decision(
    left: str,
    right: str,
    decision: str,
    method: str = "deterministic",
) -> ReconciliationDecision:
    """Helper to build a deterministic decision."""
    return ReconciliationDecision(
        decision_id="dec_" + "d" * 20,
        left_candidate_ref=left,
        right_candidate_ref=right,
        decision=decision,
        method=method,
        reason_code="test_reason",
        reason_zh="测试",
        evidence_refs=(),
        prompt_id=None,
        prompt_version=None,
        generation_provenance=None,
    )


class TestDecisionCoverageFailClosed:
    """Exact pair coverage: every explicit pair → exactly one decision,
    with state/method consistency (fail-closed regressions)."""

    def test_auto_same_decision_missing(self):
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right, state=PAIR_STATE_AUTO_SAME),)
        result = make_planning_result(entries, pair_plans, decisions=())

        client = FakeLLMClient([])
        with pytest.raises(ReconciliationSemanticError, match="no decision found"):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 0

    def test_must_not_merge_decision_missing(self):
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right, state=PAIR_STATE_MUST_NOT_MERGE),)
        result = make_planning_result(entries, pair_plans, decisions=())

        client = FakeLLMClient([])
        with pytest.raises(ReconciliationSemanticError, match="no decision found"):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 0

    def test_extra_deterministic_decision_not_in_pair_plans(self):
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        extra_left = f"{CHUNK_ID}:cand_char_010"
        extra_right = f"{CHUNK_ID}:cand_char_011"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
            make_candidate_entry(extra_left, display_name="X"),
            make_candidate_entry(extra_right, display_name="Y"),
        )
        pair_plans = (make_pair_plan(left, right, state=PAIR_STATE_AUTO_SAME),)
        valid_det = _make_det_decision(left, right, "same_entity")
        extra_det = _make_det_decision(extra_left, extra_right, "same_entity")
        result = make_planning_result(
            entries, pair_plans, decisions=(valid_det, extra_det)
        )

        client = FakeLLMClient([])
        with pytest.raises(ReconciliationSemanticError, match="not present in"):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )
        assert client.call_count == 0

    def test_deterministic_decision_for_needs_semantic_pair(self):
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right, state=PAIR_STATE_NEEDS_SEMANTIC_DECISION),)
        det = _make_det_decision(left, right, "same_entity", method="deterministic")
        result = make_planning_result(entries, pair_plans, decisions=(det,))

        client = FakeLLMClient([make_valid_payload(left, right, "same_entity")])
        with pytest.raises(ReconciliationSemanticError, match="duplicate decision"):
            resolve_semantic_ambiguity(
                result, make_profile(), make_semantic_profile(), client,
                prompt_registry=PROMPT_REGISTRY,
            )

    def test_wrong_method_for_semantic_pair_direct(self):
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right, state=PAIR_STATE_NEEDS_SEMANTIC_DECISION),)
        det = _make_det_decision(left, right, "same_entity", method="deterministic")
        result = make_planning_result(entries, pair_plans, decisions=(det,))

        with pytest.raises(ReconciliationSemanticError, match="expected 'llm'"):
            _validate_decision_coverage(result, (det,), ())

    def test_semantic_duplicate_deterministic_pair(self):
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (
            make_pair_plan(left, right, state=PAIR_STATE_AUTO_SAME),
        )
        det = _make_det_decision(left, right, "same_entity")
        result = make_planning_result(entries, pair_plans, decisions=(det,))

        client = FakeLLMClient([])
        with pytest.raises(ReconciliationSemanticError, match="duplicate decision"):
            _validate_decision_coverage(
                result, (det,), (det,)
            )
        assert client.call_count == 0

    def test_valid_mixed_auto_same_must_not_merge_semantic(self):
        a_left, a_right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        b_left, b_right = f"{CHUNK_ID}:cand_char_003", f"{CHUNK_ID}:cand_char_004"
        c_left, c_right = f"{CHUNK_ID}:cand_char_005", f"{CHUNK_ID}:cand_char_006"

        entries = (
            make_candidate_entry(a_left, display_name="A"),
            make_candidate_entry(a_right, display_name="B"),
            make_candidate_entry(b_left, display_name="C"),
            make_candidate_entry(b_right, display_name="D"),
            make_candidate_entry(c_left, display_name="E"),
            make_candidate_entry(c_right, display_name="F"),
        )
        pair_plans = (
            make_pair_plan(a_left, a_right, state=PAIR_STATE_AUTO_SAME),
            make_pair_plan(b_left, b_right, state=PAIR_STATE_MUST_NOT_MERGE),
            make_pair_plan(c_left, c_right, state=PAIR_STATE_NEEDS_SEMANTIC_DECISION),
        )
        det_same = _make_det_decision(a_left, a_right, "same_entity")
        det_diff = _make_det_decision(b_left, b_right, "different_entity")
        result = make_planning_result(
            entries, pair_plans, decisions=(det_same, det_diff)
        )

        client = FakeLLMClient(
            [make_valid_payload(c_left, c_right, "uncertain")]
        )
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )

        assert client.call_count == 1
        assert len(res.all_decisions) == 3
        pair_keys = {
            (d.left_candidate_ref, d.right_candidate_ref) for d in res.all_decisions
        }
        assert pair_keys == {
            (a_left, a_right),
            (b_left, b_right),
            (c_left, c_right),
        }


# ---------------------------------------------------------------------------
# Finding 3: Catch only typed-model rejection during semantic regeneration
# ---------------------------------------------------------------------------


class TestTypedModelExceptionNarrowing:
    """Only ReconciliationModelError triggers semantic retry. Unexpected
    non-model exceptions must propagate."""

    def test_reconciliation_model_error_triggers_retry(self):
        """ReconciliationModelError → semantic retry (2nd round attempted)."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        # First round: non-canonical pair order (left > right) triggers
        # ReconciliationModelError in from_dict(). Schema passes because both
        # refs match the pattern.
        bad_payload = {
            "decisions": [
                {
                    "left_candidate_ref": right,   # intentionally reversed
                    "right_candidate_ref": left,
                    "decision": "same_entity",
                    "reason_zh": "测试",
                    "evidence_selectors": [],
                }
            ]
        }
        good_payload = make_valid_payload(left, right, "same_entity")

        client = FakeLLMClient([bad_payload, good_payload])
        res = resolve_semantic_ambiguity(
            result, make_profile(), make_semantic_profile(), client,
            prompt_registry=PROMPT_REGISTRY,
        )

        # 2 rounds consumed: first failed (ReconciliationModelError), second succeeded
        assert client.call_count == 2
        assert len(res.semantic_decisions) == 1

    def test_unexpected_exception_propagates(self):
        """Unexpected non-model exception → propagated → no second semantic call."""
        left, right = f"{CHUNK_ID}:cand_char_001", f"{CHUNK_ID}:cand_char_002"
        entries = (
            make_candidate_entry(left, display_name="A"),
            make_candidate_entry(right, display_name="B"),
        )
        pair_plans = (make_pair_plan(left, right),)
        result = make_planning_result(entries, pair_plans)

        import short_drama.story.reconciliation_semantic as sem_mod

        original_from_dict = ReconciliationSelectorDecisionPayload.from_dict

        # Patch to raise a non-ReconciliationModelError exception
        def _raise_unexpected(value):
            raise ValueError("unexpected programming error")

        sem_mod.ReconciliationSelectorDecisionPayload.from_dict = _raise_unexpected
        try:
            client = FakeLLMClient([make_valid_payload(left, right, "same_entity")])
            with pytest.raises(ValueError, match="unexpected programming error"):
                resolve_semantic_ambiguity(
                    result, make_profile(), make_semantic_profile(), client,
                    prompt_registry=PROMPT_REGISTRY,
                )
        finally:
            sem_mod.ReconciliationSelectorDecisionPayload.from_dict = original_from_dict

        # Only 1 call: the unexpected exception propagated, no retry
        assert client.call_count == 1


def make_valid_payload(
    left: str,
    right: str,
    decision: str,
    reason_zh: str = "LLM 判断",
    evidence_selectors: list[str] | None = None,
) -> dict:
    """Build a schema-valid selector payload dict for FakeLLMClient."""
    if evidence_selectors is None:
        evidence_selectors = []
    return {
        "decisions": [
            {
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": decision,
                "reason_zh": reason_zh,
                "evidence_selectors": evidence_selectors,
            }
        ]
    }
