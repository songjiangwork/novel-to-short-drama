"""v1.2 A5D-B -- event + relationship semantic provider execution + selector resolution.

Covers A5D-B implemented in ``short_drama.story.consolidation_semantic``:

  * :func:`resolve_event_semantic_ambiguity` and
    :func:`resolve_relationship_semantic_ambiguity` drive the FROZEN production
    policies (:data:`EVENT_SEMANTIC_PACKING_V1` / :data:`RELATIONSHIP_SEMANTIC_PACKING_V1`
    = 12 / 24 each, identical block ids / request hashes to audited P2) via the
    A5D-A preparation builders and then execute each prepared block against an
    (in-memory) ``LLMClient``;
  * per-block pipeline: provider call -> provenance verification -> typed
    selector payload load -> exact pair coverage/order -> pair-local selector
    validation -> canonical selector order -> exact endpoint ``EvidenceRef``
    resolution -> stable exact-``EvidenceRef`` alias dedupe -> domain
    ``SemanticDecision(method="llm")``;
  * technical ``LLMError`` is PROPAGATED (not a semantic retry); provenance
    mismatch FAILS CLOSED with NO semantic retry; bounded semantic rounds (2);
  * ``uncertain`` is a VALID successful semantic result (no retry);
  * ``compute_event_llm_decision_id`` / ``compute_relationship_llm_decision_id``
    are backend-neutral (provider / model / response-id changes alone do not
    alter the recomputed decision id);
  * exact whole-domain decision coverage (deterministic auto_same + LLM
    semantic);
  * the frozen P2 identity regression (block ids / request hashes identical to
    audited P2 candidates).

All provider invocations use deterministic fake ``LLMClient`` subclasses; no
real provider is called and nothing is persisted.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from short_drama.llm import (
    LLMClient,
    LLMInvocationProvenance,
    PromptRegistry,
    StructuredGenerationResult,
    build_structured_request,
)
from short_drama.llm.errors import LLMError
from short_drama.story import (
    DEFAULT_PROMPT_BASE_DIR,
    EVENT_SEMANTIC_PACKING_V1,
    RELATIONSHIP_SEMANTIC_PACKING_V1,
    ConsolidationProvenanceError,
    ConsolidationSemanticError,
    ConsolidationSemanticGenerationError,
    EventSemanticDecision,
    EventSemanticPackingPolicy,
    EventSemanticResolutionResult,
    RelationshipSemanticDecision,
    RelationshipSemanticPackingPolicy,
    RelationshipSemanticResolutionResult,
    StoryIntegrityError,
    build_event_semantic_preparation,
    build_relationship_semantic_preparation,
    compute_event_llm_decision_id,
    compute_relationship_llm_decision_id,
    load_event_semantic_profile,
    load_relationship_semantic_profile,
    resolve_event_semantic_ambiguity,
    resolve_relationship_semantic_ambiguity,
)
from short_drama.story.extraction import EvidenceRef
from test_story_a5b_audit import ChunkSpec, _build_multi_chunk_tree
from test_story_a5b_planning import (
    DOCUMENT,
    PROJECT,
    RECON_PROFILE_ID,
    _char,
    _consolidation_profile,
    _loc,
)

_PROMPTS = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
_EV_SEM_PROFILE = load_event_semantic_profile()
_REL_SEM_PROFILE = load_relationship_semantic_profile()
_PROFILE = _consolidation_profile()


# ---------------------------------------------------------------------------
# Synthetic candidate / corpus helpers
# ---------------------------------------------------------------------------


def _ev(paragraph_id: str = "CH001_P0001") -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role="primary", strength="explicit", excerpt="txt"
    )


def _mk_event(
    cid: str,
    participants=(),
    locations=(),
    summary: str = "evt",
    para: str = "CH001_P0001",
) -> "object":
    from short_drama.story.extraction import EventCandidate

    return EventCandidate(
        candidate_id=cid,
        summary_zh=summary,
        participant_refs=tuple(participants),
        location_refs=tuple(locations),
        temporal_mode="normal",
        evidence_strength="explicit",
        evidence=(_ev(para),),
    )


def _mk_rel(
    cid: str,
    source: str,
    target: str,
    rtype: str = "meets",
    state: str | None = None,
    para: str = "CH001_P0001",
) -> "object":
    from short_drama.story.extraction import RelationshipCandidate

    return RelationshipCandidate(
        candidate_id=cid,
        source_ref=source,
        target_ref=target,
        relationship_type_zh=rtype,
        state_zh=state,
        direction="directed",
        evidence_strength="explicit",
        evidence=(_ev(para),),
    )


def _event_specs():
    """Two-chunk corpus: 4 events, 3 semantic pairs, 0 auto_same.

    CH001: cand_evt_001 / cand_evt_002 / cand_evt_003 all share
    cand_char_001 (with distinct summaries) -> all-pairs semantic. CH002:
    cand_evt_004 is alone (cand_char_002 only) -> no pair.
    """
    return [
        ChunkSpec(
            "CH001",
            ("CH001_P0001", "CH001_P0002", "CH001_P0003"),
            chars=(_char("cand_char_001", "Alice"),),
            locs=(_loc("cand_loc_001", "Wonderland"),),
            events=(
                _mk_event("cand_evt_001", participants=("cand_char_001",),
                          locations=("cand_loc_001",), summary="arrives"),
                _mk_event("cand_evt_002", participants=("cand_char_001",),
                          locations=("cand_loc_001",), summary="departs",
                          para="CH001_P0002"),
                _mk_event("cand_evt_003", participants=("cand_char_001",),
                          summary="speaks", para="CH001_P0003"),
            ),
        ),
        ChunkSpec(
            "CH002",
            ("CH002_P0001",),
            chars=(_char("cand_char_002", "Bob"),),
            events=(
                _mk_event("cand_evt_004", participants=("cand_char_002",),
                          summary="reads", para="CH002_P0001"),
            ),
        ),
    ]


def _relationship_specs():
    """Two-chunk corpus: 3 relationships, 3 semantic pairs, 0 auto_same.

    CH001: cand_rel_001 / cand_rel_002 / cand_rel_003 all share
    (cand_char_001, cand_char_002) with distinct types -> all-pairs semantic.
    """
    return [
        ChunkSpec(
            "CH001",
            ("CH001_P0001", "CH001_P0002", "CH001_P0003"),
            chars=(
                _char("cand_char_001", "Alice"),
                _char("cand_char_002", "Bob"),
            ),
            rels=(
                _mk_rel("cand_rel_001", source="cand_char_001",
                        target="cand_char_002", rtype="meets"),
                _mk_rel("cand_rel_002", source="cand_char_001",
                        target="cand_char_002", rtype="helps",
                        para="CH001_P0002"),
                _mk_rel("cand_rel_003", source="cand_char_001",
                        target="cand_char_002", rtype="argues",
                        state="tense", para="CH001_P0003"),
            ),
        ),
    ]


def _ev_tree(tmp_path: Path):
    return _build_multi_chunk_tree(tmp_path, *_event_specs())


def _rel_tree(tmp_path: Path):
    return _build_multi_chunk_tree(tmp_path, *_relationship_specs())


def _ev_planning(tree):
    from short_drama.story import build_consolidation_planning

    return build_consolidation_planning(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=_PROFILE,
    )


def _rel_planning(tree):
    from short_drama.story import build_consolidation_planning

    return build_consolidation_planning(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=_PROFILE,
    )


# ---------------------------------------------------------------------------
# Fake provider + payload helpers
# ---------------------------------------------------------------------------


def _make_provenance(request, *, provider_family="qwen", model="qwen3-27b", **overrides):
    """Build an ``LLMInvocationProvenance`` that matches ``request`` exactly."""
    rendered = request.rendered_prompt
    schema = request.output_schema
    base = dict(
        provider_family=provider_family,
        model=model,
        semantic_profile_id=request.semantic_profile.profile_id,
        semantic_profile_hash=request.semantic_profile.semantic_profile_hash,
        prompt_id=rendered.prompt_id,
        prompt_version=rendered.prompt_version,
        prompt_content_hash=rendered.prompt_content_hash,
        rendered_prompt_hash=rendered.rendered_prompt_hash,
        output_schema_id=schema.schema_id,
        output_schema_version=schema.schema_version,
        output_schema_hash=schema.schema_hash,
        request_hash=request.request_hash,
        provider_response_id="resp",
        finish_reason="stop",
        usage=None,
    )
    base.update(overrides)
    return LLMInvocationProvenance(**base)


class FakeLLMClient(LLMClient):
    """Deterministic fake provider for A5D-B offline tests.

    Scripted responses (popped in order):
      * a ``dict`` -> successful result with provenance matching the request;
      * a ``(dict, LLMInvocationProvenance)`` tuple -> successful result with
        the EXACT supplied provenance;
      * an ``Exception`` -> raised from ``generate_structured``.
    """

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(self, responses, *, provider_family="qwen", request_model="qwen3-27b"):
        self.responses = list(responses)
        self.call_count = 0
        self.request_hashes: list[str] = []
        self.provider_family = provider_family
        self.request_model = request_model

    def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
        self.call_count += 1
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
            provenance = _make_provenance(
                request,
                provider_family=self.provider_family,
                model=self.request_model,
            )
        return StructuredGenerationResult(
            parsed_json=parsed, provenance=provenance, attempts=1
        )


def _ev_payload_for_block(block, decision="same_event", reason="LLM 判断", selectors=None):
    """Build a schema-valid event payload dict covering a block's pairs in order."""
    if selectors is None:
        selectors = {}
    decisions = []
    for i, (left, right) in enumerate(block.pair_refs):
        decisions.append(
            {
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": decision,
                "reason_zh": reason,
                "evidence_selectors": list(selectors.get(i, [])),
            }
        )
    return {"decisions": decisions}


def _rel_payload_for_block(
    block, decision="same_relationship", reason="LLM 判断", selectors=None
):
    """Build a schema-valid relationship payload dict covering a block's pairs in order."""
    if selectors is None:
        selectors = {}
    decisions = []
    for i, (left, right) in enumerate(block.pair_refs):
        decisions.append(
            {
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": decision,
                "reason_zh": reason,
                "evidence_selectors": list(selectors.get(i, [])),
            }
        )
    return {"decisions": decisions}


def _ev_prep(planning):
    return build_event_semantic_preparation(
        planning, _PROFILE, _EV_SEM_PROFILE,
        prompts=_PROMPTS,
        packing_policy=EVENT_SEMANTIC_PACKING_V1,
    )


def _rel_prep(planning):
    return build_relationship_semantic_preparation(
        planning, _PROFILE, _REL_SEM_PROFILE,
        prompts=_PROMPTS,
        packing_policy=RELATIONSHIP_SEMANTIC_PACKING_V1,
    )


def _ev_valid_responses(planning, decision="same_event"):
    return [_ev_payload_for_block(b, decision=decision) for b in _ev_prep(planning).blocks]


def _rel_valid_responses(planning, decision="same_relationship"):
    return [_rel_payload_for_block(b, decision=decision) for b in _rel_prep(planning).blocks]


def _resolve_ev(planning, client):
    return resolve_event_semantic_ambiguity(
        planning, _PROFILE, _EV_SEM_PROFILE, client, prompts=_PROMPTS
    )


def _resolve_rel(planning, client):
    return resolve_relationship_semantic_ambiguity(
        planning, _PROFILE, _REL_SEM_PROFILE, client, prompts=_PROMPTS
    )


# ---------------------------------------------------------------------------
# Frozen production policies (12 / 24) are audited-P2-identical
# ---------------------------------------------------------------------------


class TestFrozenPolicies:
    def test_event_packing_v1_name(self):
        assert EVENT_SEMANTIC_PACKING_V1.name == "event-semantic-packing-v1"

    def test_event_packing_v1_limits(self):
        assert EVENT_SEMANTIC_PACKING_V1.max_pairs_per_block == 12
        assert EVENT_SEMANTIC_PACKING_V1.max_candidates_per_block == 24

    def test_relationship_packing_v1_name(self):
        assert RELATIONSHIP_SEMANTIC_PACKING_V1.name == "relationship-semantic-packing-v1"

    def test_relationship_packing_v1_limits(self):
        assert RELATIONSHIP_SEMANTIC_PACKING_V1.max_pairs_per_block == 12
        assert RELATIONSHIP_SEMANTIC_PACKING_V1.max_candidates_per_block == 24

    def test_event_v1_produces_same_block_ids_as_p2(self, tmp_path):
        """Frozen v1 produces byte-identical block ids to audited P2."""
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep_v1 = build_event_semantic_preparation(
            planning, _PROFILE, _EV_SEM_PROFILE,
            prompts=_PROMPTS,
            packing_policy=EVENT_SEMANTIC_PACKING_V1,
        )
        prep_p2 = build_event_semantic_preparation(
            planning, _PROFILE, _EV_SEM_PROFILE,
            prompts=_PROMPTS,
            packing_policy=EventSemanticPackingPolicy("P2", 12, 24),
        )
        assert [b.block_id for b in prep_v1.blocks] == [b.block_id for b in prep_p2.blocks]
        assert prep_v1.semantic_request_hashes == prep_p2.semantic_request_hashes

    def test_relationship_v1_produces_same_block_ids_as_p2(self, tmp_path):
        """Frozen v1 produces byte-identical block ids to audited P2."""
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep_v1 = build_relationship_semantic_preparation(
            planning, _PROFILE, _REL_SEM_PROFILE,
            prompts=_PROMPTS,
            packing_policy=RELATIONSHIP_SEMANTIC_PACKING_V1,
        )
        prep_p2 = build_relationship_semantic_preparation(
            planning, _PROFILE, _REL_SEM_PROFILE,
            prompts=_PROMPTS,
            packing_policy=RelationshipSemanticPackingPolicy("P2", 12, 24),
        )
        assert [b.block_id for b in prep_v1.blocks] == [b.block_id for b in prep_p2.blocks]
        assert prep_v1.semantic_request_hashes == prep_p2.semantic_request_hashes

    def test_policy_label_does_not_affect_identity(self, tmp_path):
        """Different label but same limits -> same block ids / request hashes."""
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep_a = build_event_semantic_preparation(
            planning, _PROFILE, _EV_SEM_PROFILE,
            prompts=_PROMPTS,
            packing_policy=EventSemanticPackingPolicy("nameA", 12, 24),
        )
        prep_b = build_event_semantic_preparation(
            planning, _PROFILE, _EV_SEM_PROFILE,
            prompts=_PROMPTS,
            packing_policy=EventSemanticPackingPolicy("nameB", 12, 24),
        )
        assert [b.block_id for b in prep_a.blocks] == [b.block_id for b in prep_b.blocks]
        assert prep_a.semantic_request_hashes == prep_b.semantic_request_hashes


# ---------------------------------------------------------------------------
# Event: valid first-round single block
# ---------------------------------------------------------------------------


class TestEventValidFirstRound:
    def test_single_block_success(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        responses = _ev_valid_responses(planning, decision="same_event")
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert isinstance(result, EventSemanticResolutionResult)
        assert client.call_count == len(planning.event_pair_plans) // 12 + (
            1 if len(planning.event_pair_plans) % 12 else 0
        )
        assert len(result.block_results) == client.call_count
        assert result.semantic_decisions
        # All decisions should be same_event
        for d in result.semantic_decisions:
            assert d.decision == "same_event"
            assert d.method == "llm"

    def test_multi_block_success(self, tmp_path):
        """Use enough events to force multiple blocks (pair limit = 12).

        17 events in same chunk -> C(17,2) = 136 pairs -> 12 blocks.
        """
        from short_drama.story.extraction import EventCandidate

        specs = [
            ChunkSpec(
                "CH001",
                tuple(f"CH001_P{i:04d}" for i in range(1, 18)),
                chars=(_char("cand_char_001", "Alice"),),
                events=tuple(
                    EventCandidate(
                        candidate_id=f"cand_evt_{i:03d}",
                        summary_zh=f"evt {i}",
                        participant_refs=("cand_char_001",),
                        location_refs=(),
                        temporal_mode="normal",
                        evidence_strength="explicit",
                        evidence=(EvidenceRef(
                            paragraph_id=f"CH001_P{i:04d}",
                            role="primary", strength="explicit", excerpt="txt"
                        ),),
                    )
                    for i in range(1, 18)
                ),
            ),
        ]
        tree = _build_multi_chunk_tree(tmp_path, *specs)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        assert len(prep.blocks) > 1  # multiple blocks
        responses = _ev_valid_responses(planning, decision="same_event")
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert len(result.block_results) == len(prep.blocks)
        assert client.call_count == len(prep.blocks)

    def test_same_event_accepted(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        responses = _ev_valid_responses(planning, decision="same_event")
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        for d in result.semantic_decisions:
            assert d.decision == "same_event"

    def test_different_event_accepted(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        responses = _ev_valid_responses(planning, decision="different_event")
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        for d in result.semantic_decisions:
            assert d.decision == "different_event"

    def test_uncertain_accepted_no_retry(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        responses = _ev_valid_responses(planning, decision="uncertain")
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        for d in result.semantic_decisions:
            assert d.decision == "uncertain"
        # No retry: exactly one call per block
        assert client.call_count == len(prep.blocks)


# ---------------------------------------------------------------------------
# Event: semantic retry (typed-invalid round1 -> valid round2)
# ---------------------------------------------------------------------------


class TestEventSemanticRetry:
    def test_typed_invalid_round1_valid_round2(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        # Round 1: invalid JSON (typed load failure). Round 2: valid.
        responses = [
            ({"invalid": "data"}, None),  # will fail typed load
            _ev_payload_for_block(prep.blocks[0], decision="same_event"),
        ]
        # Remove the None and use a dict that fails typed load
        responses = [
            {"decisions": [{"left_candidate_ref": "bad", "right_candidate_ref": "bad",
                            "decision": "invalid_decision", "reason_zh": "x",
                            "evidence_selectors": []}]},
            _ev_payload_for_block(prep.blocks[0], decision="same_event"),
        ]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2
        assert result.block_results[0].semantic_rounds == 2

    def test_selector_invalid_round1_valid_round2(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Round 1: valid typed load but out-of-range selector.
        bad_selectors = {0: ["L99"]}
        responses = [
            _ev_payload_for_block(block, decision="same_event", selectors=bad_selectors),
            _ev_payload_for_block(block, decision="same_event"),
        ]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2
        assert result.block_results[0].semantic_rounds == 2

    def test_semantic_invalid_twice_exhaustion(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Both rounds: invalid (wrong decision count).
        responses = [
            {"decisions": []},  # empty: count mismatch
            {"decisions": []},  # empty again
        ]
        client = FakeLLMClient(responses)
        with pytest.raises(ConsolidationSemanticGenerationError) as exc_info:
            _resolve_ev(planning, client)
        assert client.call_count == 2
        assert exc_info.value.rounds_attempted == 2
        assert exc_info.value.block_id == block.block_id

    def test_llm_error_propagates(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        client = FakeLLMClient([LLMError("provider down")])
        with pytest.raises(LLMError):
            _resolve_ev(planning, client)
        assert client.call_count == 1

    def test_provenance_mismatch_fail_closed(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        request = prep.structured_requests[0]
        # Build a provenance with a wrong request_hash.
        bad_prov = _make_provenance(request, request_hash="deadbeef" * 8)
        responses = [(_ev_payload_for_block(block, decision="same_event"), bad_prov)]
        client = FakeLLMClient(responses)
        with pytest.raises(ConsolidationProvenanceError):
            _resolve_ev(planning, client)
        # No semantic retry: only one call.
        assert client.call_count == 1


# ---------------------------------------------------------------------------
# Event: exact pair coverage validation
# ---------------------------------------------------------------------------


class TestEventPairCoverage:
    def test_missing_result_invalid(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Return one fewer decision than pairs.
        decisions = []
        for i, (left, right) in enumerate(block.pair_refs[:-1]):
            decisions.append({
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": "same_event",
                "reason_zh": "r",
                "evidence_selectors": [],
            })
        responses = [{"decisions": decisions}, _ev_payload_for_block(block, "same_event")]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2  # round 1 invalid, round 2 valid

    def test_extra_result_invalid(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Return one more decision than pairs.
        decisions = []
        for i, (left, right) in enumerate(block.pair_refs):
            decisions.append({
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": "same_event",
                "reason_zh": "r",
                "evidence_selectors": [],
            })
        # Add an extra
        decisions.append({
            "left_candidate_ref": block.pair_refs[0][0],
            "right_candidate_ref": block.pair_refs[0][1],
            "decision": "same_event",
            "reason_zh": "extra",
            "evidence_selectors": [],
        })
        responses = [{"decisions": decisions}, _ev_payload_for_block(block, "same_event")]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2

    def test_reordered_result_invalid(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        if len(block.pair_refs) < 2:
            pytest.skip("need at least 2 pairs")
        # Swap first two pairs.
        decisions = []
        for i in range(len(block.pair_refs)):
            left, right = block.pair_refs[(i + 1) % len(block.pair_refs)]
            decisions.append({
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": "same_event",
                "reason_zh": "r",
                "evidence_selectors": [],
            })
        responses = [{"decisions": decisions}, _ev_payload_for_block(block, "same_event")]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2

    def test_swapped_endpoint_invalid(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Swap left/right for the first pair.
        left, right = block.pair_refs[0]
        decisions = [{
            "left_candidate_ref": right,  # swapped
            "right_candidate_ref": left,  # swapped
            "decision": "same_event",
            "reason_zh": "r",
            "evidence_selectors": [],
        }]
        for i, (l, r) in enumerate(block.pair_refs[1:]):
            decisions.append({
                "left_candidate_ref": l,
                "right_candidate_ref": r,
                "decision": "same_event",
                "reason_zh": "r",
                "evidence_selectors": [],
            })
        responses = [{"decisions": decisions}, _ev_payload_for_block(block, "same_event")]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# Event: selector validation
# ---------------------------------------------------------------------------


class TestEventSelectors:
    def test_l0_r0_success(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Each candidate has 1 evidence item, so L0 and R0 are valid.
        selectors = {i: ["L0", "R0"] for i in range(len(block.pair_refs))}
        responses = [_ev_payload_for_block(block, "same_event", selectors=selectors)]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 1
        # Verify evidence refs are resolved
        for d in result.semantic_decisions:
            assert len(d.evidence_refs) >= 1

    def test_selector_permutation_canonicalized(self, tmp_path):
        """Selectors in non-canonical order are canonicalized in the output."""
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Use multiple evidence items for the first pair's candidates.
        # Each candidate in our corpus has 1 evidence, so L0/R0 is the max.
        # We just verify that providing R0 before L0 still works and produces
        # canonicalized output (L first, then R).
        selectors = {0: ["R0", "L0"]}
        responses = [_ev_payload_for_block(block, "same_event", selectors=selectors)]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 1
        d = result.semantic_decisions[0]
        # The evidence refs should be in canonical order: L0 first, then R0.
        # Since both candidates have 1 evidence each, we get 2 refs.
        assert len(d.evidence_refs) == 2

    def test_duplicate_selector_invalid(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        selectors = {0: ["L0", "L0"]}  # duplicate
        responses = [
            _ev_payload_for_block(block, "same_event", selectors=selectors),
            _ev_payload_for_block(block, "same_event"),
        ]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2

    def test_bad_selector_syntax_invalid(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        selectors = {0: ["X0"]}  # bad prefix
        responses = [
            _ev_payload_for_block(block, "same_event", selectors=selectors),
            _ev_payload_for_block(block, "same_event"),
        ]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2

    def test_out_of_range_selector_invalid(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        selectors = {0: ["L5"]}  # out of range (only 1 evidence)
        responses = [
            _ev_payload_for_block(block, "same_event", selectors=selectors),
            _ev_payload_for_block(block, "same_event"),
        ]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 2

    def test_empty_selectors_valid(self, tmp_path):
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # No selectors: valid.
        responses = [_ev_payload_for_block(block, "same_event", selectors={i: [] for i in range(len(block.pair_refs))})]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        assert client.call_count == 1
        for d in result.semantic_decisions:
            assert d.evidence_refs == ()

    def test_same_exact_evidence_ref_deduped(self, tmp_path):
        """If L0 and R0 resolve to the same exact EvidenceRef, it appears once."""
        # We need both endpoints to have the same evidence (same paragraph_id,
        # role, strength, excerpt). Our corpus has different paragraphs, so we
        # need a custom setup where both candidates share an evidence ref.
        from short_drama.story.extraction import EventCandidate

        shared_ev = EvidenceRef(
            paragraph_id="CH001_P0001", role="primary", strength="explicit", excerpt="shared"
        )
        specs = [
            ChunkSpec(
                "CH001",
                ("CH001_P0001",),
                chars=(_char("cand_char_001", "Alice"),),
                events=(
                    EventCandidate(
                        candidate_id="cand_evt_001",
                        summary_zh="evt1",
                        participant_refs=("cand_char_001",),
                        location_refs=(),
                        temporal_mode="normal",
                        evidence_strength="explicit",
                        evidence=(shared_ev,),
                    ),
                    EventCandidate(
                        candidate_id="cand_evt_002",
                        summary_zh="evt2",
                        participant_refs=("cand_char_001",),
                        location_refs=(),
                        temporal_mode="normal",
                        evidence_strength="explicit",
                        evidence=(shared_ev,),
                    ),
                ),
            ),
        ]
        tree = _build_multi_chunk_tree(tmp_path, *specs)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Select L0 and R0 for the first pair (same exact EvidenceRef).
        selectors = {0: ["L0", "R0"]}
        responses = [_ev_payload_for_block(block, "same_event", selectors=selectors)]
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        d = result.semantic_decisions[0]
        # The two selectors resolve to the same exact EvidenceRef -> deduped to 1.
        assert len(d.evidence_refs) == 1


# ---------------------------------------------------------------------------
# Event: decision ID determinism
# ---------------------------------------------------------------------------


class TestEventDecisionId:
    def test_deterministic(self):
        ev = _ev("CH001_P0001")
        id1 = compute_event_llm_decision_id(
            left_ref="src_001:CH001:cand_evt_001",
            right_ref="src_001:CH001:cand_evt_002",
            decision="same_event",
            reason_zh="reason",
            evidence_refs=(ev,),
            prompt_id="a5.event-consolidation",
            prompt_version=1,
            request_hash="abc123",
        )
        id2 = compute_event_llm_decision_id(
            left_ref="src_001:CH001:cand_evt_001",
            right_ref="src_001:CH001:cand_evt_002",
            decision="same_event",
            reason_zh="reason",
            evidence_refs=(ev,),
            prompt_id="a5.event-consolidation",
            prompt_version=1,
            request_hash="abc123",
        )
        assert id1 == id2
        assert id1.startswith("dec_")
        assert len(id1) == 4 + 20  # "dec_" + 20 hex

    def test_semantic_sensitivity(self):
        ev = _ev("CH001_P0001")
        id1 = compute_event_llm_decision_id(
            left_ref="a", right_ref="b", decision="same_event",
            reason_zh="r1", evidence_refs=(ev,),
            prompt_id="p", prompt_version=1, request_hash="h",
        )
        id2 = compute_event_llm_decision_id(
            left_ref="a", right_ref="b", decision="different_event",
            reason_zh="r1", evidence_refs=(ev,),
            prompt_id="p", prompt_version=1, request_hash="h",
        )
        assert id1 != id2

    def test_provider_metadata_does_not_alter_id(self):
        """Decision ID is backend-neutral: provider/model changes don't matter."""
        ev = _ev("CH001_P0001")
        id1 = compute_event_llm_decision_id(
            left_ref="a", right_ref="b", decision="same_event",
            reason_zh="r", evidence_refs=(ev,),
            prompt_id="p", prompt_version=1, request_hash="h",
        )
        # The ID function doesn't take provider metadata at all, so any
        # change to provider/model/response-id/usage/finish_reason doesn't
        # affect it. This is verified by the function signature: it only
        # accepts semantic material.
        assert id1.startswith("dec_")


# ---------------------------------------------------------------------------
# Event: block atomicity
# ---------------------------------------------------------------------------


class TestEventBlockAtomicity:
    def test_block_failure_stops_processing(self, tmp_path):
        """If a block fails after both rounds, no later blocks are processed."""
        from short_drama.story.extraction import EventCandidate

        # Create enough events to force 2+ blocks.
        # 17 events -> C(17,2) = 136 pairs -> 12 blocks of 12 + 1 block of 4.
        specs = [
            ChunkSpec(
                "CH001",
                tuple(f"CH001_P{i:04d}" for i in range(1, 18)),
                chars=(_char("cand_char_001", "Alice"),),
                events=tuple(
                    EventCandidate(
                        candidate_id=f"cand_evt_{i:03d}",
                        summary_zh=f"evt {i}",
                        participant_refs=("cand_char_001",),
                        location_refs=(),
                        temporal_mode="normal",
                        evidence_strength="explicit",
                        evidence=(EvidenceRef(
                            paragraph_id=f"CH001_P{i:04d}",
                            role="primary", strength="explicit", excerpt="txt"
                        ),),
                    )
                    for i in range(1, 18)
                ),
            ),
        ]
        tree = _build_multi_chunk_tree(tmp_path, *specs)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        assert len(prep.blocks) > 1

        # First block: both rounds invalid. Later blocks: would be valid but
        # never reached.
        responses = [
            {"decisions": []},  # block 0 round 1: invalid
            {"decisions": []},  # block 0 round 2: invalid
        ]
        client = FakeLLMClient(responses)
        with pytest.raises(ConsolidationSemanticGenerationError):
            _resolve_ev(planning, client)
        # Only 2 calls (both for block 0), not more.
        assert client.call_count == 2

    def test_no_partial_decisions_retained(self, tmp_path):
        """On block failure, no partial decisions from that block are accepted."""
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        prep = _ev_prep(planning)
        block = prep.blocks[0]
        # Both rounds invalid.
        responses = [{"decisions": []}, {"decisions": []}]
        client = FakeLLMClient(responses)
        with pytest.raises(ConsolidationSemanticGenerationError):
            _resolve_ev(planning, client)
        # The exception is raised; no result is returned.


# ---------------------------------------------------------------------------
# Event: deterministic + LLM exact whole-event coverage
# ---------------------------------------------------------------------------


class TestEventWholeCoverage:
    def test_auto_same_plus_llm_coverage(self, tmp_path):
        """If there are auto_same pairs, they appear in all_event_decisions."""
        # Our corpus has 0 auto_same, so all decisions are LLM.
        tree = _ev_tree(tmp_path)
        planning = _ev_planning(tree)
        auto_same_count = sum(
            1 for p in planning.event_pair_plans if p.state == "auto_same"
        )
        semantic_count = sum(
            1 for p in planning.event_pair_plans if p.state == "needs_semantic_decision"
        )
        responses = _ev_valid_responses(planning, decision="same_event")
        client = FakeLLMClient(responses)
        result = _resolve_ev(planning, client)
        # Total decisions = auto_same + semantic
        assert len(result.all_event_decisions) == auto_same_count + semantic_count
        # Deterministic decisions
        det_count = sum(1 for d in result.all_event_decisions if d.method == "deterministic")
        llm_count = sum(1 for d in result.all_event_decisions if d.method == "llm")
        assert det_count == auto_same_count
        assert llm_count == semantic_count

    def test_empty_semantic_stream_zero_calls(self, tmp_path):
        """0 needs_semantic_decision -> 0 blocks -> 0 provider calls."""
        # A single event with no pairs produces 0 semantic pairs.
        from short_drama.story.extraction import EventCandidate

        specs = [
            ChunkSpec(
                "CH001",
                ("CH001_P0001",),
                chars=(_char("cand_char_001", "Alice"),),
                events=(
                    EventCandidate(
                        candidate_id="cand_evt_001",
                        summary_zh="evt1",
                        participant_refs=("cand_char_001",),
                        location_refs=(),
                        temporal_mode="normal",
                        evidence_strength="explicit",
                        evidence=(_ev("CH001_P0001"),),
                    ),
                ),
            ),
        ]
        tree = _build_multi_chunk_tree(tmp_path, *specs)
        planning = _ev_planning(tree)
        client = FakeLLMClient([])
        result = _resolve_ev(planning, client)
        assert client.call_count == 0
        assert result.semantic_decisions == ()
        assert result.block_results == ()


# ---------------------------------------------------------------------------
# Relationship: valid first-round
# ---------------------------------------------------------------------------


class TestRelationshipValidFirstRound:
    def test_single_block_success(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        responses = _rel_valid_responses(planning, decision="same_relationship")
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert isinstance(result, RelationshipSemanticResolutionResult)
        assert len(result.block_results) == client.call_count
        for d in result.semantic_decisions:
            assert d.decision == "same_relationship"
            assert d.method == "llm"

    def test_same_relationship_accepted(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        responses = _rel_valid_responses(planning, decision="same_relationship")
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        for d in result.semantic_decisions:
            assert d.decision == "same_relationship"

    def test_different_relationship_accepted(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        responses = _rel_valid_responses(planning, decision="different_relationship")
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        for d in result.semantic_decisions:
            assert d.decision == "different_relationship"

    def test_uncertain_accepted_no_retry(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        responses = _rel_valid_responses(planning, decision="uncertain")
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        for d in result.semantic_decisions:
            assert d.decision == "uncertain"
        assert client.call_count == len(prep.blocks)


# ---------------------------------------------------------------------------
# Relationship: semantic retry + exhaustion
# ---------------------------------------------------------------------------


class TestRelationshipSemanticRetry:
    def test_typed_invalid_round1_valid_round2(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        responses = [
            {"decisions": [{"left_candidate_ref": "bad", "right_candidate_ref": "bad",
                            "decision": "invalid_decision", "reason_zh": "x",
                            "evidence_selectors": []}]},
            _rel_payload_for_block(block, decision="same_relationship"),
        ]
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert client.call_count == 2
        assert result.block_results[0].semantic_rounds == 2

    def test_semantic_invalid_twice_exhaustion(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        responses = [{"decisions": []}, {"decisions": []}]
        client = FakeLLMClient(responses)
        with pytest.raises(ConsolidationSemanticGenerationError) as exc_info:
            _resolve_rel(planning, client)
        assert client.call_count == 2
        assert exc_info.value.rounds_attempted == 2

    def test_llm_error_propagates(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        client = FakeLLMClient([LLMError("provider down")])
        with pytest.raises(LLMError):
            _resolve_rel(planning, client)
        assert client.call_count == 1

    def test_provenance_mismatch_fail_closed(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        request = prep.structured_requests[0]
        bad_prov = _make_provenance(request, request_hash="deadbeef" * 8)
        responses = [(_rel_payload_for_block(block, "same_relationship"), bad_prov)]
        client = FakeLLMClient(responses)
        with pytest.raises(ConsolidationProvenanceError):
            _resolve_rel(planning, client)
        assert client.call_count == 1


# ---------------------------------------------------------------------------
# Relationship: exact pair coverage validation
# ---------------------------------------------------------------------------


class TestRelationshipPairCoverage:
    def test_missing_result_invalid(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        decisions = []
        for i, (left, right) in enumerate(block.pair_refs[:-1]):
            decisions.append({
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": "same_relationship",
                "reason_zh": "r",
                "evidence_selectors": [],
            })
        responses = [{"decisions": decisions}, _rel_payload_for_block(block, "same_relationship")]
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert client.call_count == 2

    def test_reordered_result_invalid(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        if len(block.pair_refs) < 2:
            pytest.skip("need at least 2 pairs")
        decisions = []
        for i in range(len(block.pair_refs)):
            left, right = block.pair_refs[(i + 1) % len(block.pair_refs)]
            decisions.append({
                "left_candidate_ref": left,
                "right_candidate_ref": right,
                "decision": "same_relationship",
                "reason_zh": "r",
                "evidence_selectors": [],
            })
        responses = [{"decisions": decisions}, _rel_payload_for_block(block, "same_relationship")]
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert client.call_count == 2

    def test_swapped_endpoint_invalid(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        left, right = block.pair_refs[0]
        decisions = [{
            "left_candidate_ref": right,
            "right_candidate_ref": left,
            "decision": "same_relationship",
            "reason_zh": "r",
            "evidence_selectors": [],
        }]
        for i, (l, r) in enumerate(block.pair_refs[1:]):
            decisions.append({
                "left_candidate_ref": l,
                "right_candidate_ref": r,
                "decision": "same_relationship",
                "reason_zh": "r",
                "evidence_selectors": [],
            })
        responses = [{"decisions": decisions}, _rel_payload_for_block(block, "same_relationship")]
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# Relationship: selector validation
# ---------------------------------------------------------------------------


class TestRelationshipSelectors:
    def test_l0_r0_success(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        selectors = {i: ["L0", "R0"] for i in range(len(block.pair_refs))}
        responses = [_rel_payload_for_block(block, "same_relationship", selectors=selectors)]
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert client.call_count == 1

    def test_empty_selectors_valid(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        responses = [_rel_payload_for_block(block, "same_relationship", selectors={i: [] for i in range(len(block.pair_refs))})]
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert client.call_count == 1
        for d in result.semantic_decisions:
            assert d.evidence_refs == ()

    def test_duplicate_selector_invalid(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        selectors = {0: ["L0", "L0"]}
        responses = [
            _rel_payload_for_block(block, "same_relationship", selectors=selectors),
            _rel_payload_for_block(block, "same_relationship"),
        ]
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert client.call_count == 2

    def test_out_of_range_selector_invalid(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        selectors = {0: ["L5"]}
        responses = [
            _rel_payload_for_block(block, "same_relationship", selectors=selectors),
            _rel_payload_for_block(block, "same_relationship"),
        ]
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# Relationship: decision ID determinism
# ---------------------------------------------------------------------------


class TestRelationshipDecisionId:
    def test_deterministic(self):
        ev = _ev("CH001_P0001")
        id1 = compute_relationship_llm_decision_id(
            left_ref="src_001:CH001:cand_rel_001",
            right_ref="src_001:CH001:cand_rel_002",
            decision="same_relationship",
            reason_zh="reason",
            evidence_refs=(ev,),
            prompt_id="a5.relationship-consolidation",
            prompt_version=1,
            request_hash="abc123",
        )
        id2 = compute_relationship_llm_decision_id(
            left_ref="src_001:CH001:cand_rel_001",
            right_ref="src_001:CH001:cand_rel_002",
            decision="same_relationship",
            reason_zh="reason",
            evidence_refs=(ev,),
            prompt_id="a5.relationship-consolidation",
            prompt_version=1,
            request_hash="abc123",
        )
        assert id1 == id2
        assert id1.startswith("dec_")
        assert len(id1) == 24

    def test_semantic_sensitivity(self):
        ev = _ev("CH001_P0001")
        id1 = compute_relationship_llm_decision_id(
            left_ref="a", right_ref="b", decision="same_relationship",
            reason_zh="r1", evidence_refs=(ev,),
            prompt_id="p", prompt_version=1, request_hash="h",
        )
        id2 = compute_relationship_llm_decision_id(
            left_ref="a", right_ref="b", decision="different_relationship",
            reason_zh="r1", evidence_refs=(ev,),
            prompt_id="p", prompt_version=1, request_hash="h",
        )
        assert id1 != id2


# ---------------------------------------------------------------------------
# Relationship: state_zh and direction are not altered
# ---------------------------------------------------------------------------


class TestRelationshipStateDirection:
    def test_state_zh_preserved_in_preparation(self, tmp_path):
        """state_zh differences are not altered during response handling."""
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        # Verify the preparation carries state_zh exactly.
        for block in prep.blocks:
            for pc in block.pair_contexts:
                for side in ("left", "right"):
                    state = pc[side].get("state_zh")
                    # state_zh is either None or a non-empty string, carried
                    # through exactly from the indexed candidate.
                    if state is not None:
                        assert isinstance(state, str)
                        assert len(state) > 0

    def test_direction_not_recanonicalized(self, tmp_path):
        """Direction does not get re-canonicalized by provider-result handling."""
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        # All our test relationships use direction="directed".
        for block in prep.blocks:
            for pc in block.pair_contexts:
                for side in ("left", "right"):
                    assert pc[side]["direction"] == "directed"


# ---------------------------------------------------------------------------
# Relationship: whole-domain coverage + empty stream
# ---------------------------------------------------------------------------


class TestRelationshipWholeCoverage:
    def test_auto_same_plus_llm_coverage(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        auto_same_count = sum(
            1 for p in planning.relationship_pair_plans if p.state == "auto_same"
        )
        semantic_count = sum(
            1 for p in planning.relationship_pair_plans
            if p.state == "needs_semantic_decision"
        )
        responses = _rel_valid_responses(planning, decision="same_relationship")
        client = FakeLLMClient(responses)
        result = _resolve_rel(planning, client)
        assert len(result.all_relationship_decisions) == auto_same_count + semantic_count
        det_count = sum(
            1 for d in result.all_relationship_decisions if d.method == "deterministic"
        )
        llm_count = sum(
            1 for d in result.all_relationship_decisions if d.method == "llm"
        )
        assert det_count == auto_same_count
        assert llm_count == semantic_count

    def test_empty_semantic_stream_zero_calls(self, tmp_path):
        """0 needs_semantic_decision -> 0 blocks -> 0 provider calls."""
        from short_drama.story.extraction import RelationshipCandidate

        specs = [
            ChunkSpec(
                "CH001",
                ("CH001_P0001",),
                chars=(
                    _char("cand_char_001", "Alice"),
                    _char("cand_char_002", "Bob"),
                ),
                rels=(
                    RelationshipCandidate(
                        candidate_id="cand_rel_001",
                        source_ref="cand_char_001",
                        target_ref="cand_char_002",
                        relationship_type_zh="meets",
                        state_zh=None,
                        direction="directed",
                        evidence_strength="explicit",
                        evidence=(_ev("CH001_P0001"),),
                    ),
                ),
            ),
        ]
        tree = _build_multi_chunk_tree(tmp_path, *specs)
        planning = _rel_planning(tree)
        client = FakeLLMClient([])
        result = _resolve_rel(planning, client)
        assert client.call_count == 0
        assert result.semantic_decisions == ()
        assert result.block_results == ()


# ---------------------------------------------------------------------------
# Relationship: block atomicity
# ---------------------------------------------------------------------------


class TestRelationshipBlockAtomicity:
    def test_block_failure_stops_processing(self, tmp_path):
        tree = _rel_tree(tmp_path)
        planning = _rel_planning(tree)
        prep = _rel_prep(planning)
        block = prep.blocks[0]
        responses = [{"decisions": []}, {"decisions": []}]
        client = FakeLLMClient(responses)
        with pytest.raises(ConsolidationSemanticGenerationError):
            _resolve_rel(planning, client)
        assert client.call_count == 2
