"""v1.2 A5C-B -- fact semantic provider execution + selector resolution.

Covers A5C-B implemented in ``short_drama.story.consolidation_semantic``:

  * :func:`resolve_fact_semantic_ambiguity` drives the FROZEN production
    policy (:data:`FACT_SEMANTIC_PACKING_V1` = 12 / 24, identical block ids /
    request hashes to audited P2) via :func:`build_fact_semantic_preparation`
    and then executes each prepared block against an (in-memory) ``LLMClient``;
  * per-block pipeline: provider call -> provenance verification -> typed
    ``FactSelectorDecisionPayload`` load -> exact pair coverage/order -> pair-
    local selector validation -> canonical selector order -> exact endpoint
    ``EvidenceRef`` resolution -> stable exact-``EvidenceRef`` alias dedupe ->
    ``FactSemanticDecision(method="llm")``;
  * technical ``LLMError`` is PROPAGATED (not a semantic retry); provenance
    mismatch FAILS CLOSED with NO semantic retry; bounded semantic rounds (2);
  * ``uncertain`` is a VALID successful semantic result (no retry);
  * ``compute_fact_llm_decision_id`` is backend-neutral (provider / model /
    response-id changes alone do not alter the recomputed decision id);
  * exact whole-fact-stream coverage (deterministic auto_same + LLM semantic).

All provider invocations use deterministic fake ``LLMClient`` subclasses; no
real provider is called and nothing is persisted (no CanonicalFactSet /
StateTransition / StoryConflict). The A5C-A tests remain intact.
"""

from __future__ import annotations

import dataclasses

import pytest

from short_drama.llm import (
    LLMClient,
    LLMInvocationProvenance,
    PromptRegistry,
    StructuredGenerationResult,
    build_structured_request,
)
from short_drama.story import (
    DEFAULT_PROMPT_BASE_DIR,
    FACT_SEMANTIC_PACKING_V1,
    ConsolidationProvenanceError,
    ConsolidationSemanticError,
    ConsolidationSemanticGenerationError,
    FactSemanticDecision,
    FactSemanticResolutionResult,
    build_fact_semantic_preparation,
    compute_fact_llm_decision_id,
    load_fact_semantic_profile,
    resolve_fact_semantic_ambiguity,
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
from test_story_a5c_fact_preparation import _corpus_specs, _mk_fact, _planning, _tree

_PROMPTS = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
_SEM_PROFILE = load_fact_semantic_profile()
_PROFILE = _consolidation_profile()


def _ev(paragraph_id: str) -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role="primary", strength="explicit", excerpt="txt"
    )


# ---------------------------------------------------------------------------
# Fake provider + payload helpers
# ---------------------------------------------------------------------------


def _make_provenance(request, *, provider_family="qwen", model="qwen3-27b", **overrides):
    """Build an ``LLMInvocationProvenance`` that matches ``request`` exactly.

    The backend-neutral semantic/request identity fields are sourced from the
    exact request so provenance verification passes; provider metadata
    (family / model / response id / usage) is audit provenance only.
    """
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
    """Deterministic fake provider for A5C-B offline tests.

    Scripted responses (popped in order):
      * a ``dict`` -> successful result with provenance matching the request;
      * a ``(dict, LLMInvocationProvenance)`` tuple -> successful result with
        the EXACT supplied provenance (for provenance-mismatch tests);
      * an ``Exception`` -> raised from ``generate_structured``.

    Records ``call_count`` and the exact ``request_hash`` per call so tests can
    assert the number of provider calls and the stable per-round request hash.
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
        return StructuredGenerationResult(parsed_json=parsed, provenance=provenance, attempts=1)


def _payload_for_block(block, decision="same_fact", reason="LLM 判断", selectors=None):
    """Build a schema-valid payload dict covering a block's pairs in order.

    ``selectors`` maps pair index -> list of selector strings (default ``[]``).
    """
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


def _prep(planning):
    return build_fact_semantic_preparation(
        planning, _PROFILE, _SEM_PROFILE, prompts=_PROMPTS,
        packing_policy=FACT_SEMANTIC_PACKING_V1,
    )


def _valid_responses(planning, decision="same_fact"):
    """One valid payload per prepared block (frozen policy, in order)."""
    return [_payload_for_block(b, decision=decision) for b in _prep(planning).blocks]


def _resolve(planning, client):
    return resolve_fact_semantic_ambiguity(
        planning, _PROFILE, _SEM_PROFILE, client, prompts=_PROMPTS
    )


# ---------------------------------------------------------------------------
# Frozen production policy (12 / 24) is unchanged and audited-P2-identical
# ---------------------------------------------------------------------------


def test_frozen_production_policy_is_12_24():
    assert FACT_SEMANTIC_PACKING_V1.name == "fact-semantic-packing-v1"
    assert FACT_SEMANTIC_PACKING_V1.max_pairs_per_block == 12
    assert FACT_SEMANTIC_PACKING_V1.max_candidates_per_block == 24


def test_frozen_policy_matches_audited_p2_identity(tmp_path):
    """The production policy name does NOT enter the packing identity.

    ``FACT_SEMANTIC_PACKING_V1`` (12 / 24) must produce byte-identical block
    ids and request hashes to the audited P2 candidate (12 / 24).
    """
    planning = _planning(_tree(tmp_path, specs=_single_facts_specs(30)))
    from short_drama.story import A5C_PACKING_CANDIDATES

    p2 = A5C_PACKING_CANDIDATES[1]
    prep_prod = build_fact_semantic_preparation(
        planning, _PROFILE, _SEM_PROFILE, prompts=_PROMPTS,
        packing_policy=FACT_SEMANTIC_PACKING_V1,
    )
    prep_p2 = build_fact_semantic_preparation(
        planning, _PROFILE, _SEM_PROFILE, prompts=_PROMPTS, packing_policy=p2
    )
    assert [b.block_id for b in prep_prod.blocks] == [b.block_id for b in prep_p2.blocks]
    assert prep_prod.semantic_request_hashes == prep_p2.semantic_request_hashes


def _single_facts_specs(n_facts: int):
    """One chunk with ``n_facts`` distinct facts (all-pairs semantic)."""
    paras = tuple(f"CH001_P{i:04d}" for i in range(1, n_facts + 1))
    return [
        ChunkSpec(
            "CH001",
            paras,
            chars=(_char("cand_char_001", "Alice"),),
            facts=tuple(
                _mk_fact(
                    f"cand_fact_{i:03d}", statement_zh=f"stmt {i}", para=paras[i - 1]
                )
                for i in range(1, n_facts + 1)
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


def test_single_block_valid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    client = FakeLLMClient(_valid_responses(planning))
    result = _resolve(planning, client)
    assert isinstance(result, FactSemanticResolutionResult)
    assert client.call_count == 1
    assert len(result.block_results) == 1
    assert result.block_results[0].semantic_rounds == 1
    assert result.semantic_decisions
    assert all(d.method == "llm" for d in result.semantic_decisions)
    # The preparation carried the frozen policy.
    assert result.preparation.packing_policy is FACT_SEMANTIC_PACKING_V1


def test_multiple_blocks_valid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_single_facts_specs(30)))
    prep = _prep(planning)
    assert len(prep.blocks) > 1
    client = FakeLLMClient(_valid_responses(planning))
    result = _resolve(planning, client)
    # One provider call per block, in preparation order.
    assert client.call_count == len(prep.blocks)
    assert len(result.block_results) == len(prep.blocks)
    assert client.request_hashes == list(prep.semantic_request_hashes)
    assert sum(len(b.decisions) for b in result.block_results) == len(result.semantic_decisions)


def test_all_six_fact_decisions_accepted(tmp_path):
    six = ("same_fact", "compatible_fact", "state_change", "conflict", "unrelated", "uncertain")
    # One single-pair block (two facts); exercise each of the six decisions.
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    for dec in six:
        client = FakeLLMClient([_payload_for_block(block, decision=dec)])
        result = _resolve(planning, client)
        assert client.call_count == 1
        assert [d.decision for d in result.semantic_decisions] == [dec]


def test_uncertain_succeeds_without_retry(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    client = FakeLLMClient(_valid_responses(planning, decision="uncertain"))
    result = _resolve(planning, client)
    # uncertain is a VALID successful semantic result: one call, no retry.
    assert client.call_count == 1
    assert result.block_results[0].semantic_rounds == 1
    assert [d.decision for d in result.semantic_decisions] == ["uncertain"]


def _two_facts(left_stmt="left fact", right_stmt="right fact",
               left_paras=("CH001_P0001",), right_paras=("CH001_P0002",)):
    """One chunk with exactly two distinct facts -> one semantic pair / block."""
    return [
        ChunkSpec(
            "CH001",
            left_paras + right_paras,
            chars=(_char("cand_char_001", "Alice"),),
            facts=(
                _mk_fact("cand_fact_001", statement_zh=left_stmt, para=left_paras[0]),
                _mk_fact("cand_fact_002", statement_zh=right_stmt, para=right_paras[0]),
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Semantic retry (bounded)
# ---------------------------------------------------------------------------


def test_round1_typed_invalid_round2_valid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    invalid = _payload_for_block(block, decision="NOT_A_DECISION")  # bad enum
    valid = _payload_for_block(block)
    client = FakeLLMClient([invalid, valid])
    result = _resolve(planning, client)
    assert client.call_count == 2
    assert result.block_results[0].semantic_rounds == 2
    # Same request both rounds (semantic retry does not mutate the request).
    assert client.request_hashes[0] == client.request_hashes[1]


def test_round1_invalid_selector_round2_valid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    bad = _payload_for_block(block, selectors={0: ["X0"]})  # bad selector syntax
    valid = _payload_for_block(block)
    client = FakeLLMClient([bad, valid])
    result = _resolve(planning, client)
    assert client.call_count == 2
    assert result.block_results[0].semantic_rounds == 2


def test_semantic_exhaustion(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    bad = _payload_for_block(block, decision="NOT_A_DECISION")
    client = FakeLLMClient([bad, bad])
    with pytest.raises(ConsolidationSemanticGenerationError) as exc_info:
        _resolve(planning, client)
    assert client.call_count == 2
    err = exc_info.value
    assert err.block_id == block.block_id
    assert err.rounds_attempted == 2
    assert err.request_hash == client.request_hashes[0]
    assert err.expected_pairs == tuple(block.pair_refs)
    assert err.last_failure_details


def test_llm_error_propagates_not_semantic_retry(tmp_path):
    from short_drama.llm import LLMError

    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    client = FakeLLMClient([LLMError("transport-level failure after technical retry")])
    with pytest.raises(LLMError):
        _resolve(planning, client)
    # An LLMError is propagated immediately (not routed into a semantic retry).
    assert client.call_count == 1


# ---------------------------------------------------------------------------
# Provenance verification (FAIL CLOSED, no semantic retry)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "semantic_profile_hash",
        "prompt_content_hash",
        "rendered_prompt_hash",
        "output_schema_hash",
        "request_hash",
    ],
)
def test_provenance_mismatch_fails_closed_no_retry(tmp_path, field):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    valid = _payload_for_block(block)

    # A client whose every successful result carries provenance that mismatches
    # the requested ``field``. If the code (incorrectly) retried on provenance
    # mismatch, ``call_count`` would exceed 1.
    class MismatchClient(FakeLLMClient):
        def __init__(self, field_name):
            super().__init__([None])  # responses never consumed
            self._field = field_name
            self._mismatches = 0

        def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
            self.call_count += 1
            request = build_structured_request(
                rendered_prompt=rendered_prompt,
                output_schema=output_schema,
                semantic_profile=semantic_profile,
            )
            self.request_hashes.append(request.request_hash)
            prov = _make_provenance(request)
            prov = dataclasses.replace(prov, **{self._field: "0" * 64})
            self._mismatches += 1
            return StructuredGenerationResult(
                parsed_json=valid, provenance=prov, attempts=1
            )

    client = MismatchClient(field)
    with pytest.raises(ConsolidationProvenanceError, match="provenance field"):
        _resolve(planning, client)
    # FAIL CLOSED: exactly one provider call, no semantic retry.
    assert client.call_count == 1
    assert client._mismatches == 1


# ---------------------------------------------------------------------------
# Pair coverage / order (A4C strict behavior)
# ---------------------------------------------------------------------------


def _three_facts():
    paras = ("CH001_P0001", "CH001_P0002", "CH001_P0003")
    return [
        ChunkSpec(
            "CH001",
            paras,
            chars=(_char("cand_char_001", "Alice"),),
            facts=(
                _mk_fact("cand_fact_001", statement_zh="s1", para=paras[0]),
                _mk_fact("cand_fact_002", statement_zh="s2", para=paras[1]),
                _mk_fact("cand_fact_003", statement_zh="s3", para=paras[2]),
            ),
        )
    ]


def _exhausts_invalid(planning, bad_payload):
    """Feed ``bad_payload`` for both rounds -> semantic exhaustion."""
    client = FakeLLMClient([bad_payload, bad_payload])
    with pytest.raises(ConsolidationSemanticGenerationError):
        _resolve(planning, client)
    assert client.call_count == 2


def test_missing_pair_invalid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_three_facts()))
    block = _prep(planning).blocks[0]
    assert len(block.pair_refs) == 3
    # Drop the last pair -> count 2 != 3.
    bad = _payload_for_block(block)
    bad["decisions"] = bad["decisions"][:-1]
    _exhausts_invalid(planning, bad)


def test_extra_pair_invalid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_three_facts()))
    block = _prep(planning).blocks[0]
    bad = _payload_for_block(block)
    # Duplicate the first decision -> count 4 != 3.
    bad["decisions"] = bad["decisions"] + [dict(bad["decisions"][0])]
    _exhausts_invalid(planning, bad)


def test_duplicate_pair_invalid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_three_facts()))
    block = _prep(planning).blocks[0]
    bad = _payload_for_block(block)
    # Replace decision[1] with a copy of decision[0] -> index 1 ref mismatch.
    bad["decisions"][1] = dict(bad["decisions"][0])
    _exhausts_invalid(planning, bad)


def test_reordered_pair_invalid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_three_facts()))
    block = _prep(planning).blocks[0]
    bad = _payload_for_block(block)
    # Rotate the decisions -> every index has a valid ref but wrong pair.
    d = bad["decisions"]
    bad["decisions"] = [d[1], d[2], d[0]]
    _exhausts_invalid(planning, bad)


def test_swapped_endpoint_invalid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    (left, right) = block.pair_refs[0]
    # Return (right, left): left > right within the item -> typed load fails.
    bad = {
        "decisions": [
            {
                "left_candidate_ref": right,
                "right_candidate_ref": left,
                "decision": "same_fact",
                "reason_zh": "r",
                "evidence_selectors": [],
            }
        ]
    }
    _exhausts_invalid(planning, bad)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


def _selectors_specs(left_paras, right_paras):
    """Two facts whose evidence cites the given (non-sequential) paragraphs.

    The chunk's paragraph set is the FULL sequential range covering every cited
    paragraph (the chapter requires sequential, deterministic paragraph IDs); the
    facts cite the specific subset.
    """
    all_paras = set(left_paras) | set(right_paras)
    max_idx = max(int(p.rsplit("_P", 1)[1]) for p in all_paras)
    seq_paras = tuple(f"CH001_P{i:04d}" for i in range(1, max_idx + 1))
    return [
        ChunkSpec(
            "CH001",
            seq_paras,
            chars=(_char("cand_char_001", "Alice"),),
            facts=(
                _mk_fact(
                    "cand_fact_001", statement_zh="L", para=left_paras[0],
                    evidence=tuple(_ev(p) for p in left_paras),
                ),
                _mk_fact(
                    "cand_fact_002", statement_zh="R", para=right_paras[0],
                    evidence=tuple(_ev(p) for p in right_paras),
                ),
            ),
        )
    ]


def test_l0_r0_success(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_selectors_specs(("CH001_P0001",), ("CH001_P0002",))))
    block = _prep(planning).blocks[0]
    client = FakeLLMClient([_payload_for_block(block, selectors={0: ["L0", "R0"]})])
    result = _resolve(planning, client)
    assert client.call_count == 1
    (decision,) = result.semantic_decisions
    # L0 -> left evidence[0], R0 -> right evidence[0] (canonical L-then-R order).
    assert [e.paragraph_id for e in decision.evidence_refs] == [
        "CH001_P0001", "CH001_P0002"
    ]


def test_permuted_selector_order_canonicalized(tmp_path):
    left_paras = ("CH001_P0001", "CH001_P0003")
    right_paras = ("CH001_P0002", "CH001_P0004", "CH001_P0005")
    planning = _planning(_tree(tmp_path, specs=_selectors_specs(left_paras, right_paras)))
    block = _prep(planning).blocks[0]
    # Provider returns a permuted selector order.
    client = FakeLLMClient([_payload_for_block(block, selectors={0: ["R2", "L1", "R0", "L0"]})])
    result = _resolve(planning, client)
    (decision,) = result.semantic_decisions
    # Canonical: all L ascending, then all R ascending.
    # L0->P0001, L1->P0003, R0->P0002, R2->P0005.
    assert [e.paragraph_id for e in decision.evidence_refs] == [
        "CH001_P0001", "CH001_P0003", "CH001_P0002", "CH001_P0005"
    ]


def test_canonicalization_is_persisted_order_and_decision_id(tmp_path):
    """Equivalent selector permutations persist the SAME EvidenceRefs + id."""
    left_paras = ("CH001_P0001", "CH001_P0003")
    right_paras = ("CH001_P0002", "CH001_P0004")
    planning = _planning(_tree(tmp_path, specs=_selectors_specs(left_paras, right_paras)))
    block = _prep(planning).blocks[0]
    c1 = FakeLLMClient([_payload_for_block(block, selectors={0: ["L0", "R1"]})])
    c2 = FakeLLMClient([_payload_for_block(block, selectors={0: ["R1", "L0"]})])
    r1 = _resolve(planning, c1)
    r2 = _resolve(planning, c2)
    assert [e.to_dict() for e in r1.semantic_decisions[0].evidence_refs] == [
        e.to_dict() for e in r2.semantic_decisions[0].evidence_refs
    ]
    assert r1.semantic_decisions[0].decision_id == r2.semantic_decisions[0].decision_id


def test_duplicate_selector_invalid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    bad = _payload_for_block(block, selectors={0: ["L0", "L0"]})
    _exhausts_invalid(planning, bad)


@pytest.mark.parametrize("sel", ["l0", "X0", "L-1", "R1.0", "L", "L01foo"])
def test_bad_selector_syntax_invalid(tmp_path, sel):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    bad = _payload_for_block(block, selectors={0: [sel]})
    _exhausts_invalid(planning, bad)


def test_out_of_range_selector_invalid(tmp_path):
    # Left has one evidence item -> L1 is out of range.
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    bad = _payload_for_block(block, selectors={0: ["L1"]})
    _exhausts_invalid(planning, bad)


def test_same_evidence_ref_alias_dedupe(tmp_path):
    """L0 and R0 both resolve to the SAME exact EvidenceRef -> one persisted."""
    # Both facts cite the same paragraph with identical role/strength/excerpt.
    planning = _planning(
        _tree(tmp_path, specs=_selectors_specs(("CH001_P0001",), ("CH001_P0001",)))
    )
    block = _prep(planning).blocks[0]
    client = FakeLLMClient([_payload_for_block(block, selectors={0: ["L0", "R0"]})])
    result = _resolve(planning, client)
    (decision,) = result.semantic_decisions
    # Two legal selectors, one exact-identical EvidenceRef -> deduped to one.
    assert len(decision.evidence_refs) == 1
    assert decision.evidence_refs[0].paragraph_id == "CH001_P0001"


def test_empty_selectors_valid(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    block = _prep(planning).blocks[0]
    client = FakeLLMClient([_payload_for_block(block, selectors={0: []})])
    result = _resolve(planning, client)
    (decision,) = result.semantic_decisions
    assert decision.evidence_refs == ()


# ---------------------------------------------------------------------------
# Decision identity (backend-neutral)
# ---------------------------------------------------------------------------


def _ev_tuple():
    return (_ev("CH001_P0001"),)


def test_decision_id_deterministic_and_sensitivity():
    base_kwargs = dict(
        left_ref="CH001_C001:cand_fact_001",
        right_ref="CH001_C001:cand_fact_002",
        decision="same_fact",
        reason_zh="reason",
        evidence_refs=_ev_tuple(),
        prompt_id="a5.fact-consolidation",
        prompt_version=1,
        request_hash="f" * 64,
    )
    base = compute_fact_llm_decision_id(**base_kwargs)
    assert base == compute_fact_llm_decision_id(**base_kwargs)
    assert base.startswith("dec_") and len(base) == len("dec_") + 20
    # reason change -> different
    assert compute_fact_llm_decision_id(**{**base_kwargs, "reason_zh": "other"}) != base
    # resolved evidence change -> different
    assert compute_fact_llm_decision_id(
        **{**base_kwargs, "evidence_refs": (_ev("CH001_P0002"),)}
    ) != base
    # request_hash change -> different
    assert compute_fact_llm_decision_id(**{**base_kwargs, "request_hash": "0" * 64}) != base
    # decision change -> different
    assert compute_fact_llm_decision_id(**{**base_kwargs, "decision": "conflict"}) != base


def test_provider_metadata_change_alone_keeps_decision_id(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_two_facts()))
    responses = _valid_responses(planning)
    c1 = FakeLLMClient(list(responses), provider_family="qwen", request_model="qwen3-27b")
    c2 = FakeLLMClient(
        list(responses), provider_family="openai", request_model="some-other-model"
    )
    r1 = _resolve(planning, c1)
    r2 = _resolve(planning, c2)
    assert r1.semantic_decisions[0].decision_id == r2.semantic_decisions[0].decision_id
    # The provenance metadata itself DID differ (recorded audit provenance).
    assert r1.semantic_decisions[0].generation_provenance.provider_family == "qwen"
    assert r2.semantic_decisions[0].generation_provenance.provider_family == "openai"


# ---------------------------------------------------------------------------
# Whole-fact decision coverage (auto_same + LLM semantic)
# ---------------------------------------------------------------------------


def test_whole_fact_coverage_auto_same_plus_llm(tmp_path):
    planning = _planning(_tree(tmp_path, specs=_corpus_specs()))
    auto_pairs = {
        frozenset((p.left_ref, p.right_ref))
        for p in planning.fact_pair_plans
        if p.state == "auto_same"
    }
    assert auto_pairs, "corpus must contain an auto_same fact pair"
    client = FakeLLMClient(_valid_responses(planning))
    result = _resolve(planning, client)

    all_pairs = {(p.left_ref, p.right_ref) for p in planning.fact_pair_plans}
    result_pairs = {
        (d.left_candidate_ref, d.right_candidate_ref) for d in result.all_fact_decisions
    }
    # Exact whole-fact coverage: every explicit pair, no extra, no duplicate.
    assert result_pairs == all_pairs
    assert len(result.all_fact_decisions) == len(planning.fact_pair_plans)
    # Canonical (left, right) order.
    keys = [(d.left_candidate_ref, d.right_candidate_ref) for d in result.all_fact_decisions]
    assert keys == sorted(keys)
    # State/method consistency.
    for d in result.all_fact_decisions:
        key = (d.left_candidate_ref, d.right_candidate_ref)
        if frozenset(key) in auto_pairs:
            assert d.method == "deterministic" and d.decision == "same_fact"
        else:
            assert d.method == "llm"


def test_empty_semantic_stream_no_provider_calls(tmp_path):
    # A single fact -> no pairs -> zero blocks / zero provider calls.
    specs = [
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(_mk_fact("cand_fact_001", para="CH001_P0001"),),
        )
    ]
    planning = _planning(_tree(tmp_path, specs=specs))
    client = FakeLLMClient([])
    result = _resolve(planning, client)
    assert client.call_count == 0
    assert result.semantic_decisions == ()
    assert result.block_results == ()
    assert result.all_fact_decisions == ()


# ---------------------------------------------------------------------------
# Block atomicity
# ---------------------------------------------------------------------------


def test_block_atomicity_partial_invalid_discards_whole_block(tmp_path):
    # A block with multiple pairs: one valid decision + one invalid selector.
    left_paras = ("CH001_P0001", "CH001_P0002")
    right_paras = ("CH001_P0003",)
    # 3 facts -> 3 pairs in one block.
    planning = _planning(_tree(tmp_path, specs=_three_facts()))
    block = _prep(planning).blocks[0]
    assert block.pair_count == 3
    # Pair 0 valid (empty selectors), pair 1 invalid selector, pair 2 valid.
    bad = _payload_for_block(block, selectors={0: [], 1: ["L99"], 2: []})
    # Round 1 invalid (L99 out of range), round 2 all valid -> success.
    valid = _payload_for_block(block)
    client = FakeLLMClient([bad, valid])
    result = _resolve(planning, client)
    assert client.call_count == 2
    # The whole block is (re)produced on round 2; all 3 decisions retained.
    assert len(result.semantic_decisions) == 3
    assert result.block_results[0].semantic_rounds == 2
    # Round 1's partial (2 valid) decisions were NOT persisted.
    assert result.block_results[0].request_hash == client.request_hashes[0]
