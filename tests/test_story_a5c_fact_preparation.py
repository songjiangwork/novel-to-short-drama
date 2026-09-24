"""v1.2 A5C-A -- fact semantic preparation (zero provider).

Covers A5C-A BLOCK 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 implemented in
``short_drama.story.consolidation_semantic``:

  * explicit ``FactSemanticPackingPolicy`` with TWO independent limits
    (``max_pairs_per_block`` AND ``max_candidates_per_block``); the audit
    candidates are P1 6/12, P2 12/24, P3 24/48 (BLOCK 1);
  * no production default: the packing policy is REQUIRED (BLOCK 2);
  * each ``FactSemanticBlock`` carries ``candidate_refs`` (the stable unique
    union of its pair endpoints, ordered by candidate source_order_key then
    ref) (BLOCK 3);
  * block identity binds the frozen material (plan hash, packing limits, block
    ordinal, ordered pairs, ordered candidate refs) (BLOCK 4);
  * the fact semantic stream is validated fail closed before packing: fact
    namespace, ``left_ref < right_ref``, no duplicate pair, canonical order,
    exact ``needs_semantic_decision`` coverage, auto_same excluded (BLOCK 8);
  * pair-local endpoint packets derived from ``IndexedFactCandidate`` carry the
    exact ``evidence_strength`` and ``source_order_key``, with pair-local
    evidence selectors ``L0/L1/...`` / ``R0/R1/...`` in the EXACT indexed A5B
    evidence order (no re-sort) and NO block-wide evidence pool (BLOCK 5 / 6);
  * fail-closed verification of the consolidation profile identity INCLUDING
    ``max_generation_rounds == 2`` (BLOCK 7);
  * real ``StructuredGenerationRequest`` objects rendered through the existing
    ``PromptRegistry`` / ``OutputSchema`` / ``SemanticLLMProfile``
    infrastructure (zero provider, provider-neutral request material).

All fixtures are self-contained synthetic run trees (no dependency on the local
Alice run tree; the Alice gate lives in ``test_story_a5c_fact_audit``). No
provider is called and nothing is persisted anywhere in this file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from short_drama.llm import PromptRegistry
from short_drama.llm.models import StructuredGenerationRequest
from short_drama.story import (
    A5C_BLOCK_PREFIX,
    A5C_FACT_ENDPOINT_PACKET_FIELDS,
    A5C_FACT_MAX_GENERATION_ROUNDS,
    A5C_FACT_OUTPUT_SCHEMA_ID,
    A5C_FACT_OUTPUT_SCHEMA_VERSION,
    A5C_FACT_PROMPT_ID,
    A5C_FACT_PROMPT_VERSION,
    A5C_FACT_SEMANTIC_PROFILE_ID,
    A5C_PACKING_CANDIDATES,
    DEFAULT_PROMPT_BASE_DIR,
    FactPairPlan,
    FactSemanticBlock,
    FactSemanticPackingPolicy,
    FactSemanticPreparation,
    StoryIntegrityError,
    build_consolidation_planning,
    build_fact_endpoint_packet,
    build_fact_pair_context,
    build_fact_semantic_identity,
    build_fact_semantic_preparation,
    load_fact_semantic_profile,
)
from short_drama.story.consolidation import ConsolidationSemanticPass, IndexedFactCandidate
from short_drama.story.consolidation_semantic import _compute_block_id
from short_drama.story.extraction import EvidenceRef, FactCandidate
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
_SEM_PROFILE = load_fact_semantic_profile()
# Audit candidates (test convenience): P1 6/12, P2 12/24, P3 24/48.
_P1, _P2, _P3 = A5C_PACKING_CANDIDATES


def _ev(paragraph_id: str) -> EvidenceRef:
    return EvidenceRef(
        paragraph_id=paragraph_id, role="primary", strength="explicit", excerpt="txt"
    )


def _mk_fact(
    cid: str,
    *,
    fact_type: str = "world_fact",
    statement_zh: str = "stmt",
    subject_refs=(),
    object_refs=(),
    evidence_strength: str = "explicit",
    para: str | None = None,
    evidence: tuple[EvidenceRef, ...] | None = None,
) -> FactCandidate:
    if evidence is None:
        evidence = (_ev(para if para is not None else "CH001_P0001"),)
    return FactCandidate(
        candidate_id=cid,
        fact_type=fact_type,
        statement_zh=statement_zh,
        subject_refs=tuple(subject_refs),
        object_refs=tuple(object_refs),
        evidence_strength=evidence_strength,
        evidence=tuple(evidence),
    )


def _corpus_specs():
    """A two-chunk corpus with a known mix of semantic + auto_same fact pairs.

    CH001: cand_fact_001 (world_fact) is an exact-safe duplicate of
    cand_fact_002 (auto_same). cand_fact_003 (identity) forms same-chunk
    semantic pairs with 001/002. CH002: cand_fact_004 / cand_fact_005
    (world_fact) form same-chunk + adjacent-chunk semantic pairs.
    """
    return [
        ChunkSpec(
            "CH001",
            ("CH001_P0001", "CH001_P0002"),
            chars=(_char("cand_char_001", "Alice"),),
            locs=(_loc("cand_loc_001", "Wonderland"),),
            facts=(
                _mk_fact("cand_fact_001", statement_zh="Alice lives here",
                         subject_refs=("cand_char_001",), para="CH001_P0001"),
                _mk_fact("cand_fact_002", statement_zh="  alice   lives here ",
                         subject_refs=("cand_char_001",), para="CH001_P0002"),
                _mk_fact("cand_fact_003", fact_type="identity", statement_zh="Alice is brave",
                         subject_refs=("cand_char_001",), object_refs=("cand_loc_001",),
                         para="CH001_P0001"),
            ),
        ),
        ChunkSpec(
            "CH002",
            ("CH002_P0001",),
            chars=(_char("cand_char_002", "Bob"),),
            facts=(
                _mk_fact("cand_fact_004", statement_zh="Bob lives here",
                         subject_refs=("cand_char_002",), para="CH002_P0001"),
                _mk_fact("cand_fact_005", statement_zh="Bob runs fast",
                         subject_refs=("cand_char_002",), para="CH002_P0001"),
            ),
        ),
    ]


def _single_chunk_specs(n_facts: int):
    """One chunk with ``n_facts`` distinct facts (all-pairs semantic)."""
    paras = tuple(f"CH001_P{i:04d}" for i in range(1, n_facts + 1))
    return [
        ChunkSpec(
            "CH001",
            paras,
            chars=(_char("cand_char_001", "Alice"),),
            facts=tuple(
                _mk_fact(f"cand_fact_{i:03d}", statement_zh=f"stmt {i}", para=paras[i - 1])
                for i in range(1, n_facts + 1)
            ),
        )
    ]


def _tree(tmp_path: Path, specs=None):
    return _build_multi_chunk_tree(tmp_path, *(specs if specs is not None else _corpus_specs()))


def _planning(tree):
    return build_consolidation_planning(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=_consolidation_profile(),
    )


def _prep(tree, *, policy: FactSemanticPackingPolicy = _P2,
          profile=None, sem_profile=_SEM_PROFILE):
    planning = _planning(tree)
    prep = build_fact_semantic_preparation(
        planning,
        profile if profile is not None else _consolidation_profile(),
        sem_profile,
        prompts=_PROMPTS,
        packing_policy=policy,
    )
    return planning, prep


def _semantic_refs(planning):
    return [
        (p.left_ref, p.right_ref)
        for p in planning.fact_pair_plans
        if p.state == "needs_semantic_decision"
    ]


# ---------------------------------------------------------------------------
# BLOCK 1 -- explicit packing policy with TWO independent limits
# ---------------------------------------------------------------------------


def test_audit_candidates_are_exactly_p1_p2_p3():
    assert [(p.name, p.max_pairs_per_block, p.max_candidates_per_block)
            for p in A5C_PACKING_CANDIDATES] == [
        ("P1", 6, 12),
        ("P2", 12, 24),
        ("P3", 24, 48),
    ]
    for policy in A5C_PACKING_CANDIDATES:
        # The candidate limit is exactly twice the pair limit (2 endpoints/pair).
        assert policy.max_candidates_per_block == 2 * policy.max_pairs_per_block


@pytest.mark.parametrize("pairs,cands", [(0, 3), (1, 0), (1, 1), (-1, 3), (3, 2.0)])
def test_packing_policy_requires_valid_limits(pairs, cands):
    with pytest.raises(StoryIntegrityError):
        FactSemanticPackingPolicy("T", pairs, cands)
    with pytest.raises(StoryIntegrityError):
        FactSemanticPackingPolicy("", 6, 12)


def test_candidate_limit_tighter_than_pair_limit(tmp_path):
    """BLOCK 12: the unique-candidate limit is enforced independently of pairs.

    With a 6-fact single chunk (15 semantic pairs) and policy 10 pairs / 3
    candidates, every block must respect BOTH limits, the candidate limit is hit
    (== 3), and the pair limit is never the binding constraint (< 10 pairs).
    """
    tree = _tree(tmp_path, specs=_single_chunk_specs(6))
    policy = FactSemanticPackingPolicy("TIGHT", 10, 3)
    planning, prep = _prep(tree, policy=policy)
    assert prep.packing_policy is policy
    assert len(prep.blocks) > 0
    assert all(b.pair_count <= 10 for b in prep.blocks)
    assert all(len(b.candidate_refs) <= 3 for b in prep.blocks)
    # The candidate limit is hit ...
    assert max(len(b.candidate_refs) for b in prep.blocks) == 3
    # ... while the pair limit is never the constraint that closes a block.
    assert max(b.pair_count for b in prep.blocks) < 10
    # Far more blocks than pair-limit-only packing (ceil(15/10) == 2) would give.
    assert len(prep.blocks) > 2
    # No pair is lost or duplicated.
    assert sum(b.pair_count for b in prep.blocks) == prep.semantic_pair_count


# ---------------------------------------------------------------------------
# BLOCK 2 -- no production default: the packing policy is REQUIRED
# ---------------------------------------------------------------------------


def test_packing_policy_is_required(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    # No production default: the packing policy is REQUIRED (no keyword default).
    with pytest.raises(TypeError):
        build_fact_semantic_preparation(
            planning, _consolidation_profile(), _SEM_PROFILE, prompts=_PROMPTS
        )
    # An explicit None is rejected fail-closed.
    with pytest.raises(StoryIntegrityError, match="packing_policy is required"):
        build_fact_semantic_preparation(
            planning, _consolidation_profile(), _SEM_PROFILE,
            prompts=_PROMPTS, packing_policy=None,
        )


# ---------------------------------------------------------------------------
# BLOCK 3 -- blocks carry candidate_refs (stable union of pair endpoints)
# ---------------------------------------------------------------------------


def test_block_candidate_refs_are_stable_union(tmp_path):
    planning, prep = _prep(_tree(tmp_path), policy=_P2)
    for block in prep.blocks:
        # Recompute the stable union of the block's pair endpoints from the
        # pair contexts, ordered by candidate source_order_key then ref.
        keyed = set()
        for pc in block.pair_contexts:
            for side in ("left", "right"):
                packet = pc[side]
                keyed.add((packet["source_order_key"], packet["candidate_ref"]))
        expected = tuple(ref for _so, ref in sorted(keyed))
        assert tuple(block.candidate_refs) == expected
        # Candidate refs are unique and within the policy limit.
        assert len(set(block.candidate_refs)) == len(block.candidate_refs)
        assert len(block.candidate_refs) <= prep.packing_policy.max_candidates_per_block
        # candidate_refs are exactly the endpoints of the block's pairs.
        endpoint_union = {ref for (l, r) in block.pair_refs for ref in (l, r)}
        assert set(block.candidate_refs) == endpoint_union


# ---------------------------------------------------------------------------
# BLOCK 4 -- block identity binds the frozen material
# ---------------------------------------------------------------------------


def test_block_id_binds_plan_hash_policy_and_ordinal():
    pairs = (("A", "B"), ("B", "C"))
    cands = ("A", "B", "C")
    p24 = FactSemanticPackingPolicy("T", 24, 48)
    base = _compute_block_id("plan_hash_X", p24, 0, pairs, cands)
    # Same exact inputs -> same block id.
    assert base == _compute_block_id("plan_hash_X", p24, 0, pairs, cands)
    assert base.startswith(A5C_BLOCK_PREFIX) and len(base) == len(A5C_BLOCK_PREFIX) + 20
    # plan_hash changes -> block id changes.
    assert base != _compute_block_id("plan_hash_Y", p24, 0, pairs, cands)
    # packing limits change -> block identity changes.
    assert base != _compute_block_id("plan_hash_X", FactSemanticPackingPolicy("T2", 12, 24), 0, pairs, cands)
    # block ordinal changes -> block identity changes.
    assert base != _compute_block_id("plan_hash_X", p24, 1, pairs, cands)


def test_block_ids_deterministic_end_to_end(tmp_path):
    _, prep1 = _prep(_tree(tmp_path), policy=_P3)
    _, prep2 = _prep(_tree(tmp_path), policy=_P3)
    assert [b.block_id for b in prep1.blocks] == [b.block_id for b in prep2.blocks]
    assert prep1.semantic_request_hashes == prep2.semantic_request_hashes
    # Distinct blocks (distinct ordinal/pairs) get distinct ids.
    assert len({b.block_id for b in prep1.blocks}) == len(prep1.blocks)


# ---------------------------------------------------------------------------
# BLOCK 5 -- deterministic packing (semantic stream only, auto_same excluded)
# ---------------------------------------------------------------------------


def test_packing_deterministic_ordered_and_bounded(tmp_path):
    planning, prep = _prep(_tree(tmp_path), policy=FactSemanticPackingPolicy("S3", 3, 6))
    refs_sorted = sorted(_semantic_refs(planning))
    block_pairs = [(l, r) for b in prep.blocks for (l, r) in b.pair_refs]
    assert block_pairs == refs_sorted
    assert all(b.pair_count <= 3 for b in prep.blocks)
    assert all(len(b.candidate_refs) <= 6 for b in prep.blocks)
    if len(prep.blocks) > 1:
        assert all(b.pair_count == 3 for b in prep.blocks[:-1])
        assert 1 <= prep.blocks[-1].pair_count <= 3
    # Deterministic: a second preparation is byte-identical.
    _, prep2 = _prep(_tree(tmp_path), policy=FactSemanticPackingPolicy("S3", 3, 6))
    assert [b.block_id for b in prep.blocks] == [b.block_id for b in prep2.blocks]
    assert prep.semantic_request_hashes == prep2.semantic_request_hashes


def test_auto_same_pairs_excluded(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    auto_pairs = {
        frozenset((p.left_ref, p.right_ref))
        for p in planning.fact_pair_plans
        if p.state == "auto_same"
    }
    assert auto_pairs, "corpus must contain an auto_same fact pair"
    prep = build_fact_semantic_preparation(
        planning, _consolidation_profile(), _SEM_PROFILE, prompts=_PROMPTS,
        packing_policy=_P3,
    )
    block_pairs = {frozenset((l, r)) for b in prep.blocks for (l, r) in b.pair_refs}
    assert not (block_pairs & auto_pairs)
    assert prep.auto_same_pair_count == len(auto_pairs)
    assert prep.semantic_pair_count + prep.auto_same_pair_count == prep.total_fact_pair_count


def test_single_pair_block_policy_pairs_1(tmp_path):
    planning, prep = _prep(_tree(tmp_path), policy=FactSemanticPackingPolicy("S1", 1, 2))
    assert all(b.pair_count == 1 for b in prep.blocks)
    assert len(prep.blocks) == prep.semantic_pair_count


def test_empty_semantic_stream_yields_no_blocks(tmp_path):
    # A corpus with a single fact produces no pairs -> zero blocks / requests.
    tree = _build_multi_chunk_tree(
        tmp_path,
        ChunkSpec(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(_mk_fact("cand_fact_001", subject_refs=("cand_char_001",),
                            para="CH001_P0001"),),
        ),
    )
    planning, prep = _prep(tree, policy=_P3)
    assert planning.plan_hash
    assert prep.total_fact_pair_count == 0
    assert prep.semantic_pair_count == 0
    assert prep.auto_same_pair_count == 0
    assert prep.blocks == ()
    assert prep.structured_requests == ()
    assert prep.semantic_request_hashes == ()
    identity = build_fact_semantic_identity(prep)
    assert identity["block_count"] == 0


# ---------------------------------------------------------------------------
# BLOCK 6 -- pair-local endpoint packets (derived from IndexedFactCandidate)
# ---------------------------------------------------------------------------


def test_endpoint_packet_fields_evidence_strength_and_source_order_key(tmp_path):
    planning, prep = _prep(_tree(tmp_path), policy=_P2)
    facts_by_ref = {c.global_candidate_ref: c for c in planning.index.facts}
    for block in prep.blocks:
        for pc in block.pair_contexts:
            for side, prefix in (("left", "L"), ("right", "R")):
                packet = pc[side]
                cand = facts_by_ref[packet["candidate_ref"]]
                # The required A5C fact endpoint packet fields are all present.
                for field in A5C_FACT_ENDPOINT_PACKET_FIELDS:
                    assert field in packet, f"missing endpoint field {field!r}"
                assert packet["chunk_id"] == cand.chunk_id
                assert packet["local_candidate_id"] == cand.local_candidate_id
                assert packet["fact_type"] == cand.fact_type
                assert packet["statement_zh"] == cand.statement_zh
                assert packet["subject_refs"] == list(cand.subject_refs)
                assert packet["object_refs"] == list(cand.object_refs)
                # BLOCK 6: the exact evidence_strength is carried.
                assert packet["evidence_strength"] == cand.evidence_strength
                # source_order_key is included in the endpoint packet.
                assert packet["source_order_key"] == cand.source_order_key
                # Pair-local selectors (L0.. / R0..) in the EXACT indexed order.
                selectors = [ev["selector"] for ev in packet["evidence"]]
                assert selectors == [f"{prefix}{idx}" for idx in range(len(packet["evidence"]))]
                # The evidence order is the candidate's indexed order (not re-sorted).
                assert [ev["paragraph_id"] for ev in packet["evidence"]] == [
                    e.paragraph_id for e in cand.evidence_refs
                ]


def test_pair_context_carries_signals(tmp_path):
    planning, prep = _prep(_tree(tmp_path), policy=_P2)
    plan_by_refs = {(p.left_ref, p.right_ref): p for p in planning.fact_pair_plans}
    for block in prep.blocks:
        for pc in block.pair_contexts:
            plan = plan_by_refs[(pc["left_candidate_ref"], pc["right_candidate_ref"])]
            assert pc["signals"] == list(plan.signals)
            assert pc["left_candidate_ref"] == plan.left_ref
            assert pc["right_candidate_ref"] == plan.right_ref


def _indexed_cand(evidence):
    from short_drama.artifacts import ArtifactRef

    return IndexedFactCandidate(
        global_candidate_ref="CH001_C001:cand_fact_001", chunk_id="CH001_C001",
        local_candidate_id="cand_fact_001",
        source_order_key="000001:000000001:01:000000001:CH001_C001:cand_fact_001",
        fact_type="world_fact", statement_zh="stmt",
        subject_refs=("char_0001",), object_refs=(),
        evidence_strength="explicit",
        evidence_refs=tuple(evidence),
        candidate_extraction_ref=ArtifactRef(
            artifact_type="candidate_extraction", artifact_id="x", revision=1,
            content_hash="f" * 64,
        ),
    )


def test_evidence_selectors_preserve_indexed_order_not_lexical():
    """BLOCK 5 regression: evidence supplied out of lexical order keeps its order.

    evidence_refs = [P0002, P0003, P0001] (non-lexical). The selectors must map
    L0->P0002, L1->P0003, L2->P0001 (indexed order), NOT the sorted order.
    """
    cand = _indexed_cand(
        [_ev("CH001_P0002"), _ev("CH001_P0003"), _ev("CH001_P0001")]
    )
    left = build_fact_endpoint_packet(cand, "left")
    right = build_fact_endpoint_packet(cand, "right")
    assert [ev["paragraph_id"] for ev in left["evidence"]] == [
        "CH001_P0002", "CH001_P0003", "CH001_P0001",
    ]
    assert [ev["selector"] for ev in left["evidence"]] == ["L0", "L1", "L2"]
    assert [ev["selector"] for ev in right["evidence"]] == ["R0", "R1", "R2"]
    assert left["evidence_strength"] == cand.evidence_strength
    assert left["source_order_key"] == cand.source_order_key
    with pytest.raises(StoryIntegrityError, match="side"):
        build_fact_endpoint_packet(cand, "middle")


def test_left_and_right_can_alias_same_evidence_ref():
    """BLOCK 5: left L0 and right R0 may resolve to the SAME exact EvidenceRef.

    At preparation time they remain TWO legal selector positions (no dedupe).
    """
    shared = _ev("CH001_P0001")
    left_cand = _indexed_cand([shared])
    right_cand = dataclasses.replace(
        left_cand,
        global_candidate_ref="CH001_C001:cand_fact_002",
        local_candidate_id="cand_fact_002",
    )
    left = build_fact_endpoint_packet(left_cand, "left")
    right = build_fact_endpoint_packet(right_cand, "right")
    # Same underlying evidence (paragraph_id) ...
    assert left["evidence"][0]["paragraph_id"] == right["evidence"][0]["paragraph_id"] == "CH001_P0001"
    # ... but two distinct pair-local selector positions.
    assert left["evidence"][0]["selector"] == "L0"
    assert right["evidence"][0]["selector"] == "R0"


def test_pair_context_helper_missing_candidate_fails_closed():
    plan = FactPairPlan(
        left_ref="CH001_C001:cand_fact_001",
        right_ref="CH001_C001:cand_fact_002",
        signals=("same_chunk",),
        state="needs_semantic_decision",
    )
    with pytest.raises(StoryIntegrityError, match="missing from the index"):
        build_fact_pair_context(plan, {})


# ---------------------------------------------------------------------------
# BLOCK 7 -- profile verification must include max_generation_rounds == 2
# ---------------------------------------------------------------------------


def test_max_generation_rounds_must_be_two(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    bad_profile = dataclasses.replace(_consolidation_profile(), max_generation_rounds=3)
    with pytest.raises(StoryIntegrityError, match="max_generation_rounds"):
        build_fact_semantic_preparation(
            planning, bad_profile, _SEM_PROFILE, prompts=_PROMPTS, packing_policy=_P3
        )
    # The pinned value is 2.
    assert A5C_FACT_MAX_GENERATION_ROUNDS == 2


def test_wrong_fact_prompt_fails_closed(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    bad = dataclasses.replace(
        profile,
        fact=ConsolidationSemanticPass(
            **{**profile.fact.to_dict(), "prompt_id": "a5.event-consolidation"}
        ),
    )
    with pytest.raises(StoryIntegrityError, match="prompt"):
        build_fact_semantic_preparation(
            planning, bad, _SEM_PROFILE, prompts=_PROMPTS, packing_policy=_P3
        )


def test_wrong_fact_schema_fails_closed(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    bad = dataclasses.replace(
        profile,
        fact=ConsolidationSemanticPass(
            **{**profile.fact.to_dict(), "output_schema_id": "some-other-schema"}
        ),
    )
    with pytest.raises(StoryIntegrityError, match="output schema"):
        build_fact_semantic_preparation(
            planning, bad, _SEM_PROFILE, prompts=_PROMPTS, packing_policy=_P3
        )


def test_wrong_semantic_profile_id_fails_closed(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    bad_sem = dataclasses.replace(_SEM_PROFILE, profile_id="some-other-llm")
    with pytest.raises(StoryIntegrityError, match="semantic profile id"):
        build_fact_semantic_preparation(
            planning, _consolidation_profile(), bad_sem, prompts=_PROMPTS, packing_policy=_P3
        )


# ---------------------------------------------------------------------------
# BLOCK 8 -- semantic stream validation (fail closed before packing)
# ---------------------------------------------------------------------------


def test_semantic_stream_duplicate_pair_fails_closed(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    p0 = planning.fact_pair_plans[0]
    bad = dataclasses.replace(planning, fact_pair_plans=(p0, p0))
    with pytest.raises(StoryIntegrityError, match="duplicate"):
        build_fact_semantic_preparation(
            bad, _consolidation_profile(), _SEM_PROFILE, prompts=_PROMPTS, packing_policy=_P3
        )


def test_semantic_stream_malformed_order_fails_closed(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    p0, p1 = planning.fact_pair_plans[0], planning.fact_pair_plans[1]
    assert (p0.left_ref, p0.right_ref) < (p1.left_ref, p1.right_ref)
    bad = dataclasses.replace(planning, fact_pair_plans=(p1, p0))
    with pytest.raises(StoryIntegrityError, match="canonical"):
        build_fact_semantic_preparation(
            bad, _consolidation_profile(), _SEM_PROFILE, prompts=_PROMPTS, packing_policy=_P3
        )


# ---------------------------------------------------------------------------
# BLOCK 4 / 11 -- semantic identity pins plan + profiles + policy + requests
# ---------------------------------------------------------------------------


def test_identity_is_frozen_and_verified(tmp_path):
    planning, prep = _prep(_tree(tmp_path), policy=_P2)
    identity = build_fact_semantic_identity(prep)
    assert identity["profile_id"] == "consolidation-v1"
    assert identity["semantic_profile_id"] == A5C_FACT_SEMANTIC_PROFILE_ID
    assert identity["prompt_id"] == A5C_FACT_PROMPT_ID
    assert identity["prompt_version"] == A5C_FACT_PROMPT_VERSION
    assert identity["output_schema_id"] == A5C_FACT_OUTPUT_SCHEMA_ID
    assert identity["output_schema_version"] == A5C_FACT_OUTPUT_SCHEMA_VERSION
    assert identity["plan_hash"] == planning.plan_hash
    assert identity["block_count"] == len(prep.blocks)
    assert identity["semantic_request_hashes"] == tuple(prep.semantic_request_hashes)
    # BLOCK 11: the identity pins the explicit packing policy, not a max size.
    assert identity["packing_policy"] == {
        "max_pairs_per_block": prep.packing_policy.max_pairs_per_block,
        "max_candidates_per_block": prep.packing_policy.max_candidates_per_block,
    }
    assert "max_block_size" not in identity


def test_preparation_result_and_invariants(tmp_path):
    planning, prep = _prep(_tree(tmp_path), policy=_P2)
    assert isinstance(prep, FactSemanticPreparation)
    assert len(prep.blocks) == len(prep.structured_requests) == len(prep.semantic_request_hashes)
    assert prep.semantic_request_hashes == tuple(
        r.request_hash for r in prep.structured_requests
    )
    assert sum(b.pair_count for b in prep.blocks) == prep.semantic_pair_count
    assert [b.block_ordinal for b in prep.blocks] == list(range(len(prep.blocks)))
    # Asset identity fields are the frozen A5C-A identity.
    assert prep.prompt_id == A5C_FACT_PROMPT_ID
    assert prep.prompt_version == A5C_FACT_PROMPT_VERSION
    assert prep.output_schema_id == A5C_FACT_OUTPUT_SCHEMA_ID
    assert prep.output_schema_version == A5C_FACT_OUTPUT_SCHEMA_VERSION
    assert prep.semantic_profile.profile_id == A5C_FACT_SEMANTIC_PROFILE_ID
    assert prep.working_language == "zh"
    # Every block carries only the semantic (needs_semantic_decision) pairs.
    planned_semantic = set(_semantic_refs(planning))
    block_pairs = {(l, r) for b in prep.blocks for (l, r) in b.pair_refs}
    assert block_pairs == planned_semantic


def test_preparation_alignment_invariants_fail_closed(tmp_path):
    _, prep = _prep(_tree(tmp_path), policy=_P2)
    with pytest.raises(StoryIntegrityError, match="semantic request hash"):
        dataclasses.replace(prep, semantic_request_hashes=prep.semantic_request_hashes[:-1])
    bad_hashes = prep.semantic_request_hashes[:-1] + ("0" * 64,)
    with pytest.raises(StoryIntegrityError, match="semantic_request_hashes"):
        dataclasses.replace(prep, semantic_request_hashes=bad_hashes)


# ---------------------------------------------------------------------------
# Real StructuredGenerationRequest rendering (zero provider, provider-neutral)
# ---------------------------------------------------------------------------


def test_real_structured_requests_rendered(tmp_path):
    planning, prep = _prep(_tree(tmp_path), policy=_P2)
    assert all(isinstance(r, StructuredGenerationRequest) for r in prep.structured_requests)
    for block, request in zip(prep.blocks, prep.structured_requests):
        assert request.rendered_prompt.user_text.startswith(
            f"Fact consolidation block: {block.block_id}"
        )
        assert block.pair_contexts_json in request.rendered_prompt.user_text
        assert request.rendered_prompt.rendered_prompt_hash
        _assert_no_forbidden_keys(request.semantic_request_material())
        assert request.semantic_profile.profile_id == A5C_FACT_SEMANTIC_PROFILE_ID
        assert request.output_schema.schema_id == A5C_FACT_OUTPUT_SCHEMA_ID


_FORBIDDEN = (
    "endpoint", "url", "base_url", "hostname", "host", "api_key", "api-key",
    "credential", "token", "authorization", "timeout",
)


def _assert_no_forbidden_keys(value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert str(key) not in _FORBIDDEN, f"forbidden request key: {key!r}"
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def test_preparation_module_does_not_import_transport():
    # The preparation path never references the LLM transport/adapter layer.
    import inspect

    from short_drama.story import consolidation_semantic

    src = inspect.getsource(consolidation_semantic)
    assert "adapter" not in src
    assert "transport" not in src
    assert "http" not in src.lower().replace("https", "")
