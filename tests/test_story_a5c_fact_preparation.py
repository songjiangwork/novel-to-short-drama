"""v1.2 A5C-A -- fact semantic preparation (zero provider).

Covers A5C-A BLOCK 1 / 2 / 4 / 5 / 6 implemented in
``short_drama.story.consolidation_semantic``:

  * deterministic ``a5fblk_`` block ids (first 20 hex of the sha256 of the
    canonical block payload; distinct from A4 ``a4rblk_``);
  * the immutable ``FactSemanticPreparation`` (A5B plan + blocks + real
    ``StructuredGenerationRequest`` objects + stable per-request hashes +
    asset identity), with alignment invariants enforced fail closed;
  * fail-closed verification of the exact consolidation profile + prompt +
    schema + semantic-profile identity (BLOCK 4);
  * deterministic block packing ordered by ``(left_ref, right_ref)`` at most the
    requested max size, where the fact semantic stream is ONLY the
    ``needs_semantic_decision`` fact pairs and ``auto_same`` pairs are excluded
    (BLOCK 5);
  * pair-local endpoint packets derived from ``FactCandidate`` carrying the
    ``source_order_key`` and the exact fact-pair signals, with pair-local
    evidence selectors ``L0/L1/...`` / ``R0/R1/...`` and NO block-wide evidence
    pool (BLOCK 6);
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

from short_drama.artifacts.canonical import canonical_json_bytes, content_hash
from short_drama.llm import PromptRegistry
from short_drama.llm.models import StructuredGenerationRequest
from short_drama.story import (
    A5C_BLOCK_PREFIX,
    A5C_DEFAULT_FACT_BLOCK_SIZE,
    A5C_FACT_OUTPUT_SCHEMA_ID,
    A5C_FACT_OUTPUT_SCHEMA_VERSION,
    A5C_FACT_PROMPT_ID,
    A5C_FACT_PROMPT_VERSION,
    A5C_FACT_SEMANTIC_PROFILE_ID,
    DEFAULT_PROMPT_BASE_DIR,
    FactPairPlan,
    FactSemanticBlock,
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
    para: str,
) -> FactCandidate:
    return FactCandidate(
        candidate_id=cid,
        fact_type=fact_type,
        statement_zh=statement_zh,
        subject_refs=tuple(subject_refs),
        object_refs=tuple(object_refs),
        evidence_strength="explicit",
        evidence=(_ev(para),),
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


def _tree(tmp_path: Path):
    return _build_multi_chunk_tree(tmp_path, *_corpus_specs())


def _planning(tree):
    return build_consolidation_planning(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=_consolidation_profile(),
    )


def _prep(tree, *, max_block_size: int = A5C_DEFAULT_FACT_BLOCK_SIZE,
          profile=None, sem_profile=_SEM_PROFILE):
    planning = _planning(tree)
    prep = build_fact_semantic_preparation(
        planning, profile if profile is not None else _consolidation_profile(),
        sem_profile, prompts=_PROMPTS, max_block_size=max_block_size,
    )
    return planning, prep


def _semantic_refs(planning):
    return [
        (p.left_ref, p.right_ref)
        for p in planning.fact_pair_plans
        if p.state == "needs_semantic_decision"
    ]


# ---------------------------------------------------------------------------
# BLOCK 1 -- deterministic block id
# ---------------------------------------------------------------------------


def test_block_id_shape_and_distinct_from_a4():
    payload = [{"left_candidate_ref": "A", "right_candidate_ref": "B"}]
    block = FactSemanticBlock(
        block_id=A5C_BLOCK_PREFIX + content_hash(payload)[:20],
        pair_contexts=tuple(payload),
        pair_contexts_json=canonical_json_bytes(payload).decode("utf-8"),
        pair_refs=(("A", "B"),),
    )
    assert block.block_id.startswith(A5C_BLOCK_PREFIX)
    assert len(block.block_id) == len(A5C_BLOCK_PREFIX) + 20
    # Distinct from the A4 block id prefix.
    assert not block.block_id.startswith("a4rblk_")


def test_block_id_matches_canonical_payload():
    payload = [{"left_candidate_ref": "A", "right_candidate_ref": "B"}]
    correct = A5C_BLOCK_PREFIX + content_hash(payload)[:20]
    FactSemanticBlock(
        block_id=correct,
        pair_contexts=tuple(payload),
        pair_contexts_json=canonical_json_bytes(payload).decode("utf-8"),
        pair_refs=(("A", "B"),),
    )
    with pytest.raises(StoryIntegrityError, match="block_id"):
        FactSemanticBlock(
            block_id=A5C_BLOCK_PREFIX + "1" * 20,  # wrong hash
            pair_contexts=tuple(payload),
            pair_contexts_json=canonical_json_bytes(payload).decode("utf-8"),
            pair_refs=(("A", "B"),),
        )


def test_block_id_changes_with_payload():
    p1 = [{"left_candidate_ref": "A", "right_candidate_ref": "B"}]
    p2 = [{"left_candidate_ref": "A", "right_candidate_ref": "C"}]
    assert content_hash(p1) != content_hash(p2)
    assert (A5C_BLOCK_PREFIX + content_hash(p1)[:20]) != (A5C_BLOCK_PREFIX + content_hash(p2)[:20])


# ---------------------------------------------------------------------------
# BLOCK 2 -- FactSemanticPreparation result + alignment invariants
# ---------------------------------------------------------------------------


def test_preparation_result_and_invariants(tmp_path):
    planning, prep = _prep(_tree(tmp_path))
    assert isinstance(prep, FactSemanticPreparation)
    assert len(prep.blocks) == len(prep.structured_requests) == len(prep.semantic_request_hashes)
    assert prep.semantic_request_hashes == tuple(
        r.request_hash for r in prep.structured_requests
    )
    assert sum(b.pair_count for b in prep.blocks) == prep.semantic_pair_count
    # Asset identity fields are the frozen A5C-A identity.
    assert prep.prompt_id == A5C_FACT_PROMPT_ID
    assert prep.prompt_version == A5C_FACT_PROMPT_VERSION
    assert prep.output_schema_id == A5C_FACT_OUTPUT_SCHEMA_ID
    assert prep.output_schema_version == A5C_FACT_OUTPUT_SCHEMA_VERSION
    assert prep.semantic_profile.profile_id == A5C_FACT_SEMANTIC_PROFILE_ID
    assert prep.working_language == "zh"
    assert prep.max_block_size == A5C_DEFAULT_FACT_BLOCK_SIZE
    # Every block carries only the semantic (needs_semantic_decision) pairs.
    planned_semantic = set(_semantic_refs(planning))
    block_pairs = {(l, r) for b in prep.blocks for (l, r) in b.pair_refs}
    assert block_pairs == planned_semantic


def test_preparation_alignment_invariants_fail_closed(tmp_path):
    _, prep = _prep(_tree(tmp_path))
    with pytest.raises(StoryIntegrityError, match="semantic request hash"):
        dataclasses.replace(prep, semantic_request_hashes=prep.semantic_request_hashes[:-1])
    bad_hashes = prep.semantic_request_hashes[:-1] + ("0" * 64,)
    with pytest.raises(StoryIntegrityError, match="semantic_request_hashes"):
        dataclasses.replace(prep, semantic_request_hashes=bad_hashes)


# ---------------------------------------------------------------------------
# BLOCK 4 -- fail-closed identity verification
# ---------------------------------------------------------------------------


def _fact_pass(profile, **overrides):
    base = {
        "semantic_profile_id": profile.fact.semantic_profile_id,
        "prompt_id": profile.fact.prompt_id,
        "prompt_version": profile.fact.prompt_version,
        "output_schema_id": profile.fact.output_schema_id,
        "output_schema_version": profile.fact.output_schema_version,
    }
    base.update(overrides)
    return ConsolidationSemanticPass(**base)


def test_wrong_fact_prompt_fails_closed(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    bad = dataclasses.replace(profile, fact=_fact_pass(profile, prompt_id="a5.event-consolidation"))
    with pytest.raises(StoryIntegrityError, match="prompt"):
        build_fact_semantic_preparation(planning, bad, _SEM_PROFILE, prompts=_PROMPTS)


def test_wrong_fact_schema_fails_closed(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    bad = dataclasses.replace(profile, fact=_fact_pass(profile, output_schema_id="some-other-schema"))
    with pytest.raises(StoryIntegrityError, match="output schema"):
        build_fact_semantic_preparation(planning, bad, _SEM_PROFILE, prompts=_PROMPTS)


def test_wrong_semantic_profile_id_fails_closed(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    profile = _consolidation_profile()
    bad_sem = dataclasses.replace(_SEM_PROFILE, profile_id="some-other-llm")
    with pytest.raises(StoryIntegrityError, match="semantic profile id"):
        build_fact_semantic_preparation(planning, profile, bad_sem, prompts=_PROMPTS)


def test_identity_is_frozen_and_verified(tmp_path):
    planning, prep = _prep(_tree(tmp_path))
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
    assert identity["max_block_size"] == prep.max_block_size


# ---------------------------------------------------------------------------
# BLOCK 5 -- deterministic packing (semantic stream only, auto_same excluded)
# ---------------------------------------------------------------------------


def test_packing_deterministic_ordered_and_bounded(tmp_path):
    planning, prep = _prep(_tree(tmp_path), max_block_size=3)
    refs_sorted = sorted(_semantic_refs(planning))
    block_pairs = [(l, r) for b in prep.blocks for (l, r) in b.pair_refs]
    assert block_pairs == refs_sorted
    assert all(b.pair_count <= 3 for b in prep.blocks)
    if len(prep.blocks) > 1:
        assert all(b.pair_count == 3 for b in prep.blocks[:-1])
        assert 1 <= prep.blocks[-1].pair_count <= 3
    # Deterministic: a second preparation is byte-identical.
    _, prep2 = _prep(_tree(tmp_path), max_block_size=3)
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
        planning, _consolidation_profile(), _SEM_PROFILE, prompts=_PROMPTS
    )
    block_pairs = {frozenset((l, r)) for b in prep.blocks for (l, r) in b.pair_refs}
    assert not (block_pairs & auto_pairs)
    assert prep.auto_same_pair_count == len(auto_pairs)
    assert prep.semantic_pair_count + prep.auto_same_pair_count == prep.total_fact_pair_count


def test_single_pair_block_max_size_1(tmp_path):
    planning, prep = _prep(_tree(tmp_path), max_block_size=1)
    assert all(b.pair_count == 1 for b in prep.blocks)
    assert len(prep.blocks) == prep.semantic_pair_count


def test_max_block_size_must_be_positive(tmp_path):
    tree = _tree(tmp_path)
    planning = _planning(tree)
    with pytest.raises(StoryIntegrityError, match="max_block_size"):
        build_fact_semantic_preparation(
            planning, _consolidation_profile(), _SEM_PROFILE, prompts=_PROMPTS,
            max_block_size=0,
        )


def test_empty_semantic_stream_yields_no_blocks(tmp_path):
    # A corpus with a single fact produces no pairs -> zero blocks / requests.
    from test_story_a5b_audit import ChunkSpec as _CS
    from test_story_a5b_audit import _build_multi_chunk_tree as _btree

    tree = _btree(
        tmp_path,
        _CS(
            "CH001", ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            facts=(_mk_fact("cand_fact_001", subject_refs=("cand_char_001",),
                            para="CH001_P0001"),),
        ),
    )
    planning, prep = _prep(tree)
    assert prep.total_fact_pair_count == 0
    assert prep.semantic_pair_count == 0
    assert prep.auto_same_pair_count == 0
    assert prep.blocks == ()
    assert prep.structured_requests == ()
    assert prep.semantic_request_hashes == ()
    identity = build_fact_semantic_identity(prep)
    assert identity["block_count"] == 0


# ---------------------------------------------------------------------------
# BLOCK 6 -- pair-local endpoint packets (derived from FactCandidate)
# ---------------------------------------------------------------------------


def test_endpoint_packet_fields_and_source_order_key(tmp_path):
    planning, prep = _prep(_tree(tmp_path))
    facts_by_ref = {c.global_candidate_ref: c for c in planning.index.facts}
    for block in prep.blocks:
        for pc in block.pair_contexts:
            for side, prefix in (("left", "L"), ("right", "R")):
                packet = pc[side]
                cand = facts_by_ref[packet["candidate_ref"]]
                assert packet["chunk_id"] == cand.chunk_id
                assert packet["local_candidate_id"] == cand.local_candidate_id
                assert packet["fact_type"] == cand.fact_type
                assert packet["statement_zh"] == cand.statement_zh
                assert packet["subject_refs"] == list(cand.subject_refs)
                assert packet["object_refs"] == list(cand.object_refs)
                # source_order_key is included in the endpoint packet (BLOCK 6).
                assert packet["source_order_key"] == cand.source_order_key
                # The evidence selectors are pair-local (L0.. / R0..) and map to
                # the candidate's own evidence in canonical (sorted) order.
                selectors = [ev["selector"] for ev in packet["evidence"]]
                assert selectors == [f"{prefix}{idx}" for idx in range(len(packet["evidence"]))]
                assert [ev["paragraph_id"] for ev in packet["evidence"]] == sorted(
                    e.paragraph_id for e in cand.evidence_refs
                )


def test_pair_context_carries_signals(tmp_path):
    planning, prep = _prep(_tree(tmp_path))
    plan_by_refs = {(p.left_ref, p.right_ref): p for p in planning.fact_pair_plans}
    for block in prep.blocks:
        for pc in block.pair_contexts:
            plan = plan_by_refs[(pc["left_candidate_ref"], pc["right_candidate_ref"])]
            assert pc["signals"] == list(plan.signals)
            assert pc["left_candidate_ref"] == plan.left_ref
            assert pc["right_candidate_ref"] == plan.right_ref


def test_endpoint_packet_helper_pure():
    from short_drama.artifacts import ArtifactRef

    cand = IndexedFactCandidate(
        global_candidate_ref="CH001_C001:cand_fact_001", chunk_id="CH001_C001",
        local_candidate_id="cand_fact_001",
        source_order_key="000001:000000001:01:000000001:CH001_C001:cand_fact_001",
        fact_type="world_fact", statement_zh="stmt",
        subject_refs=("char_0001",), object_refs=(),
        evidence_strength="explicit",
        evidence_refs=(_ev("CH001_P0002"), _ev("CH001_P0001")),
        candidate_extraction_ref=ArtifactRef(
            artifact_type="candidate_extraction", artifact_id="x", revision=1,
            content_hash="f" * 64,
        ),
    )
    left = build_fact_endpoint_packet(cand, "left")
    right = build_fact_endpoint_packet(cand, "right")
    assert [ev["selector"] for ev in left["evidence"]] == ["L0", "L1"]
    assert [ev["selector"] for ev in right["evidence"]] == ["R0", "R1"]
    # The evidence order is the canonical (sorted) order, not the input order.
    assert [ev["paragraph_id"] for ev in left["evidence"]] == ["CH001_P0001", "CH001_P0002"]
    assert left["source_order_key"] == cand.source_order_key
    with pytest.raises(StoryIntegrityError, match="side"):
        build_fact_endpoint_packet(cand, "middle")


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
# Real StructuredGenerationRequest rendering (zero provider, provider-neutral)
# ---------------------------------------------------------------------------


def test_real_structured_requests_rendered(tmp_path):
    planning, prep = _prep(_tree(tmp_path))
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
