"""v1.2 A5D-A -- event + relationship semantic preparation (zero provider).

Covers A5D-A implemented in ``short_drama.story.consolidation_semantic`` for
BOTH the event and relationship domains (the equivalents of the A5C-A fact
pattern):

  * explicit ``EventSemanticPackingPolicy`` / ``RelationshipSemanticPackingPolicy``
    with TWO independent limits (``max_pairs_per_block`` AND
    ``max_candidates_per_block``); the audit candidates are P1 6/12, P2 12/24,
    P3 24/48;
  * no production default: the packing policy is REQUIRED;
  * each block carries ``candidate_refs`` (the stable unique union of its pair
    endpoints, ordered by candidate source_order_key then ref);
  * block identity binds the frozen material (plan hash, packing limits, block
    ordinal, ordered pairs, ordered candidate refs), using the event
    ``a5eblk_`` / relationship ``a5rblk_`` prefixes (distinct from the fact
    ``a5fblk_``);
  * the semantic stream is validated fail closed before packing: the exact
    domain namespace, ``left_ref < right_ref``, no duplicate pair, canonical
    ``(left_ref, right_ref)`` order, exact ``needs_semantic_decision`` coverage,
    auto_same excluded;
  * pair-local endpoint packets derived from ``IndexedEventCandidate`` /
    ``IndexedRelationshipCandidate`` carry the exact ``evidence_strength`` and
    ``source_order_key``, with pair-local evidence selectors ``L0/L1/...`` /
    ``R0/R1/...`` in the EXACT indexed A5B evidence order (no re-sort) and NO
    block-wide evidence pool;
  * fail-closed verification of the consolidation profile identity INCLUDING
    ``max_generation_rounds == 2`` and the exact event / relationship prompt +
    schema identity;
  * real ``StructuredGenerationRequest`` objects rendered through the existing
    ``PromptRegistry`` / ``OutputSchema`` / ``SemanticLLMProfile``
    infrastructure (zero provider, provider-neutral request material).

PLUS a golden regression proving the A5C fact block ids + request hashes remain
UNCHANGED after the shared two-limit packing refactor (the fact path now
delegates to ``_build_semantic_blocks`` / ``_greedy_pack``).

All fixtures are self-contained synthetic run trees (no dependency on the local
Alice run tree; the Alice gate lives in the audit script). No provider is
called and nothing is persisted anywhere in this file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

from short_drama.artifacts import ArtifactRef
from short_drama.llm import PromptRegistry
from short_drama.llm.models import StructuredGenerationRequest
from short_drama.story import (
    A5C_BLOCK_PREFIX,
    A5D_EVENT_BLOCK_PREFIX,
    A5D_EVENT_ENDPOINT_PACKET_FIELDS,
    A5D_EVENT_PACKING_CANDIDATES,
    A5D_EVENT_OUTPUT_SCHEMA_ID,
    A5D_EVENT_OUTPUT_SCHEMA_VERSION,
    A5D_EVENT_PROMPT_ID,
    A5D_EVENT_PROMPT_VERSION,
    A5D_MAX_GENERATION_ROUNDS,
    A5D_RELATIONSHIP_BLOCK_PREFIX,
    A5D_RELATIONSHIP_ENDPOINT_PACKET_FIELDS,
    A5D_RELATIONSHIP_PACKING_CANDIDATES,
    A5D_RELATIONSHIP_OUTPUT_SCHEMA_ID,
    A5D_RELATIONSHIP_OUTPUT_SCHEMA_VERSION,
    A5D_RELATIONSHIP_PROMPT_ID,
    A5D_RELATIONSHIP_PROMPT_VERSION,
    A5D_SEMANTIC_PROFILE_ID,
    DEFAULT_PROMPT_BASE_DIR,
    EventPairPlan,
    EventSemanticPackingPolicy,
    EventSemanticPreparation,
    RelationshipPairPlan,
    RelationshipSemanticPackingPolicy,
    RelationshipSemanticPreparation,
    StoryIntegrityError,
    build_consolidation_planning,
    build_event_endpoint_packet,
    build_event_pair_context,
    build_event_packing_audit,
    build_event_semantic_identity,
    build_event_semantic_preparation,
    build_fact_semantic_preparation,
    build_relationship_endpoint_packet,
    build_relationship_pair_context,
    build_relationship_packing_audit,
    build_relationship_semantic_identity,
    build_relationship_semantic_preparation,
    load_event_semantic_profile,
    load_relationship_semantic_profile,
)
from short_drama.story.consolidation import (
    ConsolidationSemanticPass,
    IndexedEventCandidate,
    IndexedRelationshipCandidate,
)
from short_drama.story.extraction import (
    EvidenceRef,
    EventCandidate,
    RelationshipCandidate,
)
from test_story_a5b_audit import ChunkSpec, _build_multi_chunk_tree
from test_story_a5b_planning import (
    DOCUMENT,
    PROJECT,
    RECON_PROFILE_ID,
    _char,
    _consolidation_profile,
    _loc,
)
from test_story_a5c_fact_preparation import (
    _P1 as _FACT_P1,
    _P2 as _FACT_P2,
    _P3 as _FACT_P3,
    _PROMPTS as _FACT_PROMPTS,
    _SEM_PROFILE as _FACT_SEM_PROFILE,
    _corpus_specs as _fact_corpus_specs,
    _single_chunk_specs as _fact_single_chunk_specs,
    _planning as _fact_planning,
    _tree as _fact_tree,
)

_PROMPTS = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
# The event / relationship semantic passes share the SAME tracked
# consolidation-llm-v1 profile (loaded once; both loaders return it).
_EV_SEM_PROFILE = load_event_semantic_profile()
_REL_SEM_PROFILE = load_relationship_semantic_profile()
# Audit candidates (test convenience): P1 6/12, P2 12/24, P3 24/48.
_EV_P1, _EV_P2, _EV_P3 = A5D_EVENT_PACKING_CANDIDATES
_REL_P1, _REL_P2, _REL_P3 = A5D_RELATIONSHIP_PACKING_CANDIDATES


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
) -> EventCandidate:
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
) -> RelationshipCandidate:
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
                _mk_event("cand_evt_002", participants=("cand_char_001",), summary="leaves"),
                _mk_event("cand_evt_003", participants=("cand_char_001",),
                          locations=("cand_loc_001",), summary="stays"),
            ),
        ),
        ChunkSpec(
            "CH002",
            ("CH002_P0001",),
            chars=(_char("cand_char_002", "Bob"),),
            events=(
                _mk_event("cand_evt_004", participants=("cand_char_002",),
                          summary="runs", para="CH002_P0001"),
            ),
        ),
    ]


def _rel_specs():
    """Two-chunk corpus: 5 relationships, 4 semantic pairs, 0 auto_same.

    Group (char_001 -> char_002): cand_rel_001 / 002 / 003 (distinct types)
    -> 3 pairs. Group (char_002 -> char_003): cand_rel_004 / 005 -> 1 pair.
    """
    return [
        ChunkSpec(
            "CH001",
            ("CH001_P0001", "CH001_P0002"),
            chars=(
                _char("cand_char_001", "Alice"),
                _char("cand_char_002", "Bob"),
                _char("cand_char_003", "Carol"),
            ),
            rels=(
                _mk_rel("cand_rel_001", "cand_char_001", "cand_char_002", "meets"),
                _mk_rel("cand_rel_002", "cand_char_001", "cand_char_002", "helps"),
                _mk_rel("cand_rel_003", "cand_char_001", "cand_char_002", "trusts"),
            ),
        ),
        ChunkSpec(
            "CH002",
            ("CH002_P0001",),
            chars=(_char("cand_char_002", "Bob"), _char("cand_char_003", "Carol")),
            rels=(
                _mk_rel("cand_rel_004", "cand_char_002", "cand_char_003", "meets",
                        para="CH002_P0001"),
                _mk_rel("cand_rel_005", "cand_char_002", "cand_char_003", "fights",
                        para="CH002_P0001"),
            ),
        ),
    ]


def _event_single_specs():
    """A single event (one chunk, one candidate) -> no pairs."""
    return [
        ChunkSpec(
            "CH001",
            ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"),),
            events=(_mk_event("cand_evt_001", participants=("cand_char_001",)),),
        ),
    ]


def _rel_single_specs():
    """A single relationship (one chunk, one candidate) -> no pairs."""
    return [
        ChunkSpec(
            "CH001",
            ("CH001_P0001",),
            chars=(_char("cand_char_001", "Alice"), _char("cand_char_002", "Bob")),
            rels=(_mk_rel("cand_rel_001", "cand_char_001", "cand_char_002"),),
        ),
    ]


# ---------------------------------------------------------------------------
# Tree / planning / preparation helpers
# ---------------------------------------------------------------------------


def _tree(tmp_path: Path, specs):
    return _build_multi_chunk_tree(tmp_path, *specs)


def _planning(tree):
    return build_consolidation_planning(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        reconciliation_profile_id=RECON_PROFILE_ID,
        consolidation_profile=_consolidation_profile(),
    )


# ---------------------------------------------------------------------------
# Domain descriptors (shared structural tests are parametrized over these)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Domain:
    name: str
    builder: object
    prep_cls: type
    policy_cls: type
    candidates: tuple
    sem_profile: object
    block_prefix: str
    prompt_prefix: str
    endpoint_fields: tuple
    pair_plans_attr: str
    index_attr: str
    total_attr: str
    pass_name: str
    prompt_id: str
    prompt_version: int
    output_schema_id: str
    output_schema_version: int
    make_specs: object
    identity_builder: object
    pair_plan_cls: type
    indexed_candidate_cls: type
    endpoint_packet_builder: object


_EV = _Domain(
    name="event",
    builder=build_event_semantic_preparation,
    prep_cls=EventSemanticPreparation,
    policy_cls=EventSemanticPackingPolicy,
    candidates=A5D_EVENT_PACKING_CANDIDATES,
    sem_profile=_EV_SEM_PROFILE,
    block_prefix=A5D_EVENT_BLOCK_PREFIX,
    prompt_prefix="Event consolidation block: ",
    endpoint_fields=A5D_EVENT_ENDPOINT_PACKET_FIELDS,
    pair_plans_attr="event_pair_plans",
    index_attr="events",
    total_attr="total_event_pair_count",
    pass_name="event",
    prompt_id=A5D_EVENT_PROMPT_ID,
    prompt_version=A5D_EVENT_PROMPT_VERSION,
    output_schema_id=A5D_EVENT_OUTPUT_SCHEMA_ID,
    output_schema_version=A5D_EVENT_OUTPUT_SCHEMA_VERSION,
    make_specs=_event_specs,
    identity_builder=build_event_semantic_identity,
    pair_plan_cls=EventPairPlan,
    indexed_candidate_cls=IndexedEventCandidate,
    endpoint_packet_builder=build_event_endpoint_packet,
)

_REL = _Domain(
    name="relationship",
    builder=build_relationship_semantic_preparation,
    prep_cls=RelationshipSemanticPreparation,
    policy_cls=RelationshipSemanticPackingPolicy,
    candidates=A5D_RELATIONSHIP_PACKING_CANDIDATES,
    sem_profile=_REL_SEM_PROFILE,
    block_prefix=A5D_RELATIONSHIP_BLOCK_PREFIX,
    prompt_prefix="Relationship consolidation block: ",
    endpoint_fields=A5D_RELATIONSHIP_ENDPOINT_PACKET_FIELDS,
    pair_plans_attr="relationship_pair_plans",
    index_attr="relationships",
    total_attr="total_relationship_pair_count",
    pass_name="relationship",
    prompt_id=A5D_RELATIONSHIP_PROMPT_ID,
    prompt_version=A5D_RELATIONSHIP_PROMPT_VERSION,
    output_schema_id=A5D_RELATIONSHIP_OUTPUT_SCHEMA_ID,
    output_schema_version=A5D_RELATIONSHIP_OUTPUT_SCHEMA_VERSION,
    make_specs=_rel_specs,
    identity_builder=build_relationship_semantic_identity,
    pair_plan_cls=RelationshipPairPlan,
    indexed_candidate_cls=IndexedRelationshipCandidate,
    endpoint_packet_builder=build_relationship_endpoint_packet,
)

_DOMAINS = [_EV, _REL]


def _prep(ctx: _Domain, tree, policy):
    planning = _planning(tree)
    prep = ctx.builder(
        planning,
        _consolidation_profile(),
        ctx.sem_profile,
        prompts=_PROMPTS,
        packing_policy=policy,
    )
    return planning, prep


def _semantic_refs(planning, ctx: _Domain):
    return [
        (p.left_ref, p.right_ref)
        for p in getattr(planning, ctx.pair_plans_attr)
        if p.state == "needs_semantic_decision"
    ]


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


# ---------------------------------------------------------------------------
# BLOCK 1 -- explicit packing policy with TWO independent limits + audit candidates
# ---------------------------------------------------------------------------


def test_event_candidates_are_exactly_p1_p2_p3():
    assert [(p.name, p.max_pairs_per_block, p.max_candidates_per_block)
            for p in A5D_EVENT_PACKING_CANDIDATES] == [
        ("P1", 6, 12),
        ("P2", 12, 24),
        ("P3", 24, 48),
    ]
    for policy in A5D_EVENT_PACKING_CANDIDATES:
        assert policy.max_candidates_per_block == 2 * policy.max_pairs_per_block


def test_relationship_candidates_are_exactly_p1_p2_p3():
    assert [(p.name, p.max_pairs_per_block, p.max_candidates_per_block)
            for p in A5D_RELATIONSHIP_PACKING_CANDIDATES] == [
        ("P1", 6, 12),
        ("P2", 12, 24),
        ("P3", 24, 48),
    ]
    for policy in A5D_RELATIONSHIP_PACKING_CANDIDATES:
        assert policy.max_candidates_per_block == 2 * policy.max_pairs_per_block


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
@pytest.mark.parametrize("pairs,cands", [(0, 3), (1, 0), (1, 1), (-1, 3), (3, 2.0)])
def test_packing_policy_requires_valid_limits(ctx, pairs, cands):
    with pytest.raises(StoryIntegrityError):
        ctx.policy_cls("T", pairs, cands)
    with pytest.raises(StoryIntegrityError):
        ctx.policy_cls("", 6, 12)


def test_packing_policy_to_dict_is_limits_only():
    assert EventSemanticPackingPolicy("T", 6, 12).to_dict() == {
        "max_pairs_per_block": 6,
        "max_candidates_per_block": 12,
    }
    assert RelationshipSemanticPackingPolicy("T", 12, 24).to_dict() == {
        "max_pairs_per_block": 12,
        "max_candidates_per_block": 24,
    }


# ---------------------------------------------------------------------------
# BLOCK 2 -- no production default: the packing policy is REQUIRED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_packing_policy_is_required(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    # No production default: the packing policy is REQUIRED (no keyword default).
    with pytest.raises(TypeError):
        ctx.builder(planning, _consolidation_profile(), ctx.sem_profile, prompts=_PROMPTS)
    # An explicit None is rejected fail-closed.
    with pytest.raises(StoryIntegrityError, match="packing_policy is required"):
        ctx.builder(
            planning, _consolidation_profile(), ctx.sem_profile,
            prompts=_PROMPTS, packing_policy=None,
        )


# ---------------------------------------------------------------------------
# BLOCK 3 -- blocks carry candidate_refs (stable union of pair endpoints)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_block_candidate_refs_are_stable_union(tmp_path, ctx):
    _, prep = _prep(ctx, _tree(tmp_path, ctx.make_specs()), _EV_P2 if ctx is _EV else _REL_P2)
    for block in prep.blocks:
        keyed = set()
        for pc in block.pair_contexts:
            for side in ("left", "right"):
                packet = pc[side]
                keyed.add((packet["source_order_key"], packet["candidate_ref"]))
        expected = tuple(ref for _so, ref in sorted(keyed))
        assert tuple(block.candidate_refs) == expected
        assert len(set(block.candidate_refs)) == len(block.candidate_refs)
        assert len(block.candidate_refs) <= prep.packing_policy.max_candidates_per_block
        endpoint_union = {ref for (l, r) in block.pair_refs for ref in (l, r)}
        assert set(block.candidate_refs) == endpoint_union


# ---------------------------------------------------------------------------
# BLOCK 4 -- block identity binds the frozen material + distinct prefixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_block_ids_binds_policy_and_are_distinct_per_block(tmp_path, ctx):
    policy = _EV_P2 if ctx is _EV else _REL_P2
    _, prep1 = _prep(ctx, _tree(tmp_path, ctx.make_specs()), policy)
    # Distinct blocks (distinct ordinal/pairs) get distinct ids.
    assert len({b.block_id for b in prep1.blocks}) == len(prep1.blocks)
    # Every block id is <domain prefix> + 20 hex.
    for block in prep1.blocks:
        assert block.block_id.startswith(ctx.block_prefix)
        hex_part = block.block_id[len(ctx.block_prefix):]
        assert len(hex_part) == 20 and all(c in "0123456789abcdef" for c in hex_part)
    # Changing the packing material changes the block ids.
    other_policy = _EV_P3 if ctx is _EV else _REL_P3
    _, prep2 = _prep(ctx, _tree(tmp_path, ctx.make_specs()), other_policy)
    if prep1.blocks and prep2.blocks and len(prep1.blocks) != len(prep2.blocks):
        assert [b.block_id for b in prep1.blocks] != [b.block_id for b in prep2.blocks]


def test_event_and_relationship_block_prefixes_are_distinct_from_fact():
    # The three domains use three distinct block-id prefixes.
    assert A5D_EVENT_BLOCK_PREFIX == "a5eblk_"
    assert A5D_RELATIONSHIP_BLOCK_PREFIX == "a5rblk_"
    assert A5C_BLOCK_PREFIX == "a5fblk_"
    assert len({A5D_EVENT_BLOCK_PREFIX, A5D_RELATIONSHIP_BLOCK_PREFIX, A5C_BLOCK_PREFIX}) == 3


def test_block_id_binds_domain_into_material():
    # A5D plan BLOCK 12: the event / relationship block-id hash material includes
    # the explicit domain (the fact block id passes no domain).
    from short_drama.story import consolidation_semantic

    pair_refs = (("CH001_C001:a", "CH001_C001:b"),)
    cand_refs = ("CH001_C001:a", "CH001_C001:b")
    base = dict(
        plan_hash="plan_hash_X",
        packing_policy_material={"max_pairs_per_block": 6, "max_candidates_per_block": 12},
        block_ordinal=0,
        pair_refs=pair_refs,
        candidate_refs=cand_refs,
    )
    ev = consolidation_semantic._compute_semantic_block_id(
        block_prefix=A5D_EVENT_BLOCK_PREFIX, domain="event", **base
    )
    ev_as_rel = consolidation_semantic._compute_semantic_block_id(
        block_prefix=A5D_EVENT_BLOCK_PREFIX, domain="relationship", **base
    )
    no_domain = consolidation_semantic._compute_semantic_block_id(
        block_prefix=A5D_EVENT_BLOCK_PREFIX, **base
    )
    # The domain is bound into the material: same prefix + different domain differ.
    assert ev != ev_as_rel
    assert ev != no_domain


# ---------------------------------------------------------------------------
# BLOCK 5 -- deterministic packing (semantic stream only, auto_same excluded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_packing_deterministic_ordered_and_bounded(tmp_path, ctx):
    planning, prep = _prep(ctx, _tree(tmp_path, ctx.make_specs()),
                           _EV_P2 if ctx is _EV else _REL_P2)
    refs_sorted = sorted(_semantic_refs(planning, ctx))
    block_pairs = [(l, r) for b in prep.blocks for (l, r) in b.pair_refs]
    assert block_pairs == refs_sorted
    assert all(b.pair_count <= 12 for b in prep.blocks)
    assert all(len(b.candidate_refs) <= 24 for b in prep.blocks)
    # Deterministic: a second preparation is byte-identical.
    _, prep2 = _prep(ctx, _tree(tmp_path, ctx.make_specs()),
                     _EV_P2 if ctx is _EV else _REL_P2)
    assert [b.block_id for b in prep.blocks] == [b.block_id for b in prep2.blocks]
    assert prep.semantic_request_hashes == prep2.semantic_request_hashes


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_auto_same_pairs_excluded_and_counts(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    all_pairs = getattr(planning, ctx.pair_plans_attr)
    auto_pairs = {
        frozenset((p.left_ref, p.right_ref)) for p in all_pairs
        if p.state == "auto_same"
    }
    # These synthetic corpora produce only semantic pairs (no auto_same).
    assert not auto_pairs
    prep = ctx.builder(
        planning, _consolidation_profile(), ctx.sem_profile, prompts=_PROMPTS,
        packing_policy=_EV_P3 if ctx is _EV else _REL_P3,
    )
    block_pairs = {frozenset((l, r)) for b in prep.blocks for (l, r) in b.pair_refs}
    assert not (block_pairs & auto_pairs)
    assert prep.auto_same_pair_count == len(auto_pairs)
    assert prep.semantic_pair_count + prep.auto_same_pair_count == getattr(
        prep, ctx.total_attr
    )


def test_exact_synthetic_counts():
    # Event: 4 candidates / 3 semantic pairs. Relationship: 5 / 4.
    tree = _tree(_tmp(), _event_specs())
    planning = _planning(tree)
    assert len(planning.index.events) == 4
    assert sum(1 for p in planning.event_pair_plans
               if p.state == "needs_semantic_decision") == 3
    tree2 = _tree(_tmp(), _rel_specs())
    planning2 = _planning(tree2)
    assert len(planning2.index.relationships) == 5
    assert sum(1 for p in planning2.relationship_pair_plans
               if p.state == "needs_semantic_decision") == 4


def _tmp():
    import tempfile

    return Path(tempfile.mkdtemp())


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_empty_semantic_stream_yields_no_blocks(tmp_path, ctx):
    specs = _event_single_specs() if ctx is _EV else _rel_single_specs()
    tree = _tree(tmp_path, specs)
    policy = _EV_P3 if ctx is _EV else _REL_P3
    planning, prep = _prep(ctx, tree, policy)
    assert planning.plan_hash
    assert getattr(prep, ctx.total_attr) == 0
    assert prep.semantic_pair_count == 0
    assert prep.auto_same_pair_count == 0
    assert prep.blocks == ()
    assert prep.structured_requests == ()
    assert prep.semantic_request_hashes == ()
    identity = ctx.identity_builder(prep)
    assert identity["block_count"] == 0


# ---------------------------------------------------------------------------
# BLOCK 6 -- pair-local endpoint packets (derived from the indexed candidate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_endpoint_packet_fields_and_selectors(tmp_path, ctx):
    planning, prep = _prep(ctx, _tree(tmp_path, ctx.make_specs()),
                           _EV_P2 if ctx is _EV else _REL_P2)
    by_ref = {c.global_candidate_ref: c for c in getattr(planning.index, ctx.index_attr)}
    for block in prep.blocks:
        for pc in block.pair_contexts:
            for side, prefix in (("left", "L"), ("right", "R")):
                packet = pc[side]
                cand = by_ref[packet["candidate_ref"]]
                for field in ctx.endpoint_fields:
                    assert field in packet, f"missing endpoint field {field!r}"
                assert packet["chunk_id"] == cand.chunk_id
                assert packet["local_candidate_id"] == cand.local_candidate_id
                # The exact evidence_strength + source_order_key are carried.
                assert packet["evidence_strength"] == cand.evidence_strength
                assert packet["source_order_key"] == cand.source_order_key
                # Pair-local selectors (L0.. / R0..) in the EXACT indexed order.
                selectors = [ev["selector"] for ev in packet["evidence"]]
                assert selectors == [f"{prefix}{idx}" for idx in range(len(packet["evidence"]))]
                assert [ev["paragraph_id"] for ev in packet["evidence"]] == [
                    e.paragraph_id for e in cand.evidence_refs
                ]


def test_event_endpoint_packet_domain_fields():
    cand = _indexed_event_candidate(
        [_ev("CH001_P0002"), _ev("CH001_P0001")]
    )
    packet = build_event_endpoint_packet(cand, "left")
    assert packet["summary_zh"] == cand.summary_zh
    assert packet["participants"] == list(cand.participants)
    assert packet["locations"] == list(cand.locations)
    assert packet["temporal_mode"] == cand.temporal_mode
    # The required A5D event endpoint fields are all present.
    for field in A5D_EVENT_ENDPOINT_PACKET_FIELDS:
        assert field in packet


def test_relationship_endpoint_packet_domain_fields_includes_none_state():
    cand = _indexed_relationship_candidate(
        [_ev("CH001_P0002"), _ev("CH001_P0001")], state_zh=None
    )
    packet = build_relationship_endpoint_packet(cand, "right")
    assert packet["source_entity_ref"] == cand.source_entity_ref
    assert packet["target_entity_ref"] == cand.target_entity_ref
    assert packet["relationship_type_zh"] == cand.relationship_type_zh
    # state_zh may be None (carried through exactly as indexed).
    assert packet["state_zh"] is None
    assert packet["direction"] == cand.direction
    for field in A5D_RELATIONSHIP_ENDPOINT_PACKET_FIELDS:
        assert field in packet


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_evidence_selectors_preserve_indexed_order_not_lexical(ctx):
    evidence = [_ev("CH001_P0002"), _ev("CH001_P0003"), _ev("CH001_P0001")]
    if ctx is _EV:
        cand = _indexed_event_candidate(evidence)
    else:
        cand = _indexed_relationship_candidate(evidence)
    left = ctx.endpoint_packet_builder(cand, "left")
    right = ctx.endpoint_packet_builder(cand, "right")
    # Non-lexical order is preserved (L0->P0002, L1->P0003, L2->P0001).
    assert [ev["paragraph_id"] for ev in left["evidence"]] == [
        "CH001_P0002", "CH001_P0003", "CH001_P0001",
    ]
    assert [ev["selector"] for ev in left["evidence"]] == ["L0", "L1", "L2"]
    assert [ev["selector"] for ev in right["evidence"]] == ["R0", "R1", "R2"]
    assert left["evidence_strength"] == cand.evidence_strength
    assert left["source_order_key"] == cand.source_order_key
    with pytest.raises(StoryIntegrityError, match="side"):
        ctx.endpoint_packet_builder(cand, "middle")


def _indexed_event_candidate(evidence) -> IndexedEventCandidate:
    return IndexedEventCandidate(
        global_candidate_ref="CH001_C001:cand_evt_001",
        chunk_id="CH001_C001",
        local_candidate_id="cand_evt_001",
        source_order_key="000001:000000001:01:000000001:CH001_C001:cand_evt_001",
        summary_zh="evt",
        participants=("char_0001",),
        locations=(),
        temporal_mode="normal",
        evidence_strength="explicit",
        evidence_refs=tuple(evidence),
        candidate_extraction_ref=ArtifactRef(
            artifact_type="candidate_extraction", artifact_id="x", revision=1,
            content_hash="f" * 64,
        ),
    )


def _indexed_relationship_candidate(evidence, state_zh=None) -> IndexedRelationshipCandidate:
    return IndexedRelationshipCandidate(
        global_candidate_ref="CH001_C001:cand_rel_001",
        chunk_id="CH001_C001",
        local_candidate_id="cand_rel_001",
        source_order_key="000001:000000001:01:000000001:CH001_C001:cand_rel_001",
        source_entity_ref="char_0001",
        target_entity_ref="char_0002",
        relationship_type_zh="meets",
        state_zh=state_zh,
        direction="directed",
        evidence_strength="explicit",
        evidence_refs=tuple(evidence),
        candidate_extraction_ref=ArtifactRef(
            artifact_type="candidate_extraction", artifact_id="x", revision=1,
            content_hash="f" * 64,
        ),
    )


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_pair_context_carries_signals(tmp_path, ctx):
    planning, prep = _prep(ctx, _tree(tmp_path, ctx.make_specs()),
                           _EV_P2 if ctx is _EV else _REL_P2)
    plans = getattr(planning, ctx.pair_plans_attr)
    plan_by_refs = {(p.left_ref, p.right_ref): p for p in plans}
    for block in prep.blocks:
        for pc in block.pair_contexts:
            plan = plan_by_refs[(pc["left_candidate_ref"], pc["right_candidate_ref"])]
            assert pc["signals"] == list(plan.signals)
            assert pc["left_candidate_ref"] == plan.left_ref
            assert pc["right_candidate_ref"] == plan.right_ref


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_pair_context_helper_missing_candidate_fails_closed(ctx):
    left = "CH001_C001:cand_evt_001" if ctx is _EV else "CH001_C001:cand_rel_001"
    right = "CH001_C001:cand_evt_002" if ctx is _EV else "CH001_C001:cand_rel_002"
    plan = ctx.pair_plan_cls(
        left_ref=left, right_ref=right, signals=("same_chunk",),
        state="needs_semantic_decision",
    )
    with pytest.raises(StoryIntegrityError, match="missing from the index"):
        if ctx is _EV:
            build_event_pair_context(plan, {})
        else:
            build_relationship_pair_context(plan, {})


# ---------------------------------------------------------------------------
# BLOCK 7 -- profile verification must include max_generation_rounds == 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_max_generation_rounds_must_be_two(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    bad_profile = dataclasses.replace(_consolidation_profile(), max_generation_rounds=3)
    with pytest.raises(StoryIntegrityError, match="max_generation_rounds"):
        ctx.builder(
            planning, bad_profile, ctx.sem_profile, prompts=_PROMPTS,
            packing_policy=_EV_P3 if ctx is _EV else _REL_P3,
        )
    assert A5D_MAX_GENERATION_ROUNDS == 2


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_wrong_prompt_fails_closed(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    profile = _consolidation_profile()
    pass_obj = getattr(profile, ctx.pass_name)
    bad = dataclasses.replace(
        profile, **{ctx.pass_name: ConsolidationSemanticPass(
            **{**pass_obj.to_dict(), "prompt_id": "some-other-prompt"})}
    )
    with pytest.raises(StoryIntegrityError, match="prompt"):
        ctx.builder(
            planning, bad, ctx.sem_profile, prompts=_PROMPTS,
            packing_policy=_EV_P3 if ctx is _EV else _REL_P3,
        )


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_wrong_schema_fails_closed(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    profile = _consolidation_profile()
    pass_obj = getattr(profile, ctx.pass_name)
    bad = dataclasses.replace(
        profile, **{ctx.pass_name: ConsolidationSemanticPass(
            **{**pass_obj.to_dict(), "output_schema_id": "some-other-schema"})}
    )
    with pytest.raises(StoryIntegrityError, match="output schema"):
        ctx.builder(
            planning, bad, ctx.sem_profile, prompts=_PROMPTS,
            packing_policy=_EV_P3 if ctx is _EV else _REL_P3,
        )


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_wrong_semantic_profile_id_fails_closed(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    bad_sem = dataclasses.replace(ctx.sem_profile, profile_id="some-other-llm")
    with pytest.raises(StoryIntegrityError, match="semantic profile id"):
        ctx.builder(
            planning, _consolidation_profile(), bad_sem, prompts=_PROMPTS,
            packing_policy=_EV_P3 if ctx is _EV else _REL_P3,
        )


# ---------------------------------------------------------------------------
# BLOCK 8 -- semantic stream validation (fail closed before packing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_semantic_stream_duplicate_pair_fails_closed(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    plans = getattr(planning, ctx.pair_plans_attr)
    p0 = plans[0]
    bad = dataclasses.replace(planning, **{ctx.pair_plans_attr: (p0, p0)})
    with pytest.raises(StoryIntegrityError, match="duplicate"):
        ctx.builder(
            bad, _consolidation_profile(), ctx.sem_profile, prompts=_PROMPTS,
            packing_policy=_EV_P3 if ctx is _EV else _REL_P3,
        )


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_semantic_stream_malformed_order_fails_closed(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    plans = list(getattr(planning, ctx.pair_plans_attr))
    p0, p1 = plans[0], plans[1]
    assert (p0.left_ref, p0.right_ref) < (p1.left_ref, p1.right_ref)
    bad = dataclasses.replace(planning, **{ctx.pair_plans_attr: (p1, p0)})
    with pytest.raises(StoryIntegrityError, match="canonical"):
        ctx.builder(
            bad, _consolidation_profile(), ctx.sem_profile, prompts=_PROMPTS,
            packing_policy=_EV_P3 if ctx is _EV else _REL_P3,
        )


# ---------------------------------------------------------------------------
# Identity + preparation invariants (BLOCK 4 / 11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_identity_is_frozen_and_verified(tmp_path, ctx):
    planning, prep = _prep(ctx, _tree(tmp_path, ctx.make_specs()),
                           _EV_P2 if ctx is _EV else _REL_P2)
    identity = ctx.identity_builder(prep)
    assert identity["profile_id"] == "consolidation-v1"
    assert identity["semantic_profile_id"] == A5D_SEMANTIC_PROFILE_ID
    assert identity["prompt_id"] == ctx.prompt_id
    assert identity["prompt_version"] == ctx.prompt_version
    assert identity["output_schema_id"] == ctx.output_schema_id
    assert identity["output_schema_version"] == ctx.output_schema_version
    assert identity["plan_hash"] == planning.plan_hash
    assert identity["block_count"] == len(prep.blocks)
    assert identity["semantic_request_hashes"] == tuple(prep.semantic_request_hashes)
    assert identity["packing_policy"] == {
        "max_pairs_per_block": prep.packing_policy.max_pairs_per_block,
        "max_candidates_per_block": prep.packing_policy.max_candidates_per_block,
    }
    assert identity[ctx.total_attr] == getattr(prep, ctx.total_attr)
    assert "max_block_size" not in identity


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_preparation_result_and_invariants(tmp_path, ctx):
    planning, prep = _prep(ctx, _tree(tmp_path, ctx.make_specs()),
                           _EV_P2 if ctx is _EV else _REL_P2)
    assert isinstance(prep, ctx.prep_cls)
    assert len(prep.blocks) == len(prep.structured_requests) == len(prep.semantic_request_hashes)
    assert prep.semantic_request_hashes == tuple(
        r.request_hash for r in prep.structured_requests
    )
    assert sum(b.pair_count for b in prep.blocks) == prep.semantic_pair_count
    assert [b.block_ordinal for b in prep.blocks] == list(range(len(prep.blocks)))
    # Asset identity fields are the frozen A5D identity.
    assert prep.prompt_id == ctx.prompt_id
    assert prep.prompt_version == ctx.prompt_version
    assert prep.output_schema_id == ctx.output_schema_id
    assert prep.output_schema_version == ctx.output_schema_version
    assert prep.semantic_profile.profile_id == A5D_SEMANTIC_PROFILE_ID
    assert prep.working_language == "zh"
    # Every block carries only the semantic (needs_semantic_decision) pairs.
    planned_semantic = set(_semantic_refs(planning, ctx))
    block_pairs = {(l, r) for b in prep.blocks for (l, r) in b.pair_refs}
    assert block_pairs == planned_semantic


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_preparation_alignment_invariants_fail_closed(tmp_path, ctx):
    _, prep = _prep(ctx, _tree(tmp_path, ctx.make_specs()),
                    _EV_P2 if ctx is _EV else _REL_P2)
    with pytest.raises(StoryIntegrityError, match="semantic request hash"):
        dataclasses.replace(prep, semantic_request_hashes=prep.semantic_request_hashes[:-1])
    bad_hashes = prep.semantic_request_hashes[:-1] + ("0" * 64,)
    with pytest.raises(StoryIntegrityError, match="semantic_request_hashes"):
        dataclasses.replace(prep, semantic_request_hashes=bad_hashes)


# ---------------------------------------------------------------------------
# Real StructuredGenerationRequest rendering (zero provider, provider-neutral)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_real_structured_requests_rendered(tmp_path, ctx):
    _, prep = _prep(ctx, _tree(tmp_path, ctx.make_specs()),
                    _EV_P2 if ctx is _EV else _REL_P2)
    assert all(isinstance(r, StructuredGenerationRequest) for r in prep.structured_requests)
    for block, request in zip(prep.blocks, prep.structured_requests):
        assert request.rendered_prompt.user_text.startswith(
            f"{ctx.prompt_prefix}{block.block_id}"
        )
        assert block.pair_contexts_json in request.rendered_prompt.user_text
        assert request.rendered_prompt.rendered_prompt_hash
        _assert_no_forbidden_keys(request.semantic_request_material())
        assert request.semantic_profile.profile_id == A5D_SEMANTIC_PROFILE_ID
        assert request.output_schema.schema_id == ctx.output_schema_id


@pytest.mark.parametrize("ctx", _DOMAINS, ids=lambda c: c.name)
def test_packing_audit_reports_deterministic_distribution(tmp_path, ctx):
    tree = _tree(tmp_path, ctx.make_specs())
    planning = _planning(tree)
    audit = (
        build_event_packing_audit(planning, _consolidation_profile(), ctx.sem_profile,
                                  prompts=_PROMPTS)
        if ctx is _EV
        else build_relationship_packing_audit(
            planning, _consolidation_profile(), ctx.sem_profile, prompts=_PROMPTS)
    )
    assert [row["packing_name"] for row in audit] == ["P1", "P2", "P3"]
    for row in audit:
        assert row["total_pairs_in_blocks"] == sum(
            1 for _ in _semantic_refs(planning, ctx)
        )
        assert row["request_hash_count"] == row["block_count"]
        assert row["unique_request_hash_count"] == row["block_count"]
        assert row["pairs_per_block"]["max"] <= row["max_pairs_per_block"]
        assert row["unique_candidates_per_block"]["max"] <= row["max_candidates_per_block"]


# ---------------------------------------------------------------------------
# Cross-domain: the correct domain stream is selected
# ---------------------------------------------------------------------------


def test_event_prep_selects_event_stream_relationship_prep_selects_relationship(tmp_path):
    # An event preparation only consumes the event pair plans (and vice versa):
    # a corpus with events but no relationships still yields event blocks, and
    # a relationship corpus with no events still yields relationship blocks.
    ev_tree = _tree(tmp_path / "ev", _event_specs())
    ev_planning = _planning(ev_tree)
    ev_prep = build_event_semantic_preparation(
        ev_planning, _consolidation_profile(), _EV_SEM_PROFILE,
        prompts=_PROMPTS, packing_policy=_EV_P2,
    )
    assert ev_prep.semantic_pair_count == 3
    assert ev_prep.total_event_pair_count == 3
    # The event tree has no relationships at all -> the relationship prep is empty.
    rel_prep = build_relationship_semantic_preparation(
        ev_planning, _consolidation_profile(), _REL_SEM_PROFILE,
        prompts=_PROMPTS, packing_policy=_REL_P2,
    )
    assert rel_prep.semantic_pair_count == 0
    assert rel_prep.blocks == ()

    rel_tree = _tree(tmp_path / "rel", _rel_specs())
    rel_planning = _planning(rel_tree)
    rel_prep2 = build_relationship_semantic_preparation(
        rel_planning, _consolidation_profile(), _REL_SEM_PROFILE,
        prompts=_PROMPTS, packing_policy=_REL_P2,
    )
    assert rel_prep2.semantic_pair_count == 4
    assert rel_prep2.total_relationship_pair_count == 4
    # The relationship tree has no events at all -> the event prep is empty.
    ev_prep2 = build_event_semantic_preparation(
        rel_planning, _consolidation_profile(), _EV_SEM_PROFILE,
        prompts=_PROMPTS, packing_policy=_EV_P2,
    )
    assert ev_prep2.semantic_pair_count == 0
    assert ev_prep2.blocks == ()


# ---------------------------------------------------------------------------
# Golden regression: the A5C fact block ids + request hashes are UNCHANGED
# after the shared two-limit packing refactor.
# ---------------------------------------------------------------------------


# --- Absolute A5C fact block ids + request hashes captured before the A5D-A
# shared-helper refactor (the fact path now delegates to _build_semantic_blocks
# / _greedy_pack). These pins prove the fact packing is byte-identical. ---

_CORPUS_P2_BLOCK_IDS = ("a5fblk_bd7713ec10ea5f8fe910",)
_CORPUS_P2_REQUEST_HASHES = ("1632f76d0a026aca7dae44f77528f060ac70b451b74eb5e5858f3d5e90203251",)

_SINGLE30_P2_BLOCK_IDS = (
    "a5fblk_16a80bf4e97c27c5f07e", "a5fblk_411fabca59a8457e4b86",
    "a5fblk_b72dac2f6412368a940d", "a5fblk_f0dddc34aa7a7d0cfe2f",
    "a5fblk_eca0cfd9b3dcad0fb7f0", "a5fblk_29faf9265bc10c356d81",
    "a5fblk_b667392ca4104c0f575c", "a5fblk_fad9344de8dead2cd606",
    "a5fblk_4ec5dbc2916d1fac0bb2", "a5fblk_a3e05b908ca42459af90",
    "a5fblk_925350e0158263bb9e33", "a5fblk_73b69cd260639d9ea167",
    "a5fblk_a8973d487dfff1bf4c1c", "a5fblk_446130cfb2d34199e657",
    "a5fblk_a50b76af86bd0ea3682f", "a5fblk_607fa9428752fa956033",
    "a5fblk_8dadb4c0315818711ae1", "a5fblk_06f0105540168b81f5d5",
    "a5fblk_8ea4d8de65e3addc4b9b", "a5fblk_7c2a99d3966c141f6386",
    "a5fblk_64d9749b935dbc5fafc0", "a5fblk_c22fbc74b34365655305",
    "a5fblk_f688233c9c3d2be6094c", "a5fblk_17afc0b38b698e3cf3d7",
    "a5fblk_18ef23081e8169ec6493", "a5fblk_57060f5bd6ccdb543c9e",
    "a5fblk_628e29846026aa76624b", "a5fblk_d457a6f31c66a2d1a8ab",
    "a5fblk_d529a38f748411148dc8", "a5fblk_29c4de938b3c5695a9af",
    "a5fblk_259d8aeb11c0f3e6b7e6", "a5fblk_4bd4155bfcfd09689916",
    "a5fblk_6abaad664a1a61923822", "a5fblk_68b79704b1f3de6d0d54",
    "a5fblk_5a031e868235fe44296e", "a5fblk_994ff3393fb5d504cc7e",
    "a5fblk_098f81e9538af65633e1",
)
_SINGLE30_P2_REQUEST_HASH_FIRST = "6f15bba95905799f14a76f6acb7bed2e7c31ad9a12e761a41518b35e27e4cf5b"
_SINGLE30_P2_REQUEST_HASH_LAST = "b7bf1c17ae34bd327c690ba3e5a2b26f6a1b8c0cbfdad9690df5a5a6c35a1fc0"

_SINGLE12_P1_BLOCK_IDS = (
    "a5fblk_ff69bb9d652c6b2391ed", "a5fblk_be2e0214a4e60ae47e39",
    "a5fblk_35417e89c712303415c6", "a5fblk_eba4f55813ffa236aacc",
    "a5fblk_97dabe84561034005860", "a5fblk_8bffe9fbef192e710cdf",
    "a5fblk_5bcb23aa9a65729e34f9", "a5fblk_9e2f9763c0f7892177bc",
    "a5fblk_02e0f02048e7de09ba8e", "a5fblk_0f0ee75227e774468f1a",
    "a5fblk_bc24e2d31b5e472e01d0",
)
_SINGLE12_P1_REQUEST_HASH_FIRST = "8714d7c6c030bb6788e06945694e3a6f2a9acb4e0b739f969e1c5c1b269ebe08"
_SINGLE12_P1_REQUEST_HASH_LAST = "9b250c68802c74428ec48bbed154a32d304c11b9f6db2c4b34524f0e71f0813f"

_SINGLE48_P2_BLOCK_IDS = (
    "a5fblk_2d8267117c69a13032b4", "a5fblk_c19f6546f808a1ba087f",
    "a5fblk_35b15c2e7c06fac4ad71", "a5fblk_f5b64291950b28100669",
    "a5fblk_81046c345008aa2e94ee", "a5fblk_1c74f88ab8e4a12417bb",
    "a5fblk_d471e288e5b56eb0ec8d", "a5fblk_4922059e38fa4544eb4c",
    "a5fblk_1504705fa06b6f826bcd", "a5fblk_8e047a03db7eab202c95",
    "a5fblk_eaca30db8e5ab0c7e6ad", "a5fblk_a98e175112d2ff35fab4",
    "a5fblk_098e4506a2b8f16103c8", "a5fblk_0d09196c7ea0009906ab",
    "a5fblk_60adee1226a434a1735d", "a5fblk_65768d98402021dae245",
    "a5fblk_a290f55f5aceb373b349", "a5fblk_b08d48bd553f43b93efb",
    "a5fblk_d45fbe60e2a18fbd415e", "a5fblk_2c50e46abaa77e0e11c7",
    "a5fblk_610e76aac8820cd41ba9", "a5fblk_12203dc8a69b6f15a8dc",
    "a5fblk_3c0f347cd9e27e36afe7", "a5fblk_bb09a1895525f4523a36",
    "a5fblk_538f65892616cee94593", "a5fblk_c3fe2e424bae63d89b1d",
    "a5fblk_65d7a788a0f549dec49b", "a5fblk_d9f949216451829967c9",
    "a5fblk_59d5f9bd2aa7454c6386", "a5fblk_a91d53a87a253fc91ad8",
    "a5fblk_a231448de8bca047843d", "a5fblk_b897535aa7c3c24f234b",
    "a5fblk_1db5bb4b93cfa47eb489", "a5fblk_2e129eb375a0250b2715",
    "a5fblk_b4af1e4168351773ae6f", "a5fblk_68fb516141273328857e",
    "a5fblk_8d9ac75730059a62b524", "a5fblk_05fdf1a82a7310c98c3a",
    "a5fblk_6d7b679ec4bfdc6cc12d", "a5fblk_3759f59900d21b21c292",
    "a5fblk_f2712ad6ccae1a568fcd", "a5fblk_05f1f789b9188af5d744",
    "a5fblk_8bdabdb942a4a67f45d1", "a5fblk_f0a2900df43da9d22f8c",
    "a5fblk_dc288b25574f6b342875", "a5fblk_d87b32559e57f1c49143",
    "a5fblk_bef0820f6c9d77c6ebf3", "a5fblk_2810baa07d3c89f7630e",
    "a5fblk_ae9d9c60d44c08854e2b", "a5fblk_33aa7c1c0391c287ec7f",
    "a5fblk_aed981afaf3744bb4bf4", "a5fblk_3e89972dad36f9ed871b",
    "a5fblk_b05e28be90cd3fbfdef1", "a5fblk_c6417e46573d55d1239b",
    "a5fblk_e91c2e6c97a636ef2ece", "a5fblk_d7fbce3a4f4f913c1b4e",
    "a5fblk_81c6b91f3bf7d32ffd6e", "a5fblk_a1ae0e23805c66e525cd",
    "a5fblk_0a0590fe3cd481d4e1b4", "a5fblk_c356f4573208efe78d66",
    "a5fblk_d502e86fd516acf70e41", "a5fblk_9fa3a3579a2bfae647ba",
    "a5fblk_3e97839c051bae158e88", "a5fblk_35c5a62a403333bbea7b",
    "a5fblk_e84df861ce1a97a060dd", "a5fblk_1e7c97e522847fca13e2",
    "a5fblk_bcf39a4a05891a5d621e", "a5fblk_9fc71bfebe6271cb1e3c",
    "a5fblk_7939e9650139ed8da0ae", "a5fblk_20f805a726c16c4b5b41",
    "a5fblk_09e9aebc41683d6b426a", "a5fblk_3a2b9b5306d1f69e85a1",
    "a5fblk_6a41e338e25f263f8ec1", "a5fblk_4d91aad9a17e1defeb56",
    "a5fblk_f605e40034546dbc032d", "a5fblk_6ff32c1e3e349b8b3667",
    "a5fblk_ccbf52a43e026f2d8fda", "a5fblk_b5e66895bfbca45443dd",
    "a5fblk_c309d283d9e5846d684d", "a5fblk_b14adaaa62671e121603",
    "a5fblk_c74c6cbe60fcc516dbb6", "a5fblk_6e3492ea5ecc4383c118",
    "a5fblk_c90a0fd6de09fa7531e6", "a5fblk_c09c9ce32c33d40c3685",
    "a5fblk_0bcdf99c141dd931450e", "a5fblk_91be66586e27c553a0f3",
    "a5fblk_5dd43067ba98917d2ddc", "a5fblk_6bf22e395981637afdbf",
    "a5fblk_6c27e5652c715c1685fa", "a5fblk_0634de928eb940bf4767",
    "a5fblk_5c6b9fa7e4896b5e45e1", "a5fblk_724c9335fed44f7751e8",
    "a5fblk_1fa33955f2b4b61b0f63", "a5fblk_e58d2d0525160647ef0b",
)
_SINGLE48_P2_REQUEST_HASH_FIRST = "97641e2f054d758bc4bc36122ed82e8ecdfe864aadf29f69987c7ea2fa1a093c"
_SINGLE48_P2_REQUEST_HASH_LAST = "db57f893bc249a566a1e2348b0e5d32748db6b76febc996d5107a50b35989515"


def _fact_prep(specs, policy):
    tree = _fact_tree(_tmp(), specs=specs)
    planning = _fact_planning(tree)
    return build_fact_semantic_preparation(
        planning, _consolidation_profile(), _FACT_SEM_PROFILE,
        prompts=_FACT_PROMPTS, packing_policy=policy,
    )


def test_golden_fact_corpus_p2_unchanged():
    prep = _fact_prep(_fact_corpus_specs(), _FACT_P2)
    assert [b.block_id for b in prep.blocks] == list(_CORPUS_P2_BLOCK_IDS)
    assert list(prep.semantic_request_hashes) == list(_CORPUS_P2_REQUEST_HASHES)


def test_golden_fact_single30_p2_unchanged():
    prep = _fact_prep(_fact_single_chunk_specs(30), _FACT_P2)
    assert [b.block_id for b in prep.blocks] == list(_SINGLE30_P2_BLOCK_IDS)
    hashes = list(prep.semantic_request_hashes)
    assert len(hashes) == len(_SINGLE30_P2_BLOCK_IDS)
    assert hashes[0] == _SINGLE30_P2_REQUEST_HASH_FIRST
    assert hashes[-1] == _SINGLE30_P2_REQUEST_HASH_LAST


def test_golden_fact_single12_p1_unchanged():
    prep = _fact_prep(_fact_single_chunk_specs(12), _FACT_P1)
    assert [b.block_id for b in prep.blocks] == list(_SINGLE12_P1_BLOCK_IDS)
    hashes = list(prep.semantic_request_hashes)
    assert len(hashes) == len(_SINGLE12_P1_BLOCK_IDS)
    assert hashes[0] == _SINGLE12_P1_REQUEST_HASH_FIRST
    assert hashes[-1] == _SINGLE12_P1_REQUEST_HASH_LAST


def test_golden_fact_single48_p2_unchanged():
    prep = _fact_prep(_fact_single_chunk_specs(48), _FACT_P2)
    assert [b.block_id for b in prep.blocks] == list(_SINGLE48_P2_BLOCK_IDS)
    hashes = list(prep.semantic_request_hashes)
    assert len(hashes) == len(_SINGLE48_P2_BLOCK_IDS)
    assert hashes[0] == _SINGLE48_P2_REQUEST_HASH_FIRST
    assert hashes[-1] == _SINGLE48_P2_REQUEST_HASH_LAST


def test_shared_packing_helper_used_by_fact():
    """The fact path now delegates to the shared two-limit packing core."""
    import inspect

    from short_drama.story import consolidation_semantic

    src = inspect.getsource(consolidation_semantic)
    # The shared core exists ...
    assert "def _build_semantic_blocks(" in src
    assert "def _greedy_pack(" in src
    # ... and the fact builder delegates to it.
    fact_src = inspect.getsource(consolidation_semantic._build_fact_blocks)
    assert "_build_semantic_blocks(" in fact_src


def test_preparation_module_does_not_import_transport():
    # The preparation path never references the LLM transport/adapter layer.
    import inspect

    from short_drama.story import consolidation_semantic

    src = inspect.getsource(consolidation_semantic)
    assert "adapter" not in src
    assert "transport" not in src
    assert "http" not in src.lower().replace("https", "")
