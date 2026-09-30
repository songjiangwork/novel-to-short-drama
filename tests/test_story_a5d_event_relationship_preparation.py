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

PLUS v1-to-v2 golden regressions pinning the exact current Fact block and
request identities after Issue #82 intentionally changes global plan identity.

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
# Historical v1 ids are retained as regression witnesses.  Issue #82 changes
# the authoritative A5B plan identity, so every downstream block/request id
# must change even though Fact packing and its prompt contract do not.
# ---------------------------------------------------------------------------


# --- Historical v1 pins are retained for the intentional identity transition. ---

_CORPUS_P2_V1_BLOCK_IDS = (
    'a5fblk_bd7713ec10ea5f8fe910',
)

_CORPUS_P2_V1_REQUEST_HASHES = (
    '1632f76d0a026aca7dae44f77528f060ac70b451b74eb5e5858f3d5e90203251',
)

_SINGLE30_P2_V1_BLOCK_IDS = (
    'a5fblk_16a80bf4e97c27c5f07e',
    'a5fblk_411fabca59a8457e4b86',
    'a5fblk_b72dac2f6412368a940d',
    'a5fblk_f0dddc34aa7a7d0cfe2f',
    'a5fblk_eca0cfd9b3dcad0fb7f0',
    'a5fblk_29faf9265bc10c356d81',
    'a5fblk_b667392ca4104c0f575c',
    'a5fblk_fad9344de8dead2cd606',
    'a5fblk_4ec5dbc2916d1fac0bb2',
    'a5fblk_a3e05b908ca42459af90',
    'a5fblk_925350e0158263bb9e33',
    'a5fblk_73b69cd260639d9ea167',
    'a5fblk_a8973d487dfff1bf4c1c',
    'a5fblk_446130cfb2d34199e657',
    'a5fblk_a50b76af86bd0ea3682f',
    'a5fblk_607fa9428752fa956033',
    'a5fblk_8dadb4c0315818711ae1',
    'a5fblk_06f0105540168b81f5d5',
    'a5fblk_8ea4d8de65e3addc4b9b',
    'a5fblk_7c2a99d3966c141f6386',
    'a5fblk_64d9749b935dbc5fafc0',
    'a5fblk_c22fbc74b34365655305',
    'a5fblk_f688233c9c3d2be6094c',
    'a5fblk_17afc0b38b698e3cf3d7',
    'a5fblk_18ef23081e8169ec6493',
    'a5fblk_57060f5bd6ccdb543c9e',
    'a5fblk_628e29846026aa76624b',
    'a5fblk_d457a6f31c66a2d1a8ab',
    'a5fblk_d529a38f748411148dc8',
    'a5fblk_29c4de938b3c5695a9af',
    'a5fblk_259d8aeb11c0f3e6b7e6',
    'a5fblk_4bd4155bfcfd09689916',
    'a5fblk_6abaad664a1a61923822',
    'a5fblk_68b79704b1f3de6d0d54',
    'a5fblk_5a031e868235fe44296e',
    'a5fblk_994ff3393fb5d504cc7e',
    'a5fblk_098f81e9538af65633e1',
)

_SINGLE30_P2_V1_REQUEST_HASH_FIRST = '6f15bba95905799f14a76f6acb7bed2e7c31ad9a12e761a41518b35e27e4cf5b'

_SINGLE30_P2_V1_REQUEST_HASH_LAST = 'b7bf1c17ae34bd327c690ba3e5a2b26f6a1b8c0cbfdad9690df5a5a6c35a1fc0'

_SINGLE12_P1_V1_BLOCK_IDS = (
    'a5fblk_ff69bb9d652c6b2391ed',
    'a5fblk_be2e0214a4e60ae47e39',
    'a5fblk_35417e89c712303415c6',
    'a5fblk_eba4f55813ffa236aacc',
    'a5fblk_97dabe84561034005860',
    'a5fblk_8bffe9fbef192e710cdf',
    'a5fblk_5bcb23aa9a65729e34f9',
    'a5fblk_9e2f9763c0f7892177bc',
    'a5fblk_02e0f02048e7de09ba8e',
    'a5fblk_0f0ee75227e774468f1a',
    'a5fblk_bc24e2d31b5e472e01d0',
)

_SINGLE12_P1_V1_REQUEST_HASH_FIRST = '8714d7c6c030bb6788e06945694e3a6f2a9acb4e0b739f969e1c5c1b269ebe08'

_SINGLE12_P1_V1_REQUEST_HASH_LAST = '9b250c68802c74428ec48bbed154a32d304c11b9f6db2c4b34524f0e71f0813f'

_SINGLE48_P2_V1_BLOCK_IDS = (
    'a5fblk_2d8267117c69a13032b4',
    'a5fblk_c19f6546f808a1ba087f',
    'a5fblk_35b15c2e7c06fac4ad71',
    'a5fblk_f5b64291950b28100669',
    'a5fblk_81046c345008aa2e94ee',
    'a5fblk_1c74f88ab8e4a12417bb',
    'a5fblk_d471e288e5b56eb0ec8d',
    'a5fblk_4922059e38fa4544eb4c',
    'a5fblk_1504705fa06b6f826bcd',
    'a5fblk_8e047a03db7eab202c95',
    'a5fblk_eaca30db8e5ab0c7e6ad',
    'a5fblk_a98e175112d2ff35fab4',
    'a5fblk_098e4506a2b8f16103c8',
    'a5fblk_0d09196c7ea0009906ab',
    'a5fblk_60adee1226a434a1735d',
    'a5fblk_65768d98402021dae245',
    'a5fblk_a290f55f5aceb373b349',
    'a5fblk_b08d48bd553f43b93efb',
    'a5fblk_d45fbe60e2a18fbd415e',
    'a5fblk_2c50e46abaa77e0e11c7',
    'a5fblk_610e76aac8820cd41ba9',
    'a5fblk_12203dc8a69b6f15a8dc',
    'a5fblk_3c0f347cd9e27e36afe7',
    'a5fblk_bb09a1895525f4523a36',
    'a5fblk_538f65892616cee94593',
    'a5fblk_c3fe2e424bae63d89b1d',
    'a5fblk_65d7a788a0f549dec49b',
    'a5fblk_d9f949216451829967c9',
    'a5fblk_59d5f9bd2aa7454c6386',
    'a5fblk_a91d53a87a253fc91ad8',
    'a5fblk_a231448de8bca047843d',
    'a5fblk_b897535aa7c3c24f234b',
    'a5fblk_1db5bb4b93cfa47eb489',
    'a5fblk_2e129eb375a0250b2715',
    'a5fblk_b4af1e4168351773ae6f',
    'a5fblk_68fb516141273328857e',
    'a5fblk_8d9ac75730059a62b524',
    'a5fblk_05fdf1a82a7310c98c3a',
    'a5fblk_6d7b679ec4bfdc6cc12d',
    'a5fblk_3759f59900d21b21c292',
    'a5fblk_f2712ad6ccae1a568fcd',
    'a5fblk_05f1f789b9188af5d744',
    'a5fblk_8bdabdb942a4a67f45d1',
    'a5fblk_f0a2900df43da9d22f8c',
    'a5fblk_dc288b25574f6b342875',
    'a5fblk_d87b32559e57f1c49143',
    'a5fblk_bef0820f6c9d77c6ebf3',
    'a5fblk_2810baa07d3c89f7630e',
    'a5fblk_ae9d9c60d44c08854e2b',
    'a5fblk_33aa7c1c0391c287ec7f',
    'a5fblk_aed981afaf3744bb4bf4',
    'a5fblk_3e89972dad36f9ed871b',
    'a5fblk_b05e28be90cd3fbfdef1',
    'a5fblk_c6417e46573d55d1239b',
    'a5fblk_e91c2e6c97a636ef2ece',
    'a5fblk_d7fbce3a4f4f913c1b4e',
    'a5fblk_81c6b91f3bf7d32ffd6e',
    'a5fblk_a1ae0e23805c66e525cd',
    'a5fblk_0a0590fe3cd481d4e1b4',
    'a5fblk_c356f4573208efe78d66',
    'a5fblk_d502e86fd516acf70e41',
    'a5fblk_9fa3a3579a2bfae647ba',
    'a5fblk_3e97839c051bae158e88',
    'a5fblk_35c5a62a403333bbea7b',
    'a5fblk_e84df861ce1a97a060dd',
    'a5fblk_1e7c97e522847fca13e2',
    'a5fblk_bcf39a4a05891a5d621e',
    'a5fblk_9fc71bfebe6271cb1e3c',
    'a5fblk_7939e9650139ed8da0ae',
    'a5fblk_20f805a726c16c4b5b41',
    'a5fblk_09e9aebc41683d6b426a',
    'a5fblk_3a2b9b5306d1f69e85a1',
    'a5fblk_6a41e338e25f263f8ec1',
    'a5fblk_4d91aad9a17e1defeb56',
    'a5fblk_f605e40034546dbc032d',
    'a5fblk_6ff32c1e3e349b8b3667',
    'a5fblk_ccbf52a43e026f2d8fda',
    'a5fblk_b5e66895bfbca45443dd',
    'a5fblk_c309d283d9e5846d684d',
    'a5fblk_b14adaaa62671e121603',
    'a5fblk_c74c6cbe60fcc516dbb6',
    'a5fblk_6e3492ea5ecc4383c118',
    'a5fblk_c90a0fd6de09fa7531e6',
    'a5fblk_c09c9ce32c33d40c3685',
    'a5fblk_0bcdf99c141dd931450e',
    'a5fblk_91be66586e27c553a0f3',
    'a5fblk_5dd43067ba98917d2ddc',
    'a5fblk_6bf22e395981637afdbf',
    'a5fblk_6c27e5652c715c1685fa',
    'a5fblk_0634de928eb940bf4767',
    'a5fblk_5c6b9fa7e4896b5e45e1',
    'a5fblk_724c9335fed44f7751e8',
    'a5fblk_1fa33955f2b4b61b0f63',
    'a5fblk_e58d2d0525160647ef0b',
)

_SINGLE48_P2_V1_REQUEST_HASH_FIRST = '97641e2f054d758bc4bc36122ed82e8ecdfe864aadf29f69987c7ea2fa1a093c'

_SINGLE48_P2_V1_REQUEST_HASH_LAST = 'db57f893bc249a566a1e2348b0e5d32748db6b76febc996d5107a50b35989515'

# --- Exact v2 pins, mechanically captured from the current deterministic implementation. ---

_CORPUS_P2_V2_BLOCK_IDS = (
    'a5fblk_31abb14a06ef51b829f1',
)

_CORPUS_P2_V2_REQUEST_HASHES = (
    '67c73eb0d481c629c54a0644f186d82f62c098dfe735cb669d98fb13549cd229',
)

_SINGLE30_P2_V2_BLOCK_IDS = (
    'a5fblk_9232f849d0a9cb1e6000',
    'a5fblk_b7a7e7985fd1f770e0a2',
    'a5fblk_cb1021cc9eed1ed46784',
    'a5fblk_9e09aace246f70cdeb7c',
    'a5fblk_081fec16aa620e868ad5',
    'a5fblk_8bc132e960e0179146f7',
    'a5fblk_8e6f3a27e9ee4dcd8f20',
    'a5fblk_61692f87ccc5f5bf0dc7',
    'a5fblk_29c88f575064e110fef9',
    'a5fblk_e6b009ae511a2d20f77b',
    'a5fblk_8e7a0d0f5f45f37ebdc6',
    'a5fblk_e7df326175b9776b30cf',
    'a5fblk_0a0fb9992c87591f2f32',
    'a5fblk_0afa0445baf5bd598670',
    'a5fblk_bca9be2d02895dc089a6',
    'a5fblk_dec18a75b6902049329c',
    'a5fblk_8214b11e6a28babb8ade',
    'a5fblk_be621fefa2f42f191e71',
    'a5fblk_dec281c96a3615a07245',
    'a5fblk_d2078e027857232fcfad',
    'a5fblk_6471fee15e0109171de8',
    'a5fblk_060d05064aa6f121df86',
    'a5fblk_2c97df0658cdd8569ab1',
    'a5fblk_6e81742bf8b73a403799',
    'a5fblk_10873c9c9cb6d945ed40',
    'a5fblk_d1f8fa5ce13563a0bc7e',
    'a5fblk_84f8cb0b031c044e7e24',
    'a5fblk_e4b03d1c43a3f09afb85',
    'a5fblk_649125a057034c484433',
    'a5fblk_34f737f3ae4de8a6a958',
    'a5fblk_1ef82d532673ff95913d',
    'a5fblk_af51a132a6b6d1e7ebf3',
    'a5fblk_c8273ac6c3c0123a4775',
    'a5fblk_3db81865b23e2125c5c7',
    'a5fblk_1a749caf9c9ed9f20ce2',
    'a5fblk_7889d5afa7bd99492c06',
    'a5fblk_094447ffb610cc9ddeee',
)

_SINGLE30_P2_V2_REQUEST_HASHES = (
    'a7d4c95c00fa5af5b19ee640a72bc1cf25ba067b250f23338c5d3885b864b6ca',
    '3b89e9378376a3e45b5102c4162f410096507518ed57fad806026d6e0a856e83',
    'c3a85a157a5798db096dc7dcbd4270eba6fffdfe45fb9e4827f1fb76310bdc53',
    '0ffd7acbc7bc6f9f9b00305720a7cdafb62cf0dfbec2041022ffa6286972430c',
    'aacded0cd9b9589042091fd2232079400b764e151e8c69cd9e7d5b06cfe828a2',
    '41fbfd729460d6b93d2f8d692485e98c9b2c2bbfa3129e336ed60fd749534c7c',
    'b160b1bd3510f830e0361e913018e842a971737240cd2f70ff2abbb25a1486db',
    '4cc680c9ca556c99b685ea4d6b69bba6bf66d27862888b1f235513296e0d8ec5',
    '621cfe1e58f684288eef637b8cb957d497cb63a0a60cc2b70da3c0700d1a4560',
    '8b2d30c73d863ffffbcba8deb63ee3aad8db69a6cb58ec63a52b7764d2e1f354',
    '6a89ad78ccc50c5e71c6eeea4fde9a0ef024cd64d80296bd6346ec5ee5d19028',
    '26e5de91e0cb046ef83a5fea679249f45f178d2b52565f36f1a978d18efbf3a3',
    'a405dde952b4287e2e65b1ac0842fed389515f9b9ef8577ef14a7c59e246a0a4',
    '8b69c4a1625b58b5287974755ced9df55c37b5a884bba0351fc986f5d1799db1',
    'f65e71490f8775a2cb6625c6362d37af6681f4597c4dcd1eee528bb105fc2930',
    'b536e396da7b5622e9affeb0e4048f51c0400fb3ca1e8576aa9057fe33ff8b58',
    '417f98ae352f2bec8a6e7b091eb16165dfc10940fa0c5e05333ef77ee349013a',
    '332e87a80a636da08676257e668532aa12266e7100b74ec60a816e5e9c5f53e0',
    '601285c5b4604c2c7629435092340cb47b85d60917a4876b6da912ba7d13e969',
    '19784fe6f8bbdcf98e50bf3306ccb1deba7a4a26c3df36e1089a186e8ba427fd',
    'ad9496963b851babad19ffa0ac2ff2f94ed07d5c9e781bd1a8b314b63a8e3b33',
    '443b745d1a580462967fbf2475df117c12dc0c4892b9ac9b6c280b34cbefd295',
    '7c5caf363d9edeb364449d551fec1f6542ef22bf8e8c3fa4fc3b21c712735e52',
    'be44efa1c83e8aa427c6b73793171a1fbe0545db82f22644700f665870f0deb0',
    '888df5937823f27149ad0d55aca17a6a2fb8c19a9bf139a2839879fd38af111f',
    'b39a46ea2b39e4707b449bd205b73016dff7fbd077964a67232c880498adcd8e',
    '569e9875f86c9a20f74312f67fd36ea42a3edee62f54c3ef664c10e9e0ea07e9',
    '6d9db790a4958037a12694136a0a6a102c34cd9a8dac23b9146e43f286b503fa',
    '23b85aee627a69338ab1ac3752d0ea298c161686536b190ce9f3cdd4085ac7c4',
    'bceffe6d1ef9342fb980ca11e1694aafaf91ef1ecd6602804c49e371b498c9cb',
    '0cf37f756ed34b91688fa428bf93d9b4d1ecafe71a821dff1b3442fbcc70a8e5',
    'a0f24f879b1efabbe74b3bc86a920c764c0c19989f6fe370b2d7dfa50b29d088',
    '45f445937e7954a7d7264d7578a24032c7e60df2c217cbb2e90adba2d78a371d',
    '151e1933bda4cb5fa3f80a26bcd32af894f3c1c55f5a586d948046a7d964f2e8',
    'c6614250a9fdc74cebb78284cbed8764f8bd4336256c5b61653f0de2e5fb9005',
    '38aa63807e419bddd4f1fb53ddf88b32a8d20453678ecd12519f3b9debf3ac84',
    '2ebcf444a2fc36ef91608ac19429c23e61d7b5cf2ea71c2f2e00e8c4a23381df',
)

_SINGLE12_P1_V2_BLOCK_IDS = (
    'a5fblk_ea7c28f654e7ccbbcab3',
    'a5fblk_1840fb4bc5a05ba16a2d',
    'a5fblk_e0008349f30b3887f75a',
    'a5fblk_4acf00646562dd8d47c7',
    'a5fblk_44d4d5d5836b9f77948c',
    'a5fblk_2572d85532f460daaa53',
    'a5fblk_66000f7be1bfc1dc551d',
    'a5fblk_4013c91aa8193b66fb77',
    'a5fblk_aa46e142c2d38864c494',
    'a5fblk_bc9edd08c242e5691a0f',
    'a5fblk_be5f28f6481475fea426',
)

_SINGLE12_P1_V2_REQUEST_HASHES = (
    '272e266c35b16ada05dea3053c79f29c08ca19f5958da71d34686adadc822f55',
    'cbea076e51c2c11ed5a18d259a8384bccc0585ae3f51f2eb9da5c72e215b1b8b',
    'ade3e11edc7ccab72c62551fcdffbd103cfdb1deff714e06d60adfcd7906484e',
    '4c0d93d62bf444b4aaa0b906e7eeb288c4c330d307eeff4efd5f6e3d5c7afb7e',
    '06c57b8673ca40100aea47b3ebbd746004a3d192c0b72d80010eaf2d3b33cb67',
    '556898bfd55aed3f676dcaf20bd12a78e43a4a0d4713b6ffbe6b0f42b4acbf23',
    '3ca75c5d9684d02e45b12cd0950455bb4c826f47d95c19f823d8ec49df35926c',
    '750f1147fdacdbe22f64afb200c6d576fde29c716fda1f9b2d296b62d1d34589',
    '0217b18d83c458f08c9762eb4ebf3110938a4dc88d3b2f41e4d586f219ebfabc',
    'b5293c4149fe2e9c2f13e14e342da8a10a634c6a93fa42bc4e28265f8244d413',
    'a8be3e0ac0df1ef1afadfba0e16dc5cd698290f9c0a697fd1b7262d2ed047f87',
)

_SINGLE48_P2_V2_BLOCK_IDS = (
    'a5fblk_9e2dec3cb8765dbac519',
    'a5fblk_0175580e5dad00aa53ae',
    'a5fblk_32ece504fa1af6140dd6',
    'a5fblk_bf30a47c3d69428dabb9',
    'a5fblk_728ad2acc68e0aa631ad',
    'a5fblk_56c1841cde1dc7f20e22',
    'a5fblk_6191927071326f8f2e75',
    'a5fblk_60ca8d653cab5c48b7d4',
    'a5fblk_ecf0b975f0f7c99dc1c1',
    'a5fblk_7f4cb1b15a93b893f1eb',
    'a5fblk_6b7bc43232b1f7915c30',
    'a5fblk_a77a95e17bdad247e817',
    'a5fblk_012edee7673ea0a046d4',
    'a5fblk_96758c3b398dbecb44d2',
    'a5fblk_c4ed1d4d4b1611c62c2a',
    'a5fblk_65e253a5d99ed9bc513b',
    'a5fblk_69eda3bbf9e840fe5ace',
    'a5fblk_52d18ea0e18a7847ce9c',
    'a5fblk_65c5fa4fbaa7fcd2c79c',
    'a5fblk_f13a70d50c5f926f11c5',
    'a5fblk_f07364fcea5cc48ba11e',
    'a5fblk_d2b4e55d6ff991ee024b',
    'a5fblk_9283f0ccb0344e5a6e58',
    'a5fblk_9c44d318af218e7e4df7',
    'a5fblk_ad8e089968d2523e9e29',
    'a5fblk_1e7a69550bb4059179b7',
    'a5fblk_8e353e90ed980cce0b2e',
    'a5fblk_49d4a5c976e50d4db3a5',
    'a5fblk_cc83e08c425e17970ef7',
    'a5fblk_a585df7e9b2ac4a97c52',
    'a5fblk_a72ea9e4011333f131bc',
    'a5fblk_b9a0f5c684e15092ef6d',
    'a5fblk_9d11483cc83ab60c3fb5',
    'a5fblk_f4b5547be565a3a8c8f8',
    'a5fblk_794bbc96a66f696ea628',
    'a5fblk_b9c4461da09beeb1dbbc',
    'a5fblk_dfe1d2081b8e2681584f',
    'a5fblk_f620b182c23c7df83347',
    'a5fblk_3a121c16b452af621e02',
    'a5fblk_3aef30a07fd52806f94a',
    'a5fblk_b5c92122d7a1f5c39ee6',
    'a5fblk_55d66249a562f1cbcfbb',
    'a5fblk_92d200f53fcae177010b',
    'a5fblk_3e2d97bc08c6b83763e7',
    'a5fblk_541cf3dec7cd19fd2962',
    'a5fblk_139b0a0a0980c42f3fdc',
    'a5fblk_7fe0cbc2e418c900b836',
    'a5fblk_139c0a18d7e793a5a3e0',
    'a5fblk_cbefd0273239e09a2c14',
    'a5fblk_7af69c03daf2acbea1af',
    'a5fblk_23651e1a4236fd59508e',
    'a5fblk_55148f686846fdb14913',
    'a5fblk_fffb2d196d2fad211758',
    'a5fblk_f53de06c511ed1505bed',
    'a5fblk_26be9222ea3bc0409d86',
    'a5fblk_761cd0bf20a7c85abd38',
    'a5fblk_fd583f5e16189eb5eb6f',
    'a5fblk_735a5c5c8d74be6df772',
    'a5fblk_4c00d144d0d3583455e1',
    'a5fblk_0d1469243962c6bf4d24',
    'a5fblk_423ddcc55e93ff29a34c',
    'a5fblk_f926ef4d66b654a2cd3a',
    'a5fblk_91d3a48d52a1d3174ef8',
    'a5fblk_4b8c86237e3dfc3b9754',
    'a5fblk_59fa2354d2a6dfab5296',
    'a5fblk_a7d2d82c8f05c6c9a249',
    'a5fblk_e9e7a593838208d1d9e7',
    'a5fblk_0b3503e97551c5122590',
    'a5fblk_bd2ae4cd9c924aca10f8',
    'a5fblk_60bf7cfcd011149fc232',
    'a5fblk_955654c2ae5a40b183a9',
    'a5fblk_f183fe3b843ade463029',
    'a5fblk_43743d42cdfb6bff78fc',
    'a5fblk_fdf0492853db146e206a',
    'a5fblk_cea72b1b404f6f67941c',
    'a5fblk_2153cca1ca40ff65e5d5',
    'a5fblk_19f13604ac2398ecd5c9',
    'a5fblk_be501c95b608bb45d029',
    'a5fblk_d27eb3d78a7fa01401af',
    'a5fblk_9d5d31e9a22d1d9de6d8',
    'a5fblk_58d99fe72917e54141d1',
    'a5fblk_283485bc8a75c6cea834',
    'a5fblk_5b82642b9621a3e283be',
    'a5fblk_38d6dcae06aa54145faf',
    'a5fblk_3b292cc310f358a8c042',
    'a5fblk_b58180e0009b1e7bb066',
    'a5fblk_99e19eb0566d95b8dc8a',
    'a5fblk_65927d8dd66976962006',
    'a5fblk_56a84f7d5a5f2ab52dea',
    'a5fblk_1a380e069b8c7ccd8c4c',
    'a5fblk_ffa794719bea574bc43e',
    'a5fblk_db5f8c1032587987c095',
    'a5fblk_ccfa106f92ddb5eb4da4',
    'a5fblk_17085632b5243c375baf',
)

_SINGLE48_P2_V2_REQUEST_HASHES = (
    'cc2881c5c78363b03ee391efc93d569d34940f617d2aa8bbc5a08ff840c2916d',
    '78d176ecd6daca713273663d0b7231fcc1a8b53b10d93b146d1e60ec83a59a12',
    '2851ce0c00b83f7a64d907dbb03c51de307c33bc248dcf6965b530d67d294ff2',
    'eb3e8ce874a3343a64de129de1d2c0cb82841df3cbae49d23ad5e56f053c6c88',
    'bb156001755e059aa26c6aa803ac3e3d6cf78dd888d558eecd82f9b36236fcd1',
    '020bb39b044cda74ae8dff963aa2a251f588017064bad53e6d32e7cce8624b3e',
    'a94810ad0abcd9803321120f3f3f1bf563ef405a3b3c86ac184519abcf46f8f0',
    '0fc9c2eda376772cfd29453b963851f0cd506d7f8e005066cdc53b3c4f32126c',
    '368c161fe04363509a94bd141c25986ed0d8c412bfb6a80974da5ea0493ebdd6',
    '14f7e587aeb5c92a3263f12137efa6a126d8f3433dbf9f0ba7f73f29168ac2bf',
    '26a63f713b5d2e2dd72112fab9d36f1b7962ceb15d411301cadefe0f6f63f595',
    '22d41929294b3d4d9e095b65a509d098c1894bffaf775880e362dd5180d687b4',
    '599098a5f8406c8dd3472668207b06880e3d17d41ba8257d291b929c722f5d57',
    '8ed52bc376f57cfbdd365b2643f600b51fe4c7c8889126e339d439d1c4550cbf',
    '6aa22f3b8260996ee005c44f46e3810e29f4a07690e1ebf8a186c48d1c8a771a',
    '1d877a57a6c07419169fba5a28e8c90e48f78b2402accdc1875b8cfde6e5c185',
    '2c615334c478cf4838ed16450a956a9eacf0c4dd96ffbe1373e82ae19ff5a941',
    '1f959797e97ebbfe1647b1abbefe3ba80817de935fd2d56ca6b5ec01b38b98f5',
    '463b9472520e7c69f7d18f89a3d617da1179fb5a930ab297f6330080535bdb86',
    '2b9e0ccdf8c37d94ee09c4f2ddc82c481a5f086ab418146ef2ab9c3249d9c33a',
    '3371b980a386767d01b480dd8a7d253d68d22f675e91c8b524235ccf5aaec66a',
    '1450c1c0c215113a12bc794696452b357d7dfef73f5a74302050e689e906ee22',
    '6a8514c38aa53a458adf00d552431d319898fd5e40f9c702146d8e9a21b26a3e',
    'c1c96a3192e70538b5dcfad2367ad1b4562a6386f12dea36176d4312d796f454',
    'd7ca32a5c1671542f394527635e1e2620869ef020056b47699c549ecfe2464d7',
    '4f3867f7a33c9c9409187be8d989e77d5d676370f97c6f0042cca768ec14785f',
    '4f0f3f97882fdab7e1d8e67aa61f0d69ea5c1268d87b3a1448118bf0f30f51d3',
    '27940f0362a8f937f4875059be0298ace0489773cf541e884a15a799cb7db6c0',
    '4f6c560faf1b6427db6351d3c9d7c5ee4a8a85edbef11e175533f43faca1daf6',
    'fa6b832fe15aa3c688a44b7f9a05c2e847e2beb7d83a7d1431b9e94e7f8756e5',
    'fceefd58831cc447c9719e667454e35e605acde6c54940ceb26772a8ffad3371',
    '7f5825d17cf1eb4fd422203d02e9ef7ed48531995e9b710ee466f2ac3e5267a5',
    '58250f96a205e1ea9f187466564fb513ca27265b3c35e4c0e5881ebf7f7d6863',
    '1a31be40e7825e808844bfae2b089c55085393811325022e17d7f1f36c5fa818',
    'ff6037364c46700921d8bd3de1bc55491b8adf6c5f07a82321e619732217fe16',
    '053c4103678820657de59828bdcb0a4bcb6ade7570041b2515b349c2b4f3262a',
    '85db94b863519afdba7f8cf0bb466adb10a9384d708c1bc9653406cb177ba67e',
    '89cdc7c6997f742b9bf032b9d3b25d840a6d1ab069070f7996666536ddf96f6d',
    '8c833ea65954425bd8650aeaa7a3ad0cf05e8e9b6363940f759a010f2caf05b9',
    '2937aba8ad7696d59698ef9369cf76a97e21d6daa86b57ea48bd35b00aa9270c',
    'a2f8a52937247445a2881bc050688d981bf5c784af91f7cb430aba53f674798a',
    'b252fa7d650e56532845b293da90b140012d58582c948bbcab1fad2f1cd5d045',
    '357b13a78b47267bd313070c49ea44243220dc3c661509fe96ae6ad18dba793b',
    'c41bf8b364a6bfd9df767d20bff7556f2d1f33044138bbfc8ca1be400865f0f9',
    'bf2b5961b52742f24f2cbc499cf3cc831216dbd5d68b14ebeab95caeec7e8ce5',
    'ea99bdaae109d144402c536a8c4258ac211ff31f7c8880ff3043d88cfc6f7ae6',
    '2da03628ecf29eb158570dcfb849251caa0c27b0ae88b4c2126f600883f2aafb',
    '834ff694c2294625d0d8fa769891c970b01f505c7c384160be6d7b92bfdf1b13',
    '8f4f8c733430a9fd12f03834be6788d603f476b3a2aa738673eb3fe18e0c98be',
    '4060cf57a0f8fc8f1837b63427ea3a8bbdd73db7a6bcf91ec37e1075cc82e78f',
    'aeadc862bacea3a045d61f93c612755b5ba9e68becf312d55888d39362041280',
    '58e63a20a06ad8364bcba452f32fb4b047032ed10698166a6d950a7cfb57bfc4',
    '24878a77619aae010aaa93547171c91eec30c9c98416ba1a9271bcc26df8bfb6',
    'ea18654a1f63a0f19b4248acf09628cf6dd092ea85d70ee546f1066bc5c301d2',
    '263fc1da349f9c624d2c44afc3868aa9cc32e9bd957698eb6c066e87a987e5c6',
    '73d2b486dba3dc123285037eb9ab886dcf2ccc959ed5ade2937f201342226803',
    'bf774363eb42e98309d117b127442586eab4a16f6703515f06b5d8065a6e546f',
    'b28caff0eb3b6b6fcc9fa74f66be303c92abbe1b5b67245e56fb2839f0136615',
    'efcc647d641b3810bc535a39ba2e0403e03843b90dfe8a557f5e75bea14dc1c1',
    '809208ffe02f57a2d78f99e6edc585f5f43af7f3671e0029cc0d7d981c577529',
    'ffaffdab3915364d2ed4b098f48441ff015bdc707a8d147785a5dd1b5c2c8189',
    '000e8c2d38ce8e84ed9f8dfc05c7b4a11928927bdc036920710ba0966251dd1c',
    'ee89b543a0bca02658695f62bc7c04b97349db0c2a1d5db1d2dfbc4ce7c5d744',
    'fbdba579ce11584ede55c3fa1cfdc0b50c22e3dad03b7b718c3ccd636549eae3',
    '32d569a9934830819d0598f2a204e6f2656a34ebc7ce61a2992bab8de5b251db',
    '9f1d8e5021664b3a6ede31ef99a5699c0944938afff6a532f021f66f0deea97d',
    '1d72f459bba750eab2dec6d68f02d1da9e2cb3e3a5a65c46ed1cc50762270d65',
    '2b3ed4a8d110046997617203150a35f649c312e2d87ec1e842c6937574972606',
    'c69ce76e257cc5b14c564b7383d4b4389bab05e33e4f1788918253ddc82305c4',
    'ae8f0a2ae784c3435c5328fa2a5f505906a621410b834c3dfefd4400045a19eb',
    '17a1c4c393338af3748056086c6b7dbd41b0a5d5199aef81ad0555f8cc53d0f1',
    '88ff6067b9237dde4aa9f53dd2979a3f0b1dc0b41226da61af90b1232fde6fb6',
    '98917bfeef26acefb12b8e05ac4d2a61cc1996e5f6a147a939f508d7bc6fea2f',
    '28cb3c572cbc994d7f44d2351693c583e8c5529857a05736123c1d97e1b4d64a',
    '224e357449b06764e10c03f9c050aa21c6c9c8beb8060a5ba60ccbe7199ae5b5',
    'b4ef6a1c716828bc26080ba4b8e1b4ada7d6e83bad3090f0bc162a4adeea87eb',
    '0e4e9d32ebb7202814007517f660918d364d7e116b8c70a5d89d1ecefde24831',
    'ba94bfdfaaa1289360136f6a31029cfe4d38e92dd5c1a55f5ffc950fcef20589',
    'bd3406c7dc30413f503c0f06015f0bfdcca08c491d6ee15ccc313c62e8fa50e2',
    '4e2567b98198f6f73c99dd8b8d51f2ba69fdc0bf639d40da6c605fbb1882daa3',
    '1166fbcea676e405b8989f6f616f37dbe4f37a5d897b8592c627ee270271ef96',
    'fd66537f3f07b2f59ed572a0a98b1992de24f7d0388a68a2f27ba48294d81fff',
    'e3478f73509d730ff5ebd28dcde7ce2c183e4a30ccca332306f26232d6514d5e',
    'baa34064a9b821f1f1bf4eb3c48c6df965227e5b69223b19e3971514110a9376',
    '41b1914eaa11208864d81176e6789010fbb59bb2bc0ebe38c5a5200b5b00dbb3',
    '2ea0ab7fa58eb3c1027b414faec04aeed0ae8f7d1c1b09af0d9dd6b7ef8d5a49',
    '3927d2afb17a1cf8dc1ca9c193c98784bae94c272bb96c7ee66c23925fd32d89',
    '42c7b569352e26dd6a8baa740ea1a5f1a70ebc08da59e7071abc5ae771fca9ab',
    '475500b22107ea455ad701f08c876e66f496b98e2678f10ec0f8973db5f62977',
    'be1ec4302aad4e478bda3f60495440b39aed74b1a694006ce59c6b9dbc808185',
    'bc75c8312a0aa38ad4519f798dd8519e994501a00c210f010a560f1ec2cb4aa2',
    '7e704168b9c51a66cadf94975f80635da2c412fd624609f834f93b99081b661b',
    'a7320562430f30c572222957c7bd809bd11259380d04a40631616a280ad9e463',
    'f10ee4e6882081b27ba5bfb32677ca865d035785d27ce630f3b0a777a83fbde4',
)


def _fact_prep(specs, policy):
    tree = _fact_tree(_tmp(), specs=specs)
    planning = _fact_planning(tree)
    return build_fact_semantic_preparation(
        planning, _consolidation_profile(), _FACT_SEM_PROFILE,
        prompts=_FACT_PROMPTS, packing_policy=policy,
    )

def test_fact_corpus_p2_pins_exact_v2_identity():
    prep = _fact_prep(_fact_corpus_specs(), _FACT_P2)
    assert tuple(block.block_id for block in prep.blocks) == _CORPUS_P2_V2_BLOCK_IDS
    assert prep.semantic_request_hashes == _CORPUS_P2_V2_REQUEST_HASHES
    assert _CORPUS_P2_V2_BLOCK_IDS != _CORPUS_P2_V1_BLOCK_IDS
    assert _CORPUS_P2_V2_REQUEST_HASHES != _CORPUS_P2_V1_REQUEST_HASHES

def test_fact_single30_p2_pins_exact_v2_identity():
    prep = _fact_prep(_fact_single_chunk_specs(30), _FACT_P2)
    assert tuple(block.block_id for block in prep.blocks) == _SINGLE30_P2_V2_BLOCK_IDS
    assert prep.semantic_request_hashes == _SINGLE30_P2_V2_REQUEST_HASHES
    assert _SINGLE30_P2_V2_BLOCK_IDS != _SINGLE30_P2_V1_BLOCK_IDS
    assert _SINGLE30_P2_V2_REQUEST_HASHES[0] != _SINGLE30_P2_V1_REQUEST_HASH_FIRST
    assert _SINGLE30_P2_V2_REQUEST_HASHES[-1] != _SINGLE30_P2_V1_REQUEST_HASH_LAST

def test_fact_single12_p1_pins_exact_v2_identity():
    prep = _fact_prep(_fact_single_chunk_specs(12), _FACT_P1)
    assert tuple(block.block_id for block in prep.blocks) == _SINGLE12_P1_V2_BLOCK_IDS
    assert prep.semantic_request_hashes == _SINGLE12_P1_V2_REQUEST_HASHES
    assert _SINGLE12_P1_V2_BLOCK_IDS != _SINGLE12_P1_V1_BLOCK_IDS
    assert _SINGLE12_P1_V2_REQUEST_HASHES[0] != _SINGLE12_P1_V1_REQUEST_HASH_FIRST
    assert _SINGLE12_P1_V2_REQUEST_HASHES[-1] != _SINGLE12_P1_V1_REQUEST_HASH_LAST

def test_fact_single48_p2_pins_exact_v2_identity():
    prep = _fact_prep(_fact_single_chunk_specs(48), _FACT_P2)
    assert tuple(block.block_id for block in prep.blocks) == _SINGLE48_P2_V2_BLOCK_IDS
    assert prep.semantic_request_hashes == _SINGLE48_P2_V2_REQUEST_HASHES
    assert _SINGLE48_P2_V2_BLOCK_IDS != _SINGLE48_P2_V1_BLOCK_IDS
    assert _SINGLE48_P2_V2_REQUEST_HASHES[0] != _SINGLE48_P2_V1_REQUEST_HASH_FIRST
    assert _SINGLE48_P2_V2_REQUEST_HASHES[-1] != _SINGLE48_P2_V1_REQUEST_HASH_LAST

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
