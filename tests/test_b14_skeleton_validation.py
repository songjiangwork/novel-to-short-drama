"""B1.4-A2 graph validation: pure graph cases plus exact-authority negatives."""
from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.adaptation import (
    AdaptedStoryBeat, AdaptedStorySkeleton, ArcCheckpoint, AudienceKnowledgeState,
    B14SkeletonAuthority, BeatAccounting, CharacterKnowledgeState,
    CharacterKnowledgeTransition, ChronologyPrecedence, GapAnchor, NarrativeDependency,
    NarrativeOutcome, PresentationEntry, RevealTransition, SetupPayoffLink, SkeletonGap,
    WithholdingBoundary, resolve_b14_skeleton_authority, validate_adapted_story_skeleton,
)
from short_drama.artifacts import ArtifactRef
from short_drama.adaptation.target import CanonicalTargetReference

from test_b13_disposition import _plan


def _ref(kind: str, name: str) -> ArtifactRef:
    return ArtifactRef(kind, name, 1, name[-1] * 64)


def _authority():
    plan = _plan()
    return B14SkeletonAuthority(_ref("adaptation_strategy", "synthetic.strategy.a"), _ref("b13_disposition_publication", "synthetic.disposition.b"), plan)


def _skeleton() -> AdaptedStorySkeleton:
    authority = _authority()
    first, second, third = authority.disposition_plan.adapted_story_beats
    character = authority.disposition_plan.adapted_characters[0].adapted_character_id
    return AdaptedStorySkeleton(
        authority.strategy_ref, authority.disposition_publication_ref,
        chronology=(ChronologyPrecedence(first.adapted_story_beat_id, second.adapted_story_beat_id), ChronologyPrecedence(second.adapted_story_beat_id, third.adapted_story_beat_id)),
        primary_presentation=(first.adapted_story_beat_id, second.adapted_story_beat_id, third.adapted_story_beat_id),
        outcomes=(NarrativeOutcome("outcome_start", first.adapted_story_beat_id, "获得行动条件。"),),
        dependencies=(NarrativeDependency("dep_action", "outcome_start", second.adapted_story_beat_id, "行动以先前结果为前提。"),),
        character_knowledge_states=(CharacterKnowledgeState("ck_initial", character, "secret", "unknown", None, "开场未知。"),),
        character_knowledge_transitions=(CharacterKnowledgeTransition("ck_learn", character, "secret", "unknown", "confirmed", second.adapted_story_beat_id),),
        audience_knowledge_states=(AudienceKnowledgeState("ak_initial", "secret", "unknown", "观众开场未知。"),),
        reveal_transitions=(RevealTransition("reveal_secret", "secret", "unknown", "confirmed", second.adapted_story_beat_id),),
        withholding_boundaries=(WithholdingBoundary("hold_secret", "secret", "audience", None, second.adapted_story_beat_id, "在此之前保留。"),),
        setup_payoffs=(SetupPayoffLink("setup_payoff", first.adapted_story_beat_id, third.adapted_story_beat_id, (CanonicalTargetReference("fact", "fact_000001"),), "改编层回收。"),),
        arc_checkpoints=(ArcCheckpoint("arc_commit", "character", character, "commitment", "作出承诺。", third.adapted_story_beat_id, "主角弧线。"),),
        gaps=(SkeletonGap("gap_ambiguity", "knowledge", "intentional_ambiguity", "细节有意保留。", "不是结构 defect。", (GapAnchor("knowledge", "secret"),)),),
        beat_accounting=tuple(BeatAccounting(beat.adapted_story_beat_id, "directly_presented", "进入主线呈现。") for beat in authority.disposition_plan.adapted_story_beats),
    )


def _codes(skeleton: AdaptedStorySkeleton) -> set[str]:
    return {item.code for item in validate_adapted_story_skeleton(skeleton, authority=_authority()).findings}


def test_complete_legal_synthetic_graph_and_b15_primary_projection():
    result = validate_adapted_story_skeleton(_skeleton(), authority=_authority())
    assert result.valid
    assert [item.presentation_index for item in result.primary_presentation] == [0, 1, 2]
    assert [item.presentation_id for item in result.primary_presentation] == list(_skeleton().primary_presentation)


def test_partial_order_allows_incomparable_beats_without_inventing_story_order():
    skeleton = _skeleton()
    first, second, third = _authority().disposition_plan.adapted_story_beats
    skeleton = replace(skeleton, chronology=(ChronologyPrecedence(first.adapted_story_beat_id, third.adapted_story_beat_id),), dependencies=(), setup_payoffs=())
    assert validate_adapted_story_skeleton(skeleton, authority=_authority()).valid
    # A semantic causal assertion cannot silently use a topological tie-break.
    skeleton = replace(skeleton, dependencies=(NarrativeDependency("needs_order", "outcome_start", second.adapted_story_beat_id, "需要明确先后。"),))
    assert "dependency.chronology.unproven" in _codes(skeleton)


def test_chronology_cycle_endpoint_and_dependency_contradiction_fail_closed():
    skeleton = _skeleton()
    first, second, _third = _authority().disposition_plan.adapted_story_beats
    assert "chronology.cycle" in _codes(replace(skeleton, chronology=(*skeleton.chronology, ChronologyPrecedence(second.adapted_story_beat_id, first.adapted_story_beat_id))))
    assert "chronology.endpoint" in _codes(replace(skeleton, chronology=(ChronologyPrecedence("missing", first.adapted_story_beat_id),)))
    assert "dependency.chronology.contradiction" in _codes(replace(skeleton, chronology=(ChronologyPrecedence(second.adapted_story_beat_id, first.adapted_story_beat_id),)))


def test_compact_explicit_identity_and_same_beat_teaser_then_reveal():
    skeleton = _skeleton()
    first, second, third = _authority().disposition_plan.adapted_story_beats
    presentation = (
        PresentationEntry("pe_teaser", third.adapted_story_beat_id, "teaser", "先见结果。"),
        first.adapted_story_beat_id,
        PresentationEntry("pe_reveal", third.adapted_story_beat_id, "delayed_reveal", "后见真相。"),
        second.adapted_story_beat_id,
    )
    skeleton = replace(skeleton, primary_presentation=presentation,
                       reveal_transitions=(RevealTransition("reveal_secret", "secret", "unknown", "confirmed", "pe_reveal"),),
                       withholding_boundaries=(WithholdingBoundary("hold_secret", "secret", "audience", None, "pe_reveal", "保留至揭示。"),))
    result = validate_adapted_story_skeleton(skeleton, authority=_authority())
    assert result.valid
    assert result.primary_presentation[0].adapted_story_beat_id == result.primary_presentation[2].adapted_story_beat_id
    with pytest.raises(Exception, match="duplicate presentation"):
        replace(skeleton, primary_presentation=(third.adapted_story_beat_id, PresentationEntry(third.adapted_story_beat_id, first.adapted_story_beat_id, "full", "冲突。")))


def test_audience_reveal_chain_direct_confirmation_and_withholding_boundary():
    skeleton = _skeleton()
    first, second, _third = _authority().disposition_plan.adapted_story_beats
    chain = replace(skeleton, primary_presentation=(
        PresentationEntry("pe_suspect", first.adapted_story_beat_id, "teaser", "给出暗示。"),
        PresentationEntry("pe_confirm", second.adapted_story_beat_id, "flashback", "正式确认。"),
        skeleton.primary_presentation[2],
    ), reveal_transitions=(
        RevealTransition("suspect", "secret", "unknown", "suspected", "pe_suspect"),
        RevealTransition("confirm", "secret", "suspected", "confirmed", "pe_confirm"),
    ), withholding_boundaries=(WithholdingBoundary("hold", "secret", "audience", None, "pe_confirm", "前置保留。"),))
    assert validate_adapted_story_skeleton(chain, authority=_authority()).valid
    direct = replace(chain, reveal_transitions=(RevealTransition("direct", "secret", "unknown", "confirmed", "pe_confirm"),))
    assert validate_adapted_story_skeleton(direct, authority=_authority()).valid
    early = replace(chain, reveal_transitions=(RevealTransition("early", "secret", "unknown", "confirmed", "pe_suspect"),))
    assert "withholding.violation" in _codes(early)


@pytest.mark.parametrize("statuses", [("unknown", "suspected"), ("unknown", "unknown")])
def test_duplicate_audience_initial_states_fail_closed_even_when_identical(statuses):
    skeleton = _skeleton()
    duplicate = replace(
        skeleton,
        audience_knowledge_states=(
            AudienceKnowledgeState("ak_first", "secret", statuses[0], "初始状态。"),
            AudienceKnowledgeState("ak_second", "secret", statuses[1], "重复初始状态。"),
        ),
    )
    assert "audience_knowledge.initial" in _codes(duplicate)


def test_initial_audience_confirmation_violates_later_withholding_boundary():
    skeleton = replace(
        _skeleton(),
        audience_knowledge_states=(AudienceKnowledgeState("ak_confirmed", "secret", "confirmed", "开场已确认。"),),
        reveal_transitions=(),
    )
    assert "withholding.violation" in _codes(skeleton)


def test_audience_withholding_requires_a_resolved_knowledge_key():
    skeleton = replace(
        _skeleton(),
        withholding_boundaries=(
            WithholdingBoundary("hold_missing", "unknown_secret", "audience", None, _skeleton().primary_presentation[1], "无效 knowledge key。"),
        ),
    )
    assert "withholding.knowledge" in _codes(skeleton)


def test_character_knowledge_requires_comparable_progression_but_flashback_is_legal():
    skeleton = _skeleton()
    first, second, third = _authority().disposition_plan.adapted_story_beats
    flashback = replace(skeleton, primary_presentation=(
        PresentationEntry("pe_late", third.adapted_story_beat_id, "teaser", "先见后果。"),
        PresentationEntry("pe_early", second.adapted_story_beat_id, "flashback", "后见学习。"), first.adapted_story_beat_id,
    ), reveal_transitions=(RevealTransition("reveal", "secret", "unknown", "confirmed", "pe_early"),),
                       withholding_boundaries=(WithholdingBoundary("hold", "secret", "audience", None, "pe_early", "观众后知。"),))
    assert validate_adapted_story_skeleton(flashback, authority=_authority()).valid
    character = _authority().disposition_plan.adapted_characters[0].adapted_character_id
    conflicted = replace(flashback, chronology=(ChronologyPrecedence(first.adapted_story_beat_id, third.adapted_story_beat_id),), character_knowledge_transitions=(
        CharacterKnowledgeTransition("suspect", character, "secret", "unknown", "suspected", second.adapted_story_beat_id),
        CharacterKnowledgeTransition("confirm", character, "secret", "suspected", "confirmed", third.adapted_story_beat_id),
    ))
    assert "character_knowledge.order" in _codes(conflicted)


def test_character_knowledge_state_snapshots_must_match_transition_progression():
    skeleton = _skeleton()
    _first, second, third = _authority().disposition_plan.adapted_story_beats
    character = _authority().disposition_plan.adapted_characters[0].adapted_character_id
    matching = replace(
        skeleton,
        character_knowledge_states=(
            *skeleton.character_knowledge_states,
            CharacterKnowledgeState("ck_after", character, "secret", "confirmed", second.adapted_story_beat_id, "学习后的快照。"),
        ),
    )
    assert validate_adapted_story_skeleton(matching, authority=_authority()).valid
    contradictory = replace(
        matching,
        character_knowledge_states=(
            skeleton.character_knowledge_states[0],
            CharacterKnowledgeState("ck_wrong", character, "secret", "suspected", second.adapted_story_beat_id, "与转换冲突。"),
        ),
    )
    assert "character_knowledge.state_conflict" in _codes(contradictory)
    unsupported = replace(
        matching,
        character_knowledge_states=(
            skeleton.character_knowledge_states[0],
            CharacterKnowledgeState("ck_unsupported", character, "secret", "confirmed", third.adapted_story_beat_id, "没有转换。"),
        ),
    )
    assert "character_knowledge.state_unsupported" in _codes(unsupported)


def test_setup_gap_accounting_and_local_invented_beat_validation():
    skeleton = _skeleton()
    first, second, third = _authority().disposition_plan.adapted_story_beats
    assert "setup_payoff.endpoint" in _codes(replace(skeleton, setup_payoffs=(SetupPayoffLink("dangling", first.adapted_story_beat_id, "missing", (), "无效。"),)))
    assert "setup_payoff.chronology.contradiction" in _codes(replace(skeleton, setup_payoffs=(SetupPayoffLink("reverse", third.adapted_story_beat_id, first.adapted_story_beat_id, (), "反序。"),)))
    assert "gap.anchor" in _codes(replace(skeleton, gaps=(SkeletonGap("bad", "knowledge", "intentional_ambiguity", "保留。", "记录。", (GapAnchor("knowledge", "not_a_state_id"),)),)))
    background = replace(
        skeleton,
        primary_presentation=skeleton.primary_presentation[1:],
        beat_accounting=tuple(
            BeatAccounting(item.adapted_story_beat_id, "background_only", "产生后续后果。")
            if item.adapted_story_beat_id == first.adapted_story_beat_id else item
            for item in skeleton.beat_accounting
        ),
    )
    assert validate_adapted_story_skeleton(background, authority=_authority()).valid
    orphan = replace(
        skeleton, primary_presentation=skeleton.primary_presentation[:2], setup_payoffs=(), arc_checkpoints=(),
        beat_accounting=(*skeleton.beat_accounting[:-1], BeatAccounting(third.adapted_story_beat_id, "background_only", "没有作用。")),
    )
    assert "accounting.orphan" in _codes(orphan)
    assert "accounting.missing" in _codes(replace(skeleton, beat_accounting=skeleton.beat_accounting[:-1]))


def test_accounting_requires_consequential_evidence_not_an_unconsumed_outcome():
    skeleton = _skeleton()
    first, _second, _third = _authority().disposition_plan.adapted_story_beats
    unconsumed = replace(
        skeleton, primary_presentation=skeleton.primary_presentation[1:], dependencies=(), setup_payoffs=(),
        beat_accounting=tuple(
            BeatAccounting(item.adapted_story_beat_id, "background_only", "误称背景贡献。")
            if item.adapted_story_beat_id == first.adapted_story_beat_id else item
            for item in skeleton.beat_accounting
        ),
    )
    assert "accounting.orphan" in _codes(unconsumed)
    dependency_path = replace(
        unconsumed, dependencies=_skeleton().dependencies,
        beat_accounting=tuple(
            BeatAccounting(item.adapted_story_beat_id, "indirectly_revealed", "通过后果间接揭示。")
            if item.adapted_story_beat_id == first.adapted_story_beat_id else item
            for item in unconsumed.beat_accounting
        ),
    )
    assert validate_adapted_story_skeleton(dependency_path, authority=_authority()).valid
    orphan_indirect = replace(dependency_path, dependencies=())
    assert "accounting.indirect_evidence" in _codes(orphan_indirect)


def test_background_only_cannot_also_have_direct_primary_presentation():
    skeleton = _skeleton()
    first = _authority().disposition_plan.adapted_story_beats[0]
    contradictory = replace(
        skeleton,
        beat_accounting=tuple(
            BeatAccounting(item.adapted_story_beat_id, "background_only", "与直接 presentation 冲突。")
            if item.adapted_story_beat_id == first.adapted_story_beat_id else item
            for item in skeleton.beat_accounting
        ),
    )
    assert "accounting.background_presented" in _codes(contradictory)


def test_background_knowledge_progression_reaches_presented_confirmation():
    skeleton = _skeleton()
    first, second, _third = _authority().disposition_plan.adapted_story_beats
    character = _authority().disposition_plan.adapted_characters[0].adapted_character_id
    knowledge_path = replace(
        skeleton,
        primary_presentation=skeleton.primary_presentation[1:],
        outcomes=(), dependencies=(), setup_payoffs=(),
        character_knowledge_transitions=(
            CharacterKnowledgeTransition("suspect_at_a", character, "secret", "unknown", "suspected", first.adapted_story_beat_id),
            CharacterKnowledgeTransition("confirm_at_b", character, "secret", "suspected", "confirmed", second.adapted_story_beat_id),
        ),
        beat_accounting=tuple(
            BeatAccounting(item.adapted_story_beat_id, "background_only", "为角色确认提供知识前提。")
            if item.adapted_story_beat_id == first.adapted_story_beat_id else item
            for item in skeleton.beat_accounting
        ),
    )
    assert validate_adapted_story_skeleton(knowledge_path, authority=_authority()).valid
    unused_transition = replace(
        knowledge_path,
        character_knowledge_transitions=knowledge_path.character_knowledge_transitions[:1],
    )
    assert "accounting.orphan" in _codes(unused_transition)


def test_self_dependency_is_not_a_meaningful_same_beat_precondition():
    skeleton = _skeleton()
    first = _authority().disposition_plan.adapted_story_beats[0]
    self_dependent = replace(
        skeleton,
        dependencies=(NarrativeDependency("self_dep", "outcome_start", first.adapted_story_beat_id, "同一 beat 不构成前提。"),),
    )
    assert "dependency.self" in _codes(self_dependent)


def test_character_specific_withholding_is_explicitly_unverifiable():
    skeleton = _skeleton()
    character = _authority().disposition_plan.adapted_characters[0].adapted_character_id
    boundary = WithholdingBoundary("character_hold", "secret", "character", character, skeleton.primary_presentation[1], "当前结构未提供角色披露时点。")
    assert "withholding.character_unverifiable" in _codes(replace(skeleton, withholding_boundaries=(*skeleton.withholding_boundaries, boundary)))


def test_local_invented_identity_conflict_and_exact_authority_fail_closed(tmp_path):
    skeleton = _skeleton()
    first = _authority().disposition_plan.adapted_story_beats[0]
    local = AdaptedStoryBeat(first.identity_key, "本地重用。", "adaptation_invented", "compatible", (), (), (CanonicalTargetReference("fact", "fact_000001"),), "bridge", "不允许。", "B1.4。")
    # A forged local ID cannot silently replace B1.3's adapted identity.
    object.__setattr__(local, "adapted_story_beat_id", first.adapted_story_beat_id)
    assert "beat.local_identity_conflict" in _codes(replace(skeleton, local_invented_beats=(local,)))
    # Exact read-only resolution rejects fake/missing/tampered pinned refs rather than accepting a caller beat set.
    from short_drama.artifacts import FileArtifactStore
    from short_drama.foundation import FilePointerStore
    store = FileArtifactStore(tmp_path / "artifacts")
    with pytest.raises(Exception, match="authority"):
        resolve_b14_skeleton_authority(skeleton, store=store, pointers=FilePointerStore(tmp_path / "pointers", store))


def test_exact_authority_uses_pinned_publication_and_strategy_not_current(tmp_path):
    """Exercise B1.3's existing historical verifier with no provider calls."""
    from test_b13_disposition_persistence import _prepared, _publication, _publish_current
    from short_drama.adaptation import verify_b13_disposition_publication

    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    publication_ref, _pointer_ref, _pointer_id = _publish_current(tree, preparation, requests)
    plan = preparation.strategy_selection
    published_plan = verify_b13_disposition_publication(_publication(preparation, requests)).plan
    candidate = AdaptedStorySkeleton(
        plan.strategy_ref, publication_ref,
        primary_presentation=tuple(item.adapted_story_beat_id for item in published_plan.adapted_story_beats),
        beat_accounting=tuple(BeatAccounting(item.adapted_story_beat_id, "directly_presented", "真实 authority 演练。") for item in published_plan.adapted_story_beats),
    )
    verified_plan = resolve_b14_skeleton_authority(
        candidate, store=tree.store, pointers=tree.pointers,
    )
    assert verified_plan.strategy_ref == plan.strategy_ref
    skeleton = AdaptedStorySkeleton(plan.strategy_ref, publication_ref)
    with pytest.raises(Exception, match="authority"):
        resolve_b14_skeleton_authority(replace(skeleton, strategy_ref=_ref("adaptation_strategy", "wrong.strategy.c")), store=tree.store, pointers=tree.pointers)
    with pytest.raises(Exception, match="authority"):
        resolve_b14_skeleton_authority(replace(skeleton, disposition_publication_ref=_ref("b13_disposition_publication", "missing.publication.d")), store=tree.store, pointers=tree.pointers)
    with pytest.raises(Exception, match="authority"):
        resolve_b14_skeleton_authority(replace(skeleton, disposition_publication_ref=replace(publication_ref, content_hash="0" * 64)), store=tree.store, pointers=tree.pointers)
    with pytest.raises(Exception, match="authority"):
        resolve_b14_skeleton_authority(replace(skeleton, disposition_publication_ref=replace(publication_ref, revision=publication_ref.revision + 1)), store=tree.store, pointers=tree.pointers)
