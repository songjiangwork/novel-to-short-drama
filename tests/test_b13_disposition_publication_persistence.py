"""B1.3-B4B2 immutable publication and stage-local CURRENT CAS tests."""
from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.adaptation import (
    B13CandidateKind, B13CandidateRef,
    B13DispositionExecutionResult, B13DispositionPersistenceError,
    B13DispositionPersistenceService, B13ReconciliationDecision,
    B13ReconciliationDecisionKind, B13ReconciliationDecisionSet,
    build_b13_reconciliation_inventory, build_b13_window_semantic_preparation,
    prepare_b13_disposition_publication, prepare_b13_disposition_windows,
    review_b13_reconciliation,
)
from short_drama.adaptation.disposition_assembly import assemble_b13_disposition_plan
from short_drama.adaptation.disposition_publication import (
    B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
    B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
    _semantic_identity,
    b13_disposition_publication_artifact_id,
)
from short_drama.artifacts import ArtifactNotFoundError, ImmutableArtifactEnvelope
from short_drama.foundation import PointerKind, PointerNotFoundError

from test_b13_disposition_persistence import _bytes, _prepared, _publication, _publish_a6_v2
from test_b13_disposition_reconciliation import (
    _accepted, _character, _source_character, _two_window_preparation,
)
from test_b13_window_semantic import _proposal


def _changed_publication(preparation, requests):
    """A valid completed B4A result with unchanged request identities."""
    value = _proposal(requests[0].packet)
    value["event_dispositions"][0]["rationale_zh"] = "改编审阅后保留，但采用不同的结构理由。"
    execution = B13DispositionExecutionResult(
        preparation.source_snapshot, preparation.strategy_selection,
        preparation.preparation_identity,
        (_accepted(requests[0], value),),
    )
    inventory = build_b13_reconciliation_inventory(execution)
    review = review_b13_reconciliation(
        inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()),
    )
    return prepare_b13_disposition_publication(
        review, assemble_b13_disposition_plan(review), execution,
    )


def _pointer_id(tree, preparation):
    return "{}.b13.{}.disposition".format(
        tree.store.get_ref(preparation.source_snapshot.story_analysis_manifest_ref).payload["project_id"],
        preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id,
    )


def _real_two_window_preparation(tmp_path, monkeypatch):
    """Produce a real A5/A6/B1.2 authority chain with two A6 windows."""
    import test_story_a5f2_current_publication as a5_fixture
    from short_drama.adaptation import (
        AdaptationSourceSnapshot, AdaptationTarget, build_b12_semantic_preparation,
        execute_b12_strategy, prepare_b12_packet, resolve_adaptation_target,
    )
    from short_drama.llm import PromptRegistry, load_semantic_profile
    from short_drama.paths import REPO_ROOT
    from short_drama.story import (
        ArcAnalysis, CharacterAnalysis, CharacterAnalysisSet,
        EvidenceBackedInterpretation, GlobalEventAnalysis, GlobalStoryBible,
        GlobalStructure, PlotWindowAnalysis, StoryAnalysisPersistenceService,
        EventSemanticDecision,
        build_story_analysis_plan_from_profile, build_story_analysis_snapshot,
        load_story_analysis_profile,
    )
    from test_adaptation_strategy import _b12_test_runtime_config
    from test_adaptation_strategy_execution import _FakeClient, _budget, _valid_proposal

    original_candidates = a5_fixture._default_candidates

    def two_event_candidates(suffix="001"):
        chars, locations, unresolved, facts, events, relationships = original_candidates(suffix)
        extra = a5_fixture.EventCandidate(
            candidate_id=f"cand_evt_{suffix}2", summary_zh="另一件独立事件。",
            participant_refs=(), location_refs=(), temporal_mode="normal",
            evidence_strength="explicit", evidence=(a5_fixture._evidence("CH001_P0002"),),
        )
        return chars, locations, unresolved, facts, (*events, extra), relationships

    monkeypatch.setattr(a5_fixture, "_default_candidates", two_event_candidates)
    tree, _ext, _a3, _index = a5_fixture._build_tree_with_a4(tmp_path)
    a5_planning = a5_fixture._planning(tree)
    facts, events, relationships = a5_fixture._make_resolutions(
        a5_planning, tree.consolidation_profile,
    )
    pair = a5_planning.event_pair_plans[0]
    event_decision = EventSemanticDecision(
        decision_id="evtd_000099", left_candidate_ref=pair.left_ref,
        right_candidate_ref=pair.right_ref, decision="different_event", method="manual",
        reason_zh="两个独立事件保留为不同 canonical event。", evidence_refs=(),
        prompt_id=None, prompt_version=None, generation_provenance=None,
    )
    events = replace(events, semantic_decisions=(event_decision,), all_event_decisions=(event_decision,))
    finalization = a5_fixture.finalize_consolidation(a5_planning, facts, events, relationships)
    a5_fixture._publish_a5(
        tree, a5_planning, fact_resolution=facts, event_resolution=events,
        relationship_resolution=relationships, finalization_result=finalization,
    )
    snapshot = build_story_analysis_snapshot(
        tree.store, tree.pointers, project_id=a5_fixture.PROJECT,
        document_id=a5_fixture.DOCUMENT,
        consolidation_profile_id=a5_fixture.CONSOLIDATION_PROFILE_ID,
    )
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml")
    profile = replace(
        profile,
        planning_policy=replace(profile.planning_policy, plot_window_owned_event_target=1),
    )
    plan = build_story_analysis_plan_from_profile(snapshot, profile)

    def interpretation():
        return EvidenceBackedInterpretation(text_zh="测试。", evidence_mode="explicit")

    characters = CharacterAnalysisSet(1, tuple(
        CharacterAnalysis(package.character_ref, interpretation(), (), (), (), (), (), (), interpretation(), ())
        for package in plan.character_packages
    ))
    events = GlobalEventAnalysis(
        1, plan.plan_hash,
        tuple(PlotWindowAnalysis(
            window.window_id, window.window_ordinal, window.owned_event_ids,
            window.context_event_ids, interpretation(),
        ) for window in plan.windows),
        (), (("canonical_event_count", len(plan.event_stream)),
             ("event_importance_count", 0), ("plot_window_count", len(plan.windows))),
    )
    a6 = StoryAnalysisPersistenceService(tree.store, tree.pointers).publish_validated(
        project_id=snapshot.consolidation_manifest.project_id,
        document_id=snapshot.consolidation_manifest.document_id, profile=profile,
        semantic_profile=load_semantic_profile(REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml"),
        prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"), snapshot=snapshot, plan=plan,
        characters=characters, events=events, arcs=ArcAnalysis(1, (), (), (), ()),
        structure=GlobalStructure(1, interpretation(), (), interpretation(), (), interpretation(), (), (), (), (), ()),
        bible=GlobalStoryBible(1, interpretation(), interpretation(), "测试", "测试", interpretation(), (), (), (), (), (), interpretation(), interpretation(), (("canonical_character_count", len(characters.analyses)), ("canonical_event_count", len(plan.event_stream)))),
    )
    source = AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, a6.story_analysis_manifest_ref)
    target = resolve_adaptation_target(AdaptationTarget("two-window-publication"), source_snapshot=source)
    b12 = build_b12_semantic_preparation(
        prepare_b12_packet(tree.store, tree.pointers, target), runtime_config=_b12_test_runtime_config(),
    )
    execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=b12, runtime_config=b12.runtime_config,
        budget=_budget(b12), llm_client=_FakeClient([(_valid_proposal(b12), False)], b12.runtime_config),
    )
    preparation = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        adaptation_target_id=target.adaptation_target_id,
    )
    requests = tuple(build_b13_window_semantic_preparation(
        packet, runtime_config=_b12_test_runtime_config(),
    ) for packet in preparation.packets)
    assert len(preparation.packets) == 2
    return tree, preparation, requests


def _two_window_publication(preparation, requests, decision_kind):
    source_character = next(
        reference for reference in preparation.source_snapshot.canonical_references
        if reference.kind == "character"
    )
    values = [_proposal(request.packet) for request in requests]
    values[0]["adapted_characters"] = [_character("lead-a", source_character)]
    values[1]["adapted_characters"] = [_character("lead-b", source_character)]
    values[1]["adapted_characters"][0]["display_name_zh"] = values[0]["adapted_characters"][0]["display_name_zh"]
    execution = B13DispositionExecutionResult(
        preparation.source_snapshot, preparation.strategy_selection,
        preparation.preparation_identity,
        tuple(_accepted(request, value) for request, value in zip(requests, values)),
    )
    inventory = build_b13_reconciliation_inventory(execution)
    decision = B13ReconciliationDecision(
        decision_kind,
        (
            B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_CHARACTER, "lead-a"),
            B13CandidateRef("window_0002", B13CandidateKind.ADAPTED_CHARACTER, "lead-b"),
        ), (), "明确的跨窗口身份决策。", "B4B2 端到端测试。",
    )
    review = review_b13_reconciliation(
        inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, (decision,)),
    )
    return prepare_b13_disposition_publication(
        review, assemble_b13_disposition_plan(review), execution,
    )


def test_initial_publication_then_final_semantic_reuse_is_zero_write(tmp_path):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    publication = _publication(preparation, requests)

    first = service.publish_validated(
        publication=publication, preparation=preparation, semantic_preparations=requests,
    )
    assert not first.reused
    assert first.publication_ref.revision == 1
    assert tree.pointers.resolve_current_pointer_ref(_pointer_id(tree, preparation)) == first.current_pointer_ref

    before = (_bytes(tree.store.root), _bytes(tree.pointers.root))
    reused = service.publish_validated(
        publication=publication, preparation=preparation, semantic_preparations=requests,
    )
    assert reused.reused
    assert reused.publication_ref == first.publication_ref
    assert reused.current_pointer_ref == first.current_pointer_ref
    assert (_bytes(tree.store.root), _bytes(tree.pointers.root)) == before


def test_changed_final_proposal_is_not_discarded_as_request_cache_hit(tmp_path):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    first = service.publish_validated(
        publication=_publication(preparation, requests), preparation=preparation,
        semantic_preparations=requests,
    )
    changed = _changed_publication(preparation, requests)
    assert changed.semantic_identity != first.publication.semantic_identity

    second = service.publish_validated(
        publication=changed, preparation=preparation, semantic_preparations=requests,
    )
    assert not second.reused
    assert second.publication_ref.revision == first.publication_ref.revision + 1
    assert second.plan != first.plan


def test_changed_reconciliation_decisions_change_completed_semantic_identity(tmp_path):
    """B4B2's final reuse discriminator includes explicit B3 decisions."""
    preparation, runtime = _two_window_preparation(tmp_path)
    requests = tuple(build_b13_window_semantic_preparation(packet, runtime_config=runtime)
                     for packet in preparation.packets)
    source_character = _source_character(B13DispositionExecutionResult(
        preparation.source_snapshot, preparation.strategy_selection,
        preparation.preparation_identity,
        tuple(_accepted(request, _proposal(request.packet)) for request in requests),
    ))
    values = [_proposal(request.packet) for request in requests]
    values[0]["adapted_characters"] = [_character("lead-a", source_character)]
    values[1]["adapted_characters"] = [_character("lead-b", source_character)]
    values[1]["adapted_characters"][0]["display_name_zh"] = values[0]["adapted_characters"][0]["display_name_zh"]
    execution = B13DispositionExecutionResult(
        preparation.source_snapshot, preparation.strategy_selection,
        preparation.preparation_identity,
        tuple(_accepted(request, value) for request, value in zip(requests, values)),
    )
    inventory = build_b13_reconciliation_inventory(execution)
    refs = (
        B13CandidateRef("window_0001", B13CandidateKind.ADAPTED_CHARACTER, "lead-a"),
        B13CandidateRef("window_0002", B13CandidateKind.ADAPTED_CHARACTER, "lead-b"),
    )
    same = B13ReconciliationDecision(
        B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT, refs, (), "确认同一角色。", "复核。",
    )
    distinct = B13ReconciliationDecision(
        B13ReconciliationDecisionKind.KEEP_DISTINCT, refs, (), "确认不同角色。", "复核。",
    )
    same_set = B13ReconciliationDecisionSet(inventory.inventory_identity, (same,))
    distinct_set = B13ReconciliationDecisionSet(inventory.inventory_identity, (distinct,))
    assert _semantic_identity(
        execution.source_snapshot, execution.strategy_selection, execution, same_set,
    ) != _semantic_identity(
        execution.source_snapshot, execution.strategy_selection, execution, distinct_set,
    )


def test_reconciliation_decision_variation_publishes_new_final_revision(tmp_path, monkeypatch):
    tree, preparation, requests = _real_two_window_preparation(tmp_path, monkeypatch)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    first_publication = _two_window_publication(
        preparation, requests, B13ReconciliationDecisionKind.IDENTIFY_EQUIVALENT,
    )
    second_publication = _two_window_publication(
        preparation, requests, B13ReconciliationDecisionKind.KEEP_DISTINCT,
    )
    assert first_publication.semantic_identity != second_publication.semantic_identity
    first = service.publish_validated(
        publication=first_publication, preparation=preparation, semantic_preparations=requests,
    )
    second = service.publish_validated(
        publication=second_publication, preparation=preparation, semantic_preparations=requests,
    )
    assert not second.reused
    assert second.publication_ref.revision == first.publication_ref.revision + 1
    assert second.current_pointer_ref.revision == first.current_pointer_ref.revision + 1
    persisted = tree.store.get_ref(second.publication_ref).payload
    assert persisted["semantic_identity"] == second_publication.semantic_identity
    current = service._resolve_verified_current(_pointer_id(tree, preparation))
    assert current is not None
    assert current.publication_ref == second.publication_ref
    assert current.publication.semantic_identity == second_publication.semantic_identity
    assert current.plan == second.plan


def test_runtime_only_provenance_difference_reuses_same_final_semantics(tmp_path):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    first = service.publish_validated(
        publication=_publication(preparation, requests), preparation=preparation,
        semantic_preparations=requests,
    )
    from test_adaptation_strategy import _b12_test_runtime_config
    from short_drama.adaptation import build_b13_window_semantic_preparation

    varied_requests = tuple(build_b13_window_semantic_preparation(
        packet,
        runtime_config=replace(
            _b12_test_runtime_config(), provider_family="alternate-route", request_model="alternate-model",
        ),
    ) for packet in preparation.packets)
    varied_publication = _publication(preparation, varied_requests)
    assert varied_publication.semantic_identity == first.publication.semantic_identity
    reused = service.publish_validated(
        publication=varied_publication, preparation=preparation,
        semantic_preparations=varied_requests,
    )
    assert reused.reused and reused.publication_ref == first.publication_ref


def test_occupied_orphan_slot_is_skipped_and_corrupt_slot_fails_closed(tmp_path):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    first_publication = _publication(preparation, requests)
    first = service.publish_validated(
        publication=first_publication, preparation=preparation, semantic_preparations=requests,
    )
    artifact_id = first.publication_ref.artifact_id
    tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, artifact_id=artifact_id,
        revision=2, schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=first_publication.to_dict(),
    ))
    published = service.publish_validated(
        publication=_changed_publication(preparation, requests), preparation=preparation,
        semantic_preparations=requests,
    )
    assert published.publication_ref.revision == 3

    # The next candidate encounters an occupied non-canonical slot: allocation
    # must fail rather than overwrite it or search old publications.
    corrupt = tree.store.root / B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE / artifact_id / "r00000004.json"
    corrupt.write_bytes(b"not canonical json")
    with pytest.raises(B13DispositionPersistenceError, match="allocate immutable"):
        service.publish_validated(
            publication=_publication(preparation, requests), preparation=preparation,
            semantic_preparations=requests,
        )


@pytest.mark.parametrize("kind", ("unsupported_schema", "malformed_b4a"))
def test_semantically_invalid_occupied_envelope_fails_before_new_publication(tmp_path, kind):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    first = service.publish_validated(
        publication=_publication(preparation, requests), preparation=preparation,
        semantic_preparations=requests,
    )
    payload = first.publication.to_dict() if kind == "unsupported_schema" else {}
    schema_version = (
        B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION + 1
        if kind == "unsupported_schema" else B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION
    )
    tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        artifact_id=first.publication_ref.artifact_id, revision=2,
        schema_version=schema_version, payload=payload,
    ))

    with pytest.raises(B13DispositionPersistenceError, match="occupied immutable.*semantic integrity"):
        service.publish_validated(
            publication=_changed_publication(preparation, requests), preparation=preparation,
            semantic_preparations=requests,
        )
    assert tree.pointers.resolve_current_pointer_ref(_pointer_id(tree, preparation)) == first.current_pointer_ref
    with pytest.raises(ArtifactNotFoundError):
        tree.store.get(B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, first.publication_ref.artifact_id, 3)


def test_historically_unverifiable_occupied_orphan_fails_before_new_publication(tmp_path):
    tree, _semantic, strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    first = service.publish_validated(
        publication=_publication(preparation, requests), preparation=preparation,
        semantic_preparations=requests,
    )
    historical_ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=strategy.strategy_ref.artifact_type,
        artifact_id=strategy.strategy_ref.artifact_id,
        revision=strategy.strategy_ref.revision + 1, schema_version=1,
        payload=strategy.artifact.to_dict(),
    ))
    orphan_preparation = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers, source_snapshot=preparation.source_snapshot,
        adaptation_target_id=preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id,
        strategy_ref=historical_ref,
    )
    orphan_requests = tuple(build_b13_window_semantic_preparation(
        packet, runtime_config=requests[0].runtime_config,
    ) for packet in orphan_preparation.packets)
    tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        artifact_id=first.publication_ref.artifact_id, revision=2,
        schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=_publication(orphan_preparation, orphan_requests).to_dict(),
    ))
    (tree.store.root / historical_ref.artifact_type / historical_ref.artifact_id
     / f"r{historical_ref.revision:08d}.json").unlink()

    with pytest.raises(B13DispositionPersistenceError, match="occupied immutable.*semantic integrity"):
        service.publish_validated(
            publication=_changed_publication(preparation, requests), preparation=preparation,
            semantic_preparations=requests,
        )
    assert tree.pointers.resolve_current_pointer_ref(_pointer_id(tree, preparation)) == first.current_pointer_ref
    with pytest.raises(ArtifactNotFoundError):
        tree.store.get(B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, first.publication_ref.artifact_id, 3)


def test_competing_cas_leaves_orphan_and_never_returns_stale_success(tmp_path, monkeypatch):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    first = service.publish_validated(
        publication=_publication(preparation, requests), preparation=preparation,
        semantic_preparations=requests,
    )
    candidate = _changed_publication(preparation, requests)
    artifact_id = first.publication_ref.artifact_id
    competitor = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, artifact_id=artifact_id,
        revision=3, schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=candidate.to_dict(),
    ))
    original = tree.pointers.compare_and_set
    raced = {"done": False}

    def compete(**kwargs):
        if not raced["done"]:
            raced["done"] = True
            original(
                pointer_id=kwargs["pointer_id"], pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=kwargs["expected_pointer_ref"], target_ref=competitor,
            )
        return original(**kwargs)

    monkeypatch.setattr(tree.pointers, "compare_and_set", compete)
    with pytest.raises(B13DispositionPersistenceError, match="CAS failed"):
        service.publish_validated(
            publication=candidate, preparation=preparation, semantic_preparations=requests,
        )
    assert tree.pointers.resolve_current(_pointer_id(tree, preparation)).target_ref == competitor
    # r2 was written before CAS and remains a harmless immutable orphan.
    assert tree.store.get(B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, artifact_id, 2)


def test_current_pinned_b12_advance_after_immutable_write_fails_closed(tmp_path, monkeypatch):
    tree, _semantic, strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    original_put = tree.store.put
    advanced = {"done": False}

    def advance_b12_after_write(envelope):
        ref = original_put(envelope)
        if (not advanced["done"]
                and envelope.artifact_type == B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE):
            advanced["done"] = True
            next_ref = original_put(ImmutableArtifactEnvelope.create(
                artifact_type=strategy.strategy_ref.artifact_type,
                artifact_id=strategy.strategy_ref.artifact_id,
                revision=strategy.strategy_ref.revision + 1,
                schema_version=1, payload=strategy.artifact.to_dict(),
            ))
            tree.pointers.compare_and_set(
                pointer_id=strategy.current_pointer_ref.artifact_id,
                pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=strategy.current_pointer_ref, target_ref=next_ref,
            )
        return ref

    monkeypatch.setattr(tree.store, "put", advance_b12_after_write)
    with pytest.raises(B13DispositionPersistenceError, match="exact live A5/A6/B1.2 authority"):
        service.publish_validated(
            publication=_publication(preparation, requests), preparation=preparation,
            semantic_preparations=requests,
        )
    artifact_id = b13_disposition_publication_artifact_id(
        preparation.source_snapshot, preparation.strategy_selection.artifact,
    )
    assert tree.store.get(B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, artifact_id, 1)


def test_a5_a6_authority_race_after_immutable_write_leaves_orphan(tmp_path, monkeypatch):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    original = service._request_current_authority
    calls = {"count": 0}

    def advance_after_write(value):
        calls["count"] += 1
        if calls["count"] == 2:
            # This is the publication race hook: the actual second authority
            # observation is no longer the A5/A6 snapshot used by B4A.
            raise B13DispositionPersistenceError("new B1.3 request no longer has exact live A5/A6/B1.2 authority")
        return original(value)

    monkeypatch.setattr(service, "_request_current_authority", advance_after_write)
    with pytest.raises(B13DispositionPersistenceError, match="exact live A5/A6/B1.2 authority"):
        service.publish_validated(
            publication=_publication(preparation, requests), preparation=preparation,
            semantic_preparations=requests,
        )
    artifact_id = b13_disposition_publication_artifact_id(
        preparation.source_snapshot, preparation.strategy_selection.artifact,
    )
    assert tree.store.get(B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, artifact_id, 1)
    with pytest.raises(PointerNotFoundError):
        tree.pointers.resolve_current_pointer_ref(_pointer_id(tree, preparation))


def test_pointer_advance_after_cas_is_detected_before_success(tmp_path, monkeypatch):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    candidate = _publication(preparation, requests)
    competitor_payload = _changed_publication(preparation, requests)
    artifact_id = b13_disposition_publication_artifact_id(
        preparation.source_snapshot, preparation.strategy_selection.artifact,
    )
    competitor = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE, artifact_id=artifact_id,
        revision=2, schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=competitor_payload.to_dict(),
    ))
    original = tree.pointers.compare_and_set

    def advance_after_cas(**kwargs):
        pointer_ref = original(**kwargs)
        original(
            pointer_id=kwargs["pointer_id"], pointer_kind=PointerKind.CURRENT,
            expected_pointer_ref=pointer_ref, target_ref=competitor,
        )
        return pointer_ref

    monkeypatch.setattr(tree.pointers, "compare_and_set", advance_after_cas)
    with pytest.raises(B13DispositionPersistenceError, match="changed after publication CAS"):
        service.publish_validated(
            publication=candidate, preparation=preparation, semantic_preparations=requests,
        )
    assert tree.pointers.resolve_current(_pointer_id(tree, preparation)).target_ref == competitor


def test_a6_artifact_change_keeps_stage_local_pointer_and_monotonic_revision(tmp_path):
    from short_drama.adaptation import (
        AdaptationSourceSnapshot, AdaptationTarget, build_b12_semantic_preparation,
        build_b13_window_semantic_preparation, execute_b12_strategy,
        prepare_b12_packet, prepare_b13_disposition_windows, resolve_adaptation_target,
    )
    from test_adaptation_strategy import _b12_test_runtime_config
    from test_adaptation_strategy_execution import _FakeClient, _budget, _valid_proposal

    tree, _semantic, _strategy, old_preparation, old_requests = _prepared(tmp_path)
    service = B13DispositionPersistenceService(tree.store, tree.pointers)
    old = service.publish_validated(
        publication=_publication(old_preparation, old_requests), preparation=old_preparation,
        semantic_preparations=old_requests,
    )
    a6_v2 = _publish_a6_v2(tree, old_preparation.source_snapshot)
    source = AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, a6_v2.story_analysis_manifest_ref)
    target = resolve_adaptation_target(AdaptationTarget("semantic-preparation"), source_snapshot=source)
    b12 = build_b12_semantic_preparation(
        prepare_b12_packet(tree.store, tree.pointers, target), runtime_config=_b12_test_runtime_config(),
    )
    execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=b12, runtime_config=b12.runtime_config,
        budget=_budget(b12), llm_client=_FakeClient([(_valid_proposal(b12), False)], b12.runtime_config),
    )
    fresh = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        adaptation_target_id=target.adaptation_target_id,
    )
    fresh_requests = tuple(build_b13_window_semantic_preparation(
        packet, runtime_config=_b12_test_runtime_config(),
    ) for packet in fresh.packets)
    new = service.publish_validated(
        publication=_publication(fresh, fresh_requests), preparation=fresh,
        semantic_preparations=fresh_requests,
    )
    assert old.publication_ref.artifact_id != new.publication_ref.artifact_id
    assert new.publication_ref.revision == old.publication_ref.revision + 1
    assert old.current_pointer_ref.artifact_id == new.current_pointer_ref.artifact_id
