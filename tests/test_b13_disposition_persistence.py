"""B1.3-B4B1 is a read-only, zero-provider CURRENT reuse boundary."""
from __future__ import annotations

from pathlib import Path

import pytest

from short_drama.adaptation import (
    B13DispositionExecutionResult, B13DispositionPersistenceError,
    B13DispositionPersistenceService, B13DispositionPacketMeasurement,
    B13DispositionPreparation, B13DispositionWindowPacket, B13ReconciliationDecisionSet,
    b13_disposition_pointer_id, build_b13_reconciliation_inventory,
    build_b13_window_semantic_preparation, prepare_b13_disposition_publication,
    prepare_b13_disposition_windows, review_b13_reconciliation,
)
from short_drama.adaptation.disposition_assembly import assemble_b13_disposition_plan
from short_drama.adaptation.disposition_publication import (
    B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
    B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION, _publication_identity,
    b13_disposition_publication_artifact_id,
)
from short_drama.artifacts import ImmutableArtifactEnvelope, canonical_json_bytes, content_hash
from short_drama.foundation import PointerKind
from short_drama.llm import LLMClient
from short_drama.story.chunking import estimate_tokens

from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_preparation import _published_strategy
from test_b13_disposition_reconciliation import _accepted
from test_b13_window_semantic import _proposal


def _bytes(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _prepared(tmp_path, *, strategy_ref=None):
    tree, semantic, strategy = _published_strategy(tmp_path)
    preparation = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
        strategy_ref=strategy_ref,
    )
    requests = tuple(
        build_b13_window_semantic_preparation(packet, runtime_config=_b12_test_runtime_config())
        for packet in preparation.packets
    )
    return tree, semantic, strategy, preparation, requests


def _publication(preparation, requests):
    execution = B13DispositionExecutionResult(
        preparation.source_snapshot, preparation.strategy_selection,
        preparation.preparation_identity,
        tuple(_accepted(request, _proposal(request.packet)) for request in requests),
    )
    inventory = build_b13_reconciliation_inventory(execution)
    review = review_b13_reconciliation(
        inventory, B13ReconciliationDecisionSet(inventory.inventory_identity, ()),
    )
    return prepare_b13_disposition_publication(
        review, assemble_b13_disposition_plan(review), execution,
    )


def _publish_current(tree, preparation, requests, *, revision=1):
    publication = _publication(preparation, requests)
    artifact_id = b13_disposition_publication_artifact_id(
        preparation.source_snapshot, preparation.strategy_selection.artifact,
    )
    ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        artifact_id=artifact_id, revision=revision,
        schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=publication.to_dict(),
    ))
    project_id = tree.store.get_ref(preparation.source_snapshot.story_analysis_manifest_ref).payload["project_id"]
    pointer_id = b13_disposition_pointer_id(
        project_id, preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id,
    )
    old = None
    try:
        old = tree.pointers.resolve_current_pointer_ref(pointer_id)
    except Exception:
        pass
    pointer_ref = tree.pointers.compare_and_set(
        pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=old, target_ref=ref,
    )
    return ref, pointer_ref, pointer_id


def _replace_current_payload(tree, preparation, requests, pointer_ref, pointer_id, payload):
    payload["publication_identity"] = _publication_identity(payload)
    ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        artifact_id=b13_disposition_publication_artifact_id(
            preparation.source_snapshot, preparation.strategy_selection.artifact,
        ),
        revision=pointer_ref.revision + 1,
        schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION, payload=payload,
    ))
    tree.pointers.compare_and_set(
        pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=pointer_ref, target_ref=ref,
    )


def _altered_event_packet_preparation(preparation):
    """Create internally coherent evidence that disagrees with real A5 bytes."""
    packets = list(preparation.packets)
    payload = packets[0].to_dict()
    payload["owned_source_events"][0]["summary_zh"] = "伪造但结构有效的 source event。"
    packets[0] = B13DispositionWindowPacket(
        preparation.source_snapshot, preparation.strategy_selection,
        packets[0].window_id, packets[0].window_ordinal, canonical_json_bytes(payload),
    )
    packets = tuple(packets)
    measurements = tuple(B13DispositionPacketMeasurement(
        packet.window_id, len(packet.canonical_bytes()),
        estimate_tokens(packet.canonical_bytes().decode("utf-8")), packet.content_hash(),
    ) for packet in packets)
    identity = content_hash({
        "stage": "B1.3-B1", "source_snapshot": preparation.source_snapshot.to_dict(),
        "strategy_ref": preparation.strategy_selection.strategy_ref.to_dict(),
        "strategy_hash": preparation.strategy_selection.strategy_hash,
        "packet_hashes": [packet.content_hash() for packet in packets],
    })
    return B13DispositionPreparation(
        preparation.source_snapshot, preparation.strategy_selection, packets, measurements,
        preparation.canonical_event_ids, identity,
    )


def _publish_a6_v2(tree, old_source):
    """Advance only A6 profile authority with valid persisted v2 artifacts."""
    from short_drama.llm import PromptRegistry, load_semantic_profile
    from short_drama.paths import REPO_ROOT
    from short_drama.story import (
        ArcAnalysis, CharacterAnalysis, CharacterAnalysisSet, EvidenceBackedInterpretation,
        GlobalEventAnalysis, GlobalStoryBible, GlobalStructure, PlotWindowAnalysis,
        StoryAnalysisPersistenceService, build_story_analysis_plan_from_profile,
        build_story_analysis_snapshot, load_story_analysis_profile,
    )
    from short_drama.story.consolidation import ConsolidationManifest

    old_manifest = ConsolidationManifest.from_dict(
        tree.store.get_ref(old_source.consolidation_manifest_ref).payload
    )
    snapshot = build_story_analysis_snapshot(
        tree.store, tree.pointers, project_id=old_manifest.project_id,
        document_id=old_manifest.document_id,
        consolidation_profile_id=old_manifest.semantic_identity.consolidation_profile_id,
    )
    profile = load_story_analysis_profile(REPO_ROOT / "profiles" / "global_story_analysis_v2.yaml")
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
        (), (("canonical_event_count", len(plan.event_stream)), ("event_importance_count", 0),
             ("plot_window_count", len(plan.windows))),
    )
    return StoryAnalysisPersistenceService(tree.store, tree.pointers).publish_validated(
        project_id=snapshot.consolidation_manifest.project_id,
        document_id=snapshot.consolidation_manifest.document_id, profile=profile,
        semantic_profile=load_semantic_profile(REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml"),
        prompts=PromptRegistry(REPO_ROOT / "prompts" / "story"), snapshot=snapshot, plan=plan,
        characters=characters, events=events, arcs=ArcAnalysis(1, (), (), (), ()),
        structure=GlobalStructure(1, interpretation(), (), interpretation(), (), interpretation(), (), (), (), (), ()),
        bible=GlobalStoryBible(1, interpretation(), interpretation(), "测试", "测试", interpretation(), (), (), (), (), (), interpretation(), interpretation(), (("canonical_character_count", len(characters.analyses)), ("canonical_event_count", len(plan.event_stream)))),
    )


def test_no_current_is_a_normal_zero_write_miss(tmp_path, monkeypatch):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    before = (_bytes(tree.store.root), _bytes(tree.pointers.root))
    monkeypatch.setattr(LLMClient, "generate_structured", lambda *_a, **_k: pytest.fail("provider called"))
    assert B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests) is None
    assert (_bytes(tree.store.root), _bytes(tree.pointers.root)) == before


def test_exact_reuse_is_verified_zero_provider_and_zero_write(tmp_path, monkeypatch):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    publication_ref, pointer_ref, _pointer_id = _publish_current(tree, preparation, requests)
    before = (_bytes(tree.store.root), _bytes(tree.pointers.root))
    monkeypatch.setattr(LLMClient, "generate_structured", lambda *_a, **_k: pytest.fail("provider called"))
    hit = B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests)
    assert hit is not None and hit.reused
    assert hit.publication_ref == publication_ref
    assert hit.current_pointer_ref == pointer_ref
    assert (_bytes(tree.store.root), _bytes(tree.pointers.root)) == before


def test_changed_current_b12_selection_is_a_healthy_miss(tmp_path):
    tree, semantic, strategy, preparation, requests = _prepared(tmp_path)
    _publish_current(tree, preparation, requests)
    # A genuine immutable B1.2 revision is selected through its advanced
    # CURRENT pointer.  It has the same content but a different exact ref,
    # which must invalidate B1.3 rather than be treated as corruption.
    next_ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=strategy.strategy_ref.artifact_type, artifact_id=strategy.strategy_ref.artifact_id,
        revision=strategy.strategy_ref.revision + 1, schema_version=1,
        payload=strategy.artifact.to_dict(),
    ))
    tree.pointers.compare_and_set(
        pointer_id=strategy.current_pointer_ref.artifact_id, pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=strategy.current_pointer_ref, target_ref=next_ref,
    )
    fresh = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers, source_snapshot=preparation.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
    )
    fresh_requests = tuple(build_b13_window_semantic_preparation(x, runtime_config=_b12_test_runtime_config()) for x in fresh.packets)
    assert B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(fresh, fresh_requests) is None


def test_old_current_from_different_a6_profile_is_a_healthy_miss(tmp_path):
    from short_drama.adaptation import (
        AdaptationSourceSnapshot, AdaptationTarget, build_b12_semantic_preparation,
        execute_b12_strategy, prepare_b12_packet, resolve_adaptation_target,
    )
    from test_adaptation_strategy_execution import _FakeClient, _budget, _valid_proposal

    tree, _semantic, _strategy, old_preparation, old_requests = _prepared(tmp_path)
    _publish_current(tree, old_preparation, old_requests)
    a6_v2 = _publish_a6_v2(tree, old_preparation.source_snapshot)
    source = AdaptationSourceSnapshot.from_store(tree.store, tree.pointers, a6_v2.story_analysis_manifest_ref)
    target = resolve_adaptation_target(AdaptationTarget("semantic-preparation"), source_snapshot=source)
    b12 = build_b12_semantic_preparation(
        prepare_b12_packet(tree.store, tree.pointers, target), runtime_config=_b12_test_runtime_config(),
    )
    execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=b12,
        runtime_config=b12.runtime_config, budget=_budget(b12),
        llm_client=_FakeClient([(_valid_proposal(b12), False)], b12.runtime_config),
    )
    fresh = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        adaptation_target_id=target.adaptation_target_id,
    )
    fresh_requests = tuple(build_b13_window_semantic_preparation(
        packet, runtime_config=_b12_test_runtime_config(),
    ) for packet in fresh.packets)
    assert old_preparation.source_snapshot.story_analysis_manifest_ref.artifact_id != source.story_analysis_manifest_ref.artifact_id
    assert B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(fresh, fresh_requests) is None


def test_explicit_historical_b12_pin_remains_reusable_after_head_advances(tmp_path):
    tree, semantic, strategy, _current_preparation, _current_requests = _prepared(tmp_path)
    preparation = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
        strategy_ref=strategy.strategy_ref,
    )
    requests = tuple(build_b13_window_semantic_preparation(x, runtime_config=_b12_test_runtime_config()) for x in preparation.packets)
    _publish_current(tree, preparation, requests)
    next_ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=strategy.strategy_ref.artifact_type, artifact_id=strategy.strategy_ref.artifact_id,
        revision=strategy.strategy_ref.revision + 1, schema_version=1, payload=strategy.artifact.to_dict(),
    ))
    tree.pointers.compare_and_set(
        pointer_id=strategy.current_pointer_ref.artifact_id, pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=strategy.current_pointer_ref, target_ref=next_ref,
    )
    assert B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests) is not None


def test_wrong_target_and_missing_historical_upstream_fail_closed_without_fallback(tmp_path):
    tree, _semantic, strategy, preparation, requests = _prepared(tmp_path)
    _publication_ref, pointer_ref, pointer_id = _publish_current(tree, preparation, requests)
    # A valid pointer snapshot that targets a different immutable artifact is
    # corruption; the old B1.3 revision must never be scanned as fallback.
    tree.pointers.compare_and_set(
        pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
        expected_pointer_ref=pointer_ref, target_ref=strategy.strategy_ref,
    )
    with pytest.raises(B13DispositionPersistenceError, match="invalid identity or target"):
        B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests)


def test_missing_historical_a5_leaf_fails_before_match_comparison(tmp_path):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    _publish_current(tree, preparation, requests)
    ref = preparation.source_snapshot.canonical_event_set_ref
    (tree.store.root / ref.artifact_type / ref.artifact_id / f"r{ref.revision:08d}.json").unlink()
    with pytest.raises(B13DispositionPersistenceError, match="historical A5/A6 authority"):
        B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests)


def test_missing_historical_b12_artifact_fails_before_match_comparison(tmp_path):
    tree, _semantic, strategy, preparation, requests = _prepared(tmp_path)
    _publish_current(tree, preparation, requests)
    ref = strategy.strategy_ref
    (tree.store.root / ref.artifact_type / ref.artifact_id / f"r{ref.revision:08d}.json").unlink()
    with pytest.raises(B13DispositionPersistenceError, match="historical B1.2 strategy authority"):
        B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests)


def test_historical_packets_must_match_real_persisted_source_authority(tmp_path):
    tree, _semantic, _strategy, preparation, _requests = _prepared(tmp_path)
    altered = _altered_event_packet_preparation(preparation)
    altered_requests = tuple(build_b13_window_semantic_preparation(
        packet, runtime_config=_b12_test_runtime_config(),
    ) for packet in altered.packets)
    _publish_current(tree, altered, altered_requests)
    with pytest.raises(B13DispositionPersistenceError, match="retained window packets"):
        B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, _requests)


@pytest.mark.parametrize("mutate", ["missing_window", "duplicate_window", "forged_plan", "forged_prompt"])
def test_malformed_publication_evidence_fails_closed(tmp_path, mutate):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    _ref, pointer_ref, pointer_id = _publish_current(tree, preparation, requests)
    payload = _publication(preparation, requests).to_dict()
    windows = payload["evidence"]["accepted_windows"]
    if mutate == "missing_window":
        del windows[-1]
    elif mutate == "duplicate_window":
        windows.append(dict(windows[0]))
    elif mutate == "forged_plan":
        payload["evidence"]["assembly"]["plan"]["event_dispositions"][0]["rationale_zh"] = "伪造。"
    else:
        windows[0]["prompt_identity"]["prompt_content_hash"] = "f" * 64
    _replace_current_payload(tree, preparation, requests, pointer_ref, pointer_id, payload)
    with pytest.raises(B13DispositionPersistenceError, match="publication"):
        B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests)


def test_runtime_routing_variation_preserves_exact_reuse_eligibility(tmp_path):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    _publish_current(tree, preparation, requests)
    from dataclasses import replace

    varied = tuple(build_b13_window_semantic_preparation(
        packet,
        runtime_config=replace(_b12_test_runtime_config(), provider_family="different", request_model="different-model"),
    ) for packet in preparation.packets)
    assert B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, varied) is not None


def test_current_advance_mid_verification_fails_closed(tmp_path, monkeypatch):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    _publication_ref, pointer_ref, pointer_id = _publish_current(tree, preparation, requests)
    # Fixture-only second immutable revision gives the race an otherwise-valid target.
    next_ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        artifact_id=b13_disposition_publication_artifact_id(preparation.source_snapshot, preparation.strategy_selection.artifact),
        revision=2, schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=_publication(preparation, requests).to_dict(),
    ))
    original = tree.pointers.resolve_current_pointer_ref
    calls = {"count": 0}

    def advance(pointer):
        if pointer == pointer_id:
            calls["count"] += 1
            if calls["count"] == 2:
                tree.pointers.compare_and_set(
                    pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
                    expected_pointer_ref=pointer_ref, target_ref=next_ref,
                )
        return original(pointer)

    monkeypatch.setattr(tree.pointers, "resolve_current_pointer_ref", advance)
    with pytest.raises(B13DispositionPersistenceError, match="CURRENT changed during verification"):
        B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests)


def test_current_advance_during_final_evidence_verification_fails_closed(tmp_path, monkeypatch):
    import short_drama.adaptation.disposition_persistence as persistence_module

    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    _publication_ref, pointer_ref, pointer_id = _publish_current(tree, preparation, requests)
    next_ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        artifact_id=b13_disposition_publication_artifact_id(preparation.source_snapshot, preparation.strategy_selection.artifact),
        revision=2, schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=_publication(preparation, requests).to_dict(),
    ))
    original = persistence_module.verify_b13_disposition_publication
    calls = {"count": 0}

    def advance_on_final(value):
        result = original(value)
        calls["count"] += 1
        if calls["count"] == 3:
            tree.pointers.compare_and_set(
                pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=pointer_ref, target_ref=next_ref,
            )
        return result

    monkeypatch.setattr(persistence_module, "verify_b13_disposition_publication", advance_on_final)
    with pytest.raises(B13DispositionPersistenceError, match="CURRENT changed during verification"):
        B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests)
    assert calls["count"] == 3


def test_current_appearing_during_missing_current_validation_fails_closed(tmp_path, monkeypatch):
    tree, _semantic, _strategy, preparation, requests = _prepared(tmp_path)
    publication = _publication(preparation, requests)
    ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=B13_DISPOSITION_PUBLICATION_ARTIFACT_TYPE,
        artifact_id=b13_disposition_publication_artifact_id(preparation.source_snapshot, preparation.strategy_selection.artifact),
        revision=1, schema_version=B13_DISPOSITION_PUBLICATION_SCHEMA_VERSION,
        payload=publication.to_dict(),
    ))
    project_id = tree.store.get_ref(preparation.source_snapshot.story_analysis_manifest_ref).payload["project_id"]
    pointer_id = b13_disposition_pointer_id(
        project_id, preparation.strategy_selection.artifact.strategy.resolved_target.adaptation_target_id,
    )
    original = tree.pointers.resolve_current_pointer_ref
    calls = {"count": 0}

    def appear(pointer):
        if pointer == pointer_id:
            calls["count"] += 1
            if calls["count"] == 2:
                tree.pointers.compare_and_set(
                    pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
                    expected_pointer_ref=None, target_ref=ref,
                )
        return original(pointer)

    monkeypatch.setattr(tree.pointers, "resolve_current_pointer_ref", appear)
    with pytest.raises(B13DispositionPersistenceError, match="appeared during missing-current"):
        B13DispositionPersistenceService(tree.store, tree.pointers).try_reuse_current(preparation, requests)
