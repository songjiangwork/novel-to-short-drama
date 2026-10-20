"""B1.3-B1 stays read-only: select B1.2 and prepare complete A6 windows."""
from __future__ import annotations

from dataclasses import replace

import pytest

from short_drama.adaptation import (
    AdaptationSourceSnapshot, B13PreparationError, B13DispositionWindowPacket,
    B12StrategySelection, prepare_b13_disposition_windows,
    resolve_b12_strategy_selection,
)
from short_drama.artifacts import ArtifactRef, ImmutableArtifactEnvelope, canonical_json_bytes
from short_drama.foundation import PointerKind
from short_drama.story.story_analysis import PlotWindowAnalysis

from test_adaptation_strategy import _semantic_preparation
from test_adaptation_strategy_execution import _FakeClient, _budget, _move_current_to_payload, _valid_proposal
from short_drama.adaptation import execute_b12_strategy
from short_drama.adaptation.disposition_preparation import _ordered_events, _validate_a6_windows
from short_drama.story.consolidation import CanonicalEventSet
from short_drama.story.story_analysis import GlobalEventAnalysis, StoryAnalysisManifest


def _published_strategy(tmp_path):
    tree, _packet, semantic = _semantic_preparation(tmp_path)
    publication = execute_b12_strategy(
        store=tree.store, pointers=tree.pointers, semantic=semantic,
        runtime_config=semantic.runtime_config, budget=_budget(semantic),
        llm_client=_FakeClient([(_valid_proposal(semantic), False)], semantic.runtime_config),
    )
    return tree, semantic, publication


def _source_and_windows(tree, semantic):
    source = semantic.preparation.packet.source_snapshot
    manifest = StoryAnalysisManifest.from_dict(
        tree.store.get_ref(source.story_analysis_manifest_ref).payload
    )
    analysis = GlobalEventAnalysis.from_dict(
        tree.store.get_ref(manifest.global_event_analysis_ref).payload
    )
    events = _ordered_events(CanonicalEventSet.from_dict(
        tree.store.get_ref(source.canonical_event_set_ref).payload
    ))
    return source, events, analysis.windows


def test_preparation_selects_verified_strategy_and_measures_complete_packets(tmp_path):
    tree, semantic, publication = _published_strategy(tmp_path)
    source = semantic.preparation.packet.source_snapshot

    prepared = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
    )

    assert prepared.strategy_selection.strategy_ref == publication.strategy_ref
    assert prepared.strategy_selection.current_pointer_ref == publication.current_pointer_ref
    assert [item.window_ordinal for item in prepared.packets] == list(range(1, len(prepared.packets) + 1))
    assert {event_id for packet in prepared.packets for event_id in packet.to_dict()["window"]["owned_event_refs"]} == set(prepared.canonical_event_ids)
    assert sum(len(packet.to_dict()["window"]["owned_event_refs"]) for packet in prepared.packets) == len(prepared.canonical_event_ids)
    assert all(item.packet_bytes == len(packet.canonical_bytes()) and item.packet_estimated_tokens >= 1 for item, packet in zip(prepared.measurements, prepared.packets))
    assert all(packet.to_dict()["global_adaptation_strategy"]["strategy"] == publication.artifact.strategy.to_dict() for packet in prepared.packets)


def test_explicit_strategy_revision_is_pinned_and_wrong_target_fails_closed(tmp_path):
    tree, semantic, publication = _published_strategy(tmp_path)
    source = semantic.preparation.packet.source_snapshot
    target_id = semantic.preparation.packet.resolved_target.adaptation_target_id
    selected = resolve_b12_strategy_selection(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        adaptation_target_id=target_id, strategy_ref=publication.strategy_ref,
    )
    assert selected.strategy_ref == publication.strategy_ref
    assert selected.current_pointer_ref is None
    with pytest.raises(B13PreparationError, match="invalid identity"):
        resolve_b12_strategy_selection(
            store=tree.store, pointers=tree.pointers, source_snapshot=source,
            adaptation_target_id="another-target", strategy_ref=publication.strategy_ref,
        )


def test_a6_cross_window_owned_sequence_must_equal_a5_narrative_order(tmp_path):
    tree, semantic, _publication = _published_strategy(tmp_path)
    _source, fixture_events, fixture_windows = _source_and_windows(tree, semantic)
    events = tuple(
        replace(fixture_events[0], event_id=f"evt_{ordinal:06d}", narrative_order=ordinal)
        for ordinal in range(1, 5)
    )
    interpretation = fixture_windows[0].interpretation
    valid = (
        PlotWindowAnalysis("window_0001", 1, ("evt_000001", "evt_000002"), (), interpretation),
        PlotWindowAnalysis("window_0002", 2, ("evt_000003", "evt_000004"), ("evt_000002",), interpretation),
    )
    assert _validate_a6_windows(valid, events) == valid
    # Every window is internally ordered and each A5 event has exactly one
    # owner, but their concatenated ownership sequence is globally swapped.
    globally_swapped = (
        PlotWindowAnalysis("window_0001", 1, ("evt_000003", "evt_000004"), ("evt_000002",), interpretation),
        PlotWindowAnalysis("window_0002", 2, ("evt_000001", "evt_000002"), (), interpretation),
    )
    with pytest.raises(B13PreparationError, match="owned-event sequence does not match A5 narrative order"):
        _validate_a6_windows(globally_swapped, events)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown", "owned_as_context"])
def test_a6_window_ownership_rejects_missing_duplicate_unknown_and_context_ownership(tmp_path, kind):
    tree, semantic, _publication = _published_strategy(tmp_path)
    _source, events, windows = _source_and_windows(tree, semantic)
    first = windows[0]
    if kind == "missing":
        changed = (replace(first, owned_event_refs=first.owned_event_refs[1:]), *windows[1:])
    elif kind == "duplicate":
        duplicate = replace(first)
        # The A6 typed decoder rejects this on load; bypass it here to ensure
        # the B1.3 ownership seam independently retains the same guard.
        object.__setattr__(duplicate, "owned_event_refs", (first.owned_event_refs[0], first.owned_event_refs[0], *first.owned_event_refs[1:]))
        changed = (duplicate, *windows[1:])
    elif kind == "unknown":
        changed = (replace(first, owned_event_refs=("evt_999999", *first.owned_event_refs[1:])), *windows[1:])
    else:
        changed = (replace(first, context_event_refs=(first.owned_event_refs[0], *first.context_event_refs)), *windows[1:])
    with pytest.raises(B13PreparationError, match="exact-one canonical coverage"):
        _validate_a6_windows(changed, events)


def test_a6_window_order_and_context_order_must_be_deterministic(tmp_path):
    tree, semantic, _publication = _published_strategy(tmp_path)
    _source, events, windows = _source_and_windows(tree, semantic)
    first = windows[0]
    # The persisted sequence itself must be ordered, even in a one-window
    # fixture.  With multiple event IDs, its contained event order is guarded
    # independently too.
    changed_ordinal = (replace(first, window_ordinal=2), *windows[1:])
    with pytest.raises(B13PreparationError, match="exact-one canonical coverage"):
        _validate_a6_windows(changed_ordinal, events)
    if len(first.owned_event_refs) > 1:
        changed = (replace(first, owned_event_refs=tuple(reversed(first.owned_event_refs))), *windows[1:])
        with pytest.raises(B13PreparationError, match="narrative order"):
            _validate_a6_windows(changed, events)


def test_packet_preserves_event_ids_and_exact_source_evidence(tmp_path):
    tree, semantic, _publication = _published_strategy(tmp_path)
    prepared = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
    )
    packet = prepared.packets[0]
    view = packet.to_dict()
    assert [item["event_id"] for item in view["owned_source_events"]] == view["window"]["owned_event_refs"]
    assert all("evidence_refs" in item for item in view["owned_source_events"])
    forged = packet.to_dict()
    forged["owned_source_events"][0]["event_id"] = "evt_999999"
    with pytest.raises(B13PreparationError, match="preserve exact IDs"):
        B13DispositionWindowPacket(
            packet.source_snapshot, packet.strategy_selection, packet.window_id,
            packet.window_ordinal, canonical_json_bytes(forged),
        )


def test_packet_identity_changes_with_global_strategy_identity(tmp_path):
    tree, semantic, _publication = _published_strategy(tmp_path)
    prepared = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
    )
    packet = prepared.packets[0]
    changed_strategy = replace(
        packet.strategy_selection.artifact.strategy, adaptation_thesis_zh="不同的全局改编论点。"
    )
    changed_selection = B12StrategySelection(
        packet.strategy_selection.strategy_ref,
        replace(packet.strategy_selection.artifact, strategy=changed_strategy),
        packet.strategy_selection.current_pointer_ref,
    )
    forged = packet.to_dict()
    forged["global_adaptation_strategy"]["strategy"] = changed_strategy.to_dict()
    forged["global_adaptation_strategy"]["strategy_hash"] = changed_strategy.content_hash()
    changed_packet = B13DispositionWindowPacket(
        packet.source_snapshot, changed_selection, packet.window_id,
        packet.window_ordinal, canonical_json_bytes(forged),
    )
    assert changed_packet.content_hash() != packet.content_hash()


def test_source_snapshot_lineage_mismatch_and_corrupt_current_do_not_fall_back(tmp_path):
    tree, semantic, publication = _published_strategy(tmp_path)
    source = semantic.preparation.packet.source_snapshot
    wrong = replace(source, consolidation_manifest_ref=ArtifactRef(
        "consolidation_manifest", publication.strategy_ref.artifact_id,
        publication.strategy_ref.revision, publication.strategy_ref.content_hash,
    ))
    with pytest.raises(B13PreparationError):
        resolve_b12_strategy_selection(
            store=tree.store, pointers=tree.pointers, source_snapshot=wrong,
            adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
        )


def test_mismatched_b12_lineage_current_fails_closed_before_window_preparation(tmp_path):
    tree, semantic, publication = _published_strategy(tmp_path)
    payload = publication.artifact.to_dict()
    # Keep the artifact structurally decodable while pinning it to a source
    # leaf that does not exist; intrinsic B1.2 lineage validation must reject
    # the CURRENT rather than treating it as an ordinary cache miss.
    forged = dict(payload["source_snapshot"]["canonical_event_set_ref"])
    forged["content_hash"] = "f" * 64
    payload["source_snapshot"]["canonical_event_set_ref"] = forged
    payload["strategy"]["resolved_target"]["source_snapshot"]["canonical_event_set_ref"] = forged
    _move_current_to_payload(tree, semantic, publication, payload)
    with pytest.raises(B13PreparationError, match="unverified"):
        prepare_b13_disposition_windows(
            store=tree.store, pointers=tree.pointers,
            source_snapshot=semantic.preparation.packet.source_snapshot,
            adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
        )


def test_b12_current_advance_between_exact_snapshot_and_recheck_fails_closed(tmp_path, monkeypatch):
    tree, semantic, publication = _published_strategy(tmp_path)
    target_id = semantic.preparation.packet.resolved_target.adaptation_target_id
    pointer_id = publication.current_pointer_ref.artifact_id
    next_strategy_ref = tree.store.put(ImmutableArtifactEnvelope.create(
        artifact_type=publication.strategy_ref.artifact_type,
        artifact_id=publication.strategy_ref.artifact_id,
        revision=publication.strategy_ref.revision + 1,
        schema_version=1,
        payload=publication.artifact.to_dict(),
    ))
    original = tree.pointers.resolve_current_pointer_ref
    calls = {"count": 0}

    def advance_after_snapshot(observed_pointer_id):
        if observed_pointer_id == pointer_id:
            calls["count"] += 1
        if observed_pointer_id == pointer_id and calls["count"] == 2:
            tree.pointers.compare_and_set(
                pointer_id=pointer_id, pointer_kind=PointerKind.CURRENT,
                expected_pointer_ref=publication.current_pointer_ref,
                target_ref=next_strategy_ref,
            )
        return original(observed_pointer_id)

    monkeypatch.setattr(tree.pointers, "resolve_current_pointer_ref", advance_after_snapshot)
    with pytest.raises(B13PreparationError, match="B1.2 CURRENT pointer changed during strategy selection"):
        resolve_b12_strategy_selection(
            store=tree.store, pointers=tree.pointers,
            source_snapshot=semantic.preparation.packet.source_snapshot,
            adaptation_target_id=target_id,
        )
    assert calls["count"] == 2


def test_current_selection_pointer_snapshot_authorizes_returned_strategy(tmp_path):
    tree, semantic, publication = _published_strategy(tmp_path)
    selected = resolve_b12_strategy_selection(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
    )
    assert selected.current_pointer_ref == publication.current_pointer_ref
    assert tree.pointers._load_pointer_ref(selected.current_pointer_ref).target_ref == selected.strategy_ref
