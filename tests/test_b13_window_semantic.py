"""B1.3-B2A window proposal and request preparation: no provider calls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import shutil

import pytest
from jsonschema import Draft202012Validator

import short_drama.adaptation.disposition_semantic as semantic_module
from short_drama.adaptation import (
    B12StrategySelection, B13DispositionWindowPacket, B13WindowSemanticBudget,
    B13WindowSemanticBudgetError, B13WindowSemanticError,
    build_b13_window_semantic_preparation, load_b13_window_semantic_assets,
    prepare_b13_disposition_windows,
)
from short_drama.adaptation.target import CanonicalTargetReference
from short_drama.artifacts import canonical_json_bytes
from short_drama.llm import PromptRegistry
from short_drama.paths import PROFILES_DIR, SCHEMAS_DIR

from test_adaptation_strategy import _b12_test_runtime_config
from test_b13_disposition_preparation import _published_strategy


def _prepared(tmp_path):
    tree, semantic, _publication = _published_strategy(tmp_path)
    prepared = prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers,
        source_snapshot=semantic.preparation.packet.source_snapshot,
        adaptation_target_id=semantic.preparation.packet.resolved_target.adaptation_target_id,
    )
    return prepared


def _packet(tmp_path):
    return _prepared(tmp_path).packets[0]


def _mapping(event_id, key, *, active=True):
    return {
        "source_ref": {"kind": "event", "canonical_id": event_id},
        "adapted_endpoint": {"kind": "story_beat", "local_key": key},
        "contribution_role": "primary",
        "retained_semantics_zh": "保留事件的因果作用。",
        "transformed_semantics_zh": "按短剧节奏重组。",
        "provenance_zh": "来自本窗口的改编候选。",
        "active": active,
    }


def _proposal(packet):
    owned = packet.to_dict()["window"]["owned_event_refs"]
    beats, mappings = [], []
    for ordinal, event_id in enumerate(owned, 1):
        key = f"beat{ordinal}"
        beats.append({
            "local_key": key, "summary_zh": f"保留 {event_id} 的核心冲突。",
            "origin": "source_direct", "canon_compatibility": "compatible",
            "source_event_refs": [{"kind": "event", "canonical_id": event_id}],
            "canonical_anchor_refs": [], "invention_purpose": None,
            "invention_rationale_zh": None, "invention_provenance_zh": None,
        })
        mappings.append(_mapping(event_id, key))
    return {
        "schema_version": 1,
        "event_dispositions": [
            {"source_event_id": event_id, "disposition": "keep", "rationale_zh": "保留核心冲突。"}
            for event_id in owned
        ],
        "adapted_characters": [], "adapted_story_beats": beats,
        "adaptation_mappings": mappings,
        "continuity_overlay_candidates": {
            "factual_divergences": [], "relationship_requirements": [],
            "story_state_requirements": [],
        },
    }


def _semantic(tmp_path):
    packet = _packet(tmp_path)
    return build_b13_window_semantic_preparation(packet, runtime_config=_b12_test_runtime_config())


def test_assets_are_closed_schema_valid_and_request_is_deterministic(tmp_path):
    assets = load_b13_window_semantic_assets()
    assert assets.version == 1
    assert assets.semantic_profile.profile_id == "b13-window-disposition-llm-v1"
    assert assets.semantic_profile.structured_output_mode == "json_schema"
    Draft202012Validator.check_schema(assets.output_schema.schema)
    first, second = _semantic(tmp_path / "one"), _semantic(tmp_path / "two")
    assert first.request_identity == second.request_identity
    assert first.request.request_hash == second.request.request_hash
    assert first.provider_request_bytes == second.provider_request_bytes
    assert first.measurement.provider_request_bytes == len(first.provider_request_bytes)
    assert first.measurement.provider_request_estimated_tokens >= 1
    assert first.measurement.reserved_output_tokens == 8192


@pytest.mark.parametrize("version", (0, 2, True, "1", None))
def test_assets_reject_unknown_versions(version):
    with pytest.raises(B13WindowSemanticError, match="unsupported B1.3 window semantic asset version"):
        load_b13_window_semantic_assets(version=version)


def test_proposal_has_exact_owned_coverage_and_never_assigns_final_ids(tmp_path):
    semantic = _semantic(tmp_path)
    proposal = _proposal(semantic.packet)
    decoded = semantic.decode_and_validate_proposal(proposal)
    assert decoded.to_dict() == proposal
    assert decoded.canonical_bytes() == canonical_json_bytes(proposal)
    for bad in (
        {**proposal, "event_dispositions": proposal["event_dispositions"][1:]},
        {**proposal, "event_dispositions": proposal["event_dispositions"] * 2},
        {**proposal, "adapted_story_beats": [{**proposal["adapted_story_beats"][0], "adapted_story_beat_id": "adapted_unauthorized"}]},
    ):
        with pytest.raises(B13WindowSemanticError):
            semantic.decode_and_validate_proposal(bad)


def test_context_event_cannot_receive_authoritative_decision(tmp_path):
    packet = _packet(tmp_path)
    view = packet.to_dict()
    context_id = "evt_000002"
    source = replace(
        packet.source_snapshot,
        canonical_references=(*packet.source_snapshot.canonical_references, CanonicalTargetReference("event", context_id)),
    )
    strategy = replace(
        packet.strategy_selection.artifact.strategy,
        resolved_target=replace(packet.strategy_selection.artifact.strategy.resolved_target, source_snapshot=source),
    )
    selection = B12StrategySelection(
        packet.strategy_selection.strategy_ref,
        replace(packet.strategy_selection.artifact, source_snapshot=source, strategy=strategy),
        packet.strategy_selection.current_pointer_ref,
    )
    view["source_snapshot"] = source.to_dict()
    view["global_adaptation_strategy"]["strategy"] = strategy.to_dict()
    view["global_adaptation_strategy"]["strategy_hash"] = strategy.content_hash()
    view["window"]["context_event_refs"] = [context_id]
    view["context_source_events"] = [{"event_id": context_id, "narrative_order": 2}]
    contextual = B13DispositionWindowPacket(
        source, selection, packet.window_id,
        packet.window_ordinal, canonical_json_bytes(view),
    )
    semantic = build_b13_window_semantic_preparation(contextual, runtime_config=_b12_test_runtime_config())
    proposal = _proposal(contextual)
    proposal["event_dispositions"].append({
        "source_event_id": context_id, "disposition": "omit", "rationale_zh": "错误地处置上下文。",
    })
    with pytest.raises(B13WindowSemanticError, match="owned events"):
        semantic.decode_and_validate_proposal(proposal)


def test_rejects_unknown_refs_invalid_local_links_and_inconsistent_mappings(tmp_path):
    semantic = _semantic(tmp_path)
    proposal = _proposal(semantic.packet)
    bad_ref = deepcopy(proposal)
    bad_ref["adapted_story_beats"][0]["source_event_refs"][0]["canonical_id"] = "evt_999999"
    with pytest.raises(B13WindowSemanticError, match="outside the exact"):
        semantic.decode_and_validate_proposal(bad_ref)
    bad_key = deepcopy(proposal)
    bad_key["adapted_story_beats"][0]["local_key"] = "adapted_final_id"
    with pytest.raises(B13WindowSemanticError, match="window-local"):
        semantic.decode_and_validate_proposal(bad_key)
    bad_link = deepcopy(proposal)
    bad_link["adaptation_mappings"][0]["adapted_endpoint"]["local_key"] = "missing"
    with pytest.raises(B13WindowSemanticError, match="does not resolve"):
        semantic.decode_and_validate_proposal(bad_link)
    omitted = deepcopy(proposal)
    omitted["event_dispositions"][0]["disposition"] = "omit"
    with pytest.raises(B13WindowSemanticError, match="omit owned event"):
        semantic.decode_and_validate_proposal(omitted)


def test_preserves_one_to_many_mapping_candidates_and_compress_merge_distinction(tmp_path):
    semantic = _semantic(tmp_path)
    proposal = _proposal(semantic.packet)
    event_id = proposal["event_dispositions"][0]["source_event_id"]
    proposal["adapted_story_beats"].append({
        "local_key": "echo", "summary_zh": "在另一节拍中保留同一后果。",
        "origin": "source_direct", "canon_compatibility": "compatible",
        "source_event_refs": [{"kind": "event", "canonical_id": event_id}],
        "canonical_anchor_refs": [], "invention_purpose": None,
        "invention_rationale_zh": None, "invention_provenance_zh": None,
    })
    proposal["adaptation_mappings"].append(_mapping(event_id, "echo"))
    decoded = semantic.decode_and_validate_proposal(proposal)
    assert len(decoded.to_dict()["adaptation_mappings"]) == len(proposal["adaptation_mappings"])
    proposal["event_dispositions"][0]["disposition"] = "compress"
    semantic.decode_and_validate_proposal(proposal)
    proposal["event_dispositions"][0]["disposition"] = "merge_member"
    with pytest.raises(B13WindowSemanticError, match="source_composite"):
        semantic.decode_and_validate_proposal(proposal)


def test_identity_binds_window_and_strategy_but_not_runtime_routing(tmp_path):
    semantic = _semantic(tmp_path)
    rerouted = replace(semantic.runtime_config, request_model="other-qwen", transport_id="other-runtime")
    changed_runtime = build_b13_window_semantic_preparation(semantic.packet, runtime_config=rerouted)
    assert changed_runtime.request_identity == semantic.request_identity
    assert changed_runtime.provider_request_bytes != semantic.provider_request_bytes
    view = semantic.packet.to_dict()
    view["window"]["window_id"] = "window_other"
    changed_window = B13DispositionWindowPacket(
        semantic.packet.source_snapshot, semantic.packet.strategy_selection, "window_other",
        semantic.packet.window_ordinal, canonical_json_bytes(view),
    )
    assert build_b13_window_semantic_preparation(changed_window, runtime_config=semantic.runtime_config).request_identity != semantic.request_identity
    changed_strategy = replace(semantic.packet.strategy_selection.artifact.strategy, adaptation_thesis_zh="另一条策略论点。")
    selection = B12StrategySelection(semantic.packet.strategy_selection.strategy_ref, replace(semantic.packet.strategy_selection.artifact, strategy=changed_strategy), semantic.packet.strategy_selection.current_pointer_ref)
    view = semantic.packet.to_dict()
    view["global_adaptation_strategy"]["strategy"] = changed_strategy.to_dict()
    view["global_adaptation_strategy"]["strategy_hash"] = changed_strategy.content_hash()
    changed_packet = B13DispositionWindowPacket(semantic.packet.source_snapshot, selection, semantic.packet.window_id, semantic.packet.window_ordinal, canonical_json_bytes(view))
    assert build_b13_window_semantic_preparation(changed_packet, runtime_config=semantic.runtime_config).request_identity != semantic.request_identity


def test_budget_requires_explicit_nctx_and_never_truncates(tmp_path):
    semantic = _semantic(tmp_path)
    with pytest.raises(B13WindowSemanticBudgetError, match="explicit semantic budget"):
        semantic.enforce_budget(None)
    with pytest.raises(B13WindowSemanticBudgetError, match="input ceiling"):
        semantic.enforce_budget(B13WindowSemanticBudget(semantic.runtime_config.request_model, 1_000_000, 1))
    required = semantic.measurement.provider_request_estimated_tokens + semantic.measurement.reserved_output_tokens
    with pytest.raises(B13WindowSemanticBudgetError, match="reserved output"):
        semantic.enforce_budget(B13WindowSemanticBudget(semantic.runtime_config.request_model, required - 1, 1_000_000))
    semantic.enforce_budget(B13WindowSemanticBudget(semantic.runtime_config.request_model, required, semantic.measurement.provider_request_estimated_tokens))


def test_modified_assets_and_injected_cross_pair_fail_closed(tmp_path, monkeypatch):
    semantic = _semantic(tmp_path / "prep")
    assets = load_b13_window_semantic_assets()
    with pytest.raises(B13WindowSemanticError, match="supplied together"):
        build_b13_window_semantic_preparation(semantic.packet, runtime_config=semantic.runtime_config, semantic_profile=assets.semantic_profile)
    asset_root = tmp_path / "assets"
    shutil.copytree(PROFILES_DIR, asset_root / "profiles")
    shutil.copytree(SCHEMAS_DIR, asset_root / "schemas")
    profile = asset_root / "profiles" / "b13_window_disposition_llm_v1.yaml"
    profile.write_text(profile.read_text(encoding="utf-8").replace("max_output_tokens: 8192", "max_output_tokens: 8193"), encoding="utf-8")
    monkeypatch.setattr(semantic_module, "REPO_ROOT", asset_root)
    with pytest.raises(B13WindowSemanticError, match="profile content hash"):
        load_b13_window_semantic_assets()


def test_modified_schema_asset_fails_closed(tmp_path, monkeypatch):
    asset_root = tmp_path / "assets"
    shutil.copytree(PROFILES_DIR, asset_root / "profiles")
    shutil.copytree(SCHEMAS_DIR, asset_root / "schemas")
    schema = asset_root / "schemas" / "b13-window-disposition-proposal-v1.schema.json"
    schema.write_text(
        schema.read_text(encoding="utf-8").replace('"schema_version": {"const": 1}', '"schema_version": {"const": 2}'),
        encoding="utf-8",
    )
    monkeypatch.setattr(semantic_module, "REPO_ROOT", asset_root)
    with pytest.raises(B13WindowSemanticError, match="schema content hash"):
        load_b13_window_semantic_assets()


def test_prompt_registry_rejects_modified_version(tmp_path):
    prompt_root = tmp_path / "prompts"
    shutil.copytree(semantic_module.B13_WINDOW_PROMPTS_DIR, prompt_root)
    system = prompt_root / "b13.window-disposition" / "v1" / "system.txt"
    system.write_text(system.read_text(encoding="utf-8") + "\n篡改。", encoding="utf-8")
    packet = _packet(tmp_path / "fixture")
    with pytest.raises(Exception, match="content_hash"):
        build_b13_window_semantic_preparation(packet, prompts=PromptRegistry(prompt_root), runtime_config=_b12_test_runtime_config())
