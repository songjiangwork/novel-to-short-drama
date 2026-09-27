"""A5G2 ``consolidate-evidence`` CLI/runtime composition tests (offline)."""

from __future__ import annotations

import json
import sys

import pytest

import short_drama.cli as cli
from short_drama.llm import LLMError
from short_drama.paths import REPO_ROOT
from short_drama.story import ConsolidationCurrentMissingError, StoryIntegrityError


FULL_STAGE_RESULT_DICT = {
    "entity_map_ref": {"artifact_type": "entity_map", "artifact_id": "p.a4", "revision": 1, "content_hash": "a" * 64},
    "candidate_extraction_refs": [{"artifact_type": "candidate_extraction", "artifact_id": "p.a3", "revision": 1, "content_hash": "b" * 64}],
    "fact_candidate_count": 2,
    "event_candidate_count": 3,
    "relationship_candidate_count": 4,
    "resolved_bound_reference_count": 5,
    "unresolved_bound_reference_count": 0,
    "fact_planned_pair_count": 1,
    "event_planned_pair_count": 2,
    "relationship_planned_pair_count": 3,
    "fact_deterministic_decision_count": 1,
    "event_deterministic_decision_count": 2,
    "relationship_deterministic_decision_count": 3,
    "fact_semantic_decision_count": 1,
    "event_semantic_decision_count": 1,
    "relationship_semantic_decision_count": 1,
    "fact_semantic_block_count": 1,
    "event_semantic_block_count": 1,
    "relationship_semantic_block_count": 1,
    "semantic_generation_call_count": 3,
    "canonical_fact_count": 2,
    "canonical_event_count": 3,
    "canonical_relationship_count": 4,
    "state_transition_count": 1,
    "story_conflict_count": 0,
    "consolidation_candidate_index_ref": {"artifact_type": "consolidation_candidate_index", "artifact_id": "p.a5.index", "revision": 1, "content_hash": "c" * 64},
    "consolidation_decision_set_ref": {"artifact_type": "consolidation_decision_set", "artifact_id": "p.a5.decisions", "revision": 1, "content_hash": "d" * 64},
    "canonical_fact_set_ref": {"artifact_type": "canonical_fact_set", "artifact_id": "p.a5.facts", "revision": 1, "content_hash": "e" * 64},
    "canonical_event_set_ref": {"artifact_type": "canonical_event_set", "artifact_id": "p.a5.events", "revision": 1, "content_hash": "f" * 64},
    "canonical_relationship_set_ref": {"artifact_type": "canonical_relationship_set", "artifact_id": "p.a5.relationships", "revision": 1, "content_hash": "0" * 64},
    "story_conflict_set_ref": {"artifact_type": "story_conflict_set", "artifact_id": "p.a5.conflicts", "revision": 1, "content_hash": "1" * 64},
    "consolidation_manifest_ref": {"artifact_type": "consolidation_manifest", "artifact_id": "p.a5.manifest", "revision": 1, "content_hash": "2" * 64},
    "validation_report_ref": {"artifact_type": "validation_report", "artifact_id": "p.a5.validation", "revision": 1, "content_hash": "3" * 64},
    "current_pointer_ref": {"artifact_type": "pointer", "artifact_id": "p.a5.current", "revision": 1, "content_hash": "4" * 64},
    "reused": False,
}


class FakeStageResult:
    def __init__(self, payload: dict):
        self._payload = payload

    def to_dict(self) -> dict:
        return self._payload


def _run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["short-drama", *args])
    return cli.main()


def _patch_success_boundaries(monkeypatch, result=None):
    captured: dict = {}
    order: list[str] = []

    def load_runtime(path):
        order.append("runtime")
        captured["runtime_path"] = path
        return "RUNTIME_CONFIG_OBJ"

    def client_factory(config):
        order.append("client")
        captured["client_config"] = config
        return "CLIENT_OBJ"

    def consolidate(project, **kwargs):
        order.append("consolidate")
        captured["project"] = project
        captured.update(kwargs)
        return result or FakeStageResult(FULL_STAGE_RESULT_DICT)

    monkeypatch.setattr(cli, "load_runtime_config", load_runtime)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", client_factory)
    monkeypatch.setattr(cli, "consolidate_evidence_project", consolidate)
    return captured, order


def test_consolidate_evidence_defaults_are_repo_root_safe_and_need_no_a3_a4_flags(monkeypatch, capsys):
    captured, _order = _patch_success_boundaries(monkeypatch)

    rc = _run_cli(monkeypatch, "consolidate-evidence", "project.yaml")

    assert rc == 0
    assert captured["project"] == "project.yaml"
    assert captured["runs_root"] == "runs"
    assert captured["reconciliation_profile_id"] == "entity-reconciliation-v2"
    assert captured["consolidation_profile_path"] == REPO_ROOT / "profiles" / "consolidation_v1.yaml"
    assert captured["semantic_profile_path"] == REPO_ROOT / "profiles" / "consolidation_llm_v1.yaml"
    assert captured["runtime_path"] == REPO_ROOT / "profiles" / "llm_local.yaml"
    assert json.loads(capsys.readouterr().out) == FULL_STAGE_RESULT_DICT
    assert cli.DEFAULT_CONSOLIDATE_PROFILES == {
        "reconciliation_profile_id": "entity-reconciliation-v2",
        "consolidation_profile": REPO_ROOT / "profiles" / "consolidation_v1.yaml",
        "runtime_config": REPO_ROOT / "profiles" / "llm_local.yaml",
        "llm_profile": REPO_ROOT / "profiles" / "consolidation_llm_v1.yaml",
    }


@pytest.mark.parametrize("forbidden_flag", ["--chunk-profile", "--extraction-profile"])
def test_consolidate_evidence_has_no_a3_a4_flags(monkeypatch, forbidden_flag):
    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "consolidate-evidence", "project.yaml", forbidden_flag, "x.yaml")
    assert exc.value.code == 2


def test_consolidate_evidence_composes_exact_runtime_and_a5_inputs(monkeypatch, capsys):
    captured, order = _patch_success_boundaries(monkeypatch)

    rc = _run_cli(
        monkeypatch,
        "consolidate-evidence", "project.yaml",
        "--runs-root", "custom-runs",
        "--reconciliation-profile-id", "a4-namespace",
        "--consolidation-profile", "consolidation.yaml",
        "--runtime-config", "runtime.yaml",
        "--llm-profile", "semantic.yaml",
    )

    assert rc == 0
    assert order == ["runtime", "client", "consolidate"]
    assert captured == {
        "runtime_path": "runtime.yaml",
        "client_config": "RUNTIME_CONFIG_OBJ",
        "project": "project.yaml",
        "runs_root": "custom-runs",
        "reconciliation_profile_id": "a4-namespace",
        "consolidation_profile_path": "consolidation.yaml",
        "semantic_profile_path": "semantic.yaml",
        "llm_client": "CLIENT_OBJ",
    }
    assert "runtime.yaml" not in {
        captured["consolidation_profile_path"],
        captured["semantic_profile_path"],
        captured["reconciliation_profile_id"],
    }
    assert json.loads(capsys.readouterr().out) == FULL_STAGE_RESULT_DICT


def test_consolidate_evidence_reuse_result_is_printed_unchanged(monkeypatch, capsys):
    reused = dict(FULL_STAGE_RESULT_DICT, reused=True, semantic_generation_call_count=0)
    _patch_success_boundaries(monkeypatch, FakeStageResult(reused))

    rc = _run_cli(monkeypatch, "consolidate-evidence", "project.yaml")

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == reused


def test_consolidate_evidence_story_error_is_structured_failure(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_runtime_config", lambda _path: "RUNTIME_CONFIG_OBJ")
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda _config: "CLIENT_OBJ")
    monkeypatch.setattr(
        cli,
        "consolidate_evidence_project",
        lambda *args, **kwargs: (_ for _ in ()).throw(StoryIntegrityError("bad A5 input")),
    )

    rc = _run_cli(monkeypatch, "consolidate-evidence", "project.yaml")

    assert rc == 2
    assert json.loads(capsys.readouterr().out) == {"valid": False, "error": "bad A5 input"}


def test_consolidate_evidence_llm_error_has_no_cli_retry(monkeypatch, capsys):
    calls = {"consolidate": 0}
    monkeypatch.setattr(cli, "load_runtime_config", lambda _path: "RUNTIME_CONFIG_OBJ")
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda _config: "CLIENT_OBJ")

    def fail(*args, **kwargs):
        calls["consolidate"] += 1
        raise LLMError("provider timeout")

    monkeypatch.setattr(cli, "consolidate_evidence_project", fail)

    rc = _run_cli(monkeypatch, "consolidate-evidence", "project.yaml")

    assert rc == 2
    assert calls == {"consolidate": 1}
    assert json.loads(capsys.readouterr().out) == {"valid": False, "error": "provider timeout"}


def test_missing_a4_current_fails_closed_without_a3_a4_autorun(monkeypatch, capsys):
    calls = {"consolidate": 0, "extract": 0, "reconcile": 0}
    monkeypatch.setattr(cli, "load_runtime_config", lambda _path: "RUNTIME_CONFIG_OBJ")
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda _config: "CLIENT_OBJ")

    def missing_current(*args, **kwargs):
        calls["consolidate"] += 1
        raise ConsolidationCurrentMissingError("A4 EntityMap is not current for p")

    def unexpected_extract(*args, **kwargs):
        calls["extract"] += 1
        raise AssertionError("A5 CLI must not auto-run A3")

    def unexpected_reconcile(*args, **kwargs):
        calls["reconcile"] += 1
        raise AssertionError("A5 CLI must not auto-run A4")

    monkeypatch.setattr(cli, "consolidate_evidence_project", missing_current)
    monkeypatch.setattr(cli, "extract_chunks_project", unexpected_extract)
    monkeypatch.setattr(cli, "reconcile_entities_project", unexpected_reconcile)

    rc = _run_cli(monkeypatch, "consolidate-evidence", "project.yaml")

    assert rc == 2
    assert calls == {"consolidate": 1, "extract": 0, "reconcile": 0}
    assert json.loads(capsys.readouterr().out) == {
        "valid": False,
        "error": "A4 EntityMap is not current for p",
    }
