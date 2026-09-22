"""A4E-B ``reconcile-entities`` CLI wiring tests.

Covers the CLI / runtime composition contract against the merged A4E-A
project authority (``reconcile_entities_project``) — NO live Qwen server, NO
real provider, and NO provider lifecycle management. The CLI must:

  * expose the ``reconcile-entities`` command with a required project path and a
    required ``--chunk-profile``;
  * resolve the tracked / repo-root-safe defaults for the extraction,
    reconciliation, runtime, and semantic-LLM profiles;
  * compose the runtime (``load_runtime_config`` +
    ``OpenAICompatibleLLMClient``) and delegate to the A4E-A authority
    ``reconcile_entities_project`` WITHOUT duplicating any A4 orchestration;
  * pass the ``--llm-profile`` (semantic profile) path through as
    ``semantic_profile_path`` — NOT a runtime-config-derived identity;
  * print ``EntityReconciliationStageResult.to_dict()`` on success (exit 0) with
    the full A4E-A reporting fields (including ``reused`` and
    ``semantic_generation_call_count``);
  * convert expected ``StoryError`` / ``LLMError`` execution failures into a
    concise structured failure (exit 2) with no stack trace;
  * NEVER auto-run A3 (``extract_chunks_project``) — a missing A3 CURRENT must
    surface the A4E-A fail-closed error with zero provider calls.

Deliberately out of scope: live backend smoke, real-novel acceptance, semantic
invalidation smoke, A4E-C.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

import short_drama.cli as cli
from short_drama.llm import LLMError
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    StoryIntegrityError,
    TOKEN_COUNTER_ID,
)


PROJECT = "classroom"
CHUNK_PROFILE_ID = "story-analysis-v1"


# ---------------------------------------------------------------------------
# Minimal valid stage-result reporting payload (the full A4E-A field set)
# ---------------------------------------------------------------------------

_REF = {
    "artifact_type": "entity_map",
    "artifact_id": "classroom.src_001.story-analysis-v1.entity-map",
    "revision": 1,
    "content_hash": "f" * 64,
}

FULL_STAGE_RESULT_DICT = {
    "source_document_ref": dict(_REF, artifact_type="source_document"),
    "chunk_manifest_ref": dict(_REF, artifact_type="chunk_manifest"),
    "candidate_extraction_refs": [dict(_REF, artifact_type="candidate_extraction")],
    "candidate_count_total": 4,
    "character_candidate_count": 4,
    "location_candidate_count": 0,
    "a3_unresolved_candidate_count": 0,
    "pair_count_total": 6,
    "auto_same_pair_count": 0,
    "must_not_merge_pair_count": 0,
    "semantic_pair_count": 6,
    "deterministic_decision_count": 0,
    "llm_same_count": 6,
    "llm_different_count": 0,
    "llm_uncertain_count": 0,
    "canonical_character_count": 1,
    "canonical_location_count": 0,
    "unresolved_entity_count": 0,
    "entity_map_ref": dict(_REF),
    "validation_report_ref": dict(_REF, artifact_type="validation_report"),
    "current_pointer_ref": dict(_REF, artifact_type="pointer"),
    "semantic_block_count": 1,
    "semantic_generation_call_count": 1,
    "reused": False,
}


class FakeStageResult:
    """A stand-in for ``EntityReconciliationStageResult`` with a stable dict."""

    def __init__(self, payload: dict):
        self._payload = payload

    def to_dict(self) -> dict:
        return self._payload


# ---------------------------------------------------------------------------
# File writers (minimal, valid inputs for the CLI)
# ---------------------------------------------------------------------------


def _write_project(project_dir: Path, text: str, project_id: str = PROJECT) -> Path:
    source_dir = project_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "novel.txt").write_text(text, encoding="utf-8")
    project = {
        "schema_version": 1,
        "project_id": project_id,
        "title": "CLI Test",
        "source": {"type": "txt", "path": "source/novel.txt", "language": "zh-CN"},
        "production": {"output_language": "zh-CN", "profile": "h3_v1"},
        "approval_policy": {
            "adaptation_requires_approval": True,
            "generation_preflight_requires_approval": True,
            "shot_qc_requires_approval": True,
        },
    }
    path = project_dir / "project.yaml"
    path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return path


def _write_chunk_profile(path: Path, profile_id: str = CHUNK_PROFILE_ID) -> Path:
    # Must exactly equal the profile used to build the A2 manifest in
    # test_story_a4e_service.setup_a1_a2 (make_chunk_profile) so the A4E-A
    # authority accepts it.
    profile = {
        "schema_version": 1,
        "profile_id": profile_id,
        "token_counter": TOKEN_COUNTER_ID,
        "ownership_token_budget": 8,
        "context_overlap_token_budget": 4,
        "context_token_budget": 24,
    }
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


def _run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["short-drama", *args])
    return cli.main()


class _ExplodeIfCalledClient:
    """Runtime client stand-in that fails loudly if a provider call is made."""

    def generate_structured(self, *args, **kwargs):
        raise AssertionError("LLM client must not be called")


# ===========================================================================
# Parser / argument contract
# ===========================================================================


def test_reconcile_entities_command_exists(monkeypatch, capsys):
    # A valid subcommand is accepted by argparse (an invalid one would raise
    # SystemExit(2) before returning). Execution then fails on the missing
    # project / runtime with a concise structured failure and a non-zero exit.
    rc = _run_cli(
        monkeypatch, "reconcile-entities", "nope.yaml", "--chunk-profile", "nope.yaml"
    )
    assert rc == 2
    data = json.loads(capsys.readouterr().out)
    assert data["valid"] is False
    assert "error" in data


def test_reconcile_entities_requires_project(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "reconcile-entities", "--chunk-profile", "p.yaml")
    assert exc.value.code == 2


def test_reconcile_entities_requires_chunk_profile(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "reconcile-entities", "p.yaml")
    assert exc.value.code == 2


def test_reconcile_entities_profile_defaults_map_resolves_to_repo_root():
    # The tracked profile/config defaults are repo-root-safe (not CWD).
    assert cli.DEFAULT_RECONCILE_PROFILES["extraction_profile"] == (
        REPO_ROOT / "profiles" / "story_extraction_v1.yaml"
    )
    assert cli.DEFAULT_RECONCILE_PROFILES["reconciliation_profile"] == (
        REPO_ROOT / "profiles" / "entity_reconciliation_v2.yaml"
    )
    assert cli.DEFAULT_RECONCILE_PROFILES["runtime_config"] == (
        REPO_ROOT / "profiles" / "llm_local.yaml"
    )
    assert cli.DEFAULT_RECONCILE_PROFILES["llm_profile"] == (
        REPO_ROOT / "profiles" / "entity_reconciliation_llm_v1.yaml"
    )
    # The chunk profile is always explicit (no default).
    assert "chunk_profile" not in cli.DEFAULT_RECONCILE_PROFILES


def test_reconcile_entities_parser_defaults_flow_to_call(monkeypatch, capsys):
    # With only the required --chunk-profile supplied, every parser default must
    # reach the A4E-A authority unchanged.
    captured: dict = {}

    def fake_reconcile(project, **kwargs):
        captured["project"] = project
        captured.update(kwargs)
        return FakeStageResult(FULL_STAGE_RESULT_DICT)

    monkeypatch.setattr(cli, "reconcile_entities_project", fake_reconcile)
    monkeypatch.setattr(
        cli, "load_runtime_config", lambda path: captured.__setitem__("runtime_path", path) or "CFG"
    )
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda cfg: "CLIENT")

    rc = _run_cli(
        monkeypatch, "reconcile-entities", "proj.yaml", "--chunk-profile", "cp.yaml"
    )
    assert rc == 0
    assert captured["project"] == "proj.yaml"
    assert captured["runs_root"] == "runs"
    assert captured["chunk_profile_path"] == "cp.yaml"
    assert captured["extraction_profile_path"] == (
        REPO_ROOT / "profiles" / "story_extraction_v1.yaml"
    )
    assert captured["reconciliation_profile_path"] == (
        REPO_ROOT / "profiles" / "entity_reconciliation_v2.yaml"
    )
    assert captured["semantic_profile_path"] == (
        REPO_ROOT / "profiles" / "entity_reconciliation_llm_v1.yaml"
    )
    assert captured["runtime_path"] == (REPO_ROOT / "profiles" / "llm_local.yaml")
    assert captured["llm_client"] == "CLIENT"


# ===========================================================================
# Successful composition (delegates to the A4E-A authority)
# ===========================================================================


def test_reconcile_entities_cli_success(tmp_path, monkeypatch, capsys):
    runs_root = tmp_path / "runs"
    project_file = _write_project(tmp_path / "proj", "占位文本。")
    chunk_profile_file = _write_chunk_profile(tmp_path / "cp.yaml")
    extraction_file = tmp_path / "extraction.yaml"
    reconciliation_file = tmp_path / "reconciliation.yaml"
    runtime_file = tmp_path / "runtime.yaml"
    llm_profile_file = tmp_path / "llm.yaml"

    calls = {"runtime_config": 0, "client": 0}
    captured: dict = {}

    def fake_load_runtime_config(path):
        calls["runtime_config"] += 1
        captured["runtime_path"] = path
        return "RUNTIME_CONFIG_OBJ"

    def fake_client_factory(cfg):
        calls["client"] += 1
        captured["client_cfg"] = cfg
        return "CLIENT_OBJ"

    def fake_reconcile(project, **kwargs):
        captured["project"] = project
        captured.update(kwargs)
        return FakeStageResult(FULL_STAGE_RESULT_DICT)

    monkeypatch.setattr(cli, "load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", fake_client_factory)
    monkeypatch.setattr(cli, "reconcile_entities_project", fake_reconcile)

    rc = _run_cli(
        monkeypatch,
        "reconcile-entities",
        str(project_file),
        "--runs-root", str(runs_root),
        "--chunk-profile", str(chunk_profile_file),
        "--extraction-profile", str(extraction_file),
        "--reconciliation-profile", str(reconciliation_file),
        "--runtime-config", str(runtime_file),
        "--llm-profile", str(llm_profile_file),
    )
    assert rc == 0

    # Exact composition: the A4E-A authority receives the precise paths / client.
    assert captured["project"] == str(project_file)
    assert captured["runs_root"] == str(runs_root)
    assert captured["chunk_profile_path"] == str(chunk_profile_file)
    assert captured["extraction_profile_path"] == str(extraction_file)
    assert captured["reconciliation_profile_path"] == str(reconciliation_file)
    assert captured["semantic_profile_path"] == str(llm_profile_file)
    assert captured["llm_client"] == "CLIENT_OBJ"

    # load_runtime_config called exactly once; client constructed exactly once
    # and from the loaded runtime config (never from the semantic profile).
    assert calls["runtime_config"] == 1
    assert captured["runtime_path"] == str(runtime_file)
    assert calls["client"] == 1
    assert captured["client_cfg"] == "RUNTIME_CONFIG_OBJ"

    # Output is exactly EntityReconciliationStageResult.to_dict().
    data = json.loads(capsys.readouterr().out)
    assert data == FULL_STAGE_RESULT_DICT
    # Reporting fields are exposed verbatim (no raw prompt / model response).
    assert data["reused"] is False
    assert data["semantic_generation_call_count"] == 1
    assert data["entity_map_ref"]["artifact_type"] == "entity_map"
    for field in (
        "source_document_ref",
        "chunk_manifest_ref",
        "candidate_extraction_refs",
        "candidate_count_total",
        "character_candidate_count",
        "location_candidate_count",
        "a3_unresolved_candidate_count",
        "pair_count_total",
        "auto_same_pair_count",
        "must_not_merge_pair_count",
        "semantic_pair_count",
        "deterministic_decision_count",
        "llm_same_count",
        "llm_different_count",
        "llm_uncertain_count",
        "canonical_character_count",
        "canonical_location_count",
        "unresolved_entity_count",
        "entity_map_ref",
        "validation_report_ref",
        "current_pointer_ref",
        "semantic_block_count",
        "semantic_generation_call_count",
        "reused",
    ):
        assert field in data


# ===========================================================================
# No runtime / semantic identity pollution
# ===========================================================================


def test_reconcile_entities_semantic_profile_is_not_runtime_derived(tmp_path, monkeypatch, capsys):
    # The project helper must receive semantic_profile_path == --llm-profile,
    # NOT anything derived from the runtime config. Use distinct, unrelated
    # paths so any conflation is detectable.
    runtime_file = tmp_path / "runtime.yaml"
    llm_profile_file = tmp_path / "semantic_llm.yaml"

    calls = {"client": 0}
    captured: dict = {}

    def fake_load_runtime_config(path):
        captured["runtime_path"] = path
        return "RUNTIME_CONFIG_OBJ"

    def fake_client_factory(cfg):
        calls["client"] += 1
        return "CLIENT_OBJ"

    def fake_reconcile(project, **kwargs):
        captured.update(kwargs)
        return FakeStageResult(FULL_STAGE_RESULT_DICT)

    monkeypatch.setattr(cli, "load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", fake_client_factory)
    monkeypatch.setattr(cli, "reconcile_entities_project", fake_reconcile)

    rc = _run_cli(
        monkeypatch,
        "reconcile-entities",
        "proj.yaml",
        "--chunk-profile", "cp.yaml",
        "--runtime-config", str(runtime_file),
        "--llm-profile", str(llm_profile_file),
    )
    assert rc == 0
    # semantic_profile_path is the --llm-profile path exactly (a path, not a
    # runtime-config-derived object).
    assert captured["semantic_profile_path"] == str(llm_profile_file)
    # The runtime config object is consumed only to build the client.
    assert captured["runtime_path"] == str(runtime_file)
    assert captured["llm_client"] == "CLIENT_OBJ"
    assert calls["client"] == 1
    # The runtime config path must never appear as the semantic profile path.
    assert captured["semantic_profile_path"] != str(runtime_file)


# ===========================================================================
# Reuse behavior through the CLI (the CLI does not decide reuse)
# ===========================================================================


def test_reconcile_entities_reuse_result_printed_unchanged(monkeypatch, capsys):
    # A reused result is simply printed as-is; the CLI adds no special logic.
    reused_result = FakeStageResult(
        dict(FULL_STAGE_RESULT_DICT, reused=True, semantic_generation_call_count=0)
    )
    monkeypatch.setattr(
        cli, "reconcile_entities_project", lambda *a, **k: reused_result
    )
    monkeypatch.setattr(cli, "load_runtime_config", lambda path: "RUNTIME_CONFIG_OBJ")
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda cfg: "CLIENT_OBJ")

    rc = _run_cli(
        monkeypatch, "reconcile-entities", "proj.yaml", "--chunk-profile", "cp.yaml"
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data == dict(
        FULL_STAGE_RESULT_DICT, reused=True, semantic_generation_call_count=0
    )
    assert data["reused"] is True
    assert data["semantic_generation_call_count"] == 0


# ===========================================================================
# Failure behavior (structured JSON + exit 2, no traceback)
# ===========================================================================


def test_reconcile_entities_story_error_is_structured_failure(monkeypatch, capsys):
    # Mock the project orchestration to raise an existing StoryError subclass.
    def fake_reconcile(*a, **k):
        raise StoryIntegrityError("A3 CandidateExtraction is not current for X")

    monkeypatch.setattr(cli, "reconcile_entities_project", fake_reconcile)
    monkeypatch.setattr(cli, "load_runtime_config", lambda path: "RUNTIME_CONFIG_OBJ")
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda cfg: "CLIENT_OBJ")

    rc = _run_cli(
        monkeypatch, "reconcile-entities", "proj.yaml", "--chunk-profile", "cp.yaml"
    )
    assert rc == 2
    out = capsys.readouterr().out
    data = json.loads(out)  # structured JSON, no traceback
    assert data == {
        "valid": False,
        "error": "A3 CandidateExtraction is not current for X",
    }


def test_reconcile_entities_llm_error_is_structured_failure(monkeypatch, capsys):
    # Mock the project orchestration to raise LLMError (no CLI retry).
    calls = {"reconcile": 0}

    def fake_reconcile(*a, **k):
        calls["reconcile"] += 1
        raise LLMError("provider timeout")

    monkeypatch.setattr(cli, "reconcile_entities_project", fake_reconcile)
    monkeypatch.setattr(cli, "load_runtime_config", lambda path: "RUNTIME_CONFIG_OBJ")
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda cfg: "CLIENT_OBJ")

    rc = _run_cli(
        monkeypatch, "reconcile-entities", "proj.yaml", "--chunk-profile", "cp.yaml"
    )
    assert rc == 2
    out = capsys.readouterr().out
    data = json.loads(out)  # structured JSON, no traceback
    assert data == {"valid": False, "error": "provider timeout"}
    # No CLI-level retry: the project orchestration is invoked exactly once.
    assert calls["reconcile"] == 1


# ===========================================================================
# Missing A3 CURRENT → fail closed; the CLI never auto-runs A3
# ===========================================================================


def test_reconcile_entities_missing_a3_never_auto_runs_a3(tmp_path, monkeypatch, capsys):
    # Real A1+A2 state (valid), NO A3. The merged A4E-A authority fails closed
    # with the missing-A3 StoryIntegrityError. The CLI must surface that error,
    # must NOT invoke extract_chunks_project, and must make zero provider calls.
    from test_story_a4e_service import setup_a1_a2

    runs_root = tmp_path / "runs"
    story_root = runs_root / PROJECT / "story"
    setup_a1_a2(story_root)  # A1 + A2 only (no A3)

    project_file = _write_project(tmp_path / "proj", "占位文本。")
    chunk_profile_file = _write_chunk_profile(tmp_path / "cp.yaml")

    extract_calls = []

    def extract_spy(*args, **kwargs):
        extract_calls.append((args, kwargs))
        raise AssertionError("reconcile-entities must not auto-run A3 extraction")

    monkeypatch.setattr(cli, "extract_chunks_project", extract_spy)
    monkeypatch.setattr(cli, "load_runtime_config", lambda path: "RUNTIME_CONFIG_OBJ")
    monkeypatch.setattr(
        cli, "OpenAICompatibleLLMClient", lambda cfg: _ExplodeIfCalledClient()
    )

    rc = _run_cli(
        monkeypatch,
        "reconcile-entities",
        str(project_file),
        "--runs-root", str(runs_root),
        "--chunk-profile", str(chunk_profile_file),
    )
    assert rc == 2
    out = capsys.readouterr().out
    data = json.loads(out)  # structured JSON, no traceback
    assert data["valid"] is False
    assert "A3 CandidateExtraction is not current" in data["error"]
    assert "run short-drama extract-chunks first" in data["error"]
    # No A3 auto-extraction, no recovery, no provider call.
    assert extract_calls == []


# ===========================================================================
# A3E-B regression: extract-chunks defaults / behavior are unchanged
# ===========================================================================


def test_extract_chunks_defaults_unchanged():
    assert cli.DEFAULT_EXTRACT_PROFILES["extraction_profile"] == (
        REPO_ROOT / "profiles" / "story_extraction_v1.yaml"
    )
    assert cli.DEFAULT_EXTRACT_PROFILES["runtime_config"] == (
        REPO_ROOT / "profiles" / "llm_local.yaml"
    )
    assert cli.DEFAULT_EXTRACT_PROFILES["llm_profile"] == (
        REPO_ROOT / "profiles" / "story_extraction_llm_v1.yaml"
    )
    assert "chunk_profile" not in cli.DEFAULT_EXTRACT_PROFILES


def test_reconcile_entities_does_not_alter_extract_chunks_defaults():
    # Adding the A4E-B defaults must not perturb the A3E-B default mapping.
    assert set(cli.DEFAULT_EXTRACT_PROFILES) == {
        "extraction_profile",
        "runtime_config",
        "llm_profile",
    }
    assert set(cli.DEFAULT_RECONCILE_PROFILES) == {
        "extraction_profile",
        "reconciliation_profile",
        "runtime_config",
        "llm_profile",
    }
