"""A3E-B ``extract-chunks`` CLI wiring tests.

Covers the CLI/application composition contract against a deterministic,
provider-neutral fake ``LLMClient`` — NO live Qwen server, NO real provider, and
NO provider lifecycle management. The CLI must:

  * expose the ``extract-chunks`` command with a required project path and a
    required ``--chunk-profile``;
  * resolve the tracked / repo-root-safe defaults for the extraction, runtime,
    and semantic-LLM profiles;
  * compose the existing authorities (project loader, profile loaders, store
    creation, the merged A3E-A :class:`ChunkExtractionBatchService`) and actually
    reach the A3E-A batch authority;
  * print a JSON-compatible structured batch summary on success (exit 0) with
    total / reused / generated / failed counts and ordered extraction /
    validation refs;
  * preserve the A3E-A reused/generated counts in the exposed summary;
  * convert expected ``StoryError`` / ``LLMError`` execution failures into a
    concise structured failure (exit 2) with no stack trace.

Deliberately out of scope: real-Qwen smoke, real-novel extraction, concurrency,
server lifecycle management, A3E-C.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

import short_drama.cli as cli
from short_drama.llm import LLMTransportError
from short_drama.paths import REPO_ROOT
from short_drama.story import TOKEN_COUNTER_ID
from test_story_a3e_a_batch import (
    CHUNK_PROFILE_ID,
    PROJECT,
    BatchFakeLLMClient,
    setup_a1_a2,
    valid_payload_for,
)


# ---------------------------------------------------------------------------
# File writers (minimal, valid inputs for the CLI)
# ---------------------------------------------------------------------------


def _write_project(project_dir: Path, text: str) -> Path:
    source_dir = project_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "novel.txt").write_text(text, encoding="utf-8")
    project = {
        "schema_version": 1,
        "project_id": PROJECT,
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


def _write_chunk_profile(path: Path) -> Path:
    # Must exactly equal the profile used to build the A2 manifest in
    # setup_a1_a2 (make_chunk_profile) so the batch authority accepts it.
    profile = {
        "schema_version": 1,
        "profile_id": CHUNK_PROFILE_ID,
        "token_counter": TOKEN_COUNTER_ID,
        "ownership_token_budget": 8,
        "context_overlap_token_budget": 4,
        "context_token_budget": 24,
    }
    path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return path


def _write_runtime_config(path: Path) -> Path:
    config = {
        "schema_version": 1,
        "transport_id": "llm-local",
        "base_url": "http://127.0.0.1:8080/v1",
        "credential_environment_name": None,
        "timeout_seconds": 120,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["short-drama", *args])
    return cli.main()


def _fake_client_factory(chunks):
    # Provider-neutral fake: one valid payload per chunk, never touches the wire.
    return lambda _cfg: BatchFakeLLMClient([valid_payload_for(c) for c in chunks])


# ===========================================================================
# Parser / argument contract
# ===========================================================================


def test_extract_chunks_command_exists(monkeypatch, capsys):
    # A valid subcommand is accepted by argparse (an invalid one would raise
    # SystemExit(2) before returning). Execution then fails on the missing
    # project with a concise structured failure and a non-zero exit code.
    rc = _run_cli(monkeypatch, "extract-chunks", "nope.yaml", "--chunk-profile", "nope.yaml")
    assert rc == 2
    data = json.loads(capsys.readouterr().out)
    assert data["valid"] is False
    assert "error" in data


def test_extract_chunks_requires_project(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "extract-chunks", "--chunk-profile", "p.yaml")
    assert exc.value.code == 2


def test_extract_chunks_requires_chunk_profile(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "extract-chunks", "p.yaml")
    assert exc.value.code == 2


def test_extract_chunks_profile_defaults_resolve_to_repo_root():
    # The three tracked profile/config defaults are repo-root-safe (not CWD).
    assert cli.DEFAULT_EXTRACT_PROFILES["extraction_profile"] == (
        REPO_ROOT / "profiles" / "story_extraction_v1.yaml"
    )
    assert cli.DEFAULT_EXTRACT_PROFILES["runtime_config"] == (
        REPO_ROOT / "profiles" / "llm_local.yaml"
    )
    assert cli.DEFAULT_EXTRACT_PROFILES["llm_profile"] == (
        REPO_ROOT / "profiles" / "story_llm_qwen_v1.yaml"
    )
    # The chunk profile is always explicit (no default).
    assert "chunk_profile" not in cli.DEFAULT_EXTRACT_PROFILES


# ===========================================================================
# Successful composition (reaches the A3E-A batch authority)
# ===========================================================================


def test_extract_chunks_cli_success(tmp_path, monkeypatch, capsys):
    runs_root = tmp_path / "runs"
    story_root = runs_root / PROJECT / "story"
    _store, _pointers, _source_ref, _manifest, chunks = setup_a1_a2(story_root)
    assert len(chunks) >= 2

    project_file = _write_project(tmp_path / "proj", "占位文本。")
    chunk_profile_file = _write_chunk_profile(tmp_path / "chunk_profile.yaml")
    runtime_file = _write_runtime_config(tmp_path / "runtime.yaml")

    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", _fake_client_factory(chunks))

    # extraction + semantic profiles use the tracked defaults.
    rc = _run_cli(
        monkeypatch,
        "extract-chunks",
        str(project_file),
        "--runs-root", str(runs_root),
        "--chunk-profile", str(chunk_profile_file),
        "--runtime-config", str(runtime_file),
    )
    assert rc == 0

    data = json.loads(capsys.readouterr().out)
    assert data["chunks_total"] == len(chunks)
    assert data["chunks_generated"] == len(chunks)
    assert data["chunks_reused"] == 0
    assert data["chunks_failed"] == 0
    assert len(data["candidate_extraction_refs"]) == len(chunks)
    assert len(data["validation_report_refs"]) == len(chunks)
    # Refs are emitted in their existing canonical ArtifactRef representation.
    for ref in data["candidate_extraction_refs"]:
        assert set(ref) == {"artifact_type", "artifact_id", "revision", "content_hash"}
    for ref in data["validation_report_refs"]:
        assert set(ref) == {"artifact_type", "artifact_id", "revision", "content_hash"}
    # Extraction and validation refs are aligned element-wise in manifest order.
    for extraction_ref, report_ref in zip(
        data["candidate_extraction_refs"], data["validation_report_refs"]
    ):
        assert report_ref["artifact_id"].startswith(extraction_ref["artifact_id"])


# ===========================================================================
# Reuse visibility (counts preserved from A3E-A across a CLI rerun)
# ===========================================================================


def test_extract_chunks_cli_reuse_visibility(tmp_path, monkeypatch, capsys):
    runs_root = tmp_path / "runs"
    story_root = runs_root / PROJECT / "story"
    _store, _pointers, _source_ref, _manifest, chunks = setup_a1_a2(story_root)

    project_file = _write_project(tmp_path / "proj", "占位文本。")
    chunk_profile_file = _write_chunk_profile(tmp_path / "chunk_profile.yaml")
    runtime_file = _write_runtime_config(tmp_path / "runtime.yaml")

    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", _fake_client_factory(chunks))
    argv = (
        "extract-chunks",
        str(project_file),
        "--runs-root", str(runs_root),
        "--chunk-profile", str(chunk_profile_file),
        "--runtime-config", str(runtime_file),
    )

    first = _run_cli(monkeypatch, *argv)
    assert first == 0
    first_data = json.loads(capsys.readouterr().out)
    assert first_data["chunks_generated"] == len(chunks)
    assert first_data["chunks_reused"] == 0

    # Second CLI run reuses the per-chunk CURRENTs persisted by the first run.
    second = _run_cli(monkeypatch, *argv)
    assert second == 0
    second_data = json.loads(capsys.readouterr().out)
    assert second_data["chunks_reused"] == len(chunks)
    assert second_data["chunks_generated"] == 0
    assert second_data["chunks_failed"] == 0
    assert second_data["chunks_total"] == len(chunks)
    assert second_data["candidate_extraction_refs"] == first_data["candidate_extraction_refs"]


# ===========================================================================
# Failure behavior (structured JSON + exit 2, no traceback)
# ===========================================================================


def test_extract_chunks_story_error_is_structured_failure(tmp_path, monkeypatch, capsys):
    # Valid project + chunk profile + runtime config, but NO A1 state -> the
    # A3E-A batch authority fails closed with a StoryIntegrityError.
    runs_root = tmp_path / "runs"
    project_file = _write_project(tmp_path / "proj", "占位文本。")
    chunk_profile_file = _write_chunk_profile(tmp_path / "chunk_profile.yaml")
    runtime_file = _write_runtime_config(tmp_path / "runtime.yaml")

    rc = _run_cli(
        monkeypatch,
        "extract-chunks",
        str(project_file),
        "--runs-root", str(runs_root),
        "--chunk-profile", str(chunk_profile_file),
        "--runtime-config", str(runtime_file),
    )
    assert rc == 2
    out = capsys.readouterr().out
    data = json.loads(out)  # structured JSON, no traceback
    assert data["valid"] is False
    assert "not current" in data["error"]


def test_extract_chunks_llm_error_is_structured_failure(tmp_path, monkeypatch, capsys):
    runs_root = tmp_path / "runs"
    story_root = runs_root / PROJECT / "story"
    _store, _pointers, _source_ref, _manifest, chunks = setup_a1_a2(story_root)
    project_file = _write_project(tmp_path / "proj", "占位文本。")
    chunk_profile_file = _write_chunk_profile(tmp_path / "chunk_profile.yaml")
    runtime_file = _write_runtime_config(tmp_path / "runtime.yaml")

    # The provider raises a transport error on the first chunk; the batch
    # propagates it unchanged and the CLI converts it to a structured failure.
    monkeypatch.setattr(
        cli,
        "OpenAICompatibleLLMClient",
        lambda _cfg: BatchFakeLLMClient([LLMTransportError("connection refused")]),
    )

    rc = _run_cli(
        monkeypatch,
        "extract-chunks",
        str(project_file),
        "--runs-root", str(runs_root),
        "--chunk-profile", str(chunk_profile_file),
        "--runtime-config", str(runtime_file),
    )
    assert rc == 2
    out = capsys.readouterr().out
    data = json.loads(out)  # structured JSON, no traceback
    assert data["valid"] is False
    assert "connection refused" in data["error"]
