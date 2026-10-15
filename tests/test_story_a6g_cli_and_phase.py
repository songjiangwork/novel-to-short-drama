"""A6G CLI and Phase-A wrapper seams (offline)."""
from __future__ import annotations

import json
import sys

import pytest

import short_drama.cli as cli
import short_drama.story.phase_a_service as phase
from short_drama.llm import LLMError
from short_drama.paths import REPO_ROOT
from short_drama.story import StoryError


class _Result:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return self.value


def _run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["short-drama", *args])
    return cli.main()


def test_build_story_bible_cli_uses_tracked_defaults(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(cli, "load_runtime_config", lambda path: seen.setdefault("runtime", path) or "runtime")
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda config: seen.setdefault("client", config) or "client")
    monkeypatch.setattr(cli, "build_story_bible_project", lambda project, **kwargs: (
        seen.update(project=project, **kwargs) or _Result({"ok": True})
    ))

    assert _run_cli(monkeypatch, "build-story-bible", "project.yaml") == 0
    assert seen["project"] == "project.yaml"
    assert seen["runs_root"] == "runs"
    assert seen["consolidation_profile_id"] == "consolidation-v1"
    assert seen["story_analysis_profile_path"] == REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
    assert seen["semantic_profile_path"] == REPO_ROOT / "profiles" / "story_analysis_llm_v1.yaml"
    assert seen["runtime"] == REPO_ROOT / "profiles" / "llm_local.yaml"
    assert json.loads(capsys.readouterr().out) == {"ok": True}


@pytest.mark.parametrize("error", [StoryError("closed"), LLMError("closed")])
def test_build_story_bible_cli_fails_closed(monkeypatch, capsys, error):
    monkeypatch.setattr(cli, "load_runtime_config", lambda path: "runtime")
    monkeypatch.setattr(cli, "OpenAICompatibleLLMClient", lambda config: "client")
    monkeypatch.setattr(cli, "build_story_bible_project", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    assert _run_cli(monkeypatch, "build-story-bible", "project.yaml") == 2
    assert json.loads(capsys.readouterr().out) == {"valid": False, "error": "closed"}


def test_analyze_story_runs_existing_stages_in_exact_order(monkeypatch):
    calls = []

    def stage(name):
        def run(*args, **kwargs):
            calls.append((name, kwargs))
            return {"stage": name}
        return run

    monkeypatch.setattr(phase, "ingest_source_project", stage("a1"))
    monkeypatch.setattr(phase, "plan_chunks_project", stage("a2"))
    monkeypatch.setattr(phase, "extract_chunks_project", stage("a3"))
    monkeypatch.setattr(phase, "reconcile_entities_project", stage("a4"))
    monkeypatch.setattr(phase, "consolidate_evidence_project", stage("a5"))
    monkeypatch.setattr(phase, "build_story_bible_project", stage("a6"))

    result = phase.analyze_story_project(
        "project.yaml", runs_root="runs", chunk_profile_path="chunk.yaml",
        extraction_profile_path="extract.yaml", extraction_semantic_profile_path="extract-llm.yaml",
        reconciliation_profile_path="reconcile.yaml", reconciliation_semantic_profile_path="reconcile-llm.yaml",
        reconciliation_profile_id="reconcile-v2", consolidation_profile_path="consolidate.yaml",
        consolidation_profile_id="consolidate-v1", consolidation_semantic_profile_path="consolidate-llm.yaml",
        story_analysis_profile_path="a6.yaml", story_analysis_semantic_profile_path="a6-llm.yaml",
        llm_client="client",
    )
    assert [name for name, _ in calls] == ["a1", "a2", "a3", "a4", "a5", "a6"]
    assert calls[4][1]["reconciliation_profile_id"] == "reconcile-v2"
    assert calls[5][1]["consolidation_profile_id"] == "consolidate-v1"
    assert result.to_dict()["build_story_bible"] == {"stage": "a6"}


def test_analyze_story_stops_on_first_stage_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(phase, "ingest_source_project", lambda *args, **kwargs: calls.append("a1") or {})
    monkeypatch.setattr(phase, "plan_chunks_project", lambda *args, **kwargs: calls.append("a2") or (_ for _ in ()).throw(StoryError("stop")))
    monkeypatch.setattr(phase, "extract_chunks_project", lambda *args, **kwargs: calls.append("a3") or {})
    with pytest.raises(StoryError, match="stop"):
        phase.analyze_story_project(
            "project.yaml", runs_root="runs", chunk_profile_path="chunk.yaml",
            extraction_profile_path="extract.yaml", extraction_semantic_profile_path="extract-llm.yaml",
            reconciliation_profile_path="reconcile.yaml", reconciliation_semantic_profile_path="reconcile-llm.yaml",
            reconciliation_profile_id="reconcile-v2", consolidation_profile_path="consolidate.yaml",
            consolidation_profile_id="consolidate-v1", consolidation_semantic_profile_path="consolidate-llm.yaml",
            story_analysis_profile_path="a6.yaml", story_analysis_semantic_profile_path="a6-llm.yaml",
            llm_client="client",
        )
    assert calls == ["a1", "a2"]
