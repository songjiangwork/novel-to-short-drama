"""Offline/read-only smoke test for the manual A5 concurrency benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "a5_concurrency_benchmark.py"
    spec = importlib.util.spec_from_file_location("a5_concurrency_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_uses_fake_client_and_leaves_existing_run_tree_unchanged(tmp_path, monkeypatch, capsys):
    benchmark = _benchmark_module()
    project_id = "demo"
    (tmp_path / project_id / "story" / "artifacts").mkdir(parents=True)
    (tmp_path / project_id / "story" / "pointers").mkdir()
    artifact = tmp_path / project_id / "story" / "artifacts" / "existing.json"
    artifact.write_text('{"immutable":true}', encoding="utf-8")

    blocks = tuple(
        SimpleNamespace(block_id=f"event-{index}", block_ordinal=index)
        for index in range(3)
    )
    requests = tuple(
        SimpleNamespace(
            rendered_prompt=SimpleNamespace(
                to_dict=lambda index=index: {"messages": ["x" * (index + 1)]}
            )
        )
        for index in range(3)
    )
    preparation = SimpleNamespace(blocks=blocks, structured_requests=requests)
    class FakeClient:
        supports_concurrent_calls = True

    fake_client = FakeClient()
    fake_result = lambda semantic_round: SimpleNamespace(
        semantic_rounds=semantic_round, generation_provenance=SimpleNamespace(usage=None)
    )
    monkeypatch.setattr(benchmark, "_load_project", lambda _path: (Path("p"), {"project_id": project_id}))
    monkeypatch.setattr(benchmark, "FileArtifactStore", lambda root: object())
    monkeypatch.setattr(benchmark, "FilePointerStore", lambda root, store: object())
    monkeypatch.setattr(benchmark, "build_consolidation_planning", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark, "load_consolidation_profile", lambda _path: object())
    monkeypatch.setattr(benchmark, "load_semantic_profile", lambda _path: object())
    monkeypatch.setattr(benchmark, "PromptRegistry", lambda _path: object())
    monkeypatch.setattr(benchmark, "build_event_semantic_preparation", lambda *args, **kwargs: preparation)
    monkeypatch.setattr(benchmark, "load_runtime_config", lambda _path: object())
    monkeypatch.setattr(benchmark, "OpenAICompatibleLLMClient", lambda _config: fake_client)
    coordinator_calls = []
    attempt_rounds = []

    def fake_attempt(planning_result, semantic_profile, client, block, request, *, semantic_round):
        assert planning_result is not None
        assert semantic_profile is not None
        assert client is fake_client
        assert semantic_round == 1
        attempt_rounds.append((block.block_id, semantic_round))
        return fake_result(semantic_round)

    real_two_stage = benchmark._execute_two_stage_semantic_blocks

    def recording_two_stage(blocks, requests, execute_one_round, *, llm_client, max_concurrency):
        assert llm_client is fake_client
        assert max_concurrency == 2
        coordinator_calls.append((blocks, requests))
        return real_two_stage(
            blocks, requests, execute_one_round,
            llm_client=llm_client, max_concurrency=max_concurrency,
        )

    monkeypatch.setattr(benchmark, "_attempt_event_semantic_block", fake_attempt)
    monkeypatch.setattr(benchmark, "_execute_two_stage_semantic_blocks", recording_two_stage)
    monkeypatch.setattr(
        sys, "argv", ["a5_concurrency_benchmark.py", "project.yaml", "--runs-root", str(tmp_path), "--max-concurrency", "2"]
    )

    assert benchmark.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["read_only_verified"] is True
    assert report["requests_completed"] == 3
    assert len(coordinator_calls) == 1
    selected_blocks, selected_requests = coordinator_calls[0]
    # The benchmark selects its largest prompts first, but the complete
    # selected set must enter one shared two-stage coordinator invocation.
    assert tuple(block.block_id for block in selected_blocks) == (
        "event-2", "event-1", "event-0"
    )
    assert selected_requests == tuple(reversed(requests))
    assert sorted(attempt_rounds) == [
        ("event-0", 1), ("event-1", 1), ("event-2", 1)
    ]
    assert artifact.read_text(encoding="utf-8") == '{"immutable":true}'
