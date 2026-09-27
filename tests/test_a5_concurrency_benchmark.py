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
    fake_client = object()
    fake_result = lambda: SimpleNamespace(
        semantic_rounds=1, generation_provenance=SimpleNamespace(usage=None)
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
    monkeypatch.setattr(benchmark, "_execute_event_semantic_block", lambda *args: fake_result())

    def fake_executor(blocks, requests, execute, *, llm_client, max_concurrency):
        assert llm_client is fake_client
        assert max_concurrency == 2
        return tuple(execute(block, request) for block, request in zip(blocks, requests))

    monkeypatch.setattr(benchmark, "_execute_prepared_blocks", fake_executor)
    monkeypatch.setattr(
        sys, "argv", ["a5_concurrency_benchmark.py", "project.yaml", "--runs-root", str(tmp_path), "--max-concurrency", "2"]
    )

    assert benchmark.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["read_only_verified"] is True
    assert report["requests_completed"] == 3
    assert artifact.read_text(encoding="utf-8") == '{"immutable":true}'
