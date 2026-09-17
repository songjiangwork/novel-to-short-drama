"""Offline tests for the A3D smoke's official-acceptance model guard.

The guard (``acceptance_model_guard``) must fail closed in acceptance mode (no
``--model``) when the configured endpoint is not verified to serve the EXACT
tracked semantic model, and must bypass the guard in diagnostic ``--model``
mode. These tests are fully offline: they exercise the pure guard function and
the ``run_smoke`` early-return (exit code 2) path with a mocked endpoint query.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from short_drama.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "a3_chunk_smoke.py"
TRACKED_MODEL = "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("a3_chunk_smoke_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke():
    return _load_smoke_module()


def _write_runtime_config(tmp_path: Path) -> str:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "schema_version: 2\n"
        "transport_id: llm-local\n"
        "base_url: http://127.0.0.1:8080/v1\n"
        "request_model: ggml-org/Qwen3.8-27B-GGUF:Q4_K_M\n"
        "provider_family: qwen\n"
        "credential_environment_name: null\n"
        "timeout_seconds: 120\n",
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# Pure guard function
# ---------------------------------------------------------------------------


def test_guard_passes_when_endpoint_serves_exact_tracked_model(smoke):
    assert (
        smoke.acceptance_model_guard(
            model_override=None,
            tracked_model=TRACKED_MODEL,
            server_model=TRACKED_MODEL,
        )
        is None
    )


def test_guard_blocks_when_served_model_differs(smoke):
    reason = smoke.acceptance_model_guard(
        model_override=None,
        tracked_model=TRACKED_MODEL,
        server_model="HauhauCS/some-other-model:Q4_K_P",
    )
    assert reason is not None
    assert "SMOKE BLOCKED_BY_RUNTIME_ENVIRONMENT" in reason
    assert TRACKED_MODEL in reason
    assert "HauhauCS/some-other-model:Q4_K_P" in reason


def test_guard_blocks_when_served_model_unknown(smoke):
    reason = smoke.acceptance_model_guard(
        model_override=None,
        tracked_model=TRACKED_MODEL,
        server_model=None,
    )
    assert reason is not None
    assert "SMOKE BLOCKED_BY_RUNTIME_ENVIRONMENT" in reason
    assert "served=unknown" in reason


def test_guard_bypassed_in_diagnostic_model_mode(smoke):
    # An explicit --model override deliberately tests a different served
    # identity; the guard must not block (even when the query is inconclusive).
    assert (
        smoke.acceptance_model_guard(
            model_override="HauhauCS/some-other-model:Q4_K_P",
            tracked_model=TRACKED_MODEL,
            server_model="HauhauCS/some-other-model:Q4_K_P",
        )
        is None
    )
    assert (
        smoke.acceptance_model_guard(
            model_override="HauhauCS/some-other-model:Q4_K_P",
            tracked_model=TRACKED_MODEL,
            server_model=None,
        )
        is None
    )


# ---------------------------------------------------------------------------
# run_smoke early-return path (acceptance mode, exit code 2)
# ---------------------------------------------------------------------------


def _run_smoke_acceptance(smoke, tmp_path, served_model) -> tuple[int, str]:
    runtime_path = _write_runtime_config(tmp_path)
    profiles = REPO_ROOT / "profiles"
    smoke.query_server_model = lambda base_url, credential_env: served_model
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = smoke.run_smoke(
            runtime_config_path=runtime_path,
            profile_path=str(profiles / "story_extraction_llm_v1.yaml"),
            extraction_profile_path=str(profiles / "story_extraction_v1.yaml"),
            model_override=None,
        )
    return rc, buffer.getvalue()


def test_run_smoke_acceptance_mode_blocks_on_wrong_served_model(smoke, tmp_path):
    rc, out = _run_smoke_acceptance(smoke, tmp_path, "HauhauCS/wrong:Q4_K_P")
    assert rc == 2
    assert "SMOKE BLOCKED_BY_RUNTIME_ENVIRONMENT" in out
    assert TRACKED_MODEL in out


def test_run_smoke_acceptance_mode_blocks_on_unknown_served_model(smoke, tmp_path):
    rc, out = _run_smoke_acceptance(smoke, tmp_path, None)
    assert rc == 2
    assert "SMOKE BLOCKED_BY_RUNTIME_ENVIRONMENT" in out


def test_run_smoke_acceptance_mode_proceeds_when_exact_tracked_served(
    smoke, tmp_path
):
    # When the endpoint serves the exact tracked model, the guard must NOT
    # block; run_smoke proceeds past the guard to generation. To keep this
    # offline, stub the provider client so generation succeeds deterministically.
    runtime_path = _write_runtime_config(tmp_path)
    profiles = REPO_ROOT / "profiles"
    smoke.query_server_model = lambda base_url, credential_env: TRACKED_MODEL

    import short_drama.llm as llm_mod

    class _StubClient(llm_mod.LLMClient):
        supported_structured_output_modes = frozenset({"json_schema"})

        def __init__(self):
            self.calls = 0

        def generate_structured(self, rendered_prompt, output_schema, semantic_profile):
            self.calls += 1
            request = llm_mod.build_structured_request(
                rendered_prompt=rendered_prompt,
                output_schema=output_schema,
                semantic_profile=semantic_profile,
            )
            # A valid single-character payload (primary evidence in ownership).
            parsed = {
                "characters": [
                    {
                        "candidate_id": "cand_char_001",
                        "display_name_original": "林晚",
                        "aliases_original": [],
                        "descriptors_zh": [],
                        "summary_zh": "林晚的简要描述。",
                        "evidence_strength": "explicit",
                        "evidence": [
                            {
                                "paragraph_id": "CH001_P0003",
                                "role": "primary",
                                "strength": "explicit",
                                "excerpt": None,
                            }
                        ],
                    }
                ],
                "locations": [],
                "facts": [],
                "events": [],
                "relationships": [],
                "unresolved_mentions": [],
            }
            return llm_mod.StructuredGenerationResult(
                parsed_json=parsed,
                provenance=llm_mod.build_provenance(
                    request,
                    llm_mod.ProviderMeta(),
                    request_model="qwen3-27b",
                    provider_family="qwen",
                ),
                attempts=1,
            )

    stub = _StubClient()
    smoke.OpenAICompatibleLLMClient = lambda runtime_config: stub

    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = smoke.run_smoke(
            runtime_config_path=runtime_path,
            profile_path=str(profiles / "story_extraction_llm_v1.yaml"),
            extraction_profile_path=str(profiles / "story_extraction_v1.yaml"),
            model_override=None,
        )
    out = buffer.getvalue()
    assert rc == 0
    assert "SMOKE BLOCKED_BY_RUNTIME_ENVIRONMENT" not in out
    assert stub.calls == 1
