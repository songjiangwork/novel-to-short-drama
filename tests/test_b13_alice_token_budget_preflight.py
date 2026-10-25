"""Offline, deterministic tests for the B1.3 Alice tokenizer & budget preflight.

No network, no generation, no production-run tree required.  Discovery and
tokenizer paths use an injected fake transport; the budget paths use the
deterministic B1.3 preparation fixture from ``test_b13_disposition_preparation``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import b13_alice_token_budget_preflight as pf  # noqa: E402

from short_drama.llm import GenerationExecutionOptions, RuntimeConfig  # noqa: E402
from test_b13_disposition_preparation import _published_strategy  # noqa: E402

REQUEST_MODEL = "test-qwen"
BASE = "http://srv"
_CEILING = 10_000_000


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(
        schema_version=2, transport_id="pf-test", base_url="http://127.0.0.1:8080/v1",
        request_model=REQUEST_MODEL, provider_family="qwen",
        credential_environment_name=None, timeout_seconds=1.0)


def _fake_transport(responses, base=BASE):
    def transport(method, url, body, timeout, max_bytes):
        value = responses.get((method, url.split(base, 1)[1]), (404, b"not found"))
        if isinstance(value, int):
            return value, b""
        return value[0], json.dumps(value[1]).encode("utf-8")
    return transport


def _client(responses) -> pf.LlamaClient:
    return pf.LlamaClient(base_url=BASE, transport=_fake_transport(responses))


def _responses(model=REQUEST_MODEL, slots=(163840, 163840, 163840, 163840), n_ctx=163840,
               tokens=5000, template="T"):
    return {
        ("GET", "/v1/models"): (200, {"models": [{"model": model}],
                                      "data": [{"id": model, "meta": {"n_ctx": n_ctx}}]}),
        ("GET", "/slots"): (200, [{"id": i, "n_ctx": n} for i, n in enumerate(slots)]),
        ("GET", "/props"): (200, {"total_slots": len(slots),
                                  "default_generation_settings": {"n_ctx": n_ctx},
                                  "chat_template": template}),
        ("POST", "/tokenize"): (200, {"tokens": list(range(tokens))}),
    }


def _preparation(tmp_path):
    tree, semantic, publication = _published_strategy(tmp_path)
    packet = semantic.preparation.packet
    return tree, packet.source_snapshot, publication, packet.resolved_target.adaptation_target_id


def _tree_files(*roots):
    return sorted(str(p.relative_to(r)) for r in roots for p in Path(r).rglob("*") if p.is_file())


def _preflight(tmp_path, *, tokens=5000, responses=None, client=None,
               input_ceiling_tokens=_CEILING, model_context_tokens=None):
    """Run the full preflight against the B1.3 fixture; return (tree, client, report, files_before)."""
    tree, source, publication, target_id = _preparation(tmp_path)
    client = client or _client(responses if responses is not None else _responses(tokens=tokens))
    files_before = _tree_files(tree.store.root, tree.pointers.root)
    report = pf.run_preflight(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        runtime_config=_runtime(), client=client, input_ceiling_tokens=input_ceiling_tokens,
        model_context_tokens=model_context_tokens, renderer=lambda t, m: "X",
        strategy_ref=publication.strategy_ref, adaptation_target_id=target_id)
    return tree, client, report, files_before


# --- Scope A: runtime discovery --------------------------------------------

def test_discovery_reports_model_and_min_per_slot_nctx():
    capacity = pf.discover_runtime(_client(_responses()), REQUEST_MODEL)
    assert capacity.ok and capacity.model_matched and capacity.per_slot_n_ctx == (163840,) * 4
    assert capacity.effective_n_ctx == 163840 and capacity.n_ctx_source == "slots"
    assert capacity.total_slots == 4 and capacity.tokenizer_supported and capacity.chat_template == "T"
    assert _client(_responses()).generation_calls == []


def test_discovery_effective_nctx_is_min_across_slots_not_division():
    capacity = pf.discover_runtime(_client(_responses(slots=(163840, 90000, 163840, 163840))), REQUEST_MODEL)
    assert capacity.effective_n_ctx == 90000 and capacity.n_ctx_source == "slots"


def test_discovery_model_mismatch_is_flagged():
    capacity = pf.discover_runtime(_client(_responses(model="other-model")), REQUEST_MODEL)
    assert not capacity.model_matched and not capacity.ok
    assert any("does not match" in n for n in capacity.notes)


def test_discovery_missing_context_degrades_to_unknown():
    responses = _responses()
    responses[("GET", "/v1/models")] = (200, {"models": [{"model": REQUEST_MODEL}], "data": [{"id": REQUEST_MODEL}]})
    responses[("GET", "/slots")] = 500  # endpoint down; no meta/default n_ctx either
    responses[("GET", "/props")] = (200, {"total_slots": 4})
    capacity = pf.discover_runtime(_client(responses), REQUEST_MODEL)
    assert capacity.model_matched and capacity.effective_n_ctx is None and not capacity.ok


def test_discovery_inconsistent_slots_fall_back_to_models():
    responses = _responses(n_ctx=55)
    responses[("GET", "/slots")] = (200, [{"id": 0, "n_ctx": 55}, {"id": 1, "n_ctx": 0}])
    capacity = pf.discover_runtime(_client(responses), REQUEST_MODEL)
    assert capacity.effective_n_ctx == 55 and capacity.n_ctx_source == "models"


def test_discovery_unavailable_server_is_unknown():
    def down(method, url, body, timeout, max_bytes):
        raise pf.LlamaHTTPError("connection failure", 0)
    capacity = pf.discover_runtime(pf.LlamaClient(base_url=BASE, transport=down), REQUEST_MODEL)
    assert not capacity.ok and capacity.effective_n_ctx is None
    assert not capacity.tokenizer_supported and capacity.notes


# --- Scope B: tokenizer -----------------------------------------------------

def _count(responses, template="T", renderer=None):
    return pf.count_prompt_tokens(_client(responses), [{"role": "user", "content": "hi"}],
                                  template, renderer=renderer or (lambda t, m: "X"))


def test_tokenizer_counts_prompt_tokens():
    assert _count(_responses(tokens=5000)) == (5000, None)


def test_tokenizer_missing_template_is_unknown():
    count, reason = _count(_responses(), template=None)
    assert count is None and "chat template" in reason


def test_tokenizer_malformed_response_is_unknown():
    responses = _responses()
    responses[("POST", "/tokenize")] = (200, {"not_tokens": True})
    count, reason = _count(responses)
    assert count is None and "malformed" in reason


def test_tokenizer_endpoint_unavailable_is_unknown():
    responses = _responses()
    responses[("POST", "/tokenize")] = 404
    count, reason = _count(responses)
    assert count is None and "unavailable" in reason


def test_chat_template_render_failure_is_unknown():
    def boom(template, messages):
        raise pf.ChatTemplateRenderError("boom")
    count, reason = _count(_responses(), renderer=boom)
    assert count is None and "render unavailable" in reason


def test_default_renderer_renders_with_jinja2():
    pytest.importorskip("jinja2")
    assert pf.default_chat_template_renderer(
        "{%- for m in messages %}{{ m.content }}{%- endfor %}", [{"role": "user", "content": "hi"}]) == "hi"


# --- Scope C: budget, regeneration, identity, safety ------------------------

def test_pass_when_all_evidence_verifiable(tmp_path):
    _, _, report, _ = _preflight(tmp_path)
    assert report["windows_total"] >= 1 and report["overall_verdict"] == "PASS"
    assert all(row["verdict"] == "PASS" for row in report["windows"])


def test_initial_and_regeneration_request_handling(tmp_path):
    tree, client, report, files_before = _preflight(tmp_path)
    for row in report["windows"]:
        # regeneration adds the cache_prompt echo, so its body is never smaller
        assert row["regen_provider_request_bytes"] >= row["initial_provider_request_bytes"]
        assert row["tokenizer_prompt_tokens"] == 5000 and row["reserved_output_tokens"] >= 0
    assert client.generation_calls == []  # no generation calls
    assert _tree_files(tree.store.root, tree.pointers.root) == files_before  # no writes


def test_stable_semantic_identities(tmp_path):
    """Same packet + runtime config -> same identity; initial and regeneration match."""
    tree, source, publication, target_id = _preparation(tmp_path)
    packet = pf.prepare_b13_disposition_windows(
        store=tree.store, pointers=tree.pointers, source_snapshot=source,
        adaptation_target_id=target_id, strategy_ref=publication.strategy_ref).packets[0]
    runtime = _runtime()
    first = pf.build_b13_window_semantic_preparation(packet, runtime_config=runtime)
    second = pf.build_b13_window_semantic_preparation(packet, runtime_config=runtime)
    regen = pf.build_b13_window_semantic_preparation(
        packet, runtime_config=runtime,
        execution_options=GenerationExecutionOptions(prompt_context_reuse="disabled"))
    assert first.request_identity == second.request_identity
    assert first.request_identity == regen.request_identity  # stable across request shapes


def test_input_ceiling_exceeded_is_fail(tmp_path):
    _, _, report, _ = _preflight(tmp_path, input_ceiling_tokens=1)
    assert all(r["estimator_budget"] == "FAIL" and r["verdict"] == "FAIL" for r in report["windows"])
    assert report["overall_verdict"] == "FAIL"


def test_total_context_exceeded_is_fail(tmp_path):
    _, _, report, _ = _preflight(tmp_path, model_context_tokens=1000)  # 1000 < input + 8192 reserve
    assert all(r["estimator_budget"] == "FAIL" and r["verdict"] == "FAIL" for r in report["windows"])


def test_overall_unknown_when_tokenizer_unsupported(tmp_path):
    responses = _responses()
    responses[("POST", "/tokenize")] = 404
    _, _, report, _ = _preflight(tmp_path, responses=responses)
    assert report["capacity"]["tokenizer_supported"] is False
    assert all(r["estimator_budget"] == "PASS" and r["tokenizer_budget"] == "UNKNOWN"
               and r["verdict"] == "UNKNOWN" for r in report["windows"])
    assert report["overall_verdict"] == "UNKNOWN"  # never fabricates a PASS


def test_report_is_bounded_and_generation_free(tmp_path):
    _, client, report, _ = _preflight(tmp_path)
    assert report["read_only"] is True and report["generation_calls"] == []
    assert report["windows_total"] == len(report["windows"])
    row = report["windows"][0]
    for key in ("window_id", "window_ordinal", "owned_event_count", "context_event_count",
                "packet_bytes", "packet_estimated_tokens", "initial_provider_request_bytes",
                "initial_provider_request_estimated_tokens", "regen_provider_request_bytes",
                "regen_provider_request_estimated_tokens", "reserved_output_tokens",
                "tokenizer_prompt_tokens", "request_identity", "estimator_budget",
                "tokenizer_budget", "verdict"):
        assert key in row
