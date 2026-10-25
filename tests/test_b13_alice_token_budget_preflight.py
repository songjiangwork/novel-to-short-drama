"""Offline, deterministic tests for the B1.3 Alice tokenizer & budget preflight.

No network, no generation, no production-run tree required.  Discovery,
tokenizer, and budget paths use an injected fake transport (including the
authoritative ``/apply-template`` + ``/tokenize`` pair); the budget paths use the
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


def _runtime(base=BASE) -> RuntimeConfig:
    # base URL carries the /v1 prefix so its server root equals the client base.
    return RuntimeConfig(
        schema_version=2, transport_id="pf-test", base_url=f"{base}/v1",
        request_model=REQUEST_MODEL, provider_family="qwen",
        credential_environment_name=None, timeout_seconds=1.0)


def _fake_transport(responses, base=BASE):
    def transport(method, url, body, timeout, max_bytes):
        value = responses.get((method, url.split(base, 1)[1]), (404, b"not found"))
        if isinstance(value, int):
            return value, b""
        status, payload = value
        if isinstance(payload, (bytes, bytearray)):
            return status, bytes(payload)
        return status, json.dumps(payload).encode("utf-8")
    return transport


def _client(responses, base=BASE) -> pf.LlamaClient:
    return pf.LlamaClient(base_url=base, transport=_fake_transport(responses, base=base))


def _responses(model=REQUEST_MODEL, slots=(163840, 163840, 163840, 163840), n_ctx=163840,
               total_slots=None, tokens=5000, prompt="PROMPT", template=True):
    if total_slots is None:
        total_slots = len(slots)
    r = {
        ("GET", "/v1/models"): (200, {"models": [{"model": model}],
                                      "data": [{"id": model, "meta": {"n_ctx": n_ctx}}]}),
        ("GET", "/slots"): (200, [{"id": i, "n_ctx": n} for i, n in enumerate(slots)]),
        ("GET", "/props"): (200, {"total_slots": total_slots,
                                  "default_generation_settings": {"n_ctx": n_ctx}}),
    }
    if template:
        r[("POST", "/apply-template")] = (200, {"prompt": prompt})
        r[("POST", "/tokenize")] = (200, {"tokens": list(range(tokens))})
    return r


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
        model_context_tokens=model_context_tokens,
        strategy_ref=publication.strategy_ref, adaptation_target_id=target_id)
    return tree, client, report, files_before


# --- Scope A: runtime discovery --------------------------------------------

def test_discovery_reports_model_and_min_per_slot_nctx():
    capacity = pf.discover_runtime(_client(_responses()), REQUEST_MODEL)
    assert capacity.model_matched and capacity.model_identity_established
    assert capacity.per_slot_n_ctx == (163840,) * 4 and capacity.slots_trusted
    assert capacity.effective_n_ctx == 163840 and capacity.tokenizer_supported
    assert _client(_responses()).generation_calls == []


def test_discovery_effective_nctx_is_min_across_slots_not_division():
    capacity = pf.discover_runtime(_client(_responses(slots=(163840, 90000, 163840, 163840))), REQUEST_MODEL)
    assert capacity.effective_n_ctx == 90000 and capacity.slots_trusted


def test_discovery_model_matched_uses_full_served_list():
    ok = pf.discover_runtime(_client(_responses()), REQUEST_MODEL)
    assert ok.model_matched  # request_model is the (and a) served id
    # a non-first served id also matches (no first-entry shortcut); an unknown one does not
    other = pf.discover_runtime(_client(_responses()), "unrelated-model")
    assert not other.model_matched and other.model_identity_established


def test_discovery_unverified_context_is_not_trusted():
    responses = _responses()
    responses[("GET", "/slots")] = 500  # slots unavailable; only untrusted metadata present
    capacity = pf.discover_runtime(_client(responses), REQUEST_MODEL)
    assert not capacity.slots_trusted and capacity.effective_n_ctx is None
    assert capacity.metadata_n_ctx == 163840  # reported, but never promoted to trusted capacity


def test_discovery_unavailable_server_is_unknown():
    def down(method, url, body, timeout, max_bytes):
        raise pf.LlamaHTTPError("connection failure", 0)
    capacity = pf.discover_runtime(pf.LlamaClient(base_url=BASE, transport=down), REQUEST_MODEL)
    assert not capacity.model_matched and not capacity.slots_trusted
    assert capacity.effective_n_ctx is None and capacity.notes


# --- Scope A, full path: model identity & server (P1-A) --------------------

def test_overall_not_pass_on_model_mismatch(tmp_path):
    _, _, report, _ = _preflight(tmp_path, responses=_responses(model="other-model"))
    assert report["capacity"]["model_matched"] is False
    assert report["policy"]["capacity_trusted"] is False
    assert report["overall_verdict"] != "PASS"


def test_overall_not_pass_when_model_identity_not_established(tmp_path):
    responses = _responses()
    responses[("GET", "/v1/models")] = (200, {"models": []})
    _, _, report, _ = _preflight(tmp_path, responses=responses)
    assert report["capacity"]["model_identity_established"] is False
    assert report["overall_verdict"] != "PASS"


def test_overall_not_pass_when_discovery_client_is_a_different_server(tmp_path):
    tree, source, publication, target_id = _preparation(tmp_path)
    client = _client(_responses(), base="http://other-server")
    report = pf.run_preflight(store=tree.store, pointers=tree.pointers, source_snapshot=source,
        runtime_config=_runtime(), client=client, input_ceiling_tokens=_CEILING,
        strategy_ref=publication.strategy_ref, adaptation_target_id=target_id)
    assert report["capacity"]["server_ok"] is False
    assert report["policy"]["capacity_trusted"] is False
    assert report["overall_verdict"] != "PASS"


# --- Scope B: authoritative tokenizer (P1-D) -------------------------------

def _count(responses, prompt="PROMPT"):
    return pf.count_prompt_tokens(_client(responses), [{"role": "user", "content": "hi"}])


def test_tokenizer_counts_prompt_tokens():
    assert _count(_responses(tokens=5000)) == (5000, None)


def test_tokenizer_apply_template_unavailable_is_unknown():
    count, reason = _count(_responses(template=False))  # no /apply-template
    assert count is None and "apply-template" in reason


def test_tokenizer_empty_prompt_is_unknown():
    responses = _responses(prompt="   ")
    count, reason = _count(responses)
    assert count is None and "empty" in reason


def test_tokenizer_invalid_responses_are_unknown():
    # tokenize endpoint down
    r = _responses()
    r[("POST", "/tokenize")] = 404
    assert _count(r)[0] is None and "unavailable" in _count(r)[1]
    # malformed: no token list
    r = _responses()
    r[("POST", "/tokenize")] = (200, {"not_tokens": True})
    assert _count(r)[0] is None and "malformed" in _count(r)[1]
    # invalid element type
    r = _responses()
    r[("POST", "/tokenize")] = (200, {"tokens": [1, "two", 3]})
    assert _count(r)[0] is None and "invalid element" in _count(r)[1]
    # empty token list for a nonempty prompt
    r = _responses(prompt="PROMPT")
    r[("POST", "/tokenize")] = (200, {"tokens": []})
    assert _count(r)[0] is None and "empty" in _count(r)[1]


def test_tokenizer_full_path_unsupported_is_unknown(tmp_path):
    _, _, report, _ = _preflight(tmp_path, responses=_responses(template=False))
    assert report["capacity"]["tokenizer_supported"] is False
    assert all(row["tokenizer_budget"] == "UNKNOWN" and row["verdict"] == "UNKNOWN"
               for row in report["windows"])
    assert report["overall_verdict"] == "UNKNOWN"  # never a fabricated PASS


# --- Scope C: budget, regeneration, identity, safety ------------------------

def test_pass_when_all_evidence_verifiable(tmp_path):
    _, _, report, _ = _preflight(tmp_path)
    assert report["windows_total"] >= 1 and report["policy"]["capacity_trusted"] is True
    assert report["overall_verdict"] == "PASS"
    assert all(row["verdict"] == "PASS" for row in report["windows"])


def test_initial_and_regeneration_request_handling(tmp_path):
    tree, client, report, files_before = _preflight(tmp_path)
    for row in report["windows"]:
        assert row["regen_provider_request_bytes"] >= row["initial_provider_request_bytes"]
        assert row["tokenizer_prompt_tokens"] == 5000 and row["reserved_output_tokens"] >= 0
    assert client.generation_calls == []  # no generation calls
    assert _tree_files(tree.store.root, tree.pointers.root) == files_before  # no writes


def test_stable_semantic_identities(tmp_path):
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
    _, _, report, _ = _preflight(tmp_path, model_context_tokens=1000)  # 1000 < input + reserve
    assert all(r["estimator_budget"] == "FAIL" and r["verdict"] == "FAIL" for r in report["windows"])


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


# --- P1-B: fail closed on inconsistent slot evidence ------------------------

def test_missing_slots_unknown(tmp_path):
    responses = _responses()
    responses[("GET", "/slots")] = 500
    _, _, report, _ = _preflight(tmp_path, responses=responses)
    assert report["capacity"]["slots_trusted"] is False
    assert report["overall_verdict"] != "PASS"


def test_malformed_slots_unknown(tmp_path):
    responses = _responses()
    responses[("GET", "/slots")] = (200, [{"id": 0, "n_ctx": 100}, {"id": 1, "n_ctx": 0}])
    _, _, report, _ = _preflight(tmp_path, responses=responses)
    assert report["capacity"]["slots_trusted"] is False
    assert report["overall_verdict"] != "PASS"


def test_duplicate_slot_identity_unknown(tmp_path):
    responses = _responses(total_slots=2)
    responses[("GET", "/slots")] = (200, [{"id": 0, "n_ctx": 100}, {"id": 0, "n_ctx": 100}])
    _, _, report, _ = _preflight(tmp_path, responses=responses)
    assert report["capacity"]["slots_trusted"] is False
    assert report["overall_verdict"] != "PASS"


def test_inconsistent_total_slots_unknown(tmp_path):
    responses = _responses(slots=(100, 200), total_slots=4)  # 2 slots vs declared 4
    _, _, report, _ = _preflight(tmp_path, responses=responses)
    assert report["capacity"]["slots_trusted"] is False
    assert report["overall_verdict"] != "PASS"


def test_conflicting_model_metadata_not_promoted(tmp_path):
    responses = _responses()
    responses[("GET", "/slots")] = 500  # slots unavailable
    responses[("GET", "/v1/models")] = (200, {"models": [{"model": REQUEST_MODEL}],
                                              "data": [{"id": REQUEST_MODEL, "meta": {"n_ctx": 999999}}]})
    responses[("GET", "/props")] = (200, {"total_slots": 4, "default_generation_settings": {"n_ctx": 888888}})
    _, _, report, _ = _preflight(tmp_path, responses=responses)
    assert report["capacity"]["slots_trusted"] is False
    assert report["capacity"]["discovered_effective_n_ctx"] is None  # metadata not promoted
    assert report["capacity"]["metadata_n_ctx"] == 999999
    assert report["overall_verdict"] != "PASS"


# --- P1-C: unsafe n_ctx override handling ----------------------------------

def test_override_cannot_raise_verified_capacity(tmp_path):
    _, _, report, _ = _preflight(tmp_path, model_context_tokens=200000)  # > discovered 163840
    assert report["policy"]["model_context_tokens"] is None
    assert report["policy"]["model_context_tokens_source"] == "operator_override_rejected_exceeds_discovered"
    assert report["policy"]["capacity_trusted"] is False
    assert report["overall_verdict"] != "PASS"


def test_override_can_lower_verified_capacity(tmp_path):
    _, _, report, _ = _preflight(tmp_path, model_context_tokens=100000)  # < discovered 163840
    assert report["policy"]["model_context_tokens"] == 100000
    assert report["policy"]["model_context_tokens_source"] == "operator_override_capped_at_discovered"
    assert report["policy"]["capacity_trusted"] is True
    assert report["overall_verdict"] == "PASS"


def test_override_alone_does_not_trust_unverified_discovery(tmp_path):
    responses = _responses()
    responses[("GET", "/slots")] = 500  # discovery unverified
    _, _, report, _ = _preflight(tmp_path, responses=responses, model_context_tokens=100000)
    assert report["policy"]["model_context_tokens"] == 100000
    assert report["policy"]["model_context_tokens_source"] == "operator_override_discovery_unverified"
    assert report["policy"]["capacity_trusted"] is False
    assert report["overall_verdict"] != "PASS"


# --- P2-A: documented exit codes -------------------------------------------

def test_exit_code_by_verdict():
    assert pf._exit_code({"overall_verdict": "PASS"}) == 0
    assert pf._exit_code({"overall_verdict": "FAIL"}) == 2
    assert pf._exit_code({"overall_verdict": "UNKNOWN"}) == 3  # never authorization to generate
    assert pf._exit_code({}) == 1
