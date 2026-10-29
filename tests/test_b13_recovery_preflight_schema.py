"""All evidence here is fabricated in memory; success proves no external fact."""
import copy
import hashlib
import json

import pytest

from short_drama.artifacts import canonical_json_bytes
from short_drama.adaptation.disposition_recovery_preflight_schema import (
    PreflightStructureError, check_preflight_structure,
)

H = "a" * 64
H2 = "b" * 64


def preflight():
    ref = {"artifact_type": "fake", "artifact_id": "fake", "revision": 1, "content_hash": H}
    node = {"path": "/synthetic/root", "dev": 0, "ino": 1, "type": "directory"}
    slot = {"status": "ABSENT", "identity": None, "bytes_sha256": None}
    return {
        "schema_version": 1, "record_type": "preflight", "created_at": "2025-01-01T00:00:00Z",
        "policy_ratification_sha256": H, "scope": {"incident_id": "synthetic", "window_id": "window_0001",
            "window_ordinal": 1, "claim_identity": H, "claim_bytes_sha256": H,
            "preparation_identity": H, "request_identity": H},
        "operator_id": "synthetic", "reason": "synthetic", "attempt_id": "a" * 32,
        "claim": {"file": {**node, "type": "regular", "bytes_sha256": H, "size": 1}, "state": "CLAIMED_UNRESOLVED"},
        "authority": {"source_snapshot_sha256": H, "a5_manifest": ref, "a6_manifest": ref,
            "a6_validation": ref, "a6_current_pointer": ref, "strategy": ref, "strategy_hash": H,
            "selection_mode": "CURRENT", "b12_current_pointer": ref},
        "roots": {k: copy.deepcopy(node) for k in ("execution", "checkpoint", "artifact", "pointer", "checkpoint_parent")},
        "root_result": None, "checkpoint": {"status": "ABSENT", "expected_path": "/synthetic/checkpoint"},
        "slots": {k: copy.deepcopy(slot) for k in ("root_intent", "root_result", "decision", "consumption", "terminal")},
        "runtime": {"observed_at": "2025-01-01T00:00:00Z", "provider_config_id": "fake",
            "provider_config_sha256": H, "model_id": "fake", "model_config_sha256": H,
            "tokenizer_id": "fake", "tokenizer_sha256": H, "effective_n_ctx": 100,
            "input_ceiling_tokens": 80, "reserved_output_tokens": 20, "estimated_input_tokens": 80,
            "provider_request_bytes": 1, "profile_id": "fake", "profile_sha256": H,
            "prompt_id": "fake", "prompt_sha256": H, "schema_id": "fake", "schema_sha256": H,
            "semantic_request_hash": H, "rendered_prompt_sha256": H, "packet_sha256": H, "budget_result": "PASS"},
        "caps": {"invocations": 1, "semantic_rounds": 2, "technical_attempts_per_round": 3,
            "provider_attempts": 6, "second_round": "schema_valid_semantic_rejection_only",
            "prior_usage": "UNKNOWN", "full_lineage_usage": "UNKNOWN"},
        "acknowledgements": {"prior_requests_may_have_completed": True, "duplicate_generation_possible": True},
    }


def check(obj, **kwargs):
    return check_preflight_structure(raw=canonical_json_bytes(obj), **kwargs)


def test_fabricated_baseline_only_returns_byte_digest():
    obj = preflight()
    raw = canonical_json_bytes(obj)
    result = check_preflight_structure(raw=raw, expected_ratification_sha256=H, expected_scope=copy.deepcopy(obj["scope"]))
    assert result.preflight_sha256 == hashlib.sha256(raw).hexdigest()
    assert not hasattr(result, "__dict__")
    assert not callable(result)
    assert result.__slots__ == ("preflight_sha256",)


def test_paired_root_result_and_explicit_selection():
    obj = preflight()
    obj["authority"].update(selection_mode="EXPLICIT", b12_current_pointer=None)
    obj["slots"]["root_intent"] = {"status": "VERIFIED", "identity": H, "bytes_sha256": H2}
    obj["slots"]["root_result"] = {"status": "VERIFIED", "identity": H2, "bytes_sha256": H}
    obj["root_result"] = {"identity": H2, "bytes_sha256": H}
    check(obj)


@pytest.mark.parametrize("path,value", [
    ("claim.file.bytes_sha256", H2), ("authority.b12_current_pointer", None),
    ("authority.selection_mode", "EXPLICIT"), ("authority.a5_manifest.revision", 0),
    ("authority.strategy.content_hash", "A" * 64), ("scope.window_ordinal", True),
    ("scope.window_id", "window_0002"), ("operator_id", "bad id"), ("reason", ""),
    ("attempt_id", "f" * 31), ("created_at", "2025-02-29T00:00:00Z"),
    ("checkpoint.status", "VERIFIED"), ("checkpoint.status", "UNKNOWN"),
    ("checkpoint.expected_path", "/synthetic/../root"), ("roots.execution.path", "/synthetic//root"),
    ("roots.pointer.type", "symlink"), ("caps.provider_attempts", 7), ("caps.prior_usage", 0),
    ("caps.full_lineage_usage", "0"), ("acknowledgements.duplicate_generation_possible", 1),
    ("runtime.input_ceiling_tokens", 0), ("runtime.estimated_input_tokens", 81),
    ("runtime.reserved_output_tokens", 21), ("runtime.provider_request_bytes", -1),
    ("runtime.effective_n_ctx", 9007199254740992), ("runtime.input_ceiling_tokens", True),
    ("runtime.budget_result", "FAIL"), ("slots.root_intent.status", "VERIFIED"),
    ("slots.root_result.identity", H), ("root_result", {"identity": H, "bytes_sha256": H}),
    ("slots.decision", {"status": "VERIFIED", "identity": H, "bytes_sha256": H}),
    ("slots.consumption", {"status": "VERIFIED", "identity": H, "bytes_sha256": H}),
    ("slots.terminal", {"status": "VERIFIED", "identity": H, "bytes_sha256": H}),
])
def test_invalid_fields(path, value):
    obj = preflight()
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(PreflightStructureError):
        check_preflight_structure(raw=json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def test_root_result_mismatch():
    obj = preflight()
    for name in ("root_intent", "root_result"):
        obj["slots"][name] = {"status": "VERIFIED", "identity": H, "bytes_sha256": H}
    obj["root_result"] = {"identity": H, "bytes_sha256": H2}
    with pytest.raises(PreflightStructureError):
        check(obj)


@pytest.mark.parametrize("mutation", ["missing", "extra", "scope", "ratification"])
def test_closed_fields_and_external_expectations(mutation):
    obj = preflight()
    kwargs = {}
    if mutation == "missing":
        del obj["runtime"]["model_id"]
    elif mutation == "extra":
        obj["claim"]["extra"] = 1
    elif mutation == "scope":
        kwargs["expected_scope"] = {**obj["scope"], "claim_identity": H2}
    else:
        kwargs["expected_ratification_sha256"] = H2
    with pytest.raises(PreflightStructureError):
        check(obj, **kwargs)


@pytest.mark.parametrize("raw", [b'{} ', b'{"x":1,"x":2}', b'\xff', b'\xef\xbb\xbf{}',
                                 b'{"x":"\\ud800"}', b'{"x":1.0}', b' ' * 65537, b''])
def test_bad_raw_bytes(raw):
    with pytest.raises(PreflightStructureError):
        check_preflight_structure(raw=raw)


def test_noncanonical_and_illegal_values_in_complete_document():
    raw = canonical_json_bytes(preflight())
    variants = [raw + b'\n', raw.replace(b'"schema_version":1', b'"schema_version":1.0'),
                raw.replace(b'"reason":"synthetic"', b'"reason":"bad\\ntext"'),
                raw.replace(b'"record_type":"preflight"', b'"record_type":"preflight","record_type":"preflight"')]
    for variant in variants:
        with pytest.raises(PreflightStructureError):
            check_preflight_structure(raw=variant)


def test_paired_slots_require_result_and_exact_hashes():
    obj = preflight()
    for name in ("root_intent", "root_result"):
        obj["slots"][name] = {"status": "VERIFIED", "identity": H, "bytes_sha256": H}
    with pytest.raises(PreflightStructureError):
        check(obj)
    obj["root_result"] = {"identity": H, "bytes_sha256": H2}
    with pytest.raises(PreflightStructureError):
        check(obj)
