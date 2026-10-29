"""Synthetic/offline preflight byte structure checks, never evidence or authority.

A completely fabricated preflight can pass. No claim, policy, manifest, root,
checkpoint, CURRENT, runtime, freshness, approval or permission is verified.
Runtime configuration evidence codec and trusted discovery remain OPEN DECISION.
Path checks are syntactic; symlinks and actual filesystem identities are unknown.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

from short_drama.artifacts import canonical_json_bytes, strict_json_loads
from short_drama.artifacts.errors import CanonicalSerializationError
from .disposition_recovery_signatures import _check_values, EnvelopeIntegrityError


class PreflightStructureError(ValueError):
    """Canonical bytes, proposed structure, or supplied expectation is invalid."""


@dataclass(frozen=True, slots=True)
class PreflightByteIntegrity:
    """Digest of the exact structurally checked bytes; conveys no trust."""
    preflight_sha256: str


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,127}\Z", re.ASCII)
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ATTEMPT = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z", re.ASCII)
_MAX = 9007199254740991


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names.split()):
        raise PreflightStructureError("invalid fields")


def _match(value, pattern):
    if type(value) is not str or not pattern.fullmatch(value):
        raise PreflightStructureError("invalid string format")


def _uint(value, minimum=0):
    if type(value) is not int or not minimum <= value <= _MAX:
        raise PreflightStructureError("invalid integer")


def _literal(value, expected):
    if type(value) is not type(expected) or value != expected:
        raise PreflightStructureError("invalid literal")


def _time(value):
    _match(value, _TIME)
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PreflightStructureError("invalid timestamp") from exc


def _string(value, limit):
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= limit:
        raise PreflightStructureError("invalid string length")


def _path(value):
    _string(value, 4096)
    if not value.startswith("/") or (value != "/" and any(
            part in ("", ".", "..") for part in value[1:].split("/"))):
        raise PreflightStructureError("invalid absolute path syntax")


def _scope(value):
    _fields(value, "incident_id window_id window_ordinal claim_identity claim_bytes_sha256 preparation_identity request_identity")
    _match(value["incident_id"], _ID)
    _literal(value["window_id"], "window_0001")
    _uint(value["window_ordinal"], 1)
    for name in ("claim_identity", "claim_bytes_sha256", "preparation_identity", "request_identity"):
        _match(value[name], _HASH)


def _ref(value):
    _fields(value, "artifact_type artifact_id revision content_hash")
    for name in ("artifact_type", "artifact_id"):
        _match(value[name], _ID)
    _uint(value["revision"], 1)
    _match(value["content_hash"], _HASH)


def _node(value, file=False):
    _fields(value, "path dev ino type bytes_sha256 size" if file else "path dev ino type")
    _path(value["path"])
    _uint(value["dev"])
    _uint(value["ino"], 1)
    _literal(value["type"], "regular" if file else "directory")
    if file:
        _match(value["bytes_sha256"], _HASH)
        _uint(value["size"], 1)


def _slot(value):
    _fields(value, "status identity bytes_sha256")
    if value["status"] == "ABSENT":
        if value["identity"] is not None or value["bytes_sha256"] is not None:
            raise PreflightStructureError("absent slot has hashes")
    elif value["status"] == "VERIFIED":
        _match(value["identity"], _HASH)
        _match(value["bytes_sha256"], _HASH)
    else:
        raise PreflightStructureError("invalid slot status")


def _validate(obj):
    _fields(obj, "schema_version record_type created_at policy_ratification_sha256 scope operator_id reason attempt_id claim authority roots root_result checkpoint slots runtime caps acknowledgements")
    _literal(obj["schema_version"], 1)
    _literal(obj["record_type"], "preflight")
    _time(obj["created_at"])
    _match(obj["policy_ratification_sha256"], _HASH)
    _scope(obj["scope"])
    _match(obj["operator_id"], _ID)
    _string(obj["reason"], 256)
    _match(obj["attempt_id"], _ATTEMPT)
    claim = obj["claim"]
    _fields(claim, "file state")
    _node(claim["file"], file=True)
    _literal(claim["state"], "CLAIMED_UNRESOLVED")
    if claim["file"]["bytes_sha256"] != obj["scope"]["claim_bytes_sha256"]:
        raise PreflightStructureError("claim hash mismatch")
    authority = obj["authority"]
    _fields(authority, "source_snapshot_sha256 a5_manifest a6_manifest a6_validation a6_current_pointer strategy strategy_hash selection_mode b12_current_pointer")
    for name in ("source_snapshot_sha256", "strategy_hash"):
        _match(authority[name], _HASH)
    for name in ("a5_manifest", "a6_manifest", "a6_validation", "a6_current_pointer", "strategy"):
        _ref(authority[name])
    mode, pointer = authority["selection_mode"], authority["b12_current_pointer"]
    if mode == "CURRENT":
        _ref(pointer)
    elif mode != "EXPLICIT" or pointer is not None:
        raise PreflightStructureError("invalid selection/pointer")
    roots = obj["roots"]
    _fields(roots, "execution checkpoint artifact pointer checkpoint_parent")
    for node in roots.values():
        _node(node)
    checkpoint = obj["checkpoint"]
    _fields(checkpoint, "status expected_path")
    _literal(checkpoint["status"], "ABSENT")
    _path(checkpoint["expected_path"])
    slots = obj["slots"]
    _fields(slots, "root_intent root_result decision consumption terminal")
    for slot in slots.values():
        _slot(slot)
    if any(slots[name]["status"] != "ABSENT" for name in ("decision", "consumption", "terminal")):
        raise PreflightStructureError("non-baseline slot")
    if slots["root_intent"]["status"] != slots["root_result"]["status"]:
        raise PreflightStructureError("unpaired root slots")
    result = obj["root_result"]
    if slots["root_result"]["status"] == "ABSENT":
        if result is not None:
            raise PreflightStructureError("unexpected root result")
    else:
        _fields(result, "identity bytes_sha256")
        if any(result[name] != slots["root_result"][name] for name in ("identity", "bytes_sha256")):
            raise PreflightStructureError("root result mismatch")
    runtime = obj["runtime"]
    _fields(runtime, "observed_at provider_config_id provider_config_sha256 model_id model_config_sha256 tokenizer_id tokenizer_sha256 effective_n_ctx input_ceiling_tokens reserved_output_tokens estimated_input_tokens provider_request_bytes profile_id profile_sha256 prompt_id prompt_sha256 schema_id schema_sha256 semantic_request_hash rendered_prompt_sha256 packet_sha256 budget_result")
    _time(runtime["observed_at"])
    for name in ("provider_config_id", "model_id", "tokenizer_id", "profile_id", "prompt_id", "schema_id"):
        _match(runtime[name], _ID)
    for name in ("provider_config_sha256", "model_config_sha256", "tokenizer_sha256", "profile_sha256", "prompt_sha256", "schema_sha256", "semantic_request_hash", "rendered_prompt_sha256", "packet_sha256"):
        _match(runtime[name], _HASH)
    for name in ("effective_n_ctx", "input_ceiling_tokens", "reserved_output_tokens", "estimated_input_tokens", "provider_request_bytes"):
        _uint(runtime[name], 1)
    _literal(runtime["budget_result"], "PASS")
    if (runtime["estimated_input_tokens"] > runtime["input_ceiling_tokens"] or
            runtime["estimated_input_tokens"] + runtime["reserved_output_tokens"] > runtime["effective_n_ctx"]):
        raise PreflightStructureError("invalid token budget")
    caps = obj["caps"]
    expected = {"invocations": 1, "semantic_rounds": 2, "technical_attempts_per_round": 3,
                "provider_attempts": 6, "second_round": "schema_valid_semantic_rejection_only",
                "prior_usage": "UNKNOWN", "full_lineage_usage": "UNKNOWN"}
    _fields(caps, " ".join(expected))
    for name, value in expected.items():
        _literal(caps[name], value)
    ack = obj["acknowledgements"]
    _fields(ack, "prior_requests_may_have_completed duplicate_generation_possible")
    for value in ack.values():
        _literal(value, True)


def check_preflight_structure(*, raw: bytes, expected_ratification_sha256: str | None = None,
                              expected_scope: dict | None = None) -> PreflightByteIntegrity:
    """Check proposed shape and optional caller-pinned equality, with no trust inference.

    Omitted expectations authenticate nothing; supplied values are assertions,
    not independently verified here. No external evidence or clock is consulted.
    """
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise PreflightStructureError("invalid preflight size or type")
    try:
        obj = strict_json_loads(raw.decode("utf-8"))
        _check_values(obj)
        if canonical_json_bytes(obj) != raw:
            raise PreflightStructureError("noncanonical preflight")
        _validate(obj)
        if expected_ratification_sha256 is not None:
            _match(expected_ratification_sha256, _HASH)
            if obj["policy_ratification_sha256"] != expected_ratification_sha256:
                raise PreflightStructureError("ratification hash mismatch")
        if expected_scope is not None:
            _scope(expected_scope)
            if obj["scope"] != expected_scope:
                raise PreflightStructureError("scope mismatch")
    except PreflightStructureError:
        raise
    except (CanonicalSerializationError, EnvelopeIntegrityError, UnicodeError, ValueError, RecursionError) as exc:
        raise PreflightStructureError("invalid preflight structure") from exc
    return PreflightByteIntegrity(hashlib.sha256(raw).hexdigest())
