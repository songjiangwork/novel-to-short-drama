"""Synthetic keys, expectations and documents are generated only in memory."""
import base64
import copy
import hashlib
import socket
from dataclasses import FrozenInstanceError, fields

import pytest
import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from short_drama.adaptation import disposition_recovery_policy_ratification as policy
from short_drama.adaptation import disposition_recovery_trust_manifest as trust


NOW = "2026-06-01T00:00:00Z"
MANIFEST_DOMAIN = b"B1.3-R0D3/trust-anchor-manifest/v1\0"
POLICY_DOMAIN = b"B1.3-R0D1/approval/v1\0"


def public(key):
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id(key):
    return hashlib.sha256(public(key)).hexdigest()


def signed(key, payload, domain):
    return rfc8785.dumps({"payload": payload, "signature": base64.b64encode(
        key.sign(domain + rfc8785.dumps(payload))).decode("ascii")})


@pytest.fixture
def sample():
    root, ratifier, live = (Ed25519PrivateKey.generate() for _ in range(3))
    def entry(key, role):
        return {"key_id": key_id(key), "public_key": base64.b64encode(public(key)).decode(),
                "principal_id": "synthetic-person", "role": role, "not_before": "2026-02-01T00:00:00Z",
                "not_after": "2026-11-01T00:00:00Z", "status": "active", "revoked_at": None}
    policy_bytes, contract_bytes = b"synthetic policy\n", b"synthetic contract\n"
    def doc(path, raw):
        return {"path": path, "version": "v1", "sha256": hashlib.sha256(raw).hexdigest()}
    policy_doc, contract_doc = doc("docs/policy", policy_bytes), doc("docs/contract", contract_bytes)
    scope = {"incident_id": "synthetic", "window_id": "window_0001", "window_ordinal": 1,
             "claim_identity": "a" * 64, "claim_bytes_sha256": "b" * 64,
             "preparation_identity": "c" * 64, "request_identity": "d" * 64}
    manifest = {"schema_version": 1, "record_type": "trust_anchor_manifest", "root_key_id": key_id(root),
                "manifest_version": 1, "issued_at": "2026-01-01T00:00:00Z", "expires_at": "2026-12-01T00:00:00Z",
                "revocation_checked_at": "2026-05-01T00:00:00Z", "revocation_valid_until": "2026-10-01T00:00:00Z",
                "bootstrap_provenance_ref": "synthetic", "keys": [entry(ratifier, "policy_ratifier"), entry(live, "live_authorizer")],
                "allowlist": [{"policy_document": policy_doc, "contract_document": contract_doc}]}
    payload = {"schema_version": 1, "record_type": "policy_ratification", "key_id": key_id(ratifier),
               "authority_id": "synthetic-person", "issued_at": "2026-03-01T00:00:00Z",
               "expires_at": "2026-09-01T00:00:00Z", "policy_document": copy.deepcopy(policy_doc),
               "contract_document": copy.deepcopy(contract_doc), "scope": copy.deepcopy(scope),
               "decision_a": "approve", "decision_b": "approve"}
    args = {"expected_authority_id": "synthetic-person", "expected_scope": copy.deepcopy(scope), "trusted_now": NOW,
            "policy_bytes": policy_bytes, "policy_path": "docs/policy", "policy_version": "v1",
            "contract_bytes": contract_bytes, "contract_path": "docs/contract", "contract_version": "v1"}
    return root, ratifier, live, manifest, payload, args


def manifest_for(sample):
    root, _, _, data, _, _ = sample
    return trust.verify_manifest(raw=signed(root, data, MANIFEST_DOMAIN), expected_root_public_key=public(root),
                                 expected_manifest_version=1, trusted_now=NOW, currentness_confirmed=True)


def verify(sample, raw=None, **overrides):
    _, key, _, _, payload, args = sample
    return policy.verify_policy_ratification(raw=raw if raw is not None else signed(key, payload, POLICY_DOMAIN),
                                             manifest=manifest_for(sample), **(args | overrides))


def test_valid_is_only_immutable_digest_and_no_io(sample, monkeypatch):
    raw = signed(sample[1], sample[4], POLICY_DOMAIN)
    manifest = manifest_for(sample)
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("file access"))
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    result = policy.verify_policy_ratification(raw=raw, manifest=manifest, **sample[5])
    assert result.envelope_sha256 == hashlib.sha256(raw).hexdigest()
    assert [f.name for f in fields(result)] == ["envelope_sha256"]
    with pytest.raises(FrozenInstanceError):
        result.envelope_sha256 = "changed"


@pytest.mark.parametrize("a,b", [("reject", "approve"), ("approve", "defer"), ("reject", "defer"), ("defer", "reject")])
def test_signed_non_grant_decisions_are_only_digest(sample, a, b):
    sample[4].update(decision_a=a, decision_b=b)
    result = verify(sample)
    assert [f.name for f in fields(result)] == ["envelope_sha256"]
    assert not hasattr(result, "approved") and not hasattr(result, "authorized")


@pytest.mark.parametrize("change", [
    lambda p: p.update(extra="x"), lambda p: p.pop("scope"), lambda p: p.update(schema_version=True),
    lambda p: p.update(record_type="live_authorization"), lambda p: p.update(key_id="0" * 64),
    lambda p: p.update(authority_id="other"), lambda p: p.update(authority_id=None),
    lambda p: p.update(decision_a="yes"), lambda p: p.update(decision_b=None), lambda p: p.update(decision_b=True),
    lambda p: p.update(scope=None), lambda p: p["scope"].update(extra="x"),
    lambda p: p["scope"].pop("request_identity"), lambda p: p["scope"].update(window_ordinal=True),
    lambda p: p["scope"].update(window_id="window_0002"), lambda p: p["scope"].update(claim_identity="e" * 64),
    lambda p: p["scope"].update(claim_bytes_sha256="e" * 64), lambda p: p["scope"].update(request_identity="e" * 64),
    lambda p: p.update(policy_document=None), lambda p: p["policy_document"].update(extra="x"),
    lambda p: p["policy_document"].pop("version"), lambda p: p["policy_document"].update(path="docs/other"),
    lambda p: p["contract_document"].update(sha256="f" * 64), lambda p: p["contract_document"].update(version=None),
    lambda p: p.update(issued_at="2026-02-30T00:00:00Z"), lambda p: p.update(expires_at="2026-06-01T00:00:00Z"),
    lambda p: p.update(issued_at="2026-06-02T00:00:00Z"), lambda p: p.update(issued_at="2026-06-01T00:00:00Z", expires_at="2026-06-01T00:00:00Z"),
    lambda p: p.update(issued_at="2026-01-31T23:59:59Z"), lambda p: p.update(expires_at="2026-11-01T00:00:01Z"),
    lambda p: p.update(issued_at="2025-12-31T23:59:59Z"), lambda p: p.update(expires_at="2026-12-01T00:00:01Z"),
])
def test_bad_signed_payload(sample, change):
    change(sample[4])
    with pytest.raises(policy.PolicyRatificationError, match="^invalid policy ratification$"):
        verify(sample)


@pytest.mark.parametrize("overrides", [
    {"expected_authority_id": "other"}, {"expected_authority_id": None}, {"trusted_now": "2026-06-01T00:00:01Z"},
    {"trusted_now": "invalid"}, {"expected_scope": None}, {"expected_scope": {}},
    {"policy_bytes": b"other"}, {"contract_bytes": b"other"},
    {"policy_bytes": b"synthetic contract\n", "contract_bytes": b"synthetic policy\n"},
    {"policy_path": "docs/other"}, {"contract_path": "docs/other"},
    {"policy_version": "v2"}, {"contract_version": "v2"}, {"policy_path": "../other"},
    {"policy_bytes": b"x" * 1048577}, {"contract_bytes": None},
])
def test_independent_expectations(sample, overrides):
    with pytest.raises(policy.PolicyRatificationError, match="^invalid policy ratification$"):
        verify(sample, **overrides)


def test_allowlist_pair_is_mandatory(sample):
    sample[3]["allowlist"][0]["policy_document"]["sha256"] = "a" * 64
    with pytest.raises(policy.PolicyRatificationError):
        verify(sample)


def test_wrong_role_key_and_signature(sample):
    _, ratifier, live, _, payload, _ = sample
    outsider = Ed25519PrivateKey.generate()
    for key in (live, outsider):
        modified = copy.deepcopy(payload)
        modified["key_id"] = key_id(key)
        with pytest.raises(policy.PolicyRatificationError):
            verify(sample, raw=signed(key, modified, POLICY_DOMAIN))
    with pytest.raises(policy.PolicyRatificationError):
        verify(sample, raw=signed(outsider, payload, POLICY_DOMAIN))
    with pytest.raises(policy.PolicyRatificationError):
        verify(sample, raw=signed(ratifier, payload, MANIFEST_DOMAIN))


def test_manifest_and_key_intervals_inclusive_end_boundaries(sample):
    # The ratification end may equal both exclusive manifest/key ends.
    sample[3]["keys"][0].update(not_before="2026-01-01T00:00:00Z", not_after="2026-12-01T00:00:00Z")
    sample[4].update(issued_at="2026-01-01T00:00:00Z", expires_at="2026-12-01T00:00:00Z")
    verify(sample)
    sample[4].update(issued_at=NOW)
    verify(sample)
    sample[4].update(expires_at=NOW)
    with pytest.raises(policy.PolicyRatificationError):
        verify(sample)


def test_manifest_interval_separately_enforced(sample):
    sample[3]["keys"][0].update(not_before="2025-01-01T00:00:00Z", not_after="2027-01-01T00:00:00Z")
    for field, value in [("issued_at", "2025-12-31T23:59:59Z"), ("expires_at", "2026-12-01T00:00:01Z")]:
        original = sample[4][field]
        sample[4][field] = value
        with pytest.raises(policy.PolicyRatificationError):
            verify(sample)
        sample[4][field] = original


def test_invalid_envelopes(sample):
    raw = signed(sample[1], sample[4], POLICY_DOMAIN)
    bad = [b"", b"x" * 32769, b"\xff", b"{}", raw + b"\n", raw.replace(b'"payload":', b'"payload" :', 1),
           raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
           raw.replace(b'"window_ordinal":1', b'"window_ordinal":1.0'),
           raw.replace(b'"decision_a":"approve"', b'"decision_a":"reject"')]
    for value in bad:
        with pytest.raises(policy.PolicyRatificationError, match="^invalid policy ratification$"):
            verify(sample, raw=value)
    with pytest.raises(policy.PolicyRatificationError):
        policy.verify_policy_ratification(raw=raw, manifest=None, **sample[5])


@pytest.mark.parametrize("change", [
    {"status": "revoked", "revoked_at": "2026-05-01T00:00:00Z"},
    {"not_before": "2026-06-02T00:00:00Z"},
    {"not_after": NOW},
    {"role": "live_authorizer"},
    {"principal_id": "other"},
])
def test_manifest_key_must_be_current_and_role_bound(sample, change):
    sample[3]["keys"][0].update(change)
    with pytest.raises(policy.PolicyRatificationError, match="^invalid policy ratification$"):
        verify(sample)


def test_expected_scope_must_be_independent_and_well_formed(sample):
    other = copy.deepcopy(sample[5]["expected_scope"])
    other["request_identity"] = "e" * 64
    with pytest.raises(policy.PolicyRatificationError):
        verify(sample, expected_scope=other)
    other["window_ordinal"] = True
    with pytest.raises(policy.PolicyRatificationError):
        verify(sample, expected_scope=other)


def test_document_pair_cannot_be_swapped_even_if_individually_known(sample):
    pair = sample[3]["allowlist"][0]
    pair["policy_document"], pair["contract_document"] = pair["contract_document"], pair["policy_document"]
    with pytest.raises(policy.PolicyRatificationError):
        verify(sample)
