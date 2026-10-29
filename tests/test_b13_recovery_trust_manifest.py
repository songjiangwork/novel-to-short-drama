"""All keys, identities and currentness assertions are synthetic and in memory."""
import base64
import copy
import hashlib
import socket
from dataclasses import FrozenInstanceError

import pytest
import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from short_drama.adaptation import disposition_recovery_trust_manifest as trust
from short_drama.adaptation.disposition_recovery_signatures import verify_envelope_integrity


def public(key):
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id(key):
    return hashlib.sha256(public(key)).hexdigest()


def signed(key, payload, domain=b"B1.3-R0D3/trust-anchor-manifest/v1\0"):
    return rfc8785.dumps({"payload": payload, "signature": base64.b64encode(key.sign(domain + rfc8785.dumps(payload))).decode()})


@pytest.fixture
def sample():
    root, policy, live = (Ed25519PrivateKey.generate() for _ in range(3))
    def entry(key, role):
        return {"key_id": key_id(key), "public_key": base64.b64encode(public(key)).decode(),
                "principal_id": role + "-person", "role": role, "not_before": "2026-01-01T00:00:00Z",
                "not_after": "2027-01-01T00:00:00Z", "status": "active", "revoked_at": None}
    doc = lambda name: {"path": "docs/" + name, "version": "v1", "sha256": "a" * 64}
    payload = {"schema_version": 1, "record_type": "trust_anchor_manifest", "root_key_id": key_id(root),
               "manifest_version": 2, "issued_at": "2026-01-01T00:00:00Z", "expires_at": "2027-01-01T00:00:00Z",
               "revocation_checked_at": "2026-06-01T00:00:00Z", "revocation_valid_until": "2026-12-01T00:00:00Z",
               "bootstrap_provenance_ref": "synthetic", "keys": [entry(policy, "policy_ratifier"), entry(live, "live_authorizer")],
               "allowlist": [{"policy_document": doc("policy"), "contract_document": doc("contract")}]}
    return root, policy, live, payload


def verify(root, payload, **kwargs):
    args = dict(raw=signed(root, payload), expected_root_public_key=public(root), expected_manifest_version=2,
                trusted_now="2026-10-01T00:00:00Z", currentness_confirmed=True)
    args.update(kwargs)
    return trust.verify_manifest(**args)


def test_positive_and_wrong_role_signature(sample, monkeypatch):
    root, policy, live, payload = sample
    raw = signed(root, payload)
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("file access"))
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    manifest = verify(root, payload, raw=raw)
    trust.check_document_pair(manifest, **payload["allowlist"][0])
    for key, role, kind in [(policy, "policy_ratifier", "policy_ratification"), (live, "live_authorizer", "live_authorization")]:
        selected = trust.select_signer_key(manifest, expected_key_id=key_id(key), expected_principal_id=role + "-person", expected_role=role)
        approval = {"schema_version": 1, "record_type": kind, "key_id": key_id(key)}
        verify_envelope_integrity(expected_kind=kind, raw=signed(key, approval, b"B1.3-R0D1/approval/v1\0"), expected_public_key=selected)
        other = "live_authorizer" if role == "policy_ratifier" else "policy_ratifier"
        with pytest.raises(trust.TrustManifestError):
            trust.select_signer_key(manifest, expected_key_id=key_id(key), expected_principal_id=role + "-person", expected_role=other)
    with pytest.raises(FrozenInstanceError):
        manifest._now = None
    with pytest.raises(trust.TrustManifestError):
        trust.VerifiedManifest(None, (), (), None)


@pytest.mark.parametrize("overrides", [
    {"expected_manifest_version": 1}, {"expected_manifest_version": 3}, {"expected_manifest_version": None},
    {"expected_manifest_version": True}, {"currentness_confirmed": None}, {"currentness_confirmed": False},
    {"currentness_confirmed": 1}, {"trusted_now": None}, {"trusted_now": "2026-12-01T00:00:00Z"},
    {"trusted_now": "2027-01-01T00:00:00Z"}, {"trusted_now": "2026-02-30T00:00:00Z"},
    {"expected_root_public_key": b"x" * 32}, {"expected_root_public_key": None},
])
def test_external_expectations_fail_closed(sample, overrides):
    root, _, _, payload = sample
    with pytest.raises(trust.TrustManifestError):
        verify(root, payload, **overrides)


def test_forged_root_and_invalid_bytes(sample):
    root, _, _, payload = sample
    attacker = Ed25519PrivateKey.generate()
    forged = copy.deepcopy(payload)
    forged["root_key_id"] = key_id(attacker)
    for raw in (signed(attacker, forged), signed(attacker, payload), signed(root, payload) + b"\n", b"x" * 65537):
        with pytest.raises(trust.TrustManifestError):
            verify(root, payload, raw=raw)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(extra=1), lambda p: p.pop("keys"), lambda p: p.update(manifest_version=True),
    lambda p: p.update(bootstrap_provenance_ref=""), lambda p: p.update(issued_at="2026-01-01T00:00:00+00:00"),
    lambda p: p.update(revocation_checked_at="2026-11-01T00:00:00Z"),
    lambda p: p.update(revocation_valid_until="2028-01-01T00:00:00Z"),
    lambda p: p.update(expires_at=p["issued_at"]), lambda p: p.update(keys=[]),
    lambda p: p["keys"].append(copy.deepcopy(p["keys"][0])),
    lambda p: p["keys"][0].update(key_id="0" * 64), lambda p: p["keys"][0].update(public_key="bad"),
    lambda p: p["keys"][0].update(role="unknown"), lambda p: p["keys"][0].update(status=True),
    lambda p: p["keys"][0].update(revoked_at="2026-01-01T00:00:00Z"),
    lambda p: p["keys"][0].update(status="revoked", revoked_at=None),
    lambda p: p["keys"][0].update(status="revoked", revoked_at="2026-11-01T00:00:00Z"),
    lambda p: p["keys"][0].update(not_after=p["keys"][0]["not_before"]),
    lambda p: p["keys"][0].update(extra=1), lambda p: p["keys"][0].pop("principal_id"),
    lambda p: p.update(allowlist=[]), lambda p: p["allowlist"].append(copy.deepcopy(p["allowlist"][0])),
    lambda p: p["allowlist"][0].update(extra=1), lambda p: p["allowlist"][0]["policy_document"].update(extra=1),
    lambda p: p["allowlist"][0]["policy_document"].update(path="docs/../policy"),
    lambda p: p["allowlist"][0]["policy_document"].update(sha256="A" * 64),
    lambda p: p["allowlist"][0]["policy_document"].update(version=None),
])
def test_bad_schema(sample, mutate):
    root, _, _, payload = sample
    mutate(payload)
    with pytest.raises(trust.TrustManifestError):
        verify(root, payload)


@pytest.mark.parametrize("changes", [
    {"status": "revoked", "revoked_at": "2026-09-01T00:00:00Z"},
    {"not_before": "2026-11-01T00:00:00Z"}, {"not_after": "2026-09-01T00:00:00Z"},
])
def test_key_not_usable(sample, changes):
    root, policy, _, payload = sample
    payload["keys"][0].update(changes)
    manifest = verify(root, payload)
    with pytest.raises(trust.TrustManifestError):
        trust.select_signer_key(manifest, expected_key_id=key_id(policy), expected_principal_id="policy_ratifier-person", expected_role="policy_ratifier")


def test_binding_and_documents(sample):
    root, policy, _, payload = sample
    manifest = verify(root, payload)
    for kid, principal, role in [(key_id(policy), "other", "policy_ratifier"), ("0" * 64, "policy_ratifier-person", "policy_ratifier"), (key_id(policy), "policy_ratifier-person", "unknown")]:
        with pytest.raises(trust.TrustManifestError):
            trust.select_signer_key(manifest, expected_key_id=kid, expected_principal_id=principal, expected_role=role)
    for name in ("policy_document", "contract_document"):
        for field, value in [("path", "docs/other"), ("version", "v2"), ("sha256", "b" * 64)]:
            pair = copy.deepcopy(payload["allowlist"][0])
            pair[name][field] = value
            with pytest.raises(trust.TrustManifestError):
                trust.check_document_pair(manifest, **pair)


def test_valid_untrusted_signer_is_not_a_role_match(sample):
    root, _, _, payload = sample
    outsider = Ed25519PrivateKey.generate()
    approval = {"schema_version": 1, "record_type": "policy_ratification", "key_id": key_id(outsider)}
    raw = signed(outsider, approval, b"B1.3-R0D1/approval/v1\0")
    verify_envelope_integrity(expected_kind="policy_ratification", raw=raw, expected_public_key=public(outsider))
    manifest = verify(root, payload)
    with pytest.raises(trust.TrustManifestError):
        trust.select_signer_key(manifest, expected_key_id=key_id(outsider),
                                expected_principal_id="policy_ratifier-person", expected_role="policy_ratifier")


def test_invalid_expected_document_is_typed(sample):
    root, _, _, payload = sample
    manifest = verify(root, payload)
    pair = copy.deepcopy(payload["allowlist"][0])
    pair["policy_document"]["path"] = "\ud800"
    with pytest.raises(trust.TrustManifestError):
        trust.check_document_pair(manifest, **pair)
