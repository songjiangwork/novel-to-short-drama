"""Synthetic in-memory signatures only; integrity is not authorization."""
import base64
import hashlib
import json
from dataclasses import FrozenInstanceError, fields

import pytest
import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from short_drama.adaptation import disposition_recovery_signatures as signatures


KINDS = ("policy_ratification", "live_authorization", "trust_anchor_manifest")
APPROVAL_DOMAIN = b"B1.3-R0D1/approval/v1\x00"
MANIFEST_DOMAIN = b"B1.3-R0D3/trust-anchor-manifest/v1\x00"


@pytest.fixture
def key():
    return Ed25519PrivateKey.generate()


def public(key):
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def payload(key, kind="policy_ratification", **extra):
    name = "root_key_id" if kind == "trust_anchor_manifest" else "key_id"
    return {"schema_version": 1, "record_type": kind,
            name: hashlib.sha256(public(key)).hexdigest(), **extra}


def envelope(key, data, domain=None):
    if domain is None:
        domain = MANIFEST_DOMAIN if data["record_type"] == "trust_anchor_manifest" else APPROVAL_DOMAIN
    signature = key.sign(domain + rfc8785.dumps(data))
    return rfc8785.dumps({"payload": data, "signature": base64.b64encode(signature).decode("ascii")})


def verify(raw, key, kind="policy_ratification"):
    return signatures.verify_envelope_integrity(expected_kind=kind, raw=raw,
                                                 expected_public_key=public(key))


@pytest.mark.parametrize("kind", KINDS)
def test_valid_signature_proves_integrity_only(key, kind, monkeypatch):
    data = payload(key, kind, decision_a="approve", claim="synthetic-unverified")
    raw = envelope(key, data)
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("unexpected file access"))
    result = verify(raw, key, kind)
    assert result.envelope_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.payload_bytes == rfc8785.dumps(data)
    assert {f.name for f in fields(result)} == {"envelope_sha256", "payload_bytes"}
    with pytest.raises(FrozenInstanceError):
        result.envelope_sha256 = "changed"


@pytest.mark.parametrize("kind,wrong_domain", [(KINDS[0], MANIFEST_DOMAIN), (KINDS[1], MANIFEST_DOMAIN), (KINDS[2], APPROVAL_DOMAIN)])
def test_wrong_domain(key, kind, wrong_domain):
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(envelope(key, payload(key, kind), wrong_domain), key, kind)


def test_wrong_kind_key_and_identity(key):
    raw = envelope(key, payload(key))
    other = Ed25519PrivateKey.generate()
    for kind, key_bytes, candidate in [
        ("live_authorization", public(key), raw),
        ("invalid", public(key), raw),
        ("policy_ratification", public(other), raw),
        ("policy_ratification", public(key), envelope(key, payload(other))),
        ("trust_anchor_manifest", public(key), envelope(key, payload(other, "trust_anchor_manifest"))),
    ]:
        with pytest.raises(signatures.EnvelopeIntegrityError):
            signatures.verify_envelope_integrity(expected_kind=kind, raw=candidate, expected_public_key=key_bytes)


def test_signature_and_payload_tamper(key):
    obj = json.loads(envelope(key, payload(key, test="original")))
    obj["payload"]["test"] = "changed"
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(rfc8785.dumps(obj), key)
    obj = json.loads(envelope(key, payload(key)))
    sig = bytearray(base64.b64decode(obj["signature"]))
    sig[0] ^= 1
    obj["signature"] = base64.b64encode(sig).decode()
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(rfc8785.dumps(obj), key)


@pytest.mark.parametrize("bad", [b"", b"x", b"x" * 31, b"x" * 33, "x" * 32, bytearray(32), None])
def test_bad_key(key, bad):
    with pytest.raises(signatures.EnvelopeIntegrityError):
        signatures.verify_envelope_integrity(expected_kind=KINDS[0], raw=envelope(key, payload(key)), expected_public_key=bad)


@pytest.mark.parametrize("bad", [None, "{}", bytearray(b"{}"), b"", b"\xef\xbb\xbf{}", b'\xff', b'{', b'{}\n', b'{} ', b'\xff\xfe{\x00}\x00'])
def test_bad_raw(key, bad):
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(bad, key)


@pytest.mark.parametrize("bad", ["", "a", "AAAA", "!" * 88, "A" * 88, "A" * 86 + "==\n", "A" * 86 + "==extra", "A" * 85 + "B==", None, 1])
def test_bad_base64(key, bad):
    obj = json.loads(envelope(key, payload(key)))
    obj["signature"] = bad
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(rfc8785.dumps(obj), key)


@pytest.mark.parametrize("mutation", [
    lambda s: b' ' + s,
    lambda s: s + b'\n',
    lambda s: s.replace(b'"payload":', b'"payload" :', 1),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":1.0'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":true'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":-1'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":9007199254740992'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":NaN'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":Infinity'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
    lambda s: s.replace(b'"payload":{', b'"payload":{"nested":{"x":1,"x":2},'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":1,"evil":"\\ud800"'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":1,"evil":"\\u0001"'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":1,"evil":"\\u007f"'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":1,"evil":0.0'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":1,"evil":-1'),
    lambda s: s.replace(b'"schema_version":1', b'"schema_version":1,"evil":9007199254740992'),
])
def test_reject_noncanonical_or_unsafe_json(key, mutation):
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(mutation(envelope(key, payload(key))), key)


@pytest.mark.parametrize("obj", [{}, {"payload": {}}, {"signature": "x"}, {"payload": {}, "signature": "x", "extra": 1}, {"payload": [], "signature": "x"}])
def test_closed_envelope(key, obj):
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(rfc8785.dumps(obj), key)


@pytest.mark.parametrize("kind,limit", [(KINDS[0], 32768), (KINDS[1], 32768), (KINDS[2], 65536)])
def test_exact_size_limits(key, kind, limit):
    data = payload(key, kind, padding="")
    base = len(envelope(key, data))
    data["padding"] = "x" * (limit - base)
    raw = envelope(key, data)
    assert len(raw) == limit
    verify(raw, key, kind)
    data["padding"] += "x"
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(envelope(key, data), key, kind)


def test_noncanonical_order_and_escapes(key):
    raw = envelope(key, payload(key, label="alpha"))
    obj = json.loads(raw)
    bad_order = json.dumps({"signature": obj["signature"], "payload": obj["payload"]}, separators=(",", ":")).encode()
    bad_escape = raw.replace(b'"alpha"', b'"\\u0061lpha"')
    for bad in (bad_order, bad_escape):
        with pytest.raises(signatures.EnvelopeIntegrityError):
            verify(bad, key)


def test_excessive_nesting_is_typed_error(key):
    raw = b'{"payload":' + b'[' * 1100 + b'0' + b']' * 1100 + b',"signature":""}'
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(raw, key)


@pytest.mark.parametrize("extra", [
    {"value": 1.5}, {"value": -1}, {"value": {"x": "\x01"}},
    {"value": ["\x85"]}, {"schema_version": True}, {"schema_version": 2},
    {"key_id": "0" * 64}, {"record_type": "live_authorization"},
])
def test_signed_but_invalid_values_rejected(key, extra):
    data = payload(key, **extra)
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(envelope(key, data, APPROVAL_DOMAIN), key)


def test_noncanonical_base64_pad_bits(key):
    obj = json.loads(envelope(key, payload(key)))
    encoded = obj["signature"]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    index = alphabet.index(encoded[-3])
    obj["signature"] = encoded[:-3] + alphabet[index + 1] + "=="
    assert base64.b64decode(obj["signature"]) == base64.b64decode(encoded)
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(rfc8785.dumps(obj), key)


def test_crypto_backend_error_is_typed(key, monkeypatch):
    from cryptography.exceptions import UnsupportedAlgorithm

    raw = envelope(key, payload(key))
    def fail(_):
        raise UnsupportedAlgorithm("synthetic failure")
    monkeypatch.setattr(signatures.Ed25519PublicKey, "from_public_bytes", fail)
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(raw, key)


def test_matching_declared_id_does_not_replace_external_key_verification(key):
    other = Ed25519PrivateKey.generate()
    raw = envelope(key, payload(other))
    with pytest.raises(signatures.EnvelopeIntegrityError):
        verify(raw, other)


def test_max_safe_integer_and_nested_boolean_are_accepted_as_bytes(key):
    data = payload(key, example={"count": 9007199254740991, "flag": True})
    assert verify(envelope(key, data), key).payload_bytes == rfc8785.dumps(data)
