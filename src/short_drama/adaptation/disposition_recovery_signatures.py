"""Pure envelope byte/signature integrity checks; never an authority decision.

The caller supplies the expected kind and public key. Successful verification
establishes neither trust, roles, time validity, full payload schema, nor
incident evidence or permission to execute. No keys are discovered here.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import unicodedata
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from short_drama.artifacts import canonical_json_bytes, strict_json_loads
from short_drama.artifacts.errors import CanonicalSerializationError


class EnvelopeIntegrityError(ValueError):
    """Envelope bytes or signature failed integrity verification."""


@dataclass(frozen=True, slots=True)
class EnvelopeIntegrity:
    envelope_sha256: str
    payload_bytes: bytes


_DOMAINS = {
    "policy_ratification": (b"B1.3-R0D1/approval/v1\x00", 32768, "key_id"),
    "live_authorization": (b"B1.3-R0D1/approval/v1\x00", 32768, "key_id"),
    "trust_anchor_manifest": (b"B1.3-R0D3/trust-anchor-manifest/v1\x00", 65536, "root_key_id"),
}
_MAX_INTEGER = 9007199254740991


def _check_values(value: object) -> None:
    # Values are already JSON parsed; an iterative walk also validates keys.
    pending = [value]
    while pending:
        item = pending.pop()
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            if not 0 <= item <= _MAX_INTEGER:
                raise EnvelopeIntegrityError("integer outside v1 range")
        elif type(item) is str:
            if any(unicodedata.category(c) in ("Cc", "Cs") for c in item):
                raise EnvelopeIntegrityError("invalid string character")
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        else:
            raise EnvelopeIntegrityError("unsupported JSON value")


def verify_envelope_integrity(*, expected_kind: str, raw: bytes,
                              expected_public_key: bytes) -> EnvelopeIntegrity:
    """Verify canonical bytes and a signature against an externally supplied key.

    Payload-specific schema, trust and authorization are intentionally outside
    this primitive. Returned payload bytes are immutable, not a validated policy.
    """
    if type(expected_kind) is not str or expected_kind not in _DOMAINS:
        raise EnvelopeIntegrityError("unknown envelope kind")
    domain, limit, key_field = _DOMAINS[expected_kind]
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise EnvelopeIntegrityError("invalid envelope size or type")
    if type(expected_public_key) is not bytes or len(expected_public_key) != 32:
        raise EnvelopeIntegrityError("invalid public key")
    try:
        # Decode explicitly to prohibit JSON's UTF-16/32 auto-detection.
        envelope = strict_json_loads(raw.decode("utf-8"))
        _check_values(envelope)
        if type(envelope) is not dict or set(envelope) != {"payload", "signature"}:
            raise EnvelopeIntegrityError("invalid envelope fields")
        payload = envelope["payload"]
        if (type(payload) is not dict or type(payload.get("schema_version")) is not int
                or payload["schema_version"] != 1 or payload.get("record_type") != expected_kind
                or payload.get(key_field) != hashlib.sha256(expected_public_key).hexdigest()):
            raise EnvelopeIntegrityError("invalid payload identity")
        if canonical_json_bytes(envelope) != raw:
            raise EnvelopeIntegrityError("noncanonical envelope")
        encoded = envelope["signature"]
        if type(encoded) is not str:
            raise EnvelopeIntegrityError("invalid signature encoding")
        signature = base64.b64decode(encoded, validate=True)
        if len(signature) != 64 or base64.b64encode(signature).decode("ascii") != encoded:
            raise EnvelopeIntegrityError("invalid signature encoding")
        payload_bytes = canonical_json_bytes(payload)
        Ed25519PublicKey.from_public_bytes(expected_public_key).verify(signature, domain + payload_bytes)
    except EnvelopeIntegrityError:
        raise
    except (CanonicalSerializationError, UnicodeError, ValueError, binascii.Error,
            InvalidSignature, UnsupportedAlgorithm, RecursionError) as exc:
        raise EnvelopeIntegrityError("invalid envelope integrity") from exc
    return EnvelopeIntegrity(hashlib.sha256(raw).hexdigest(), payload_bytes)
