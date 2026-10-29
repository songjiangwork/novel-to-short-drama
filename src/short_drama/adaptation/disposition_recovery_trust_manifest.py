"""Offline manifest checks conditional on caller-provided trust assertions.

No bootstrap, clock or currentness source is verified here. A snapshot is valid
only for the supplied time and external currentness assertion; callers must
reverify when either changes. None of these checks grants authority.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from short_drama.artifacts import strict_json_loads
from short_drama.adaptation.disposition_recovery_signatures import (
    EnvelopeIntegrityError, verify_envelope_integrity,
)


class TrustManifestError(ValueError):
    """Manifest or caller-supplied expectation failed a local check."""


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,127}\Z", re.ASCII)
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z", re.ASCII)
_ROLES = frozenset({"policy_ratifier", "live_authorizer"})
_TOKEN = object()


def _fields(obj, names):
    if type(obj) is not dict or set(obj) != set(names.split()):
        raise TrustManifestError("invalid fields")


def _id(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise TrustManifestError("invalid ID")
    return value


def _hash(value):
    if type(value) is not str or not _HASH.fullmatch(value):
        raise TrustManifestError("invalid hash")
    return value


def _time(value):
    if type(value) is not str or not _TIME.fullmatch(value):
        raise TrustManifestError("invalid timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise TrustManifestError("invalid timestamp") from exc


def _document(value):
    _fields(value, "path version sha256")
    path = value["path"]
    if (type(path) is not str or any(0xD800 <= ord(c) <= 0xDFFF for c in path)
            or not 1 <= len(path.encode("utf-8")) <= 256
            or path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/"))
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in path)):
        raise TrustManifestError("invalid document path")
    return (path, _id(value["version"]), _hash(value["sha256"]))


@dataclass(frozen=True, slots=True)
class _Key:
    key_id: str
    public_key: bytes
    principal_id: str
    role: str
    not_before: datetime
    not_after: datetime
    status: str


@dataclass(frozen=True, slots=True, init=False)
class VerifiedManifest:
    """Immutable, time-specific local verification result, not a trust root."""
    _keys: tuple[_Key, ...]
    _allowlist: tuple[tuple[tuple[str, str, str], tuple[str, str, str]], ...]
    _now: datetime

    def __init__(self, token, keys, allowlist, now):
        if token is not _TOKEN:
            raise TrustManifestError("manifest must be verified")
        object.__setattr__(self, "_keys", keys)
        object.__setattr__(self, "_allowlist", allowlist)
        object.__setattr__(self, "_now", now)


def verify_manifest(*, raw: bytes, expected_root_public_key: bytes,
                    expected_manifest_version: int, trusted_now: str,
                    currentness_confirmed: bool) -> VerifiedManifest:
    """Verify using independently supplied assumptions, never manifest bootstrap.

    The caller must independently establish the root, current version, clock,
    and that this manifest's revocation view is current. True is an assertion,
    not proof; missing or ambiguous assertions fail closed.
    """
    if (type(expected_manifest_version) is not int or not 1 <= expected_manifest_version <= 9007199254740991
            or currentness_confirmed is not True):
        raise TrustManifestError("missing currentness expectation")
    now = _time(trusted_now)
    try:
        verified = verify_envelope_integrity(expected_kind="trust_anchor_manifest", raw=raw,
                                             expected_public_key=expected_root_public_key)
        obj = strict_json_loads(verified.payload_bytes.decode("utf-8"))
    except (EnvelopeIntegrityError, ValueError, UnicodeError) as exc:
        raise TrustManifestError("invalid manifest integrity") from exc
    _fields(obj, "schema_version record_type root_key_id manifest_version issued_at expires_at revocation_checked_at revocation_valid_until bootstrap_provenance_ref keys allowlist")
    version = obj["manifest_version"]
    if type(version) is not int or version != expected_manifest_version:
        raise TrustManifestError("unexpected manifest version")
    _hash(obj["root_key_id"])
    _id(obj["bootstrap_provenance_ref"])
    issued, expires, checked, until = (_time(obj[k]) for k in
        ("issued_at", "expires_at", "revocation_checked_at", "revocation_valid_until"))
    if not (issued < expires and issued <= checked <= now < until <= expires and issued <= now < expires):
        raise TrustManifestError("invalid manifest time window")
    if type(obj["keys"]) is not list or not 1 <= len(obj["keys"]) <= 32:
        raise TrustManifestError("invalid keys")
    keys = []
    ids = set()
    for item in obj["keys"]:
        _fields(item, "key_id public_key principal_id role not_before not_after status revoked_at")
        key_id = _hash(item["key_id"])
        encoded = item["public_key"]
        try:
            if type(encoded) is not str:
                raise ValueError("invalid public key")
            public = base64.b64decode(encoded, validate=True)
            if len(public) != 32 or base64.b64encode(public).decode("ascii") != encoded:
                raise ValueError("invalid public key")
        except (ValueError, binascii.Error) as exc:
            raise TrustManifestError("invalid public key") from exc
        if key_id in ids or hashlib.sha256(public).hexdigest() != key_id:
            raise TrustManifestError("duplicate or mismatched key ID")
        ids.add(key_id)
        principal = _id(item["principal_id"])
        role, status = item["role"], item["status"]
        if type(role) is not str or role not in _ROLES or type(status) is not str or status not in ("active", "revoked"):
            raise TrustManifestError("invalid role or status")
        before, after = _time(item["not_before"]), _time(item["not_after"])
        if before >= after:
            raise TrustManifestError("invalid key window")
        revoked = item["revoked_at"]
        if (status == "active" and revoked is not None) or (status == "revoked" and (revoked is None or _time(revoked) > now)):
            raise TrustManifestError("invalid revocation")
        keys.append(_Key(key_id, public, principal, role, before, after, status))
    if type(obj["allowlist"]) is not list or not 1 <= len(obj["allowlist"]) <= 32:
        raise TrustManifestError("invalid allowlist")
    allowed = []
    for item in obj["allowlist"]:
        _fields(item, "policy_document contract_document")
        pair = (_document(item["policy_document"]), _document(item["contract_document"]))
        if pair in allowed:
            raise TrustManifestError("duplicate allowlist pair")
        allowed.append(pair)
    return VerifiedManifest(_TOKEN, tuple(keys), tuple(allowed), now)


def select_signer_key(manifest: VerifiedManifest, *, expected_key_id: str,
                      expected_principal_id: str, expected_role: str) -> bytes:
    """Return only the exact active role-bound key at the snapshot time."""
    if type(manifest) is not VerifiedManifest:
        raise TrustManifestError("manifest must be verified")
    _hash(expected_key_id)
    _id(expected_principal_id)
    if type(expected_role) is not str or expected_role not in _ROLES:
        raise TrustManifestError("invalid role")
    for key in manifest._keys:
        if (key.key_id == expected_key_id and key.principal_id == expected_principal_id
                and key.role == expected_role and key.status == "active"
                and key.not_before <= manifest._now < key.not_after):
            return key.public_key
    raise TrustManifestError("no matching active key")


def check_document_pair(manifest: VerifiedManifest, *, policy_document: dict,
                        contract_document: dict) -> None:
    """Match caller-pinned complete document tuples; does not read documents."""
    if type(manifest) is not VerifiedManifest:
        raise TrustManifestError("manifest must be verified")
    if (_document(policy_document), _document(contract_document)) not in manifest._allowlist:
        raise TrustManifestError("document pair mismatch")
