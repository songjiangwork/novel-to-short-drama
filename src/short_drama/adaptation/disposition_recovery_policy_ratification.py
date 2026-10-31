"""Offline policy contract checks under caller-supplied trust assumptions.

No bootstrap, trusted clock, currentness, principal identity, incident evidence,
real human approval or execution permission is established. The R0D1 digest
allowlist remains an open decision; this result is not a production trust chain.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from short_drama.artifacts import strict_json_loads
from .disposition_recovery_preflight_schema import PreflightStructureError, _scope
from .disposition_recovery_signatures import EnvelopeIntegrityError, verify_envelope_integrity
from .disposition_recovery_trust_manifest import (
    TrustManifestError, check_document_pair, select_signer_key_for_interval,
)


class PolicyRatificationError(ValueError):
    """Offline contract or a caller-provided expectation failed."""


@dataclass(frozen=True, slots=True)
class PolicyRatificationContract:
    """Exact envelope digest only; not an approval or an authorization token."""
    envelope_sha256: str


def verify_policy_ratification(*, raw: bytes, manifest, expected_authority_id: str,
                               expected_scope: dict, trusted_now: str,
                               policy_bytes: bytes, policy_path: str, policy_version: str,
                               contract_bytes: bytes, contract_path: str,
                               contract_version: str) -> PolicyRatificationContract:
    """Check the proposed closed contract against independently supplied assertions.

    Callers must independently establish all expectations, including manifest
    root/currentness and clock. That trust bootstrap is not implemented here.
    Signed reject/defer values are valid contract data, never permission.
    Document bytes are limited to 1 MiB each for this offline interface.
    """
    if (type(raw) is not bytes or not 0 < len(raw) <= 32768
            or type(policy_bytes) is not bytes or len(policy_bytes) > 1048576
            or type(contract_bytes) is not bytes or len(contract_bytes) > 1048576):
        raise PolicyRatificationError("invalid policy ratification")
    try:
        # Untrusted parse is bounded and used only to select a candidate key.
        envelope = strict_json_loads(raw.decode("utf-8"))
        if type(envelope) is not dict or type(envelope.get("payload")) is not dict:
            raise PolicyRatificationError("invalid policy ratification")
        candidate = envelope["payload"]
        public = select_signer_key_for_interval(
            manifest, expected_key_id=candidate.get("key_id"),
            expected_principal_id=expected_authority_id, expected_role="policy_ratifier",
            trusted_now=trusted_now, issued_at=candidate.get("issued_at"),
            expires_at=candidate.get("expires_at"))
        integrity = verify_envelope_integrity(expected_kind="policy_ratification", raw=raw,
                                              expected_public_key=public)
        payload = strict_json_loads(integrity.payload_bytes.decode("utf-8"))
        fields = {"schema_version", "record_type", "key_id", "authority_id", "issued_at", "expires_at",
                  "policy_document", "contract_document", "scope", "decision_a", "decision_b"}
        if (type(payload) is not dict or set(payload) != fields
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or payload["record_type"] != "policy_ratification"
                or payload["authority_id"] != expected_authority_id
                or any(type(payload[k]) is not str or payload[k] not in ("approve", "reject", "defer")
                       for k in ("decision_a", "decision_b"))):
            raise PolicyRatificationError("invalid policy ratification")
        _scope(expected_scope)
        _scope(payload["scope"])
        if payload["scope"] != expected_scope:
            raise PolicyRatificationError("invalid policy ratification")
        policy = {"path": policy_path, "version": policy_version,
                  "sha256": hashlib.sha256(policy_bytes).hexdigest()}
        contract = {"path": contract_path, "version": contract_version,
                    "sha256": hashlib.sha256(contract_bytes).hexdigest()}
        # Validates both independently pinned tuples and their exact allowlisted pair.
        check_document_pair(manifest, policy_document=policy, contract_document=contract)
        if payload["policy_document"] != policy or payload["contract_document"] != contract:
            raise PolicyRatificationError("invalid policy ratification")
    except (PolicyRatificationError, TrustManifestError, PreflightStructureError,
            EnvelopeIntegrityError, UnicodeError, ValueError, RecursionError, TypeError):
        raise PolicyRatificationError("invalid policy ratification") from None
    return PolicyRatificationContract(integrity.envelope_sha256)
