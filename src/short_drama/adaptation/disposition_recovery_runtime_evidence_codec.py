"""Pure checks of proposed runtime evidence bytes, not runtime verification.

These unratified codecs establish neither truth, discovery, freshness, trust nor
permission. Timeout is only a bounded string: its numeric normalization and
mapping to RuntimeConfig remain OPEN DECISION. No runtime data is obtained here.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from short_drama.artifacts import canonical_json_bytes, strict_json_loads
from short_drama.artifacts.errors import CanonicalSerializationError


class RuntimeEvidenceCodecError(ValueError):
    """The supplied candidate codec or bytes are invalid."""


@dataclass(frozen=True, slots=True)
class EvidenceByteIntegrity:
    """Only a codec label and the SHA-256 of its exact canonical bytes."""

    codec: str
    evidence_sha256: str


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,127}\Z", re.ASCII)
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_MAX_INT = 9007199254740991
_SCHEMAS = {
    "b13.provider-config.v1": {
        "id": "ID", "provider_family": "ID", "request_model": "ID",
        "transport_binding_ref": "ID", "timeout_decimal_seconds": "S",
    },
    "b13.model-config.v1": {
        "id": "ID", "deployment_ref": "ID", "model_revision_ref": "ID",
        "context_limit_tokens": "U+", "discovery_method": "ID",
    },
    "b13.tokenizer-config.v1": {
        "id": "ID", "implementation_ref": "ID", "version": "ID",
        "vocabulary_sha256": "H", "special_tokens_sha256": "H",
        "chat_template_sha256": "H", "discovery_method": "ID",
    },
}


def check_runtime_evidence_bytes(*, raw: bytes, expected_codec: str) -> EvidenceByteIntegrity:
    """Check candidate shape and exact byte identity; no evidence is authenticated.

    A fabricated object can pass. This function does not compare timeout values
    numerically or establish any relationship to a real runtime configuration.
    """
    if (type(expected_codec) is not str or expected_codec not in _SCHEMAS
            or type(raw) is not bytes or not 0 < len(raw) <= 65536):
        raise RuntimeEvidenceCodecError("invalid runtime evidence")
    try:
        obj = strict_json_loads(raw.decode("utf-8"))
        schema = _SCHEMAS[expected_codec]
        if (type(obj) is not dict or set(obj) != set(schema) | {"schema_version", "codec"}
                or type(obj["schema_version"]) is not int or obj["schema_version"] != 1
                or obj["codec"] != expected_codec or type(obj["codec"]) is not str):
            raise RuntimeEvidenceCodecError("invalid runtime evidence")
        for name, kind in schema.items():
            value = obj[name]
            if kind == "U+":
                valid = type(value) is int and 1 <= value <= _MAX_INT
            elif kind == "ID":
                valid = type(value) is str and _ID.fullmatch(value) is not None
            elif kind == "H":
                valid = type(value) is str and _HASH.fullmatch(value) is not None
            else:
                valid = (type(value) is str and 1 <= len(value.encode("utf-8")) <= 256
                         and not any(unicodedata.category(c) in ("Cc", "Cs") for c in value))
            if not valid:
                raise RuntimeEvidenceCodecError("invalid runtime evidence")
        if canonical_json_bytes(obj) != raw:
            raise RuntimeEvidenceCodecError("invalid runtime evidence")
    except RuntimeEvidenceCodecError:
        raise
    except (CanonicalSerializationError, UnicodeError, ValueError, RecursionError):
        raise RuntimeEvidenceCodecError("invalid runtime evidence") from None
    return EvidenceByteIntegrity(expected_codec, hashlib.sha256(raw).hexdigest())
