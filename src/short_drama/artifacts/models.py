from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .canonical import (
    JSONValue,
    canonical_json_bytes,
    content_hash as hash_json_content,
    strict_json_loads,
)
from .errors import ArtifactValidationError, CanonicalSerializationError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_identity(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ArtifactValidationError(f"{field_name} must be a non-empty string without NUL")
    return value


def _require_revision(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ArtifactValidationError("revision must be an integer >= 1")
    return value


def _require_schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ArtifactValidationError("schema_version must be an integer >= 1")
    return value


def _require_hash(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactValidationError(
            "content_hash must be a lowercase 64-character SHA-256 hex digest"
        )
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_type: str
    artifact_id: str
    revision: int
    content_hash: str

    def __post_init__(self) -> None:
        _require_identity(self.artifact_type, "artifact_type")
        _require_identity(self.artifact_id, "artifact_id")
        _require_revision(self.revision)
        _require_hash(self.content_hash)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "revision": self.revision,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactRef":
        expected = {"artifact_type", "artifact_id", "revision", "content_hash"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ArtifactValidationError("ArtifactRef must contain exactly its four canonical fields")
        return cls(
            artifact_type=value["artifact_type"],
            artifact_id=value["artifact_id"],
            revision=value["revision"],
            content_hash=value["content_hash"],
        )


@dataclass(frozen=True, slots=True)
class ImmutableArtifactEnvelope:
    """Immutable artifact identity plus an immutable snapshot of JSON payload bytes.

    `content_hash` is SHA-256 over canonical JSON containing `schema_version` +
    `payload`.
    Identity, revision, and the hash field itself are intentionally excluded. Including the
    schema version makes ArtifactRef indirectly pin the interpretation contract as well as
    payload bytes, while identical content across revisions retains the same hash.
    """

    artifact_type: str
    artifact_id: str
    revision: int
    schema_version: int
    content_hash: str
    _payload_bytes: bytes

    def __post_init__(self) -> None:
        _require_identity(self.artifact_type, "artifact_type")
        _require_identity(self.artifact_id, "artifact_id")
        _require_revision(self.revision)
        _require_schema_version(self.schema_version)
        _require_hash(self.content_hash)
        if not isinstance(self._payload_bytes, bytes):
            raise ArtifactValidationError("artifact payload snapshot must be bytes")
        try:
            payload = strict_json_loads(self._payload_bytes)
            canonical = canonical_json_bytes(payload)
        except CanonicalSerializationError as exc:
            raise ArtifactValidationError(f"invalid canonical artifact payload: {exc}") from exc
        if canonical != self._payload_bytes:
            raise ArtifactValidationError("artifact payload bytes are not canonical JSON")
        actual_hash = hash_json_content(
            {"schema_version": self.schema_version, "payload": payload}
        )
        if actual_hash != self.content_hash:
            raise ArtifactValidationError("artifact content_hash does not match payload")

    @classmethod
    def create(
        cls,
        *,
        artifact_type: str,
        artifact_id: str,
        revision: int,
        schema_version: int,
        payload: JSONValue,
    ) -> "ImmutableArtifactEnvelope":
        payload_bytes = canonical_json_bytes(payload)
        digest = hash_json_content(
            {"schema_version": schema_version, "payload": strict_json_loads(payload_bytes)}
        )
        return cls(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            revision=revision,
            schema_version=schema_version,
            content_hash=digest,
            _payload_bytes=payload_bytes,
        )

    @property
    def payload(self) -> JSONValue:
        # A fresh parse prevents callers from mutating the stored snapshot in place.
        return strict_json_loads(self._payload_bytes)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(
            artifact_type=self.artifact_type,
            artifact_id=self.artifact_id,
            revision=self.revision,
            content_hash=self.content_hash,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "payload": self.payload,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImmutableArtifactEnvelope":
        expected = {
            "artifact_type",
            "artifact_id",
            "revision",
            "schema_version",
            "content_hash",
            "payload",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ArtifactValidationError(
                "ImmutableArtifactEnvelope must contain exactly its six canonical fields"
            )
        payload_bytes = canonical_json_bytes(value["payload"])
        return cls(
            artifact_type=value["artifact_type"],
            artifact_id=value["artifact_id"],
            revision=value["revision"],
            schema_version=value["schema_version"],
            content_hash=value["content_hash"],
            _payload_bytes=payload_bytes,
        )
