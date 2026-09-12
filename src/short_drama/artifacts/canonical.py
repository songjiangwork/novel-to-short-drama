from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .errors import CanonicalSerializationError

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def _validate_json_value(value: Any, *, _containers: set[int] | None = None) -> None:
    """Validate the deliberately small JSON value domain used for canonical artifacts.

    Canonicalization is representation-only: it never normalizes user strings, numbers,
    or array ordering. Object keys must already be strings, and non-finite floats are
    rejected instead of relying on Python's non-standard NaN/Infinity JSON extension.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalSerializationError("non-finite floats are not valid canonical JSON")
        return

    if not isinstance(value, (dict, list)):
        raise CanonicalSerializationError(
            f"unsupported canonical JSON value: {type(value).__name__}"
        )

    containers = _containers if _containers is not None else set()
    marker = id(value)
    if marker in containers:
        raise CanonicalSerializationError("cyclic containers are not valid canonical JSON")
    containers.add(marker)
    try:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise CanonicalSerializationError("canonical JSON object keys must be strings")
                _validate_json_value(child, _containers=containers)
        else:
            for child in value:
                _validate_json_value(child, _containers=containers)
    finally:
        containers.remove(marker)


def canonical_json_bytes(value: JSONValue) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a validated JSON-compatible value.

    The v1.2 content hash is SHA-256 over these bytes. No trailing newline is included.
    """

    _validate_json_value(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:  # Defensive: validation should catch these first.
        raise CanonicalSerializationError(str(exc)) from exc
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalSerializationError("canonical JSON strings must be valid UTF-8") from exc


def content_hash(value: JSONValue) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_json_loads(data: bytes | str) -> JSONValue:
    """Parse JSON while rejecting duplicate keys and non-standard numeric constants."""

    def reject_constant(token: str) -> None:
        raise CanonicalSerializationError(f"non-standard JSON numeric constant: {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalSerializationError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CanonicalSerializationError(f"invalid JSON: {exc}") from exc
    _validate_json_value(value)
    return value
