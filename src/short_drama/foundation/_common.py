from __future__ import annotations

import re
from typing import Any

from short_drama.artifacts import ArtifactRef

from .errors import FoundationError

_STORAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def require_text(value: Any, field_name: str, error_type: type[FoundationError]) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise error_type(f"{field_name} must be a non-empty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise error_type(f"{field_name} must contain valid UTF-8 text") from exc
    return value


def require_optional_text(
    value: Any,
    field_name: str,
    error_type: type[FoundationError],
) -> str | None:
    if value is None:
        return None
    return require_text(value, field_name, error_type)


def require_storage_id(value: Any, field_name: str, error_type: type[FoundationError]) -> str:
    require_text(value, field_name, error_type)
    if _STORAGE_ID_RE.fullmatch(value) is None or value.endswith("."):
        raise error_type(
            f"{field_name} must match {_STORAGE_ID_RE.pattern!r} and must not end with '.'"
        )
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_BASENAMES:
        raise error_type(f"{field_name} uses a Windows-reserved filesystem name: {value!r}")
    return value


def require_artifact_ref(
    value: Any,
    field_name: str,
    error_type: type[FoundationError],
) -> ArtifactRef:
    if not isinstance(value, ArtifactRef):
        raise error_type(f"{field_name} must be an ArtifactRef")
    return value


def artifact_ref_sort_key(ref: ArtifactRef) -> tuple[str, str, int, str]:
    return (ref.artifact_type, ref.artifact_id, ref.revision, ref.content_hash)


def require_exact_keys(
    value: Any,
    expected: set[str],
    model_name: str,
    error_type: type[FoundationError],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise error_type(
            f"{model_name} must contain exactly: {', '.join(sorted(expected))}"
        )
    return value
