from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from short_drama.io import load_yaml

from .errors import LLMConfigError
from .models import SemanticLLMProfile

RUNTIME_CONFIG_SCHEMA_VERSION = 1
_TRANSPORT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BASE_URL_RE = re.compile(r"^https?://[^\s@/]+(:\d+)?(/.*)?$")


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise LLMConfigError(f"{field_name} must be a non-empty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LLMConfigError(f"{field_name} must contain valid UTF-8 text") from exc
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime transport configuration.

    Describes connection/runtime details only: which endpoint to reach, which
    environment variable holds the credential, and the timeout. It never
    participates in downstream semantic identity.
    """

    schema_version: int
    transport_id: str
    base_url: str
    credential_environment_name: str | None
    timeout_seconds: float

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_CONFIG_SCHEMA_VERSION:
            raise LLMConfigError(
                "RuntimeConfig.schema_version must be "
                f"{RUNTIME_CONFIG_SCHEMA_VERSION}"
            )
        _require_text(self.transport_id, "transport_id")
        if _TRANSPORT_ID_RE.fullmatch(self.transport_id) is None:
            raise LLMConfigError("transport_id must be a safe lowercase storage identifier")
        _require_text(self.base_url, "base_url")
        if _BASE_URL_RE.fullmatch(self.base_url) is None:
            raise LLMConfigError("base_url must be an http(s) URL")
        authority = self.base_url.split("://", 1)[1].split("/", 1)[0]
        if "@" in authority:
            raise LLMConfigError(
                "base_url must not embed credentials; use credential_environment_name"
            )
        if self.credential_environment_name is not None:
            _require_text(self.credential_environment_name, "credential_environment_name")
            if _ENV_NAME_RE.fullmatch(self.credential_environment_name) is None:
                raise LLMConfigError(
                    "credential_environment_name must be a valid environment variable name"
                )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise LLMConfigError("timeout_seconds must be a number")
        timeout = float(self.timeout_seconds)
        if not 0 < timeout <= 3600:
            raise LLMConfigError("timeout_seconds must be within (0, 3600]")
        object.__setattr__(self, "timeout_seconds", timeout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transport_id": self.transport_id,
            "base_url": self.base_url,
            "credential_environment_name": self.credential_environment_name,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeConfig":
        expected = {
            "schema_version",
            "transport_id",
            "base_url",
            "credential_environment_name",
            "timeout_seconds",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise LLMConfigError(
                "RuntimeConfig must contain exactly: " + ", ".join(sorted(expected))
            )
        return cls(**value)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    path = Path(path).expanduser()
    if not path.is_file():
        raise LLMConfigError(f"runtime config not found: {path}")
    try:
        data = load_yaml(path)
    except Exception as exc:  # noqa: BLE001 - report any load failure
        raise LLMConfigError(f"failed to load runtime config: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMConfigError("runtime config must contain an object")
    try:
        return RuntimeConfig.from_dict(data)
    except LLMConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMConfigError(f"invalid runtime config: {exc}") from exc


def load_semantic_profile(path: str | Path) -> SemanticLLMProfile:
    path = Path(path).expanduser()
    if not path.is_file():
        raise LLMConfigError(f"semantic profile not found: {path}")
    try:
        data = load_yaml(path)
    except Exception as exc:  # noqa: BLE001
        raise LLMConfigError(f"failed to load semantic profile: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMConfigError("semantic profile must contain an object")
    try:
        return SemanticLLMProfile.from_dict(data)
    except LLMConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMConfigError(f"invalid semantic profile: {exc}") from exc


def resolve_auth_header(runtime_config: RuntimeConfig) -> str | None:
    """Resolve the Authorization header value from the runtime environment.

    The credential is only ever read from the environment at request time. It is
    never persisted, printed, hashed, or stored in provenance. A missing or
    empty environment variable resolves to no Authorization header (local
    servers typically need no credential).
    """

    if not isinstance(runtime_config, RuntimeConfig):
        raise LLMConfigError("runtime_config must be a RuntimeConfig")
    name = runtime_config.credential_environment_name
    if name is None:
        return None
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return f"Bearer {value}"
