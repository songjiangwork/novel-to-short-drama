from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from short_drama.io import load_yaml

from .errors import LLMConfigError
from .models import SemanticLLMProfile, require_storage_id

RUNTIME_CONFIG_SCHEMA_VERSION = 1
_TRANSPORT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise LLMConfigError(f"{field_name} must be a non-empty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LLMConfigError(f"{field_name} must contain valid UTF-8 text") from exc
    return value


def _validate_base_url(value: Any) -> str:
    """Validate a runtime base URL using proper URL parsing.

    A base URL is the endpoint prefix the adapter appends ``/chat/completions``
    to (e.g. ``http://127.0.0.1:8080/v1``). It must be http(s), carry a
    hostname, embed no credentials, use a valid port, carry no query or
    fragment, and have a path of exactly ``/v1`` (the llama.cpp convention).
    Non-canonical local URLs (trailing slashes, missing ``/v1``, embedded
    credentials, ...) fail closed. Failures are non-retryable configuration
    errors.
    """

    if not isinstance(value, str) or not value:
        raise LLMConfigError("base_url must be a non-empty string")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https"):
        raise LLMConfigError("base_url must use an http or https scheme")
    if not parts.hostname:
        raise LLMConfigError("base_url must include a hostname")
    if parts.username is not None or parts.password is not None:
        raise LLMConfigError(
            "base_url must not embed credentials; use credential_environment_name"
        )
    if parts.query or parts.fragment:
        raise LLMConfigError("base_url must not include a query or fragment")
    # Validate the port explicitly (urlsplit reports out-of-range ports as None).
    netloc = parts.netloc
    if netloc and not netloc.startswith("["):
        if ":" in netloc:
            port_str = netloc.rsplit(":", 1)[1]
            if not port_str.isdigit() or not 1 <= int(port_str) <= 65535:
                raise LLMConfigError(
                    "base_url port must be an integer in [1, 65535]"
                )
    # The adapter appends /chat/completions to this prefix. Requiring the path
    # to be exactly /v1 keeps the request path canonical (the llama.cpp
    # convention) and prevents double-pathing or a silently wrong host.
    if parts.path != "/v1":
        raise LLMConfigError(
            "base_url path must be exactly /v1 "
            f"(the adapter appends /chat/completions): {value!r}"
        )
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
        require_storage_id(self.transport_id, "transport_id", LLMConfigError)
        _validate_base_url(self.base_url)
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
    never persisted, printed, hashed, or stored in provenance. When
    ``credential_environment_name`` is ``None``, unauthenticated transport is
    intentional and valid. When a credential environment-variable name IS
    configured, that variable must exist and be non-empty; otherwise we fail
    closed with a secret-safe :class:`LLMConfigError` before any HTTP request is
    sent. The error message names the variable but never the credential value.
    """

    if not isinstance(runtime_config, RuntimeConfig):
        raise LLMConfigError("runtime_config must be a RuntimeConfig")
    name = runtime_config.credential_environment_name
    if name is None:
        return None
    value = os.environ.get(name)
    if value is None or value == "":
        raise LLMConfigError(
            f"credential environment variable {name!r} is not set or is empty"
        )
    return f"Bearer {value}"
