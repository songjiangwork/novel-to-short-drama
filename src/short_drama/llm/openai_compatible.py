from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from jsonschema import Draft202012Validator

from short_drama.artifacts import CanonicalSerializationError, strict_json_loads

from .client import LLMClient
from .config import RuntimeConfig, resolve_auth_header
from .errors import (
    LLMConfigError,
    LLMHTTPError,
    LLMResponseError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTransportError,
)
from .models import (
    LLMInvocationProvenance,
    OutputSchema,
    RenderedPrompt,
    SemanticLLMProfile,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    build_structured_request,
)
from .retry import DEFAULT_MAX_ATTEMPTS, real_sleeper, run_with_retry, validate_max_attempts

CHAT_COMPLETIONS_PATH = "/chat/completions"
_SNIPPET_LIMIT = 200
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """Raw HTTP transport result: status code plus body bytes."""

    status: int
    body: bytes


class LLMTransport(Protocol):
    """Injectable low-level HTTP seam.

    Implementations raise :class:`LLMTimeoutError` / :class:`LLMTransportError`
    for timeout/connection failures and return a :class:`TransportResponse` for
    any HTTP status (including 4xx/5xx), leaving status classification to the
    adapter.
    """

    def post(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> TransportResponse:
        ...


class UrllibTransport:
    """v1 transport built on Python stdlib ``urllib.request`` (no new deps)."""

    def post(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> TransportResponse:
        request = urllib.request.Request(
            url, data=body, method="POST", headers=dict(headers)
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return TransportResponse(status=response.status, body=response.read())
        except urllib.error.HTTPError as exc:
            try:
                body_bytes = exc.read()
            except Exception:  # noqa: BLE001 - a missing body is still an error
                body_bytes = b""
            return TransportResponse(status=exc.code, body=body_bytes)
        except (TimeoutError, socket.timeout) as exc:
            raise LLMTimeoutError(
                f"LLM request timed out after {timeout_seconds:g}s: {url}"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMTimeoutError(
                    f"LLM request timed out after {timeout_seconds:g}s: {url}"
                ) from exc
            raise LLMTransportError(
                f"LLM connection failure for {url}: {exc.reason!r}"
            ) from exc
        except (ConnectionError, OSError) as exc:
            raise LLMTransportError(
                f"LLM connection failure for {url}: {exc!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ProviderMeta:
    """Optional provider metadata; ``None`` means the provider did not return it."""

    response_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


def _bounded_snippet(body: bytes) -> str | None:
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    text = " ".join(text.split())
    if len(text) > _SNIPPET_LIMIT:
        text = text[:_SNIPPET_LIMIT] + "..."
    return text or None


def _normalize_usage(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    normalized: dict[str, int] = {}
    for key in _USAGE_FIELDS:
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            normalized[key] = value
    return normalized or None


def _format_schema_error(error: Any) -> str:
    path = "/".join(str(part) for part in error.absolute_path) or "<root>"
    return f"{path}: {error.message}"


def validate_against_output_schema(
    parsed: Any, output_schema: OutputSchema
) -> None:
    """Local authoritative JSON Schema validation of the model output."""

    validator = Draft202012Validator(output_schema.schema)
    errors = sorted(
        validator.iter_errors(parsed),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            str(error.validator),
            error.message,
        ),
    )
    if errors:
        detail = "; ".join(_format_schema_error(error) for error in errors[:5])
        if len(errors) > 5:
            detail += f"; and {len(errors) - 5} more"
        raise LLMStructuredOutputError(
            f"model output violates output schema: {detail}"
        )


def parse_and_validate_response(
    response: TransportResponse, output_schema: OutputSchema
) -> tuple[Any, ProviderMeta]:
    """Provider-envelope parsing -> strict JSON parsing -> local schema validation.

    Raises typed, correctly-classified errors. The provider envelope is a
    transport concern; the parsed payload is only trusted after local validation.
    """

    if not 200 <= response.status < 300:
        retryable = response.status == 429 or 500 <= response.status < 600
        raise LLMHTTPError(
            f"LLM provider returned HTTP {response.status}",
            status=response.status,
            retryable=retryable,
            detail=_bounded_snippet(response.body),
        )

    try:
        envelope = strict_json_loads(response.body)
    except CanonicalSerializationError as exc:
        raise LLMResponseError(f"malformed provider envelope: {exc}") from exc
    if not isinstance(envelope, dict):
        raise LLMResponseError("malformed provider envelope: not a JSON object")

    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("malformed provider envelope: missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseError("malformed provider envelope: invalid choice entry")

    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("malformed provider envelope: missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise LLMResponseError(
            "malformed provider envelope: message.content is not a string"
        )

    finish_reason = first.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise LLMResponseError("malformed provider envelope: finish_reason not a string")
    response_id = envelope.get("id")
    if response_id is not None and not isinstance(response_id, str):
        raise LLMResponseError("malformed provider envelope: id is not a string")

    if not content.strip():
        raise LLMStructuredOutputError("model returned empty content")

    try:
        parsed = strict_json_loads(content)
    except CanonicalSerializationError as exc:
        raise LLMStructuredOutputError(f"model content is not valid JSON: {exc}") from exc

    validate_against_output_schema(parsed, output_schema)

    meta = ProviderMeta(
        response_id=response_id,
        finish_reason=finish_reason,
        usage=_normalize_usage(envelope.get("usage")),
    )
    return parsed, meta


def build_provenance(
    request: StructuredGenerationRequest, meta: ProviderMeta
) -> LLMInvocationProvenance:
    profile = request.semantic_profile
    rendered = request.rendered_prompt
    schema = request.output_schema
    return LLMInvocationProvenance(
        provider_family=profile.provider_family,
        model=request.model,
        semantic_profile_id=profile.profile_id,
        semantic_profile_hash=profile.semantic_profile_hash,
        prompt_id=rendered.prompt_id,
        prompt_version=rendered.prompt_version,
        prompt_content_hash=rendered.prompt_content_hash,
        rendered_prompt_hash=rendered.rendered_prompt_hash,
        output_schema_id=schema.schema_id,
        output_schema_version=schema.schema_version,
        output_schema_hash=schema.schema_hash,
        request_hash=request.request_hash,
        provider_response_id=meta.response_id,
        finish_reason=meta.finish_reason,
        usage=meta.usage,
    )


class OpenAICompatibleLLMClient(LLMClient):
    """OpenAI-compatible Chat Completions adapter over the injectable transport.

    The first intended runtime provider is a local Qwen model behind an
    OpenAI-compatible (llama.cpp) server, but this class is named and coded
    provider-neutrally: business code never references the model, server,
    localhost, or any provider-specific request/response field.
    """

    supported_structured_output_modes = frozenset({"none", "json_object", "json_schema"})

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        *,
        transport: LLMTransport | None = None,
        sleeper: Callable[[float], None] = real_sleeper,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if not isinstance(runtime_config, RuntimeConfig):
            raise LLMConfigError("runtime_config must be a RuntimeConfig")
        if transport is None:
            transport = UrllibTransport()
        if not callable(sleeper):
            raise LLMConfigError("sleeper must be callable")
        self._runtime = runtime_config
        self._transport = transport
        self._sleeper = sleeper
        self._max_attempts = validate_max_attempts(max_attempts)

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime

    def generate_structured(
        self,
        rendered_prompt: RenderedPrompt,
        output_schema: OutputSchema,
        semantic_profile: SemanticLLMProfile,
    ) -> StructuredGenerationResult:
        request = build_structured_request(
            rendered_prompt=rendered_prompt,
            output_schema=output_schema,
            semantic_profile=semantic_profile,
        )
        self._require_structured_output_capability(semantic_profile)
        return run_with_retry(
            max_attempts=self._max_attempts,
            sleeper=self._sleeper,
            attempt=lambda number: self._attempt(request, number),
        )

    def _require_structured_output_capability(
        self, semantic_profile: SemanticLLMProfile
    ) -> None:
        mode = semantic_profile.structured_output_mode
        if mode not in self.supported_structured_output_modes:
            raise LLMConfigError(
                "adapter does not support structured_output_mode="
                f"{mode!r}; supported={sorted(self.supported_structured_output_modes)}"
            )

    def _attempt(
        self, request: StructuredGenerationRequest, attempt_number: int
    ) -> StructuredGenerationResult:
        url, headers, body = self._map_request(request)
        response = self._transport.post(
            url=url,
            headers=headers,
            body=body,
            timeout_seconds=self._runtime.timeout_seconds,
        )
        parsed, meta = parse_and_validate_response(response, request.output_schema)
        provenance = build_provenance(request, meta)
        return StructuredGenerationResult(
            parsed_json=parsed,
            provenance=provenance,
            attempts=attempt_number,
        )

    def _map_request(
        self, request: StructuredGenerationRequest
    ) -> tuple[str, dict[str, str], bytes]:
        profile = request.semantic_profile
        body: dict[str, Any] = {
            "model": request.model,
            "messages": list(request.messages),
            "temperature": profile.temperature,
            "max_tokens": profile.max_output_tokens,
        }
        mode = profile.structured_output_mode
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.schema_id,
                    "schema": request.output_schema.schema,
                    "strict": True,
                },
            }
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}

        url = self._runtime.base_url.rstrip("/") + CHAT_COMPLETIONS_PATH
        headers = {"Content-Type": "application/json"}
        auth = resolve_auth_header(self._runtime)
        if auth is not None:
            headers["Authorization"] = auth
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        return url, headers, body_bytes
