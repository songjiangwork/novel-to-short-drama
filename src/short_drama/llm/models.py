from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from short_drama.artifacts import (
    CanonicalSerializationError,
    canonical_json_bytes,
    content_hash,
    strict_json_loads,
)

from .errors import LLMConfigError, LLMPromptError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STORAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_VAR_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_STRUCTURED_OUTPUT_MODES = frozenset({"none", "json_object", "json_schema"})
# Explicit provider reasoning-effort values the OpenAI-compatible adapter can
# map. Disabled reasoning maps to "none"; enabled reasoning must name one of
# these so the server startup default never silently decides behavior.
SUPPORTED_REASONING_EFFORTS = frozenset({"low", "medium", "high"})
REASONING_DISABLED_EFFORT = "none"

# Schema version constants (bumped explicitly on material change).
# The A-I3 backend/runtime split (request_model / provider_family moved to the
# runtime config and dropped from the semantic profile, the structured request,
# and the request fingerprint) is a material v2 change: the old v1 shapes must
# fail closed. PROMPT_SPEC_SCHEMA_VERSION is unchanged (it was not part of it).
SEMANTIC_PROFILE_SCHEMA_VERSION = 2
PROMPT_SPEC_SCHEMA_VERSION = 1
LLM_REQUEST_SCHEMA_VERSION = 2
FINGERPRINT_SCHEMA_VERSION = 2


def _require_text(value: Any, field_name: str, error_type: type) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise error_type(f"{field_name} must be a non-empty string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise error_type(f"{field_name} must contain valid UTF-8 text") from exc
    return value


def _require_valid_text(value: Any, field_name: str, error_type: type) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise error_type(f"{field_name} must be text without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise error_type(f"{field_name} must contain valid UTF-8 text") from exc
    return value


def _require_positive_int(value: Any, field_name: str, error_type: type) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error_type(f"{field_name} must be an integer >= 1")
    return value


def _require_hash(value: Any, field_name: str, error_type: type) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise error_type(f"{field_name} must be a lowercase 64-char SHA-256 hex digest")
    return value


def require_storage_id(value: Any, field_name: str, error_type: type) -> str:
    """Validate a safe lowercase storage identifier (no path traversal)."""

    _require_text(value, field_name, error_type)
    if STORAGE_ID_RE.fullmatch(value) is None:
        raise error_type(f"{field_name} must be a safe lowercase storage identifier")
    return value


# ---------------------------------------------------------------------------
# Strict deterministic prompt-template placeholder scanning / substitution.
#
# The only supported placeholder syntax is ``{{name}}`` where ``name`` matches
# ``_VAR_NAME_RE``. A ``{{`` that is not immediately a valid ``{{name}}`` is an
# unsupported placeholder and fails closed. Single braces (``{`` / ``}``) are
# literal text so JSON examples in prompts pass through untouched.
# ---------------------------------------------------------------------------


def _scan_template(template: str) -> list[str]:
    names: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        if template[i : i + 2] == "{{":
            end = template.find("}}", i + 2)
            if end == -1:
                raise LLMPromptError(
                    "unsupported placeholder: unterminated '{{' "
                    f"(near {template[i : i + 24]!r})"
                )
            inner = template[i + 2 : end]
            if _VAR_NAME_RE.fullmatch(inner) is None:
                raise LLMPromptError(f"unsupported placeholder: {inner!r}")
            names.append(inner)
            i = end + 2
        else:
            i += 1
    return names


def extract_placeholders(template: str) -> set[str]:
    """Return the set of ``{{name}}`` placeholder names in a template.

    Fails closed on unsupported placeholder syntax.
    """

    _require_valid_text(template, "template", LLMPromptError)
    return set(_scan_template(template))


def substitute_template(template: str, variables: Mapping[str, str]) -> str:
    """Strictly substitute ``{{name}}`` placeholders with ``variables[name]``.

    Fails closed on a missing variable or an unsupported placeholder. No silent
    defaults or undefined-to-empty substitution is ever performed.
    """

    out: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        if template[i : i + 2] == "{{":
            end = template.find("}}", i + 2)
            if end == -1:
                raise LLMPromptError(
                    "unsupported placeholder: unterminated '{{' "
                    f"(near {template[i : i + 24]!r})"
                )
            inner = template[i + 2 : end]
            if _VAR_NAME_RE.fullmatch(inner) is None:
                raise LLMPromptError(f"unsupported placeholder: {inner!r}")
            if inner not in variables:
                raise LLMPromptError(f"missing variable: {inner!r}")
            value = variables[inner]
            if not isinstance(value, str):
                raise LLMPromptError(f"variable {inner!r} must be a string value")
            out.append(value)
            i = end + 2
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Semantic LLM profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReasoningSettings:
    """Coherent reasoning semantics.

    Invariants (enforced in Python and mirrored in the semantic-profile schema):
      * disabled reasoning (``enabled`` False) MUST carry ``effort is None``;
      * enabled reasoning (``enabled`` True) MUST carry an explicit supported
        effort (one of :data:`SUPPORTED_REASONING_EFFORTS`).

    The adapter maps disabled reasoning to the provider request
    ``reasoning_effort: "none"`` and enabled reasoning to
    ``reasoning_effort: <effort>``, so the provider request always carries an
    explicit value and the server default never decides behavior.
    """

    enabled: bool
    effort: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise LLMConfigError("reasoning.enabled must be a boolean")
        if self.enabled:
            if self.effort is None:
                raise LLMConfigError(
                    "reasoning.effort is required when reasoning is enabled"
                )
            if not isinstance(self.effort, str) or self.effort not in SUPPORTED_REASONING_EFFORTS:
                raise LLMConfigError(
                    "reasoning.effort must be one of: "
                    + ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
                )
        else:
            if self.effort is not None:
                raise LLMConfigError(
                    "reasoning.effort must be null when reasoning is disabled"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "effort": self.effort}

    @property
    def request_effort(self) -> str:
        """The explicit provider ``reasoning_effort`` value to send."""

        return self.effort if self.enabled else REASONING_DISABLED_EFFORT

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReasoningSettings":
        if not isinstance(value, dict) or set(value) != {"enabled", "effort"}:
            raise LLMConfigError(
                "reasoning must contain exactly: enabled, effort"
            )
        return cls(enabled=value["enabled"], effort=value["effort"])


@dataclass(frozen=True, slots=True)
class SemanticLLMProfile:
    """Result-affecting generation semantics.

    This is deliberately separate from runtime transport configuration: it
    carries no endpoint, timeout, credential, or hostname, so its hash is a
    stable semantic identity that is not invalidated by connection changes.

    It is ALSO deliberately separate from the backend routing identity: it
    carries no ``provider_family`` and no concrete ``model`` name. Those are the
    *requested routing backend* (declared by the operator and supplied via
    ``RuntimeConfig.provider_family`` / ``RuntimeConfig.request_model``), not the
    result-affecting generation semantics, so changing the backend (e.g. Qwen ->
    Gemma) must NOT invalidate this semantic identity or any downstream
    CandidateExtraction reuse identity.
    """

    schema_version: int
    profile_id: str
    temperature: float
    max_output_tokens: int
    structured_output_mode: str
    reasoning: ReasoningSettings

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_PROFILE_SCHEMA_VERSION:
            raise LLMConfigError(
                "SemanticLLMProfile.schema_version must be "
                f"{SEMANTIC_PROFILE_SCHEMA_VERSION}"
            )
        require_storage_id(self.profile_id, "profile_id", LLMConfigError)
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise LLMConfigError("temperature must be a number")
        temperature = float(self.temperature)
        if not 0.0 <= temperature <= 2.0:
            raise LLMConfigError("temperature must be within [0.0, 2.0]")
        object.__setattr__(self, "temperature", temperature)
        _require_positive_int(self.max_output_tokens, "max_output_tokens", LLMConfigError)
        if self.structured_output_mode not in _STRUCTURED_OUTPUT_MODES:
            raise LLMConfigError(
                "structured_output_mode must be one of: "
                + ", ".join(sorted(_STRUCTURED_OUTPUT_MODES))
            )
        if not isinstance(self.reasoning, ReasoningSettings):
            raise LLMConfigError("reasoning must be ReasoningSettings")

    @property
    def semantic_profile_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "structured_output_mode": self.structured_output_mode,
            "reasoning": self.reasoning.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticLLMProfile":
        expected = {
            "schema_version",
            "profile_id",
            "temperature",
            "max_output_tokens",
            "structured_output_mode",
            "reasoning",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise LLMConfigError(
                "SemanticLLMProfile must contain exactly: " + ", ".join(sorted(expected))
            )
        reasoning = value["reasoning"]
        if not isinstance(reasoning, dict):
            raise LLMConfigError("SemanticLLMProfile.reasoning must be an object")
        return cls(
            schema_version=value["schema_version"],
            profile_id=value["profile_id"],
            temperature=value["temperature"],
            max_output_tokens=value["max_output_tokens"],
            structured_output_mode=value["structured_output_mode"],
            reasoning=ReasoningSettings.from_dict(reasoning),
        )


# ---------------------------------------------------------------------------
# Output schema (pinned identity for structured generation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutputSchema:
    """Pinned output ``schema_id + schema_version + schema_hash``.

    The schema is stored as immutable canonical bytes. It is validated as a
    well-formed Draft 2020-12 JSON Schema; an invalid local output schema is a
    non-retryable configuration failure.
    """

    schema_id: str
    schema_version: int
    schema_hash: str
    _schema_bytes: bytes

    def __post_init__(self) -> None:
        require_storage_id(self.schema_id, "schema_id", LLMConfigError)
        _require_positive_int(self.schema_version, "schema_version", LLMConfigError)
        _require_hash(self.schema_hash, "schema_hash", LLMConfigError)
        if not isinstance(self._schema_bytes, bytes):
            raise LLMConfigError("schema snapshot must be bytes")
        try:
            schema = strict_json_loads(self._schema_bytes)
        except CanonicalSerializationError as exc:
            raise LLMConfigError(f"invalid schema snapshot: {exc}") from exc
        if not isinstance(schema, dict):
            raise LLMConfigError("schema must be a JSON object")
        self._check_valid_schema(schema)
        if self.schema_hash != content_hash(schema):
            raise LLMConfigError("schema_hash does not match schema content")

    @staticmethod
    def _check_valid_schema(schema: dict[str, Any]) -> None:
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:  # noqa: BLE001 - report any schema defect
            raise LLMConfigError(f"invalid output JSON Schema: {exc}") from exc

    @property
    def schema(self) -> dict[str, Any]:
        return strict_json_loads(self._schema_bytes)

    @classmethod
    def create(
        cls, *, schema_id: str, schema_version: int, schema: dict[str, Any]
    ) -> "OutputSchema":
        if not isinstance(schema, dict):
            raise LLMConfigError("schema must be a JSON object")
        # Canonicalization / strict-snapshot construction of a user-supplied
        # schema can fail (e.g. a non-canonical value such as NaN/Inf, a
        # non-finite float, or an unserializable type). Any such failure is a
        # non-retryable configuration error and must surface as LLMConfigError,
        # never a raw CanonicalSerializationError / ArtifactError / TypeError /
        # ValueError.
        try:
            schema_bytes = canonical_json_bytes(schema)
            snapshot = strict_json_loads(schema_bytes)
            schema_hash = content_hash(schema)
        except (CanonicalSerializationError, TypeError, ValueError) as exc:
            raise LLMConfigError(f"invalid output schema: {exc}") from exc
        cls._check_valid_schema(snapshot)
        return cls(
            schema_id=schema_id,
            schema_version=schema_version,
            schema_hash=schema_hash,
            _schema_bytes=schema_bytes,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OutputSchema":
        if not isinstance(value, dict) or set(value) != {
            "schema_id",
            "schema_version",
            "schema_hash",
            "schema",
        }:
            raise LLMConfigError(
                "OutputSchema must contain exactly: "
                "schema_id, schema_version, schema_hash, schema"
            )
        # As in create(): any canonicalization failure of a user-supplied schema
        # must surface as a non-retryable LLMConfigError, not a raw parser error.
        try:
            schema_bytes = canonical_json_bytes(value["schema"])
        except (CanonicalSerializationError, TypeError, ValueError) as exc:
            raise LLMConfigError(f"invalid output schema: {exc}") from exc
        return cls(
            schema_id=value["schema_id"],
            schema_version=value["schema_version"],
            schema_hash=value["schema_hash"],
            _schema_bytes=schema_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "schema": self.schema,
        }


# ---------------------------------------------------------------------------
# PromptSpec + RenderedPrompt
# ---------------------------------------------------------------------------


def compute_prompt_content_hash(
    *,
    prompt_id: str,
    version: int,
    system_template: str,
    user_template: str,
    required_variables: Any,
) -> str:
    """Canonical hash over the exact prompt semantic material.

    Covers the exact system/user template text, semantic metadata, and the
    normalized required-variable list -- not merely the metadata file.
    """

    return content_hash(
        {
            "prompt_spec_schema_version": PROMPT_SPEC_SCHEMA_VERSION,
            "prompt_id": prompt_id,
            "version": version,
            "system_template": system_template,
            "user_template": user_template,
            "required_variables": sorted(required_variables),
        }
    )


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    version: int
    system_template: str
    user_template: str
    required_variables: tuple[str, ...]
    content_hash: str

    def __post_init__(self) -> None:
        require_storage_id(self.prompt_id, "prompt_id", LLMPromptError)
        _require_positive_int(self.version, "version", LLMPromptError)
        _require_valid_text(self.system_template, "system_template", LLMPromptError)
        _require_text(self.user_template, "user_template", LLMPromptError)

        raw_variables = tuple(self.required_variables)
        for name in raw_variables:
            if not isinstance(name, str) or _VAR_NAME_RE.fullmatch(name) is None:
                raise LLMPromptError(f"invalid required_variable: {name!r}")
        if len(raw_variables) != len(set(raw_variables)):
            raise LLMPromptError("required_variables must not contain duplicates")
        object.__setattr__(self, "required_variables", tuple(sorted(raw_variables)))

        _require_hash(self.content_hash, "content_hash", LLMPromptError)
        expected = compute_prompt_content_hash(
            prompt_id=self.prompt_id,
            version=self.version,
            system_template=self.system_template,
            user_template=self.user_template,
            required_variables=self.required_variables,
        )
        if self.content_hash != expected:
            raise LLMPromptError("content_hash does not match prompt content")

        actual = extract_placeholders(self.system_template) | extract_placeholders(
            self.user_template
        )
        required = set(self.required_variables)
        if actual != required:
            raise LLMPromptError(
                "required_variables must exactly match template placeholders; "
                f"missing={sorted(actual - required)}, "
                f"unexpected={sorted(required - actual)}"
            )

    @classmethod
    def create(
        cls,
        *,
        prompt_id: str,
        version: int,
        system_template: str,
        user_template: str,
        required_variables: Any,
    ) -> "PromptSpec":
        return cls(
            prompt_id=prompt_id,
            version=version,
            system_template=system_template,
            user_template=user_template,
            required_variables=tuple(required_variables),
            content_hash=compute_prompt_content_hash(
                prompt_id=prompt_id,
                version=version,
                system_template=system_template,
                user_template=user_template,
                required_variables=required_variables,
            ),
        )

    def render(self, variables: Any) -> "RenderedPrompt":
        """Convenience wrapper matching the documented A-I4 usage.

        Delegates to the strict deterministic renderer in ``prompts``.
        """

        from .prompts import render_prompt

        return render_prompt(self, variables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "system_template": self.system_template,
            "user_template": self.user_template,
            "required_variables": list(self.required_variables),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PromptSpec":
        if not isinstance(value, dict) or set(value) != {
            "prompt_id",
            "version",
            "system_template",
            "user_template",
            "required_variables",
            "content_hash",
        }:
            raise LLMPromptError(
                "PromptSpec must contain exactly: prompt_id, version, "
                "system_template, user_template, required_variables, content_hash"
            )
        return cls(
            prompt_id=value["prompt_id"],
            version=value["version"],
            system_template=value["system_template"],
            user_template=value["user_template"],
            required_variables=tuple(value["required_variables"]),
            content_hash=value["content_hash"],
        )


def compute_rendered_prompt_hash(
    *,
    prompt_id: str,
    prompt_version: int,
    prompt_content_hash: str,
    variables_hash: str,
    system_text: str,
    user_text: str,
) -> str:
    return content_hash(
        {
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "prompt_content_hash": prompt_content_hash,
            "variables_hash": variables_hash,
            "system_text": system_text,
            "user_text": user_text,
        }
    )


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    variables_hash: str
    system_text: str
    user_text: str
    rendered_prompt_hash: str

    def __post_init__(self) -> None:
        _require_text(self.prompt_id, "prompt_id", LLMPromptError)
        _require_positive_int(self.prompt_version, "prompt_version", LLMPromptError)
        _require_hash(self.prompt_content_hash, "prompt_content_hash", LLMPromptError)
        _require_hash(self.variables_hash, "variables_hash", LLMPromptError)
        _require_valid_text(self.system_text, "system_text", LLMPromptError)
        _require_text(self.user_text, "user_text", LLMPromptError)
        _require_hash(self.rendered_prompt_hash, "rendered_prompt_hash", LLMPromptError)
        expected = compute_rendered_prompt_hash(
            prompt_id=self.prompt_id,
            prompt_version=self.prompt_version,
            prompt_content_hash=self.prompt_content_hash,
            variables_hash=self.variables_hash,
            system_text=self.system_text,
            user_text=self.user_text,
        )
        if self.rendered_prompt_hash != expected:
            raise LLMPromptError("rendered_prompt_hash does not match rendered content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_content_hash": self.prompt_content_hash,
            "variables_hash": self.variables_hash,
            "system_text": self.system_text,
            "user_text": self.user_text,
            "rendered_prompt_hash": self.rendered_prompt_hash,
        }


# ---------------------------------------------------------------------------
# Structured generation request + fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructuredGenerationRequest:
    """Provider-neutral semantic request.

    Carries no endpoint, timeout, credential, or hostname; only result-affecting
    semantics. It also carries NO backend routing identity: the *requested
    routing model* lives in ``RuntimeConfig.request_model`` and is supplied by the
    transport adapter, not by this request. That keeps
    ``request_hash`` (the canonical semantic request identity) independent of the
    backend in use, so changing the backend does not invalidate reuse.
    """

    schema_version: int
    output_schema: OutputSchema
    semantic_profile: SemanticLLMProfile
    rendered_prompt: RenderedPrompt

    def __post_init__(self) -> None:
        if self.schema_version != LLM_REQUEST_SCHEMA_VERSION:
            raise LLMConfigError(
                "StructuredGenerationRequest.schema_version must be "
                f"{LLM_REQUEST_SCHEMA_VERSION}"
            )
        if not isinstance(self.output_schema, OutputSchema):
            raise LLMConfigError("output_schema must be OutputSchema")
        if not isinstance(self.semantic_profile, SemanticLLMProfile):
            raise LLMConfigError("semantic_profile must be SemanticLLMProfile")
        if not isinstance(self.rendered_prompt, RenderedPrompt):
            raise LLMPromptError("rendered_prompt must be RenderedPrompt")

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        messages: list[dict[str, str]] = []
        if self.rendered_prompt.system_text:
            messages.append(
                {"role": "system", "content": self.rendered_prompt.system_text}
            )
        messages.append({"role": "user", "content": self.rendered_prompt.user_text})
        return tuple(messages)

    def semantic_request_material(self) -> dict[str, Any]:
        """Exact object whose canonical bytes define the semantic request identity."""

        return {
            "request_schema_version": LLM_REQUEST_SCHEMA_VERSION,
            "semantic_profile": self.semantic_profile.to_dict(),
            "rendered_prompt": self.rendered_prompt.to_dict(),
            "output_schema": {
                "schema_id": self.output_schema.schema_id,
                "schema_version": self.output_schema.schema_version,
                "schema": self.output_schema.schema,
                "schema_hash": self.output_schema.schema_hash,
            },
        }

    @property
    def request_hash(self) -> str:
        return content_hash(self.semantic_request_material())

    @property
    def fingerprint(self) -> "LLMRequestFingerprint":
        rendered = self.rendered_prompt
        return LLMRequestFingerprint(
            schema_version=FINGERPRINT_SCHEMA_VERSION,
            semantic_profile_hash=self.semantic_profile.semantic_profile_hash,
            prompt_id=rendered.prompt_id,
            prompt_version=rendered.prompt_version,
            prompt_content_hash=rendered.prompt_content_hash,
            rendered_prompt_hash=rendered.rendered_prompt_hash,
            output_schema_hash=self.output_schema.schema_hash,
            request_hash=self.request_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "messages": list(self.messages),
            "output_schema": self.output_schema.to_dict(),
            "semantic_profile": self.semantic_profile.to_dict(),
            "rendered_prompt": self.rendered_prompt.to_dict(),
            "request_hash": self.request_hash,
        }


def build_structured_request(
    *,
    rendered_prompt: RenderedPrompt,
    output_schema: OutputSchema,
    semantic_profile: SemanticLLMProfile,
) -> StructuredGenerationRequest:
    if not isinstance(rendered_prompt, RenderedPrompt):
        raise LLMPromptError("rendered_prompt must be a RenderedPrompt")
    if not isinstance(output_schema, OutputSchema):
        raise LLMConfigError("output_schema must be an OutputSchema")
    if not isinstance(semantic_profile, SemanticLLMProfile):
        raise LLMConfigError("semantic_profile must be a SemanticLLMProfile")
    return StructuredGenerationRequest(
        schema_version=LLM_REQUEST_SCHEMA_VERSION,
        output_schema=output_schema,
        semantic_profile=semantic_profile,
        rendered_prompt=rendered_prompt,
    )


@dataclass(frozen=True, slots=True)
class LLMRequestFingerprint:
    """Semantic request identity.

    Identifies the semantic request only; it does NOT guarantee byte-identical
    model output on re-invocation. Runtime transport details must never change
    it, and the declared backend routing model does NOT participate in it either:
    that is backend routing identity (``RuntimeConfig.request_model`` / the
    invocation provenance), not the semantic request identity.
    """

    schema_version: int
    semantic_profile_hash: str
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    rendered_prompt_hash: str
    output_schema_hash: str
    request_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != FINGERPRINT_SCHEMA_VERSION:
            raise LLMConfigError(
                "LLMRequestFingerprint.schema_version must be "
                f"{FINGERPRINT_SCHEMA_VERSION}"
            )
        _require_hash(self.semantic_profile_hash, "semantic_profile_hash", LLMConfigError)
        _require_text(self.prompt_id, "prompt_id", LLMConfigError)
        _require_positive_int(self.prompt_version, "prompt_version", LLMConfigError)
        _require_hash(self.prompt_content_hash, "prompt_content_hash", LLMConfigError)
        _require_hash(self.rendered_prompt_hash, "rendered_prompt_hash", LLMConfigError)
        _require_hash(self.output_schema_hash, "output_schema_hash", LLMConfigError)
        _require_hash(self.request_hash, "request_hash", LLMConfigError)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_profile_hash": self.semantic_profile_hash,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_content_hash": self.prompt_content_hash,
            "rendered_prompt_hash": self.rendered_prompt_hash,
            "output_schema_hash": self.output_schema_hash,
            "request_hash": self.request_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LLMRequestFingerprint":
        expected = {
            "schema_version",
            "semantic_profile_hash",
            "prompt_id",
            "prompt_version",
            "prompt_content_hash",
            "rendered_prompt_hash",
            "output_schema_hash",
            "request_hash",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise LLMConfigError(
                "LLMRequestFingerprint must contain exactly: "
                + ", ".join(sorted(expected))
            )
        return cls(**value)


# ---------------------------------------------------------------------------
# Invocation provenance + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMInvocationProvenance:
    """Exact provenance of one provider call (A-I3).

    ``provider_family`` and ``model`` record the *declared backend routing
    metadata* for the call, NOT an independently observed or verified statement
    of the actual backend implementation that served the request. They are
    sourced from the ``RuntimeConfig`` in effect when the call was made
    (``provider_family`` / ``request_model``), NOT from the semantic profile.
    They are recorded metadata, not part of the semantic/reuse identity (see
    :func:`request_semantic_fields`), so changing the backend changes what is
    recorded without invalidating reuse.
    """

    provider_family: str
    model: str
    semantic_profile_id: str
    semantic_profile_hash: str
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    rendered_prompt_hash: str
    output_schema_id: str
    output_schema_version: int
    output_schema_hash: str
    request_hash: str
    provider_response_id: str | None
    finish_reason: str | None
    usage: dict[str, int] | None

    def __post_init__(self) -> None:
        _require_text(self.provider_family, "provider_family", LLMConfigError)
        _require_text(self.model, "model", LLMConfigError)
        _require_text(self.semantic_profile_id, "semantic_profile_id", LLMConfigError)
        _require_hash(self.semantic_profile_hash, "semantic_profile_hash", LLMConfigError)
        _require_text(self.prompt_id, "prompt_id", LLMConfigError)
        _require_positive_int(self.prompt_version, "prompt_version", LLMConfigError)
        _require_hash(self.prompt_content_hash, "prompt_content_hash", LLMConfigError)
        _require_hash(self.rendered_prompt_hash, "rendered_prompt_hash", LLMConfigError)
        _require_text(self.output_schema_id, "output_schema_id", LLMConfigError)
        _require_positive_int(self.output_schema_version, "output_schema_version", LLMConfigError)
        _require_hash(self.output_schema_hash, "output_schema_hash", LLMConfigError)
        _require_hash(self.request_hash, "request_hash", LLMConfigError)
        if self.provider_response_id is not None:
            _require_text(self.provider_response_id, "provider_response_id", LLMConfigError)
        if self.finish_reason is not None:
            _require_text(self.finish_reason, "finish_reason", LLMConfigError)
        if self.usage is not None:
            try:
                canonical_json_bytes(self.usage)
            except CanonicalSerializationError as exc:
                raise LLMConfigError(f"usage must be canonical JSON: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_family": self.provider_family,
            "model": self.model,
            "semantic_profile_id": self.semantic_profile_id,
            "semantic_profile_hash": self.semantic_profile_hash,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_content_hash": self.prompt_content_hash,
            "rendered_prompt_hash": self.rendered_prompt_hash,
            "output_schema_id": self.output_schema_id,
            "output_schema_version": self.output_schema_version,
            "output_schema_hash": self.output_schema_hash,
            "request_hash": self.request_hash,
            "provider_response_id": self.provider_response_id,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


@dataclass(frozen=True, slots=True)
class StructuredGenerationResult:
    parsed_json: Any
    provenance: LLMInvocationProvenance
    attempts: int

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, LLMInvocationProvenance):
            raise LLMConfigError("provenance must be LLMInvocationProvenance")
        _require_positive_int(self.attempts, "attempts", LLMConfigError)
        try:
            canonical_json_bytes(self.parsed_json)
        except CanonicalSerializationError as exc:
            raise LLMConfigError(f"parsed_json must be canonical JSON: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed_json": self.parsed_json,
            "provenance": self.provenance.to_dict(),
            "attempts": self.attempts,
        }
