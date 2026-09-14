"""v1.2 A-I3 — common LLM adapter + versioned prompt registry.

Provider-neutral structured-generation infrastructure shared by the later A3-A6
semantic stages. Business code depends only on :class:`LLMClient` and the typed
semantic models; it never references a concrete model, server, transport field,
provider envelope, credential, or retry loop.

Trust boundary: provider-side structured output is only a generation constraint.
Local strict JSON + JSON Schema validation is always the authority.

A-I3 deliberately provides NO generic persisted raw-LLM cache, no CURRENT
pointer, and no Foundation approval artifacts. It provides reusable primitives:
:class:`LLMRequestFingerprint` and :class:`LLMInvocationProvenance`.
"""

from .client import LLMClient
from .config import (
    RUNTIME_CONFIG_SCHEMA_VERSION,
    RuntimeConfig,
    load_runtime_config,
    load_semantic_profile,
    resolve_auth_header,
)
from .errors import (
    LLMConfigError,
    LLMError,
    LLMHTTPError,
    LLMResponseError,
    LLMRetryExhaustedError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTransportError,
    LLMPromptError,
)
from .models import (
    FINGERPRINT_SCHEMA_VERSION,
    LLM_REQUEST_SCHEMA_VERSION,
    PROMPT_SPEC_SCHEMA_VERSION,
    REASONING_DISABLED_EFFORT,
    SEMANTIC_PROFILE_SCHEMA_VERSION,
    SUPPORTED_REASONING_EFFORTS,
    LLMInvocationProvenance,
    LLMRequestFingerprint,
    OutputSchema,
    PromptSpec,
    ReasoningSettings,
    RenderedPrompt,
    SemanticLLMProfile,
    StructuredGenerationRequest,
    StructuredGenerationResult,
    build_structured_request,
    compute_prompt_content_hash,
    compute_rendered_prompt_hash,
    extract_placeholders,
    require_storage_id,
    substitute_template,
)
from .openai_compatible import (
    CHAT_COMPLETIONS_PATH,
    OpenAICompatibleLLMClient,
    ProviderMeta,
    TransportResponse,
    UrllibTransport,
    build_provenance,
    parse_and_validate_response,
    validate_against_output_schema,
)
from .prompts import PromptRegistry, render_prompt
from .retry import (
    DEFAULT_MAX_ATTEMPTS,
    compute_backoff_seconds,
    real_sleeper,
    run_with_retry,
    validate_max_attempts,
)

__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "DEFAULT_MAX_ATTEMPTS",
    "FINGERPRINT_SCHEMA_VERSION",
    "LLM_REQUEST_SCHEMA_VERSION",
    "LLMConfigError",
    "LLMClient",
    "LLMError",
    "LLMHTTPError",
    "LLMInvocationProvenance",
    "LLMRequestFingerprint",
    "LLMResponseError",
    "LLMRetryExhaustedError",
    "LLMStructuredOutputError",
    "LLMTimeoutError",
    "LLMTransportError",
    "LLMPromptError",
    "OpenAICompatibleLLMClient",
    "PROMPT_SPEC_SCHEMA_VERSION",
    "ProviderMeta",
    "PromptRegistry",
    "PromptSpec",
    "REASONING_DISABLED_EFFORT",
    "ReasoningSettings",
    "RenderedPrompt",
    "RUNTIME_CONFIG_SCHEMA_VERSION",
    "SEMANTIC_PROFILE_SCHEMA_VERSION",
    "SUPPORTED_REASONING_EFFORTS",
    "SemanticLLMProfile",
    "StructuredGenerationRequest",
    "StructuredGenerationResult",
    "TransportResponse",
    "UrllibTransport",
    "build_provenance",
    "build_structured_request",
    "compute_backoff_seconds",
    "compute_prompt_content_hash",
    "compute_rendered_prompt_hash",
    "extract_placeholders",
    "load_runtime_config",
    "require_storage_id",
    "load_semantic_profile",
    "parse_and_validate_response",
    "real_sleeper",
    "render_prompt",
    "resolve_auth_header",
    "run_with_retry",
    "validate_against_output_schema",
    "validate_max_attempts",
]
