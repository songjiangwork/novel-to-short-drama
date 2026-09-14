from __future__ import annotations


class LLMError(Exception):
    """Base class for all v1.2 A-I3 common LLM infrastructure failures.

    ``retryable`` is a machine-readable, deterministic classification consumed by
    the bounded retry loop. Configuration/programming failures are non-retryable;
    transient technical failures and invalid structured output are retryable.

    Error messages are secret-safe: they never include credentials, connection
    secrets, or full prompt/raw-provider-response bodies.
    """

    retryable = False


class LLMConfigError(LLMError):
    """Runtime transport config, semantic profile, or a required adapter
    capability is invalid. Non-retryable (programming/configuration failure)."""

    retryable = False


class LLMPromptError(LLMError):
    """A PromptSpec or prompt rendering failed: missing/unexpected variable,
    duplicate declaration, unsupported placeholder, invalid text, or a content
    hash mismatch. Non-retryable."""

    retryable = False


class LLMTransportError(LLMError):
    """Transient transport-level failure (connection failure, DNS, refused).
    Retryable."""

    retryable = True


class LLMTimeoutError(LLMTransportError):
    """The provider request exceeded the configured timeout. Retryable."""

    retryable = True


class LLMHTTPError(LLMError):
    """The provider returned a non-2xx HTTP status. Retryable for HTTP 429 and
    5xx; non-retryable for ordinary request/auth/not-found 4xx failures.

    ``retryable`` is set per instance from the status code. ``detail`` is a
    bounded, truncated snippet of the provider error body (never the request
    headers or any credential).
    """

    def __init__(
        self,
        message: str,
        *,
        status: int,
        retryable: bool,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.detail = detail


class LLMResponseError(LLMError):
    """The provider response envelope is structurally malformed (temporarily).
    Retryable as a temporarily malformed provider envelope."""

    retryable = True


class LLMStructuredOutputError(LLMError):
    """The provider returned content that is empty, invalid JSON, or violates the
    local authoritative output JSON Schema. Retryable as invalid structured
    output. Local validation is the authority, not the provider constraint."""

    retryable = True


class LLMRetryExhaustedError(LLMError):
    """All allowed attempts were used without a valid structured result. Not
    itself retryable: the bounded retry loop has already terminated."""

    retryable = False

    def __init__(self, *, attempts: int, message: str) -> None:
        super().__init__(message)
        self.attempts = attempts
