from __future__ import annotations

import time
from typing import Callable, TypeVar

from .errors import LLMConfigError, LLMError, LLMRetryExhaustedError

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 3
# The contract caps the total attempt budget at 1..3 (initial attempt plus at
# most two retries). This is enforced fail-closed by validate_max_attempts.
MAX_ALLOWED_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.25
_BACKOFF_CAP_SECONDS = 2.0


def validate_max_attempts(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > MAX_ALLOWED_ATTEMPTS
    ):
        raise LLMConfigError(
            f"max_attempts must be an integer in [1, {MAX_ALLOWED_ATTEMPTS}]"
        )
    return value


def compute_backoff_seconds(attempt_number: int) -> float:
    """Deterministic exponential backoff delay before the next attempt.

    ``attempt_number`` is the attempt that just failed (1-based). The delay is
    small and bounded; it is always routed through an injectable sleeper so unit
    tests never sleep for real.
    """

    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int) or attempt_number < 1:
        raise LLMConfigError("attempt_number must be an integer >= 1")
    delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt_number - 1))
    return min(_BACKOFF_CAP_SECONDS, delay)


def real_sleeper(delay_seconds: float) -> None:
    if delay_seconds and delay_seconds > 0:
        time.sleep(delay_seconds)


def run_with_retry(
    *,
    max_attempts: int,
    sleeper: Callable[[float], None],
    attempt: Callable[[int], T],
) -> T:
    """Run ``attempt(n)`` with a bounded retry policy.

    ``attempt`` is invoked with a 1-based attempt number. Retryable ``LLMError``
    subtypes are retried (using the identical semantic request) up to
    ``max_attempts`` total attempts. Non-retryable errors propagate immediately.
    If every attempt is retryable and all fail, ``LLMRetryExhaustedError`` is
    raised. Non-LLM exceptions propagate unchanged.
    """

    # Defensively validate the budget here as well; callers must not be the
    # sole guard against an out-of-range attempt budget.
    max_attempts = validate_max_attempts(max_attempts)
    if not callable(sleeper):
        raise LLMConfigError("sleeper must be callable")
    last_error: LLMError | None = None
    for attempt_number in range(1, max_attempts + 1):
        try:
            return attempt(attempt_number)
        except LLMError as exc:
            last_error = exc
            if not exc.retryable:
                raise
            if attempt_number >= max_attempts:
                raise LLMRetryExhaustedError(
                    attempts=attempt_number,
                    message=(
                        f"LLM generation failed after {attempt_number} attempt(s); "
                        f"last error: {type(exc).__name__}: {exc}"
                    ),
                ) from exc
            sleeper(compute_backoff_seconds(attempt_number))
    assert last_error is not None
    raise LLMRetryExhaustedError(
        attempts=max_attempts,
        message=(
            f"LLM generation failed after {max_attempts} attempt(s); "
            f"last error: {type(last_error).__name__}: {last_error}"
        ),
    ) from last_error
