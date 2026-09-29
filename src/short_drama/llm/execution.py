"""Provider-neutral, execution-only generation options.

These options govern how an already prepared generation is executed.  They are
intentionally absent from semantic requests, hashes, profiles, provenance, and
all artifact/reuse identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import LLMConfigError


PromptContextReuse = Literal["provider_default", "disabled"]


@dataclass(frozen=True, slots=True)
class GenerationExecutionOptions:
    """Execution controls that never affect semantic or reuse identity.

    ``provider_default`` leaves provider prompt-context behavior unspecified.
    ``disabled`` asks a supporting adapter to disable prompt-context reuse for
    this execution only.
    """

    prompt_context_reuse: PromptContextReuse = "provider_default"

    def __post_init__(self) -> None:
        if self.prompt_context_reuse not in ("provider_default", "disabled"):
            raise LLMConfigError(
                "prompt_context_reuse must be 'provider_default' or 'disabled'"
            )
