from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .execution import GenerationExecutionOptions
from .models import (
    OutputSchema,
    RenderedPrompt,
    SemanticLLMProfile,
    StructuredGenerationResult,
)


class LLMClient(ABC):
    """Provider-neutral structured-generation boundary.

    Business stages depend only on :meth:`generate_structured`. They must not
    know about transport fields, provider response envelopes, HTTP headers,
    credentials, retry loops, or a specific model/server.

    ``supported_structured_output_modes`` declares the structured-output
    capabilities the concrete adapter can express. A semantic profile that
    requires a mode the adapter cannot provide fails closed before any request
    is sent (no silent fallback to plain text).
    """

    supported_structured_output_modes: ClassVar[frozenset[str]] = frozenset({"none"})
    # Adapters must opt in explicitly before one instance is shared by the A5
    # bounded executor.  The conservative default keeps future adapters from
    # accidentally acquiring a thread-safety contract merely by subclassing.
    supports_concurrent_calls: ClassVar[bool] = False

    @abstractmethod
    def generate_structured(
        self,
        rendered_prompt: RenderedPrompt,
        output_schema: OutputSchema,
        semantic_profile: SemanticLLMProfile,
        *,
        execution_options: GenerationExecutionOptions | None = None,
    ) -> StructuredGenerationResult:
        """Generate and locally validate a structured result.

        Always performs strict local JSON + JSON Schema validation as the
        authoritative trust boundary, regardless of any provider-side
        structured-output constraint.

        A non-default ``execution_options`` value must be supported and
        applied by a concrete adapter; it must never be silently ignored.
        """
        raise NotImplementedError
