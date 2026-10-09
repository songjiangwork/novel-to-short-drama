"""v1.2 A6E-1 -- deterministic global-skeleton input packet assembly + measurement.

This module is the **A6E-1 measurement checkpoint** (Issue #90). It constructs
the *exact complete* A6E global-skeleton **input** packet and measures it,
WITHOUT implementing the full A6E global-skeleton semantic pass (no A6E provider
execution, no arc / turning-point / reveal allocation, no A6 persistence, no
CURRENT mutation).

Frozen architecture (``docs/v1.2-A/06-A6-global-story-bible.md`` + the A6
implementation plan, sections 23-24) requires the global-skeleton pass to see the
**whole-story compressed representation**:

    ALL character dossiers          (A6C CharacterAnalysis set)
  + ALL plot-window analyses        (A6D PlotWindowAnalysis set)
  + compact canonical character index   (GlobalIndexBase.character_descriptors)
  + compact canonical location index    (GlobalIndexBase.location_descriptors)
  + canonical relationship summaries    (GlobalIndexBase.relationship_summaries)
  + all state-transition summaries      (GlobalIndexBase.state_transition_summaries)
  + all unresolved entities             (GlobalIndexBase.unresolved_entities)
  + all story conflicts                 (GlobalIndexBase.conflict_summaries)
  + selected core/major event details   (GlobalIndexBase.event_index)

The last seven are exactly the deterministic A5-derived ``GlobalIndexBase``
(produced by A6B). This module therefore assembles a single deterministic packet
from the three components:

    character_analyses    : every A6C CharacterAnalysis, canonical planned order
    plot_window_analyses  : every A6D PlotWindowAnalysis, canonical planned order
    global_index          : the complete A5-derived GlobalIndexBase

No sampling, no first-N truncation, no omission of later windows, and no
independent partial-story concatenation. The packet is serialized as RFC 8785
canonical JSON and measured with the single supported estimator
(``utf8-bytes-div3-v1``, the same authority A6B / A6C / A6D use). It is NOT an
A5-only proxy: it joins the *actual* A6C / A6D semantic outputs.

This module is **measurement/assembly only**: it performs NO provider calls.
The A6E-1 checkpoint measures the exact rendered global-skeleton request so that
``global_skeleton_packet_max_estimated_tokens`` can be frozen from real, complete
input (the staged ceiling authority in the merged #97 contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from short_drama.artifacts.canonical import canonical_json_bytes, content_hash
from short_drama.llm import PromptSpec, RenderedPrompt, render_prompt
from short_drama.paths import REPO_ROOT

from .chunking import estimate_tokens
from .errors import StoryAnalysisSemanticError
from .story_analysis import CharacterAnalysis, PlotWindowAnalysis
from .story_analysis_planning import GlobalIndexBase, StoryAnalysisPlan

# The exact A6E global-skeleton semantic pass prompt (frozen, reviewed).
A6E_GLOBAL_SKELETON_PROMPT_ID = "a6.global-skeleton"
A6E_GLOBAL_SKELETON_PROMPT_VERSION = 1
#: The single prompt variable rendered for the global-skeleton request.
_GLOBAL_SKELETON_PROMPT_VARIABLE = "global_context_json"

#: Default prompt base directory (same convention as the other A6 semantic
#: passes); A6E reuses the tracked prompt family.
DEFAULT_PROMPT_BASE_DIR = REPO_ROOT / "prompts" / "story"


@dataclass(frozen=True, slots=True)
class GlobalSkeletonContext:
    """The deterministic whole-story compressed global-skeleton input packet.

    This is the exact ``global_context_json`` payload supplied to the A6E
    global-skeleton request. It joins the *actual* A6C / A6D semantic outputs
    with the deterministic A5-derived ``GlobalIndexBase``. It carries no
    provider / model / runtime material.
    """

    character_analyses: tuple[CharacterAnalysis, ...]
    plot_window_analyses: tuple[PlotWindowAnalysis, ...]
    global_index: GlobalIndexBase

    # -- canonical payload ------------------------------------------------

    def character_analyses_payload(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.character_analyses]

    def plot_window_analyses_payload(self) -> list[dict[str, Any]]:
        return [w.to_dict() for w in self.plot_window_analyses]

    def global_index_payload(self) -> dict[str, Any]:
        return self.global_index.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_analyses": self.character_analyses_payload(),
            "plot_window_analyses": self.plot_window_analyses_payload(),
            "global_index": self.global_index_payload(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.canonical_bytes().decode("utf-8"))

    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    # -- per-component canonical sizes (measurement) ----------------------

    def _component_bytes(self, payload: Any) -> int:
        return len(canonical_json_bytes(payload))

    def _component_tokens(self, payload: Any) -> int:
        data = canonical_json_bytes(payload)
        return estimate_tokens(data.decode("utf-8"))

    def character_analysis_component(self) -> dict[str, int]:
        payload = self.character_analyses_payload()
        return {
            "count": len(self.character_analyses),
            "canonical_bytes": self._component_bytes(payload),
            "estimated_tokens": self._component_tokens(payload),
        }

    def plot_window_analysis_component(self) -> dict[str, int]:
        payload = self.plot_window_analyses_payload()
        return {
            "count": len(self.plot_window_analyses),
            "canonical_bytes": self._component_bytes(payload),
            "estimated_tokens": self._component_tokens(payload),
        }

    def global_index_component(self) -> dict[str, int]:
        payload = self.global_index_payload()
        return {
            "canonical_bytes": self._component_bytes(payload),
            "estimated_tokens": self._component_tokens(payload),
            "content_hash": self.global_index.content_hash(),
        }


def build_global_skeleton_context(
    character_analyses: Sequence[CharacterAnalysis],
    plot_window_analyses: Sequence[PlotWindowAnalysis],
    global_index: GlobalIndexBase,
) -> GlobalSkeletonContext:
    """Assemble the deterministic whole-story global-skeleton input packet.

    The input sequences are taken verbatim (order preserved) and wrapped as
    immutable tuples. No filtering, sampling, or truncation is applied.
    """
    if not isinstance(global_index, GlobalIndexBase):
        raise StoryAnalysisSemanticError("global_index must be a GlobalIndexBase")
    return GlobalSkeletonContext(
        character_analyses=tuple(character_analyses),
        plot_window_analyses=tuple(plot_window_analyses),
        global_index=global_index,
    )


def validate_global_skeleton_coverage(
    context: GlobalSkeletonContext, plan: StoryAnalysisPlan
) -> None:
    """Fail closed unless the packet has complete, exact coverage.

    Invariants (frozen A6 sections 9 / 16 / 19 / 23):
      * every canonical planned character -> exactly one CharacterAnalysis, no
        missing / duplicate / extra character refs;
      * every planned window -> exactly one PlotWindowAnalysis, in canonical
        planned order, with matching window id / ordinal; no omission of later
        windows and no extra / duplicate windows;
      * the packet's character / window sets exactly match the authoritative A6B
        plan (so the whole-story representation is complete and ordered).
    """
    expected_chars = frozenset(p.character_ref for p in plan.character_packages)
    char_refs = [a.character_ref for a in context.character_analyses]
    seen_chars: set[str] = set()
    for ref in char_refs:
        if ref in seen_chars:
            raise StoryAnalysisSemanticError(
                f"duplicate character analysis in global-skeleton packet: {ref!r}"
            )
        seen_chars.add(ref)
    missing_chars = sorted(expected_chars - seen_chars)
    extra_chars = sorted(seen_chars - expected_chars)
    if missing_chars:
        raise StoryAnalysisSemanticError(
            f"global-skeleton packet missing character analyses: {missing_chars!r}"
        )
    if extra_chars:
        raise StoryAnalysisSemanticError(
            f"global-skeleton packet has extra/unknown character analyses: "
            f"{extra_chars!r}"
        )

    expected_windows = list(plan.windows)
    window_analyses = list(context.plot_window_analyses)
    if len(window_analyses) != len(expected_windows):
        raise StoryAnalysisSemanticError(
            "global-skeleton packet window count "
            f"{len(window_analyses)} != planned window count {len(expected_windows)}"
        )
    for analysis, window in zip(window_analyses, expected_windows):
        if analysis.window_id != window.window_id:
            raise StoryAnalysisSemanticError(
                f"global-skeleton packet window id mismatch: "
                f"{analysis.window_id!r} != {window.window_id!r}"
            )
        if analysis.window_ordinal != window.window_ordinal:
            raise StoryAnalysisSemanticError(
                f"global-skeleton packet window ordinal mismatch for "
                f"{analysis.window_id!r}: {analysis.window_ordinal} != "
                f"{window.window_ordinal}"
            )


def build_global_skeleton_rendered_request(
    context: GlobalSkeletonContext, prompt_spec: PromptSpec
) -> RenderedPrompt:
    """Render the exact A6E global-skeleton provider request.

    The single ``global_context_json`` variable is the RFC 8785 canonical JSON
    of the complete packet. This mirrors the A6C / A6D request-rendering seam
    (``render_prompt``) so the measured request is the *actual* request that the
    A6E pass would send, not an A5-only proxy.
    """
    return render_prompt(
        prompt_spec, {_GLOBAL_SKELETON_PROMPT_VARIABLE: context.canonical_bytes().decode("utf-8")}
    )


def estimate_global_skeleton_request_tokens(rendered: RenderedPrompt) -> int:
    """``utf8-bytes-div3-v1`` estimate of the complete rendered global-skeleton
    request (system + user framing), mirroring A6C / A6D."""
    return estimate_tokens(rendered.system_text) + estimate_tokens(rendered.user_text)
