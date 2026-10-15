"""A6F deterministic whole-story packet assembly (no provider calls)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from short_drama.artifacts.canonical import canonical_json_bytes, content_hash
from short_drama.llm import PromptSpec, RenderedPrompt, render_prompt

from .chunking import estimate_tokens
from .errors import StoryAnalysisSemanticError
from .story_analysis import ArcAnalysis, CharacterAnalysis, GlobalEventAnalysis, GlobalStructure
from .story_analysis_planning import GlobalIndexBase, StoryAnalysisPlan

A6F_STORY_BIBLE_PROMPT_ID = "a6.story-bible-synthesis"
A6F_STORY_BIBLE_PROMPT_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoryBibleContext:
    """Complete compressed story representation for the final A6 semantic pass.

    Window analyses occur exactly once, inside ``global_event_analysis``.  This
    deliberately avoids a second, divergent copy while retaining every A6C/D/E
    semantic result plus the compact A5 global index and its unresolved/conflict
    evidence.
    """
    character_analyses: tuple[CharacterAnalysis, ...]
    global_event_analysis: GlobalEventAnalysis
    arc_analysis: ArcAnalysis
    global_structure: GlobalStructure
    global_index: GlobalIndexBase

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_analyses": [item.to_dict() for item in self.character_analyses],
            "global_event_analysis": self.global_event_analysis.to_dict(),
            "arc_analysis": self.arc_analysis.to_dict(),
            "global_structure": self.global_structure.to_dict(),
            "global_index": self.global_index.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.canonical_bytes().decode("utf-8"))


def build_story_bible_context(
    character_analyses: Sequence[CharacterAnalysis],
    global_event_analysis: GlobalEventAnalysis,
    arc_analysis: ArcAnalysis,
    global_structure: GlobalStructure,
    global_index: GlobalIndexBase,
) -> StoryBibleContext:
    if not isinstance(global_event_analysis, GlobalEventAnalysis):
        raise StoryAnalysisSemanticError("global_event_analysis must be GlobalEventAnalysis")
    if not isinstance(arc_analysis, ArcAnalysis) or not isinstance(global_structure, GlobalStructure):
        raise StoryAnalysisSemanticError("A6E outputs must be typed artifacts")
    if not isinstance(global_index, GlobalIndexBase):
        raise StoryAnalysisSemanticError("global_index must be GlobalIndexBase")
    return StoryBibleContext(tuple(character_analyses), global_event_analysis, arc_analysis, global_structure, global_index)


def validate_story_bible_context(context: StoryBibleContext, plan: StoryAnalysisPlan) -> None:
    expected_chars = [p.character_ref for p in plan.character_packages]
    got_chars = [a.character_ref for a in context.character_analyses]
    if got_chars != expected_chars or len(set(got_chars)) != len(got_chars):
        raise StoryAnalysisSemanticError("story-bible packet lacks exact ordered character coverage")
    windows = context.global_event_analysis.windows
    if len(windows) != len(plan.windows):
        raise StoryAnalysisSemanticError("story-bible packet lacks complete plot-window coverage")
    for got, expected in zip(windows, plan.windows):
        if (got.window_id, got.window_ordinal, tuple(got.owned_event_refs), tuple(got.context_event_refs)) != (
            expected.window_id, expected.window_ordinal, tuple(expected.owned_event_ids), tuple(expected.context_event_ids)
        ):
            raise StoryAnalysisSemanticError("story-bible packet window differs from exact plan")
    if context.global_event_analysis.plot_window_plan_hash != plan.plan_hash:
        raise StoryAnalysisSemanticError("GlobalEventAnalysis plot-window plan hash mismatch")
    if context.global_index.content_hash() != plan.global_index.content_hash():
        raise StoryAnalysisSemanticError("story-bible packet global index differs from exact plan")


def build_story_bible_rendered_request(context: StoryBibleContext, prompt_spec: PromptSpec) -> RenderedPrompt:
    return render_prompt(prompt_spec, {"story_context_json": context.canonical_bytes().decode("utf-8")})


def estimate_story_bible_request_tokens(rendered: RenderedPrompt) -> int:
    return estimate_tokens(rendered.system_text) + estimate_tokens(rendered.user_text)
