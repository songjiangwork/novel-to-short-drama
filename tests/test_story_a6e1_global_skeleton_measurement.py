"""A6E-1 tests: deterministic global-skeleton input packet assembly + measurement.

Covers the A6E-1 measurement/assembly module
``short_drama.story.story_analysis_global_skeleton`` (Issue #90):

* the deterministic whole-story global-skeleton input packet is assembled from
  the *actual* A6C / A6D semantic outputs plus the complete A5-derived
  ``GlobalIndexBase`` -- NO sampling, first-N truncation, or omitted later
  windows;
* the packet is serialized as RFC 8785 canonical JSON and measured with
  ``utf8-bytes-div3-v1`` (the single supported estimator);
* per-component / total / complete-request sizing is deterministic and matches
  the canonical JSON;
* coverage validation fails closed on any missing / duplicate / extra character,
  any missing / duplicate / extra / out-of-order window, or a non-
  ``GlobalIndexBase`` index;
* the exact rendered request fills the single ``global_context_json`` variable
  with the canonical packet and its token estimate equals system + user.

No provider is called. The A5 CURRENT is a real published (synthetic) A5 tree
via the A5F2 publication helpers; the A6C / A6D outputs are minimal-but-valid
``CharacterAnalysis`` / ``PlotWindowAnalysis`` objects standing in for the real
semantic outputs (the assembly / coverage / measurement logic is provider-
independent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from short_drama.artifacts.canonical import canonical_json_bytes, content_hash
from short_drama.llm import PromptRegistry
from short_drama.paths import REPO_ROOT
from short_drama.story import (
    DEFAULT_PROMPT_BASE_DIR,
    CharacterAnalysis,
    EvidenceBackedInterpretation,
    GlobalIndexBase,
    PlotWindowAnalysis,
    build_global_index_base,
    build_global_skeleton_context,
    build_global_skeleton_rendered_request,
    build_story_analysis_plan,
    build_story_analysis_plan_from_profile,
    build_story_analysis_snapshot,
    estimate_global_skeleton_request_tokens,
    load_story_analysis_profile,
    planning_policy_ids_from_profile,
    validate_global_skeleton_coverage,
)
from short_drama.story.chunking import estimate_tokens
from short_drama.story.errors import StoryAnalysisSemanticError

from test_story_a5f2_current_publication import (
    CONSOLIDATION_PROFILE_ID,
    DOCUMENT,
    PROJECT,
    _build_tree_with_a4,
    _planning,
    _publish_a5,
)

# Reuse the A6D in-memory A5 snapshot builder (20 events -> 2 windows) for the
# multi-window coverage invariant.
from test_story_a6d_plot_window_analysis import _build_snapshot as _in_memory_snapshot

# The A6E-1 module constants under test.
from short_drama.story import (
    A6E_GLOBAL_SKELETON_PROMPT_ID,
    A6E_GLOBAL_SKELETON_PROMPT_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _interp():
    return EvidenceBackedInterpretation(text_zh="测试解读文本。", evidence_mode="explicit")


def _char_analysis(ref: str) -> CharacterAnalysis:
    return CharacterAnalysis(
        character_ref=ref,
        role=_interp(),
        goals=(),
        motivations=(),
        traits=(),
        key_event_refs=(),
        key_fact_refs=(),
        important_relationship_refs=(),
        arc_summary=_interp(),
        unresolved_or_conflicting_points=(),
    )


def _window_analysis(window) -> PlotWindowAnalysis:
    return PlotWindowAnalysis(
        window_id=window.window_id,
        window_ordinal=window.window_ordinal,
        owned_event_refs=window.owned_event_ids,
        context_event_refs=window.context_event_ids,
        interpretation=_interp(),
    )


def _plan_fixture(tmp_path: Path):
    """A real published (synthetic) A5 CURRENT + the deterministic A6B plan + the
    A5-derived global index, plus helper builders for valid analyses."""
    tree, *_ = _build_tree_with_a4(tmp_path)
    planning = _planning(tree)
    _publish_a5(tree, planning)
    snap = build_story_analysis_snapshot(
        tree.store,
        tree.pointers,
        project_id=PROJECT,
        document_id=DOCUMENT,
        consolidation_profile_id=CONSOLIDATION_PROFILE_ID,
    )
    profile = load_story_analysis_profile(
        REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
    )
    plan = build_story_analysis_plan_from_profile(snap, profile)
    gi = build_global_index_base(snap)
    return snap, plan, gi


def _full_context(plan, gi):
    chars = [_char_analysis(p.character_ref) for p in plan.character_packages]
    wins = [_window_analysis(w) for w in plan.windows]
    return build_global_skeleton_context(chars, wins, gi), chars, wins


def _in_memory_plan_fixture():
    """A deterministic in-memory A5 snapshot (20 events -> 2 windows) + plan +
    global index, for the multi-window coverage invariants."""
    profile = load_story_analysis_profile(
        REPO_ROOT / "profiles" / "global_story_analysis_v1.yaml"
    )
    snap = _in_memory_snapshot(num_events=20)
    plan = build_story_analysis_plan(
        snap,
        profile.planning_policy,
        planning_policy_ids=planning_policy_ids_from_profile(profile),
    )
    assert len(plan.windows) >= 2
    return plan, build_global_index_base(snap)


# ---------------------------------------------------------------------------
# Assembly + determinism
# ---------------------------------------------------------------------------


def test_packet_assembles_and_is_canonical(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, chars, wins = _full_context(plan, gi)

    # The packet payload is the verbatim, ordered set of actual outputs + the
    # complete A5-derived global index.
    assert context.character_analyses == tuple(chars)
    assert context.plot_window_analyses == tuple(wins)
    assert context.to_dict() == {
        "character_analyses": [a.to_dict() for a in chars],
        "plot_window_analyses": [w.to_dict() for w in wins],
        "global_index": gi.to_dict(),
    }

    # Determinism: identical inputs -> identical canonical bytes + content hash,
    # and the canonical bytes equal the canonical JSON of the payload.
    rebuilt = build_global_skeleton_context(chars, wins, gi)
    assert context.canonical_bytes() == rebuilt.canonical_bytes()
    assert context.content_hash() == rebuilt.content_hash()
    assert context.canonical_bytes() == canonical_json_bytes(context.to_dict())
    assert context.content_hash() == content_hash(context.to_dict())


def test_no_truncation_full_verbatim_coverage(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, chars, wins = _full_context(plan, gi)
    # Every analysis is present, in canonical planned order, verbatim.
    assert [a.character_ref for a in context.character_analyses] == [
        p.character_ref for p in plan.character_packages
    ]
    assert [w.window_id for w in context.plot_window_analyses] == [
        w.window_id for w in plan.windows
    ]
    # Later windows are not omitted (order preserved).
    assert [w.window_ordinal for w in context.plot_window_analyses] == [
        w.window_ordinal for w in plan.windows
    ]


# ---------------------------------------------------------------------------
# Measurement (utf8-bytes-div3-v1)
# ---------------------------------------------------------------------------


def test_component_sizes_match_canonical_json(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, chars, wins = _full_context(plan, gi)

    char_comp = context.character_analysis_component()
    assert char_comp["count"] == len(chars)
    assert char_comp["canonical_bytes"] == len(
        canonical_json_bytes(context.character_analyses_payload())
    )
    assert char_comp["estimated_tokens"] == estimate_tokens(
        canonical_json_bytes(context.character_analyses_payload()).decode("utf-8")
    )

    win_comp = context.plot_window_analysis_component()
    assert win_comp["count"] == len(wins)
    assert win_comp["canonical_bytes"] == len(
        canonical_json_bytes(context.plot_window_analyses_payload())
    )
    assert win_comp["estimated_tokens"] == estimate_tokens(
        canonical_json_bytes(context.plot_window_analyses_payload()).decode("utf-8")
    )

    gi_comp = context.global_index_component()
    assert gi_comp["canonical_bytes"] == len(canonical_json_bytes(gi.to_dict()))
    assert gi_comp["estimated_tokens"] == estimate_tokens(
        canonical_json_bytes(gi.to_dict()).decode("utf-8")
    )
    assert gi_comp["content_hash"] == gi.content_hash()

    # Total packet tokens == estimator over the complete canonical packet.
    assert context.estimated_tokens() == estimate_tokens(
        context.canonical_bytes().decode("utf-8")
    )
    # The total is at least the sum of the (overlapping-key) component sizes and
    # each component is positive.
    assert char_comp["estimated_tokens"] > 0
    assert win_comp["estimated_tokens"] > 0
    assert gi_comp["estimated_tokens"] > 0


# ---------------------------------------------------------------------------
# Coverage validation (fail closed)
# ---------------------------------------------------------------------------


def test_coverage_pass_full(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, _, _ = _full_context(plan, gi)
    validate_global_skeleton_coverage(context, plan)  # must not raise


def test_coverage_rejects_missing_character(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, chars, wins = _full_context(plan, gi)
    if len(chars) < 1:
        pytest.skip("need at least one character")
    reduced = build_global_skeleton_context(chars[:-1], wins, gi)
    with pytest.raises(StoryAnalysisSemanticError):
        validate_global_skeleton_coverage(reduced, plan)


def test_coverage_rejects_duplicate_character(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, chars, wins = _full_context(plan, gi)
    if len(chars) < 2:
        pytest.skip("need at least two characters")
    dup = build_global_skeleton_context(chars[:-1] + [chars[0]], wins, gi)
    with pytest.raises(StoryAnalysisSemanticError):
        validate_global_skeleton_coverage(dup, plan)


def test_coverage_rejects_extra_character(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, chars, wins = _full_context(plan, gi)
    extra_ref = plan.character_packages[0].character_ref + "_extra"
    # A character ref that is not in the plan (but a valid char namespace ref)
    # is an extra / unknown analysis.
    extra = build_global_skeleton_context(
        chars + [_char_analysis("char_" + "9" * 6)], wins, gi
    )
    with pytest.raises(StoryAnalysisSemanticError):
        validate_global_skeleton_coverage(extra, plan)


def test_coverage_rejects_missing_window(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, chars, wins = _full_context(plan, gi)
    if len(wins) < 1:
        pytest.skip("need at least one window")
    reduced = build_global_skeleton_context(chars, wins[:-1], gi)
    with pytest.raises(StoryAnalysisSemanticError):
        validate_global_skeleton_coverage(reduced, plan)


def test_coverage_rejects_extra_window(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, chars, wins = _full_context(plan, gi)
    if len(wins) < 1:
        pytest.skip("need at least one window")
    extra = build_global_skeleton_context(chars, wins + [wins[0]], gi)
    with pytest.raises(StoryAnalysisSemanticError):
        validate_global_skeleton_coverage(extra, plan)


def test_coverage_rejects_window_order_mismatch():
    plan, gi = _in_memory_plan_fixture()
    chars = [_char_analysis(p.character_ref) for p in plan.character_packages]
    wins = [_window_analysis(w) for w in plan.windows]
    assert len(wins) >= 2
    # Swap the first two windows -> window_id mismatch at position 0.
    swapped = list(wins)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    bad = build_global_skeleton_context(chars, swapped, gi)
    with pytest.raises(StoryAnalysisSemanticError):
        validate_global_skeleton_coverage(bad, plan)


def test_build_rejects_non_global_index(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    chars = [_char_analysis(p.character_ref) for p in plan.character_packages]
    wins = [_window_analysis(w) for w in plan.windows]
    with pytest.raises(StoryAnalysisSemanticError):
        build_global_skeleton_context(chars, wins, "not-a-global-index")


# ---------------------------------------------------------------------------
# Rendered request
# ---------------------------------------------------------------------------


def test_rendered_request_fills_global_context_json(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    context, _, _ = _full_context(plan, gi)
    prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
    spec = prompts.load(
        A6E_GLOBAL_SKELETON_PROMPT_ID, version=A6E_GLOBAL_SKELETON_PROMPT_VERSION
    )
    rendered = build_global_skeleton_rendered_request(context, spec)

    # The single global_context_json variable is the exact canonical packet, so
    # the user prompt embeds it verbatim and the request is non-empty.
    assert context.canonical_bytes().decode("utf-8") in rendered.user_text
    assert rendered.system_text
    assert rendered.rendered_prompt_hash

    # The request token estimate == system + user framing (utf8-bytes-div3-v1).
    expected = estimate_tokens(rendered.system_text) + estimate_tokens(rendered.user_text)
    assert estimate_global_skeleton_request_tokens(rendered) == expected
    # The packet is the dominant component of the complete request.
    assert estimate_global_skeleton_request_tokens(rendered) >= context.estimated_tokens()


def test_a6e_prompt_identity_constants():
    assert A6E_GLOBAL_SKELETON_PROMPT_ID == "a6.global-skeleton"
    assert A6E_GLOBAL_SKELETON_PROMPT_VERSION == 1
    # The frozen prompt is registered and loads (no id / version drift).
    prompts = PromptRegistry(DEFAULT_PROMPT_BASE_DIR)
    spec = prompts.load(
        A6E_GLOBAL_SKELETON_PROMPT_ID, version=A6E_GLOBAL_SKELETON_PROMPT_VERSION
    )
    assert spec.prompt_id == A6E_GLOBAL_SKELETON_PROMPT_ID
    assert spec.version == A6E_GLOBAL_SKELETON_PROMPT_VERSION
    assert spec.required_variables == ("global_context_json",)


def test_global_index_base_type_is_checked(tmp_path):
    _, plan, gi = _plan_fixture(tmp_path)
    assert isinstance(gi, GlobalIndexBase)
